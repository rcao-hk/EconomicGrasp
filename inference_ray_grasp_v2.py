#!/usr/bin/env python3
"""Inference for the P2-v2 ray-wise cross-depth Transformer selector."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

# No models import before P2/v2 arguments are consumed.
from utils.ray_grasp_v2_runtime import parse_v2_cli, load_v2_model

SPLITS = {
    "test_seen": (100, 130),
    "test_similar": (130, 160),
    "test_novel": (160, 190),
}


def _evaluate(cfg, args):
    from utils.ray_grasp_runtime import json_write

    root = Path(cfg.save_dir)
    protocol_path = root / "ray_v2_inference_summary.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing P2-v2 inference protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("complete_sampled_split", False):
        raise RuntimeError("Refusing AP evaluation for an incomplete P2-v2 inference split.")
    for key, expected in (("split", cfg.test_mode), ("camera", cfg.camera)):
        if protocol.get(key) != expected:
            raise ValueError(f"Saved P2-v2 protocol {key}={protocol.get(key)!r} != {expected!r}.")
    stride = int(protocol["sample_stride"])
    lo, hi = SPLITS[cfg.test_mode]
    missing = [
        str(root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy")
        for scene in range(lo, hi)
        for anno in range(0, 256, stride)
        if not (root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"P2-v2 dump misses {len(missing)} files; first={missing[:5]}")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval.py"),
        "--dataset_root", cfg.dataset_root,
        "--dump_dir", str(root),
        "--camera", cfg.camera,
        "--split", cfg.test_mode,
        "--num_workers", str(args.v2_eval_workers),
        "--sample_interval", str(stride),
    ]
    with (root / "evaluation.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    result_path = root / f"ap_{cfg.test_mode}_{cfg.camera}.npy"
    result = np.asarray(np.load(result_path, allow_pickle=False), dtype=np.float64)
    if result.size == 0 or result.ndim < 1 or result.shape[-1] < 4 or not np.isfinite(result).all():
        raise RuntimeError(f"Malformed P2-v2 AP tensor: {result.shape}")
    summary = {
        "experiment": "p2-v2-ray-wise-cross-depth-transformer",
        "split": cfg.test_mode,
        "result_shape": list(result.shape),
        "ap": float(result.mean()),
        "ap0.4": float(result[..., 1].mean()),
        "ap0.8": float(result[..., 3].mean()),
        "sample_stride": stride,
        "result_npy": str(result_path.resolve()),
        "inference_protocol": protocol,
    }
    json_write(root / "ray_v2_eval_summary.json", summary)
    print(
        f"[P2-V2][EVAL] {cfg.test_mode}: AP={100*summary['ap']:.4f}, "
        f"AP0.4={100*summary['ap0.4']:.4f}, AP0.8={100*summary['ap0.8']:.4f}",
        flush=True,
    )


def main():
    args, ray_args, cfg = parse_v2_cli(training=False)
    from utils.ray_grasp_runtime import (
        init_distributed, cleanup, build_dataset, make_loader,
        move_batch, json_write, stride_from_fraction,
    )

    if cfg.test_mode not in SPLITS or not cfg.save_dir:
        raise ValueError("Specify --test_mode test_seen/test_similar/test_novel and --save_dir.")
    if cfg.collision_thresh < 0:
        raise ValueError("collision_thresh must be nonnegative.")
    if args.v2_run_eval and args.v2_max_batches:
        raise ValueError("Do not run AP evaluation with --v2_max_batches smoke inference.")
    if args.v2_eval_only:
        if int(os.environ.get("RANK", 0)) == 0:
            _evaluate(cfg, args)
        return

    rank, world, device = init_distributed(cfg.seed)
    complete = False
    try:
        model, ck, is_v2, offsets = load_v2_model(
            args, ray_args, cfg, device, require_v2=True
        )
        if not is_v2:
            raise RuntimeError("Internal error: v2 inference loaded a non-v2 checkpoint.")
        from models.economicgrasp_ray_v2 import decode_ray_grasps_v2
        from graspnetAPI import GraspGroup

        model.eval()
        base, dataset, indices = build_dataset(
            cfg, cfg.test_mode, cfg.sample_interval, labels=False
        )
        cfg.eval_num_workers = cfg.num_workers
        loader, _ = make_loader(dataset, indices, cfg, rank, world, False)
        shard = indices[rank::world]
        root = Path(cfg.save_dir)
        root.mkdir(parents=True, exist_ok=True)
        protocol = {
            "experiment": "p2-v2-ray-wise-cross-depth-transformer",
            "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
            "split": cfg.test_mode,
            "camera": cfg.camera,
            "ray_offsets_mm": list(offsets),
            "selection": args.v2_selection,
            "final_score": args.v2_final_score,
            "sample_fraction_requested": cfg.sample_interval,
            "sample_stride": stride_from_fraction(cfg.sample_interval),
            "expected_samples": len(indices),
            "rays": cfg.m_point,
            "candidate_budget_before_collision": "one grasp per native image-FPS ray",
            "network_input": "RGB+camera calibration/pose only",
            "gt_depth_network_input": False,
            "base_p2_v1_frozen": bool(ck.get("base_p2_v1_frozen", False)),
            "checkpoint_selection": ck.get("checkpoint_selection"),
            "best_validation_selection_regret": ck.get("best_validation_selection_regret"),
            "collision_thresh": cfg.collision_thresh,
            "collision_uses_observed_cloud": cfg.collision_thresh > 0,
            "world_size": world,
            "complete_sampled_split": False,
            "checkpoint_is_smoke": bool(ck.get("ray_v2_smoke_max_batches", 0)),
        }
        if protocol["checkpoint_is_smoke"]:
            raise ValueError("Do not report inference from a P2-v2 smoke-training checkpoint.")
        if rank == 0:
            # Invalidate old completion metadata before touching prediction files.
            json_write(root / "ray_v2_inference_summary.json", protocol)
        if world > 1:
            dist.barrier()

        count = 0
        hist = torch.zeros(len(offsets), dtype=torch.float64, device=device)
        contextual_sum = torch.zeros((), dtype=torch.float64, device=device)
        contextual_count = torch.zeros((), dtype=torch.float64, device=device)
        started = time.monotonic()
        for step, batch in enumerate(loader):
            if args.v2_max_batches and step >= args.v2_max_batches:
                break
            batch.pop("gt_depth_m", None)
            batch = move_batch(batch, device)
            with torch.inference_mode():
                out = model(batch, with_labels=False)
                predictions, selected_k, selected_context = decode_ray_grasps_v2(
                    out,
                    selection=args.v2_selection,
                    final_score=args.v2_final_score,
                )
                hist += torch.bincount(
                    selected_k.flatten(), minlength=len(offsets)
                ).to(hist)
                contextual_sum += selected_context.double().sum()
                contextual_count += float(selected_context.numel())

            for bi, grasp in enumerate(predictions):
                local_position = step * cfg.batch_size + bi
                if local_position >= len(shard):
                    raise IndexError("P2-v2 dataloader/shard ordering mismatch.")
                data_idx = int(shard[local_position])
                scene = base.scenename[data_idx]
                frame = data_idx % 256
                gg = GraspGroup(grasp.detach().cpu().numpy())

                if cfg.save_nocollision:
                    path = Path(str(root) + "_nocollision") / scene / cfg.camera / f"{frame:04d}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    gg.save_npy(str(path))

                if cfg.collision_thresh > 0:
                    from utils.collision_detector import ModelFreeCollisionDetectorTorch
                    cloud, _ = base.get_data(data_idx, return_raw_cloud=True)
                    detector = ModelFreeCollisionDetectorTorch(
                        cloud.reshape(-1, 3), voxel_size=cfg.collision_voxel_size
                    )
                    collision = detector.detect(
                        gg, approach_dist=0.05, collision_thresh=cfg.collision_thresh
                    )
                    gg = gg[~collision.detach().cpu().numpy()]

                path = root / scene / cfg.camera / f"{frame:04d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                gg.save_npy(str(path))
                count += 1

            if rank == 0 and step % 20 == 0:
                print(
                    f"[P2-V2][INFER] step={step}/{len(loader)} frames={count}/{len(shard)} "
                    f"selection={args.v2_selection} final_score={args.v2_final_score} "
                    f"elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
            del out, predictions, selected_k, selected_context, batch

        packed = torch.cat((
            torch.tensor([count], device=device, dtype=torch.float64),
            hist,
            contextual_sum.reshape(1),
            contextual_count.reshape(1),
        ))
        if world > 1:
            dist.all_reduce(packed)
        processed = int(packed[0].item())
        complete = processed == len(indices) and not args.v2_max_batches
        if not args.v2_max_batches and not complete:
            raise RuntimeError(f"P2-v2 saved {processed} frames; expected {len(indices)}.")
        if rank == 0:
            histogram = packed[1:1 + len(offsets)].cpu().tolist()
            csum = float(packed[-2].item())
            ccount = float(packed[-1].item())
            protocol.update(
                processed_samples=processed,
                complete_sampled_split=complete,
                selected_depth_histogram=histogram,
                selected_nonzero_ratio=(
                    1.0 - histogram[list(offsets).index(0.0)] / max(sum(histogram), 1.0)
                ),
                selected_contextual_score_mean=(csum / ccount if ccount else None),
                elapsed_seconds=time.monotonic() - started,
            )
            json_write(root / "ray_v2_inference_summary.json", protocol)
            print(
                f"[P2-V2] saved {processed} predictions to {root}; complete={complete}",
                flush=True,
            )
        if world > 1:
            dist.barrier()
    finally:
        cleanup()

    # Release CUDA/NCCL before the CPU-heavy official evaluator.
    if args.v2_run_eval and rank == 0 and complete:
        _evaluate(cfg, args)


if __name__ == "__main__":
    main()
