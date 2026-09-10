#!/usr/bin/env python3
"""P3 RGB-only ray-evidence inference and optional offline GraspNet evaluation."""
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

from utils.p3_ray_runtime import parse_p3_cli, load_p3_model

SPLITS = {"test_seen": (100, 130), "test_similar": (130, 160), "test_novel": (160, 190)}


def _evaluate(cfg, args):
    from utils.p3_ray_runtime import json_write
    root = Path(cfg.save_dir)
    protocol_path = root / "p3_inference_summary.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing P3 inference protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("complete_sampled_split", False):
        raise RuntimeError("Refusing AP evaluation for an incomplete P3 sampled split.")
    for name, value in (("split", cfg.test_mode), ("camera", cfg.camera)):
        if protocol.get(name) != value:
            raise ValueError(f"P3 evaluation {name} differs from saved inference protocol.")
    stride = int(protocol["sample_stride"])
    lo, hi = SPLITS[cfg.test_mode]
    missing = [
        str(root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy")
        for scene in range(lo, hi)
        for anno in range(0, 256, stride)
        if not (root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"P3 evaluation is missing {len(missing)} prediction files: {missing[:5]}")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval.py"),
        "--dataset_root", cfg.dataset_root,
        "--dump_dir", str(root),
        "--camera", cfg.camera,
        "--split", cfg.test_mode,
        "--num_workers", str(args.p3_eval_workers),
        "--sample_interval", str(stride),
    ]
    print(f"[P3][EVAL] split={cfg.test_mode} stride={stride}", flush=True)
    with (root / "evaluation.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    result_path = root / f"ap_{cfg.test_mode}_{cfg.camera}.npy"
    result = np.asarray(np.load(result_path, allow_pickle=False), dtype=np.float64)
    if result.size == 0 or result.ndim < 1 or result.shape[-1] < 4 or not np.isfinite(result).all():
        raise RuntimeError(f"Malformed P3 AP tensor: shape={result.shape}")
    summary = {
        "split": cfg.test_mode,
        "result_shape": list(result.shape),
        "ap": float(result.mean()),
        "ap0.4": float(result[..., 1].mean()),
        "ap0.8": float(result[..., 3].mean()),
        "sample_stride": stride,
        "result_npy": str(result_path.resolve()),
        "inference_protocol": protocol,
    }
    json_write(root / "p3_eval_summary.json", summary)
    print(
        f"[P3][EVAL] {cfg.test_mode}: AP={100*summary['ap']:.4f} "
        f"AP0.4={100*summary['ap0.4']:.4f} AP0.8={100*summary['ap0.8']:.4f}",
        flush=True,
    )


def main():
    args, cfg = parse_p3_cli(training=False)
    if cfg.test_mode not in SPLITS or not cfg.save_dir:
        raise ValueError("Specify --test_mode test_seen/test_similar/test_novel and --save_dir.")
    if cfg.collision_thresh < 0:
        raise ValueError("collision_thresh must be non-negative.")
    if args.p3_run_eval and args.p3_max_batches:
        raise ValueError("P3 AP evaluation requires p3_max_batches=0.")
    if args.p3_eval_only:
        if int(os.environ.get("RANK", "0")) == 0:
            _evaluate(cfg, args)
        return

    from utils.p3_ray_runtime import (
        init_distributed, cleanup_distributed, build_dataset, make_loader,
        move_batch, json_write, stride_from_fraction,
    )
    rank, world, device = init_distributed(cfg.seed)
    complete = False
    try:
        model, checkpoint, _is_p3, offsets = load_p3_model(args, cfg, device, require_p3=True)
        from models.economicgrasp_ray_p3 import decode_p3_grasps, P3_BASE_MAIN_SHA
        from graspnetAPI import GraspGroup
        model.eval()

        base, dataset, indices = build_dataset(cfg, cfg.test_mode, cfg.sample_interval, labels=False)
        cfg.eval_num_workers = cfg.num_workers
        loader, _ = make_loader(dataset, indices, cfg, rank, world, False)
        shard = indices[rank::world]
        root = Path(cfg.save_dir)
        root.mkdir(parents=True, exist_ok=True)
        protocol = {
            "experiment": "p3-selection-free-cross-depth-evidence-aggregation",
            "base_main_sha": P3_BASE_MAIN_SHA,
            "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
            "split": cfg.test_mode,
            "camera": cfg.camera,
            "ray_offsets_mm": list(offsets),
            "selection_score": args.p3_selection_score,
            "final_score": args.p3_final_score,
            "force_zero": bool(args.p3_force_zero),
            "sample_fraction_requested": float(cfg.sample_interval),
            "sample_stride": stride_from_fraction(cfg.sample_interval),
            "expected_samples": len(indices),
            "native_image_rays": int(cfg.m_point),
            "candidate_budget_before_collision": "one final grasp per native image-FPS ray",
            "hard_depth_selection_before_grasp_head": False,
            "network_input": "RGB+camera calibration/pose metadata required by Stage-1",
            "gt_depth_network_input": False,
            "collision_thresh": float(cfg.collision_thresh),
            "collision_uses_observed_cloud": bool(cfg.collision_thresh > 0),
            "pose_depth_mode": cfg.pose_depth_mode,
            "use_fuse_depth_dataset": bool(cfg.use_fuse_depth),
            "world_size": world,
            "checkpoint_is_smoke": bool(checkpoint.get("p3_smoke_max_batches", 0)),
            "complete_sampled_split": False,
        }
        # Prevent stale completed metadata from validating an interrupted rerun.
        if rank == 0:
            json_write(root / "p3_inference_summary.json", protocol)
        if world > 1:
            dist.barrier()

        hist = torch.zeros(len(offsets), dtype=torch.float64, device=device)
        processed = 0
        started = time.monotonic()
        for step, batch in enumerate(loader):
            if args.p3_max_batches and step >= args.p3_max_batches:
                break
            batch.pop("gt_depth_m", None)
            batch = move_batch(batch, device)
            with torch.inference_mode():
                end_points = model(batch, with_labels=False)
                grasps, selected_k = decode_p3_grasps(
                    end_points,
                    selection_score=args.p3_selection_score,
                    final_score=args.p3_final_score,
                    force_zero=args.p3_force_zero,
                )
                hist += torch.bincount(selected_k.flatten(), minlength=len(offsets)).to(hist)

            for local_i, grasp in enumerate(grasps):
                global_idx = shard[step * cfg.batch_size + local_i]
                scene = str(base.scenename[global_idx])
                anno = int(global_idx % 256)
                group = GraspGroup(grasp.detach().cpu().numpy())
                if cfg.save_nocollision:
                    nc_path = Path(str(root) + "_nocollision") / scene / cfg.camera / f"{anno:04d}.npy"
                    nc_path.parent.mkdir(parents=True, exist_ok=True)
                    group.save_npy(str(nc_path))
                if cfg.collision_thresh > 0:
                    from utils.collision_detector import ModelFreeCollisionDetectorTorch
                    cloud, _ = base.get_data(global_idx, return_raw_cloud=True)
                    detector = ModelFreeCollisionDetectorTorch(
                        cloud.reshape(-1, 3), voxel_size=cfg.collision_voxel_size
                    )
                    collision = detector.detect(
                        group, approach_dist=0.05, collision_thresh=cfg.collision_thresh
                    )
                    group = group[~collision.detach().cpu().numpy()]
                path = root / scene / cfg.camera / f"{anno:04d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                group.save_npy(str(path))
                processed += 1
            if rank == 0 and step % 20 == 0:
                print(
                    f"[P3][INFER] step={step}/{len(loader)} local_frames={processed}/{len(shard)} "
                    f"elapsed={time.monotonic()-started:.1f}s "
                    f"selection={args.p3_selection_score} final={args.p3_final_score} zero={int(args.p3_force_zero)}",
                    flush=True,
                )
            del end_points, grasps, batch

        totals = torch.cat((torch.tensor([processed], dtype=hist.dtype, device=device), hist))
        if world > 1:
            dist.all_reduce(totals)
        total_processed = int(totals[0].item())
        complete = total_processed == len(indices) and args.p3_max_batches == 0
        if args.p3_max_batches == 0 and not complete:
            raise RuntimeError(f"P3 saved {total_processed} frames but expected {len(indices)}.")
        if rank == 0:
            protocol.update(
                processed_samples=total_processed,
                complete_sampled_split=complete,
                selected_depth_histogram=totals[1:].cpu().tolist(),
                elapsed_seconds=time.monotonic() - started,
            )
            json_write(root / "p3_inference_summary.json", protocol)
            print(f"[P3] saved {total_processed} frames to {root}; complete={complete}", flush=True)
        if world > 1:
            dist.barrier()
    finally:
        cleanup_distributed()

    # Release CUDA/DDP before starting the CPU-heavy evaluator.
    if args.p3_run_eval and rank == 0 and complete:
        _evaluate(cfg, args)


if __name__ == "__main__":
    main()
