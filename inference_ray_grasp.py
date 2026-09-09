#!/usr/bin/env python3
"""RGB-network ray-field inference and optional *offline* GraspNet evaluation.

Stage-1 checkpoint + raw scoring is a zero-shot multi-depth feasibility run.
P2 checkpoint + supported scoring is the trained representation. Emit one grasp
per ray before optional collision filtering, not K times more proposals.
"""
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

from utils.ray_grasp_runtime import (
    parse_cli, init_distributed, cleanup, load_model, build_dataset, make_loader,
    move_batch, json_write, stride_from_fraction,
)

SPLITS = {"test_seen": (100, 130), "test_similar": (130, 160), "test_novel": (160, 190)}


def evaluate(cfg, args):
    """Run the repository evaluator on a verified, complete sampled split."""
    root = Path(cfg.save_dir)
    protocol_path = root / "ray_inference_summary.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing inference protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    if not protocol["complete_sampled_split"]:
        raise RuntimeError("Refusing AP evaluation of a smoke/incomplete sampled split.")
    for key, value in (("split", cfg.test_mode), ("camera", cfg.camera)):
        if protocol[key] != value:
            raise ValueError(f"Evaluation {key} differs from saved inference protocol.")
    stride = protocol["sample_stride"]
    lo, hi = SPLITS[cfg.test_mode]
    missing = [str(root / f"scene_{s:04d}" / cfg.camera / f"{a:04d}.npy")
               for s in range(lo, hi) for a in range(0, 256, stride)
               if not (root / f"scene_{s:04d}" / cfg.camera / f"{a:04d}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} predicted frames: {missing[:5]}")
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "eval.py"),
           "--dataset_root", cfg.dataset_root, "--dump_dir", str(root),
           "--camera", cfg.camera, "--split", cfg.test_mode,
           "--num_workers", str(args.ray_eval_workers), "--sample_interval", str(stride)]
    print("[RAY][EVAL] offline evaluator; frame stride=", stride, flush=True)
    with (root / "evaluation.log").open("w") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    path = root / f"ap_{cfg.test_mode}_{cfg.camera}.npy"
    result = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if result.size == 0 or result.ndim < 1 or result.shape[-1] < 4 or not np.isfinite(result).all():
        raise RuntimeError(f"Malformed/nonfinite AP tensor: shape={result.shape}")
    summary = {"split": cfg.test_mode, "result_shape": list(result.shape),
               "ap": float(result.mean()), "ap0.4": float(result[..., 1].mean()),
               "ap0.8": float(result[..., 3].mean()), "sample_stride": stride,
               "result_npy": str(path.resolve()), "inference_protocol": protocol}
    json_write(root / "ray_eval_summary.json", summary)
    print(f"[RAY][EVAL] {cfg.test_mode}: AP={100*summary['ap']:.4f}, "
          f"AP0.4={100*summary['ap0.4']:.4f}, AP0.8={100*summary['ap0.8']:.4f}", flush=True)


def main():
    args, cfg = parse_cli(training=False)
    if cfg.test_mode not in SPLITS or not cfg.save_dir:
        raise ValueError("Specify --test_mode test_seen/test_similar/test_novel and --save_dir.")
    if args.ray_eval_workers <= 0 or cfg.collision_thresh < 0:
        raise ValueError("Evaluation workers must be positive, collision threshold nonnegative.")
    if args.ray_run_eval and args.ray_max_batches:
        raise ValueError("Do not request AP evaluation with --ray_max_batches smoke cap.")
    if args.ray_eval_only:
        # No checkpoint, CUDA, model construction or inference is needed here.
        if int(os.environ.get("RANK", 0)) == 0:
            evaluate(cfg, args)
        return
    rank, world, device = init_distributed(cfg.seed)
    complete = False
    try:
        model, ck, is_ray, offsets = load_model(args, cfg, device)
        from models.economicgrasp_ray import decode_ray_grasps, MAIN_BASE_SHA
        from graspnetAPI import GraspGroup
        score_mode = args.ray_score_mode
        if score_mode == "auto":
            score_mode = "supported" if is_ray and ck["ray_loss_weights"]["support"] > 0 else "raw"
        if score_mode == "supported" and (not is_ray or ck["ray_loss_weights"]["support"] <= 0):
            raise ValueError("Supported scoring requires a P2 checkpoint trained with support loss.")
        model.eval()
        base, dataset, indices = build_dataset(cfg, cfg.test_mode, cfg.sample_interval, labels=False)
        # Inference uses num_workers, not the trainer's separate validation setting.
        cfg.eval_num_workers = cfg.num_workers
        loader, _ = make_loader(dataset, indices, cfg, rank, world, False)
        shard = indices[rank::world]
        root = Path(cfg.save_dir)
        root.mkdir(parents=True, exist_ok=True)
        protocol = {"experiment": "ray-conditioned-grasp-v1", "base_main_sha": MAIN_BASE_SHA,
            "checkpoint": str(Path(cfg.checkpoint_path).resolve()), "trained_ray_model": is_ray,
            "split": cfg.test_mode, "camera": cfg.camera, "ray_offsets_mm": list(offsets),
            "score_mode": score_mode, "selection": args.ray_selection, "rays": cfg.m_point,
            "sample_fraction_requested": cfg.sample_interval, "sample_stride": stride_from_fraction(cfg.sample_interval),
            "expected_samples": len(indices), "network_input": "RGB+camera_calibration/pose",
            "gt_depth_network_input": False, "collision_thresh": cfg.collision_thresh,
            "collision_uses_observed_cloud": cfg.collision_thresh > 0,
            "use_fuse_depth_dataset": cfg.use_fuse_depth, "pose_depth_mode": cfg.pose_depth_mode,
            "candidate_budget_before_collision": "one per native image-FPS ray",
            "world_size": world, "complete_sampled_split": False,
            "checkpoint_is_smoke": bool(ck.get("ray_smoke_max_batches", 0))}
        # Mark incomplete before writing any predictions. Old complete metadata
        # must never make an interrupted rerun look like a valid new evaluation.
        if rank == 0:
            json_write(root / "ray_inference_summary.json", protocol)
        if world > 1:
            dist.barrier()
        count = 0
        hist = torch.zeros(len(offsets), dtype=torch.float64, device=device)
        t0 = time.monotonic()
        for step, batch in enumerate(loader):
            if args.ray_max_batches and step >= args.ray_max_batches:
                break
            batch.pop("gt_depth_m", None)
            batch = move_batch(batch, device)
            with torch.inference_mode():
                ep = model(batch, with_labels=False)
                pred, k = decode_ray_grasps(ep, supported=score_mode == "supported", selection=args.ray_selection)
                hist += torch.bincount(k.flatten(), minlength=len(offsets)).to(hist)
            for bi, grasp in enumerate(pred):
                idx = shard[step * cfg.batch_size + bi]
                scene = base.scenename[idx]
                frame = idx % 256  # main GraspNetMultiDataset enumerates each scene's 256 frames in order
                gg = GraspGroup(grasp.cpu().numpy())
                if cfg.save_nocollision:
                    path = Path(str(root) + "_nocollision") / scene / cfg.camera / f"{frame:04d}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    gg.save_npy(str(path))
                if cfg.collision_thresh > 0:
                    from utils.collision_detector import ModelFreeCollisionDetectorTorch
                    cloud, _ = base.get_data(idx, return_raw_cloud=True)
                    detector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfg.collision_voxel_size)
                    col = detector.detect(gg, approach_dist=.05, collision_thresh=cfg.collision_thresh)
                    gg = gg[~col.detach().cpu().numpy()]
                path = root / scene / cfg.camera / f"{frame:04d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                gg.save_npy(str(path))
                count += 1
            if rank == 0 and step % 20 == 0:
                print(f"[RAY][INFER] local_step={step}/{len(loader)} frames={count}/{len(shard)} "
                      f"seconds={time.monotonic()-t0:.1f} mode={score_mode}/{args.ray_selection}", flush=True)
            del ep, pred, batch
        totals = torch.cat((torch.tensor([count], device=device, dtype=hist.dtype), hist))
        if world > 1:
            dist.all_reduce(totals)
        processed = int(totals[0].item())
        complete = processed == len(indices) and not args.ray_max_batches
        if not args.ray_max_batches and not complete:
            raise RuntimeError(f"Saved {processed} frames, expected {len(indices)}.")
        if rank == 0:
            protocol.update(processed_samples=processed, complete_sampled_split=complete,
                            selected_depth_histogram=totals[1:].cpu().tolist(),
                            elapsed_seconds=time.monotonic()-t0)
            json_write(root / "ray_inference_summary.json", protocol)
            print(f"[RAY] saved {processed} predictions to {root}; complete={complete}", flush=True)
        if world > 1:
            dist.barrier()
    finally:
        cleanup()
    # All CUDA/DDP processes are released before the CPU-heavy offline evaluator.
    if args.ray_run_eval and rank == 0 and complete:
        evaluate(cfg, args)


if __name__ == "__main__":
    main()
