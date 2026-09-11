#!/usr/bin/env python3
"""Inference for score-only ray-conditioned grasp confidence calibration."""
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

from utils.ray_confidence_runtime import parse_rc_cli, load_rc_model

SPLITS = {"test_seen": (100, 130), "test_similar": (130, 160), "test_novel": (160, 190)}


def _evaluate(cfg, args):
    from utils.ray_confidence_runtime import json_write
    root = Path(cfg.save_dir)
    protocol_path = root / "rc_inference_summary.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing RC inference protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("complete_sampled_split", False):
        raise RuntimeError("Refusing AP evaluation for incomplete RC predictions.")
    for name, value in (("split", cfg.test_mode), ("camera", cfg.camera)):
        if protocol.get(name) != value:
            raise ValueError(f"RC evaluation {name} differs from saved inference protocol.")
    stride = int(protocol["sample_stride"])
    lo, hi = SPLITS[cfg.test_mode]
    missing = [
        str(root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy")
        for scene in range(lo, hi)
        for anno in range(0, 256, stride)
        if not (root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"RC evaluation is missing {len(missing)} prediction files: {missing[:5]}")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval.py"),
        "--dataset_root", cfg.dataset_root,
        "--dump_dir", str(root),
        "--camera", cfg.camera,
        "--split", cfg.test_mode,
        "--num_workers", str(args.rc_eval_workers),
        "--sample_interval", str(stride),
    ]
    print(f"[RC][EVAL] split={cfg.test_mode} stride={stride}", flush=True)
    with (root / "evaluation.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    result_path = root / f"ap_{cfg.test_mode}_{cfg.camera}.npy"
    result = np.asarray(np.load(result_path, allow_pickle=False), dtype=np.float64)
    if result.size == 0 or result.ndim < 1 or result.shape[-1] < 4 or not np.isfinite(result).all():
        raise RuntimeError(f"Malformed RC AP tensor: shape={result.shape}")
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
    json_write(root / "rc_eval_summary.json", summary)
    print(
        f"[RC][EVAL] {cfg.test_mode}: AP={100*summary['ap']:.4f} "
        f"AP0.4={100*summary['ap0.4']:.4f} AP0.8={100*summary['ap0.8']:.4f}",
        flush=True,
    )


def main():
    args, cfg = parse_rc_cli(training=False)
    if cfg.test_mode not in SPLITS or not cfg.save_dir:
        raise ValueError("Specify --test_mode test_seen/test_similar/test_novel and --save_dir.")
    if cfg.collision_thresh < 0:
        raise ValueError("collision_thresh must be non-negative.")
    if args.rc_run_eval and args.rc_max_batches:
        raise ValueError("AP evaluation requires rc_max_batches=0.")
    if args.rc_eval_only:
        if int(os.environ.get("RANK", "0")) == 0:
            _evaluate(cfg, args)
        return

    from utils.ray_confidence_runtime import (
        init_distributed, cleanup_distributed, build_dataset, make_loader,
        move_batch, json_write, stride_from_fraction,
    )
    rank, world, device = init_distributed(cfg.seed)
    complete = False
    try:
        model, checkpoint, _is_rc, offsets = load_rc_model(args, cfg, device, require_rc=True)
        from models.economicgrasp_ray_confidence import decode_ray_confidence_grasps, RC_BASE_MAIN_SHA
        from graspnetAPI import GraspGroup
        model.eval()

        base, dataset, indices = build_dataset(cfg, cfg.test_mode, cfg.sample_interval, labels=False)
        cfg.eval_num_workers = cfg.num_workers
        loader, _ = make_loader(dataset, indices, cfg, rank, world, False)
        shard = indices[rank::world]
        root = Path(cfg.save_dir)
        root.mkdir(parents=True, exist_ok=True)
        protocol = {
            "experiment": "ray-conditioned-grasp-confidence-calibration",
            "base_main_sha": RC_BASE_MAIN_SHA,
            "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
            "split": cfg.test_mode,
            "camera": cfg.camera,
            "ray_offsets_mm": list(offsets),
            "score_mode": args.rc_score_mode,
            "sample_fraction_requested": float(cfg.sample_interval),
            "sample_stride": stride_from_fraction(cfg.sample_interval),
            "expected_samples": len(indices),
            "native_image_rays": int(cfg.m_point),
            "candidate_budget_before_collision": "one native Stage-1 grasp per image-FPS ray",
            "center_changed": False,
            "view_changed": False,
            "angle_changed": False,
            "insertion_depth_changed": False,
            "width_changed": False,
            "only_possible_network_output_change": "grasp score/ranking",
            "network_input": "RGB+camera calibration/pose metadata required by Stage-1",
            "gt_depth_network_input": False,
            "collision_thresh": float(cfg.collision_thresh),
            "collision_uses_observed_cloud": bool(cfg.collision_thresh > 0),
            "pose_depth_mode": cfg.pose_depth_mode,
            "use_fuse_depth_dataset": bool(cfg.use_fuse_depth),
            "world_size": world,
            "checkpoint_is_smoke": bool(checkpoint.get("rc_smoke_max_batches", 0)),
            "complete_sampled_split": False,
        }
        if rank == 0:
            json_write(root / "rc_inference_summary.json", protocol)
        if world > 1:
            dist.barrier()

        processed = 0
        gate_sum = torch.zeros(1, dtype=torch.float64, device=device)
        gate_sq_sum = torch.zeros(1, dtype=torch.float64, device=device)
        gate_count = torch.zeros(1, dtype=torch.float64, device=device)
        suppress_sum = torch.zeros(1, dtype=torch.float64, device=device)
        started = time.monotonic()
        for step, batch in enumerate(loader):
            if args.rc_max_batches and step >= args.rc_max_batches:
                break
            batch.pop("gt_depth_m", None)
            batch = move_batch(batch, device)
            with torch.inference_mode():
                end_points = model(batch, with_labels=False)
                grasps = decode_ray_confidence_grasps(end_points, score_mode=args.rc_score_mode)
                gate = end_points["rc_confidence_gate"].double()
                gate_sum += gate.sum()
                gate_sq_sum += (gate * gate).sum()
                gate_count += gate.new_tensor(float(gate.numel()))
                suppress_sum += (1.0 - gate).clamp_min(0.0).sum()

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
                    f"[RC][INFER] step={step}/{len(loader)} local_frames={processed}/{len(shard)} "
                    f"elapsed={time.monotonic()-started:.1f}s score_mode={args.rc_score_mode}",
                    flush=True,
                )
            del end_points, grasps, batch

        totals = torch.cat((
            torch.tensor([processed], dtype=torch.float64, device=device),
            gate_sum, gate_sq_sum, gate_count, suppress_sum,
        ))
        if world > 1:
            dist.all_reduce(totals)
        total_processed = int(totals[0].item())
        count = max(float(totals[3].item()), 1.0)
        gate_mean = float(totals[1].item()) / count
        gate_var = max(float(totals[2].item()) / count - gate_mean * gate_mean, 0.0)
        complete = total_processed == len(indices) and args.rc_max_batches == 0
        if args.rc_max_batches == 0 and not complete:
            raise RuntimeError(f"RC saved {total_processed} frames but expected {len(indices)}.")
        if rank == 0:
            protocol.update(
                processed_samples=total_processed,
                complete_sampled_split=complete,
                confidence_gate_mean=gate_mean,
                confidence_gate_std=float(np.sqrt(gate_var)),
                confidence_suppression_mean=float(totals[4].item()) / count,
                elapsed_seconds=time.monotonic() - started,
            )
            json_write(root / "rc_inference_summary.json", protocol)
            print(f"[RC] saved {total_processed} frames to {root}; complete={complete}", flush=True)
        if world > 1:
            dist.barrier()
    finally:
        cleanup_distributed()

    if args.rc_run_eval and rank == 0 and complete:
        _evaluate(cfg, args)


if __name__ == "__main__":
    main()
