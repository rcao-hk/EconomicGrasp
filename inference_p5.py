#!/usr/bin/env python3
"""P5 grasp-repair inference and optional offline GraspNet evaluation."""
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

from utils.p5_runtime import parse_p5_cli, load_p5_model

SPLITS = {"test_seen": (100, 130), "test_similar": (130, 160), "test_novel": (160, 190)}


def _evaluate(cfg, args):
    from utils.p5_runtime import json_write
    root = Path(cfg.save_dir)
    meta_path = root / "p5_inference_summary.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing P5 inference metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete_sampled_split", False):
        raise RuntimeError("Refusing AP evaluation for incomplete P5 inference.")
    if meta.get("split") != cfg.test_mode or meta.get("camera") != cfg.camera:
        raise ValueError("P5 evaluation protocol differs from saved inference metadata.")
    stride = int(meta["sample_stride"])
    lo, hi = SPLITS[cfg.test_mode]
    missing = [
        str(root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy")
        for scene in range(lo, hi) for anno in range(0, 256, stride)
        if not (root / f"scene_{scene:04d}" / cfg.camera / f"{anno:04d}.npy").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"P5 evaluation is missing {len(missing)} prediction files: {missing[:5]}")
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "eval.py"),
           "--dataset_root", cfg.dataset_root, "--dump_dir", str(root),
           "--camera", cfg.camera, "--split", cfg.test_mode,
           "--num_workers", str(args.p5_eval_workers), "--sample_interval", str(stride)]
    print(f"[P5][EVAL] split={cfg.test_mode} stride={stride}", flush=True)
    with (root / "evaluation.log").open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    result_path = root / f"ap_{cfg.test_mode}_{cfg.camera}.npy"
    result = np.asarray(np.load(result_path, allow_pickle=False), dtype=np.float64)
    if result.size == 0 or result.shape[-1] < 4 or not np.isfinite(result).all():
        raise RuntimeError(f"Malformed P5 AP tensor: {result.shape}")
    summary = {
        "split": cfg.test_mode, "result_shape": list(result.shape),
        "ap": float(result.mean()), "ap0.4": float(result[..., 1].mean()),
        "ap0.8": float(result[..., 3].mean()), "sample_stride": stride,
        "result_npy": str(result_path.resolve()), "inference_protocol": meta,
    }
    json_write(root / "p5_eval_summary.json", summary)
    print(f"[P5][EVAL] {cfg.test_mode}: AP={100*summary['ap']:.4f} "
          f"AP0.4={100*summary['ap0.4']:.4f} AP0.8={100*summary['ap0.8']:.4f}", flush=True)


def main():
    args, cfg = parse_p5_cli(training=False)
    if cfg.test_mode not in SPLITS or not cfg.save_dir:
        raise ValueError("Specify --test_mode test_seen/test_similar/test_novel and --save_dir.")
    if cfg.collision_thresh < 0:
        raise ValueError("P5 collision_thresh must be non-negative.")
    if args.p5_run_eval and args.p5_max_batches:
        raise ValueError("P5 AP evaluation requires p5_max_batches=0.")
    if args.p5_eval_only:
        if int(os.environ.get("RANK", "0")) == 0:
            _evaluate(cfg, args)
        return

    from utils.p5_runtime import (
        init_distributed, cleanup_distributed, build_dataset, make_loader,
        move_batch, json_write, stride_from_fraction,
    )
    rank, world, device = init_distributed(cfg.seed)
    complete = False
    try:
        model, ck, _is_p5 = load_p5_model(args, cfg, device, require_p5=True)
        from models.economicgrasp_p5 import decode_p5_grasps, P5_CONTROLLED_MAIN_SHA
        from graspnetAPI import GraspGroup
        model.eval()
        base, dataset, indices = build_dataset(cfg, cfg.test_mode, cfg.sample_interval, labels=False)
        cfg.eval_num_workers = cfg.num_workers
        loader, _ = make_loader(dataset, indices, cfg, rank, world, False)
        shard = indices[rank::world]
        root = Path(cfg.save_dir); root.mkdir(parents=True, exist_ok=True)
        protocol = {
            "experiment": "p5-geometry-error-aware-gripper-repair",
            "controlled_main_sha": P5_CONTROLLED_MAIN_SHA,
            "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
            "split": cfg.test_mode, "camera": cfg.camera, "mode": args.p5_mode,
            "sample_fraction_requested": float(cfg.sample_interval),
            "sample_stride": stride_from_fraction(cfg.sample_interval),
            "expected_samples": len(indices), "native_image_rays": int(cfg.m_point),
            "network_input": "RGB+camera calibration/pose metadata required by Stage-1",
            "gt_depth_network_input": False, "synthetic_corruption_at_inference": False,
            "score_view_angle_depth_width_unchanged": True,
            "translation_repair_only": args.p5_mode == "repair",
            "collision_thresh": float(cfg.collision_thresh),
            "collision_uses_observed_cloud": bool(cfg.collision_thresh > 0),
            "world_size": world,
            "checkpoint_is_smoke": bool(ck.get("p5_smoke_max_batches", 0)),
            "complete_sampled_split": False,
        }
        if rank == 0:
            json_write(root / "p5_inference_summary.json", protocol)
        if world > 1:
            dist.barrier()

        processed = 0
        delta_sum = torch.zeros(1, dtype=torch.float64, device=device)
        delta_gt5 = torch.zeros(1, dtype=torch.float64, device=device)
        delta_count = torch.zeros(1, dtype=torch.float64, device=device)
        t0 = time.monotonic()
        for step, batch in enumerate(loader):
            if args.p5_max_batches and step >= args.p5_max_batches:
                break
            batch = move_batch(batch, device)
            with torch.inference_mode():
                ep = model(batch, with_labels=False)
                grasps = decode_p5_grasps(ep, use_repair=args.p5_mode == "repair")
                dn = ep["p5_delta_local"].float().norm(dim=-1)
                delta_sum += dn.double().sum()
                delta_gt5 += (dn > 0.005).double().sum()
                delta_count += dn.new_tensor(float(dn.numel()), dtype=torch.float64)
            for local_i, grasp in enumerate(grasps):
                global_idx = shard[step * cfg.batch_size + local_i]
                scene = str(base.scenename[global_idx]); anno = int(global_idx % 256)
                gg = GraspGroup(grasp.detach().cpu().numpy())
                if cfg.save_nocollision:
                    p = Path(str(root) + "_nocollision") / scene / cfg.camera / f"{anno:04d}.npy"
                    p.parent.mkdir(parents=True, exist_ok=True); gg.save_npy(str(p))
                if cfg.collision_thresh > 0:
                    from utils.collision_detector import ModelFreeCollisionDetectorTorch
                    cloud, _ = base.get_data(global_idx, return_raw_cloud=True)
                    detector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfg.collision_voxel_size)
                    collision = detector.detect(gg, approach_dist=.05, collision_thresh=cfg.collision_thresh)
                    gg = gg[~collision.detach().cpu().numpy()]
                p = root / scene / cfg.camera / f"{anno:04d}.npy"
                p.parent.mkdir(parents=True, exist_ok=True); gg.save_npy(str(p)); processed += 1
            if rank == 0 and step % 20 == 0:
                print(f"[P5][INFER] step={step}/{len(loader)} frames={processed}/{len(shard)} "
                      f"mode={args.p5_mode} elapsed={time.monotonic()-t0:.1f}s", flush=True)
            del ep, grasps, batch

        totals = torch.cat((torch.tensor([processed], dtype=torch.float64, device=device),
                            delta_sum, delta_gt5, delta_count))
        if world > 1:
            dist.all_reduce(totals)
        total_processed = int(totals[0].item())
        complete = total_processed == len(indices) and args.p5_max_batches == 0
        if args.p5_max_batches == 0 and not complete:
            raise RuntimeError(f"P5 saved {total_processed} frames but expected {len(indices)}.")
        if rank == 0:
            count = max(float(totals[3].item()), 1.0)
            protocol.update(
                processed_samples=total_processed, complete_sampled_split=complete,
                mean_predicted_repair_m=float(totals[1].item()) / count,
                predicted_repair_gt5mm_ratio=float(totals[2].item()) / count,
                elapsed_seconds=time.monotonic() - t0,
            )
            json_write(root / "p5_inference_summary.json", protocol)
            print(f"[P5] saved {total_processed} frames to {root}; complete={complete}", flush=True)
        if world > 1:
            dist.barrier()
    finally:
        cleanup_distributed()

    if args.p5_run_eval and rank == 0 and complete:
        _evaluate(cfg, args)


if __name__ == "__main__":
    main()
