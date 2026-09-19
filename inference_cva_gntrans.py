#!/usr/bin/env python3
"""Strict EconomicGrasp CVA-CDF inference on GN-Trans RGB.

The network path remains RGB-only: metric depth used by EconomicGrasp is the
model prediction.  GN-Trans rendered GT depth is present in the dataset only for
preprocessing/diagnostics and is never selected as the network geometry source.

Collision filtering is explicit because it is outside the RGB network:
  none            : strict RGB-only grasp dump (collision_thresh must be 0)
  original_sensor : use the paired original GraspNet sensor cloud; this matches
                    the standard GraspNet post-filter and isolates the RGB shift
  virtual_gt      : use GN-Trans rendered GT geometry (privileged diagnostic)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup


def _parse_gntrans_flags():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gntrans_rgb_root", required=True)
    p.add_argument(
        "--gntrans_collision_source",
        choices=("none", "original_sensor", "virtual_gt"),
        default="none",
    )
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


G = _parse_gntrans_flags()
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import (
    GraspNetMultiDataset,
    GraspNetTransDataset,
    collate_fn,
)
from models.economicgrasp_bip3d import (
    economicgrasp_dpt,
    pred_decode_center_view_angle,
)


def _worker_init(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _build_subset(dataset, sample_fraction: float) -> Tuple[torch.utils.data.Dataset, List[int]]:
    if sample_fraction <= 0:
        raise ValueError("sample_interval must be positive.")
    if sample_fraction >= 1.0:
        indices = list(range(len(dataset)))
        return dataset, indices
    stride = max(1, int(round(1.0 / float(sample_fraction))))
    groups = {}
    for idx, scene in enumerate(dataset.scenename):
        groups.setdefault(str(scene), []).append(idx)
    indices = []
    for scene in sorted(groups):
        rows = sorted(groups[scene], key=lambda i: int(dataset.frameid[i]))
        indices.extend(rows[::stride])
    return Subset(dataset, indices), indices


def _move_fixed_inputs(batch, device):
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"GN-Trans inference received list-valued key {key!r}; "
                "construct the dataset with load_label=False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def _load_checkpoint(checkpoint_path: str):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    return ck, state


def inference() -> None:
    if not cfgs.multi_modal or not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("GN-Trans experiment requires --multi_modal --use_cdf.")
    if not cfgs.save_dir or not cfgs.test_mode:
        raise ValueError("--save_dir and --test_mode are required.")
    if cfgs.test_mode not in ("test_seen", "test_similar", "test_novel"):
        raise ValueError(f"Unsupported test_mode={cfgs.test_mode!r}.")
    if bool(getattr(cfgs, "use_obs_depth", False)) or bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError("GN-Trans RGB-only inference forbids --use_obs_depth/--use_gt_depth.")
    if cfgs.collision_thresh > 0 and G.gntrans_collision_source == "none":
        raise ValueError(
            "collision_thresh>0 requires explicit --gntrans_collision_source "
            "original_sensor or virtual_gt. Set collision_thresh=0 for strict RGB-only output."
        )
    if cfgs.collision_thresh <= 0 and G.gntrans_collision_source != "none":
        print("[GNTRANS] collision source ignored because collision_thresh=0", flush=True)

    os.makedirs(cfgs.save_dir, exist_ok=True)
    full = GraspNetTransDataset(
        cfgs.dataset_root,
        G.gntrans_rgb_root,
        camera=cfgs.camera,
        split=cfgs.test_mode,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=True,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        depth_strides=1,
    )
    selected, indices = _build_subset(full, float(cfgs.sample_interval))
    loader = DataLoader(
        selected,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=cfgs.num_workers,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(cfgs.num_workers > 0),
    )

    original = None
    if cfgs.collision_thresh > 0 and G.gntrans_collision_source == "original_sensor":
        original = GraspNetMultiDataset(
            cfgs.dataset_root,
            camera=cfgs.camera,
            split=cfgs.test_mode,
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=False,
            load_label=False,
            use_gt_depth=False,
        )
        if len(original) != len(full):
            raise RuntimeError("Original/GN-Trans split lengths differ.")

    ck, state = _load_checkpoint(cfgs.checkpoint_path)
    saved_pose_mode = ck.get("pose_depth_mode") if isinstance(ck, dict) else None
    pose_mode = str(getattr(cfgs, "pose_depth_mode", "none"))
    use_top4_view_infer = bool(
        getattr(cfgs, "use_top4_view_infer", False)
    )
    if saved_pose_mode is not None and str(saved_pose_mode) != pose_mode:
        raise ValueError(
            f"Checkpoint pose_depth_mode={saved_pose_mode!r}, CLI={pose_mode!r}."
        )

    print(
        f"[GNTRANS-INFER] total={len(full)} selected={len(selected)} "
        f"cdf=1 top4={use_top4_view_infer} batch={cfgs.batch_size} "
        f"pose_depth_mode={pose_mode}",
        flush=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = economicgrasp_dpt(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=pose_mode,
        use_cdf=True,
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    ).to(device)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.eval()

    protocol = {
        "experiment": "gntrans-mixed-cva-cdf",
        "checkpoint": str(Path(cfgs.checkpoint_path).resolve()),
        "split": cfgs.test_mode,
        "camera": cfgs.camera,
        "gntrans_rgb_root": str(Path(G.gntrans_rgb_root).resolve()),
        "sample_fraction": float(cfgs.sample_interval),
        "selected_samples": len(indices),
        "network_geometry": "predicted_metric_depth",
        "observed_or_gt_depth_network_input": False,
        "collision_thresh": float(cfgs.collision_thresh),
        "collision_source": G.gntrans_collision_source,
        "pose_depth_mode": pose_mode,
        "use_top4_view_infer": use_top4_view_infer,
    }
    Path(cfgs.save_dir, "gntrans_inference_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )

    processed = 0
    start = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        batch = _move_fixed_inputs(batch, device)
        batch["cva_export_angle_feature"] = False
        with torch.inference_mode():
            ep = model(batch)
            pred_depth = ep.get("depth_net_pred")
            used_depth = ep.get("depth_map_used_for_geometry")
            if pred_depth is None or used_depth is None:
                raise RuntimeError("Model did not expose predicted/used geometry depth.")
            if pred_depth.shape != used_depth.shape or float((pred_depth - used_depth).abs().max().item()) > 1e-6:
                raise RuntimeError("GN-Trans inference violated predicted-depth RGB-only geometry contract.")
            grasps = pred_decode_center_view_angle(ep, use_cdf=True)

        for sample_i, pred in enumerate(grasps):
            subset_pos = batch_idx * cfgs.batch_size + sample_i
            if subset_pos >= len(indices):
                raise IndexError("Dataloader produced more samples than selected indices.")
            data_idx = indices[subset_pos]
            gg = GraspGroup(pred.detach().cpu().numpy())

            if cfgs.save_nocollision:
                out_dir = Path(str(cfgs.save_dir) + "_nocollision") / full.scenename[data_idx] / cfgs.camera
                out_dir.mkdir(parents=True, exist_ok=True)
                gg.save_npy(str(out_dir / f"{int(full.frameid[data_idx]):04d}.npy"))

            if cfgs.collision_thresh > 0:
                if G.gntrans_collision_source == "original_sensor":
                    cloud, _ = original.get_data(data_idx, return_raw_cloud=True)
                elif G.gntrans_collision_source == "virtual_gt":
                    cloud, _ = full.get_data(data_idx, return_raw_cloud=True)
                else:
                    raise AssertionError(G.gntrans_collision_source)
                detector = ModelFreeCollisionDetectorTorch(
                    np.asarray(cloud).reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size,
                )
                collision = detector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                gg = gg[~collision.detach().cpu().numpy()]

            out_dir = Path(cfgs.save_dir) / full.scenename[data_idx] / cfgs.camera
            out_dir.mkdir(parents=True, exist_ok=True)
            gg.save_npy(str(out_dir / f"{int(full.frameid[data_idx]):04d}.npy"))
            processed += 1

        if batch_idx % 20 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[GNTRANS-INFER] batch={batch_idx}/{len(loader)} "
                f"samples={processed}/{len(indices)} sec/sample={elapsed/max(processed,1):.3f}",
                flush=True,
            )

    if processed != len(indices):
        raise RuntimeError(f"Saved {processed} predictions, expected {len(indices)}.")
    print(f"[GNTRANS-INFER] complete: {processed} -> {cfgs.save_dir}", flush=True)


if __name__ == "__main__":
    inference()
