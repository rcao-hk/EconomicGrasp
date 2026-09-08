#!/usr/bin/env python3
"""Strict RGB-only inference for P1 grasp-query center recentering.

The dense DPT metric-depth map is still predicted from RGB.  P1 applies its
learned residual only to student-selected sparse grasp centers before view/CVA
reasoning.  This entry point removes ``gt_depth_m`` from every inference batch
before the model forward, so the learned correction cannot consume privileged
geometry at deployment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup


def _parse_p1_infer_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--p1_force_zero_recenter",
        action="store_true",
        help=(
            "Diagnostic ablation: force the learned sparse residual to zero. "
            "For frozen-base P1 checkpoints this recovers the Stage-1 centers."
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


P1_INFER_ARGS = _parse_p1_infer_args()

# Reuse the current controlled Stage-0/1/2 inference utilities.  Importing the
# module is safe: its inference() is guarded by __main__.
import inference_cva_distill as base

from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.p1_grasp_center_recenter import (
    P1_CENTER_RECENTER_CONTRACT_VERSION,
    economicgrasp_dpt_p1_center_recenter,
)


cfgs = base.cfgs


def _validate_p1_checkpoint(checkpoint: Dict[str, Any]) -> None:
    stage = base._resolve_distill_stage(checkpoint)
    if stage != 1:
        raise RuntimeError(
            f"P1 inference requires distill_stage=1 metadata, got {stage}."
        )
    base._validate_checkpoint_contract(checkpoint, stage)

    if not bool(checkpoint.get("p1_center_recenter", False)):
        raise RuntimeError(
            "Checkpoint is not a P1 grasp-query center-recentering checkpoint."
        )
    if int(checkpoint.get("p1_contract_version", -1)) != (
        P1_CENTER_RECENTER_CONTRACT_VERSION
    ):
        raise RuntimeError(
            "P1 contract mismatch: expected "
            f"{P1_CENTER_RECENTER_CONTRACT_VERSION}, got "
            f"{checkpoint.get('p1_contract_version')!r}."
        )
    if bool(checkpoint.get("p1_inference_requires_gt_depth", True)):
        raise RuntimeError(
            "Checkpoint metadata says P1 inference requires GT depth; refusing "
            "to run it as an RGB-only model."
        )
    if bool(checkpoint.get("p1_uses_dexnet_evaluator", True)):
        raise RuntimeError(
            "P1 checkpoint unexpectedly records evaluator-dependent training."
        )
    if int(checkpoint.get("p1_hidden_dim", 0)) <= 0:
        raise RuntimeError("P1 checkpoint has invalid p1_hidden_dim.")
    max_residual = float(checkpoint.get("p1_max_residual_m", -1.0))
    if not (0.0 < max_residual <= 0.25):
        raise RuntimeError(
            f"P1 checkpoint has invalid p1_max_residual_m={max_residual}."
        )


def _save_json(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def inference() -> None:
    if not bool(getattr(cfgs, "multi_modal", False)):
        raise RuntimeError("P1 CVA inference requires --multi_modal.")
    if not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("P1 supports the current CVA-CDF model only; add --use_cdf.")
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError("P1 deployment is RGB-only; remove --use_obs_depth.")
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError(
            "P1 deployment must not enable the legacy dataset --use_gt_depth switch."
        )
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError(
            "P1 CVA-CDF has no learned collision head; remove --kview_use_collision."
        )
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required.")
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required.")

    checkpoint, state = base._read_checkpoint(cfgs.checkpoint_path)
    _validate_p1_checkpoint(checkpoint)

    checkpoint_use_fuse_depth = bool(checkpoint["use_fuse_depth"])
    requested_use_fuse_depth = bool(getattr(cfgs, "use_fuse_depth", False))
    if requested_use_fuse_depth != checkpoint_use_fuse_depth:
        raise RuntimeError(
            "--use_fuse_depth must match the P1 checkpoint: checkpoint="
            f"{checkpoint_use_fuse_depth}, requested={requested_use_fuse_depth}."
        )
    checkpoint_pose_depth_mode = str(checkpoint.get("pose_depth_mode", ""))
    requested_pose = str(getattr(cfgs, "pose_depth_mode", "none") or "none")
    if requested_pose != checkpoint_pose_depth_mode:
        raise RuntimeError(
            "--pose_depth_mode must match the P1 checkpoint: checkpoint="
            f"{checkpoint_pose_depth_mode!r}, requested={requested_pose!r}."
        )

    os.makedirs(cfgs.save_dir, exist_ok=True)
    full_dataset = base.GraspNetMultiDataset(
        cfgs.dataset_root,
        split=cfgs.test_mode,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=requested_use_fuse_depth,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
    )
    eval_dataset, sampled_indices = base._build_subset(
        full_dataset,
        float(getattr(cfgs, "sample_interval", 1.0)),
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=cfgs.num_workers,
        worker_init_fn=base._worker_init,
        collate_fn=base.collate_fn,
        pin_memory=False,
        persistent_workers=(cfgs.num_workers > 0),
    )
    scene_list = full_dataset.scene_list()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = economicgrasp_dpt_p1_center_recenter(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_cdf=True,
        use_obs_depth=False,
        pose_depth_mode=checkpoint_pose_depth_mode,
        camera_pose_key=str(
            checkpoint.get("camera_pose_key", "camera_pose_vec")
        ),
        camera_gravity_key=str(
            checkpoint.get("camera_gravity_key", "camera_gravity_vec")
        ),
        pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(
            checkpoint.get("ray_gravity_hidden_dim", 64)
        ),
        ray_gravity_mid_dim=int(
            checkpoint.get("ray_gravity_mid_dim", 32)
        ),
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
        p1_hidden_dim=int(checkpoint["p1_hidden_dim"]),
        p1_max_residual_m=float(checkpoint["p1_max_residual_m"]),
        p1_freeze_base=False,
        p1_detach_head_features=False,
        p1_detach_corrected_for_downstream=False,
        p1_compute_gt_targets=False,
        p1_force_zero_residual=bool(
            P1_INFER_ARGS.p1_force_zero_recenter
        ),
    ).to(device)
    base._load_checkpoint_strict(model, state)
    model.eval()

    print(f"[P1-INFER] total={len(full_dataset)} selected={len(eval_dataset)}")
    print(
        "[P1-INFER] RGB-only sparse center recenter: "
        f"mode={checkpoint.get('p1_train_mode')}, "
        f"max_residual={float(checkpoint['p1_max_residual_m']):.4f}m, "
        f"pose_depth_mode={checkpoint_pose_depth_mode}, "
        f"use_fuse_depth={int(requested_use_fuse_depth)}, "
        f"force_zero={int(P1_INFER_ARGS.p1_force_zero_recenter)}, "
        f"top4={int(bool(getattr(cfgs, 'use_top4_view_infer', False)))}, "
        "GT_depth_forward_input=0 DexNet=0",
        flush=True,
    )

    diag_keys = (
        "D: P1 residual abs mean",
        "D: P1 residual abs max",
        "D: P1 residual positive ratio",
        "D: P1 residual clipped ratio",
    )
    diag_sum = {key: 0.0 for key in diag_keys}
    diag_count = {key: 0 for key in diag_keys}

    start = time.perf_counter()
    processed = 0
    role_checked = False
    for batch_idx, batch in enumerate(dataloader):
        # Strong deployment check: even if the dataset returns clean synthetic
        # depth for bookkeeping, P1 does not receive it at forward time.
        batch.pop("gt_depth_m", None)
        batch = base._move_fixed_inputs(batch, device)
        if "gt_depth_m" in batch:
            raise RuntimeError("P1 RGB-only inference still contains gt_depth_m.")
        batch["cva_export_angle_feature"] = False

        with torch.inference_mode():
            end_points = model(batch)
            if not role_checked:
                base._assert_inference_geometry_role(end_points, "pred")
                if not bool(
                    round(float(end_points["D: P1 enabled"].detach().item()))
                ):
                    raise RuntimeError("P1 model did not execute its recenter head.")
                role_checked = True

            for key in diag_keys:
                value = end_points.get(key)
                if torch.is_tensor(value) and value.numel() == 1:
                    diag_sum[key] += float(value.detach().item())
                    diag_count[key] += 1

            grasp_preds = pred_decode_center_view_angle(
                end_points,
                use_cdf=True,
            )

        for sample_i, pred in enumerate(grasp_preds):
            subset_idx = batch_idx * cfgs.batch_size + sample_i
            if subset_idx >= len(sampled_indices):
                raise IndexError(
                    f"Subset index {subset_idx} exceeds {len(sampled_indices)}."
                )
            data_idx = sampled_indices[subset_idx]
            gg = GraspGroup(pred.detach().cpu().numpy())

            if cfgs.save_nocollision:
                out_dir = os.path.join(
                    cfgs.save_dir + "_nocollision",
                    scene_list[data_idx],
                    cfgs.camera,
                )
                os.makedirs(out_dir, exist_ok=True)
                gg.save_npy(
                    os.path.join(out_dir, f"{data_idx % 256:04d}.npy")
                )

            if cfgs.collision_thresh > 0:
                cloud, _ = full_dataset.get_data(
                    data_idx,
                    return_raw_cloud=True,
                )
                detector = base.ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size,
                )
                collision = detector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                gg = gg[~collision.detach().cpu().numpy()]

            out_dir = os.path.join(
                cfgs.save_dir,
                scene_list[data_idx],
                cfgs.camera,
            )
            os.makedirs(out_dir, exist_ok=True)
            gg.save_npy(
                os.path.join(out_dir, f"{data_idx % 256:04d}.npy")
            )
            processed += 1

        if batch_idx % 20 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[P1-INFER] batch={batch_idx}/{len(dataloader)} "
                f"samples={processed}/{len(eval_dataset)} "
                f"sec_per_sample={elapsed / max(processed, 1):.3f}",
                flush=True,
            )

    summary = {
        "experiment": "p1_grasp_query_center_recenter_v1",
        "checkpoint": str(cfgs.checkpoint_path),
        "test_mode": str(cfgs.test_mode),
        "selected_samples": int(processed),
        "p1_train_mode": str(checkpoint.get("p1_train_mode", "")),
        "p1_force_zero_recenter": bool(
            P1_INFER_ARGS.p1_force_zero_recenter
        ),
        "p1_max_residual_m": float(checkpoint["p1_max_residual_m"]),
        "pose_depth_mode": checkpoint_pose_depth_mode,
        "use_fuse_depth": requested_use_fuse_depth,
        "gt_depth_forward_input": False,
        "uses_dexnet_evaluator": False,
        "top4": bool(getattr(cfgs, "use_top4_view_infer", False)),
        "diagnostics": {
            key: (
                diag_sum[key] / diag_count[key]
                if diag_count[key] > 0
                else float("nan")
            )
            for key in diag_keys
        },
    }
    _save_json(
        summary,
        os.path.join(cfgs.save_dir, "p1_inference_summary.json"),
    )
    print(
        "[P1-INFER] correction mean_abs="
        f"{summary['diagnostics']['D: P1 residual abs mean']:.6f}m, "
        "clip_ratio="
        f"{summary['diagnostics']['D: P1 residual clipped ratio']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    inference()
