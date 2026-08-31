#!/usr/bin/env python3
"""Current-only Top-1 inference for the exact-action CDF-head probe.

``base`` and ``exact`` execute the identical corrected Stage-1 RGB-only graph.
The exact checkpoint differs from the base checkpoint only in the existing
``decoder.cdf_head`` weight and bias. No legacy candidate scorer, Top-K view
expansion, learned collision head, or extra inference module is used.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple


def _consume_probe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--exact_action_mode",
        required=True,
        choices=("base", "exact"),
    )
    parser.add_argument(
        "--exact_action_expected_pose_depth_mode",
        default="global_film",
        choices=("global_film", "ray_gravity_film", "none"),
    )
    parser.add_argument(
        "--exact_action_reference_base_checkpoint",
        required=True,
        help=(
            "Untouched current Stage-1 checkpoint used to mine the cache. "
            "Both base and exact modes are checked against its SHA256."
        ),
    )
    custom, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *remaining]
    return custom


PROBE_ARGS = _consume_probe_args()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup

from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student

from exact_action_cdf_common import (
    PROBE_VERSION,
    atomic_save_json,
    load_model_state_strict,
    sha256_file,
    validate_current_stage1_cdf_checkpoint,
)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed((seed + worker_id) % (2**32 - 1))


def _build_subset(
    dataset,
    sample_interval: float,
    annos_per_scene: int = 256,
) -> Tuple[torch.utils.data.Dataset, List[int]]:
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    total = len(dataset)
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices
    stride = max(1, int(round(1.0 / sample_interval)))
    indices: List[int] = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        indices.extend(range(start, end, stride))
    return Subset(dataset, indices), indices


def _move_current_inputs(
    batch: MutableMapping[str, Any],
    device: torch.device,
) -> MutableMapping[str, Any]:
    for key in ("point_clouds", "cloud_colors", "coordinates_for_voxel"):
        batch.pop(key, None)
    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Unexpected list-valued test input {key!r}; load_label must be False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    batch["cva_compute_diagnostics"] = False
    batch["geometry_compute_diagnostics"] = False
    batch["cva_export_angle_feature"] = False
    return batch


def _assert_predicted_depth_role(end_points: Mapping[str, Any]) -> None:
    required = (
        "D: Geometry depth source GT",
        "D: Depth head executed",
        "depth_map_used_for_geometry",
        "depth_net_pred",
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise RuntimeError(f"Inference geometry contract is missing {missing}")
    source_is_gt = bool(round(float(end_points["D: Geometry depth source GT"].item())))
    head_executed = bool(round(float(end_points["D: Depth head executed"].item())))
    if source_is_gt or not head_executed:
        raise RuntimeError("Inference did not execute the Stage-1 predicted-depth path.")
    used = end_points["depth_map_used_for_geometry"]
    predicted = end_points["depth_net_pred"]
    if used.shape != predicted.shape or float((used - predicted).abs().max().item()) > 1e-6:
        raise RuntimeError("Inference geometry does not equal the RGB metric-depth output.")


def _validate_probe_mode(
    checkpoint: Mapping[str, Any],
    mode: str,
    *,
    loaded_checkpoint_sha256: str,
    reference_base_sha256: str,
) -> Dict[str, Any]:
    probe = checkpoint.get("exact_action_cdf_probe", None)
    if mode == "base":
        if probe is not None:
            raise RuntimeError(
                "--exact_action_mode base received an exact-action checkpoint. "
                "Use the untouched Stage-1 checkpoint."
            )
        if loaded_checkpoint_sha256 != reference_base_sha256:
            raise RuntimeError(
                "Base mode checkpoint is not the requested untouched Stage-1 "
                "reference checkpoint."
            )
        return {}
    if not isinstance(probe, Mapping):
        raise RuntimeError(
            "--exact_action_mode exact requires checkpoint['exact_action_cdf_probe']."
        )
    if str(probe.get("version", "")) != PROBE_VERSION:
        raise RuntimeError(
            f"Unexpected exact-action probe version {probe.get('version')!r}; "
            f"expected {PROBE_VERSION!r}."
        )
    if not bool(probe.get("head_only_update", False)):
        raise RuntimeError("Exact checkpoint does not declare a CDF-head-only update.")
    recorded_base_sha = str(probe.get("base_checkpoint_sha256", ""))
    if recorded_base_sha != reference_base_sha256:
        raise RuntimeError(
            "Exact checkpoint was not trained from the supplied reference base: "
            f"recorded={recorded_base_sha!r}, reference={reference_base_sha256!r}."
        )
    updated = list(probe.get("updated_state_keys", []))
    if len(updated) != 2 or not all("decoder.cdf_head." in str(key) for key in updated):
        raise RuntimeError(f"Unexpected exact-action updated_state_keys={updated}")
    return dict(probe)


def main() -> None:
    if not cfgs.checkpoint_path:
        raise ValueError("--checkpoint_path is required")
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required")
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required")
    if not bool(getattr(cfgs, "multi_modal", False)):
        raise RuntimeError("Pass --multi_modal for the current RGB student.")
    if not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("Pass --use_cdf; legacy explicit-angle inference is unsupported.")
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError("Observed-depth inference is unsupported in this probe.")
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError("Remove the deprecated dataset --use_gt_depth switch.")
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError("The current CDF-only model has no learned collision head.")
    if bool(getattr(cfgs, "use_top4_view_infer", False)):
        raise RuntimeError("This controlled exact-action probe is Top-1 only.")

    requested_fuse = bool(getattr(cfgs, "use_fuse_depth", False))
    reference_base_path = os.path.abspath(
        PROBE_ARGS.exact_action_reference_base_checkpoint
    )
    reference_base_sha = sha256_file(reference_base_path)
    checkpoint, contract = validate_current_stage1_cdf_checkpoint(
        cfgs.checkpoint_path,
        expected_pose_depth_mode=PROBE_ARGS.exact_action_expected_pose_depth_mode,
        expected_use_fuse_depth=requested_fuse,
    )
    probe_metadata = _validate_probe_mode(
        checkpoint,
        str(PROBE_ARGS.exact_action_mode),
        loaded_checkpoint_sha256=contract.checkpoint_sha256,
        reference_base_sha256=reference_base_sha,
    )

    seed = int(getattr(cfgs, "seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        split=str(cfgs.test_mode),
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=contract.use_fuse_depth,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
    )
    eval_dataset, sampled_indices = _build_subset(
        full_dataset,
        float(getattr(cfgs, "sample_interval", 1.0)),
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=int(cfgs.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfgs.num_workers)),
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=int(cfgs.num_workers) > 0,
        worker_init_fn=_worker_init,
    )

    model = economicgrasp_dpt_student(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=contract.pose_depth_mode,
        camera_pose_key=str(checkpoint.get("camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(checkpoint.get("camera_gravity_key", "camera_gravity_vec")),
        pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(checkpoint.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(checkpoint.get("ray_gravity_mid_dim", 32)),
        use_cdf=True,
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    ).to(device)
    load_model_state_strict(model, checkpoint["model_state_dict"])
    model.eval()

    output_root = os.path.abspath(cfgs.save_dir)
    os.makedirs(output_root, exist_ok=True)
    manifest = {
        "mode": str(PROBE_ARGS.exact_action_mode),
        "checkpoint_contract": contract.to_dict(),
        "exact_action_metadata": probe_metadata,
        "reference_base_checkpoint": reference_base_path,
        "reference_base_checkpoint_sha256": reference_base_sha,
        "test_mode": str(cfgs.test_mode),
        "camera": str(cfgs.camera),
        "num_samples": len(sampled_indices),
        "sample_interval": float(getattr(cfgs, "sample_interval", 1.0)),
        "batch_size": int(cfgs.batch_size),
        "collision_thresh": float(cfgs.collision_thresh),
        "collision_voxel_size": float(cfgs.collision_voxel_size),
        "graspness_mode": str(cfgs.graspness_mode),
        "top_views": 1,
        "seed_selection_mode": "image_fps",
        "geometry_depth_source": "pred",
    }
    atomic_save_json(manifest, os.path.join(output_root, "_inference_config.json"))

    scene_list = full_dataset.scene_list()
    processed = 0
    start = time.perf_counter()
    role_checked = False
    for batch_index, batch in enumerate(loader):
        batch = _move_current_inputs(batch, device)
        with torch.inference_mode():
            end_points = model(batch)
            if not role_checked:
                _assert_predicted_depth_role(end_points)
                role_checked = True
            grasp_predictions = pred_decode_center_view_angle(
                end_points,
                use_cdf=True,
            )

        for sample_index, prediction in enumerate(grasp_predictions):
            subset_index = batch_index * int(cfgs.batch_size) + sample_index
            if subset_index >= len(sampled_indices):
                raise IndexError("Dataloader produced more samples than the subset index list.")
            dataset_index = int(sampled_indices[subset_index])
            grasp_group = GraspGroup(prediction.detach().cpu().numpy())

            if bool(getattr(cfgs, "save_nocollision", False)):
                raw_dir = os.path.join(
                    output_root + "_nocollision",
                    scene_list[dataset_index],
                    cfgs.camera,
                )
                os.makedirs(raw_dir, exist_ok=True)
                grasp_group.save_npy(
                    os.path.join(raw_dir, f"{dataset_index % 256:04d}.npy")
                )

            if float(cfgs.collision_thresh) > 0:
                cloud, _ = full_dataset.get_data(
                    dataset_index,
                    return_raw_cloud=True,
                )
                detector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size,
                )
                collision = detector.detect(
                    grasp_group,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                grasp_group = grasp_group[~collision.detach().cpu().numpy()]

            out_dir = os.path.join(
                output_root,
                scene_list[dataset_index],
                cfgs.camera,
            )
            os.makedirs(out_dir, exist_ok=True)
            grasp_group.save_npy(
                os.path.join(out_dir, f"{dataset_index % 256:04d}.npy")
            )
            processed += 1

        if batch_index % 20 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[INFER] mode={PROBE_ARGS.exact_action_mode} "
                f"split={cfgs.test_mode} batch={batch_index}/{len(loader)} "
                f"samples={processed}/{len(eval_dataset)} "
                f"sec_per_sample={elapsed/max(processed,1):.3f}",
                flush=True,
            )

    if processed != len(eval_dataset):
        raise RuntimeError(f"Processed {processed} samples, expected {len(eval_dataset)}")
    manifest["processed_samples"] = processed
    manifest["elapsed_sec"] = time.perf_counter() - start
    manifest["status"] = "complete"
    atomic_save_json(manifest, os.path.join(output_root, "_inference_complete.json"))
    print(f"[DONE] {output_root}", flush=True)


if __name__ == "__main__":
    main()
