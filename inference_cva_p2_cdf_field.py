#!/usr/bin/env python3
"""RGB-only inference for Base and four scratch P2 CDF-field variants."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple


def _consume_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--p2_mode",
        required=True,
        choices=("base", "p2_0", "p2_a", "p2_b", "p2_c"),
    )
    parser.add_argument("--p2_reference_base_checkpoint", required=True)
    parser.add_argument("--p2_predictor_checkpoint", default="")
    parser.add_argument("--p2_expected_pose_depth_mode", default="global_film")
    parser.add_argument("--p2_expected_use_fuse_depth", type=int, choices=(0, 1), default=1)
    parser.add_argument("--p2_row_chunk", type=int, default=512)
    custom, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *remaining]
    return custom


P2 = _consume_args()

import numpy as np
import torch
from graspnetAPI import GraspGroup
from torch.utils.data import DataLoader, Subset

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_cdf_common import (
    atomic_save_json,
    load_model_state_strict,
    resolve_current_cdf_decoder,
    sha256_file,
)
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from models.p2_gripper_cdf_field import (
    GripperFieldSampler,
    active_evidence_blocks,
    build_action_pose_feature,
    validate_variant,
)
from p2_gripper_field_common import (
    CdfHeadIoCapture,
    assert_current_top1_rgb_output,
    load_p2_predictor_checkpoint,
    validate_base_checkpoint,
)
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.label_generation import batch_viewpoint_params_to_matrix


FIXED_COLLISION_THRESH = 0.01
FIXED_COLLISION_VOXEL_SIZE = 0.01


def _worker_init(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed((seed + worker_id) % (2**32 - 1))


def _build_subset(dataset, sample_interval: float, annos_per_scene: int = 256):
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    total = len(dataset)
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices
    stride = max(1, int(round(1.0 / sample_interval)))
    indices = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        indices.extend(range(start, end, stride))
    return Subset(dataset, indices), indices


def _move_inputs(batch: MutableMapping[str, Any], device: torch.device):
    for key in ("point_clouds", "cloud_colors", "coordinates_for_voxel"):
        batch.pop(key, None)
    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(f"Unexpected list-valued P2 inference input {key!r}")
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    batch["cva_compute_diagnostics"] = False
    batch["geometry_compute_diagnostics"] = False
    batch["cva_export_angle_feature"] = False
    return batch


def _score_full_lattice(
    end_points: Mapping[str, Any],
    captured_input: torch.Tensor,
    predictor,
    field_sampler: GripperFieldSampler,
    row_chunk: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    t, q, a, d = assert_current_top1_rgb_output(
        end_points, m_point=int(cfgs.m_point)
    )
    if captured_input.shape != (1, predictor.base_feature_dim, q * a):
        raise RuntimeError(
            f"Captured feature shape={tuple(captured_input.shape)}, expected "
            f"{(1, predictor.base_feature_dim, q*a)}"
        )
    if predictor.num_depths != d or predictor.num_thresholds != t:
        raise RuntimeError(
            f"Runtime D/T={d}/{t}, predictor={predictor.num_depths}/"
            f"{predictor.num_thresholds}"
        )
    image_dim = int(end_points["img_feat_dpt"].shape[1])
    if image_dim != predictor.image_feature_dim:
        raise RuntimeError(
            f"Runtime image feature dim={image_dim}, predictor={predictor.image_feature_dim}"
        )

    feature_qa = (
        captured_input[0]
        .transpose(0, 1)
        .contiguous()
        .view(q * a, predictor.base_feature_dim)
        .float()
    )
    width_qad = (
        end_points["grasp_width_pred_angle_depth"][0]
        .permute(1, 2, 0)
        .contiguous()
        .float()
    )  # [Q,A,D]
    centers = end_points["xyz_graspable"][0].float()
    views = end_points["grasp_top_view_xyz"][0].float()
    num_rows = q * a
    output_logits = torch.empty(
        (num_rows, d, t), device=centers.device, dtype=torch.float32
    )
    active = set(active_evidence_blocks(predictor.variant))
    total_valid = 0.0
    total_depth_valid = 0.0
    total_field_actions = 0.0
    chunk = max(1, int(row_chunk))

    depth_values = (
        torch.arange(d, device=centers.device, dtype=torch.float32) + 1.0
    ) * 0.01

    for start in range(0, num_rows, chunk):
        stop = min(start + chunk, num_rows)
        row_id = torch.arange(start, stop, device=centers.device, dtype=torch.long)
        center_id = row_id // a
        angle_id = row_id % a
        rows = int(row_id.numel())
        base_feature = feature_qa[row_id]

        kwargs: Dict[str, torch.Tensor] = {}
        if "pose" in active:
            center = centers[center_id]
            view = views[center_id]
            angle_rad = angle_id.float() * (np.pi / float(a))
            width_m = torch.clamp(
                1.2 * width_qad[center_id, angle_id] / 10.0,
                min=0.0,
                max=float(cfgs.grasp_max_width),
            )  # [R,D]
            depth_m = depth_values.view(1, d).expand(rows, -1)
            center_actions = center[:, None, :].expand(rows, d, 3).reshape(-1, 3)
            view_actions = view[:, None, :].expand(rows, d, 3).reshape(-1, 3)
            angle_actions = angle_rad[:, None].expand(rows, d).reshape(-1)
            width_actions = width_m.reshape(-1)
            depth_actions = depth_m.reshape(-1)
            pose = build_action_pose_feature(
                center_actions,
                view_actions,
                angle_actions,
                depth_actions,
                width_actions,
                field_sampler.config,
            ).view(rows, d, -1)
            kwargs["action_pose_feature"] = pose

            if "projected" in active:
                rotation_rows = batch_viewpoint_params_to_matrix(
                    -view, angle_rad
                ).float()
                rotation_actions = (
                    rotation_rows[:, None, :, :]
                    .expand(rows, d, 3, 3)
                    .reshape(-1, 3, 3)
                )
                projected, ray, diag = field_sampler(
                    end_points["img_feat_dpt"],
                    end_points["depth_map_used_for_geometry"],
                    end_points["K"],
                    center_actions,
                    rotation_actions,
                    width_actions,
                    depth_actions,
                    action_chunk=max(1, chunk * d),
                )
                kwargs["projected_field_feature"] = projected.view(rows, d, -1)
                if "ray_depth" in active:
                    kwargs["ray_depth_feature"] = ray.view(rows, d, -1)
                total_valid += diag["valid_ratio"] * diag["num_actions"]
                total_depth_valid += diag["depth_valid_ratio"] * diag["num_actions"]
                total_field_actions += diag["num_actions"]

        logits, _ = predictor(base_feature, **kwargs)
        output_logits[start:stop] = logits.float()

    qadt = output_logits.view(q, a, d, t)
    endpoint = qadt.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    diagnostics = {
        "num_rows": float(num_rows),
        "num_actions": float(num_rows * d),
        "field_valid_ratio": total_valid / max(total_field_actions, 1.0),
        "field_depth_valid_ratio": total_depth_valid / max(
            total_field_actions, 1.0
        ),
    }
    return endpoint, diagnostics


def main():
    mode = str(P2.p2_mode)
    if not cfgs.checkpoint_path or not cfgs.save_dir or not cfgs.test_mode:
        raise ValueError("checkpoint_path, save_dir, and test_mode are required")
    if int(cfgs.batch_size) != 1:
        raise RuntimeError("P2 inference requires batch_size=1")
    if not bool(cfgs.multi_modal) or not bool(cfgs.use_cdf):
        raise RuntimeError("P2 inference requires --multi_modal --use_cdf")
    if bool(cfgs.use_obs_depth) or bool(cfgs.use_gt_depth):
        raise RuntimeError("P2 inference remains RGB-only")
    if bool(cfgs.kview_use_collision) or bool(cfgs.use_top4_view_infer):
        raise RuntimeError("P2 probe is Top-1 and has no learned collision head")
    if abs(float(cfgs.collision_thresh) - FIXED_COLLISION_THRESH) > 1e-12:
        raise RuntimeError("P2 collision_thresh is fixed to 0.01")
    if abs(float(cfgs.collision_voxel_size) - FIXED_COLLISION_VOXEL_SIZE) > 1e-12:
        raise RuntimeError("P2 collision_voxel_size is fixed to 0.01")

    base_checkpoint, base_contract, base_sha = validate_base_checkpoint(
        P2.p2_reference_base_checkpoint,
        expected_pose_depth_mode=P2.p2_expected_pose_depth_mode,
        expected_use_fuse_depth=bool(P2.p2_expected_use_fuse_depth),
    )
    expected_checkpoint = os.path.abspath(P2.p2_reference_base_checkpoint)
    if os.path.abspath(cfgs.checkpoint_path) != expected_checkpoint:
        raise RuntimeError(
            f"All P2 modes require checkpoint_path={expected_checkpoint}, got "
            f"{os.path.abspath(cfgs.checkpoint_path)}"
        )

    predictor = None
    field_sampler = None
    predictor_checkpoint_sha = ""
    if mode != "base":
        variant = validate_variant(mode)
        if not P2.p2_predictor_checkpoint:
            raise ValueError(f"Mode {mode} requires --p2_predictor_checkpoint")
        _, predictor, field_config = load_p2_predictor_checkpoint(
            P2.p2_predictor_checkpoint,
            expected_variant=variant,
            expected_source_base_checkpoint_sha256=base_sha,
        )
        runtime_contract = {
            "max_grasp_width_m": float(cfgs.grasp_max_width),
            "min_metric_depth_m": float(cfgs.min_depth),
            "max_metric_depth_m": float(cfgs.max_depth),
        }
        for key, actual in runtime_contract.items():
            expected = float(getattr(field_config, key))
            if abs(actual - expected) > 1e-9:
                raise RuntimeError(
                    f"P2 runtime {key}={actual} differs from predictor/cache "
                    f"contract {expected}."
                )
        predictor_checkpoint_sha = sha256_file(P2.p2_predictor_checkpoint)
    elif P2.p2_predictor_checkpoint:
        raise RuntimeError("Base mode must not receive a P2 predictor checkpoint")

    seed = int(cfgs.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        split=str(cfgs.test_mode),
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=base_contract.use_fuse_depth,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
    )
    eval_dataset, sampled_indices = _build_subset(
        dataset, float(cfgs.sample_interval)
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=1,
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
        pose_depth_mode=base_contract.pose_depth_mode,
        camera_pose_key=str(base_checkpoint.get("camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(
            base_checkpoint.get("camera_gravity_key", "camera_gravity_vec")
        ),
        pose_hidden_dim=int(base_checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(
            base_checkpoint.get("ray_gravity_hidden_dim", 64)
        ),
        ray_gravity_mid_dim=int(base_checkpoint.get("ray_gravity_mid_dim", 32)),
        use_cdf=True,
        vis_dir=None,
    ).to(device)
    load_model_state_strict(model, base_checkpoint["model_state_dict"])
    model.eval().requires_grad_(False)
    capture: Optional[CdfHeadIoCapture] = None
    if predictor is not None:
        predictor = predictor.to(device).eval().requires_grad_(False)
        field_sampler = GripperFieldSampler(field_config).to(device).eval()
        capture = CdfHeadIoCapture(resolve_current_cdf_decoder(model).cdf_head)

    output_root = os.path.abspath(cfgs.save_dir)
    os.makedirs(output_root, exist_ok=True)
    manifest = {
        "mode": mode,
        "checkpoint_path": os.path.abspath(cfgs.checkpoint_path),
        "checkpoint_sha256": base_contract.checkpoint_sha256,
        "reference_base_checkpoint": os.path.abspath(
            P2.p2_reference_base_checkpoint
        ),
        "reference_base_checkpoint_sha256": base_sha,
        "p2_predictor_checkpoint": (
            os.path.abspath(P2.p2_predictor_checkpoint)
            if P2.p2_predictor_checkpoint
            else ""
        ),
        "p2_predictor_checkpoint_sha256": predictor_checkpoint_sha,
        "test_mode": str(cfgs.test_mode),
        "camera": str(cfgs.camera),
        "sample_interval": float(cfgs.sample_interval),
        "collision_thresh": float(cfgs.collision_thresh),
        "collision_voxel_size": float(cfgs.collision_voxel_size),
        "top_views": 1,
        "geometry_depth_source": "pred",
        "cdf_only": True,
        "scratch_three_layer_mlp": bool(predictor is not None),
        "uses_p1_checkpoint": False,
        "uses_stage1_or_p1_residual": False,
    }
    atomic_save_json(manifest, os.path.join(output_root, "_inference_config.json"))

    scene_list = dataset.scene_list()
    processed = 0
    start = time.perf_counter()
    field_valid_sum = 0.0
    field_depth_valid_sum = 0.0
    try:
        for batch_index, batch in enumerate(loader):
            batch = _move_inputs(batch, device)
            if capture is not None:
                capture.reset()
            with torch.inference_mode():
                end_points = model(batch)
                assert_current_top1_rgb_output(
                    end_points, m_point=int(cfgs.m_point)
                )
                if predictor is not None:
                    head_input, _ = capture.pop()
                    updated, diag = _score_full_lattice(
                        end_points,
                        head_input,
                        predictor,
                        field_sampler,
                        int(P2.p2_row_chunk),
                    )
                    end_points["grasp_cdf_pred_angle_depth"] = updated
                    field_valid_sum += diag["field_valid_ratio"]
                    field_depth_valid_sum += diag["field_depth_valid_ratio"]
                predictions = pred_decode_center_view_angle(
                    end_points, use_cdf=True
                )

            for sample_index, prediction in enumerate(predictions):
                subset_index = batch_index + sample_index
                dataset_index = int(sampled_indices[subset_index])
                grasp_group = GraspGroup(prediction.detach().cpu().numpy())
                if bool(cfgs.save_nocollision):
                    raw_dir = os.path.join(
                        output_root + "_nocollision",
                        scene_list[dataset_index],
                        cfgs.camera,
                    )
                    os.makedirs(raw_dir, exist_ok=True)
                    grasp_group.save_npy(
                        os.path.join(raw_dir, f"{dataset_index % 256:04d}.npy")
                    )
                cloud, _ = dataset.get_data(
                    dataset_index, return_raw_cloud=True
                )
                detector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size
                )
                collision = detector.detect(
                    grasp_group,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                grasp_group = grasp_group[~collision.detach().cpu().numpy()]
                out_dir = os.path.join(
                    output_root, scene_list[dataset_index], cfgs.camera
                )
                os.makedirs(out_dir, exist_ok=True)
                grasp_group.save_npy(
                    os.path.join(out_dir, f"{dataset_index % 256:04d}.npy")
                )
                processed += 1
            if batch_index % 20 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[P2-INFER] mode={mode} split={cfgs.test_mode} "
                    f"batch={batch_index}/{len(loader)} "
                    f"samples={processed}/{len(eval_dataset)} "
                    f"sec_per_sample={elapsed/max(processed,1):.3f}",
                    flush=True,
                )
    finally:
        if capture is not None:
            capture.close()

    if processed != len(eval_dataset):
        raise RuntimeError(f"Processed {processed}, expected {len(eval_dataset)}")
    manifest.update(
        {
            "status": "complete",
            "processed_samples": processed,
            "elapsed_sec": time.perf_counter() - start,
            "mean_field_valid_ratio": field_valid_sum / max(processed, 1),
            "mean_field_depth_valid_ratio": field_depth_valid_sum
            / max(processed, 1),
        }
    )
    atomic_save_json(
        manifest, os.path.join(output_root, "_inference_complete.json")
    )
    print(f"[DONE] {output_root}", flush=True)


if __name__ == "__main__":
    main()
