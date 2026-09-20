#!/usr/bin/env python3
"""Enrich an exact-action cache with P2 scratch-MLP representation blocks.

This script never calls the CAD/DexNet evaluator. It reuses the exact friction,
collision, and empty labels already stored by the P1 mining stage, but reruns the
*untouched original Stage-1 RGB student* to recover dense DPT maps and the exact
pre-CDF feature. No P1 trained checkpoint or P1 CDF prediction is consumed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _consume_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--p2_source_cache_dir", required=True)
    parser.add_argument("--p2_cache_dir", required=True)
    parser.add_argument("--p2_reference_base_checkpoint", required=True)
    parser.add_argument("--p2_scene_ids", default="")
    parser.add_argument("--p2_action_chunk", type=int, default=2048)
    parser.add_argument("--p2_store_dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--p2_overwrite", type=int, choices=(0, 1), default=0)
    parser.add_argument("--p2_strict", type=int, choices=(0, 1), default=1)
    parser.add_argument("--p2_seed", type=int, default=0)
    parser.add_argument("--p2_worker_tag", default="worker")
    parser.add_argument("--p2_expected_pose_depth_mode", default="global_film")
    parser.add_argument("--p2_expected_use_fuse_depth", type=int, choices=(0, 1), default=1)
    parser.add_argument("--p2_residual_tau_m", type=float, default=0.02)
    parser.add_argument("--p2_surface_tau_m", type=float, default=0.01)
    custom, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *remaining]
    return custom


P2 = _consume_args()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_cdf_cache import validate_cache_arrays
from exact_action_cdf_common import (
    CACHE_SCHEMA_VERSION as EXACT_ACTION_CACHE_SCHEMA_VERSION,
    atomic_save_json,
    atomic_save_npz,
    load_model_state_strict,
    resolve_current_cdf_decoder,
)
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from models.p2_gripper_cdf_field import (
    ACTION_POSE_DIM,
    P2_CACHE_SCHEMA_VERSION,
    P2FieldConfig,
    GripperFieldSampler,
    RAY_FEATURE_DIM,
    build_action_pose_feature,
    monotonic_cdf_logits_from_raw,
    projected_feature_dim,
)
from p2_gripper_field_common import (
    CdfHeadIoCapture,
    assert_current_top1_rgb_output,
    validate_base_checkpoint,
)
from utils.arguments import cfgs
from utils.label_generation import batch_viewpoint_params_to_matrix


class SourceFrameDataset(Dataset):
    def __init__(self, dataset: Dataset, metadata: Sequence[Any], seed: int) -> None:
        self.dataset = dataset
        self.metadata = list(metadata)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        meta = self.metadata[int(index)]
        # exact_action_cdf_cache.CacheFileMeta intentionally keeps only compact
        # shape/contract metadata.  The authoritative dataset index remains in
        # the frame NPZ and must be read here instead of assuming scene-major
        # indexing or a non-existent ``meta.dataset_idx`` attribute.
        with np.load(meta.path, allow_pickle=False) as data:
            dataset_index_array = np.asarray(data["dataset_idx"]).reshape(-1)
        if dataset_index_array.size != 1:
            raise ValueError(
                f"{meta.path}: dataset_idx must contain one scalar, got "
                f"shape={dataset_index_array.shape}"
            )
        dataset_index = int(dataset_index_array[0])

        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            deterministic = self.seed + dataset_index * 1_000_003
            np.random.seed(deterministic % (2**32 - 1))
            random.seed(deterministic)
            sample = self.dataset[dataset_index]
            sample["p2_source_local_index"] = torch.tensor(index, dtype=torch.long)
            return sample
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def _csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _move_inputs(batch: MutableMapping[str, Any], device: torch.device) -> MutableMapping[str, Any]:
    for key in ("point_clouds", "cloud_colors", "coordinates_for_voxel"):
        batch.pop(key, None)
    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(f"Unexpected list-valued P2 input {key!r}")
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    batch["cva_compute_diagnostics"] = False
    batch["geometry_compute_diagnostics"] = False
    batch["cva_export_angle_feature"] = False
    return batch


def _scalar_string(value: np.ndarray) -> str:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError("Expected scalar string")
    return str(array[0])


def _load_source(path: str, base_sha: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    validate_cache_arrays(
        arrays,
        path=path,
        expected_checkpoint_sha256=base_sha,
        check_values=True,
    )
    return arrays



def _scan_selected_source_metadata(
    cache_dir: str,
    *,
    base_sha: str,
    scene_filter: Sequence[int],
    strict: bool,
):
    """Validate only the source-cache scenes assigned to this GPU worker.

    The launcher shards enrichment by scene.  Calling the generic ``scan_cache``
    in every worker would make all GPUs reread and validate the complete P1
    cache before doing any useful work.  This selective scan preserves the same
    per-file validator while avoiding that O(num_workers) I/O amplification.
    """
    root = os.path.abspath(cache_dir)
    requested = sorted({int(value) for value in scene_filter})
    if requested:
        files = []
        for scene_id in requested:
            files.extend(
                sorted(
                    glob.glob(
                        os.path.join(
                            root,
                            f"scene_{scene_id:04d}",
                            "ann_*.npz",
                        )
                    )
                )
            )
    else:
        files = sorted(glob.glob(os.path.join(root, "scene_*", "ann_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No exact-action source cache files under {root} for scenes={requested}"
        )

    metadata = []
    failures = []
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            metadata.append(
                validate_cache_arrays(
                    arrays,
                    path=path,
                    expected_checkpoint_sha256=base_sha,
                    check_values=True,
                )
            )
        except Exception as exc:
            failures.append((path, repr(exc)))
            if strict:
                raise
    if not metadata:
        raise RuntimeError(
            "No valid exact-action source cache frame remains after validation"
        )
    return metadata, failures

def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def _build_cache_arrays(
    source: Mapping[str, np.ndarray],
    end_points: Mapping[str, Any],
    captured_input: torch.Tensor,
    captured_raw_output: torch.Tensor,
    *,
    source_base_sha256: str,
    cdf_increment_bias: float,
    field_sampler: GripperFieldSampler,
    field_config: P2FieldConfig,
    action_chunk: int,
    store_dtype: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    num_thresholds, q, num_angles, num_depths = assert_current_top1_rgb_output(
        end_points, m_point=int(cfgs.m_point)
    )
    if captured_input.shape[0] != 1 or captured_raw_output.shape[0] != 1:
        raise RuntimeError("P2 cache requires batch size 1")
    _, feature_dim, flat_qa = captured_input.shape
    if flat_qa != q * num_angles:
        raise RuntimeError("P2 captured CDF feature ordering mismatch")
    expected_raw_shape = (1, num_depths * num_thresholds, q * num_angles)
    if tuple(captured_raw_output.shape) != expected_raw_shape:
        raise RuntimeError(
            f"P2 raw CDF output shape={tuple(captured_raw_output.shape)}, "
            f"expected={expected_raw_shape}"
        )

    row_center = torch.as_tensor(source["center_id"], device=captured_input.device).long()
    row_angle = torch.as_tensor(source["angle_id"], device=captured_input.device).long()
    rows = int(row_center.numel())
    if row_angle.shape != (rows,):
        raise RuntimeError("source center/angle shapes differ")

    feature_qa = (
        captured_input[0].transpose(0, 1).contiguous().view(q, num_angles, feature_dim).float()
    )
    feature_rows = feature_qa[row_center, row_angle]
    source_feature = torch.as_tensor(source["cdf_head_feature"], device=feature_rows.device).float()
    source_feature_error = _max_abs(feature_rows, source_feature)

    raw_qadt = (
        captured_raw_output.view(1, num_depths, num_thresholds, q, num_angles)[0]
        .permute(2, 3, 0, 1)
        .contiguous()
        .float()
    )
    base_raw_rows = raw_qadt[row_center, row_angle]
    base_logits_rows = monotonic_cdf_logits_from_raw(
        base_raw_rows, float(cdf_increment_bias)
    )
    endpoint_qadt = (
        end_points["grasp_cdf_pred_angle_depth"][0]
        .permute(1, 2, 3, 0)
        .contiguous()
        .float()
    )
    endpoint_rows = endpoint_qadt[row_center, row_angle]
    endpoint_error = _max_abs(base_logits_rows, endpoint_rows)
    source_base_logits = torch.as_tensor(
        source["base_cdf_logits"], device=base_logits_rows.device
    ).float()
    source_base_logits_error = _max_abs(base_logits_rows, source_base_logits)

    width_qad = (
        end_points["grasp_width_pred_angle_depth"][0]
        .permute(1, 2, 0)
        .contiguous()
        .float()
    )
    width_rows = width_qad[row_center, row_angle]
    source_width = torch.as_tensor(source["width_raw"], device=width_rows.device).float()
    width_error = _max_abs(width_rows, source_width)

    centers_q = end_points["xyz_graspable"][0].float()
    views_q = end_points["grasp_top_view_xyz"][0].float()
    tokens_q = end_points["token_sel_idx"][0].long()
    center_rows = centers_q[row_center]
    view_rows = views_q[row_center]
    token_rows = tokens_q[row_center]
    source_center = torch.as_tensor(source["center_xyz"], device=center_rows.device).float()
    source_view = torch.as_tensor(source["view_xyz"], device=view_rows.device).float()
    source_token = torch.as_tensor(source["token_sel_idx"], device=token_rows.device).long()
    center_error = _max_abs(center_rows, source_center)
    view_error = _max_abs(view_rows, source_view)
    token_exact = bool(torch.equal(token_rows, source_token))

    strict_errors = {
        "source_feature_max_abs": source_feature_error,
        "source_center_max_abs": center_error,
        "source_view_max_abs": view_error,
        "source_width_max_abs": width_error,
        "base_endpoint_reconstruction_max_abs": endpoint_error,
        "source_base_logits_max_abs": source_base_logits_error,
    }
    limits = {
        "source_feature_max_abs": 5e-4,
        "source_center_max_abs": 5e-5,
        "source_view_max_abs": 5e-5,
        "source_width_max_abs": 5e-5,
        "base_endpoint_reconstruction_max_abs": 2e-5,
        "source_base_logits_max_abs": 5e-4,
    }
    bad = {key: value for key, value in strict_errors.items() if not np.isfinite(value) or value > limits[key]}
    if bad or not token_exact:
        raise RuntimeError(
            "P2 re-forward does not reproduce the exact-action cache: "
            f"errors={bad}, token_exact={token_exact}.  Use the exact same "
            "checkpoint/config/seed and current image-FPS code."
        )

    # Convert current width head output to the physical width used by decoding.
    width_m = torch.clamp(
        1.2 * width_rows / 10.0,
        min=0.0,
        max=float(cfgs.grasp_max_width),
    )
    depth_values = (
        torch.arange(num_depths, device=width_m.device, dtype=torch.float32) + 1.0
    ) * 0.01
    angle_rad_rows = row_angle.float() * (np.pi / float(num_angles))
    rotation_rows = batch_viewpoint_params_to_matrix(-view_rows, angle_rad_rows).float()

    center_actions = center_rows[:, None, :].expand(rows, num_depths, 3).reshape(-1, 3)
    view_actions = view_rows[:, None, :].expand(rows, num_depths, 3).reshape(-1, 3)
    angle_actions = angle_rad_rows[:, None].expand(rows, num_depths).reshape(-1)
    rotation_actions = rotation_rows[:, None, :, :].expand(rows, num_depths, 3, 3).reshape(-1, 3, 3)
    width_actions = width_m.reshape(-1)
    depth_actions = depth_values.view(1, num_depths).expand(rows, -1).reshape(-1)

    action_pose = build_action_pose_feature(
        center_actions,
        view_actions,
        angle_actions,
        depth_actions,
        width_actions,
        field_config,
    )
    projected, ray, field_diag = field_sampler(
        end_points["img_feat_dpt"],
        end_points["depth_map_used_for_geometry"],
        end_points["K"],
        center_actions,
        rotation_actions,
        width_actions,
        depth_actions,
        action_chunk=int(action_chunk),
    )
    image_feature_dim = int(end_points["img_feat_dpt"].shape[1])
    projected_dim = projected_feature_dim(image_feature_dim)
    disk_dtype = np.float16 if str(store_dtype) == "float16" else np.float32

    def source_copy(name: str, dtype=None):
        value = np.asarray(source[name])
        return value.astype(dtype) if dtype is not None else value.copy()

    arrays: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(P2_CACHE_SCHEMA_VERSION),
        "source_exact_action_cache_schema_version": np.asarray(
            _scalar_string(source["schema_version"])
        ),
        "source_base_checkpoint_sha256": np.asarray(str(source_base_sha256)),
        "field_config_json": np.asarray(field_config.canonical_json()),
        "field_config_sha256": np.asarray(field_config.sha256()),
        "cdf_increment_bias": np.asarray([float(cdf_increment_bias)], dtype=np.float32),
        "cdf_head_feature": feature_rows.cpu().numpy().astype(np.float32),
        "base_cdf_logits": base_logits_rows.cpu().numpy().astype(np.float32),
        "action_pose_feature": action_pose.view(rows, num_depths, -1).cpu().numpy().astype(np.float32),
        "projected_field_feature": projected.view(rows, num_depths, -1).cpu().numpy().astype(disk_dtype),
        "ray_depth_feature": ray.view(rows, num_depths, -1).cpu().numpy().astype(disk_dtype),
        "friction": source_copy("friction", np.float32),
        "collision_or_empty": source_copy("collision_or_empty", np.uint8),
        "pure_collision": source_copy("pure_collision", np.uint8),
        "empty": source_copy("empty", np.uint8),
        "assigned_obj": source_copy("assigned_obj", np.int16),
        "center_xyz": center_rows.cpu().numpy().astype(np.float32),
        "view_xyz": view_rows.cpu().numpy().astype(np.float32),
        "width_raw": width_rows.cpu().numpy().astype(np.float32),
        "center_id": row_center.cpu().numpy().astype(np.int16),
        "angle_id": row_angle.cpu().numpy().astype(np.int8),
        "token_sel_idx": token_rows.cpu().numpy().astype(np.int32),
        "scene_id": source_copy("scene_id", np.int16),
        "anno_id": source_copy("anno_id", np.int16),
        "dataset_idx": source_copy("dataset_idx", np.int32),
        "feature_dim": np.asarray([feature_dim], dtype=np.int16),
        "image_feature_dim": np.asarray([image_feature_dim], dtype=np.int16),
        "num_depths": np.asarray([num_depths], dtype=np.int16),
        "num_thresholds": np.asarray([num_thresholds], dtype=np.int16),
        "action_pose_dim": np.asarray([ACTION_POSE_DIM], dtype=np.int16),
        "projected_feature_dim": np.asarray([projected_dim], dtype=np.int32),
        "ray_feature_dim": np.asarray([RAY_FEATURE_DIM], dtype=np.int16),
        "source_feature_max_abs": np.asarray([source_feature_error], dtype=np.float32),
        "source_center_max_abs": np.asarray([center_error], dtype=np.float32),
        "source_view_max_abs": np.asarray([view_error], dtype=np.float32),
        "source_width_max_abs": np.asarray([width_error], dtype=np.float32),
        "base_endpoint_reconstruction_max_abs": np.asarray([endpoint_error], dtype=np.float32),
        "source_base_logits_max_abs": np.asarray(
            [source_base_logits_error], dtype=np.float32
        ),
        "field_valid_ratio": np.asarray([field_diag["valid_ratio"]], dtype=np.float32),
        "field_depth_valid_ratio": np.asarray([field_diag["depth_valid_ratio"]], dtype=np.float32),
        "field_samples_per_action": np.asarray([field_diag["samples_per_action"]], dtype=np.int16),
    }
    diagnostics = {
        **strict_errors,
        "token_exact": float(token_exact),
        "field_valid_ratio": float(field_diag["valid_ratio"]),
        "field_depth_valid_ratio": float(field_diag["depth_valid_ratio"]),
        "num_rows": float(rows),
        "num_actions": float(rows * num_depths),
        "image_feature_dim": float(image_feature_dim),
    }
    return arrays, diagnostics


def main() -> None:
    if int(cfgs.batch_size) != 1:
        raise RuntimeError("P2 cache enrichment requires --batch_size 1")
    if not bool(cfgs.multi_modal) or not bool(cfgs.use_cdf):
        raise RuntimeError("P2 requires --multi_modal --use_cdf")
    if bool(cfgs.use_obs_depth) or bool(cfgs.use_gt_depth):
        raise RuntimeError("P2 remains RGB-only; remove observed/GT-depth switches")
    if bool(cfgs.kview_use_collision) or bool(cfgs.use_top4_view_infer):
        raise RuntimeError("P2 controlled probe has no learned collision head and is Top-1 only")

    base_checkpoint, base_contract, base_sha = validate_base_checkpoint(
        P2.p2_reference_base_checkpoint,
        expected_pose_depth_mode=str(P2.p2_expected_pose_depth_mode),
        expected_use_fuse_depth=bool(P2.p2_expected_use_fuse_depth),
    )
    scene_filter = set(_csv_ints(P2.p2_scene_ids))
    source_metadata, failures = _scan_selected_source_metadata(
        P2.p2_source_cache_dir,
        base_sha=base_sha,
        scene_filter=sorted(scene_filter),
        strict=bool(P2.p2_strict),
    )
    if failures and bool(P2.p2_strict):
        raise RuntimeError(f"exact-action source cache has failures: {failures[:3]}")
    metadata = [
        m for m in source_metadata if not scene_filter or m.scene_id in scene_filter
    ]
    if not metadata:
        raise RuntimeError("No exact-action source cache frames selected")
    selected_metadata = list(metadata)
    output_root = os.path.abspath(P2.p2_cache_dir)
    os.makedirs(output_root, exist_ok=True)
    pre_skipped = 0
    if not bool(P2.p2_overwrite):
        pending_metadata = []
        for meta in metadata:
            out_path = os.path.join(
                output_root,
                f"scene_{meta.scene_id:04d}",
                f"ann_{meta.anno_id:04d}.npz",
            )
            if os.path.isfile(out_path):
                pre_skipped += 1
            else:
                pending_metadata.append(meta)
        metadata = pending_metadata

    field_config = P2FieldConfig(
        image_height=448,
        image_width=448,
        max_grasp_width_m=float(cfgs.grasp_max_width),
        min_metric_depth_m=float(cfgs.min_depth),
        max_metric_depth_m=float(cfgs.max_depth),
        residual_tau_m=float(P2.p2_residual_tau_m),
        surface_tau_m=float(P2.p2_surface_tau_m),
    )
    if field_config.surface_tau_m <= 0 or field_config.residual_tau_m <= 0:
        raise ValueError("P2 residual/surface tau must be positive")

    seed = int(P2.p2_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        split="train",
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
        extend_angle=False,
    )
    loader = DataLoader(
        SourceFrameDataset(dataset, metadata, seed),
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(cfgs.num_workers)),
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )

    model = economicgrasp_dpt_student(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=base_contract.pose_depth_mode,
        camera_pose_key=str(base_checkpoint.get("camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(base_checkpoint.get("camera_gravity_key", "camera_gravity_vec")),
        pose_hidden_dim=int(base_checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(base_checkpoint.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(base_checkpoint.get("ray_gravity_mid_dim", 32)),
        use_cdf=True,
        vis_dir=None,
    ).to(device)
    load_model_state_strict(model, base_checkpoint["model_state_dict"])
    model.eval().requires_grad_(False)
    decoder = resolve_current_cdf_decoder(model)
    increment_bias = float(getattr(decoder, "cdf_increment_bias", float("nan")))
    if not np.isfinite(increment_bias):
        raise RuntimeError("Current decoder has no finite cdf_increment_bias")
    capture = CdfHeadIoCapture(decoder.cdf_head)
    field_sampler = GripperFieldSampler(field_config).to(device)

    config_payload = {
        "schema_version": P2_CACHE_SCHEMA_VERSION,
        "source_exact_action_cache_dir": os.path.abspath(P2.p2_source_cache_dir),
        "source_exact_action_cache_schema_version": EXACT_ACTION_CACHE_SCHEMA_VERSION,
        "source_base_checkpoint": os.path.abspath(P2.p2_reference_base_checkpoint),
        "source_base_checkpoint_sha256": base_sha,
        "field_config": field_config.to_dict(),
        "field_config_sha256": field_config.sha256(),
        "selected_scene_ids": sorted({m.scene_id for m in selected_metadata}),
        "num_selected_frames": len(selected_metadata),
        "num_pending_frames_at_start": len(metadata),
        "num_existing_frames_at_start": pre_skipped,
        "action_chunk": int(P2.p2_action_chunk),
        "store_dtype": str(P2.p2_store_dtype),
        "worker_tag": str(P2.p2_worker_tag),
    }
    atomic_save_json(
        config_payload,
        os.path.join(output_root, f"_enrich_config_{P2.p2_worker_tag}.json"),
    )

    summaries = []
    started = time.time()
    processed = 0
    skipped = int(pre_skipped)
    try:
        for batch in loader:
            local_index = int(batch.pop("p2_source_local_index").reshape(-1)[0].item())
            meta = metadata[local_index]
            out_path = os.path.join(
                output_root,
                f"scene_{meta.scene_id:04d}",
                f"ann_{meta.anno_id:04d}.npz",
            )
            if os.path.isfile(out_path) and not bool(P2.p2_overwrite):
                print(f"[SKIP] {out_path}", flush=True)
                skipped += 1
                continue
            source = _load_source(meta.path, base_sha)
            batch = _move_inputs(batch, device)
            capture.reset()
            frame_start = time.perf_counter()
            with torch.inference_mode():
                end_points = model(batch)
                head_input, head_output = capture.pop()
                arrays, diagnostics = _build_cache_arrays(
                    source,
                    end_points,
                    head_input,
                    head_output,
                    source_base_sha256=base_sha,
                    cdf_increment_bias=increment_bias,
                    field_sampler=field_sampler,
                    field_config=field_config,
                    action_chunk=int(P2.p2_action_chunk),
                    store_dtype=str(P2.p2_store_dtype),
                )
            atomic_save_npz(out_path, arrays, compress=False)
            frame_sec = time.perf_counter() - frame_start
            row = {
                "scene_id": meta.scene_id,
                "anno_id": meta.anno_id,
                "dataset_idx": int(np.asarray(source["dataset_idx"]).reshape(-1)[0]),
                "frame_sec": frame_sec,
                **diagnostics,
                "cache_path": out_path,
            }
            summaries.append(row)
            processed += 1
            print(
                f"[P2-CACHE] scene={meta.scene_id:04d} anno={meta.anno_id:04d} "
                f"actions={int(diagnostics['num_actions'])} "
                f"valid={diagnostics['field_valid_ratio']:.3f} "
                f"depth_valid={diagnostics['field_depth_valid_ratio']:.3f} "
                f"sec={frame_sec:.2f} processed={processed} skipped={skipped}",
                flush=True,
            )
    finally:
        capture.close()

    completion = {
        **config_payload,
        "status": "complete",
        "processed_new_frames": processed,
        "skipped_existing_frames": skipped,
        "elapsed_sec": time.time() - started,
        "mean_frame_sec": float(np.mean([x["frame_sec"] for x in summaries])) if summaries else None,
        "summaries": summaries,
    }
    atomic_save_json(
        completion,
        os.path.join(output_root, f"_enrich_complete_{P2.p2_worker_tag}.json"),
    )
    print(
        f"[DONE] worker={P2.p2_worker_tag} new={processed} skipped={skipped} "
        f"root={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
