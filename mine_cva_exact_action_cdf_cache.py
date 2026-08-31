#!/usr/bin/env python3
"""Mine current CVA-CDF head inputs with exact on-policy grasp labels.

Supported model contract
------------------------
Only the corrected Stage-1 RGB student is accepted:

* predicted metric depth (typically ``global_film``),
* deterministic image-space FPS,
* current CDF-only center-view-angle-depth head,
* no learned collision head and no legacy score/depth candidate heads.

For the model's deterministic Top-1 view, the miner selects a small set of
centers, retains all 12 in-plane angles and all four physical depths, and
labels the resulting exact ``(center, view, angle, depth, predicted width)``
actions with GraspNet CAD collision and DexNet force closure. It caches the
input to the existing ``decoder.cdf_head``. This permits head-only fine-tuning
without changing any cached action geometry.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _consume_exact_action_args() -> argparse.Namespace:
    env_key = "CVA_EXACT_ACTION_CDF_MINE_ARGS_JSON"
    defaults: Dict[str, Any] = {}
    raw = os.environ.get(env_key, "").strip()
    if raw:
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"{env_key} must encode a JSON object")
        defaults = loaded

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ea_cache_dir", default="")
    parser.add_argument(
        "--ea_dataset_split",
        default="train",
        choices=("train", "test_seen"),
        help=(
            "GraspNet split used to construct frames and exact evaluator geometry. "
            "Use train for scenes 0-99 and test_seen for scenes 100-129."
        ),
    )
    parser.add_argument("--ea_scene_ids", default="")
    parser.add_argument("--ea_anno_ids", default="")
    parser.add_argument("--ea_sample_interval", type=float, default=0.1)
    parser.add_argument("--ea_max_samples", type=int, default=-1)
    parser.add_argument("--ea_top_centers", type=int, default=16)
    parser.add_argument("--ea_random_centers", type=int, default=4)
    parser.add_argument("--ea_eval_workers", type=int, default=4)
    parser.add_argument("--ea_max_pending", type=int, default=8)
    parser.add_argument("--ea_collision_chunk", type=int, default=512)
    parser.add_argument("--ea_eval_threads", type=int, default=1)
    parser.add_argument(
        "--ea_fc_mode",
        choices=("official", "reuse_contacts"),
        default="reuse_contacts",
    )
    parser.add_argument("--ea_fc_verify_n", type=int, default=0)
    parser.add_argument("--ea_cdf_increment_bias", type=float, default=-4.0)
    parser.add_argument("--ea_compress", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ea_overwrite", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ea_strict", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ea_seed", type=int, default=0)
    parser.add_argument("--ea_worker_tag", default="worker")
    parser.add_argument(
        "--ea_expected_pose_depth_mode",
        default="global_film",
        choices=("global_film", "ray_gravity_film", "none"),
    )
    if defaults:
        known = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in known})

    custom, remaining = parser.parse_known_args(sys.argv[1:])
    os.environ[env_key] = json.dumps(vars(custom), sort_keys=True)
    sys.argv[:] = [sys.argv[0], *remaining]
    return custom


EA = _consume_exact_action_args()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from utils.label_generation import batch_viewpoint_params_to_matrix
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student

from exact_action_cdf_common import (
    CACHE_SCHEMA_VERSION,
    FRICTION_THRESHOLDS,
    atomic_save_json,
    atomic_save_npz,
    load_model_state_strict,
    resolve_current_cdf_decoder,
    validate_current_stage1_cdf_checkpoint,
)
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator


class DeterministicIndexDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], seed: int) -> None:
        self.dataset = dataset
        self.indices = [int(value) for value in indices]
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int):
        dataset_index = self.indices[int(local_index)]
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            deterministic_seed = self.seed + dataset_index * 1_000_003
            np.random.seed(deterministic_seed % (2**32 - 1))
            random.seed(deterministic_seed)
            return self.dataset[dataset_index]
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def _csv_ints(text: str) -> List[int]:
    return [int(value.strip()) for value in str(text).split(",") if value.strip()]


def _validate_requested_scene_ids_for_split() -> None:
    split = str(EA.ea_dataset_split)
    bounds = {
        "train": (0, 99),
        "test_seen": (100, 129),
    }
    lo, hi = bounds[split]
    requested = _csv_ints(EA.ea_scene_ids)
    bad = [scene_id for scene_id in requested if not (lo <= scene_id <= hi)]
    if bad:
        raise ValueError(
            f"--ea_dataset_split={split!r} only contains scenes {lo}-{hi}, "
            f"but --ea_scene_ids includes {bad[:16]}."
        )


def _select_indices(dataset: GraspNetMultiDataset) -> List[int]:
    scene_filter = set(_csv_ints(EA.ea_scene_ids))
    anno_filter = set(_csv_ints(EA.ea_anno_ids))
    interval = float(EA.ea_sample_interval)
    if interval <= 0:
        raise ValueError("--ea_sample_interval must be positive")
    stride = 1 if interval >= 1.0 else max(1, int(round(1.0 / interval)))

    selected: List[int] = []
    for dataset_index, scene_name in enumerate(dataset.scene_list()):
        scene_id = int(str(scene_name).split("_")[-1])
        anno_id = int(dataset_index % 256)
        if scene_filter and scene_id not in scene_filter:
            continue
        if anno_filter:
            if anno_id not in anno_filter:
                continue
        elif anno_id % stride != 0:
            continue
        selected.append(dataset_index)
        if int(EA.ea_max_samples) > 0 and len(selected) >= int(EA.ea_max_samples):
            break
    if not selected:
        raise RuntimeError("No frame selected; check scene/annotation filters.")
    return selected


def _move_current_inputs(
    batch: MutableMapping[str, Any],
    device: torch.device,
) -> MutableMapping[str, Any]:
    # The current RGB student does not consume sampled/captured point tensors.
    for key in ("point_clouds", "cloud_colors", "coordinates_for_voxel"):
        batch.pop(key, None)
    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Unexpected list-valued input {key!r}; cache mining requires load_label=False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    batch["cva_compute_diagnostics"] = False
    batch["geometry_compute_diagnostics"] = False
    batch["cva_export_angle_feature"] = False
    return batch


def _assert_predicted_depth_top1(end_points: Mapping[str, Any]) -> None:
    required = (
        "D: Geometry depth source GT",
        "D: Depth head executed",
        "depth_map_used_for_geometry",
        "depth_net_pred",
        "grasp_cdf_pred_angle_depth",
        "grasp_width_pred_angle_depth",
        "xyz_graspable",
        "grasp_top_view_xyz",
        "grasp_top_view_inds",
        "token_sel_idx",
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise RuntimeError(f"Current model output is missing {missing}")
    source_is_gt = bool(round(float(end_points["D: Geometry depth source GT"].item())))
    depth_head_executed = bool(round(float(end_points["D: Depth head executed"].item())))
    if source_is_gt or not depth_head_executed:
        raise RuntimeError(
            "Cache mining did not execute the Stage-1 predicted-depth geometry path."
        )
    used = end_points["depth_map_used_for_geometry"]
    predicted = end_points["depth_net_pred"]
    if used.shape != predicted.shape or float((used - predicted).abs().max().item()) > 1e-6:
        raise RuntimeError("Geometry depth differs from the current RGB depth-head output.")
    cdf = end_points["grasp_cdf_pred_angle_depth"]
    width = end_points["grasp_width_pred_angle_depth"]
    if cdf.dim() != 5:
        raise RuntimeError(f"CDF logits must be [B,T,Q,A,D], got {tuple(cdf.shape)}")
    batch_size, _t, q, a, d = cdf.shape
    if width.shape != (batch_size, d, q, a):
        raise RuntimeError(
            f"Depth-wise width shape {tuple(width.shape)} is incompatible with "
            f"CDF shape {tuple(cdf.shape)}."
        )
    if q != int(getattr(cfgs, "m_point", 1024)):
        raise RuntimeError(
            f"Top-1 current model should emit Q=m_point, got Q={q}, "
            f"m_point={getattr(cfgs, 'm_point', None)}."
        )


class _CdfHeadIoCapture:
    """Capture the exact input and raw output of the deployed current CDF head.

    The raw output is required for a numerically sound cache-alignment check.
    Recomputing a Conv1d output with F.linear is mathematically equivalent but
    can differ by about 1e-2 on Ampere GPUs because cuDNN and cuBLAS may select
    different TF32 kernels. That discrepancy must not be mistaken for an
    ordering or feature-capture failure.
    """

    def __init__(self, cdf_head: torch.nn.Module) -> None:
        self.input_value: Optional[torch.Tensor] = None
        self.output_value: Optional[torch.Tensor] = None

        def hook(_module, args, output):
            if len(args) != 1 or not torch.is_tensor(args[0]):
                raise RuntimeError("Current cdf_head hook expected one tensor input.")
            if not torch.is_tensor(output):
                raise RuntimeError("Current cdf_head hook expected one tensor output.")
            if self.input_value is not None or self.output_value is not None:
                raise RuntimeError("Current cdf_head executed more than once in one forward.")
            self.input_value = args[0].detach()
            self.output_value = output.detach()

        self.handle = cdf_head.register_forward_hook(hook)

    def reset(self) -> None:
        self.input_value = None
        self.output_value = None

    def pop(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_value is None or self.output_value is None:
            raise RuntimeError("Current cdf_head input/output was not captured.")
        input_value = self.input_value
        output_value = self.output_value
        self.input_value = None
        self.output_value = None
        return input_value, output_value

    def close(self) -> None:
        self.handle.remove()


def _choose_centers(
    cdf_logits: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    # cdf_logits: [T,Q,A,D]. Candidate utility is exactly current inference utility.
    utility = torch.sigmoid(cdf_logits.float()).mean(dim=0)  # [Q,A,D]
    center_score = utility.amax(dim=(-1, -2))
    q = int(center_score.numel())
    top_count = min(max(int(EA.ea_top_centers), 0), q)
    random_count = min(
        max(int(EA.ea_random_centers), 0),
        max(q - top_count, 0),
    )
    if top_count + random_count <= 0:
        raise ValueError("At least one top/random center must be retained.")
    top = (
        torch.topk(center_score, k=top_count, largest=True, sorted=True).indices
        if top_count > 0
        else torch.empty(0, dtype=torch.long, device=center_score.device)
    )
    if random_count > 0:
        all_ids = np.arange(q, dtype=np.int64)
        top_np = top.detach().cpu().numpy()
        remaining = np.setdiff1d(all_ids, top_np, assume_unique=False)
        random_np = rng.choice(remaining, size=random_count, replace=False)
        random_ids = torch.as_tensor(
            random_np, device=center_score.device, dtype=torch.long
        )
        return torch.cat([top, random_ids], dim=0)
    return top


def _build_rows_and_grasps(
    end_points: Mapping[str, Any],
    captured_head_input: torch.Tensor,
    captured_head_output: torch.Tensor,
    center_ids: torch.Tensor,
    checkpoint_sha256: str,
    cdf_head_weight: torch.Tensor,
    cdf_head_bias: torch.Tensor,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int, int]:
    cdf = end_points["grasp_cdf_pred_angle_depth"]
    width = end_points["grasp_width_pred_angle_depth"]
    if (
        cdf.shape[0] != 1
        or captured_head_input.shape[0] != 1
        or captured_head_output.shape[0] != 1
    ):
        raise RuntimeError("Cache miner intentionally requires batch_size=1.")
    _, num_thresholds, q, num_angles, num_depths = cdf.shape
    if num_thresholds != len(FRICTION_THRESHOLDS):
        raise RuntimeError(
            f"Current CDF threshold count={num_thresholds}, expected "
            f"{len(FRICTION_THRESHOLDS)}."
        )
    if num_angles != int(getattr(cfgs, "num_angle", 12)):
        raise RuntimeError("Current CDF angle count differs from cfgs.num_angle.")
    if num_depths != int(getattr(cfgs, "num_depth", 4)):
        raise RuntimeError("Current CDF depth count differs from cfgs.num_depth.")

    _, feature_dim, flat_qa = captured_head_input.shape
    if flat_qa != q * num_angles:
        raise RuntimeError(
            f"Captured cdf_head input has QA={flat_qa}, expected {q}*{num_angles}."
        )
    feature_qa = (
        captured_head_input[0]
        .transpose(0, 1)
        .contiguous()
        .view(q, num_angles, feature_dim)
        .float()
    )
    logits_qadt = cdf[0].permute(1, 2, 3, 0).contiguous().float()
    width_qad = width[0].permute(1, 2, 0).contiguous().float()
    centers = end_points["xyz_graspable"][0].float()
    views = end_points["grasp_top_view_xyz"][0].float()
    view_ids = end_points["grasp_top_view_inds"][0].long()
    token_ids = end_points["token_sel_idx"][0].long()

    center_ids = center_ids.long()
    row_center = center_ids[:, None].expand(-1, num_angles).reshape(-1)
    row_angle = (
        torch.arange(num_angles, device=center_ids.device, dtype=torch.long)
        .view(1, num_angles)
        .expand(center_ids.numel(), -1)
        .reshape(-1)
    )
    num_rows = int(row_center.numel())
    feature_rows = feature_qa[row_center, row_angle]
    logits_rows = logits_qadt[row_center, row_angle]  # [R,D,T]
    width_rows = width_qad[row_center, row_angle]     # [R,D]

    expected_head_output_shape = (
        1,
        num_depths * num_thresholds,
        q * num_angles,
    )
    if tuple(captured_head_output.shape) != expected_head_output_shape:
        raise RuntimeError(
            "Captured current cdf_head output has shape "
            f"{tuple(captured_head_output.shape)}, expected "
            f"{expected_head_output_shape}."
        )

    # Validate cache alignment from the actual deployed Conv1d output. The
    # hooked raw output, after the decoder's D x T reshape and monotonic
    # transform, must reproduce the final endpoint at near machine precision.
    captured_raw_qadt = (
        captured_head_output
        .view(1, num_depths, num_thresholds, q, num_angles)[0]
        .permute(2, 3, 0, 1)
        .contiguous()
    )  # [Q,A,D,T]
    captured_raw_rows = captured_raw_qadt[row_center, row_angle].float()
    captured_base = captured_raw_rows[..., :1]
    captured_increments = F.softplus(
        captured_raw_rows[..., 1:] + float(EA.ea_cdf_increment_bias)
    )
    captured_logits_rows = torch.cat(
        [
            captured_base,
            captured_base + torch.cumsum(captured_increments, dim=-1),
        ],
        dim=-1,
    )
    reconstruction_error = float(
        (captured_logits_rows - logits_rows).abs().max().item()
    )
    if reconstruction_error > 2e-6:
        raise RuntimeError(
            "Captured current cdf_head raw output does not map back to the "
            "model CDF endpoint; this indicates a real ordering/capture error "
            f"(max_abs_error={reconstruction_error:.3e})."
        )

    # A compact row-wise F.linear replay is retained only as a numerical
    # diagnostic. On Ampere GPUs, the original Conv1d and F.linear may use
    # different TF32 kernels, so an ~1e-2 maximum difference is expected and
    # must not invalidate an otherwise exact cache.
    compact_raw_rows = F.linear(
        feature_rows,
        cdf_head_weight[..., 0].to(feature_rows),
        cdf_head_bias.to(feature_rows),
    ).view(num_rows, num_depths, num_thresholds)
    compact_difference = (compact_raw_rows - captured_raw_rows).abs()
    compact_replay_max_abs = float(compact_difference.max().item())
    compact_replay_mean_abs = float(compact_difference.mean().item())
    if compact_replay_max_abs > 5e-2:
        raise RuntimeError(
            "Compact CDF-head replay differs too much from the deployed Conv1d "
            "output. This exceeds the numerical TF32 allowance and suggests a "
            f"real feature/weight mismatch (max_abs_error="
            f"{compact_replay_max_abs:.3e})."
        )
    center_rows = centers.index_select(0, row_center)
    view_rows = views.index_select(0, row_center)
    view_id_rows = view_ids.index_select(0, row_center)
    token_rows = token_ids.index_select(0, row_center)

    angle_rad = row_angle.float() * (np.pi / float(num_angles))
    rotation = batch_viewpoint_params_to_matrix(-view_rows, angle_rad).reshape(
        num_rows, 9
    )
    base_utility = torch.sigmoid(logits_rows).mean(dim=-1)
    width_m = torch.clamp(
        1.2 * width_rows / 10.0,
        min=0.0,
        max=float(getattr(cfgs, "grasp_max_width", 0.1)),
    )
    depth_m = (
        torch.arange(num_depths, device=width_rows.device, dtype=torch.float32)
        + 1.0
    ) * 0.01

    score = base_utility
    height = torch.full_like(score, 0.02)
    depth = depth_m.view(1, num_depths).expand(num_rows, num_depths)
    rotation_rd = rotation[:, None, :].expand(num_rows, num_depths, 9)
    center_rd = center_rows[:, None, :].expand(num_rows, num_depths, 3)
    obj_id = torch.full_like(score, -1.0)
    grasps = torch.cat(
        [
            score.unsqueeze(-1),
            width_m.unsqueeze(-1),
            height.unsqueeze(-1),
            depth.unsqueeze(-1),
            rotation_rd,
            center_rd,
            obj_id.unsqueeze(-1),
        ],
        dim=-1,
    )
    if grasps.shape != (num_rows, num_depths, 17):
        raise RuntimeError(f"Constructed grasp tensor has shape {tuple(grasps.shape)}")

    arrays: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION),
        "checkpoint_sha256": np.asarray(str(checkpoint_sha256)),
        "cdf_increment_bias": np.asarray([float(EA.ea_cdf_increment_bias)], dtype=np.float32),
        # Float32 is mandatory: the untouched current head must be exactly
        # reconstructible from the cache before any optimization.
        "cdf_head_feature": feature_rows.to(torch.float32).cpu().numpy(),
        "base_cdf_logits": logits_rows.to(torch.float32).cpu().numpy(),
        "base_utility": base_utility.to(torch.float32).cpu().numpy(),
        "width_raw": width_rows.to(torch.float32).cpu().numpy(),
        "center_xyz": center_rows.to(torch.float32).cpu().numpy(),
        "view_xyz": view_rows.to(torch.float32).cpu().numpy(),
        "center_id": row_center.to(torch.int16).cpu().numpy(),
        "angle_id": row_angle.to(torch.int8).cpu().numpy(),
        "view_id": view_id_rows.to(torch.int16).cpu().numpy(),
        "token_sel_idx": token_rows.to(torch.int32).cpu().numpy(),
        "num_angles": np.asarray([num_angles], dtype=np.int16),
        "num_depths": np.asarray([num_depths], dtype=np.int16),
        "num_thresholds": np.asarray([num_thresholds], dtype=np.int16),
        "feature_dim": np.asarray([feature_dim], dtype=np.int16),
        # Strict deployed-head-output -> endpoint ordering check.
        "base_reconstruction_max_abs": np.asarray(
            [reconstruction_error], dtype=np.float32
        ),
        # Conv1d-vs-F.linear numerical replay diagnostics only.
        "compact_replay_max_abs": np.asarray(
            [compact_replay_max_abs], dtype=np.float32
        ),
        "compact_replay_mean_abs": np.asarray(
            [compact_replay_mean_abs], dtype=np.float32
        ),
    }
    return arrays, grasps.reshape(-1, 17).float().cpu().numpy(), num_rows, num_depths


_EVALUATOR: Optional[ExactGraspNetActionEvaluator] = None


def _evaluator_worker_init(
    dataset_root: str,
    camera: str,
    dataset_split: str,
    collision_chunk: int,
    fc_mode: str,
    verify_n: int,
    strict: bool,
    threads: int,
) -> None:
    global _EVALUATOR
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    threads = max(1, int(threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _EVALUATOR = ExactGraspNetActionEvaluator(
        dataset_root,
        camera,
        split=str(dataset_split),
        collision_chunk=collision_chunk,
        fc_mode=fc_mode,
        verify_n=verify_n,
        strict=strict,
    )


def _evaluator_worker_task(
    scene_id: int,
    anno_id: int,
    grasps: np.ndarray,
) -> Dict[str, Any]:
    if _EVALUATOR is None:
        raise RuntimeError("Evaluator worker is not initialized")
    started = time.perf_counter()
    result = _EVALUATOR.evaluate(scene_id, anno_id, grasps)
    return {
        "assigned_obj": result.assigned_obj,
        "collision_or_empty": result.collision_or_empty,
        "pure_collision": result.pure_collision,
        "empty": result.empty,
        "friction": result.friction,
        "eval_sec": time.perf_counter() - started,
        "worker_pid": os.getpid(),
        **result.stats,
    }


@dataclass
class PendingFrame:
    scene_id: int
    anno_id: int
    dataset_idx: int
    out_path: str
    arrays: Dict[str, np.ndarray]
    num_rows: int
    num_depths: int
    gpu_sec: float
    submitted_at: float


def _finalize(
    item: PendingFrame,
    payload: Mapping[str, Any],
    summaries: List[Dict[str, Any]],
    start_wall: float,
) -> None:
    expected = item.num_rows * item.num_depths

    def reshape(name: str, dtype) -> np.ndarray:
        value = np.asarray(payload[name], dtype=dtype)
        if value.size != expected:
            raise RuntimeError(
                f"Evaluator field {name} has {value.size} values; expected {expected}."
            )
        return value.reshape(item.num_rows, item.num_depths)

    friction = reshape("friction", np.float32)
    collision = reshape("collision_or_empty", bool)
    pure_collision = reshape("pure_collision", bool)
    empty = reshape("empty", bool)
    assigned_obj = reshape("assigned_obj", np.int64)
    arrays = {
        **item.arrays,
        # Preserve the exact friction-grid value.  In particular, float16
        # represents 1.2 as a value slightly larger than 1.2 and would corrupt
        # the final CDF target when using a <= threshold test.
        "friction": friction.astype(np.float32),
        "collision_or_empty": collision.astype(np.uint8),
        "pure_collision": pure_collision.astype(np.uint8),
        "empty": empty.astype(np.uint8),
        "assigned_obj": assigned_obj.astype(np.int16),
        "scene_id": np.asarray([item.scene_id], dtype=np.int16),
        "anno_id": np.asarray([item.anno_id], dtype=np.int16),
        "dataset_idx": np.asarray([item.dataset_idx], dtype=np.int32),
    }
    save_start = time.perf_counter()
    atomic_save_npz(item.out_path, arrays, compress=bool(EA.ea_compress))
    save_sec = time.perf_counter() - save_start

    thresholds = np.asarray(FRICTION_THRESHOLDS, dtype=np.float32)
    target = (
        (friction[..., None] > 0.0)
        & (friction[..., None] <= thresholds.reshape(1, 1, -1))
    )
    utility = target.mean(axis=-1)
    summary = {
        "scene_id": item.scene_id,
        "anno_id": item.anno_id,
        "num_rows": item.num_rows,
        "num_actions": expected,
        "valid_ratio": float((friction > 0.0).mean()),
        "safe08_ratio": float(((friction > 0.0) & (friction <= 0.8)).mean()),
        "target_utility_mean": float(utility.mean()),
        "collision_or_empty_ratio": float(collision.mean()),
        "pure_collision_ratio": float(pure_collision.mean()),
        "empty_ratio": float(empty.mean()),
        "deployed_reconstruction_max_abs": float(
            np.asarray(item.arrays["base_reconstruction_max_abs"]).reshape(-1)[0]
        ),
        "compact_replay_max_abs": float(
            np.asarray(item.arrays["compact_replay_max_abs"]).reshape(-1)[0]
        ),
        "compact_replay_mean_abs": float(
            np.asarray(item.arrays["compact_replay_mean_abs"]).reshape(-1)[0]
        ),
        "gpu_sec": float(item.gpu_sec),
        "eval_sec": float(payload.get("eval_sec", float("nan"))),
        "collision_sec": float(payload.get("collision_sec", float("nan"))),
        "force_closure_sec": float(payload.get("force_closure_sec", float("nan"))),
        "fc_candidates": int(payload.get("fc_candidates", 0)),
        "fc_quality_calls": int(payload.get("fc_quality_calls", 0)),
        "fc_verify_count": int(payload.get("fc_verify_count", 0)),
        "fc_verify_mismatches": int(payload.get("fc_verify_mismatches", 0)),
        "queue_sec": float(time.perf_counter() - item.submitted_at),
        "save_sec": float(save_sec),
        "evaluator_pid": int(payload.get("worker_pid", -1)),
        "cache_path": item.out_path,
    }
    summaries.append(summary)
    print(
        f"[CACHE] scene={item.scene_id:04d} anno={item.anno_id:04d} "
        f"actions={expected} valid={summary['valid_ratio']:.3f} "
        f"safe08={summary['safe08_ratio']:.3f} col={summary['pure_collision_ratio']:.3f} "
        f"empty={summary['empty_ratio']:.3f} "
        f"maperr={summary['deployed_reconstruction_max_abs']:.1e} "
        f"replay={summary['compact_replay_max_abs']:.2e} "
        f"gpu={item.gpu_sec:.2f}s eval={summary['eval_sec']:.2f}s "
        f"elapsed={(time.time()-start_wall)/60.0:.1f}m",
        flush=True,
    )


def _terminate_executor(executor: Optional[cf.ProcessPoolExecutor]) -> None:
    if executor is None:
        return
    processes = []
    try:
        process_map = getattr(executor, "_processes", None)
        if process_map:
            processes = list(process_map.values())
    except Exception:
        processes = []
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            pass
    for process in processes:
        try:
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
        except Exception:
            pass


def main() -> None:
    if not str(EA.ea_cache_dir).strip():
        raise SystemExit("--ea_cache_dir is required")
    if not cfgs.checkpoint_path:
        raise SystemExit("--checkpoint_path must point to the current Stage-1 CDF checkpoint")
    if not bool(getattr(cfgs, "multi_modal", False)):
        raise RuntimeError("Pass --multi_modal for the current RGB student.")
    if not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("Pass --use_cdf; legacy explicit-angle heads are unsupported.")
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError("Captured/observed-depth models are unsupported. Remove --use_obs_depth.")
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError("Remove the deprecated dataset --use_gt_depth switch.")
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError("The current CDF-only model has no learned collision head.")
    if bool(getattr(cfgs, "use_top4_view_infer", False)):
        raise RuntimeError("Exact-action probe is fixed to the current Top-1 protocol.")
    if int(getattr(cfgs, "batch_size", 1)) != 1:
        raise RuntimeError("Cache mining requires --batch_size 1.")

    _validate_requested_scene_ids_for_split()

    expected_fuse = bool(getattr(cfgs, "use_fuse_depth", False))
    checkpoint, contract = validate_current_stage1_cdf_checkpoint(
        cfgs.checkpoint_path,
        expected_pose_depth_mode=str(EA.ea_expected_pose_depth_mode),
        expected_use_fuse_depth=expected_fuse,
        expected_num_depths=int(getattr(cfgs, "num_depth", 4)),
        expected_num_thresholds=len(FRICTION_THRESHOLDS),
    )

    seed = int(EA.ea_seed)
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
        split=str(EA.ea_dataset_split),
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
        extend_angle=False,
    )
    indices = _select_indices(dataset)
    selected_dataset = DeterministicIndexDataset(dataset, indices, seed)
    num_workers = max(0, int(getattr(cfgs, "num_workers", 2)))
    loader = DataLoader(
        selected_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
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
        pose_depth_mode=contract.pose_depth_mode,
        camera_pose_key=str(checkpoint.get("camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(checkpoint.get("camera_gravity_key", "camera_gravity_vec")),
        pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(checkpoint.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(checkpoint.get("ray_gravity_mid_dim", 32)),
        use_cdf=True,
        vis_dir=None,
    ).to(device)
    load_model_state_strict(model, checkpoint["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    decoder = resolve_current_cdf_decoder(model)
    decoder_increment_bias = float(getattr(decoder, "cdf_increment_bias", float("nan")))
    if not np.isfinite(decoder_increment_bias) or abs(
        decoder_increment_bias - float(EA.ea_cdf_increment_bias)
    ) > 1e-8:
        raise RuntimeError(
            "Cache cdf_increment_bias must exactly match the current decoder: "
            f"decoder={decoder_increment_bias}, requested={EA.ea_cdf_increment_bias}."
        )
    capture = _CdfHeadIoCapture(decoder.cdf_head)

    output_root = os.path.abspath(EA.ea_cache_dir)
    os.makedirs(output_root, exist_ok=True)
    config = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "worker_tag": str(EA.ea_worker_tag),
        "dataset_root": os.path.abspath(cfgs.dataset_root),
        "camera": str(cfgs.camera),
        "dataset_split": str(EA.ea_dataset_split),
        "selected_scene_ids": _csv_ints(EA.ea_scene_ids),
        "selected_anno_ids": _csv_ints(EA.ea_anno_ids),
        "sample_interval": float(EA.ea_sample_interval),
        "num_selected_frames": len(indices),
        "top_centers": int(EA.ea_top_centers),
        "random_centers": int(EA.ea_random_centers),
        "all_angles": True,
        "top_views": 1,
        "num_depths": contract.num_depths,
        "num_thresholds": contract.num_thresholds,
        "feature_dim": contract.feature_dim,
        "cache_float_dtype": "float32",
        "cdf_increment_bias": float(EA.ea_cdf_increment_bias),
        "fc_mode": str(EA.ea_fc_mode),
        "fc_verify_n": int(EA.ea_fc_verify_n),
        "checkpoint_contract": contract.to_dict(),
        "graspness_mode": str(cfgs.graspness_mode),
    }
    atomic_save_json(
        config,
        os.path.join(output_root, f"_mine_config_{EA.ea_worker_tag}.json"),
    )

    eval_workers = max(0, int(EA.ea_eval_workers))
    executor: Optional[cf.ProcessPoolExecutor] = None
    serial_evaluator: Optional[ExactGraspNetActionEvaluator] = None
    if eval_workers > 0:
        context = mp.get_context("spawn")
        executor = cf.ProcessPoolExecutor(
            max_workers=eval_workers,
            mp_context=context,
            initializer=_evaluator_worker_init,
            initargs=(
                cfgs.dataset_root,
                cfgs.camera,
                str(EA.ea_dataset_split),
                int(EA.ea_collision_chunk),
                str(EA.ea_fc_mode),
                int(EA.ea_fc_verify_n),
                bool(EA.ea_strict),
                int(EA.ea_eval_threads),
            ),
        )
    else:
        serial_evaluator = ExactGraspNetActionEvaluator(
            cfgs.dataset_root,
            cfgs.camera,
            split=str(EA.ea_dataset_split),
            collision_chunk=int(EA.ea_collision_chunk),
            fc_mode=str(EA.ea_fc_mode),
            verify_n=int(EA.ea_fc_verify_n),
            strict=bool(EA.ea_strict),
        )

    max_pending = max(1, int(EA.ea_max_pending))
    pending: Dict[cf.Future, PendingFrame] = {}
    summaries: List[Dict[str, Any]] = []
    start_wall = time.time()
    first_contract_check = True

    def drain(block: bool) -> None:
        if not pending:
            return
        done, _ = cf.wait(
            list(pending.keys()),
            timeout=None if block else 0.0,
            return_when=cf.FIRST_COMPLETED,
        )
        for future in done:
            item = pending.pop(future)
            _finalize(item, future.result(), summaries, start_wall)

    try:
        for local_index, batch in enumerate(loader):
            while executor is not None and len(pending) >= max_pending:
                drain(block=True)

            dataset_index = int(indices[local_index])
            scene_name = dataset.scene_list()[dataset_index]
            scene_id = int(str(scene_name).split("_")[-1])
            anno_id = int(dataset_index % 256)
            out_path = os.path.join(
                output_root,
                f"scene_{scene_id:04d}",
                f"ann_{anno_id:04d}.npz",
            )
            if os.path.isfile(out_path) and not bool(EA.ea_overwrite):
                print(f"[SKIP] {out_path}", flush=True)
                continue

            gpu_start = time.perf_counter()
            batch = _move_current_inputs(batch, device)
            capture.reset()
            with torch.inference_mode():
                end_points = model(batch)
            head_input, head_output = capture.pop()
            if first_contract_check:
                _assert_predicted_depth_top1(end_points)
                if int(head_input.shape[1]) != contract.feature_dim:
                    raise RuntimeError(
                        f"Runtime CDF feature dim={head_input.shape[1]}, checkpoint "
                        f"contract={contract.feature_dim}."
                    )
                first_contract_check = False

            rng = np.random.default_rng(seed + scene_id * 1009 + anno_id * 9176)
            center_ids = _choose_centers(
                end_points["grasp_cdf_pred_angle_depth"][0], rng
            )
            arrays, grasps, num_rows, num_depths = _build_rows_and_grasps(
                end_points,
                head_input,
                head_output,
                center_ids,
                contract.checkpoint_sha256,
                decoder.cdf_head.weight.detach(),
                decoder.cdf_head.bias.detach(),
            )
            gpu_sec = time.perf_counter() - gpu_start
            item = PendingFrame(
                scene_id=scene_id,
                anno_id=anno_id,
                dataset_idx=dataset_index,
                out_path=out_path,
                arrays=arrays,
                num_rows=num_rows,
                num_depths=num_depths,
                gpu_sec=gpu_sec,
                submitted_at=time.perf_counter(),
            )

            if executor is None:
                assert serial_evaluator is not None
                eval_start = time.perf_counter()
                result = serial_evaluator.evaluate(scene_id, anno_id, grasps)
                payload = {
                    "assigned_obj": result.assigned_obj,
                    "collision_or_empty": result.collision_or_empty,
                    "pure_collision": result.pure_collision,
                    "empty": result.empty,
                    "friction": result.friction,
                    "eval_sec": time.perf_counter() - eval_start,
                    "worker_pid": os.getpid(),
                    **result.stats,
                }
                _finalize(item, payload, summaries, start_wall)
            else:
                future = executor.submit(
                    _evaluator_worker_task, scene_id, anno_id, grasps
                )
                pending[future] = item
                drain(block=False)

            del batch, end_points, head_input, head_output, arrays, grasps

        while pending:
            drain(block=True)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            executor = None
    except BaseException:
        _terminate_executor(executor)
        executor = None
        raise
    finally:
        capture.close()

    elapsed = time.time() - start_wall
    completion = {
        **config,
        "status": "complete",
        "num_new_files": len(summaries),
        "elapsed_sec": elapsed,
        "mean_gpu_sec": float(np.mean([row["gpu_sec"] for row in summaries]))
        if summaries
        else None,
        "mean_eval_sec": float(np.mean([row["eval_sec"] for row in summaries]))
        if summaries
        else None,
        "summaries": summaries,
    }
    atomic_save_json(
        completion,
        os.path.join(output_root, f"_mine_complete_{EA.ea_worker_tag}.json"),
    )
    print(
        f"[DONE] worker={EA.ea_worker_tag} wrote {len(summaries)} files to "
        f"{output_root} in {elapsed/60.0:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
