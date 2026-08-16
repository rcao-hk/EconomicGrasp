"""CVA inference with pose-stratified metric-depth diagnostics.

This script preserves the strict grasp inference / saving behavior of
``inference_cva.py`` and additionally exports the depth diagnostics needed to
verify whether pose-aware depth resolves view-dependent metric-depth errors.

Diagnostics
-----------
1. Dense valid-pixel depth MAE, signed bias, RMSE, and P90.
2. Sampled point-aligned depth metrics for all/object/table regions.
3. Dense depth-gradient error and surface-normal angular error.
4. Grasp-center z error, obtained by projecting ``xyz_graspable`` to GT depth.
5. Summaries by camera tilt, annotation id, and explicitly anno 0 / anno 160.

The current test dataloader returns sampled segmentation labels rather than a
full-resolution segmentation map. Therefore object/table metrics are computed
on the 20k sampled point-aligned pixels used by the grasp pipeline. Table
samples are background points whose GT 3D location lies within
``TABLE_PLANE_TOL_M`` of z=0 in the GraspNet table frame.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from graspnetAPI import GraspGroup
from torch.utils.data import DataLoader, Subset

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_bip3d import (
    economicgrasp_dpt,
    pred_decode_center_view_angle,
)
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch


# -----------------------------------------------------------------------------
# Diagnostic protocol. Keep these fixed across no-pose / pose-aware runs.
# -----------------------------------------------------------------------------
ANNOS_PER_SCENE = 256
FORCE_INCLUDE_ANNOS = (0, 160)
TILT_EDGES_DEG = (20.0, 30.0, 40.0)
TABLE_PLANE_TOL_M = 0.015
NORMAL_DOWNSAMPLE = 4
DIAG_SUFFIX = "_pose_depth_diag"

DEPTH_PREFIXES = (
    "dense",
    "sampled_all",
    "object",
    "table",
    "grasp_center",
)

PER_SAMPLE_FIELDS = [
    "data_idx",
    "scene_name",
    "scene_idx",
    "anno_idx",
    "tilt_deg",
    "tilt_bin",
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_film_gamma_abs_mean",
    "pose_film_beta_abs_mean",
    "depth_valid_ratio",
]
for _prefix in DEPTH_PREFIXES:
    PER_SAMPLE_FIELDS.extend(
        [
            f"{_prefix}_count",
            f"{_prefix}_sum_abs",
            f"{_prefix}_sum_signed",
            f"{_prefix}_sum_sq",
            f"{_prefix}_mae_m",
            f"{_prefix}_bias_m",
            f"{_prefix}_rmse_m",
            f"{_prefix}_p90_m",
            f"{_prefix}_gt_mean_m",
            f"{_prefix}_pred_mean_m",
        ]
    )
PER_SAMPLE_FIELDS.extend(
    [
        "table_mask_ratio",
        "table_height_abs_mean_m",
        "grad_count",
        "grad_sum_abs",
        "grad_mae_m_per_px",
        "grad_p90_m_per_px",
        "normal_count",
        "normal_sum_deg",
        "normal_mean_deg",
        "normal_p90_deg",
    ]
)


def _worker_init(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _build_subset(
    dataset,
    sample_interval: float,
    annos_per_scene: int = ANNOS_PER_SCENE,
    force_include_annos: Sequence[int] = FORCE_INCLUDE_ANNOS,
) -> Tuple[torch.utils.data.Dataset, List[int]]:
    """Build a deterministic per-scene subset and always retain key views."""
    if sample_interval <= 0:
        raise ValueError(
            f"sample_interval must be positive, got {sample_interval}."
        )

    total = len(dataset)
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices: List[int] = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        local = set(range(start, end, stride))
        for anno in force_include_annos:
            idx = start + int(anno)
            if start <= idx < end:
                local.add(idx)
        indices.extend(sorted(local))

    return Subset(dataset, indices), indices


def _move_fixed_inputs(batch, device):
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Inference received unexpected list-valued key '{key}'. "
                "The test dataset must be constructed with load_label=False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def _load_checkpoint_strict(model, checkpoint_path: str) -> None:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"CVA model checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Strict CVA checkpoint loading produced missing/unexpected keys: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def _as_b1hw(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        x = x[:, :1]
    else:
        raise ValueError(f"Unexpected {name} shape: {tuple(x.shape)}")
    return x


def _finite_float(x, default: float = float("nan")) -> float:
    if x is None:
        return default
    if torch.is_tensor(x):
        if x.numel() == 0:
            return default
        x = x.detach().float().mean().cpu().item()
    try:
        value = float(x)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float(), q).detach().cpu().item())


def _empty_depth_stats(prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_count": 0,
        f"{prefix}_sum_abs": 0.0,
        f"{prefix}_sum_signed": 0.0,
        f"{prefix}_sum_sq": 0.0,
        f"{prefix}_mae_m": float("nan"),
        f"{prefix}_bias_m": float("nan"),
        f"{prefix}_rmse_m": float("nan"),
        f"{prefix}_p90_m": float("nan"),
        f"{prefix}_gt_mean_m": float("nan"),
        f"{prefix}_pred_mean_m": float("nan"),
    }


def _depth_stats(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    """Compute additive accumulators plus human-readable depth statistics."""
    mask = mask.bool() & torch.isfinite(pred) & torch.isfinite(gt)
    if not bool(mask.any()):
        return _empty_depth_stats(prefix)

    pred_v = pred[mask].float()
    gt_v = gt[mask].float()
    signed = pred_v - gt_v
    abs_err = signed.abs()
    sq_err = signed.square()
    count = int(abs_err.numel())

    return {
        f"{prefix}_count": count,
        f"{prefix}_sum_abs": float(abs_err.sum().detach().cpu().item()),
        f"{prefix}_sum_signed": float(signed.sum().detach().cpu().item()),
        f"{prefix}_sum_sq": float(sq_err.sum().detach().cpu().item()),
        f"{prefix}_mae_m": float(abs_err.mean().detach().cpu().item()),
        f"{prefix}_bias_m": float(signed.mean().detach().cpu().item()),
        f"{prefix}_rmse_m": float(sq_err.mean().sqrt().detach().cpu().item()),
        f"{prefix}_p90_m": _quantile(abs_err, 0.90),
        f"{prefix}_gt_mean_m": float(gt_v.mean().detach().cpu().item()),
        f"{prefix}_pred_mean_m": float(pred_v.mean().detach().cpu().item()),
    }


def _tilt_bin_label(tilt_deg: float) -> str:
    e0, e1, e2 = TILT_EDGES_DEG
    if tilt_deg < e0:
        return f"lt_{e0:g}"
    if tilt_deg < e1:
        return f"{e0:g}_{e1:g}"
    if tilt_deg < e2:
        return f"{e1:g}_{e2:g}"
    return f"ge_{e2:g}"


def _backproject_indexed(
    depth_values: torch.Tensor,
    flat_indices: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Backproject indexed z-depth samples to camera-frame XYZ."""
    idx = flat_indices.long().clamp(0, height * width - 1)
    z = depth_values.float()
    u = (idx % width).float()
    v = (idx // width).float()

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)


class _CameraGeometryCache:
    """Cache per-scene camera trajectories and table alignment matrices."""

    def __init__(self, root: str, camera: str):
        self.root = root
        self.camera = camera
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def table_from_camera(
        self,
        scene_name: str,
        anno_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if scene_name not in self._cache:
            base = os.path.join(
                self.root,
                "scenes",
                scene_name,
                self.camera,
            )
            poses_path = os.path.join(base, "camera_poses.npy")
            align_path = os.path.join(base, "cam0_wrt_table.npy")
            if not os.path.isfile(poses_path):
                raise FileNotFoundError(poses_path)
            if not os.path.isfile(align_path):
                raise FileNotFoundError(align_path)
            camera_poses = np.load(poses_path).astype(np.float32)
            cam0_wrt_table = np.load(align_path).astype(np.float32)
            if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
                raise ValueError(
                    f"Unexpected camera_poses shape {camera_poses.shape}: "
                    f"{poses_path}"
                )
            if cam0_wrt_table.shape != (4, 4):
                raise ValueError(
                    f"Unexpected cam0_wrt_table shape {cam0_wrt_table.shape}: "
                    f"{align_path}"
                )
            self._cache[scene_name] = (camera_poses, cam0_wrt_table)

        camera_poses, cam0_wrt_table = self._cache[scene_name]
        if not 0 <= int(anno_idx) < camera_poses.shape[0]:
            raise IndexError(
                f"anno_idx={anno_idx} outside camera trajectory "
                f"[0,{camera_poses.shape[0]}) for {scene_name}."
            )
        transform = cam0_wrt_table @ camera_poses[int(anno_idx)]
        return torch.as_tensor(transform, device=device, dtype=dtype)


def _sampled_region_stats(
    pred_hw: torch.Tensor,
    gt_hw: torch.Tensor,
    valid_hw: torch.Tensor,
    img_idxs: torch.Tensor,
    seg_sampled: torch.Tensor,
    K: torch.Tensor,
    table_from_camera: torch.Tensor,
) -> Dict[str, float]:
    """Compute point-aligned all/object/table depth diagnostics."""
    H, W = pred_hw.shape
    idx = img_idxs.long().clamp(0, H * W - 1)
    pred_s = pred_hw.reshape(-1).gather(0, idx)
    gt_s = gt_hw.reshape(-1).gather(0, idx)
    valid_s = valid_hw.reshape(-1).gather(0, idx)

    seg_s = seg_sampled.reshape(-1)
    if seg_s.numel() != idx.numel():
        raise ValueError(
            "seg/img_idxs length mismatch: "
            f"seg={seg_s.numel()}, idx={idx.numel()}"
        )

    output: Dict[str, float] = {}
    output.update(
        _depth_stats(pred_s, gt_s, valid_s, prefix="sampled_all")
    )

    object_mask = valid_s & (seg_s > 0)
    output.update(
        _depth_stats(pred_s, gt_s, object_mask, prefix="object")
    )

    xyz_gt_cam = _backproject_indexed(gt_s, idx, K, H, W)
    R = table_from_camera[:3, :3]
    t = table_from_camera[:3, 3]
    xyz_gt_table = xyz_gt_cam @ R.transpose(0, 1) + t
    table_height = xyz_gt_table[:, 2]

    background_mask = seg_s <= 0
    table_mask = (
        valid_s
        & background_mask
        & torch.isfinite(table_height)
        & (table_height.abs() <= TABLE_PLANE_TOL_M)
    )
    output.update(
        _depth_stats(pred_s, gt_s, table_mask, prefix="table")
    )

    background_valid_count = int((valid_s & background_mask).sum().item())
    table_count = int(table_mask.sum().item())
    output["table_mask_ratio"] = (
        float(table_count / background_valid_count)
        if background_valid_count > 0
        else float("nan")
    )
    output["table_height_abs_mean_m"] = (
        float(table_height[table_mask].abs().mean().detach().cpu().item())
        if table_count > 0
        else float("nan")
    )
    return output


def _gradient_stats(
    pred_hw: torch.Tensor,
    gt_hw: torch.Tensor,
    valid_hw: torch.Tensor,
) -> Dict[str, float]:
    dx_mask = valid_hw[:, 1:] & valid_hw[:, :-1]
    dy_mask = valid_hw[1:, :] & valid_hw[:-1, :]

    dx_err = (
        (pred_hw[:, 1:] - pred_hw[:, :-1])
        - (gt_hw[:, 1:] - gt_hw[:, :-1])
    ).abs()
    dy_err = (
        (pred_hw[1:, :] - pred_hw[:-1, :])
        - (gt_hw[1:, :] - gt_hw[:-1, :])
    ).abs()

    values = []
    if bool(dx_mask.any()):
        values.append(dx_err[dx_mask])
    if bool(dy_mask.any()):
        values.append(dy_err[dy_mask])
    if not values:
        return {
            "grad_count": 0,
            "grad_sum_abs": 0.0,
            "grad_mae_m_per_px": float("nan"),
            "grad_p90_m_per_px": float("nan"),
        }

    err = torch.cat(values).float()
    return {
        "grad_count": int(err.numel()),
        "grad_sum_abs": float(err.sum().detach().cpu().item()),
        "grad_mae_m_per_px": float(err.mean().detach().cpu().item()),
        "grad_p90_m_per_px": _quantile(err, 0.90),
    }


def _dense_xyz(depth_hw: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    H, W = depth_hw.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=depth_hw.device, dtype=depth_hw.dtype),
        torch.arange(W, device=depth_hw.device, dtype=depth_hw.dtype),
        indexing="ij",
    )
    z = depth_hw
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    return torch.stack([x, y, z], dim=-1)


def _normal_stats(
    pred_hw: torch.Tensor,
    gt_hw: torch.Tensor,
    valid_hw: torch.Tensor,
    K: torch.Tensor,
    downsample: int = NORMAL_DOWNSAMPLE,
) -> Dict[str, float]:
    if downsample < 1:
        raise ValueError(f"downsample must be >=1, got {downsample}")

    pred = pred_hw[::downsample, ::downsample]
    gt = gt_hw[::downsample, ::downsample]
    valid = valid_hw[::downsample, ::downsample]

    K_s = K.clone()
    K_s[0, 0] /= downsample
    K_s[1, 1] /= downsample
    K_s[0, 2] /= downsample
    K_s[1, 2] /= downsample

    if pred.shape[0] < 3 or pred.shape[1] < 3:
        return {
            "normal_count": 0,
            "normal_sum_deg": 0.0,
            "normal_mean_deg": float("nan"),
            "normal_p90_deg": float("nan"),
        }

    xyz_pred = _dense_xyz(pred, K_s)
    xyz_gt = _dense_xyz(gt, K_s)

    vx_pred = xyz_pred[1:-1, 2:, :] - xyz_pred[1:-1, :-2, :]
    vy_pred = xyz_pred[2:, 1:-1, :] - xyz_pred[:-2, 1:-1, :]
    vx_gt = xyz_gt[1:-1, 2:, :] - xyz_gt[1:-1, :-2, :]
    vy_gt = xyz_gt[2:, 1:-1, :] - xyz_gt[:-2, 1:-1, :]

    n_pred = torch.cross(vx_pred, vy_pred, dim=-1)
    n_gt = torch.cross(vx_gt, vy_gt, dim=-1)
    pred_norm = torch.linalg.norm(n_pred, dim=-1)
    gt_norm = torch.linalg.norm(n_gt, dim=-1)

    neighborhood_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
        & torch.isfinite(pred_norm)
        & torch.isfinite(gt_norm)
        & (pred_norm > 1e-8)
        & (gt_norm > 1e-8)
    )
    if not bool(neighborhood_valid.any()):
        return {
            "normal_count": 0,
            "normal_sum_deg": 0.0,
            "normal_mean_deg": float("nan"),
            "normal_p90_deg": float("nan"),
        }

    n_pred = n_pred / pred_norm.clamp_min(1e-8).unsqueeze(-1)
    n_gt = n_gt / gt_norm.clamp_min(1e-8).unsqueeze(-1)
    # Surface normals have a sign ambiguity; compare orientation using |dot|.
    dot = (n_pred * n_gt).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(dot))[neighborhood_valid].float()

    return {
        "normal_count": int(angle_deg.numel()),
        "normal_sum_deg": float(angle_deg.sum().detach().cpu().item()),
        "normal_mean_deg": float(angle_deg.mean().detach().cpu().item()),
        "normal_p90_deg": _quantile(angle_deg, 0.90),
    }


def _grasp_center_stats(
    end_points: Mapping[str, torch.Tensor],
    sample_i: int,
    K: torch.Tensor,
    gt_hw: torch.Tensor,
    valid_hw: torch.Tensor,
) -> Dict[str, float]:
    centers_all = end_points.get("xyz_graspable", None)
    if centers_all is None:
        return _empty_depth_stats("grasp_center")

    centers = centers_all[sample_i]
    if centers.dim() != 2:
        raise ValueError(
            f"Unexpected xyz_graspable sample shape: {tuple(centers.shape)}"
        )
    if centers.shape[-1] == 3:
        pass
    elif centers.shape[0] == 3:
        centers = centers.transpose(0, 1)
    else:
        raise ValueError(
            f"Cannot identify xyz_graspable XYZ dimension: {tuple(centers.shape)}"
        )

    centers = centers.float()
    z = centers[:, 2]
    finite = torch.isfinite(centers).all(dim=1) & (z > 1e-6)
    u = torch.round(K[0, 0] * centers[:, 0] / z + K[0, 2]).long()
    v = torch.round(K[1, 1] * centers[:, 1] / z + K[1, 2]).long()
    H, W = gt_hw.shape
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    safe_u = u.clamp(0, W - 1)
    safe_v = v.clamp(0, H - 1)
    gt_z = gt_hw[safe_v, safe_u]
    gt_valid = valid_hw[safe_v, safe_u]
    mask = finite & inside & gt_valid
    return _depth_stats(z, gt_z, mask, prefix="grasp_center")


def _numeric_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def _numeric_std(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(clean)) if clean else float("nan")


def _summarize_rows(
    rows: Sequence[Mapping[str, object]],
    group_name: str,
    group_value: object,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "group_name": group_name,
        "group_value": group_value,
        "num_samples": len(rows),
        "tilt_deg_mean": _numeric_mean(float(r["tilt_deg"]) for r in rows),
        "tilt_deg_std": _numeric_std(float(r["tilt_deg"]) for r in rows),
    }

    for prefix in DEPTH_PREFIXES:
        count = sum(int(r[f"{prefix}_count"]) for r in rows)
        sum_abs = sum(float(r[f"{prefix}_sum_abs"]) for r in rows)
        sum_signed = sum(float(r[f"{prefix}_sum_signed"]) for r in rows)
        sum_sq = sum(float(r[f"{prefix}_sum_sq"]) for r in rows)
        out[f"{prefix}_count"] = count
        out[f"{prefix}_pooled_mae_m"] = (
            sum_abs / count if count > 0 else float("nan")
        )
        out[f"{prefix}_pooled_bias_m"] = (
            sum_signed / count if count > 0 else float("nan")
        )
        out[f"{prefix}_pooled_rmse_m"] = (
            math.sqrt(sum_sq / count) if count > 0 else float("nan")
        )
        out[f"{prefix}_mean_sample_p90_m"] = _numeric_mean(
            float(r[f"{prefix}_p90_m"]) for r in rows
        )

    grad_count = sum(int(r["grad_count"]) for r in rows)
    grad_sum = sum(float(r["grad_sum_abs"]) for r in rows)
    out["grad_count"] = grad_count
    out["grad_pooled_mae_m_per_px"] = (
        grad_sum / grad_count if grad_count > 0 else float("nan")
    )
    out["grad_mean_sample_p90_m_per_px"] = _numeric_mean(
        float(r["grad_p90_m_per_px"]) for r in rows
    )

    normal_count = sum(int(r["normal_count"]) for r in rows)
    normal_sum = sum(float(r["normal_sum_deg"]) for r in rows)
    out["normal_count"] = normal_count
    out["normal_pooled_mean_deg"] = (
        normal_sum / normal_count if normal_count > 0 else float("nan")
    )
    out["normal_mean_sample_p90_deg"] = _numeric_mean(
        float(r["normal_p90_deg"]) for r in rows
    )

    out["pose_film_gamma_abs_mean"] = _numeric_mean(
        float(r["pose_film_gamma_abs_mean"]) for r in rows
    )
    out["pose_film_beta_abs_mean"] = _numeric_mean(
        float(r["pose_film_beta_abs_mean"]) for r in rows
    )
    out["table_mask_ratio_mean"] = _numeric_mean(
        float(r["table_mask_ratio"]) for r in rows
    )
    return out


def _write_rows_csv(
    path: str,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _export_summaries(
    diag_dir: str,
    rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> None:
    by_tilt: MutableMapping[str, List[Mapping[str, object]]] = defaultdict(list)
    by_anno: MutableMapping[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_tilt[str(row["tilt_bin"])].append(row)
        by_anno[int(row["anno_idx"])].append(row)

    tilt_order = [
        f"lt_{TILT_EDGES_DEG[0]:g}",
        f"{TILT_EDGES_DEG[0]:g}_{TILT_EDGES_DEG[1]:g}",
        f"{TILT_EDGES_DEG[1]:g}_{TILT_EDGES_DEG[2]:g}",
        f"ge_{TILT_EDGES_DEG[2]:g}",
    ]
    tilt_rows = [
        _summarize_rows(by_tilt[label], "tilt_bin", label)
        for label in tilt_order
        if label in by_tilt
    ]
    anno_rows = [
        _summarize_rows(by_anno[anno], "anno_idx", anno)
        for anno in sorted(by_anno)
    ]
    selected_anno_rows = [
        _summarize_rows(by_anno[anno], "anno_idx", anno)
        for anno in FORCE_INCLUDE_ANNOS
        if anno in by_anno
    ]
    overall = _summarize_rows(rows, "overall", "all") if rows else {}

    _write_rows_csv(
        os.path.join(diag_dir, "pose_depth_by_tilt.csv"),
        tilt_rows,
    )
    _write_rows_csv(
        os.path.join(diag_dir, "pose_depth_by_anno.csv"),
        anno_rows,
    )
    _write_rows_csv(
        os.path.join(diag_dir, "pose_depth_anno0_anno160.csv"),
        selected_anno_rows,
    )

    payload = {
        "manifest": dict(manifest),
        "overall": overall,
        "tilt_bins": tilt_rows,
        "selected_annos": selected_anno_rows,
    }
    with open(
        os.path.join(diag_dir, "pose_depth_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def _batch_sample_row(
    *,
    batch: Mapping[str, torch.Tensor],
    end_points: Mapping[str, torch.Tensor],
    sample_i: int,
    data_idx: int,
    scene_name: str,
    geometry_cache: _CameraGeometryCache,
) -> Dict[str, object]:
    pred = _as_b1hw(end_points["depth_map_pred"], "depth_map_pred")
    gt = _as_b1hw(batch["gt_depth_m"], "gt_depth_m")

    pred_hw = pred[sample_i, 0].float()
    gt_hw = gt[sample_i, 0].float()
    valid_hw = (
        torch.isfinite(gt_hw)
        & (gt_hw >= float(cfgs.min_depth))
        & (gt_hw <= float(cfgs.max_depth))
        & torch.isfinite(pred_hw)
    )

    pose = batch["camera_pose_vec"][sample_i].float().reshape(-1)
    if pose.numel() != 3:
        raise ValueError(
            f"camera_pose_vec must have 3 values, got {tuple(pose.shape)}"
        )
    pose = pose / torch.linalg.norm(pose).clamp_min(1e-8)

    if "camera_tilt_deg" in batch:
        tilt_deg = float(
            batch["camera_tilt_deg"][sample_i].detach().cpu().item()
        )
    else:
        tilt_deg = float(
            torch.rad2deg(
                torch.acos(pose[2].clamp(-1.0, 1.0))
            ).detach().cpu().item()
        )

    scene_idx = int(batch["scene_idx"][sample_i].detach().cpu().item())
    anno_idx = int(batch["anno_idx"][sample_i].detach().cpu().item())
    K = batch["K"][sample_i].float()

    row: Dict[str, object] = {
        "data_idx": int(data_idx),
        "scene_name": scene_name,
        "scene_idx": scene_idx,
        "anno_idx": anno_idx,
        "tilt_deg": tilt_deg,
        "tilt_bin": _tilt_bin_label(tilt_deg),
        "pose_x": float(pose[0].detach().cpu().item()),
        "pose_y": float(pose[1].detach().cpu().item()),
        "pose_z": float(pose[2].detach().cpu().item()),
        "pose_film_gamma_abs_mean": _finite_float(
            end_points.get("pose_film_gamma_abs_mean", None)
        ),
        "pose_film_beta_abs_mean": _finite_float(
            end_points.get("pose_film_beta_abs_mean", None)
        ),
        "depth_valid_ratio": float(valid_hw.float().mean().cpu().item()),
    }

    row.update(_depth_stats(pred_hw, gt_hw, valid_hw, prefix="dense"))

    table_from_camera = geometry_cache.table_from_camera(
        scene_name,
        anno_idx,
        device=pred_hw.device,
        dtype=pred_hw.dtype,
    )
    row.update(
        _sampled_region_stats(
            pred_hw=pred_hw,
            gt_hw=gt_hw,
            valid_hw=valid_hw,
            img_idxs=batch["img_idxs"][sample_i],
            seg_sampled=batch["seg"][sample_i],
            K=K,
            table_from_camera=table_from_camera,
        )
    )
    row.update(_gradient_stats(pred_hw, gt_hw, valid_hw))
    row.update(_normal_stats(pred_hw, gt_hw, valid_hw, K))
    row.update(_grasp_center_stats(end_points, sample_i, K, gt_hw, valid_hw))

    # Ensure stable CSV schema even when a metric is missing.
    for field in PER_SAMPLE_FIELDS:
        row.setdefault(field, "")
    return row


def inference() -> None:
    if not cfgs.multi_modal:
        raise RuntimeError("CVA inference requires --multi_modal.")
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError(
            "This CVA configuration has no learned collision head. Remove "
            "--kview_use_collision."
        )
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required for inference.")
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required for inference.")

    use_obs_depth = bool(getattr(cfgs, "use_obs_depth", False))
    pose_depth_mode = getattr(cfgs, "pose_depth_mode", None)
    pose_depth_mode = str(pose_depth_mode)
    if use_obs_depth and pose_depth_mode != "none":
        raise RuntimeError(
            f"pose_depth_mode={pose_depth_mode!r} and observed-depth "
            "refinement are mutually exclusive."
        )

    os.makedirs(cfgs.save_dir, exist_ok=True)
    diag_dir = cfgs.save_dir.rstrip(os.sep) + DIAG_SUFFIX
    os.makedirs(diag_dir, exist_ok=True)
    per_sample_path = os.path.join(diag_dir, "pose_depth_per_sample.csv")

    full_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        split=cfgs.test_mode,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    eval_dataset, sampled_indices = _build_subset(
        full_dataset,
        float(getattr(cfgs, "sample_interval", 1.0)),
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=cfgs.num_workers,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(cfgs.num_workers > 0),
    )
    scene_list = full_dataset.scene_list()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cdf = bool(getattr(cfgs, "use_cdf", False))
    use_top4 = bool(getattr(cfgs, "use_top4_view_infer", False))

    print(f"[INFER] total={len(full_dataset)} selected={len(eval_dataset)}")
    print(
        f"[INFER] cdf={int(use_cdf)} top4={int(use_top4)} "
        f"batch={cfgs.batch_size} observed_depth={int(use_obs_depth)} "
        f"pose_depth_mode={pose_depth_mode}"
    )
    print(
        f"[DEPTH-DIAG] output={diag_dir} "
        f"tilt_edges={TILT_EDGES_DEG} "
        f"force_annos={FORCE_INCLUDE_ANNOS} "
        f"table_tol={TABLE_PLANE_TOL_M:.3f}m"
    )

    model = economicgrasp_dpt(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=use_obs_depth,
        pose_depth_mode=pose_depth_mode,
        camera_pose_key=str(
            getattr(cfgs, "camera_pose_key", "camera_pose_vec")
        ),
        camera_gravity_key=str(
            getattr(cfgs, "camera_gravity_key", "camera_gravity_vec")
        ),
        pose_hidden_dim=int(getattr(cfgs, "pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(
            getattr(cfgs, "ray_gravity_hidden_dim", 64)
        ),
        ray_gravity_mid_dim=int(
            getattr(cfgs, "ray_gravity_mid_dim", 32)
        ),
        use_cdf=use_cdf,
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    ).to(device)
    _load_checkpoint_strict(model, cfgs.checkpoint_path)
    model.eval()

    geometry_cache = _CameraGeometryCache(cfgs.dataset_root, cfgs.camera)
    manifest = {
        "checkpoint_path": cfgs.checkpoint_path,
        "dataset_root": cfgs.dataset_root,
        "test_mode": cfgs.test_mode,
        "camera": cfgs.camera,
        "num_selected_samples": len(eval_dataset),
        "sample_interval": float(getattr(cfgs, "sample_interval", 1.0)),
        "pose_depth_mode": pose_depth_mode,
        "use_obs_depth": use_obs_depth,
        "use_cdf": use_cdf,
        "use_top4_view_infer": use_top4,
        "tilt_edges_deg": list(TILT_EDGES_DEG),
        "force_include_annos": list(FORCE_INCLUDE_ANNOS),
        "table_plane_tolerance_m": TABLE_PLANE_TOL_M,
        "normal_downsample": NORMAL_DOWNSAMPLE,
        "object_table_metric_domain": "sampled_point_aligned_pixels",
    }
    with open(
        os.path.join(diag_dir, "pose_depth_manifest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    rows: List[Dict[str, object]] = []
    start = time.perf_counter()
    processed = 0

    with open(per_sample_path, "w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.DictWriter(csv_f, fieldnames=PER_SAMPLE_FIELDS)
        csv_writer.writeheader()

        for batch_idx, batch in enumerate(dataloader):
            batch = _move_fixed_inputs(batch, device)
            batch["cva_export_angle_feature"] = False

            with torch.inference_mode():
                end_points = model(batch)
                if "depth_map_pred" not in end_points:
                    raise KeyError(
                        "Model output lacks end_points['depth_map_pred']; "
                        "pose-stratified depth diagnostics cannot run."
                    )
                grasp_preds = pred_decode_center_view_angle(
                    end_points,
                    use_cdf=use_cdf,
                )

            for sample_i, pred in enumerate(grasp_preds):
                subset_idx = batch_idx * cfgs.batch_size + sample_i
                if subset_idx >= len(sampled_indices):
                    raise IndexError(
                        f"Subset index {subset_idx} exceeds "
                        f"{len(sampled_indices)}."
                    )
                data_idx = sampled_indices[subset_idx]
                scene_name = scene_list[data_idx]

                row = _batch_sample_row(
                    batch=batch,
                    end_points=end_points,
                    sample_i=sample_i,
                    data_idx=data_idx,
                    scene_name=scene_name,
                    geometry_cache=geometry_cache,
                )
                rows.append(row)
                csv_writer.writerow(
                    {key: row.get(key, "") for key in PER_SAMPLE_FIELDS}
                )

                gg = GraspGroup(pred.detach().cpu().numpy())
                if cfgs.save_nocollision:
                    out_dir = os.path.join(
                        cfgs.save_dir + "_nocollision",
                        scene_name,
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
                    detector = ModelFreeCollisionDetectorTorch(
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
                    scene_name,
                    cfgs.camera,
                )
                os.makedirs(out_dir, exist_ok=True)
                gg.save_npy(
                    os.path.join(out_dir, f"{data_idx % 256:04d}.npy")
                )
                processed += 1

            csv_f.flush()

            if batch_idx % 20 == 0:
                elapsed = time.perf_counter() - start
                latest = rows[-1] if rows else {}
                print(
                    f"[INFER] batch={batch_idx}/{len(dataloader)} "
                    f"samples={processed}/{len(eval_dataset)} "
                    f"sec_per_sample={elapsed / max(processed, 1):.3f} "
                    f"last_anno={latest.get('anno_idx', 'NA')} "
                    f"last_tilt={_finite_float(latest.get('tilt_deg')):.2f} "
                    f"last_mae_mm={1000.0 * _finite_float(latest.get('dense_mae_m')):.2f} "
                    f"last_bias_mm={1000.0 * _finite_float(latest.get('dense_bias_m')):.2f}",
                    flush=True,
                )
                # Keep partial summaries recoverable if a long run is interrupted.
                _export_summaries(diag_dir, rows, manifest)

    _export_summaries(diag_dir, rows, manifest)

    selected = [r for r in rows if int(r["anno_idx"]) in FORCE_INCLUDE_ANNOS]
    by_anno = defaultdict(list)
    for row in selected:
        by_anno[int(row["anno_idx"])].append(row)
    for anno in FORCE_INCLUDE_ANNOS:
        if anno not in by_anno:
            continue
        summary = _summarize_rows(by_anno[anno], "anno_idx", anno)
        print(
            f"[DEPTH-DIAG] anno={anno:03d} "
            f"tilt={summary['tilt_deg_mean']:.2f}deg "
            f"dense_mae={1000.0 * summary['dense_pooled_mae_m']:.3f}mm "
            f"dense_bias={1000.0 * summary['dense_pooled_bias_m']:.3f}mm "
            f"object_mae={1000.0 * summary['object_pooled_mae_m']:.3f}mm "
            f"table_bias={1000.0 * summary['table_pooled_bias_m']:.3f}mm "
            f"center_z_mae={1000.0 * summary['grasp_center_pooled_mae_m']:.3f}mm",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    print(
        f"[DONE] samples={processed} elapsed_h={elapsed / 3600.0:.3f} "
        f"grasp_dir={cfgs.save_dir} diag_dir={diag_dir}",
        flush=True,
    )


if __name__ == "__main__":
    inference()
