"""Common utilities for Rep-P0 fixed-action geometry-source diagnosis.

Rep-P0 keeps each physical grasp action fixed and changes only the geometry
source from which gripper-centric evidence is extracted.

Geometry sources:
  pred      : Stage-1 RGB-predicted metric depth
  sensor    : captured RealSense depth
  rendered  : clean virtual/rendered visible depth from virtual_scenes
  cad_full  : complete CAD scene + table in the current camera frame (privileged)

The descriptor is intentionally transparent and source-agnostic: a local
gripper-frame occupancy histogram, collision/contact-zone occupancies, local
shape statistics, and the complete action parameters. All sources are voxelized
at the same resolution before feature extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


FRICTION_THRESHOLDS = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
GEOMETRY_SOURCES = ("pred", "sensor", "rendered", "cad_full")\nREP_P0_EVIDENCE_VOXEL_SIZE = 0.005


def parse_offsets_mm(text: str) -> np.ndarray:
    values = np.asarray([float(x.strip()) for x in str(text).split(",") if x.strip()], dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("offsets_mm must contain at least one value.")
    if int(np.isclose(values, 0.0, atol=1e-8).sum()) != 1:
        raise ValueError("offsets_mm must contain exactly one zero offset.")
    return values


def zero_offset_index(offsets_mm: Sequence[float]) -> int:
    arr = np.asarray(offsets_mm, dtype=np.float32)
    ids = np.flatnonzero(np.isclose(arr, 0.0, atol=1e-8))
    if len(ids) != 1:
        raise ValueError("Expected exactly one zero offset.")
    return int(ids[0])


def friction_to_cdf_targets(friction: np.ndarray) -> np.ndarray:
    f = np.asarray(friction, dtype=np.float32)
    return (
        np.isfinite(f)[..., None]
        & (f[..., None] > 0.0)
        & (f[..., None] <= FRICTION_THRESHOLDS.reshape((1,) * f.ndim + (-1,)) + 1e-6)
    ).astype(np.float32)


def friction_utility(friction: np.ndarray) -> np.ndarray:
    return friction_to_cdf_targets(friction).mean(axis=-1).astype(np.float32)


def success08(friction: np.ndarray) -> np.ndarray:
    f = np.asarray(friction, dtype=np.float32)
    return np.isfinite(f) & (f > 0.0) & (f <= 0.8 + 1e-6)


def voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {pts.shape}")
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    size = max(float(voxel_size), 1e-6)
    key = np.floor(pts / size).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(first)].astype(np.float32, copy=False)


def backproject_depth_map(
    depth_m: np.ndarray,
    K: np.ndarray,
    *,
    voxel_size: float = 0.005,
    min_depth: float = 0.05,
    max_depth: float = 2.0,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be [H,W], got {depth.shape}")
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be [3,3], got {K.shape}")

    valid = np.isfinite(depth) & (depth > float(min_depth)) & (depth < float(max_depth))
    vv, uu = np.nonzero(valid)
    if len(vv) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[vv, uu]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (uu.astype(np.float32) - cx) / fx * z
    y = (vv.astype(np.float32) - cy) / fy * z
    pts = np.stack((x, y, z), axis=1)
    return voxel_downsample_numpy(pts, voxel_size)


def build_translation_ray_actions(
    native_actions: np.ndarray,
    offsets_mm: Sequence[float],
    *,
    min_depth: float = 0.2,
    max_depth: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create same-ray translation hypotheses while preserving the full grasp decision.

    The native decoded grasp fixes width, height, insertion depth and rotation.
    Only camera-ray translation changes. A shift dz keeps the image ray fixed by
    scaling x/y/z by (z+dz)/z.
    """
    native = np.asarray(native_actions, dtype=np.float32)
    if native.ndim != 2 or native.shape[1] != 17:
        raise ValueError(f"native_actions must be [Q,17], got {native.shape}")
    offsets = np.asarray(offsets_mm, dtype=np.float32)
    K = len(offsets)
    Q = len(native)
    actions = np.repeat(native[None, :, :], K, axis=0)
    valid = np.ones((K, Q), dtype=bool)

    t0 = native[:, 13:16]
    z0 = t0[:, 2]
    base_valid = np.isfinite(t0).all(axis=1) & np.isfinite(z0) & (z0 > 1e-6)
    for k, off_mm in enumerate(offsets):
        z = z0 + float(off_mm) / 1000.0
        vk = base_valid & (z >= float(min_depth)) & (z <= float(max_depth))
        scale = np.ones_like(z, dtype=np.float32)
        scale[vk] = z[vk] / z0[vk]
        actions[k, :, 13:16] = t0 * scale[:, None]
        valid[k] = vk

    zidx = zero_offset_index(offsets)
    valid[zidx] = base_valid
    actions[zidx] = native
    return actions, valid


def select_query_indices(native_actions: torch.Tensor, count: int, mode: str) -> torch.Tensor:
    if native_actions.dim() != 2 or native_actions.shape[-1] != 17:
        raise ValueError("native_actions must be [N,17].")
    n = native_actions.shape[0]
    ids = torch.arange(n, device=native_actions.device)
    if count <= 0 or count >= n or mode == "all":
        return ids
    score = native_actions[:, 0]
    ranked = torch.argsort(score, descending=True, stable=True)
    if mode == "topk":
        return ranked[:count]
    if mode == "uniform":
        pos = torch.round(torch.linspace(0, n - 1, steps=count, device=ids.device)).long()
        return ids.index_select(0, torch.unique(pos, sorted=True)[:count])
    if mode != "topk_uniform":
        raise ValueError(f"Unknown query mode {mode!r}.")
    n_top = count // 2
    remain = ranked[n_top:]
    need = count - n_top
    if need >= len(remain):
        tail = remain
    else:
        pos = torch.round(torch.linspace(0, len(remain) - 1, steps=need, device=ids.device)).long()
        pos = torch.unique(pos, sorted=True)
        if len(pos) < need:
            used = torch.zeros(len(remain), dtype=torch.bool, device=ids.device)
            used[pos] = True
            fill = torch.nonzero(~used, as_tuple=False).squeeze(1)[: need - len(pos)]
            pos = torch.sort(torch.cat((pos, fill))).values
        tail = remain.index_select(0, pos[:need])
    return torch.cat((ranked[:n_top], tail))


@dataclass(frozen=True)
class DescriptorConfig:
    voxel_size: float = 0.005
    finger_width: float = 0.01
    finger_length: float = 0.06
    approach_dist: float = 0.03
    context_margin: float = 0.02
    hist_bins: Tuple[int, int, int] = (5, 7, 3)
    neighbor_radius: float = 0.18

    @property
    def hist_dim(self) -> int:
        return int(np.prod(self.hist_bins))

    @property
    def feature_dim(self) -> int:
        # histogram + 8 region occupancies + 12 local statistics + 15 action scalars
        return self.hist_dim + 8 + 12 + 15


def _occupancy(count: int, volume: float, voxel_size: float) -> float:
    expected = max(float(volume) / max(float(voxel_size) ** 3, 1e-12), 1.0)
    return float(np.clip(float(count) / expected, 0.0, 1.0))


def _safe_min_abs(values: np.ndarray, target: float, fallback: float) -> float:
    if values.size == 0:
        return float(fallback)
    return float(np.min(np.abs(values - float(target))))


def describe_actions(
    points: np.ndarray,
    actions_kq17: np.ndarray,
    valid_kq: np.ndarray,
    config: DescriptorConfig,
) -> np.ndarray:
    """Extract source-agnostic gripper-centric geometry descriptors [K,Q,F]."""
    pts = voxel_downsample_numpy(points, config.voxel_size)
    actions = np.asarray(actions_kq17, dtype=np.float32)
    valid = np.asarray(valid_kq, dtype=bool)
    if actions.ndim != 3 or actions.shape[-1] != 17:
        raise ValueError("actions_kq17 must be [K,Q,17].")
    if valid.shape != actions.shape[:2]:
        raise ValueError("valid_kq must match actions [K,Q].")

    Kc, Q, _ = actions.shape
    out = np.zeros((Kc, Q, config.feature_dim), dtype=np.float32)
    tree = cKDTree(pts) if len(pts) else None
    bins = config.hist_bins

    for k in range(Kc):
        for q in range(Q):
            if not valid[k, q]:
                continue
            g = actions[k, q]
            width = max(float(g[1]), 1e-4)
            height = max(float(g[2]), 1e-4)
            depth = float(g[3])
            R = g[4:13].reshape(3, 3).astype(np.float32)
            t = g[13:16].astype(np.float32)

            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                continue
            idx = [] if tree is None else tree.query_ball_point(t, r=float(config.neighbor_radius))
            local = np.empty((0, 3), dtype=np.float32)
            if len(idx):
                local = (pts[np.asarray(idx, dtype=np.int64)] - t[None, :]) @ R

            fw = float(config.finger_width)
            fl = float(config.finger_length)
            ad = max(float(config.approach_dist), fw)
            m = float(config.context_margin)

            xmin = depth - fl - fw - ad - m
            xmax = depth + m
            ymax = width / 2.0 + fw + m
            zmax = height / 2.0 + m

            if len(local):
                crop_mask = (
                    (local[:, 0] >= xmin) & (local[:, 0] <= xmax)
                    & (np.abs(local[:, 1]) <= ymax)
                    & (np.abs(local[:, 2]) <= zmax)
                )
                crop = local[crop_mask]
            else:
                crop = local

            # 3D normalized occupancy histogram.
            hist = np.zeros(bins, dtype=np.float32)
            if len(crop):
                nx = np.clip((crop[:, 0] - xmin) / max(xmax - xmin, 1e-6), 0.0, 1.0 - 1e-7)
                ny = np.clip((crop[:, 1] + ymax) / max(2.0 * ymax, 1e-6), 0.0, 1.0 - 1e-7)
                nz = np.clip((crop[:, 2] + zmax) / max(2.0 * zmax, 1e-6), 0.0, 1.0 - 1e-7)
                hist, _ = np.histogramdd(
                    np.stack((nx, ny, nz), axis=1),
                    bins=bins,
                    range=((0, 1), (0, 1), (0, 1)),
                )
                bin_volume = (
                    (xmax - xmin) / bins[0]
                    * (2.0 * ymax) / bins[1]
                    * (2.0 * zmax) / bins[2]
                )
                hist = np.clip(
                    hist.astype(np.float32)
                    / max(bin_volume / (config.voxel_size ** 3), 1.0),
                    0.0,
                    1.0,
                )

            # Collision/contact-inspired zones, matching GraspNet gripper axes.
            if len(local):
                hm = np.abs(local[:, 2]) < height / 2.0
                xm = (local[:, 0] > depth - fl) & (local[:, 0] < depth)
                left = hm & xm & (local[:, 1] > -(width / 2.0 + fw)) & (local[:, 1] < -width / 2.0)
                right = hm & xm & (local[:, 1] < (width / 2.0 + fw)) & (local[:, 1] > width / 2.0)
                bottom = (
                    hm
                    & (local[:, 1] > -(width / 2.0 + fw))
                    & (local[:, 1] < (width / 2.0 + fw))
                    & (local[:, 0] <= depth - fl)
                    & (local[:, 0] > depth - fl - fw)
                )
                approach = (
                    hm
                    & (local[:, 1] > -(width / 2.0 + fw))
                    & (local[:, 1] < (width / 2.0 + fw))
                    & (local[:, 0] <= depth - fl - fw)
                    & (local[:, 0] > depth - fl - fw - ad)
                )
                inner = hm & xm & (local[:, 1] >= -width / 2.0) & (local[:, 1] <= width / 2.0)
                inner_front = inner & (local[:, 0] > depth - fl / 2.0)
                inner_back = inner & ~inner_front
            else:
                left = right = bottom = approach = inner = inner_front = inner_back = np.zeros(0, dtype=bool)

            lr_vol = height * fl * fw
            bottom_vol = height * (width + 2 * fw) * fw
            approach_vol = height * (width + 2 * fw) * ad
            inner_vol = height * fl * width
            region = np.asarray([
                _occupancy(int(left.sum()), lr_vol, config.voxel_size),
                _occupancy(int(right.sum()), lr_vol, config.voxel_size),
                _occupancy(int(bottom.sum()), bottom_vol, config.voxel_size),
                _occupancy(int(approach.sum()), approach_vol, config.voxel_size),
                _occupancy(int(inner.sum()), inner_vol, config.voxel_size),
                _occupancy(int(inner_front.sum()), inner_vol / 2.0, config.voxel_size),
                _occupancy(int(inner_back.sum()), inner_vol / 2.0, config.voxel_size),
                _occupancy(len(crop), max((xmax-xmin)*2*ymax*2*zmax, 1e-8), config.voxel_size),
            ], dtype=np.float32)

            if len(crop):
                scale_xyz = np.asarray(
                    [max(xmax-xmin, 1e-4), max(2*ymax, 1e-4), max(2*zmax, 1e-4)],
                    dtype=np.float32,
                )
                mean_xyz = crop.mean(axis=0) / scale_xyz
                std_xyz = crop.std(axis=0) / scale_xyz
                min_xyz = crop.min(axis=0) / scale_xyz
                max_xyz = crop.max(axis=0) / scale_xyz
                stats = np.concatenate((mean_xyz, std_xyz, min_xyz, max_xyz)).astype(np.float32)
            else:
                stats = np.zeros(12, dtype=np.float32)

            action = np.concatenate((
                np.asarray([width, height, depth], dtype=np.float32),
                R.reshape(-1),
                t,
            )).astype(np.float32)
            feat = np.concatenate((hist.reshape(-1), region, stats, action)).astype(np.float32)
            if feat.size != config.feature_dim:
                raise RuntimeError(f"Descriptor size mismatch: {feat.size} vs {config.feature_dim}")
            out[k, q] = feat
    return out


def load_full_cad_scene_cloud(evaluator, scene_id: int, anno_id: int, voxel_size: float = 0.005) -> np.ndarray:
    """Return complete CAD objects + table at the Rep-P0 evidence resolution.

    This is a privileged geometry upper bound. It does not call the
    ExactActionGraspNetEvaluator._scene_models() helper, because that helper
    first downsamples object CAD points to 8 mm for the official label
    evaluator. Rep-P0 instead reads raw scene CAD points, transforms them to
    the current camera frame, builds a table directly at voxel_size, and then
    applies the same final voxelization used by pred/sensor/rendered evidence.

    Exact-action labels are still produced by the untouched official
    evaluator; only Rep-P0 input evidence is standardized to 5 mm.
    """
    from graspnetAPI.utils.eval_utils import create_table_points, transform_points

    # Raw CAD geometry: avoid the evaluator cached 8-mm object sampling.
    raw_models, _, _ = evaluator.eval.get_scene_models(int(scene_id), ann_id=0)
    _, poses, camera_pose, align_mat = evaluator.eval.get_model_poses(
        int(scene_id), int(anno_id)
    )
    models_cam = [
        transform_points(np.asarray(model, dtype=np.float32), poses[obj_index])
        for obj_index, model in enumerate(raw_models)
    ]

    # Build the table at the same spatial resolution as all Rep-P0 evidence.
    table = create_table_points(
        1.0,
        1.0,
        0.05,
        dx=-0.5,
        dy=-0.5,
        dz=-0.05,
        grid_size=float(voxel_size),
    )
    table_cam = transform_points(
        table,
        np.linalg.inv(np.matmul(align_mat, camera_pose)),
    )

    parts = [p for p in models_cam if len(p)] + [table_cam]
    if not parts:
        return np.empty((0, 3), dtype=np.float32)
    return voxel_downsample_numpy(
        np.concatenate(parts, axis=0).astype(np.float32, copy=False),
        voxel_size,
    )

class GeometrySourceProbe(nn.Module):
    """Same lightweight six-threshold CDF probe for every geometry source."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, len(FRICTION_THRESHOLDS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def predicted_utility_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float()).mean(dim=-1)
