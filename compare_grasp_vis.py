#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two GraspNet AP result folders.

This script reads split-level GraspNetEval outputs named like:
    ap_test_seen_realsense.npy
    ap_test_similar_realsense.npy
    ap_test_novel_realsense.npy

Supported array formats:
1) Scalar or shape (3,): already summarized AP values.
   If shape (3,), interpreted as [AP, AP0.8, AP0.4] by default.
2) Shape (..., 3): summarized rows with last dim [AP, AP0.8, AP0.4].
3) GraspNetEval accuracy array, e.g. (num_scenes, num_annos, top_k, num_friction)
   commonly (30, 256, 50, 6) or already sampled (30, 26, 50, 6).
   In this case:
       AP    = mean over all dimensions
       AP0.8 = mean over friction index --ap08-index, default 3
       AP0.4 = mean over friction index --ap04-index, default 1

Important: This script refuses to read grasp prediction arrays with shape (N,17).
Those are not AP/eval files.

Sample interval behavior:
- If the AP file is full per-anno, e.g. (30, 256, 50, 6), --sample-interval 0.1
  samples anno indices 0,10,20,... per scene.
- If the AP file is already sampled, e.g. (30, 26, 50, 6), the script will NOT
  sample again by default. Use --force-resample if you explicitly want that.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional visualization dependencies.
try:
    import cv2
except Exception:
    cv2 = None

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    from graspnetAPI import GraspGroup  # type: ignore
    HAS_GRASPNET_API = True
except Exception:
    GraspGroup = None  # type: ignore
    HAS_GRASPNET_API = False


SPLITS = ["test_seen", "test_similar", "test_novel"]
SPLIT_SHORT = {
    "test_seen": "seen",
    "test_similar": "similar",
    "test_novel": "novel",
}
# GraspNet test scenes: seen 100-129, similar 130-159, novel 160-189.
SPLIT_SCENE_START = {
    "test_seen": 100,
    "test_similar": 130,
    "test_novel": 160,
}

DEFAULT_GRASP_TEMPLATES = [
    "{scene}/{camera}/{ann:04d}.npy",
    "{scene}/{camera}/grasp_{ann:04d}.npy",
    "{scene}/{camera}/pred_{ann:04d}.npy",
    "{scene}/{ann:04d}.npy",
    "{scene}/grasp_{ann:04d}.npy",
    "{scene}_{camera}_{ann:04d}.npy",
    "{scene}_{ann:04d}.npy",
]
DEFAULT_RGB_TEMPLATES = [
    "scenes/{scene}/{camera}/rgb/{ann:04d}.png",
    "{scene}/{camera}/rgb/{ann:04d}.png",
]
DEFAULT_DEPTH_TEMPLATES = [
    "scenes/{scene}/{camera}/depth/{ann:04d}.png",
    "{scene}/{camera}/depth/{ann:04d}.png",
]
DEFAULT_CAMK_TEMPLATES = [
    "scenes/{scene}/{camera}/camK.npy",
    "{scene}/{camera}/camK.npy",
]


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method1-name", type=str, default="method1")
    parser.add_argument("--method1-eval-root", type=str, required=True)
    parser.add_argument("--method2-name", type=str, default="method2")
    parser.add_argument("--method2-eval-root", type=str, required=True)
    parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "kinect"])
    parser.add_argument("--output-root", type=str, required=True)

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        help="Which split AP files to compare.",
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help=(
            "Sample annos per scene before aggregation. 1.0 means use all annos in the AP file. "
            "0.1 means keep anno indices 0,10,20,... but only if the file still has annos_per_scene entries."
        ),
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--annos-per-scene", type=int, default=256)
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Resample even if AP file appears already sampled, e.g. second dim is not annos_per_scene.",
    )

    parser.add_argument(
        "--friction-axis",
        type=int,
        default=-1,
        help="Axis corresponding to friction thresholds for high-dimensional GraspNetEval arrays.",
    )
    parser.add_argument(
        "--anno-axis",
        type=int,
        default=1,
        help="Axis corresponding to anno/frame id for high-dimensional GraspNetEval arrays.",
    )
    parser.add_argument(
        "--scene-axis",
        type=int,
        default=0,
        help="Axis corresponding to local scene index for high-dimensional GraspNetEval arrays.",
    )
    parser.add_argument(
        "--ap04-index",
        type=int,
        default=1,
        help="Friction index for AP0.4. Default assumes friction list contains 0.4 at index 1.",
    )
    parser.add_argument(
        "--ap08-index",
        type=int,
        default=3,
        help="Friction index for AP0.8. Default assumes friction list contains 0.8 at index 3.",
    )
    parser.add_argument(
        "--summary-last-dim-order",
        type=str,
        default="AP,AP0.8,AP0.4",
        help="If an AP file has last dim=3, interpret the three values in this order.",
    )
    parser.add_argument("--no-plots", action="store_true")

    # Optional visualization args. If omitted, the original AP comparison behavior stays unchanged.
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--method1-grasp-root", type=str, default="")
    parser.add_argument("--method2-grasp-root", type=str, default="")
    parser.add_argument("--skip-vis", action="store_true")
    parser.add_argument(
        "--vis-topk-scene-worst",
        type=int,
        default=3,
        help=(
            "Backward-compatible name. Number of samples per scene to save in each "
            "visualization group. Prefer --vis-topk-scene-samples for new runs."
        ),
    )
    parser.add_argument(
        "--vis-topk-scene-samples",
        type=int,
        default=None,
        help="Number of samples per scene to save for each visualization group.",
    )
    parser.add_argument("--vis-topk-grasps", type=int, default=50)
    parser.add_argument("--grasp-mesh-sample-points", type=int, default=1000)
    parser.add_argument("--pcd-stride", type=int, default=2)
    parser.add_argument(
        "--save-pcd-num-points",
        type=int,
        default=0,
        help=(
            "If > 0, randomly downsample the RGB-D scene point cloud to at most this many "
            "points for each side before saving the visualization PLY. Grasp mesh samples are "
            "still controlled by --grasp-mesh-sample-points."
        ),
    )
    parser.add_argument(
        "--save-pcd-random-seed",
        type=int,
        default=0,
        help="Random seed used by --save-pcd-num-points.",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-trunc", type=float, default=2.0)
    parser.add_argument("--side-by-side-gap", type=float, default=0.10)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# IO / parsing
# -----------------------------------------------------------------------------
def _safe_load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    arr = np.load(str(path), allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()
    if isinstance(arr, dict):
        # Common fallbacks if someone saved a dict.
        for key in ["ap", "accuracy", "grasp_accuracy", "result", "data"]:
            if key in arr:
                arr = arr[key]
                break
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 17:
        raise ValueError(
            f"{path} has shape {arr.shape}, which looks like a grasp prediction array "
            "[score,width,height,depth,rot(9),center(3),obj_id], not an AP eval file. "
            "Please pass the directory containing ap_test_*_*.npy files as eval-root."
        )
    return arr


def _ap_file(root: Path, split: str, camera: str) -> Path:
    return root / f"ap_{split}_{camera}.npy"


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Invalid axis {axis} for ndim={ndim}")
    return axis


def _sample_along_axis(arr: np.ndarray, axis: int, indices: Sequence[int]) -> np.ndarray:
    return np.take(arr, indices=np.asarray(indices, dtype=np.int64), axis=axis)


def _build_sample_indices(length: int, sample_interval: float, offset: int) -> List[int]:
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")
    if sample_interval >= 1.0:
        return list(range(length))
    stride = max(1, int(round(1.0 / sample_interval)))
    offset = int(offset) % stride
    return list(range(offset, length, stride))


def maybe_apply_sample_interval(
    arr: np.ndarray,
    args: argparse.Namespace,
    split: str,
    method_name: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply sample_interval only when the AP array is full per-anno by default."""
    info: Dict[str, Any] = {
        "shape_before_sampling": list(arr.shape),
        "sample_interval": float(args.sample_interval),
        "sample_offset": int(args.sample_offset),
        "sample_applied": False,
        "sample_indices": None,
        "warning": None,
    }

    if args.sample_interval >= 1.0:
        info["shape_after_sampling"] = list(arr.shape)
        return arr, info

    if arr.ndim < 2:
        info["warning"] = "sample_interval requested but AP file is summary/scalar; skipped."
        info["shape_after_sampling"] = list(arr.shape)
        return arr, info

    anno_axis = _normalize_axis(args.anno_axis, arr.ndim)
    anno_len = int(arr.shape[anno_axis])

    if (anno_len != int(args.annos_per_scene)) and (not args.force_resample):
        info["warning"] = (
            f"sample_interval requested, but anno axis length is {anno_len}, not annos_per_scene="
            f"{args.annos_per_scene}. This AP file is likely already sampled; skipped. "
            "Use --force-resample to resample anyway."
        )
        info["shape_after_sampling"] = list(arr.shape)
        return arr, info

    indices = _build_sample_indices(anno_len, float(args.sample_interval), int(args.sample_offset))
    arr_s = _sample_along_axis(arr, anno_axis, indices)
    info["sample_applied"] = True
    info["sample_indices"] = indices
    info["shape_after_sampling"] = list(arr_s.shape)
    return arr_s, info


def _parse_summary_order(order: str) -> Dict[str, int]:
    names = [x.strip() for x in order.split(",")]
    out: Dict[str, int] = {}
    for i, name in enumerate(names):
        out[name] = i
    required = ["AP", "AP0.8", "AP0.4"]
    miss = [r for r in required if r not in out]
    if miss:
        raise ValueError(f"summary-last-dim-order must include {required}, got {names}")
    return out


def parse_ap_array(
    arr: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Return split-level metrics plus optional per-scene and per-anno rows.

    Metrics keys: AP, AP0.8, AP0.4, plus friction_mean_* when applicable.
    """
    arr = np.asarray(arr, dtype=np.float64)
    meta: Dict[str, Any] = {
        "shape": list(arr.shape),
        "ndim": int(arr.ndim),
        "raw_min": float(np.nanmin(arr)) if arr.size else None,
        "raw_max": float(np.nanmax(arr)) if arr.size else None,
        "raw_mean": float(np.nanmean(arr)) if arr.size else None,
        "parse_mode": None,
        "friction_means": None,
    }

    if arr.size == 0:
        raise ValueError("Empty AP array")

    # Scalar summary.
    if arr.ndim == 0 or arr.size == 1:
        val = float(np.asarray(arr).reshape(-1)[0])
        meta["parse_mode"] = "scalar_summary"
        return {"AP": val, "AP0.8": math.nan, "AP0.4": math.nan}, meta, None, None

    # Summary vector (3,).
    if arr.ndim == 1 and arr.shape[0] == 3:
        order = _parse_summary_order(args.summary_last_dim_order)
        meta["parse_mode"] = "summary_vector_3"
        return {
            "AP": float(arr[order["AP"]]),
            "AP0.8": float(arr[order["AP0.8"]]),
            "AP0.4": float(arr[order["AP0.4"]]),
        }, meta, None, None

    # Summary rows with last dim 3, e.g. per-scene [AP, AP0.8, AP0.4].
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        order = _parse_summary_order(args.summary_last_dim_order)
        flat = arr.reshape(-1, 3)
        meta["parse_mode"] = "summary_last_dim_3"
        metrics = {
            "AP": float(np.nanmean(flat[:, order["AP"]])),
            "AP0.8": float(np.nanmean(flat[:, order["AP0.8"]])),
            "AP0.4": float(np.nanmean(flat[:, order["AP0.4"]])),
        }
        return metrics, meta, None, None

    # GraspNetEval accuracy array, typically (..., friction_dim), e.g. (30, 256, 50, 6).
    friction_axis = _normalize_axis(args.friction_axis, arr.ndim)
    if arr.shape[friction_axis] < 2:
        raise ValueError(
            f"Unsupported AP array shape {arr.shape}: friction axis {friction_axis} has length "
            f"{arr.shape[friction_axis]}. If this is a summary array, use shape (...,3)."
        )

    # Move friction to the last dim for easier handling.
    arr_f = np.moveaxis(arr, friction_axis, -1)
    num_friction = arr_f.shape[-1]
    if args.ap04_index >= num_friction or args.ap08_index >= num_friction:
        raise ValueError(
            f"ap04/ap08 index out of range for friction dim={num_friction}: "
            f"ap04={args.ap04_index}, ap08={args.ap08_index}"
        )

    friction_means = np.nanmean(arr_f.reshape(-1, num_friction), axis=0)
    meta["parse_mode"] = "graspnet_accuracy_array"
    meta["friction_means"] = [float(x) for x in friction_means]
    meta["ap04_index"] = int(args.ap04_index)
    meta["ap08_index"] = int(args.ap08_index)

    metrics = {
        "AP": float(np.nanmean(arr_f)),
        "AP0.8": float(friction_means[int(args.ap08_index)]),
        "AP0.4": float(friction_means[int(args.ap04_index)]),
    }
    for i, v in enumerate(friction_means):
        metrics[f"friction_mean_{i}"] = float(v)

    # Optional per-scene and per-anno analysis if axes are available.
    per_scene_df = None
    per_anno_df = None

    try:
        scene_axis_orig = _normalize_axis(args.scene_axis, arr.ndim)
        anno_axis_orig = _normalize_axis(args.anno_axis, arr.ndim)
        # After moveaxis friction -> last, update scene/anno axes.
        axis_order = [i for i in range(arr.ndim) if i != friction_axis] + [friction_axis]
        scene_axis = axis_order.index(scene_axis_orig)
        anno_axis = axis_order.index(anno_axis_orig)

        if scene_axis != anno_axis and arr_f.ndim >= 4:
            # Per-scene AP: mean over all axes except scene.
            scene_vals_ap = np.nanmean(arr_f, axis=tuple(i for i in range(arr_f.ndim) if i != scene_axis))
            scene_vals_ap08 = np.nanmean(
                np.take(arr_f, int(args.ap08_index), axis=-1),
                axis=tuple(i for i in range(arr_f.ndim - 1) if i != scene_axis),
            )
            scene_vals_ap04 = np.nanmean(
                np.take(arr_f, int(args.ap04_index), axis=-1),
                axis=tuple(i for i in range(arr_f.ndim - 1) if i != scene_axis),
            )
            per_scene_df = pd.DataFrame({
                "local_scene_index": np.arange(scene_vals_ap.shape[0], dtype=np.int64),
                "AP": scene_vals_ap.reshape(-1),
                "AP0.8": scene_vals_ap08.reshape(-1),
                "AP0.4": scene_vals_ap04.reshape(-1),
            })

            # Per-anno AP: mean over all axes except scene and anno.
            # Bring scene, anno to first two dims to flatten into rows.
            arr_sa = np.moveaxis(arr_f, [scene_axis, anno_axis], [0, 1])
            ap_sa = np.nanmean(arr_sa, axis=tuple(range(2, arr_sa.ndim)))
            ap08_sa = np.nanmean(arr_sa[..., int(args.ap08_index)], axis=tuple(range(2, arr_sa.ndim - 1)))
            ap04_sa = np.nanmean(arr_sa[..., int(args.ap04_index)], axis=tuple(range(2, arr_sa.ndim - 1)))
            rows: List[Dict[str, Any]] = []
            for s in range(ap_sa.shape[0]):
                for a in range(ap_sa.shape[1]):
                    rows.append({
                        "local_scene_index": int(s),
                        "local_anno_index": int(a),
                        "AP": float(ap_sa[s, a]),
                        "AP0.8": float(ap08_sa[s, a]),
                        "AP0.4": float(ap04_sa[s, a]),
                    })
            per_anno_df = pd.DataFrame(rows)
    except Exception as e:
        meta["per_scene_warning"] = repr(e)

    return metrics, meta, per_scene_df, per_anno_df


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def resolve_path(root: Path, scene_id: int, ann_id: int, camera: str, templates: Sequence[str]) -> Optional[Path]:
    scene = f"scene_{scene_id:04d}"
    for tmpl in templates:
        p = root / tmpl.format(scene=scene, camera=camera, ann=ann_id)
        if p.exists():
            return p
    return None


def load_grasp_array(path: Path) -> np.ndarray:
    arr = np.load(str(path), allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()
    if isinstance(arr, dict):
        for k in ["grasp_group", "grasps", "pred", "data"]:
            if k in arr:
                arr = arr[k]
                break
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected grasp shape: {arr.shape} @ {path}")
    return arr.astype(np.float32)


def sort_nms_topk_grasp_array(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return arr
    if HAS_GRASPNET_API:
        gg = GraspGroup(arr)
        gg = gg.sort_by_score()
        gg = gg.nms()
        gg_vis = gg[:min(k, len(gg))]
        return np.asarray(gg_vis.grasp_group_array, dtype=np.float32)
    order = np.argsort(-arr[:, 0])
    return arr[order[: min(k, arr.shape[0])]]


def load_camK(dataset_root: Path, scene_id: int, camera: str) -> np.ndarray:
    p = resolve_path(dataset_root, scene_id, 0, camera, DEFAULT_CAMK_TEMPLATES)
    if p is None:
        raise FileNotFoundError(f"camK.npy not found for scene_{scene_id:04d}/{camera}")
    K = np.load(str(p))
    K = np.asarray(K, dtype=np.float32)
    if K.shape == (3, 3):
        return K
    if K.shape == (3, 4):
        return K[:, :3]
    raise ValueError(f"Unexpected camK shape: {K.shape} @ {p}")


def load_rgb_depth(dataset_root: Path, scene_id: int, ann_id: int, camera: str) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("cv2 is required for visualization but is not available.")
    rgb_path = resolve_path(dataset_root, scene_id, ann_id, camera, DEFAULT_RGB_TEMPLATES)
    depth_path = resolve_path(dataset_root, scene_id, ann_id, camera, DEFAULT_DEPTH_TEMPLATES)
    if rgb_path is None:
        raise FileNotFoundError(f"RGB not found for scene_{scene_id:04d}/{camera}/{ann_id:04d}")
    if depth_path is None:
        raise FileNotFoundError(f"Depth not found for scene_{scene_id:04d}/{camera}/{ann_id:04d}")
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise RuntimeError(f"Failed to read rgb/depth @ {rgb_path} / {depth_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb, depth


def backproject_depth_to_pcd(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    depth_trunc: float,
    stride: int,
):
    if o3d is None:
        raise RuntimeError("open3d is required for visualization but is not available.")
    h, w = depth.shape[:2]
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    xv, yv = np.meshgrid(xs, ys)
    z = depth[yv, xv].astype(np.float32) / depth_scale
    valid = (z > 0) & np.isfinite(z) & (z < depth_trunc)
    xv = xv[valid].astype(np.float32)
    yv = yv[valid].astype(np.float32)
    z = z[valid]

    x = (xv - float(K[0, 2])) * z / float(K[0, 0])
    y = (yv - float(K[1, 2])) * z / float(K[1, 1])
    pts = np.stack([x, y, z], axis=1)

    cols = rgb[yv.astype(np.int32), xv.astype(np.int32)].astype(np.float32) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd


def sample_pcd_points(pcd, num_points: int, seed: int = 0):
    """Randomly keep at most num_points points from an Open3D point cloud."""
    if o3d is None:
        return pcd
    num_points = int(num_points)
    if num_points <= 0:
        return pcd
    n = len(pcd.points)
    if n <= num_points:
        return pcd
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=num_points, replace=False)
    return pcd.select_by_index(idx.astype(np.int64).tolist())


def _json_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def tint_pcd(pcd, tint_rgb: Tuple[float, float, float], strength: float = 0.35):
    if o3d is None:
        return pcd
    out = pcd.clone() if hasattr(pcd, "clone") else o3d.geometry.PointCloud(pcd)
    cols = np.asarray(out.colors)
    if cols.size == 0:
        cols = np.zeros((len(out.points), 3), dtype=np.float64)
    tint = np.asarray(tint_rgb, dtype=np.float64)[None, :]
    cols = np.clip(cols * (1.0 - strength) + tint * strength, 0.0, 1.0)
    out.colors = o3d.utility.Vector3dVector(cols)
    return out


def make_grasp_samples(arr: np.ndarray, sample_points: int):
    if o3d is None:
        raise RuntimeError("open3d is required for visualization but is not available.")
    pcd = o3d.geometry.PointCloud()
    if arr.shape[0] == 0:
        return pcd

    if HAS_GRASPNET_API:
        gg = GraspGroup(arr)
        geoms = gg.to_open3d_geometry_list()
        for g in geoms:
            # Keep the original GraspNetAPI geometry colors. In common GraspNetAPI
            # visualizations, grasp color encodes grasp score, so do not repaint them.
            sampled = g.sample_points_uniformly(number_of_points=sample_points)
            pcd += sampled
        return pcd

    # Fallback path when graspnetAPI is unavailable: keep the original behavior by
    # visualizing grasp centers only. This path cannot recover GraspNetAPI's score
    # color mapping.
    centers = arr[:, 13:16]
    pcd.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
    fallback = np.array((1.0, 0.0, 0.0), dtype=np.float64)[None, :]
    colors = np.tile(fallback, (centers.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def translate_geometry(g, t: np.ndarray):
    if o3d is None:
        return g
    out = g.clone() if hasattr(g, "clone") else o3d.geometry.PointCloud(g)
    out.translate(t)
    return out


def save_combined_visualization(
    dataset_root: Path,
    scene_id: int,
    ann_id: int,
    camera: str,
    m1_name: str,
    m2_name: str,
    m1_arr: np.ndarray,
    m2_arr: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
    metric_info: Optional[Dict[str, Any]] = None,
) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required for visualization but is not available.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth = load_rgb_depth(dataset_root, scene_id, ann_id, camera)
    K = load_camK(dataset_root, scene_id, camera)
    base_pcd = backproject_depth_to_pcd(
        depth=depth,
        rgb=rgb,
        K=K,
        depth_scale=args.depth_scale,
        depth_trunc=args.depth_trunc,
        stride=args.pcd_stride,
    )
    num_scene_points_before_sampling = int(len(base_pcd.points))
    sample_seed = int(args.save_pcd_random_seed) + int(scene_id) * 100000 + int(ann_id)
    base_pcd = sample_pcd_points(base_pcd, int(args.save_pcd_num_points), seed=sample_seed)
    num_scene_points_after_sampling = int(len(base_pcd.points))

    bbox = base_pcd.get_axis_aligned_bounding_box()
    extent_x = max(float(bbox.get_extent()[0]), 1e-6)
    shift_x = extent_x + float(args.side_by_side_gap)

    left_pcd = tint_pcd(base_pcd, (1.0, 0.35, 0.35), strength=0.30)
    right_pcd = tint_pcd(translate_geometry(base_pcd, np.array([shift_x, 0.0, 0.0], dtype=np.float64)), (0.35, 0.35, 1.0), strength=0.30)

    left_grasp = make_grasp_samples(m1_arr, args.grasp_mesh_sample_points)
    right_grasp = make_grasp_samples(m2_arr, args.grasp_mesh_sample_points)
    right_grasp.translate((shift_x, 0.0, 0.0))

    combo_all = left_pcd + right_pcd + left_grasp + right_grasp

    tag = f"scene_{scene_id:04d}_ann_{ann_id:04d}_{m1_name}_vs_{m2_name}"
    ply_path = out_dir / f"{tag}_all_top{args.vis_topk_grasps}_nms.ply"
    o3d.io.write_point_cloud(str(ply_path), combo_all)

    cfg = {
        "ply_file": str(ply_path),
        "scene_id": int(scene_id),
        "ann_id": int(ann_id),
        "camera": camera,
        "selection": {
            "group": (metric_info or {}).get("selection_group"),
            "description": (metric_info or {}).get("selection_description"),
            "delta_sort": (metric_info or {}).get("selection_delta_sort"),
        },
        "layout": "side_by_side_along_x",
        "red_left_point_cloud": {
            "method": m1_name,
            "description": "scene point cloud tinted red; grasps preserve original GraspNetAPI score colors",
            "translation_xyz": [0.0, 0.0, 0.0],
            "point_cloud_tint_rgb": [1.0, 0.35, 0.35],
        },
        "blue_right_point_cloud": {
            "method": m2_name,
            "description": "same scene point cloud shifted along +x and tinted blue; grasps preserve original GraspNetAPI score colors",
            "translation_xyz": [float(shift_x), 0.0, 0.0],
            "point_cloud_tint_rgb": [0.35, 0.35, 1.0],
        },
        "grasp_color_policy": "preserve original colors from GraspGroup.to_open3d_geometry_list; no uniform red/blue repainting",
        "ap_delta_definition": "blue/right method minus red/left method",
        "metrics": metric_info or {},
        "visualization_settings": {
            "vis_topk_grasps_after_sort_nms": int(args.vis_topk_grasps),
            "grasp_mesh_sample_points_per_grasp": int(args.grasp_mesh_sample_points),
            "pcd_stride": int(args.pcd_stride),
            "save_pcd_num_points_per_side": int(args.save_pcd_num_points),
            "save_pcd_random_seed": int(args.save_pcd_random_seed),
            "num_scene_points_before_sampling": num_scene_points_before_sampling,
            "num_scene_points_after_sampling_per_side": num_scene_points_after_sampling,
            "num_method1_grasps_visualized": int(m1_arr.shape[0]),
            "num_method2_grasps_visualized": int(m2_arr.shape[0]),
            "side_by_side_gap": float(args.side_by_side_gap),
        },
    }
    with open(ply_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def split_to_scene_id(split: str, local_scene_index: int) -> int:
    return int(SPLIT_SCENE_START.get(split, 0) + local_scene_index)


def build_original_anno_lookup(
    anno_len: int,
    sample_info: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[int], str]:
    """Map local anno indices in a parsed/evaluated AP array to original GraspNet anno ids.

    Why this is needed:
    - When sample_interval=0.1 is applied to a full 256-anno AP array, the parsed
      per-anno table has local indices 0..25, but the corresponding original anno ids
      are 0,10,20,...,250.
    - When the AP file is already sampled, maybe_apply_sample_interval() intentionally
      skips resampling. If the sampled length matches the requested sample interval,
      we still infer the same original anno ids for visualization.
    """
    anno_len = int(anno_len)
    if anno_len <= 0:
        return [], "empty"

    sample_indices = sample_info.get("sample_indices")
    if bool(sample_info.get("sample_applied", False)) and isinstance(sample_indices, list):
        lookup = [int(x) for x in sample_indices]
        if len(lookup) >= anno_len:
            return lookup[:anno_len], "sample_info.sample_indices"

    # If the AP file appears to be already sampled, infer the mapping from the requested
    # sample interval. This fixes the common case: local 0..25 -> original 0,10,...,250.
    if float(args.sample_interval) < 1.0:
        candidate = _build_sample_indices(
            int(args.annos_per_scene),
            float(args.sample_interval),
            int(args.sample_offset),
        )
        if len(candidate) == anno_len:
            return [int(x) for x in candidate], "inferred_from_sample_interval"

    return list(range(anno_len)), "identity"


def make_bar_plot(df: pd.DataFrame, method1: str, method2: str, out_path: Path) -> None:
    plot_df = df[df["metric"].isin(["AP", "AP0.8", "AP0.4"])].copy()
    if plot_df.empty:
        return
    labels = [f"{row.split_short}\n{row.metric}" for row in plot_df.itertuples(index=False)]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(9, len(labels) * 0.55), 4.8))
    plt.bar(x - width / 2, plot_df["method1_value"].to_numpy(), width, label=method1)
    plt.bar(x + width / 2, plot_df["method2_value"].to_numpy(), width, label=method2)
    plt.xticks(x, labels, rotation=0, fontsize=8)
    plt.ylabel("AP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_delta_plot(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[df["metric"].isin(["AP", "AP0.8", "AP0.4"])].copy()
    if plot_df.empty:
        return
    labels = [f"{row.split_short}\n{row.metric}" for row in plot_df.itertuples(index=False)]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(9, len(labels) * 0.55), 4.5))
    plt.axhline(0, color="black", linewidth=1)
    plt.bar(x, plot_df["delta_m2_minus_m1"].to_numpy())
    plt.xticks(x, labels, rotation=0, fontsize=8)
    plt.ylabel("method2 - method1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_scene_delta_plot(scene_df: pd.DataFrame, out_path: Path) -> None:
    if scene_df.empty:
        return
    df = scene_df.sort_values("delta_AP_m2_minus_m1")
    x = np.arange(len(df))
    plt.figure(figsize=(14, 4.5))
    plt.axhline(0, color="black", linewidth=1)
    plt.bar(x, df["delta_AP_m2_minus_m1"].to_numpy())
    plt.xticks(x, df["scene_id"].astype(str).to_list(), rotation=90, fontsize=6)
    plt.ylabel("Scene AP delta (method2 - method1)")
    plt.xlabel("Scene ID")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    method1_root = Path(args.method1_eval_root)
    method2_root = Path(args.method2_eval_root)

    summary_rows: List[Dict[str, Any]] = []
    per_scene_rows: List[pd.DataFrame] = []
    per_anno_rows: List[pd.DataFrame] = []
    overview: Dict[str, Any] = {
        "method1": args.method1_name,
        "method1_eval_root": str(method1_root),
        "method2": args.method2_name,
        "method2_eval_root": str(method2_root),
        "camera": args.camera,
        "sample_interval": float(args.sample_interval),
        "sample_offset": int(args.sample_offset),
        "annos_per_scene": int(args.annos_per_scene),
        "force_resample": bool(args.force_resample),
        "friction_axis": int(args.friction_axis),
        "anno_axis": int(args.anno_axis),
        "scene_axis": int(args.scene_axis),
        "ap04_index": int(args.ap04_index),
        "ap08_index": int(args.ap08_index),
        "save_pcd_num_points": int(args.save_pcd_num_points),
        "save_pcd_random_seed": int(args.save_pcd_random_seed),
        "splits": {},
    }

    for split in args.splits:
        p1 = _ap_file(method1_root, split, args.camera)
        p2 = _ap_file(method2_root, split, args.camera)
        arr1_raw = _safe_load_npy(p1)
        arr2_raw = _safe_load_npy(p2)

        arr1, sample_info1 = maybe_apply_sample_interval(arr1_raw, args, split, args.method1_name)
        arr2, sample_info2 = maybe_apply_sample_interval(arr2_raw, args, split, args.method2_name)

        metrics1, meta1, scene1, anno1 = parse_ap_array(arr1, args)
        metrics2, meta2, scene2, anno2 = parse_ap_array(arr2, args)

        split_short = SPLIT_SHORT[split]
        overview["splits"][split] = {
            "method1_file": str(p1),
            "method2_file": str(p2),
            "method1_raw_shape": list(arr1_raw.shape),
            "method2_raw_shape": list(arr2_raw.shape),
            "method1_sample_info": sample_info1,
            "method2_sample_info": sample_info2,
            "method1_parse_meta": meta1,
            "method2_parse_meta": meta2,
        }

        metric_keys = sorted(set(metrics1.keys()) | set(metrics2.keys()))
        metric_keys = [k for k in ["AP", "AP0.8", "AP0.4"] if k in metric_keys] + [
            k for k in metric_keys if k not in {"AP", "AP0.8", "AP0.4"}
        ]
        for metric in metric_keys:
            v1 = float(metrics1.get(metric, math.nan))
            v2 = float(metrics2.get(metric, math.nan))
            summary_rows.append({
                "split": split,
                "split_short": split_short,
                "metric": metric,
                "method1": args.method1_name,
                "method2": args.method2_name,
                "method1_value": v1,
                "method2_value": v2,
                "delta_m2_minus_m1": v2 - v1,
                "relative_delta_m2_vs_m1": (v2 - v1) / v1 if np.isfinite(v1) and abs(v1) > 1e-12 else math.nan,
            })

        if scene1 is not None and scene2 is not None and len(scene1) == len(scene2):
            s = scene1.merge(scene2, on="local_scene_index", suffixes=("_m1", "_m2"))
            s.insert(0, "split", split)
            s.insert(1, "split_short", split_short)
            s["scene_id"] = [split_to_scene_id(split, int(x)) for x in s["local_scene_index"]]
            s["delta_AP_m2_minus_m1"] = s["AP_m2"] - s["AP_m1"]
            s["delta_AP0.8_m2_minus_m1"] = s["AP0.8_m2"] - s["AP0.8_m1"]
            s["delta_AP0.4_m2_minus_m1"] = s["AP0.4_m2"] - s["AP0.4_m1"]
            per_scene_rows.append(s)

        if anno1 is not None and anno2 is not None and len(anno1) == len(anno2):
            a = anno1.merge(anno2, on=["local_scene_index", "local_anno_index"], suffixes=("_m1", "_m2"))
            a.insert(0, "split", split)
            a.insert(1, "split_short", split_short)
            a["scene_id"] = [split_to_scene_id(split, int(x)) for x in a["local_scene_index"]]

            anno_len_after_parse = int(a["local_anno_index"].max()) + 1 if len(a) else 0
            orig_lookup1, lookup_source1 = build_original_anno_lookup(anno_len_after_parse, sample_info1, args)
            orig_lookup2, lookup_source2 = build_original_anno_lookup(anno_len_after_parse, sample_info2, args)
            if orig_lookup1 != orig_lookup2:
                overview["splits"][split]["anno_mapping_warning"] = (
                    "method1 and method2 inferred different original anno mappings; "
                    "using method1 mapping for visualization."
                )
            overview["splits"][split]["original_anno_mapping"] = {
                "method1_source": lookup_source1,
                "method2_source": lookup_source2,
                "lookup_used": orig_lookup1,
            }

            a["eval_local_anno_index"] = a["local_anno_index"].astype(np.int64)
            a["orig_ann_id"] = [
                int(orig_lookup1[int(x)]) if int(x) < len(orig_lookup1) else int(x)
                for x in a["local_anno_index"]
            ]
            # ann_id is used to locate RGB/depth/grasp files, so it must be the original
            # GraspNet annotation id, not the local index in the sampled AP array.
            a["ann_id"] = a["orig_ann_id"].astype(np.int64)
            a["anno_mapping_source"] = lookup_source1

            a["delta_AP_m2_minus_m1"] = a["AP_m2"] - a["AP_m1"]
            a["delta_AP0.8_m2_minus_m1"] = a["AP0.8_m2"] - a["AP0.8_m1"]
            a["delta_AP0.4_m2_minus_m1"] = a["AP0.4_m2"] - a["AP0.4_m1"]
            per_anno_rows.append(a)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "ap_summary.csv", index=False)

    paper_rows: List[Dict[str, Any]] = []
    for method_label, value_col in [(args.method1_name, "method1_value"), (args.method2_name, "method2_value")]:
        vec: List[float] = []
        for split in args.splits:
            for metric in ["AP", "AP0.8", "AP0.4"]:
                r = summary_df[(summary_df["split"] == split) & (summary_df["metric"] == metric)]
                vec.append(float(r.iloc[0][value_col]) if not r.empty else math.nan)
        split_ap_values = []
        for split in args.splits:
            r = summary_df[(summary_df["split"] == split) & (summary_df["metric"] == "AP")]
            if not r.empty:
                split_ap_values.append(float(r.iloc[0][value_col]))
        vec.append(float(np.nanmean(split_ap_values)) if split_ap_values else math.nan)
        paper_rows.append({"method": method_label, "paper_style_vector": vec})
        overview[f"paper_style_vector_{method_label}"] = vec

    with open(out_root / "paper_style_vectors.json", "w", encoding="utf-8") as f:
        json.dump(paper_rows, f, indent=2, ensure_ascii=False)

    main_df = summary_df[summary_df["metric"].isin(["AP", "AP0.8", "AP0.4"])].copy()
    main_df.to_csv(out_root / "ap_main_metrics.csv", index=False)

    if per_scene_rows:
        scene_df = pd.concat(per_scene_rows, ignore_index=True)
        scene_df.to_csv(out_root / "per_scene_ap.csv", index=False)
        overview["scene_delta_AP_m2_minus_m1"] = {
            "mean": float(scene_df["delta_AP_m2_minus_m1"].mean()),
            "median": float(scene_df["delta_AP_m2_minus_m1"].median()),
            "min": float(scene_df["delta_AP_m2_minus_m1"].min()),
            "max": float(scene_df["delta_AP_m2_minus_m1"].max()),
            "num_scene_m2_better": int((scene_df["delta_AP_m2_minus_m1"] > 0).sum()),
            "num_scene_m1_better": int((scene_df["delta_AP_m2_minus_m1"] < 0).sum()),
        }
    else:
        scene_df = pd.DataFrame()

    if per_anno_rows:
        anno_df = pd.concat(per_anno_rows, ignore_index=True)
        anno_df.to_csv(out_root / "per_anno_ap.csv", index=False)
        overview["anno_delta_AP_m2_minus_m1"] = {
            "mean": float(anno_df["delta_AP_m2_minus_m1"].mean()),
            "median": float(anno_df["delta_AP_m2_minus_m1"].median()),
            "min": float(anno_df["delta_AP_m2_minus_m1"].min()),
            "max": float(anno_df["delta_AP_m2_minus_m1"].max()),
            "num_anno_m2_better": int((anno_df["delta_AP_m2_minus_m1"] > 0).sum()),
            "num_anno_m1_better": int((anno_df["delta_AP_m2_minus_m1"] < 0).sum()),
        }
    else:
        anno_df = pd.DataFrame()

    # Optional visualization: for each scene, save two groups of samples:
    #   1) top_method2_worse  : method2/blue/right is worse than method1/red/left, most negative delta first.
    #   2) top_method2_better : method2/blue/right is better than method1/red/left, most positive delta first.
    # Here delta means AP(method2) - AP(method1).
    vis_enabled = (
        (not args.skip_vis)
        and bool(args.dataset_root)
        and bool(args.method1_grasp_root)
        and bool(args.method2_grasp_root)
        and (not anno_df.empty)
    )
    vis_records: List[Dict[str, Any]] = []
    if vis_enabled:
        dataset_root = Path(args.dataset_root)
        method1_grasp_root = Path(args.method1_grasp_root)
        method2_grasp_root = Path(args.method2_grasp_root)
        vis_root = out_root / "visualizations"

        vis_topk_per_scene = (
            int(args.vis_topk_scene_samples)
            if args.vis_topk_scene_samples is not None
            else int(args.vis_topk_scene_worst)
        )

        selection_specs = [
            {
                "name": "top_method2_worse",
                "description": "method2/blue/right AP is lower than method1/red/left; sorted by most negative delta_AP_m2_minus_m1",
                "ascending": True,
                "sign": "negative",
            },
            {
                "name": "top_method2_better",
                "description": "method2/blue/right AP is higher than method1/red/left; sorted by most positive delta_AP_m2_minus_m1",
                "ascending": False,
                "sign": "positive",
            },
        ]

        for spec in selection_specs:
            group_name = str(spec["name"])
            group_records: List[Dict[str, Any]] = []
            group_root = vis_root / group_name
            for scene_id, g in anno_df.groupby("scene_id"):
                if spec["sign"] == "negative":
                    candidates = g[g["delta_AP_m2_minus_m1"] < 0].copy()
                else:
                    candidates = g[g["delta_AP_m2_minus_m1"] > 0].copy()
                if candidates.empty:
                    continue
                selected = candidates.sort_values(
                    "delta_AP_m2_minus_m1",
                    ascending=bool(spec["ascending"]),
                ).head(vis_topk_per_scene)

                for _, row in selected.iterrows():
                    ann_id = int(row.get("orig_ann_id", row.get("ann_id")))
                    local_anno_index = int(row.get("eval_local_anno_index", row.get("local_anno_index", ann_id)))
                    g1 = resolve_path(method1_grasp_root, int(scene_id), ann_id, args.camera, DEFAULT_GRASP_TEMPLATES)
                    g2 = resolve_path(method2_grasp_root, int(scene_id), ann_id, args.camera, DEFAULT_GRASP_TEMPLATES)
                    metric_info = {
                        "selection_group": group_name,
                        "selection_description": str(spec["description"]),
                        "selection_delta_sort": "ascending" if bool(spec["ascending"]) else "descending",
                        "red_left_method": args.method1_name,
                        "blue_right_method": args.method2_name,
                        "eval_local_anno_index": local_anno_index,
                        "orig_ann_id": ann_id,
                        "ann_id_used_for_files": ann_id,
                        "anno_mapping_source": row.get("anno_mapping_source", ""),
                        "AP_red_left": _json_float(row.get("AP_m1")),
                        "AP_blue_right": _json_float(row.get("AP_m2")),
                        "delta_AP_blue_minus_red": _json_float(row.get("delta_AP_m2_minus_m1")),
                        "AP0.8_red_left": _json_float(row.get("AP0.8_m1")),
                        "AP0.8_blue_right": _json_float(row.get("AP0.8_m2")),
                        "delta_AP0.8_blue_minus_red": _json_float(row.get("delta_AP0.8_m2_minus_m1")),
                        "AP0.4_red_left": _json_float(row.get("AP0.4_m1")),
                        "AP0.4_blue_right": _json_float(row.get("AP0.4_m2")),
                        "delta_AP0.4_blue_minus_red": _json_float(row.get("delta_AP0.4_m2_minus_m1")),
                    }
                    base_record = {
                        "selection_group": group_name,
                        "selection_description": str(spec["description"]),
                        "scene_id": int(scene_id),
                        "local_anno_index": local_anno_index,
                        "orig_ann_id": ann_id,
                        "ann_id": ann_id,
                    }
                    if g1 is None or g2 is None:
                        rec = {
                            **base_record,
                            "status": "missing_grasp_file",
                            "method1_grasp_path": str(g1) if g1 is not None else "",
                            "method2_grasp_path": str(g2) if g2 is not None else "",
                            **metric_info,
                        }
                        vis_records.append(rec)
                        group_records.append(rec)
                        continue
                    try:
                        arr1 = sort_nms_topk_grasp_array(load_grasp_array(g1), int(args.vis_topk_grasps))
                        arr2 = sort_nms_topk_grasp_array(load_grasp_array(g2), int(args.vis_topk_grasps))
                        save_combined_visualization(
                            dataset_root=dataset_root,
                            scene_id=int(scene_id),
                            ann_id=ann_id,
                            camera=args.camera,
                            m1_name=args.method1_name,
                            m2_name=args.method2_name,
                            m1_arr=arr1,
                            m2_arr=arr2,
                            out_dir=group_root / f"scene_{int(scene_id):04d}",
                            args=args,
                            metric_info=metric_info,
                        )
                        rec = {
                            **base_record,
                            "status": "ok",
                            "method1_grasp_path": str(g1),
                            "method2_grasp_path": str(g2),
                            **metric_info,
                        }
                        vis_records.append(rec)
                        group_records.append(rec)
                    except Exception as e:
                        rec = {
                            **base_record,
                            "status": f"error: {repr(e)}",
                            "method1_grasp_path": str(g1),
                            "method2_grasp_path": str(g2),
                            **metric_info,
                        }
                        vis_records.append(rec)
                        group_records.append(rec)

            if group_records:
                pd.DataFrame(group_records).to_csv(out_root / f"visualization_records_{group_name}.csv", index=False)

        if vis_records:
            vis_df = pd.DataFrame(vis_records)
            vis_df.to_csv(out_root / "visualization_records.csv", index=False)
            overview["visualization"] = {
                "enabled": True,
                "dataset_root": str(dataset_root),
                "method1_grasp_root": str(method1_grasp_root),
                "method2_grasp_root": str(method2_grasp_root),
                "delta_definition": "method2/blue/right minus method1/red/left",
                "vis_topk_per_scene_per_group": int(vis_topk_per_scene),
                "anno_id_mapping": "ann_id/orig_ann_id used for file IO; local_anno_index kept for eval-array index",
                "groups": {
                    name: {
                        "num_records": int((vis_df["selection_group"] == name).sum()),
                        "num_ok": int(((vis_df["selection_group"] == name) & (vis_df["status"] == "ok")).sum()),
                    }
                    for name in sorted(vis_df["selection_group"].unique().tolist())
                },
                "num_records": len(vis_records),
                "num_ok": int(sum(1 for x in vis_records if x.get("status") == "ok")),
            }
        else:
            overview["visualization"] = {
                "enabled": True,
                "dataset_root": str(dataset_root),
                "method1_grasp_root": str(method1_grasp_root),
                "method2_grasp_root": str(method2_grasp_root),
                "num_records": 0,
                "num_ok": 0,
                "reason": "no positive/negative per-anno AP deltas found after sign filtering",
            }
    else:
        overview["visualization"] = {
            "enabled": False,
            "reason": "need dataset-root + method1-grasp-root + method2-grasp-root, and per-anno AP must be available",
        }

    with open(out_root / "overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)

    if not args.no_plots:
        make_bar_plot(summary_df, args.method1_name, args.method2_name, out_root / "ap_grouped_bars.png")
        make_delta_plot(summary_df, out_root / "ap_delta_bars.png")
        if not scene_df.empty:
            make_scene_delta_plot(scene_df, out_root / "scene_delta_bars.png")

    print("[DONE] Saved AP comparison to:", out_root)
    print("[SUMMARY]")
    print(main_df.to_string(index=False))
    print("[PAPER STYLE VECTORS]")
    for r in paper_rows:
        print(r["method"], r["paper_style_vector"])
    if overview.get("visualization", {}).get("enabled"):
        print("[VIS] Visualization records saved to:", out_root / "visualization_records.csv")


if __name__ == "__main__":
    main()
