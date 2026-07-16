#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate point-wise graspness labels for GraspClutter6D for ecograsp_dpt.

This is a debug-friendly version of the GC6D graspness generator. It keeps the
same default behavior as the previous generator, but adds optional diagnostics
for common failure modes:

1) background leakage: background/bin points receiving non-zero graspness from
   nearest-neighbor propagation;
2) KNN distance statistics: how far masked scene points are from their nearest
   object grasp-label point;
3) mask-count diagnostics: depth / label / bbox / workspace masks;
4) optional 2D debug panels, NPZ dumps, and colored PLYs;
5) optional switches to test fixes without changing the default behavior:
     --zero_background
     --knn_dist_thresh
     --final_norm {minmax,foreground_minmax,none}

Default output is unchanged:
    real depth, scene-level mode:
        ROOT/graspness/<scene_id:06d>/<camera>/<img_id:06d>.npy
    real depth, instance-normalized mode:
        ROOT/graspness_instance/<scene_id:06d>/<camera>/<img_id:06d>.npy
    virtual depth, scene-level mode:
        ROOT/virtual_graspness/<scene_id:06d>/<camera>/<img_id:06d>.npy
    virtual depth, instance-normalized mode:
        ROOT/virtual_graspness_instance/<scene_id:06d>/<camera>/<img_id:06d>.npy

Recommended debug command for one frame:
    python gc6d_generate_graspness_debug.py \
      --dataset_root /data/robotarm/dataset/GraspClutter6D \
      --scene_ids 5 --camera realsense-d435 --img_ids 2 \
      --depth_type real --mode instance \
      --mask_mode workspace_depth --workspace_outlier 0.02 --workspace_pose_mode json \
      --write_stats --debug_dir /tmp/gc6d_graspness_debug \
      --debug_save_npz --debug_save_2d --debug_save_ply
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


CAMERA_OFFSETS = {
    "realsense-d415": 1,
    "realsense-d435": 2,
    "azure-kinect": 3,
    "zivid": 4,
}
CAMERA_ALIASES = {
    "realsense": "realsense-d435",
    "kinect": "azure-kinect",
}
NUM_VIEWS = 300
NUM_ANGLES = 12
NUM_DEPTHS = 4


def canonical_camera(camera: str) -> str:
    camera = CAMERA_ALIASES.get(str(camera), str(camera))
    if camera not in CAMERA_OFFSETS:
        raise ValueError(f"Unknown camera={camera}. Expected one of {list(CAMERA_OFFSETS)}")
    return camera


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    return int(ann_id) * 4 + CAMERA_OFFSETS[canonical_camera(camera)]


def img_id_to_ann_id(img_id: int, camera: str) -> int:
    offset = CAMERA_OFFSETS[canonical_camera(camera)]
    img_id = int(img_id)
    if img_id < offset or ((img_id - offset) % 4) != 0:
        raise ValueError(f"img_id={img_id} does not belong to camera={camera}")
    return (img_id - offset) // 4


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out))


def read_split_scene_ids(dataset_root: Path, split: str) -> List[int]:
    split = str(split).lower()
    split_dir = dataset_root / "split_info"
    if split == "train":
        data = load_json(split_dir / "grasp_train_scene_ids.json")
    elif split == "test":
        data = load_json(split_dir / "grasp_test_scene_ids.json")
    elif split == "all":
        data = list(load_json(split_dir / "grasp_train_scene_ids.json"))
        data += list(load_json(split_dir / "grasp_test_scene_ids.json"))
    else:
        return parse_int_list(split)

    if isinstance(data, dict):
        vals = list(data.keys()) if all(str(k).isdigit() for k in data.keys()) else list(data.values())
    else:
        vals = list(data)
    return sorted({int(x) for x in vals})


def get_camera_img_ids(scene_camera: Dict[str, dict], camera: str, max_ann: Optional[int]) -> List[int]:
    camera = canonical_camera(camera)
    offset = CAMERA_OFFSETS[camera]
    img_ids: List[int] = []
    for k in scene_camera.keys():
        try:
            img_id = int(k)
        except Exception:
            continue
        if img_id >= offset and ((img_id - offset) % 4) == 0:
            ann_id = (img_id - offset) // 4
            if max_ann is None or ann_id < int(max_ann):
                img_ids.append(img_id)
    return sorted(img_ids)


def safe_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < eps:
        if x_max > eps:
            return np.ones_like(x, dtype=np.float32)
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def foreground_minmax(values: np.ndarray, fg_mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    fg_mask = np.asarray(fg_mask, dtype=bool).reshape(-1)
    out = values.copy()
    valid = fg_mask & (out.reshape(-1) > eps)
    out[:] = 0.0
    if np.any(valid):
        out[valid, 0] = safe_minmax(values[valid, 0], eps=eps)
    return out.astype(np.float32)


def get_factor_depth(camera: str, cam_info: dict, mode: str, fixed: Optional[float]) -> float:
    """Return denominator so that depth_m = raw_depth / factor_depth."""
    mode = str(mode).lower()
    camera = canonical_camera(camera)
    if mode == "fixed":
        if fixed is None:
            raise ValueError("--fixed_factor_depth is required when --factor_depth_mode fixed")
        return float(fixed)
    if mode == "camera":
        if camera in ["realsense-d415", "realsense-d435"]:
            return 1000.0
        if camera in ["azure-kinect", "zivid"]:
            return 10000.0
        raise ValueError(camera)
    if mode == "bop":
        ds = float(cam_info.get("depth_scale", 1.0))
        if ds <= 0:
            raise ValueError(f"Invalid depth_scale={ds}")
        return 1000.0 / ds
    raise ValueError(f"Unknown factor_depth_mode={mode}")


def depth_to_cloud(depth_raw: np.ndarray, K: np.ndarray, factor_depth: float) -> np.ndarray:
    depth = depth_raw.astype(np.float32) / float(factor_depth)
    H, W = depth.shape[:2]
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth
    x = (xs - float(K[0, 2])) / float(K[0, 0]) * z
    y = (ys - float(K[1, 2])) / float(K[1, 1]) * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def pose_from_scene_camera(cam_info: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(cam_info["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(cam_info["cam_t_w2c"], dtype=np.float32).reshape(3) / 1000.0
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    return (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def get_camera_to_workspace_transform(cam_info: dict, mode: str) -> Optional[np.ndarray]:
    mode = str(mode).lower()
    if mode == "none":
        return None
    T = pose_from_scene_camera(cam_info)
    if mode == "json":
        return T.astype(np.float32)
    if mode == "inverse":
        return np.linalg.inv(T).astype(np.float32)
    raise ValueError(f"Unsupported workspace_pose_mode={mode}")


def build_workspace_mask(
    cloud: np.ndarray,
    label: np.ndarray,
    depth_mask: np.ndarray,
    cam_info: dict,
    outlier: float,
    pose_mode: str,
    depth_trunc: float,
) -> np.ndarray:
    if cloud.shape[:2] != depth_mask.shape or label.shape != depth_mask.shape:
        raise ValueError(
            f"workspace shape mismatch: cloud={cloud.shape}, label={label.shape}, depth_mask={depth_mask.shape}"
        )
    valid = depth_mask.copy()
    if depth_trunc is not None and float(depth_trunc) > 0:
        valid &= (cloud[..., 2] > 0) & (cloud[..., 2] < float(depth_trunc))
    fg = valid & (label > 0)
    if not np.any(fg):
        return valid

    H, W = depth_mask.shape
    cloud_flat = cloud.reshape(-1, 3).astype(np.float32)
    T = get_camera_to_workspace_transform(cam_info, pose_mode)
    cloud_box_flat = transform_points(cloud_flat, T) if T is not None else cloud_flat

    fg_points = cloud_box_flat[fg.reshape(-1)]
    xyz_min = fg_points.min(axis=0) - float(outlier)
    xyz_max = fg_points.max(axis=0) + float(outlier)
    in_box = (
        (cloud_box_flat[:, 0] > xyz_min[0]) & (cloud_box_flat[:, 0] < xyz_max[0]) &
        (cloud_box_flat[:, 1] > xyz_min[1]) & (cloud_box_flat[:, 1] < xyz_max[1]) &
        (cloud_box_flat[:, 2] > xyz_min[2]) & (cloud_box_flat[:, 2] < xyz_max[2])
    )
    return (valid.reshape(-1) & in_box).reshape(H, W)


def load_label(path: Path) -> np.ndarray:
    label = np.array(Image.open(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def pose_from_gc6d_obj(obj: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(obj["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
    return T


def load_scene_object_list(scene_gt: Dict[str, list], img_id: int) -> Tuple[List[int], List[np.ndarray]]:
    objs = scene_gt[str(int(img_id))]
    obj_list: List[int] = []
    pose_list: List[np.ndarray] = []
    for obj in objs:
        obj_list.append(int(obj["obj_id"]))
        pose_list.append(pose_from_gc6d_obj(obj))
    return obj_list, pose_list


def load_grasp_label(dataset_root: Path, obj_id: int, cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    obj_id = int(obj_id)
    if obj_id not in cache:
        path = dataset_root / "grasp_label" / f"obj_{obj_id:06d}_labels.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        cache[obj_id] = (
            data["points"].astype(np.float32),
            data["offsets"].astype(np.float32),
            data["scores"].astype(np.float32),
        )
    return cache[obj_id]


def load_collision_dump(dataset_root: Path, scene_id: int) -> List[np.ndarray]:
    path = dataset_root / "collision_label" / f"{int(scene_id):06d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = np.load(path)
    dump = []
    for j in range(len(labels.files)):
        dump.append(labels[f"arr_{j}"])
    return dump


def build_mask(
    depth: np.ndarray,
    label: np.ndarray,
    cloud: np.ndarray,
    cam_info: dict,
    mode: str,
    bbox_margin_ratio: float,
    workspace_outlier: float,
    workspace_pose_mode: str,
    workspace_depth_trunc: float,
) -> np.ndarray:
    depth_mask = depth > 0
    mode = str(mode).lower()
    if mode == "depth":
        return depth_mask
    if mode == "label_depth":
        return depth_mask & (label > 0)
    if mode == "bbox_depth":
        H, W = depth.shape[:2]
        fg = label > 0
        if not np.any(fg):
            return depth_mask
        ys, xs = np.where(fg)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mx = int(round(W * float(bbox_margin_ratio)))
        my = int(round(H * float(bbox_margin_ratio)))
        x0 = max(0, x0 - mx)
        x1 = min(W, x1 + mx)
        y0 = max(0, y0 - my)
        y1 = min(H, y1 + my)
        mask = np.zeros_like(depth_mask, dtype=bool)
        mask[y0:y1, x0:x1] = True
        return depth_mask & mask
    if mode == "workspace_depth":
        return build_workspace_mask(
            cloud=cloud,
            label=label,
            depth_mask=depth_mask,
            cam_info=cam_info,
            outlier=float(workspace_outlier),
            pose_mode=workspace_pose_mode,
            depth_trunc=float(workspace_depth_trunc),
        )
    raise ValueError(f"Unsupported mask_mode={mode}")


def get_mask_counts(depth: np.ndarray, label: np.ndarray, cloud: np.ndarray, cam_info: dict, cfg) -> Dict[str, int]:
    out = {}
    for mode in ["depth", "label_depth", "bbox_depth", "workspace_depth"]:
        try:
            out[f"mask_count_{mode}"] = int(build_mask(
                depth=depth,
                label=label,
                cloud=cloud,
                cam_info=cam_info,
                mode=mode,
                bbox_margin_ratio=cfg.bbox_margin_ratio,
                workspace_outlier=cfg.workspace_outlier,
                workspace_pose_mode=cfg.workspace_pose_mode,
                workspace_depth_trunc=cfg.workspace_depth_trunc,
            ).sum())
        except Exception:
            out[f"mask_count_{mode}"] = -1
    return out


def get_save_root(dataset_root: Path, depth_type: str, mode: str, save_root: Optional[str]) -> Path:
    if save_root is not None:
        return Path(save_root)
    if depth_type == "virtual":
        base = "virtual_graspness"
    elif depth_type == "real":
        base = "graspness"
    else:
        raise ValueError(f"Unsupported depth_type={depth_type}")
    if mode == "instance":
        base += "_instance"
    return dataset_root / base


def resolve_depth_path(dataset_root: Path, virtual_root: Path, scene_id: int, camera: str, img_id: int, depth_type: str) -> Path:
    s6 = f"{int(scene_id):06d}"
    if depth_type == "real":
        return dataset_root / "scenes" / s6 / "depth" / f"{int(img_id):06d}.png"
    if depth_type == "virtual":
        candidates = [
            virtual_root / s6 / canonical_camera(camera) / f"{int(img_id):06d}_depth.png",
            virtual_root / s6 / f"{int(img_id):06d}_depth.png",
            virtual_root / s6 / canonical_camera(camera) / f"{int(img_id):06d}.png",
            virtual_root / s6 / f"{int(img_id):06d}.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]
    raise ValueError(f"Unsupported depth_type={depth_type}")


def resolve_rgb_path(dataset_root: Path, scene_id: int, img_id: int) -> Path:
    return dataset_root / "scenes" / f"{int(scene_id):06d}" / "rgb" / f"{int(img_id):06d}.png"


def nearest_graspness_scipy_with_dist(
    cloud_points: np.ndarray,
    grasp_points: np.ndarray,
    graspness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree
    tree = cKDTree(grasp_points.astype(np.float32))
    dists, inds = tree.query(cloud_points.astype(np.float32), k=1, workers=-1)
    return graspness[np.asarray(inds, dtype=np.int64)].astype(np.float32), dists.reshape(-1, 1).astype(np.float32)


def nearest_graspness_torch_with_dist(
    cloud_points: np.ndarray,
    grasp_points: np.ndarray,
    graspness: np.ndarray,
    device: str,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    dev = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    gp = torch.from_numpy(grasp_points.astype(np.float32)).to(dev)
    gv = torch.from_numpy(graspness.astype(np.float32)).to(dev)
    out_g = []
    out_d = []
    for start in range(0, cloud_points.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), cloud_points.shape[0])
        cp = torch.from_numpy(cloud_points[start:end].astype(np.float32)).to(dev)
        dist = torch.cdist(cp[None], gp[None], p=2).squeeze(0)
        min_d, inds = torch.min(dist, dim=1)
        out_g.append(gv[inds].detach().cpu().numpy())
        out_d.append(min_d.detach().cpu().numpy().reshape(-1, 1))
        del cp, dist, min_d, inds
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(out_g, axis=0).astype(np.float32), np.concatenate(out_d, axis=0).astype(np.float32)


def assign_nearest_graspness_with_dist(
    cloud_points: np.ndarray,
    grasp_points: np.ndarray,
    graspness: np.ndarray,
    backend: str,
    torch_device: str,
    torch_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if cloud_points.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    if grasp_points.shape[0] == 0:
        return np.zeros((cloud_points.shape[0], 1), dtype=np.float32), np.full((cloud_points.shape[0], 1), np.inf, dtype=np.float32)
    backend = str(backend).lower()
    if backend == "scipy":
        return nearest_graspness_scipy_with_dist(cloud_points, grasp_points, graspness)
    if backend == "torch":
        return nearest_graspness_torch_with_dist(cloud_points, grasp_points, graspness, torch_device, torch_chunk_size)
    if backend == "auto":
        try:
            return nearest_graspness_scipy_with_dist(cloud_points, grasp_points, graspness)
        except Exception:
            return nearest_graspness_torch_with_dist(cloud_points, grasp_points, graspness, torch_device, torch_chunk_size)
    raise ValueError(f"Unknown knn_backend={backend}")


def stat_prefix(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
        }
    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
    }


def make_label_color(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.int32)
    H, W = label.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    ids = np.unique(label)
    for idx in ids:
        if idx <= 0:
            continue
        rng = np.random.default_rng(int(idx) * 9973)
        out[label == idx] = rng.integers(40, 256, size=3, dtype=np.uint8)
    return out


def colorize_map(x: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    try:
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)
        y = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
        return (cmap(y)[..., :3] * 255).astype(np.uint8)
    except Exception:
        y = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
        return np.stack([y, np.zeros_like(y), 1.0 - y], axis=-1).astype(np.float32) * 255


def overlay(rgb: np.ndarray, heat: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    heat = heat.astype(np.float32)
    a = np.zeros((*mask.shape, 1), dtype=np.float32)
    a[mask] = float(alpha)
    return np.clip(rgb * (1 - a) + heat * a, 0, 255).astype(np.uint8)


def title_bar(img: np.ndarray, title: str, bar_h: int = 28) -> np.ndarray:
    im = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    canvas = Image.new("RGB", (im.width, im.height + bar_h), (255, 255, 255))
    canvas.paste(im, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(0, 0, 0))
    return np.array(canvas)


def save_debug_2d_panel(
    path: Path,
    rgb: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    graspness_final: np.ndarray,
    nn_dist: np.ndarray,
    high_thresh: float,
    cmap: str,
    overlay_alpha: float,
) -> None:
    H, W = mask.shape
    gmap = np.zeros((H, W), dtype=np.float32)
    dmap = np.zeros((H, W), dtype=np.float32)
    gmap[mask] = graspness_final.reshape(-1)
    d = nn_dist.reshape(-1)
    if np.any(np.isfinite(d)):
        finite = d[np.isfinite(d)]
        d_scale = max(float(np.percentile(finite, 99)), 1e-6)
        d_norm = np.clip(d / d_scale, 0, 1)
    else:
        d_norm = np.zeros_like(d)
    dmap[mask] = d_norm
    heat = colorize_map(gmap, cmap)
    dist_heat = colorize_map(dmap, "magma")
    label_vis = make_label_color(label)
    mask_vis = np.zeros_like(rgb, dtype=np.uint8)
    mask_vis[mask] = np.array([255, 255, 255], dtype=np.uint8)
    bg_high = ((label <= 0) & mask & (gmap > float(high_thresh)))
    leak_vis = rgb.copy()
    leak_vis[bg_high] = np.array([255, 0, 0], dtype=np.uint8)
    ov = overlay(rgb, heat, mask, overlay_alpha)
    panels = [
        title_bar(rgb, "RGB"),
        title_bar(label_vis, "Label"),
        title_bar(mask_vis, "Mask"),
        title_bar(heat, "Graspness"),
        title_bar(ov, "Overlay"),
        title_bar(dist_heat, "NN distance"),
        title_bar(leak_vis, f"BG high > {high_thresh}"),
    ]
    gap = 6
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x = 0
    for p in panels:
        canvas[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def save_debug_ply(path: Path, cloud_points: np.ndarray, graspness_final: np.ndarray, max_points: int, cmap: str) -> None:
    import open3d as o3d
    pts = cloud_points.astype(np.float32)
    g = np.asarray(graspness_final, dtype=np.float32).reshape(-1)
    if max_points and max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        g = g[idx]
    colors = colorize_map(g, cmap).reshape(-1, 3).astype(np.float32) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


@dataclass
class RunStats:
    generated: int = 0
    skipped_done: int = 0
    empty_cloud: int = 0
    empty_grasp: int = 0
    missing: int = 0
    failed: int = 0

    def add(self, d: Dict[str, int]) -> None:
        self.generated += int(d.get("generated", 0))
        self.skipped_done += int(d.get("skipped_done", 0))
        self.empty_cloud += int(d.get("empty_cloud", 0))
        self.empty_grasp += int(d.get("empty_grasp", 0))
        self.missing += int(d.get("missing", 0))
        self.failed += int(d.get("failed", 0))

    def as_dict(self) -> Dict[str, int]:
        return dict(
            generated=self.generated,
            skipped_done=self.skipped_done,
            empty_cloud=self.empty_cloud,
            empty_grasp=self.empty_grasp,
            missing=self.missing,
            failed=self.failed,
        )


def normalize_graspness(values: np.ndarray, label_masked: np.ndarray, final_norm: str) -> np.ndarray:
    final_norm = str(final_norm).lower()
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if final_norm == "minmax":
        return safe_minmax(values).reshape(-1, 1).astype(np.float32)
    if final_norm == "foreground_minmax":
        return foreground_minmax(values, label_masked.reshape(-1) > 0).astype(np.float32)
    if final_norm == "none":
        return values.astype(np.float32)
    raise ValueError(f"Unknown final_norm={final_norm}")


def process_one_scene(scene_id: int, cfg) -> Dict[str, object]:
    stats = RunStats()
    rows = []
    dataset_root = Path(cfg.dataset_root)
    virtual_root = Path(cfg.virtual_dataset_root) if cfg.virtual_dataset_root else (dataset_root / "virtual_scenes")
    save_root = get_save_root(dataset_root, cfg.depth_type, cfg.mode, cfg.save_root)
    debug_root = Path(cfg.debug_dir) if cfg.debug_dir else (save_root / "debug")
    camera = canonical_camera(cfg.camera)
    scene_dir = dataset_root / "scenes" / f"{int(scene_id):06d}"
    grasp_label_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    try:
        scene_gt_path = scene_dir / "scene_gt.json"
        scene_camera_path = scene_dir / "scene_camera.json"
        if not scene_gt_path.exists() or not scene_camera_path.exists():
            stats.missing += 1
            return {"ok": False, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": "missing scene_gt or scene_camera"}

        scene_gt = load_json(scene_gt_path)
        scene_camera = load_json(scene_camera_path)
        collision_dump = load_collision_dump(dataset_root, scene_id)

        img_ids = get_camera_img_ids(scene_camera, camera, cfg.max_ann)
        if cfg.img_ids:
            wanted = set(parse_int_list(cfg.img_ids))
            img_ids = [x for x in img_ids if x in wanted]
        if not img_ids:
            return {"ok": False, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": "no image ids"}

        for local_frame_idx, img_id in enumerate(img_ids):
            out_dir = save_root / f"{int(scene_id):06d}" / camera
            out_path = out_dir / f"{int(img_id):06d}.npy"
            if cfg.skip_done and out_path.exists():
                stats.skipped_done += 1
                continue
            if str(int(img_id)) not in scene_gt or str(int(img_id)) not in scene_camera:
                stats.missing += 1
                continue

            depth_path = resolve_depth_path(dataset_root, virtual_root, scene_id, camera, img_id, cfg.depth_type)
            label_path = scene_dir / "label" / f"{int(img_id):06d}.png"
            if not depth_path.exists() or not label_path.exists():
                stats.missing += 1
                if cfg.verbose:
                    print(f"[missing] scene={scene_id:06d} img={img_id:06d} depth={depth_path.exists()} label={label_path.exists()}")
                continue

            depth = np.array(Image.open(depth_path))
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            label = load_label(label_path)
            if label.shape != depth.shape:
                raise ValueError(f"label/depth shape mismatch: {label.shape} vs {depth.shape}")

            cam_info = scene_camera[str(int(img_id))]
            K = np.asarray(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
            factor_depth = get_factor_depth(camera, cam_info, cfg.factor_depth_mode, cfg.fixed_factor_depth)
            cloud = depth_to_cloud(depth, K, factor_depth)

            mask = build_mask(
                depth=depth,
                label=label,
                cloud=cloud,
                cam_info=cam_info,
                mode=cfg.mask_mode,
                bbox_margin_ratio=cfg.bbox_margin_ratio,
                workspace_outlier=cfg.workspace_outlier,
                workspace_pose_mode=cfg.workspace_pose_mode,
                workspace_depth_trunc=cfg.workspace_depth_trunc,
            )
            cloud_masked = cloud[mask].astype(np.float32)
            label_masked = label[mask].reshape(-1)
            if cloud_masked.shape[0] == 0:
                stats.empty_cloud += 1
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_path, np.zeros((0, 1), dtype=np.float16))
                continue

            obj_list, pose_list = load_scene_object_list(scene_gt, img_id)
            grasp_points_all = []
            graspness_all = []
            object_debug = []

            for local_i, (obj_id, pose) in enumerate(zip(obj_list, pose_list)):
                if local_i >= len(collision_dump):
                    raise IndexError(f"collision_dump has {len(collision_dump)} arrays but scene_gt local index={local_i}")
                sampled_points, offsets, fric_coefs = load_grasp_label(dataset_root, obj_id, grasp_label_cache)
                collision = collision_dump[local_i]
                if collision.shape != fric_coefs.shape:
                    raise ValueError(
                        f"collision/fric shape mismatch for scene={scene_id:06d}, img={img_id:06d}, "
                        f"local_i={local_i}, obj={obj_id}: collision={collision.shape}, fric={fric_coefs.shape}"
                    )
                num_points = sampled_points.shape[0]
                valid_grasp_mask = (
                    (fric_coefs <= float(cfg.fric_coef_thresh)) &
                    (fric_coefs > 0) &
                    (~collision.astype(bool))
                ).reshape(num_points, -1)
                raw_graspness = valid_grasp_mask.sum(axis=1).astype(np.float32) / float(NUM_VIEWS * NUM_ANGLES * NUM_DEPTHS)
                if cfg.mode == "scene":
                    graspness = raw_graspness
                elif cfg.mode == "instance":
                    graspness = safe_minmax(raw_graspness)
                else:
                    raise ValueError(f"Unsupported mode={cfg.mode}")
                target_points = transform_points(sampled_points, pose)
                grasp_points_all.append(target_points.astype(np.float32))
                graspness_all.append(graspness.reshape(num_points, 1).astype(np.float32))
                if cfg.debug_object_stats:
                    object_debug.append({
                        "local_i": int(local_i),
                        "obj_id": int(obj_id),
                        "num_label_points": int(num_points),
                        "raw_min": float(np.min(raw_graspness)) if raw_graspness.size else 0.0,
                        "raw_max": float(np.max(raw_graspness)) if raw_graspness.size else 0.0,
                        "raw_mean": float(np.mean(raw_graspness)) if raw_graspness.size else 0.0,
                        "mode_min": float(np.min(graspness)) if graspness.size else 0.0,
                        "mode_max": float(np.max(graspness)) if graspness.size else 0.0,
                        "mode_mean": float(np.mean(graspness)) if graspness.size else 0.0,
                    })

            if len(grasp_points_all) == 0:
                stats.empty_grasp += 1
                cloud_graspness_raw = np.zeros((cloud_masked.shape[0], 1), dtype=np.float32)
                nn_dist = np.full((cloud_masked.shape[0], 1), np.inf, dtype=np.float32)
            else:
                grasp_points = np.vstack(grasp_points_all).astype(np.float32)
                graspness = np.vstack(graspness_all).astype(np.float32)
                cloud_graspness_raw, nn_dist = assign_nearest_graspness_with_dist(
                    cloud_masked,
                    grasp_points,
                    graspness,
                    backend=cfg.knn_backend,
                    torch_device=cfg.torch_device,
                    torch_chunk_size=cfg.torch_chunk_size,
                )
                cloud_graspness_raw = cloud_graspness_raw.astype(np.float32).reshape(-1, 1)
                nn_dist = nn_dist.astype(np.float32).reshape(-1, 1)

            cloud_graspness_pre = cloud_graspness_raw.copy()
            zero_by_background = np.zeros((cloud_masked.shape[0],), dtype=bool)
            zero_by_dist = np.zeros((cloud_masked.shape[0],), dtype=bool)
            if cfg.zero_background:
                zero_by_background = label_masked <= 0
                cloud_graspness_pre[zero_by_background, 0] = 0.0
            if cfg.knn_dist_thresh is not None and float(cfg.knn_dist_thresh) > 0:
                zero_by_dist = nn_dist.reshape(-1) > float(cfg.knn_dist_thresh)
                cloud_graspness_pre[zero_by_dist, 0] = 0.0

            cloud_graspness = normalize_graspness(cloud_graspness_pre, label_masked, cfg.final_norm)

            if cloud_graspness.shape[0] != cloud_masked.shape[0]:
                raise RuntimeError(f"Internal length mismatch: graspness={cloud_graspness.shape}, cloud={cloud_masked.shape}")

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_path, cloud_graspness.astype(np.float16))
            stats.generated += 1

            fg_m = label_masked.reshape(-1) > 0
            bg_m = ~fg_m
            high = cloud_graspness.reshape(-1) > float(cfg.debug_high_thresh)
            raw_high = cloud_graspness_raw.reshape(-1) > float(cfg.debug_high_thresh)
            finite_dist = np.isfinite(nn_dist.reshape(-1))
            row = {
                "scene_id": int(scene_id),
                "camera": camera,
                "img_id": int(img_id),
                "ann_id": int(img_id_to_ann_id(img_id, camera)),
                "depth_type": cfg.depth_type,
                "mode": cfg.mode,
                "mask_mode": cfg.mask_mode,
                "factor_depth": float(factor_depth),
                "num_masked_points": int(cloud_masked.shape[0]),
                "num_fg_masked": int(fg_m.sum()),
                "num_bg_masked": int(bg_m.sum()),
                "num_objects": int(len(obj_list)),
                "obj_ids": ";".join(str(x) for x in obj_list),
                "num_grasp_points": int(sum(x.shape[0] for x in grasp_points_all)),
                "zero_background": int(bool(cfg.zero_background)),
                "knn_dist_thresh": float(cfg.knn_dist_thresh),
                "final_norm": cfg.final_norm,
                "num_zero_by_background": int(zero_by_background.sum()),
                "num_zero_by_dist": int(zero_by_dist.sum()),
                "bg_high_count_raw": int((bg_m & raw_high).sum()),
                "bg_high_ratio_raw": float((bg_m & raw_high).sum() / max(int(bg_m.sum()), 1)),
                "bg_high_count_final": int((bg_m & high).sum()),
                "bg_high_ratio_final": float((bg_m & high).sum() / max(int(bg_m.sum()), 1)),
                "fg_high_count_final": int((fg_m & high).sum()),
                "fg_high_ratio_final": float((fg_m & high).sum() / max(int(fg_m.sum()), 1)),
                "save_path": str(out_path),
            }
            row.update(get_mask_counts(depth, label, cloud, cam_info, cfg))
            row.update(stat_prefix(cloud_graspness_raw, "grasp_raw"))
            row.update(stat_prefix(cloud_graspness_pre, "grasp_pre_norm"))
            row.update(stat_prefix(cloud_graspness, "grasp_final"))
            row.update(stat_prefix(cloud_graspness[fg_m], "grasp_final_fg"))
            row.update(stat_prefix(cloud_graspness[bg_m], "grasp_final_bg"))
            row.update(stat_prefix(nn_dist[finite_dist], "nn_dist"))
            row.update(stat_prefix(nn_dist[fg_m & finite_dist], "nn_dist_fg"))
            row.update(stat_prefix(nn_dist[bg_m & finite_dist], "nn_dist_bg"))
            if cfg.debug_object_stats:
                row["object_debug_json"] = json.dumps(object_debug, ensure_ascii=False, separators=(",", ":"))
            if cfg.write_stats or cfg.debug_dir:
                rows.append(row)

            should_debug = bool(cfg.debug_dir) and (cfg.debug_every <= 1 or (local_frame_idx % int(cfg.debug_every) == 0))
            if should_debug:
                s6 = f"{int(scene_id):06d}"
                i6 = f"{int(img_id):06d}"
                if cfg.debug_save_npz:
                    npz_path = debug_root / "npz" / s6 / camera / f"{i6}.npz"
                    npz_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        npz_path,
                        cloud_masked=cloud_masked.astype(np.float32),
                        label_masked=label_masked.astype(np.int32),
                        graspness_raw=cloud_graspness_raw.astype(np.float32),
                        graspness_pre_norm=cloud_graspness_pre.astype(np.float32),
                        graspness_final=cloud_graspness.astype(np.float32),
                        nn_dist=nn_dist.astype(np.float32),
                        mask=mask.astype(np.uint8),
                        label=label.astype(np.int32),
                        depth=depth,
                        obj_ids=np.asarray(obj_list, dtype=np.int32),
                    )
                if cfg.debug_save_2d:
                    rgb_path = resolve_rgb_path(dataset_root, scene_id, img_id)
                    if rgb_path.exists():
                        rgb = np.array(Image.open(rgb_path).convert("RGB"))
                    else:
                        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    panel_path = debug_root / "panel" / s6 / camera / f"{i6}.png"
                    save_debug_2d_panel(
                        panel_path,
                        rgb=rgb,
                        label=label,
                        mask=mask,
                        graspness_final=cloud_graspness,
                        nn_dist=nn_dist,
                        high_thresh=cfg.debug_high_thresh,
                        cmap=cfg.debug_cmap,
                        overlay_alpha=cfg.debug_overlay_alpha,
                    )
                if cfg.debug_save_ply:
                    ply_path = debug_root / "ply" / s6 / camera / f"{i6}.ply"
                    save_debug_ply(
                        ply_path,
                        cloud_points=cloud_masked,
                        graspness_final=cloud_graspness,
                        max_points=cfg.debug_ply_max_points,
                        cmap=cfg.debug_cmap,
                    )

        if (cfg.write_stats or cfg.debug_dir) and rows:
            stats_dir = (debug_root / "stats") if cfg.debug_dir else (save_root / "stats")
            stats_dir.mkdir(parents=True, exist_ok=True)
            write_csv_rows(stats_dir / f"{int(scene_id):06d}_{camera}_{cfg.depth_type}_{cfg.mode}.csv", rows)

        return {"ok": True, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": "done"}

    except Exception:
        stats.failed += 1
        return {"ok": False, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": traceback.format_exc()}


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def _worker(args):
    scene_id, cfg = args
    return process_one_scene(scene_id, cfg)


def run(scene_ids: Sequence[int], cfg) -> List[Dict[str, object]]:
    scene_ids = list(scene_ids)
    if cfg.proc <= 1:
        return [process_one_scene(sid, cfg) for sid in tqdm(scene_ids, desc="Scenes")]
    ctx = mp.get_context(cfg.mp_start_method)
    jobs = [(sid, cfg) for sid in scene_ids]
    results = []
    with ctx.Pool(processes=int(cfg.proc), maxtasksperchild=1) as pool:
        for ret in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc="Scenes"):
            results.append(ret)
            status = "OK" if ret["ok"] else "FAIL"
            msg = str(ret.get("message", ""))
            msg0 = msg.splitlines()[0] if msg else ""
            print(f"[scene {int(ret['scene_id']):06d}] {status} {ret['stats']} {msg0}")
            if not ret["ok"]:
                print(msg)
    return results


def summarize(results: Sequence[Dict[str, object]]) -> None:
    total = RunStats()
    failed = []
    for r in results:
        total.add(r.get("stats", {}))
        if not r.get("ok", False):
            failed.append(r.get("scene_id"))
    print("\n[Summary]")
    print(f"  scenes:       {len(results)}")
    print(f"  generated:    {total.generated}")
    print(f"  skipped_done: {total.skipped_done}")
    print(f"  empty_cloud:  {total.empty_cloud}")
    print(f"  empty_grasp:  {total.empty_grasp}")
    print(f"  missing:      {total.missing}")
    print(f"  failed:       {total.failed}")
    if failed:
        print(f"  failed scenes: {failed[:50]}{' ...' if len(failed) > 50 else ''}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate GraspClutter6D point-wise graspness for ecograsp_dpt with debug support.")
    p.add_argument("--dataset_root", default=os.environ.get("GC6D_ROOT"), help="GraspClutter6D root.")
    p.add_argument("--virtual_dataset_root", default=None, help="Default: dataset_root/virtual_scenes.")
    p.add_argument("--save_root", default=None, help="Optional explicit output root.")
    p.add_argument("--split", default="train", help="train/test/all or explicit scene ids like '5,7,10-20'.")
    p.add_argument("--scene_ids", default=None, help="Override --split.")
    p.add_argument("--camera", default="realsense-d435", choices=list(CAMERA_OFFSETS.keys()) + list(CAMERA_ALIASES.keys()))
    p.add_argument("--max_ann", type=int, default=13, help="GC6D default ann_id is 0..12 per camera. Use <=0 for all matching ids.")
    p.add_argument("--img_ids", default=None, help="Optional explicit image ids, e.g. '2,6,10'.")

    p.add_argument("--depth_type", default="real", choices=["real", "virtual"])
    p.add_argument("--mode", default="instance", choices=["scene", "instance"], help="Object-level normalization mode before final normalization.")
    p.add_argument("--mask_mode", default="depth", choices=["depth", "label_depth", "bbox_depth", "workspace_depth"])
    p.add_argument("--bbox_margin_ratio", type=float, default=0.1)
    p.add_argument("--workspace_outlier", type=float, default=0.02)
    p.add_argument("--workspace_pose_mode", default="json", choices=["json", "inverse", "none"])
    p.add_argument("--workspace_depth_trunc", type=float, default=0.0)

    p.add_argument("--factor_depth_mode", default="bop", choices=["bop", "camera", "fixed"])
    p.add_argument("--fixed_factor_depth", type=float, default=None)

    p.add_argument("--fric_coef_thresh", type=float, default=0.8)
    p.add_argument("--knn_backend", default="torch", choices=["auto", "scipy", "torch"])
    p.add_argument("--torch_device", default="cuda:0")
    p.add_argument("--torch_chunk_size", type=int, default=65536)

    # Debug / ablation switches. Defaults preserve previous behavior.
    p.add_argument("--zero_background", action="store_true", help="Set label==0 masked points to zero after KNN assignment.")
    p.add_argument("--knn_dist_thresh", type=float, default=-1.0, help="If >0, set points whose nearest grasp-label point is farther than this meter threshold to zero.")
    p.add_argument("--final_norm", default="minmax", choices=["minmax", "foreground_minmax", "none"], help="Final normalization after optional background/distance filtering.")
    p.add_argument("--debug_high_thresh", type=float, default=0.3, help="Threshold for reporting high-graspness background leakage.")
    p.add_argument("--debug_object_stats", action="store_true", help="Add per-object raw/mode graspness statistics as JSON in the CSV.")
    p.add_argument("--debug_dir", default=None, help="If set, write debug CSV / NPZ / panel / PLY under this directory.")
    p.add_argument("--debug_every", type=int, default=1, help="Save debug artifacts every N frames when --debug_dir is set.")
    p.add_argument("--debug_save_npz", action="store_true")
    p.add_argument("--debug_save_2d", action="store_true")
    p.add_argument("--debug_save_ply", action="store_true")
    p.add_argument("--debug_ply_max_points", type=int, default=200000)
    p.add_argument("--debug_cmap", default="jet")
    p.add_argument("--debug_overlay_alpha", type=float, default=0.55)

    p.add_argument("--proc", type=int, default=1)
    p.add_argument("--mp_start_method", default="spawn", choices=["spawn", "forkserver", "fork"])
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--write_stats", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    cfg = build_argparser().parse_args()
    if cfg.dataset_root is None:
        raise ValueError("--dataset_root is required or set $GC6D_ROOT")
    cfg.dataset_root = str(Path(cfg.dataset_root).expanduser())
    if cfg.max_ann is not None and cfg.max_ann <= 0:
        cfg.max_ann = None
    if cfg.debug_every <= 0:
        cfg.debug_every = 1

    if cfg.scene_ids:
        scene_ids = parse_int_list(cfg.scene_ids)
    else:
        scene_ids = read_split_scene_ids(Path(cfg.dataset_root), cfg.split)

    save_root = get_save_root(Path(cfg.dataset_root), cfg.depth_type, cfg.mode, cfg.save_root)
    print(f"[Main] dataset_root={cfg.dataset_root}")
    print(f"[Main] split={cfg.split}, scenes={len(scene_ids)}")
    print(f"[Main] camera={canonical_camera(cfg.camera)}, depth_type={cfg.depth_type}, mode={cfg.mode}")
    print(f"[Main] mask_mode={cfg.mask_mode}; output sequence must match the dataloader mask")
    if cfg.mask_mode == "workspace_depth":
        print(f"[Main] workspace_outlier={cfg.workspace_outlier}, workspace_pose_mode={cfg.workspace_pose_mode}, workspace_depth_trunc={cfg.workspace_depth_trunc}")
    print(f"[Main] save_root={save_root}")
    print(f"[Main] factor_depth_mode={cfg.factor_depth_mode}, knn_backend={cfg.knn_backend}, proc={cfg.proc}")
    print(f"[Main] zero_background={cfg.zero_background}, knn_dist_thresh={cfg.knn_dist_thresh}, final_norm={cfg.final_norm}")
    if cfg.debug_dir:
        print(f"[Main] debug_dir={cfg.debug_dir}")

    results = run(scene_ids, cfg)
    summarize(results)


if __name__ == "__main__":
    main()
