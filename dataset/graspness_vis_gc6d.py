#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize GraspClutter6D graspness in both 3D and 2D.

Features
--------
1) Save colored point cloud (.ply), where point color is mapped from graspness.
2) Save 2D visualization panel: RGB | graspness heatmap | RGB+graspness overlay.

This script is based on gc6d_visualize_graspness_v2.py and keeps the same
workspace-depth masking logic so that the visualization mask matches the
graspness-generation mask.

Typical usage
-------------
python gc6d_visualize_graspness_v3.py \
    --dataset_root /data/robotarm/dataset/GraspClutter6D \
    --scene_id 5 \
    --camera realsense-d435 \
    --img_id 2 \
    --depth_type real \
    --graspness_root /data/robotarm/dataset/GraspClutter6D/graspness_instance \
    --mask_mode workspace_depth \
    --workspace_outlier 0.02 \
    --workspace_pose_mode json \
    --print_mask_counts \
    --save_ply /tmp/gc6d_graspness.ply \
    --save_panel /tmp/gc6d_graspness_panel.png \
    --save_overlay_png /tmp/gc6d_graspness_overlay.png \
    --save_graspness_png /tmp/gc6d_graspness_heatmap.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from matplotlib import cm


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


def canonical_camera(camera: str) -> str:
    camera = CAMERA_ALIASES.get(str(camera), str(camera))
    if camera not in CAMERA_OFFSETS:
        raise ValueError(f"Unknown camera={camera}. Expected one of {list(CAMERA_OFFSETS)}")
    return camera


def img_id_to_ann_id(img_id: int, camera: str) -> int:
    offset = CAMERA_OFFSETS[canonical_camera(camera)]
    img_id = int(img_id)
    if img_id < offset or ((img_id - offset) % 4) != 0:
        raise ValueError(f"img_id={img_id} does not belong to camera={camera}")
    return (img_id - offset) // 4


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    return int(ann_id) * 4 + CAMERA_OFFSETS[canonical_camera(camera)]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label(path: Path) -> np.ndarray:
    label = np.array(Image.open(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def get_factor_depth(camera: str, cam_info: dict, mode: str, fixed: Optional[float]) -> float:
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
    """
    Same semantics as gc6d_generate_graspness_v2.py.

    Modes:
      none:    compute workspace bbox in current camera frame.
      json:    use raw scene_camera matrix as camera-to-workspace.
      inverse: use inverse(raw scene_camera matrix) as camera-to-workspace.
    """
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
    """Exact workspace_depth mask used by gc6d_generate_graspness_v2.py."""
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
    if T is not None:
        cloud_box_flat = transform_points(cloud_flat, T)
    else:
        cloud_box_flat = cloud_flat

    fg_flat = fg.reshape(-1)
    valid_flat = valid.reshape(-1)

    fg_points = cloud_box_flat[fg_flat]
    xyz_min = fg_points.min(axis=0) - float(outlier)
    xyz_max = fg_points.max(axis=0) + float(outlier)

    in_box = (
        (cloud_box_flat[:, 0] > xyz_min[0]) & (cloud_box_flat[:, 0] < xyz_max[0]) &
        (cloud_box_flat[:, 1] > xyz_min[1]) & (cloud_box_flat[:, 1] < xyz_max[1]) &
        (cloud_box_flat[:, 2] > xyz_min[2]) & (cloud_box_flat[:, 2] < xyz_max[2])
    )
    return (valid_flat & in_box).reshape(H, W)


def build_mask(
    cloud: np.ndarray,
    depth: np.ndarray,
    label: np.ndarray,
    cam_info: dict,
    mask_mode: str,
    workspace_outlier: float,
    workspace_pose_mode: str,
    workspace_depth_trunc: float,
) -> np.ndarray:
    depth_mask = depth > 0
    mask_mode = str(mask_mode).lower()

    if mask_mode == "depth":
        return depth_mask

    if mask_mode == "label_depth":
        return depth_mask & (label > 0)

    if mask_mode == "bbox_depth":
        H, W = depth.shape[:2]
        fg = label > 0
        if not np.any(fg):
            return depth_mask
        ys, xs = np.where(fg)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mx = int(round(W * 0.1))
        my = int(round(H * 0.1))
        x0 = max(0, x0 - mx)
        x1 = min(W, x1 + mx)
        y0 = max(0, y0 - my)
        y1 = min(H, y1 + my)
        mask = np.zeros_like(depth_mask, dtype=bool)
        mask[y0:y1, x0:x1] = True
        return depth_mask & mask

    if mask_mode == "workspace_depth":
        return build_workspace_mask(
            cloud=cloud,
            label=label,
            depth_mask=depth_mask,
            cam_info=cam_info,
            outlier=float(workspace_outlier),
            pose_mode=workspace_pose_mode,
            depth_trunc=float(workspace_depth_trunc),
        )

    raise ValueError(f"Unsupported mask_mode={mask_mode}")


def resolve_rgb_path(dataset_root: Path, scene_id: int, img_id: int) -> Path:
    return dataset_root / "scenes" / f"{int(scene_id):06d}" / "rgb" / f"{int(img_id):06d}.png"


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


def resolve_graspness_path(graspness_root: Path, scene_id: int, camera: str, img_id: int) -> Path:
    candidates = [
        graspness_root / f"{int(scene_id):06d}" / canonical_camera(camera) / f"{int(img_id):06d}.npy",
        graspness_root / f"scene_{int(scene_id):04d}" / canonical_camera(camera) / f"{int(img_id):06d}.npy",
        graspness_root / f"{int(scene_id):06d}" / canonical_camera(camera) / f"{int(img_id):04d}.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def colorize_graspness_values(graspness: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    g = np.asarray(graspness, dtype=np.float32).reshape(-1)
    g = np.clip(g, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    return cmap(g)[:, :3].astype(np.float32)


def save_colored_ply(points: np.ndarray, colors: np.ndarray, save_ply: Path) -> None:
    save_ply.parent.mkdir(parents=True, exist_ok=True)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    ok = o3d.io.write_point_cloud(str(save_ply), pc)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud to {save_ply}")


def make_graspness_images(
    rgb: np.ndarray,
    mask: np.ndarray,
    graspness: np.ndarray,
    cmap_name: str = "jet",
    overlay_alpha: float = 0.55,
    overlay_mask_only: bool = True,
):
    """
    Returns:
        heatmap_rgb_uint8: HxWx3
        overlay_rgb_uint8: HxWx3
        graspness_map: HxW float32
    """
    H, W = mask.shape
    if rgb.shape[:2] != (H, W):
        raise ValueError(f"RGB shape {rgb.shape[:2]} != mask shape {(H, W)}")

    graspness_map = np.zeros((H, W), dtype=np.float32)
    graspness_map[mask] = np.asarray(graspness, dtype=np.float32).reshape(-1)

    cmap = cm.get_cmap(cmap_name)
    heat = (cmap(np.clip(graspness_map, 0.0, 1.0))[:, :, :3] * 255.0).astype(np.uint8)

    rgb_u8 = rgb.astype(np.uint8)
    overlay = rgb_u8.copy()

    if overlay_mask_only:
        alpha_map = np.zeros((H, W, 1), dtype=np.float32)
        alpha_map[mask] = float(overlay_alpha)
    else:
        alpha_map = np.full((H, W, 1), float(overlay_alpha), dtype=np.float32)

    overlay = (
        rgb_u8.astype(np.float32) * (1.0 - alpha_map) +
        heat.astype(np.float32) * alpha_map
    ).clip(0, 255).astype(np.uint8)

    return heat, overlay, graspness_map


def add_title_bar(img: np.ndarray, title: str, bar_h: int = 34) -> np.ndarray:
    """Add a simple white title bar above an RGB image."""
    H, W = img.shape[:2]
    canvas = Image.new("RGB", (W, H + bar_h), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    return np.array(canvas)


def build_panel(rgb: np.ndarray, heat: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    a = add_title_bar(rgb, "RGB")
    b = add_title_bar(heat, "Graspness")
    c = add_title_bar(overlay, "Overlay")
    gap = 8
    H, W = a.shape[:2]
    panel = np.full((H, W * 3 + gap * 2, 3), 255, dtype=np.uint8)
    panel[:, 0:W] = a
    panel[:, W + gap:2 * W + gap] = b
    panel[:, 2 * W + 2 * gap:3 * W + 2 * gap] = c
    return panel


def save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser("Visualize GC6D graspness in 3D and 2D.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--virtual_dataset_root", default=None, help="Default: dataset_root/virtual_scenes")
    parser.add_argument("--graspness_root", required=True, help="Path to graspness / graspness_instance / virtual_graspness / virtual_graspness_instance")
    parser.add_argument("--scene_id", type=int, required=True)
    parser.add_argument("--camera", default="realsense-d435", choices=list(CAMERA_OFFSETS.keys()) + list(CAMERA_ALIASES.keys()))
    parser.add_argument("--img_id", type=int, default=None)
    parser.add_argument("--ann_id", type=int, default=None)
    parser.add_argument("--depth_type", default="real", choices=["real", "virtual"])

    parser.add_argument("--mask_mode", default="workspace_depth", choices=["depth", "label_depth", "bbox_depth", "workspace_depth"])
    parser.add_argument("--workspace_outlier", type=float, default=0.02)
    parser.add_argument("--workspace_pose_mode", default="json", choices=["json", "inverse", "none"])
    parser.add_argument("--workspace_depth_trunc", type=float, default=0.0)

    parser.add_argument("--factor_depth_mode", default="bop", choices=["bop", "camera", "fixed"])
    parser.add_argument("--fixed_factor_depth", type=float, default=None)

    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--overlay_mask_only", action="store_true",
                        help="If set, only overlay heatmap on masked pixels; otherwise blend over the full image.")

    parser.add_argument("--save_ply", default=None)
    parser.add_argument("--save_panel", default=None, help="Save panel image: RGB | graspness | overlay")
    parser.add_argument("--save_graspness_png", default=None, help="Save graspness heatmap image")
    parser.add_argument("--save_overlay_png", default=None, help="Save overlay image")
    parser.add_argument("--save_rgb_png", default=None, help="Optional re-save the original RGB image")
    parser.add_argument("--save_npz", default=None)
    parser.add_argument("--print_mask_counts", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    virtual_root = Path(args.virtual_dataset_root).expanduser() if args.virtual_dataset_root else (dataset_root / "virtual_scenes")
    graspness_root = Path(args.graspness_root).expanduser()
    camera = canonical_camera(args.camera)

    if args.img_id is None and args.ann_id is None:
        raise ValueError("Provide either --img_id or --ann_id")
    if args.img_id is not None and args.ann_id is not None:
        raise ValueError("Provide only one of --img_id or --ann_id")
    img_id = int(args.img_id) if args.img_id is not None else ann_id_to_img_id(int(args.ann_id), camera)

    scene_dir = dataset_root / "scenes" / f"{int(args.scene_id):06d}"
    scene_camera = load_json(scene_dir / "scene_camera.json")
    if str(img_id) not in scene_camera:
        raise KeyError(f"img_id={img_id} not found in {scene_dir/'scene_camera.json'}")
    cam_info = scene_camera[str(img_id)]

    rgb_path = resolve_rgb_path(dataset_root, args.scene_id, img_id)
    depth_path = resolve_depth_path(dataset_root, virtual_root, args.scene_id, camera, img_id, args.depth_type)
    label_path = scene_dir / "label" / f"{int(img_id):06d}.png"
    graspness_path = resolve_graspness_path(graspness_root, args.scene_id, camera, img_id)

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")
    if not graspness_path.exists():
        raise FileNotFoundError(f"Graspness not found: {graspness_path}")

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path))
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    label = load_label(label_path)

    K = np.asarray(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
    factor_depth = get_factor_depth(camera, cam_info, args.factor_depth_mode, args.fixed_factor_depth)
    cloud = depth_to_cloud(depth, K, factor_depth)

    if args.print_mask_counts:
        for m in ["depth", "label_depth", "bbox_depth", "workspace_depth"]:
            mm = build_mask(
                cloud=cloud,
                depth=depth,
                label=label,
                cam_info=cam_info,
                mask_mode=m,
                workspace_outlier=args.workspace_outlier,
                workspace_pose_mode=args.workspace_pose_mode,
                workspace_depth_trunc=args.workspace_depth_trunc,
            )
            print(f"[mask-count] {m:16s}: {int(mm.sum())}")

    mask = build_mask(
        cloud=cloud,
        depth=depth,
        label=label,
        cam_info=cam_info,
        mask_mode=args.mask_mode,
        workspace_outlier=args.workspace_outlier,
        workspace_pose_mode=args.workspace_pose_mode,
        workspace_depth_trunc=args.workspace_depth_trunc,
    )

    points = cloud[mask].astype(np.float32)
    graspness = np.load(graspness_path)
    graspness = np.asarray(graspness, dtype=np.float32).reshape(-1, 1)

    if points.shape[0] != graspness.shape[0]:
        raise ValueError(
            f"Length mismatch: points={points.shape[0]}, graspness={graspness.shape[0]}. "
            f"Check mask_mode/workspace_pose_mode/workspace_depth_trunc/factor_depth_mode and depth_type. "
            f"depth_path={depth_path}, graspness_path={graspness_path}"
        )

    point_colors = colorize_graspness_values(graspness, cmap_name=args.cmap)

    if args.save_ply:
        save_colored_ply(points, point_colors, Path(args.save_ply).expanduser())

    heat, overlay, graspness_map = make_graspness_images(
        rgb=rgb,
        mask=mask,
        graspness=graspness,
        cmap_name=args.cmap,
        overlay_alpha=args.overlay_alpha,
        overlay_mask_only=args.overlay_mask_only,
    )

    if args.save_graspness_png:
        save_image(heat, Path(args.save_graspness_png).expanduser())
    if args.save_overlay_png:
        save_image(overlay, Path(args.save_overlay_png).expanduser())
    if args.save_rgb_png:
        save_image(rgb, Path(args.save_rgb_png).expanduser())
    if args.save_panel:
        panel = build_panel(rgb, heat, overlay)
        save_image(panel, Path(args.save_panel).expanduser())

    if args.save_npz:
        save_npz = Path(args.save_npz).expanduser()
        save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_npz,
            points=points.astype(np.float32),
            point_colors=point_colors.astype(np.float32),
            graspness=graspness.astype(np.float32),
            graspness_map=graspness_map.astype(np.float32),
            mask=mask.astype(np.uint8),
            rgb=rgb.astype(np.uint8),
            heatmap=heat.astype(np.uint8),
            overlay=overlay.astype(np.uint8),
        )

    print(f"[OK] scene_id={args.scene_id:06d} camera={camera} img_id={img_id:06d} ann_id={img_id_to_ann_id(img_id, camera)}")
    print(f"[OK] rgb:        {rgb_path}")
    print(f"[OK] depth:      {depth_path}")
    print(f"[OK] label:      {label_path}")
    print(f"[OK] graspness:  {graspness_path}")
    print(f"[OK] mask_mode:  {args.mask_mode}")
    print(f"[OK] #points:    {points.shape[0]}")
    print(f"[OK] graspness range: [{float(graspness.min()):.4f}, {float(graspness.max()):.4f}]  mean={float(graspness.mean()):.4f}")
    if args.save_ply:
        print(f"[OK] ply saved:         {Path(args.save_ply).expanduser()}")
    if args.save_panel:
        print(f"[OK] panel saved:       {Path(args.save_panel).expanduser()}")
    if args.save_graspness_png:
        print(f"[OK] graspness png:     {Path(args.save_graspness_png).expanduser()}")
    if args.save_overlay_png:
        print(f"[OK] overlay png:       {Path(args.save_overlay_png).expanduser()}")
    if args.save_rgb_png:
        print(f"[OK] rgb png:           {Path(args.save_rgb_png).expanduser()}")


if __name__ == "__main__":
    main()
