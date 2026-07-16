#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize GraspClutter6D rendered/virtual depth against observed depth and GT grasps.

Visualization convention:
  - observed sensor depth point cloud: red
  - rendered/virtual GT depth point cloud: blue
  - GT 6D grasps: GraspGroup.to_open3d_geometry_list(), preserving GraspNetAPI colors

The script uses GraspClutter6D.loadGrasp(...) to read ground-truth grasps, then
sorts/NMS/top-k filters them before visualization.

Example:
  python gc6d_visualize_rendered_depth_gt_grasps.py \
    --dataset_root /data/robotarm/dataset/GraspClutter6D \
    --scene_id 5 --ann_id 0 --camera realsense-d435 \
    --virtual_root /data/robotarm/dataset/GraspClutter6D/virtual_scenes \
    --topk 80 --nms --save_ply /tmp/gc6d_scene000005_ann0000_obs_red_render_blue_gtgrasps.ply

If your generated virtual depth is stored by image id rather than ann id, the script
will automatically resolve:
  virtual_scenes/<scene>/<camera>/<img_id>_depth.png
  virtual_scenes/<scene>/<img_id>_depth.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from PIL import Image


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


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    return int(ann_id) * 4 + CAMERA_OFFSETS[canonical_camera(camera)]


def img_id_to_ann_id(img_id: int, camera: str) -> int:
    img_id = int(img_id)
    offset = CAMERA_OFFSETS[canonical_camera(camera)]
    if img_id < offset or ((img_id - offset) % 4) != 0:
        raise ValueError(f"img_id={img_id} does not belong to camera={camera}")
    return (img_id - offset) // 4


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_virtual_depth_path(virtual_root: Path, scene_id: int, camera: str, img_id: int, ann_id: int) -> Path:
    s6 = f"{int(scene_id):06d}"
    cam = canonical_camera(camera)
    candidates = [
        virtual_root / s6 / cam / f"{int(img_id):06d}_depth.png",
        virtual_root / s6 / cam / f"{int(img_id):06d}.png",
        virtual_root / s6 / f"{int(img_id):06d}_depth.png",
        virtual_root / s6 / f"{int(img_id):06d}.png",
        # Backward-compatible variants.
        virtual_root / s6 / cam / f"{int(ann_id):04d}_depth.png",
        virtual_root / s6 / cam / f"{int(ann_id):04d}.png",
        virtual_root / f"scene_{int(scene_id):04d}" / cam / f"{int(ann_id):04d}_depth.png",
        virtual_root / f"scene_{int(scene_id):04d}" / cam / f"{int(ann_id):04d}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("virtual/rendered depth not found. Tried:\n" + "\n".join(str(x) for x in candidates))


def depth_raw_to_m(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    """BOP convention: depth_m = raw * depth_scale / 1000."""
    return depth_raw.astype(np.float32) * float(depth_scale) / 1000.0


def depth_to_points(
    depth_raw: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    depth_trunc: float,
    stride: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    depth_m = depth_raw_to_m(depth_raw, depth_scale)
    H, W = depth_m.shape[:2]
    stride = max(1, int(stride))
    ys = np.arange(0, H, stride, dtype=np.int64)
    xs = np.arange(0, W, stride, dtype=np.int64)
    xv, yv = np.meshgrid(xs, ys)
    z = depth_m[yv, xv].astype(np.float32)
    valid = np.isfinite(z) & (z > 0) & (z < float(depth_trunc))
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    xv = xv[valid].astype(np.float32)
    yv = yv[valid].astype(np.float32)
    z = z[valid]
    x = (xv - float(K[0, 2])) * z / float(K[0, 0])
    y = (yv - float(K[1, 2])) * z / float(K[1, 1])
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    max_points = int(max_points)
    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def make_pcd(points: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if points.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def translate_pcd(pcd: o3d.geometry.PointCloud, offset: Sequence[float]) -> o3d.geometry.PointCloud:
    out = pcd.clone() if hasattr(pcd, "clone") else o3d.geometry.PointCloud(pcd)
    out.translate(np.asarray(offset, dtype=np.float64))
    return out


def import_gc6d_api(api_root: Optional[str]):
    if api_root:
        sys.path.insert(0, str(Path(api_root).expanduser()))
    try:
        from graspclutter6dAPI import GraspClutter6D  # type: ignore
        return GraspClutter6D
    except Exception as e1:
        try:
            from graspclutter6d import GraspClutter6D  # type: ignore
            return GraspClutter6D
        except Exception as e2:
            raise ImportError(
                "Failed to import GraspClutter6D. Use --api_root to point to the parent directory "
                "containing graspclutter6dAPI.\n"
                f"First error: {repr(e1)}\nSecond error: {repr(e2)}"
            )


def get_obj_ids_for_image(dataset_root: Path, scene_id: int, img_id: int) -> List[int]:
    scene_gt = load_json(dataset_root / "scenes" / f"{int(scene_id):06d}" / "scene_gt.json")
    obj_ids = [int(x["obj_id"]) for x in scene_gt[str(int(img_id))]]
    return list(dict.fromkeys(obj_ids))


def load_gt_grasps(args, scene_id: int, ann_id: int, img_id: int):
    GraspClutter6D = import_gc6d_api(args.api_root)
    camera = canonical_camera(args.camera)
    # split only affects API's internal path list; loadGrasp itself reads the requested scene/ann directly.
    g = GraspClutter6D(args.dataset_root, camera=camera, split=args.split)

    obj_ids = get_obj_ids_for_image(Path(args.dataset_root), scene_id, img_id)
    grasp_labels = g.loadGraspLabels(objIds=obj_ids)
    collision_labels = g.loadCollisionLabels(scene_id)
    gg = g.loadGrasp(
        sceneId=scene_id,
        annId=ann_id,
        format="6d",
        camera=camera,
        grasp_labels=grasp_labels,
        collision_labels=collision_labels,
        fric_coef_thresh=float(args.fric_coef_thresh),
        remove_invisible=bool(args.remove_invisible),
    )
    if args.sort_by_score:
        gg = gg.sort_by_score()
    if args.nms:
        gg = gg.nms()
    if args.topk > 0:
        gg = gg[: min(int(args.topk), len(gg))]
    return gg


def grasp_geometries_to_pcd(geoms: Sequence[o3d.geometry.Geometry], sample_points_per_grasp: int) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    for g in geoms:
        if isinstance(g, o3d.geometry.PointCloud):
            pcd += g
        elif isinstance(g, o3d.geometry.TriangleMesh):
            n = max(10, int(sample_points_per_grasp))
            pcd += g.sample_points_uniformly(number_of_points=n)
    return pcd


def save_combined_ply(
    path: Path,
    obs_pcd: o3d.geometry.PointCloud,
    rendered_pcd: o3d.geometry.PointCloud,
    grasp_geoms: Sequence[o3d.geometry.Geometry],
    sample_points_per_grasp: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = o3d.geometry.PointCloud()
    combined += obs_pcd
    combined += rendered_pcd
    combined += grasp_geometries_to_pcd(grasp_geoms, sample_points_per_grasp)
    ok = o3d.io.write_point_cloud(str(path), combined)
    if not ok:
        raise IOError(f"Failed to write point cloud: {path}")


def main() -> None:
    parser = argparse.ArgumentParser("Visualize GC6D observed depth, rendered virtual depth, and GT grasps.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--api_root", default=None, help="Optional parent dir containing graspclutter6dAPI.")
    parser.add_argument("--virtual_root", default=None, help="Default: dataset_root/virtual_scenes")
    parser.add_argument("--scene_id", type=int, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ann_id", type=int, default=None, help="GC6D annotation id, normally 0..12 for each camera.")
    group.add_argument("--img_id", type=int, default=None, help="GC6D image id, normally 1..52 across four cameras.")
    parser.add_argument("--camera", default="realsense-d435", choices=list(CAMERA_OFFSETS.keys()) + list(CAMERA_ALIASES.keys()))
    parser.add_argument("--split", default="all", choices=["all", "train", "test"], help="API split used only for initialization.")

    parser.add_argument("--observed_depth_scale", type=float, default=None, help="If None, use scene_camera depth_scale.")
    parser.add_argument("--rendered_depth_scale", type=float, default=None, help="If None, use scene_camera depth_scale.")
    parser.add_argument("--depth_trunc", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max_points", type=int, default=100000)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--rendered_offset_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Optional offset for blue rendered point cloud. Default overlays it with observed depth.")

    parser.add_argument("--fric_coef_thresh", type=float, default=0.4)
    parser.add_argument("--remove_invisible", action="store_true", help="Use API visibility filtering; requires mask/mask_visib files.")
    parser.add_argument("--sort_by_score", action="store_true", default=True)
    parser.add_argument("--no_sort_by_score", dest="sort_by_score", action="store_false")
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--grasp_mesh_sample_points", type=int, default=300)

    parser.add_argument("--save_ply", default=None, help="Path to save combined PLY. If omitted, uses output_root-like default under /tmp.")
    parser.add_argument("--save_grasps_npy", default=None, help="Optional path to save visualized GraspGroup array.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    args.dataset_root = str(Path(args.dataset_root).expanduser())
    dataset_root = Path(args.dataset_root)
    virtual_root = Path(args.virtual_root).expanduser() if args.virtual_root else dataset_root / "virtual_scenes"
    camera = canonical_camera(args.camera)
    scene_id = int(args.scene_id)
    if args.ann_id is None:
        img_id = int(args.img_id)
        ann_id = img_id_to_ann_id(img_id, camera)
    else:
        ann_id = int(args.ann_id)
        img_id = ann_id_to_img_id(ann_id, camera)

    scene_dir = dataset_root / "scenes" / f"{scene_id:06d}"
    scene_camera = load_json(scene_dir / "scene_camera.json")
    if str(img_id) not in scene_camera:
        raise KeyError(f"img_id={img_id} not found in {scene_dir / 'scene_camera.json'}")
    cam_info = scene_camera[str(img_id)]
    K = np.asarray(cam_info["cam_K"], dtype=np.float64).reshape(3, 3)
    camera_depth_scale = float(cam_info.get("depth_scale", 1.0))
    obs_scale = camera_depth_scale if args.observed_depth_scale is None else float(args.observed_depth_scale)
    ren_scale = camera_depth_scale if args.rendered_depth_scale is None else float(args.rendered_depth_scale)

    obs_path = scene_dir / "depth" / f"{img_id:06d}.png"
    if not obs_path.exists():
        raise FileNotFoundError(obs_path)
    virtual_path = resolve_virtual_depth_path(virtual_root, scene_id, camera, img_id, ann_id)

    observed_raw = np.array(Image.open(obs_path))
    rendered_raw = np.array(Image.open(virtual_path))
    if observed_raw.shape != rendered_raw.shape:
        raise ValueError(f"depth shape mismatch: observed={observed_raw.shape}, rendered={rendered_raw.shape}")

    obs_pts = depth_to_points(observed_raw, K, obs_scale, args.depth_trunc, args.stride, args.max_points, args.random_seed + 13)
    ren_pts = depth_to_points(rendered_raw, K, ren_scale, args.depth_trunc, args.stride, args.max_points, args.random_seed + 29)
    obs_pcd = make_pcd(obs_pts, (1.0, 0.0, 0.0))
    rendered_pcd = make_pcd(ren_pts, (0.0, 0.0, 1.0))
    if np.linalg.norm(np.asarray(args.rendered_offset_xyz, dtype=np.float64)) > 0:
        rendered_pcd = translate_pcd(rendered_pcd, args.rendered_offset_xyz)

    gg = load_gt_grasps(args, scene_id=scene_id, ann_id=ann_id, img_id=img_id)
    grasp_geoms = gg.to_open3d_geometry_list()

    if args.save_grasps_npy:
        out_npy = Path(args.save_grasps_npy).expanduser()
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_npy), np.asarray(gg.grasp_group_array, dtype=np.float32))

    if args.save_ply is None:
        save_ply = Path("/tmp") / f"gc6d_scene_{scene_id:06d}_{camera}_ann_{ann_id:04d}_img_{img_id:06d}_obs_red_render_blue_gtgrasps.ply"
    else:
        save_ply = Path(args.save_ply).expanduser()
    save_combined_ply(save_ply, obs_pcd, rendered_pcd, grasp_geoms, args.grasp_mesh_sample_points)

    print("[Info]")
    print(f"  scene_id={scene_id:06d}, camera={camera}, ann_id={ann_id}, img_id={img_id}")
    print(f"  observed depth: {obs_path}")
    print(f"  rendered depth: {virtual_path}")
    print(f"  K=\n{K}")
    print(f"  depth scales: observed={obs_scale}, rendered={ren_scale}, scene_camera={camera_depth_scale}")
    print(f"  points: observed(red)={len(obs_pcd.points)}, rendered(blue)={len(rendered_pcd.points)}")
    print(f"  grasps visualized: {len(gg)}")
    print(f"  saved ply: {save_ply}")

    if args.show:
        o3d.visualization.draw_geometries(
            [obs_pcd, rendered_pcd] + grasp_geoms,
            window_name=f"GC6D scene {scene_id:06d} ann {ann_id:04d} | red=observed, blue=rendered GT",
        )


if __name__ == "__main__":
    main()
