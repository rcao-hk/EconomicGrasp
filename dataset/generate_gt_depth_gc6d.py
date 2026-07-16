#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate GraspClutter6D virtual_scenes depth for ecograsp_dpt.

Corrected behavior:
- TSDF depth IS generated, but NOT saved separately.
- GraspClutter6D cam_R_w2c/cam_t_w2c is inverted by default for TSDF alignment.
- For each scene and camera, the script first integrates a TSDF volume from the
  observed depth frames of that camera.
- It then extracts a scene mesh and re-renders a per-frame TSDF scene depth.
- Finally, it writes:

    virtual_scenes = rendered TSDF background + rendered CAD object foreground

That is, background comes from the on-the-fly TSDF render, while object
foreground is overwritten using object-pose CAD rendering.

Expected GraspClutter6D layout:
  ROOT/
    split_info/grasp_train_scene_ids.json
    split_info/grasp_test_scene_ids.json
    scenes/<scene_id:06d>/
      rgb/<img_id:06d>.png                 # not required for TSDF, only depth is used
      depth/<img_id:06d>.png
      label/<img_id:06d>.png
      scene_gt.json
      scene_camera.json
    models_m/obj_<obj_id:06d>.ply

Default output:
  ROOT/virtual_scenes/<scene_id:06d>/<camera>/<img_id:06d>_depth.png

Recommended usage:
  python gc6d_generate_virtual_scenes_v3.py \
    --dataset_root /data/robotarm/dataset/GraspClutter6D \
    --split train --camera realsense-d435 --proc 8 --skip_done --write_stats
"""

from __future__ import annotations

import argparse
import copy
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
import open3d as o3d
from PIL import Image
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


def canonical_camera(camera: str) -> str:
    camera = CAMERA_ALIASES.get(str(camera), str(camera))
    if camera not in CAMERA_OFFSETS:
        raise ValueError(f"Unknown camera={camera}. Expected one of {list(CAMERA_OFFSETS)}")
    return camera


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


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    return int(ann_id) * 4 + CAMERA_OFFSETS[canonical_camera(camera)]


def img_id_to_ann_id(img_id: int, camera: str) -> int:
    offset = CAMERA_OFFSETS[canonical_camera(camera)]
    if img_id < offset or ((img_id - offset) % 4) != 0:
        raise ValueError(f"img_id={img_id} does not belong to camera={camera}")
    return (img_id - offset) // 4


def get_camera_img_ids(scene_camera: Dict[str, dict], camera: str, max_ann: Optional[int]) -> List[int]:
    cam = canonical_camera(camera)
    offset = CAMERA_OFFSETS[cam]
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


def output_depth_path(output_root: Path, scene_id: int, camera: str, img_id: int, flat_output: bool) -> Path:
    s6 = f"{int(scene_id):06d}"
    cam = canonical_camera(camera)
    if flat_output:
        return output_root / s6 / f"{int(img_id):06d}_depth.png"
    return output_root / s6 / cam / f"{int(img_id):06d}_depth.png"


def output_object_depth_path(output_root: Path, scene_id: int, camera: str, img_id: int, flat_output: bool) -> Path:
    s6 = f"{int(scene_id):06d}"
    cam = canonical_camera(camera)
    if flat_output:
        return output_root / "rendered_object_depth" / s6 / f"{int(img_id):06d}_depth.png"
    return output_root / "rendered_object_depth" / s6 / cam / f"{int(img_id):06d}_depth.png"


def depth_m_to_raw(depth_m: np.ndarray, depth_scale: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    depth_np = np.asarray(depth) * 1000.0 / depth_scale
    return depth_np.astype(np.uint16)
    # raw = np.rint(depth * 1000.0 / float(depth_scale))
    # return np.clip(raw, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def load_label(label_path: Path) -> np.ndarray:
    label = np.array(Image.open(label_path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def make_intrinsic_o3d(width: int, height: int, K: np.ndarray):
    return o3d.camera.PinholeCameraIntrinsic(
        int(width), int(height),
        float(K[0, 0]), float(K[1, 1]),
        float(K[0, 2]), float(K[1, 2]),
    )


def get_integration_module():
    if hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "integration"):
        return o3d.pipelines.integration
    return o3d.integration


def make_tsdf_volume(cfg):
    integration = get_integration_module()
    kwargs = dict(
        voxel_length=float(cfg.voxel_length),
        sdf_trunc=float(cfg.sdf_trunc),
        color_type=integration.TSDFVolumeColorType.NoColor,
    )
    try:
        return integration.ScalableTSDFVolume(
            volume_unit_resolution=int(cfg.volume_unit_resolution),
            depth_sampling_stride=int(cfg.depth_sampling_stride),
            **kwargs,
        )
    except TypeError:
        return integration.ScalableTSDFVolume(**kwargs)


def make_dummy_rgbd_from_depth(depth_raw: np.ndarray, depth_scale_bop: float, depth_trunc: float):
    """
    BOP convention: depth_m = depth_raw * depth_scale / 1000
    Open3D convention: depth_m = depth_raw / depth_scale_open3d
    => depth_scale_open3d = 1000 / depth_scale_bop
    """
    h, w = depth_raw.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color_img = o3d.geometry.Image(color)
    depth_img = o3d.geometry.Image(depth_raw.astype(np.uint16))
    depth_scale_open3d = 1000.0 / float(depth_scale_bop)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img,
        depth_img,
        depth_scale=float(depth_scale_open3d),
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False,
    )
    return rgbd


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0:
        return mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def build_w2c_from_scene_camera(cam_info: dict, invert: bool = True) -> np.ndarray:
    """Build the camera transform used by TSDF integration/rendering.

    GraspClutter6D names the fields cam_R_w2c/cam_t_w2c, but empirical
    visualization shows that using the inverse transform aligns the TSDF render
    with the observed depth. Therefore invert=True is the default.

    This function is only for the scene camera pose used by TSDF. Do NOT apply
    this inversion to object cam_R_m2c/cam_t_m2c, which are already object-to-camera.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(cam_info["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(cam_info["cam_t_w2c"], dtype=np.float64).reshape(3) / 1000.0
    return np.linalg.inv(T) if invert else T


def build_oc_from_scene_gt(obj: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(obj["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(obj["cam_t_m2c"], dtype=np.float64).reshape(3) / 1000.0
    return T


def integrate_scene_tsdf_mesh(scene_dir: Path, img_ids: Sequence[int], cfg) -> o3d.geometry.TriangleMesh:
    volume = make_tsdf_volume(cfg)
    scene_camera = load_json(scene_dir / "scene_camera.json")

    integrated = 0
    for idx, img_id in enumerate(img_ids):
        depth_path = scene_dir / "depth" / f"{int(img_id):06d}.png"
        if not depth_path.exists():
            continue
        depth_raw = np.array(Image.open(depth_path))
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        if not np.any(depth_raw > 0):
            continue

        cam_info = scene_camera[str(int(img_id))]
        K = np.asarray(cam_info["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale_bop = float(cam_info.get("depth_scale", cfg.default_depth_scale))
        rgbd = make_dummy_rgbd_from_depth(depth_raw, depth_scale_bop, cfg.depth_trunc)
        intrinsic = make_intrinsic_o3d(depth_raw.shape[1], depth_raw.shape[0], K)
        extrinsic = build_w2c_from_scene_camera(cam_info, invert=not cfg.no_invert_scene_camera_pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated += 1

    if integrated == 0:
        return o3d.geometry.TriangleMesh()

    mesh = volume.extract_triangle_mesh()
    mesh = clean_mesh(mesh)
    if cfg.mesh_smooth and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(cfg.mesh_smooth_iter))
        mesh = clean_mesh(mesh)
    return mesh


def load_mesh_model(model_root: Path, obj_id: int, model_unit: str, cache: Dict[int, o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    obj_id = int(obj_id)
    if obj_id not in cache:
        path = model_root / f"obj_{obj_id:06d}.ply"
        if not path.exists():
            raise FileNotFoundError(f"Missing CAD model: {path}")
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.vertices) == 0:
            raise ValueError(f"Empty mesh: {path}")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        if model_unit == "mm":
            mesh.scale(1.0 / 1000.0, center=(0.0, 0.0, 0.0))
        elif model_unit != "m":
            raise ValueError(f"model_unit must be m or mm, got {model_unit}")
        cache[obj_id] = mesh
    return copy.deepcopy(cache[obj_id])


def transform_scene_meshes(objs: Sequence[dict], model_root: Path, model_unit: str, mesh_cache: Dict[int, o3d.geometry.TriangleMesh]) -> List[o3d.geometry.TriangleMesh]:
    meshes: List[o3d.geometry.TriangleMesh] = []
    for obj in objs:
        mesh = load_mesh_model(model_root, int(obj["obj_id"]), model_unit, mesh_cache)
        mesh.transform(build_oc_from_scene_gt(obj))
        meshes.append(mesh)
    return meshes


def render_depth_raycast(meshes: Sequence[o3d.geometry.TriangleMesh], K: np.ndarray, width: int, height: int, depth_trunc: float) -> np.ndarray:
    if len(meshes) == 0:
        return np.zeros((int(height), int(width)), dtype=np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(np.asarray(K, dtype=np.float32)),
        extrinsic_matrix=o3d.core.Tensor(np.eye(4, dtype=np.float32)),
        width_px=int(width),
        height_px=int(height),
    )
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    rays_np = rays.numpy()
    valid = np.isfinite(t_hit)
    points = rays_np[..., :3] + rays_np[..., 3:] * t_hit[..., None]
    depth = points[..., 2].astype(np.float32)
    depth[~valid] = 0.0
    depth[depth <= 0.0] = 0.0
    depth[depth > float(depth_trunc)] = 0.0
    return depth


class OffscreenDepthRenderer:
    def __init__(self, width: int, height: int, depth_trunc: float):
        self.width = int(width)
        self.height = int(height)
        self.depth_trunc = float(depth_trunc)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self.renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
        self.renderer.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        )
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"

    def render(self, meshes: Sequence[o3d.geometry.TriangleMesh], K: np.ndarray) -> np.ndarray:
        self.renderer.scene.clear_geometry()
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            self.width, self.height,
            float(K[0, 0]), float(K[1, 1]),
            float(K[0, 2]), float(K[1, 2]),
        )
        self.renderer.setup_camera(intrinsic, np.eye(4, dtype=np.float64))
        for i, mesh in enumerate(meshes):
            self.renderer.scene.add_geometry(f"obj_{i}", mesh, self.material)
        depth = np.asarray(self.renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0.0] = 0.0
        depth[depth > self.depth_trunc] = 0.0
        return depth


def render_depth(meshes: Sequence[o3d.geometry.TriangleMesh], K: np.ndarray, width: int, height: int, cfg, offscreen_renderer: Optional[OffscreenDepthRenderer] = None) -> np.ndarray:
    if cfg.renderer == "raycast":
        return render_depth_raycast(meshes, K, width, height, cfg.depth_trunc)
    if cfg.renderer == "offscreen":
        if offscreen_renderer is None:
            offscreen_renderer = OffscreenDepthRenderer(width, height, cfg.depth_trunc)
        return offscreen_renderer.render(meshes, K)
    raise ValueError(f"Unknown renderer={cfg.renderer}")


def render_tsdf_background_depth(tsdf_mesh_world: o3d.geometry.TriangleMesh, cam_info: dict, width: int, height: int, cfg, offscreen_renderer: Optional[OffscreenDepthRenderer] = None) -> np.ndarray:
    K = np.asarray(cam_info["cam_K"], dtype=np.float64).reshape(3, 3)
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size for TSDF rendering.")
    if len(tsdf_mesh_world.vertices) == 0:
        return np.zeros((height, width), dtype=np.float32)
    mesh_cam = copy.deepcopy(tsdf_mesh_world)
    mesh_cam.transform(build_w2c_from_scene_camera(cam_info, invert=not cfg.no_invert_scene_camera_pose))
    return render_depth([mesh_cam], K, width, height, cfg, offscreen_renderer)


def render_object_depth(objs: Sequence[dict], K: np.ndarray, width: int, height: int, cfg, mesh_cache: Dict[int, o3d.geometry.TriangleMesh], offscreen_renderer: Optional[OffscreenDepthRenderer] = None) -> np.ndarray:
    meshes = transform_scene_meshes(objs, Path(cfg.model_root), cfg.model_unit, mesh_cache)
    return render_depth(meshes, K, width, height, cfg, offscreen_renderer)


def foreground_mask(label: np.ndarray, rendered_raw: np.ndarray, mask_source: str) -> np.ndarray:
    label_fg = label > 0
    render_fg = rendered_raw > 0
    if mask_source == "label":
        return label_fg
    if mask_source == "render":
        return render_fg
    if mask_source == "union":
        return label_fg | render_fg
    if mask_source == "intersection":
        return label_fg & render_fg
    raise ValueError(f"Unknown mask_source={mask_source}")


def compose_virtual_raw(tsdf_bg_raw: np.ndarray, rendered_obj_raw: np.ndarray, label: np.ndarray, mask_source: str, fg_invalid_fallback: str) -> Tuple[np.ndarray, Dict[str, float]]:
    if tsdf_bg_raw.shape != rendered_obj_raw.shape or label.shape != rendered_obj_raw.shape:
        raise ValueError(f"shape mismatch: tsdf={tsdf_bg_raw.shape}, obj={rendered_obj_raw.shape}, label={label.shape}")

    fg = foreground_mask(label, rendered_obj_raw, mask_source)
    obj_valid = rendered_obj_raw > 0
    bg_valid = tsdf_bg_raw > 0

    target = tsdf_bg_raw.astype(np.uint16).copy()
    if fg_invalid_fallback == "zero":
        target[fg] = 0
    elif fg_invalid_fallback == "tsdf":
        pass
    else:
        raise ValueError(f"Unknown fg_invalid_fallback={fg_invalid_fallback}")

    write_mask = fg & obj_valid
    target[write_mask] = rendered_obj_raw[write_mask]

    bg = ~fg
    stats = {
        "fg_ratio": float(fg.mean()),
        "fg_render_valid_ratio": float(write_mask.sum() / max(int(fg.sum()), 1)),
        "bg_tsdf_valid_ratio": float((bg & bg_valid).sum() / max(int(bg.sum()), 1)),
        "target_valid_ratio": float((target > 0).mean()),
        "background_from_tsdf": 1.0,
    }
    return target, stats


@dataclass
class RunStats:
    scenes_done: int = 0
    rendered: int = 0
    skipped_done: int = 0
    missing_gt: int = 0
    failed: int = 0

    def add(self, other: Dict[str, int]) -> None:
        self.scenes_done += int(other.get("scenes_done", 0))
        self.rendered += int(other.get("rendered", 0))
        self.skipped_done += int(other.get("skipped_done", 0))
        self.missing_gt += int(other.get("missing_gt", 0))
        self.failed += int(other.get("failed", 0))

    def as_dict(self) -> Dict[str, int]:
        return dict(
            scenes_done=self.scenes_done,
            rendered=self.rendered,
            skipped_done=self.skipped_done,
            missing_gt=self.missing_gt,
            failed=self.failed,
        )


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def process_one_scene(scene_id: int, cfg) -> Dict[str, object]:
    stats = RunStats()
    rows: List[Dict[str, object]] = []
    camera = canonical_camera(cfg.camera)
    scene_dir = Path(cfg.dataset_root) / "scenes" / f"{int(scene_id):06d}"
    mesh_cache: Dict[int, o3d.geometry.TriangleMesh] = {}

    try:
        scene_gt = load_json(scene_dir / "scene_gt.json")
        scene_camera = load_json(scene_dir / "scene_camera.json")
        img_ids = get_camera_img_ids(scene_camera, camera, cfg.max_ann)
        if cfg.img_ids:
            wanted = set(parse_int_list(cfg.img_ids))
            img_ids = [x for x in img_ids if x in wanted]
        if not img_ids:
            return {"ok": False, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": "no image ids"}

        # Optional scene-level early skip.
        if cfg.skip_done:
            all_done = True
            for img_id in img_ids:
                out_path = output_depth_path(Path(cfg.output_root), scene_id, camera, img_id, cfg.flat_output)
                if not out_path.exists():
                    all_done = False
                    break
            if all_done:
                stats.skipped_done += len(img_ids)
                stats.scenes_done += 1
                return {"ok": True, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": "all_done"}

        tsdf_mesh_world = integrate_scene_tsdf_mesh(scene_dir, img_ids, cfg)

        offscreen_renderer: Optional[OffscreenDepthRenderer] = None
        if cfg.renderer == "offscreen":
            first_depth = np.array(Image.open(scene_dir / "depth" / f"{int(img_ids[0]):06d}.png"))
            h, w = first_depth.shape[:2]
            offscreen_renderer = OffscreenDepthRenderer(w, h, cfg.depth_trunc)

        for img_id in img_ids:
            out_path = output_depth_path(Path(cfg.output_root), scene_id, camera, img_id, cfg.flat_output)
            if cfg.skip_done and out_path.exists():
                stats.skipped_done += 1
                continue

            if str(img_id) not in scene_gt or str(img_id) not in scene_camera:
                stats.missing_gt += 1
                continue

            depth_path = scene_dir / "depth" / f"{int(img_id):06d}.png"
            label_path = scene_dir / "label" / f"{int(img_id):06d}.png"
            if not depth_path.exists() or not label_path.exists():
                stats.missing_gt += 1
                continue

            obs_depth_raw = np.array(Image.open(depth_path))
            height, width = obs_depth_raw.shape[:2]
            label = load_label(label_path)
            cam_info = scene_camera[str(img_id)]
            K = np.asarray(cam_info["cam_K"], dtype=np.float64).reshape(3, 3)
            camera_depth_scale = float(cam_info.get("depth_scale", cfg.default_depth_scale))
            render_depth_scale = camera_depth_scale if cfg.render_depth_scale is None else float(cfg.render_depth_scale)

            tsdf_bg_m = render_tsdf_background_depth(tsdf_mesh_world, cam_info, width, height, cfg, offscreen_renderer=offscreen_renderer)
            tsdf_bg_raw = depth_m_to_raw(tsdf_bg_m, render_depth_scale)

            objs = scene_gt[str(img_id)]
            rendered_obj_m = render_object_depth(objs, K, width, height, cfg, mesh_cache, offscreen_renderer=offscreen_renderer)
            rendered_obj_raw = depth_m_to_raw(rendered_obj_m, render_depth_scale)

            target_raw, frame_stats = compose_virtual_raw(tsdf_bg_raw, rendered_obj_raw, label, cfg.mask_source, cfg.fg_invalid_fallback)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target_raw).save(out_path)

            if cfg.save_rendered_object_depth:
                obj_path = output_object_depth_path(Path(cfg.output_root), scene_id, camera, img_id, cfg.flat_output)
                obj_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rendered_obj_raw).save(obj_path)

            frame_stats.update({
                "scene_id": int(scene_id),
                "camera": camera,
                "img_id": int(img_id),
                "ann_id": int(img_id_to_ann_id(img_id, camera)),
                "H": int(height),
                "W": int(width),
                "camera_depth_scale": camera_depth_scale,
                "render_depth_scale": render_depth_scale,
                "obs_valid_ratio": float((obs_depth_raw > 0).mean()),
                "tsdf_valid_ratio": float((tsdf_bg_raw > 0).mean()),
                "obj_render_valid_ratio": float((rendered_obj_raw > 0).mean()),
                "output_path": str(out_path),
            })
            rows.append(frame_stats)
            stats.rendered += 1

        stats.scenes_done += 1
        if cfg.write_stats and rows:
            stats_dir = Path(cfg.output_root) / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            write_csv_rows(stats_dir / f"{int(scene_id):06d}_{camera}.csv", rows)

        return {"ok": True, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": "done"}

    except Exception:
        stats.failed += 1
        return {"ok": False, "scene_id": int(scene_id), "stats": stats.as_dict(), "rows": rows, "message": traceback.format_exc()}


def _scene_worker(args):
    scene_id, cfg = args
    return process_one_scene(scene_id, cfg)


def run_scene_parallel(scene_ids: Sequence[int], cfg) -> List[Dict[str, object]]:
    if cfg.proc <= 1:
        return [process_one_scene(sid, cfg) for sid in tqdm(scene_ids, desc="Scenes")]
    ctx = mp.get_context(cfg.mp_start_method)
    jobs = [(sid, cfg) for sid in scene_ids]
    out: List[Dict[str, object]] = []
    with ctx.Pool(processes=int(cfg.proc), maxtasksperchild=1) as pool:
        for ret in tqdm(pool.imap_unordered(_scene_worker, jobs), total=len(jobs), desc="Scenes"):
            out.append(ret)
            status = "OK" if ret["ok"] else "FAIL"
            msg0 = str(ret["message"]).splitlines()[0] if ret.get("message") else ""
            print(f"[scene {int(ret['scene_id']):06d}] {status} {ret['stats']} {msg0}")
    return out


def summarize(results: Sequence[Dict[str, object]], n_items: int) -> None:
    total = RunStats()
    failed = []
    for r in results:
        total.add(r.get("stats", {}))
        if not r.get("ok", False):
            failed.append(r.get("scene_id"))
    print("\n[Summary]")
    print(f"  scenes:        {n_items}")
    print(f"  scenes_done:   {total.scenes_done}")
    print(f"  rendered:      {total.rendered}")
    print(f"  skipped_done:  {total.skipped_done}")
    print(f"  missing_gt:    {total.missing_gt}")
    print(f"  failed:        {total.failed}")
    if failed:
        print(f"  failed scenes: {failed[:50]}{' ...' if len(failed) > 50 else ''}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate GC6D virtual_scenes by on-the-fly TSDF background rendering and CAD foreground rendering.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="train", help="train/test/all or explicit scene ids like '5,7,10-20'.")
    p.add_argument("--scene_ids", default=None, help="Override --split, e.g. '5,7,10-20'.")
    p.add_argument("--camera", default="realsense-d435", choices=list(CAMERA_OFFSETS.keys()) + list(CAMERA_ALIASES.keys()))
    p.add_argument("--max_ann", type=int, default=13, help="GC6D default ann_id is 0..12 per camera. Use <=0 for all matching image ids.")
    p.add_argument("--img_ids", default=None, help="Optional explicit GC6D image ids, e.g. '2,6,10'.")

    p.add_argument("--model_root", default=None, help="Default: dataset_root/models_m")
    p.add_argument("--model_unit", default="m", choices=["m", "mm"], help="GC6D models_m is normally meters.")
    p.add_argument("--output_root", default=None, help="Default: dataset_root/virtual_scenes")
    p.add_argument("--flat_output", action="store_true", help="Save without camera folder.")

    p.add_argument("--mask_source", default="label", choices=["label", "render", "union", "intersection"])
    p.add_argument("--fg_invalid_fallback", default="zero", choices=["zero", "tsdf"], help="For foreground pixels where CAD rendering is invalid.")
    p.add_argument("--renderer", default="raycast", choices=["raycast", "offscreen"], help="raycast is more stable for multiprocessing/headless servers.")
    p.add_argument("--depth_trunc", type=float, default=2.0)
    p.add_argument("--default_depth_scale", type=float, default=1.0)
    p.add_argument("--render_depth_scale", type=float, default=None, help="Raw encoding for saved virtual depth. If None, use scene_camera depth_scale.")
    p.add_argument("--save_rendered_object_depth", action="store_true", help="Also save object-only rendered depth under output_root/rendered_object_depth.")

    # TSDF parameters
    p.add_argument("--voxel_length", type=float, default=0.003)
    p.add_argument("--sdf_trunc", type=float, default=0.015)
    p.add_argument("--volume_unit_resolution", type=int, default=16)
    p.add_argument("--depth_sampling_stride", type=int, default=1)
    p.add_argument("--mesh_smooth", action="store_true")
    p.add_argument("--mesh_smooth_iter", type=int, default=5)
    p.add_argument("--default_width", type=int, default=1280)
    p.add_argument("--default_height", type=int, default=720)
    p.add_argument("--no_invert_scene_camera_pose", action="store_true",
                   help="Use cam_R_w2c/cam_t_w2c directly for TSDF instead of the empirically correct inverse.")

    # Parallelism
    p.add_argument("--proc", type=int, default=1, help="Parallel scenes. Each worker integrates its own scene TSDF then renders all frames.")
    p.add_argument("--mp_start_method", default="spawn", choices=["spawn", "forkserver", "fork"])
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--write_stats", action="store_true")
    return p


def main() -> None:
    cfg = build_argparser().parse_args()
    cfg.dataset_root = str(Path(cfg.dataset_root).expanduser())
    if cfg.model_root is None:
        cfg.model_root = str(Path(cfg.dataset_root) / "models_m")
    if cfg.output_root is None:
        cfg.output_root = str(Path(cfg.dataset_root) / "virtual_scenes")
    if cfg.max_ann is not None and cfg.max_ann <= 0:
        cfg.max_ann = None

    if cfg.scene_ids:
        scene_ids = parse_int_list(cfg.scene_ids)
    else:
        scene_ids = read_split_scene_ids(Path(cfg.dataset_root), cfg.split)

    print(f"[Main] dataset_root={cfg.dataset_root}")
    print(f"[Main] camera={canonical_camera(cfg.camera)} split={cfg.split} scenes={len(scene_ids)}")
    print(f"[Main] output_root={cfg.output_root}")
    print(f"[Main] renderer={cfg.renderer} proc={cfg.proc}")
    print("[Main] composition: integrate TSDF per scene -> render TSDF background per frame -> overwrite foreground with CAD rendered depth")

    results = run_scene_parallel(scene_ids, cfg)
    summarize(results, len(scene_ids))


if __name__ == "__main__":
    main()
