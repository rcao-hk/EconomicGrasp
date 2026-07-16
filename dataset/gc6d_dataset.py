#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraspClutter6D dataloader for ecograsp_dpt.

This class is a GC6D counterpart of the existing GraspNetMultiDataset used by
ecograsp_dpt. It does NOT require pre-generated meta.mat files. Instead, it
reads GC6D BOP-style metadata directly from:
    scenes/<scene_id:06d>/scene_gt.json
    scenes/<scene_id:06d>/scene_camera.json

Required pre-generated data:
    virtual_scenes/<scene_id>/<camera>/<img_id>_depth.png
    graspness or graspness_instance
    virtual_graspness or virtual_graspness_instance, only if use_gt_depth=True
    economic_grasp_label_300views or economic_grasp_label_300views_extend_angle

Expected GC6D raw layout:
    ROOT/scenes/<scene_id:06d>/rgb/<img_id:06d>.png
    ROOT/scenes/<scene_id:06d>/depth/<img_id:06d>.png
    ROOT/scenes/<scene_id:06d>/label/<img_id:06d>.png
    ROOT/scenes/<scene_id:06d>/scene_gt.json
    ROOT/scenes/<scene_id:06d>/scene_camera.json

Important:
    If graspness was generated with --mask_mode workspace_depth, instantiate this
    dataloader with mask_mode="workspace_depth" and exactly the same
    workspace_outlier / workspace_pose_mode / workspace_depth_trunc /
    factor_depth_mode. Otherwise graspness length will not match.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    return int(ann_id) * 4 + CAMERA_OFFSETS[canonical_camera(camera)]


def img_id_to_ann_id(img_id: int, camera: str) -> int:
    camera = canonical_camera(camera)
    offset = CAMERA_OFFSETS[camera]
    img_id = int(img_id)
    if img_id < offset or ((img_id - offset) % 4) != 0:
        raise ValueError(f"img_id={img_id} does not belong to camera={camera}")
    return (img_id - offset) // 4


def load_json(path: Union[str, Path]):
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


def read_split_scene_ids(root: Union[str, Path], split: str) -> List[int]:
    root = Path(root)
    split_l = str(split).lower()
    split_dir = root / "split_info"

    if split_l == "train":
        data = load_json(split_dir / "grasp_train_scene_ids.json")
    elif split_l in ("train_remove_overlap", "remove_overlap", "train_no_overlap", "train_filtered"):
        data = load_json(split_dir / "grasp_train_remove_overlap_scene_ids.json")
    elif split_l == "test":
        data = load_json(split_dir / "grasp_test_scene_ids.json")
    elif split_l == "all":
        data = list(load_json(split_dir / "grasp_train_scene_ids.json"))
        data += list(load_json(split_dir / "grasp_test_scene_ids.json"))
    else:
        return parse_int_list(split)

    if isinstance(data, dict):
        vals = list(data.keys()) if all(str(k).isdigit() for k in data.keys()) else list(data.values())
    else:
        vals = list(data)
    return sorted({int(x) for x in vals})


def get_camera_img_ids(scene_camera: Dict[str, dict], camera: str, max_ann: Optional[int] = 13) -> List[int]:
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


def load_label(path: Union[str, Path]) -> np.ndarray:
    label = np.array(Image.open(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def get_factor_depth(camera: str, cam_info: dict, mode: str = "bop", fixed: Optional[float] = None) -> float:
    """Return denominator so that depth_m = raw_depth / factor_depth."""
    mode = str(mode).lower()
    camera = canonical_camera(camera)

    if mode == "fixed":
        if fixed is None:
            raise ValueError("fixed_factor_depth is required when factor_depth_mode='fixed'")
        return float(fixed)

    if mode == "camera":
        if camera in ["realsense-d415", "realsense-d435"]:
            return 1000.0
        if camera in ["azure-kinect", "zivid"]:
            return 10000.0
        raise ValueError(camera)

    if mode == "bop":
        # BOP convention: depth_m = raw * depth_scale / 1000,
        # so factor_depth = 1000 / depth_scale.
        ds = float(cam_info.get("depth_scale", 1.0))
        if ds <= 0:
            raise ValueError(f"Invalid depth_scale={ds}")
        return 1000.0 / ds

    raise ValueError(f"Unknown factor_depth_mode={mode}")


def depth_to_cloud(depth_raw: np.ndarray, K: np.ndarray, factor_depth: float) -> np.ndarray:
    """Back-project depth to organized camera-frame cloud, shape HxWx3."""
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


def transform_points_np(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    return (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def get_camera_to_workspace_transform(cam_info: dict, mode: str) -> Optional[np.ndarray]:
    """Same semantics as gc6d_generate_graspness_v2/debug.py.

    none:    workspace bbox in current camera frame.
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
    outlier: float = 0.02,
    pose_mode: str = "json",
    depth_trunc: float = 0.0,
) -> np.ndarray:
    """GraspNet-1B-style workspace mask for GC6D."""
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
        cloud_box_flat = transform_points_np(cloud_flat, T)
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
    depth: np.ndarray,
    label: np.ndarray,
    cloud: np.ndarray,
    cam_info: dict,
    mode: str,
    bbox_margin_ratio: float = 0.1,
    workspace_outlier: float = 0.02,
    workspace_pose_mode: str = "json",
    workspace_depth_trunc: float = 0.0,
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
            outlier=workspace_outlier,
            pose_mode=workspace_pose_mode,
            depth_trunc=workspace_depth_trunc,
        )

    raise ValueError(f"Unsupported mask_mode={mode}")


def pose_from_gc6d_obj(obj: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(obj["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
    return T


def scene_gt_pose_map(scene_gt: Dict[str, list], img_id: int) -> Dict[int, np.ndarray]:
    """Map obj_id -> 3x4 object-to-camera pose for current image."""
    out: Dict[int, np.ndarray] = {}
    for obj in scene_gt[str(int(img_id))]:
        obj_id = int(obj["obj_id"])
        T = pose_from_gc6d_obj(obj)
        out[obj_id] = T[:3, :4].astype(np.float32)
    return out


def transform_point_cloud(points: np.ndarray, transform: np.ndarray, format: str = "3x3") -> np.ndarray:
    if format == "3x3":
        return points @ transform.T
    if format == "4x4":
        return transform_points_np(points, transform)
    raise ValueError(format)


def _read_depth_png(path: Union[str, Path]) -> np.ndarray:
    depth = np.array(Image.open(path))
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


class GraspClutter6DMultiDataset(Dataset):
    def __init__(
        self,
        root: str,
        camera: str = "realsense-d435",
        split: str = "train",
        num_points: int = 20000,
        voxel_size: float = 0.005,
        remove_outlier: Optional[bool] = None,
        remove_invisible: bool = True,
        augment: bool = False,
        load_label: bool = True,
        use_gt_depth: bool = False,
        use_fuse_depth: bool = False,
        graspness_mode: str = "instance",
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        depth_strides: int = 2,
        extend_angle: bool = False,
        # GC6D-specific
        scene_ids: Optional[Union[str, Sequence[int]]] = None,
        max_ann: Optional[int] = 13,
        mask_mode: str = "workspace_depth",
        bbox_margin_ratio: float = 0.1,
        workspace_outlier: float = 0.02,
        workspace_pose_mode: str = "json",
        workspace_depth_trunc: float = 0.0,
        factor_depth_mode: str = "bop",
        fixed_factor_depth: Optional[float] = None,
        virtual_root: Optional[str] = None,
        economic_label_root: Optional[str] = None,
        scene_name_format: str = "plain6",
        strict_label_check: bool = True,
        fallback_to_depth_mask_on_empty: bool = False,
    ):
        self.root = Path(root)
        self.camera = canonical_camera(camera)
        self.split = split
        self.voxel_size = float(voxel_size)
        self.num_points = int(num_points)
        self.remove_invisible = remove_invisible
        self.augment = bool(augment)
        self.load_label = bool(load_label)
        self.use_gt_depth = bool(use_gt_depth)
        self.use_fuse_depth = bool(use_fuse_depth)
        self.graspness_mode = str(graspness_mode)
        self.extend_angle = bool(extend_angle)

        # If remove_outlier is explicitly passed, map it to mask_mode for
        # backward compatibility. Otherwise trust mask_mode.
        if remove_outlier is not None:
            self.mask_mode = "workspace_depth" if bool(remove_outlier) else "depth"
        else:
            self.mask_mode = str(mask_mode)
        self.bbox_margin_ratio = float(bbox_margin_ratio)
        self.workspace_outlier = float(workspace_outlier)
        self.workspace_pose_mode = str(workspace_pose_mode)
        self.workspace_depth_trunc = float(workspace_depth_trunc)
        self.factor_depth_mode = str(factor_depth_mode)
        self.fixed_factor_depth = fixed_factor_depth
        self.virtual_root = Path(virtual_root) if virtual_root is not None else (self.root / "virtual_scenes")
        self.strict_label_check = bool(strict_label_check)
        self.fallback_to_depth_mask_on_empty = bool(fallback_to_depth_mask_on_empty)

        self.resize_shape = (448, 448)  # (H, W)
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.depth_prob_min = min_depth
        self.depth_prob_max = max_depth
        self.depth_prob_bins = int(bin_num)
        self.depth_prob_strides = int(depth_strides)
        self.depth_prob_valid_threshold = -1
        self.gt_factor_depth = None

        self.scene_camera_cache: Dict[int, dict] = {}
        self.scene_gt_cache: Dict[int, dict] = {}

        if scene_ids is not None:
            if isinstance(scene_ids, str):
                self.scene_ids = parse_int_list(scene_ids)
            else:
                self.scene_ids = [int(x) for x in scene_ids]
        else:
            self.scene_ids = read_split_scene_ids(self.root, split)

        self.max_ann = max_ann if (max_ann is None or int(max_ann) > 0) else None

        self.samples: List[Tuple[int, int, int]] = []  # scene_id, img_id, ann_id
        self.colorpath: List[str] = []
        self.depthpath: List[str] = []
        self.gtdepthpath: List[str] = []
        self.fusedepthpath: List[str] = []
        self.labelpath: List[str] = []
        self.scenename: List[str] = []
        self.frameid: List[int] = []  # ann_id, not img_id
        self.imgid: List[int] = []
        self.graspnesspath: List[str] = []
        self.gtgraspnesspath: List[str] = []
        self.grasp_labels: Dict[str, str] = {}

        if economic_label_root is not None:
            self.economic_label_root = Path(economic_label_root)
        else:
            folder = "economic_grasp_label_300views_extend_angle" if self.extend_angle else "economic_grasp_label_300views"
            self.economic_label_root = self.root / folder

        self.scene_name_format = scene_name_format

        for scene_id in tqdm(self.scene_ids, desc="Loading GC6D scene paths"):
            scene_camera = self._load_scene_camera(scene_id)
            img_ids = get_camera_img_ids(scene_camera, self.camera, self.max_ann)

            scene_key = self._scene_key(scene_id)
            if self.load_label:
                self.grasp_labels[scene_key] = str(self._resolve_economic_label_path(scene_id))

            for img_id in img_ids:
                ann_id = img_id_to_ann_id(img_id, self.camera)
                self.samples.append((int(scene_id), int(img_id), int(ann_id)))

                s6 = f"{int(scene_id):06d}"
                self.colorpath.append(str(self.root / "scenes" / s6 / "rgb" / f"{int(img_id):06d}.png"))
                self.depthpath.append(str(self.root / "scenes" / s6 / "depth" / f"{int(img_id):06d}.png"))
                self.gtdepthpath.append(str(self._resolve_virtual_depth_path(scene_id, img_id)))
                self.fusedepthpath.append(str(self._resolve_tsdf_depth_path(scene_id, img_id)))
                self.labelpath.append(str(self.root / "scenes" / s6 / "label" / f"{int(img_id):06d}.png"))

                if self.graspness_mode == "scene":
                    g_root = self.root / "graspness"
                    vg_root = self.root / "virtual_graspness"
                elif self.graspness_mode == "instance":
                    g_root = self.root / "graspness_instance"
                    vg_root = self.root / "virtual_graspness_instance"
                else:
                    raise ValueError(f"Unsupported graspness_mode={self.graspness_mode}")

                self.graspnesspath.append(str(g_root / s6 / self.camera / f"{int(img_id):06d}.npy"))
                self.gtgraspnesspath.append(str(vg_root / s6 / self.camera / f"{int(img_id):06d}.npy"))
                self.scenename.append(scene_key)
                self.frameid.append(int(ann_id))
                self.imgid.append(int(img_id))

    def _scene_key(self, scene_id: int) -> str:
        if self.scene_name_format == "plain6":
            return f"{int(scene_id):06d}"
        if self.scene_name_format == "scene6":
            return f"scene_{int(scene_id):06d}"
        if self.scene_name_format == "scene4":
            return f"scene_{int(scene_id):04d}"
        raise ValueError(f"Unknown scene_name_format={self.scene_name_format}")

    def _load_scene_camera(self, scene_id: int) -> dict:
        scene_id = int(scene_id)
        if scene_id not in self.scene_camera_cache:
            self.scene_camera_cache[scene_id] = load_json(self.root / "scenes" / f"{scene_id:06d}" / "scene_camera.json")
        return self.scene_camera_cache[scene_id]

    def _load_scene_gt(self, scene_id: int) -> dict:
        scene_id = int(scene_id)
        if scene_id not in self.scene_gt_cache:
            self.scene_gt_cache[scene_id] = load_json(self.root / "scenes" / f"{scene_id:06d}" / "scene_gt.json")
        return self.scene_gt_cache[scene_id]

    def _resolve_virtual_depth_path(self, scene_id: int, img_id: int) -> Path:
        s6 = f"{int(scene_id):06d}"
        candidates = [
            self.virtual_root / s6 / self.camera / f"{int(img_id):06d}_depth.png",
            self.virtual_root / s6 / f"{int(img_id):06d}_depth.png",
            self.virtual_root / s6 / self.camera / f"{int(img_id):06d}.png",
            self.virtual_root / s6 / f"{int(img_id):06d}.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]

    def _resolve_tsdf_depth_path(self, scene_id: int, img_id: int) -> Path:
        # Usually unused for GC6D because virtual_scenes is already fused.
        s6 = f"{int(scene_id):06d}"
        return self.root / "tsdf_depth" / s6 / self.camera / f"{int(img_id):06d}_depth.png"

    def _resolve_economic_label_path(self, scene_id: int) -> Path:
        candidates = [
            self.economic_label_root / f"{int(scene_id):06d}_labels.npz",
            self.economic_label_root / f"scene_{int(scene_id):06d}_labels.npz",
            self.economic_label_root / f"scene_{int(scene_id):04d}_labels.npz",
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]

    def __len__(self):
        return len(self.samples)

    def scene_list(self):
        return self.scenename

    def augment_data(self, point_clouds, object_poses_list):
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [ 0, 1, 0],
                                 [ 0, 0, 1]], dtype=np.float32)
            point_clouds = transform_point_cloud(point_clouds, flip_mat, "3x3")
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        return self.get_data(index)

    # ---------------------------------------------------------------------
    # image / geometry helpers, same interface as GraspNetMultiDataset
    # ---------------------------------------------------------------------
    def _crop_box_from_mask(self, mask):
        H, W = mask.shape
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0, 0, W, H
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        return int(x0), int(y0), int(x1), int(y1)

    def get_resized_idxs_from_flat_crop(self, pix_flat, orig_hw, crop_box, out_hw=(448, 448)):
        H, W = orig_hw
        outH, outW = out_hw
        x0, y0, x1, y1 = crop_box
        cw, ch = (x1 - x0), (y1 - y0)

        ys, xs = np.unravel_index(pix_flat, (H, W))
        xs = xs.astype(np.float32) - float(x0)
        ys = ys.astype(np.float32) - float(y0)

        xs = np.clip(xs, 0, cw - 1e-6)
        ys = np.clip(ys, 0, ch - 1e-6)

        xf = np.floor(xs * (outW / float(cw))).astype(np.int64)
        yf = np.floor(ys * (outH / float(ch))).astype(np.int64)
        xf = np.clip(xf, 0, outW - 1)
        yf = np.clip(yf, 0, outH - 1)
        return (yf * outW + xf).astype(np.int64)

    def resize_intrinsics_with_crop(self, intrinsic, crop_box, out_hw=(448, 448)):
        x0, y0, x1, y1 = crop_box
        outH, outW = out_hw
        cw = float(x1 - x0)
        ch = float(y1 - y0)

        sx = outW / cw
        sy = outH / ch

        K = intrinsic.astype(np.float32).copy()
        K[0, 2] -= float(x0)
        K[1, 2] -= float(y0)

        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    def build_resized_depth_m_with_crop(self, depth_raw, factor_depth, crop_box, out_hw=(448, 448)):
        x0, y0, x1, y1 = crop_box
        outH, outW = out_hw

        depth_m = depth_raw.astype(np.float32) / float(factor_depth)
        depth_crop = depth_m[y0:y1, x0:x1].copy()
        depth_m_resized = cv2.resize(depth_crop, (outW, outH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return depth_m_resized

    def build_depth_prob_gt(self, depth_m_resized: np.ndarray):
        Hr, Wr = self.resize_shape
        s = self.depth_prob_strides
        Ht, Wt = Hr // s, Wr // s
        Nfeat = Ht * Wt
        D = int(self.depth_prob_bins)
        min_d, max_d = float(self.depth_prob_min), float(self.depth_prob_max)

        depth = depth_m_resized.astype(np.float32)
        valid = depth > 0

        depth_clip = np.clip(depth, min_d, max_d)
        t = (depth_clip - min_d) / (max_d - min_d + 1e-12) * (D - 1)

        i0 = np.floor(t).astype(np.int64)
        i0 = np.clip(i0, 0, D - 1)
        i1 = np.clip(i0 + 1, 0, D - 1)

        w1 = (t - i0.astype(np.float32)).astype(np.float32)
        w0 = (1.0 - w1).astype(np.float32)

        inv = ~valid
        i0[inv] = 0
        i1[inv] = 0
        w0[inv] = 0.0
        w1[inv] = 0.0

        ys, xs = np.indices((Hr, Wr))
        pid = (ys // s) * Wt + (xs // s)
        pid_flat = pid.reshape(-1).astype(np.int64)

        flat_i0 = i0.reshape(-1)
        flat_i1 = i1.reshape(-1)
        flat_w0 = w0.reshape(-1)
        flat_w1 = w1.reshape(-1)
        flat_valid = valid.reshape(-1).astype(np.float32)

        prob = np.zeros((Nfeat, D), dtype=np.float32)
        np.add.at(prob, (pid_flat, flat_i0), flat_w0)
        np.add.at(prob, (pid_flat, flat_i1), flat_w1)

        vcnt = np.bincount(pid_flat, weights=flat_valid, minlength=Nfeat).astype(np.float32)
        den = np.maximum(vcnt, 1.0)
        prob = prob / den[:, None]
        prob[vcnt < 1.0] = 0.0

        vratio = vcnt / float(s * s)
        return prob[None].astype(np.float32), vratio[None].astype(np.float32)

    def build_fused_gt_depth_m(self, gt_depth_raw, seg_raw, index, factor_depth):
        """GC6D virtual_scenes is normally already fused, so use_fuse_depth=False is recommended."""
        fd = float(self.gt_factor_depth) if (self.gt_factor_depth is not None) else float(factor_depth)
        gt_depth_m = gt_depth_raw.astype(np.float32) / fd

        if not self.use_fuse_depth:
            return gt_depth_m

        fuse_path = self.fusedepthpath[index]
        if not os.path.exists(fuse_path):
            raise FileNotFoundError(f"use_fuse_depth=True but fused depth not found: {fuse_path}")

        fuse_depth_raw = np.array(Image.open(fuse_path))
        if fuse_depth_raw.ndim == 3:
            fuse_depth_raw = fuse_depth_raw[:, :, 0]
        fuse_depth_m = fuse_depth_raw.astype(np.float32) / fd

        if fuse_depth_m.shape != gt_depth_m.shape:
            raise ValueError(f"Fused depth shape mismatch: fuse={fuse_depth_m.shape}, gt={gt_depth_m.shape}")

        if seg_raw.shape != gt_depth_m.shape:
            raise ValueError(f"Seg shape mismatch: seg={seg_raw.shape}, gt={gt_depth_m.shape}")

        obj_region = seg_raw > 0
        bg_region = ~obj_region

        target_depth_m = np.zeros_like(gt_depth_m, dtype=np.float32)
        obj_valid = obj_region & (gt_depth_m > 0)
        target_depth_m[obj_valid] = gt_depth_m[obj_valid]
        bg_valid = bg_region & (fuse_depth_m > 0)
        target_depth_m[bg_valid] = fuse_depth_m[bg_valid]
        return target_depth_m

    def _mask_and_sample(self, depth, seg, cloud, mask):
        H, W = depth.shape
        valid_flat = np.flatnonzero(mask)
        cloud_masked = cloud[mask]
        seg_masked = seg[mask]
        return H, W, valid_flat, cloud_masked, seg_masked

    def _get_sample_meta(self, index: int):
        scene_id, img_id, ann_id = self.samples[index]
        scene_camera = self._load_scene_camera(scene_id)
        scene_gt = self._load_scene_gt(scene_id)
        cam_info = scene_camera[str(int(img_id))]
        K = np.asarray(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
        factor_depth = get_factor_depth(
            self.camera, cam_info, mode=self.factor_depth_mode, fixed=self.fixed_factor_depth
        )
        return scene_id, img_id, ann_id, scene_gt, cam_info, K, factor_depth

    def _build_current_object_meta(self, scene_id: int, img_id: int, grasp_labels_npz=None):
        """Build cls_indexes and poses in the same order as economic label pointid."""
        scene_gt = self._load_scene_gt(scene_id)
        pose_map = scene_gt_pose_map(scene_gt, img_id)

        if grasp_labels_npz is not None and "obj_ids" in grasp_labels_npz.files:
            obj_ids = grasp_labels_npz["obj_ids"].astype(np.int32).tolist()
        else:
            # Fallback: current frame order. Safe only if economic labels used the same order.
            obj_ids = [int(o["obj_id"]) for o in scene_gt[str(int(img_id))]]

        poses = []
        missing = []
        for obj_id in obj_ids:
            if int(obj_id) not in pose_map:
                missing.append(int(obj_id))
                poses.append(np.zeros((3, 4), dtype=np.float32))
            else:
                poses.append(pose_map[int(obj_id)])

        if missing and self.strict_label_check:
            raise KeyError(f"Missing poses for obj_ids={missing} in scene={scene_id:06d}, img={img_id:06d}")

        poses_arr = np.stack(poses, axis=2).astype(np.float32) if poses else np.zeros((3, 4, 0), dtype=np.float32)
        obj_idxs = np.asarray(obj_ids, dtype=np.int32)
        return obj_idxs, poses_arr

    def _load_and_normalize_graspness(self, path: str) -> np.ndarray:
        graspness = np.load(path)
        graspness = np.asarray(graspness, dtype=np.float32)
        if graspness.ndim == 2 and graspness.shape[1] == 1:
            graspness = graspness[:, 0]
        elif graspness.ndim == 2 and graspness.shape[0] == 1:
            graspness = graspness[0, :]
        elif graspness.ndim != 1:
            graspness = graspness.reshape(graspness.shape[0], -1)[:, 0]
        return graspness.reshape(-1)

    # ---------------------------------------------------------------------
    # Public data paths
    # ---------------------------------------------------------------------
    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]).convert("RGB"), dtype=np.float32) / 255.0
        sensor_depth = _read_depth_png(self.depthpath[index])
        depth = sensor_depth.copy()
        seg = load_label(self.labelpath[index])
        gt_depth = _read_depth_png(self.gtdepthpath[index])

        scene_id, img_id, ann_id, scene_gt, cam_info, intrinsic, factor_depth = self._get_sample_meta(index)

        if self.use_gt_depth:
            depth = gt_depth

        cloud = depth_to_cloud(depth, intrinsic, factor_depth)
        mask = build_mask(
            depth=depth,
            label=seg,
            cloud=cloud,
            cam_info=cam_info,
            mode=self.mask_mode,
            bbox_margin_ratio=self.bbox_margin_ratio,
            workspace_outlier=self.workspace_outlier,
            workspace_pose_mode=self.workspace_pose_mode,
            workspace_depth_trunc=self.workspace_depth_trunc,
        )
        if self.fallback_to_depth_mask_on_empty and not np.any(mask):
            mask = depth > 0

        H, W, valid_flat, cloud_masked, seg_masked = self._mask_and_sample(depth, seg, cloud, mask)
        color_masked = color[mask]

        if return_raw_cloud:
            return cloud_masked, color_masked

        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]

        pix_flat = valid_flat[idxs]
        crop_box = self._crop_box_from_mask(mask)
        x0, y0, x1, y1 = crop_box
        color_crop = color[y0:y1, x0:x1].copy()
        img = self.img_transforms(color_crop)
        resized_idxs = self.get_resized_idxs_from_flat_crop(pix_flat, (H, W), crop_box, out_hw=self.resize_shape)

        sensor_depth_m_resized = self.build_resized_depth_m_with_crop(
            sensor_depth, factor_depth, crop_box, out_hw=self.resize_shape
        )

        gt_depth_m = self.build_fused_gt_depth_m(gt_depth, seg, index, factor_depth)
        K_resized = self.resize_intrinsics_with_crop(intrinsic, crop_box, out_hw=self.resize_shape)

        Hr, Wr = self.resize_shape
        gt_depth_crop = gt_depth_m[y0:y1, x0:x1]
        gt_depth_m_resized = cv2.resize(gt_depth_crop, (Wr, Hr), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        depth_prob_gt, depth_prob_w = self.build_depth_prob_gt(gt_depth_m_resized)

        ret_dict = {
            "point_clouds": cloud_sampled.astype(np.float32),
            "cloud_colors": color_sampled.astype(np.float32),
            "coordinates_for_voxel": (cloud_sampled.astype(np.float32) / self.voxel_size),
            "seg": seg_sampled.astype(np.float32),
            "img": img,
            "img_idxs": resized_idxs.astype(np.int64),
            "K": K_resized.astype(np.float32),
            "gt_depth_m": gt_depth_m_resized.astype(np.float32),
            "depth_prob_gt": depth_prob_gt.astype(np.float32),
            "depth_prob_weight": depth_prob_w.astype(np.float32),
            "sensor_depth_m": sensor_depth_m_resized.astype(np.float32),
            "scene_idx": np.int64(scene_id),
            "anno_idx": np.int64(ann_id),
            "img_id": np.int64(img_id),
            "dataset_idx": np.int64(index),
        }
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]).convert("RGB"), dtype=np.float32) / 255.0
        sensor_depth = _read_depth_png(self.depthpath[index])
        depth = sensor_depth.copy()
        seg = load_label(self.labelpath[index])
        gt_depth = _read_depth_png(self.gtdepthpath[index])

        scene = self.scenename[index]
        scene_id, img_id, ann_id, scene_gt, cam_info, intrinsic, factor_depth = self._get_sample_meta(index)

        graspness_path = self.graspnesspath[index]
        if self.use_gt_depth:
            depth = gt_depth
            graspness_path = self.gtgraspnesspath[index]
        graspness = self._load_and_normalize_graspness(graspness_path)

        cloud = depth_to_cloud(depth, intrinsic, factor_depth)

        mask = build_mask(
            depth=depth,
            label=seg,
            cloud=cloud,
            cam_info=cam_info,
            mode=self.mask_mode,
            bbox_margin_ratio=self.bbox_margin_ratio,
            workspace_outlier=self.workspace_outlier,
            workspace_pose_mode=self.workspace_pose_mode,
            workspace_depth_trunc=self.workspace_depth_trunc,
        )
        if self.fallback_to_depth_mask_on_empty and not np.any(mask):
            mask = depth > 0

        H, W = depth.shape
        Hr, Wr = self.resize_shape

        crop_box = self._crop_box_from_mask(mask)
        x0, y0, x1, y1 = crop_box

        color_crop = color[y0:y1, x0:x1].copy()
        img = self.img_transforms(color_crop)

        sensor_depth_m_resized = self.build_resized_depth_m_with_crop(
            sensor_depth, factor_depth, crop_box, out_hw=self.resize_shape
        )

        gt_depth_m = self.build_fused_gt_depth_m(gt_depth, seg, index, factor_depth)
        gt_depth_crop = gt_depth_m[y0:y1, x0:x1].copy()
        gt_depth_m_resized = cv2.resize(gt_depth_crop, (Wr, Hr), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        K_resized = self.resize_intrinsics_with_crop(intrinsic, crop_box, out_hw=self.resize_shape)
        depth_prob_gt, depth_prob_w = self.build_depth_prob_gt(gt_depth_m_resized)

        H, W, valid_flat, cloud_masked, seg_masked = self._mask_and_sample(depth, seg, cloud, mask)
        color_masked = color[mask]

        if graspness.shape[0] != valid_flat.shape[0]:
            raise ValueError(
                f"graspness length mismatch: graspness={graspness.shape}, masked_points={valid_flat.shape}. "
                f"scene={scene}, img_id={img_id:06d}, ann_id={ann_id}, mask_mode={self.mask_mode}, "
                f"graspness_path={graspness_path}"
            )

        resized_valid_all = self.get_resized_idxs_from_flat_crop(
            valid_flat, (H, W), crop_box, out_hw=self.resize_shape
        )
        resized_valid_all = np.asarray(resized_valid_all, dtype=np.int64).reshape(-1)

        obj_masked = seg_masked.copy()
        obj_masked[obj_masked > 1] = 1
        obj_masked = np.asarray(obj_masked, dtype=np.int64).reshape(-1)

        grasp_masked = graspness.astype(np.float32).reshape(-1)

        Nmasked = resized_valid_all.shape[0]
        if not (obj_masked.shape[0] == Nmasked and grasp_masked.shape[0] == Nmasked):
            raise ValueError(
                f"Masked arrays length mismatch: resized_valid_all={Nmasked}, "
                f"obj_masked={obj_masked.shape}, grasp_masked={grasp_masked.shape}"
            )

        obj_flat = np.zeros((Hr * Wr,), dtype=np.int64)
        gsum = np.zeros((Hr * Wr,), dtype=np.float32)
        gcnt = np.zeros((Hr * Wr,), dtype=np.float32)

        np.maximum.at(obj_flat, resized_valid_all, obj_masked)
        np.add.at(gsum, resized_valid_all, grasp_masked)
        np.add.at(gcnt, resized_valid_all, 1.0)

        grasp_flat = gsum / np.maximum(gcnt, 1.0)
        valid_flat_resized = (gcnt > 0).astype(np.float32)

        obj_map_448 = obj_flat.reshape(Hr, Wr)
        grasp_map_448 = grasp_flat.reshape(Hr, Wr)
        valid_map_448 = valid_flat_resized.reshape(Hr, Wr)

        objectness_label_tok = obj_map_448.reshape(-1).astype(np.int64)
        objectness_label_tok[valid_map_448.reshape(-1) < 1.0] = -1
        graspness_label_tok = grasp_map_448.reshape(-1).astype(np.float32)
        token_valid_mask = (valid_map_448.reshape(-1) >= 1.0)

        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = grasp_masked[idxs].astype(np.float32)

        objectness_label = seg_sampled.copy()
        segmentation_label = objectness_label.copy()
        objectness_label[objectness_label > 1] = 1

        pix_flat = valid_flat[idxs]
        resized_idxs = self.get_resized_idxs_from_flat_crop(
            pix_flat, (H, W), crop_box, out_hw=self.resize_shape
        )

        # -----------------------------
        # Economic grasp labels
        # -----------------------------
        grasp_labels = np.load(self.grasp_labels[scene])
        points = grasp_labels["points"]
        rotations = grasp_labels["rotations"].astype(np.int32)
        depth_l = grasp_labels["depth"].astype(np.int32)
        scores = grasp_labels["scores"].astype(np.float32) / 10.0
        widths = grasp_labels["widths"].astype(np.float32) / 1000.0
        topview = grasp_labels["topview"].astype(np.int32)
        view_graspness = grasp_labels["vgraspness"].astype(np.float32)
        pointid = grasp_labels["pointid"]
        collisions = grasp_labels["collisions"].astype(np.float32)

        obj_idxs, poses = self._build_current_object_meta(scene_id, img_id, grasp_labels)

        object_poses_list = []
        grasp_points_list = []
        grasp_rotations_list = []
        grasp_depth_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        view_graspness_list = []
        top_view_index_list = []
        grasp_collision_list = []

        for i, obj_idx in enumerate(obj_idxs):
            object_poses_list.append(poses[:, :, i])
            mask_i = (pointid == i)
            grasp_points_list.append(points[mask_i])
            grasp_rotations_list.append(rotations[mask_i])
            grasp_depth_list.append(depth_l[mask_i])
            grasp_scores_list.append(scores[mask_i])
            grasp_widths_list.append(widths[mask_i])
            view_graspness_list.append(view_graspness[mask_i])
            top_view_index_list.append(topview[mask_i])
            grasp_collision_list.append(collisions[mask_i])

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {
            # point-level
            "point_clouds": cloud_sampled.astype(np.float32),
            "cloud_colors": color_sampled.astype(np.float32),
            "coordinates_for_voxel": (cloud_sampled.astype(np.float32) / self.voxel_size),

            "img": img,
            "img_idxs": resized_idxs.astype(np.int64),

            "graspness_label": graspness_sampled.astype(np.float32),
            "objectness_label": objectness_label.astype(np.int64),
            "segmentation_label": segmentation_label.astype(np.int64),

            # token-level
            "objectness_label_tok": objectness_label_tok,
            "graspness_label_tok": graspness_label_tok,
            "token_valid_mask": token_valid_mask.astype(np.bool_),

            # economic grasp labels
            "object_poses_list": object_poses_list,
            "grasp_points_list": grasp_points_list,
            "grasp_rotations_list": grasp_rotations_list,
            "grasp_depth_list": grasp_depth_list,
            "grasp_widths_list": grasp_widths_list,
            "grasp_scores_list": grasp_scores_list,
            "view_graspness_list": view_graspness_list,
            "top_view_index_list": top_view_index_list,
            "grasp_collision_list": grasp_collision_list,

            # debug / bookkeeping
            "sampled_masked_idxs": idxs.astype(np.int64),
            "pix_flat": pix_flat.astype(np.int64),
            "crop_box": np.asarray(crop_box, dtype=np.int64),

            # camera / depth supervision
            "K": K_resized.astype(np.float32),
            "gt_depth_m": gt_depth_m_resized.astype(np.float32),
            "depth_prob_gt": depth_prob_gt.astype(np.float32),
            "depth_prob_weight": depth_prob_w.astype(np.float32),
            "sensor_depth_m": sensor_depth_m_resized.astype(np.float32),

            "scene_idx": np.int64(scene_id),
            "anno_idx": np.int64(ann_id),
            "img_id": np.int64(img_id),
            "dataset_idx": np.int64(index),
        }
        return ret_dict


# Backward-compatible alias if your training script expects this symbol.
GC6DMultiDataset = GraspClutter6DMultiDataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--camera", default="realsense-d435")
    parser.add_argument("--scene_ids", default=None)
    parser.add_argument("--img_id", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--load_label", action="store_true")
    parser.add_argument("--use_gt_depth", action="store_true")
    parser.add_argument("--mask_mode", default="workspace_depth")
    parser.add_argument("--graspness_mode", default="instance")
    parser.add_argument("--extend_angle", action="store_true")
    args = parser.parse_args()

    ds = GraspClutter6DMultiDataset(
        root=args.root,
        split=args.split,
        camera=args.camera,
        scene_ids=args.scene_ids,
        load_label=args.load_label,
        use_gt_depth=args.use_gt_depth,
        mask_mode=args.mask_mode,
        graspness_mode=args.graspness_mode,
        extend_angle=args.extend_angle,
        num_points=20000,
    )
    print(f"Dataset length: {len(ds)}")

    idx = int(args.index)
    if args.img_id is not None:
        for j, (_, img_id, _) in enumerate(ds.samples):
            if int(img_id) == int(args.img_id):
                idx = j
                break
    sample = ds[idx]
    print("Sample keys:", sorted(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        elif torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        elif isinstance(v, list):
            print(k, "list", len(v), [np.asarray(x).shape for x in v[:3]])
        else:
            print(k, type(v), v)
