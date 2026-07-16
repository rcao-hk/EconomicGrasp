#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate EconomicGrasp / ecograsp_dpt scene-level grasp labels for GraspClutter6D.

This adapts the GraspNet-1Billion generate_economic.py to GraspClutter6D.

Input GC6D layout:
    ROOT/
      split_info/grasp_train_scene_ids.json
      split_info/grasp_test_scene_ids.json
      split_info/obj_ids_per_scene.json                 # optional
      grasp_label/obj_<obj_id:06d>_labels.npz
      collision_label/<scene_id:06d>.npz
      scenes/<scene_id:06d>/scene_gt.json
      scenes/<scene_id:06d>/scene_camera.json

Default output:
    ROOT/economic_grasp_label_300views/<scene_id:06d>_labels.npz

Output fields match the GraspNet/EconomicGrasp dataloader:
    points, rotations, depth, scores, widths, pointid, vgraspness,
    topview, collisions, extend_angle

If --extend_angle:
    rotations, depth, scores, widths, collisions, valids have shape [Ns, K, A].

Notes:
- GC6D obj_id is already the dataset object id, so this script reads:
      grasp_label/obj_<obj_id:06d>_labels.npz
  There is no GraspNet-style obj_idx - 1 conversion here.
- pointid is the local scene-object index. Your GC6D meta.mat generation must use
  the same object order as this file, otherwise dataloader pointid == i will mismatch.
- The safest default object order is --obj_order_source auto and
  --collision_align auto. The script checks collision tensor shapes against object
  grasp label shapes and stores obj_ids / collision_arr_indices in the npz for audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
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


# -----------------------------------------------------------------------------
# Basic IO
# -----------------------------------------------------------------------------
def canonical_camera(camera: str) -> str:
    camera = CAMERA_ALIASES.get(str(camera), str(camera))
    if camera not in CAMERA_OFFSETS:
        raise ValueError(f"Unknown camera={camera}. Expected one of {list(CAMERA_OFFSETS)}")
    return camera


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    return int(ann_id) * 4 + CAMERA_OFFSETS[canonical_camera(camera)]


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


def output_folder_name(keeping_views_numbers: int, extend_angle: bool, save_folder: Optional[str]) -> str:
    if save_folder is not None:
        return str(save_folder)
    if extend_angle:
        return f"economic_grasp_label_{int(keeping_views_numbers)}views_extend_angle"
    return f"economic_grasp_label_{int(keeping_views_numbers)}views"


def scene_label_name(scene_id: int, scene_name_format: str) -> str:
    if scene_name_format == "plain6":
        return f"{int(scene_id):06d}"
    if scene_name_format == "scene6":
        return f"scene_{int(scene_id):06d}"
    if scene_name_format == "scene4":
        return f"scene_{int(scene_id):04d}"
    raise ValueError(f"Unknown scene_name_format={scene_name_format}")


def label_path_for_scene(output_root: Path, scene_id: int, scene_name_format: str) -> Path:
    return output_root / f"{scene_label_name(scene_id, scene_name_format)}_labels.npz"


def load_collision_dump(dataset_root: Path, scene_id: int) -> List[np.ndarray]:
    path = dataset_root / "collision_label" / f"{int(scene_id):06d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = np.load(path)
    # Keep arr_0, arr_1, ... order. npz.files order is normally insertion order,
    # but sort numerically to be explicit.
    keys = sorted(labels.files, key=lambda x: int(x.split("_")[-1]) if x.startswith("arr_") else x)
    return [labels[k] for k in keys]


def load_object_label(dataset_root: Path, obj_id: int):
    path = dataset_root / "grasp_label" / f"obj_{int(obj_id):06d}_labels.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    points = data["points"].astype(np.float32)
    offsets = data["offsets"].astype(np.float32)
    scores = data["scores"].astype(np.float32)  # friction coefficients, smaller is better
    return points, offsets, scores


# -----------------------------------------------------------------------------
# Object order and collision alignment
# -----------------------------------------------------------------------------
def obj_ids_from_scene_gt_frame(scene_gt: dict, img_id: int) -> List[int]:
    return [int(o["obj_id"]) for o in scene_gt[str(int(img_id))]]


def obj_ids_from_first_scene_gt(scene_gt: dict) -> List[int]:
    keys = sorted([int(k) for k in scene_gt.keys()])
    if not keys:
        return []
    return obj_ids_from_scene_gt_frame(scene_gt, keys[0])


def obj_ids_from_camera_first(scene_gt: dict, camera: str, max_ann: int = 13) -> List[int]:
    camera = canonical_camera(camera)
    for ann_id in range(int(max_ann)):
        img_id = ann_id_to_img_id(ann_id, camera)
        if str(img_id) in scene_gt:
            return obj_ids_from_scene_gt_frame(scene_gt, img_id)
    return obj_ids_from_first_scene_gt(scene_gt)


def obj_ids_from_split_info(dataset_root: Path, scene_id: int) -> Optional[List[int]]:
    path = dataset_root / "split_info" / "obj_ids_per_scene.json"
    if not path.exists():
        return None
    data = load_json(path)
    key_candidates = [str(int(scene_id)), f"{int(scene_id):06d}"]
    for k in key_candidates:
        if k in data:
            return [int(x) for x in data[k]]
    return None


def object_score_shape(dataset_root: Path, obj_id: int, shape_cache: Dict[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    obj_id = int(obj_id)
    if obj_id not in shape_cache:
        _, _, scores = load_object_label(dataset_root, obj_id)
        shape_cache[obj_id] = tuple(scores.shape)
    return shape_cache[obj_id]


def count_index_shape_matches(dataset_root: Path, obj_ids: Sequence[int], collision_dump: Sequence[np.ndarray],
                              shape_cache: Dict[int, Tuple[int, ...]]) -> int:
    if len(obj_ids) != len(collision_dump):
        return -1
    n = 0
    for i, obj_id in enumerate(obj_ids):
        if tuple(collision_dump[i].shape) == object_score_shape(dataset_root, int(obj_id), shape_cache):
            n += 1
    return n


def choose_object_order(dataset_root: Path, scene_id: int, camera: str, scene_gt: dict,
                        collision_dump: Sequence[np.ndarray], source: str) -> Tuple[List[int], str, Dict[str, int]]:
    """Choose canonical object order for a scene.

    Returns:
        obj_ids, chosen_source, diagnostics
    """
    source = str(source).lower()
    shape_cache: Dict[int, Tuple[int, ...]] = {}

    candidates: List[Tuple[str, Optional[List[int]]]] = []
    candidates.append(("scene_gt_camera_first", obj_ids_from_camera_first(scene_gt, camera)))
    candidates.append(("scene_gt_first", obj_ids_from_first_scene_gt(scene_gt)))
    candidates.append(("obj_ids_per_scene", obj_ids_from_split_info(dataset_root, scene_id)))

    if source != "auto":
        mapping = dict(candidates)
        if source not in mapping:
            raise ValueError(f"Unknown obj_order_source={source}")
        obj_ids = mapping[source]
        if obj_ids is None:
            raise RuntimeError(f"obj_order_source={source} unavailable for scene {scene_id:06d}")
        diag = {
            "num_collision_arrays": int(len(collision_dump)),
            "num_objects": int(len(obj_ids)),
            "index_shape_matches": int(count_index_shape_matches(dataset_root, obj_ids, collision_dump, shape_cache)),
        }
        return list(obj_ids), source, diag

    best_name = None
    best_ids = None
    best_score = -999
    scores = {}
    for name, ids in candidates:
        if ids is None:
            scores[name] = -999
            continue
        score = count_index_shape_matches(dataset_root, ids, collision_dump, shape_cache)
        scores[name] = score
        if score > best_score:
            best_name = name
            best_ids = ids
            best_score = score

    if best_ids is None:
        raise RuntimeError(f"Could not obtain object list for scene {scene_id:06d}")

    diag = {
        "num_collision_arrays": int(len(collision_dump)),
        "num_objects": int(len(best_ids)),
        "index_shape_matches": int(best_score),
    }
    for name, score in scores.items():
        diag[f"shape_matches_{name}"] = int(score)
    return list(best_ids), str(best_name), diag


def align_collision_indices(dataset_root: Path, obj_ids: Sequence[int], collision_dump: Sequence[np.ndarray],
                            mode: str) -> Tuple[List[int], str, Dict[str, int]]:
    """Map each object index to a collision arr index.

    mode:
      index: require collision_dump[i] to match obj_ids[i].
      shape_match: greedily match by tensor shape.
      auto: use index if all match; otherwise shape_match.
    """
    mode = str(mode).lower()
    shape_cache: Dict[int, Tuple[int, ...]] = {}
    index_matches = count_index_shape_matches(dataset_root, obj_ids, collision_dump, shape_cache)

    if mode == "auto":
        mode = "index" if index_matches == len(obj_ids) else "shape_match"

    if mode == "index":
        if len(obj_ids) != len(collision_dump):
            raise ValueError(f"Object/collision length mismatch: obj={len(obj_ids)} collision={len(collision_dump)}")
        bad = []
        for i, obj_id in enumerate(obj_ids):
            if tuple(collision_dump[i].shape) != object_score_shape(dataset_root, obj_id, shape_cache):
                bad.append((i, int(obj_id), tuple(collision_dump[i].shape), object_score_shape(dataset_root, obj_id, shape_cache)))
        if bad:
            msg = "\n".join([f"local={i}, obj={o}, collision={cs}, object_scores={os}" for i, o, cs, os in bad[:20]])
            raise ValueError(f"collision/index shape mismatch. First bad entries:\n{msg}")
        diag = {
            "index_shape_matches": int(index_matches),
            "shape_match_ambiguous": 0,
        }
        return list(range(len(obj_ids))), "index", diag

    if mode == "shape_match":
        unused = set(range(len(collision_dump)))
        arr_indices = []
        ambiguous = 0
        for local_i, obj_id in enumerate(obj_ids):
            target_shape = object_score_shape(dataset_root, obj_id, shape_cache)
            matches = [j for j in sorted(unused) if tuple(collision_dump[j].shape) == target_shape]
            if not matches:
                raise ValueError(
                    f"No collision array shape matches obj_id={obj_id}, local={local_i}, "
                    f"target_shape={target_shape}; unused_shapes={[tuple(collision_dump[j].shape) for j in sorted(unused)[:20]]}"
                )
            if len(matches) > 1:
                ambiguous += 1
            j = matches[0]
            arr_indices.append(j)
            unused.remove(j)
        diag = {
            "index_shape_matches": int(index_matches),
            "shape_match_ambiguous": int(ambiguous),
        }
        return arr_indices, "shape_match", diag

    raise ValueError(f"Unknown collision_align={mode}")


# -----------------------------------------------------------------------------
# Economic label compression
# -----------------------------------------------------------------------------
def _normalize_view_graspness(grasp_view_graspness: torch.Tensor) -> torch.Tensor:
    vmin, _ = torch.min(grasp_view_graspness, dim=-1, keepdim=True)
    vmax, _ = torch.max(grasp_view_graspness, dim=-1, keepdim=True)
    return (grasp_view_graspness - vmin) / (vmax - vmin + 1e-5)


def _build_original_view_labels(scene_scores: torch.Tensor, scene_width: torch.Tensor, scene_collisions: torch.Tensor):
    """Original EconomicGrasp label compression."""
    score_max_depth, depth_idx = scene_scores.max(-1)  # [Ns,V,A]
    width_at_depth = scene_width.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)
    collision_at_depth = scene_collisions.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)

    score_max_angle, angle_idx = score_max_depth.max(-1)  # [Ns,V]
    depth = depth_idx.gather(-1, angle_idx.unsqueeze(-1)).squeeze(-1)
    widths = width_at_depth.gather(-1, angle_idx.unsqueeze(-1)).squeeze(-1)
    collisions = collision_at_depth.gather(-1, angle_idx.unsqueeze(-1)).squeeze(-1)
    return angle_idx, depth, score_max_angle, widths, collisions


def _build_extended_angle_labels(scene_scores: torch.Tensor, scene_width: torch.Tensor, scene_collisions: torch.Tensor):
    """Per-view-per-angle labels."""
    Ns, V, A, D = scene_scores.shape
    score_max_depth, depth_idx = scene_scores.max(-1)  # [Ns,V,A]
    widths = scene_width.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)
    collisions = scene_collisions.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)
    angle_ids = torch.arange(A, device=scene_scores.device, dtype=torch.long)
    rotations = angle_ids.view(1, 1, A).expand(Ns, V, A).contiguous()
    valids = score_max_depth > 0
    return rotations, depth_idx, score_max_depth, widths, collisions, valids


def _gather_top_views_2d(x: torch.Tensor, view_index: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, view_index)


def _gather_top_views_3d(x: torch.Tensor, view_index: torch.Tensor) -> torch.Tensor:
    A = x.shape[-1]
    idx = view_index.unsqueeze(-1).expand(-1, -1, A)
    return torch.gather(x, 1, idx)


def to_uint8_clipped(x: np.ndarray, scale: float = 1.0, name: str = "") -> np.ndarray:
    y = np.asarray(x, dtype=np.float32) * float(scale)
    if np.any(y < 0) or np.any(y > 255):
        # Keep behavior safe. Original script casts directly to uint8; clipping avoids wrap-around.
        y = np.clip(y, 0, 255)
    return y.astype(np.uint8)


@dataclass
class SceneResult:
    scene_id: int
    ok: bool
    skipped: bool
    message: str
    stats: Dict[str, object]


def generate_one_scene(scene_id: int, cfg) -> SceneResult:
    dataset_root = Path(cfg.dataset_root)
    camera = canonical_camera(cfg.camera)
    output_root = Path(cfg.output_root)
    out_path = label_path_for_scene(output_root, scene_id, cfg.scene_name_format)

    try:
        if cfg.skip_done and out_path.exists():
            return SceneResult(scene_id, True, True, f"skip_done: {out_path}", {"scene_id": scene_id})

        scene_dir = dataset_root / "scenes" / f"{int(scene_id):06d}"
        scene_gt_path = scene_dir / "scene_gt.json"
        if not scene_gt_path.exists():
            raise FileNotFoundError(scene_gt_path)
        scene_gt = load_json(scene_gt_path)

        collision_dump = load_collision_dump(dataset_root, scene_id)
        obj_ids, order_source, order_diag = choose_object_order(
            dataset_root=dataset_root,
            scene_id=scene_id,
            camera=camera,
            scene_gt=scene_gt,
            collision_dump=collision_dump,
            source=cfg.obj_order_source,
        )
        collision_arr_indices, collision_align_used, collision_diag = align_collision_indices(
            dataset_root=dataset_root,
            obj_ids=obj_ids,
            collision_dump=collision_dump,
            mode=cfg.collision_align,
        )

        keeping_views_numbers = int(cfg.keeping_views_numbers)
        if keeping_views_numbers <= 0 or keeping_views_numbers > NUM_VIEWS:
            raise ValueError(f"keeping_views_numbers must be in [1,{NUM_VIEWS}], got {keeping_views_numbers}")

        scene_points = []
        scene_pointid = []
        scene_scores = []
        scene_width = []
        scene_collisions = []

        ori_number = 0
        filtered_number = 0
        per_object_rows = []

        # Load and pre-filter object labels object-by-object to reduce peak memory.
        for local_i, obj_id in enumerate(obj_ids):
            points_np, offsets_np, scores_np = load_object_label(dataset_root, obj_id)
            collision_np = collision_dump[collision_arr_indices[local_i]]

            if tuple(collision_np.shape) != tuple(scores_np.shape):
                raise ValueError(
                    f"After alignment, collision/friction shape mismatch: scene={scene_id:06d}, "
                    f"local={local_i}, obj_id={obj_id}, collision={collision_np.shape}, scores={scores_np.shape}"
                )

            points = torch.from_numpy(points_np).to(cfg.device)
            width = torch.from_numpy(offsets_np[..., 2]).to(cfg.device)
            scores = torch.from_numpy(scores_np).to(cfg.device)
            collision = torch.from_numpy(collision_np.astype(np.bool_)).to(cfg.device)

            # Collision grasps are invalid for score learning; keep collision tensor separately.
            scores = scores.clone()
            scores[collision] = 0
            scores[scores < 0] = 0

            num_points = int(points.shape[0])
            ori_number += num_points

            grasp_mask = (scores <= float(cfg.point_filter_fric_thresh)) & (scores > 0)
            graspness = grasp_mask.float().view(num_points, -1).sum(dim=-1) / float(NUM_VIEWS * NUM_ANGLES * NUM_DEPTHS)
            filter_mask = graspness > 0

            keep_n = int(filter_mask.sum().item())
            filtered_number += keep_n

            if keep_n > 0:
                scene_points.append(points[filter_mask].cpu())
                scene_pointid.append(torch.ones(keep_n, dtype=torch.long) * int(local_i))
                scene_scores.append(scores[filter_mask].cpu())
                scene_width.append(width[filter_mask].cpu())
                scene_collisions.append(collision[filter_mask].cpu())

            per_object_rows.append({
                "scene_id": int(scene_id),
                "local_i": int(local_i),
                "obj_id": int(obj_id),
                "collision_arr_i": int(collision_arr_indices[local_i]),
                "num_points": int(num_points),
                "num_kept": int(keep_n),
                "keep_ratio": float(keep_n / max(num_points, 1)),
                "shape": "x".join(map(str, tuple(scores_np.shape))),
            })

            del points, width, scores, collision

        if len(scene_points) == 0:
            # Save an empty but structurally valid label file.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_empty_label(out_path, cfg.extend_angle)
            stats = {
                "scene_id": int(scene_id),
                "ok": 1,
                "empty": 1,
                "num_objects": int(len(obj_ids)),
                "num_points_raw": int(ori_number),
                "num_points_kept": 0,
                "order_source": order_source,
                "collision_align": collision_align_used,
            }
            return SceneResult(scene_id, True, False, "empty label saved", stats)

        scene_points_t = torch.cat(scene_points, dim=0).to(cfg.device)
        scene_pointid_t = torch.cat(scene_pointid, dim=0).to(cfg.device)
        scene_scores_t = torch.cat(scene_scores, dim=0).to(cfg.device)
        scene_width_t = torch.cat(scene_width, dim=0).to(cfg.device)
        scene_collisions_t = torch.cat(scene_collisions, dim=0).to(cfg.device)

        Ns, V, A, D = scene_scores_t.shape

        # Compute view graspness using friction-coefficient threshold before score inversion.
        grasp_view_valid_mask = (scene_scores_t <= float(cfg.view_fric_thresh)) & (scene_scores_t > 0)
        grasp_view_graspness = grasp_view_valid_mask.float().sum(dim=-1).sum(dim=-1) / float(A * D)  # [Ns,V]
        grasp_view_graspness = _normalize_view_graspness(grasp_view_graspness)

        # Normalize score: smaller friction is better; EconomicGrasp uses larger-is-better score.
        label_mask = (scene_scores_t > 0) & (scene_width_t <= float(cfg.max_grasp_width))
        scene_scores_t[~label_mask] = 0
        po_mask = scene_scores_t > 0
        scene_scores_t[po_mask] = 1.1 - scene_scores_t[po_mask]

        if cfg.extend_angle:
            scene_rotations, scene_depth, scene_scores_out, scene_width_out, scene_collision_selected, scene_valids = \
                _build_extended_angle_labels(scene_scores_t, scene_width_t, scene_collisions_t)
        else:
            scene_rotations, scene_depth, scene_scores_out, scene_width_out, scene_collision_selected = \
                _build_original_view_labels(scene_scores_t, scene_width_t, scene_collisions_t)
            scene_valids = None

        _, index = torch.topk(grasp_view_graspness, k=keeping_views_numbers)
        scene_top_view_index = index

        if cfg.extend_angle:
            scene_rotations = _gather_top_views_3d(scene_rotations, index)
            scene_depth = _gather_top_views_3d(scene_depth, index)
            scene_scores_out = _gather_top_views_3d(scene_scores_out, index)
            scene_width_out = _gather_top_views_3d(scene_width_out, index)
            scene_collision_selected = _gather_top_views_3d(scene_collision_selected.to(torch.uint8), index).bool()
            scene_valids = _gather_top_views_3d(scene_valids.to(torch.uint8), index).bool()
        else:
            scene_rotations = _gather_top_views_2d(scene_rotations, index)
            scene_depth = _gather_top_views_2d(scene_depth, index)
            scene_scores_out = _gather_top_views_2d(scene_scores_out, index)
            scene_width_out = _gather_top_views_2d(scene_width_out, index)
            scene_collision_selected = _gather_top_views_2d(scene_collision_selected.to(torch.uint8), index).bool()

        # Save. Keep dtypes compatible with current dataloader.
        scene_points_np = scene_points_t.cpu().numpy().astype(np.float32)
        grasp_rotations_np = scene_rotations.cpu().numpy().astype(np.uint8)
        grasp_depth_np = scene_depth.cpu().numpy().astype(np.uint8)
        grasp_scores_np = to_uint8_clipped(scene_scores_out.cpu().numpy(), scale=10.0, name="scores")
        grasp_widths_np = to_uint8_clipped(scene_width_out.cpu().numpy(), scale=1000.0, name="widths")
        scene_pointid_np = scene_pointid_t.cpu().numpy().astype(np.uint8)
        grasp_view_graspness_np = grasp_view_graspness.cpu().numpy().astype(np.float32)
        grasp_top_view_index_np = scene_top_view_index.cpu().numpy().astype(np.uint16)
        grasp_collisions_np = scene_collision_selected.cpu().numpy().astype(np.uint8)

        save_dict = dict(
            points=scene_points_np,
            rotations=grasp_rotations_np,
            depth=grasp_depth_np,
            scores=grasp_scores_np,
            widths=grasp_widths_np,
            pointid=scene_pointid_np,
            vgraspness=grasp_view_graspness_np,
            topview=grasp_top_view_index_np,
            collisions=grasp_collisions_np,
            extend_angle=np.array([1 if cfg.extend_angle else 0], dtype=np.uint8),
            # Extra audit fields. Existing dataloader will ignore these.
            obj_ids=np.asarray(obj_ids, dtype=np.int32),
            collision_arr_indices=np.asarray(collision_arr_indices, dtype=np.int32),
            scene_id=np.asarray([int(scene_id)], dtype=np.int32),
        )
        if cfg.extend_angle:
            save_dict["valids"] = scene_valids.cpu().numpy().astype(np.uint8)
            save_dict["num_angle"] = np.array([A], dtype=np.uint8)
            save_dict["num_depth"] = np.array([D], dtype=np.uint8)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **save_dict)

        if cfg.save_alias_scene6:
            alias = output_root / f"scene_{int(scene_id):06d}_labels.npz"
            if alias != out_path:
                shutil.copyfile(out_path, alias)
        if cfg.save_alias_scene4:
            alias = output_root / f"scene_{int(scene_id):04d}_labels.npz"
            if alias != out_path:
                shutil.copyfile(out_path, alias)

        if cfg.write_object_stats:
            obj_stats_path = output_root / "stats" / f"{int(scene_id):06d}_objects.csv"
            write_csv(obj_stats_path, per_object_rows)

        stats = {
            "scene_id": int(scene_id),
            "ok": 1,
            "empty": 0,
            "num_objects": int(len(obj_ids)),
            "num_collision_arrays": int(len(collision_dump)),
            "num_points_raw": int(ori_number),
            "num_points_kept": int(filtered_number),
            "keep_ratio": float(filtered_number / max(ori_number, 1)),
            "Ns": int(Ns),
            "K": int(keeping_views_numbers),
            "extend_angle": int(cfg.extend_angle),
            "order_source": order_source,
            "collision_align": collision_align_used,
        }
        stats.update({f"order_{k}": v for k, v in order_diag.items()})
        stats.update({f"collision_{k}": v for k, v in collision_diag.items()})
        return SceneResult(scene_id, True, False, str(out_path), stats)

    except Exception:
        return SceneResult(scene_id, False, False, traceback.format_exc(), {
            "scene_id": int(scene_id),
            "ok": 0,
        })


def save_empty_label(out_path: Path, extend_angle: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if extend_angle:
        save_dict = dict(
            points=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, NUM_VIEWS, NUM_ANGLES), dtype=np.uint8),
            depth=np.zeros((0, NUM_VIEWS, NUM_ANGLES), dtype=np.uint8),
            scores=np.zeros((0, NUM_VIEWS, NUM_ANGLES), dtype=np.uint8),
            widths=np.zeros((0, NUM_VIEWS, NUM_ANGLES), dtype=np.uint8),
            pointid=np.zeros((0,), dtype=np.uint8),
            vgraspness=np.zeros((0, NUM_VIEWS), dtype=np.float32),
            topview=np.zeros((0, NUM_VIEWS), dtype=np.uint16),
            collisions=np.zeros((0, NUM_VIEWS, NUM_ANGLES), dtype=np.uint8),
            valids=np.zeros((0, NUM_VIEWS, NUM_ANGLES), dtype=np.uint8),
            extend_angle=np.array([1], dtype=np.uint8),
            num_angle=np.array([NUM_ANGLES], dtype=np.uint8),
            num_depth=np.array([NUM_DEPTHS], dtype=np.uint8),
        )
    else:
        save_dict = dict(
            points=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, NUM_VIEWS), dtype=np.uint8),
            depth=np.zeros((0, NUM_VIEWS), dtype=np.uint8),
            scores=np.zeros((0, NUM_VIEWS), dtype=np.uint8),
            widths=np.zeros((0, NUM_VIEWS), dtype=np.uint8),
            pointid=np.zeros((0,), dtype=np.uint8),
            vgraspness=np.zeros((0, NUM_VIEWS), dtype=np.float32),
            topview=np.zeros((0, NUM_VIEWS), dtype=np.uint16),
            collisions=np.zeros((0, NUM_VIEWS), dtype=np.uint8),
            extend_angle=np.array([0], dtype=np.uint8),
        )
    np.savez(out_path, **save_dict)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _worker(args):
    scene_id, cfg = args
    return generate_one_scene(scene_id, cfg)


def run(scene_ids: Sequence[int], cfg) -> List[SceneResult]:
    scene_ids = list(scene_ids)
    if cfg.proc <= 1:
        return [generate_one_scene(sid, cfg) for sid in tqdm(scene_ids, desc="Scenes")]

    ctx = mp.get_context(cfg.mp_start_method)
    jobs = [(sid, cfg) for sid in scene_ids]
    out: List[SceneResult] = []
    with ctx.Pool(processes=int(cfg.proc), maxtasksperchild=1) as pool:
        for ret in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc="Scenes"):
            out.append(ret)
            status = "OK" if ret.ok else "FAIL"
            skip = " skipped" if ret.skipped else ""
            msg0 = str(ret.message).splitlines()[0] if ret.message else ""
            print(f"[scene {ret.scene_id:06d}] {status}{skip} {msg0}")
            if (not ret.ok) and cfg.print_traceback:
                print(ret.message)
    return out


def summarize_and_write(results: Sequence[SceneResult], output_root: Path, write_stats: bool) -> None:
    ok = [r for r in results if r.ok and not r.skipped]
    skipped = [r for r in results if r.skipped]
    fail = [r for r in results if not r.ok]
    print("\n[Summary]")
    print(f"  total:   {len(results)}")
    print(f"  ok:      {len(ok)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed:  {len(fail)}")
    if fail:
        print(f"  failed scenes: {[r.scene_id for r in fail[:50]]}{' ...' if len(fail) > 50 else ''}")

    if write_stats:
        rows = [r.stats for r in results]
        write_csv(output_root / "stats" / "scene_summary.csv", rows)
        if fail:
            err_dir = output_root / "stats" / "errors"
            err_dir.mkdir(parents=True, exist_ok=True)
            for r in fail:
                with open(err_dir / f"{r.scene_id:06d}.txt", "w", encoding="utf-8") as f:
                    f.write(r.message)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate GC6D EconomicGrasp labels.")
    p.add_argument("--dataset_root", default=os.environ.get("GC6D_ROOT"))
    p.add_argument("--split", default="train", help="train/test/all or explicit scene ids like '5,7,10-20'.")
    p.add_argument("--scene_ids", default=None, help="Override --split.")
    p.add_argument("--camera", default="realsense-d435", choices=list(CAMERA_OFFSETS.keys()) + list(CAMERA_ALIASES.keys()))

    p.add_argument("--keeping_views_numbers", type=int, default=300)
    p.add_argument("--extend_angle", action="store_true")
    p.add_argument("--save_folder", default=None)
    p.add_argument("--output_root", default=None,
                   help="Default: dataset_root/<economic_grasp_label_{K}views[_extend_angle]>")
    p.add_argument("--scene_name_format", default="plain6", choices=["plain6", "scene6", "scene4"],
                   help="Filename stem before _labels.npz.")
    p.add_argument("--save_alias_scene6", action="store_true",
                   help="Also save scene_<scene_id:06d>_labels.npz.")
    p.add_argument("--save_alias_scene4", action="store_true",
                   help="Also save scene_<scene_id:04d>_labels.npz.")

    p.add_argument("--obj_order_source", default="auto",
                   choices=["auto", "scene_gt_camera_first", "scene_gt_first", "obj_ids_per_scene"])
    p.add_argument("--collision_align", default="auto", choices=["auto", "index", "shape_match"])

    p.add_argument("--point_filter_fric_thresh", type=float, default=0.4,
                   help="Keep object grasp point if any friction score in (0, threshold].")
    p.add_argument("--view_fric_thresh", type=float, default=0.6,
                   help="Threshold for view graspness before score inversion.")
    p.add_argument("--max_grasp_width", type=float, default=0.1,
                   help="EconomicGrasp width mask before score inversion.")

    p.add_argument("--device", default="cpu", help="cpu or cuda:0. CPU is safer for large scenes.")
    p.add_argument("--torch_num_threads", type=int, default=1)
    p.add_argument("--proc", type=int, default=1)
    p.add_argument("--mp_start_method", default="spawn", choices=["spawn", "forkserver", "fork"])
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--write_stats", action="store_true")
    p.add_argument("--write_object_stats", action="store_true")
    p.add_argument("--print_traceback", action="store_true")
    return p


def main() -> None:
    cfg = build_argparser().parse_args()
    if cfg.dataset_root is None:
        raise ValueError("--dataset_root is required or set $GC6D_ROOT")
    cfg.dataset_root = str(Path(cfg.dataset_root).expanduser())

    torch.set_num_threads(max(1, int(cfg.torch_num_threads)))

    if cfg.scene_ids:
        scene_ids = parse_int_list(cfg.scene_ids)
    else:
        scene_ids = read_split_scene_ids(Path(cfg.dataset_root), cfg.split)

    save_folder = output_folder_name(cfg.keeping_views_numbers, cfg.extend_angle, cfg.save_folder)
    if cfg.output_root is None:
        cfg.output_root = str(Path(cfg.dataset_root) / save_folder)
    cfg.output_root = str(Path(cfg.output_root).expanduser())
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[Main] dataset_root={cfg.dataset_root}")
    print(f"[Main] split={cfg.split}, scenes={len(scene_ids)}")
    print(f"[Main] camera={canonical_camera(cfg.camera)}")
    print(f"[Main] output_root={cfg.output_root}")
    print(f"[Main] K={cfg.keeping_views_numbers}, extend_angle={cfg.extend_angle}")
    print(f"[Main] obj_order_source={cfg.obj_order_source}, collision_align={cfg.collision_align}")
    print(f"[Main] device={cfg.device}, proc={cfg.proc}, torch_num_threads={cfg.torch_num_threads}")

    results = run(scene_ids, cfg)
    summarize_and_write(results, output_root, cfg.write_stats)


if __name__ == "__main__":
    main()
