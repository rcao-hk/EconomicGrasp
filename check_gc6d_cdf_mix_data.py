#!/usr/bin/env python3
"""Preflight a *filtered* GraspClutter6D CVA-CDF training source.

Never silently substitute the GC6D unfiltered train split. Files are checked
for every selected scene and image; --load_one exercises CVA collate semantics.
No model checkpoint or GPU is necessary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset.gc6d_dataset import (
    GraspClutter6DMultiDataset,
    read_split_scene_ids,
)
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
from dataset.graspnet_dataset import collate_fn

CDF_KEYS = {
    "points", "pointid", "vgraspness", "topview", "obj_ids",
    "extend_angle", "num_angle", "num_depth",
    "cdf_bins", "cdf_thresholds", "widths_depth_mm", "width_valids_depth",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gc6d_root", required=True)
    p.add_argument("--gc6d_scene_ids_file", required=True)
    p.add_argument("--gc6d_camera", default="realsense-d435")
    p.add_argument("--gc6d_label_folder", default="economic_grasp_label_300views_extend_angle_cdf_depth")
    p.add_argument("--expected_scenes", type=int, default=232)
    p.add_argument("--expected_frames", type=int, default=3016)
    p.add_argument("--max_ann", type=int, default=13)
    p.add_argument("--graspness_mode", default="scene")
    p.add_argument("--workspace_pose_mode", default="json")
    p.add_argument("--load_one", action="store_true")
    p.add_argument("--check_all_caches", action="store_true")
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def check_cache(path: Path, full: bool = False):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as labels:
        missing = sorted(CDF_KEYS - set(labels.files))
        if missing:
            raise KeyError(f"GC6D CDF cache missing {missing}: {path}")
        if int(labels["extend_angle"].reshape(-1)[0]) != 1:
            raise ValueError(f"Not extended-angle cache: {path}")
        if int(labels["num_angle"].reshape(-1)[0]) != 12:
            raise ValueError(f"Cache num_angle != 12: {path}")
        if int(labels["num_depth"].reshape(-1)[0]) != 4:
            raise ValueError(f"Cache num_depth != 4: {path}")
        t = labels["cdf_thresholds"]
        if t.shape != (6,) or not np.allclose(t, np.arange(0.2, 1.21, 0.2), atol=1e-5):
            raise ValueError(f"Unexpected CDF friction thresholds: {path}: {t}")
        ids = labels["obj_ids"]
        if ids.ndim != 1 or len(ids) == 0 or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"Invalid obj_ids cache mapping: {path}")
        pointid = labels["pointid"]
        if pointid.ndim != 1 or pointid.size == 0 or pointid.max() >= len(ids):
            raise ValueError(f"Invalid local pointid: {path}")
        if full:
            points = labels["points"]
            cdf = labels["cdf_bins"]
            width = labels["widths_depth_mm"]
            valid = labels["width_valids_depth"]
            topview = labels["topview"]
            expected = (points.shape[0], topview.shape[1], 12, 4)
            if any(a.shape != expected for a in (cdf, width, valid)):
                raise ValueError(f"CDF/width shape mismatch in {path}, expected {expected}")
            if cdf.dtype != np.uint8 or width.dtype != np.uint16 or valid.dtype != np.uint8:
                raise TypeError(f"GC6D cache dtype mismatch: {path}")
            if cdf.max() > 6 or not np.all(np.isfinite(points)):
                raise ValueError(f"Invalid GC6D cache values: {path}")
    return len(ids)


def main():
    a = parse_args()
    root = Path(a.gc6d_root).expanduser().resolve()
    split = Path(a.gc6d_scene_ids_file).expanduser().resolve()
    with split.open("r", encoding="utf-8") as fin:
        ids = json.load(fin)
    if not isinstance(ids, list):
        raise ValueError(f"Filtered scene-list must be a JSON array: {split}")
    ids = [int(i) for i in ids]
    if len(ids) != len(set(ids)) or len(ids) != a.expected_scenes:
        raise ValueError(f"Expected {a.expected_scenes} unique GC6D scenes; got {len(ids)}.")
    official_train = set(read_split_scene_ids(root, "train"))
    assert not (set(ids) - official_train), "Filtered list contains GC6D non-training scenes."

    base = GraspClutter6DMultiDataset(
        root=str(root), camera=a.gc6d_camera, split="train", scene_ids=sorted(ids),
        max_ann=a.max_ann, load_label=True, load_grasp_payload=False,
        use_gt_depth=False, use_fuse_depth=False, graspness_mode=a.graspness_mode,
        extend_angle=True, economic_label_root=str(root / a.gc6d_label_folder),
        scene_name_format="plain6", mask_mode="workspace_depth", remove_outlier=None,
        workspace_pose_mode=a.workspace_pose_mode, depth_strides=1,
    )
    if len(base) != a.expected_frames:
        raise ValueError(f"Expected {a.expected_frames} images, found {len(base)}.")
    seen = set()
    errors = []
    for i in range(len(base)):
        scene = str(base.scenename[i])
        seen.add(scene)
        for name, path in (
            ("rgb", base.colorpath[i]), ("sensor_depth", base.depthpath[i]),
            ("gt_depth", base.gtdepthpath[i]), ("segmentation", base.labelpath[i]),
            ("graspness", base.graspnesspath[i]),
        ):
            if not Path(path).is_file():
                errors.append(f"{scene} ann={base.frameid[i]} missing {name}: {path}")
    if errors:
        raise FileNotFoundError(
            f"GC6D has {len(errors)} missing frame files; first ten:\n"
            + "\n".join(errors[:10])
        )
    per_scene = {}
    for scene in sorted(seen):
        path = root / a.gc6d_label_folder / f"{scene}_labels.npz"
        per_scene[scene] = check_cache(path, full=a.check_all_caches)
    if len(seen) != a.expected_scenes:
        raise ValueError(f"Scene count mismatch: indexed {len(seen)} vs {a.expected_scenes}")

    if a.load_one:
        adapter = CVAExtendedLabelAdapter(
            base, dataset_root=str(root), label_folder=a.gc6d_label_folder,
            use_cdf=True, num_angle=12, num_depth=4,
        )
        item = adapter[0]
        required = {
            "img", "img_idxs", "K", "gt_depth_m", "camera_pose_vec",
            "camera_gravity_vec", "camera_tilt_deg", "object_poses_list",
            "grasp_cdf_bins_list", "grasp_widths_depth_list",
            "grasp_width_valids_depth_list", "cdf_thresholds",
        }
        if missing := (required - set(item)):
            raise KeyError(f"GC6D sample missing CVA-CDF keys: {sorted(missing)}")
        batch = collate_fn([item])
        for k in required:
            if k not in batch:
                raise KeyError(f"Collate missing {k}")
        if len(batch["object_poses_list"][0]) != len(batch["grasp_cdf_bins_list"][0]):
            raise ValueError("Pose/label object ordering mismatch")

    result = {
        "status": "PASS",
        "gc6d_root": str(root),
        "scene_list": str(split),
        "scenes": len(seen),
        "frames": len(base),
        "camera": base.camera,
        "label_folder": a.gc6d_label_folder,
        "sampled_obj_count_range": [min(per_scene.values()), max(per_scene.values())],
        "load_one_pass": bool(a.load_one),
        "all_caches_shape_checked": bool(a.check_all_caches),
    }
    print("[GC6D-CDF-PREFLIGHT] " + json.dumps(result, indent=2))
    if a.output_json:
        out = Path(a.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
