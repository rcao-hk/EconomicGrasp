"""Shared, dependency-light protocol helpers for the point-depth mixed baseline.

Do not import training code here: train/inference/test must agree on the exact
per-scene 10% selection before potentially expensive model initialization.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict


def scene_stride_indices(dataset, fraction: float):
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"fraction must be in (0,1], received {fraction}")
    scenes = getattr(dataset, "scenename", None)
    frames = getattr(dataset, "frameid", None)
    if scenes is None or frames is None:
        raise TypeError("Dataset needs scenename and frameid")
    if len(dataset) != len(scenes) or len(scenes) != len(frames):
        raise RuntimeError("Dataset length and scene/frame metadata disagree")
    stride = 1 if fraction >= 1.0 else max(1, round(1.0 / float(fraction)))
    groups = defaultdict(list)
    for index, (scene, frame) in enumerate(zip(scenes, frames)):
        groups[str(scene)].append((int(frame), index))
    selected = []
    for scene in sorted(groups):
        rows = sorted(groups[scene])
        if len({frame for frame, _ in rows}) != len(rows):
            raise RuntimeError(f"Duplicate annotations in {scene}")
        selected += [index for frame, index in rows[::stride]]
    return selected


def scene_frame_signature(dataset, indices):
    pairs = [(str(dataset.scenename[i]), int(dataset.frameid[i])) for i in indices]
    blob = json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assert_paired_frames(original, trans, original_indices, trans_indices, name):
    a = [(str(original.scenename[i]), int(original.frameid[i])) for i in original_indices]
    b = [(str(trans.scenename[i]), int(trans.frameid[i])) for i in trans_indices]
    if a != b:
        raise RuntimeError(f"{name}: Original and GN-Trans selected frame IDs differ")
    return len(a)


def network_rgb_pose_inputs(batch, *, pose_depth_mode="global_film"):
    """Allowlist, not blacklist: sensor/rendered/GT depth never enters inference."""
    required = ("img", "K", "img_idxs")
    if pose_depth_mode == "global_film":
        required += ("camera_pose_vec",)
    elif pose_depth_mode == "ray_gravity_film":
        required += ("camera_gravity_vec",)
    elif pose_depth_mode != "none":
        raise ValueError(f"Unsupported pose depth mode: {pose_depth_mode}")
    missing = set(required).difference(batch)
    if missing:
        raise KeyError(f"RGB-only inference missing inputs: {sorted(missing)}")
    result = {key: batch[key] for key in required}
    for key in ("scene_idx", "anno_idx"):
        if key in batch:
            result[key] = batch[key]
    forbidden = ("point_clouds", "gt_depth_m", "sensor_depth_m", "depth_prob_gt")
    if set(forbidden).intersection(result):
        raise AssertionError("Depth or externally constructed point cloud leaked into network")
    return result
