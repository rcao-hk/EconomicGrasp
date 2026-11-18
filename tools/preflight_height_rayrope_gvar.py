#!/usr/bin/env python3
"""Read-only GraspNet/GN-Trans camera-plane and mix50 sampling audit.

No neural inference, no depth target is used to estimate the support plane.
Prints explicit physical-unit assumptions and refuses missing per-scene metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from height_analytic_geometry import plane_from_camera_to_table


def audit(root, gntrans, *, camera="realsense", train_fraction=0.5, min_dot=0.1,
          test_all=True, check_rgb_files=False):
    root, gntrans = Path(root), Path(gntrans)
    if train_fraction != 0.5:
        raise ValueError("This is the registered exact 50% PER-DOMAIN protocol.")
    results = {"protocol": "mix50-per-domain-128-of-256; GraspNet RealSense",
               "plane_definition": "T_table_from_cam=cam0_wrt_table@camera_poses[anno]",
               "plane_is_privileged_calibration": True,
               "geometry_gt_not_read": True,
               "train_samples_per_domain": 12800,
               "train_total_samples": 25600,
               "full_inference_frames_per_split": 7680,
               "train_scenes": 100,
               "full_eval_scenes": 90,
               "height_ray_min_downward_dot": min_dot}
    stats = {"train": [], "test": []}
    h = hashlib.sha256()
    for sid in range(190 if test_all else 100):
        scene = f"scene_{sid:04d}"
        d = root / "scenes" / scene / camera
        poses = np.load(d / "camera_poses.npy", allow_pickle=False)
        align = np.load(d / "cam0_wrt_table.npy", allow_pickle=False)
        if poses.shape != (256, 4, 4):
            raise ValueError(f"{scene}: 256 camera extrinsics required, got {poses.shape}")
        # The full test contains all 256 views; mixed train uses every second view.
        annos = range(0, 256, 2) if sid < 100 else range(256)
        for anno in annos:
            T = align @ poses[anno]
            plane = plane_from_camera_to_table(T)
            n, b = plane[:3], float(plane[3])
            # Principal-ray check is a necessary but not sufficient coverage check.
            if n[2] >= -min_dot:
                raise ValueError(
                    f"{scene} frame={anno}: optical ray nearly parallel/upward "
                    f"n_z={n[2]:.4f}; height decoding not supported for this camera pose."
                )
            h.update(plane.tobytes())
            stats["train" if sid < 100 else "test"].append((b, float(n[2])))
        if sid < 100 and check_rgb_files:
            for anno in range(0, 256, 2):
                p = gntrans / "scenes" / f"{sid:05d}" / f"{anno:04d}_color.png"
                if not p.is_file():
                    raise FileNotFoundError(p)
    for tag, values in stats.items():
        if not values:
            continue
        x = np.asarray(values)
        results[tag] = {
            "num_frames": int(len(x)),
            "camera_height_m_min_median_max": [
                float(x[:, 0].min()), float(np.median(x[:, 0])), float(x[:, 0].max())
            ],
            "optical_ray_n_dot_min_median_max": [
                float(x[:, 1].min()), float(np.median(x[:, 1])), float(x[:, 1].max())
            ],
        }
    results["metadata_signature_sha256"] = h.hexdigest()
    if len(stats["train"]) != 12800:
        raise AssertionError("Expected exactly 12800 original/GNTrans frame pairs.")
    if test_all and len(stats["test"]) != 23040:
        raise AssertionError("Expected 90*256=23040 full GraspNet test frames.")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--gntrans_rgb_root", required=True)
    p.add_argument("--camera", default="realsense")
    p.add_argument("--train_fraction", type=float, default=.5)
    p.add_argument("--height_min_downward_dot", type=float, default=.1)
    p.add_argument("--train_only", action="store_true")
    p.add_argument("--check_rgb_files", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    path = Path(args.output)
    if path.exists(): raise FileExistsError(f"Refusing to replace existing audit: {path}")
    r = audit(args.dataset_root, args.gntrans_rgb_root, camera=args.camera,
              train_fraction=args.train_fraction,
              min_dot=args.height_min_downward_dot,
              test_all=not args.train_only, check_rgb_files=args.check_rgb_files)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2, sort_keys=True)+"\n")
    print(json.dumps(r, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
