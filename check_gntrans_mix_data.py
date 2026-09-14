#!/usr/bin/env python3
"""Validate paired GraspNet/GN-Trans sampling before mixed CVA-CDF training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np

from dataset.graspnet_dataset import GraspNetMultiDataset, GraspNetTransDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--gntrans_rgb_root", required=True)
    p.add_argument("--camera", default="realsense")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument(
        "--cdf_label_folder",
        default="economic_grasp_label_300views_extend_angle_cdf_depth",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--check_all_selected", action="store_true")
    p.add_argument("--load_one", action="store_true")
    return p.parse_args()


def scene_stride_indices(dataset, fraction: float) -> List[int]:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0,1].")
    if fraction >= 1.0:
        return list(range(len(dataset)))
    stride = max(1, int(round(1.0 / float(fraction))))
    groups = {}
    for i, scene in enumerate(dataset.scenename):
        groups.setdefault(str(scene), []).append(i)
    out = []
    for scene in sorted(groups):
        rows = sorted(groups[scene], key=lambda i: int(dataset.frameid[i]))
        out.extend(rows[::stride])
    return out


def assert_pairs(a, b, ia, ib):
    if len(ia) != len(ib):
        raise RuntimeError(f"sample count mismatch {len(ia)} != {len(ib)}")
    for xa, xb in zip(ia, ib):
        pa = (str(a.scenename[xa]), int(a.frameid[xa]))
        pb = (str(b.scenename[xb]), int(b.frameid[xb]))
        if pa != pb:
            raise RuntimeError(f"paired sample mismatch: {pa} != {pb}")


def selected_paths(original, trans, indices):
    for i in indices:
        scene = str(original.scenename[i])
        yield {
            "original_rgb": original.colorpath[i],
            "original_sensor_depth": original.depthpath[i],
            "original_gt_depth": original.gtdepthpath[i],
            "gntrans_rgb": trans.colorpath[i],
            "gntrans_gt_depth": trans.gtdepthpath[i],
            "label": original.labelpath[i],
            "meta": original.metapath[i],
            "virtual_graspness": trans.graspnesspath[i],
            "cdf_cache": os.path.join(
                original.root,
                ARGS.cdf_label_folder,
                f"{scene}_labels.npz",
            ),
        }


def check_paths(paths, all_selected: bool):
    rows = list(paths)
    inspect = rows if all_selected else ([rows[0], rows[-1]] if rows else [])
    missing = []
    for row in inspect:
        for key, path in row.items():
            if not os.path.isfile(path):
                missing.append((key, path))
    if missing:
        text = "\n".join(f"{k}: {p}" for k, p in missing[:30])
        raise FileNotFoundError(f"Missing {len(missing)} required files:\n{text}")
    return len(inspect)


def build(split: str):
    common = dict(
        camera=ARGS.camera,
        split=split,
        num_points=20000,
        remove_outlier=True,
        augment=False,
        load_label=True,
        min_depth=0.2,
        max_depth=1.0,
        bin_num=256,
        depth_strides=1,
    )
    original = GraspNetMultiDataset(
        ARGS.dataset_root,
        voxel_size=0.005,
        use_gt_depth=False,
        use_fuse_depth=False,
        graspness_mode="scene",
        extend_angle=True,
        load_grasp_payload=False,
        **common,
    )
    trans = GraspNetTransDataset(
        ARGS.dataset_root,
        ARGS.gntrans_rgb_root,
        voxel_size=0.005,
        use_gt_depth=True,
        **common,
    )
    trans.extend_angle = True
    trans.load_grasp_payload = False
    return original, trans


def main():
    report = {
        "fraction_each_domain": float(ARGS.fraction),
        "splits": {},
    }
    for split in ("train", "test_seen", "test_similar", "test_novel"):
        original, trans = build(split)
        oi = scene_stride_indices(original, ARGS.fraction)
        ti = scene_stride_indices(trans, ARGS.fraction)
        assert_pairs(original, trans, oi, ti)
        checked = check_paths(
            selected_paths(original, trans, oi),
            all_selected=bool(ARGS.check_all_selected),
        )
        row = {
            "original_full": len(original),
            "gntrans_full": len(trans),
            "original_selected": len(oi),
            "gntrans_selected": len(ti),
            "mixed_selected": len(oi) + len(ti),
            "checked_sample_path_records": checked,
            "first_frames": [
                [str(original.scenename[i]), int(original.frameid[i])]
                for i in oi[:5]
            ],
        }
        report["splits"][split] = row
        print(f"[{split}] {json.dumps(row, sort_keys=True)}")

        if ARGS.load_one and split == "train":
            a = original[oi[0]]
            b = trans[ti[0]]
            for name, sample in (("original", a), ("gntrans", b)):
                required = (
                    "img", "K", "gt_depth_m", "object_poses_list",
                    "objectness_label_tok", "graspness_label_tok",
                )
                missing = [k for k in required if k not in sample]
                if missing:
                    raise KeyError(f"{name} sample missing {missing}")
                print(
                    f"[{name}] img={tuple(sample['img'].shape)} "
                    f"gt_depth={np.asarray(sample['gt_depth_m']).shape} "
                    f"objects={len(sample['object_poses_list'])}"
                )

    text = json.dumps(report, indent=2, sort_keys=True)
    if ARGS.output:
        Path(ARGS.output).write_text(text, encoding="utf-8")
    print(text)


ARGS = parse_args()
if __name__ == "__main__":
    main()
