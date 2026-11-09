#!/usr/bin/env python3
"""Validate the P2 scratch-MLP gripper-field cache.

Important: consume this script's CLI before importing any EconomicGrasp modules.
Some project modules may import ``utils.arguments``, whose global parser calls
``parse_args()`` at import time.  If P2-specific flags are still present in
``sys.argv`` at that point, the global parser reports every P2 flag as
"unrecognized arguments" before this validator gets a chance to parse them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def _consume_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--expected_pose_depth_mode", default="global_film")
    parser.add_argument("--expected_use_fuse_depth", type=int, choices=(0, 1), default=1)
    parser.add_argument("--residual_tau_m", type=float, default=0.02)
    parser.add_argument("--surface_tau_m", type=float, default=0.01)
    parser.add_argument("--max_grasp_width_m", type=float, default=0.10)
    parser.add_argument("--min_metric_depth_m", type=float, default=0.20)
    parser.add_argument("--max_metric_depth_m", type=float, default=1.00)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--strict", type=int, choices=(0, 1), default=1)
    parser.add_argument("--require_all_scenes", type=int, choices=(0, 1), default=0)
    parser.add_argument("--min_frames_per_scene", type=int, default=26)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    # Prevent a transitive import of utils.arguments from re-parsing P2 flags.
    sys.argv[:] = [sys.argv[0]]
    return args


ARGS = _consume_args()

# Project imports intentionally come *after* P2 CLI consumption.
from exact_action_cdf_common import atomic_save_json
from models.p2_gripper_cdf_field import P2FieldConfig
from p2_gripper_field_cache import save_inventory, scan_p2_cache
from p2_gripper_field_common import validate_base_checkpoint


def main() -> None:
    args = ARGS
    _, _, base_sha = validate_base_checkpoint(
        args.base_checkpoint,
        expected_pose_depth_mode=args.expected_pose_depth_mode,
        expected_use_fuse_depth=bool(args.expected_use_fuse_depth),
    )
    field_config = P2FieldConfig(
        max_grasp_width_m=float(args.max_grasp_width_m),
        min_metric_depth_m=float(args.min_metric_depth_m),
        max_metric_depth_m=float(args.max_metric_depth_m),
        residual_tau_m=float(args.residual_tau_m),
        surface_tau_m=float(args.surface_tau_m),
    )
    metadata, inventory, failures = scan_p2_cache(
        args.cache_dir,
        expected_source_base_checkpoint_sha256=base_sha,
        expected_field_config_sha256=field_config.sha256(),
        max_files=int(args.max_files),
        strict=bool(args.strict),
        check_values=True,
    )
    frame_counts = Counter(item.scene_id for item in metadata)
    payload = inventory.to_dict()
    payload["num_failures"] = len(failures)
    payload["failures"] = failures
    payload["frames_per_scene"] = {
        str(scene): int(count) for scene, count in sorted(frame_counts.items())
    }
    if bool(args.require_all_scenes):
        missing = sorted(set(range(100)) - set(frame_counts))
        insufficient = {
            int(scene): int(frame_counts.get(scene, 0))
            for scene in range(100)
            if int(frame_counts.get(scene, 0)) < int(args.min_frames_per_scene)
        }
        if missing or insufficient:
            raise RuntimeError(
                f"Formal P2 cache incomplete: missing={missing}, insufficient={insufficient}"
            )

    output = args.output_json or os.path.join(
        os.path.abspath(args.cache_dir), "cache_inventory.json"
    )
    save_inventory(inventory, output)
    detailed = os.path.splitext(output)[0] + "_detailed.json"
    atomic_save_json(payload, detailed)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"[SAVE] {output}\n[SAVE] {detailed}", flush=True)


if __name__ == "__main__":
    main()
