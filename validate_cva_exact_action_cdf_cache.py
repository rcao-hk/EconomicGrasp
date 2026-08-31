#!/usr/bin/env python3
"""Validate the current-only CVA-CDF exact-action cache."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from exact_action_cdf_cache import scan_cache, save_inventory
from exact_action_cdf_common import validate_current_stage1_cdf_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--expected_pose_depth_mode", default="global_film")
    parser.add_argument("--expected_use_fuse_depth", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--strict", type=int, choices=(0, 1), default=1)
    parser.add_argument("--require_all_scenes", type=int, choices=(0, 1), default=0)
    parser.add_argument("--min_frames_per_scene", type=int, default=26)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, contract = validate_current_stage1_cdf_checkpoint(
        args.base_checkpoint,
        expected_pose_depth_mode=args.expected_pose_depth_mode,
        expected_use_fuse_depth=bool(args.expected_use_fuse_depth),
    )
    metadata, inventory, failures = scan_cache(
        args.cache_dir,
        expected_checkpoint_sha256=contract.checkpoint_sha256,
        max_files=int(args.max_files),
        strict=bool(args.strict),
        check_values=True,
    )
    payload = inventory.to_dict()
    payload["checkpoint_contract"] = contract.to_dict()
    payload["num_failures"] = len(failures)
    payload["failures"] = failures
    payload["frame_examples"] = [
        {
            "scene_id": item.scene_id,
            "anno_id": item.anno_id,
            "num_rows": item.num_rows,
            "path": item.path,
        }
        for item in metadata[:10]
    ]
    frame_counts = Counter(item.scene_id for item in metadata)
    payload["frames_per_scene"] = {
        str(scene_id): int(count)
        for scene_id, count in sorted(frame_counts.items())
    }
    if bool(args.require_all_scenes):
        missing_scenes = sorted(set(range(100)) - set(frame_counts))
        insufficient = {
            int(scene_id): int(frame_counts.get(scene_id, 0))
            for scene_id in range(100)
            if int(frame_counts.get(scene_id, 0)) < int(args.min_frames_per_scene)
        }
        if missing_scenes or insufficient:
            raise RuntimeError(
                "Formal cache is incomplete: "
                f"missing_scenes={missing_scenes}, "
                f"frames_below_{int(args.min_frames_per_scene)}={insufficient}."
            )
    output = args.output_json or os.path.join(
        os.path.abspath(args.cache_dir), "cache_inventory.json"
    )
    save_inventory(inventory, output)
    # Preserve the richer checkpoint/failure context as a companion file.
    detailed = os.path.splitext(output)[0] + "_detailed.json"
    from exact_action_cdf_common import atomic_save_json

    atomic_save_json(payload, detailed)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"[SAVE] {output}", flush=True)
    print(f"[SAVE] {detailed}", flush=True)


if __name__ == "__main__":
    main()
