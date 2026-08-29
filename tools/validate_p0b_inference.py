#!/usr/bin/env python3
"""Validate P0-B multi-worker inference outputs and exact-query contracts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


DEFAULT_VARIANTS = (
    "student",
    "teacher_full",
    "teacher_common",
    "oracle_hybrid",
)


def _parse_csv(text: str) -> Tuple[str, ...]:
    values = tuple(x.strip() for x in str(text).split(",") if x.strip())
    if not values:
        raise ValueError("Comma-separated argument cannot be empty.")
    return values


def _scene_ids(split: str) -> range:
    if split == "test_seen":
        return range(100, 130)
    if split == "test_similar":
        return range(130, 160)
    if split == "test_novel":
        return range(160, 190)
    raise ValueError(f"Unsupported split: {split!r}.")


def _expected_ann_ids(sample_fraction: float) -> List[int]:
    fraction = float(sample_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("sample_fraction must be positive and finite.")
    if fraction >= 1.0:
        return list(range(256))
    stride = max(1, int(round(1.0 / fraction)))
    return list(range(0, 256, stride))


def _relative_prediction_files(
    root: Path,
    *,
    split: str,
    camera: str,
) -> Set[str]:
    result: Set[str] = set()
    for scene_id in _scene_ids(split):
        directory = root / f"scene_{scene_id:04d}" / camera
        if not directory.is_dir():
            continue
        for path in directory.glob("*.npy"):
            result.add(str(path.relative_to(root)))
    return result


def _checkpoint_signature(record: Dict[str, object]) -> Tuple[object, ...]:
    return (
        record.get("path"),
        record.get("size_bytes"),
        record.get("mtime_ns"),
        record.get("distill_stage"),
        record.get("epoch"),
        record.get("geometry_depth_source"),
        record.get("pose_depth_mode"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=("test_seen", "test_similar", "test_novel"),
    )
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--expected_effective_k", type=int, default=1)
    parser.add_argument(
        "--expected_gate_mode",
        choices=("balanced", "ordinary"),
        default="balanced",
    )
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    if args.world_size <= 0:
        raise ValueError("--world_size must be positive.")
    if args.expected_effective_k <= 0:
        raise ValueError("--expected_effective_k must be positive.")

    save_root = Path(args.save_root).resolve()
    meta_dir = save_root / "_p0b_meta"
    variants = _parse_csv(args.variants)
    summary_paths = sorted(meta_dir.glob("worker_*_summary.json"))
    if len(summary_paths) != args.world_size:
        raise RuntimeError(
            f"Expected {args.world_size} worker summaries in {meta_dir}, "
            f"found {len(summary_paths)}: {summary_paths}"
        )

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    ranks = sorted(int(record["worker_rank"]) for record in summaries)
    expected_ranks = list(range(args.world_size))
    if ranks != expected_ranks:
        raise RuntimeError(f"Worker ranks mismatch: got {ranks}, expected {expected_ranks}.")

    teacher_signatures = {
        _checkpoint_signature(record["teacher_checkpoint"])
        for record in summaries
    }
    student_signatures = {
        _checkpoint_signature(record["student_checkpoint"])
        for record in summaries
    }
    if len(teacher_signatures) != 1 or len(student_signatures) != 1:
        raise RuntimeError("P0-B workers did not use identical teacher/student checkpoints.")

    exact_min = {"seed": 1.0, "pixel": 1.0, "view": 1.0}
    processed_total = 0
    weighted_diagnostics: Dict[str, float] = {}
    diagnostic_weight = 0

    for record in summaries:
        if record.get("protocol") != "P0-B-official-AP-oracle-hybrid-v1":
            raise RuntimeError(f"Unexpected protocol in worker summary: {record.get('protocol')!r}.")
        if record.get("split") != args.split or record.get("camera") != args.camera:
            raise RuntimeError(
                "Worker summary split/camera mismatch: "
                f"{record.get('split')}/{record.get('camera')} vs "
                f"{args.split}/{args.camera}."
            )
        if int(record.get("world_size", -1)) != args.world_size:
            raise RuntimeError("Worker summary world_size mismatch.")
        effective_k = int(record.get("effective_k", -1))
        if effective_k != int(args.expected_effective_k):
            raise RuntimeError(
                f"Worker effective_k mismatch: got {effective_k}, "
                f"expected {args.expected_effective_k}."
            )
        expected_top4 = int(args.expected_effective_k) == 4
        if bool(record.get("use_top4_view_infer", False)) != expected_top4:
            raise RuntimeError(
                "Worker use_top4_view_infer metadata is inconsistent with "
                f"expected_effective_k={args.expected_effective_k}."
            )
        if str(record.get("gate_mode", "")) != args.expected_gate_mode:
            raise RuntimeError(
                "Worker gate_mode mismatch: got "
                f"{record.get('gate_mode')!r}, expected "
                f"{args.expected_gate_mode!r}."
            )
        if tuple(record.get("variants", [])) != variants:
            raise RuntimeError(
                f"Worker variant mismatch: {record.get('variants')} vs {variants}."
            )
        processed = int(record.get("processed_samples", 0))
        processed_total += processed
        if processed > 0:
            for key in exact_min:
                value = float(record["exact_min"][key])
                exact_min[key] = min(exact_min[key], value)
                if abs(value - 1.0) > 1.0e-8:
                    raise RuntimeError(
                        f"Exact-query contract failed for {key}: worker "
                        f"{record['worker_rank']} reported {value}."
                    )
            diagnostics = record.get("diagnostic_means", {})
            for key, value in diagnostics.items():
                weighted_diagnostics[key] = (
                    weighted_diagnostics.get(key, 0.0) + float(value) * processed
                )
            diagnostic_weight += processed

    file_sets: Dict[str, Set[str]] = {}
    for variant in variants:
        variant_root = save_root / variant
        if not variant_root.is_dir():
            raise FileNotFoundError(f"Variant output directory is missing: {variant_root}")
        file_sets[variant] = _relative_prediction_files(
            variant_root,
            split=args.split,
            camera=args.camera,
        )

    reference_variant = variants[0]
    reference_files = file_sets[reference_variant]
    for variant in variants[1:]:
        if file_sets[variant] != reference_files:
            only_reference = sorted(reference_files - file_sets[variant])[:20]
            only_variant = sorted(file_sets[variant] - reference_files)[:20]
            raise RuntimeError(
                f"P0-B output file sets differ between {reference_variant} and {variant}. "
                f"missing_in_{variant}={only_reference}, extra_in_{variant}={only_variant}."
            )

    expected_ann_ids = _expected_ann_ids(args.sample_fraction)
    expected_total = len(list(_scene_ids(args.split))) * len(expected_ann_ids)
    actual_total = len(reference_files)
    if not args.allow_incomplete and actual_total != expected_total:
        raise RuntimeError(
            f"P0-B output is incomplete: actual={actual_total}, expected={expected_total}."
        )
    if args.allow_incomplete and actual_total <= 0:
        raise RuntimeError("P0-B smoke output contains no prediction files.")

    diagnostic_means = {
        key: value / max(diagnostic_weight, 1)
        for key, value in sorted(weighted_diagnostics.items())
    }
    combined = {
        "protocol": "P0-B-official-AP-oracle-hybrid-v1",
        "split": args.split,
        "camera": args.camera,
        "world_size": args.world_size,
        "sample_fraction": float(args.sample_fraction),
        "variants": list(variants),
        "effective_k": int(args.expected_effective_k),
        "use_top4_view_infer": int(args.expected_effective_k) == 4,
        "gate_mode": args.expected_gate_mode,
        "worker_summary_paths": [str(path) for path in summary_paths],
        "processed_in_current_invocation": processed_total,
        "prediction_files_per_variant": actual_total,
        "expected_prediction_files_per_variant": expected_total,
        "complete": actual_total == expected_total,
        "exact_min": exact_min,
        "teacher_checkpoint": summaries[0]["teacher_checkpoint"],
        "student_checkpoint": summaries[0]["student_checkpoint"],
        "diagnostic_means": diagnostic_means,
    }
    output_path = meta_dir / f"inference_summary_{args.split}_{args.camera}.json"
    output_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"[P0-B][CHECK] split={args.split} files/variant={actual_total}/{expected_total} "
        f"processed_now={processed_total} exact={exact_min} complete={combined['complete']}"
    )
    print(f"[P0-B][CHECK] saved {output_path}")


if __name__ == "__main__":
    main()
