#!/usr/bin/env python3
"""Validate analytic EconomicGrasp CDF/depth-wise-width scene caches."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


LEGACY_KEYS = (
    "points",
    "rotations",
    "depth",
    "scores",
    "widths",
    "pointid",
    "vgraspness",
    "topview",
    "collisions",
    "extend_angle",
    "valids",
    "num_angle",
    "num_depth",
)
NEW_KEYS = (
    "cdf_bins",
    "cdf_thresholds",
    "widths_depth_mm",
    "width_valids_depth",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir", required=True)
    p.add_argument(
        "--legacy_label_dir",
        default=None,
        help="Optional legacy extend-angle directory for field-by-field comparison.",
    )
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--strict", type=int, default=1)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def iter_files(root: Path, max_files: int) -> Iterable[Path]:
    files = sorted(root.glob("scene_*_labels.npz"))
    if max_files > 0:
        files = files[:max_files]
    return files


def main() -> None:
    args = parse_args()
    root = Path(args.label_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)

    files = list(iter_files(root, args.max_files))
    if not files:
        raise RuntimeError(f"No scene_*_labels.npz under {root}")

    legacy_root = Path(args.legacy_label_dir) if args.legacy_label_dir else None
    failures: list[str] = []
    schemas = Counter()
    thresholds_ref = None
    bin_hist = Counter()
    total_candidates = 0
    total_width_valid = 0
    total_points = 0
    legacy_compared = 0
    legacy_differences = Counter()

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = [k for k in LEGACY_KEYS + NEW_KEYS if k not in data]
                if missing:
                    raise RuntimeError(f"missing keys {missing}")

                rotations = data["rotations"]
                cdf_bins = data["cdf_bins"]
                widths_depth = data["widths_depth_mm"]
                width_valid = data["width_valids_depth"]
                thresholds = data["cdf_thresholds"].astype(np.float32)
                num_angle = int(np.asarray(data["num_angle"]).reshape(-1)[0])
                num_depth = int(np.asarray(data["num_depth"]).reshape(-1)[0])

                if rotations.ndim != 3:
                    raise RuntimeError(f"rotations must be [P,K,A], got {rotations.shape}")
                expected = rotations.shape + (num_depth,)
                if cdf_bins.shape != expected:
                    raise RuntimeError(
                        f"cdf_bins {cdf_bins.shape} != rotations+depth {expected}"
                    )
                if widths_depth.shape != expected or width_valid.shape != expected:
                    raise RuntimeError(
                        f"depth-wise width shapes differ: cdf={cdf_bins.shape}, "
                        f"width={widths_depth.shape}, valid={width_valid.shape}"
                    )
                if rotations.shape[-1] != num_angle:
                    raise RuntimeError(
                        f"rotations A={rotations.shape[-1]} != num_angle={num_angle}"
                    )
                if thresholds.ndim != 1 or thresholds.size < 2:
                    raise RuntimeError(f"bad thresholds shape {thresholds.shape}")
                if not np.all(np.isfinite(thresholds)) or not np.all(
                    thresholds[1:] > thresholds[:-1]
                ):
                    raise RuntimeError(f"thresholds not finite/increasing: {thresholds}")
                if cdf_bins.min(initial=0) < 0 or cdf_bins.max(initial=0) > len(thresholds):
                    raise RuntimeError(
                        f"cdf bins out of [0,{len(thresholds)}]: "
                        f"min={cdf_bins.min()}, max={cdf_bins.max()}"
                    )
                if width_valid.dtype not in (np.uint8, np.bool_):
                    raise RuntimeError(f"unexpected width_valid dtype {width_valid.dtype}")

                if thresholds_ref is None:
                    thresholds_ref = thresholds
                elif not np.allclose(thresholds, thresholds_ref, atol=0, rtol=0):
                    raise RuntimeError(
                        f"threshold mismatch {thresholds} vs {thresholds_ref}"
                    )

                schemas[(rotations.shape[1], num_angle, num_depth, len(thresholds))] += 1
                vals, counts = np.unique(cdf_bins, return_counts=True)
                for v, c in zip(vals.tolist(), counts.tolist()):
                    bin_hist[int(v)] += int(c)
                total_candidates += int(cdf_bins.size)
                total_width_valid += int(width_valid.astype(bool).sum())
                total_points += int(rotations.shape[0])

                if legacy_root is not None:
                    old_path = legacy_root / path.name
                    if not old_path.is_file():
                        raise FileNotFoundError(f"legacy counterpart missing: {old_path}")
                    with np.load(old_path, allow_pickle=False) as old:
                        legacy_compared += 1
                        for key in LEGACY_KEYS:
                            if key not in old:
                                legacy_differences[f"{key}:missing_old"] += 1
                                continue
                            a, b = old[key], data[key]
                            same = (
                                np.allclose(a, b, atol=1e-6, rtol=1e-6)
                                if np.issubdtype(a.dtype, np.floating)
                                else np.array_equal(a, b)
                            )
                            if not same:
                                legacy_differences[key] += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.name}: {exc}")

    summary = {
        "label_dir": str(root),
        "files": len(files),
        "failures": len(failures),
        "failure_examples": failures[:20],
        "schemas_K_A_D_T": [list(k) + [v] for k, v in sorted(schemas.items())],
        "thresholds": thresholds_ref.tolist() if thresholds_ref is not None else None,
        "points": total_points,
        "candidates": total_candidates,
        "cdf_bin_histogram": {str(k): v for k, v in sorted(bin_hist.items())},
        "cdf_positive_ratio": (
            1.0 - bin_hist.get(0, 0) / total_candidates if total_candidates else 0.0
        ),
        "width_valid_ratio": (
            total_width_valid / total_candidates if total_candidates else 0.0
        ),
        "legacy_files_compared": legacy_compared,
        "legacy_difference_counts": dict(legacy_differences),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.strict and failures:
        raise SystemExit(2)
    if args.strict and legacy_differences:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
