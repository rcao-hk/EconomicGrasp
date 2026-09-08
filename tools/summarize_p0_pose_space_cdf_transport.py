#!/usr/bin/env python3
"""Summarize P0 Seen/Similar/Novel summary.json files into one CSV/Markdown."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List


KEYS = [
    "student_valid_query_ratio",
    "teacher_valid_query_ratio",
    "common_valid_query_ratio",
    "support_iou_before",
    "support_iou_after_continuous",
    "support_iou_after_nearest",
    "teacher_support_recall_before",
    "teacher_support_recall_after_nearest",
    "p0_recovered_continuous_fraction_of_missing_teacher_support",
    "p0_recovered_nearest_fraction_of_missing_teacher_support",
    "center_error_mean_m",
    "center_parallel_abs_mean_m",
    "center_lateral_mean_m",
    "center_nearest_action_residual_mean_m",
    "center_continuous_transportable_5mm",
    "center_nearest_transportable_5mm",
    "center_depth_retained_fraction",
    "label_point_dist_mean_m",
    "label_lateral_mean_m",
    "label_nearest_action_residual_mean_m",
    "common_gt_cdf_element_exact_before",
    "paired_gt_cdf_element_exact_before",
    "paired_gt_cdf_element_exact_after_nearest",
    "teacher_cdf_bce_before_transport",
    "teacher_cdf_bce_after_nearest_transport",
    "teacher_cdf_transport_improved_query_ratio",
]


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "nan"
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="OUTPUT_ROOT containing test_seen/test_similar/test_novel")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for split in ("test_seen", "test_similar", "test_novel"):
        path = os.path.join(args.root, split, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", {})
        row: Dict[str, Any] = {
            "split": split,
            "global_images": payload.get("global_images", ""),
        }
        for key in KEYS:
            value = metrics.get(key, float("nan"))
            if key.endswith("_mean_m") and isinstance(value, (int, float)):
                row[key[:-2] + "_mm"] = float(value) * 1000.0
            else:
                row[key] = value
        row["support_iou_gain_nearest"] = (
            float(metrics.get("support_iou_after_nearest", float("nan")))
            - float(metrics.get("support_iou_before", float("nan")))
        )
        row["cdf_bce_delta_after_minus_before"] = (
            float(metrics.get("teacher_cdf_bce_after_nearest_transport", float("nan")))
            - float(metrics.get("teacher_cdf_bce_before_transport", float("nan")))
        )
        rows.append(row)

    if not rows:
        raise SystemExit(f"No P0 summary.json files found under {args.root}")

    csv_path = args.output_csv or os.path.join(args.root, "p0_split_summary.csv")
    md_path = args.output_md or os.path.join(args.root, "p0_split_summary.md")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# P0 Pose-Space CDF Transport — Split Summary\n\n")
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_fmt(row.get(k, "")) for k in fieldnames) + " |\n")

    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
