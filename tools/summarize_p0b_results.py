#!/usr/bin/env python3
"""Summarize P0-B official AP JSON files across splits and variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_VARIANTS = (
    "student",
    "teacher_full",
    "teacher_common",
    "oracle_hybrid",
)
DEFAULT_SPLITS = ("test_seen", "test_similar", "test_novel")


def _parse_csv(text: str) -> Tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("Comma-separated argument cannot be empty.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = _parse_csv(args.variants)
    splits = _parse_csv(args.splits)

    records: List[Dict[str, object]] = []
    by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    missing: List[str] = []

    for split in splits:
        meta = root / split / "_p0b_meta"
        for variant in variants:
            path = meta / f"eval_{variant}_{split}_{args.camera}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            by_key[(variant, split)] = record
            records.append(record)

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing P0-B evaluation JSON files: " + ", ".join(missing[:20])
        )
    if not records:
        raise RuntimeError(f"No P0-B evaluation records found under {root}.")

    student_by_split = {
        split: by_key[("student", split)]
        for split in splits
        if ("student", split) in by_key
    }

    rows: List[Dict[str, object]] = []
    summary_json: Dict[str, object] = {
        "protocol": "P0-B-official-AP-oracle-hybrid-v1",
        "camera": args.camera,
        "root": str(root),
        "variants": {},
    }

    for variant in variants:
        variant_records = [
            by_key[(variant, split)]
            for split in splits
            if (variant, split) in by_key
        ]
        if not variant_records:
            continue
        variant_json: Dict[str, object] = {"splits": {}}
        split_ap_values: List[float] = []
        split_ap04_values: List[float] = []
        split_ap08_values: List[float] = []

        for split in splits:
            record = by_key.get((variant, split))
            if record is None:
                continue
            student = student_by_split.get(split)
            ap = float(record["ap"])
            ap04 = float(record["ap0.4"])
            ap08 = float(record["ap0.8"])
            delta_ap = ap - float(student["ap"]) if student is not None else 0.0
            delta_ap04 = ap04 - float(student["ap0.4"]) if student is not None else 0.0
            delta_ap08 = ap08 - float(student["ap0.8"]) if student is not None else 0.0
            row = {
                "variant": variant,
                "split": split,
                "ap": ap,
                "ap0.8": ap08,
                "ap0.4": ap04,
                "delta_ap_vs_student": delta_ap,
                "delta_ap0.8_vs_student": delta_ap08,
                "delta_ap0.4_vs_student": delta_ap04,
                "ap_percent": 100.0 * ap,
                "delta_ap_points_vs_student": 100.0 * delta_ap,
            }
            rows.append(row)
            variant_json["splits"][split] = row
            split_ap_values.append(ap)
            split_ap04_values.append(ap04)
            split_ap08_values.append(ap08)

        variant_json["mean_over_available_splits"] = {
            "ap": sum(split_ap_values) / len(split_ap_values),
            "ap0.4": sum(split_ap04_values) / len(split_ap04_values),
            "ap0.8": sum(split_ap08_values) / len(split_ap08_values),
        }
        summary_json["variants"][variant] = variant_json

    student_mean = None
    student_entry = summary_json["variants"].get("student")
    if student_entry is not None:
        student_mean = float(student_entry["mean_over_available_splits"]["ap"])
    for variant, entry in summary_json["variants"].items():
        mean_ap = float(entry["mean_over_available_splits"]["ap"])
        entry["mean_over_available_splits"]["delta_ap_vs_student"] = (
            mean_ap - student_mean if student_mean is not None else 0.0
        )

    csv_path = output_dir / "p0b_official_ap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "p0b_official_ap_summary.json"
    json_path.write_text(
        json.dumps(summary_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    split_headers = [split.replace("test_", "").title() for split in splits]
    md_lines = [
        "# P0-B Official GraspNet AP Summary",
        "",
        "All values below are AP percentage points. Parentheses report the delta against the student on the same split.",
        "",
        "| Variant | " + " | ".join(split_headers) + " | Mean |",
        "|---|" + "|".join(["---:" for _ in range(len(split_headers) + 1)]) + "|",
    ]
    for variant in variants:
        entry = summary_json["variants"].get(variant)
        if entry is None:
            continue
        cells = []
        for split in splits:
            record = entry["splits"].get(split)
            if record is None:
                cells.append("—")
            else:
                cells.append(
                    f"{100.0 * float(record['ap']):.3f} "
                    f"({100.0 * float(record['delta_ap_vs_student']):+.3f})"
                )
        mean_record = entry["mean_over_available_splits"]
        cells.append(
            f"{100.0 * float(mean_record['ap']):.3f} "
            f"({100.0 * float(mean_record['delta_ap_vs_student']):+.3f})"
        )
        md_lines.append(f"| {variant} | " + " | ".join(cells) + " |")

    md_lines.extend(
        [
            "",
            "## P0-B decision",
            "",
            "- `oracle_hybrid > student`: the selective privileged CDF signal survives full decoding and official evaluation.",
            "- Local utility improves but official AP does not: stop treating analytic local CDF utility as an evaluator-aligned target.",
            "- `teacher_full` / `teacher_common < student`: uniform or support-only teacher replacement remains invalid.",
        ]
    )
    md_path = output_dir / "p0b_official_ap_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(md_path.read_text(encoding="utf-8"))
    print(f"[P0-B] CSV:  {csv_path}")
    print(f"[P0-B] JSON: {json_path}")
    print(f"[P0-B] MD:   {md_path}")


if __name__ == "__main__":
    main()
