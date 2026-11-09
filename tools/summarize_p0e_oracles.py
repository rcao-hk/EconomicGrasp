#!/usr/bin/env python3
"""Aggregate official P0-E AP and select the next technical priority."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_VARIANTS = (
    "student_original",
    "exact_action_rerank",
    "local_field_oracle",
    "proposal_union_oracle",
)
DEFAULT_SPLITS = ("test_seen", "test_similar", "test_novel")


def _csv_tuple(text: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("Comma-separated argument cannot be empty.")
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    keys.extend(sorted({key for row in rows for key in row.keys()} - set(keys)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_eval(
    meta_root: Path,
    variant: str,
    split: str,
    camera: str,
) -> Dict[str, object]:
    path = meta_root / f"eval_{variant}_{split}_{camera}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("variant")) != variant or str(payload.get("split")) != split:
        raise RuntimeError(f"Mismatched P0-E eval record: {path}")
    return payload


def _mean(values: Sequence[float]) -> float:
    return float(np.asarray(values, dtype=np.float64).mean())


def _decision(gains: Mapping[str, float], threshold_pp: float) -> Dict[str, object]:
    labels = {
        "ranking_gain_pp": "action-field ranking/calibration",
        "local_increment_pp": "local pose refinement",
        "proposal_increment_pp": "off-policy proposal distillation",
    }
    best_key = max(gains, key=lambda key: float(gains[key]))
    best_gain = float(gains[best_key])
    if best_gain < float(threshold_pp):
        return {
            "recommended_priority": "none",
            "recommended_reason": (
                f"No P0-E oracle exceeds {threshold_pp:.3f} AP points; audit the "
                "candidate/evaluator contract before further method work."
            ),
        }
    return {
        "recommended_priority": labels[best_key],
        "recommended_reason": (
            f"The largest mean ceiling is {best_gain:.3f} AP points from "
            f"{labels[best_key]}."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--min_actionable_gain_pp", type=float, default=0.5)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    prediction_root = Path(args.prediction_root).resolve()
    meta_root = prediction_root / "_p0e_meta"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else meta_root
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = _csv_tuple(args.splits)
    variants = _csv_tuple(args.variants)

    records: Dict[Tuple[str, str], Dict[str, object]] = {}
    long_rows: List[Dict[str, object]] = []
    for split in splits:
        for variant in variants:
            payload = _load_eval(meta_root, variant, split, args.camera)
            records[(split, variant)] = payload
            long_rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "ap": float(payload["ap"]),
                    "ap0.4": float(payload["ap0.4"]),
                    "ap0.8": float(payload["ap0.8"]),
                    "ap_percent": float(payload["ap_percent"]),
                    "ap0.4_percent": float(payload["ap0.4_percent"]),
                    "ap0.8_percent": float(payload["ap0.8_percent"]),
                }
            )

    mean_by_variant: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        mean_by_variant[variant] = {
            metric: _mean(
                [float(records[(split, variant)][metric]) for split in splits]
            )
            for metric in ("ap", "ap0.4", "ap0.8")
        }
        long_rows.append(
            {
                "split": "mean_of_splits",
                "variant": variant,
                "ap": mean_by_variant[variant]["ap"],
                "ap0.4": mean_by_variant[variant]["ap0.4"],
                "ap0.8": mean_by_variant[variant]["ap0.8"],
                "ap_percent": 100.0 * mean_by_variant[variant]["ap"],
                "ap0.4_percent": 100.0 * mean_by_variant[variant]["ap0.4"],
                "ap0.8_percent": 100.0 * mean_by_variant[variant]["ap0.8"],
            }
        )

    missing = sorted(set(DEFAULT_VARIANTS) - set(variants))
    if missing:
        raise ValueError(
            f"Gap decomposition requires all canonical variants; missing {missing}."
        )

    decomposition_rows: List[Dict[str, object]] = []
    for split in (*splits, "mean_of_splits"):
        if split == "mean_of_splits":
            ap = {variant: mean_by_variant[variant]["ap"] for variant in variants}
        else:
            ap = {
                variant: float(records[(split, variant)]["ap"])
                for variant in variants
            }
        baseline = ap["student_original"]
        rerank = ap["exact_action_rerank"]
        local = ap["local_field_oracle"]
        union = ap["proposal_union_oracle"]
        decomposition_rows.append(
            {
                "split": split,
                "student_ap_percent": 100.0 * baseline,
                "rerank_ap_percent": 100.0 * rerank,
                "local_field_ap_percent": 100.0 * local,
                "proposal_union_ap_percent": 100.0 * union,
                "ranking_gain_pp": 100.0 * (rerank - baseline),
                "local_increment_pp": 100.0 * (local - rerank),
                "proposal_increment_pp": 100.0 * (union - rerank),
                "local_total_gain_pp": 100.0 * (local - baseline),
                "proposal_total_gain_pp": 100.0 * (union - baseline),
            }
        )

    mean_row = decomposition_rows[-1]
    gains = {
        key: float(mean_row[key])
        for key in (
            "ranking_gain_pp",
            "local_increment_pp",
            "proposal_increment_pp",
        )
    }
    decision = _decision(gains, float(args.min_actionable_gain_pp))
    mean_row.update(decision)

    long_path = output_dir / "p0e_official_ap_long.csv"
    decomposition_path = output_dir / "p0e_gap_decomposition.csv"
    _write_csv(long_path, long_rows)
    _write_csv(decomposition_path, decomposition_rows)

    report = {
        "prediction_root": str(prediction_root),
        "camera": args.camera,
        "splits": list(splits),
        "variants": list(variants),
        "mean_gains_pp": gains,
        **decision,
        "official_ap_csv": str(long_path),
        "gap_decomposition_csv": str(decomposition_path),
    }
    report_path = output_dir / "p0e_decision.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    markdown = [
        "# P0-E oracle decomposition",
        "",
        "| Split | Student AP | Exact rerank | Local field | Proposal union | "
        "Ranking gain | Local incremental | Proposal incremental |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decomposition_rows:
        markdown.append(
            "| {split} | {student_ap_percent:.3f} | {rerank_ap_percent:.3f} | "
            "{local_field_ap_percent:.3f} | {proposal_union_ap_percent:.3f} | "
            "{ranking_gain_pp:+.3f} | {local_increment_pp:+.3f} | "
            "{proposal_increment_pp:+.3f} |".format(**row)
        )
    markdown.extend(
        [
            "",
            f"**Recommended next priority:** {decision['recommended_priority']}.",
            "",
            str(decision["recommended_reason"]),
            "",
            "All three oracle rows are privileged upper bounds, not deployable RGB-only results.",
        ]
    )
    markdown_path = output_dir / "P0E_REPORT.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
