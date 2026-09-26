#!/usr/bin/env python3
"""Combine split-wise Rep-P1 full-K audit tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLES = (
    "summary.csv",
    "native_offset_accuracy.csv",
    "argmax_confusion.csv",
    "policy_selected_offset_stats.csv",
    "advantage_deciles.csv",
    "focus_harmful.csv",
    "focus_beneficial.csv",
    "img_point_vs_img_region_disagreements.csv",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit_root", required=True)
    p.add_argument("--output_dir", default="")
    p.add_argument("--splits", default="test_similar,test_novel")
    return p.parse_args()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def main():
    args = parse_args()
    root = Path(args.audit_root)
    out = Path(args.output_dir) if args.output_dir else root
    out.mkdir(parents=True, exist_ok=True)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    combined = {}
    metas = []
    for split in splits:
        split_dir = root / split
        meta_path = split_dir / "audit_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        metas.append(json.loads(meta_path.read_text()))
        for table in TABLES:
            p = split_dir / table
            if p.is_file():
                combined.setdefault(table, []).extend(read_csv(p))

    for table, rows in combined.items():
        write_csv(out / table, rows)

    summary = combined.get("summary.csv", [])
    compact = []
    for row in summary:
        if row.get("subset") not in ("all", "top10"):
            continue
        compact.append({
            "split": row["split"],
            "variant": row["variant"],
            "subset": row["subset"],
            "checkpoint_epoch": row["checkpoint_epoch"],
            "selection_margin": row["selection_margin"],
            "curve_spearman": row["within_ray_spearman_mean"],
            "native_centered_pair_acc":
                row["native_centered_pair_accuracy"],
            "endpoint_direction_acc":
                row["endpoint_direction_accuracy"],
            "raw_argmax_match":
                row["raw_argmax_exact_oracle_match"],
            "raw_argmax_offset_mae_mm":
                row["raw_argmax_offset_mae_mm"],
            "raw_boundary_rate":
                row["raw_argmax_boundary_rate"],
            "boundary_overreach":
                row["boundary_overreach_rate"],
            "raw_utility_gain":
                row["raw_argmax_utility_gain"],
            "policy_utility_gain":
                row["policy_utility_gain"],
            "policy_rescue08":
                row["policy_rescue08"],
            "policy_harm08":
                row["policy_harm08"],
        })
    write_csv(out / "compact_comparison.csv", compact)

    # Representation deltas versus action-only and predicted geometry.
    delta_rows = []
    metric_keys = (
        "within_ray_spearman_mean",
        "native_centered_pair_accuracy",
        "endpoint_direction_accuracy",
        "raw_argmax_exact_oracle_match",
        "raw_argmax_offset_mae_mm",
        "raw_argmax_utility_gain",
        "policy_utility_gain",
        "policy_rescue08",
        "policy_harm08",
    )
    for row in summary:
        if row.get("subset") not in ("all", "top10"):
            continue
        for baseline, tag in (
            ("action_only", "action"),
            ("geo_pred", "geo"),
        ):
            base = next(
                (
                    x for x in summary
                    if x.get("split") == row.get("split")
                    and x.get("subset") == row.get("subset")
                    and x.get("variant") == baseline
                ),
                None,
            )
            if base is None:
                continue
            out_row = {
                "split": row["split"],
                "subset": row["subset"],
                "variant": row["variant"],
                "baseline": baseline,
            }
            for key in metric_keys:
                out_row[f"delta_vs_{tag}_{key}"] = f(row, key) - f(base, key)
            delta_rows.append(out_row)
    write_csv(out / "deltas.csv", delta_rows)

    report = {
        "experiment": "Rep-P1 full-K curve audit",
        "splits": splits,
        "split_meta": metas,
        "tables": sorted(combined),
        "primary_readout": (
            "Use native-centered pair accuracy, per-offset sign accuracy, "
            "raw argmax/oracle confusion, and boundary overreach to determine "
            "whether an evidence representation resolves same-ray metric depth. "
            "Policy metrics are secondary because they depend on the Seen-tuned margin."
        ),
    }
    (out / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    print("[REP-P1-FULL-K] combined comparison")
    for row in compact:
        print(
            f"  {row['split']:12s} {row['variant']:11s} {row['subset']:5s} "
            f"rho={float(row['curve_spearman']):+.3f} "
            f"native_pair={float(row['native_centered_pair_acc']):.3f} "
            f"endpoint={float(row['endpoint_direction_acc']):.3f} "
            f"argmax={float(row['raw_argmax_match']):.3f} "
            f"mae={float(row['raw_argmax_offset_mae_mm']):.1f}mm "
            f"overreach={100*float(row['boundary_overreach']):.1f}%"
        )


if __name__ == "__main__":
    main()
