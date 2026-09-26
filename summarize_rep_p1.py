#!/usr/bin/env python3
"""Summarize Rep-P1 representation decodability across held-out splits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--variants", default="geo_pred,img_point,img_region"
    )
    p.add_argument("--splits", default="test_similar,test_novel")
    p.add_argument("--alignment_tol", type=float, default=1e-7)
    return p.parse_args()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.test_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    rows = []
    raw = {}
    for variant in variants:
        for split in splits:
            p = root / variant / split / "summary.json"
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            raw[(variant, split)] = d
            c = d["candidate_metrics"]
            m = d["policy"]
            t = d["policy_top10_native_score"]
            rows.append({
                "variant": variant,
                "split": split,
                "checkpoint_epoch": d["checkpoint_epoch"],
                "trainable_parameters": d["trainable_parameters"],
                "candidate_spearman": c["pred_exact_utility_spearman"],
                "candidate_success08_auroc": c["success08_auroc"],
                "candidate_success08_auprc": c["success08_auprc"],
                "within_ray_pairwise_accuracy":
                    c["within_ray_pairwise_accuracy"],
                "cdf_bce": c["cdf_bce"],
                "utility_gain": m["utility_gain"],
                "headroom_recovery": m["utility_headroom_recovery"],
                "success08_gain": m["success08_gain"],
                "rescue08": m["rescue08"],
                "harm08": m["harm08"],
                "change_rate": m["change_rate"],
                "top10_utility_gain": t["utility_gain"],
                "top10_headroom_recovery": t["utility_headroom_recovery"],
                "top10_success08_gain": t["success08_gain"],
                "top10_rescue08": t["rescue08"],
                "top10_harm08": t["harm08"],
                "selection_margin": d["selection_margin"],
            })
    if not rows:
        raise SystemExit("No Rep-P1 summary.json files found")

    # Exact action/label parity is a hard scientific contract, not a convenience.
    for split in splits:
        present = [
            raw[(v, split)] for v in variants if (v, split) in raw
        ]
        if len(present) < 2:
            continue
        ref = present[0]
        for d in present[1:]:
            for section, keys in (
                ("policy", ("native_utility", "oracle_utility", "native_success08")),
                ("policy_top10_native_score",
                 ("native_utility", "oracle_utility", "native_success08")),
            ):
                for key in keys:
                    a = float(ref[section][key])
                    b = float(d[section][key])
                    if abs(a - b) > args.alignment_tol:
                        raise RuntimeError(
                            f"Rep-P1 action/label alignment violated on "
                            f"{split}/{section}/{key}: {a} vs {b}"
                        )

    # Deltas against the predicted-geometry baseline.
    for row in rows:
        base = next(
            (
                x for x in rows
                if x["variant"] == "geo_pred" and x["split"] == row["split"]
            ),
            None,
        )
        if base is None:
            continue
        for key in (
            "candidate_spearman",
            "candidate_success08_auroc",
            "candidate_success08_auprc",
            "within_ray_pairwise_accuracy",
            "utility_gain",
            "headroom_recovery",
            "success08_gain",
            "rescue08",
            "harm08",
            "top10_utility_gain",
            "top10_headroom_recovery",
            "top10_success08_gain",
            "top10_rescue08",
            "top10_harm08",
        ):
            row[f"delta_vs_geo_{key}"] = float(row[key]) - float(base[key])

    write_csv(out / "comparison.csv", rows)
    (out / "comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True)
    )

    print("[REP-P1] fixed-action representation comparison")
    for row in rows:
        print(
            f"  {row['variant']:10s} {row['split']:12s} "
            f"rho={row['candidate_spearman']:+.3f} "
            f"AUROC08={row['candidate_success08_auroc']:.3f} "
            f"pair={row['within_ray_pairwise_accuracy']:.3f} "
            f"dU={row['utility_gain']:+.5f} "
            f"headroom={100*row['headroom_recovery']:+.1f}% "
            f"top10_dU={row['top10_utility_gain']:+.5f} "
            f"harm={100*row['harm08']:.2f}%"
        )


if __name__ == "__main__":
    main()
