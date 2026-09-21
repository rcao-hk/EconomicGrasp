#!/usr/bin/env python3
"""Summarize Rep-B B0/B1/B2 with paired scene-level contrasts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rep_a_common import save_json, seed_for


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def scene_means(rows, metric):
    by = {}
    for r in rows:
        by.setdefault(int(r["scene_id"]), []).append(float(r[metric]))
    return {s: float(np.mean(v)) for s, v in by.items()}


def bootstrap_diff(a, b, *, seed, n=10000):
    scenes = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError("Paired Rep-B scene sets differ")
    d = np.array([b[s] - a[s] for s in scenes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "effect_b_minus_a": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "num_scenes": len(scenes),
        "positive_scenes": int((d > 0).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-root", required=True)
    p.add_argument("--splits", default="test_similar,test_novel")
    p.add_argument("--variants", default="B0,B1,B2")
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    root = Path(args.test_root)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    metrics = (
        "selected_utility",
        "success08",
        "utility_gain",
        "harm08",
        "move_rate",
        "within_ray_pair_accuracy",
        "probability_drift",
        "representation_cosine_distance",
        "representation_relative_l2",
    )
    contrasts = (("B0", "B1"), ("B1", "B2"), ("B0", "B2"))

    summaries = []
    effects = []
    for split in splits:
        data = {}
        for variant in variants:
            folder = root / variant / split
            summary_path = folder / "summary.json"
            csv_path = folder / "per_frame.csv"
            if not summary_path.is_file() or not csv_path.is_file():
                raise FileNotFoundError(f"Missing Rep-B result: {folder}")
            summary = json.loads(summary_path.read_text())
            if summary["variant"] != variant or summary["split"] != split:
                raise ValueError(f"Rep-B metadata mismatch: {folder}")
            data[variant] = read_rows(csv_path)
            for case, values in summary["cases"].items():
                summaries.append({
                    "variant": variant,
                    "split": split,
                    "case": case,
                    **values,
                })

        # All variants must contain exactly the same frame/case/action identities.
        keysets = {}
        lookup = {}
        for variant, rows in data.items():
            d = {
                (int(r["scene_id"]), int(r["anno_id"]), r["case"]): r
                for r in rows
            }
            lookup[variant] = d
            keysets[variant] = set(d)
        reference = keysets[variants[0]]
        for variant in variants[1:]:
            if keysets[variant] != reference:
                raise ValueError(f"Unpaired Rep-B frames/cases: {split}/{variant}")
        for key in reference:
            hashes = {lookup[v][key]["action_sha"] for v in variants}
            if len(hashes) != 1:
                raise ValueError(f"Action mismatch at {split}/{key}")

        cases = sorted({k[2] for k in reference})
        for case in cases:
            case_rows = {
                v: [r for k, r in lookup[v].items() if k[2] == case]
                for v in variants
            }
            for a, b in contrasts:
                if a not in variants or b not in variants:
                    continue
                for metric in metrics:
                    sa = scene_means(case_rows[a], metric)
                    sb = scene_means(case_rows[b], metric)
                    result = bootstrap_diff(
                        sa, sb,
                        seed=seed_for("rep_b_summary", split, case, metric, a, b),
                        n=args.bootstrap,
                    )
                    effects.append({
                        "split": split,
                        "case": case,
                        "metric": metric,
                        "contrast": f"{b}-{a}",
                        **result,
                        "note": "scene percentile bootstrap; single training seed",
                    })

    for name, rows in (("comparison", summaries), ("paired_effects", effects)):
        save_json(root / f"{name}.json", rows)
        with open(root / f"{name}.csv", "w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[REP-B] wrote {root/'comparison.csv'} and {root/'paired_effects.csv'}")


if __name__ == "__main__":
    main()
