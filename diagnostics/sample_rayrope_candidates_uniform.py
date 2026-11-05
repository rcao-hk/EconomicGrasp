#!/usr/bin/env python3
"""Uniformly sample fixed candidate sets independently of predicted scores.

Use this on a complete pre-filter candidate pool exported before score-based
top-K selection. The explicit --input_is_full_pool acknowledgement prevents
mistaking the existing 64 score-stratified rows for a full proposal pool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--manifest_json", default=None)
    p.add_argument("--sample_col", default="sample_id")
    p.add_argument("--candidate_col", default="candidate_id")
    p.add_argument("--n_per_sample", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--input_is_full_pool", action="store_true",
                   help="Required acknowledgement that input contains the complete pre-score-filter pool.")
    args = p.parse_args()
    if not args.input_is_full_pool:
        raise SystemExit("Refusing to sample: pass --input_is_full_pool only after verifying this is the complete pre-score-filter pool.")
    if args.n_per_sample < 1:
        raise ValueError("--n_per_sample must be >=1")

    src = Path(args.input_csv)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    with src.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for col in (args.sample_col, args.candidate_col):
        if col not in fields:
            raise ValueError(f"Missing column {col!r}; available headers: {fields}")

    groups = defaultdict(list)
    for row in rows:
        groups[row[args.sample_col]].append(row)
    rng = random.Random(args.seed)
    selected = []
    summary = {}
    for sample in sorted(groups):
        candidates = groups[sample]
        ids = [r[args.candidate_col] for r in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate candidate IDs in sample {sample!r}; deduplicate before sampling.")
        n = min(args.n_per_sample, len(candidates))
        chosen = rng.sample(candidates, n)
        selected.extend(chosen)
        summary[sample] = {"pool_size": len(candidates), "selected": n}

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    tmp.replace(out)

    manifest = {
        "input_csv": str(src),
        "input_sha256": digest,
        "output_csv": str(out),
        "sampling": "uniform_without_replacement_within_each_sample",
        "score_independent": True,
        "n_per_sample_requested": args.n_per_sample,
        "seed": args.seed,
        "num_samples": len(groups),
        "num_input_candidates": len(rows),
        "num_selected_candidates": len(selected),
        "per_sample": summary,
        "warning": "Uniformity is valid only if input_csv is the complete candidate pool before score-based filtering or truncation.",
    }
    manifest_path = Path(args.manifest_json) if args.manifest_json else out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_csv": str(out), "manifest_json": str(manifest_path),
                      "num_samples": len(groups), "num_selected": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
