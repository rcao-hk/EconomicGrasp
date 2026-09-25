#!/usr/bin/env python3
"""Merge multi-GPU grasp-behavior visualization shards into one index/summary."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tools.grasp_behavior_viz import write_csv, write_html_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)

    rows = []
    for path in sorted(root.glob("summary_shard*.csv")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    if rows:
        write_csv(root / "summary.csv", rows)

    groups = {}
    for split in ("test_seen", "test_similar", "test_novel"):
        entries = []
        for index in sorted((root / split).glob("scene_*/ann_*/index.html")):
            label = f"{index.parents[1].name} {index.parent.name}"
            entries.append((label, index))
        if entries:
            groups[split] = entries
    write_html_index(
        root / "index.html",
        "EconomicGrasp behavior visualization",
        groups, base_dir=root)
    print(f"[GRASP-VIZ] merged {len(rows)} case rows -> {root}")


if __name__ == "__main__":
    main()
