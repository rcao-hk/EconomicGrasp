#!/usr/bin/env python3
"""Summarize DCR-E1-4 with same-ranking causal controls."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path

from e1e2_common import atomic_file, save_json


def _write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, restval="")
    writer.writeheader()
    writer.writerows(rows)
    with atomic_file(path) as f:
        f.write(buf.getvalue().encode())


def _load_rows(root: Path, label: str):
    rows = []
    for path in sorted((root / "test" / label / "official").glob("*/*/*/summary.json")):
        d = json.loads(path.read_text())
        rows.append({
            "source_model": label,
            "variant": d["variant"],
            "method": d["method"],
            "case": d["case"],
            "split": d["split"],
            "mean_accuracy": float(d["mean_accuracy"]),
            "sample_interval": float(d["sample_interval"]),
            "seen_role": d["seen_role"],
        })
    return rows


def _index(rows):
    return {(r["split"], r["case"], r["method"]): r for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", required=True)
    p.add_argument("--dcr-label", default="dcr")
    p.add_argument("--e1-label", default="e1_ref")
    args = p.parse_args()

    root = Path(args.work_root)
    dcr_rows = _load_rows(root, args.dcr_label)
    e1_rows = _load_rows(root, args.e1_label)
    if not dcr_rows:
        raise RuntimeError("No DCR official summaries found")
    if not e1_rows:
        raise RuntimeError("No E1-reference official summaries found")

    all_rows = dcr_rows + e1_rows
    _write_csv(root / "comparison.csv", all_rows)
    save_json(root / "comparison.json", all_rows)

    dcr = _index(dcr_rows)
    e1 = _index(e1_rows)
    keys = sorted(
        set((s, c) for s, c, m in dcr if m == "stage1")
        & set((s, c) for s, c, m in e1 if m == "stage1")
    )
    if not keys:
        raise RuntimeError("No matched DCR/E1 Stage-1-ranking cells")

    effects = []
    for split, case in keys:
        d = dcr[(split, case, "stage1")]["mean_accuracy"]
        e = e1[(split, case, "stage1")]["mean_accuracy"]
        native_row = dcr.get((split, case, "native"))
        row = {
            "split": split,
            "case": case,
            "e1_stage1": e,
            "dcr_stage1": d,
            "dcr_over_e1": d - e,
        }
        if native_row is not None:
            n = native_row["mean_accuracy"]
            row.update(
                native=n,
                e1_over_native=e - n,
                dcr_over_native=d - n,
            )
        effects.append(row)

    _write_csv(root / "correction_effects.csv", effects)
    save_json(root / "correction_effects.json", effects)

    def mean(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else None

    bias_rows = [r for r in effects if r["case"].startswith("bias:")]
    scale_rows = [r for r in effects if r["case"].startswith("scale:")]
    smooth_rows = [r for r in effects if r["case"].startswith("smooth:")]
    nominal_rows = [r for r in effects if r["case"] == "nominal"]
    heldout = [r for r in effects if r["split"] in ("test_similar", "test_novel")]

    macro = {
        "cells": len(effects),
        "dcr_over_e1_all": mean(effects, "dcr_over_e1"),
        "dcr_over_e1_nominal": mean(nominal_rows, "dcr_over_e1"),
        "dcr_over_e1_bias": mean(bias_rows, "dcr_over_e1"),
        "dcr_over_e1_scale": mean(scale_rows, "dcr_over_e1"),
        "dcr_over_e1_smooth": mean(smooth_rows, "dcr_over_e1"),
        "dcr_over_e1_heldout": mean(heldout, "dcr_over_e1"),
        "dcr_over_native_all": mean(effects, "dcr_over_native"),
        "dcr_over_native_smooth": mean(smooth_rows, "dcr_over_native"),
    }
    save_json(root / "macro_summary.json", macro)

    print(f"[DCR-E1-4] {len(effects)} matched cells")
    print(json.dumps(macro, indent=2, sort_keys=True))
    print(f"[DCR-E1-4] detailed effects: {root / 'correction_effects.csv'}")


if __name__ == "__main__":
    main()
