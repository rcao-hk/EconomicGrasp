#!/usr/bin/env python3
"""Summarize evaluator tensors without re-evaluation or mixing full/subset AP."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(root):
    rows = []
    for manifest in sorted(root.rglob("gvar_inference_protocol.json")):
        p = json.loads(manifest.read_text())
        summary = manifest.with_name("gvar_inference_summary.json")
        if not summary.exists() or not json.loads(summary.read_text()).get("complete"):
            continue
        matches = list(manifest.parent.glob(f"ap_{p['split']}_{p['camera']}.npy"))
        if len(matches) != 1:
            continue
        a = np.load(matches[0], allow_pickle=False)
        if a.shape != (30, 26, 50, 6) or not np.isfinite(a).all():
            raise ValueError(f"Expected [30,26,50,6] finite AP tensor: {matches[0]} has {a.shape}")
        row = {"variant": p["gvar_config"]["variant"], "epoch": p["completed_epoch"],
               "split": p["split"], "AP": 100 * float(a.mean()), "AP_mu04": 100 * float(a[..., 1].mean()),
               "AP_mu08": 100 * float(a[..., 3].mean()), "checkpoint_sha256": p["checkpoint_sha256"],
               "frame_fingerprint": p["frame_fingerprint"], "source": str(matches[0])}
        for k in (1, 5, 10, 20, 50):
            row[f"prefix_precision_{k}"] = 100 * float(a[:, :, k - 1, :].mean())
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.root)
    if not rows:
        raise SystemExit("No complete GVAR AP tensors found")
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "gvar_ap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report = ["# GVAR evaluation summary", "", "All values: AP percentage points. Prefix precision is not candidate recall.", "",
              "| Variant | Epoch | Split | AP | AP mu=0.4 | AP mu=0.8 |", "|---|---:|---|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['variant']} | {r['epoch']} | {r['split']} | {r['AP']:.3f} | {r['AP_mu04']:.3f} | {r['AP_mu08']:.3f} |")
    groups = {}
    for r in rows:
        groups.setdefault((r["variant"], r["epoch"], r["checkpoint_sha256"]), []).append(r)
    report += ["", "## Three-split means (complete checkpoints only)"]
    for (v, e, sha), rr in groups.items():
        if len(rr) == 3 and {r["split"] for r in rr} == {"test_seen", "test_similar", "test_novel"}:
            report.append(f"\n{v} epoch {e}: mean AP = {np.mean([r['AP'] for r in rr]):.3f}")
    (args.output / "RESULTS.md").write_text("\n".join(report) + "\n")
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
