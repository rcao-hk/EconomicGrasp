#!/usr/bin/env python3
"""Summarize measured CVA depth trajectories without inferring a mechanism."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Incomplete or invalid JSON at {path}:{number}") from exc
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def avg(values):
    values = [v for v in values if finite(v)]
    return mean(values) if values else None


def flatten_metrics(record):
    """Accept one per-image record or an explicitly nested metric list."""
    if "regions" in record:
        return [record]
    for key in ("depth_metrics", "metrics", "images", "depth"):
        value = record.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and "regions" in value:
            return [value]
    return []


def summarize(root, output, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    sources = []
    for path in sorted(root.rglob("fixed_probe.jsonl")):
        sources.append(str(path.resolve()))
        for record in read_jsonl(path):
            step = record.get("step", record.get("optimizer_step", record.get("update", 0)))
            arm = record.get("arm", path.parent.name)
            split = record.get("split", "unspecified")
            mode = record.get("mode", record.get("module_mode", record.get("measurement_mode", "unspecified")))
            grouped[(arm, split, mode, step)].extend(flatten_metrics(record))
    rows = []
    for (arm, split, mode, step), images in sorted(grouped.items()):
        valid = [im.get("regions", {}).get("valid", {}) for im in images]
        local = [im.get("local", {}) for im in images]
        foreground = [im.get("regions", {}).get("foreground", {}) for im in images]
        eligible = [x for x in foreground if finite(x.get("gt_std")) and x["gt_std"] > .005]
        flat = [x for x in eligible if finite(x.get("std_ratio")) and x["std_ratio"] < .1]
        rows.append({"arm": arm, "split": split, "mode": mode, "step": step,
                     "image_count": len(images), "eligible_image_count": len(eligible),
                     "mae_m": avg(x.get("mae") for x in valid),
                     "bias_m": avg(x.get("bias") for x in valid),
                     "mean_image_std_ratio": avg(x.get("std_ratio") for x in valid),
                     "below_min_fraction": avg(x.get("below_min_fraction") for x in valid),
                     "flat_image_fraction": len(flat) / len(eligible) if eligible else None,
                     "local_contrast_ratio": avg(x.get("contrast_ratio") for x in local),
                     "local_slope": avg(x.get("slope") for x in local),
                     "local_difference_mae_m": avg(x.get("difference_mae") for x in local),
                     "sigmoid_derivative_mean": avg(im.get("raw", {}).get("sigmoid_derivative_mean") for im in images)})
    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("arm,split,mode,step,image_count\n")

    measured = defaultdict(list)
    for row in rows:
        measured[(row["arm"], row["split"], row["mode"])].append(row)
    report = ["# CVA metric-depth dynamics: measured results", "",
              "These tables summarize fixed-frame depth measurements. They do not establish "
              "a necessary/sufficient route or a causal mechanism.", ""]
    if not rows:
        report.append("No fixed-probe measurements are available; no training conclusion is supported.")
    else:
        report += ["| Arm | Split / mode | Last update | MAE (m) | Mean image std ratio | Flat image fraction |",
                   "| --- | --- | ---: | ---: | ---: | ---: |"]
        fmt = lambda x: f"{x:.6g}" if finite(x) else "missing"
        for (arm, split, mode), trajectory in sorted(measured.items()):
            last = trajectory[-1]
            report.append(f"| {arm} | {split} / {mode} | {last['step']} | {fmt(last['mae_m'])} | "
                          f"{fmt(last['mean_image_std_ratio'])} | {fmt(last['flat_image_fraction'])} |")
    contracts = []
    for path in sorted(root.rglob("contract.json")):
        contract = json.loads(path.read_text(encoding="utf-8"))
        contracts.append({"path": str(path.resolve()), "contract": contract})
    report += ["", "## Scope and unresolved questions", "",
               "- Reproduction: interpret only the measured initialization, route, data protocol and update budget.",
               "- Necessary/sufficient path: not established by these summary statistics.",
               "- Mechanism: requires a matched failure plus pre-failure actual-update interventions.",
               "- Exclusions: no absent mechanism is ruled out by a short stable trajectory.",
               "- Validation: test_seen is user-designated validation; its original split identity remains in the contract.",
               "- Next step: inspect audit connectivity and paired curves before extending the update budget or opening P3.",
               "", "Source contracts and measured files are indexed in `summary_sources.json`."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "summary_sources.json").write_text(json.dumps({"fixed_probe_files": sources,
        "contracts": contracts}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if plots and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        metrics = [("mae_m", "Depth MAE (m)"), ("bias_m", "Depth bias (m)"),
                   ("mean_image_std_ratio", "Mean within-image std ratio"),
                   ("local_contrast_ratio", "Local contrast ratio"),
                   ("below_min_fraction", "Depth below minimum"),
                   ("sigmoid_derivative_mean", "Mean sigmoid derivative")]
        for ax, (metric, title) in zip(axes.flat, metrics):
            for (arm, split, mode), trajectory in sorted(measured.items()):
                points = [(r["step"], r[metric]) for r in trajectory if finite(r[metric])]
                if points:
                    ax.plot(*zip(*points), marker=".", label=f"{arm}/{split}/{mode}")
            ax.set(title=title, xlabel="Successful optimizer updates")
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        figure.savefig(figures / "depth_dynamics.png", dpi=160)
        plt.close(figure)
    return {"rows": len(rows), "summary": str(csv_path), "report": str(output / "report.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Run diagnostics root")
    parser.add_argument("--output", required=True, type=Path, help="Summary directory")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output, not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
