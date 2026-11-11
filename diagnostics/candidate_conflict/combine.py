"""Combine the three full GraspNet splits with 90-scene paired bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import SCHEMA, scene_cluster_bootstrap, write_json


VARIANTS = (
    "native", "quality_nms_native_rank", "quality_rank_native_nms",
    "quality_nms_quality_rank", "component_set_native_rank",
    "component_set_quality_rank", "per_mu_success_oracle",
)
EXPECTED = ("test_seen", "test_similar", "test_novel")


def _parse(values):
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--split-dir must use SPLIT=PATH")
        name, path = item.split("=", 1)
        if name in out or name not in EXPECTED:
            raise ValueError(f"Invalid/duplicate split mapping: {name}")
        out[name] = Path(path)
    if set(out) != set(EXPECTED):
        raise ValueError(f"Require exactly {EXPECTED}")
    return out


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    split_dirs = _parse(a.split_dir)
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Combined output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    summaries = {}
    rows_by_split = {}
    methods = None
    for split in EXPECTED:
        root = split_dirs[split]
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("schema") != SCHEMA or summary.get("split") != split or not summary.get("complete"):
            raise ValueError(f"{split}: wrong/incomplete analysis summary")
        current = set(summary["methods"])
        if methods is None:
            methods = current
        elif current != methods:
            raise ValueError("Method sets differ across splits")
        failed = [m for m, r in summary["methods"].items()
                  if not r["official_ap_parity"]["passed"]]
        if failed:
            raise RuntimeError(f"{split}: official AP parity failed for {failed}")
        rows = _read_csv(root / "per_scene.csv")
        for method in methods:
            mr = [r for r in rows if r["method"] == method]
            if len(mr) != 30:
                raise ValueError(f"{split}/{method}: expected 30 scene rows, got {len(mr)}")
        summaries[split] = summary
        rows_by_split[split] = rows

    combined_scene_rows = []
    method_results = {}
    for method in sorted(methods):
        method_rows = []
        for split in EXPECTED:
            method_rows.extend(r for r in rows_by_split[split] if r["method"] == method)
        scene_ids = [int(r["scene"]) for r in method_rows]
        if len(scene_ids) != 90 or len(set(scene_ids)) != 90:
            raise ValueError(f"{method}: expected 90 unique scene IDs across all splits")
        values = {
            variant: {int(r["scene"]): float(r[f"{variant}_ap_percent"]) / 100.0
                      for r in method_rows}
            for variant in VARIANTS
        }
        mean_ap = {
            variant: float(np.mean(list(values[variant].values())) * 100.0)
            for variant in VARIANTS
        }
        boot = {}
        for variant in VARIANTS:
            if variant == "native":
                continue
            delta = {
                scene: (values[variant][scene] - values["native"][scene]) * 100.0
                for scene in values["native"]
            }
            boot[variant] = scene_cluster_bootstrap(delta, a.bootstrap, a.seed)

        graph_totals = {
            key: int(sum(summaries[split]["methods"][method]["graph_totals"][key]
                         for split in EXPECTED))
            for key in ("nodes", "edges", "components", "native_survivors")
        }
        selection_keys = (
            "suppressed", "suppressed_q_better", "component_exact_nodes",
            "component_total_nodes", "component_exact_components",
            "component_fallback_components",
        )
        selection = {
            key: int(sum(summaries[split]["methods"][method]["selection_totals"][key]
                         for split in EXPECTED))
            for key in selection_keys
        }
        selection["fraction_suppressed_q_better"] = (
            selection["suppressed_q_better"] / selection["suppressed"]
            if selection["suppressed"] else 0.0
        )
        selection["component_exact_node_coverage"] = (
            selection["component_exact_nodes"] / selection["component_total_nodes"]
            if selection["component_total_nodes"] else 1.0
        )
        selection["component_oracle_globally_exact"] = selection["component_fallback_components"] == 0
        parity_errors = [
            summaries[split]["methods"][method]["official_ap_parity"]["max_abs"]
            for split in EXPECTED
        ]
        method_results[method] = {
            "num_scenes": 90,
            "mean_ap_percent": mean_ap,
            "scene_bootstrap": boot,
            "graph_totals": graph_totals,
            "selection_totals": selection,
            "official_ap_parity_max_abs": max(parity_errors),
            "ap_by_mu_percent": {
                variant: np.mean([
                    summaries[split]["methods"][method]["ap_by_mu_percent"][variant]
                    for split in EXPECTED
                ], axis=0).tolist()
                for variant in VARIANTS
            },
        }
        for row in method_rows:
            combined_scene_rows.append(row)

    combined = {
        "schema": SCHEMA,
        "status": "complete",
        "splits": list(EXPECTED),
        "num_scenes": 90,
        "bootstrap_unit": "scene",
        "bootstrap_replicates": a.bootstrap,
        "seed": a.seed,
        "methods": method_results,
        "source_summary_sha256": {
            split: __import__("hashlib").sha256(
                (split_dirs[split] / "summary.json").read_bytes()
            ).hexdigest() for split in EXPECTED
        },
    }
    write_json(out / "summary_all_splits.json", combined)
    with (out / "per_scene_all_splits.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(combined_scene_rows[0]) if combined_scene_rows else []
        writer = csv.DictWriter(f, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(combined_scene_rows)

    lines = [
        "# Candidate Conflict Graph + Quality-aware Selection Oracle — All 90 Scenes",
        "",
        "All three full splits passed native official AP replay parity. Scene is the bootstrap unit.",
        "",
        "| Method | Native AP | Q-NMS/native rank | Q-rank/native NMS | Q-NMS+Q-rank | Component set/native rank | Per-μ oracle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in method_results.items():
        m = result["mean_ap_percent"]
        lines.append(
            f"| {method} | {m['native']:.4f} | {m['quality_nms_native_rank']:.4f} | "
            f"{m['quality_rank_native_nms']:.4f} | {m['quality_nms_quality_rank']:.4f} | "
            f"{m['component_set_native_rank']:.4f} | {m['per_mu_success_oracle']:.4f} |"
        )
    lines += ["", "## Paired 90-scene oracle gains", "",
              "| Method | Oracle | Δ AP (pp) | 95% CI (pp) |", "|---|---|---:|---:|"]
    for method, result in method_results.items():
        for variant, boot in result["scene_bootstrap"].items():
            lines.append(
                f"| {method} | {variant} | {boot['mean']:.4f} | "
                f"[{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}] |"
            )
    lines += ["", "## Candidate graph and suppression totals", "",
              "| Method | Nodes | NMS edges | Components | Native survivors | Suppressed Q-better | Exact component coverage |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method, result in method_results.items():
        g = result["graph_totals"]
        sel = result["selection_totals"]
        lines.append(
            f"| {method} | {g['nodes']} | {g['edges']} | {g['components']} | "
            f"{g['native_survivors']} | {sel['suppressed_q_better']} | "
            f"{sel['component_exact_node_coverage']:.3f} |"
        )
    lines += ["", "## Interpretation limits", "",
              "- Q-based policies use exact labels and are not deployable models.",
              "- Per-μ oracle uses six different threshold-specific policies.",
              "- Component oracle is exact only if fallback component count is zero.",
              "- Oracle headroom is conditional on the current candidate pool.",
              "- Scene bootstrap does not estimate training-seed variability.", ""]
    (out / "RESULTS_CN_ALL_SPLITS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "num_scenes": 90,
        "methods": {m: r["mean_ap_percent"] for m, r in method_results.items()},
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
