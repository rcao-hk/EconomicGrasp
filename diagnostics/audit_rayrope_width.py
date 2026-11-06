#!/usr/bin/env python3
"""Audit reported grasp widths against the EconomicGrasp decode contract.

The last interval [0.10, infinity) INCLUDES clamped 0.10m widths.
Count below/near/strictly-above the cap with explicit tolerance, rather than
calling every sample in the last histogram bin a decoder violation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .analyze_rayrope_candidates import load_candidates
    from .rayrope_action_lattice import count_width_status
except ImportError:  # direct script invocation
    from analyze_rayrope_candidates import load_candidates
    from rayrope_action_lattice import count_width_status


def audit_widths(rows, *, cap_m=0.1, tol_m=1e-6, condition=None):
    """Return per-condition/scene and global clamp diagnostics."""
    selected = [r for r in rows if (condition is None or r["condition"] == condition)]
    groups = defaultdict(list)
    raw_key_variants = ("width_unclamped_m", "raw_width_m", "unclamped_width_m", "width_before_clamp_m")
    for r in selected:
        if math.isfinite(r["width_m"]):
            groups[r["condition"]].append(r)
    if not groups:
        raise ValueError("No finite metric width values found in CSV.")
    out = {}
    for key, data in sorted(groups.items()):
        widths = np.asarray([r["width_m"] for r in data], dtype=float)
        status = count_width_status(widths, cap_m, tol_m)
        raw = []
        for r in data:
            candidate = next((r["raw"][k] for k in raw_key_variants if k in r["raw"] and r["raw"][k] not in ("", None)), None)
            if candidate is None:
                continue
            try:
                v = float(candidate)
                if math.isfinite(v):
                    raw.append(v)
            except ValueError:
                continue
        near = np.abs(widths-cap_m) <= tol_m
        per_scene = {}
        for scene in sorted({r["scene"] for r in data if r["scene"] is not None}):
            subset = [r["width_m"] for r in data if r["scene"] == scene]
            per_scene[str(scene)] = count_width_status(subset, cap_m, tol_m)
        out[key] = {
            **status,
            "at_or_above_nominal_bin_count": int(np.sum(widths >= cap_m)),
            "fraction_near_cap": float(np.mean(near)),
            "fraction_strictly_above_cap": float(np.mean(widths > cap_m+tol_m)),
            "num_raw_unclamped_values": len(raw),
            "raw_unclamped_max_m": float(max(raw)) if raw else None,
            "raw_unclamped_exceeds_cap": int(sum(v > cap_m + tol_m for v in raw)),
            "per_scene": per_scene,
            "warning": (
                "Executed widths exceed the declared clamp by more than tolerance; "
                "check CLI --grasp_max_width, raw/pre-clamp column, and units."
                if status["above_cap"] else
                "No executed-width violation beyond numerical tolerance. "
                "Rows in the [cap,infinity) histogram bin may simply be clamped."
            ),
        }
    return {
        "cap_m": cap_m, "tolerance_m": tol_m,
        "selected_condition": condition, "by_condition": out,
        "interpretation": "The upper histogram bin [cap,infinity) includes exact cap values; do not equate bin membership to illegal overflow.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--columns_json", default=None)
    p.add_argument("--width_unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument("--max_width_m", type=float, default=0.1)
    p.add_argument("--tol_m", type=float, default=1e-6)
    p.add_argument("--condition", default=None, help="Optional: audit native only.")
    args = p.parse_args()
    rows, schema = load_candidates(args.input_csv, args.columns_json, args.width_unit)
    result = audit_widths(rows, cap_m=args.max_width_m,
                          tol_m=args.tol_m, condition=args.condition)
    result["schema"] = schema
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: {"below": x["below_cap"], "near": x["near_cap"], "above": x["above_cap"]}
                      for name, x in result["by_condition"].items()}, indent=2))


if __name__ == "__main__":
    main()
