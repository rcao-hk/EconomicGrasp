#!/usr/bin/env python3
"""Offline analysis primitives for RayRoPE P1/P2 candidate diagnostics.

This module deliberately does not alter model inference. It consumes candidate
records exported by a model-side runner, so ranking/evaluator analysis can be
tested independently and reused across checkpoint variants.

CSV schema (one row per physical candidate and intervention):
  sample_id,candidate_id,intervention,score,quality
Optional: action_id,center_x,center_y,center_z,approach_x,...,width_m,depth_m

Semantics:
- P1 evidence-only: candidate_id identifies the SAME physical grasp in every
  intervention; only reader geometry/evidence changes.
- P1 placement/full-system: candidate_id identifies the actual executed grasp
  after the corresponding geometry intervention. Do not mix this with the
  evidence-only table.
- P2: candidates in a sample share one RGB observation but differ in physical
  action parameters. quality must be evaluated for that exact action.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REQUIRED = ("sample_id", "candidate_id", "intervention", "score", "quality")


def depth_bias(depth: np.ndarray, bias_m: float) -> np.ndarray:
    """Add a metric-Z bias while preserving invalid pixels and input dtype."""
    x = np.asarray(depth)
    out = x.astype(np.float64, copy=True)
    valid = np.isfinite(out) & (out > 0)
    out[valid] += float(bias_m)
    out[valid] = np.maximum(out[valid], 1e-4)
    return out.astype(x.dtype, copy=False)


def depth_affine(depth: np.ndarray, scale: float = 1.0, bias_m: float = 0.0) -> np.ndarray:
    """Apply a declared global metric-depth intervention to valid pixels."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    x = np.asarray(depth)
    out = x.astype(np.float64, copy=True)
    valid = np.isfinite(out) & (out > 0)
    out[valid] = scale * out[valid] + float(bias_m)
    out[valid] = np.maximum(out[valid], 1e-4)
    return out.astype(x.dtype, copy=False)


def depth_local_plane(depth: np.ndarray, bias_m: float, slope_x: float = 0.0,
                      slope_y: float = 0.0) -> np.ndarray:
    """Add a deterministic image-plane depth field; coordinates span [-1,1]."""
    x = np.asarray(depth)
    if x.ndim < 2:
        raise ValueError("depth must have at least two spatial dimensions")
    h, w = x.shape[-2:]
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    field = float(bias_m) + float(slope_x) * xx + float(slope_y) * yy
    out = x.astype(np.float64, copy=True)
    valid = np.isfinite(out) & (out > 0)
    out[valid] += np.broadcast_to(field, out.shape)[valid]
    out[valid] = np.maximum(out[valid], 1e-4)
    return out.astype(x.dtype, copy=False)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        da = np.sign(a[i + 1:] - a[i])
        db = np.sign(b[i + 1:] - b[i])
        concordant += int(np.sum((da * db) > 0))
        discordant += int(np.sum((da * db) < 0))
        ties_a += int(np.sum((da == 0) & (db != 0)))
        ties_b += int(np.sum((db == 0) & (da != 0)))
    denom = math.sqrt((concordant + discordant + ties_a) *
                      (concordant + discordant + ties_b))
    return float((concordant - discordant) / denom) if denom else float("nan")


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [k for k in REQUIRED if k not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        return list(reader)


def _number(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {key!r}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}: {value}")
    return value


def analyze_candidate_table(rows: Sequence[Mapping[str, str]], *, success_threshold: float = 0.0,
                            high_score_quantile: float = 0.95) -> Dict[str, object]:
    """Compute per-intervention ranking, regret, and high-confidence failure.

    quality is a higher-is-better evaluator utility on the exact candidate.
    success is quality > success_threshold. Scores are model ranking scores.
    """
    if not 0 < high_score_quantile < 1:
        raise ValueError("high_score_quantile must be in (0,1)")
    groups: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        score, quality = _number(row, "score"), _number(row, "quality")
        if not row["sample_id"] or not row["candidate_id"] or not row["intervention"]:
            raise ValueError("sample_id/candidate_id/intervention must be non-empty")
        groups[(row["sample_id"], row["intervention"])].append(row)

    per_group = []
    for (sample_id, intervention), items in sorted(groups.items()):
        # Candidate IDs must be unique within one sample/intervention.
        ids = [x["candidate_id"] for x in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate candidate_id in {sample_id}/{intervention}")
        scores = np.asarray([_number(x, "score") for x in items])
        quality = np.asarray([_number(x, "quality") for x in items])
        order = np.argsort(-scores, kind="mergesort")
        selected = int(order[0])
        oracle = float(np.max(quality))
        q = float(np.quantile(scores, high_score_quantile))
        high = scores >= q
        per_group.append({
            "sample_id": sample_id, "intervention": intervention, "n": len(items),
            "spearman": _spearman(scores, quality),
            "kendall_tau_b": _kendall_tau_b(scores, quality),
            "top1_quality": float(quality[selected]),
            "top1_success": float(quality[selected] > success_threshold),
            "oracle_quality": oracle,
            "top1_regret": float(oracle - quality[selected]),
            "high_score_threshold": q,
            "high_conf_failure_rate": float(np.mean(quality[high] <= success_threshold)),
            "top5_quality": float(np.mean(quality[order[:min(5, len(order))]])),
        })

    by_intervention: Dict[str, List[dict]] = defaultdict(list)
    for item in per_group:
        by_intervention[item["intervention"]].append(item)
    summary = {}
    for name, vals in sorted(by_intervention.items()):
        keys = ("spearman", "kendall_tau_b", "top1_quality", "top1_success",
                "top1_regret", "high_conf_failure_rate", "top5_quality")
        summary[name] = {
            "num_samples": len(vals),
            "num_candidates": int(sum(x["n"] for x in vals)),
            **{k: float(np.nanmean([x[k] for x in vals])) for k in keys},
        }

    # P1 paired ranking stability: same candidate IDs must exist in both modes.
    intervention_names = sorted(by_intervention)
    paired = []
    for i, left in enumerate(intervention_names):
        for right in intervention_names[i + 1:]:
            left_map = {x["sample_id"]: x for x in by_intervention[left]}
            right_map = {x["sample_id"]: x for x in by_intervention[right]}
            common = sorted(set(left_map) & set(right_map))
            deltas, rank_corr = [], []
            for sid in common:
                lm = {x["candidate_id"]: _number(x, "score") for x in groups[(sid, left)]}
                rm = {x["candidate_id"]: _number(x, "score") for x in groups[(sid, right)]}
                shared = sorted(set(lm) & set(rm))
                if len(shared) < 2:
                    continue
                deltas.append(float(np.mean([abs(lm[k] - rm[k]) for k in shared])))
                rank_corr.append(_spearman(np.asarray([lm[k] for k in shared]),
                                           np.asarray([rm[k] for k in shared])))
            paired.append({"left": left, "right": right, "paired_samples": len(deltas),
                           "mean_abs_score_change": float(np.mean(deltas)) if deltas else float("nan"),
                           "mean_rank_stability": float(np.nanmean(rank_corr)) if rank_corr else float("nan")})
    return {"summary": summary, "paired_intervention_stability": paired,
            "per_sample": per_group,
            "protocol": {"success_threshold": success_threshold,
                         "high_score_quantile": high_score_quantile,
                         "quality_semantics": "higher_is_better_exact_action_evaluator"}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--success_threshold", type=float, default=0.0)
    p.add_argument("--high_score_quantile", type=float, default=0.95)
    args = p.parse_args()
    result = analyze_candidate_table(_read_rows(args.input_csv),
                                     success_threshold=args.success_threshold,
                                     high_score_quantile=args.high_score_quantile)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    tmp.replace(out)
    print(json.dumps(result["summary"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
