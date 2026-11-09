#!/usr/bin/env python3
"""P0-C1: test whether teacher CDF carries query-specific privileged information.

The script consumes per-query arrays exported by the repository's existing
privileged-KD diagnosis. It compares the real teacher against two controls:

1. entropy-matched ordinary label smoothing;
2. within-hard-label shuffled teacher probabilities.

Both controls preserve non-privileged statistics while removing query-specific
teacher information. Results are reported globally and by correspondence bins.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pkd_p0.common import atomic_json_dump
from pkd_p0.paired_cache import load_standard_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diagnosis_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mapping_json", default="")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--shuffle_repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z_bins_mm", default="0,5,10,20,40,inf")
    p.add_argument("--support_bins", default="0,0.25,0.5,0.75,1.000001")
    p.add_argument("--min_bin_rows", type=int, default=100)
    return p.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def bce(probability: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(target, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def entropy(probability: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def smoothed_target(target: np.ndarray, epsilon: float) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    return y * (1.0 - epsilon) + (1.0 - y) * epsilon


def entropy_matched_epsilon(target: np.ndarray, teacher_probability: np.ndarray) -> float:
    desired = float(entropy(teacher_probability).mean())
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = (lo + hi) * 0.5
        current = float(entropy(smoothed_target(target, mid)).mean())
        if current < desired:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def hard_label_codes(target: np.ndarray) -> np.ndarray:
    bits = (np.asarray(target) >= 0.5).astype(np.int64)
    powers = (1 << np.arange(bits.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum(bits * powers, axis=1)


def within_label_shuffle(probability: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    codes = hard_label_codes(target)
    shuffled = np.asarray(probability).copy()
    for code in np.unique(codes):
        indices = np.flatnonzero(codes == code)
        if len(indices) > 1:
            shuffled[indices] = probability[rng.permutation(indices)]
    return shuffled


def ece(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(p)
    value = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= boundaries[i]) & (p <= boundaries[i + 1])
        else:
            mask = (p >= boundaries[i]) & (p < boundaries[i + 1])
        if not mask.any():
            continue
        value += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def summarize(
    student_probability: np.ndarray,
    teacher_probability: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    smoothing_probability: np.ndarray,
    shuffled_probabilities: Sequence[np.ndarray],
) -> Dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(target):
        raise ValueError("row mask shape mismatch")
    count = int(mask.sum())
    if count == 0:
        return {"num_rows": 0}
    sp = student_probability[mask]
    tp = teacher_probability[mask]
    y = target[mask]
    smooth = smoothing_probability[mask]
    teacher_bce_row = bce(tp, y).mean(axis=1)
    student_bce_row = bce(sp, y).mean(axis=1)
    smooth_bce_row = bce(smooth, y).mean(axis=1)
    shuffled_bces = [bce(probability[mask], y).mean(axis=1) for probability in shuffled_probabilities]
    shuffled_mean = np.mean(np.stack(shuffled_bces, axis=0), axis=0)
    teacher_utility = tp.mean(axis=1)
    student_utility = sp.mean(axis=1)
    target_utility = y.mean(axis=1)
    return {
        "num_rows": count,
        "student_bce": float(student_bce_row.mean()),
        "teacher_bce": float(teacher_bce_row.mean()),
        "teacher_minus_student_bce": float((teacher_bce_row - student_bce_row).mean()),
        "teacher_better_ratio": float((teacher_bce_row < student_bce_row).mean()),
        "teacher_entropy": float(entropy(tp).mean()),
        "student_entropy": float(entropy(sp).mean()),
        "teacher_student_abs_disagreement": float(np.abs(tp - sp).mean()),
        "teacher_utility_mae": float(np.abs(teacher_utility - target_utility).mean()),
        "student_utility_mae": float(np.abs(student_utility - target_utility).mean()),
        "entropy_matched_smoothing_bce": float(smooth_bce_row.mean()),
        "teacher_gain_over_smoothing": float((smooth_bce_row - teacher_bce_row).mean()),
        "within_label_shuffled_teacher_bce": float(shuffled_mean.mean()),
        "teacher_gain_over_within_label_shuffle": float((shuffled_mean - teacher_bce_row).mean()),
        "teacher_ece": ece(tp, y),
        "student_ece": ece(sp, y),
        "teacher_student_utility_correlation": float(np.corrcoef(teacher_utility, student_utility)[0, 1]) if count > 2 else float("nan"),
    }


def parse_edges(text: str, *, scale: float = 1.0) -> np.ndarray:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        values.append(float("inf") if token in {"inf", "+inf", "infinity"} else float(token) * scale)
    edges = np.asarray(values, dtype=np.float64)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"Invalid bin edges {text!r}")
    return edges


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bin_rows(
    name: str,
    values: np.ndarray,
    edges: np.ndarray,
    base_mask: np.ndarray,
    summarizer,
    min_rows: int,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = base_mask & (values >= lower) & (values < upper)
        if int(mask.sum()) < min_rows:
            continue
        row = {
            "condition": name,
            "lower": float(lower),
            "upper": float(upper),
            **summarizer(mask),
        }
        result.append(row)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[P0-C1][START] diagnosis_dir={Path(args.diagnosis_dir).expanduser().resolve()} "
        f"output_dir={output_dir} max_files={int(args.max_files)} "
        f"shuffle_repeats={int(args.shuffle_repeats)}",
        flush=True,
    )
    rows = load_standard_rows(args.diagnosis_dir, args.mapping_json, int(args.max_files))
    valid = rows.valid_mask.astype(bool)
    if rows.common_valid is not None:
        valid &= rows.common_valid.astype(bool)
    student_probability = sigmoid(rows.student_logits)
    teacher_probability = sigmoid(rows.teacher_logits)
    target = np.asarray(rows.gt_cdf, dtype=np.float64)

    print(
        f"[P0-C1][ANALYZE] total_rows={rows.num_rows} valid_rows={int(valid.sum())}",
        flush=True,
    )
    if not valid.any():
        raise RuntimeError("P0-C1 found no valid/common-valid rows in the paired cache.")
    epsilon = entropy_matched_epsilon(target[valid], teacher_probability[valid])
    print(f"[P0-C1][CONTROL] entropy_matched_epsilon={epsilon:.8f}", flush=True)
    smooth = smoothed_target(target, epsilon)
    rng = np.random.default_rng(int(args.seed))
    shuffled = [within_label_shuffle(teacher_probability, target, rng) for _ in range(int(args.shuffle_repeats))]

    def summarize_mask(mask: np.ndarray) -> Dict[str, Any]:
        return summarize(
            student_probability,
            teacher_probability,
            target,
            mask,
            smoothing_probability=smooth,
            shuffled_probabilities=shuffled,
        )

    overall = summarize_mask(valid)
    conditions: List[Dict[str, Any]] = [{"condition": "all_valid", "lower": float("nan"), "upper": float("nan"), **overall}]

    if rows.teacher_better is not None:
        for value, label in ((True, "teacher_better"), (False, "teacher_not_better")):
            mask = valid & (rows.teacher_better.astype(bool) == value)
            if int(mask.sum()) >= int(args.min_bin_rows):
                conditions.append({"condition": label, "lower": 0.0, "upper": 1.0, **summarize_mask(mask)})
    if rows.center_z_error is not None:
        z_edges = parse_edges(args.z_bins_mm, scale=1e-3)
        conditions.extend(bin_rows("center_z_error_m", np.abs(rows.center_z_error), z_edges, valid, summarize_mask, int(args.min_bin_rows)))
    if rows.support_iou is not None:
        support_edges = parse_edges(args.support_bins)
        conditions.extend(bin_rows("support_iou", rows.support_iou, support_edges, valid, summarize_mask, int(args.min_bin_rows)))
    if rows.scene_id is not None:
        for domain, predicate in (
            ("train_scene", rows.scene_id < 100),
            ("seen_scene", (rows.scene_id >= 100) & (rows.scene_id < 130)),
            ("similar_scene", (rows.scene_id >= 130) & (rows.scene_id < 160)),
            ("novel_scene", (rows.scene_id >= 160) & (rows.scene_id < 190)),
        ):
            mask = valid & predicate
            if int(mask.sum()) >= int(args.min_bin_rows):
                conditions.append({"condition": domain, "lower": float("nan"), "upper": float("nan"), **summarize_mask(mask)})

    print(f"[P0-C1][WRITE] conditions={len(conditions)} output_dir={output_dir}", flush=True)
    write_csv(output_dir / "condition_summary.csv", conditions)
    payload = {
        "experiment": "P0-C teacher information controls",
        "diagnosis_dir": str(Path(args.diagnosis_dir).expanduser().resolve()),
        "metadata": rows.metadata,
        "num_rows_total": rows.num_rows,
        "num_rows_valid": int(valid.sum()),
        "entropy_matched_smoothing_epsilon": epsilon,
        "shuffle_repeats": int(args.shuffle_repeats),
        "overall": overall,
        "interpretation_keys": {
            "teacher_gain_over_smoothing": "positive means query-specific teacher probabilities beat entropy-matched label smoothing",
            "teacher_gain_over_within_label_shuffle": "positive means teacher-query correspondence adds information beyond hard-label-conditioned teacher statistics",
            "teacher_minus_student_bce": "negative means teacher fits GT CDF better than student on the same rows",
        },
    }
    atomic_json_dump(payload, output_dir / "summary.json")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"[DONE] P0-C1 outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
