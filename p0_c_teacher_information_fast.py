#!/usr/bin/env python3
"""Fast P0-C1 teacher-information controls for paired-query NPZ caches.

This implementation preserves the scientific quantities of the original P0-C1
analysis while avoiding its largest costs:

* it lazily loads only P0-C1 arrays from each NPZ (never layer features);
* it computes probability-derived statistics in bounded CPU chunks;
* it uses PyTorch CPU kernels with configurable intra-op threads;
* it never materializes R full shuffled [N,T] teacher-probability tensors;
* for binary CDF targets it can analytically marginalize the within-hard-label
  shuffle control (the exact expectation of random within-label shuffling).

Use --shuffle_mode monte_carlo --shuffle_repeats 8 to reproduce the original
finite-shuffle control semantics.  Use --shuffle_mode expected for the same
shuffle control averaged over all possible within-label permutations, with zero
Monte-Carlo noise and much lower runtime.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from pkd_p0.common import atomic_json_dump
from pkd_p0.paired_cache import (
    concatenate_rows,
    discover_payload_files,
    load_aliases,
    load_payload,
    normalize_key,
    standardize_payload,
)


C1_KEYS = (
    "student_logits",
    "teacher_logits",
    "gt_cdf",
    "friction",
    "valid_mask",
    "common_valid",
    "center_z_error",
    "support_iou",
    "teacher_better",
    "scene_id",
    "anno_id",
)


@dataclass(frozen=True)
class Condition:
    condition: str
    lower: float
    upper: float
    kind: str
    arg0: float = 0.0
    arg1: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diagnosis_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mapping_json", default="")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--shuffle_repeats", type=int, default=8)
    p.add_argument("--shuffle_mode", choices=("expected", "monte_carlo"), default="expected")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z_bins_mm", default="0,5,10,20,40,inf")
    p.add_argument("--support_bins", default="0,0.25,0.5,0.75,1.000001")
    p.add_argument("--min_bin_rows", type=int, default=100)
    p.add_argument("--cpu_threads", type=int, default=8)
    p.add_argument("--chunk_rows", type=int, default=250000)
    p.add_argument("--ece_bins", type=int, default=15)
    p.add_argument("--progress_every", type=int, default=50)
    return p.parse_args()


def _resolve_key(keys: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    key_set = set(keys)
    for alias in aliases:
        if alias in key_set:
            return alias
    normalized = {normalize_key(key): key for key in keys}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key is not None:
            return key
    for alias in aliases:
        norm_alias = normalize_key(alias)
        matches = [key for key in keys if normalize_key(key).endswith(norm_alias)]
        if len(matches) == 1:
            return matches[0]
    return None


def _load_c1_payload(path: Path, aliases: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    # NPZ is the normal paired-cache format. np.load is lazy for individual
    # members, so feature arrays are not decompressed/read at all.
    if path.suffix == ".npz":
        payload: Dict[str, Any] = {}
        with np.load(path, allow_pickle=False) as data:
            keys = tuple(str(key) for key in data.files)
            for canonical in C1_KEYS:
                names = aliases.get(canonical, ())
                key = _resolve_key(keys, names)
                if key is not None:
                    payload[key] = np.asarray(data[key])
        return payload

    # Compatibility fallback for older PT/PTH caches. These formats cannot be
    # selectively decompressed with the existing repository reader.
    full = load_payload(path)
    payload = {}
    keys = tuple(full.keys())
    for canonical in C1_KEYS:
        key = _resolve_key(keys, aliases.get(canonical, ()))
        if key is not None:
            payload[key] = full[key]
    return payload


def load_standard_rows_c1(root: str, mapping_json: str, max_files: int, progress_every: int):
    aliases = load_aliases(mapping_json)
    files = discover_payload_files(root)
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No NPZ/PT paired diagnostic files under {root}")

    print(
        f"[P0-C1-FAST][LOAD] files={len(files)} selective_npz=1 "
        f"progress_every={progress_every}",
        flush=True,
    )
    items = []
    errors: List[str] = []
    t0 = time.perf_counter()
    for i, path in enumerate(files, start=1):
        try:
            payload = _load_c1_payload(path, aliases)
            items.append(standardize_payload(payload, aliases=aliases, source=str(path)))
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
        if i == 1 or i % progress_every == 0 or i == len(files):
            print(
                f"[P0-C1-FAST][LOAD] {i}/{len(files)} accepted={len(items)} "
                f"rejected={len(errors)} elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )
    if not items:
        raise RuntimeError("No compatible paired-query cache file. Errors:\n" + "\n".join(errors[:30]))

    print(f"[P0-C1-FAST][CONCAT] accepted={len(items)}", flush=True)
    rows = concatenate_rows(items)
    del items
    gc.collect()
    rows.metadata["num_files_loaded"] = int(len(files) - len(errors))
    rows.metadata["num_files_rejected"] = int(len(errors))
    rows.metadata["rejected_examples"] = errors[:20]
    rows.metadata["c1_selective_npz_loader"] = True
    print(
        f"[P0-C1-FAST][READY] rows={rows.num_rows} "
        f"student_logits={rows.student_logits.dtype}{tuple(rows.student_logits.shape)}",
        flush=True,
    )
    return rows


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


def _is_binary_target(target: np.ndarray, chunk_rows: int) -> bool:
    for start in range(0, len(target), chunk_rows):
        part = target[start : start + chunk_rows]
        if not bool(np.all((part == 0.0) | (part == 1.0))):
            return False
    return True


def _hard_label_codes_u8(target: np.ndarray, chunk_rows: int) -> np.ndarray:
    t = int(target.shape[1])
    if t > 8:
        raise ValueError(f"Fast uint8 hard-label code supports <=8 thresholds; got T={t}")
    powers = (1 << np.arange(t, dtype=np.uint16)).reshape(1, -1)
    out = np.empty(len(target), dtype=np.uint8)
    for start in range(0, len(target), chunk_rows):
        end = min(len(target), start + chunk_rows)
        bits = (target[start:end] >= 0.5).astype(np.uint16, copy=False)
        out[start:end] = np.sum(bits * powers, axis=1, dtype=np.uint16).astype(np.uint8)
    return out


def _condition_mask(cond: Condition, valid: np.ndarray, rows, start: int = 0, end: Optional[int] = None) -> np.ndarray:
    if end is None:
        end = len(valid)
    mask = valid[start:end].copy()
    if cond.kind == "all":
        return mask
    if cond.kind == "teacher_better":
        return mask & (rows.teacher_better[start:end].astype(bool) == bool(int(cond.arg0)))
    if cond.kind == "z":
        values = np.abs(rows.center_z_error[start:end])
        return mask & (values >= cond.arg0) & (values < cond.arg1)
    if cond.kind == "support":
        values = rows.support_iou[start:end]
        return mask & (values >= cond.arg0) & (values < cond.arg1)
    if cond.kind == "scene":
        scene = rows.scene_id[start:end]
        return mask & (scene >= int(cond.arg0)) & (scene < int(cond.arg1))
    raise ValueError(f"Unknown condition kind: {cond.kind}")


def build_conditions(args, rows, valid: np.ndarray) -> List[Condition]:
    conditions = [Condition("all_valid", float("nan"), float("nan"), "all")]
    min_rows = int(args.min_bin_rows)

    if rows.teacher_better is not None:
        for value, label in ((1, "teacher_better"), (0, "teacher_not_better")):
            cond = Condition(label, 0.0, 1.0, "teacher_better", float(value), 0.0)
            if int(_condition_mask(cond, valid, rows).sum()) >= min_rows:
                conditions.append(cond)

    if rows.center_z_error is not None:
        edges = parse_edges(args.z_bins_mm, scale=1e-3)
        for lower, upper in zip(edges[:-1], edges[1:]):
            cond = Condition("center_z_error_m", float(lower), float(upper), "z", float(lower), float(upper))
            if int(_condition_mask(cond, valid, rows).sum()) >= min_rows:
                conditions.append(cond)

    if rows.support_iou is not None:
        edges = parse_edges(args.support_bins)
        for lower, upper in zip(edges[:-1], edges[1:]):
            cond = Condition("support_iou", float(lower), float(upper), "support", float(lower), float(upper))
            if int(_condition_mask(cond, valid, rows).sum()) >= min_rows:
                conditions.append(cond)

    if rows.scene_id is not None:
        for name, lo, hi in (
            ("train_scene", -10**9, 100),
            ("seen_scene", 100, 130),
            ("similar_scene", 130, 160),
            ("novel_scene", 160, 190),
        ):
            cond = Condition(name, float("nan"), float("nan"), "scene", float(lo), float(hi))
            if int(_condition_mask(cond, valid, rows).sum()) >= min_rows:
                conditions.append(cond)

    return conditions


def _ece_update(counts: np.ndarray, sum_p: np.ndarray, sum_y: np.ndarray, probability: np.ndarray, target: np.ndarray, bins: int) -> None:
    if probability.size == 0:
        return
    p = probability.reshape(-1).astype(np.float64, copy=False)
    y = target.reshape(-1).astype(np.float64, copy=False)
    ids = np.floor(p * bins).astype(np.int16)
    np.clip(ids, 0, bins - 1, out=ids)
    counts += np.bincount(ids, minlength=bins).astype(np.int64)
    sum_p += np.bincount(ids, weights=p, minlength=bins)
    sum_y += np.bincount(ids, weights=y, minlength=bins)


def _ece_finalize(counts: np.ndarray, sum_p: np.ndarray, sum_y: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return float("nan")
    valid = counts > 0
    mean_p = np.zeros_like(sum_p)
    mean_y = np.zeros_like(sum_y)
    mean_p[valid] = sum_p[valid] / counts[valid]
    mean_y[valid] = sum_y[valid] / counts[valid]
    return float(np.sum((counts[valid] / float(total)) * np.abs(mean_p[valid] - mean_y[valid])))


def compute_row_statistics(args, rows, valid: np.ndarray, conditions: Sequence[Condition]):
    n = rows.num_rows
    t = int(rows.student_logits.shape[1])
    chunk = max(1, int(args.chunk_rows))
    eps = 1e-7

    # Row-level sufficient statistics. float32 is enough for per-row values;
    # final reductions below use float64 accumulation.
    names = (
        "student_bce",
        "teacher_bce",
        "student_entropy",
        "teacher_entropy",
        "abs_disagreement",
        "student_utility",
        "teacher_utility",
        "target_utility",
        "student_utility_mae",
        "teacher_utility_mae",
    )
    stats = {name: np.empty(n, dtype=np.float32) for name in names}

    ece = {}
    for ci in range(len(conditions)):
        ece[(ci, "student")] = [np.zeros(args.ece_bins, np.int64), np.zeros(args.ece_bins, np.float64), np.zeros(args.ece_bins, np.float64)]
        ece[(ci, "teacher")] = [np.zeros(args.ece_bins, np.int64), np.zeros(args.ece_bins, np.float64), np.zeros(args.ece_bins, np.float64)]

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print(
        f"[P0-C1-FAST][ANALYZE] rows={n} valid={int(valid.sum())} T={t} "
        f"chunk_rows={chunk} cpu_threads={torch.get_num_threads()} conditions={len(conditions)}",
        flush=True,
    )
    t0 = time.perf_counter()
    total_chunks = math.ceil(n / chunk)
    with torch.inference_mode():
        for chunk_i, start in enumerate(range(0, n, chunk), start=1):
            end = min(n, start + chunk)
            s_logit = torch.from_numpy(rows.student_logits[start:end]).float()
            t_logit = torch.from_numpy(rows.teacher_logits[start:end]).float()
            y = torch.from_numpy(rows.gt_cdf[start:end]).float()

            sp = torch.sigmoid(s_logit)
            tp = torch.sigmoid(t_logit)
            sp_clip = sp.clamp(eps, 1.0 - eps)
            tp_clip = tp.clamp(eps, 1.0 - eps)

            sbce = -(y * torch.log(sp_clip) + (1.0 - y) * torch.log1p(-sp_clip)).mean(dim=1)
            tbce = -(y * torch.log(tp_clip) + (1.0 - y) * torch.log1p(-tp_clip)).mean(dim=1)
            sent = -(sp_clip * torch.log(sp_clip) + (1.0 - sp_clip) * torch.log1p(-sp_clip)).mean(dim=1)
            tent = -(tp_clip * torch.log(tp_clip) + (1.0 - tp_clip) * torch.log1p(-tp_clip)).mean(dim=1)
            su = sp.mean(dim=1)
            tu = tp.mean(dim=1)
            yu = y.mean(dim=1)

            stats["student_bce"][start:end] = sbce.numpy()
            stats["teacher_bce"][start:end] = tbce.numpy()
            stats["student_entropy"][start:end] = sent.numpy()
            stats["teacher_entropy"][start:end] = tent.numpy()
            stats["abs_disagreement"][start:end] = torch.abs(tp - sp).mean(dim=1).numpy()
            stats["student_utility"][start:end] = su.numpy()
            stats["teacher_utility"][start:end] = tu.numpy()
            stats["target_utility"][start:end] = yu.numpy()
            stats["student_utility_mae"][start:end] = torch.abs(su - yu).numpy()
            stats["teacher_utility_mae"][start:end] = torch.abs(tu - yu).numpy()

            sp_np = sp.numpy()
            tp_np = tp.numpy()
            y_np = y.numpy()
            for ci, cond in enumerate(conditions):
                cmask = _condition_mask(cond, valid, rows, start, end)
                if not cmask.any():
                    continue
                _ece_update(*ece[(ci, "student")], sp_np[cmask], y_np[cmask], args.ece_bins)
                _ece_update(*ece[(ci, "teacher")], tp_np[cmask], y_np[cmask], args.ece_bins)

            if chunk_i == 1 or chunk_i % 10 == 0 or chunk_i == total_chunks:
                elapsed = time.perf_counter() - t0
                rate = end / max(elapsed, 1e-9)
                eta = (n - end) / max(rate, 1e-9)
                print(
                    f"[P0-C1-FAST][ANALYZE] chunk={chunk_i}/{total_chunks} "
                    f"rows={end}/{n} rate={rate:,.0f} rows/s eta={eta/60:.1f} min",
                    flush=True,
                )

    ece_values = {}
    for ci in range(len(conditions)):
        ece_values[(ci, "student")] = _ece_finalize(*ece[(ci, "student")])
        ece_values[(ci, "teacher")] = _ece_finalize(*ece[(ci, "teacher")])
    print(f"[P0-C1-FAST][ANALYZE-DONE] elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return stats, ece_values


def _binary_entropy(epsilon: float) -> float:
    p = min(max(float(epsilon), 1e-15), 1.0 - 1e-15)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def entropy_matched_epsilon_binary(desired_entropy: float) -> float:
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binary_entropy(mid) < desired_entropy:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_shuffle_bce_by_condition(
    teacher_bce: np.ndarray,
    codes: np.ndarray,
    valid: np.ndarray,
    rows,
    conditions: Sequence[Condition],
) -> np.ndarray:
    unique_codes = np.unique(codes)
    group_mean = {int(code): float(np.mean(teacher_bce[codes == code], dtype=np.float64)) for code in unique_codes}
    outputs = np.empty(len(conditions), dtype=np.float64)
    for ci, cond in enumerate(conditions):
        mask = _condition_mask(cond, valid, rows)
        count = int(mask.sum())
        if count == 0:
            outputs[ci] = float("nan")
            continue
        total = 0.0
        for code in unique_codes:
            c = int(np.sum(mask & (codes == code)))
            if c:
                total += c * group_mean[int(code)]
        outputs[ci] = total / count
    return outputs


def monte_carlo_shuffle_row_bce(
    teacher_bce: np.ndarray,
    codes: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    if repeats <= 0:
        raise ValueError("shuffle_repeats must be positive for monte_carlo mode")
    n = len(teacher_bce)
    # float64 accumulator keeps repeated sums stable; one vector only.
    accum = np.zeros(n, dtype=np.float64)
    unique_codes = np.unique(codes)
    groups = []
    for code in unique_codes:
        idx = np.flatnonzero(codes == code)
        # uint32 halves index-memory when possible and remains valid for numpy indexing.
        if n < np.iinfo(np.uint32).max:
            idx = idx.astype(np.uint32)
        groups.append((int(code), idx))
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    for repeat in range(repeats):
        for _code, idx in groups:
            if len(idx) <= 1:
                accum[idx] += teacher_bce[idx]
            else:
                perm = rng.permutation(idx)
                accum[idx] += teacher_bce[perm]
        print(
            f"[P0-C1-FAST][SHUFFLE] repeat={repeat+1}/{repeats} "
            f"elapsed={time.perf_counter()-t0:.1f}s",
            flush=True,
        )
    return (accum / repeats).astype(np.float32)


def _corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) <= 2:
        return float("nan")
    xv = x[mask].astype(np.float64, copy=False)
    yv = y[mask].astype(np.float64, copy=False)
    xm = float(xv.mean())
    ym = float(yv.mean())
    dx = xv - xm
    dy = yv - ym
    denom = math.sqrt(float(np.dot(dx, dx)) * float(np.dot(dy, dy)))
    if denom <= 0:
        return float("nan")
    return float(np.dot(dx, dy) / denom)


def summarize_condition(
    ci: int,
    cond: Condition,
    valid: np.ndarray,
    rows,
    stats: Mapping[str, np.ndarray],
    ece_values: Mapping[Tuple[int, str], float],
    smooth_bce_scalar: float,
    shuffled_bce_row: Optional[np.ndarray],
    shuffled_bce_condition: Optional[np.ndarray],
) -> Dict[str, Any]:
    mask = _condition_mask(cond, valid, rows)
    count = int(mask.sum())
    if count == 0:
        return {"num_rows": 0}

    def mean(name: str) -> float:
        return float(np.mean(stats[name][mask], dtype=np.float64))

    sb = stats["student_bce"]
    tb = stats["teacher_bce"]
    teacher_bce = mean("teacher_bce")
    if shuffled_bce_condition is not None:
        shuffled = float(shuffled_bce_condition[ci])
    else:
        assert shuffled_bce_row is not None
        shuffled = float(np.mean(shuffled_bce_row[mask], dtype=np.float64))

    return {
        "num_rows": count,
        "student_bce": mean("student_bce"),
        "teacher_bce": teacher_bce,
        "teacher_minus_student_bce": float(np.mean((tb[mask] - sb[mask]).astype(np.float64))),
        "teacher_better_ratio": float(np.mean(tb[mask] < sb[mask])),
        "teacher_entropy": mean("teacher_entropy"),
        "student_entropy": mean("student_entropy"),
        "teacher_student_abs_disagreement": mean("abs_disagreement"),
        "teacher_utility_mae": mean("teacher_utility_mae"),
        "student_utility_mae": mean("student_utility_mae"),
        "entropy_matched_smoothing_bce": float(smooth_bce_scalar),
        "teacher_gain_over_smoothing": float(smooth_bce_scalar - teacher_bce),
        "within_label_shuffled_teacher_bce": shuffled,
        "teacher_gain_over_within_label_shuffle": float(shuffled - teacher_bce),
        "teacher_ece": float(ece_values[(ci, "teacher")]),
        "student_ece": float(ece_values[(ci, "student")]),
        "teacher_student_utility_correlation": _corr(
            stats["teacher_utility"], stats["student_utility"], mask
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[P0-C1-FAST][START] diagnosis_dir={Path(args.diagnosis_dir).expanduser().resolve()} "
        f"output_dir={output_dir} max_files={args.max_files} shuffle_mode={args.shuffle_mode} "
        f"shuffle_repeats={args.shuffle_repeats} cpu_threads={args.cpu_threads} chunk_rows={args.chunk_rows}",
        flush=True,
    )

    rows = load_standard_rows_c1(
        args.diagnosis_dir,
        args.mapping_json,
        int(args.max_files),
        max(1, int(args.progress_every)),
    )
    valid = rows.valid_mask.astype(bool, copy=False)
    if rows.common_valid is not None:
        valid = valid & rows.common_valid.astype(bool, copy=False)
    if not valid.any():
        raise RuntimeError("P0-C1 found no valid/common-valid rows in the paired cache.")

    # The current paired-cache writer converts compact onset bins into explicit
    # binary T=6 CDF targets. Fast analytical controls rely on this invariant.
    binary_target = _is_binary_target(rows.gt_cdf, max(1, int(args.chunk_rows)))
    if not binary_target:
        raise RuntimeError(
            "Fast P0-C1 requires binary CDF targets (0/1). The current paired-query "
            "collector satisfies this contract; use the legacy script for a non-binary cache."
        )

    conditions = build_conditions(args, rows, valid)
    print(
        "[P0-C1-FAST][CONDITIONS] "
        + ", ".join(
            f"{c.condition}[{c.lower},{c.upper}]" if math.isfinite(c.lower) else c.condition
            for c in conditions
        ),
        flush=True,
    )

    codes = _hard_label_codes_u8(rows.gt_cdf, max(1, int(args.chunk_rows)))
    stats, ece_values = compute_row_statistics(args, rows, valid, conditions)

    desired_entropy = float(np.mean(stats["teacher_entropy"][valid], dtype=np.float64))
    epsilon = entropy_matched_epsilon_binary(desired_entropy)
    smooth_bce_scalar = -math.log(max(1.0 - epsilon, 1e-15))
    print(
        f"[P0-C1-FAST][CONTROL] teacher_entropy={desired_entropy:.8f} "
        f"entropy_matched_epsilon={epsilon:.8f} smooth_bce={smooth_bce_scalar:.8f}",
        flush=True,
    )

    shuffled_bce_row = None
    shuffled_bce_condition = None
    if args.shuffle_mode == "expected":
        t0 = time.perf_counter()
        shuffled_bce_condition = expected_shuffle_bce_by_condition(
            stats["teacher_bce"], codes, valid, rows, conditions
        )
        print(
            f"[P0-C1-FAST][SHUFFLE] analytical expectation done in "
            f"{time.perf_counter()-t0:.1f}s",
            flush=True,
        )
    else:
        shuffled_bce_row = monte_carlo_shuffle_row_bce(
            stats["teacher_bce"], codes, int(args.shuffle_repeats), int(args.seed)
        )

    condition_rows: List[Dict[str, Any]] = []
    for ci, cond in enumerate(conditions):
        summary = summarize_condition(
            ci,
            cond,
            valid,
            rows,
            stats,
            ece_values,
            smooth_bce_scalar,
            shuffled_bce_row,
            shuffled_bce_condition,
        )
        condition_rows.append({
            "condition": cond.condition,
            "lower": cond.lower,
            "upper": cond.upper,
            **summary,
        })
        print(
            f"[P0-C1-FAST][SUMMARY] {cond.condition} rows={summary.get('num_rows', 0)} "
            f"teacher_bce={summary.get('teacher_bce', float('nan')):.6f} "
            f"gain_shuffle={summary.get('teacher_gain_over_within_label_shuffle', float('nan')):.6f}",
            flush=True,
        )

    overall = condition_rows[0].copy()
    overall.pop("condition", None)
    overall.pop("lower", None)
    overall.pop("upper", None)

    write_csv(output_dir / "condition_summary.csv", condition_rows)
    payload = {
        "experiment": "P0-C teacher information controls",
        "implementation": "p0_c_teacher_information_fast_v1_2",
        "diagnosis_dir": str(Path(args.diagnosis_dir).expanduser().resolve()),
        "metadata": rows.metadata,
        "num_rows_total": rows.num_rows,
        "num_rows_valid": int(valid.sum()),
        "entropy_matched_smoothing_epsilon": epsilon,
        "shuffle_mode": str(args.shuffle_mode),
        "shuffle_repeats": int(args.shuffle_repeats) if args.shuffle_mode == "monte_carlo" else 0,
        "shuffle_control_note": (
            "expected mode analytically marginalizes the same random within-hard-label permutation control"
            if args.shuffle_mode == "expected"
            else "monte_carlo mode uses finite random within-hard-label permutations"
        ),
        "cpu_threads": int(args.cpu_threads),
        "chunk_rows": int(args.chunk_rows),
        "overall": overall,
        "interpretation_keys": {
            "teacher_gain_over_smoothing": "positive means query-specific teacher probabilities beat entropy-matched label smoothing",
            "teacher_gain_over_within_label_shuffle": "positive means teacher-query correspondence adds information beyond hard-label-conditioned teacher statistics",
            "teacher_minus_student_bce": "negative means teacher fits GT CDF better than student on the same rows",
        },
    }
    atomic_json_dump(payload, output_dir / "summary.json")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"[DONE] P0-C1 FAST outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
