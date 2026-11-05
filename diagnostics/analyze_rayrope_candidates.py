#!/usr/bin/env python3
"""Analyze RayRoPE P1/P2 candidate-level GraspNet/Dex-Net diagnostics.

The input is the lossless candidates.csv exported by the P1/P2 inference run.
This script does not rerun inference or the CAD evaluator. It derives failure
classes, official six-friction-threshold outcomes, width-conditioned quality,
matched width sweeps, intervention pairing, and scene-clustered bootstrap CIs.

The parser accepts common column aliases and an optional JSON column map. If
your server CSV uses different headers, pass --columns_json rather than
silently editing the source.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MUS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
ALIASES = {
    "sample": ("sample_id", "frame_id", "sample", "frame_key", "image_id", "sample_key"),
    "scene": ("scene_id", "scene", "scene_idx", "scene_index", "sceneid"),
    "anno": ("anno_id", "annotation_id", "frame_idx", "image_idx", "frame_index"),
    "split": ("split", "object_split", "test_split", "eval_split"),
    "candidate": ("candidate_id", "action_id", "grasp_id", "candidate_idx", "action_idx", "id"),
    "parent": ("parent_id", "parent_candidate_id", "base_action_id", "geometry_id", "action_family_id", "source_candidate_id"),
    "condition": ("intervention", "condition", "mode", "variant", "setting"),
    "score": ("score", "model_score", "pred_score", "confidence", "grasp_score", "score_pred"),
    "quality": ("quality", "exact_quality", "quality_ratio", "success_ratio", "mu_success_ratio"),
    "friction": ("friction", "friction_score", "friction_threshold", "max_mu", "dexnet_score", "fc_score"),
    "collision_or_empty": ("collision_or_empty", "collision_empty", "invalid_collision", "collision_flag", "collision"),
    "pure_collision": ("pure_collision", "collision_only", "is_collision"),
    "empty": ("empty", "empty_grasp", "is_empty", "empty_flag", "no_contact"),
    "width": ("width_m", "width", "predicted_width", "grasp_width", "width_pred", "executed_width"),
    "width_mode": ("width_mode", "width_source", "width_type", "is_predicted_width", "width_is_predicted"),
    "center_x": ("center_x", "x", "tx", "translation_x", "grasp_x"),
    "center_y": ("center_y", "y", "ty", "translation_y", "grasp_y"),
    "center_z": ("center_z", "z", "tz", "translation_z", "grasp_z"),
    "u": ("center_u", "u", "pixel_u", "image_u", "center_px_x"),
    "v": ("center_v", "v", "pixel_v", "image_v", "center_px_y"),
    "insertion": ("insertion_depth_m", "insertion_depth", "depth_m", "grasp_depth", "depth"),
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    try:
        x = float(s)
        if math.isfinite(x):
            return bool(x)
    except ValueError:
        pass
    return None

def parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default

def resolve_columns(fields: Sequence[str], overrides: Mapping[str, str]) -> Dict[str, Optional[str]]:
    by_norm = {norm(f): f for f in fields}
    out: Dict[str, Optional[str]] = {}
    for key, aliases in ALIASES.items():
        requested = overrides.get(key)
        if requested:
            if requested not in fields:
                raise ValueError(f"Column map for {key!r} points to missing column {requested!r}. Available: {list(fields)}")
            out[key] = requested
            continue
        out[key] = next((by_norm[norm(a)] for a in aliases if norm(a) in by_norm), None)
    return out

def discover_mu_columns(fields: Sequence[str]) -> Dict[float, Optional[str]]:
    result: Dict[float, Optional[str]] = {}
    for mu in MUS:
        token = f"{int(round(mu * 10)):02d}"
        matches = []
        for field in fields:
            n = norm(field)
            if n in ("mu", "muthreshold", "friction", "frictionscore"):
                continue
            has_mu_token = n.endswith(token) or re.search(rf"(?:mu|success|pass|fc|forceclosure).*{token}(?:success|pass|flag)?$", n) is not None
            has_success_marker = any(k in n for k in ("success", "pass", "closure", "forceclosure", "fc"))
            if has_mu_token and has_success_marker:
                matches.append(field)
        result[mu] = matches[0] if len(matches) == 1 else None
    return result

def scene_number(value: Any, sample: str = "") -> Optional[int]:
    if value is not None and str(value).strip():
        s = str(value).strip()
        m = re.search(r"(\d{1,5})", s)
        if m:
            return int(m.group(1))
    m = re.search(r"scene[_-]?(\d{3,5})", sample, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def infer_split(raw: Any, scene: Optional[int]) -> str:
    if raw is not None and str(raw).strip():
        s = str(raw).strip().lower().replace("test_", "").replace("test-", "")
        if "seen" in s:
            return "seen"
        if "similar" in s:
            return "similar"
        if "novel" in s:
            return "novel"
        return s
    if scene is None:
        return "unknown"
    if 100 <= scene <= 129:
        return "seen"
    if 130 <= scene <= 159:
        return "similar"
    if 160 <= scene <= 189:
        return "novel"
    return "unknown"

def _mu_flags(raw: Mapping[str, Any], mu_cols: Mapping[float, Optional[str]],
              friction: float) -> Dict[float, Optional[bool]]:
    flags: Dict[float, Optional[bool]] = {}
    all_explicit = all(mu_cols.get(mu) for mu in MUS)
    for mu in MUS:
        col = mu_cols.get(mu)
        if all_explicit and col:
            flags[mu] = parse_bool(raw.get(col))
        elif math.isfinite(friction):
            # Avoid mixing a partial exported bitset with derived bits. If the
            # six-bit package is incomplete, derive the complete official
            # threshold vector from the evaluator's scalar friction score.
            flags[mu] = bool(friction > 0.0 and friction <= mu + 1e-8)
        elif col:
            flags[mu] = parse_bool(raw.get(col))
        else:
            flags[mu] = None
    return flags

def failure_class(collision_or_empty: Optional[bool], pure_collision: Optional[bool],
                  empty: Optional[bool], quality: float, flags: Mapping[float, Optional[bool]]) -> str:
    if empty is True:
        return "empty_grasp"
    if pure_collision is True:
        return "collision"
    if collision_or_empty is True:
        return "collision"
    known_flags = [v for v in flags.values() if v is not None]
    if known_flags and not any(known_flags):
        return "force_closure_failure"
    if math.isfinite(quality) and quality <= 0.0:
        return "force_closure_failure"
    if any(v is True for v in known_flags) or (math.isfinite(quality) and quality > 0.0):
        return "success"
    return "unknown"

def load_candidates(path: str, columns_json: Optional[str], width_unit: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        raw_rows = list(reader)
    if not fields or not raw_rows:
        raise ValueError(f"Candidate CSV is empty: {path}")
    overrides = json.loads(Path(columns_json).read_text(encoding="utf-8")) if columns_json else {}
    cols = resolve_columns(fields, overrides)
    mu_cols = discover_mu_columns(fields)
    if cols["score"] is None:
        raise ValueError(f"Cannot find score column. Available headers: {fields}")
    if cols["sample"] is None and (cols["scene"] is None or cols["anno"] is None):
        raise ValueError("Need sample_id/frame_id or both scene_id and anno_id columns.")
    if cols["sample"] is None:
        sample_col = None
    else:
        sample_col = cols["sample"]

    widths = [parse_float(r.get(cols["width"])) for r in raw_rows] if cols["width"] else []
    valid_widths = [x for x in widths if math.isfinite(x)]
    resolved_unit = width_unit
    if width_unit == "auto":
        resolved_unit = "mm" if valid_widths and float(np.median(valid_widths)) > 1.0 else "m"
    width_scale = 0.001 if resolved_unit == "mm" else 1.0

    rows: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_rows):
        scene = scene_number(raw.get(cols["scene"]) if cols["scene"] else None,
                             raw.get(sample_col, "") if sample_col else "")
        if sample_col:
            sample = str(raw.get(sample_col, "")).strip()
        else:
            sample = f"scene{scene:04d}_frame{int(parse_float(raw.get(cols['anno']), i)):04d}"
        if not sample:
            raise ValueError(f"Empty sample identifier at CSV row {i + 2}")
        candidate = str(raw.get(cols["candidate"], "")).strip() if cols["candidate"] else str(i)
        condition = str(raw.get(cols["condition"], "native")).strip() if cols["condition"] else "native"
        score = parse_float(raw.get(cols["score"]))
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score at CSV row {i + 2}")
        friction = parse_float(raw.get(cols["friction"])) if cols["friction"] else float("nan")
        flags = _mu_flags(raw, mu_cols, friction)
        known = [v for v in flags.values() if v is not None]
        quality = float(np.mean(known)) if len(known) == len(MUS) else (
            parse_float(raw.get(cols["quality"])) if cols["quality"] else float("nan")
        )
        # A scalar quality field is a fallback only. If the six official
        # threshold bits exist, their mean is the canonical 0..1 quality.
        if not known and math.isfinite(friction):
            quality = float(np.mean([friction > 0.0 and friction <= mu + 1e-8 for mu in MUS]))
        collision_or_empty = parse_bool(raw.get(cols["collision_or_empty"])) if cols["collision_or_empty"] else None
        pure_collision = parse_bool(raw.get(cols["pure_collision"])) if cols["pure_collision"] else None
        empty = parse_bool(raw.get(cols["empty"])) if cols["empty"] else None
        if pure_collision is None and collision_or_empty is True and empty is not True:
            pure_collision = True
        if collision_or_empty is None and pure_collision is not None:
            collision_or_empty = pure_collision
        width = parse_float(raw.get(cols["width"])) * width_scale if cols["width"] else float("nan")
        row = {
            "raw": raw, "row_index": i, "sample": sample, "candidate": candidate,
            "condition": condition or "native", "scene": scene,
            "split": infer_split(raw.get(cols["split"]) if cols["split"] else None, scene),
            "score": score, "quality": quality, "friction": friction, "mu_flags": flags,
            "collision_or_empty": collision_or_empty, "pure_collision": pure_collision,
            "empty": empty, "failure": failure_class(collision_or_empty, pure_collision, empty, quality, flags),
            "width_m": width,
            "parent": str(raw.get(cols["parent"], "")).strip() if cols["parent"] else "",
            "width_mode": str(raw.get(cols["width_mode"], "")).strip() if cols["width_mode"] else "",
            "center": tuple(parse_float(raw.get(cols[k])) for k in ("center_x", "center_y", "center_z")),
            "insertion": parse_float(raw.get(cols["insertion"])) if cols["insertion"] else float("nan"),
            "u": parse_float(raw.get(cols["u"])) if cols["u"] else float("nan"),
            "v": parse_float(raw.get(cols["v"])) if cols["v"] else float("nan"),
        }
        rows.append(row)
    # A candidate ID must be unique within a physical frame and intervention;
    # otherwise paired score/rank comparisons would silently overwrite rows.
    seen_ids = set()
    for row in rows:
        key = (row["sample"], row["condition"], row["candidate"])
        if key in seen_ids:
            raise ValueError(f"Duplicate candidate_id {row['candidate']!r} in sample={row['sample']!r}, condition={row['condition']!r}")
        seen_ids.add(key)
    detected_mu = {str(mu): col for mu, col in mu_cols.items() if col}
    return rows, {"columns": cols, "mu_columns": detected_mu, "width_unit_input": width_unit,
                  "width_unit_resolved": resolved_unit, "headers": fields}

def rank_corr(x: Sequence[float], y: Sequence[float], kendall: bool = False) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 2:
        return float("nan")
    if kendall:
        c = d = ta = tb = 0
        for i in range(len(a)):
            sa = np.sign(a[i + 1:] - a[i])
            sb = np.sign(b[i + 1:] - b[i])
            c += int(np.sum(sa * sb > 0))
            d += int(np.sum(sa * sb < 0))
            ta += int(np.sum((sa == 0) & (sb != 0)))
            tb += int(np.sum((sb == 0) & (sa != 0)))
        den = math.sqrt((c + d + ta) * (c + d + tb))
        return float((c - d) / den) if den else float("nan")
    def ranks(z):
        order = np.argsort(z, kind="mergesort")
        out = np.empty(len(z), dtype=float)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and z[order[j]] == z[order[i]]:
                j += 1
            out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
            i = j
        return out
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])

def select_high_score(items: Sequence[Mapping[str, Any]], quantile: float) -> List[Mapping[str, Any]]:
    n = max(1, int(math.ceil(len(items) * (1.0 - quantile))))
    return sorted(items, key=lambda x: (-x["score"], x["candidate"]))[:n]

def sample_metrics(items: Sequence[Mapping[str, Any]], high_q: float) -> Dict[str, Any]:
    ordered = sorted(items, key=lambda x: (-x["score"], x["candidate"]))
    high = select_high_score(items, high_q)
    qualities = np.asarray([x["quality"] for x in items], dtype=float)
    scores = np.asarray([x["score"] for x in items], dtype=float)
    valid = np.isfinite(qualities)
    q = qualities[valid]
    top1 = ordered[0]
    top5 = ordered[:min(5, len(ordered))]
    oracle = float(np.nanmax(qualities)) if np.any(valid) else float("nan")
    top1q = float(top1["quality"])
    result = {
        "sample": items[0]["sample"], "scene": items[0]["scene"], "split": items[0]["split"],
        "condition": items[0]["condition"], "n": len(items),
        "spearman": rank_corr(scores, qualities),
        "kendall_tau_b": rank_corr(scores, qualities, kendall=True),
        "top1_quality": top1q,
        "top5_quality": float(np.nanmean([x["quality"] for x in top5])),
        "oracle_quality": oracle,
        "top1_regret": float(oracle - top1q) if math.isfinite(oracle) and math.isfinite(top1q) else float("nan"),
        "top1_candidate": top1["candidate"],
        "top1_score": top1["score"],
        "top1_failure_class": top1["failure"],
        "high_score_failure_rate": float(np.mean([x["quality"] <= 0 for x in high if math.isfinite(x["quality"])])) if any(math.isfinite(x["quality"]) for x in high) else float("nan"),
        "high_score_n": len(high),
        "high_score_failure_classes": {k: sum(x["failure"] == k for x in high) for k in ("collision", "empty_grasp", "force_closure_failure", "success", "unknown")},
    }
    for mu in MUS:
        vals = [x["mu_flags"][mu] for x in items]
        known_vals = [v for v in vals if v is not None]
        top1_mu = top1["mu_flags"][mu]
        top5_mu = [x["mu_flags"][mu] for x in top5 if x["mu_flags"][mu] is not None]
        high_mu = [x["mu_flags"][mu] for x in high if x["mu_flags"][mu] is not None]
        key = f"mu_{mu:.1f}"
        result[key] = {
            "all_candidate_success_rate": float(np.mean(known_vals)) if known_vals else None,
            "top1_success": bool(top1_mu) if top1_mu is not None else None,
            "top5_success_rate": float(np.mean(top5_mu)) if top5_mu else None,
            "high_score_failure_rate": float(np.mean([not v for v in high_mu])) if high_mu else None,
        }
    return result

def mean_or_none(values: Sequence[float]) -> Optional[float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else None

def bootstrap_scene_ci(per_sample: Sequence[Mapping[str, Any]], metric: str,
                       n_boot: int, seed: int) -> Dict[str, Any]:
    by_scene: Dict[Any, List[float]] = defaultdict(list)
    for row in per_sample:
        scene = row.get("scene")
        value = row.get(metric)
        if scene is not None and value is not None and math.isfinite(float(value)):
            by_scene[scene].append(float(value))
    scene_means = np.asarray([np.mean(v) for _, v in sorted(by_scene.items())], dtype=float)
    if len(scene_means) == 0:
        return {"n_scenes": 0, "mean": None, "ci95": None}
    if len(scene_means) == 1 or n_boot <= 0:
        return {"n_scenes": len(scene_means), "mean": float(scene_means.mean()), "ci95": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scene_means), size=(n_boot, len(scene_means)))
    boot = scene_means[draws].mean(axis=1)
    return {"n_scenes": len(scene_means), "mean": float(scene_means.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}

def summarize_samples(samples: Sequence[Mapping[str, Any]], n_boot: int, seed: int) -> Dict[str, Any]:
    scalar_metrics = ("spearman", "kendall_tau_b", "top1_quality", "top5_quality",
                      "oracle_quality", "top1_regret", "high_score_failure_rate")
    out: Dict[str, Any] = {"num_frames": len(samples), "num_scenes": len({x["scene"] for x in samples if x["scene"] is not None})}
    for metric in scalar_metrics:
        vals = [x[metric] for x in samples]
        out[metric] = mean_or_none(vals)
        out[f"{metric}_scene_bootstrap"] = bootstrap_scene_ci(samples, metric, n_boot, seed)
    out["top1_failure_classes"] = {
        k: sum(x["top1_failure_class"] == k for x in samples) for k in
        ("collision", "empty_grasp", "force_closure_failure", "success", "unknown")
    }
    out["high_score_failure_classes"] = {
        k: sum(x["high_score_failure_classes"][k] for x in samples) for k in
        ("collision", "empty_grasp", "force_closure_failure", "success", "unknown")
    }
    out["friction_thresholds"] = {}
    for mu in MUS:
        key = f"mu_{mu:.1f}"
        out["friction_thresholds"][key] = {
            name: mean_or_none([x[key][name] for x in samples if x[key][name] is not None])
            for name in ("all_candidate_success_rate", "top1_success", "top5_success_rate", "high_score_failure_rate")
        }
    return out

def candidate_failure_breakdown(rows: Sequence[Mapping[str, Any]], high_q: float) -> Dict[str, Any]:
    def counts(items):
        total = len(items)
        return {"n": total, **{k: {"count": sum(x["failure"] == k for x in items),
                                   "rate": float(sum(x["failure"] == k for x in items) / total) if total else None}
                              for k in ("collision", "empty_grasp", "force_closure_failure", "success", "unknown")}}
    high = []
    top1 = []
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["sample"], row["condition"])].append(row)
    for items in groups.values():
        high.extend(select_high_score(items, high_q))
        top1.append(sorted(items, key=lambda x: (-x["score"], x["candidate"]))[0])
    return {"all_candidates": counts(rows), "high_score_candidates": counts(high), "top1_candidates": counts(top1)}

def width_bins(rows: Sequence[Mapping[str, Any]], edges: Sequence[float], condition: str) -> Dict[str, Any]:
    selected = [x for x in rows if x["condition"] == condition and math.isfinite(x["width_m"])]
    result = {"condition": condition, "edges_m": list(edges), "bins": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        items = [x for x in selected if lo <= x["width_m"] < hi]
        if not items:
            result["bins"].append({"range_m": [lo, hi], "n": 0})
            continue
        classes = {k: sum(x["failure"] == k for x in items) for k in
                   ("collision", "empty_grasp", "force_closure_failure", "success", "unknown")}
        entry = {
            "range_m": [lo, hi], "n": len(items), "num_scenes": len({x["scene"] for x in items if x["scene"] is not None}),
            "mean_width_m": mean_or_none([x["width_m"] for x in items]),
            "mean_score": mean_or_none([x["score"] for x in items]),
            "mean_quality": mean_or_none([x["quality"] for x in items]),
            "any_mu_success_rate": mean_or_none([float(x["quality"] > 0) for x in items if math.isfinite(x["quality"])]),
            "failure_class_counts": classes,
            "friction_threshold_success_rate": {
                f"mu_{mu:.1f}": mean_or_none([float(x["mu_flags"][mu]) for x in items if x["mu_flags"][mu] is not None])
                for mu in MUS
            },
        }
        by_sample: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for x in items:
            by_sample[x["sample"]].append(x)
        entry["within_bin_top1_quality"] = mean_or_none([
            sorted(v, key=lambda x: (-x["score"], x["candidate"]))[0]["quality"]
            for v in by_sample.values()
        ])
        result["bins"].append(entry)
    return result

def insertion_depth_bins(rows: Sequence[Mapping[str, Any]], edges: Sequence[float], condition: str) -> Dict[str, Any]:
    selected = [x for x in rows if x["condition"] == condition and math.isfinite(x["insertion"])]
    result = {"condition": condition, "edges_m": list(edges), "bins": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        items = [x for x in selected if lo <= x["insertion"] < hi]
        if not items:
            result["bins"].append({"range_m": [lo, hi], "n": 0})
            continue
        by_sample: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for x in items:
            by_sample[x["sample"]].append(x)
        result["bins"].append({
            "range_m": [lo, hi], "n": len(items),
            "num_scenes": len({x["scene"] for x in items if x["scene"] is not None}),
            "mean_insertion_depth_m": mean_or_none([x["insertion"] for x in items]),
            "mean_score": mean_or_none([x["score"] for x in items]),
            "mean_quality": mean_or_none([x["quality"] for x in items]),
            "any_mu_success_rate": mean_or_none([float(x["quality"] > 0) for x in items if math.isfinite(x["quality"])]),
            "top1_within_bin_quality": mean_or_none([
                sorted(v, key=lambda x: (-x["score"], x["candidate"]))[0]["quality"]
                for v in by_sample.values()
            ]),
            "friction_threshold_success_rate": {
                f"mu_{mu:.1f}": mean_or_none([float(x["mu_flags"][mu]) for x in items if x["mu_flags"][mu] is not None])
                for mu in MUS
            },
        })
    return result


def _geometry_family(row: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    if row["parent"]:
        return (row["sample"], "parent", row["parent"])
    c = row["center"]
    if all(math.isfinite(v) for v in c) and math.isfinite(row["insertion"]):
        # Rotation fields are optional in the report's schema. Without them,
        # do not call different orientations the same physical action family.
        raw = row["raw"]
        rot = []
        for i in range(3):
            for j in range(3):
                keys = (f"r{i}{j}", f"rot{i}{j}", f"rotation_{i}{j}", f"rotation_{i}_{j}")
                found = next((k for k in raw if norm(k) in {norm(x) for x in keys}), None)
                if found is None:
                    rot = []
                    break
                rot.append(round(parse_float(raw[found]), 4))
            if not rot:
                break
        if not rot:
            return None
        return (row["sample"], "geometry",
                *(round(v, 5) for v in c), *rot, round(row["insertion"], 5))
    return None

def matched_width_sweeps(rows: Sequence[Mapping[str, Any]], condition: str) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] != condition or not math.isfinite(row["width_m"]):
            continue
        key = _geometry_family(row)
        if key is not None:
            groups[key].append(row)
    families = []
    for key, items in groups.items():
        widths = {round(x["width_m"], 6) for x in items}
        if len(items) < 2 or len(widths) < 2:
            continue
        selected = max(items, key=lambda x: (x["score"], x["candidate"]))
        oracle = max(items, key=lambda x: (x["quality"], x["candidate"]))
        families.append({
            "sample": items[0]["sample"], "family": str(key), "n": len(items),
            "num_distinct_widths": len(widths), "width_min_m": min(widths), "width_max_m": max(widths),
            "score_selected_width_m": selected["width_m"], "score_selected_quality": selected["quality"],
            "oracle_width_m": oracle["width_m"], "oracle_quality": oracle["quality"],
            "within_family_regret": oracle["quality"] - selected["quality"],
            "score_quality_spearman": rank_corr([x["score"] for x in items], [x["quality"] for x in items]),
            "width_quality_spearman": rank_corr([x["width_m"] for x in items], [x["quality"] for x in items]),
        })
    return {
        "condition": condition, "num_matched_width_families": len(families),
        "mean_score_selected_quality": mean_or_none([x["score_selected_quality"] for x in families]),
        "mean_oracle_quality": mean_or_none([x["oracle_quality"] for x in families]),
        "mean_within_family_regret": mean_or_none([x["within_family_regret"] for x in families]),
        "mean_score_quality_spearman": mean_or_none([x["score_quality_spearman"] for x in families]),
        "mean_width_quality_spearman": mean_or_none([x["width_quality_spearman"] for x in families]),
        "families": families,
        "note": "Requires parent/action-family ID or identical center+rotation+insertion across width variants. No family is inferred from center alone.",
    }

def width_mode_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    modes = sorted({x["width_mode"] for x in rows if x["width_mode"]})
    if not modes:
        return {"available": False, "reason": "No width_mode/width_source column was detected."}
    out = {"available": True, "modes": {}}
    for mode in modes:
        items = [x for x in rows if x["width_mode"] == mode]
        out["modes"][mode] = {
            "n": len(items), "mean_width_m": mean_or_none([x["width_m"] for x in items]),
            "mean_quality": mean_or_none([x["quality"] for x in items]),
            "any_mu_success_rate": mean_or_none([float(x["quality"] > 0) for x in items if math.isfinite(x["quality"])]),
            "failure_classes": {k: sum(x["failure"] == k for x in items) for k in
                                ("collision", "empty_grasp", "force_closure_failure", "success", "unknown")},
        }
    return out

def paired_interventions(rows: Sequence[Mapping[str, Any]], reference: str) -> List[Dict[str, Any]]:
    by: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    split_scene: Dict[str, Tuple[str, Optional[int]]] = {}
    for x in rows:
        by[(x["sample"], x["condition"])][x["candidate"]] = x
        split_scene[x["sample"]] = (x["split"], x["scene"])
    conditions = sorted({x["condition"] for x in rows})
    if reference not in conditions:
        # Common aliases are accepted; otherwise choose the first condition and
        # report the choice in the output rather than silently inventing native.
        aliases = ("native", "r0", "baseline", "original")
        reference = next((c for c in aliases if c in conditions), conditions[0] if conditions else reference)
    results = []
    for condition in conditions:
        if condition == reference:
            continue
        per_sample = []
        for sample in sorted({x["sample"] for x in rows}):
            a = by.get((sample, reference), {})
            b = by.get((sample, condition), {})
            common = sorted(set(a) & set(b))
            if len(common) < 2:
                continue
            sa = np.asarray([a[k]["score"] for k in common])
            sb = np.asarray([b[k]["score"] for k in common])
            rank = rank_corr(sa, sb)
            top_a = max(common, key=lambda k: (a[k]["score"], k))
            top_b = max(common, key=lambda k: (b[k]["score"], k))
            qa, qb = a[top_a]["quality"], b[top_b]["quality"]
            per_sample.append({
                "sample": sample, "scene": split_scene[sample][1], "split": split_scene[sample][0],
                "n_common_candidates": len(common),
                "mean_abs_score_change": float(np.mean(np.abs(sb - sa))),
                "rank_stability_spearman": rank,
                "top1_changed": top_a != top_b,
                "reference_top1_quality": qa, "condition_top1_quality": qb,
                "top1_quality_change": qb - qa if math.isfinite(qa) and math.isfinite(qb) else float("nan"),
            })
        results.append({
            "reference": reference, "condition": condition, "paired_samples": len(per_sample),
            "mean_abs_score_change": mean_or_none([x["mean_abs_score_change"] for x in per_sample]),
            "mean_rank_stability_spearman": mean_or_none([x["rank_stability_spearman"] for x in per_sample]),
            "top1_changed_frames": sum(x["top1_changed"] for x in per_sample),
            "mean_top1_quality_change": mean_or_none([x["top1_quality_change"] for x in per_sample]),
            "per_sample": per_sample,
            "warning": None if per_sample else "No shared candidate IDs; do not interpret this as a paired intervention.",
        })
    return results

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value

def parse_edges(text: str) -> List[float]:
    vals = []
    for token in text.split(","):
        token = token.strip().lower()
        vals.append(float("inf") if token in ("inf", "+inf") else float(token))
    if len(vals) < 2 or any(b <= a for a, b in zip(vals[:-1], vals[1:])):
        raise ValueError("--width_bins must be strictly increasing, e.g. 0,0.02,0.04,0.06,0.08,0.10,inf")
    return vals

def render_markdown(result: Mapping[str, Any]) -> str:
    lines = ["# RayRoPE candidate diagnostics", "",
             f"- Input rows: {result['data']['num_rows']}",
             f"- Frames: {result['data']['num_frames']}",
             f"- Scenes: {result['data']['num_scenes']}",
             f"- Conditions: {', '.join(result['data']['conditions'])}", "",
             "## Ranking and regret", "",
             "| Split / condition | Frames | Spearman | Kendall τ-b | Top-1 quality | Top-5 quality | Oracle | Regret | High-score failure |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, conds in result["by_split_condition"].items():
        for condition, s in conds.items():
            def f(k):
                v = s.get(k)
                return "—" if v is None else f"{v:.4f}"
            lines.append(f"| {split} / {condition} | {s['num_frames']} | {f('spearman')} | {f('kendall_tau_b')} | {f('top1_quality')} | {f('top5_quality')} | {f('oracle_quality')} | {f('top1_regret')} | {f('high_score_failure_rate')} |")
    lines += ["", "## Failure attribution", "", "Failure classes are mutually exclusive with priority: empty grasp, collision, force-closure failure, success, unknown.", ""]
    for split, groups in result["failure_breakdown_by_split"].items():
        lines.append(f"### {split}")
        for population, counts in groups.items():
            lines.append(f"- {population}: n={counts['n']}; " + ", ".join(f"{k}={v['count']} ({v['rate']:.1%})" for k, v in counts.items() if isinstance(v, dict)))
        lines.append("")
    lines += ["## Width-conditioned quality", "", "Widths are in metres. These are observational bins, not a causal width ablation.", ""]
    for item in result["width_bins"]:
        lines.append(f"- {item['range_m']}: n={item['n']}, mean quality={item.get('mean_quality')}")
    lines += ["", "See the JSON output for all six friction thresholds, per-frame results, scene-bootstrap intervals, and paired interventions.", ""]
    return "\n".join(lines)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", default=None)
    p.add_argument("--columns_json", default=None, help="Optional canonical-to-source column map.")
    p.add_argument("--width_unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument("--reference_condition", default="native")
    p.add_argument("--high_score_quantile", type=float, default=0.95)
    p.add_argument("--width_bins", default="0,0.02,0.04,0.06,0.08,0.10,inf")
    p.add_argument("--insertion_bins", default="0,0.01,0.02,0.03,0.04,0.05,inf")
    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--sampling_protocol", choices=("score_stratified", "uniform", "full_pool", "unknown"),
                   default="score_stratified",
                   help="Current P1/P2 report used score-stratified 64-candidate sampling.")
    args = p.parse_args()
    if not 0 < args.high_score_quantile < 1:
        raise ValueError("--high_score_quantile must be in (0,1)")
    rows, schema = load_candidates(args.input_csv, args.columns_json, args.width_unit)
    conditions = sorted({x["condition"] for x in rows})
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for x in rows:
        grouped[(x["split"], x["condition"])].append(x)
    sample_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for x in rows:
        sample_groups[(x["sample"], x["condition"])].append(x)
    per_sample = [sample_metrics(v, args.high_score_quantile) for _, v in sorted(sample_groups.items())]
    by_split_condition: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for split in sorted({x["split"] for x in rows}):
        for condition in conditions:
            subset = [x for x in per_sample if x["split"] == split and x["condition"] == condition]
            if subset:
                by_split_condition[split][condition] = summarize_samples(subset, args.bootstrap_samples, args.seed)
    overall = {}
    for condition in conditions:
        subset = [x for x in per_sample if x["condition"] == condition]
        if subset:
            overall[condition] = summarize_samples(subset, args.bootstrap_samples, args.seed)
    fail_by_split = {}
    for split in sorted({x["split"] for x in rows}):
        subset = [x for x in rows if x["split"] == split and x["condition"] == args.reference_condition]
        if not subset and split in by_split_condition:
            fallback = next(iter(by_split_condition[split]))
            subset = [x for x in rows if x["split"] == split and x["condition"] == fallback]
        if subset:
            fail_by_split[split] = candidate_failure_breakdown(subset, args.high_score_quantile)
    ref_rows = [x for x in rows if x["condition"] == args.reference_condition]
    if not ref_rows and rows:
        chosen = next(iter(conditions))
        ref_rows = [x for x in rows if x["condition"] == chosen]
    edges = parse_edges(args.width_bins)
    insertion_edges = parse_edges(args.insertion_bins)
    analysis_condition = args.reference_condition if any(x["condition"] == args.reference_condition for x in rows) else (conditions[0] if conditions else "native")
    width_result = width_bins(rows, edges, analysis_condition)
    insertion_result = insertion_depth_bins(rows, insertion_edges, analysis_condition)
    width_sweep = matched_width_sweeps(rows, analysis_condition)
    width_modes = width_mode_summary(ref_rows)
    sample_count = len({x["sample"] for x in rows})
    scene_count = len({x["scene"] for x in rows if x["scene"] is not None})
    sampling_warning = []
    if args.sampling_protocol == "score_stratified":
        sampling_warning.append("Candidate pool is score-stratified; candidate-level rates and oracle quality are conditional on this selection, not an unbiased proposal-distribution estimate.")
    if scene_count < 20:
        sampling_warning.append(f"Only {scene_count} scenes are present; use scene-clustered intervals and avoid frame/candidate-level pseudo-replication.")
    if not any(schema["mu_columns"].values()) and not schema["columns"]["friction"]:
        sampling_warning.append("No six-threshold flags or friction score detected; friction-threshold decomposition is unavailable.")
    if not schema["columns"]["width"]:
        sampling_warning.append("No width column detected; width-conditioned analysis is unavailable.")
    result = {
        "data": {
            "input_csv": str(Path(args.input_csv)), "num_rows": len(rows),
            "num_frames": sample_count, "num_scenes": scene_count,
            "conditions": conditions, "sampling_protocol": args.sampling_protocol,
            "high_score_quantile": args.high_score_quantile,
            "width_unit_resolved": schema["width_unit_resolved"],
            "schema": schema,
        },
        "overall_by_condition": overall,
        "by_split_condition": by_split_condition,
        "failure_breakdown_by_split": fail_by_split,
        "width_bins": width_result["bins"],
        "width_bins_metadata": {"condition": width_result["condition"], "edges_m": edges},
        "insertion_depth_bins": insertion_result["bins"],
        "insertion_depth_bins_metadata": {"condition": insertion_result["condition"], "edges_m": insertion_edges},
        "matched_width_sweep": width_sweep,
        "width_mode_summary": width_modes,
        "paired_interventions_vs_reference": paired_interventions(rows, args.reference_condition),
        "per_sample": per_sample,
        "warnings": sampling_warning,
        "interpretation_limits": [
            "Quality is the mean of six official friction-threshold success bits when available; it is not official AP.",
            "High-score failure is measured within each frame using the configured score quantile.",
            "Oracle quality is limited to candidates present in this CSV.",
            "Scene bootstrap captures scene sampling uncertainty, not training-seed variability.",
            "Width-bin comparisons are observational; matched width sweeps require the same physical action family evaluated at multiple widths.",
            "A score-stratified candidate sample cannot estimate unbiased full-pool candidate quality.",
        ],
    }
    result = json_safe(result)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(out)
    if args.output_md:
        md = Path(args.output_md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"output_json": str(out), "frames": sample_count, "scenes": scene_count,
                      "conditions": conditions, "warnings": sampling_warning}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
