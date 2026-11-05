#!/usr/bin/env python3
"""Relate measured metric-depth error to RayRoPE +10 mm reader interventions.

Manifest CSV columns:
  sample_id,pred_path,gt_path
Optional:
  mask_path,pred_key,gt_key,mask_key

Depth arrays must be aligned, in the same metric unit, and refer to the exact
RGB frame/depth tensor used by the model. The script never assumes that the
saved depth files contain ground truth; a separate gt_path is mandatory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    from .analyze_rayrope_candidates import load_candidates, rank_corr
except ImportError:  # direct execution: python diagnostics/analyze_rayrope_depth_error.py
    from analyze_rayrope_candidates import load_candidates, rank_corr


def load_array(path: str, key: str = "") -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        names = list(z.files)
        chosen = key or next((k for k in ("depth", "pred_depth", "gt_depth", "arr_0") if k in names), names[0] if names else "")
        if not chosen or chosen not in names:
            raise KeyError(f"Cannot select key {key!r} from {path}; keys={names}")
        arr = z[chosen]
    elif suffix in (".png", ".tif", ".tiff"):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to read PNG/TIFF depth images.") from exc
        arr = np.asarray(Image.open(p))
    else:
        raise ValueError(f"Unsupported depth format {suffix!r}: use .npy/.npz/.png/.tif")
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Depth array must reduce to HxW, got {arr.shape} from {path}")
    return arr.astype(np.float64, copy=False)


def read_manifest(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"sample_id", "pred_path", "gt_path"}
        missing = required - fields
        if missing:
            raise ValueError(f"Depth manifest missing columns {sorted(missing)}; found {sorted(fields)}")
        out = {}
        for row in reader:
            sid = row["sample_id"].strip()
            if not sid or sid in out:
                raise ValueError(f"Empty or duplicate sample_id in depth manifest: {sid!r}")
            out[sid] = row
    return out


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray], unit: str) -> Dict[str, Any]:
    if pred.shape != gt.shape:
        raise ValueError(f"Pred/GT shape mismatch: {pred.shape} vs {gt.shape}")
    scale = 0.001 if unit == "mm" else 1.0
    pred = pred * scale
    gt = gt * scale
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if mask is not None:
        if mask.shape != pred.shape:
            raise ValueError(f"Mask shape mismatch: {mask.shape} vs {pred.shape}")
        valid &= np.isfinite(mask) & (mask > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No valid overlapping positive depth pixels.")
    err = pred[valid] - gt[valid]
    abs_err = np.abs(err)
    return {
        "valid_pixels": n_valid,
        "total_pixels": int(pred.size),
        "valid_fraction": float(n_valid / pred.size),
        "mean_signed_error_m": float(np.mean(err)),
        "median_signed_error_m": float(np.median(err)),
        "mae_m": float(np.mean(abs_err)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_error_m": float(np.median(abs_err)),
        "p95_abs_error_m": float(np.quantile(abs_err, 0.95)),
        "error_map_m": np.where(valid, pred - gt, np.nan),
    }


def local_error(error_map: np.ndarray, u: float, v: float, radius: int) -> Optional[float]:
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    h, w = error_map.shape
    x, y = int(round(u)), int(round(v))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = error_map[y0:y1, x0:x1]
    valid = np.isfinite(patch)
    return float(np.median(patch[valid])) if valid.any() else None


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def paired_candidate_effect(rows, native: str, perturbed: str) -> Dict[str, Dict[str, Any]]:
    grouped = defaultdict(dict)
    for row in rows:
        if row["condition"] in (native, perturbed):
            grouped[(row["sample"], row["condition"])][row["candidate"]] = row
    samples = sorted({sample for sample, _ in grouped})
    out = {}
    for sample in samples:
        a = grouped.get((sample, native), {})
        b = grouped.get((sample, perturbed), {})
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            continue
        native_top = max(common, key=lambda k: (a[k]["score"], k))
        perturbed_top = max(common, key=lambda k: (b[k]["score"], k))
        score_delta = np.asarray([b[k]["score"] - a[k]["score"] for k in common])
        q_native = a[native_top]["quality"]
        q_perturbed = b[perturbed_top]["quality"]
        out[sample] = {
            "n_common_candidates": len(common),
            "mean_abs_score_change": float(np.mean(np.abs(score_delta))),
            "mean_signed_score_change": float(np.mean(score_delta)),
            "native_top1_candidate": native_top,
            "perturbed_top1_candidate": perturbed_top,
            "top1_changed": native_top != perturbed_top,
            "native_top1_quality": q_native,
            "perturbed_top1_quality": q_perturbed,
            "top1_quality_change": q_perturbed - q_native if math.isfinite(q_native) and math.isfinite(q_perturbed) else float("nan"),
            "native_top1_score_change": b[native_top]["score"] - a[native_top]["score"],
            "rank_stability_spearman": rank_corr([a[k]["score"] for k in common], [b[k]["score"] for k in common]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth_manifest_csv", required=True)
    p.add_argument("--candidate_csv", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--columns_json", default=None)
    p.add_argument("--depth_unit", choices=("m", "mm"), default="m")
    p.add_argument("--native_condition", default="native")
    p.add_argument("--perturbed_condition", default="reader_plus10mm")
    p.add_argument("--local_radius_px", type=int, default=5)
    args = p.parse_args()
    if args.local_radius_px < 0:
        raise ValueError("--local_radius_px must be >=0")

    manifest = read_manifest(args.depth_manifest_csv)
    candidates, schema = load_candidates(args.candidate_csv, args.columns_json, "auto")
    effects = paired_candidate_effect(candidates, args.native_condition, args.perturbed_condition)
    per_sample = {}
    for sample, item in manifest.items():
        pred = load_array(item["pred_path"], item.get("pred_key", ""))
        gt = load_array(item["gt_path"], item.get("gt_key", ""))
        mask = load_array(item["mask_path"], item.get("mask_key", "")) if item.get("mask_path", "").strip() else None
        stats = depth_metrics(pred, gt, mask, args.depth_unit)
        error_map = stats.pop("error_map_m")
        sample_rows = [x for x in candidates if x["sample"] == sample and x["condition"] == args.native_condition]
        local_values = [local_error(error_map, x["u"], x["v"], args.local_radius_px) for x in sample_rows]
        local_values = [x for x in local_values if x is not None]
        stats["candidate_local_median_signed_error_m"] = float(np.median(local_values)) if local_values else None
        stats["candidate_local_error_count"] = len(local_values)
        stats.update(effects.get(sample, {}))
        per_sample[sample] = stats

    common = sorted(set(manifest) & set(effects))
    associations = {}
    for xkey in ("median_signed_error_m", "candidate_local_median_signed_error_m", "mae_m", "p95_abs_error_m"):
        for ykey in ("top1_quality_change", "native_top1_score_change", "mean_abs_score_change"):
            pairs = [(per_sample[s].get(xkey), effects[s].get(ykey)) for s in common]
            xx = [a for a, b in pairs if a is not None and b is not None]
            yy = [b for a, b in pairs if a is not None and b is not None]
            associations[f"{xkey}__{ykey}"] = {
                "n_frames": len(xx), "pearson": pearson(xx, yy),
                "spearman": rank_corr(xx, yy),
            }
    result = {
        "depth_manifest": args.depth_manifest_csv,
        "candidate_csv": args.candidate_csv,
        "depth_unit_input": args.depth_unit,
        "native_condition": args.native_condition,
        "perturbed_condition": args.perturbed_condition,
        "num_depth_frames": len(manifest),
        "num_paired_candidate_frames": len(effects),
        "per_sample": per_sample,
        "depth_error_vs_reader_intervention_association": associations,
        "schema": schema,
        "limitations": [
            "Predicted and GT depth must be registered, same-resolution, same-camera-Z metric maps.",
            "Global image depth error can differ from error at grasp support; local error is computed only when candidate pixel u/v columns are available.",
            "Association with +10 mm score/selection changes is observational and does not prove that the perturbation compensates depth bias.",
            "This script cannot use a SHA256-only depth manifest; actual pred and GT arrays are required.",
        ],
    }
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if math.isfinite(float(x)) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        return x
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({"output_json": str(out), "depth_frames": len(manifest),
                      "paired_frames": len(effects), "association_count": len(associations)}, indent=2))


if __name__ == "__main__":
    main()
