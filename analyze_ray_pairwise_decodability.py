#!/usr/bin/env python3
"""Train-vs-validation decodability analysis for the pairwise ray selector.

This is cache-only analysis. It never runs Stage-1, never calls the CAD/DexNet
analytic evaluator, and never updates any weight. Exact-action outcomes and
frozen local features are read from the previously mined cache.

The analysis separates two questions:

1. Candidate-level decodability: can frozen center-conditioned evidence predict
   the exact utility advantage ΔU_k = U(g_k) - U(g_native)?
2. Ray-level selection transfer: how much exact best-of-K headroom does the
   learned selector recover on train scenes versus validation scenes?

Because the validation split is also used for checkpoint/threshold selection,
thresholded validation metrics are diagnostic rather than untouched test results.
Threshold-free candidate metrics (correlation/AUROC/AUPRC) and the threshold-0
selection control are therefore reported explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata

from utils.ray_bestofk_diagnostic import gather_kn, select_exact_oracle, select_raw_score
from utils.ray_pairwise_selector import (
    RayPairwiseSelector,
    compose_pairwise_features,
    select_with_native_fallback,
)


EPS = 1.0e-8


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--selector_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_scene_start", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override checkpoint native-fallback threshold.")
    p.add_argument("--max_train_frames", type=int, default=0)
    p.add_argument("--max_val_frames", type=int, default=0)
    p.add_argument("--top_fracs", default="0.05,0.10,0.20")
    p.add_argument("--progress_every", type=int, default=200)
    return p.parse_args()


ARGS = _parse_args()


def _scene_id(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def _cache_paths(root: Path) -> List[Path]:
    return sorted(root.glob("scene_*/ann_*.npz"))


def _split_paths(paths: Sequence[Path], val_scene_start: int) -> Tuple[List[Path], List[Path]]:
    train = [p for p in paths if _scene_id(p) < val_scene_start]
    val = [p for p in paths if _scene_id(p) >= val_scene_start]
    if not train or not val:
        raise RuntimeError(
            f"Need both train and validation caches; got train={len(train)}, val={len(val)} "
            f"with val_scene_start={val_scene_start}."
        )
    return train, val


def _parse_top_fracs(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if not values or any((x <= 0.0 or x > 1.0) for x in values):
        raise ValueError("top_fracs must contain values in (0,1].")
    return values


def _load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    required = (
        "selector_state_dict", "feature_dim", "hidden_dim", "dropout",
        "feature_mean", "feature_std",
    )
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f"Selector checkpoint missing keys: {missing}")
    model = RayPairwiseSelector(
        int(ckpt["feature_dim"]), int(ckpt["hidden_dim"]), float(ckpt["dropout"])
    ).to(device)
    model.load_state_dict(ckpt["selector_state_dict"])
    model.eval()
    mean = ckpt["feature_mean"].to(device).float()
    std = ckpt["feature_std"].to(device).float()
    threshold = float(ckpt.get("selector_threshold", 0.0))
    if ARGS.threshold is not None:
        threshold = float(ARGS.threshold)
    return ckpt, model, mean, std, threshold


def _load_frame(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as d:
        selected = torch.from_numpy(d["selected_feature"].astype(np.float32)).to(device)
        mean = torch.from_numpy(d["mean_feature"].astype(np.float32)).to(device)
        raw = torch.from_numpy(d["raw_score"].astype(np.float32)).to(device)
        utility = d["utility"].astype(np.float32)
        friction = d["friction"].astype(np.float32)
        valid = d["valid"].astype(bool)
        offsets = torch.from_numpy(d["offsets_mm"].astype(np.float32)).to(device)
        zero = int(np.asarray(d["zero_index"]).reshape(-1)[0])
        scene_id = int(np.asarray(d["scene_id"]).reshape(-1)[0])
        anno_id = int(np.asarray(d["anno_id"]).reshape(-1)[0])
    return selected, mean, raw, utility, friction, valid, offsets, zero, scene_id, anno_id


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(x.mean()) if x.size else float("nan")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(float(b)) > 1.0e-12 else float("nan")


def _success(friction: np.ndarray, threshold: float) -> np.ndarray:
    f = np.asarray(friction, dtype=np.float32)
    return np.isfinite(f) & (f > 0.0) & (f <= float(threshold) + 1.0e-6)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    return _pearson(rx, ry)


def _binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    sum_pos = float(ranks[y].sum())
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    yy = y[order].astype(np.float64)
    tp = np.cumsum(yy)
    precision = tp / (np.arange(len(yy), dtype=np.float64) + 1.0)
    return float((precision * yy).sum() / pos)


def _candidate_metrics(score: np.ndarray, target_delta: np.ndarray, top_fracs: Sequence[float]) -> Dict[str, float]:
    score = np.asarray(score, dtype=np.float64)
    delta = np.asarray(target_delta, dtype=np.float64)
    if score.shape != delta.shape:
        raise ValueError("score and target_delta must have equal shape")
    informative = np.abs(delta) > EPS
    beneficial = delta > EPS
    harmful = delta < -EPS
    out: Dict[str, float] = {
        "num_candidates": int(delta.size),
        "informative_fraction": _safe_mean(informative),
        "beneficial_fraction": _safe_mean(beneficial),
        "harmful_fraction": _safe_mean(harmful),
        "mae_delta": float(np.mean(np.abs(score - delta))) if delta.size else float("nan"),
        "pearson_all": _pearson(score, delta),
        "spearman_all": _spearman(score, delta),
        # Deployment-oriented: beneficial alternative vs tie/harmful alternative.
        "beneficial_vs_all_auroc": _binary_auroc(beneficial, score),
        "beneficial_vs_all_auprc": _average_precision(beneficial, score),
    }
    if informative.any():
        si = score[informative]
        di = delta[informative]
        yi = di > 0
        out.update({
            "pearson_informative": _pearson(si, di),
            "spearman_informative": _spearman(si, di),
            "sign_auroc": _binary_auroc(yi, si),
            "sign_auprc_beneficial": _average_precision(yi, si),
            "sign_auprc_harmful": _average_precision(~yi, -si),
        })
    else:
        out.update({
            "pearson_informative": float("nan"),
            "spearman_informative": float("nan"),
            "sign_auroc": float("nan"),
            "sign_auprc_beneficial": float("nan"),
            "sign_auprc_harmful": float("nan"),
        })

    order = np.argsort(-score, kind="stable")
    for frac in top_fracs:
        n = max(1, int(math.ceil(len(order) * float(frac))))
        idx = order[:n]
        tag = f"top{int(round(frac * 100)):02d}"
        out[f"{tag}_beneficial_precision"] = _safe_mean(beneficial[idx])
        out[f"{tag}_harmful_fraction"] = _safe_mean(harmful[idx])
        out[f"{tag}_mean_exact_delta"] = _safe_mean(delta[idx])
    return out


def _policy_metrics(
    utility: np.ndarray,
    friction: np.ndarray,
    selected_k: np.ndarray,
    native_k: np.ndarray,
    oracle_k: np.ndarray,
) -> Dict[str, np.ndarray]:
    us = gather_kn(utility, selected_k)
    un = gather_kn(utility, native_k)
    fs = gather_kn(friction, selected_k)
    fn = gather_kn(friction, native_k)
    ss = _success(fs, 0.8)
    sn = _success(fn, 0.8)
    return {
        "utility": us,
        "native_utility": un,
        "success08": ss,
        "native_success08": sn,
        "rescue08": (~sn) & ss,
        "harm08": sn & (~ss),
        "changed": selected_k != native_k,
        "match_oracle": selected_k == oracle_k,
        "beneficial_switch": (selected_k != native_k) & (us > un + EPS),
        "harmful_switch": (selected_k != native_k) & (us < un - EPS),
        "tie_switch": (selected_k != native_k) & (np.abs(us - un) <= EPS),
    }


def _aggregate_policy(chunks: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    keys = chunks[0].keys() if chunks else []
    cat = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    changed = cat["changed"]
    out = {
        "utility": _safe_mean(cat["utility"]),
        "native_utility": _safe_mean(cat["native_utility"]),
        "utility_gain": _safe_mean(cat["utility"] - cat["native_utility"]),
        "success08": _safe_mean(cat["success08"]),
        "native_success08": _safe_mean(cat["native_success08"]),
        "success08_gain": _safe_mean(cat["success08"].astype(np.float32) - cat["native_success08"].astype(np.float32)),
        "rescue08": _safe_mean(cat["rescue08"]),
        "harm08": _safe_mean(cat["harm08"]),
        "change_rate": _safe_mean(changed),
        "match_oracle": _safe_mean(cat["match_oracle"]),
        "beneficial_switch_rate_all": _safe_mean(cat["beneficial_switch"]),
        "harmful_switch_rate_all": _safe_mean(cat["harmful_switch"]),
    }
    if changed.any():
        out["switch_precision_beneficial"] = _safe_mean(cat["beneficial_switch"][changed])
        out["switch_harmful_fraction"] = _safe_mean(cat["harmful_switch"][changed])
        out["switch_tie_fraction"] = _safe_mean(cat["tie_switch"][changed])
    else:
        out["switch_precision_beneficial"] = float("nan")
        out["switch_harmful_fraction"] = float("nan")
        out["switch_tie_fraction"] = float("nan")
    return out


@torch.no_grad()
def _analyze_split(
    name: str,
    paths: Sequence[Path],
    model: RayPairwiseSelector,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    threshold: float,
    device: torch.device,
    top_fracs: Sequence[float],
):
    learned_scores: List[np.ndarray] = []
    raw_scores: List[np.ndarray] = []
    target_deltas: List[np.ndarray] = []

    policy_chunks: Dict[str, List[Dict[str, np.ndarray]]] = {
        "raw": [],
        "learned_t0": [],
        "learned_threshold": [],
        "oracle": [],
    }
    oracle_headroom_u: List[np.ndarray] = []
    oracle_headroom_s: List[np.ndarray] = []
    improvable: List[np.ndarray] = []

    for i, path in enumerate(paths):
        selected, mean, raw_t, utility, friction, valid, offsets, zero, scene_id, anno_id = _load_frame(path, device)
        K, N = utility.shape
        if raw_t.shape != (K, N) or valid.shape != (K, N):
            raise RuntimeError(f"Malformed cache {path}: utility={utility.shape} raw={tuple(raw_t.shape)} valid={valid.shape}")
        if not np.all(valid[zero]):
            raise RuntimeError(f"Native candidate invalid in cache {path}")

        pair = compose_pairwise_features(selected, mean, raw_t, offsets, zero)
        if pair.shape[-1] != model.feature_dim:
            raise RuntimeError(
                f"Feature dim mismatch in {path}: cache={pair.shape[-1]} selector={model.feature_dim}"
            )
        pred = model(((pair - feat_mean) / feat_std.clamp_min(1.0e-5)).reshape(-1, pair.shape[-1])).reshape(K, N)
        pred[zero] = 0.0
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        raw = raw_t.detach().cpu().numpy().astype(np.float32)

        target_delta = utility - utility[zero : zero + 1]
        alt_mask = valid.copy()
        alt_mask[zero] = False
        learned_scores.append(pred_np[alt_mask])
        raw_scores.append((raw - raw[zero : zero + 1])[alt_mask])
        target_deltas.append(target_delta[alt_mask])

        native_k = np.full(N, zero, dtype=np.int64)
        raw_k = select_raw_score(raw, valid)
        oracle_k = select_exact_oracle(utility, raw, valid)
        learned_t0_k = select_with_native_fallback(pred_np, valid, zero, 0.0)
        learned_thr_k = select_with_native_fallback(pred_np, valid, zero, threshold)

        for key, kk in (
            ("raw", raw_k),
            ("learned_t0", learned_t0_k),
            ("learned_threshold", learned_thr_k),
            ("oracle", oracle_k),
        ):
            policy_chunks[key].append(_policy_metrics(utility, friction, kk, native_k, oracle_k))

        oracle_u = gather_kn(utility, oracle_k)
        native_u = gather_kn(utility, native_k)
        oracle_s = _success(gather_kn(friction, oracle_k), 0.8)
        native_s = _success(gather_kn(friction, native_k), 0.8)
        oracle_headroom_u.append(oracle_u - native_u)
        oracle_headroom_s.append(oracle_s.astype(np.float32) - native_s.astype(np.float32))
        improvable.append(oracle_u > native_u + EPS)

        if ARGS.progress_every > 0 and (i + 1) % ARGS.progress_every == 0:
            print(f"[DECODE][{name}] {i+1}/{len(paths)} frames", flush=True)

    learned_score = np.concatenate(learned_scores)
    raw_score = np.concatenate(raw_scores)
    delta = np.concatenate(target_deltas)
    oracle_u_gap = np.concatenate(oracle_headroom_u)
    oracle_s_gap = np.concatenate(oracle_headroom_s)
    improvable_arr = np.concatenate(improvable)

    candidate = {
        "learned": _candidate_metrics(learned_score, delta, top_fracs),
        "raw_delta": _candidate_metrics(raw_score, delta, top_fracs),
    }
    policies = {k: _aggregate_policy(v) for k, v in policy_chunks.items()}
    oracle_u_headroom = _safe_mean(oracle_u_gap)
    oracle_s_headroom = _safe_mean(oracle_s_gap)
    for key in ("raw", "learned_t0", "learned_threshold"):
        policies[key]["utility_headroom_recovery"] = _safe_div(
            policies[key]["utility_gain"], oracle_u_headroom
        )
        policies[key]["success08_headroom_recovery"] = _safe_div(
            policies[key]["success08_gain"], oracle_s_headroom
        )

    return {
        "split": name,
        "num_frames": len(paths),
        "num_candidate_pairs": int(delta.size),
        "num_rays": int(sum(np.load(p, allow_pickle=False)["utility"].shape[1] for p in paths)),
        "checkpoint_threshold": float(threshold),
        "candidate_level": candidate,
        "ray_level": {
            "oracle_utility_headroom": oracle_u_headroom,
            "oracle_success08_headroom": oracle_s_headroom,
            "improvable_ray_fraction": _safe_mean(improvable_arr),
            "policies": policies,
        },
    }


def _flatten(prefix: str, obj, out: Dict[str, object]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        out[prefix] = obj


def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.4f}"
    return str(x)


def _write_report(path: Path, checkpoint: Path, threshold: float, train: Dict, val: Dict):
    lines = []
    lines.append("# Train-vs-Val Pairwise Selector Decodability")
    lines.append("")
    lines.append(f"- checkpoint: `{checkpoint}`")
    lines.append(f"- checkpoint/native-fallback threshold: `{threshold:.6f}`")
    lines.append(f"- train frames: {train['num_frames']}")
    lines.append(f"- validation frames: {val['num_frames']}")
    lines.append("")
    lines.append("> Validation is the checkpoint/threshold-selection split, not an untouched test split. "
                 "Use candidate-level threshold-free metrics and learned_t0 for decodability transfer; "
                 "use learned_threshold to inspect the deployed validation-selected operating point.")
    lines.append("")
    lines.append("## Candidate-level decodability")
    lines.append("")
    lines.append("| Split | Score | Spearman all | Spearman informative | Sign AUROC | Beneficial AUPRC | Harmful AUPRC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for split_name, result in (("train", train), ("val", val)):
        for score_name in ("raw_delta", "learned"):
            m = result["candidate_level"][score_name]
            lines.append(
                f"| {split_name} | {score_name} | {_fmt(m['spearman_all'])} | "
                f"{_fmt(m['spearman_informative'])} | {_fmt(m['sign_auroc'])} | "
                f"{_fmt(m['sign_auprc_beneficial'])} | {_fmt(m['sign_auprc_harmful'])} |"
            )
    lines.append("")
    lines.append("## Ray-level exact-action selection")
    lines.append("")
    lines.append("| Split | Policy | Utility | ΔUtility | Success@0.8 | ΔSuccess@0.8 | Rescue | Harm | Change | Headroom recovery |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split_name, result in (("train", train), ("val", val)):
        for policy in ("raw", "learned_t0", "learned_threshold", "oracle"):
            m = result["ray_level"]["policies"][policy]
            recovery = m.get("utility_headroom_recovery", 1.0 if policy == "oracle" else float("nan"))
            lines.append(
                f"| {split_name} | {policy} | {_fmt(m['utility'])} | {_fmt(m['utility_gain'])} | "
                f"{_fmt(m['success08'])} | {_fmt(m['success08_gain'])} | {_fmt(m['rescue08'])} | "
                f"{_fmt(m['harm08'])} | {_fmt(m['change_rate'])} | {_fmt(recovery)} |"
            )
    lines.append("")
    lines.append("## Generalization gaps")
    lines.append("")
    learned_train = train["candidate_level"]["learned"]
    learned_val = val["candidate_level"]["learned"]
    t0_train = train["ray_level"]["policies"]["learned_t0"]
    t0_val = val["ray_level"]["policies"]["learned_t0"]
    lines.append(f"- learned sign-AUROC gap (train - val): {_fmt(learned_train['sign_auroc'] - learned_val['sign_auroc'])}")
    lines.append(f"- learned Spearman gap (train - val): {_fmt(learned_train['spearman_all'] - learned_val['spearman_all'])}")
    lines.append(f"- threshold-0 utility-gain gap (train - val): {_fmt(t0_train['utility_gain'] - t0_val['utility_gain'])}")
    lines.append("")
    lines.append("Interpretation gate:")
    lines.append("- train and val both weak -> frozen evidence / selector decodability is the bottleneck;")
    lines.append("- train strong, val much weaker -> cross-scene/domain generalization is the bottleneck;")
    lines.append("- candidate metrics transfer but ray selection remains weak -> objective/ranking/fallback policy is the bottleneck.")
    path.write_text("\n".join(lines) + "\n")


def main():
    cache_root = Path(ARGS.cache_root)
    checkpoint_path = Path(ARGS.selector_checkpoint)
    out = Path(ARGS.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    top_fracs = _parse_top_fracs(ARGS.top_fracs)
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")

    paths = _cache_paths(cache_root)
    train_paths, val_paths = _split_paths(paths, ARGS.val_scene_start)
    if ARGS.max_train_frames > 0:
        train_paths = train_paths[: ARGS.max_train_frames]
    if ARGS.max_val_frames > 0:
        val_paths = val_paths[: ARGS.max_val_frames]

    ckpt, model, feat_mean, feat_std, threshold = _load_checkpoint(str(checkpoint_path), device)
    print(
        f"[DECODE] checkpoint={checkpoint_path} threshold={threshold:.6f} "
        f"train_frames={len(train_paths)} val_frames={len(val_paths)} device={device}",
        flush=True,
    )

    train = _analyze_split(
        "train", train_paths, model, feat_mean, feat_std, threshold, device, top_fracs
    )
    val = _analyze_split(
        "val", val_paths, model, feat_mean, feat_std, threshold, device, top_fracs
    )

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_threshold": float(threshold),
        "val_scene_start": int(ARGS.val_scene_start),
        "cache_root": str(cache_root.resolve()),
        "train": train,
        "val": val,
        "notes": {
            "val_is_selection_split": True,
            "candidate_metrics_are_threshold_free": True,
            "learned_t0_uses_threshold_zero": True,
            "learned_threshold_uses_checkpoint_selected_threshold": True,
        },
    }
    with (out / "decodability.json").open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    rows = []
    for split_name, split_result in (("train", train), ("val", val)):
        flat: Dict[str, object] = {"split": split_name}
        _flatten("", split_result, flat)
        rows.append(flat)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (out / "decodability_flat.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    _write_report(out / "REPORT.md", checkpoint_path, threshold, train, val)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
