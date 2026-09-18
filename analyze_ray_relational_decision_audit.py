#!/usr/bin/env python3
"""Cache-only train-vs-val audit for relational selective correction.

No Stage-1 forward, CAD model, DexNet evaluator, or weight update is used.

The audit separates:
A) opportunity: any valid alternative has U_k > U_0;
B) where: the selector-proposed alternative is actually better than native;
C) action confidence: the delta head predicts whether that proposed action is better.

It also evaluates counterfactual policies that replace only the gate, only the
center selector, or only the deployment confidence rule.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from utils.ray_bestofk_diagnostic import select_exact_oracle
from utils.ray_relational_selective import (
    CrossCenterRelationalSelective,
    compose_relational_tokens,
    select_relational_correction,
)

EPS = 1.0e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_cache_root", required=True)
    p.add_argument("--val_cache_root", required=True)
    p.add_argument("--selector_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_scene_start", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_train_frames", type=int, default=0)
    p.add_argument("--max_val_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=200)
    p.add_argument("--save_per_query", action="store_true")
    p.add_argument("--delta_sweep_points", type=int, default=81)
    return p.parse_args()


ARGS = parse_args()


def scene_id(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def filtered_paths(root: Path, train: bool):
    paths = sorted(root.glob("scene_*/ann_*.npz"))
    if train:
        return [p for p in paths if scene_id(p) < ARGS.val_scene_start]
    return [p for p in paths if scene_id(p) >= ARGS.val_scene_start]


def safe_mean(x):
    x = np.asarray(x)
    return float(x.mean()) if x.size else float("nan")


def safe_div(a, b):
    return float(a / b) if abs(float(b)) > 1e-12 else float("nan")


def pearson(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 2:
        return float("nan")
    return pearson(rankdata(x, method="average"), rankdata(y, method="average"))


def auroc(y_true, score):
    y = np.asarray(y_true, bool)
    s = np.asarray(score, np.float64)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(y_true, score):
    y = np.asarray(y_true, bool)
    s = np.asarray(score, np.float64)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    yy = y[order].astype(np.float64)
    precision = np.cumsum(yy) / (np.arange(len(yy), dtype=np.float64) + 1.0)
    return float((precision * yy).sum() / pos)


def binary_at_threshold(y_true, score, threshold):
    y = np.asarray(y_true, bool)
    pred = np.asarray(score, np.float64) > float(threshold)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    return {
        "threshold": float(threshold),
        "positive_rate": safe_mean(pred),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "fpr": safe_div(fp, fp + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def success08(friction):
    f = np.asarray(friction, np.float32)
    return np.isfinite(f) & (f > 0.0) & (f <= 0.8 + 1e-6)


def gather_kn(arr_kn, k_n):
    arr = np.asarray(arr_kn)
    k = np.asarray(k_n, np.int64)
    return arr[k, np.arange(arr.shape[1])]


def load_frame(path: Path, device):
    with np.load(path, allow_pickle=False) as d:
        selected = torch.from_numpy(d["selected_feature"].astype(np.float32)).to(device)
        mean = torch.from_numpy(d["mean_feature"].astype(np.float32)).to(device)
        raw = torch.from_numpy(d["raw_score"].astype(np.float32)).to(device)
        utility = d["utility"].astype(np.float32)
        friction = d["friction"].astype(np.float32)
        valid = d["valid"].astype(bool)
        offsets = torch.from_numpy(d["offsets_mm"].astype(np.float32)).to(device)
        zero = int(np.asarray(d["zero_index"]).reshape(-1)[0])
        sid = int(np.asarray(d["scene_id"]).reshape(-1)[0])
        aid = int(np.asarray(d["anno_id"]).reshape(-1)[0])
    tokens = compose_relational_tokens(selected, mean, raw, offsets, zero)
    return {
        "tokens": tokens,
        "raw": raw.detach().cpu().numpy(),
        "utility": utility,
        "friction": friction,
        "valid": valid,
        "offsets": offsets.detach().cpu().numpy(),
        "zero": zero,
        "scene_id": sid,
        "anno_id": aid,
    }


def load_model(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("selector_type") != "cross_center_relational_selective_v1":
        raise RuntimeError("Unexpected selector_type: %r" % ckpt.get("selector_type"))
    model = CrossCenterRelationalSelective(
        token_dim=int(ckpt["token_dim"]),
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
        ff_dim=int(ckpt["ff_dim"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return (
        ckpt,
        model,
        ckpt["token_mean"].to(device).float(),
        ckpt["token_std"].to(device).float(),
        float(ckpt.get("move_threshold", 0.5)),
    )


def exact_best_alt(utility, valid, raw, zero):
    alt_valid = valid.copy()
    alt_valid[zero] = False
    masked_u = np.where(alt_valid, utility, -np.inf)
    best_u = masked_u.max(axis=0)
    candidate = masked_u == best_u[None, :]
    masked_raw = np.where(candidate & alt_valid, raw, -np.inf)
    best_k = np.argmax(masked_raw, axis=0).astype(np.int64)
    has_alt = alt_valid.any(axis=0)
    best_k = np.where(has_alt, best_k, zero).astype(np.int64)
    best_u = np.where(has_alt, best_u, utility[zero])
    return best_k, best_u, has_alt


def policy_summary(k, utility, friction, zero, oracle_k):
    n = utility.shape[1]
    native_k = np.full(n, zero, np.int64)
    us = gather_kn(utility, k)
    un = utility[zero]
    uo = gather_kn(utility, oracle_k)
    fs = gather_kn(friction, k)
    fn = friction[zero]
    ss = success08(fs)
    sn = success08(fn)
    changed = k != zero
    beneficial = changed & (us > un + EPS)
    harmful = changed & (us < un - EPS)
    tie = changed & (np.abs(us - un) <= EPS)
    return {
        "utility": safe_mean(us),
        "native_utility": safe_mean(un),
        "utility_gain": safe_mean(us - un),
        "oracle_utility": safe_mean(uo),
        "oracle_headroom": safe_mean(uo - un),
        "utility_headroom_recovery": safe_div(safe_mean(us - un), safe_mean(uo - un)),
        "success08": safe_mean(ss),
        "native_success08": safe_mean(sn),
        "success08_gain": safe_mean(ss.astype(np.float32) - sn.astype(np.float32)),
        "rescue08": safe_mean((~sn) & ss),
        "harm08": safe_mean(sn & (~ss)),
        "change_rate": safe_mean(changed),
        "beneficial_change_rate": safe_mean(beneficial),
        "harmful_change_rate": safe_mean(harmful),
        "tie_change_rate": safe_mean(tie),
        "switch_precision_beneficial": safe_mean(beneficial[changed]) if changed.any() else float("nan"),
        "switch_harmful_fraction": safe_mean(harmful[changed]) if changed.any() else float("nan"),
        "match_oracle": safe_mean(k == oracle_k),
    }


def delta_thresholds(values):
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.array([0.0])
    lo, hi = np.quantile(v, [0.01, 0.99])
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)
    if abs(hi - lo) < 1e-8:
        return np.array([0.0, lo])
    t = np.linspace(lo, hi, max(3, ARGS.delta_sweep_points))
    return np.unique(np.concatenate((t, np.array([0.0]))))


def sweep_policy(scores, builder, utility, friction, zero, oracle_k):
    rows = []
    for t in delta_thresholds(scores):
        k = builder(float(t))
        rows.append({"delta_threshold": float(t), **policy_summary(k, utility, friction, zero, oracle_k)})
    best = max(rows, key=lambda r: (r["utility"], -r["harm08"], -r["change_rate"]))
    return best, rows


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def analyze_split(name, paths, model, token_mean, token_std, move_threshold, device, split_out):
    A_chunks = []
    B_chunks = []
    gate_logit_chunks = []
    gate_prob_chunks = []
    native_raw_chunks = []
    sel_delta_pred_chunks = []
    sel_exact_delta_chunks = []
    max_delta_pred_chunks = []
    max_delta_exact_chunks = []

    cand_delta_pred = []
    cand_exact_delta = []
    cand_selector_score = []
    cand_A = []

    where_better = []
    where_tie = []
    where_worse = []
    where_top_tie = []
    where_exact_idx = []
    where_regret = []

    utility_chunks = []
    friction_chunks = []
    valid_chunks = []
    raw_chunks = []
    current_k_chunks = []
    selector_k_chunks = []
    oracle_alt_k_chunks = []
    oracle_k_chunks = []
    delta_k_chunks = []
    query_rows = []

    zero_ref = None

    for i, path in enumerate(paths):
        fr = load_frame(path, device)
        z = fr["zero"]
        if zero_ref is None:
            zero_ref = z
        elif z != zero_ref:
            raise RuntimeError("zero_index changed across cache files.")

        valid_nk_t = torch.from_numpy(fr["valid"].T).to(device)
        tokens = (fr["tokens"] - token_mean) / token_std.clamp_min(1e-5)
        out = model(tokens, valid_nk_t, z)
        gate_logit = out["gate_logit"].detach().cpu().numpy()
        gate_prob = 1.0 / (1.0 + np.exp(-np.clip(gate_logit, -50.0, 50.0)))
        selector = out["selector_logits"].detach().cpu().numpy()
        delta = out["delta_pred"].detach().cpu().numpy()

        u = fr["utility"]
        v = fr["valid"]
        raw = fr["raw"]
        native_u = u[z]
        n = u.shape[1]

        alt_valid_nk = v.T.copy()
        alt_valid_nk[:, z] = False
        has_alt = alt_valid_nk.any(axis=1)

        sel_score = np.where(alt_valid_nk, selector, -np.inf)
        sel_k = np.argmax(sel_score, axis=1).astype(np.int64)
        sel_k = np.where(has_alt, sel_k, z)
        sel_exact_delta = u[sel_k, np.arange(n)] - native_u
        sel_delta_pred = delta[np.arange(n), sel_k]

        delta_score = np.where(alt_valid_nk, delta, -np.inf)
        delta_k = np.argmax(delta_score, axis=1).astype(np.int64)
        delta_k = np.where(has_alt, delta_k, z)
        max_delta_pred = delta[np.arange(n), delta_k]
        max_delta_exact = u[delta_k, np.arange(n)] - native_u

        oracle_alt_k, best_alt_u, _ = exact_best_alt(u, v, raw, z)
        A = best_alt_u > native_u + EPS
        B = has_alt & (sel_exact_delta > EPS)
        oracle_k = select_exact_oracle(u, raw, v)
        current_k, _, _ = select_relational_correction(
            gate_logit, selector, v.T, z, move_threshold
        )

        exact_delta_nk = (u - native_u[None, :]).T
        alt_mask = alt_valid_nk
        cand_delta_pred.append(delta[alt_mask])
        cand_exact_delta.append(exact_delta_nk[alt_mask])
        cand_selector_score.append(selector[alt_mask])
        cand_A.append(np.broadcast_to(A[:, None], alt_mask.shape)[alt_mask])

        if A.any():
            chosen_u = u[sel_k[A], np.arange(n)[A]]
            best_u = best_alt_u[A]
            native_A = native_u[A]
            where_better.append(chosen_u > native_A + EPS)
            where_tie.append(np.abs(chosen_u - native_A) <= EPS)
            where_worse.append(chosen_u < native_A - EPS)
            where_top_tie.append(np.abs(chosen_u - best_u) <= EPS)
            where_exact_idx.append(sel_k[A] == oracle_alt_k[A])
            where_regret.append(best_u - chosen_u)

        A_chunks.append(A)
        B_chunks.append(B)
        gate_logit_chunks.append(gate_logit)
        gate_prob_chunks.append(gate_prob)
        native_raw_chunks.append(raw[z])
        sel_delta_pred_chunks.append(sel_delta_pred)
        sel_exact_delta_chunks.append(sel_exact_delta)
        max_delta_pred_chunks.append(max_delta_pred)
        max_delta_exact_chunks.append(max_delta_exact)

        utility_chunks.append(u)
        friction_chunks.append(fr["friction"])
        valid_chunks.append(v)
        raw_chunks.append(raw)
        current_k_chunks.append(current_k)
        selector_k_chunks.append(sel_k)
        oracle_alt_k_chunks.append(oracle_alt_k)
        oracle_k_chunks.append(oracle_k)
        delta_k_chunks.append(delta_k)

        if ARGS.save_per_query:
            offs = fr["offsets"]
            for q in range(n):
                query_rows.append({
                    "split": name,
                    "scene_id": fr["scene_id"],
                    "anno_id": fr["anno_id"],
                    "query_id": q,
                    "A_any_better": int(A[q]),
                    "B_selector_better": int(B[q]),
                    "gate_logit": float(gate_logit[q]),
                    "gate_prob": float(gate_prob[q]),
                    "native_raw_score": float(raw[z, q]),
                    "native_utility": float(native_u[q]),
                    "selector_k": int(sel_k[q]),
                    "selector_offset_mm": float(offs[sel_k[q]]),
                    "selector_exact_delta": float(sel_exact_delta[q]),
                    "selector_delta_pred": float(sel_delta_pred[q]),
                    "delta_argmax_k": int(delta_k[q]),
                    "delta_argmax_offset_mm": float(offs[delta_k[q]]),
                    "delta_argmax_pred": float(max_delta_pred[q]),
                    "delta_argmax_exact_delta": float(max_delta_exact[q]),
                    "oracle_alt_k": int(oracle_alt_k[q]),
                    "oracle_alt_offset_mm": float(offs[oracle_alt_k[q]]),
                    "oracle_alt_delta": float(best_alt_u[q] - native_u[q]),
                    "current_selected_k": int(current_k[q]),
                })

        if ARGS.progress_every > 0 and (i + 1) % ARGS.progress_every == 0:
            print("[REL-AUDIT][%s] %d/%d frames" % (name, i + 1, len(paths)), flush=True)

    z = int(zero_ref)
    utility = np.concatenate(utility_chunks, axis=1)
    friction = np.concatenate(friction_chunks, axis=1)
    valid = np.concatenate(valid_chunks, axis=1)
    raw = np.concatenate(raw_chunks, axis=1)
    current_k = np.concatenate(current_k_chunks)
    selector_k = np.concatenate(selector_k_chunks)
    oracle_alt_k = np.concatenate(oracle_alt_k_chunks)
    oracle_k = np.concatenate(oracle_k_chunks)
    delta_k = np.concatenate(delta_k_chunks)

    A = np.concatenate(A_chunks)
    B = np.concatenate(B_chunks)
    gate_logit = np.concatenate(gate_logit_chunks)
    gate_prob = np.concatenate(gate_prob_chunks)
    native_raw = np.concatenate(native_raw_chunks)
    sel_delta_pred = np.concatenate(sel_delta_pred_chunks)
    sel_exact_delta = np.concatenate(sel_exact_delta_chunks)
    max_delta_pred = np.concatenate(max_delta_pred_chunks)
    max_delta_exact = np.concatenate(max_delta_exact_chunks)

    cdp = np.concatenate(cand_delta_pred)
    ced = np.concatenate(cand_exact_delta)
    css = np.concatenate(cand_selector_score)
    ca = np.concatenate(cand_A)
    beneficial = ced > EPS

    gate_metrics = {
        "target_positive_fraction": safe_mean(A),
        "auroc": auroc(A, gate_logit),
        "auprc": average_precision(A, gate_logit),
        "checkpoint_operating_point": binary_at_threshold(A, gate_prob, move_threshold),
        "spearman_gate_vs_native_raw": spearman(gate_logit, native_raw),
    }
    action_metrics = {
        "target_positive_fraction": safe_mean(B),
        "gate_score_for_B_auroc": auroc(B, gate_logit),
        "gate_score_for_B_auprc": average_precision(B, gate_logit),
        "selected_delta_for_B_auroc": auroc(B, sel_delta_pred),
        "selected_delta_for_B_auprc": average_precision(B, sel_delta_pred),
        "selected_delta_pearson_exact": pearson(sel_delta_pred, sel_exact_delta),
        "selected_delta_spearman_exact": spearman(sel_delta_pred, sel_exact_delta),
    }
    candidate_metrics = {
        "num_alternative_candidates": int(ced.size),
        "beneficial_fraction": safe_mean(beneficial),
        "delta_mae": safe_mean(np.abs(cdp - ced)),
        "delta_pearson": pearson(cdp, ced),
        "delta_spearman": spearman(cdp, ced),
        "delta_sign_auroc": auroc(beneficial, cdp),
        "delta_sign_auprc": average_precision(beneficial, cdp),
        "selector_score_vs_exact_delta_spearman_all": spearman(css, ced),
    }
    if ca.any():
        candidate_metrics.update({
            "selector_score_vs_exact_delta_spearman_A_positive": spearman(css[ca], ced[ca]),
            "selector_beneficial_auroc_A_positive": auroc(beneficial[ca], css[ca]),
            "selector_beneficial_auprc_A_positive": average_precision(beneficial[ca], css[ca]),
        })

    where = {
        "num_A_positive": int(A.sum()),
        "chosen_beneficial_fraction_given_A": safe_mean(np.concatenate(where_better)) if where_better else float("nan"),
        "chosen_tie_native_fraction_given_A": safe_mean(np.concatenate(where_tie)) if where_tie else float("nan"),
        "chosen_harmful_fraction_given_A": safe_mean(np.concatenate(where_worse)) if where_worse else float("nan"),
        "chosen_exact_best_index_fraction_given_A": safe_mean(np.concatenate(where_exact_idx)) if where_exact_idx else float("nan"),
        "chosen_top_utility_tie_fraction_given_A": safe_mean(np.concatenate(where_top_tie)) if where_top_tie else float("nan"),
        "mean_alt_selection_regret_given_A": safe_mean(np.concatenate(where_regret)) if where_regret else float("nan"),
    }

    native_k = np.full(A.shape, z, np.int64)
    current_gate = gate_prob > move_threshold
    policies = {
        "native": policy_summary(native_k, utility, friction, z, oracle_k),
        "current_gate_plus_selector": policy_summary(current_k, utility, friction, z, oracle_k),
        "always_selector": policy_summary(selector_k, utility, friction, z, oracle_k),
        "oracle_A_plus_selector": policy_summary(np.where(A, selector_k, z), utility, friction, z, oracle_k),
        "current_gate_plus_oracle_alt": policy_summary(np.where(current_gate, oracle_alt_k, z), utility, friction, z, oracle_k),
        "oracle_A_plus_oracle_alt": policy_summary(np.where(A, oracle_alt_k, z), utility, friction, z, oracle_k),
        "delta_argmax_threshold0": policy_summary(np.where(max_delta_pred > 0.0, delta_k, z), utility, friction, z, oracle_k),
        "selector_plus_selected_delta_threshold0": policy_summary(np.where(sel_delta_pred > 0.0, selector_k, z), utility, friction, z, oracle_k),
        "current_gate_and_selected_delta_threshold0": policy_summary(np.where(current_gate & (sel_delta_pred > 0.0), selector_k, z), utility, friction, z, oracle_k),
    }

    best_sel_delta, sweep_sel = sweep_policy(
        sel_delta_pred,
        lambda t: np.where(sel_delta_pred > t, selector_k, z),
        utility, friction, z, oracle_k,
    )
    best_delta_argmax, sweep_argmax = sweep_policy(
        max_delta_pred,
        lambda t: np.where(max_delta_pred > t, delta_k, z),
        utility, friction, z, oracle_k,
    )
    best_gate_delta, sweep_gate_delta = sweep_policy(
        sel_delta_pred,
        lambda t: np.where(current_gate & (sel_delta_pred > t), selector_k, z),
        utility, friction, z, oracle_k,
    )

    split_out.mkdir(parents=True, exist_ok=True)
    write_rows(split_out / "sweep_selector_selected_delta.csv", sweep_sel)
    write_rows(split_out / "sweep_delta_argmax.csv", sweep_argmax)
    write_rows(split_out / "sweep_current_gate_plus_selected_delta.csv", sweep_gate_delta)
    if ARGS.save_per_query and query_rows:
        with gzip.open(split_out / "per_query_audit.csv.gz", "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(query_rows[0].keys()))
            w.writeheader()
            w.writerows(query_rows)

    return {
        "split": name,
        "num_frames": len(paths),
        "num_queries": int(A.size),
        "checkpoint_move_threshold": float(move_threshold),
        "A_opportunity_gate": gate_metrics,
        "B_selected_action_confidence": action_metrics,
        "candidate_heads": candidate_metrics,
        "where_selector": where,
        "policies": policies,
        "delta_policy_diagnostics": {
            "best_selector_selected_delta": best_sel_delta,
            "best_delta_argmax": best_delta_argmax,
            "best_current_gate_plus_selected_delta": best_gate_delta,
            "note": "Thresholds are optimized on this diagnostic split only; they are not held-out results.",
        },
    }


def pct(x):
    try:
        return "%.2f%%" % (100.0 * float(x))
    except Exception:
        return "nan"


def num(x):
    try:
        x = float(x)
        return "nan" if not np.isfinite(x) else "%.4f" % x
    except Exception:
        return str(x)


def write_report(path, checkpoint, train, val):
    lines = [
        "# Relational Selective Two-Level Decision Audit",
        "",
        "- checkpoint: %s" % checkpoint,
        "- cache-only; no Stage-1 forward, CAD/DexNet call, or weight update",
        "",
        "## A. Opportunity gate: does any better alternative exist?",
        "",
        "| Split | A positive | Gate AUROC | Gate AUPRC | Precision@ckpt | Recall@ckpt | FPR@ckpt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("train", train), ("val", val)):
        m = result["A_opportunity_gate"]
        op = m["checkpoint_operating_point"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            name, pct(m["target_positive_fraction"]), num(m["auroc"]), num(m["auprc"]),
            pct(op["precision"]), pct(op["recall"]), pct(op["fpr"])
        ))

    lines += [
        "",
        "## B. Is the selector-proposed action actually better?",
        "",
        "| Split | B positive | Gate->B AUROC | Delta->B AUROC | Delta Spearman exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in (("train", train), ("val", val)):
        m = result["B_selected_action_confidence"]
        lines.append("| %s | %s | %s | %s | %s |" % (
            name, pct(m["target_positive_fraction"]), num(m["gate_score_for_B_auroc"]),
            num(m["selected_delta_for_B_auroc"]), num(m["selected_delta_spearman_exact"])
        ))

    lines += [
        "",
        "## C. Conditional selector on A-positive rays",
        "",
        "| Split | Chosen better | Tie native | Chosen worse | Top-utility tie | Mean regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("train", train), ("val", val)):
        m = result["where_selector"]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            name, pct(m["chosen_beneficial_fraction_given_A"]),
            pct(m["chosen_tie_native_fraction_given_A"]),
            pct(m["chosen_harmful_fraction_given_A"]),
            pct(m["chosen_top_utility_tie_fraction_given_A"]),
            num(m["mean_alt_selection_regret_given_A"])
        ))

    lines += [
        "",
        "## Counterfactual policy decomposition",
        "",
        "| Split | Policy | dUtility | dS@0.8 | Rescue | Harm | Headroom recovery |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    keys = (
        "current_gate_plus_selector",
        "oracle_A_plus_selector",
        "current_gate_plus_oracle_alt",
        "oracle_A_plus_oracle_alt",
        "delta_argmax_threshold0",
        "selector_plus_selected_delta_threshold0",
        "current_gate_and_selected_delta_threshold0",
    )
    for name, result in (("train", train), ("val", val)):
        for key in keys:
            m = result["policies"][key]
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                name, key, num(m["utility_gain"]), pct(m["success08_gain"]),
                pct(m["rescue08"]), pct(m["harm08"]), pct(m["utility_headroom_recovery"])
            ))

    lines += [
        "",
        "## Interpretation gate",
        "",
        "- Train gate strong, val gate weak: opportunity recognition fails to generalize.",
        "- Gate weak on both: current frozen evidence/gate does not decode opportunity.",
        "- Selector strong on train, weak on val: center ranking is the generalization bottleneck.",
        "- Delta->B is materially stronger than Gate->A/B and its validation counterfactual improves utility: deployment target mismatch; action-specific gating is worth testing.",
        "- Delta also weak on val: changing only threshold/deployment logic is unlikely to solve the problem.",
    ]
    path.write_text("\\n".join(lines) + "\\n")


def main():
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    checkpoint = Path(ARGS.selector_checkpoint)
    ckpt, model, mean, std, threshold = load_model(checkpoint, device)

    train_paths = filtered_paths(Path(ARGS.train_cache_root), True)
    val_paths = filtered_paths(Path(ARGS.val_cache_root), False)
    if ARGS.max_train_frames > 0:
        train_paths = train_paths[:ARGS.max_train_frames]
    if ARGS.max_val_frames > 0:
        val_paths = val_paths[:ARGS.max_val_frames]
    if not train_paths or not val_paths:
        raise RuntimeError("Need both train and validation caches.")

    out = Path(ARGS.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("[REL-AUDIT] checkpoint=%s epoch=%s threshold=%.4f" % (
        checkpoint, ckpt.get("epoch"), threshold
    ), flush=True)
    print("[REL-AUDIT] train=%d val=%d device=%s" % (
        len(train_paths), len(val_paths), device
    ), flush=True)

    train = analyze_split("train", train_paths, model, mean, std, threshold, device, out / "train")
    val = analyze_split("val", val_paths, model, mean, std, threshold, device, out / "val")

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_move_threshold": threshold,
        "train_cache_root": str(Path(ARGS.train_cache_root).resolve()),
        "val_cache_root": str(Path(ARGS.val_cache_root).resolve()),
        "train": train,
        "val": val,
        "notes": {
            "cache_only": True,
            "A": "any valid non-native exact utility > native exact utility",
            "B": "selector-chosen alternative exact utility > native exact utility",
            "counterfactual_threshold_sweeps_are_diagnostics_not_heldout_results": True,
        },
    }
    with (out / "decision_audit.json").open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    write_report(out / "REPORT.md", checkpoint, train, val)

    print(json.dumps({
        "checkpoint_epoch": result["checkpoint_epoch"],
        "move_threshold": threshold,
        "train_A_gate_AUROC": train["A_opportunity_gate"]["auroc"],
        "val_A_gate_AUROC": val["A_opportunity_gate"]["auroc"],
        "train_delta_to_B_AUROC": train["B_selected_action_confidence"]["selected_delta_for_B_auroc"],
        "val_delta_to_B_AUROC": val["B_selected_action_confidence"]["selected_delta_for_B_auroc"],
        "train_where_better_given_A": train["where_selector"]["chosen_beneficial_fraction_given_A"],
        "val_where_better_given_A": val["where_selector"]["chosen_beneficial_fraction_given_A"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
