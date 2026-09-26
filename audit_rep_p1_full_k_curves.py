#!/usr/bin/env python3
"""Audit full K-ray utility curves for Rep-P1 evidence probes.

This diagnostic never regenerates grasp actions and never calls the CAD/DexNet
evaluator.  It reuses the formal Rep-P0 K=7 physical-action cache and evaluates
trained Rep-P1 probes on every fixed candidate:

    [-40, -20, -10, 0, +10, +20, +40] mm

For each query it stores the exact K-curve and every requested predicted K-curve,
then derives diagnostics that are hidden by one selected action:

- within-ray utility-curve Spearman correlation;
- native-vs-alternative sign accuracy for every offset;
- endpoint direction accuracy;
- raw argmax / oracle-offset confusion;
- boundary collapse / boundary overreach;
- predicted-advantage calibration against exact gain;
- checkpoint-margin rescue/harm and selected-offset statistics;
- img_point vs img_region policy disagreement.

The canonical output is curves.npz.  All CSV/JSON files are deterministic
derivatives of that paired full-K artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from rep_p0_geometry_common import success08
from rep_p1_common import (
    IMAGE_VARIANTS,
    REP_P1_VARIANTS,
    REP_P1_VERSION,
    build_probe,
    cache_paths,
    image_cache_path,
    load_image_frame,
    load_p0_frame,
    predicted_utility_from_logits,
)

EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p0_cache_root", required=True)
    p.add_argument("--image_cache_root", default="")
    p.add_argument("--train_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument(
        "--variants",
        default="action_only,geo_pred,img_point,img_region",
    )
    p.add_argument(
        "--checkpoint_kind",
        default="latest",
        choices=("best", "latest"),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--focus_top_n", type=int, default=200)
    return p.parse_args()


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pearson(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return pearson(
        rankdata(x, method="average"),
        rankdata(y, method="average"),
    )


def safe_nanmean(values):
    values = np.asarray(values, np.float64)
    good = np.isfinite(values)
    return float(values[good].mean()) if bool(good.any()) else float("nan")


def safe_nanmedian(values):
    values = np.asarray(values, np.float64)
    good = np.isfinite(values)
    return float(np.median(values[good])) if bool(good.any()) else float("nan")


def gather_nk(arr_nk, k_n):
    arr = np.asarray(arr_nk)
    ids = np.asarray(k_n, np.int64)
    return arr[np.arange(len(ids)), ids]


def load_probe(path, device):
    ck = torch.load(path, map_location="cpu")
    if ck.get("version") != REP_P1_VERSION:
        raise RuntimeError(
            f"{path} is not a Rep-P1 checkpoint: {ck.get('version')!r}"
        )
    variant = str(ck["variant"])
    model = build_probe(
        variant,
        feature_dim=int(ck["feature_dim"]),
        hidden=int(ck["hidden_dim"]),
        dropout=float(ck["dropout"]),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    mean = (
        ck["feature_mean"].cpu().numpy().astype(np.float32)
        if ck.get("feature_mean") is not None else None
    )
    std = (
        ck["feature_std"].cpu().numpy().astype(np.float32)
        if ck.get("feature_std") is not None else None
    )
    return {
        "model": model,
        "epoch": int(ck.get("epoch", -1)),
        "margin": float(ck.get("selection_margin", 0.0)),
        "feature_mean": mean,
        "feature_std": std,
        "feature_dim": int(ck["feature_dim"]),
    }


def score_variant(variant, state, p0, image, device):
    model = state["model"]
    if variant == "action_only":
        actions = torch.from_numpy(p0["actions"]).to(device)
        logits = model(actions)
    elif variant == "geo_pred":
        feat = p0["feat_pred"]
        mean, std = state["feature_mean"], state["feature_std"]
        if mean is None or std is None:
            raise RuntimeError("geo_pred checkpoint missing feature normalization")
        x = (feat - mean) / np.maximum(std, 1e-5)
        K, Q, Fdim = x.shape
        x = torch.from_numpy(x.astype(np.float32)).to(device)
        logits = model(x.reshape(K * Q, Fdim)).reshape(K, Q, -1)
    elif variant in IMAGE_VARIANTS:
        feature = torch.from_numpy(image["pre_feature"]).unsqueeze(0).to(device)
        Kcam = torch.from_numpy(image["K"]).to(device)
        actions = torch.from_numpy(p0["actions"]).to(device)
        valid = torch.from_numpy(p0["valid"]).to(device)
        logits, _ = model(
            feature,
            Kcam,
            tuple(image["image_hw"].tolist()),
            actions,
            valid,
        )
    else:
        raise ValueError(variant)
    return predicted_utility_from_logits(logits).cpu().numpy().astype(np.float32)


def exact_oracle_k(exact, valid, zero):
    masked = np.where(valid, exact, -np.inf)
    arg = np.argmax(masked, axis=1)
    best = gather_nk(masked, arg)
    native = exact[:, zero]
    return np.where(best > native + EPS, arg, zero).astype(np.int16)


def raw_argmax_k(pred, valid):
    return np.argmax(np.where(valid, pred, -np.inf), axis=1).astype(np.int16)


def policy_selection(pred, valid, zero, margin):
    alt = valid.copy()
    alt[:, zero] = False
    score = np.where(alt, pred, -np.inf)
    best_alt = np.argmax(score, axis=1).astype(np.int16)
    has_alt = alt.any(axis=1)
    best_alt = np.where(has_alt, best_alt, zero).astype(np.int16)
    adv = gather_nk(pred, best_alt) - pred[:, zero]
    selected = np.where(
        has_alt & (adv > float(margin)), best_alt, zero
    ).astype(np.int16)
    return selected, best_alt, adv.astype(np.float32)


def per_query_curve_spearman(pred, exact, valid):
    out = np.full(len(pred), np.nan, np.float32)
    for i in range(len(pred)):
        m = valid[i]
        if int(m.sum()) < 2:
            continue
        out[i] = spearman(pred[i, m], exact[i, m])
    return out


def top10_mask(scene, anno, native_score):
    mask = np.zeros(len(native_score), dtype=bool)
    # Each frame contributes its own Stage-1 Top-10 queries.
    key = np.stack((scene, anno), axis=1)
    _, starts = np.unique(key, axis=0, return_index=True)
    order = np.argsort(starts)
    starts = starts[order]
    ends = np.r_[starts[1:], len(scene)]
    for start, end in zip(starts, ends):
        local = np.argsort(-native_score[start:end], kind="stable")
        local = local[: min(10, len(local))]
        mask[start + local] = True
    return mask


def subset_masks(oracle_k, zero, top10):
    headroom = oracle_k != zero
    return {
        "all": np.ones(len(oracle_k), dtype=bool),
        "top10": np.asarray(top10, bool),
        "headroom": headroom,
        "no_headroom": ~headroom,
    }


def sign_accuracy(pred_delta, exact_delta):
    comparable = np.abs(exact_delta) > EPS
    if not bool(comparable.any()):
        return float("nan"), 0
    correct = (
        np.sign(pred_delta[comparable]) ==
        np.sign(exact_delta[comparable])
    )
    return float(correct.mean()), int(comparable.sum())


def curve_summary_row(
    *,
    split,
    checkpoint_kind,
    variant,
    state,
    mask,
    offsets,
    zero,
    exact,
    friction,
    valid,
    pred,
    oracle_k,
    query_spearman,
):
    ids = np.flatnonzero(mask)
    e = exact[ids]
    f = friction[ids]
    v = valid[ids]
    p = pred[ids]
    o = oracle_k[ids]
    native = e[:, zero]
    native_f = f[:, zero]
    native_success = success08(native_f)

    raw_k = raw_argmax_k(p, v)
    raw_exact = gather_nk(e, raw_k)
    raw_gain = raw_exact - native
    raw_success = success08(gather_nk(f, raw_k))

    sel_k, best_alt, advantage = policy_selection(
        p, v, zero, state["margin"]
    )
    sel_exact = gather_nk(e, sel_k)
    sel_gain = sel_exact - native
    sel_success = success08(gather_nk(f, sel_k))

    oracle_u = gather_nk(e, o)
    headroom = oracle_u - native
    mean_headroom = float(headroom.mean())

    native_pair_correct = native_pair_total = 0
    for k in range(len(offsets)):
        if k == zero:
            continue
        ok = v[:, k] & v[:, zero] & (np.abs(e[:, k] - native) > EPS)
        native_pair_total += int(ok.sum())
        if bool(ok.any()):
            native_pair_correct += int(
                (
                    np.sign(p[ok, k] - p[ok, zero]) ==
                    np.sign(e[ok, k] - native[ok])
                ).sum()
            )

    lo, hi = 0, len(offsets) - 1
    endpoint_ok = (
        v[:, lo] & v[:, hi] &
        (np.abs(e[:, hi] - e[:, lo]) > EPS)
    )
    endpoint_acc = float("nan")
    if bool(endpoint_ok.any()):
        endpoint_acc = float(
            (
                np.sign(p[endpoint_ok, hi] - p[endpoint_ok, lo]) ==
                np.sign(e[endpoint_ok, hi] - e[endpoint_ok, lo])
            ).mean()
        )

    pred_offsets = offsets[raw_k]
    oracle_offsets = offsets[o]
    oracle_nonzero = o != zero
    direction_ok = oracle_nonzero & (raw_k != zero)
    direction_acc = float("nan")
    if bool(direction_ok.any()):
        direction_acc = float(
            (
                np.sign(pred_offsets[direction_ok]) ==
                np.sign(oracle_offsets[direction_ok])
            ).mean()
        )

    boundary = (raw_k == lo) | (raw_k == hi)
    oracle_boundary = (o == lo) | (o == hi)
    boundary_overreach = boundary & (raw_k != o)

    best_alt_exact_gain = gather_nk(e, best_alt) - native
    adv_gain_spearman = spearman(advantage, best_alt_exact_gain)
    adv_gain_pearson = pearson(advantage, best_alt_exact_gain)

    exact_match = float((raw_k == o).mean())
    offset_mae = float(
        np.abs(offsets[raw_k] - offsets[o]).mean()
    )

    qsp = query_spearman[ids]
    return {
        "split": split,
        "checkpoint_kind": checkpoint_kind,
        "variant": variant,
        "subset": "",
        "checkpoint_epoch": state["epoch"],
        "selection_margin": state["margin"],
        "num_queries": int(len(ids)),
        "within_ray_spearman_mean": safe_nanmean(qsp),
        "within_ray_spearman_median": safe_nanmedian(qsp),
        "native_centered_pair_accuracy": (
            float(native_pair_correct / native_pair_total)
            if native_pair_total else float("nan")
        ),
        "native_centered_pair_count": int(native_pair_total),
        "endpoint_direction_accuracy": endpoint_acc,
        "endpoint_direction_count": int(endpoint_ok.sum()),
        "raw_argmax_exact_oracle_match": exact_match,
        "raw_argmax_offset_mae_mm": offset_mae,
        "raw_argmax_direction_accuracy_on_headroom": direction_acc,
        "raw_argmax_boundary_rate": float(boundary.mean()),
        "oracle_boundary_rate": float(oracle_boundary.mean()),
        "boundary_overreach_rate": float(boundary_overreach.mean()),
        "raw_argmax_utility_gain": float(raw_gain.mean()),
        "raw_argmax_success08_gain": float(
            (raw_success.astype(np.float32) -
             native_success.astype(np.float32)).mean()
        ),
        "raw_argmax_headroom_recovery": (
            float(raw_gain.mean() / mean_headroom)
            if abs(mean_headroom) > 1e-12 else float("nan")
        ),
        "best_alt_advantage_exact_gain_spearman": adv_gain_spearman,
        "best_alt_advantage_exact_gain_pearson": adv_gain_pearson,
        "policy_change_rate": float((sel_k != zero).mean()),
        "policy_utility_gain": float(sel_gain.mean()),
        "policy_success08_gain": float(
            (sel_success.astype(np.float32) -
             native_success.astype(np.float32)).mean()
        ),
        "policy_rescue08": float(((~native_success) & sel_success).mean()),
        "policy_harm08": float((native_success & (~sel_success)).mean()),
        "policy_headroom_recovery": (
            float(sel_gain.mean() / mean_headroom)
            if abs(mean_headroom) > 1e-12 else float("nan")
        ),
        "mean_oracle_headroom": mean_headroom,
    }


def offset_sign_rows(
    *,
    split,
    checkpoint_kind,
    variant,
    subset_name,
    mask,
    offsets,
    zero,
    exact,
    valid,
    pred,
):
    rows = []
    ids = np.flatnonzero(mask)
    e, v, p = exact[ids], valid[ids], pred[ids]
    native_e, native_p = e[:, zero], p[:, zero]
    for k, off in enumerate(offsets):
        if k == zero:
            continue
        ok = v[:, k] & v[:, zero]
        exact_delta = e[:, k] - native_e
        pred_delta = p[:, k] - native_p
        comparable = ok & (np.abs(exact_delta) > EPS)
        acc, count = sign_accuracy(
            pred_delta[ok], exact_delta[ok]
        )
        rows.append({
            "split": split,
            "checkpoint_kind": checkpoint_kind,
            "variant": variant,
            "subset": subset_name,
            "offset_mm": float(off),
            "valid_queries": int(ok.sum()),
            "comparable_queries": count,
            "sign_accuracy": acc,
            "exact_beneficial_rate": (
                float((exact_delta[ok] > EPS).mean())
                if bool(ok.any()) else float("nan")
            ),
            "pred_beneficial_rate": (
                float((pred_delta[ok] > 0).mean())
                if bool(ok.any()) else float("nan")
            ),
            "mean_exact_delta": (
                float(exact_delta[ok].mean())
                if bool(ok.any()) else float("nan")
            ),
            "mean_pred_delta": (
                float(pred_delta[ok].mean())
                if bool(ok.any()) else float("nan")
            ),
            "delta_spearman": (
                spearman(pred_delta[comparable], exact_delta[comparable])
                if bool(comparable.any()) else float("nan")
            ),
        })
    return rows


def argmax_confusion_rows(
    *,
    split,
    checkpoint_kind,
    variant,
    subset_name,
    mask,
    offsets,
    zero,
    exact,
    valid,
    pred,
):
    ids = np.flatnonzero(mask)
    e, v, p = exact[ids], valid[ids], pred[ids]
    oracle = exact_oracle_k(e, v, zero)
    raw = raw_argmax_k(p, v)
    rows = []
    for ok, off_o in enumerate(offsets):
        row_total = int((oracle == ok).sum())
        for pk, off_p in enumerate(offsets):
            count = int(((oracle == ok) & (raw == pk)).sum())
            rows.append({
                "split": split,
                "checkpoint_kind": checkpoint_kind,
                "variant": variant,
                "subset": subset_name,
                "oracle_offset_mm": float(off_o),
                "pred_argmax_offset_mm": float(off_p),
                "count": count,
                "row_fraction": (
                    float(count / row_total) if row_total else float("nan")
                ),
            })
    return rows


def policy_offset_rows(
    *,
    split,
    checkpoint_kind,
    variant,
    subset_name,
    mask,
    offsets,
    zero,
    exact,
    friction,
    valid,
    pred,
    margin,
):
    ids = np.flatnonzero(mask)
    e, f, v, p = exact[ids], friction[ids], valid[ids], pred[ids]
    selected, best_alt, advantage = policy_selection(
        p, v, zero, margin
    )
    native = e[:, zero]
    native_s = success08(f[:, zero])
    selected_u = gather_nk(e, selected)
    selected_s = success08(gather_nk(f, selected))
    best_alt_gain = gather_nk(e, best_alt) - native
    rows = []
    for k, off in enumerate(offsets):
        use = selected == k
        if not bool(use.any()):
            continue
        rows.append({
            "split": split,
            "checkpoint_kind": checkpoint_kind,
            "variant": variant,
            "subset": subset_name,
            "selected_offset_mm": float(off),
            "count": int(use.sum()),
            "fraction": float(use.mean()),
            "mean_exact_gain": float((selected_u[use] - native[use]).mean()),
            "rescue08": float(((~native_s[use]) & selected_s[use]).mean()),
            "harm08": float((native_s[use] & (~selected_s[use])).mean()),
            "mean_pred_advantage_best_alt": float(advantage[use].mean()),
            "mean_exact_gain_best_alt": float(best_alt_gain[use].mean()),
        })
    return rows


def advantage_bin_rows(
    *,
    split,
    checkpoint_kind,
    variant,
    subset_name,
    mask,
    exact,
    friction,
    valid,
    pred,
    zero,
):
    ids = np.flatnonzero(mask)
    e, f, v, p = exact[ids], friction[ids], valid[ids], pred[ids]
    _, best_alt, advantage = policy_selection(p, v, zero, -np.inf)
    native = e[:, zero]
    exact_gain = gather_nk(e, best_alt) - native
    native_s = success08(f[:, zero])
    alt_s = success08(gather_nk(f, best_alt))

    finite = np.isfinite(advantage)
    if not bool(finite.any()):
        return []
    values = advantage[finite]
    # Equal-count bins are representation-scale invariant.
    edges = np.quantile(values, np.linspace(0.0, 1.0, 11))
    rows = []
    for b in range(10):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == 9:
            use = finite & (advantage >= lo) & (advantage <= hi)
        else:
            use = finite & (advantage >= lo) & (advantage < hi)
        if not bool(use.any()):
            continue
        rows.append({
            "split": split,
            "checkpoint_kind": checkpoint_kind,
            "variant": variant,
            "subset": subset_name,
            "advantage_decile": b,
            "count": int(use.sum()),
            "advantage_min": lo,
            "advantage_max": hi,
            "mean_pred_advantage": float(advantage[use].mean()),
            "mean_exact_gain_best_alt": float(exact_gain[use].mean()),
            "beneficial_rate": float((exact_gain[use] > EPS).mean()),
            "rescue08_if_best_alt": float(
                ((~native_s[use]) & alt_s[use]).mean()
            ),
            "harm08_if_best_alt": float(
                (native_s[use] & (~alt_s[use])).mean()
            ),
        })
    return rows


def focus_rows(
    *,
    split,
    variant,
    scene,
    anno,
    query_id,
    native_score,
    offsets,
    zero,
    exact,
    friction,
    valid,
    pred,
    margin,
    top_n,
):
    selected, best_alt, advantage = policy_selection(
        pred, valid, zero, margin
    )
    native = exact[:, zero]
    gain = gather_nk(exact, selected) - native
    changed = selected != zero
    if not bool(changed.any()):
        return [], []
    ids = np.flatnonzero(changed)
    harmful = ids[np.argsort(gain[ids], kind="stable")[:top_n]]
    beneficial = ids[
        np.argsort(-gain[ids], kind="stable")[:top_n]
    ]

    def encode_curve(row):
        return "|".join(f"{float(x):.6f}" for x in row)

    def make(indices):
        rows = []
        for i in indices:
            rows.append({
                "split": split,
                "variant": variant,
                "scene_id": int(scene[i]),
                "anno_id": int(anno[i]),
                "query_id": int(query_id[i]),
                "native_score": float(native_score[i]),
                "selected_offset_mm": float(offsets[selected[i]]),
                "best_alt_offset_mm": float(offsets[best_alt[i]]),
                "pred_advantage": float(advantage[i]),
                "exact_gain": float(gain[i]),
                "native_exact_utility": float(native[i]),
                "selected_exact_utility": float(
                    exact[i, selected[i]]
                ),
                "exact_curve": encode_curve(exact[i]),
                "pred_curve": encode_curve(pred[i]),
                "valid_curve": "|".join(
                    str(int(x)) for x in valid[i]
                ),
            })
        return rows

    return make(harmful), make(beneficial)


def point_region_disagreement_rows(
    *,
    split,
    scene,
    anno,
    query_id,
    native_score,
    offsets,
    zero,
    exact,
    friction,
    valid,
    pred_point,
    pred_region,
    state_point,
    state_region,
):
    sp, bp, ap = policy_selection(
        pred_point, valid, zero, state_point["margin"]
    )
    sr, br, ar = policy_selection(
        pred_region, valid, zero, state_region["margin"]
    )
    disagreement = sp != sr
    ids = np.flatnonzero(disagreement)
    native = exact[:, zero]
    native_s = success08(friction[:, zero])
    rows = []
    for i in ids:
        up = exact[i, sp[i]]
        ur = exact[i, sr[i]]
        fp = friction[i, sp[i]]
        fr = friction[i, sr[i]]
        rows.append({
            "split": split,
            "scene_id": int(scene[i]),
            "anno_id": int(anno[i]),
            "query_id": int(query_id[i]),
            "native_score": float(native_score[i]),
            "native_exact_utility": float(native[i]),
            "oracle_exact_utility": float(np.nanmax(
                np.where(valid[i], exact[i], np.nan)
            )),
            "point_selected_offset_mm": float(offsets[sp[i]]),
            "region_selected_offset_mm": float(offsets[sr[i]]),
            "point_pred_advantage": float(ap[i]),
            "region_pred_advantage": float(ar[i]),
            "point_exact_gain": float(up - native[i]),
            "region_exact_gain": float(ur - native[i]),
            "point_rescue08": int(
                (not bool(native_s[i])) and bool(success08(np.asarray([fp]))[0])
            ),
            "point_harm08": int(
                bool(native_s[i]) and (not bool(success08(np.asarray([fp]))[0]))
            ),
            "region_rescue08": int(
                (not bool(native_s[i])) and bool(success08(np.asarray([fr]))[0])
            ),
            "region_harm08": int(
                bool(native_s[i]) and (not bool(success08(np.asarray([fr]))[0]))
            ),
        })
    return rows


@torch.no_grad()
def main():
    args = parse_args()
    variants = tuple(
        x.strip() for x in args.variants.split(",") if x.strip()
    )
    unknown = sorted(set(variants) - set(REP_P1_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown Rep-P1 variants: {unknown}")
    if not variants:
        raise ValueError("No variants requested")
    if any(v in IMAGE_VARIANTS for v in variants) and not args.image_cache_root:
        raise ValueError("Image variants require --image_cache_root")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    states = {}
    for variant in variants:
        ckpt = (
            Path(args.train_root) / variant /
            f"checkpoint_{args.checkpoint_kind}.tar"
        )
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        state = load_probe(ckpt, device)
        if state["model"].__class__.__name__ == "":
            raise RuntimeError("Impossible empty model class")
        states[variant] = state

    p0_root = Path(args.p0_cache_root)
    paths = cache_paths(p0_root)
    if args.max_frames > 0:
        paths = paths[: args.max_frames]
    if not paths:
        raise RuntimeError(f"No Rep-P0 caches under {p0_root}")

    need_geo = "geo_pred" in variants
    need_image = any(v in IMAGE_VARIANTS for v in variants)

    exact_chunks = []
    friction_chunks = []
    valid_chunks = []
    scene_chunks = []
    anno_chunks = []
    query_chunks = []
    native_score_chunks = []
    top10_chunks = []
    pred_chunks = {v: [] for v in variants}
    offsets_ref = None
    zero_ref = None

    for frame_i, path in enumerate(paths):
        p0 = load_p0_frame(path, need_geo=need_geo)
        if offsets_ref is None:
            offsets_ref = p0["offsets"].astype(np.float32)
            zero_ref = int(p0["zero"])
        else:
            if (
                p0["zero"] != zero_ref or
                p0["offsets"].shape != offsets_ref.shape or
                not np.allclose(p0["offsets"], offsets_ref, atol=1e-7, rtol=0)
            ):
                raise RuntimeError(f"K-grid mismatch at {path}")

        image = None
        if need_image:
            ipath = image_cache_path(
                args.image_cache_root, path, p0_root
            )
            if not ipath.is_file():
                raise FileNotFoundError(ipath)
            image = load_image_frame(ipath, p0)

        Q = p0["actions"].shape[1]
        exact_chunks.append(p0["utility"].T.astype(np.float32))
        friction_chunks.append(p0["friction"].T.astype(np.float32))
        valid_chunks.append(p0["valid"].T.astype(bool))
        scene_chunks.append(
            np.full(Q, p0["scene_id"], dtype=np.int16)
        )
        anno_chunks.append(
            np.full(Q, p0["anno_id"], dtype=np.int16)
        )
        query_chunks.append(p0["query_ids"].astype(np.int32))
        native_score_chunks.append(p0["native_score"].astype(np.float32))
        local_top10 = np.zeros(Q, dtype=bool)
        rank = np.argsort(-p0["native_score"], kind="stable")
        local_top10[rank[: min(10, Q)]] = True
        top10_chunks.append(local_top10)

        for variant in variants:
            pred_kq = score_variant(
                variant, states[variant], p0, image, device
            )
            if pred_kq.shape != p0["utility"].shape:
                raise RuntimeError(
                    f"{variant} prediction shape {pred_kq.shape} != "
                    f"{p0['utility'].shape}"
                )
            pred_chunks[variant].append(pred_kq.T.astype(np.float32))

        if args.progress_every > 0 and (frame_i + 1) % args.progress_every == 0:
            print(
                f"[REP-P1-FULL-K] {args.split} "
                f"{frame_i+1}/{len(paths)} frames",
                flush=True,
            )

    exact = np.concatenate(exact_chunks, axis=0)
    friction = np.concatenate(friction_chunks, axis=0)
    valid = np.concatenate(valid_chunks, axis=0)
    scene = np.concatenate(scene_chunks)
    anno = np.concatenate(anno_chunks)
    query_id = np.concatenate(query_chunks)
    native_score = np.concatenate(native_score_chunks)
    top10 = np.concatenate(top10_chunks)
    pred = {
        v: np.concatenate(pred_chunks[v], axis=0)
        for v in variants
    }
    offsets = np.asarray(offsets_ref, np.float32)
    zero = int(zero_ref)

    oracle_k = exact_oracle_k(exact, valid, zero)
    masks = subset_masks(oracle_k, zero, top10)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "version": np.asarray(REP_P1_VERSION),
        "split": np.asarray(args.split),
        "checkpoint_kind": np.asarray(args.checkpoint_kind),
        "offsets_mm": offsets,
        "zero_index": np.asarray(zero, np.int16),
        "scene_id": scene,
        "anno_id": anno,
        "query_id": query_id,
        "native_score": native_score,
        "top10_mask": top10,
        "valid": valid.astype(np.uint8),
        "exact_utility": exact,
        "friction": friction,
        "oracle_k": oracle_k,
    }
    for variant in variants:
        npz_payload[f"pred_{variant}"] = pred[variant]
        sel, best_alt, adv = policy_selection(
            pred[variant], valid, zero, states[variant]["margin"]
        )
        npz_payload[f"raw_argmax_k_{variant}"] = raw_argmax_k(
            pred[variant], valid
        )
        npz_payload[f"policy_selected_k_{variant}"] = sel
        npz_payload[f"best_alt_k_{variant}"] = best_alt
        npz_payload[f"best_alt_advantage_{variant}"] = adv
    np.savez_compressed(out / "curves.npz", **npz_payload)

    summary_rows = []
    offset_rows = []
    confusion_rows = []
    policy_rows = []
    advantage_rows = []
    harmful_rows = []
    beneficial_rows = []

    query_spearman = {
        v: per_query_curve_spearman(pred[v], exact, valid)
        for v in variants
    }

    for variant in variants:
        for subset_name, mask in masks.items():
            row = curve_summary_row(
                split=args.split,
                checkpoint_kind=args.checkpoint_kind,
                variant=variant,
                state=states[variant],
                mask=mask,
                offsets=offsets,
                zero=zero,
                exact=exact,
                friction=friction,
                valid=valid,
                pred=pred[variant],
                oracle_k=oracle_k,
                query_spearman=query_spearman[variant],
            )
            row["subset"] = subset_name
            summary_rows.append(row)
            offset_rows.extend(offset_sign_rows(
                split=args.split,
                checkpoint_kind=args.checkpoint_kind,
                variant=variant,
                subset_name=subset_name,
                mask=mask,
                offsets=offsets,
                zero=zero,
                exact=exact,
                valid=valid,
                pred=pred[variant],
            ))
            confusion_rows.extend(argmax_confusion_rows(
                split=args.split,
                checkpoint_kind=args.checkpoint_kind,
                variant=variant,
                subset_name=subset_name,
                mask=mask,
                offsets=offsets,
                zero=zero,
                exact=exact,
                valid=valid,
                pred=pred[variant],
            ))
            policy_rows.extend(policy_offset_rows(
                split=args.split,
                checkpoint_kind=args.checkpoint_kind,
                variant=variant,
                subset_name=subset_name,
                mask=mask,
                offsets=offsets,
                zero=zero,
                exact=exact,
                friction=friction,
                valid=valid,
                pred=pred[variant],
                margin=states[variant]["margin"],
            ))
            advantage_rows.extend(advantage_bin_rows(
                split=args.split,
                checkpoint_kind=args.checkpoint_kind,
                variant=variant,
                subset_name=subset_name,
                mask=mask,
                exact=exact,
                friction=friction,
                valid=valid,
                pred=pred[variant],
                zero=zero,
            ))

        harm, benefit = focus_rows(
            split=args.split,
            variant=variant,
            scene=scene,
            anno=anno,
            query_id=query_id,
            native_score=native_score,
            offsets=offsets,
            zero=zero,
            exact=exact,
            friction=friction,
            valid=valid,
            pred=pred[variant],
            margin=states[variant]["margin"],
            top_n=args.focus_top_n,
        )
        harmful_rows.extend(harm)
        beneficial_rows.extend(benefit)

    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "native_offset_accuracy.csv", offset_rows)
    write_csv(out / "argmax_confusion.csv", confusion_rows)
    write_csv(out / "policy_selected_offset_stats.csv", policy_rows)
    write_csv(out / "advantage_deciles.csv", advantage_rows)
    write_csv(out / "focus_harmful.csv", harmful_rows)
    write_csv(out / "focus_beneficial.csv", beneficial_rows)

    disagreement_rows = []
    if "img_point" in variants and "img_region" in variants:
        disagreement_rows = point_region_disagreement_rows(
            split=args.split,
            scene=scene,
            anno=anno,
            query_id=query_id,
            native_score=native_score,
            offsets=offsets,
            zero=zero,
            exact=exact,
            friction=friction,
            valid=valid,
            pred_point=pred["img_point"],
            pred_region=pred["img_region"],
            state_point=states["img_point"],
            state_region=states["img_region"],
        )
        write_csv(
            out / "img_point_vs_img_region_disagreements.csv",
            disagreement_rows,
        )

    meta = {
        "version": REP_P1_VERSION,
        "experiment": "Rep-P1 full-K curve audit",
        "split": args.split,
        "checkpoint_kind": args.checkpoint_kind,
        "variants": list(variants),
        "num_frames": len(paths),
        "num_queries": int(len(exact)),
        "offsets_mm": offsets.tolist(),
        "zero_index": zero,
        "checkpoints": {
            v: {
                "epoch": states[v]["epoch"],
                "selection_margin": states[v]["margin"],
                "path": str(
                    (
                        Path(args.train_root) / v /
                        f"checkpoint_{args.checkpoint_kind}.tar"
                    ).resolve()
                ),
            }
            for v in variants
        },
        "contracts": {
            "actions_and_labels":
                "read only from formal Rep-P0 cache; never regenerated",
            "curves":
                "all requested Rep-P1 variants score the same K=7 physical actions",
            "margin_free_metrics":
                "within-ray curves/raw argmax/native-centered sign use no selection margin",
            "policy_metrics":
                "use the checkpoint Seen-selected native-fallback margin",
        },
        "artifacts": [
            "curves.npz",
            "summary.csv",
            "native_offset_accuracy.csv",
            "argmax_confusion.csv",
            "policy_selected_offset_stats.csv",
            "advantage_deciles.csv",
            "focus_harmful.csv",
            "focus_beneficial.csv",
            "img_point_vs_img_region_disagreements.csv",
        ],
        "point_region_policy_disagreements": int(len(disagreement_rows)),
    }
    (out / "audit_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )

    print(
        f"[REP-P1-FULL-K] split={args.split} queries={len(exact)} "
        f"checkpoint={args.checkpoint_kind}"
    )
    for row in summary_rows:
        if row["subset"] not in ("all", "top10"):
            continue
        print(
            f"  {row['variant']:11s} {row['subset']:5s} "
            f"curve_rho={row['within_ray_spearman_mean']:+.3f} "
            f"native_pair={row['native_centered_pair_accuracy']:.3f} "
            f"raw_dU={row['raw_argmax_utility_gain']:+.5f} "
            f"policy_dU={row['policy_utility_gain']:+.5f} "
            f"boundary={100*row['raw_argmax_boundary_rate']:.1f}% "
            f"overreach={100*row['boundary_overreach_rate']:.1f}%"
        )


if __name__ == "__main__":
    main()
