#!/usr/bin/env python3
"""Evaluate a trained Rep-P1 evidence probe on fixed Rep-P0 actions.

This script never regenerates actions and never calls the CAD/DexNet evaluator.
It reports representation decodability and same-ray selection quality on the
already paired exact-action labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rep_p1_common import (
    IMAGE_VARIANTS,
    REP_P1_VERSION,
    build_probe,
    cache_paths,
    candidate_metrics,
    count_trainable_parameters,
    friction_to_cdf_targets,
    image_cache_path,
    load_image_frame,
    load_p0_frame,
    policy_metrics,
    predicted_utility_from_logits,
    select_policy,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p0_cache_root", required=True)
    p.add_argument("--image_cache_root", default="")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--save_per_query", action="store_true")
    return p.parse_args()


def normalize(x, mean, std):
    return (x - mean) / np.maximum(std, 1e-5)


def paired_image(path, p0_root, image_root, p0):
    ipath = image_cache_path(image_root, path, p0_root)
    if not ipath.is_file():
        raise FileNotFoundError(f"Missing Rep-P1 image cache: {ipath}")
    return load_image_frame(ipath, p0)


def score_frame(model, variant, p0, image, device, mean=None, std=None):\n    if variant == "action_only":\n        actions = torch.from_numpy(p0["actions"]).to(device)\n        logits = model(actions)\n        diag = {}\n    elif variant == "geo_pred":\n        K, Q, Fdim = p0["feat_pred"].shape
        x = torch.from_numpy(
            normalize(p0["feat_pred"], mean, std)
        ).to(device)
        logits = model(x.reshape(K * Q, Fdim)).reshape(K, Q, -1)
        diag = {}
    else:
        feature = torch.from_numpy(image["pre_feature"]).unsqueeze(0).to(device)
        Kcam = torch.from_numpy(image["K"]).to(device)
        actions = torch.from_numpy(p0["actions"]).to(device)
        valid = torch.from_numpy(p0["valid"]).to(device)
        logits, diag = model(
            feature, Kcam, tuple(image["image_hw"].tolist()), actions, valid
        )
    return logits, diag


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Keep optimizer/checkpoint payload on host memory; only model weights are copied to GPU.\n    ckpt = torch.load(args.checkpoint, map_location="cpu")\n    if ckpt.get("version") != REP_P1_VERSION:
        raise RuntimeError(
            f"Unsupported Rep-P1 checkpoint version {ckpt.get('version')!r}"
        )
    variant = str(ckpt["variant"])
    if variant in IMAGE_VARIANTS and not args.image_cache_root:
        raise ValueError("Image variant requires --image_cache_root")

    model = build_probe(
        variant,
        feature_dim=int(ckpt["feature_dim"]),
        hidden=int(ckpt["hidden_dim"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    mean = (
        ckpt["feature_mean"].cpu().numpy().astype(np.float32)
        if ckpt.get("feature_mean") is not None else None
    )
    std = (
        ckpt["feature_std"].cpu().numpy().astype(np.float32)
        if ckpt.get("feature_std") is not None else None
    )
    margin = float(
        ckpt.get("selection_margin", 0.0)
        if args.margin is None else args.margin
    )

    p0_root = Path(args.p0_cache_root)
    paths = cache_paths(p0_root)
    if args.max_frames > 0:
        paths = paths[: args.max_frames]
    if not paths:
        raise RuntimeError(f"No Rep-P0 cache under {p0_root}")

    frames = []
    cdf_num = cdf_den = 0.0
    per_frame = []
    per_query = []
    visible = []
    region_visible = []

    for i, path in enumerate(paths):
        p0 = load_p0_frame(path, need_geo=(variant == "geo_pred"))
        image = (
            paired_image(path, p0_root, args.image_cache_root, p0)
            if variant in IMAGE_VARIANTS else None
        )
        logits, diag = score_frame(
            model, variant, p0, image, device, mean, std
        )
        valid_t = torch.from_numpy(p0["valid"]).to(device)
        target = torch.from_numpy(
            friction_to_cdf_targets(p0["friction"])
        ).to(device)
        cdf = F.binary_cross_entropy_with_logits(
            logits[valid_t], target[valid_t], reduction="sum"
        )
        cdf_num += float(cdf)
        cdf_den += int(valid_t.sum()) * target.shape[-1]
        pred_u = predicted_utility_from_logits(logits).cpu().numpy().astype(np.float32)
        frame = {**p0, "pred_u": pred_u}
        frames.append(frame)

        if "visible_ratio" in diag:
            vr = diag["visible_ratio"][valid_t].float()
            visible.append(float(vr.mean().cpu()))
        if "region_visible_ratio" in diag:
            rv = diag["region_visible_ratio"][valid_t].float()
            region_visible.append(float(rv.mean().cpu()))

        selected, oracle, best, advantage = select_policy(
            pred_u, p0["utility"], p0["valid"], p0["zero"], margin
        )
        q = np.arange(pred_u.shape[1])
        u_sel = p0["utility"][selected, q]
        u_nat = p0["utility"][p0["zero"], q]
        u_orc = p0["utility"][oracle, q]
        f_sel = p0["friction"][selected, q]
        f_nat = p0["friction"][p0["zero"], q]
        s_sel = np.isfinite(f_sel) & (f_sel > 0) & (f_sel <= 0.8 + 1e-6)
        s_nat = np.isfinite(f_nat) & (f_nat > 0) & (f_nat <= 0.8 + 1e-6)

        per_frame.append({
            "variant": variant,
            "scene_id": p0["scene_id"],
            "anno_id": p0["anno_id"],
            "queries": len(q),
            "native_utility": float(u_nat.mean()),
            "selected_utility": float(u_sel.mean()),
            "oracle_utility": float(u_orc.mean()),
            "utility_gain": float((u_sel - u_nat).mean()),
            "native_success08": float(s_nat.mean()),
            "selected_success08": float(s_sel.mean()),
            "success08_gain": float(
                (s_sel.astype(np.float32) - s_nat.astype(np.float32)).mean()
            ),
            "rescue08": float(((~s_nat) & s_sel).mean()),
            "harm08": float((s_nat & (~s_sel)).mean()),
            "change_rate": float((selected != p0["zero"]).mean()),
        })

        if args.save_per_query:
            for qi in range(len(q)):
                per_query.append({
                    "variant": variant,
                    "scene_id": p0["scene_id"],
                    "anno_id": p0["anno_id"],
                    "query_id": int(p0["query_ids"][qi]),
                    "native_score": float(p0["native_score"][qi]),
                    "selected_k": int(selected[qi]),
                    "selected_offset_mm": float(p0["offsets"][selected[qi]]),
                    "best_pred_k": int(best[qi]),
                    "pred_advantage": float(advantage[qi]),
                    "pred_native_utility": float(pred_u[p0["zero"], qi]),
                    "pred_selected_utility": float(pred_u[selected[qi], qi]),
                    "native_exact_utility": float(u_nat[qi]),
                    "selected_exact_utility": float(u_sel[qi]),
                    "oracle_exact_utility": float(u_orc[qi]),
                    "native_success08": int(s_nat[qi]),
                    "selected_success08": int(s_sel[qi]),
                    "rescue08": int((not s_nat[qi]) and s_sel[qi]),
                    "harm08": int(s_nat[qi] and (not s_sel[qi])),
                })

        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(
                f"[REP-P1-TEST][{variant}] {i+1}/{len(paths)}",
                flush=True,
            )

    cand = candidate_metrics(frames)
    cand["cdf_bce"] = cdf_num / max(cdf_den, 1.0)
    if visible:
        cand["visible_ratio"] = float(np.mean(visible))
    if region_visible:
        cand["region_visible_ratio"] = float(np.mean(region_visible))

    policy = policy_metrics(frames, margin)
    policy_top10 = policy_metrics(frames, margin, top_native_k=10)

    summary = {
        "version": REP_P1_VERSION,
        "experiment": "Rep-P1 fixed-action action-conditioned evidence",
        "variant": variant,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "trainable_parameters": count_trainable_parameters(model),
        "selection_margin": margin,
        "num_frames": len(paths),
        "num_queries": int(sum(x["pred_u"].shape[1] for x in frames)),
        "candidate_metrics": cand,
        "policy": policy,
        "policy_top10_native_score": policy_top10,
        "action_contract":
            "same Rep-P0 K=7 actions/labels; no action regeneration at test",
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    write_csv(out / "per_frame.csv", per_frame)
    if args.save_per_query:
        write_csv(out / "per_query.csv", per_query)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
