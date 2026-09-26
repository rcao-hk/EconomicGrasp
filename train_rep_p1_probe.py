#!/usr/bin/env python3
"""Train Rep-P1 fixed-action evidence probes.

All variants use exactly the same Rep-P0 K=7 physical actions and exact-action
labels.  No candidate generation, CAD evaluation, or grasp decoding occurs in
this script.

Formal comparison:
  action_only: explicit physical action only; controls action/dataset priors.
  geo_pred   : predicted-depth geometry descriptor baseline.
  img_point  : sparse 13-point action-aligned image readout.
  img_region : structured region-level action-aligned image readout.

All probes optimize the same six-threshold exact-action CDF objective plus the
same optional within-ray pairwise ranking objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rep_p1_common import (
    IMAGE_VARIANTS,
    REP_P1_VARIANTS,
    REP_P1_VERSION,
    build_probe,
    cache_paths,
    candidate_metrics,
    count_trainable_parameters,
    friction_to_cdf_targets,
    image_cache_path,
    load_image_frame,
    load_p0_frame,
    pairwise_rank_loss,
    policy_metrics,
    predicted_utility_from_logits,
    tune_margin,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_p0_root", required=True)
    p.add_argument("--val_p0_root", required=True)
    p.add_argument("--train_image_root", default="")
    p.add_argument("--val_image_root", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variant", required=True, choices=REP_P1_VARIANTS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cdf_weight", type=float, default=1.0)
    p.add_argument("--rank_weight", type=float, default=0.5)
    p.add_argument("--rank_temperature", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--margin_max", type=float, default=0.5)
    p.add_argument("--margin_steps", type=int, default=51)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=2101)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_train_frames", type=int, default=0)
    p.add_argument("--max_val_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=100)
    return p.parse_args()


def feature_stats(paths):
    total = total2 = None
    count = 0
    for i, path in enumerate(paths):
        fr = load_p0_frame(path, need_geo=True)
        rows = fr["feat_pred"][fr["valid"]]
        if not len(rows):
            continue
        x = rows.astype(np.float64)
        s, s2 = x.sum(0), np.square(x).sum(0)
        total = s if total is None else total + s
        total2 = s2 if total2 is None else total2 + s2
        count += len(rows)
    if count == 0:
        raise RuntimeError("No valid geometry features")
    mean = total / count
    var = np.maximum(total2 / count - mean ** 2, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), count


def normalize(x, mean, std):
    return (x - mean) / np.maximum(std, 1e-5)


def paired_image(path, p0_root, image_root, p0):
    ipath = image_cache_path(image_root, path, p0_root)
    if not ipath.is_file():
        raise FileNotFoundError(f"Missing Rep-P1 image cache: {ipath}")
    return load_image_frame(ipath, p0)


def infer_feature_dim(variant, first_path, p0_root, image_root):
    p0 = load_p0_frame(first_path, need_geo=(variant == "geo_pred"))
    if variant == "action_only":
        return 15
    if variant == "geo_pred":
        return int(p0["feat_pred"].shape[-1])
    img = paired_image(first_path, p0_root, image_root, p0)
    return int(img["pre_feature"].shape[0])


def score_frame(model, variant, p0, image, device, mean=None, std=None):
    if variant == "action_only":
        actions = torch.from_numpy(p0["actions"]).to(device)
        logits = model(actions)
        diagnostics = {}
    elif variant == "geo_pred":\n        K, Q, Fdim = p0["feat_pred"].shape
        x = torch.from_numpy(
            normalize(p0["feat_pred"], mean, std)
        ).to(device)
        logits = model(x.reshape(K * Q, Fdim)).reshape(K, Q, -1)
        diagnostics = {}
    else:
        feature = torch.from_numpy(image["pre_feature"]).unsqueeze(0).to(device)
        Kcam = torch.from_numpy(image["K"]).to(device)
        actions = torch.from_numpy(p0["actions"]).to(device)
        valid = torch.from_numpy(p0["valid"]).to(device)
        logits, diagnostics = model(
            feature, Kcam, tuple(image["image_hw"].tolist()), actions, valid
        )
    return logits, diagnostics


@torch.no_grad()
def collect_validation(
    model,
    variant,
    paths,
    p0_root,
    image_root,
    device,
    mean=None,
    std=None,
    progress_every=0,
):
    model.eval()
    frames = []
    cdf_num = cdf_den = 0.0
    vis = []
    for i, path in enumerate(paths):
        p0 = load_p0_frame(path, need_geo=(variant == "geo_pred"))
        image = (
            paired_image(path, p0_root, image_root, p0)
            if variant in IMAGE_VARIANTS else None
        )
        logits, diagnostics = score_frame(
            model, variant, p0, image, device, mean, std
        )
        valid_t = torch.from_numpy(p0["valid"]).to(device)
        target = torch.from_numpy(
            friction_to_cdf_targets(p0["friction"])
        ).to(device)
        if bool(valid_t.any()):
            cdf = F.binary_cross_entropy_with_logits(
                logits[valid_t], target[valid_t], reduction="sum"
            )
            cdf_num += float(cdf)
            cdf_den += int(valid_t.sum()) * target.shape[-1]
        pred_u = predicted_utility_from_logits(logits).cpu().numpy().astype(np.float32)
        frames.append({**p0, "pred_u": pred_u})
        if diagnostics and "visible_ratio" in diagnostics:
            vis.append(float(diagnostics["visible_ratio"][valid_t].mean().cpu()))
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(
                f"[REP-P1][VAL][{variant}] {i+1}/{len(paths)}",
                flush=True,
            )
    cand = candidate_metrics(frames)
    cand["cdf_bce"] = cdf_num / max(cdf_den, 1.0)
    if vis:
        cand["visible_ratio"] = float(np.mean(vis))
    return frames, cand


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    variant,
    feature_dim,
    mean,
    std,
    margin,
    val_candidate,
    val_policy,
    args,
):
    torch.save({
        "version": REP_P1_VERSION,
        "experiment": "Rep-P1 fixed-action action-conditioned evidence",
        "variant": variant,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "feature_dim": int(feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "feature_mean": (
            torch.from_numpy(mean) if mean is not None else None
        ),
        "feature_std": (
            torch.from_numpy(std) if std is not None else None
        ),
        "selection_margin": float(margin),
        "val_candidate_metrics": dict(val_candidate),
        "val_policy": dict(val_policy),
        "cdf_weight": float(args.cdf_weight),
        "rank_weight": float(args.rank_weight),
        "rank_temperature": float(args.rank_temperature),
        "training_target":
            "six-threshold exact-action CDF BCE + within-ray pairwise utility ranking",
        "trainable_parameters": count_trainable_parameters(model),
    }, path)


def main():
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >=1")
    if args.learning_rate <= 0 or args.rank_temperature <= 0:
        raise ValueError("Invalid learning rate/rank temperature")
    if min(args.cdf_weight, args.rank_weight) < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.variant in IMAGE_VARIANTS and (
        not args.train_image_root or not args.val_image_root
    ):
        raise ValueError("Image variants require train/val image cache roots")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_root = Path(args.train_p0_root)
    val_root = Path(args.val_p0_root)
    train_paths = cache_paths(train_root)
    val_paths = cache_paths(val_root)
    if args.max_train_frames > 0:
        train_paths = train_paths[: args.max_train_frames]
    if args.max_val_frames > 0:
        val_paths = val_paths[: args.max_val_frames]
    if not train_paths or not val_paths:
        raise RuntimeError(
            f"Need train+val Rep-P0 cache; got {len(train_paths)}/{len(val_paths)}"
        )

    feature_dim = infer_feature_dim(
        args.variant,
        train_paths[0],
        train_root,
        args.train_image_root,
    )
    mean = std = None
    stat_rows = 0
    if args.variant == "geo_pred":
        mean, std, stat_rows = feature_stats(train_paths)

    model = build_probe(
        args.variant,
        feature_dim=feature_dim,
        hidden=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_utility = -math.inf
    best_harm = math.inf
    no_improve = 0
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        model.train()
        order = list(train_paths)
        rng.shuffle(order)
        loss_sum = cdf_sum = rank_sum = 0.0
        frames_seen = updates = 0

        for start in range(0, len(order), args.grad_accum_steps):
            chunk = order[start : start + args.grad_accum_steps]
            optimizer.zero_grad(set_to_none=True)
            contributing = 0
            for path in chunk:
                p0 = load_p0_frame(
                    path, need_geo=(args.variant == "geo_pred")
                )
                image = (
                    paired_image(path, train_root, args.train_image_root, p0)
                    if args.variant in IMAGE_VARIANTS else None
                )
                logits, _ = score_frame(
                    model, args.variant, p0, image, device, mean, std
                )
                valid = torch.from_numpy(p0["valid"]).to(device)
                target = torch.from_numpy(
                    friction_to_cdf_targets(p0["friction"])
                ).to(device)
                exact_u = torch.from_numpy(p0["utility"]).to(device)
                if not bool(valid.any()):
                    continue
                cdf_loss = F.binary_cross_entropy_with_logits(
                    logits[valid], target[valid]
                )
                rank_loss = pairwise_rank_loss(
                    logits, exact_u, valid,
                    temperature=args.rank_temperature,
                )
                loss = (
                    args.cdf_weight * cdf_loss +
                    args.rank_weight * rank_loss
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite Rep-P1 loss at {path}"
                    )
                (loss / max(len(chunk), 1)).backward()
                contributing += 1
                frames_seen += 1
                loss_sum += float(loss.detach())
                cdf_sum += float(cdf_loss.detach())
                rank_sum += float(rank_loss.detach())

            if contributing:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                optimizer.step()
                updates += 1

            if (
                args.progress_every > 0 and frames_seen > 0 and
                frames_seen % args.progress_every < max(1, len(chunk))
            ):
                print(
                    f"[REP-P1][TRAIN][{args.variant}] epoch={epoch} "
                    f"frames={frames_seen}/{len(order)} "
                    f"loss={loss_sum/max(frames_seen,1):.5f}",
                    flush=True,
                )

        val_frames, val_candidate = collect_validation(
            model,
            args.variant,
            val_paths,
            val_root,
            args.val_image_root,
            device,
            mean,
            std,
            args.progress_every,
        )
        best_margin, sweep = tune_margin(
            val_frames, args.margin_max, args.margin_steps
        )
        top10 = policy_metrics(
            val_frames, best_margin["margin"], top_native_k=10
        )
        row = {
            "epoch": epoch,
            "variant": args.variant,
            "train_loss": loss_sum / max(frames_seen, 1),
            "train_cdf_loss": cdf_sum / max(frames_seen, 1),
            "train_rank_loss": rank_sum / max(frames_seen, 1),
            "optimizer_updates": updates,
            "val_candidate": val_candidate,
            "val_policy": best_margin,
            "val_policy_top10_native_score": top10,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        with (out / "margin_sweep_latest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            writer.writeheader()
            writer.writerows(sweep)

        save_checkpoint(
            out / "checkpoint_latest.tar",
            model, optimizer, epoch, args.variant, feature_dim,
            mean, std, best_margin["margin"],
            val_candidate, best_margin, args,
        )

        improved = (
            best_margin["selected_utility"] > best_utility + 1e-9 or
            (
                abs(best_margin["selected_utility"] - best_utility) <= 1e-9
                and best_margin["harm08"] < best_harm
            )
        )
        if improved:
            best_utility = best_margin["selected_utility"]
            best_harm = best_margin["harm08"]
            no_improve = 0
            save_checkpoint(
                out / "checkpoint_best.tar",
                model, optimizer, epoch, args.variant, feature_dim,
                mean, std, best_margin["margin"],
                val_candidate, best_margin, args,
            )
            (out / "best.json").write_text(
                json.dumps(row, indent=2, sort_keys=True)
            )
        else:
            no_improve += 1

        with (out / "metrics.jsonl").open("w") as f:
            for h in history:
                f.write(json.dumps(h, sort_keys=True) + "\n")

        if (
            args.early_stop_patience > 0 and
            no_improve >= args.early_stop_patience
        ):
            print(
                f"[REP-P1] early stop variant={args.variant} epoch={epoch}",
                flush=True,
            )
            break

    protocol = {
        "version": REP_P1_VERSION,
        "experiment": "Rep-P1 fixed-action action-conditioned evidence",
        "variant": args.variant,
        "train_p0_root": str(train_root.resolve()),
        "val_p0_root": str(val_root.resolve()),
        "train_image_root": (
            str(Path(args.train_image_root).resolve())
            if args.variant in IMAGE_VARIANTS else None
        ),
        "val_image_root": (
            str(Path(args.val_image_root).resolve())
            if args.variant in IMAGE_VARIANTS else None
        ),
        "train_frames": len(train_paths),
        "val_frames": len(val_paths),
        "feature_dim": feature_dim,
        "feature_stat_rows": stat_rows,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "trainable_parameters": count_trainable_parameters(model),
        "cdf_weight": args.cdf_weight,
        "rank_weight": args.rank_weight,
        "rank_temperature": args.rank_temperature,
        "checkpoint_selection":
            "Seen selected exact utility after validation-only margin tuning",
        "action_contract":
            "same Rep-P0 K=7 physical actions and exact-action labels for every variant",
        "test_leakage": "test_similar/test_novel never tune checkpoint or margin",
    }
    (out / "training_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
