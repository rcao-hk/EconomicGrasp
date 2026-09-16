#!/usr/bin/env python3
"""Train a lightweight relative-advantage selector from frozen exact-action cache.

Only the MLP in ``RayPairwiseSelector`` is optimized.  Stage-1 RGB/DPT, view,
local grouping, CDF and width heads are absent from the training graph because
all frozen features and CAD/DexNet outcomes were cached beforehand.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.ray_bestofk_diagnostic import select_exact_oracle, select_raw_score
from utils.ray_pairwise_selector import (
    RayPairwiseSelector,
    balanced_sign_bce,
    compose_pairwise_features,
    listwise_exact_utility_loss,
    pairwise_feature_dim,
    select_with_native_fallback,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_scene_start", type=int, default=80)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--reg_weight", type=float, default=1.0)
    p.add_argument("--sign_weight", type=float, default=0.5)
    p.add_argument("--listwise_weight", type=float, default=0.5)
    p.add_argument("--target_temperature", type=float, default=0.15)
    p.add_argument("--threshold_max", type=float, default=0.30)
    p.add_argument("--threshold_steps", type=int, default=31)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--resume", default="")
    p.add_argument("--max_train_frames", type=int, default=0)
    p.add_argument("--max_val_frames", type=int, default=0)
    return p.parse_args()


ARGS = _parse_args()


def _scene_id(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def _paths(root: Path) -> List[Path]:
    return sorted(root.glob("scene_*/ann_*.npz"))


def _split_paths(paths: List[Path], val_scene_start: int):
    train = [p for p in paths if _scene_id(p) < val_scene_start]
    val = [p for p in paths if _scene_id(p) >= val_scene_start]
    if not train or not val:
        raise RuntimeError(
            f"Need both train and validation caches; got train={len(train)}, val={len(val)} "
            f"with val_scene_start={val_scene_start}."
        )
    return train, val


def _load_frame(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as d:
        selected = torch.from_numpy(d["selected_feature"].astype(np.float32)).to(device)
        mean = torch.from_numpy(d["mean_feature"].astype(np.float32)).to(device)
        raw = torch.from_numpy(d["raw_score"].astype(np.float32)).to(device)
        utility = torch.from_numpy(d["utility"].astype(np.float32)).to(device)
        valid = torch.from_numpy(d["valid"].astype(bool)).to(device)
        friction = d["friction"].astype(np.float32)
        offsets = torch.from_numpy(d["offsets_mm"].astype(np.float32)).to(device)
        zero = int(np.asarray(d["zero_index"]).reshape(-1)[0])
        scene_id = int(np.asarray(d["scene_id"]).reshape(-1)[0])
        anno_id = int(np.asarray(d["anno_id"]).reshape(-1)[0])
    x = compose_pairwise_features(selected, mean, raw, offsets, zero)
    return {
        "x": x,
        "raw": raw,
        "utility": utility,
        "valid": valid,
        "friction": friction,
        "offsets": offsets,
        "zero": zero,
        "scene_id": scene_id,
        "anno_id": anno_id,
    }


def _feature_stats(paths: List[Path], device: torch.device):
    total = None
    total2 = None
    count = 0
    for i, path in enumerate(paths):
        frame = _load_frame(path, device)
        x, valid, zero = frame["x"], frame["valid"], frame["zero"]
        mask = valid.clone()
        mask[zero] = False
        rows = x[mask]
        if rows.numel() == 0:
            continue
        s = rows.double().sum(dim=0)
        s2 = (rows.double() ** 2).sum(dim=0)
        total = s if total is None else total + s
        total2 = s2 if total2 is None else total2 + s2
        count += rows.shape[0]
        if (i + 1) % 200 == 0:
            print(f"[PAIR-TRAIN][STATS] {i+1}/{len(paths)} rows={count}", flush=True)
    if count == 0:
        raise RuntimeError("No valid non-native rows for feature normalization.")
    mean = total / count
    var = (total2 / count - mean ** 2).clamp_min(1e-8)
    std = var.sqrt()
    return mean.float(), std.float(), count


def _normalize(x, mean, std):
    return (x - mean) / std.clamp_min(1e-5)


def _frame_loss(model, frame, feat_mean, feat_std):
    x = _normalize(frame["x"], feat_mean, feat_std)
    utility = frame["utility"]
    valid = frame["valid"]
    zero = frame["zero"]
    K, N, Fdim = x.shape
    pred = model(x.reshape(K * N, Fdim)).reshape(K, N)
    # Native is the reference action.  Its relative advantage is defined as 0,
    # not learned from an arbitrary MLP bias.
    pred_rel = pred.clone()
    pred_rel[zero] = 0.0
    target_delta = utility - utility[zero : zero + 1]
    mask = valid.clone()
    mask[zero] = False
    if not bool(mask.any()):
        zero_loss = pred.sum() * 0.0
        return zero_loss, {"reg": 0.0, "sign": 0.0, "list": 0.0}
    reg = F.smooth_l1_loss(pred_rel[mask], target_delta[mask], beta=0.1)
    sign = balanced_sign_bce(pred_rel[mask], target_delta[mask])
    list_loss = listwise_exact_utility_loss(
        pred_rel, utility, valid, target_temperature=ARGS.target_temperature
    )
    loss = ARGS.reg_weight * reg + ARGS.sign_weight * sign + ARGS.listwise_weight * list_loss
    return loss, {"reg": float(reg.detach()), "sign": float(sign.detach()), "list": float(list_loss.detach())}


def _success(friction: np.ndarray, threshold: float):
    return np.isfinite(friction) & (friction > 0.0) & (friction <= threshold + 1e-6)


def _gather_kn(arr, k):
    arr = np.asarray(arr)
    k = np.asarray(k, dtype=np.int64)
    return arr[k, np.arange(arr.shape[1])]


@torch.no_grad()
def _collect_validation(model, paths, feat_mean, feat_std, device):
    frames = []
    for path in paths:
        frame = _load_frame(path, device)
        x = _normalize(frame["x"], feat_mean, feat_std)
        K, N, Fdim = x.shape
        pred = model(x.reshape(K * N, Fdim)).reshape(K, N)
        pred[frame["zero"]] = 0.0
        frames.append({
            "pred": pred.cpu().numpy(),
            "utility": frame["utility"].cpu().numpy(),
            "valid": frame["valid"].cpu().numpy(),
            "raw": frame["raw"].cpu().numpy(),
            "friction": frame["friction"],
            "zero": frame["zero"],
            "scene_id": frame["scene_id"],
            "anno_id": frame["anno_id"],
        })
    return frames


def _metrics_at_threshold(frames, threshold):
    util_sel = []
    util_native = []
    util_raw = []
    util_oracle = []
    s08_sel = []
    s08_native = []
    rescue = []
    harm = []
    change = []
    match_oracle = []
    for fr in frames:
        p, u, v, raw = fr["pred"], fr["utility"], fr["valid"], fr["raw"]
        z = fr["zero"]
        selected = select_with_native_fallback(p, v, z, threshold)
        native = np.full(selected.shape, z, dtype=np.int64)
        raw_k = select_raw_score(raw, v)
        oracle_k = select_exact_oracle(u, raw, v)
        us = _gather_kn(u, selected)
        un = _gather_kn(u, native)
        ur = _gather_kn(u, raw_k)
        uo = _gather_kn(u, oracle_k)
        fs = _gather_kn(fr["friction"], selected)
        fn = _gather_kn(fr["friction"], native)
        ss = _success(fs, 0.8)
        sn = _success(fn, 0.8)
        util_sel.append(us); util_native.append(un); util_raw.append(ur); util_oracle.append(uo)
        s08_sel.append(ss); s08_native.append(sn)
        rescue.append((~sn) & ss); harm.append(sn & (~ss))
        change.append(selected != z); match_oracle.append(selected == oracle_k)
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0)
    out = {
        "threshold": float(threshold),
        "selected_utility": float(cat(util_sel).mean()),
        "native_utility": float(cat(util_native).mean()),
        "raw_utility": float(cat(util_raw).mean()),
        "oracle_utility": float(cat(util_oracle).mean()),
        "selected_success08": float(cat(s08_sel).mean()),
        "native_success08": float(cat(s08_native).mean()),
        "rescue08": float(cat(rescue).mean()),
        "harm08": float(cat(harm).mean()),
        "change_rate": float(cat(change).mean()),
        "match_oracle": float(cat(match_oracle).mean()),
    }
    out["selection_regret"] = out["oracle_utility"] - out["selected_utility"]
    out["utility_gain"] = out["selected_utility"] - out["native_utility"]
    out["success08_gain"] = out["selected_success08"] - out["native_success08"]
    return out


def _tune_threshold(frames):
    values = np.linspace(0.0, ARGS.threshold_max, max(2, ARGS.threshold_steps))
    rows = [_metrics_at_threshold(frames, x) for x in values]
    # Primary objective: exact utility. Tie-break by lower harm, then lower change.
    best = max(rows, key=lambda r: (r["selected_utility"], -r["harm08"], -r["change_rate"]))
    return best, rows


def _save_checkpoint(path, model, optimizer, epoch, feat_mean, feat_std, threshold, feature_dim, best_metrics):
    torch.save({
        "selector_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "feature_dim": int(feature_dim),
        "hidden_dim": int(ARGS.hidden_dim),
        "dropout": float(ARGS.dropout),
        "feature_mean": feat_mean.detach().cpu(),
        "feature_std": feat_std.detach().cpu(),
        "selector_threshold": float(threshold),
        "val_metrics": dict(best_metrics),
        "val_scene_start": int(ARGS.val_scene_start),
        "training_target": "exact-action relative utility delta vs native center",
    }, path)


def main():
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    root = Path(ARGS.cache_root)
    out = Path(ARGS.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = _paths(root)
    train_paths, val_paths = _split_paths(paths, ARGS.val_scene_start)
    if ARGS.max_train_frames > 0:
        train_paths = train_paths[: ARGS.max_train_frames]
    if ARGS.max_val_frames > 0:
        val_paths = val_paths[: ARGS.max_val_frames]
    print(f"[PAIR-TRAIN] cache frames train={len(train_paths)} val={len(val_paths)}")

    first = _load_frame(train_paths[0], device)
    group_dim = int(first["x"].shape[-1] - 4) // 5
    feature_dim = pairwise_feature_dim(group_dim)
    if feature_dim != first["x"].shape[-1]:
        raise RuntimeError("Pairwise feature dimension contract mismatch.")

    feat_mean, feat_std, stat_rows = _feature_stats(train_paths, device)
    model = RayPairwiseSelector(feature_dim, ARGS.hidden_dim, ARGS.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
    start_epoch = 0
    if ARGS.resume:
        ckpt = torch.load(ARGS.resume, map_location=device)
        model.load_state_dict(ckpt["selector_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if "feature_mean" in ckpt:
            feat_mean = ckpt["feature_mean"].to(device)
            feat_std = ckpt["feature_std"].to(device)

    history = []
    best_utility = -math.inf
    best_harm = math.inf
    rng = random.Random(ARGS.seed)
    for epoch in range(start_epoch, ARGS.epochs):
        model.train()
        order = list(train_paths)
        rng.shuffle(order)
        loss_sum = reg_sum = sign_sum = list_sum = 0.0
        steps = 0
        for step, path in enumerate(order):
            frame = _load_frame(path, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _frame_loss(model, frame, feat_mean, feat_std)
            loss.backward()
            if ARGS.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), ARGS.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
            reg_sum += parts["reg"]; sign_sum += parts["sign"]; list_sum += parts["list"]
            steps += 1
            if (step + 1) % 200 == 0:
                print(f"[PAIR-TRAIN] epoch={epoch} step={step+1}/{len(order)} loss={loss_sum/steps:.4f}", flush=True)

        model.eval()
        frames = _collect_validation(model, val_paths, feat_mean, feat_std, device)
        best_thr, threshold_rows = _tune_threshold(frames)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(steps, 1),
            "train_reg": reg_sum / max(steps, 1),
            "train_sign": sign_sum / max(steps, 1),
            "train_list": list_sum / max(steps, 1),
            **{f"val_{k}": v for k, v in best_thr.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        with (out / "threshold_sweep_latest.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
            w.writeheader(); w.writerows(threshold_rows)
        _save_checkpoint(
            out / "checkpoint_latest.tar", model, optimizer, epoch, feat_mean, feat_std,
            best_thr["threshold"], feature_dim, best_thr,
        )
        improved = (
            best_thr["selected_utility"] > best_utility + 1e-9
            or (abs(best_thr["selected_utility"] - best_utility) <= 1e-9 and best_thr["harm08"] < best_harm)
        )
        if improved:
            best_utility = best_thr["selected_utility"]
            best_harm = best_thr["harm08"]
            _save_checkpoint(
                out / "checkpoint_best.tar", model, optimizer, epoch, feat_mean, feat_std,
                best_thr["threshold"], feature_dim, best_thr,
            )
            with (out / "best.json").open("w") as f:
                json.dump(row, f, indent=2, sort_keys=True)

        with (out / "metrics.jsonl").open("w") as f:
            for h in history:
                f.write(json.dumps(h, sort_keys=True) + "\n")

    protocol = {
        "cache_root": str(root.resolve()),
        "train_frames": len(train_paths),
        "val_frames": len(val_paths),
        "val_scene_start": ARGS.val_scene_start,
        "feature_dim": feature_dim,
        "feature_stat_rows": stat_rows,
        "loss": {
            "reg_weight": ARGS.reg_weight,
            "sign_weight": ARGS.sign_weight,
            "listwise_weight": ARGS.listwise_weight,
            "target_temperature": ARGS.target_temperature,
        },
        "selection": "predict relative exact utility; switch from native only above tuned threshold",
    }
    with (out / "training_protocol.json").open("w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
