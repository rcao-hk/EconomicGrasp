#!/usr/bin/env python3
"""Train one controlled feature-ablation selector from frozen exact-action cache.

The grasp network is absent from this graph.  Every variant uses the same
3-linear-layer RayPairwiseSelector and the same exact-action relative-utility
loss.  Only the selector input feature subset changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from utils.ray_bestofk_diagnostic import select_exact_oracle, select_raw_score
from utils.ray_pairwise_feature_ablation import (
    FEATURE_MODES,
    compose_feature_ablation,
    feature_ablation_dim,
)
from utils.ray_pairwise_selector import (
    RayPairwiseSelector,
    balanced_sign_bce,
    listwise_exact_utility_loss,
    select_with_native_fallback,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_cache_root", required=True)
    p.add_argument("--val_cache_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_mode", required=True, choices=FEATURE_MODES)
    p.add_argument("--val_scene_start", type=int, default=100)
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
    p.add_argument("--max_train_frames", type=int, default=0)
    p.add_argument("--max_val_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=200)
    return p.parse_args()


ARGS = parse_args()


def scene_id(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def filtered_paths(root: Path, *, train: bool) -> List[Path]:
    paths = sorted(root.glob("scene_*/ann_*.npz"))
    if train:
        return [p for p in paths if scene_id(p) < ARGS.val_scene_start]
    return [p for p in paths if scene_id(p) >= ARGS.val_scene_start]


def load_frame(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as d:
        selected = torch.from_numpy(d["selected_feature"].astype(np.float32)).to(device)
        mean = torch.from_numpy(d["mean_feature"].astype(np.float32)).to(device)
        raw = torch.from_numpy(d["raw_score"].astype(np.float32)).to(device)
        utility = torch.from_numpy(d["utility"].astype(np.float32)).to(device)
        valid = torch.from_numpy(d["valid"].astype(bool)).to(device)
        friction = d["friction"].astype(np.float32)
        offsets = torch.from_numpy(d["offsets_mm"].astype(np.float32)).to(device)
        zero = int(np.asarray(d["zero_index"]).reshape(-1)[0])
        sid = int(np.asarray(d["scene_id"]).reshape(-1)[0])
        aid = int(np.asarray(d["anno_id"]).reshape(-1)[0])
    x = compose_feature_ablation(selected, mean, raw, offsets, zero, ARGS.feature_mode)
    return {
        "x": x,
        "raw": raw,
        "utility": utility,
        "valid": valid,
        "friction": friction,
        "zero": zero,
        "scene_id": sid,
        "anno_id": aid,
        "group_dim": int(selected.shape[-1]),
    }


def feature_stats(paths: List[Path], device: torch.device):
    total = total2 = None
    count = 0
    for i, path in enumerate(paths):
        f = load_frame(path, device)
        mask = f["valid"].clone(); mask[f["zero"]] = False
        rows = f["x"][mask]
        if rows.numel() == 0:
            continue
        s = rows.double().sum(0); s2 = (rows.double() ** 2).sum(0)
        total = s if total is None else total + s
        total2 = s2 if total2 is None else total2 + s2
        count += int(rows.shape[0])
        if ARGS.progress_every > 0 and (i + 1) % ARGS.progress_every == 0:
            print(f"[ABLATE][STATS][{ARGS.feature_mode}] {i+1}/{len(paths)} rows={count}", flush=True)
    if count == 0:
        raise RuntimeError("No valid non-native rows for normalization.")
    mean = total / count
    var = (total2 / count - mean ** 2).clamp_min(1e-8)
    return mean.float(), var.sqrt().float(), count


def normalize(x, mean, std):
    return (x - mean) / std.clamp_min(1e-5)


def frame_loss(model, frame, feat_mean, feat_std):
    x = normalize(frame["x"], feat_mean, feat_std)
    utility, valid, z = frame["utility"], frame["valid"], frame["zero"]
    K, N, Fdim = x.shape
    pred = model(x.reshape(K * N, Fdim)).reshape(K, N)
    pred_rel = pred.clone(); pred_rel[z] = 0.0
    delta = utility - utility[z:z+1]
    mask = valid.clone(); mask[z] = False
    if not bool(mask.any()):
        zero = pred.sum() * 0.0
        return zero, {"reg": 0.0, "sign": 0.0, "list": 0.0}
    reg = F.smooth_l1_loss(pred_rel[mask], delta[mask], beta=0.1)
    sign = balanced_sign_bce(pred_rel[mask], delta[mask])
    lst = listwise_exact_utility_loss(pred_rel, utility, valid, ARGS.target_temperature)
    loss = ARGS.reg_weight * reg + ARGS.sign_weight * sign + ARGS.listwise_weight * lst
    return loss, {"reg": float(reg.detach()), "sign": float(sign.detach()), "list": float(lst.detach())}


def success(friction, threshold=0.8):
    f = np.asarray(friction, np.float32)
    return np.isfinite(f) & (f > 0) & (f <= threshold + 1e-6)


def gather(arr, k):
    arr = np.asarray(arr); k = np.asarray(k, np.int64)
    return arr[k, np.arange(arr.shape[1])]


@torch.no_grad()
def collect_validation(model, paths, feat_mean, feat_std, device):
    out = []
    for i, path in enumerate(paths):
        f = load_frame(path, device)
        x = normalize(f["x"], feat_mean, feat_std)
        K, N, D = x.shape
        pred = model(x.reshape(K*N, D)).reshape(K, N)
        pred[f["zero"]] = 0.0
        out.append({
            "pred": pred.cpu().numpy(),
            "utility": f["utility"].cpu().numpy(),
            "valid": f["valid"].cpu().numpy(),
            "raw": f["raw"].cpu().numpy(),
            "friction": f["friction"],
            "zero": f["zero"],
        })
        if ARGS.progress_every > 0 and (i + 1) % ARGS.progress_every == 0:
            print(f"[ABLATE][VAL][{ARGS.feature_mode}] {i+1}/{len(paths)}", flush=True)
    return out


def metrics_at_threshold(frames, threshold):
    us=[]; un=[]; ur=[]; uo=[]; ss=[]; sn=[]; rescue=[]; harm=[]; change=[]; match=[]
    for fr in frames:
        p,u,v,raw,z = fr["pred"],fr["utility"],fr["valid"],fr["raw"],fr["zero"]
        ks = select_with_native_fallback(p,v,z,threshold)
        kn = np.full(ks.shape,z,np.int64)
        kr = select_raw_score(raw,v); ko = select_exact_oracle(u,raw,v)
        usi,uni = gather(u,ks),gather(u,kn)
        fsi,fni = gather(fr["friction"],ks),gather(fr["friction"],kn)
        ssi,sni = success(fsi),success(fni)
        us.append(usi); un.append(uni); ur.append(gather(u,kr)); uo.append(gather(u,ko))
        ss.append(ssi); sn.append(sni); rescue.append((~sni)&ssi); harm.append(sni&(~ssi))
        change.append(ks!=z); match.append(ks==ko)
    cat=lambda xs: np.concatenate(xs)
    m={
        "threshold":float(threshold),
        "selected_utility":float(cat(us).mean()),
        "native_utility":float(cat(un).mean()),
        "raw_utility":float(cat(ur).mean()),
        "oracle_utility":float(cat(uo).mean()),
        "selected_success08":float(cat(ss).mean()),
        "native_success08":float(cat(sn).mean()),
        "rescue08":float(cat(rescue).mean()),
        "harm08":float(cat(harm).mean()),
        "change_rate":float(cat(change).mean()),
        "match_oracle":float(cat(match).mean()),
    }
    m["utility_gain"] = m["selected_utility"] - m["native_utility"]
    m["success08_gain"] = m["selected_success08"] - m["native_success08"]
    m["selection_regret"] = m["oracle_utility"] - m["selected_utility"]
    headroom = m["oracle_utility"] - m["native_utility"]
    m["utility_headroom_recovery"] = m["utility_gain"] / headroom if abs(headroom)>1e-12 else float("nan")
    return m


def tune_threshold(frames):
    rows=[metrics_at_threshold(frames,x) for x in np.linspace(0,ARGS.threshold_max,max(2,ARGS.threshold_steps))]
    best=max(rows,key=lambda r:(r["selected_utility"],-r["harm08"],-r["change_rate"]))
    return best, rows


def save_checkpoint(path, model, optimizer, epoch, mean, std, threshold, feature_dim, group_dim, metrics):
    torch.save({
        "selector_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "epoch":int(epoch),
        "feature_mode":ARGS.feature_mode,
        "feature_dim":int(feature_dim),
        "group_feature_dim":int(group_dim),
        "hidden_dim":int(ARGS.hidden_dim),
        "dropout":float(ARGS.dropout),
        "feature_mean":mean.detach().cpu(),
        "feature_std":std.detach().cpu(),
        "selector_threshold":float(threshold),
        "val_metrics":dict(metrics),
        "architecture":"3-linear-layer MLP: Linear-LN-GELU-Dropout-Linear-GELU-Dropout-Linear",
        "training_target":"exact-action relative utility delta vs native center",
    }, path)


def main():
    random.seed(ARGS.seed); np.random.seed(ARGS.seed); torch.manual_seed(ARGS.seed)
    device=torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    train_paths=filtered_paths(Path(ARGS.train_cache_root),train=True)
    val_paths=filtered_paths(Path(ARGS.val_cache_root),train=False)
    if ARGS.max_train_frames>0: train_paths=train_paths[:ARGS.max_train_frames]
    if ARGS.max_val_frames>0: val_paths=val_paths[:ARGS.max_val_frames]
    if not train_paths or not val_paths:
        raise RuntimeError(f"Need train+val caches; got train={len(train_paths)} val={len(val_paths)}")
    out=Path(ARGS.output_dir); out.mkdir(parents=True,exist_ok=True)
    print(f"[ABLATE] mode={ARGS.feature_mode} train={len(train_paths)} val={len(val_paths)} device={device}")

    first=load_frame(train_paths[0],device)
    group_dim=first["group_dim"]
    expected=feature_ablation_dim(group_dim,ARGS.feature_mode)
    if first["x"].shape[-1] != expected:
        raise RuntimeError(f"Feature dimension mismatch {first['x'].shape[-1]} vs {expected}")
    mean,std,stat_rows=feature_stats(train_paths,device)
    model=RayPairwiseSelector(expected,ARGS.hidden_dim,ARGS.dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=ARGS.learning_rate,weight_decay=ARGS.weight_decay)

    history=[]; best_utility=-math.inf; best_harm=math.inf
    rng=random.Random(ARGS.seed)
    for epoch in range(ARGS.epochs):
        model.train(); order=list(train_paths); rng.shuffle(order)
        sums={"loss":0.0,"reg":0.0,"sign":0.0,"list":0.0}; steps=0
        for i,path in enumerate(order):
            frame=load_frame(path,device); optimizer.zero_grad(set_to_none=True)
            loss,parts=frame_loss(model,frame,mean,std); loss.backward()
            if ARGS.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),ARGS.grad_clip)
            optimizer.step(); steps+=1
            sums["loss"]+=float(loss.detach()); sums["reg"]+=parts["reg"]; sums["sign"]+=parts["sign"]; sums["list"]+=parts["list"]
            if ARGS.progress_every>0 and (i+1)%ARGS.progress_every==0:
                print(f"[ABLATE][TRAIN] mode={ARGS.feature_mode} epoch={epoch} {i+1}/{len(order)} loss={sums['loss']/steps:.4f}",flush=True)
        model.eval(); frames=collect_validation(model,val_paths,mean,std,device)
        best_thr,sweep=tune_threshold(frames)
        row={"epoch":epoch,"feature_mode":ARGS.feature_mode,
             "train_loss":sums["loss"]/steps,"train_reg":sums["reg"]/steps,
             "train_sign":sums["sign"]/steps,"train_list":sums["list"]/steps,
             **{f"val_{k}":v for k,v in best_thr.items()}}
        history.append(row); print(json.dumps(row,sort_keys=True),flush=True)
        with (out/"threshold_sweep_latest.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
        save_checkpoint(out/"checkpoint_latest.tar",model,optimizer,epoch,mean,std,best_thr["threshold"],expected,group_dim,best_thr)
        improved=(best_thr["selected_utility"]>best_utility+1e-9 or
                  (abs(best_thr["selected_utility"]-best_utility)<=1e-9 and best_thr["harm08"]<best_harm))
        if improved:
            best_utility=best_thr["selected_utility"]; best_harm=best_thr["harm08"]
            save_checkpoint(out/"checkpoint_best.tar",model,optimizer,epoch,mean,std,best_thr["threshold"],expected,group_dim,best_thr)
            with (out/"best.json").open("w") as f: json.dump(row,f,indent=2,sort_keys=True)
        with (out/"metrics.jsonl").open("w") as f:
            for h in history: f.write(json.dumps(h,sort_keys=True)+"\n")

    protocol={"feature_mode":ARGS.feature_mode,"feature_dim":expected,"group_feature_dim":group_dim,
              "train_frames":len(train_paths),"val_frames":len(val_paths),"feature_stat_rows":stat_rows,
              "architecture":"fixed 3-linear-layer MLP","val_scene_start":ARGS.val_scene_start,
              "train_cache_root":str(Path(ARGS.train_cache_root).resolve()),
              "val_cache_root":str(Path(ARGS.val_cache_root).resolve())}
    with (out/"training_protocol.json").open("w") as f: json.dump(protocol,f,indent=2,sort_keys=True)


if __name__=="__main__":
    main()
