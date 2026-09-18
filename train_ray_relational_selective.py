#!/usr/bin/env python3
"""Train Cross-Center Relational Selective Correction from frozen K-ray cache.

No grasp-network parameter is present in this graph. The cached K-center local
features and exact CAD/DexNet utilities supervise:
  * a move / keep-native gate;
  * a conditional relational K-center selector;
  * an auxiliary contextual relative-utility head.

Train scenes and validation scenes are read from separate cache roots and are
filtered by global GraspNet scene id.
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

from utils.ray_bestofk_diagnostic import select_exact_oracle, select_raw_score
from utils.ray_relational_selective import (
    CrossCenterRelationalSelective,
    compose_relational_tokens,
    relational_exact_action_losses,
    relational_token_dim,
    select_relational_correction,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_cache_root", required=True)
    p.add_argument("--val_cache_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_scene_start", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gate_weight", type=float, default=1.0)
    p.add_argument("--selector_weight", type=float, default=1.0)
    p.add_argument("--delta_weight", type=float, default=0.5)
    p.add_argument("--selector_temperature", type=float, default=0.15)
    p.add_argument("--delta_beta", type=float, default=0.1)
    p.add_argument("--threshold_steps", type=int, default=41)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--max_train_frames", type=int, default=0)
    p.add_argument("--max_val_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=200)
    return p.parse_args()


ARGS = parse_args()
EPS = 1.0e-8


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
    tokens = compose_relational_tokens(selected, mean, raw, offsets, zero)
    valid_nk = valid.transpose(0, 1).contiguous()
    return {
        "tokens": tokens,
        "raw": raw,
        "utility": utility,
        "valid": valid,
        "valid_nk": valid_nk,
        "friction": friction,
        "offsets": offsets,
        "zero": zero,
        "scene_id": sid,
        "anno_id": aid,
        "group_dim": int(selected.shape[-1]),
    }


def token_stats(paths: List[Path], device: torch.device):
    total = total2 = None
    count = 0
    for i, path in enumerate(paths):
        f = load_frame(path, device)
        rows = f["tokens"][f["valid_nk"]]
        if rows.numel() == 0:
            continue
        s = rows.double().sum(dim=0)
        s2 = (rows.double() ** 2).sum(dim=0)
        total = s if total is None else total + s
        total2 = s2 if total2 is None else total2 + s2
        count += int(rows.shape[0])
        if ARGS.progress_every > 0 and (i + 1) % ARGS.progress_every == 0:
            print(f"[REL][STATS] {i+1}/{len(paths)} rows={count}", flush=True)
    if count == 0:
        raise RuntimeError("No valid K-center tokens for normalization.")
    mean = total / count
    var = (total2 / count - mean ** 2).clamp_min(1e-8)
    return mean.float(), var.sqrt().float(), count


def normalize(tokens, mean, std):
    return (tokens - mean) / std.clamp_min(1e-5)


def success(friction: np.ndarray, threshold: float = 0.8):
    f = np.asarray(friction, dtype=np.float32)
    return np.isfinite(f) & (f > 0.0) & (f <= float(threshold) + 1e-6)


def gather(arr, k):
    arr = np.asarray(arr)
    k = np.asarray(k, dtype=np.int64)
    return arr[k, np.arange(arr.shape[1])]


@torch.no_grad()
def collect_validation(model, paths, token_mean, token_std, device):
    frames = []
    for i, path in enumerate(paths):
        f = load_frame(path, device)
        out = model(
            normalize(f["tokens"], token_mean, token_std),
            f["valid_nk"],
            f["zero"],
        )
        frames.append({
            "gate_logit": out["gate_logit"].detach().cpu().numpy(),
            "selector_logits": out["selector_logits"].detach().cpu().numpy(),
            "delta_pred": out["delta_pred"].detach().cpu().numpy(),
            "utility": f["utility"].detach().cpu().numpy(),
            "valid": f["valid"].detach().cpu().numpy(),
            "raw": f["raw"].detach().cpu().numpy(),
            "friction": f["friction"],
            "zero": f["zero"],
            "scene_id": f["scene_id"],
            "anno_id": f["anno_id"],
        })
        if ARGS.progress_every > 0 and (i + 1) % ARGS.progress_every == 0:
            print(f"[REL][VAL] {i+1}/{len(paths)}", flush=True)
    return frames


def metrics_at_threshold(frames, threshold: float):
    util_sel=[]; util_native=[]; util_raw=[]; util_oracle=[]
    s08_sel=[]; s08_native=[]; rescue=[]; harm=[]; change=[]; match=[]
    move_prob_all=[]; move_target_all=[]; best_alt_beneficial=[]
    for fr in frames:
        u,v,raw,z = fr["utility"],fr["valid"],fr["raw"],fr["zero"]
        selected,best_alt,move_prob = select_relational_correction(
            fr["gate_logit"], fr["selector_logits"], v.T, z, threshold
        )
        native = np.full(selected.shape, z, dtype=np.int64)
        raw_k = select_raw_score(raw, v)
        oracle_k = select_exact_oracle(u, raw, v)

        us = gather(u, selected); un = gather(u, native)
        ur = gather(u, raw_k); uo = gather(u, oracle_k)
        fs = gather(fr["friction"], selected); fn = gather(fr["friction"], native)
        ss = success(fs); sn = success(fn)
        util_sel.append(us); util_native.append(un); util_raw.append(ur); util_oracle.append(uo)
        s08_sel.append(ss); s08_native.append(sn)
        rescue.append((~sn) & ss); harm.append(sn & (~ss))
        change.append(selected != z); match.append(selected == oracle_k)

        alt_u = gather(u, best_alt)
        move_target = np.max(np.where(v & (np.arange(v.shape[0])[:,None] != z), u, -np.inf), axis=0) > (un + EPS)
        move_prob_all.append(move_prob)
        move_target_all.append(move_target)
        best_alt_beneficial.append(alt_u > un + EPS)

    cat=lambda xs: np.concatenate(xs) if xs else np.empty(0)
    us=cat(util_sel); un=cat(util_native); uo=cat(util_oracle)
    out={
        "move_threshold":float(threshold),
        "selected_utility":float(us.mean()),
        "native_utility":float(un.mean()),
        "raw_utility":float(cat(util_raw).mean()),
        "oracle_utility":float(uo.mean()),
        "selected_success08":float(cat(s08_sel).mean()),
        "native_success08":float(cat(s08_native).mean()),
        "rescue08":float(cat(rescue).mean()),
        "harm08":float(cat(harm).mean()),
        "change_rate":float(cat(change).mean()),
        "match_oracle":float(cat(match).mean()),
        "move_probability_mean":float(cat(move_prob_all).mean()),
        "move_target_fraction":float(cat(move_target_all).mean()),
        "best_alt_beneficial_fraction":float(cat(best_alt_beneficial).mean()),
    }
    out["utility_gain"]=out["selected_utility"]-out["native_utility"]
    out["success08_gain"]=out["selected_success08"]-out["native_success08"]
    out["selection_regret"]=out["oracle_utility"]-out["selected_utility"]
    headroom=out["oracle_utility"]-out["native_utility"]
    out["utility_headroom_recovery"]=out["utility_gain"]/headroom if abs(headroom)>1e-12 else float("nan")
    return out


def tune_threshold(frames):
    values=np.linspace(0.0,1.0,max(2,ARGS.threshold_steps))
    rows=[metrics_at_threshold(frames,float(x)) for x in values]
    best=max(rows,key=lambda r:(r["selected_utility"],-r["harm08"],-r["change_rate"]))
    return best, rows


def save_checkpoint(path, model, optimizer, epoch, mean, std, threshold, token_dim, group_dim, metrics):
    torch.save({
        "selector_type":"cross_center_relational_selective_v1",
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "epoch":int(epoch),
        "token_dim":int(token_dim),
        "group_feature_dim":int(group_dim),
        "d_model":int(ARGS.d_model),
        "nhead":int(ARGS.nhead),
        "num_layers":int(ARGS.num_layers),
        "ff_dim":int(ARGS.ff_dim),
        "dropout":float(ARGS.dropout),
        "token_mean":mean.detach().cpu(),
        "token_std":std.detach().cpu(),
        "move_threshold":float(threshold),
        "val_metrics":dict(metrics),
        "loss_weights":{
            "gate":float(ARGS.gate_weight),
            "selector":float(ARGS.selector_weight),
            "delta":float(ARGS.delta_weight),
        },
        "selector_temperature":float(ARGS.selector_temperature),
        "architecture":"joint K-center Transformer + move gate + conditional selector + delta head",
        "training_target":"exact decoded-grasp utility; strict beneficial move target",
    },path)


def main():
    random.seed(ARGS.seed); np.random.seed(ARGS.seed); torch.manual_seed(ARGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ARGS.seed)
    device=torch.device(ARGS.device if torch.cuda.is_available() else "cpu")

    train_paths=filtered_paths(Path(ARGS.train_cache_root),train=True)
    val_paths=filtered_paths(Path(ARGS.val_cache_root),train=False)
    if ARGS.max_train_frames>0: train_paths=train_paths[:ARGS.max_train_frames]
    if ARGS.max_val_frames>0: val_paths=val_paths[:ARGS.max_val_frames]
    if not train_paths or not val_paths:
        raise RuntimeError(f"Need train+val caches; got train={len(train_paths)} val={len(val_paths)}")

    out_dir=Path(ARGS.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    print(f"[REL] train_frames={len(train_paths)} val_frames={len(val_paths)} device={device}",flush=True)

    first=load_frame(train_paths[0],device)
    group_dim=first["group_dim"]
    token_dim=relational_token_dim(group_dim)
    if first["tokens"].shape[-1] != token_dim:
        raise RuntimeError(f"Token dim mismatch {first['tokens'].shape[-1]} vs {token_dim}")

    mean,std,stat_rows=token_stats(train_paths,device)
    model=CrossCenterRelationalSelective(
        token_dim=token_dim,
        d_model=ARGS.d_model,
        nhead=ARGS.nhead,
        num_layers=ARGS.num_layers,
        ff_dim=ARGS.ff_dim,
        dropout=ARGS.dropout,
    ).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=ARGS.learning_rate,weight_decay=ARGS.weight_decay)

    best_utility=-math.inf; best_harm=math.inf; epochs_no_improve=0
    history=[]; rng=random.Random(ARGS.seed)
    for epoch in range(ARGS.epochs):
        model.train(); order=list(train_paths); rng.shuffle(order)
        sums={"loss":0.0,"gate":0.0,"selector":0.0,"delta":0.0,"move_frac":0.0}; steps=0
        for i,path in enumerate(order):
            f=load_frame(path,device)
            optimizer.zero_grad(set_to_none=True)
            outputs=model(normalize(f["tokens"],mean,std),f["valid_nk"],f["zero"])
            losses,targets=relational_exact_action_losses(
                outputs,f["utility"],f["valid"],f["zero"],
                selector_temperature=ARGS.selector_temperature,
                delta_beta=ARGS.delta_beta,
            )
            loss=(ARGS.gate_weight*losses["gate"]+
                  ARGS.selector_weight*losses["selector"]+
                  ARGS.delta_weight*losses["delta"])
            loss.backward()
            if ARGS.grad_clip>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),ARGS.grad_clip)
            optimizer.step()
            steps+=1
            sums["loss"]+=float(loss.detach())
            for key in ("gate","selector","delta"):
                sums[key]+=float(losses[key].detach())
            sums["move_frac"]+=float(targets["move_target"].float().mean().detach())
            if ARGS.progress_every>0 and (i+1)%ARGS.progress_every==0:
                print(f"[REL][TRAIN] epoch={epoch} {i+1}/{len(order)} loss={sums['loss']/steps:.4f}",flush=True)

        model.eval()
        frames=collect_validation(model,val_paths,mean,std,device)
        best_thr,sweep=tune_threshold(frames)
        row={
            "epoch":epoch,
            "train_loss":sums["loss"]/max(steps,1),
            "train_gate":sums["gate"]/max(steps,1),
            "train_selector":sums["selector"]/max(steps,1),
            "train_delta":sums["delta"]/max(steps,1),
            "train_move_target_fraction":sums["move_frac"]/max(steps,1),
            **{f"val_{k}":v for k,v in best_thr.items()},
        }
        history.append(row)
        print(json.dumps(row,sort_keys=True),flush=True)

        with (out_dir/"threshold_sweep_latest.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
        save_checkpoint(out_dir/"checkpoint_latest.tar",model,optimizer,epoch,mean,std,best_thr["move_threshold"],token_dim,group_dim,best_thr)

        improved=(best_thr["selected_utility"]>best_utility+1e-9 or
                  (abs(best_thr["selected_utility"]-best_utility)<=1e-9 and best_thr["harm08"]<best_harm))
        if improved:
            best_utility=best_thr["selected_utility"]; best_harm=best_thr["harm08"]; epochs_no_improve=0
            save_checkpoint(out_dir/"checkpoint_best.tar",model,optimizer,epoch,mean,std,best_thr["move_threshold"],token_dim,group_dim,best_thr)
            with (out_dir/"best.json").open("w") as f:
                json.dump(row,f,indent=2,sort_keys=True)
        else:
            epochs_no_improve+=1

        with (out_dir/"metrics.jsonl").open("w") as f:
            for h in history: f.write(json.dumps(h,sort_keys=True)+"\n")

        if ARGS.early_stop_patience>0 and epochs_no_improve>=ARGS.early_stop_patience:
            print(f"[REL] early stop at epoch={epoch}; no validation utility improvement for {epochs_no_improve} epochs",flush=True)
            break

    protocol={
        "train_cache_root":str(Path(ARGS.train_cache_root).resolve()),
        "val_cache_root":str(Path(ARGS.val_cache_root).resolve()),
        "train_frames":len(train_paths),
        "val_frames":len(val_paths),
        "val_scene_start":ARGS.val_scene_start,
        "group_feature_dim":group_dim,
        "token_dim":token_dim,
        "token_stat_rows":stat_rows,
        "architecture":{
            "d_model":ARGS.d_model,"nhead":ARGS.nhead,"num_layers":ARGS.num_layers,
            "ff_dim":ARGS.ff_dim,"dropout":ARGS.dropout,
        },
        "loss_weights":{"gate":ARGS.gate_weight,"selector":ARGS.selector_weight,"delta":ARGS.delta_weight},
        "decision":"move gate first; conditional alternative-center ranking second",
    }
    with (out_dir/"training_protocol.json").open("w") as f:
        json.dump(protocol,f,indent=2,sort_keys=True)


if __name__=="__main__":
    main()
