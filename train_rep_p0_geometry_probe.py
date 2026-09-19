#!/usr/bin/env python3
"""Train one Rep-P0 geometry-source action-quality probe.

Each source uses the same MLP, the same fixed physical actions, and the same
exact-action labels. The probe predicts the six GraspNet friction-threshold CDF
labels from a gripper-centric geometry descriptor.

Validation selects only a native-fallback margin; the action set is never
changed by the geometry source.
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
from scipy.stats import rankdata

from rep_p0_geometry_common import (
    FRICTION_THRESHOLDS,
    GEOMETRY_SOURCES,
    GeometrySourceProbe,
    friction_to_cdf_targets,
    friction_utility,
    predicted_utility_from_logits,
    success08,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_cache_root", required=True)
    p.add_argument("--val_cache_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source", required=True, choices=GEOMETRY_SOURCES)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--margin_max", type=float, default=0.5)
    p.add_argument("--margin_steps", type=int, default=51)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_train_frames", type=int, default=0)
    p.add_argument("--max_val_frames", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=200)
    return p.parse_args()


ARGS = parse_args()
EPS = 1e-8


def cache_paths(root: Path):
    return sorted(root.glob("scene_*/ann_*.npz"))


def load_frame(path: Path, source: str):
    key = f"feat_{source}"
    with np.load(path, allow_pickle=False) as d:
        if key not in d.files:
            raise KeyError(f"{path} does not contain {key}")
        feat = d[key].astype(np.float32)
        valid = d["valid"].astype(bool)
        friction = d["friction"].astype(np.float32)
        utility = d["utility"].astype(np.float32)
        offsets = d["offsets_mm"].astype(np.float32)
        zero = int(np.asarray(d["zero_index"]).reshape(-1)[0])
        scene_id = int(np.asarray(d["scene_id"]).reshape(-1)[0])
        anno_id = int(np.asarray(d["anno_id"]).reshape(-1)[0])
    if feat.shape[:2] != valid.shape or friction.shape != valid.shape:
        raise RuntimeError(f"Malformed cache shapes in {path}")
    return {
        "feat": feat,
        "valid": valid,
        "friction": friction,
        "utility": utility,
        "offsets": offsets,
        "zero": zero,
        "scene_id": scene_id,
        "anno_id": anno_id,
    }


def feature_stats(paths, source):
    total = total2 = None
    count = 0
    for i, path in enumerate(paths):
        fr = load_frame(path, source)
        rows = fr["feat"][fr["valid"]]
        if not len(rows):
            continue
        rows64 = rows.astype(np.float64)
        s = rows64.sum(axis=0)
        s2 = np.square(rows64).sum(axis=0)
        total = s if total is None else total + s
        total2 = s2 if total2 is None else total2 + s2
        count += len(rows)
        if ARGS.progress_every > 0 and (i + 1) % ARGS.progress_every == 0:
            print(f"[REP-P0][STATS][{source}] {i+1}/{len(paths)} rows={count}", flush=True)
    if count == 0:
        raise RuntimeError("No valid rows for feature normalization.")
    mean = total / count
    var = np.maximum(total2 / count - mean ** 2, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), count


def normalize(x, mean, std):
    return (x - mean) / np.maximum(std, 1e-5)


def gather_kn(arr, k):
    arr = np.asarray(arr)
    k = np.asarray(k, np.int64)
    return arr[k, np.arange(arr.shape[1])]


def select_policy(pred_u, exact_u, valid, zero, margin):
    K, Q = pred_u.shape
    alt_valid = valid.copy()
    alt_valid[zero] = False
    score = np.where(alt_valid, pred_u, -np.inf)
    best_alt = np.argmax(score, axis=0).astype(np.int64)
    has_alt = alt_valid.any(axis=0)
    best_alt = np.where(has_alt, best_alt, zero)
    best_score = pred_u[best_alt, np.arange(Q)]
    adv = best_score - pred_u[zero]
    selected = np.where(has_alt & (adv > float(margin)), best_alt, zero).astype(np.int64)

    masked_exact = np.where(valid, exact_u, -np.inf)
    exact_best = masked_exact.max(axis=0)
    oracle_argmax = np.argmax(masked_exact, axis=0).astype(np.int64)
    oracle = np.where(exact_best > exact_u[zero] + EPS, oracle_argmax, zero).astype(np.int64)
    return selected, oracle, best_alt, adv


def policy_metrics(frames, margin):
    us=[]; un=[]; uo=[]; ss=[]; sn=[]; rescue=[]; harm=[]; change=[]
    for fr in frames:
        selected, oracle, _, _ = select_policy(
            fr["pred_u"], fr["utility"], fr["valid"], fr["zero"], margin
        )
        u_sel = gather_kn(fr["utility"], selected)
        u_nat = fr["utility"][fr["zero"]]
        u_orc = gather_kn(fr["utility"], oracle)
        s_sel = success08(gather_kn(fr["friction"], selected))
        s_nat = success08(fr["friction"][fr["zero"]])
        us.append(u_sel); un.append(u_nat); uo.append(u_orc)
        ss.append(s_sel); sn.append(s_nat)
        rescue.append((~s_nat) & s_sel); harm.append(s_nat & (~s_sel))
        change.append(selected != fr["zero"])
    cat=lambda xs: np.concatenate(xs)
    us=cat(us); un=cat(un); uo=cat(uo); ss=cat(ss); sn=cat(sn)
    out={
        "margin":float(margin),
        "selected_utility":float(us.mean()),
        "native_utility":float(un.mean()),
        "oracle_utility":float(uo.mean()),
        "utility_gain":float((us-un).mean()),
        "selected_success08":float(ss.mean()),
        "native_success08":float(sn.mean()),
        "success08_gain":float((ss.astype(np.float32)-sn.astype(np.float32)).mean()),
        "rescue08":float(cat(rescue).mean()),
        "harm08":float(cat(harm).mean()),
        "change_rate":float(cat(change).mean()),
    }
    head=out["oracle_utility"]-out["native_utility"]
    out["utility_headroom_recovery"]=out["utility_gain"]/head if abs(head)>1e-12 else float("nan")
    return out


def tune_margin(frames):
    margins=np.linspace(0.0,ARGS.margin_max,max(2,ARGS.margin_steps))
    rows=[policy_metrics(frames,float(m)) for m in margins]
    best=max(rows,key=lambda r:(r["selected_utility"],-r["harm08"],-r["change_rate"]))
    return best,rows


@torch.no_grad()
def collect_validation(model, paths, source, feat_mean, feat_std, device):
    model.eval()
    frames=[]
    for i,path in enumerate(paths):
        fr=load_frame(path,source)
        K,Q,Fdim=fr["feat"].shape
        x=torch.from_numpy(normalize(fr["feat"],feat_mean,feat_std)).to(device)
        logits=model(x.reshape(K*Q,Fdim)).reshape(K,Q,-1)
        pred_u=predicted_utility_from_logits(logits).cpu().numpy().astype(np.float32)
        frames.append({**fr,"pred_u":pred_u})
        if ARGS.progress_every>0 and (i+1)%ARGS.progress_every==0:
            print(f"[REP-P0][VAL][{source}] {i+1}/{len(paths)}",flush=True)
    return frames


def save_checkpoint(path,model,optimizer,epoch,mean,std,margin,feature_dim,val_metrics):
    torch.save({
        "experiment":"Rep-P0 fixed-action geometry-source probe",
        "source":ARGS.source,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "epoch":int(epoch),
        "feature_dim":int(feature_dim),
        "hidden_dim":int(ARGS.hidden_dim),
        "dropout":float(ARGS.dropout),
        "feature_mean":torch.from_numpy(mean),
        "feature_std":torch.from_numpy(std),
        "selection_margin":float(margin),
        "val_metrics":dict(val_metrics),
        "friction_thresholds":FRICTION_THRESHOLDS.tolist(),
        "training_target":"six-threshold exact-action CDF BCE",
    },path)


def main():
    random.seed(ARGS.seed); np.random.seed(ARGS.seed); torch.manual_seed(ARGS.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(ARGS.seed)
    if ARGS.grad_accum_steps<1:
        raise ValueError("grad_accum_steps must be >=1")
    device=torch.device(ARGS.device if torch.cuda.is_available() else "cpu")

    train_paths=cache_paths(Path(ARGS.train_cache_root))
    val_paths=cache_paths(Path(ARGS.val_cache_root))
    if ARGS.max_train_frames>0: train_paths=train_paths[:ARGS.max_train_frames]
    if ARGS.max_val_frames>0: val_paths=val_paths[:ARGS.max_val_frames]
    if not train_paths or not val_paths:
        raise RuntimeError(f"Need train+val caches; got train={len(train_paths)} val={len(val_paths)}")

    first=load_frame(train_paths[0],ARGS.source)
    feature_dim=int(first["feat"].shape[-1])
    feat_mean,feat_std,stat_rows=feature_stats(train_paths,ARGS.source)

    out=Path(ARGS.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=GeometrySourceProbe(feature_dim,ARGS.hidden_dim,ARGS.dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=ARGS.learning_rate,weight_decay=ARGS.weight_decay)

    history=[]; best_utility=-math.inf; best_harm=math.inf; no_improve=0
    rng=random.Random(ARGS.seed)

    for epoch in range(ARGS.epochs):
        model.train(); order=list(train_paths); rng.shuffle(order)
        loss_sum=0.0; frames_seen=0; updates=0
        for start in range(0,len(order),ARGS.grad_accum_steps):
            chunk=order[start:start+ARGS.grad_accum_steps]
            optimizer.zero_grad(set_to_none=True)
            for path in chunk:
                fr=load_frame(path,ARGS.source)
                mask=fr["valid"]
                rows=fr["feat"][mask]
                friction=fr["friction"][mask]
                if len(rows)==0:
                    continue
                x=torch.from_numpy(normalize(rows,feat_mean,feat_std)).to(device)
                y=torch.from_numpy(friction_to_cdf_targets(friction)).to(device)
                logits=model(x)
                loss=F.binary_cross_entropy_with_logits(logits,y)
                (loss/max(len(chunk),1)).backward()
                loss_sum+=float(loss.detach())
                frames_seen+=1
            if frames_seen:
                if ARGS.grad_clip>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),ARGS.grad_clip)
                optimizer.step(); updates+=1
            if ARGS.progress_every>0 and frames_seen>0 and frames_seen%ARGS.progress_every<len(chunk):
                print(
                    f"[REP-P0][TRAIN][{ARGS.source}] epoch={epoch} "
                    f"frames={frames_seen}/{len(order)} loss={loss_sum/frames_seen:.4f}",
                    flush=True,
                )

        val_frames=collect_validation(model,val_paths,ARGS.source,feat_mean,feat_std,device)
        best_margin,sweep=tune_margin(val_frames)
        row={
            "epoch":epoch,
            "source":ARGS.source,
            "train_loss":loss_sum/max(frames_seen,1),
            "optimizer_updates":updates,
            **{f"val_{k}":v for k,v in best_margin.items()},
        }
        history.append(row)
        print(json.dumps(row,sort_keys=True),flush=True)

        with (out/"margin_sweep_latest.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
        save_checkpoint(
            out/"checkpoint_latest.tar",model,optimizer,epoch,feat_mean,feat_std,
            best_margin["margin"],feature_dim,best_margin
        )
        improved=(
            best_margin["selected_utility"]>best_utility+1e-9
            or (
                abs(best_margin["selected_utility"]-best_utility)<=1e-9
                and best_margin["harm08"]<best_harm
            )
        )
        if improved:
            best_utility=best_margin["selected_utility"]; best_harm=best_margin["harm08"]; no_improve=0
            save_checkpoint(
                out/"checkpoint_best.tar",model,optimizer,epoch,feat_mean,feat_std,
                best_margin["margin"],feature_dim,best_margin
            )
            with (out/"best.json").open("w") as f:
                json.dump(row,f,indent=2,sort_keys=True)
        else:
            no_improve+=1

        with (out/"metrics.jsonl").open("w") as f:
            for h in history: f.write(json.dumps(h,sort_keys=True)+"\n")

        if ARGS.early_stop_patience>0 and no_improve>=ARGS.early_stop_patience:
            print(f"[REP-P0] early stop source={ARGS.source} epoch={epoch}",flush=True)
            break

    protocol={
        "experiment":"Rep-P0 fixed-action geometry-source diagnosis",
        "source":ARGS.source,
        "train_cache_root":str(Path(ARGS.train_cache_root).resolve()),
        "val_cache_root":str(Path(ARGS.val_cache_root).resolve()),
        "train_frames":len(train_paths),
        "val_frames":len(val_paths),
        "feature_dim":feature_dim,
        "feature_stat_rows":stat_rows,
        "hidden_dim":ARGS.hidden_dim,
        "dropout":ARGS.dropout,
        "grad_accum_steps":ARGS.grad_accum_steps,
        "objective":"same six-threshold CDF BCE for all geometry sources",
    }
    with (out/"training_protocol.json").open("w") as f:
        json.dump(protocol,f,indent=2,sort_keys=True)


if __name__=="__main__":
    main()
