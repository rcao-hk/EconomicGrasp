#!/usr/bin/env python3
"""Cache-only inference/evaluation for a trained feature-ablation selector.

Loads one 3-layer MLP checkpoint, reconstructs the checkpoint-declared feature
subset, and reports candidate-level decodability plus native/raw/learned/oracle
exact-action selection metrics.  No Stage-1 forward and no DexNet call occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.stats import rankdata

from utils.ray_bestofk_diagnostic import gather_kn, select_exact_oracle, select_raw_score
from utils.ray_pairwise_feature_ablation import FEATURE_MODES, compose_feature_ablation
from utils.ray_pairwise_selector import RayPairwiseSelector, select_with_native_fallback

EPS=1e-8


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root",required=True)
    p.add_argument("--selector_checkpoint",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--split_name",default="eval")
    p.add_argument("--scene_min",type=int,default=-1)
    p.add_argument("--scene_max",type=int,default=-1,help="Exclusive upper bound; <0 means no upper bound")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--threshold",type=float,default=None)
    p.add_argument("--max_frames",type=int,default=0)
    p.add_argument("--progress_every",type=int,default=200)
    return p.parse_args()

ARGS=parse_args()


def sid(path:Path)->int:
    return int(path.parent.name.split("_")[-1])


def paths(root:Path)->List[Path]:
    out=[]
    for p in sorted(root.glob("scene_*/ann_*.npz")):
        s=sid(p)
        if ARGS.scene_min>=0 and s<ARGS.scene_min: continue
        if ARGS.scene_max>=0 and s>=ARGS.scene_max: continue
        out.append(p)
    return out[:ARGS.max_frames] if ARGS.max_frames>0 else out


def load_frame(path:Path,device):
    with np.load(path,allow_pickle=False) as d:
        selected=torch.from_numpy(d["selected_feature"].astype(np.float32)).to(device)
        mean=torch.from_numpy(d["mean_feature"].astype(np.float32)).to(device)
        raw=torch.from_numpy(d["raw_score"].astype(np.float32)).to(device)
        utility=d["utility"].astype(np.float32)
        friction=d["friction"].astype(np.float32)
        valid=d["valid"].astype(bool)
        offsets=torch.from_numpy(d["offsets_mm"].astype(np.float32)).to(device)
        zero=int(np.asarray(d["zero_index"]).reshape(-1)[0])
    return selected,mean,raw,utility,friction,valid,offsets,zero


def success(f,t=0.8):
    f=np.asarray(f,np.float32)
    return np.isfinite(f)&(f>0)&(f<=t+1e-6)


def pearson(x,y):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    if x.size<2 or np.std(x)<=1e-12 or np.std(y)<=1e-12: return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def spearman(x,y):
    if len(x)<2: return float("nan")
    return pearson(rankdata(x,method="average"),rankdata(y,method="average"))


def auroc(y,score):
    y=np.asarray(y,bool); score=np.asarray(score,np.float64)
    pos=int(y.sum()); neg=int((~y).sum())
    if pos==0 or neg==0: return float("nan")
    r=rankdata(score,method="average")
    return float((r[y].sum()-pos*(pos+1)/2)/(pos*neg))


def ap(y,score):
    y=np.asarray(y,bool); score=np.asarray(score,np.float64); pos=int(y.sum())
    if pos==0: return float("nan")
    order=np.argsort(-score,kind="stable"); yy=y[order].astype(np.float64)
    prec=np.cumsum(yy)/(np.arange(len(yy))+1)
    return float((prec*yy).sum()/pos)


def gather(arr,k):
    arr=np.asarray(arr); k=np.asarray(k,np.int64)
    return arr[k,np.arange(arr.shape[1])]


def policy_accumulate(store,u,f,k,z,ko):
    kn=np.full(k.shape,z,np.int64)
    us,un=gather(u,k),gather(u,kn)
    fs,fn=gather(f,k),gather(f,kn)
    ss,sn=success(fs),success(fn)
    for key,val in {
        "utility":us,"native_utility":un,"success08":ss,"native_success08":sn,
        "rescue08":(~sn)&ss,"harm08":sn&(~ss),"change":k!=z,"match_oracle":k==ko,
        "beneficial_switch":(k!=z)&(us>un+EPS),"harmful_switch":(k!=z)&(us<un-EPS),
    }.items(): store.setdefault(key,[]).append(val)


def aggregate(store):
    cat={k:np.concatenate(v) for k,v in store.items()}
    change=cat["change"]
    m={
        "utility":float(cat["utility"].mean()),
        "native_utility":float(cat["native_utility"].mean()),
        "success08":float(cat["success08"].mean()),
        "native_success08":float(cat["native_success08"].mean()),
        "rescue08":float(cat["rescue08"].mean()),
        "harm08":float(cat["harm08"].mean()),
        "change_rate":float(change.mean()),
        "match_oracle":float(cat["match_oracle"].mean()),
    }
    m["utility_gain"]=m["utility"]-m["native_utility"]
    m["success08_gain"]=m["success08"]-m["native_success08"]
    if change.any():
        m["switch_precision_beneficial"]=float(cat["beneficial_switch"][change].mean())
        m["switch_harmful_fraction"]=float(cat["harmful_switch"][change].mean())
    else:
        m["switch_precision_beneficial"]=float("nan"); m["switch_harmful_fraction"]=float("nan")
    return m


def main():
    device=torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(ARGS.selector_checkpoint,map_location=device)
    mode=str(ckpt.get("feature_mode",""))
    if mode not in FEATURE_MODES: raise RuntimeError(f"Checkpoint has invalid/missing feature_mode={mode!r}")
    model=RayPairwiseSelector(int(ckpt["feature_dim"]),int(ckpt["hidden_dim"]),float(ckpt["dropout"])).to(device)
    model.load_state_dict(ckpt["selector_state_dict"]); model.eval()
    mean=ckpt["feature_mean"].to(device).float(); std=ckpt["feature_std"].to(device).float()
    threshold=float(ckpt.get("selector_threshold",0.0) if ARGS.threshold is None else ARGS.threshold)
    pp=paths(Path(ARGS.cache_root))
    if not pp: raise RuntimeError("No cache frames selected.")
    print(f"[ABLATE-INFER] split={ARGS.split_name} mode={mode} frames={len(pp)} threshold={threshold:.4f}")

    pred_all=[]; delta_all=[]; stores={x:{} for x in ("native","raw","learned_t0","learned_threshold","oracle")}
    with torch.no_grad():
        for i,p in enumerate(pp):
            selected,mn,raw_t,u,f,v,offsets,z=load_frame(p,device)
            x=compose_feature_ablation(selected,mn,raw_t,offsets,z,mode)
            K,N,D=x.shape
            if D!=model.feature_dim: raise RuntimeError(f"Feature dim mismatch in {p}: {D} vs {model.feature_dim}")
            pred=model(((x-mean)/std.clamp_min(1e-5)).reshape(K*N,D)).reshape(K,N)
            pred[z]=0.0; pred_np=pred.cpu().numpy(); raw=raw_t.cpu().numpy()
            delta=u-u[z:z+1]; mask=v.copy(); mask[z]=False
            pred_all.append(pred_np[mask]); delta_all.append(delta[mask])
            kn=np.full(N,z,np.int64); kr=select_raw_score(raw,v); ko=select_exact_oracle(u,raw,v)
            k0=select_with_native_fallback(pred_np,v,z,0.0); kt=select_with_native_fallback(pred_np,v,z,threshold)
            for name,k in (("native",kn),("raw",kr),("learned_t0",k0),("learned_threshold",kt),("oracle",ko)):
                policy_accumulate(stores[name],u,f,k,z,ko)
            if ARGS.progress_every>0 and (i+1)%ARGS.progress_every==0:
                print(f"[ABLATE-INFER] {i+1}/{len(pp)}",flush=True)

    pred=np.concatenate(pred_all); delta=np.concatenate(delta_all); inf=np.abs(delta)>EPS; bene=delta>EPS
    candidate={
        "num_candidates":int(len(delta)),"spearman_all":spearman(pred,delta),"pearson_all":pearson(pred,delta),
        "beneficial_vs_all_auroc":auroc(bene,pred),"beneficial_vs_all_auprc":ap(bene,pred),
        "informative_fraction":float(inf.mean()),
    }
    if inf.any():
        candidate.update({"spearman_informative":spearman(pred[inf],delta[inf]),
                          "sign_auroc":auroc(delta[inf]>0,pred[inf]),
                          "sign_auprc_beneficial":ap(delta[inf]>0,pred[inf])})
    policies={k:aggregate(v) for k,v in stores.items()}
    headroom=policies["oracle"]["utility"]-policies["native"]["utility"]
    for name in ("raw","learned_t0","learned_threshold"):
        policies[name]["utility_headroom_recovery"]=(policies[name]["utility_gain"]/headroom if abs(headroom)>1e-12 else float("nan"))
    result={"split":ARGS.split_name,"cache_root":str(Path(ARGS.cache_root).resolve()),
            "checkpoint":str(Path(ARGS.selector_checkpoint).resolve()),"feature_mode":mode,
            "checkpoint_epoch":int(ckpt.get("epoch",-1)),"threshold":threshold,"num_frames":len(pp),
            "candidate_level":candidate,"policies":policies,
            "oracle_utility_headroom":headroom}
    out=Path(ARGS.output_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.json").open("w") as fh: json.dump(result,fh,indent=2,sort_keys=True)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
