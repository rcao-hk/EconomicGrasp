#!/usr/bin/env python3
"""Evaluate a trained Rep-P0 geometry-source probe on a fixed-action cache.

No Stage-1, CAD evaluator, or action generation is run here. The test consumes
the frozen Rep-P0 cache, uses the validation-selected native-fallback margin from
the checkpoint, and reports action decodability plus K-ray selection quality.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from rep_p0_geometry_common import (
    GeometrySourceProbe,
    friction_to_cdf_targets,
    friction_utility,
    predicted_utility_from_logits,
    success08,
)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--margin",type=float,default=None)
    p.add_argument("--max_frames",type=int,default=0)
    p.add_argument("--progress_every",type=int,default=200)
    p.add_argument("--save_per_query",action="store_true")
    return p.parse_args()


ARGS=parse_args()
EPS=1e-8


def cache_paths(root):
    return sorted(Path(root).glob("scene_*/ann_*.npz"))


def load_frame(path,source):
    key=f"feat_{source}"
    with np.load(path,allow_pickle=False) as d:
        if key not in d.files:
            raise KeyError(f"{path} missing {key}")
        return {
            "feat":d[key].astype(np.float32),
            "valid":d["valid"].astype(bool),
            "friction":d["friction"].astype(np.float32),
            "utility":d["utility"].astype(np.float32),
            "offsets":d["offsets_mm"].astype(np.float32),
            "zero":int(np.asarray(d["zero_index"]).reshape(-1)[0]),
            "scene_id":int(np.asarray(d["scene_id"]).reshape(-1)[0]),
            "anno_id":int(np.asarray(d["anno_id"]).reshape(-1)[0]),
            "native_score":d["native_score"].astype(np.float32),
        }


def normalize(x,mean,std):
    return (x-mean)/np.maximum(std,1e-5)


def gather_kn(arr,k):
    arr=np.asarray(arr); k=np.asarray(k,np.int64)
    return arr[k,np.arange(arr.shape[1])]


def pearson(x,y):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    if x.size<2 or np.std(x)<=1e-12 or np.std(y)<=1e-12:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def spearman(x,y):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    if x.size<2:
        return float("nan")
    return pearson(rankdata(x,method="average"),rankdata(y,method="average"))


def auroc(y_true,score):
    y=np.asarray(y_true,bool); s=np.asarray(score,np.float64)
    pos=int(y.sum()); neg=int((~y).sum())
    if pos==0 or neg==0: return float("nan")
    ranks=rankdata(s,method="average")
    return float((ranks[y].sum()-pos*(pos+1)/2.0)/(pos*neg))


def average_precision(y_true,score):
    y=np.asarray(y_true,bool); s=np.asarray(score,np.float64)
    pos=int(y.sum())
    if pos==0: return float("nan")
    order=np.argsort(-s,kind="stable")
    yy=y[order].astype(np.float64)
    precision=np.cumsum(yy)/(np.arange(len(yy),dtype=np.float64)+1.0)
    return float((precision*yy).sum()/pos)


def select_policy(pred_u,exact_u,valid,zero,margin):
    K,Q=pred_u.shape
    alt=valid.copy(); alt[zero]=False
    score=np.where(alt,pred_u,-np.inf)
    best=np.argmax(score,axis=0).astype(np.int64)
    has_alt=alt.any(axis=0)
    best=np.where(has_alt,best,zero)
    adv=pred_u[best,np.arange(Q)]-pred_u[zero]
    selected=np.where(has_alt&(adv>float(margin)),best,zero).astype(np.int64)

    masked=np.where(valid,exact_u,-np.inf)
    exact_best=masked.max(axis=0)
    oracle_arg=np.argmax(masked,axis=0).astype(np.int64)
    oracle=np.where(exact_best>exact_u[zero]+EPS,oracle_arg,zero).astype(np.int64)
    return selected,oracle,best,adv


@torch.no_grad()
def main():
    device=torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(ARGS.checkpoint,map_location=device)
    source=str(ckpt["source"])
    model=GeometrySourceProbe(
        int(ckpt["feature_dim"]),
        int(ckpt["hidden_dim"]),
        float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    mean=ckpt["feature_mean"].cpu().numpy().astype(np.float32)
    std=ckpt["feature_std"].cpu().numpy().astype(np.float32)
    margin=float(ckpt.get("selection_margin",0.0) if ARGS.margin is None else ARGS.margin)

    paths=cache_paths(ARGS.cache_root)
    if ARGS.max_frames>0: paths=paths[:ARGS.max_frames]
    if not paths: raise RuntimeError("No cache files found.")

    pred_all=[]; exact_all=[]; succ_all=[]; cdf_losses=[]
    us=[]; un=[]; uo=[]; ss=[]; sn=[]; rescue=[]; harm=[]; change=[]
    scene_rows=[]; query_rows=[]; offset_hist={}
    total_queries=0

    for i,path in enumerate(paths):
        fr=load_frame(path,source)
        K,Q,Fdim=fr["feat"].shape
        x=torch.from_numpy(normalize(fr["feat"],mean,std)).to(device)
        logits=model(x.reshape(K*Q,Fdim)).reshape(K,Q,-1)
        pred_u=predicted_utility_from_logits(logits).cpu().numpy().astype(np.float32)

        mask=fr["valid"]
        mask_t=torch.from_numpy(mask).to(device=device,dtype=torch.bool)
        y=torch.from_numpy(friction_to_cdf_targets(fr["friction"][mask])).to(device)
        cdf_losses.append(float(F.binary_cross_entropy_with_logits(logits[mask_t],y).item()))

        pred_all.append(pred_u[mask])
        exact_all.append(fr["utility"][mask])
        succ_all.append(success08(fr["friction"][mask]))

        selected,oracle,best,adv=select_policy(pred_u,fr["utility"],fr["valid"],fr["zero"],margin)
        u_sel=gather_kn(fr["utility"],selected)
        u_nat=fr["utility"][fr["zero"]]
        u_orc=gather_kn(fr["utility"],oracle)
        s_sel=success08(gather_kn(fr["friction"],selected))
        s_nat=success08(fr["friction"][fr["zero"]])

        us.append(u_sel); un.append(u_nat); uo.append(u_orc)
        ss.append(s_sel); sn.append(s_nat)
        rescue.append((~s_nat)&s_sel); harm.append(s_nat&(~s_sel))
        change.append(selected!=fr["zero"])

        for k in selected:
            off=float(fr["offsets"][int(k)])
            offset_hist[str(off)]=offset_hist.get(str(off),0)+1

        scene_rows.append({
            "source":source,
            "scene_id":fr["scene_id"],
            "anno_id":fr["anno_id"],
            "num_queries":Q,
            "native_utility":float(u_nat.mean()),
            "selected_utility":float(u_sel.mean()),
            "oracle_utility":float(u_orc.mean()),
            "utility_gain":float((u_sel-u_nat).mean()),
            "native_success08":float(s_nat.mean()),
            "selected_success08":float(s_sel.mean()),
            "success08_gain":float((s_sel.astype(np.float32)-s_nat.astype(np.float32)).mean()),
            "rescue08":float(((~s_nat)&s_sel).mean()),
            "harm08":float((s_nat&(~s_sel)).mean()),
            "change_rate":float((selected!=fr["zero"]).mean()),
        })

        if ARGS.save_per_query:
            for q in range(Q):
                query_rows.append({
                    "source":source,
                    "scene_id":fr["scene_id"],
                    "anno_id":fr["anno_id"],
                    "query_id":q,
                    "native_score":float(fr["native_score"][q]),
                    "selected_k":int(selected[q]),
                    "selected_offset_mm":float(fr["offsets"][selected[q]]),
                    "best_pred_k":int(best[q]),
                    "pred_advantage":float(adv[q]),
                    "native_utility":float(u_nat[q]),
                    "selected_utility":float(u_sel[q]),
                    "oracle_utility":float(u_orc[q]),
                    "native_success08":int(s_nat[q]),
                    "selected_success08":int(s_sel[q]),
                    "rescue08":int((not s_nat[q]) and s_sel[q]),
                    "harm08":int(s_nat[q] and (not s_sel[q])),
                })
        total_queries+=Q
        if ARGS.progress_every>0 and (i+1)%ARGS.progress_every==0:
            print(f"[REP-P0-TEST][{source}] {i+1}/{len(paths)} frames",flush=True)

    cat=lambda xs:np.concatenate(xs)
    pred=cat(pred_all); exact=cat(exact_all); succ=cat(succ_all)
    us=cat(us); un=cat(un); uo=cat(uo); ss=cat(ss); sn=cat(sn)
    head=float((uo-un).mean())
    gain=float((us-un).mean())

    summary={
        "experiment":"Rep-P0 fixed-action geometry-source diagnosis",
        "source":source,
        "checkpoint":str(Path(ARGS.checkpoint).resolve()),
        "checkpoint_epoch":int(ckpt.get("epoch",-1)),
        "selection_margin":margin,
        "num_frames":len(paths),
        "num_queries":total_queries,
        "candidate_metrics":{
            "cdf_bce":float(np.mean(cdf_losses)),
            "pred_exact_utility_pearson":pearson(pred,exact),
            "pred_exact_utility_spearman":spearman(pred,exact),
            "success08_auroc":auroc(succ,pred),
            "success08_auprc":average_precision(succ,pred),
            "success08_positive_fraction":float(succ.mean()),
        },
        "policy":{
            "native_utility":float(un.mean()),
            "selected_utility":float(us.mean()),
            "oracle_utility":float(uo.mean()),
            "utility_gain":gain,
            "utility_headroom_recovery":gain/head if abs(head)>1e-12 else float("nan"),
            "native_success08":float(sn.mean()),
            "selected_success08":float(ss.mean()),
            "success08_gain":float((ss.astype(np.float32)-sn.astype(np.float32)).mean()),
            "rescue08":float(cat(rescue).mean()),
            "harm08":float(cat(harm).mean()),
            "change_rate":float(cat(change).mean()),
            "offset_hist":offset_hist,
        },
    }

    out=Path(ARGS.output_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.json").open("w") as f:
        json.dump(summary,f,indent=2,sort_keys=True)
    with (out/"per_frame.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(scene_rows[0].keys())); w.writeheader(); w.writerows(scene_rows)
    if ARGS.save_per_query and query_rows:
        with (out/"per_query.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(query_rows[0].keys())); w.writeheader(); w.writerows(query_rows)
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
