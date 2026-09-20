#!/usr/bin/env python3
"""Rep-A-P0: selection-policy diagnostic without retraining.

Separates representation/scoring quality from conservative native fallback.
For A0-A3, evaluate the SAME checkpoints under:
  1) each checkpoint validation-selected margin;
  2) common fixed margins (default 0 and 0.1);
  3) a validation-only margin matched to A0 nominal validation move rate.

Matched margins are selected from nominal test_seen predictions only. No
Similar/Novel label or corruption severity is used to select a policy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from rep_a_common import (
    DEFAULT_CASES, VARIANTS, aggregate, array_sha, check_runtime_sources,
    exclusive_run, file_sha, list_frames, load_torch, metrics, parse_case,
    perturb_depth, read_frame, save_json, seed_for, tensors,
)
from rep_a_model import RepAReaderScorer


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--train-root", required=True,
                   help="Rep-A training root containing A0...A3/checkpoint_best.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=("test_similar","test_novel"), required=True)
    p.add_argument("--variants", default="A0,A1,A2,A3")
    p.add_argument("--reference-variant", default="A0", choices=VARIANTS)
    p.add_argument("--fixed-margins", default="0,0.1")
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-val-frames", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    return p


def load_model(cache_root, ckpt_path, device):
    ck=load_torch(ckpt_path)
    init=load_torch(Path(cache_root)/"reader_init.pt")
    if ck["contract"] != init["contract"] or ck["model_config"] != init["model_config"]:
        raise RuntimeError(f"Checkpoint/cache contract mismatch: {ckpt_path}")
    model=RepAReaderScorer(init, ck["variant"]).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    info={"variant":ck["variant"], "margin":float(ck["margin"]),
          "epoch":int(ck["epoch"]), "contract":ck["contract"],
          "checkpoint_sha256":file_sha(ckpt_path)}
    del ck, init
    return model, info


def advantage(prob, d):
    pu=np.asarray(prob, np.float64).mean(-1)
    valid=d["valid"].astype(bool)
    z=int(d["zero_index"])
    score=np.where(valid,pu,-np.inf).copy()
    score[z]=-np.inf
    has=np.isfinite(score).any(0)
    best=np.where(has, score.max(0), -np.inf)
    return np.where(has, best-pu[z], -np.inf)


def margin_for_target(advantages, target_rate):
    """Find nonnegative strict-threshold margin closest to target move rate."""
    adv=np.concatenate(advantages).astype(np.float64)
    finite=adv[np.isfinite(adv)]
    n=len(adv)
    if n==0:
        raise ValueError("No validation queries")
    if not 0 <= target_rate <= 1:
        raise ValueError(target_rate)
    positive=np.sort(finite[finite>0])[::-1]
    target_count=int(round(target_rate*n))
    candidates=[0.0,1.0]
    if len(positive):
        idx=min(max(target_count,0),len(positive))
        if 0 < idx < len(positive):
            a,b=positive[idx-1],positive[idx]
            candidates.extend([float(a),float(b),float((a+b)/2),
                               float(np.nextafter(a,np.inf)),
                               float(np.nextafter(a,-np.inf))])
        elif idx==0:
            candidates.append(float(np.nextafter(positive[0],np.inf)))
        else:
            candidates.append(0.0)
    candidates=[min(1.0,max(0.0,float(x))) for x in candidates if np.isfinite(x)]
    best=None
    for m in sorted(set(candidates)):
        rate=float(np.mean(adv>m))
        key=(abs(rate-target_rate), abs(m-0.0), m)
        if best is None or key<best[0]:
            best=(key,m,rate)
    return float(best[1]), float(best[2])


@torch.no_grad()
def nominal_val_stats(model, info, paths, device):
    advantages=[]; rows=[]
    for path in paths:
        d=read_frame(path, info["contract"])
        t=tensors(d,device)
        prob=model(t).sigmoid().cpu().numpy()
        advantages.append(advantage(prob,d))
        rows.append(metrics(prob,d,info["margin"])[0])
        del d,t,prob
    return aggregate(rows), advantages


def summarize_rows(rows):
    out={}
    variants=sorted({r["variant"] for r in rows})
    policies=sorted({r["policy"] for r in rows})
    cases=sorted({r["case"] for r in rows})
    extras=("probability_drift","representation_cosine_distance",
            "representation_relative_l2","selection_turnover",
            "depth_rms_mm","depth_bias_mm","clamped_fraction",
            "top_half_utility_gain","top_half_success08_gain")
    for variant in variants:
        out[variant]={}
        for policy in policies:
            chosen=[x for x in rows if x["variant"]==variant and x["policy"]==policy]
            if not chosen: continue
            out[variant][policy]={}
            for case in cases:
                rr=[x for x in chosen if x["case"]==case]
                if not rr: continue
                agg=aggregate(rr)
                nq=sum(x["num_queries"] for x in rr)
                for key in extras:
                    agg[key]=sum(x[key]*x["num_queries"] for x in rr)/nq
                out[variant][policy][case]=agg
            clean=out[variant][policy].get("nominal")
            if clean is not None:
                for item in out[variant][policy].values():
                    item["utility_drop_from_nominal"]=clean["selected_utility"]-item["selected_utility"]
                    item["success08_drop_from_nominal"]=clean["success08"]-item["success08"]
    return out


@torch.no_grad()
def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    check_runtime_sources(args.cache_root)
    variants=[x.strip() for x in args.variants.split(",") if x.strip()]
    if not variants or any(v not in VARIANTS for v in variants):
        raise ValueError(f"Invalid variants: {variants}")
    if args.reference_variant not in variants:
        raise ValueError("reference-variant must be included in --variants")
    fixed=[float(x) for x in args.fixed_margins.split(",") if x.strip()]
    if any(not 0<=x<=1 for x in fixed): raise ValueError("fixed margins must be in [0,1]")
    cases=list(dict.fromkeys(["nominal"]+[x.strip() for x in args.cases.split(",") if x.strip()]))
    for case in cases: parse_case(case)
    device=torch.device(args.device)
    val_paths=list_frames(args.cache_root,"test_seen",args.max_val_frames)
    test_paths=list_frames(args.cache_root,args.split,args.max_frames)
    train_root=Path(args.train_root)
    ckpts={v:train_root/v/"checkpoint_best.pt" for v in variants}
    for v,p in ckpts.items():
        if not p.is_file(): raise FileNotFoundError(p)

    # Determine ONE target move rate from A0 nominal validation using A0 own
    # validation-selected margin. This target is independent of test split/case.
    ref_model,ref_info=load_model(args.cache_root,ckpts[args.reference_variant],device)
    ref_val,_=nominal_val_stats(ref_model,ref_info,val_paths,device)
    target_rate=float(ref_val["move_rate"])
    del ref_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; policy_meta={}
    with exclusive_run(out/".p0.lock"):
        for vi,variant in enumerate(variants):
            model,info=load_model(args.cache_root,ckpts[variant],device)
            _,advs=nominal_val_stats(model,info,val_paths,device)
            matched_margin,matched_rate=margin_for_target(advs,target_rate)
            policies={"val_selected":float(info["margin"]),
                      "matched_move_rate":matched_margin}
            for m in fixed:
                policies[f"fixed_{m:g}"]=m
            policy_meta[variant]={"checkpoint_margin":float(info["margin"]),
                "matched_margin":matched_margin,"matched_val_move_rate":matched_rate,
                "target_val_move_rate":target_rate,"checkpoint_epoch":info["epoch"],
                "checkpoint_sha256":info["checkpoint_sha256"]}
            print(f"[REP-A-P0] {variant}: val_margin={info['margin']:.3f} "
                  f"matched_margin={matched_margin:.6f} target_move={target_rate:.4f} "
                  f"matched_move={matched_rate:.4f}",flush=True)

            for i,path in enumerate(test_paths):
                d=read_frame(path,info["contract"]); t=tensors(d,device)
                action_sha=array_sha(d["actions"],d["valid"],d["friction"],d["query_ids"])
                logits0,rep0=model(t,return_repr=True)
                p0=logits0.sigmoid().cpu().numpy()
                nominal_sel={name:metrics(p0,d,m)[1] for name,m in policies.items()}
                valid=d["valid"].astype(bool)
                for case in cases:
                    seed=seed_for(args.seed,args.split,int(d["scene_id"]),int(d["anno_id"]),parse_case(case)[0])
                    dep,err=perturb_depth(t["depth"],case,seed)
                    logits,rep=(logits0,rep0) if case=="nominal" else model(t,dep,return_repr=True)
                    prob=logits.sigmoid().cpu().numpy()
                    cosine=(1-F.cosine_similarity(rep.float(),rep0.float(),dim=-1)).reshape(valid.shape).cpu().numpy()
                    relative=((rep-rep0).norm(dim=-1)/rep0.norm(dim=-1).clamp_min(1e-6)).reshape(valid.shape).cpu().numpy()
                    ids=np.argsort(-d["native_score"],kind="stable")[:max(1,len(d["native_score"])//2)]
                    small={key:d[key][:,ids] for key in ("valid","friction","utility")}; small["zero_index"]=d["zero_index"]
                    for pname,margin in policies.items():
                        met,selected=metrics(prob,d,margin)
                        top,_=metrics(prob[:,ids],small,margin)
                        rows.append({"variant":variant,"split":args.split,
                            "scene_id":int(d["scene_id"]),"anno_id":int(d["anno_id"]),
                            "action_sha":action_sha,"case":case,"policy":pname,
                            "margin":float(margin),**met,**err,
                            "probability_drift":float(np.abs(prob-p0)[valid].mean()),
                            "representation_cosine_distance":float(cosine[valid].mean()),
                            "representation_relative_l2":float(relative[valid].mean()),
                            "selection_turnover":float((selected!=nominal_sel[pname]).mean()),
                            "top_half_utility_gain":top["utility_gain"],
                            "top_half_success08_gain":top["success08_gain"]})
                if action_sha!=array_sha(d["actions"],d["valid"],d["friction"],d["query_ids"]):
                    raise RuntimeError("Actions/labels were mutated")
                del d,t,logits0,rep0,p0,logits,rep,prob,dep
                if (i+1)%max(args.progress_every,1)==0:
                    print(f"[REP-A-P0 {variant}] {i+1}/{len(test_paths)} frames",flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        summary=summarize_rows(rows)
        signature={"split":args.split,"seed":args.seed,"cases":cases,
                   "variants":variants,"reference_variant":args.reference_variant,
                   "target_val_move_rate":target_rate,"policies":policy_meta,
                   "fixed_margins":fixed}
        save_json(out/"summary.json",{"protocol":signature,"results":summary})
        with open(out/"per_frame.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[REP-A-P0] wrote {out}/summary.json and per_frame.csv",flush=True)


if __name__=="__main__":
    main()
