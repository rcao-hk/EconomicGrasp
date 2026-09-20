#!/usr/bin/env python3
"""Rep-A-P1: inference-only branch intervention on the trained A3 checkpoint.

Runs the SAME A3 weights as:
  full       = depth-conditioned reader + 0.1 RGB branch + action embedding
  no_rgb     = depth-conditioned reader + action embedding
  rgb_only   = 0.1 RGB branch + action embedding

No branch is retrained or renormalized. All A3 interventions use the SAME A3
validation-selected margin. A separately trained A1 checkpoint is reported only
as a reference under its own validation-selected margin.
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
    aggregate, array_sha, check_runtime_sources, exclusive_run, file_sha,
    list_frames, load_torch, metrics, parse_case, perturb_depth, read_frame,
    save_json, seed_for, tensors,
)
from rep_a_model import RepAReaderScorer

DEFAULT_P1_CASES="nominal,bias:-20,bias:20,bias:-40,bias:40,smooth:20"
INTERVENTIONS=("full","no_rgb","rgb_only")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=("test_similar","test_novel"), required=True)
    p.add_argument("--cases", default=DEFAULT_P1_CASES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--no-a1-reference", action="store_true")
    return p


def load_model(cache_root, ckpt_path, expected_variant, device):
    ck=load_torch(ckpt_path)
    if ck["variant"]!=expected_variant:
        raise RuntimeError(f"{ckpt_path} is {ck['variant']}, expected {expected_variant}")
    init=load_torch(Path(cache_root)/"reader_init.pt")
    if ck["contract"]!=init["contract"] or ck["model_config"]!=init["model_config"]:
        raise RuntimeError(f"Checkpoint/cache mismatch: {ckpt_path}")
    model=RepAReaderScorer(init,expected_variant).to(device)
    model.load_state_dict(ck["model"],strict=True); model.eval()
    info={"variant":expected_variant,"margin":float(ck["margin"]),
          "epoch":int(ck["epoch"]),"contract":ck["contract"],
          "checkpoint_sha256":file_sha(ckpt_path)}
    del ck,init
    return model,info


def fuse_from_components(model, comps, action_shape):
    if comps["rgb"] is None:
        raise RuntimeError("A3 checkpoint has no RGB branch")
    reps={
        "full":comps["depth"]+comps["rgb"]+comps["action"],
        "no_rgb":comps["depth"]+comps["action"],
        "rgb_only":comps["rgb"]+comps["action"],
    }
    outputs={}
    for name,rep in reps.items():
        logits=model.scorer(rep).reshape(*action_shape,6)
        outputs[name]=(logits,rep)
    return outputs


def component_stats(comps, nominal=None):
    out={}
    for key in ("depth","rgb","action"):
        x=comps[key]
        if x is None: continue
        out[f"{key}_norm"]=float(x.float().norm(dim=-1).mean())
        if nominal is not None and nominal.get(key) is not None:
            y=nominal[key]
            out[f"{key}_cosine_drift"]=float((1-F.cosine_similarity(x.float(),y.float(),dim=-1)).mean())
            out[f"{key}_relative_l2_drift"]=float(((x-y).float().norm(dim=-1)/y.float().norm(dim=-1).clamp_min(1e-6)).mean())
    if comps.get("rgb") is not None:
        out["rgb_to_depth_norm_ratio"]=out["rgb_norm"]/max(out["depth_norm"],1e-12)
    return out


def summarize(rows):
    result={}
    names=sorted({r["intervention"] for r in rows})
    cases=sorted({r["case"] for r in rows})
    extras=("probability_drift","representation_cosine_distance",
            "representation_relative_l2","selection_turnover",
            "depth_rms_mm","depth_bias_mm","clamped_fraction",
            "full_case_probability_distance","full_case_representation_cosine_distance",
            "full_case_representation_relative_l2","top_half_utility_gain",
            "top_half_success08_gain","depth_norm","rgb_norm","action_norm",
            "rgb_to_depth_norm_ratio","depth_cosine_drift","depth_relative_l2_drift",
            "rgb_cosine_drift","rgb_relative_l2_drift","action_cosine_drift",
            "action_relative_l2_drift")
    for name in names:
        result[name]={}
        for case in cases:
            rr=[x for x in rows if x["intervention"]==name and x["case"]==case]
            if not rr: continue
            agg=aggregate(rr); nq=sum(x["num_queries"] for x in rr)
            for key in extras:
                vals=[x for x in rr if key in x and x[key] is not None]
                if vals:
                    agg[key]=sum(x[key]*x["num_queries"] for x in vals)/sum(x["num_queries"] for x in vals)
            result[name][case]=agg
        clean=result[name].get("nominal")
        if clean:
            for item in result[name].values():
                item["utility_drop_from_nominal"]=clean["selected_utility"]-item["selected_utility"]
                item["success08_drop_from_nominal"]=clean["success08"]-item["success08"]
    return result


@torch.no_grad()
def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    check_runtime_sources(args.cache_root)
    cases=list(dict.fromkeys(["nominal"]+[x.strip() for x in args.cases.split(",") if x.strip()]))
    for case in cases: parse_case(case)
    device=torch.device(args.device)
    paths=list_frames(args.cache_root,args.split,args.max_frames)
    train_root=Path(args.train_root)
    a3_path=train_root/"A3"/"checkpoint_best.pt"
    a1_path=train_root/"A1"/"checkpoint_best.pt"
    if not a3_path.is_file(): raise FileNotFoundError(a3_path)
    if not args.no_a1_reference and not a1_path.is_file(): raise FileNotFoundError(a1_path)
    a3,a3i=load_model(args.cache_root,a3_path,"A3",device)
    a1=a1i=None
    if not args.no_a1_reference:
        a1,a1i=load_model(args.cache_root,a1_path,"A1",device)
    if a3.rgb_reader is None: raise RuntimeError("A3 has no RGB reader")

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with exclusive_run(out/".p1.lock"):
        for i,path in enumerate(paths):
            d=read_frame(path,a3i["contract"]); t=tensors(d,device)
            action_sha=array_sha(d["actions"],d["valid"],d["friction"],d["query_ids"])
            valid=d["valid"].astype(bool); shape=d["actions"].shape[:2]

            # Nominal branch-specific references.
            _,rep_full0,comps0=a3(t,return_repr=True,return_components=True,intervention="full")
            outs0=fuse_from_components(a3,comps0,shape)
            prob0={name:lr[0].sigmoid().cpu().numpy() for name,lr in outs0.items()}
            sel0={name:metrics(prob0[name],d,a3i["margin"])[1] for name in INTERVENTIONS}
            a1_logits0=a1(t) if a1 is not None else None
            a1_prob0=a1_logits0.sigmoid().cpu().numpy() if a1 is not None else None
            a1_sel0=metrics(a1_prob0,d,a1i["margin"])[1] if a1 is not None else None

            for case in cases:
                seed=seed_for(args.seed,args.split,int(d["scene_id"]),int(d["anno_id"]),parse_case(case)[0])
                dep,err=perturb_depth(t["depth"],case,seed)
                if case=="nominal":
                    comps=comps0; outs=outs0
                else:
                    _,_,comps=a3(t,dep,return_repr=True,return_components=True,intervention="full")
                    outs=fuse_from_components(a3,comps,shape)
                comp=component_stats(comps,comps0)
                full_logits,full_rep=outs["full"]
                full_prob=full_logits.sigmoid().cpu().numpy()

                ids=np.argsort(-d["native_score"],kind="stable")[:max(1,len(d["native_score"])//2)]
                small={key:d[key][:,ids] for key in ("valid","friction","utility")}; small["zero_index"]=d["zero_index"]
                for name in INTERVENTIONS:
                    logits,rep=outs[name]; prob=logits.sigmoid().cpu().numpy()
                    met,selected=metrics(prob,d,a3i["margin"]); top,_=metrics(prob[:,ids],small,a3i["margin"])
                    rep0=outs0[name][1]
                    cosine=(1-F.cosine_similarity(rep.float(),rep0.float(),dim=-1)).reshape(valid.shape).cpu().numpy()
                    relative=((rep-rep0).norm(dim=-1)/rep0.norm(dim=-1).clamp_min(1e-6)).reshape(valid.shape).cpu().numpy()
                    fcos=(1-F.cosine_similarity(rep.float(),full_rep.float(),dim=-1)).reshape(valid.shape).cpu().numpy()
                    frel=((rep-full_rep).norm(dim=-1)/full_rep.norm(dim=-1).clamp_min(1e-6)).reshape(valid.shape).cpu().numpy()
                    rows.append({"intervention":name,"source_checkpoint":"A3","split":args.split,
                        "scene_id":int(d["scene_id"]),"anno_id":int(d["anno_id"]),
                        "action_sha":action_sha,"case":case,"margin":a3i["margin"],**met,**err,**comp,
                        "probability_drift":float(np.abs(prob-prob0[name])[valid].mean()),
                        "representation_cosine_distance":float(cosine[valid].mean()),
                        "representation_relative_l2":float(relative[valid].mean()),
                        "selection_turnover":float((selected!=sel0[name]).mean()),
                        "full_case_probability_distance":float(np.abs(prob-full_prob)[valid].mean()),
                        "full_case_representation_cosine_distance":float(fcos[valid].mean()),
                        "full_case_representation_relative_l2":float(frel[valid].mean()),
                        "top_half_utility_gain":top["utility_gain"],
                        "top_half_success08_gain":top["success08_gain"]})

                if a1 is not None:
                    logits1=a1_logits0 if case=="nominal" else a1(t,dep)
                    prob1=logits1.sigmoid().cpu().numpy()
                    met1,sel1=metrics(prob1,d,a1i["margin"]); top1,_=metrics(prob1[:,ids],small,a1i["margin"])
                    rows.append({"intervention":"a1_reference","source_checkpoint":"A1","split":args.split,
                        "scene_id":int(d["scene_id"]),"anno_id":int(d["anno_id"]),
                        "action_sha":action_sha,"case":case,"margin":a1i["margin"],**met1,**err,
                        "probability_drift":float(np.abs(prob1-a1_prob0)[valid].mean()),
                        "representation_cosine_distance":0.0,"representation_relative_l2":0.0,
                        "selection_turnover":float((sel1!=a1_sel0).mean()),
                        "full_case_probability_distance":0.0,
                        "full_case_representation_cosine_distance":0.0,
                        "full_case_representation_relative_l2":0.0,
                        "top_half_utility_gain":top1["utility_gain"],
                        "top_half_success08_gain":top1["success08_gain"]})

            if action_sha!=array_sha(d["actions"],d["valid"],d["friction"],d["query_ids"]):
                raise RuntimeError("Actions/labels were mutated")
            del d,t,rep_full0,comps0,outs0,prob0,full_logits,full_rep,full_prob,dep
            if (i+1)%max(args.progress_every,1)==0:
                print(f"[REP-A-P1] {i+1}/{len(paths)} frames x {len(cases)} cases",flush=True)

        summary=summarize(rows)
        protocol={"split":args.split,"cases":cases,"seed":args.seed,
            "a3_checkpoint":str(a3_path.resolve()),"a3_checkpoint_sha256":a3i["checkpoint_sha256"],
            "a3_margin_shared_across_interventions":a3i["margin"],
            "a1_reference_checkpoint":str(a1_path.resolve()) if a1 is not None else None,
            "a1_reference_margin":a1i["margin"] if a1 is not None else None,
            "interventions":list(INTERVENTIONS),
            "note":"rgb_only keeps trained 0.1 RGB scale; no intervention is retrained or recalibrated"}
        save_json(out/"summary.json",{"protocol":protocol,"results":summary})
        with open(out/"per_frame.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r.keys()})); w.writeheader(); w.writerows(rows)
    print(f"[REP-A-P1] wrote {out}/summary.json and per_frame.csv",flush=True)


if __name__=="__main__":
    main()
