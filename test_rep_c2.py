#!/usr/bin/env python3
"""Evaluate Rep-C2 accept/reject verifiers without threshold retuning."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch

from rep_a_common import (
    check_runtime_sources, choose, list_frames, read_frame, perturb_depth,
    seed_for, tensors, atomic_file, save_json
)
from rep_followup_common import load_scorer
from rep_c2_model import EVIDENCE, RepC2Verifier, apply_evidence, build_pair_evidence


DEFAULT_CASES=(
    "nominal,bias:-10,bias:10,bias:-20,bias:20,bias:-25,bias:25,"
    "scale:-0.05,scale:0.05,smooth:20,edge:2"
)


def auc_binary(score,label):
    score=np.asarray(score,float); label=np.asarray(label,bool)
    pos=score[label]; neg=score[~label]
    if not len(pos) or not len(neg): return None
    # Mann-Whitney formulation, ties receive half credit.
    vals=np.concatenate([pos,neg]); labs=np.concatenate([np.ones(len(pos),bool),np.zeros(len(neg),bool)])
    order=np.argsort(vals,kind="stable"); ranks=np.empty(len(vals),float)
    i=0
    while i<len(vals):
        j=i+1
        while j<len(vals) and vals[order[j]]==vals[order[i]]: j+=1
        ranks[order[i:j]]=(i+j+1)/2.
        i=j
    u=ranks[labs].sum()-len(pos)*(len(pos)+1)/2
    return float(u/(len(pos)*len(neg)))


def frame_metrics(d,proposal,gate,threshold,case,evidence):
    z=int(d["zero_index"]); q=np.arange(len(proposal)); u=d["utility"]
    move=proposal!=z
    true_gain=u[proposal,q]-u[z]
    accept=move&(gate>=threshold)
    selected=np.where(accept,proposal,z)
    gain=u[selected,q]-u[z]
    benefit=move&(true_gain>1e-7); harm=move&(true_gain<-1e-7)
    return {
        "evidence":evidence,"case":case,"num_queries":len(q),
        "proposal_rate":float(move.mean()),"accept_rate":float(accept.mean()),
        "native_utility":float(u[z].mean()),"selected_utility":float(u[selected,q].mean()),
        "utility_gain":float(gain.mean()),"a1_fixed0_gain":float(true_gain.mean()),
        "beneficial_retention":float((accept&benefit).sum()/max(1,benefit.sum())),
        "harmful_rejection":float(1-(accept&harm).sum()/max(1,harm.sum())),
        "accept_precision":float((accept&benefit).sum()/max(1,accept.sum())),
        "beneficial_proposals":int(benefit.sum()),"harmful_proposals":int(harm.sum()),
        "gate_auc":auc_binary(gate[move],benefit[move]) if move.any() else None,
    }


def aggregate(rows):
    nq=sum(r["num_queries"] for r in rows)
    out={"num_frames":len(rows),"num_queries":nq}
    for k in ("proposal_rate","accept_rate","native_utility","selected_utility","utility_gain","a1_fixed0_gain"):
        out[k]=sum(r[k]*r["num_queries"] for r in rows)/nq
    for k,num,den in (
        ("beneficial_retention","beneficial_retention","beneficial_proposals"),
        ("harmful_rejection","harmful_rejection","harmful_proposals"),
    ):
        d=sum(r[den] for r in rows)
        out[k]=sum(r[num]*r[den] for r in rows)/d if d else None
    vals=[r["gate_auc"] for r in rows if r["gate_auc"] is not None]
    out["mean_frame_gate_auc"]=float(np.mean(vals)) if vals else None
    out["beneficial_proposals"]=sum(r["beneficial_proposals"] for r in rows)
    out["harmful_proposals"]=sum(r["harmful_proposals"] for r in rows)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root",required=True)
    p.add_argument("--a1-checkpoint",required=True)
    p.add_argument("--c2-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--split",choices=("test_seen","test_similar","test_novel"),required=True)
    p.add_argument("--cases",default=DEFAULT_CASES)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--max-frames",type=int,default=0)
    args=p.parse_args()

    check_runtime_sources(args.cache_root)
    device=torch.device(args.device)
    a1,meta=load_scorer(args.a1_checkpoint,args.cache_root,device)
    if meta["variant"]!="A1": raise ValueError("A1 checkpoint required")
    a1.requires_grad_(False)
    models={}; thresholds={}
    for e in EVIDENCE:
        ck=torch.load(Path(args.c2_dir)/f"checkpoint_{e}.pt",map_location="cpu",weights_only=False)
        if ck["evidence"]!=e or ck["contract"]!=meta["contract"]: raise RuntimeError(f"Bad C2 checkpoint {e}")
        m=RepC2Verifier().to(device); m.load_state_dict(ck["model"]); m.eval()
        models[e]=m; thresholds[e]=float(ck["threshold"])
    cases=list(dict.fromkeys(["nominal"]+[x.strip() for x in args.cases.split(",") if x.strip()]))
    paths=list_frames(args.cache_root,args.split,args.max_frames)
    rows=[]
    with torch.no_grad():
        for path in paths:
            d=read_frame(path,meta["contract"])
            t=tensors(d,device)
            t["valid"]=torch.as_tensor(d["valid"],device=device,dtype=torch.bool)
            t["offsets_mm"]=torch.as_tensor(d["offsets_mm"],device=device,dtype=torch.float32)
            t["zero_index"]=int(d["zero_index"])
            t["native_score"]=torch.as_tensor(d["native_score"],device=device,dtype=torch.float32)
            for case in cases:
                s=seed_for(args.seed,"c2-test",args.split,int(d["scene_id"]),int(d["anno_id"]),case)
                dep,_=perturb_depth(t["depth"],case,s)
                prob=a1(t,dep).sigmoid()
                proposal_np=choose(prob.mean(-1).cpu().numpy(),d["valid"],int(d["zero_index"]),0.)
                proposal=torch.as_tensor(proposal_np,device=device)
                feat=build_pair_evidence(t,dep,prob,proposal)
                for e in EVIDENCE:
                    gate=models[e](apply_evidence(feat,e)).sigmoid().cpu().numpy()
                    rec=frame_metrics(d,proposal_np,gate,thresholds[e],case,e)
                    rec.update(split=args.split,scene_id=int(d["scene_id"]),anno_id=int(d["anno_id"]),threshold=thresholds[e])
                    rows.append(rec)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with atomic_file(out/"per_frame.csv") as f:
        import io
        buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        f.write(buf.getvalue().encode())
    summary=[]
    for e in EVIDENCE:
        for case in cases:
            rr=[r for r in rows if r["evidence"]==e and r["case"]==case]
            summary.append({"evidence":e,"split":args.split,"case":case,"threshold":thresholds[e],**aggregate(rr)})
    save_json(out/"summary.json",{"split":args.split,"cases":cases,"results":summary})
    with atomic_file(out/"comparison.csv") as f:
        import io
        keys=list(dict.fromkeys(k for r in summary for k in r)); buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=keys,restval="")
        w.writeheader(); w.writerows(summary); f.write(buf.getvalue().encode())
    print(f"[REP-C2 TEST] {args.split}: {out/'comparison.csv'}")


if __name__=="__main__":
    main()
