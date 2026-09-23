#!/usr/bin/env python3
"""Evaluate Rep-C2-v2 decision rules with frozen Seen calibration.

No verifier weights are updated and no CAD/DexNet evaluation is run. The script
uses the formal full-path source and its existing exact labels.

For Similar/Novel, all thresholds and ridge calibrators are frozen from Seen.
A case-specific-threshold policy is also reported as a PRIVILEGED diagnostic to
measure how much of the failure is regime calibration versus representation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from rep_a_common import atomic_file, read_frame, save_json
from rep_c2v2_common import (
    compact_from_source, frame_cache_path, full_query_gate_metrics,
    source_eval_path, source_files,
)
from rep_c2v2_decision import (
    CONTEXT_FEATURE_NAMES, QUERY_FEATURE_NAMES, RULES, RidgeCalibrator,
    context_features, load_calibration, predict_verifier_outputs,
    query_features, raw_rule_scores,
)
from test_rep_c2v2 import load_models


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",required=True)
    p.add_argument("--cache-root",required=True)
    p.add_argument("--c2v2-dir",required=True)
    p.add_argument("--calibration",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--split",choices=("test_seen","test_similar","test_novel"),required=True)
    p.add_argument("--cases",default="nominal,bias:-20,bias:20")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--query-chunk",type=int,default=128)
    p.add_argument("--seen-subset",choices=("val","train","all"),default="val")
    p.add_argument("--max-files",type=int,default=0)
    return p


def _scene(path):
    return int(path.parent.name.split("_")[-1])


def _auc(score,delta):
    s=np.asarray(score,np.float64)
    d=np.asarray(delta,np.float64)
    keep=np.abs(d)>1e-7
    if not keep.any(): return None
    s=s[keep]; y=d[keep]>0
    n1=int(y.sum()); n0=int((~y).sum())
    if n1==0 or n0==0: return None
    order=np.argsort(s,kind="stable")
    ranks=np.empty(len(s),np.float64)
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]: j+=1
        ranks[order[i:j]]=(i+j+1)/2.
        i=j
    u=ranks[y].sum()-n1*(n1+1)/2.
    return float(u/(n1*n0))


def _pearson(score,delta):
    s=np.asarray(score,np.float64); d=np.asarray(delta,np.float64)
    if len(s)<2 or s.std()<1e-12 or d.std()<1e-12: return None
    return float(np.corrcoef(s,d)[0,1])


def _write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf=io.StringIO(newline="")
    w=csv.DictWriter(buf,fieldnames=keys,restval="")
    w.writeheader(); w.writerows(rows)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with atomic_file(path) as f: f.write(buf.getvalue().encode())


def _aggregate(records,threshold):
    score=np.concatenate([r["score"] for r in records]) if records else np.empty(0)
    delta=np.concatenate([r["delta"] for r in records]) if records else np.empty(0)
    total_q=sum(r["total_queries"] for r in records)
    if total_q<=0: raise ValueError("No queries")
    m=full_query_gate_metrics(score,delta,threshold,total_q)
    out=vars(m)
    out["benefit_harm_auc"]=_auc(score,delta)
    out["pearson_with_delta"]=_pearson(score,delta)
    return out


def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    if args.query_chunk<1: raise ValueError("query-chunk must be positive")
    device=torch.device(args.device)
    calib=load_calibration(args.calibration)
    models,meta,contract=load_models(args.c2v2_dir,device)
    cases=tuple(x.strip() for x in args.cases.split(",") if x.strip())

    if set(calib["variants"])!=set(models):
        raise RuntimeError("Calibration/model variant mismatch")
    for v in models:
        if calib["checkpoint_signatures"][v]!=meta[v]["signature"]:
            raise RuntimeError(f"Calibration uses a different {v} checkpoint")

    qr={
        v:RidgeCalibrator.from_json(calib["variants"][v]["query_ridge"])
        for v in models
    }
    cr={
        v:RidgeCalibrator.from_json(calib["variants"][v]["context_ridge"])
        for v in models
    }
    if any(tuple(x.feature_names)!=QUERY_FEATURE_NAMES for x in qr.values()):
        raise RuntimeError("Query-ridge feature contract changed")
    if any(tuple(x.feature_names)!=CONTEXT_FEATURE_NAMES for x in cr.values()):
        raise RuntimeError("Context-ridge feature contract changed")

    files=source_files(args.source_root,args.split,"joint",cases)
    if args.split=="test_seen" and args.seen_subset!="all":
        scenes=set(calib["seen_val_scenes"] if args.seen_subset=="val" else calib["seen_train_scenes"])
        files=[p for p in files if _scene(p) in scenes]
    if args.max_files>0: files=files[:args.max_files]
    if not files: raise RuntimeError("No diagnostic files after filtering")

    # records[variant][rule] = list(frame records)
    records={v:{r:[] for r in RULES} for v in models}
    per_frame=[]
    for i,path in enumerate(files):
        sid=_scene(path)
        with np.load(path,allow_pickle=False) as z:
            payload={k:z[k] for k in z.files}
        ev=source_eval_path(args.source_root,path,args.split)
        if not ev.is_file(): raise FileNotFoundError(ev)
        with np.load(ev,allow_pickle=False) as z:
            labels={k:z[k] for k in z.files}
        ex=compact_from_source(payload,labels)
        aid=int(np.asarray(payload["anno_id"]).reshape(()))
        case=str(payload["case"])
        q_total=int(payload["actions"].shape[1])
        cf=read_frame(frame_cache_path(args.cache_root,args.split,sid,aid),contract)

        for v,m in models.items():
            cp,pd=predict_verifier_outputs(m,ex,cf,device,args.query_chunk)
            scores=raw_rule_scores(cp,pd)
            qf=query_features(ex,cp,pd)
            xf=context_features(ex,cp,pd,q_total)
            scores["query_ridge"]=qr[v].predict(qf)
            scores["context_ridge"]=cr[v].predict(xf)

            for rule in RULES:
                for policy in ("global","case_specific_privileged"):
                    spec=calib["variants"][v]["rules"][rule]
                    thr=(
                        float(spec["global"]["threshold"])
                        if policy=="global"
                        else float(spec["case_specific_privileged"][case]["threshold"])
                    )
                    frame_rec={
                        "case":case,"score":scores[rule],
                        "delta":np.asarray(ex["delta_utility"],np.float32),
                        "total_queries":q_total,
                    }
                    # Store each frame once per rule; aggregate policy applies a
                    # different fixed threshold to the same score distribution.
                    if policy=="global":
                        records[v][rule].append(frame_rec)
                    fm=full_query_gate_metrics(
                        frame_rec["score"],frame_rec["delta"],thr,q_total
                    )
                    per_frame.append({
                        "variant":v,"rule":rule,"policy":policy,
                        "split":args.split,
                        "seen_subset":args.seen_subset if args.split=="test_seen" else "held_out",
                        "scene_id":sid,"anno_id":aid,"case":case,
                        "threshold":thr,**vars(fm),
                    })
        if (i+1)%50==0:
            print(f"[C2-v2 DECISION EVAL] {args.split} {i+1}/{len(files)}",flush=True)

    summary=[]
    for v in models:
        for rule in RULES:
            for policy in ("global","case_specific_privileged"):
                spec=calib["variants"][v]["rules"][rule]
                for case in cases:
                    rr=[r for r in records[v][rule] if r["case"]==case]
                    if not rr: continue
                    thr=(
                        float(spec["global"]["threshold"])
                        if policy=="global"
                        else float(spec["case_specific_privileged"][case]["threshold"])
                    )
                    summary.append({
                        "variant":v,"rule":rule,"policy":policy,
                        "split":args.split,
                        "seen_subset":args.seen_subset if args.split=="test_seen" else "held_out",
                        "case":case,"threshold":thr,
                        **_aggregate(rr,thr),
                    })

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/"per_frame.csv",per_frame)
    _write_csv(out/"comparison.csv",summary)
    save_json(out/"summary.json",{
        "split":args.split,
        "seen_subset":args.seen_subset if args.split=="test_seen" else "held_out",
        "num_source_files":len(files),
        "global_policy":"deployable-style: one Seen-calibrated threshold per variant/rule",
        "case_specific_policy":"PRIVILEGED synthetic-case diagnostic only; not deployable",
        "results":summary,
    })
    print(f"[C2-v2 DECISION EVAL] wrote {out/'comparison.csv'}")


if __name__=="__main__":
    main()
