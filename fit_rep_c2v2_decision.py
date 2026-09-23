#!/usr/bin/env python3
"""Fit Rep-C2-v2 decision diagnostics on Seen only.

This does NOT retrain the verifier architecture and does NOT call CAD/DexNet.
It reuses:
  * trained C2-v2 checkpoints,
  * formal full-path inference outputs,
  * fresh exact labels already produced for Rep-C1/C2-v2.

Seen scenes are deterministically divided into:
  calibration-train: fit lightweight query/context ridge delta calibrators.
  calibration-val:   choose one global decision threshold for every rule.

Similar/Novel remain untouched until eval_rep_c2v2_decision.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from rep_a_common import atomic_file, digest, read_frame, save_json
from rep_c2v2_common import (
    compact_from_source, frame_cache_path, full_query_gate_metrics,
    source_eval_path, source_files,
)
from rep_c2v2_decision import (
    CONTEXT_FEATURE_NAMES, QUERY_FEATURE_NAMES, RULES,
    context_features, fit_ridge, predict_verifier_outputs,
    query_features, raw_rule_scores, save_calibration,
    select_case_thresholds, select_global_threshold, split_seen_scenes,
)
from test_rep_c2v2 import load_models


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",required=True)
    p.add_argument("--cache-root",required=True)
    p.add_argument("--c2v2-dir",required=True)
    p.add_argument("--output-root",required=True)
    p.add_argument("--cases",default="nominal,bias:-20,bias:20")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--query-chunk",type=int,default=128)
    p.add_argument("--seen-train-fraction",type=float,default=.5)
    p.add_argument("--split-seed",type=int,default=2029)
    p.add_argument("--ridge",type=float,default=1e-2)
    p.add_argument("--ridge-gain-weight",type=float,default=4.)
    p.add_argument("--threshold-grid",type=int,default=201)
    p.add_argument("--max-files",type=int,default=0)
    return p


def _scene(path):
    return int(path.parent.name.split("_")[-1])


def _records_for_rule(frames,rule):
    return [
        {
            "case":r["case"],
            "score":r["scores"][rule],
            "delta":r["delta"],
            "total_queries":r["total_queries"],
        }
        for r in frames
    ]


def _aggregate(frames,rule,threshold,case=None):
    rr=[r for r in frames if case is None or r["case"]==case]
    if not rr:
        return None
    score=np.concatenate([r["scores"][rule] for r in rr])
    delta=np.concatenate([r["delta"] for r in rr])
    total_q=sum(r["total_queries"] for r in rr)
    return full_query_gate_metrics(score,delta,threshold,total_q)


def _write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf=io.StringIO(newline="")
    w=csv.DictWriter(buf,fieldnames=keys,restval="")
    w.writeheader(); w.writerows(rows)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with atomic_file(path) as f: f.write(buf.getvalue().encode())


def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    if not 0<args.seen_train_fraction<1:
        raise ValueError("seen-train-fraction must be in (0,1)")
    if args.query_chunk<1 or args.threshold_grid<3 or args.ridge<=0:
        raise ValueError("Invalid chunk/grid/ridge")
    device=torch.device(args.device)
    models,meta,contract=load_models(args.c2v2_dir,device)
    cases=tuple(x.strip() for x in args.cases.split(",") if x.strip())
    source_protocol=json.loads((Path(args.source_root)/"protocol.json").read_text())
    if int(source_protocol.get("query_limit",-1))!=0:
        raise RuntimeError("Decision diagnostic requires formal source query_limit=0")

    files=source_files(args.source_root,"test_seen","joint",cases)
    if args.max_files>0: files=files[:args.max_files]
    train_scenes,val_scenes=split_seen_scenes(
        [_scene(p) for p in files],args.seen_train_fraction,args.split_seed
    )
    train_set=set(map(int,train_scenes)); val_set=set(map(int,val_scenes))
    frames={v:{"train":[],"val":[]} for v in models}

    for i,path in enumerate(files):
        sid=_scene(path)
        subset="train" if sid in train_set else "val"
        if sid not in train_set and sid not in val_set:
            raise RuntimeError("Seen calibration scene split bug")
        with np.load(path,allow_pickle=False) as z:
            payload={k:z[k] for k in z.files}
        ev=source_eval_path(args.source_root,path,"test_seen")
        if not ev.is_file(): raise FileNotFoundError(ev)
        with np.load(ev,allow_pickle=False) as z:
            labels={k:z[k] for k in z.files}
        ex=compact_from_source(payload,labels)
        aid=int(np.asarray(payload["anno_id"]).reshape(()))
        case=str(payload["case"])
        q_total=int(payload["actions"].shape[1])
        cf=read_frame(frame_cache_path(args.cache_root,"test_seen",sid,aid),contract)

        for v,m in models.items():
            cp,pd=predict_verifier_outputs(m,ex,cf,device,args.query_chunk)
            raw=raw_rule_scores(cp,pd)
            qf=query_features(ex,cp,pd)
            xf=context_features(ex,cp,pd,q_total)
            frames[v][subset].append({
                "scene_id":sid,"anno_id":aid,"case":case,
                "total_queries":q_total,
                "delta":np.asarray(ex["delta_utility"],np.float32),
                "query_features":qf,
                "context_features":xf,
                "scores":raw,
            })
        if (i+1)%50==0:
            print(f"[C2-v2 DECISION FIT] Seen files {i+1}/{len(files)}",flush=True)

    for subset_name,scene_set in (("train",train_set),("val",val_set)):
        present={str(r["case"]) for r in frames[next(iter(models))][subset_name]}
        missing=set(cases)-present
        if missing:
            raise RuntimeError(
                f"Seen calibration {subset_name} lacks cases {sorted(missing)}. "
                "Increase --max-files or use the complete formal source."
            )

    out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)
    calibration={
        "version":1,
        "experiment":"Rep-C2-v2 decision diagnostic",
        "source_root":str(Path(args.source_root).resolve()),
        "source_protocol_digest":digest(source_protocol),
        "c2v2_dir":str(Path(args.c2v2_dir).resolve()),
        "checkpoint_signatures":{v:meta[v]["signature"] for v in models},
        "cases":list(cases),
        "seen_train_scenes":train_scenes.tolist(),
        "seen_val_scenes":val_scenes.tolist(),
        "seen_train_fraction":args.seen_train_fraction,
        "split_seed":args.split_seed,
        "ridge":args.ridge,
        "ridge_gain_weight":args.ridge_gain_weight,
        "threshold_grid":args.threshold_grid,
        "rules":list(RULES),
        "variants":{},
        "note":"No case/error identity is used as a feature. Case-specific thresholds are privileged diagnostics only.",
    }
    csv_rows=[]

    for v in models:
        tr=frames[v]["train"]; va=frames[v]["val"]
        if not tr or not va: raise RuntimeError(f"Empty Seen calibration subset for {v}")

        y=np.concatenate([r["delta"] for r in tr])
        xq=np.concatenate([r["query_features"] for r in tr])
        xc=np.concatenate([r["context_features"] for r in tr])
        qridge=fit_ridge(
            xq,y,QUERY_FEATURE_NAMES,args.ridge,args.ridge_gain_weight
        )
        cringe=fit_ridge(
            xc,y,CONTEXT_FEATURE_NAMES,args.ridge,args.ridge_gain_weight
        )

        for subset_frames in (tr,va):
            for r in subset_frames:
                r["scores"]["query_ridge"]=qridge.predict(r["query_features"])
                r["scores"]["context_ridge"]=cringe.predict(r["context_features"])

        calibration["variants"][v]={
            "query_ridge":qridge.to_json(),
            "context_ridge":cringe.to_json(),
            "rules":{},
        }

        for rule in RULES:
            val_records=_records_for_rule(va,rule)
            best,sweep=select_global_threshold(val_records,args.threshold_grid)
            case_best,_=select_case_thresholds(val_records,args.threshold_grid)
            calibration["variants"][v]["rules"][rule]={
                "global":best,
                "case_specific_privileged":case_best,
            }
            save_json(out/f"threshold_sweep_{v}_{rule}.json",sweep)

            for subset_name,subset_frames in (("seen_cal_train",tr),("seen_cal_val",va)):
                for policy in ("global","case_specific_privileged"):
                    for case in cases:
                        if policy=="global":
                            thr=float(best["threshold"])
                        else:
                            thr=float(case_best[case]["threshold"])
                        m=_aggregate(subset_frames,rule,thr,case)
                        if m is None: continue
                        csv_rows.append({
                            "variant":v,"rule":rule,"policy":policy,
                            "subset":subset_name,"case":case,
                            "threshold":thr,**vars(m),
                        })

    save_calibration(out/"calibration.json",calibration)
    _write_csv(out/"seen_calibration.csv",csv_rows)
    print(f"[C2-v2 DECISION FIT] wrote {out/'calibration.json'}")
    print(f"[C2-v2 DECISION FIT] wrote {out/'seen_calibration.csv'}")


if __name__=="__main__":
    main()
