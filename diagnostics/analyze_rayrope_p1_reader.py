#!/usr/bin/env python3
"""Summarize P1 reader depth dose response with paired fixed physical actions.

Input:
  p1_fixed_actions_labeled.csv, p1_interventions.json
Outputs:
  per-condition score changes, rank stability, Top-1 quality change,
  mask flip statistics, and scene-clustered paired intervals.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .analyze_rayrope_candidates import load_candidates, rank_corr
except ImportError:
    from analyze_rayrope_candidates import load_candidates, rank_corr


def _mean(items):
    values=np.asarray(list(items),dtype=float)
    values=values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def _paired_scene_ci(per_frame, metric, n_boot=5000, seed=20261106):
    by_split=defaultdict(lambda:defaultdict(list))
    for r in per_frame:
        val=r.get(metric)
        if r["scene"] is not None and val is not None and math.isfinite(float(val)):
            by_split[r["split"]][r["scene"]].append(float(val))
    if not by_split:
        return {"num_scenes":0,"mean":None,"ci95":None}
    scene_means={split:np.asarray([np.mean(v) for v in scene_map.values()],float)
                 for split,scene_map in by_split.items()}
    estimate=float(np.mean([a.mean() for a in scene_means.values()]))
    rng=np.random.default_rng(seed)
    dist=[]
    for _ in range(n_boot):
        dist.append(float(np.mean([
            np.mean(rng.choice(a,len(a),replace=True)) for a in scene_means.values()
        ])))
    return {
        "num_scenes":sum(len(a) for a in scene_means.values()),
        "num_frames":len(per_frame),
        "mean":estimate,
        "ci95":list(map(float,np.quantile(dist,[.025,.975]))) if dist else None,
        "unit":"scene_cluster_stratified_by_split",
    }


def analyze(csv_path,interventions_path,*,n_boot=5000,seed=20261106):
    rows,schema=load_candidates(csv_path,None,"m")
    provenance=json.loads(Path(interventions_path).read_text(encoding="utf-8"))
    by_sample=defaultdict(lambda:defaultdict(dict))
    for x in rows:
        sample,cond,cid=x["sample"],x["condition"],x["candidate"]
        by_sample[sample][cond][cid]=x
    if not by_sample:
        raise ValueError("Empty P1 candidate set")
    stats={(x["sample"],x["condition"]):x for x in provenance.get("mask_statistics",[])}
    per_condition=defaultdict(list)
    for sample,groups in sorted(by_sample.items()):
        native=groups.get("native",{})
        if not native:
            raise ValueError(f"No native condition for frame {sample}")
        for cond,target in sorted(groups.items()):
            if cond=="native":
                continue
            if set(native)!=set(target):
                raise ValueError(f"P1 {sample}/{cond}: non-identical physical candidate IDs")
            # Frozen score context: score can change; physical grasp
            # center, rotation, depth and width must NOT.
            for key in native:
                a,b=native[key],target[key]
                if abs(a["width_m"]-b["width_m"])>1e-7 or abs(a["insertion"]-b["insertion"])>1e-7:
                    raise ValueError(f"Reader-only P1 changed action width/depth: {sample}/{cond}/{key}")
                if not np.allclose(a["center"],b["center"],rtol=0,atol=1e-7):
                    raise ValueError(f"Reader-only P1 changed physical grasp center: {sample}/{cond}/{key}")
                if math.isfinite(a["quality"]) and math.isfinite(b["quality"]) and abs(a["quality"]-b["quality"])>1e-7:
                    raise ValueError(f"Reader-only P1 relabeled a frozen physical action: {sample}/{cond}/{key}")
            ids=sorted(native)
            sa=np.asarray([native[x]["score"] for x in ids])
            sb=np.asarray([target[x]["score"] for x in ids])
            # Stable deterministic tie policy on candidate IDs.
            base_top=max(ids,key=lambda i:(native[i]["score"],i))
            changed_top=max(ids,key=lambda i:(target[i]["score"],i))
            change=float(target[changed_top]["quality"]-native[base_top]["quality"])
            st=stats.get((sample,cond),{})
            row={
                "sample":sample,"split":native[ids[0]]["split"],
                "scene":native[ids[0]]["scene"],"condition":cond,
                "num_actions":len(ids),
                "mean_abs_score_change":float(np.mean(np.abs(sb-sa))),
                "rank_stability":rank_corr(sa,sb),
                "top1_changed":float(base_top!=changed_top),
                "top1_quality_change":change,
                "mask_flip_count":int(st.get("mask_flip_count",0)) if st else None,
                "mask_flip_fraction":(float(st.get("mask_flip_count",0))/float(st["token_count"])
                                      if st and st.get("token_count",0)>0 else None),
                "query_mask_flip_fraction":(float(st.get("queries_with_mask_flip",0))/float(st["query_count"])
                                            if st and st.get("query_count",0)>0 else None),
            }
            per_condition[cond].append(row)
    contrasts={}
    for condition,values in sorted(per_condition.items()):
        metrics=("mean_abs_score_change","rank_stability","top1_changed",
                 "top1_quality_change","mask_flip_fraction","query_mask_flip_fraction")
        contrasts[condition]={
            "frames":len(values),
            "metrics":{
                k:_paired_scene_ci(values,k,n_boot=n_boot,seed=seed)
                for k in metrics
            },
            "per_frame":values,
        }
    return {
        "candidate_csv":csv_path,
        "interventions_json":interventions_path,
        "schema":schema,
        "native_checkpoint_sha256":provenance.get("checkpoint_sha256"),
        "num_frames":len(by_sample),
        "conditions":contrasts,
        "limits":[
            "Reader-only interventions modify RoPE coordinates and/or validity; they do not modify the evaluated physical grasp.",
            "Score changes are measured at the frozen query-angle-depth index; the modified model may predict another width but executed width remains native.",
            "A zero mask-only score change does not establish mask irrelevance if no token validity flips occurred.",
            "A positive depth bias should not be selected using the same test frames used to report the gain.",
            "Scene bootstrap cannot measure training-seed variance.",
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate_csv",required=True)
    p.add_argument("--interventions_json",required=True)
    p.add_argument("--output_json",required=True)
    p.add_argument("--n_boot",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261106)
    args=p.parse_args()
    out=analyze(args.candidate_csv,args.interventions_json,n_boot=args.n_boot,seed=args.seed)
    target=Path(args.output_json)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:{
        "score_delta":v["metrics"]["mean_abs_score_change"]["mean"],
        "top1_delta":v["metrics"]["top1_quality_change"]["mean"],
        "mask_flip_fraction":v["metrics"]["mask_flip_fraction"]["mean"],
    } for k,v in out["conditions"].items()},indent=2))


if __name__=="__main__":
    main()
