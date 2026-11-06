#!/usr/bin/env python3
"""Paired scene-bootstrap comparison of two fixed-budget candidate pools.

Accepts two exact-evaluated CSVs (e.g. score-stratified64 vs uniform64).
The candidate pools may differ; only frame/scene pairing is claimed.
No conclusion about causal sampling effects is licensed when the source
proposal populations differ (per-query winners versus full angle-depth lattice).
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .analyze_rayrope_candidates import load_candidates, sample_metrics
except ImportError:
    from analyze_rayrope_candidates import load_candidates, sample_metrics


def frame_metrics(path, condition=None, width_unit="auto", high_quantile=0.95):
    rows, source = load_candidates(path, None, width_unit)
    groups = defaultdict(list)
    for row in rows:
        if condition is None or row["condition"] == condition:
            groups[row["sample"]].append(row)
    if not groups:
        raise ValueError(f"No candidate records in {path} for condition={condition}")
    metrics = {sample: sample_metrics(group, high_quantile) for sample,group in groups.items()}
    return metrics,source


def paired_scene_bootstrap(left, right, metric, *, n_boot=5000, seed=20261106):
    joined = defaultdict(list)
    for sample in sorted(set(left)&set(right)):
        a,b=left[sample],right[sample]
        if a["scene"] is None or b["scene"] is None:
            raise ValueError(f"Missing scene for {sample}")
        if a["scene"]!=b["scene"] or a["split"]!=b["split"]:
            raise ValueError(f"Scene/split mismatch for paired frame {sample}")
        x,y=a.get(metric),b.get(metric)
        if x is None or y is None or not(math.isfinite(x) and math.isfinite(y)):
            continue
        joined[(a["split"],a["scene"])].append(float(y-x))
    by_split=defaultdict(list)
    for (split,scene),values in sorted(joined.items()):
        by_split[split].append(float(np.mean(values)))
    if not by_split:
        return {"metric":metric,"paired_frames":0,"paired_scenes":0,"delta_right_minus_left":None,"ci95":None}
    value=float(np.mean([np.mean(x) for x in by_split.values()]))
    rng=np.random.default_rng(seed)
    boot=np.empty(n_boot,dtype=float)
    for i in range(n_boot):
        boot[i]=float(np.mean([np.mean(rng.choice(v,size=len(v),replace=True)) for v in by_split.values()]))
    return {
        "metric":metric, "paired_frames":int(sum(len(v) for v in joined.values())),
        "paired_scenes":len(joined), "num_splits":len(by_split),
        "delta_right_minus_left":value,
        "ci95":list(map(float,np.quantile(boot,[.025,.975]))) if n_boot else None,
        "bootstrap_unit":"scene_cluster_stratified_by_split",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left_csv",required=True)
    p.add_argument("--right_csv",required=True)
    p.add_argument("--left_condition",default=None)
    p.add_argument("--right_condition",default=None)
    p.add_argument("--output_json",required=True)
    p.add_argument("--n_boot",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261106)
    p.add_argument("--high_score_quantile",type=float,default=.95)
    args=p.parse_args()
    if args.n_boot<0:
        raise ValueError("--n_boot must be >=0")
    left,meta_left=frame_metrics(args.left_csv,args.left_condition,high_quantile=args.high_score_quantile)
    right,meta_right=frame_metrics(args.right_csv,args.right_condition,high_quantile=args.high_score_quantile)
    metrics=("spearman","kendall_tau_b","top1_quality","top5_quality","oracle_quality","top1_regret","high_score_failure_rate")
    result={
        "left":{"csv":args.left_csv,"condition":args.left_condition},
        "right":{"csv":args.right_csv,"condition":args.right_condition},
        "metrics":{k:paired_scene_bootstrap(left,right,k,n_boot=args.n_boot,seed=args.seed) for k in metrics},
        "high_score_rule":"Top ceil(n*(1-q)) candidates per frame. For n=64 and q=.95, 4/64=6.25%, not exactly 5%.",
        "limitations":[
            "Different candidate selection populations can cause all reported differences. No sampler-only causal attribution.",
            "Scene bootstrap does not measure model initialization or training-seed variance.",
            "Oracle and regret are defined only within the respective 64-candidate sets.",
            "Top-1/Top-5 success quality is not official GraspNet AP.",
        ],
    }
    out=Path(args.output_json)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v["delta_right_minus_left"] for k,v in result["metrics"].items()},indent=2))


if __name__=="__main__":
    main()
