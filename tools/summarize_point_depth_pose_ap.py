#!/usr/bin/env python3
"""Summarize official GraspNet AP arrays for 10% point-depth baseline.

Official evaluator output remains the source of truth. This script only computes
reproducible descriptive metrics and optional 30-scene paired CIs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

SPLITS = ("test_seen","test_similar","test_novel")
PREFIX = (1,5,10,20,50)
FRICTION = (0.2,0.4,0.6,0.8,1.0,1.2)


def read_results(root: Path):
    arrays={}
    for split in SPLITS:
        path=root/split/f"ap_{split}_realsense.npy"
        arr=np.load(path, allow_pickle=False)
        if arr.ndim!=4 or arr.shape[0]!=30 or arr.shape[1] not in (26,256) or arr.shape[2:]!=(50,6) or not np.isfinite(arr).all():
            raise ValueError(f"{path}: requires finite [30,26|256,50,6], got {arr.shape}")
        arrays[split]=arr.astype(np.float64,copy=False)
    return arrays


def metrics(arr):
    out = {
        "mean_ap_pct_mu_0p2_to_1p0": float(100*arr[...,:5].mean()),
        "ap_mu_0p4_pct":float(100*arr[...,1].mean()),
        "ap_mu_0p8_pct":float(100*arr[...,3].mean()),
    }
    for k in PREFIX:
        out[f"prefix_{k}_mean_ap_pct"]=float(100*arr[:,:,:k,:5].mean())
    for j,mu in enumerate(FRICTION):
        out[f"mu_{mu:.1f}_all_ranks_pct"]=float(100*arr[...,j].mean())
    return out


def paired_scene_bootstrap(arrays,ref_arrays,replicates=50000,seed=20261009):
    rng=np.random.default_rng(seed)
    means=[]
    per_split={}
    for split in SPLITS:
        a=arrays[split][...,:5].mean(axis=(1,2,3))
        if arrays[split].shape!=ref_arrays[split].shape:
            raise ValueError(f"Paired AP requires matching frames per scene: {split}")
        b=ref_arrays[split][...,:5].mean(axis=(1,2,3))
        diff=100*(a-b)
        indices=rng.integers(0,30,size=(replicates,30))
        bootstrap=diff[indices].mean(axis=1)
        per_split[split]={
            "delta_ap_pp":float(diff.mean()),
            "ci95_pp":np.percentile(bootstrap,[2.5,97.5]).tolist(),
            "positive_scenes":int((diff>0).sum()),
        }
        means.append(bootstrap)
    overall=(means[0]+means[1]+means[2])/3.0
    return {
        "per_split":per_split,
        "mean_delta_ap_pp":float(np.mean([per_split[x]["delta_ap_pp"] for x in SPLITS])),
        "mean_ci95_pp":np.percentile(overall,[2.5,97.5]).tolist(),
        "cluster_unit":"scene, 30 per split, frame annotations correlated",
        "bootstrap_seed":seed,
        "bootstrap_iterations":replicates,
        "notes":"Does not cover train-seed/geometry-provider differences.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--reference-root",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--bootstrap",type=int,default=50000)
    args=p.parse_args()
    if args.bootstrap<100:
        p.error("--bootstrap must be >=100")
    ours=read_results(args.root)
    report={
        "dataset":"GraspNet-1Billion Original RealSense official AP arrays",
        "frames_per_scene":int(next(iter(ours.values())).shape[1]),
        "splits":{split:metrics(arr) for split,arr in ours.items()},
        "mean_ap_pct":float(np.mean([metrics(arr)["mean_ap_pct_mu_0p2_to_1p0"] for arr in ours.values()])),
    }
    if len({arr.shape for arr in ours.values()})!=1:
        raise ValueError("Three official AP arrays use different evaluation fractions")
    if args.reference_root:
        reference=read_results(args.reference_root)
        report["reference_root"]=str(args.reference_root.resolve())
        report["paired"]=paired_scene_bootstrap(ours,reference,args.bootstrap)
    result=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.output:
        if args.output.exists():
            p.error(f"Refusing to overwrite report: {args.output}")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result)
    print(result)


if __name__=="__main__":
    main()
