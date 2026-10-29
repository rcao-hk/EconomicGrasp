#!/usr/bin/env python3
"""P1-0: scene-paired comparison of native GC6D (235,13,50,6) accuracy arrays.

Runs may represent two-source Point, three-source Point/GVAR, or collision
on/off of the same checkpoint. No fabricated AP if a baseline is unavailable.
The paired CI resamples SCENES (not 3055 correlated views as IID).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mixed_p1_historical import sha256

MUS = (0.2,0.4,0.6,0.8,1.,1.2)
RANKS = (1,5,10,50)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",action="append",required=True,
                   help="NAME=/path/to/official_accuracy.npy; repeat for each run")
    p.add_argument("--compare",action="append",required=True,
                   help="A:B (AP of B minus A); e.g. two_point_off:three_point_off")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--bootstrap",type=int,default=10000)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--gc6d-root",default=None,
                   help="Optional official GC6D root for test-scene identity audit.")
    return p.parse_args(argv)


def load_accuracy(file):
    file=Path(file)
    if not file.is_file():
        raise FileNotFoundError(f"GC6D AP array missing: {file}")
    arr=np.load(file,allow_pickle=False)
    if arr.shape!=(235,13,50,6) or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:
        raise ValueError(f"Expected finite [235,13,50,6] in [0,1]: {file}, got {arr.shape}")
    return arr.astype(np.float64,copy=False)


def per_scene_ap(a,k=50,mu_idx=None):
    ranks=a[:,:,:k,:] if mu_idx is None else a[:,:,:k,mu_idx:mu_idx+1]
    return ranks.mean(axis=(1,2,3))*100


def percentile(draw):
    return [float(v) for v in np.percentile(draw,[2.5,97.5])]


def compare_arrays(a,b,draw_indices):
    out=[]
    for k in RANKS:
        for midx,mu in [(None,"mean_six_mu")]+[(i,str(v)) for i,v in enumerate(MUS)]:
            aa=per_scene_ap(a,k,midx)
            bb=per_scene_ap(b,k,midx)
            diff=bb-aa
            bootstrap=diff[draw_indices].mean(axis=1)
            lo,hi=percentile(bootstrap)
            out.append(dict(rank_k=k,mu=mu,
                ap_a_percent=float(aa.mean()),ap_b_percent=float(bb.mean()),
                delta_pp=float(diff.mean()),ci95_low_pp=lo,ci95_high_pp=hi,
                scene_delta_sd_pp=float(np.std(diff,ddof=1)),
                bootstrap_fraction_positive=float(np.mean(bootstrap>0))))
    return out


def save_csv(file,rows):
    if not rows:
        return
    with Path(file).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    a=parse_args(argv)
    if a.bootstrap<100:
        raise ValueError("--bootstrap must be ≥100")
    ids=None
    if a.gc6d_root:
        path=Path(a.gc6d_root)/"split_info"/"grasp_test_scene_ids.json"
        ids=[int(i) for i in json.loads(path.read_text(encoding="utf-8"))]
        if len(ids)!=235 or len(set(ids))!=235:
            raise ValueError("Official GC6D test requires 235 unique scene IDs in evaluator's file order.")
    run_files={}
    arrays={}
    for arg in a.run:
        if "=" not in arg:
            raise ValueError("--run must be NAME=FILE")
        name,path=arg.split("=",1)
        if not name or name in arrays:
            raise ValueError("Empty/repeated run name")
        arrays[name]=load_accuracy(path)
        run_files[name]=dict(file=str(Path(path).resolve()),sha256=sha256(path))
    pairs=[]
    for arg in a.compare:
        if ":" not in arg:
            raise ValueError(f"--compare expects A:B: {arg}")
        left,right=arg.split(":",1)
        if left not in arrays or right not in arrays or left==right:
            raise ValueError(f"Invalid pair {arg}; provided runs={list(arrays)}")
        pairs.append((left,right))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed)
    selected=rng.integers(0,235,size=(a.bootstrap,235))
    rows=[];scenes=[]
    for left,right in pairs:
        for row in compare_arrays(arrays[left],arrays[right],selected):
            rows.append(dict(run_a=left,run_b=right,**row))
        for sceneidx,(va,vb) in enumerate(zip(
            per_scene_ap(arrays[left]),per_scene_ap(arrays[right]))):
            scenes.append(dict(run_a=left,run_b=right,
                scene_idx=int(sceneidx),scene_id=int(ids[sceneidx]) if ids else "",
                ap_a_percent=float(va),ap_b_percent=float(vb),
                delta_pp=float(vb-va)))
    save_csv(out/"paired_gc6d_ap.csv",rows)
    save_csv(out/"paired_gc6d_scenes.csv",scenes)
    summary=dict(status="complete",input_arrays=run_files,pairs=pairs,
        scene_ids=ids,scene_count=235,views_per_scene=13,
        ranks=RANKS,friction_coefficients=MUS,bootstrap_replicates=a.bootstrap,
        bootstrap_seed=a.seed,bootstrap_unit="scene",
        metric="official native GC6D average precision prefix, in percent",
        confidence_interpretation="scene resampling uncertainty; excludes training seed variation",
        metrics=[r for r in rows if r["rank_k"]==50 and r["mu"]=="mean_six_mu"])
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(output=str(out),comparisons=summary["metrics"]),indent=2))


if __name__=="__main__":
    main()
