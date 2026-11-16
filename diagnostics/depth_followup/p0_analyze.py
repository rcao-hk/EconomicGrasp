"""P0 compare *native* train/GC6D crop-scale and reference-distance support.

Descriptive scene-clustered statistics only; no learned depth calibration,
no per-frame rescaling, no use of the repeatedly inspected GC6D AP for tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics.frozen_depth_response.analyze import spearman,slope
from .common import sha256,write_json

FIELDS=("median_pred","median_reference","median_ratio","median_error",
        "mae","absrel","fx_crop","fy_crop","fx_raw","fy_raw",
        "crop_fraction_w","crop_fraction_h","crop_area_fraction",
        "foreground_fraction","crop_foreground_fraction",
        "focal_mean_crop","camera_pose_tilt_deg")
DEPTH_BINS=(0.,.45,.55,.65,.75,.85,1.,float("inf"))
FX_BINS=(0.,300.,400.,500.,600.,800.,float("inf"))


def load_domain(path,name):
    root=Path(path)
    mf=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if (mf.get("schema")!="frozen-depth-p0-domain-v1" or
        mf.get("domain")!=name or not mf.get("complete")):
        raise ValueError(f"Incomplete/mismatched P0 domain: {name}")
    raw=root/"per_frame.csv"
    if sha256(raw)!=mf.get("per_frame_sha256"):
        raise ValueError(f"Changed source P0 CSV for {name}")
    with raw.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    target=1014 if name=="gc6d" else 400
    scenes=78 if name=="gc6d" else 100
    if len(rows)!=target or mf.get("frames")!=target or mf.get("scenes")!=scenes:
        raise ValueError(f"Expected {target} frames/{scenes} scenes: {name}")
    seen=set()
    for r in rows:
        r["scene"]=int(r["scene"]);r["anno"]=int(r["anno"])
        key=(r["scene"],r["anno"])
        if key in seen:raise ValueError(f"Repeated scene/frame: {name}/{key}")
        seen.add(key)
        for f in FIELDS:
            r[f]=float(r[f])
            if not np.isfinite(r[f]):
                raise ValueError(f"Nonfinite P0 feature {f} in {name}/{key}")
    if name=="gc6d" and not all(
            len([r for r in rows if r["scene"]==s])==13
            for s in {x["scene"] for x in rows}):
        raise ValueError("GC6D must have 13 distinct views per scene")
    if name!="gc6d" and not all(
            len([r for r in rows if r["scene"]==s])==4
            for s in range(100)):
        raise ValueError("MIX50 train scene subset must have 4 frames per scene")
    return mf,rows


def scene_bootstrap(rows,key,*,draws=4000,seed=20261117):
    groups=defaultdict(list)
    for x in rows:groups[x["scene"]].append(x[key])
    if len(groups)<2:raise ValueError("Need independent scenes for confidence intervals")
    vals=np.array([np.mean(v) for _,v in sorted(groups.items())],dtype=float)
    rng=np.random.default_rng(seed)
    ids=rng.integers(0,len(vals),size=(draws,len(vals)))
    return dict(mean=float(vals.mean()),ci95=np.quantile(vals[ids].mean(1),[.025,.975]).tolist(),
                scene_median=float(np.median(vals)),scenes=len(vals))


def binned(rows,key,edges):
    x=np.array([r[key] for r in rows],dtype=float)
    bins=np.digitize(x,edges[1:-1],right=False)
    results=[]
    for i in range(len(edges)-1):
        part=[r for r,b in zip(rows,bins) if b==i]
        results.append(dict(bin=i,lower=float(edges[i]),upper=None if
             not np.isfinite(edges[i+1]) else float(edges[i+1]),
             frames=len(part),scenes=len({r["scene"] for r in part}),
             median_ref=float(np.median([r["median_reference"] for r in part]))
                 if part else None,
             median_pred=float(np.median([r["median_pred"] for r in part]))
                 if part else None,
             mean_ratio=float(np.mean([r["median_ratio"] for r in part]))
                 if part else None))
    return results


def _outside_train(rows,quantiles):
    features=("median_reference","fx_crop","fy_crop","crop_fraction_w",
              "crop_fraction_h","foreground_fraction")
    count={k:0 for k in features}
    joint=0
    for r in rows:
        out={k:not (quantiles[k][0]<=r[k]<=quantiles[k][1]) for k in features}
        for k,v in out.items():
            count[k]+=int(v)
        joint+=int(any(out.values()))
    return dict(n=len(rows),out_of_original_p05_p95={k:v/len(rows) for k,v in count.items()},
                outside_any_feature_fraction=joint/len(rows))


def run(a):
    if a.bootstrap<1:raise ValueError("Invalid bootstrap draws")
    inputs=dict(original=a.original,gntrans=a.gntrans,gc6d=a.gc6d)
    datasets={k:load_domain(path,k) for k,path in inputs.items()}
    base=datasets["original"][0]
    checkpoint=base["base_checkpoint_sha256"]
    for domain,(mf,rows) in datasets.items():
        if mf["base_checkpoint_sha256"]!=checkpoint or mf.get("train_updates")!=0:
            raise ValueError("Frozen base/checkpoint or train updates mismatch")
        if mf["base_config_digest"]!=base["base_config_digest"]:
            raise ValueError("P0 input geometry source base differs across domains")
    ref_values={}
    orig_rows=datasets["original"][1]
    for feature in ("median_reference","fx_crop","fy_crop",
                    "crop_fraction_w","crop_fraction_h","foreground_fraction"):
        x=np.array([r[feature] for r in orig_rows])
        ref_values[feature]=np.quantile(x,[.05,.95]).tolist()
    report={}
    details={}
    for i,(domain,(mf,rows)) in enumerate(datasets.items()):
        r=np.array([x["median_reference"] for x in rows],dtype=float)
        p=np.array([x["median_pred"] for x in rows],dtype=float)
        fx=np.array([x["fx_crop"] for x in rows],dtype=float)
        byscene=defaultdict(list)
        for x in rows:byscene[x["scene"]].append(x)
        slopes=[];rank=[]
        for scene,items in sorted(byscene.items()):
            xx=np.array([x["median_reference"] for x in items])
            yy=np.array([x["median_pred"] for x in items])
            if len(items)>=3 and np.ptp(xx)>=.03:
                q=slope(xx,yy,min_span=.03)
                if q is not None:slopes.append(q)
                rho=spearman(xx,yy)
                if rho is not None:rank.append(rho)
        report[domain]=dict(
            reference_source=mf["reference"],
            frames=len(rows),scenes=len(byscene),
            metrics={f:scene_bootstrap(rows,f,draws=a.bootstrap,seed=a.seed+i*123+j)
                     for j,f in enumerate(FIELDS)},
            spearman_ref_pred=spearman(r,p),
            spearman_reference_fx_crop=spearman(r,fx),
            spearman_predicted_fx_crop=spearman(p,fx),
            mean_within_scene_slope=float(np.mean(slopes)) if slopes else None,
            mean_within_scene_spearman=float(np.mean(rank)) if rank else None,
            scenes_with_valid_slope=len(slopes),
            by_reference_distance=binned(rows,"median_reference",DEPTH_BINS),
            by_effective_focal=binned(rows,"fx_crop",FX_BINS),
            train_support_gap=(_outside_train(rows,ref_values) if domain=="gc6d" else None))
    paired=[]
    gn={(r["scene"],r["anno"]):r for r in datasets["gntrans"][1]}
    if set(gn)!={(r["scene"],r["anno"]) for r in orig_rows}:
        raise ValueError("GraspNet Original and GNTrans input frames are not paired")
    for r in orig_rows:
        trans=gn[(r["scene"],r["anno"])]
        paired.append(dict(scene=r["scene"],anno=r["anno"],
            original_reference=r["median_reference"],
            gntrans_virtual_reference=trans["median_reference"],
            delta_crop_fx=trans["fx_crop"]-r["fx_crop"],
            delta_crop_area=trans["crop_area_fraction"]-r["crop_area_fraction"],
            delta_pred_depth=trans["median_pred"]-r["median_pred"]))
    output=dict(schema="frozen-depth-p0-comparison-v1",
                base_checkpoint_sha256=checkpoint,
                source_manifests={k:sha256(Path(p)/"manifest.json") for k,p in inputs.items()},
                original_p05_p95_thresholds=ref_values,by_domain=report,
                paired_original_gntrans=dict(
                    frames=len(paired),
                    avg_delta_pred_depth=float(np.mean([r["delta_pred_depth"] for r in paired])),
                    avg_delta_crop_fx=float(np.mean([r["delta_crop_fx"] for r in paired])),
                    virtual_reference_not_sensor=True),
                limitations=[
                    "Original/GNTrans train scenes were seen by the frozen backbone",
                    "GC6D was repeatedly inspected; descriptive not untouched",
                    "GC6D and GraspNet have different cameras, object populations and native crop masks",
                    "GNTrans uses virtual depth and material augmentation; comparison cannot be interpreted as equal sensor noise",
                    "Native crop dimensions/focal depend on observed depth and segmentation",
                    "No correction factor, camera-aware depth fit, trainable model or grasp AP was produced",
                    "Spearman is correlation and does not establish K/crop causal effect",
                ])
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"summary.json",output)
    with (out/"paired_original_gntrans.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    rows=[]
    for d,val in report.items():
        rows.append(dict(domain=d,frames=val["frames"],scenes=val["scenes"],
            depth_reference=val["metrics"]["median_reference"]["mean"],
            depth_predicted=val["metrics"]["median_pred"]["mean"],
            depth_ratio=val["metrics"]["median_ratio"]["mean"],
            fx_crop=val["metrics"]["fx_crop"]["mean"],
            crop_fraction_w=val["metrics"]["crop_fraction_w"]["mean"],
            rho_ref_fx=val["spearman_reference_fx_crop"],
            rho_pred_fx=val["spearman_predicted_fx_crop"],
            rho_ref_pred=val["spearman_ref_pred"],
            mean_within_scene_slope=val["mean_within_scene_slope"]))
    with (out/"domain_summary.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=["# P0 冻结米制深度的训练域／GC6D距离与crop支持域审计","",
           "| 域 | 参考深度(m) | 预测深度(m) | 比值 | fx' | crop宽比 | ref–fx' Spearman | pred–fx' Spearman |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['domain']} | {r['depth_reference']:.4f} | "
                     f"{r['depth_predicted']:.4f} | {r['depth_ratio']:.3f} | "
                     f"{r['fx_crop']:.1f} | {r['crop_fraction_w']:.3f} | "
                     f"{r['rho_ref_fx']} | {r['rho_pred_fx']} |")
    lines.extend(["","Original和GC6D参考均为传感器深度，GNTrans参考为virtual depth；不能将其误称为同一种传感器GT。",
                  "Original分位支持域外比例是描述性指标，不能证明训练分布单独造成跨域误差。",
                  "未拟合深度尺度、不更新模型、不进行任何新AP评估。"])
    (out/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",required=True)
    p.add_argument("--gntrans",required=True)
    p.add_argument("--gc6d",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=3000)
    p.add_argument("--seed",type=int,default=20261117)
    run(p.parse_args())


if __name__=="__main__":
    main()
