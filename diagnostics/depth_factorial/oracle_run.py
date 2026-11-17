"""A: offline shape-vs-metric-depth limit using accepted GC6D P1 cache.

The only fitted parameters are per-frame, per-fold reference-depth ORACLES.
All quality/shape metrics are on spatially nonoverlapping validation pixels.
No scaler is saved for deployment, no model updates, no grasp AP or CAD labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from diagnostics.depth_followup.common import digest,sha256,write_json
from diagnostics.depth_followup.p1_track import load_cache
from diagnostics.frozen_depth_response.geometry import crop_K_from_raw
from .oracle_core import MODES,SCHEMA,evaluate_frame,paired_scene_bootstrap


def read_frame(rec):
    """Use actual signed P1 raw sensor depth and segmentation, no virtual GT."""
    source=rec["source_paths"];signatures=rec["source_sha256"]
    for key in ("rgb","depth","seg","camera"):
        path=Path(source[key])
        if not path.is_file() or sha256(path)!=signatures[key]:
            raise RuntimeError(f"Source {key} modified/missing: {path}")
    path=Path(rec["prediction_file"])
    if not path.is_file() or sha256(path)!=rec["prediction_sha256"]:
        raise RuntimeError("Frozen prediction checksum mismatch")
    with np.load(path,allow_pickle=False) as z:
        pred=np.asarray(z["pred_depth"],dtype=np.float32)
    if pred.shape!=(448,448) or not np.isfinite(pred).all() or (pred<=0).any():
        raise ValueError("Frozen metric depth shape or numerical validity failure")
    depth=np.asarray(Image.open(source["depth"]))
    seg=np.asarray(Image.open(source["seg"]))
    if seg.ndim==3:seg=seg[...,0]
    if depth.shape!=seg.shape or depth.ndim!=2:
        raise ValueError("Reference and segmentation not aligned")
    factor=float(rec["depth_factor_divisor"])
    if not np.isfinite(factor) or factor<=0:
        raise ValueError("GC6D depth factor missing")
    box=tuple(map(int,rec["crop_box"]))
    K=np.asarray(rec["K"],dtype=np.float32)
    Kraw=np.asarray(rec["raw_K"],dtype=np.float32)
    generated,_=crop_K_from_raw(Kraw,box,(448,448))
    if np.max(np.abs(K-generated))>2e-3:
        raise RuntimeError("Source P1 crop K no longer agrees with raw K")
    x0,y0,x1,y1=box
    ref=cv2.resize((depth.astype(np.float32)/factor)[y0:y1,x0:x1],
                   (448,448),interpolation=cv2.INTER_NEAREST)
    fg=cv2.resize(np.asarray((seg>0)&(depth>0),dtype=np.uint8)[y0:y1,x0:x1],
                  (448,448),interpolation=cv2.INTER_NEAREST)>0
    return pred,ref,fg


def _csv(path,rows):
    if not rows:return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)


def _paired_deltas(rows,metric):
    keyed={(r["scene"],r["frame"],r["fold"],r["mode"]):r
           for r in rows}
    output={}
    for mode in MODES[1:]:
        paired=[]
        for key,ref in keyed.items():
            scene,frame,fold,m=key
            if m!="identity" or not ref["valid"]:continue
            other=keyed.get((scene,frame,fold,mode))
            if not other or not other["valid"]:continue
            if ref.get(metric) is None or other.get(metric) is None:continue
            paired.append(dict(scene=scene,frame=frame,fold=fold,valid=True,
                               delta=float(other[metric]-ref[metric])))
        output[mode]=paired_scene_bootstrap(paired,"delta")
        output[mode]["paired_folds"]=len(paired)
    return output


def run(a):
    if a.bootstrap<1 or a.min_train<128 or a.min_test<32 or a.guard<0:
        raise ValueError("Invalid spatial oracle evaluation budget")
    manifest=load_cache(a.cache_manifest,complete=not a.smoke)
    records=sorted(manifest["records"],key=lambda r:(r["scene"],r["anno"]))
    if a.smoke:
        if not 0<a.limit<len(records):
            raise ValueError("Explicit smoke limit required")
        records=records[:a.limit]
    elif a.limit:
        raise ValueError("Formal A cannot silently skip cached views")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use independent output")
    root.mkdir(parents=True,exist_ok=True)
    raw=[]
    for index,rec in enumerate(records,1):
        p,r,fg=read_frame(rec)
        fold_rows=evaluate_frame(p,r,fg,min_train=a.min_train,
                                 min_test=a.min_test,guard=a.guard,
                                 affine_min_std=a.affine_min_std)
        for v in fold_rows:
            v.update(scene=int(rec["scene"]),frame=int(rec["img_id"]),
                     anno=int(rec["anno"]),
                     source_prediction_sha256=rec["prediction_sha256"])
        raw.extend(fold_rows)
        if index%26==0 or index==len(records):
            print(f"[A] {index}/{len(records)} scene={rec['scene']}",flush=True)
    _csv(root/"per_fold.csv",raw)
    summary={}
    for mode in MODES:
        subset=[r for r in raw if r["mode"]==mode]
        ok=[r for r in subset if r["valid"]]
        summary[mode]=dict(
            valid_folds=len(ok),total_folds=len(subset),
            skipped_reasons=dict(Counter(r.get("reason") for r in subset if not r["valid"])),
            negative_affine_folds=sum(r.get("affine_nonphysical") is True for r in ok),
            mae=paired_scene_bootstrap(ok,"mae",draws=a.bootstrap,seed=a.seed),
            rmse=paired_scene_bootstrap(ok,"rmse",draws=a.bootstrap,seed=a.seed+1),
            absrel=paired_scene_bootstrap(ok,"absrel",draws=a.bootstrap,seed=a.seed+2),
            gradient_mae=paired_scene_bootstrap(ok,"gradient_mae",draws=a.bootstrap,seed=a.seed+3),
            rank_spearman=paired_scene_bootstrap(ok,"rank_spearman",draws=a.bootstrap,seed=a.seed+4))
    result=dict(schema=SCHEMA,
        source_cache_manifest_sha256=sha256(a.cache_manifest),
        cache_identity_sha256=manifest["manifest_sha256"],
        status="smoke" if a.smoke else "complete",
        frames=len(records),scenes=len({r["scene"] for r in records}),
        folds=len(raw),guard_pixels=a.guard,
        min_train_pixels=a.min_train,min_test_pixels=a.min_test,
        affine_identifiability_std_m=a.affine_min_std,
        results=summary,
        paired_delta_mae=_paired_deltas(raw,"mae"),
        paired_delta_gradient=_paired_deltas(raw,"gradient_mae"),
        interpretation_limits=[
            "Only per-frame oracle coefficients inferred from observed sensor reference on disjoint regions",
            "All fit coefficients are diagnostic ONLY; no network or grasp center adjusted",
            "Rank ordering cannot be repaired by positive affine; negative affine is nonphysical",
            "Affine fit is refused when predicted depth variance is too small",
            "Spatial quadrants are separated by fit-side guard pixels, not entirely independent natural scenes",
            "This is on previously examined GC6D scenes, no untouched generalization claim",
            "Raw sensor depth and segmentation provide metric reference, not CAD-complete geometry"
        ])
    write_json(root/"summary.json",result)
    lines=["# A：GC6D单帧深度的空间留出 Oracle 对齐上限","",
           "三象限拟合、第四象限评价；与测试象限之间在拟合侧留有guard带。",
           "| 模式 | 有效fold | MAE(m) | 同fold相对原始 ΔMAE(m) | 局部梯度MAE(m/pixel) |",
           "|---|---:|---:|---:|---:|"]
    for mode,v in summary.items():
        delta=result["paired_delta_mae"].get(mode,{}).get("mean")
        d="—" if delta is None else f"{delta:+.5f}"
        lines.append(f"| {mode} | {v['valid_folds']}/{v['total_folds']} | "
                     f"{v['mae']['mean']} | {d} | {v['gradient_mae']['mean']} |")
    lines.extend(["","affine包含不受物理约束的有符号斜率；负斜率不能作为部署修复。"
                  ,"不得将每帧Oracle归因于已实现的camera-aware尺度修复。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=3000)
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--min-train",type=int,default=512)
    p.add_argument("--min-test",type=int,default=128)
    p.add_argument("--guard",type=int,default=8)
    p.add_argument("--affine-min-std",type=float,default=.003)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args()
    run(a)


if __name__=="__main__":
    main()
