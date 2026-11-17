"""P0: same-fold CONSTANT and physical nonnegative-affine depth oracles.

Reuses accepted GC6D P1 78*13 FLOAT32 depth cache and the same quadrants,
guard, fit/test sizes and pixels as the previous A oracle. Real sensor reference
depth is used to FIT these privileged per-frame controls, never fed to the
network, cached as a deployable transform, or used for grasp AP.

For all oracles, evaluate identical held-out pixels. Stratify gradient pairs
by reference-defined edge/smooth surface, and report a constant-predictor
shape baseline before claiming retained local geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from diagnostics.depth_followup.p1_track import load_cache
from .oracle_core import (MODES,fit_coefficients,apply_coefficients,
                          depth_metrics,local_gradient_mae,rank_agreement,
                          paired_scene_bootstrap,spatial_fold_masks)
from .oracle_run import read_frame
from .factorial_contract import sha256,write_json

SCHEMA="depth-oracle-constant-nonnegative-followup-v1"
ARMS=("identity","scale","shift","affine","constant_mean",
      "constant_median","affine_nonnegative")


def fit_followup(mode,pred,reference,mask,*,min_std=.003):
    """Fit on EXACT same three quadrants, without holdout pixel access."""
    x=np.asarray(pred,dtype=np.float64)[mask]
    y=np.asarray(reference,dtype=np.float64)[mask]
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid oracle fit support")
    if mode=="constant_mean":
        return dict(a=0.,b=float(y.mean()),valid=True,
                    raw_a=0.,clamped=False,nonphysical=False)
    if mode=="constant_median":
        return dict(a=0.,b=float(np.median(y)),valid=True,
                    raw_a=0.,clamped=False,nonphysical=False)
    if mode=="affine_nonnegative":
        original=fit_coefficients("affine",pred,reference,mask,min_std=min_std)
        if not original["valid"]:
            return dict(valid=False,reason=original["reason"],
                        raw_a=None,clamped=None,nonphysical=None)
        a=max(0.,float(original["a"]))
        # A constrained L2 fit with intercept: for a>=0, optimum intercept
        # equals mean(y)-a*mean(x). If unconstrained slope<0, boundary a=0.
        b=float(y.mean()-a*x.mean())
        return dict(a=a,b=b,valid=True,raw_a=float(original["a"]),
                    clamped=bool(original["a"]<0),nonphysical=False)
    if mode in MODES:
        old=fit_coefficients(mode,pred,reference,mask,min_std=min_std)
        return dict(old,clamped=False)
    raise ValueError(f"Unregistered P0 oracle mode {mode}")


def edge_pair_metrics(pred,reference,heldout,*,smooth_max=.002,edge_min=.006):
    """Reference-gradient stratification ONLY for OFFLINE evaluation.

    Each adjacent pair must lie wholly inside the heldout foreground.
    "Edge" is a depth discontinuity proxy, not a verified object boundary.
    Gradients are computed on original predictions with the chosen oracle
    transform, not on any network re-forward.
    """
    if not 0<smooth_max<edge_min:
        raise ValueError("Require preregistered nonoverlapping smooth/edge thresholds")
    p=np.asarray(pred,dtype=np.float64)
    r=np.asarray(reference,dtype=np.float64)
    mask=np.asarray(heldout,dtype=bool)
    if p.shape!=r.shape or mask.shape!=p.shape:
        raise ValueError("Gradient evaluation geometry mismatch")
    metrics={name:dict(n_pairs=0,mae_sum=0.,ref_strength_sum=0.,
                       sign_matches=0) for name in ("smooth","middle","edge")}
    for direction in (0,1):
        p0,p1=(p[:-1,:],p[1:,:]) if direction==0 else (p[:,:-1],p[:,1:])
        r0,r1=(r[:-1,:],r[1:,:]) if direction==0 else (r[:,:-1],r[:,1:])
        m0,m1=(mask[:-1,:],mask[1:,:]) if direction==0 else (mask[:,:-1],mask[:,1:])
        valid=m0&m1&np.isfinite(p0)&np.isfinite(p1)&np.isfinite(r0)&np.isfinite(r1)
        delta_r=r1-r0
        delta_p=p1-p0
        groups={"smooth":valid&(np.abs(delta_r)<=smooth_max),
                "edge":valid&(np.abs(delta_r)>=edge_min),
                "middle":valid&(np.abs(delta_r)>smooth_max)&(np.abs(delta_r)<edge_min)}
        for group,m in groups.items():
            if not m.any():continue
            v=metrics[group]
            v["n_pairs"]+=int(m.sum())
            v["mae_sum"]+=float(np.abs(delta_p[m]-delta_r[m]).sum())
            v["ref_strength_sum"]+=float(np.abs(delta_r[m]).sum())
            v["sign_matches"]+=int(np.sum(
                (np.sign(delta_p[m])==np.sign(delta_r[m]))&
                (np.abs(delta_r[m])>smooth_max)))
    result={}
    for name,values in metrics.items():
        n=values["n_pairs"]
        result[f"{name}_pairs"]=n
        result[f"{name}_grad_mae"]=values["mae_sum"]/n if n else None
        result[f"{name}_ref_grad_abs"]=values["ref_strength_sum"]/n if n else None
        if name=="edge":
            result["edge_direction_agreement"]=(
                values["sign_matches"]/n if n else None)
    return result


def mean_centered_shape(pred,reference,mask):
    """Non-oracle mean-centered residual; constant predicts no local shape."""
    pred=np.asarray(pred,dtype=np.float64)
    ref=np.asarray(reference,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    if not m.any():return None
    x=pred[m];y=ref[m]
    return float(np.mean(np.abs((x-x.mean())-(y-y.mean()))))


def evaluate_all(pred,ref,fg,*,guard=8,min_train=512,min_test=128,
                 affine_min_std=.003,smooth_max=.002,edge_min=.006):
    pred=np.asarray(pred,dtype=np.float64)
    ref=np.asarray(ref,dtype=np.float64)
    fg=np.asarray(fg,dtype=bool)
    if pred.shape!=ref.shape or fg.shape!=pred.shape or pred.ndim!=2:
        raise ValueError("All oracle tensors must be same 2D shape")
    valid=fg&(pred>0)&(ref>0)&np.isfinite(pred)&np.isfinite(ref)
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Nonfinite depth map")
    out=[]
    for fold in range(4):
        fit,test=spatial_fold_masks(pred.shape,fold,guard)
        fit &= valid;test &= valid
        nt=int(fit.sum());nh=int(test.sum())
        for mode in ARMS:
            base=dict(fold=fold,mode=mode,fit_pixels=nt,
                      test_pixels=nh,guard_pixels=guard)
            if nt<min_train or nh<min_test:
                out.append(dict(base,valid=False,reason="insufficient spatial support"))
                continue
            coef=fit_followup(mode,pred,ref,fit,min_std=affine_min_std)
            if not coef["valid"]:
                out.append(dict(base,valid=False,reason=coef.get("reason"),
                                raw_a=coef.get("raw_a")))
                continue
            y=apply_coefficients(pred,coef)
            metric=depth_metrics(y,ref,test)
            if metric is None:raise RuntimeError("Enough heldout pixels but metrics empty")
            ranks=rank_agreement(y,ref,test)
            result=dict(base,valid=True,reason=None,
                a=float(coef["a"]),b=float(coef["b"]),
                raw_a=float(coef["raw_a"]),clamped=bool(coef["clamped"]),
                nonphysical=bool(coef["nonphysical"]),
                **metric,
                gradient_mae=local_gradient_mae(y,ref,test),
                rank_spearman=ranks,
                centered_shape_mae=mean_centered_shape(y,ref,test),
                **edge_pair_metrics(y,ref,test,smooth_max=smooth_max,
                                    edge_min=edge_min))
            out.append(result)
    if not any(r["valid"] for r in out):
        raise RuntimeError("No valid spatially held-out oracle folds")
    return out


def _csv(path,rows):
    if not rows:return
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)


def _paired(rows,a,b,key,draws,seed):
    d={(r["scene"],r["frame"],r["fold"],r["mode"]):r for r in rows}
    v=[]
    for (scene,frame,fold,mode),row in d.items():
        if mode!=a or not row["valid"] or row.get(key) is None:continue
        other=d.get((scene,frame,fold,b))
        if other is None or not other["valid"] or other.get(key) is None:continue
        v.append(dict(scene=scene,frame=frame,fold=fold,valid=True,
                      delta=float(other[key]-row[key])))
    stat=paired_scene_bootstrap(v,"delta",draws=draws,seed=seed)
    stat["paired_folds"]=len(v)
    return stat


def run(a):
    if not (0<a.smooth_max<a.edge_min and a.bootstrap>0):
        raise ValueError("Invalid preregistered edge thresholds or bootstrap")
    if a.guard!=8 or a.min_train!=512 or a.min_test!=128:
        if not a.smoke:
            raise ValueError("Formal P0 must reproduce prior A 8px,512/128 fold protocol")
    cache=load_cache(a.cache_manifest,complete=not a.smoke)
    records=sorted(cache["records"],key=lambda r:(r["scene"],r["anno"]))
    if a.smoke:
        if not 0<a.limit<len(records):raise ValueError("Smoke requires positive <total frame limit")
        records=records[:a.limit]
    elif a.limit:raise ValueError("Formal oracle cannot skip frames")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use NEW output root")
    root.mkdir(parents=True,exist_ok=True)
    data=[]
    for i,record in enumerate(records,1):
        pred,ref,fg=read_frame(record)
        scores=evaluate_all(pred,ref,fg,guard=a.guard,min_train=a.min_train,
                 min_test=a.min_test,affine_min_std=a.affine_min_std,
                 smooth_max=a.smooth_max,edge_min=a.edge_min)
        data.extend(dict(row,scene=int(record["scene"]),
                         frame=int(record["img_id"]),
                         anno=int(record["anno"])) for row in scores)
        if i%26==0 or i==len(records):
            print(f"[P0 shape oracle] {i}/{len(records)}",flush=True)
    _csv(root/"per_fold.csv",data)
    summary={}
    for i,mode in enumerate(ARMS):
        rows=[r for r in data if r["mode"]==mode]
        valid=[r for r in rows if r["valid"]]
        summary[mode]=dict(
            valid_folds=len(valid),total_folds=len(rows),
            skipped=dict(Counter(r.get("reason") for r in rows if not r["valid"])),
            negative_fitted_slopes=sum(r.get("raw_a",0)<0 for r in valid),
            clamped_folds=sum(bool(r.get("clamped")) for r in valid),
            fitted_a_median=float(np.median([r["a"] for r in valid]))
                            if valid else None,
            metrics={metric:paired_scene_bootstrap(valid,metric,draws=a.bootstrap,
                            seed=a.seed+i*101+j)
                for j,metric in enumerate((
                    "mae","gradient_mae","centered_shape_mae","edge_grad_mae",
                    "smooth_grad_mae","edge_direction_agreement","rank_spearman"))},
        )
    delta={}
    for mode in ("shift","affine","affine_nonnegative"):
        delta[f"{mode}_vs_constant_mean"]={
            "mae":_paired(data,"constant_mean",mode,"mae",a.bootstrap,a.seed+17),
            "edge_grad_mae":_paired(data,"constant_mean",mode,
                                    "edge_grad_mae",a.bootstrap,a.seed+18),
            "centered_shape_mae":_paired(data,"constant_mean",mode,
                                         "centered_shape_mae",a.bootstrap,a.seed+19)}
    for mode in ("constant_mean","constant_median","affine_nonnegative"):
        delta[f"{mode}_vs_unconstrained_affine"]={
            "mae":_paired(data,"affine",mode,"mae",a.bootstrap,a.seed+20),
            "gradient_mae":_paired(data,"affine",mode,
                                  "gradient_mae",a.bootstrap,a.seed+21)}
    result=dict(schema=SCHEMA,status="smoke" if a.smoke else "complete",
       accepted_source_cache_sha256=sha256(a.cache_manifest),
       accepted_source_cache_signature=cache["manifest_sha256"],
       frames=len(records),scenes=len({r["scene"] for r in records}),
       modes=list(ARMS),fits_per_frame=4,
       spatial_guard=a.guard,min_fit_pixels=a.min_train,
       min_eval_pixels=a.min_test,edge_abs_depth_step_m=a.edge_min,
       smooth_abs_depth_step_m=a.smooth_max,
       affine_min_prediction_std_m=a.affine_min_std,
       by_mode=summary,paired_comparisons=delta,
       code_source_sha256=sha256(__file__),
       original_oracle_core_sha256=sha256(Path(__file__).with_name("oracle_core.py")),
       training_updates=0,official_grasp_AP_evaluations=0,
       interpretation_limits=[
          "Constant and constrained affine use PRIVILEGED observed test-frame depth outside heldout quadrants",
          "Positive-shape evidence requires beating constant on same fold in local gradients/rank and MAE",
          "Near-zero or negative affine slopes do not prove reliable local metric shape",
          "Quadrant holdout pixels share the original physical scene; not independent test scenes",
          "GC6D 78 scenes already viewed; scene-cluster CIs descriptive only",
          "Reference-defined edges are an evaluation stratum, not model input",
          "Never transfer coefficients from this oracle into deployable RGB-only depth"
       ])
    write_json(root/"summary.json",result)
    lines=["# GC6D A 后续：Constant vs Nonnegative Affine（空间留出）","",
        f"源帧：{len(records)}；有效模式一致性由各自valid_folds核查；所有oracle仅用于离线诊断。",
        "| 模式 | 有效折 | 留出MAE(m) | 局部形状MAE(m) | Edge梯度MAE(m/px) |",
        "|---|---:|---:|---:|---:|"]
    for mode in ARMS:
        d=summary[mode];mm=d["metrics"]
        fmt=lambda key:"NA" if mm[key]["mean"] is None else f"{mm[key]['mean']:.5f}"
        lines.append(f"| {mode} | {d['valid_folds']}/{d['total_folds']} | "
                     f"{fmt('mae')} | {fmt('centered_shape_mae')} | "
                     f"{fmt('edge_grad_mae')} |")
    lines.extend(["","每折测试像素完全相同。只依据同fold配对的ΔMAE、梯度、排序和对应scene CI判断。",
                  "Constant_mean/median使用拟合区真实参考深度；并非部署能力。",
                  "若Constrained Affine被压到a=0，则退化为Constant-mean。"
                  ])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=3000)
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--guard",type=int,default=8)
    p.add_argument("--min-train",type=int,default=512)
    p.add_argument("--min-test",type=int,default=128)
    p.add_argument("--affine-min-std",type=float,default=.003)
    p.add_argument("--smooth-max",type=float,default=.002)
    p.add_argument("--edge-min",type=float,default=.006)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--limit",type=int,default=0)
    run(p.parse_args())


if __name__=="__main__":
    main()
