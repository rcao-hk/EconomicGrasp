"""Spatially held-out per-frame depth alignment, purely OFFLINE.

Only a spatially separated training region sees reference depth while fitting
scale, shift or affine coefficients. Metrics are computed on disjoint 448x448
quadrants. This measures retained relative geometry, NOT deployable metric depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MODES=("identity","scale","shift","affine")
SCHEMA="depth-oracle-spatial-v1"


def _vector(a,mask):
    result=np.asarray(a,dtype=np.float64)[mask]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite masked depth")
    return result


def spatial_fold_masks(shape,fold,guard=8):
    """Train on three quadrants, hold out the fourth, with guard lines.

    No single holdout pixel or near-boundary training pixel participates in
    fitting and evaluating the same oracle.
    """
    h,w=map(int,shape)
    if h<32 or w<32 or fold not in range(4) or guard<0 or guard>=min(h,w)//4:
        raise ValueError("Invalid fold / image / guard")
    y,x=np.indices((h,w))
    cy,cx=h//2,w//2
    quad=(y>=cy).astype(int)*2+(x>=cx).astype(int)
    test=quad==fold
    train=(quad!=fold)&(abs(x-cx)>=guard)&(abs(y-cy)>=guard)
    if (train&test).any():
        raise RuntimeError("Oracle fit and test spatial masks intersect")
    return train,test


def fit_coefficients(mode,pred,ref,train_mask,min_std=.003):
    """Constrain scale>=0; unconstrained affine is marked nonphysical if a<0."""
    if mode not in MODES:raise ValueError(mode)
    p=_vector(pred,train_mask);r=_vector(ref,train_mask)
    if len(p)<2 or np.any(p<=0) or np.any(r<=0):
        raise ValueError("Need enough positive train reference pixels")
    if mode=="identity":
        return dict(a=1.,b=0.,raw_a=1.,valid=True,
                    train_std=float(p.std()),nonphysical=False)
    if mode=="scale":
        denominator=float(np.dot(p,p))
        a=float(np.dot(p,r)/denominator)
        return dict(a=max(0.,a),b=0.,raw_a=a,valid=True,
                    train_std=float(p.std()),nonphysical=a<0)
    if mode=="shift":
        b=float(np.mean(r-p))
        return dict(a=1.,b=b,raw_a=1.,valid=True,
                    train_std=float(p.std()),nonphysical=False)
    sd=float(np.std(p))
    if sd<min_std:
        # A nearly constant prediction cannot identify a reliable affine
        # slope. Do not silently report a near-singular fitted line.
        return dict(a=None,b=None,raw_a=None,valid=False,
                    reason="predicted-depth variance too small for affine identification",
                    train_std=sd,nonphysical=None)
    m_p=float(p.mean());m_r=float(r.mean())
    a=float(np.dot(p-m_p,r-m_r)/np.dot(p-m_p,p-m_p))
    b=float(m_r-a*m_p)
    return dict(a=a,b=b,raw_a=a,valid=True,
                train_std=sd,nonphysical=bool(a<0))


def apply_coefficients(pred,coef):
    if not coef["valid"]:raise ValueError("Cannot apply unidentified affine oracle")
    p=np.asarray(pred,dtype=np.float64)
    return p*float(coef["a"])+float(coef["b"])


def depth_metrics(pred,ref,mask):
    p=np.asarray(pred,dtype=np.float64)
    r=np.asarray(ref,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    if p.shape!=r.shape or p.shape!=mask.shape:
        raise ValueError("Depth/reference/test shape mismatch")
    support=mask&np.isfinite(p)&np.isfinite(r)&(r>0)
    if not support.any():return None
    pp=p[support];rr=r[support]
    err=pp-rr
    return dict(n=int(len(err)),
                mae=float(np.mean(np.abs(err))),
                rmse=float(np.sqrt(np.mean(err**2))),
                absrel=float(np.mean(np.abs(err)/rr)),
                median_error=float(np.median(err)),
                p90_abs=float(np.quantile(np.abs(err),.9)),
                negative_pred_fraction=float((pp<=0).mean()))


def local_gradient_mae(pred,ref,mask):
    """Spatial local-shape fidelity on held-out adjacent visible pairs only."""
    p=np.asarray(pred,dtype=np.float64)
    r=np.asarray(ref,dtype=np.float64)
    valid=np.asarray(mask,dtype=bool)&np.isfinite(p)&np.isfinite(r)&(r>0)
    values=[]
    counts=[]
    for sl0,sl1 in (((slice(None),slice(None,-1)),
                     (slice(None),slice(1,None))),
                    ((slice(None,-1),slice(None)),
                     (slice(1,None),slice(None)))):
        m=valid[sl0]&valid[sl1]
        if m.any():
            delta=(p[sl1]-p[sl0])-(r[sl1]-r[sl0])
            values.append(float(np.abs(delta[m]).sum()))
            counts.append(int(m.sum()))
    return float(sum(values)/sum(counts)) if counts else None


def rank_agreement(pred,ref,mask,max_sample=4096):
    """Tie-aware Spearman via stable rank, fixed grid subsampling."""
    p=np.asarray(pred,dtype=np.float64)
    r=np.asarray(ref,dtype=np.float64)
    valid=np.asarray(mask,dtype=bool)&(p>0)&(r>0)&np.isfinite(p)&np.isfinite(r)
    ids=np.flatnonzero(valid)
    if len(ids)<3:return None
    if len(ids)>max_sample:
        ids=ids[np.linspace(0,len(ids)-1,max_sample,dtype=int)]
    def rank(x):
        order=np.argsort(x,kind="mergesort")
        vals=x[order];out=np.empty(len(x),dtype=float)
        i=0
        while i<len(x):
            j=i+1
            while j<len(x) and vals[j]==vals[i]:j+=1
            out[order[i:j]]=(i+j+1)/2.
            i=j
        return out
    pp=rank(p.flat[ids]);rr=rank(r.flat[ids])
    pp-=pp.mean();rr-=rr.mean()
    den=float(np.linalg.norm(pp)*np.linalg.norm(rr))
    return float(np.dot(pp,rr)/den) if den>0 else None


def evaluate_frame(pred,ref,foreground,*,min_train=512,min_test=128,
                   guard=8,affine_min_std=.003):
    pred=np.asarray(pred,dtype=np.float64)
    ref=np.asarray(ref,dtype=np.float64)
    fg=np.asarray(foreground,dtype=bool)
    if pred.shape!=ref.shape or fg.shape!=pred.shape or pred.ndim!=2:
        raise ValueError("Expected same-shape 2D depth/reference/foreground")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Nonfinite predicted or reference depth")
    valid=fg&(pred>0)&(ref>0)
    rows=[]
    for fold in range(4):
        train,test=spatial_fold_masks(pred.shape,fold,guard)
        train &= valid;test &= valid
        nt=int(train.sum());nv=int(test.sum())
        for mode in MODES:
            base=dict(fold=fold,mode=mode,fit_pixels=nt,test_pixels=nv,
                      spatial_guard=guard)
            if nt<min_train or nv<min_test:
                rows.append(dict(base,valid=False,reason="insufficient independent spatial support"))
                continue
            coef=fit_coefficients(mode,pred,ref,train,min_std=affine_min_std)
            if not coef["valid"]:
                rows.append(dict(base,valid=False,reason=coef["reason"],
                                 fit_depth_std=coef["train_std"]))
                continue
            calibrated=apply_coefficients(pred,coef)
            metrics=depth_metrics(calibrated,ref,test)
            grad=local_gradient_mae(calibrated,ref,test)
            rho=rank_agreement(calibrated,ref,test)
            rows.append(dict(base,valid=True,reason=None,
                fitted_a=coef["a"],fitted_b=coef["b"],
                affine_nonphysical=coef["nonphysical"],
                fit_depth_std=coef["train_std"],
                gradient_mae=grad,rank_spearman=rho,
                **metrics))
    if not any(r["valid"] for r in rows):
        raise RuntimeError("No valid spatially disjoint held-out depth region")
    return rows


def paired_scene_bootstrap(rows,key,*,draws=3000,seed=20261117):
    """Compute fold->frame->scene means, then bootstrap scenes."""
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in rows:
        if r.get("valid") and r.get(key) is not None:
            grouped[(r["scene"],r["frame"])].append(float(r[key]))
    frame_mean={f:float(np.mean(v)) for f,v in grouped.items()}
    by_scene=defaultdict(list)
    for (scene,frame),v in frame_mean.items():by_scene[scene].append(v)
    values=np.asarray([np.mean(v) for _,v in sorted(by_scene.items())],dtype=float)
    if not len(values):return dict(mean=None,ci95=None,scenes=0)
    if len(values)==1:return dict(mean=float(values[0]),ci95=None,scenes=1)
    rng=np.random.default_rng(seed)
    ids=rng.integers(0,len(values),size=(draws,len(values)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[ids].mean(1),[.025,.975]).tolist(),
                scenes=len(values))
