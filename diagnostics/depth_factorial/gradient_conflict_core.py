"""Parameter-space diagnostic of same-physical-ray supervision across crops.

Positive cosine means a small gradient step on crop A locally decreases the
crop B loss to first order; negative means locally opposing directions.
This is a local gradient alignment test, NOT proof that SGD will converge or
that a particular feature is the unique cause of domain shift.
"""
from __future__ import annotations

import hashlib
import numpy as np

CROPS=("fixed_full","fixed_roi","adaptive_workspace")
PAIR_ORDER=(("fixed_full","fixed_roi"),("fixed_full","adaptive_workspace"),
            ("fixed_roi","adaptive_workspace"))
DISTANCE_INDICES=(0,2,4)


def deterministic_seed(mesh_sha,distance_id,seed=20261117):
    token=f"grad-crosscrop-v1:{mesh_sha}:{distance_id}:{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8],"little")


def shared_raw_points(mask,reference,boxes,*,max_points=128,margin=3,seed=0):
    """Select EXACT same source-image rays; independent of predicted scores."""
    mask=np.asarray(mask,dtype=bool)
    ref=np.asarray(reference,dtype=np.float64)
    if mask.ndim!=2 or mask.shape!=ref.shape or set(boxes)!=set(CROPS):
        raise ValueError("Need matched raw segmentation/depth and 3 crop boxes")
    h,w=mask.shape
    yy,xx=np.indices(mask.shape)
    valid=mask&(ref>0)&np.isfinite(ref)
    for crop in CROPS:
        x0,y0,x1,y1=map(int,boxes[crop])
        if not(0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError(f"Invalid {crop} raw crop box")
        # Exclude boundaries to avoid padded or cropped-out interpolation.
        valid &= (xx>=x0+margin)&(xx<x1-margin)&(yy>=y0+margin)&(yy<y1-margin)
    locations=np.flatnonzero(valid)
    if len(locations)<max_points:
        raise ValueError(f"Only {len(locations)} common physical rays; need {max_points}")
    ids=np.random.default_rng(seed).choice(locations,size=max_points,replace=False)
    v,u=np.divmod(ids,w)
    return u.astype(np.int64),v.astype(np.int64),ref[v,u].astype(np.float32)


def gradient_pair(g1,g2):
    x=np.asarray(g1,dtype=np.float64).reshape(-1)
    y=np.asarray(g2,dtype=np.float64).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Gradient vectors nonfinite or shape mismatch")
    dot=float(np.dot(x,y))
    norms=(float(np.linalg.norm(x)),float(np.linalg.norm(y)))
    if min(norms)<=1e-14:
        return dict(cosine=None,dot=dot,norm_a=norms[0],norm_b=norms[1],
                    first_order_conflict=None,source_step_transfer=None)
    cos=float(dot/(norms[0]*norms[1]))
    return dict(cosine=float(np.clip(cos,-1.,1.)),dot=dot,
                norm_a=norms[0],norm_b=norms[1],
                first_order_conflict=bool(dot<0),
                # Expected local loss-B direction under step -g_A,
                # normalized by source gradient energy.
                source_step_transfer=dot/(norms[0]**2))


def per_group_vectors(names,grads):
    """Group Torch parameter gradients by top-level DPT decoder block."""
    if len(names)!=len(grads):raise ValueError("Parameter/group count mismatch")
    groups={}
    for name,g in zip(names,grads):
        prefix=name.split(".")[0]
        groups.setdefault(prefix,[]).append(np.asarray(g,dtype=np.float32).ravel())
    return {k:np.concatenate(v) for k,v in sorted(groups.items())}


def summarize_cases(rows):
    """Do NOT treat gradient parameter counts or repeated crops as IID scenes."""
    out={}
    for state in sorted({r["state"] for r in rows}):
        for pair in PAIR_ORDER:
            selected=[r for r in rows if r["state"]==state and
                       (r["crop_a"],r["crop_b"])==pair and
                       r.get("layer")=="all"]
            if not selected:continue
            vals=[r["cosine"] for r in selected if r.get("cosine") is not None]
            out[f"{state}/{pair[0]}/{pair[1]}"]=dict(
                cases=len(selected),valid_cosines=len(vals),
                mean_cosine=float(np.mean(vals)) if vals else None,
                median_cosine=float(np.median(vals)) if vals else None,
                negative_fraction=float(np.mean([v<0 for v in vals]))
                                  if vals else None,
                samples_by_distance={str(d):len([r for r in selected
                                  if r["distance_id"]==d]) for d in DISTANCE_INDICES},
            )
    return out
