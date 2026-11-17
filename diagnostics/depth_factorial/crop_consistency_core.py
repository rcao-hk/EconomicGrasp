"""Pure same-camera, same-raw-ray comparison of 448-depth across 3 crops.

A physical camera view is rendered ONCE and then reparameterized by cropping.
Crop invariance is evaluated on identical measured object pixels, not on
unpaired crop-average depth statistics. These are diagnostic, not grasp AP.
"""
from __future__ import annotations

import numpy as np

from diagnostics.depth_followup.multiview import to_crop,bilinear
from .factorial_contract import CROP_POLICIES


def sample_crop_predictions(raw_u,raw_v,ref_z,masks_and_maps,*,min_points=32):
    u=np.asarray(raw_u,dtype=float).reshape(-1)
    v=np.asarray(raw_v,dtype=float).reshape(-1)
    ref=np.asarray(ref_z,dtype=float).reshape(-1)
    if len(u)!=len(v) or len(v)!=len(ref):
        raise ValueError("Different source-point array lengths")
    if not np.isfinite(ref).all() or np.any(ref<=0):
        raise ValueError("Need exact positive measured optical-z at raw world surface")
    if set(masks_and_maps)!=set(CROP_POLICIES):
        raise ValueError("All 3 source-crop observations required")
    predictions={}
    keep=np.ones(len(ref),dtype=bool)
    for crop in CROP_POLICIES:
        record=masks_and_maps[crop]
        depth=np.asarray(record["pred"],dtype=float)
        box=np.asarray(record["crop_box"],dtype=int)
        if depth.shape!=(448,448) or box.shape!=(4,) or not np.isfinite(depth).all():
            raise ValueError(f"Invalid frozen 448-depth or raw crop for {crop}")
        uc,vc=to_crop(u,v,box,(448,448))
        # Restrict to interior: bilinear interpolation must not touch outside
        # crop/padded area, and must produce one-to-one raw ray comparisons.
        valid=((uc>=1)&(uc<=446)&(vc>=1)&(vc<=446))
        p=bilinear(depth,uc,vc)
        valid &= np.isfinite(p)&(p>0)
        keep &=valid
        predictions[crop]=p
    if int(keep.sum())<min_points:
        raise RuntimeError("Too few same raw world-surface pixels in all three crop strategies")
    p={crop:predictions[crop][keep] for crop in CROP_POLICIES}
    target=ref[keep]
    roi=p["fixed_roi"]
    mu=np.stack([p[c] for c in CROP_POLICIES],axis=0)
    result=dict(
       points=int(keep.sum()),accepted_raw_index=np.flatnonzero(keep),
       true_z=target,pred_by_crop=p,
       crosscrop_std=float(np.std(mu,axis=0).mean()),
       crosscrop_range=float(np.ptp(mu,axis=0).mean()),
       contrast_adaptive_roi=float(np.mean(p["adaptive_workspace"]-roi)),
       contrast_full_roi=float(np.mean(p["fixed_full"]-roi)),
       abs_adaptive_roi=float(np.mean(np.abs(p["adaptive_workspace"]-roi))),
       abs_full_roi=float(np.mean(np.abs(p["fixed_full"]-roi))),
    )
    return result


def evaluate_shared_points(result):
    p=result["pred_by_crop"];target=result["true_z"]
    if not len(target):raise ValueError("No physical surface points")
    rows=[]
    for crop in CROP_POLICIES:
        pred=p[crop]
        rows.append(dict(crop_policy=crop,points=len(target),
            mae=float(np.mean(np.abs(pred-target))),
            mean_bias=float(np.mean(pred-target)),
            median_prediction=float(np.median(pred)),
            median_reference=float(np.median(target)),
            delta_from_roi_m=float(np.mean(pred-p["fixed_roi"])),
            mean_abs_delta_from_roi_m=float(np.mean(np.abs(pred-p["fixed_roi"]))),
            shared_three_crop_std_m=result["crosscrop_std"],
            shared_three_crop_range_m=result["crosscrop_range"]))
    return rows
