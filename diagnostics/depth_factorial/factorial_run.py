"""B: frozen RGB-only depth on crossed PHYSICAL distance × crop policy.

The input source is rendered using Blender camera TRANSLATION with fixed
object/material/illumination/camera K/rotation. Each rendered RGB is reused for
fixed_full, fixed_roi, and native adaptive_workspace crops. Depth/seg data is
ONLY used to determine diagnostic crop and evaluate predictions (privileged).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from diagnostics.decision_supervision.core import BASE_SHA
from diagnostics.depth_followup.common import check_checkpoint
from diagnostics.frozen_depth_response.geometry import HW,crop_K_from_raw,paired_depth_metrics
from rare_grasp.runtime import Runtime
from .factorial_contract import (
    CROP_POLICIES,SOURCE_SCHEMA,sha256,digest,write_json,
    verify_fixed_physics,source_image
)

SCHEMA="physical-distance-by-crop-frozen-pred-v1"


def _boxed(mask,margin=.2):
    yy,xx=np.nonzero(mask)
    H,W=mask.shape
    if not len(xx):raise ValueError("No object pixels for fixed ROI")
    x0=int(xx.min());x1=int(xx.max()+1)
    y0=int(yy.min());y1=int(yy.max()+1)
    dx=int(round(max(8,x1-x0)*margin))
    dy=int(round(max(8,y1-y0)*margin))
    box=(max(0,x0-dx),max(0,y0-dy),min(W,x1+dx),min(H,y1+dy))
    if box[2]<=box[0] or box[3]<=box[1]:
        raise ValueError("Degenerate fixed ROI")
    return box


def _workspace_box(depth,seg,K,T):
    from dataset.gc6d_dataset import depth_to_cloud,build_mask
    depth=np.asarray(depth,dtype=np.float32)
    seg=np.asarray(seg,dtype=np.uint8)
    R=np.asarray(T,dtype=float)[:3,:3]
    t=np.asarray(T,dtype=float)[:3,3]*1000.
    meta={"cam_R_w2c":R.reshape(-1).tolist(),
          "cam_t_w2c":t.tolist()}
    cloud=depth_to_cloud(depth,K,1.)
    workspace=build_mask(
        depth=depth,label=seg,cloud=cloud,cam_info=meta,
        mode="workspace_depth",workspace_outlier=.02,
        workspace_pose_mode="json",workspace_depth_trunc=0.)
    if not workspace.any():
        raise ValueError("Adaptive workspace crop has zero valid pixels")
    yy,xx=np.where(workspace)
    return (int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))


def crop_inputs(rgb,depth,mask,K,box):
    """Exact dataset-style ToTensor → Resize(448) → ImageNet Normalize."""
    H,W=mask.shape
    x0,y0,x1,y1=map(int,box)
    if not (0<=x0<x1<=W and 0<=y0<y1<=H):
        raise ValueError("Invalid physical render crop")
    src=np.asarray(rgb,dtype=np.float32)/255.
    trans=transforms.Compose([
        transforms.ToTensor(),transforms.Resize(HW),
        transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
    ])
    image=trans(src[y0:y1,x0:x1].copy())
    crop_K,_=crop_K_from_raw(K,box,HW)
    label=cv2.resize(np.asarray(mask[y0:y1,x0:x1],dtype=np.uint8),
                     HW[::-1],interpolation=cv2.INTER_NEAREST)>0
    ref=cv2.resize(np.asarray(depth[y0:y1,x0:x1],dtype=np.float32),
                   HW[::-1],interpolation=cv2.INTER_NEAREST)
    if image.shape!=(3,*HW) or not torch.isfinite(image).all():
        raise RuntimeError("Unexpected normalized 448 RGB crop")
    return image,np.asarray(crop_K,dtype=np.float32),ref,label


def run(a):
    source_file=Path(a.source_manifest)
    manifest=json.loads(source_file.read_text(encoding="utf-8"))
    physics=verify_fixed_physics(manifest,strict_files=True)
    plan=manifest["plan"]
    if plan["object"]["kind"]=="procedural_box" and a.require_cad:
        raise ValueError("Formal CAD experiment needs an actual independent object mesh")
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    depth_net=check_checkpoint(rt)
    if rt.config["checkpoint_sha256"]!=BASE_SHA:
        raise ValueError("Mismatched frozen base")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use new output, not prior GC6D runs")
    root.mkdir(parents=True,exist_ok=True)
    W,H=plan["camera"]["width"],plan["camera"]["height"]
    K=np.array([[plan["camera"]["fx"],0.,plan["camera"]["cx"]],
                [0.,plan["camera"]["fy"],plan["camera"]["cy"]],[0.,0.,1.]],
               dtype=np.float32)
    R=np.asarray(plan["camera"]["R_c2w"],dtype=np.float32)
    pose=-R[:,2]
    gravity=R.T@np.array([0.,0.,1.],dtype=np.float32)
    poses=np.linalg.norm(pose);g=np.linalg.norm(gravity)
    if abs(poses-1)>1e-4 or abs(g-1)>1e-4:
        raise RuntimeError("Perspective camera pose is not a unit vector")
    first=manifest["views"][0]
    rgb0,depth0,mask0=source_image(first)
    fixed_roi=_boxed(mask0,a.fixed_roi_margin)
    rows=[];records=[]
    for i,v in enumerate(manifest["views"]):
        rgb,depth,seg=source_image(v)
        if rgb.shape[:2]!=(H,W):
            raise ValueError("Physical rendering resolution changed with distance")
        for policy in CROP_POLICIES:
            if policy=="fixed_full":box=(0,0,W,H)
            elif policy=="fixed_roi":box=fixed_roi
            elif policy=="adaptive_workspace":
                box=_workspace_box(depth,seg,K,v["T_c2w"])
            else:raise ValueError(policy)
            inp,Kn,ref,fg=crop_inputs(rgb,depth,seg,K,box)
            with torch.inference_mode():
                result=depth_net(
                    inp[None].to(rt.device),
                    camera_pose_vec=torch.as_tensor(pose,device=rt.device)[None],
                    camera_gravity_vec=torch.as_tensor(gravity,device=rt.device)[None],
                    camera_K=torch.as_tensor(Kn,device=rt.device)[None],
                    return_raw=True,return_pose_aux=True)
            pred=result[0].detach().float().cpu().numpy()
            if pred.shape!=(1,1,*HW) or not np.isfinite(pred).all() or (pred<=0).any():
                raise RuntimeError("Nonfinite frozen depth result")
            pred=pred[0,0]
            stats=paired_depth_metrics(pred,ref,fg,a.min_fg_pixels)
            if not stats["valid"]:
                raise ValueError(f"Insufficient foreground for distance {v['distance_m']} policy {policy}")
            out=root/"predictions"/policy/f"distance_{i:03d}.npy"
            out.parent.mkdir(parents=True,exist_ok=True)
            np.save(out,pred.astype(np.float32),allow_pickle=False)
            input_sha=hashlib.sha256(np.ascontiguousarray(inp.cpu().numpy()).tobytes()).hexdigest()
            rows.append(dict(distance_id=i,distance_m=float(v["distance_m"]),
                crop_policy=policy,crop_x0=box[0],crop_y0=box[1],
                crop_x1=box[2],crop_y1=box[3],
                crop_fraction=float((box[2]-box[0])*(box[3]-box[1])/(W*H)),
                crop_fx=float(Kn[0,0]),crop_fy=float(Kn[1,1]),
                object_pixels=int(fg.sum()),input_sha256=input_sha,
                pred_sha256=sha256(out),reference_sha256=v["files"]["depth"]["sha256"],
                **stats))
            records.append(dict(distance_id=i,distance_m=float(v["distance_m"]),
                crop_policy=policy,crop_box=list(box),K=Kn.tolist(),
                prediction_file=str(out.resolve()),prediction_sha256=sha256(out),
                frozen_input_sha256=input_sha,
                render_source_sha256=v["files"],
                rgb_source_file=v["files"]["rgb"]["path"],
                depth_source_file=v["files"]["depth"]["path"],
                seg_source_file=v["files"]["seg"]["path"]))
        print(f"[B frozen] physical distance {i+1}/{len(manifest['views'])} "
              f"{v['distance_m']:.3f}m × {len(CROP_POLICIES)} crops",flush=True)
    with (root/"per_condition.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)
    result=dict(schema=SCHEMA,complete=True,
        source_manifest_sha256=sha256(source_file),
        source_plan_sha256=manifest["plan_sha256"],
        base_checkpoint_sha256=BASE_SHA,
        bound_config_sha256=sha256(a.bound_config),
        model="frozen global_film metric depth on RGB+pose",
        data_origin="controlled physically translated Blender rendered RGB/depth/seg",
        camera=physics,crop_policies=list(CROP_POLICIES),
        fixed_roi_box=list(fixed_roi),
        records=records,
        per_condition_sha256=sha256(root/"per_condition.csv"),
        min_foreground_pixels=a.min_fg_pixels,
        no_training=True,reference_used_only_for_crop_and_metric=True,
        no_grasp_AP=True,
        interpretation_limits=[
            "Rendered depth is privileged in adaptive workspace crop, not supplied as depth-network input",
            "Fixed full/ROI crops are compared to a privileged depth-based workspace crop",
            "Synthetic Blender style/domain may be unlike training and real GC6D",
            "A physically translated fixed-K RGB renderer is not a real moving camera observation",
            "No depth rescaling or data-driven model/threshold selection on previously inspected GC6D",
            "Single-object results cannot establish robust unseen-object generalization",
        ])
    result["manifest_sha256"]=digest(result)
    write_json(root/"manifest.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest",required=True)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--min-fg-pixels",type=int,default=128)
    p.add_argument("--fixed-roi-margin",type=float,default=.2)
    p.add_argument("--require-cad",action="store_true")
    a=p.parse_args()
    if not .0<=a.fixed_roi_margin<=1.:p.error("Invalid fixed ROI margin")
    run(a)


if __name__=="__main__":
    main()
