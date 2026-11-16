"""Frozen MIX50 GVAR e15 metric-depth response on GC6D D435.

(A) Same-scene 13-view metric distance response.
(B) Same-scene SAME-RAY image crop/zoom with K'=A K (digital, not optical).
(C) K-only negative control and same-scene alternative-pose FiLM stress.

No depth-head training, no grasp quality evaluator, no AP improvement claim.
Input sensor depth/segmentation is used for the ORIGINAL GC6D loader crop and
OFFLINE references, never passed to the frozen DPT predictor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from rare_grasp.runtime import Runtime
from rare_grasp.core import sha256
from diagnostics.decision_supervision.core import BASE_SHA
from .geometry import (HW, INTERVENTIONS, pixel_affine, update_K,
                       warp_tensor, recover_crop_box, paired_depth_metrics,
                       matched_warp_response, check_projective_consistency)


def _scene_list(path):
    p=Path(path)
    data=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data,list) or not data or not all(
        isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in data):
        raise ValueError("Use an explicit JSON list of GC6D integer test scene IDs")
    if len(data)!=len(set(data)):
        raise ValueError("Duplicate GC6D scene IDs in protocol")
    return sorted(data)


def _sha_array(a):
    x=np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(x.tobytes()).hexdigest()


def _init_dataset(a):
    from dataset.gc6d_dataset import GraspClutter6DMultiDataset
    scenes=_scene_list(a.scenes_json)
    ds=GraspClutter6DMultiDataset(
        root=a.dataset_root,camera="realsense-d435",split=a.split,
        scene_ids=scenes,max_ann=a.max_ann,load_label=False,augment=False,
        use_gt_depth=False,use_fuse_depth=False,
        num_points=20000,voxel_size=.005,remove_outlier=None,
        mask_mode=a.mask_mode,workspace_pose_mode=a.workspace_pose_mode,
        workspace_outlier=a.workspace_outlier,
        workspace_depth_trunc=a.workspace_depth_trunc,
        factor_depth_mode=a.factor_depth_mode,
        fallback_to_depth_mask_on_empty=False)
    all_scenes={int(r[0]) for r in ds.samples}
    if all_scenes!=set(scenes):
        raise ValueError("GC6D loader missing source scenes")
    if not a.allow_partial and (
        len(scenes)!=a.expected_scenes or len(ds)!=a.expected_frames):
        raise ValueError(f"Formal GC6D protocol requires {a.expected_scenes} scenes "
                         f"and {a.expected_frames} frames, got {len(scenes)}/{len(ds)}")
    return ds,scenes


def _pose_info(ds,idx):
    _,_,_,_,cam_info,K_raw,factor=ds._get_sample_meta(idx)
    pose,gravity,tilt=ds.build_camera_orientation_metadata(cam_info)
    if np.linalg.norm(pose-0)>1.01 or not np.isclose(np.linalg.norm(pose),1,atol=1e-4):
        raise ValueError("GC6D pose orientation not unit length")
    return pose,gravity,float(tilt),K_raw,cam_info,float(factor)


def _mask_and_crop(ds,idx,K_crop,raw_K,source):
    """Verify native crop intrinsics, then align foreground segmentation."""
    import cv2
    from dataset.gc6d_dataset import load_label
    seg=load_label(ds.labelpath[idx])
    if seg.ndim!=2:
        raise ValueError("GC6D exact foreground mask must be 2D")
    box,error=recover_crop_box(raw_K,K_crop,seg.shape,HW)
    x0,y0,x1,y1=box
    mask=cv2.resize(np.asarray(seg[y0:y1,x0:x1]>0,dtype=np.uint8),
                    (HW[1],HW[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
    ref=np.asarray(source["sensor_depth_m"],dtype=np.float32)
    gt=np.asarray(source["gt_depth_m"],dtype=np.float32)
    if ref.shape!=HW or gt.shape!=HW or mask.shape!=HW:
        raise ValueError("Dataset crop/ref/seg dimensions are misaligned")
    if not np.isfinite(ref).all() or np.any(ref<0):
        raise ValueError("Invalid GC6D observed reference depth")
    if not np.isfinite(gt).all() or np.any(gt<0):
        raise ValueError("Invalid GC6D virtual/rendered reference depth")
    return mask,ref,gt,box,error


def _farthest_pose(idx,ds,cache):
    scene=int(ds.samples[idx][0])
    this=cache[idx]
    possible=[j for j in range(len(ds.samples))
              if int(ds.samples[j][0])==scene and j!=idx]
    if not possible:
        return None,None
    other=max(possible,key=lambda j:-float(np.dot(this,cache[j])))
    delta=float(np.rad2deg(np.arccos(
        np.clip(float(np.dot(this,cache[other])),-1,1))))
    return other,delta


def _depth_forward(depth_model,img,pose,gravity,K,device):
    with torch.inference_mode():
        out=depth_model(
            img.to(device),
            camera_pose_vec=torch.as_tensor(pose,dtype=torch.float32,
                                            device=device).reshape(1,3),
            camera_gravity_vec=torch.as_tensor(gravity,dtype=torch.float32,
                                               device=device).reshape(1,3),
            camera_K=torch.as_tensor(K,dtype=torch.float32,device=device).reshape(1,3,3),
            return_raw=True,return_pose_aux=True)
    if len(out)!=5:
        raise RuntimeError("Frozen DPT output contract changed")
    z=out[0]
    if z.shape!=(1,1,*HW) or not torch.isfinite(z).all() or (z<=0).any():
        raise ValueError("Frozen DPT depth must be strictly positive finite 448x448")
    return z.detach().float().cpu(),out[-1]


def _row_identity(ds,idx):
    scene,imgid,anno=map(int,ds.samples[idx])
    return dict(scene=scene,img_id=imgid,anno=anno)


def _base_metrics(z,ref,gt,mask,min_pixels):
    res=paired_depth_metrics(z,ref,mask,min_pixels)
    secondary=paired_depth_metrics(z,gt,mask,min_pixels)
    return dict(
        **{f"sensor_{k}":v for k,v in res.items()},
        **{f"rendered_{k}":v for k,v in secondary.items()})


def _write_rows(path,rows):
    if not rows:raise ValueError("No depth response rows to write")
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def run(a):
    if a.k_only_scale<=0 or a.min_foreground_pixels<32 or a.limit<0:
        raise ValueError("Invalid intervention protocol")
    if len(a.variants)!=len(set(a.variants)) or any(
            v not in INTERVENTIONS or v=="identity" for v in a.variants):
        raise ValueError("Use unique pre-registered nonidentity zoom/crop interventions")
    if a.intervention_views_per_scene not in (0,1,2,13):
        raise ValueError("intervention_views_per_scene must be 0,1,2 or 13")
    ds,scenes=_init_dataset(a)
    cfg_sha=sha256(a.bound_config)
    rt=Runtime(a.bound_config,a.dataset_root,"",a.device)
    if (rt.config["checkpoint_sha256"]!=BASE_SHA or
        int(rt.config.get("expected_checkpoint_epoch",-1))!=16 or
        float(rt.config.get("mix_training_fraction",0))!=.5 or
        rt.config.get("backbone")!="point_gvar" or
        rt.model.geometry_depth_source!="pred" or
        rt.model.pose_depth_mode!="global_film" or
        rt.model.use_obs_depth):
        raise ValueError("Only strict frozen MIX50 e15 RGB-only global_film GVAR branch supported")
    if any(p.requires_grad for p in rt.model.parameters()):
        raise RuntimeError("Entire base/checkpoint must be frozen")
    depth_model=rt.model.depth_net
    depth_model.eval().requires_grad_(False)
    device=rt.device
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new, empty output root (smoke runs separately)")
    root.mkdir(parents=True,exist_ok=True)
    # Deterministic intervention frames: first/last from each 13-view scene.
    per_scene={}
    for idx,(scene,im,anno) in enumerate(ds.samples):
        per_scene.setdefault(int(scene),[]).append(idx)
    intervention_ids=set()
    for group in per_scene.values():
        if a.intervention_views_per_scene==1:
            intervention_ids.add(group[0])
        elif a.intervention_views_per_scene==2:
            intervention_ids.update((group[0],group[-1]))
        elif a.intervention_views_per_scene==13:
            intervention_ids.update(group)
    pose_by_idx={idx:_pose_info(ds,idx)[0] for idx in range(len(ds))}
    selected=list(range(len(ds)))
    if a.limit:
        if a.limit>=len(selected):raise ValueError("Limit is smoke only, smaller than full coverage")
        selected=selected[:a.limit]
    formal=(not a.limit and not a.allow_partial and len(selected)==a.expected_frames)
    metadata=dict(schema="frozen-gvar-depth-camera-response-v1",
        stage="frozen_depth_response",status="complete" if formal else "smoke_or_partial",
        base_checkpoint_sha256=rt.config["checkpoint_sha256"],
        bound_config_sha256=cfg_sha,base_config_digest=rt.config_digest,
        scene_ids_sha256=sha256(a.scenes_json),
        geometry_code_sha256=sha256(Path(__file__).with_name("geometry.py")),
        runner_code_sha256=sha256(__file__),
        camera="realsense-d435",dataset="GC6D",split=a.split,
        scenes=len(scenes),expected_frames=a.expected_frames,processed_frames=len(selected),
        crop_source="GC6D native workspace mask; recovered and K-verified crop box",
        reference="sensor_depth_m, rendered gt_depth_m (both used only OFFLINE)",
        model="strict RGB-predicted metric depth, global_film",
        train_updates=0,ap_evaluations=0,
        zoom_is="digital crop/resize, NOT physical camera focal-length change",
        variants=a.variants,intervention_views_per_scene=a.intervention_views_per_scene,
        pose="max-angle same-scene alternative pose vector; metadata-only OOD stress",
        K_only="same RGB and pose, changed K; global_film direct-K negative control",
        min_foreground_pixels=a.min_foreground_pixels)
    (root/"protocol.json").write_text(json.dumps(metadata,indent=2))
    rows=[]
    for count,idx in enumerate(selected,1):
        scene_info=_row_identity(ds,idx)
        seed=a.seed+scene_info["scene"]*1009+scene_info["anno"]
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        sample=ds.get_data(idx)
        image=sample["img"].float()
        if image.shape!=(3,*HW) or not torch.isfinite(image).all():
            raise ValueError("Unexpected GC6D native normalized 448x448 RGB crop")
        K=np.asarray(sample["K"],dtype=np.float32)
        pose,gravity,tilt,K_raw,cam_info,factor=_pose_info(ds,idx)
        mask,ref,gt,box,K_error=_mask_and_crop(ds,idx,K,K_raw,sample)
        z0,aux=_depth_forward(depth_model,image[None],pose,gravity,K,device)
        base=z0[0,0].numpy()
        identity=dict(**scene_info,crop_x0=box[0],crop_y0=box[1],
            crop_x1=box[2],crop_y1=box[3],crop_K_error=K_error,
            K_fx=float(K[0,0]),K_fy=float(K[1,1]),
            K_cx=float(K[0,2]),K_cy=float(K[1,2]),
            pose_x=float(pose[0]),pose_y=float(pose[1]),pose_z=float(pose[2]),
            camera_tilt_deg=tilt,
            rgb_sha256=_sha_array(image.numpy()),
            native_K_sha256=_sha_array(K),
            reference_sensor_sha256=_sha_array(ref),
            reference_rendered_sha256=_sha_array(gt),
            foreground_mask_sha256=_sha_array(mask),
            rgb_source_sha256=sha256(ds.colorpath[idx]),
            sensor_depth_source_sha256=sha256(ds.depthpath[idx]),
            segmentation_source_sha256=sha256(ds.labelpath[idx]),
            rendered_depth_source_sha256=sha256(ds.gtdepthpath[idx]),
            scene_camera_sha256=sha256(
                Path(a.dataset_root)/"scenes"/f"{scene_info['scene']:06d}"/"scene_camera.json"),
            factor_depth=factor)
        base_row=dict(**identity,intervention="identity",
            image_warped=False,pose_swapped=False,
            physical_camera_changed=False,
            K_fx_modified=float(K[0,0]),K_fy_modified=float(K[1,1]),
            **_base_metrics(base,ref,gt,mask,a.min_foreground_pixels))
        base_row.update(same_ray_median_delta=0.,same_ray_median_abs_delta=0.,
                        same_ray_median_ratio=1.,common_ray_n=int((mask&(ref>0)).sum()),
                        overlap_fraction=1.,pose_delta_deg=0.,K_only_maxabs=0.)
        rows.append(base_row)
        if idx in intervention_ids:
            # In the current global_film branch, the focal matrix is passed
            # downstream but NOT consumed by the metric-depth adapter.
            K_only=K.copy()
            K_only[0,0]*=a.k_only_scale;K_only[1,1]*=a.k_only_scale
            zk,_=_depth_forward(depth_model,image[None],pose,gravity,K_only,device)
            e=float(np.max(np.abs(base-zk[0,0].numpy())))
            if e>a.k_only_tolerance:
                raise RuntimeError(f"K-only invariant unexpectedly failed: {e:.9g}")
            rows.append(dict(**identity,intervention="K_only_negative_control",
                image_warped=False,pose_swapped=False,
                physical_camera_changed=False,
                K_fx_modified=float(K_only[0,0]),K_fy_modified=float(K_only[1,1]),
                **_base_metrics(zk[0,0].numpy(),ref,gt,mask,a.min_foreground_pixels),
                same_ray_median_delta=0.,same_ray_median_abs_delta=0.,
                same_ray_median_ratio=1.,common_ray_n=int((mask&(ref>0)).sum()),
                overlap_fraction=1.,pose_delta_deg=0.,K_only_maxabs=e))
            for name in a.variants:
                A=pixel_affine(name,HW)
                Knew=update_K(K,A)
                K_parity=check_projective_consistency(K,A)
                img_warp,valid=warp_tensor(image[None].to(device),A)
                new_z,_=_depth_forward(depth_model,img_warp,pose,gravity,Knew,device)
                estimate=new_z[0,0].numpy()
                response=matched_warp_response(
                    base,estimate,ref,mask,A,min_pixels=a.min_foreground_pixels)
                # Also report rendered-depth reference under SAME warp.
                ref_w,_=warp_tensor(torch.as_tensor(gt)[None,None].to(device),A,mode="nearest")
                mask_w,_=warp_tensor(torch.as_tensor(mask,dtype=torch.float32)[None,None].to(device),A,mode="nearest")
                rendered=paired_depth_metrics(
                    estimate,ref_w[0,0].cpu().numpy(),
                    (mask_w[0,0].cpu().numpy()>.5)&valid.cpu().numpy(),
                    a.min_foreground_pixels)
                row=dict(**identity,intervention=name,
                    image_warped=True,pose_swapped=False,
                    physical_camera_changed=False,K_projective_max_error_px=K_parity,
                    K_fx_modified=float(Knew[0,0]),K_fy_modified=float(Knew[1,1]),
                    pose_delta_deg=0.,K_only_maxabs=None)
                row.update({f"sensor_{k}":v for k,v in response.items()
                            if k in ("valid_pixels","valid","median_pred",
                                     "median_reference","median_ratio","median_error",
                                     "mae","absrel")})
                row.update({f"rendered_{k}":v for k,v in rendered.items()})
                row.update({k:v for k,v in response.items()
                            if k in ("common_ray_n","same_ray_median_delta",
                                     "same_ray_median_abs_delta",
                                     "same_ray_median_ratio","overlap_fraction")})
                rows.append(row)
            other_idx,delta=_farthest_pose(idx,ds,pose_by_idx)
            if other_idx is not None:
                swapped_pose=pose_by_idx[other_idx]
                zp,_=_depth_forward(depth_model,image[None],swapped_pose,gravity,K,device)
                res=paired_depth_metrics(zp[0,0].numpy(),ref,mask,a.min_foreground_pixels)
                on=(mask&(ref>0)&(base>0))
                rows.append(dict(**identity,intervention="pose_same_scene_swap",
                    image_warped=False,pose_swapped=True,
                    physical_camera_changed=False,
                    K_fx_modified=float(K[0,0]),K_fy_modified=float(K[1,1]),
                    pose_delta_deg=delta,
                    **{f"sensor_{k}":v for k,v in res.items()},
                    same_ray_median_delta=float(np.median(zp[0,0].numpy()[on]-base[on]))
                                             if on.any() else None,
                    same_ray_median_abs_delta=float(np.median(np.abs(
                        zp[0,0].numpy()[on]-base[on]))) if on.any() else None,
                    same_ray_median_ratio=float(np.median(zp[0,0].numpy()[on]/base[on]))
                                          if on.any() else None,
                    common_ray_n=int(on.sum()),overlap_fraction=1.,
                    K_only_maxabs=None))
            if a.save_maps:
                dump=root/"maps"/f"scene_{scene_info['scene']:06d}"/f"img_{scene_info['img_id']:06d}.npz"
                dump.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dump,
                    baseline=base.astype(np.float16),
                    sensor_reference=ref.astype(np.float16),
                    rendered_reference=gt.astype(np.float16),
                    foreground=mask.astype(np.uint8),K=K,
                    pose=pose,crop=np.asarray(box,dtype=np.int64))
                base_row["map_file"]=str(dump.relative_to(root))
                base_row["map_sha256"]=sha256(dump)
        if count%26==0 or count==len(selected):
            print(f"[GC6D frozen depth] {count}/{len(selected)}, scene={scene_info['scene']}, "
                  f"image={scene_info['img_id']}",flush=True)
    _write_rows(root/"per_frame_response.csv",rows)
    totals=dict(metadata,rows=len(rows),intervention_frames=len(intervention_ids&set(selected)),
                output_csv_sha256=sha256(root/"per_frame_response.csv"))
    (root/"manifest.json").write_text(json.dumps(totals,indent=2))
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--scenes-json",required=True,help="Explicit list of 78 GC6D D435 scene IDs")
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--split",default="test")
    p.add_argument("--seed",type=int,default=20261116)
    p.add_argument("--max-ann",type=int,default=13)
    p.add_argument("--expected-scenes",type=int,default=78)
    p.add_argument("--expected-frames",type=int,default=1014)
    p.add_argument("--allow-partial",action="store_true")
    p.add_argument("--limit",type=int,default=0,help="Smoke; never a formal report")
    p.add_argument("--mask-mode",default="workspace_depth")
    p.add_argument("--workspace-pose-mode",default="json")
    p.add_argument("--workspace-outlier",type=float,default=.02)
    p.add_argument("--workspace-depth-trunc",type=float,default=0.)
    p.add_argument("--factor-depth-mode",default="bop")
    p.add_argument("--min-foreground-pixels",type=int,default=128)
    p.add_argument("--intervention-views-per-scene",type=int,choices=(0,1,2,13),default=2)
    p.add_argument("--variants",nargs="+",default=[
        "zoom_in_115","zoom_out_085","shift_right_08","shift_down_08"])
    p.add_argument("--k-only-scale",type=float,default=1.2)
    p.add_argument("--k-only-tolerance",type=float,default=1e-7)
    p.add_argument("--save-maps",action="store_true")
    a=p.parse_args()
    run(a)


if __name__=="__main__":
    main()
