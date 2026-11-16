"""Exact native-crop input adapters shared by frozen P0/P1 diagnostics.

No rendered GT is required. Sensor depth and segmentation may determine the
native crop and OFFLINE evaluation mask; they never enter depth_net.forward().
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA
from diagnostics.frozen_depth_response.geometry import crop_K_from_raw, HW


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False,
             separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,
                    allow_nan=False),encoding="utf-8")


def check_checkpoint(rt):
    c=rt.config
    model=rt.model
    if (c["checkpoint_sha256"]!=BASE_SHA or
        c.get("backbone")!="point_gvar" or
        float(c.get("mix_training_fraction",-1))!=.5 or
        int(c.get("expected_checkpoint_epoch",-1))!=16 or
        model.pose_depth_mode!="global_film" or
        model.geometry_depth_source!="pred" or
        model.use_obs_depth):
        raise ValueError("Require unchanged frozen MIX50 GVAR36 e15 global_film predicted-depth base")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Frozen experiment cannot have trainable model parameters")
    model.eval()
    return model.depth_net.eval()


def _validate_crop(native_K,raw_K,box,h,w):
    k=np.asarray(native_K,dtype=np.float64)
    if k.shape!=(3,3) or min(k[0,0],k[1,1])<=0:
        raise ValueError("Invalid source camera intrinsics")
    regenerated,_=crop_K_from_raw(raw_K,box,out_hw=HW)
    err=float(np.max(np.abs(k-regenerated)))
    if err>2e-3:
        raise RuntimeError(f"Crop K'=A K parity failed: {err:.8g}")
    x0,y0,x1,y1=box
    if not 0<=x0<x1<=w or not 0<=y0<y1<=h:
        raise ValueError("Native crop outside original frame")
    return err


def _frame_meta(domain,scene,img_id,anno,raw_rgb,raw_depth,raw_seg,
                img,crop_K,raw_K,box,pose,gravity,T_c2w,depth_factor,
                input_paths,*,native_mask,raw_size,reference_source,retain_raw=False):
    import cv2
    h,w=raw_depth.shape
    if raw_rgb.shape!=(h,w,3) or raw_seg.shape!=(h,w) or raw_size!=(h,w):
        raise ValueError("RGB/depth/seg raw resolution mismatch")
    if not np.isfinite(raw_rgb).all() or not np.isfinite(raw_depth).all():
        raise ValueError("Nonfinite raw RGB/sensor")
    if not np.asarray(raw_K).shape==(3,3):
        raise ValueError("Missing raw camera K")
    if not np.isfinite(depth_factor) or depth_factor<=0:
        raise ValueError("Invalid meter depth conversion factor")
    if not native_mask.any():
        raise ValueError("Native workspace crop has zero valid pixels; no invented fallback")
    crop_err=_validate_crop(crop_K,raw_K,box,h,w)
    x0,y0,x1,y1=box
    fg=(np.asarray(raw_seg)>0)&(np.asarray(raw_depth)>0)
    if not fg.any():
        raise ValueError("No exact observable foreground")
    fg_crop=cv2.resize(fg[y0:y1,x0:x1].astype(np.uint8),
                        (HW[1],HW[0]),interpolation=cv2.INTER_NEAREST)>0
    z_raw=np.asarray(raw_depth,dtype=np.float32)/float(depth_factor)
    z_crop=cv2.resize(z_raw[y0:y1,x0:x1],(HW[1],HW[0]),
                       interpolation=cv2.INTER_NEAREST)
    if tuple(img.shape)!=(3,*HW):
        raise ValueError("Original loader must produce normalized [3,448,448] RGB")
    pose=np.asarray(pose,dtype=np.float64).reshape(3)
    gravity=np.asarray(gravity,dtype=np.float64).reshape(3)
    for v in (pose,gravity):
        if not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v),1.,atol=1e-4):
            raise ValueError("Camera unit orientation unavailable")
    T=np.asarray(T_c2w,dtype=np.float64).reshape(4,4)
    if not np.allclose(T[3], [0,0,0,1],atol=1e-6):
        raise ValueError("Invalid 4x4 C2W camera transform")
    if (not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=2e-3) or
            np.linalg.det(T[:3,:3])<.99):
        raise ValueError("Non-rigid or reflected camera-to-world transform")
    return dict(
        domain=domain,scene=int(scene),img_id=int(img_id),anno=int(anno),
        image=img,raw_K=np.asarray(raw_K,dtype=np.float32),
        K=np.asarray(crop_K,dtype=np.float32),
        pose=pose.astype(np.float32),gravity=gravity.astype(np.float32),
        T_c2w=T.astype(np.float32),crop_box=np.asarray(box,dtype=np.int32),
        crop_K_error=crop_err,foreground=fg_crop,reference=z_crop,
        foreground_full_pixels=int(fg.sum()),
        native_mask_pixels=int(native_mask.sum()),
        foreground_fraction=float(fg.mean()),
        crop_area_fraction=float((x1-x0)*(y1-y0)/(w*h)),
        crop_hw=(y1-y0,x1-x0),raw_hw=(h,w),
        reference_source=reference_source,
        depth_factor_divisor=float(depth_factor),
        source_paths={k:str(Path(v).resolve()) for k,v in input_paths.items()},
        source_sha256={k:sha256(v) for k,v in input_paths.items()},
        **(dict(depth_raw_m=z_raw,seg_raw=np.asarray(raw_seg,dtype=np.int32))
           if retain_raw else {}),
    )


def gc6d_frame(dataset,index,*,retain_raw=False):
    """GC6D exact workspace mask but NO virtual_scenes access.

    Uses the same read, crop and normalization functions as
    GraspClutter6DMultiDataset.get_data(), omitting only expensive graspness
    and absent rendered/virtual GT dependencies.
    """
    import cv2
    from PIL import Image
    from dataset.gc6d_dataset import (
        _read_depth_png,load_label,depth_to_cloud,build_mask,
        pose_from_scene_camera,
    )
    scene,imgid,anno=map(int,dataset.samples[index])
    scene_gt_path=Path(dataset.root)/"scenes"/f"{scene:06d}"/"scene_gt.json"
    camera_path=Path(dataset.root)/"scenes"/f"{scene:06d}"/"scene_camera.json"
    color=np.asarray(Image.open(dataset.colorpath[index]).convert("RGB"),
                     dtype=np.float32)/255.
    depth=_read_depth_png(dataset.depthpath[index])
    seg=load_label(dataset.labelpath[index])
    _,_,_,_,cam_info,K_raw,depth_factor=dataset._get_sample_meta(index)
    cloud=depth_to_cloud(depth,K_raw,depth_factor)
    mask=build_mask(depth=depth,label=seg,cloud=cloud,cam_info=cam_info,
                    mode=dataset.mask_mode,
                    bbox_margin_ratio=dataset.bbox_margin_ratio,
                    workspace_outlier=dataset.workspace_outlier,
                    workspace_pose_mode=dataset.workspace_pose_mode,
                    workspace_depth_trunc=dataset.workspace_depth_trunc)
    box=dataset._crop_box_from_mask(mask)
    x0,y0,x1,y1=box
    img=dataset.img_transforms(color[y0:y1,x0:x1].copy())
    K=dataset.resize_intrinsics_with_crop(K_raw,box,out_hw=HW)
    pose,gravity,_=dataset.build_camera_orientation_metadata(cam_info)
    T=pose_from_scene_camera(cam_info)  # verified GC6D C2W semantics
    return _frame_meta(
        "gc6d",scene,imgid,anno,color,depth,seg,img,K,K_raw,box,
        pose,gravity,T,depth_factor,
        dict(rgb=dataset.colorpath[index],depth=dataset.depthpath[index],
             seg=dataset.labelpath[index],camera=camera_path,scene_gt=scene_gt_path),
        native_mask=np.asarray(mask,dtype=bool),raw_size=depth.shape,
        reference_source="sensor_depth_m",retain_raw=retain_raw)


def graspnet_frame(dataset,index,*,domain,retain_raw=False):
    """Native GraspNet workspace crop and current RGB path.

    GN-Trans inherits the original GraspNet camera metadata but uses
    augmented RGB and a rendered/virtual depth source; tag this separately.
    """
    import scipy.io as scio
    from PIL import Image
    from utils.data_utils import (
        CameraInfo,create_point_cloud_from_depth_image,get_workspace_mask
    )
    if domain not in ("original","gntrans"):
        raise ValueError("Unsupported GraspNet data domain")
    scene_name=str(dataset.scenename[index])
    scene=int(scene_name.split("_")[-1])
    anno=int(dataset.frameid[index])
    rgb=np.asarray(Image.open(dataset.colorpath[index]).convert("RGB"),
                   dtype=np.float32)/255.
    depth=np.asarray(Image.open(dataset.depthpath[index]))
    seg=np.asarray(Image.open(dataset.labelpath[index]))
    if seg.ndim==3:seg=seg[...,0]
    meta=scio.loadmat(dataset.metapath[index])
    K_raw=np.asarray(meta["intrinsic_matrix"],dtype=np.float32).reshape(3,3)
    depth_factor=float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
    h,w=depth.shape
    cam=CameraInfo(float(w),float(h),K_raw[0,0],K_raw[1,1],
                   K_raw[0,2],K_raw[1,2],depth_factor)
    cloud=create_point_cloud_from_depth_image(depth,cam,organized=True)
    depth_mask=depth>0
    T_raw,T_table=dataset._load_camera_geometry(scene_name)
    T=np.asarray(T_table@T_raw[anno],dtype=np.float32)
    mask=(depth_mask&
          get_workspace_mask(cloud,seg,trans=T,organized=True,outlier=.02))
    box=dataset._crop_box_from_mask(mask)
    x0,y0,x1,y1=box
    image=dataset.img_transforms(rgb[y0:y1,x0:x1].copy())
    K=dataset.resize_intrinsics_with_crop(K_raw,box,out_hw=HW)
    pose,gravity,_=dataset.build_camera_orientation_metadata(scene_name,anno)
    root=Path(dataset.root)/"scenes"/scene_name/dataset.camera
    paths=dict(rgb=dataset.colorpath[index],depth=dataset.depthpath[index],
               seg=dataset.labelpath[index],meta=dataset.metapath[index],
               camera_poses=root/"camera_poses.npy",
               cam0_wrt_table=root/"cam0_wrt_table.npy")
    return _frame_meta(
        domain,scene,anno,anno,rgb,depth,seg,image,K,K_raw,box,
        pose,gravity,T,depth_factor,paths,native_mask=mask,
        raw_size=(h,w),
        reference_source=("observed_sensor_depth_m" if domain=="original"
                          else "virtual_depth_m_not_sensor"),
        retain_raw=retain_raw)


def frozen_depth(depth_net,frame,device):
    """Run ONLY frozen depth head; never pass depth/masks to model."""
    import torch
    with torch.inference_mode():
        out=depth_net(frame["image"][None].to(device),
                      camera_pose_vec=torch.as_tensor(frame["pose"],device=device)[None],
                      camera_gravity_vec=torch.as_tensor(frame["gravity"],device=device)[None],
                      camera_K=torch.as_tensor(frame["K"],device=device)[None],
                      return_raw=True,return_pose_aux=True)
    z=out[0].detach().float().cpu().numpy()
    if z.shape!=(1,1,*HW) or not np.isfinite(z).all() or (z<=0).any():
        raise ValueError("Frozen metric-depth result must be positive finite [1,1,448,448]")
    return z[0,0]
