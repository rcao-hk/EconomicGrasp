"""Rigid-camera multiview same-surface tracking with measured occlusion gates.

The GC6D JSON keys cam_R_w2c/cam_t_w2c are known in this release to
represent C2W. We additionally verify composed model-to-world consistency
against per-view scene_gt before trusting projections.

Point identity is from source measured depth transformed by C2W and checked
against target measured depth, 3D backprojection, and segmentation label.
It is NOT inferred from frozen monocular depth predictions.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def check_transform(T,tol=2e-3):
    t=np.asarray(T,dtype=np.float64)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(
            t[-1], [0,0,0,1],atol=1e-6):
        raise ValueError("Malformed 4x4 homogeneous rigid transform")
    R=t[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=tol) or abs(np.linalg.det(R)-1)>tol:
        raise ValueError("Non-rigid C2W input")
    return t


def camera_to_world(points,T):
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    T=check_transform(T)
    return points@T[:3,:3].T+T[:3,3]


def world_to_camera(points,T):
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    T=check_transform(T)
    return (points-T[:3,3])@T[:3,:3]


def unproject(u,v,z,K):
    k=np.asarray(K,dtype=np.float64)
    if k.shape!=(3,3) or min(k[0,0],k[1,1])<=0:
        raise ValueError("Invalid camera intrinsics")
    u,v,z=np.broadcast_arrays(np.asarray(u,dtype=np.float64),
                              np.asarray(v,dtype=np.float64),
                              np.asarray(z,dtype=np.float64))
    if np.any(z<=0) or not np.isfinite(z).all():
        raise ValueError("Unprojection requires finite positive metric depth")
    xyz=np.stack(((u-k[0,2])*z/k[0,0],
                  (v-k[1,2])*z/k[1,1],z),axis=-1)
    return xyz.reshape(-1,3)


def project(points,K):
    p=np.asarray(points,dtype=np.float64).reshape(-1,3)
    k=np.asarray(K,dtype=np.float64)
    if k.shape!=(3,3) or min(k[0,0],k[1,1])<=0:
        raise ValueError("Invalid camera K")
    z=p[:,2]
    safe=np.where(np.abs(z)>1e-9,z,1.)
    u=(k[0,0]*p[:,0]+k[0,2]*z)/safe
    v=(k[1,1]*p[:,1]+k[1,2]*z)/safe
    return u,v,z


def to_crop(u,v,box,out_hw=(448,448)):
    x0,y0,x1,y1=map(int,box)
    h,w=out_hw
    if x1<=x0 or y1<=y0:
        raise ValueError("Invalid raw crop bounds")
    return ((np.asarray(u,dtype=np.float64)-x0)*w/(x1-x0),
            (np.asarray(v,dtype=np.float64)-y0)*h/(y1-y0))


def bilinear(map_2d,u,v):
    """Strict bilinear sampling, NaN outside valid input bounds."""
    z=np.asarray(map_2d,dtype=np.float64)
    uu=np.asarray(u,dtype=np.float64).reshape(-1)
    vv=np.asarray(v,dtype=np.float64).reshape(-1)
    H,W=z.shape
    good=(np.isfinite(uu)&np.isfinite(vv)&(uu>=0)&(uu<=W-1)&
          (vv>=0)&(vv<=H-1))
    result=np.full(len(uu),np.nan,dtype=np.float64)
    if not good.any():
        return result
    u0=np.floor(uu[good]).astype(int);v0=np.floor(vv[good]).astype(int)
    u1=np.minimum(u0+1,W-1);v1=np.minimum(v0+1,H-1)
    du=uu[good]-u0;dv=vv[good]-v0
    result[good]=((1-du)*(1-dv)*z[v0,u0]+du*(1-dv)*z[v0,u1]+
                  (1-du)*dv*z[v1,u0]+du*dv*z[v1,u1])
    return result


def sample_anchors(depth,label,*,scene,img_id,
                   points_per_object=64,max_points=512,margin=3):
    """Deterministic per-segmentation-label sampling, no predictions involved."""
    depth=np.asarray(depth,dtype=np.float64)
    label=np.asarray(label,dtype=np.int32)
    if depth.shape!=label.shape or depth.ndim!=2:
        raise ValueError("Expected co-aligned raw sensor depth and 2D segmentation")
    H,W=depth.shape
    fg=(depth>0)&np.isfinite(depth)&(label>0)
    fg[:margin,:]=False;fg[-margin:,:]=False
    fg[:,:margin]=False;fg[:,-margin:]=False
    if not fg.any():
        raise ValueError("No measurable foreground surface points")
    picked=[]
    for oid in sorted(set(label[fg].tolist())):
        ids=np.flatnonzero(fg & (label==oid))
        salt=f"gc6d-track-v1:{scene}:{img_id}:{int(oid)}".encode()
        seed=int.from_bytes(hashlib.sha256(salt).digest()[:8],"little")
        rng=np.random.default_rng(seed)
        sel=rng.choice(ids,size=min(points_per_object,len(ids)),replace=False)
        picked.extend(sel.tolist())
    if len(picked)>max_points:
        salt=f"gc6d-cap-v1:{scene}:{img_id}".encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(salt).digest()[:8],"little"))
        picked=rng.choice(picked,size=max_points,replace=False).tolist()
    flat=np.asarray(sorted(picked),dtype=np.int64)
    v,u=np.divmod(flat,W)
    return u,v,label[v,u].astype(np.int64)


def _obj_transform(obj):
    T=np.eye(4,dtype=np.float64)
    T[:3,:3]=np.asarray(obj["cam_R_m2c"],dtype=np.float64).reshape(3,3)
    T[:3,3]=np.asarray(obj["cam_t_m2c"],dtype=np.float64).reshape(3)/1000.
    return check_transform(T)


def verify_c2w_scene_poses(frames,scene_gt,*,translation_tol=.025,angle_tol_deg=5.):
    """Check C2W against observed object-to-camera poses across all views.

    Repeated BOP obj_id within a single frame is ambiguous and omitted, never
    assigned to the first arbitrary instance.
    """
    by_obj={}
    for frame in frames:
        view=str(int(frame["img_id"]))
        if view not in scene_gt:
            raise ValueError(f"GC6D scene_gt missing view {view}")
        objs=scene_gt[view]
        counts={}
        for obj in objs:
            key=int(obj["obj_id"])
            counts[key]=counts.get(key,0)+1
        for obj in objs:
            key=int(obj["obj_id"])
            if counts[key]!=1:continue
            world=check_transform(frame["T_c2w"])@_obj_transform(obj)
            by_obj.setdefault(key,[]).append(world)
    errors=[]
    for oid,poses in by_obj.items():
        if len(poses)<2:continue
        ref=poses[0]
        for other in poses[1:]:
            translation=float(np.linalg.norm(other[:3,3]-ref[:3,3]))
            cos=float(np.clip((np.trace(ref[:3,:3].T@other[:3,:3])-1)/2,-1,1))
            angle=float(np.degrees(np.arccos(cos)))
            errors.append((translation,angle,oid))
    if not errors:
        raise ValueError("No repeated unique CAD object across GC6D views to validate C2W")
    errs=np.asarray([(e[0],e[1]) for e in errors])
    if float(np.median(errs[:,0]))>translation_tol or \
            float(np.median(errs[:,1]))>angle_tol_deg:
        raise RuntimeError("GC6D C2W frame convention inconsistent with object world-pose stability")
    return dict(object_ids=len({e[2] for e in errors}),
                measured_instances=len(errors),
                median_translation_m=float(np.median(errs[:,0])),
                p90_translation_m=float(np.quantile(errs[:,0],.9)),
                median_rotation_deg=float(np.median(errs[:,1])),
                p90_rotation_deg=float(np.quantile(errs[:,1],.9)))


def match_pair(src,dst,u,v,label,*,abs_tol=.008,rel_tol=.012,
               xyz_tol=.018,edge_margin=2):
    """Validate projected same 3D surface across two measured views.

    Return only points that pass raw target z-buffer consistency, equal
    segmentation numeric label and 3D backprojection. Numeric label equality
    is a correspondence gate, not verified CAD-instance identity.
    """
    u=np.asarray(u,dtype=int);v=np.asarray(v,dtype=int)
    label=np.asarray(label,dtype=np.int64)
    if u.shape!=v.shape or label.shape!=u.shape:
        raise ValueError("Anchor coordinates/labels differ")
    src_sensor=np.asarray(src["depth_raw_m"],dtype=np.float64)
    dst_sensor=np.asarray(dst["depth_raw_m"],dtype=np.float64)
    src_seg=np.asarray(src["seg_raw"])
    dst_seg=np.asarray(dst["seg_raw"])
    if src_sensor.shape!=src_seg.shape or dst_sensor.shape!=dst_seg.shape:
        raise ValueError("Reference depth and segmentation resolution differ")
    zsrc=src_sensor[v,u]
    if np.any(zsrc<=0) or not np.isfinite(zsrc).all():
        raise ValueError("Measured anchor source depth invalid")
    world=camera_to_world(unproject(u,v,zsrc,src["raw_K"]),src["T_c2w"])
    source_back=world_to_camera(world,src["T_c2w"])
    if not np.allclose(source_back[:,2],zsrc,atol=1e-6,rtol=0):
        raise RuntimeError("Source camera C2W inverse roundtrip failed")
    target_xyz=world_to_camera(world,dst["T_c2w"])
    ut,vt,zt=project(target_xyz,dst["raw_K"])
    H,W=dst_sensor.shape
    in_view=(np.isfinite(ut)&np.isfinite(vt)&(zt>0)&
             (ut>=edge_margin)&(ut<W-edge_margin)&
             (vt>=edge_margin)&(vt<H-edge_margin))
    u_int=np.rint(np.clip(ut,0,W-1)).astype(int)
    v_int=np.rint(np.clip(vt,0,H-1)).astype(int)
    depth_t=dst_sensor[v_int,u_int]
    seg_t=dst_seg[v_int,u_int].astype(np.int64)
    same_seg=in_view&(seg_t==label)&(label>0)
    known=same_seg&np.isfinite(depth_t)&(depth_t>0)
    occlusion_tol=np.maximum(abs_tol,rel_tol*np.maximum(zt,0))
    visible=known&(np.abs(depth_t-zt)<=occlusion_tol)
    xyz_back=camera_to_world(unproject(
        u_int[known],v_int[known],depth_t[known],dst["raw_K"]),dst["T_c2w"]) if known.any() else np.zeros((0,3))
    back_err=np.full(len(u),np.inf,dtype=np.float64)
    if known.any():back_err[known]=np.linalg.norm(xyz_back-world[known],axis=1)
    visible &= back_err<=xyz_tol
    us,vs=to_crop(u,v,src["crop_box"])
    ud,vd=to_crop(ut,vt,dst["crop_box"])
    pred_src=bilinear(src["pred_depth"],us,vs)
    pred_dst=bilinear(dst["pred_depth"],ud,vd)
    matched=visible&np.isfinite(pred_src)&np.isfinite(pred_dst)&\
            (pred_src>0)&(pred_dst>0)
    records=[]
    for i in np.flatnonzero(matched):
        records.append(dict(
            src_img_id=int(src["img_id"]),dst_img_id=int(dst["img_id"]),
            src_u=int(u[i]),src_v=int(v[i]),seg_label=int(label[i]),
            true_depth_src_m=float(zsrc[i]),
            true_depth_dst_m=float(depth_t[i]),
            pred_depth_src_m=float(pred_src[i]),
            pred_depth_dst_m=float(pred_dst[i]),
            delta_reference_m=float(depth_t[i]-zsrc[i]),
            delta_predicted_m=float(pred_dst[i]-pred_src[i]),
            source_error_m=float(pred_src[i]-zsrc[i]),
            target_error_m=float(pred_dst[i]-depth_t[i]),
            abs_target_geometry_reprojection_m=float(back_err[i]),
            ref_pose_distance_m=float(np.linalg.norm(
                np.asarray(dst["T_c2w"])[:3,3]-np.asarray(src["T_c2w"])[:3,3])),
            crop_fx_src=float(np.asarray(src["K"])[0,0]),
            crop_fx_dst=float(np.asarray(dst["K"])[0,0]),
        ))
    counts=dict(anchors=len(u),inside_fov=int(in_view.sum()),
                same_seg=int(same_seg.sum()),valid_sensor=int(known.sum()),
                depth_consistent=int(visible.sum()),
                matched_pred=int(matched.sum()))
    return records,counts
