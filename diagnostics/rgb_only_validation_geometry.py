"""Geometry utilities for the four RGB-only grasp validation experiments.

All transforms operate on pixel-centre coordinates and metric camera-z depth.
The functions are deterministic and deliberately independent of the network.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

ROLE_NAMES = ("left_contact", "right_contact", "closing", "finger_body", "palm", "approach")
ROLE_MASKS = {
    "all": (1,1,1,1,1,1),
    "contact_closing": (1,1,1,0,0,0),
    "clearance_approach": (0,0,0,1,1,1),
    "only_left_contact": (1,0,0,0,0,0),
    "only_right_contact": (0,1,0,0,0,0),
    "only_closing": (0,0,1,0,0,0),
    "only_finger_body": (0,0,0,1,0,0),
    "only_palm": (0,0,0,0,1,0),
    "only_approach": (0,0,0,0,0,1),
    "leave_left_contact": (0,1,1,1,1,1),
    "leave_right_contact": (1,0,1,1,1,1),
    "leave_closing": (1,1,0,1,1,1),
    "leave_finger_body": (1,1,1,0,1,1),
    "leave_palm": (1,1,1,1,0,1),
    "leave_approach": (1,1,1,1,1,0),
}

def normalize_role_mask(value) -> np.ndarray:
    if isinstance(value, str):
        if value not in ROLE_MASKS:
            raise ValueError(f"Unknown role mask {value!r}; choices={sorted(ROLE_MASKS)}")
        value = ROLE_MASKS[value]
    mask = np.asarray(value, dtype=bool).reshape(-1)
    if mask.shape != (6,) or not mask.any():
        raise ValueError("A GVAR role mask must keep at least one of six roles")
    return mask

def make_zoom_homography(height: int, width: int, scale: float=1.25,
                         center: Optional[Tuple[float,float]]=None) -> np.ndarray:
    if height < 2 or width < 2 or not np.isfinite(scale) or scale <= 1.0:
        raise ValueError("height/width must be >=2 and zoom scale must be >1")
    cx,cy = center if center is not None else ((width-1)/2,(height-1)/2)
    cw,ch = width/scale,height/scale
    x0,y0 = cx-(cw-1)/2,cy-(ch-1)/2
    sx,sy = width/cw,height/ch
    return np.asarray([[sx,0.,-sx*x0],[0.,sy,-sy*y0],[0.,0.,1.]],dtype=np.float64)

def make_asymmetric_crop_homography(height: int,width: int,crop_box: Sequence[float]) -> np.ndarray:
    if height < 2 or width < 2 or len(crop_box)!=4:
        raise ValueError("Expected image size >=2 and crop_box=(x0,y0,x1,y1)")
    x0,y0,x1,y1=map(float,crop_box)
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):
        raise ValueError(f"Invalid crop box {crop_box} for {(height,width)}")
    sx,sy=width/(x1-x0),height/(y1-y0)
    return np.asarray([[sx,0.,-sx*x0],[0.,sy,-sy*y0],[0.,0.,1.]],dtype=np.float64)

def make_camera_roll_homography(K: np.ndarray,roll_deg: float):
    """New camera coordinates are Rz.T @ X_old; K is unchanged."""
    K=np.asarray(K,dtype=np.float64)
    if K.shape!=(3,3) or not np.isfinite(K).all(): raise ValueError("K must be finite 3x3")
    t=np.deg2rad(float(roll_deg)); c,s=np.cos(t),np.sin(t)
    Rz=np.asarray([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return K@Rz.T@np.linalg.inv(K),Rz

def transform_intrinsics(K: np.ndarray,H: np.ndarray)->np.ndarray:
    K=np.asarray(K,dtype=np.float64); H=np.asarray(H,dtype=np.float64)
    if K.shape!=(3,3) or H.shape!=(3,3): raise ValueError("K and H must be 3x3")
    out=H@K
    if abs(out[2,2])<1e-12: raise ValueError("Invalid transformed intrinsics")
    return out/out[2,2]

def warp_array(array: np.ndarray,H: np.ndarray,out_hw: Tuple[int,int],
               interpolation: str="linear",border_value: float=0.)->np.ndarray:
    try: import cv2
    except Exception as exc: raise ImportError("OpenCV is required for observation warping") from exc
    array=np.asarray(array); h,w=map(int,out_hw)
    if interpolation not in ("linear","nearest"): raise ValueError("interpolation must be linear or nearest")
    flag=cv2.INTER_LINEAR if interpolation=="linear" else cv2.INTER_NEAREST
    return cv2.warpPerspective(array,np.asarray(H,np.float64),(w,h),flags=flag,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=float(border_value))

def unproject_pixels(uv: np.ndarray,depth_z: np.ndarray,K: np.ndarray)->np.ndarray:
    uv=np.asarray(uv,dtype=np.float64); z=np.asarray(depth_z,dtype=np.float64); K=np.asarray(K,dtype=np.float64)
    if uv.shape[-1]!=2 or K.shape!=(3,3): raise ValueError("uv must end in 2 and K must be 3x3")
    hom=np.concatenate((uv,np.ones((*uv.shape[:-1],1))),axis=-1)
    rays=hom@np.linalg.inv(K).T
    rays=rays/np.maximum(rays[...,2:3],1e-12)
    return rays*z[...,None]

def project_points(xyz: np.ndarray,K: np.ndarray)->np.ndarray:
    xyz=np.asarray(xyz,dtype=np.float64); K=np.asarray(K,dtype=np.float64)
    if xyz.shape[-1]!=3 or K.shape!=(3,3): raise ValueError("xyz must end in 3 and K must be 3x3")
    hom=xyz@K.T
    return hom[...,:2]/np.maximum(hom[...,2:3],1e-12)

def depth_to_xyz(depth: np.ndarray,K: np.ndarray)->np.ndarray:
    depth=np.asarray(depth,dtype=np.float64); h,w=depth.shape
    vv,uu=np.indices((h,w),dtype=np.float64)
    return unproject_pixels(np.stack((uu,vv),-1),depth,K)

def fit_plane_svd(depth: np.ndarray,K: np.ndarray,mask: np.ndarray,max_points: int=50000,seed: int=0):
    depth=np.asarray(depth,dtype=np.float64); mask=np.asarray(mask,dtype=bool)
    if depth.shape!=mask.shape: raise ValueError("depth/mask shape mismatch")
    ys,xs=np.nonzero(mask&np.isfinite(depth)&(depth>0))
    if len(xs)<3: raise ValueError(f"Need >=3 support-plane pixels, got {len(xs)}")
    if len(xs)>max_points:
        rng=np.random.default_rng(int(seed)); keep=np.sort(rng.choice(len(xs),int(max_points),replace=False))
        xs,ys=xs[keep],ys[keep]
    xyz=unproject_pixels(np.stack((xs,ys),-1),depth[ys,xs],K)
    center=xyz.mean(0); _,_,vh=np.linalg.svd(xyz-center,full_matrices=False)
    normal=vh[-1]; normal=normal/max(np.linalg.norm(normal),1e-12)
    if normal[2]<0: normal=-normal
    offset=-float(normal@center); residual=xyz@normal+offset
    return np.asarray([*normal,offset]),{"plane_points":int(len(xyz)),
      "plane_rmse_m":float(np.sqrt(np.mean(residual**2))),
      "plane_median_abs_m":float(np.median(np.abs(residual)))}

def plane_depth_map(K: np.ndarray,height: int,width: int,plane: Sequence[float]):
    K=np.asarray(K,dtype=np.float64); plane=np.asarray(plane,dtype=np.float64).reshape(-1)
    if K.shape!=(3,3) or plane.shape!=(4,): raise ValueError("Expected K[3,3] and plane[4]")
    vv,uu=np.indices((int(height),int(width)),dtype=np.float64)
    pix=np.stack((uu,vv,np.ones_like(uu)),-1)
    rays=pix@np.linalg.inv(K).T; rays=rays/np.maximum(rays[...,2:3],1e-12)
    denom=np.einsum("hwc,c->hw",rays,plane[:3])
    z=-plane[3]/np.where(np.abs(denom)>1e-10,denom,np.nan)
    valid=np.isfinite(z)&(z>0)
    return np.where(valid,z,0.).astype(np.float32),valid

def align_global_median_ratio(pred: np.ndarray,gt: np.ndarray,mask: np.ndarray):
    pred=np.asarray(pred,dtype=np.float64); gt=np.asarray(gt,dtype=np.float64); mask=np.asarray(mask,dtype=bool)
    valid=mask&np.isfinite(pred)&np.isfinite(gt)&(pred>0)&(gt>0)
    if valid.sum()<16: raise ValueError(f"Too few valid foreground pixels: {int(valid.sum())}")
    pm,gm=np.median(pred[valid]),np.median(gt[valid])
    if pm<=0 or not np.isfinite(pm) or not np.isfinite(gm): raise ValueError("Invalid median depth")
    alpha=float(gm/pm)
    return (pred*alpha).astype(np.float32),alpha

def align_support_plane(pred: np.ndarray,plane_z: np.ndarray,background_mask: np.ndarray):
    pred=np.asarray(pred,dtype=np.float64); plane_z=np.asarray(plane_z,dtype=np.float64)
    mask=np.asarray(background_mask,dtype=bool)
    if pred.shape!=plane_z.shape or pred.shape!=mask.shape: raise ValueError("shape mismatch")
    valid=mask&np.isfinite(pred)&np.isfinite(plane_z)&(pred>0)&(plane_z>0)
    if valid.sum()<16: raise ValueError(f"Too few valid background pixels: {int(valid.sum())}")
    alpha=float(np.median(plane_z[valid]/pred[valid]))
    if not np.isfinite(alpha) or alpha<=0: raise ValueError(f"Invalid plane scale {alpha}")
    return (pred*alpha).astype(np.float32),alpha

def _normal_map(depth: np.ndarray,K: np.ndarray):
    xyz=depth_to_xyz(depth,K); dx=np.gradient(xyz,axis=1); dy=np.gradient(xyz,axis=0)
    normal=np.cross(dx,dy); norm=np.linalg.norm(normal,axis=-1,keepdims=True)
    valid=np.isfinite(normal).all(-1)&(norm[...,0]>1e-8)
    return normal/np.maximum(norm,1e-8),valid

def depth_geometry_metrics(pred: np.ndarray,gt: np.ndarray,K: np.ndarray,
                           mask: Optional[np.ndarray]=None,patch_size: int=16)->Dict[str,float]:
    pred=np.asarray(pred,dtype=np.float64); gt=np.asarray(gt,dtype=np.float64)
    if pred.shape!=gt.shape or pred.ndim!=2: raise ValueError("pred/gt must be same HxW")
    if mask is None: mask=np.ones_like(gt,dtype=bool)
    mask=np.asarray(mask,dtype=bool)
    valid=mask&np.isfinite(pred)&np.isfinite(gt)&(pred>0)&(gt>0)
    if not valid.any(): raise ValueError("No valid pixels")
    err=pred[valid]-gt[valid]
    out={"valid_pixels":int(valid.sum()),"depth_mae_m":float(np.mean(np.abs(err))),
         "depth_rmse_m":float(np.sqrt(np.mean(err**2))),"depth_bias_m":float(np.mean(err)),
         "depth_p90_abs_m":float(np.percentile(np.abs(err),90))}
    rel_errors=[]; grad_errors=[]; h,w=gt.shape; ps=max(2,int(patch_size))
    for y0 in range(0,h,ps):
      for x0 in range(0,w,ps):
        sl=np.s_[y0:min(y0+ps,h),x0:min(x0+ps,w)]; m=valid[sl]
        if m.sum()<8: continue
        p,g=pred[sl],gt[sl]; pm,gm=np.median(p[m]),np.median(g[m])
        rel_errors.extend(np.abs((p[m]-pm)-(g[m]-gm)).tolist())
        if p.shape[1]>1:
          mp=m[:,1:]&m[:,:-1]
          if mp.any(): grad_errors.extend(np.abs(np.diff(p,axis=1)[mp]-np.diff(g,axis=1)[mp]).tolist())
        if p.shape[0]>1:
          mp=m[1:,:]&m[:-1,:]
          if mp.any(): grad_errors.extend(np.abs(np.diff(p,axis=0)[mp]-np.diff(g,axis=0)[mp]).tolist())
    out["local_relative_depth_mae_m"]=float(np.mean(rel_errors)) if rel_errors else float("nan")
    out["local_gradient_mae_m"]=float(np.mean(grad_errors)) if grad_errors else float("nan")
    npred,vp=_normal_map(pred,K); ngt,vg=_normal_map(gt,K); nm=valid&vp&vg
    if nm.any():
      dot=np.clip(np.sum(npred[nm]*ngt[nm],axis=-1),-1.,1.)
      angles=np.rad2deg(np.arccos(dot))
      out["normal_angle_mean_deg"]=float(np.mean(angles))
      out["normal_angle_p90_deg"]=float(np.percentile(angles,90))
    else:
      out["normal_angle_mean_deg"]=float("nan"); out["normal_angle_p90_deg"]=float("nan")
    return out

def rotation_geodesic_deg(R0: np.ndarray,R1: np.ndarray)->float:
    R0=np.asarray(R0,dtype=np.float64).reshape(3,3); R1=np.asarray(R1,dtype=np.float64).reshape(3,3)
    d=R0.T@R1; c=np.clip((np.trace(d)-1.)/2.,-1.,1.)
    return float(np.rad2deg(np.arccos(c)))

def spearman_rank_correlation(a: np.ndarray,b: np.ndarray)->float:
    from scipy.stats import rankdata
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.size<2 or a.shape!=b.shape: return float("nan")
    ra,rb=rankdata(a),rankdata(b)
    if np.std(ra)<1e-12 or np.std(rb)<1e-12: return float("nan")
    return float(np.corrcoef(ra,rb)[0,1])

def match_topk_actions(pred0: np.ndarray,pred1: np.ndarray,topk: int=50)->Dict[str,float]:
    from scipy.optimize import linear_sum_assignment
    p0=np.asarray(pred0,dtype=np.float64); p1=np.asarray(pred1,dtype=np.float64)
    if p0.ndim!=2 or p1.ndim!=2 or p0.shape[1]!=17 or p1.shape[1]!=17:
        raise ValueError("predictions must be [N,17] GraspGroup arrays")
    p0=p0[np.argsort(-p0[:,0])[:max(1,int(topk))]]
    p1=p1[np.argsort(-p1[:,0])[:max(1,int(topk))]]
    if not len(p0) or not len(p1):
      return {"matched":0,"center_displacement_mm":float("nan"),
              "orientation_error_deg":float("nan"),"score_spearman":float("nan")}
    c0,c1=p0[:,13:16],p1[:,13:16]; R0=p0[:,4:13].reshape(-1,3,3); R1=p1[:,4:13].reshape(-1,3,3)
    cd=np.linalg.norm(c0[:,None,:]-c1[None,:,:],axis=-1); rd=np.empty((len(p0),len(p1)))
    for i in range(len(p0)):
      for j in range(len(p1)): rd[i,j]=rotation_geodesic_deg(R0[i],R1[j])
    rows,cols=linear_sum_assignment(cd/0.03+rd/30.)
    cm=cd[rows,cols]*1000.; rm=rd[rows,cols]
    return {"matched":int(len(rows)),"center_displacement_mm":float(np.mean(cm)),
      "center_displacement_p90_mm":float(np.percentile(cm,90)),
      "orientation_error_deg":float(np.mean(rm)),
      "orientation_error_p90_deg":float(np.percentile(rm,90)),
      "score_spearman":spearman_rank_correlation(p0[rows,0],p1[cols,0])}

def assert_fixed_action_bank_identity(reference: np.ndarray,candidate: np.ndarray,atol: float=1e-6):
    reference=np.asarray(reference,dtype=np.float64); candidate=np.asarray(candidate,dtype=np.float64)
    if reference.shape!=candidate.shape or reference.ndim!=2 or reference.shape[1]!=17:
      raise AssertionError(f"Fixed action bank shape mismatch: {reference.shape} vs {candidate.shape}")
    if not np.allclose(reference[:,1:],candidate[:,1:],atol=atol,rtol=0.):
      delta=np.max(np.abs(reference[:,1:]-candidate[:,1:]))
      raise AssertionError(f"Fixed action bank geometry changed (max abs delta={delta})")
