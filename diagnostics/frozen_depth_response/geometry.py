"""Pixel-exact, metric-depth-preserving crop/zoom diagnostics.

A maps ORIGINAL 448×448 crop pixels to perturbation pixels. We sample
I'(u')=I(A^-1u'), K'=AK, and compare depths at the SAME source rays.
Digital crop/zoom is NOT a physical focal-length or camera-motion change.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch
from torch.nn import functional as F

HW = (448,448)
INTERVENTIONS = {
    "identity": (1.,0.,0.),
    "zoom_in_115": (1.15,0.,0.),
    "zoom_out_085": (.85,0.,0.),
    "shift_right_08": (1.,.08,0.),
    "shift_down_08": (1.,0.,.08),
}


def crop_K_from_raw(raw_K, box, out_hw=HW):
    """K' = diag(sx,sy,1) @ translate(-x0,-y0) @ K."""
    k=np.asarray(raw_K,dtype=np.float64)
    x0,y0,x1,y1=map(int,box)
    h,w=map(int,out_hw)
    if k.shape!=(3,3) or not (x1>x0 and y1>y0 and h>1 and w>1):
        raise ValueError("Invalid K/crop contract")
    a=np.array([[w/(x1-x0),0.,-x0*w/(x1-x0)],
                [0.,h/(y1-y0),-y0*h/(y1-y0)],
                [0.,0.,1.]])
    return a@k,a


def recover_crop_box(raw_K, resized_K, image_hw, out_hw=HW, tol=2e-3):
    """Infer and independently VERIFY the loader's integer crop rectangle.

    GC6D get_data(load_label=False) does not return its crop box. Its
    native helper applies uniform affine scale/translation to intrinsics.
    No segmentation bounding box is guessed if the equality does not hold.
    """
    k0=np.asarray(raw_K,dtype=np.float64)
    k=np.asarray(resized_K,dtype=np.float64)
    h0,w0=map(int,image_hw);h,w=map(int,out_hw)
    if k0.shape!=(3,3) or k.shape!=(3,3):
        raise ValueError("Expected 3x3 camera K")
    if min(k0[0,0],k0[1,1],k[0,0],k[1,1])<=0:
        raise ValueError("Invalid focal length")
    sx=k[0,0]/k0[0,0]
    sy=k[1,1]/k0[1,1]
    cw=w/sx;ch=h/sy
    x0=int(round(k0[0,2]-k[0,2]/sx))
    y0=int(round(k0[1,2]-k[1,2]/sy))
    box=(x0,y0,int(round(x0+cw)),int(round(y0+ch)))
    x0,y0,x1,y1=box
    if x0<0 or y0<0 or x1>w0 or y1>h0:
        raise ValueError(f"Recovered crop outside original image: {box}")
    regenerated,_=crop_K_from_raw(k0,box,out_hw)
    err=float(np.max(np.abs(regenerated-k)))
    if err>tol:
        raise RuntimeError(f"Crop K does not reproduce dataset native transform; max error={err:.7f}")
    return box,err


def pixel_affine(name, hw=HW):
    if name not in INTERVENTIONS:raise ValueError(f"Unknown intervention {name}")
    s,dx,dy=INTERVENTIONS[name]
    h,w=hw;cx=(w-1)/2.;cy=(h-1)/2.
    return np.array([[s,0.,(1.-s)*cx+dx*w],
                     [0.,s,(1.-s)*cy+dy*h],
                     [0.,0.,1.]],dtype=np.float64)


def update_K(K,A):
    k=np.asarray(K,dtype=np.float64)
    a=np.asarray(A,dtype=np.float64)
    if k.shape!=(3,3) or a.shape!=(3,3):
        raise ValueError("Invalid homogeneous camera matrices")
    result=a@k
    if not np.allclose(result[2], [0,0,1],atol=1e-7):
        raise ValueError("Invalid homogeneous affine camera transform")
    if not np.isfinite(result).all() or min(result[0,0],result[1,1])<=0:
        raise ValueError("Invalid transformed focal lengths")
    return result.astype(np.float32)


def inverse_grid(A,hw=HW,device="cpu"):
    """Output pixels to source pixels, align_corners=True."""
    h,w=map(int,hw)
    if h<2 or w<2:raise ValueError("Invalid output grid")
    inv=np.linalg.inv(np.asarray(A,dtype=np.float64))
    yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=torch.float64),
                         torch.arange(w,device=device,dtype=torch.float64),
                         indexing="ij")
    u=float(inv[0,0])*xx+float(inv[0,1])*yy+float(inv[0,2])
    v=float(inv[1,0])*xx+float(inv[1,1])*yy+float(inv[1,2])
    valid=(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
    grid=torch.stack((u*2/(w-1)-1,v*2/(h-1)-1),-1)[None].float()
    return grid,valid


def warp_tensor(x,A,mode="bilinear",padding_mode="zeros"):
    """Warp a 1CHW or BCHW tensor with the SAME pixel affine as K'."""
    if not torch.is_tensor(x) or x.ndim!=4 or x.shape[0]!=1:
        raise ValueError("Expect a single-frame image/depth [1,C,H,W]")
    H,W=x.shape[-2:]
    grid,valid=inverse_grid(A,(H,W),device=x.device)
    result=F.grid_sample(x,grid.to(x.dtype),
                         mode=mode,padding_mode=padding_mode,align_corners=True)
    return result,valid


def project_rays(K,xyz):
    """Pinhole projection for checking A K consistency."""
    k=np.asarray(K,dtype=np.float64)
    xyz=np.asarray(xyz,dtype=np.float64)
    p=(k@xyz.T).T
    return p[:,:2]/p[:,2,None]


def check_projective_consistency(K,A,tol=3e-4):
    xyz=np.array([[.04,.03,.6],[-.06,.04,.75],[.02,-.03,.45]],dtype=float)
    p0=project_rays(K,xyz)
    p1=project_rays(update_K(K,A),xyz)
    p2=(A@np.c_[p0,np.ones(len(p0))].T).T[:,:2]
    err=float(np.max(np.abs(p1-p2)))
    if err>tol:raise RuntimeError(f"Metric K'=A K projection parity failed: {err}")
    return err


def paired_depth_metrics(pred,reference,foreground,min_pixels=128):
    """Paired same-ray foreground-only metrics; no padding/unknown as zeros."""
    pred=np.asarray(pred,dtype=np.float64)
    ref=np.asarray(reference,dtype=np.float64)
    fg=np.asarray(foreground,dtype=bool)
    if pred.shape!=ref.shape or pred.shape!=fg.shape:
        raise ValueError("Paired depth/mask shape mismatch")
    valid=fg&np.isfinite(pred)&np.isfinite(ref)&(pred>0)&(ref>0)
    n=int(valid.sum())
    if n<min_pixels:
        return dict(valid_pixels=n,valid=False,median_pred=None,
                    median_reference=None,median_ratio=None,median_error=None,
                    mae=None,absrel=None)
    p=pred[valid];r=ref[valid]
    return dict(valid_pixels=n,valid=True,
                median_pred=float(np.median(p)),
                median_reference=float(np.median(r)),
                median_ratio=float(np.median(p/r)),
                median_error=float(np.median(p-r)),
                mae=float(np.mean(np.abs(p-r))),
                absrel=float(np.mean(np.abs(p-r)/r)))


def matched_warp_response(baseline,perturbed,ref,foreground,A,min_pixels=128):
    """Compare perturbed prediction to SAME-RAY warped baseline and reference."""
    def tensor(a):
        return torch.as_tensor(np.asarray(a,dtype=np.float32))[None,None]
    warped_base,source_valid=warp_tensor(tensor(baseline),A)
    warped_ref,_=warp_tensor(tensor(ref),A,mode="nearest")
    warped_fg,_=warp_tensor(tensor(foreground).float(),A,mode="nearest")
    base=warped_base[0,0].cpu().numpy()
    reference=warped_ref[0,0].cpu().numpy()
    mask=(warped_fg[0,0].cpu().numpy()>.5)&source_valid.cpu().numpy()
    result=paired_depth_metrics(perturbed,reference,mask,min_pixels)
    true=(mask&(np.asarray(perturbed)>0)&(base>0)&(reference>0))
    if int(true.sum())<min_pixels:
        result.update(common_ray_n=int(true.sum()),same_ray_median_delta=None,
                      same_ray_median_abs_delta=None,
                      same_ray_median_ratio=None)
    else:
        current=np.asarray(perturbed)[true]
        common=base[true]
        result.update(common_ray_n=int(true.sum()),
                      same_ray_median_delta=float(np.median(current-common)),
                      same_ray_median_abs_delta=float(np.median(np.abs(current-common))),
                      same_ray_median_ratio=float(np.median(current/common)))
    result["overlap_fraction"]=float(mask.mean())
    return result
