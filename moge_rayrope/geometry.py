"""Pure PyTorch geometry and bounded-cost MoGe-inspired alignment losses.

The global solver is the untruncated weighted-L1 scale+camera-Z-shift problem
on a bounded set of samples. It is NOT the complete truncated ROE solver in
MoGe. Local losses use independently median-centered, scale-aligned patches.
No target-derived transform is used to construct inference geometry.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def image_rays(K, hw, image_hw=None):
    if K.ndim != 3 or K.shape[-2:] != (3,3):
        raise ValueError("K must be [B,3,3]")
    h,w=map(int,hw); H,W=map(int,image_hw or hw)
    if min(h,w,H,W)<2 or not torch.isfinite(K).all() or bool((K[:,(0,1),(0,1)]<=0).any()):
        raise ValueError("Invalid intrinsics/image dimensions")
    yy,xx=torch.meshgrid(torch.linspace(0,H-1,h,device=K.device,dtype=K.dtype),
                         torch.linspace(0,W-1,w,device=K.device,dtype=K.dtype),indexing="ij")
    rx=(xx[None]-K[:,0,2,None,None])/K[:,0,0,None,None]
    ry=(yy[None]-K[:,1,2,None,None])/K[:,1,1,None,None]
    return torch.stack((rx,ry,torch.ones_like(rx)),1)


def rays_at_pixels(uv,K):
    shape=(K.shape[0],)+(1,)*(uv.ndim-2)
    rx=(uv[...,0]-K[:,0,2].reshape(shape))/K[:,0,0].reshape(shape)
    ry=(uv[...,1]-K[:,1,2].reshape(shape))/K[:,1,1].reshape(shape)
    return torch.stack((rx,ry,torch.ones_like(rx)),-1)


def sample_map(x,uv,image_hw):
    H,W=image_hw
    grid=torch.stack((2*uv[...,0]/(W-1)-1,2*uv[...,1]/(H-1)-1),-1)
    b=x.shape[0]
    out=F.grid_sample(x,grid.reshape(b,-1,1,2).to(x.dtype),mode="bilinear",
                      padding_mode="zeros",align_corners=True)
    return out[...,0].transpose(1,2).reshape(*uv.shape[:-1],x.shape[1])


def known_K_z_shift(points,rays):
    """Fit the projective camera-Z shift from prediction and K ONLY.

    p_xy ~ ray_xy * (p_z + t). Differentiable least-squares inference fit;
    the metric scale is unobservable here and is predicted separately.
    """
    if points.shape != rays.shape or points.shape[1]!=3:
        raise ValueError("Expected point/ray maps [B,3,H,W]")
    r=rays[:,:2].double(); p=points.double()
    rr=r.square().sum(1,keepdim=True)
    numerator=(r*p[:,:2]).sum(1,keepdim=True)-rr*p[:,2:3]
    return (numerator.sum((2,3),keepdim=True)/rr.sum((2,3),keepdim=True).clamp_min(1e-8)).to(points.dtype)


def canonicalize_points(points,rays):
    """Fix positive-scale/Z-shift gauge using predictions only.

    Returns canonical points, fitted Z shift and transverse RMS scale. No GT is
    involved. Gauge equivariance is exact away from numerical floor bounds.
    """
    t=known_K_z_shift(points,rays)
    shifted=torch.cat((points[:,:2],points[:,2:3]+t),1)
    # Compute the gauge in float64: finite large pointmaps can overflow a float32 square.
    scale=shifted[:,:2].double().square().mean((1,2,3),keepdim=True).clamp_min(1e-6).sqrt().to(points.dtype)
    return shifted/scale,t,scale


def metric_depth_from_shape(canonical,metric_anchor):
    """A scalar metric anchor sets median optical Z; local shape is detached."""
    z=canonical[:,2:3].detach().clamp_min(1e-3)
    med=z.flatten(1).median(1).values[:,None,None,None].clamp_min(1e-3)
    return metric_anchor*z/med


@torch.no_grad()
def weighted_median(x,w):
    if x.shape!=w.shape: raise ValueError("weighted median shape mismatch")
    order=x.argsort(dim=-1,stable=True)
    xs=x.gather(-1,order); ws=w.gather(-1,order).clamp_min(0)
    total=ws.sum(-1,keepdim=True)
    idx=(ws.cumsum(-1)<.5*total).sum(-1,keepdim=True).clamp_max(x.shape[-1]-1)
    out=xs.gather(-1,idx).squeeze(-1)
    return torch.where(total.squeeze(-1)>1e-12,out,torch.ones_like(out))


@torch.no_grad()
def fit_scale_l1(x,y,w):
    """Positive scalar argmin sum w*|s*x-y|, batched in final axis."""
    good=x.abs()>1e-10
    ratio=torch.where(good,y/torch.where(good,x,torch.ones_like(x)),torch.zeros_like(x))
    weight=w*torch.where(good,x.abs(),torch.zeros_like(x))
    return weighted_median(ratio,weight).clamp_min(0.)


@torch.no_grad()
def fit_scale_zshift_l1(pred,target,weight):
    """Enumerate zero-Z-residual anchors, solve weighted L1 scale per anchor.

    For fixed s, a weighted-median Z residual gives an optimal t and at least one
    active anchor. Enumerating anchors gives the untruncated convex optimum on
    the supplied samples (including the nonnegative-scale boundary).
    """
    if pred.ndim!=2 or pred.shape[-1]!=3 or pred.shape!=target.shape:
        raise ValueError("Alignment expects [N,3] point pairs")
    if weight.shape!=(pred.shape[0],) or pred.shape[0]<2:
        raise ValueError("Need >=2 aligned weights")
    p=pred.detach().double(); y=target.detach().double(); w=weight.detach().double()
    n=p.shape[0]
    x=p[None].expand(n,-1,-1).clone(); yy=y[None].expand(n,-1,-1).clone()
    x[:,:,2]-=p[:,None,2]; yy[:,:,2]-=y[:,None,2]
    ww=w[None,:,None].expand(n,n,3)
    s=fit_scale_l1(x.reshape(n,-1),yy.reshape(n,-1),ww.reshape(n,-1))
    t=y[:,2]-s*p[:,2]
    err=s[:,None,None]*p[None]-y[None]
    err[:,:,2]+=t[:,None]
    objective=(err.abs()*ww).sum((1,2))
    best=objective.argmin()
    return s[best].float(),t[best].float()


def _subsample(ids,n):
    if ids.numel()<=n: return ids
    i=torch.linspace(0,ids.numel()-1,n,device=ids.device).round().long()
    return ids[i]


def shape_losses(points,gt_depth,K,image_hw,cfg):
    """Full-image affine alignment and multiscale local-shape objectives.

    Fits are detached (envelope-gradient treatment). GT values <=0/nonfinite are
    invalid; caller restricts valid metric depth range. No matching cache.
    """
    B,_,h,w=points.shape
    gt=F.interpolate(gt_depth.float(),(h,w),mode="nearest")
    valid=torch.isfinite(gt[:,0])&(gt[:,0]>0)
    gt=torch.nan_to_num(gt,nan=0.,posinf=0.,neginf=0.)
    rays=image_rays(K.float(),(h,w),image_hw)
    target=rays*gt
    zero=points.sum()*0.
    glob,local=zero,zero; ng=nl=0
    for b in range(B):
        ids=torch.where(valid[b].flatten())[0]
        if ids.numel()<4: continue
        p=points[b].permute(1,2,0).reshape(-1,3)
        y=target[b].permute(1,2,0).reshape(-1,3)
        sel=_subsample(ids,cfg.global_points)
        ww=y[sel,2].clamp_min(.05).reciprocal()
        s,t=fit_scale_zshift_l1(p[sel],y[sel],ww)
        error=s*p[sel]+torch.stack((t*0,t*0,t))-y[sel]
        glob=glob+(error.abs()*ww[:,None]).sum()/(3*ww.sum().clamp_min(1e-6)); ng+=1
        for size in cfg.patch_sizes:
            centers=_subsample(ids,cfg.patches_per_scale)
            for center in centers:
                cy,cx=int(center)//w,int(center)%w
                ya,yb=max(0,cy-size//2),min(h,cy+(size+1)//2)
                xa,xb=max(0,cx-size//2),min(w,cx+(size+1)//2)
                yy,xx=torch.meshgrid(torch.arange(ya,yb,device=p.device),torch.arange(xa,xb,device=p.device),indexing='ij')
                jj=(yy*w+xx).flatten(); jj=jj[valid[b].flatten()[jj]]
                if jj.numel()<4: continue
                jj=_subsample(jj,cfg.local_points)
                pp=p[jj]; tt=y[jj]
                # Per-patch translation-free coordinates (median centering).
                pp=pp-pp.detach().median(0).values; tt=tt-tt.median(0).values
                ss=fit_scale_l1(pp.detach().flatten(),tt.flatten(),torch.ones_like(tt).flatten())
                normalizer=tt.norm(dim=-1).mean().clamp_min(.01)
                local=local+(ss*pp-tt).abs().mean()/normalizer; nl+=1
    return {'global_shape':glob/max(ng,1),'local_shape':local/max(nl,1)}


def ray_consistency_loss(canonical,rays):
    z=canonical[:,2:3]
    reproj=(canonical[:,:2]/z.clamp_min(.01)-rays[:,:2]).abs().clamp_max(10).mean()
    positivity=F.relu(.01-z).mean()
    return reproj+positivity


def interval_score(mean,halfwidth,target,valid,coverage=.9):
    """Proper central-interval score; geometry adaptation, not RayRoPE's loss.

    Mean is detached, so only halfwidth is trained. A concentrated interval is
    NOT asserted to be calibrated; coverage/width must be logged on validation.
    """
    if not 0<coverage<1: raise ValueError("Invalid interval coverage")
    mu=mean.detach(); y=torch.nan_to_num(target,nan=0.,posinf=0.,neginf=0.)
    score=2*halfwidth+2/(1-coverage)*(F.relu(mu-halfwidth-y)+F.relu(y-mu-halfwidth))
    m=valid.to(score.dtype)
    return (score*m).sum()/m.sum().clamp_min(1)
