"""Grasp-frame RayRoPE local cross-attention.

Adaptation of the query-frame projective endpoint + 3D ray-origin formulation.
The source paper is multi-view; here each center/view/angle defines a virtual
query camera. Uniform coordinate-interval RoPE moments use sinc analytically.
This is NOT E[softmax(attention)] and NOT Gaussian positional encoding.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .geometry import sample_map, rays_at_pixels


def uniform_rope_moments(lower,upper,frequencies):
    if lower.shape != upper.shape or lower.shape[-1]!=6:
        raise ValueError("Expected matching [...,6] ray-coordinate bounds")
    lo=torch.minimum(lower,upper); hi=torch.maximum(lower,upper)
    mid=.5*(lo+hi); half=.5*(hi-lo)
    phase=mid[...,None]*frequencies
    # torch.sinc(t) = sin(pi*t)/(pi*t); analytic integral has argument omega*h.
    attenuation=torch.sinc(half[...,None]*frequencies/math.pi)
    return torch.cos(phase)*attenuation,torch.sin(phase)*attenuation


def rotate_channels(x,cosine,sine,inverse=False):
    """x [...,heads,12*F]; moments [...,6,F]; conjugate, NEVER inverse damping."""
    if x.shape[-1] != 12*cosine.shape[-1]:
        raise ValueError("RoPE/head width mismatch")
    xx=x.reshape(*x.shape[:-1],6,cosine.shape[-1],2)
    c=cosine.unsqueeze(-3); s=sine.unsqueeze(-3)
    if inverse: s=-s
    a,b=xx.unbind(-1)
    return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-3)


def ray_coordinates_in_grasp_frame(center,rotation,low_points,high_points,
                                    standoff=.15,length_unit=.1,camera_origin=None):
    """All XYZ are expressed in the same physical frame.

    Rotation columns are [approach, closing, vertical]; virtual query camera
    columns are [closing, vertical, approach]. Returned key coordinates are
    [origin_XYZ/unit, projected_x, projected_y, unit/projected_Z].
    The source-ray endpoints are projected first. The uniform marginal interval
    in projected coordinates is an approximation, as in RayRoPE's construction.
    """
    if center.shape[-1]!=3 or rotation.shape!=(*center.shape[:-1],3,3):
        raise ValueError("Bad query frame")
    axes=rotation[..., [1,2,0]]
    origin=center-standoff*rotation[...,0]
    cam=torch.zeros_like(center) if camera_origin is None else camera_origin.expand_as(center)
    camera_local=torch.einsum('bmc,bmcd->bmd',cam-origin,axes)/length_unit
    def project(p):
        local=torch.einsum('bmlc,bmcd->bmld',p-origin[:,:,None],axes)
        z=local[...,2].clamp_min(.01)
        proj=torch.stack((local[...,0]/z,local[...,1]/z,length_unit/z),-1)
        # Numerical safeguard near the virtual camera plane, not occupancy.
        return proj.clamp(-100,100),local[...,2]
    lo,zlo=project(low_points); hi,zhi=project(high_points)
    camera_local=camera_local[:,:,None].expand_as(lo)
    lower=torch.cat((camera_local,torch.minimum(lo,hi)),-1)
    upper=torch.cat((camera_local,torch.maximum(lo,hi)),-1)
    query=center.new_zeros(*center.shape[:-1],6)
    query[...,5]=length_unit/standoff
    # Identical validity for point/fixed/expected: do not make uncertainty itself
    # a hard token-deletion rule. Near-plane interval crossings are attenuated.
    valid=(.5*(zlo+zhi)>.01)&torch.isfinite(lower).all(-1)&torch.isfinite(upper).all(-1)
    return query,lower,upper,valid


class GraspRayRoPEGrouping(nn.Module):
    """Drop-in CVA `group`: [B,C,Q*A] -> [B,decoder_channels,Q*A].

    This groups center/view/angle queries BEFORE insertion-depth and width are
    decoded. It is not the earlier 27-support-point action residual module.
    Fixed pixel support avoids an uncertain depth setting the support radius.
    """
    def __init__(self,cfg,feature_dim=128,output_dim=256):
        super().__init__(); self.cfg=cfg; self.feature_dim=feature_dim
        self.geometry_context=None
        # Inference-only state; no parameters/buffers/state_dict changes.
        self.diagnostic_mode=None
        self.diagnostic_bias_m=0.0
        self.last_diagnostic_stats={}
        d=cfg.ray_dim; self.head_dim=d//cfg.ray_heads
        self.query=nn.Sequential(nn.Linear(feature_dim,d),nn.LayerNorm(d),nn.GELU())
        # Camera/shape scalars only exist in this branch; geometry is detached.
        extra=3 if cfg.use_shape_tokens else 0
        self.tokens=nn.Sequential(nn.Linear(feature_dim+extra,d),nn.LayerNorm(d),nn.GELU())
        self.q_proj=nn.Linear(d,d); self.k_proj=nn.Linear(d,d); self.v_proj=nn.Linear(d,d)
        self.out=nn.Linear(d,output_dim); self.skip=nn.Linear(feature_dim,output_dim)
        self.norm=nn.LayerNorm(output_dim)
        self.ff=nn.Sequential(nn.Linear(output_dim,output_dim*2),nn.GELU(),nn.Linear(output_dim*2,output_dim))
        freq=math.pi*cfg.ray_freq_base**torch.arange(self.head_dim//12,dtype=torch.float32)
        self.register_buffer('frequencies',freq,persistent=True)
        steps=torch.linspace(-1,1,cfg.ray_grid)
        yy,xx=torch.meshgrid(steps,steps,indexing='ij')
        self.register_buffer('offsets',torch.stack((xx,yy),-1).reshape(-1,2)*cfg.ray_radius_px,persistent=True)

    def set_diagnostic(self, mode=None, bias_m=0.0):
        """Set an opt-in P1 geometry intervention on the *reader only*.

        Modes: None (disabled), native, coords_only, mask_only, both.
        Seed centers, feature map, view/angle frame, and native decoded
        actions stay fixed. Never call during training. This has no learned
        parameters and does not alter checkpoint compatibility.
        """
        if mode not in (None, "native", "coords_only", "mask_only", "both"):
            raise ValueError(f"Unsupported RayRoPE diagnostic mode {mode!r}")
        if mode is not None and self.training:
            raise RuntimeError("RayRoPE diagnostic is inference-only")
        if mode is not None and not math.isfinite(float(bias_m)):
            raise ValueError("Non-finite reader depth bias")
        self.diagnostic_mode=mode
        self.diagnostic_bias_m=float(bias_m)
        self.last_diagnostic_stats={}

    def _chunk(self,seed,indices,center,R,features,depth,K,sigma,shape,image_hw):
        B,M,C=seed.shape; H,W=image_hw; L=len(self.offsets)
        uv=torch.stack(((indices%W).float(),torch.div(indices,W,rounding_mode='floor').float()),-1)
        pix=uv[:,:,None]+self.offsets[None,None]
        valid=(pix[...,0]>=0)&(pix[...,0]<=W-1)&(pix[...,1]>=0)&(pix[...,1]<=H-1)
        visual=sample_map(features.float(),pix,image_hw)
        mu=sample_map(depth.float(),pix,image_hw)[...,0]
        hw=sample_map(sigma.float(),pix,image_hw)[...,0]
        rays=rays_at_pixels(pix,K.float())
        if self.cfg.ray_encoding!='expected': hw=torch.zeros_like(hw)
        lo=rays*(mu-hw).clamp_min(.001)[...,None]
        hi=rays*(mu+hw).clamp_min(.001)[...,None]
        query,lower,upper,infront=ray_coordinates_in_grasp_frame(center,R,lo,hi,
                                        self.cfg.virtual_standoff,self.cfg.ray_length_unit)
        valid=valid&infront&torch.isfinite(mu)&(mu>0)
        diag_stats=None
        if self.diagnostic_mode is not None:
            # Native and perturbed data share the same RGB samples, seed,
            # physical query frame, and *fixed* pixel coordinates. Only the
            # RoPE-coordinate stream and/or token-validity stream changes.
            mode=self.diagnostic_mode
            corrected=torch.where(
                torch.isfinite(depth)&(depth>0),
                (depth.float()+self.diagnostic_bias_m).clamp_min(1e-4),
                depth.float(),
            )
            mu_p=sample_map(corrected,pix,image_hw)[...,0]
            lo_p=rays*(mu_p-hw).clamp_min(.001)[...,None]
            hi_p=rays*(mu_p+hw).clamp_min(.001)[...,None]
            query_p,lower_p,upper_p,infront_p=ray_coordinates_in_grasp_frame(
                center,R,lo_p,hi_p,
                self.cfg.virtual_standoff,self.cfg.ray_length_unit)
            valid_p=(pix[...,0]>=0)&(pix[...,0]<=W-1)&(pix[...,1]>=0)&(pix[...,1]<=H-1)
            valid_p=valid_p&infront_p&torch.isfinite(mu_p)&(mu_p>0)
            flips=(valid_p!=valid)
            diag_stats=torch.stack((
                torch.as_tensor(valid.numel(),device=depth.device,dtype=torch.float64),
                valid.to(torch.float64).sum(),
                valid_p.to(torch.float64).sum(),
                flips.to(torch.float64).sum(),
                flips.any(-1).to(torch.float64).sum(),
                torch.as_tensor(valid.shape[0]*valid.shape[1],device=depth.device,dtype=torch.float64),
            ))
            if mode in ("coords_only", "both"):
                query,lower,upper=query_p,lower_p,upper_p
            if mode in ("mask_only", "both"):
                valid=valid_p
        if self.cfg.use_shape_tokens:
            p=sample_map(shape.float(),pix,image_hw)
            p0=sample_map(shape.float(),uv,image_hw)
            rel=p-p0[:,:,None]
            rel=torch.einsum('bmlc,bmcd->bmld',rel,R)
            # Scale-free local descriptor; action itself remains metrically grounded.
            norm=rel.norm(dim=-1).mean(-1,keepdim=True).clamp_min(.01)
            visual=torch.cat((visual,torch.tanh(rel/norm[...,None])), -1)
        token=self.tokens(visual); query_feat=self.query(seed)
        q=self.q_proj(query_feat).reshape(B,M,self.cfg.ray_heads,self.head_dim)
        k=self.k_proj(token).reshape(B,M,L,self.cfg.ray_heads,self.head_dim)
        v=self.v_proj(token).reshape_as(k)
        if self.cfg.ray_encoding!='none':
            qc,qs=uniform_rope_moments(query,query,self.frequencies)
            kc,ks=uniform_rope_moments(lower,upper,self.frequencies)
            q=rotate_channels(q,qc,qs,inverse=True)
            k=rotate_channels(k,kc,ks,inverse=True)
            if self.cfg.ray_apply_vo: v=rotate_channels(v,kc,ks,inverse=True)
        score=torch.einsum('bmhd,bmlhd->bmhl',q,k)/math.sqrt(self.head_dim)
        score=score.masked_fill(~valid[:,:,None],-1e4)
        attn=score.softmax(-1)*valid[:,:,None].to(score.dtype)
        attn=attn/attn.sum(-1,keepdim=True).clamp_min(1e-8)
        ctx=torch.einsum('bmhl,bmlhd->bmhd',attn,v)
        if self.cfg.ray_encoding!='none' and self.cfg.ray_apply_vo:
            ctx=rotate_channels(ctx,qc,qs)  # query is deterministic: transpose=exact inverse
        out=self.norm(self.out(ctx.flatten(-2))+self.skip(seed))
        result=out+self.ff(out)
        return (result,diag_stats) if diag_stats is not None else result

    def forward(self,seed_features,token_sel_idx,seed_xyz,top_view_rot,
                feat_map,depth_map,camera_K,end_points=None,**kwargs):
        B,C,M=seed_features.shape
        if C!=self.feature_dim or token_sel_idx.shape!=(B,M) or seed_xyz.shape!=(B,M,3):
            raise ValueError("CVA grouping input contract changed")
        if self.geometry_context is None:
            raise RuntimeError("No per-forward geometry context: use EconomicGraspMoGeRayRoPE")
        context=self.geometry_context
        depth=depth_map.detach(); sigma=context['sigma'].detach()
        shape=context.get('shape')
        if shape is None:
            shape=depth.new_zeros(B,3,*depth.shape[-2:])
        else: shape=shape.detach()
        outputs=[]; image_hw=depth.shape[-2:]
        diag_total=None
        self.last_diagnostic_stats={}
        for start in range(0,M,self.cfg.group_chunk):
            sl=slice(start,start+self.cfg.group_chunk)
            inputs=(seed_features[:,:,sl].transpose(1,2).contiguous(),token_sel_idx[:,sl],
                    seed_xyz[:,sl].detach().float(),top_view_rot[:,sl].detach().float(),
                    feat_map,depth,camera_K.detach(),sigma,shape)
            if self.training and self.cfg.checkpoint_chunks and torch.is_grad_enabled():
                def run(*args): return self._chunk(*args,image_hw)
                value=checkpoint(run,*inputs,use_reentrant=False)
            else: value=self._chunk(*inputs,image_hw)
            if self.diagnostic_mode is not None:
                if not isinstance(value,tuple) or len(value)!=2:
                    raise RuntimeError("Missing diagnostic geometry statistics")
                value,stats=value
                diag_total=stats if diag_total is None else diag_total+stats
            outputs.append(value)
        if diag_total is not None:
            n,valid0,valid1,flips,queries_changed,n_queries=diag_total.detach().cpu().tolist()
            self.last_diagnostic_stats={
                "mode":self.diagnostic_mode,
                "depth_bias_m":self.diagnostic_bias_m,
                "token_count":int(round(n)),
                "valid_native":int(round(valid0)),
                "valid_perturbed":int(round(valid1)),
                "mask_flip_count":int(round(flips)),
                "queries_with_mask_flip":int(round(queries_changed)),
                "query_count":int(round(n_queries)),
            }
        return torch.cat(outputs,1).transpose(1,2).contiguous()
