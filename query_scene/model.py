"""Same local state, three context sources; no graph, leader rule or action update."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from .core import VARIANTS


@contextmanager
def head_precision():
    # Frozen provider may require historical TF32 for native parity. New heads
    # use the same FP32 setting in training and inference, then restore it.
    mat=torch.backends.cuda.matmul.allow_tf32; conv=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=mat;torch.backends.cudnn.allow_tf32=conv


@dataclass(frozen=True)
class Config:
    feature_dim:int
    scene_dim:int
    variant:str='scene'
    hidden:int=128
    heads:int=4
    layers:int=2
    seed:int=0
    residual_scale:float=.2
    def __post_init__(self):
        if self.variant not in VARIANTS or min(self.feature_dim,self.scene_dim,self.hidden,self.heads,self.layers)<1 or self.hidden%self.heads:
            raise ValueError('Bad model dimensions/variant')
        if not 0<self.residual_scale<=1:raise ValueError('Common preregistered residual scale in (0,1]')


class Access(nn.Module):
    def __init__(self,c):
        super().__init__();h=c.hidden;self.heads=c.heads
        self.q=nn.Linear(h,h);self.k=nn.Linear(h,h);self.v=nn.Linear(h,h);self.out=nn.Linear(h,h)
    def forward(self,q,m,valid):
        b,n,h=q.shape;p=m.shape[1];heads=self.heads
        any_valid=valid.any(-1);safe=valid.clone();safe[~any_valid,0]=True
        m=torch.where(valid[...,None],m,0.)
        qt=self.q(q).reshape(b,n,heads,h//heads).transpose(1,2)
        kt=self.k(m).reshape(b,p,heads,h//heads).transpose(1,2)
        vt=self.v(m).reshape(b,p,heads,h//heads).transpose(1,2)
        y=F.scaled_dot_product_attention(qt,kt,vt,attn_mask=safe[:,None,None,:],dropout_p=0.)
        y=self.out(y.transpose(1,2).reshape(b,n,h))
        return torch.where(any_valid[:,None,None],y,0.)


class ContextScorer(nn.Module):
    def __init__(self,c):
        super().__init__();self.cfg=c;h=c.hidden
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed)
            self.local=nn.Sequential(nn.LayerNorm(c.feature_dim),nn.Linear(c.feature_dim,h))
            self.geometry=nn.Sequential(nn.Linear(15,h),nn.GELU(),nn.Linear(h,h))
        context_dim=c.feature_dim+15 if c.variant=='candidate' else c.scene_dim+8
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed+101)
            if c.variant!='local':
                self.context=nn.Sequential(nn.LayerNorm(context_dim),nn.Linear(context_dim,h))
            else:
                self.context=None
        self.mix=nn.ModuleList();self.norm=nn.ModuleList();self.ffn=nn.ModuleList()
        # Replace all context/access parameters with active per-action MLP capacity.
        context_cost=2*context_dim+context_dim*h+h
        target_dim=max(c.feature_dim+15,c.scene_dim+8)
        target_cost=2*target_dim+target_dim*h+h
        per_layer_target=4*h*h+4*h+target_cost/c.layers
        ff_mid=2*h if c.variant=='local' else 2*h+round((target_cost-context_cost)/(c.layers*(2*h+1)))
        local_mid=max(1,round((per_layer_target-h)/(2*h+1)))
        for i in range(c.layers):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(c.seed+1009*(i+1))
                self.norm.append(nn.LayerNorm(h))
                self.mix.append(nn.Sequential(nn.Linear(h,local_mid),nn.GELU(),nn.Linear(local_mid,h)) if c.variant=='local' else Access(c))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(c.seed+1009*(i+1)+1)
                self.ffn.append(nn.Sequential(nn.LayerNorm(h),nn.Linear(h,ff_mid),nn.GELU(),nn.Linear(ff_mid,h)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed+65537)
            self.final=nn.LayerNorm(h);self.score=nn.Linear(h,1)
            nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias)
    @staticmethod
    def action_geometry(g):
        x=g[...,1:16].clone();x[...,:3]/=.1;return x
    def forward(self,features,grasps,base_logit,scene_features,scene_geometry,scene_valid,action_valid=None):
        with head_precision():
            return self._forward(features,grasps,base_logit,scene_features,scene_geometry,scene_valid,action_valid)
    def _forward(self,features,grasps,base_logit,scene_features,scene_geometry,scene_valid,action_valid=None):
        b,n,f=features.shape
        if f!=self.cfg.feature_dim or grasps.shape!=(b,n,17) or base_logit.shape!=(b,n):raise ValueError('Action shape mismatch')
        if action_valid is None:action_valid=torch.ones_like(base_logit,dtype=torch.bool)
        if action_valid.dtype!=torch.bool or action_valid.shape!=(b,n):raise ValueError('Action mask')
        g=torch.where(action_valid[...,None],grasps,0.)
        features=torch.where(action_valid[...,None],features,0.)
        geo=self.action_geometry(g);x=self.local(features)+self.geometry(geo)
        if self.cfg.variant=='candidate':
            raw=torch.cat((features,geo),-1);mask=action_valid
            memory=self.context(raw)
        elif self.cfg.variant=='scene':
            if scene_features.shape[:2]!=scene_geometry.shape[:2] or scene_features.shape[-1]!=self.cfg.scene_dim or scene_geometry.shape[-1]!=8:
                raise ValueError('Scene feature/geometry dimensions')
            if scene_valid.dtype!=torch.bool or scene_valid.shape!=scene_features.shape[:2]:raise ValueError('Scene mask')
            raw=torch.cat((scene_features,scene_geometry),-1)
            raw=torch.where(scene_valid[...,None],raw,0.)
            memory=self.context(raw);mask=scene_valid
        for norm,mix,ffn in zip(self.norm,self.mix,self.ffn):
            d=mix(norm(x)) if self.cfg.variant=='local' else mix(norm(x),memory,mask)
            x=x+d;x=x+ffn(x);x=torch.where(action_valid[...,None],x,0.)
        delta=self.score(self.final(x)).squeeze(-1)
        return base_logit+self.cfg.residual_scale*torch.where(action_valid,delta,0.)
    def describe(self):
        return {**asdict(self.cfg),'parameters':sum(p.numel() for p in self.parameters()),
                'fixed_physical_actions':True,'graph':False,'leader_prior':False,
                'context_states':'static source tokens; iterative query reads','sensor_mask_in_model':False}


def intervene(x,mode,seed=0,donor=None):
    """Validation-only controls; joint permutation is NOT a spatial mismatch."""
    out=dict(x);gen=torch.Generator(device=x['features'].device).manual_seed(seed)
    if mode=='aligned':return out
    if mode in ('scene_joint_permute','scene_appearance_shuffle'):
        ids=torch.randperm(x['scene_features'].shape[1],device=x['features'].device,generator=gen)
        out['scene_features']=x['scene_features'][:,ids]
        if mode=='scene_joint_permute':
            for k in ('scene_geometry','scene_valid'):out[k]=x[k][:,ids]
    elif mode=='scene_swap':
        if donor is None:raise ValueError('Need predeclared other-scene same-domain donor')
        for k in ('scene_features','scene_geometry','scene_valid'):out[k]=donor[k]
    elif mode=='scene_mean':
        mask=x['scene_valid'][...,None];den=mask.sum(1,keepdim=True).clamp_min(1)
        out['scene_features']=(torch.where(mask,x['scene_features'],0.).sum(1,keepdim=True)/den).expand_as(x['scene_features'])
    elif mode=='duplicate_candidates':
        ids=torch.arange(min(64,x['features'].shape[1]),device=x['features'].device)
        for k in ('features','grasps','base_logit','action_valid'):out[k]=torch.cat((x[k],x[k][:,ids]),1)
    else:raise ValueError(mode)
    return out
