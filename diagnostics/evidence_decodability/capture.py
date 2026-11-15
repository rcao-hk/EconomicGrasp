"""Non-invasive hooks on the existing frozen GVAR volume reader.

The reader's action ordering is Q * A * D. Hook results are streamed as
float16 CPU chunks; the forward's official score/geometry outputs are unchanged.
Predicted-depth relative geometry is recorded BEFORE label lookup.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch

from rayrope_gvar import reader as reader_lib


class GVARCapture:
    def __init__(self, rt, *, capture_geometry=True):
        module = rt.model.kview_grasp_module.decoder
        if not hasattr(module,"reader") or module.reader is None:
            raise ValueError("Require checkpoint with actual GVAR reader")
        self.reader=module.reader
        if getattr(self.reader.cfg,"variant",None)!="volume_rel":
            raise ValueError("Only frozen GVAR volume_rel evidence is comparable")
        if int(self.reader.cfg.reader_dim)!=64:
            raise ValueError("Expected 64D attention role tokens")
        self.capture_geometry=bool(capture_geometry)
        self.pre=None
        self.post=None
        self.role_chunks=[]
        self.mass_chunks=[]
        self.valid_chunks=[]
        self.geom_chunks=[]
        self._handles=[]
        self._old_geom=None
        self.context=None
        self._entered=False

    def _pre_hook(self,module,args):
        if self.pre is not None:
            raise RuntimeError("GVAR reader called more than once in one snapshot")
        if len(args)<2 or not isinstance(args[1],dict):
            raise RuntimeError("GVAR reader context missing")
        data=args[0]
        if data.ndim!=5 or data.shape[0]!=1:
            raise RuntimeError("Expected single frame [1,Q,A,D,H] GVAR actions")
        self.pre=data.detach().to(dtype=torch.float16,device="cpu").numpy()
        self.context=args[1]

    def _post_hook(self,module,args,output):
        if self.post is not None or output.ndim!=5:
            raise RuntimeError("Malformed GVAR post-reader output")
        self.post=output.detach().to(dtype=torch.float16,device="cpu").numpy()

    def _attn_hook(self,module,args,kwargs,output):
        if not isinstance(output,tuple) or len(output)<2:
            raise RuntimeError("Reader MHA weights were not returned")
        tokens=args[1].detach().float()
        weights=output[1]
        padding=kwargs.get("key_padding_mask")
        if weights is None or padding is None:
            raise RuntimeError("Require attention weights and valid-token padding")
        w=weights.detach().float()
        if w.ndim==3 and w.shape[1]==1:
            w=w[:,0,:]
        elif w.ndim==4 and w.shape[-2]==1:
            w=w.mean(dim=1)[:,0,:]
        else:
            raise RuntimeError(f"Unexpected GVAR role attention weights {tuple(w.shape)}")
        n=len(tokens)
        if tokens.shape!=(n,36,64) or w.shape!=(n,36) or padding.shape!=(n,36):
            raise RuntimeError("GVAR role evidence tensors not aligned")
        w=w.reshape(n,6,6)
        tok=tokens.reshape(n,6,6,64)
        mass=w.sum(-1)
        pooled=(w.unsqueeze(-1)*tok).sum(dim=2)/mass.clamp_min(1e-9).unsqueeze(-1)
        valid=(~padding.bool()).reshape(n,6,6).float().mean(dim=2)
        self.role_chunks.append(pooled.cpu().half().numpy())
        self.mass_chunks.append(mass.cpu().half().numpy())
        self.valid_chunks.append(valid.cpu().half().numpy())

    def _geometry(self,*args,**kwargs):
        raw=self._old_geom(*args,**kwargs)
        if raw.ndim!=3 or raw.shape[1:]!=(36,6):
            raise RuntimeError("Unexpected predicted-depth raw volume geometry")
        n=len(raw)
        mean=raw.detach().float().reshape(n,6,6,6).mean(dim=2)
        self.geom_chunks.append(mean.cpu().half().numpy())
        return raw

    def __enter__(self):
        if self._entered:raise RuntimeError("Nested evidence capture is forbidden")
        self._entered=True
        self._handles.append(self.reader.register_forward_pre_hook(self._pre_hook))
        self._handles.append(self.reader.register_forward_hook(self._post_hook))
        self._handles.append(self.reader.attn.register_forward_hook(
            self._attn_hook, with_kwargs=True))
        if self.capture_geometry:
            self._old_geom=reader_lib.action_relative_geometry
            reader_lib.action_relative_geometry=self._geometry
        return self

    def __exit__(self,*_):
        if self._old_geom is not None:
            reader_lib.action_relative_geometry=self._old_geom
        for h in reversed(self._handles):h.remove()
        self._handles.clear()
        self._entered=False
        return False

    def collect(self,snap):
        if self._entered or self.pre is None or self.post is None:
            raise RuntimeError("Read capture only after complete forward/unhook")
        pre=self.pre
        post=self.post
        if pre.shape!=post.shape or pre.shape[:4]!=(1,1024,12,4):
            raise RuntimeError(f"Frozen GVAR reader lattice changed: {pre.shape}/{post.shape}")
        length=1024*12*4
        role=np.concatenate(self.role_chunks,axis=0)
        mass=np.concatenate(self.mass_chunks,axis=0)
        valid=np.concatenate(self.valid_chunks,axis=0)
        if role.shape!=(length,6,64) or mass.shape!=(length,6) or valid.shape!=(length,6):
            raise RuntimeError("Reader MHA chunk order/count disagrees with Q*A*D")
        if len(self.geom_chunks):
            geom=np.concatenate(self.geom_chunks,axis=0)
            if geom.shape!=(length,6,6):
                raise RuntimeError("Predicted-depth geometry chunk count differs")
        else:
            if self.capture_geometry:
                raise RuntimeError("GVAR did not execute relative geometry")
            geom=np.zeros((length,6,6),dtype=np.float16)
        angle=np.asarray(snap["angles"],dtype=np.int64)
        if angle.shape!=(1024,) or np.any((angle<0)|(angle>=12)):
            raise RuntimeError("Native angle identity differs from GVAR forward")
        return dict(hidden_dim=int(pre.shape[-1]),pre=pre.reshape(length,-1),
                    post=post.reshape(length,-1),role=role,mass=mass,
                    valid=valid,geom=geom,angle=angle)

    def chosen(self,snap,query_ids,depth_ids):
        tensors=self.collect(snap)
        q=np.asarray(query_ids,dtype=np.int64)
        d=np.asarray(depth_ids,dtype=np.int64)
        if q.shape!=d.shape or np.any((q<0)|(q>=1024)|(d<0)|(d>=4)):
            raise ValueError("Invalid queried action identities")
        indices=(q*12+tensors["angle"][q])*4+d
        out=dict(pre_reader=tensors["pre"][indices].astype(np.float32),
                 post_reader=tensors["post"][indices].astype(np.float32),
                 reader_delta=(tensors["post"][indices].astype(np.float32)-
                               tensors["pre"][indices].astype(np.float32)),
                 role_tokens=tensors["role"][indices].astype(np.float32),
                 role_mass=tensors["mass"][indices].astype(np.float32),
                 role_valid=tensors["valid"][indices].astype(np.float32),
                 pred_geometry=tensors["geom"][indices].astype(np.float32))
        return out


def frozen_snapshot(rt,row,seed,query_ids=None,depth_ids=None):
    """Use the *existing* strict RARE Runtime snapshot and opt-in evidence hooks."""
    with GVARCapture(rt) as capture:
        snap=rt.snapshot(row,seed)
    if query_ids is None:
        query_ids=np.repeat(np.arange(1024,dtype=np.int64),4)
        depth_ids=np.tile(np.arange(4,dtype=np.int64),1024)
    evidence=capture.chosen(snap,query_ids,depth_ids)
    return snap,evidence


def privileged_geometry_from_depth(rt,capture,snap,query_ids,depth_ids,
                                   depth_reference,unit="m"):
    """Explicit offline CAD/GT geometry only for train-scene oracle *diagnosis*.

    It does not change the neural network, reader state or action geometry.
    The caller must supply a pre-registered, pixel-aligned H×W depth map
    with declared physical units. NEVER use this for official test inference.
    """
    if unit not in ("m","mm"):
        raise ValueError("Reference depth units must be explicitly m or mm")
    ctx=capture.context
    if ctx is None:
        raise ValueError("GVAR context not captured")
    ref=np.asarray(depth_reference,dtype=np.float32)
    H,W=ctx["image_hw"]
    if ref.shape!=(H,W) or not np.isfinite(ref).all() or np.any(ref<0):
        raise ValueError(f"Reference depth must be aligned finite [{H},{W}]")
    if unit=="mm":ref=ref/1000.
    with torch.inference_mode():
        K=ctx["K"].detach().float()
        if K.shape!=(1,3,3):
            raise ValueError("Expected single camera intrinsic")
        centers=ctx["centers"].detach().reshape(1,1024,12,3)[0]
        rot=ctx["rotations"].detach().reshape(1,1024,12,3,3)[0]
        q=torch.as_tensor(query_ids,device=centers.device,dtype=torch.long)
        d=torch.as_tensor(depth_ids,device=centers.device,dtype=torch.long)
        a=torch.as_tensor(snap["angles"][q.cpu().numpy()],device=centers.device,dtype=torch.long)
        center=centers[q,a]
        rotation=rot[q,a]
        depth_ids=(d.float()+1.)*.01
        probes,_=reader_lib.canonical_probes(depth_ids,capture.reader.cfg)
        xyz,uv,grid,valid=reader_lib.project_probes(center,rotation,probes,K[0],(H,W))
        depth=torch.as_tensor(ref,device=center.device,dtype=torch.float32)[None,None]
        geom=reader_lib.action_relative_geometry(
            depth,grid,uv,xyz,center,rotation,K[0],valid,
            capture.reader.cfg.geometry_scale_m)
        return geom.reshape(-1,6,6,6).mean(dim=2).cpu().numpy().astype(np.float32)
