"""Separate anchor and reader depth in the ACTUAL Stage-1 forward.

No change to repository models or parameters. One nominal RGB forward caches
frozen depth/backbone/proposal outputs. Replays rerun spatial enhancement,
image-FPS/backprojection, ViewNet, CVA grouping, CDF/width and decoding.
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
from rep_a_common import load_torch
from rep_followup_common import release_memory

MODES = ('reader_only', 'anchor_only', 'joint')


def routed_depths(nominal, perturbed, mode):
    if mode not in MODES:
        raise ValueError(mode)
    anchor = perturbed if mode in ('anchor_only','joint') else nominal
    reader = perturbed if mode in ('reader_only','joint') else nominal
    return anchor, reader


@contextmanager
def replace_method(obj, name, value):
    # Restore inherited methods without leaving bound-method objects in module state.
    present = name in obj.__dict__; old = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if present: setattr(obj, name, old)
        else: delattr(obj, name)


class FullpathReplay:
    def __init__(self, model, graspness_threshold):
        self.model = model
        self.threshold = float(graspness_threshold)
        if model.training or model.is_training or model.geometry_depth_source != 'pred' or model.use_obs_depth:
            raise ValueError('FullpathReplay requires an eval-mode RGB-only predicted-depth Stage-1')
        if model.seed_selection_mode != 'image_fps':
            raise ValueError('This protocol requires the original deterministic image-FPS')
        self.depth_output = self.proposal_output = None
        self.nominal_depth = None

    @torch.no_grad()
    def capture(self, batch):
        def depth_hook(_m, _i, out): self.depth_output = out
        def proposal_hook(_m, _i, out): self.proposal_output = out
        hd = self.model.depth_net.register_forward_hook(depth_hook)
        hp = self.model.proposal_head.register_forward_hook(proposal_hook)
        try:
            ep = self.model(dict(batch))
        finally:
            hd.remove(); hp.remove()
        if not isinstance(self.depth_output, (tuple,list)) or len(self.depth_output)!=6:
            raise RuntimeError('Stage-1 depth output contract changed; expected 6 outputs')
        if not isinstance(self.proposal_output, (tuple,list)) or len(self.proposal_output)!=2:
            raise RuntimeError('Stage-1 proposal output contract changed')
        self.nominal_depth = ep['depth_map_used_for_geometry'].detach().clone()
        if not torch.equal(self.nominal_depth, self.depth_output[0]):
            raise RuntimeError('Active geometry differs from RGB depth output; unsupported refinement path')
        return ep

    @property
    def image_feature(self):
        return self.proposal_output[0][0]

    @torch.no_grad()
    def run(self, batch, anchor_depth, reader_depth):
        if self.depth_output is None: raise RuntimeError('capture() must run first')
        if anchor_depth.shape!=self.nominal_depth.shape or reader_depth.shape!=self.nominal_depth.shape:
            raise ValueError('Depth intervention must retain [1,1,H,W] shape')
        original_selector = self.model._select_graspable_seed_queries
        record = {}
        def selector(**kw):
            # Depth-validity and image-FPS membership belong to the ANCHOR path.
            # Image features passed here already reflect READER depth.
            ep = kw['end_points']; b = anchor_depth.shape[0]
            valid = torch.isfinite(anchor_depth) & (anchor_depth>self.model.min_depth) & (anchor_depth<self.model.max_depth)
            valid = valid.reshape(b,-1)
            if 'token_valid_mask' in ep: valid = valid & ep['token_valid_mask'].bool()
            mask = valid & (ep['objectness_score'].argmax(1)==1) & (kw['grasp_score']>self.threshold)
            kw.update(depth_map=anchor_depth, valid_tok=valid, graspable_mask=mask)
            record['anchor_valid_fraction'] = float(valid.float().mean())
            out = original_selector(**kw)
            record['seed_xyz'] = out[1].detach()
            record['seed_tokens'] = out[2].detach()
            return out
        def depth_forward(*_a, **_kw):
            return (reader_depth, *self.depth_output[1:])
        def proposal_forward(*_a, **_kw):
            return tuple(x.clone() for x in self.proposal_output)
        with replace_method(self.model.depth_net,'forward',depth_forward), \
             replace_method(self.model.proposal_head,'forward',proposal_forward), \
             replace_method(self.model,'_select_graspable_seed_queries',selector):
            ep = self.model(dict(batch))
        if not torch.equal(ep['depth_map_used_for_geometry'],reader_depth):
            raise RuntimeError('Reader intervention did not reach active geometry')
        if not record: raise RuntimeError('Anchor selector was not executed')
        ep['fullpath_anchor_valid_fraction'] = record['anchor_valid_fraction']
        # Check actual anchor placement against the ORIGINAL selector clamp policy.
        ids = record['seed_tokens']
        expected_z = torch.gather(anchor_depth[:,0].flatten(1),1,ids)
        expected_z = torch.nan_to_num(expected_z,nan=self.model.min_depth,
                                     posinf=self.model.max_depth,neginf=self.model.min_depth)
        expected_z = expected_z.clamp(self.model.min_depth,self.model.max_depth)
        err = (record['seed_xyz'][...,2]-expected_z).abs().max()
        if float(err)>1e-6: raise RuntimeError(f'Anchor backprojection mismatch: {float(err)}')
        return ep

    def clear(self):
        self.depth_output = self.proposal_output = self.nominal_depth = None


def load_stage1(checkpoint, device, pose_depth_mode='global_film'):
    # Caller parses CLI then clears sys.argv before importing utils.arguments.
    from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
    from utils.arguments import cfgs
    cfgs.use_top4_view_infer=False; cfgs.kview_mode='A1'; cfgs.kview_k=1
    cfgs.use_cdf=True; cfgs.use_obs_depth=False; cfgs.pose_depth_mode=pose_depth_mode
    ck=load_torch(checkpoint)
    if ck.get('geometry_depth_source','pred')!='pred' or ck.get('use_obs_depth',False):
        raise ValueError('Requires RGB-only Stage-1 checkpoint')
    if ck.get('pose_depth_mode',pose_depth_mode)!=pose_depth_mode:
        raise ValueError('Stage-1 POSE_DEPTH_MODE mismatch')
    kwargs={key:ck.get(key,default) for key,default in
        (('camera_pose_key','camera_pose_vec'),('camera_gravity_key','camera_gravity_vec'),
         ('pose_hidden_dim',64),('ray_gravity_hidden_dim',64),('ray_gravity_mid_dim',32))}
    model=economicgrasp_dpt_student(is_training=False,use_obs_depth=False,use_cdf=True,
        min_depth=.2,max_depth=1.,bin_num=256,pose_depth_mode=pose_depth_mode,vis_dir=None,**kwargs).to(device)
    state=ck['model_state_dict']
    if all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    result=model.load_state_dict(state,strict=False)
    incompatible=[k for k in result.missing_keys+result.unexpected_keys if not k.startswith('rgb_geometry_diagnostics.')]
    if incompatible: raise RuntimeError(f'Stage-1 checkpoint mismatch: {incompatible}')
    use_fuse=bool(ck.get('use_fuse_depth',False))
    del ck,state; release_memory()
    model.eval().requires_grad_(False)
    return model,FullpathReplay(model,cfgs.graspness_threshold),use_fuse


def parse_offsets(text):
    off=np.asarray([float(x) for x in text.split(',')],np.float32)
    if not np.isfinite(off).all() or len(np.unique(off))!=len(off) or np.count_nonzero(off==0)!=1:
        raise ValueError('Offsets must be finite, unique, and contain exactly one zero')
    return off


def expand_actions(native, offsets, min_depth=.2, max_depth=1.):
    """After Stage-1 has REGENERATED R,w,d, expand translations at fixed ray offsets."""
    g=np.asarray(native,np.float32)
    if g.ndim!=2 or g.shape[1]!=17 or not np.isfinite(g).all() or np.any(g[:,15]<=0):
        raise ValueError('Non-finite/invalid decoded native grasps')
    a=np.repeat(g[None],len(offsets),axis=0)
    z=g[None,:,15]+np.asarray(offsets)[:,None]/1000.
    a[...,13:16]=g[None,:,13:16]*(z/g[None,:,15])[...,None]
    valid=(z>=min_depth)&(z<=max_depth)
    zero=int(np.flatnonzero(np.asarray(offsets)==0)[0]); a[zero]=g
    if not valid[zero].all(): raise ValueError('Stage-1 emitted native centers outside geometry limits')
    return a,valid,zero


def case_key(case):
    return case.replace(':','_').replace('-','m').replace('+','p').replace('.','p')
