"""Read-only native GVAR capture through temporary hooks.

Never changes a production forward or state_dict. Full native Q*A*D execution
is retained. Evidence is captured by REPLAYING the SAME native reader at the
same action_chunk lengths; no shortened Conv1d/TF32 approximation.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import numpy as np
import torch
from torch import nn


class FrozenScalarSelector(nn.Module):
    """State-name compatible with existing RARE ResidualRanker (4673 params)."""
    def __init__(self, feature_dim=64, hidden=64, depths=4):
        super().__init__();self.depths=depths
        self.norm=nn.LayerNorm(feature_dim)
        self.net=nn.Sequential(nn.Linear(feature_dim+depths+1,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,features,base_logit,depth_ids):
        x=torch.cat((self.norm(features),torch.nn.functional.one_hot(depth_ids,self.depths).to(features.dtype),
                     base_logit[...,None]),-1)
        return base_logit+self.net(x).squeeze(-1)


@contextmanager
def capture_hooks(decoder):
    data={};handles=[]
    def save_input(name):
        def hook(_m, args):
            if name in data: raise RuntimeError(f'Unexpected repeated native {name} hook')
            data[name]=args[0].detach().clone()
        return hook
    def save_output(name):
        def hook(_m,_args,value):
            if name in data: raise RuntimeError(f'Unexpected repeated native {name} hook')
            data[name]=value.detach().clone()
        return hook
    def save_context(_m,args):
        if 'reader_queries' in data:raise RuntimeError('Multiple reader forward calls')
        data['reader_queries']=args[0].detach()
        data['context']=dict(args[1])
    try:
        handles.append(decoder.width_proj.register_forward_pre_hook(save_input('prebranch')))
        handles.append(decoder.score_dropout.register_forward_hook(save_output('score_feature')))
        handles.append(decoder.reader.register_forward_pre_hook(save_context))
        handles.append(decoder.reader.register_forward_hook(save_output('reader_output')))
        yield data
    finally:
        for handle in reversed(handles):handle.remove()


def selected_evidence(reader, queries, context, original_output, selected_flat,
                      true_valid_fn, tolerance=1e-6):
    """Capture K/V tokens for selected actions by full native reader replay.

    true_valid_fn(context, selected_flat, Q,A,D) supplies the ACTUAL projection
    mask. Native MHA temporarily unmasks probe0 for all-invalid actions; that
    numerical workaround must never be treated as real visual evidence.
    """
    if reader.training or getattr(reader,'counterfactual_mode','r0')!='r0':
        raise ValueError('Only frozen native r0 reader can be captured')
    B,Q,A,D,H=queries.shape
    if B!=1:raise ValueError('One physical RGB frame per frozen forward')
    ids=selected_flat.long()
    if ids.ndim!=1 or len(ids)!=Q or not torch.equal(ids,ids.sort().values):
        raise ValueError('Expected increasing one-action-per-query indices')
    tokens=[];pads=[];positions=[];cursor=0
    def tap(_m,args,kwargs):
        nonlocal cursor
        if len(args)<3:raise RuntimeError('Native reader attention call changed')
        _,keys,values=args[:3]
        if not torch.equal(keys,values):raise RuntimeError('Native reader K/V diverge')
        end=cursor+keys.shape[0]
        loc=torch.nonzero((ids>=cursor)&(ids<end),as_tuple=False).flatten()
        if len(loc):
            idx=ids[loc]-cursor
            tokens.append(keys[idx].detach().clone())
            pad=kwargs.get('key_padding_mask')
            if pad is None:raise RuntimeError('Native evidence padding mask missing')
            pads.append(pad[idx].detach().clone());positions.append(loc)
        cursor=end
    old_debug=copy.copy(reader.last_debug)
    h=reader.attn.register_forward_pre_hook(tap,with_kwargs=True)
    try:
        with torch.inference_mode():replay=reader(queries,context)
    finally:
        h.remove();reader.last_debug=old_debug
    if cursor!=Q*A*D or not tokens or not torch.equal(torch.cat(positions),torch.arange(Q,device=ids.device)):
        raise RuntimeError('Evidence capture did not cover full native reader lattice')
    err=(replay-original_output).abs().max().item()
    if not np.isfinite(err) or err>tolerance:raise RuntimeError(f'Native reader replay parity failed: {err}')
    mem=torch.cat(tokens);pad=torch.cat(pads)
    valid=true_valid_fn(context,ids,Q,A,D).bool()
    if valid.shape!=pad.shape:raise ValueError('True evidence mask shape differs')
    expected_pad=~valid
    expected_pad=expected_pad.clone();expected_pad[~valid.any(-1),0]=False
    if not torch.equal(expected_pad,pad):raise RuntimeError('Projected validity != native attention mask')
    if not torch.isfinite(mem).all():raise FloatingPointError('Nonfinite native evidence')
    mem=torch.where(valid[...,None],mem,0.0)
    return mem,valid,err


@torch.inference_mode()
def snapshot(model,batch,decode,rotation_fn,true_valid_fn,selector=None,
             expected_queries=1024,tolerance=1e-6):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen base must eval() and requires_grad_(False)')
    decoder=model.kview_grasp_module.decoder
    if decoder.reader is None:raise ValueError('Stage1 requires a GVAR reader')
    with capture_hooks(decoder) as captured:
        ep=model(dict(batch))
        native=decode(ep,use_cdf=True)[0].detach().float()
    pred=ep.get('depth_net_pred');used=ep.get('depth_map_used_for_geometry')
    if pred is None or used is None or pred.shape!=used.shape or not torch.allclose(pred,used,rtol=0,atol=tolerance):
        raise RuntimeError('Network geometry must be RGB predicted depth')
    cdf=ep['grasp_cdf_pred_angle_depth'][0].detach().float()
    T,Q,A,D=cdf.shape
    if (T,Q,A,D)!=(6,expected_queries,12,4):raise ValueError(f'Unexpected CDF lattice {cdf.shape}')
    utility=cdf.sigmoid().mean(0)
    flat=utility.reshape(Q,A*D).argmax(-1)
    angle=flat//D;native_depth=flat%D;q=torch.arange(Q,device=cdf.device)
    score_features=captured['score_feature'].float()
    if score_features.shape!=(Q*A*D,64):raise ValueError('Unexpected native score feature contract')
    score_features=score_features.reshape(Q,A,D,64)[q,angle]
    baseline=utility[q,angle]
    base_logit=torch.logit(baseline.clamp(1e-7,1-1e-7))
    source_scores=baseline
    if selector is not None:
        if selector.training or any(p.requires_grad for p in selector.parameters()):raise ValueError('Selector must be frozen')
        d=torch.arange(D,device=cdf.device)[None].expand(Q,D)
        source_scores=selector(score_features,base_logit,d).sigmoid()
    source_depth=source_scores.argmax(-1)
    widths=(1.2*ep['grasp_width_pred_angle_depth'][0].detach().float().permute(1,2,0)/10).clamp(0,.1)
    centers=ep['xyz_graspable'][0].detach().float()
    R=rotation_fn(-ep['grasp_top_view_xyz'][0].detach().float(),angle.float()*(np.pi/A))
    bank=cdf.new_zeros(Q,D,17)
    bank[...,0]=baseline;bank[...,1]=widths[q,angle];bank[...,2]=.02
    bank[...,3]=torch.arange(1,D+1,device=cdf.device)[None]*.01
    bank[...,4:13]=R.reshape(Q,1,9);bank[...,13:16]=centers[:,None];bank[...,16]=-1
    native_replay=bank[q,native_depth]
    if native.shape!=native_replay.shape:raise RuntimeError('Production/native decode shape mismatch')
    err=(native-native_replay).abs().max().item()
    if not np.isfinite(err) or err>tolerance:raise RuntimeError(f'Native 17D decode parity failed: {err}')
    grasps=bank[q,source_depth].clone();grasps[:,0]=source_scores.max(-1).values
    ctx=captured['context'];parents=q*A+angle
    if not torch.allclose(ctx['centers'].reshape(-1,3)[parents],centers,rtol=0,atol=tolerance):
        raise RuntimeError('Reader center differs from executed center')
    if not torch.allclose(ctx['rotations'].reshape(-1,3,3)[parents],R,rtol=0,atol=tolerance):
        raise RuntimeError('Reader rotation differs from executed orientation')
    idx=parents*D+source_depth
    features=captured['prebranch'][0].transpose(0,1)[idx].float()
    mem,valid,read_err=selected_evidence(decoder.reader,captured['reader_queries'],ctx,
                        captured['reader_output'],idx,true_valid_fn,tolerance)
    z=torch.logit(grasps[:,0].clamp(1e-7,1-1e-7))
    def arr(x):return x.detach().cpu().numpy().copy()
    return dict(grasps=arr(grasps),features=arr(features),base_logit=arr(z),
                memory=arr(mem.float()),memory_valid=arr(valid),query_id=arr(q),
                angle_id=arr(angle),depth_id=arr(source_depth)),{
                    'native_decode_max_abs':err,'native_reader_replay_max_abs':read_err}
