"""CPU tests; integration uses the repository's actual enhancer/group classes."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from rep_a_common import (aggregate, array_sha, cdf_targets, choose, fixed_data, metrics,
    parse_case, perturb_depth, read_frame, save_npz, seed_for, training_case)
from rep_a_model import IndependentImageReader, RepAReaderScorer


def actions():
    a=np.zeros((3,4,17),np.float32)
    a[...,0]=.6; a[...,1]=.06; a[...,2]=.02; a[...,3]=.03
    a[...,4:13]=np.eye(3,dtype=np.float32).reshape(9)
    a[:,:,15]=np.array([.49,.5,.51])[:,None]
    return a


def labels():
    f=np.array([[-1,.4,.6,-1],[.8,-1,.4,-1],[.4,-1,.6,-1]],np.float32)
    return {'actions':actions(), 'valid':np.ones((3,4),bool), 'friction':f,
            'utility':cdf_targets(f).mean(-1),'zero_index':np.array(1),
            'offsets_mm':np.array([-10,0,10],np.float32),'query_ids':np.arange(4),
            'scene_id':np.array(0),'anno_id':np.array(0),'native_score':np.ones(4)}


@pytest.mark.parametrize('case',['nominal','bias:10','bias:-20','scale:.02','smooth:10','edge:2'])
def test_perturbation_finite_deterministic_and_does_not_mutate(case):
    dep=torch.linspace(.3,.8,1024).reshape(1,32,32)
    original=dep.clone()
    a, info=perturb_depth(dep,case,seed_for('scene',1))
    b,_=perturb_depth(dep,case,seed_for('scene',1))
    assert torch.equal(a,b) and torch.equal(dep,original)
    assert torch.isfinite(a).all()
    if case=='bias:10': assert abs(info['depth_rms_mm']-10)<.001
    if case=='smooth:10': assert abs(info['depth_rms_mm']-10)<.001
    if case=='nominal': assert torch.equal(a,dep)


def test_separate_augmentation_rng():
    s=seed_for(0,1,2)
    a=training_case(s)
    torch.randn(500)
    assert a==training_case(s)
    assert seed_for(0,1,2)!=seed_for(0,1,3)


def test_invalid_depth_preserved():
    d=torch.ones(1,16,16)*.5; d[0,0,0]=0
    for case in ('bias:10','smooth:5','edge:2'):
        x,_=perturb_depth(d,case,2)
        assert x[0,0,0]==0


def test_reader_visual_path_gradients_and_action_geometry():
    torch.manual_seed(0)
    reader=IndependentImageReader(8,32,4,0.)
    feat=torch.randn(1,8,32,32,requires_grad=True)
    aa=torch.from_numpy(actions().reshape(-1,17))
    K=torch.tensor([[[60.,0,15.5],[0,60.,15.5],[0,0,1.]]])
    local, xyz=reader.sample_points(aa)
    assert xyz.shape==(12,40,3)
    assert torch.allclose(xyz, local+aa[:,None,13:16])
    out=reader(feat,aa,K,(32,32))
    assert out.shape==(12,32) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert feat.grad.abs().sum()>0
    assert all(p.grad is not None for p in reader.parameters())


def test_reader_out_of_view_safe():
    r=IndependentImageReader(8,32,4,0.)
    a=torch.from_numpy(actions().reshape(-1,17)); a[:,13]=10
    K=torch.tensor([[[60.,0,15.5],[0,60.,15.5],[0,0,1.]]])
    out=r(torch.randn(1,8,32,32),a,K,(32,32))
    assert torch.equal(out,torch.zeros_like(out))


def test_projection_same_ray_center_not_sufficient():
    r=IndependentImageReader(8,32,4,0.)
    a=torch.from_numpy(actions().reshape(-1,17))
    _,x=r.sample_points(a)
    uv=x[:,:,:2]/x[:,:,2:]
    assert not torch.allclose(uv[0],uv[-1])  # finite-size gripper footprint changes


def test_exact_oracle_and_native_fallback():
    d=labels(); p=cdf_targets(d['friction'])*.998+.001
    m, k=metrics(p,d,0)
    assert m['utility_gain']>0 and m['harm08']==0
    assert np.array_equal(choose(p.mean(-1),d['valid'],1,1),np.ones(4,int))
    result=aggregate([m,m])
    assert result['num_queries']==8 and result['headroom_recovery']>0.99
    d['valid'][0,0]=False
    q=choose(np.array([[10,1,1,1],[.4,1,1,1],[.3,1,1,1]]),d['valid'],1,0)
    assert q[0]==1


def test_fixed_p0_label_checks(tmp_path):
    p=tmp_path/'scene_0000'/'ann_0000.npz'
    d=labels(); save_npz(p,d); fixed_data(p)
    d['actions'][0,0,1]+=.01; save_npz(p,d)
    with pytest.raises(ValueError,match='fixed-action'): fixed_data(p)


def test_atomic_integrity_and_protocol(tmp_path):
    d=labels(); d.update(version=np.array(1), contract=np.array('abc'),
        image_feature=np.ones((8,8,8)),depth=np.ones((1,32,32)),
        objectness=np.ones((2,32,32)),graspness=np.ones((1,32,32)),
        K=np.eye(3),token_ids=np.array([400,401,402,403]))
    d['action_sha']=np.array(array_sha(d['actions'],d['valid'],d['friction'],d['query_ids']))
    p=tmp_path/'f.npz'; save_npz(p,d)
    assert not list(tmp_path.glob('*.tmp.*'))
    read_frame(p,'abc')
    with pytest.raises(ValueError,match='contract'): read_frame(p,'other')
    d['actions'][0,0,15]+=.01; save_npz(p,d)
    with pytest.raises(ValueError,match='fingerprint'): read_frame(p)


def has_upstream():
    try:
        return importlib.util.find_spec('models.kview_query_transformer') is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not has_upstream(), reason='Run inside EconomicGrasp for upstream reader integration')
@pytest.mark.parametrize('variant',['A0','A1','A2','A3'])
def test_actual_upstream_reader_backward(variant):
    import dataclasses
    from models.grasp_spatial_enhancer import GraspSpatialEnhancer
    from models.kview_query_transformer import KViewQueryTransformerConfig,ViewConditionedAttentionGrouping
    cfg=KViewQueryTransformerConfig(grouping_model_dim=32,head_model_dim=32,patch_size=4,
                                    grouping_num_heads=4,grouping_max_queries_per_chunk=16)
    econf={'embed_dims':8,'feature_3d_dim':8,'vis_dir':None}
    e=GraspSpatialEnhancer(**econf)
    g=ViewConditionedAttentionGrouping(8,8,32,cfg)
    init={'model_config':{'channels':8,'out_dim':32,'enhancer':econf,'group_config':dataclasses.asdict(cfg)},
          'enhancer_state':e.state_dict(),'group_state':g.state_dict()}
    model=RepAReaderScorer(init,variant)
    a=torch.from_numpy(actions())
    data={'actions':a,'image_feature':torch.randn(8,8,8),'depth':torch.ones(1,32,32)*.5,
          'K':torch.tensor([[60.,0,15.5],[0,60.,15.5],[0,0,1.]]),
          'objectness':torch.randn(2,32,32),'graspness':torch.rand(1,32,32),
          'token_ids':torch.tensor([400,401,402,403])}
    old=data['actions'].clone()
    out=model(data)
    assert out.shape==(3,4,6)
    torch.nn.functional.binary_cross_entropy_with_logits(out,torch.rand_like(out)).backward()
    for name, part in model.named_children():
        assert sum(float(p.grad.norm()) for p in part.parameters() if p.grad is not None)>0,name
    assert torch.equal(old,data['actions'])


def test_p0_matched_move_rate_margin():
    from diagnose_rep_a_p0_selection import margin_for_target
    adv=[np.array([.5,.4,.3,-np.inf],np.float64)]
    margin,rate=margin_for_target(adv,.5)
    assert 0.3 <= margin < 0.4
    assert abs(rate-.5)<1e-12


@pytest.mark.skipif(not has_upstream(), reason='Run inside EconomicGrasp for upstream reader integration')
def test_a3_intervention_full_is_exact_original_forward():
    import dataclasses
    from models.grasp_spatial_enhancer import GraspSpatialEnhancer
    from models.kview_query_transformer import KViewQueryTransformerConfig,ViewConditionedAttentionGrouping
    torch.manual_seed(7)
    cfg=KViewQueryTransformerConfig(grouping_model_dim=32,head_model_dim=32,patch_size=4,
                                    grouping_num_heads=4,grouping_max_queries_per_chunk=16)
    econf={'embed_dims':8,'feature_3d_dim':8,'vis_dir':None}
    e=GraspSpatialEnhancer(**econf)
    g=ViewConditionedAttentionGrouping(8,8,32,cfg)
    init={'model_config':{'channels':8,'out_dim':32,'enhancer':econf,'group_config':dataclasses.asdict(cfg)},
          'enhancer_state':e.state_dict(),'group_state':g.state_dict()}
    model=RepAReaderScorer(init,'A3').eval()
    data={'actions':torch.from_numpy(actions()),'image_feature':torch.randn(8,8,8),
          'depth':torch.ones(1,32,32)*.5,
          'K':torch.tensor([[60.,0,15.5],[0,60.,15.5],[0,0,1.]]),
          'objectness':torch.randn(2,32,32),'graspness':torch.rand(1,32,32),
          'token_ids':torch.tensor([400,401,402,403])}
    with torch.no_grad():
        original=model(data)
        full,rep,comps=model(data,intervention='full',return_repr=True,return_components=True)
        _,no_rgb=model(data,intervention='no_rgb',return_repr=True)
        _,rgb_only=model(data,intervention='rgb_only',return_repr=True)
    assert torch.equal(original,full)
    assert torch.allclose(rep,comps['depth']+comps['rgb']+comps['action'])
    assert torch.allclose(no_rgb,comps['depth']+comps['action'])
    assert torch.allclose(rgb_only,comps['rgb']+comps['action'])
