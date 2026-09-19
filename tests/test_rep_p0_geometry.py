import numpy as np
import torch

from rep_p0_geometry_common import (
    DescriptorConfig,
    backproject_depth_map,
    build_translation_ray_actions,
    describe_actions,
    friction_to_cdf_targets,
    friction_utility,
    select_query_indices,
)


def _native_actions(q=4):
    g=np.zeros((q,17),dtype=np.float32)
    g[:,0]=np.linspace(0.9,0.6,q)
    g[:,1]=0.06
    g[:,2]=0.02
    g[:,3]=0.03
    g[:,4:13]=np.eye(3,dtype=np.float32).reshape(1,9)
    g[:,13]=0.02
    g[:,14]=0.01
    g[:,15]=0.60
    return g


def test_same_ray_translation_preserves_nontranslation_action():
    native=_native_actions()
    offsets=[-40,-20,-10,0,10,20,40]
    actions,valid=build_translation_ray_actions(native,offsets,min_depth=0.2,max_depth=1.0)
    assert actions.shape==(7,4,17)
    assert valid.all()
    zero=3
    assert np.array_equal(actions[zero],native)
    for k in range(7):
        assert np.array_equal(actions[k,:,0:13],native[:,0:13])
        # same image ray => x/z and y/z are unchanged
        assert np.allclose(actions[k,:,13]/actions[k,:,15],native[:,13]/native[:,15],atol=1e-6)
        assert np.allclose(actions[k,:,14]/actions[k,:,15],native[:,14]/native[:,15],atol=1e-6)


def test_depth_backprojection_and_descriptor_contract():
    H=W=32
    depth=np.full((H,W),0.6,dtype=np.float32)
    K=np.array([[30.0,0,15.5],[0,30.0,15.5],[0,0,1]],dtype=np.float32)
    points=backproject_depth_map(depth,K,voxel_size=0.008)
    assert points.ndim==2 and points.shape[1]==3 and len(points)>0

    native=_native_actions(q=2)
    actions,valid=build_translation_ray_actions(native,[-10,0,10])
    cfg=DescriptorConfig(voxel_size=0.008)
    feat=describe_actions(points,actions,valid,cfg)
    assert feat.shape==(3,2,cfg.feature_dim)
    assert np.isfinite(feat).all()


def test_friction_cdf_targets():
    f=np.array([-1.0,0.3,0.9],dtype=np.float32)
    y=friction_to_cdf_targets(f)
    assert y.shape==(3,6)
    assert y[0].sum()==0
    assert y[1].tolist()==[0,1,1,1,1,1]
    assert y[2].tolist()==[0,0,0,0,1,1]
    u=friction_utility(f)
    assert np.allclose(u,[0,5/6,2/6])


def test_topk_uniform_query_count():
    g=torch.from_numpy(_native_actions(q=64))
    ids=select_query_indices(g,32,"topk_uniform")
    assert ids.numel()==32
    assert torch.unique(ids).numel()==32
