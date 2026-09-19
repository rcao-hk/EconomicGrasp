import numpy as np
import torch

from rep_p0_geometry_common import (
    DescriptorConfig,
    backproject_depth_map,
    build_translation_ray_actions,
    describe_actions,
    friction_to_cdf_targets,
    friction_utility,
    scene_sharded_indices,
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
    points=backproject_depth_map(depth,K,voxel_size=0.005)
    assert points.ndim==2 and points.shape[1]==3 and len(points)>0

    native=_native_actions(q=2)
    actions,valid=build_translation_ray_actions(native,[-10,0,10])
    cfg=DescriptorConfig(voxel_size=0.005)
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


def test_rep_p0_default_evidence_resolution_is_5mm():
    cfg=DescriptorConfig()
    assert abs(cfg.voxel_size-0.005)<1e-12
    H=W=16
    depth=np.full((H,W),0.6,dtype=np.float32)
    K=np.array([[20.0,0,7.5],[0,20.0,7.5],[0,0,1]],dtype=np.float32)
    pts_default=backproject_depth_map(depth,K)
    pts_5mm=backproject_depth_map(depth,K,voxel_size=0.005)
    assert np.array_equal(pts_default,pts_5mm)


def test_scene_level_sharding_has_disjoint_scene_ownership():
    # 6 scenes x 256 frames, sampled every 10th frame.
    total=6*256
    shards=[
        scene_sharded_indices(
            total,
            0.1,
            shard_id=s,
            num_shards=3,
            frames_per_scene=256,
        )
        for s in range(3)
    ]
    scene_sets=[{idx//256 for idx in ids} for ids in shards]
    assert scene_sets[0]=={0,3}
    assert scene_sets[1]=={1,4}
    assert scene_sets[2]=={2,5}
    assert scene_sets[0].isdisjoint(scene_sets[1])
    assert scene_sets[0].isdisjoint(scene_sets[2])
    assert scene_sets[1].isdisjoint(scene_sets[2])
    # Every selected frame in a scene belongs to exactly one shard.
    merged=[idx for ids in shards for idx in ids]
    assert len(merged)==len(set(merged))


def test_scene_level_sharding_max_samples():
    ids=scene_sharded_indices(
        4*256,
        0.1,
        shard_id=0,
        num_shards=2,
        max_samples=5,
    )
    assert len(ids)==5
    assert all(idx//256==0 for idx in ids)
