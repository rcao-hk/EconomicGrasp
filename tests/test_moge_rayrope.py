"""CPU numerical/contracts tests. Does not substitute for main CUDA integration."""
from __future__ import annotations
import dataclasses,json,math,subprocess,sys,types
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moge_rayrope.config import ModelConfig,LossConfig
from moge_rayrope.geometry import (image_rays,known_K_z_shift,canonicalize_points,
    metric_depth_from_shape,fit_scale_zshift_l1,shape_losses,interval_score)
from moge_rayrope.attention import (uniform_rope_moments,rotate_channels,
    ray_coordinates_in_grasp_frame,GraspRayRoPEGrouping)
from moge_rayrope.model import filter_empty_objects,CPU_LABELS


def camera(B=1,h=32,w=32):
    return torch.tensor([[[32.,0.,(w-1)/2],[0.,32.,(h-1)/2],[0.,0.,1.]]]).expand(B,-1,-1).clone()


def test_known_intrinsics_shift_and_affine_gauge():
    K=camera(); rays=image_rays(K,(16,16),(32,32))
    z=torch.linspace(.3,.8,256).reshape(1,1,16,16)
    true=rays*z
    p=(true-torch.tensor([0.,0.,.7])[None,:,None,None])/2.5
    shift=known_K_z_shift(p,rays)
    torch.testing.assert_close(shift,torch.full_like(shift,.7/2.5),atol=1e-6,rtol=1e-6)
    norm,_,_=canonicalize_points(p,rays)
    norm2,_,_=canonicalize_points(3.2*p+torch.tensor([0.,0.,-.4])[None,:,None,None],rays)
    torch.testing.assert_close(norm,norm2,atol=3e-6,rtol=3e-6)


def test_metric_grounding_does_not_rewrite_shape():
    shape=torch.randn(2,3,8,8,requires_grad=True)
    anchor=torch.tensor([.4,.6],requires_grad=True).view(2,1,1,1)
    depth=metric_depth_from_shape(shape,anchor)
    sg,ag=torch.autograd.grad(depth.mean(),(shape,anchor),allow_unused=True)
    assert sg is None and ag is not None
    assert torch.isfinite(depth).all() and bool((depth>0).all())


@pytest.mark.parametrize('seed',range(5))
def test_sampled_weighted_l1_alignment_matches_linear_program(seed):
    rng=np.random.default_rng(seed); n=10
    x=rng.normal(size=(n,3)); y=rng.normal(size=(n,3)); w=rng.uniform(.5,2,n)
    s,t=fit_scale_zshift_l1(torch.tensor(x),torch.tensor(y),torch.tensor(w))
    # Unknowns s,t,3*N absolute residual slacks.
    c=np.r_[0.,0.,np.repeat(w,3)]
    ab=[]; bb=[]
    for i in range(n):
        for j in range(3):
            k=i*3+j
            for sign in (-1,1):
                row=np.zeros(2+3*n); row[0]=sign*x[i,j]; row[1]=sign*(j==2); row[2+k]=-1
                ab.append(row); bb.append(sign*y[i,j])
    opt=linprog(c,A_ub=np.array(ab),b_ub=np.array(bb),bounds=[(0,None),(None,None)]+[(0,None)]*(3*n),method='highs')
    assert opt.success
    got=np.sum(np.abs(float(s)*x+np.array([0,0,float(t)])-y)*w[:,None])
    assert got==pytest.approx(opt.fun,abs=2e-5,rel=2e-6)


def test_alignment_perfect_scale_zshift():
    p=torch.randn(30,3); t=2.3*p+torch.tensor([0,0,.5])
    s,z=fit_scale_zshift_l1(p,t,torch.ones(30))
    torch.testing.assert_close(s,torch.tensor(2.3)); torch.testing.assert_close(z,torch.tensor(.5))


def test_shape_loss_zero_on_correct_affine_shape():
    K=camera(); rays=image_rays(K,(16,16),(32,32))
    gt=torch.linspace(.3,.8,32*32).view(1,1,32,32)
    low=torch.nn.functional.interpolate(gt,(16,16),mode='nearest')
    p=((rays*low-torch.tensor([0,0,.17])[None,:,None,None])/2).requires_grad_()
    logs=shape_losses(p,gt,K,(32,32),LossConfig(global_points=16,local_points=12,patches_per_scale=2,patch_sizes=(4,8)))
    assert float(logs['global_shape'].detach())<1e-6
    assert float(logs['local_shape'].detach())<1e-5
    sum(logs.values()).backward(); assert p.grad is not None and torch.isfinite(p.grad).all()


def test_empty_shape_supervision_stays_finite_connected():
    p=torch.randn(1,3,8,8,requires_grad=True)
    logs=shape_losses(p,torch.zeros(1,1,32,32),camera(),(32,32),LossConfig())
    assert all(float(v)==0 for v in logs.values())
    sum(logs.values()).backward(); assert torch.count_nonzero(p.grad)==0


def test_interval_loss_has_no_mean_gradient_and_penalizes_missed_coverage():
    mu=torch.tensor([.5,.5],requires_grad=True); hw=torch.tensor([.02,.02],requires_grad=True)
    y=torch.tensor([.5,.6]); valid=torch.tensor([True,True])
    loss=interval_score(mu,hw,y,valid,.9)
    mg,hg=torch.autograd.grad(loss,(mu,hw),allow_unused=True)
    assert mg is None and hg[0]>0 and hg[1]<0


def test_uniform_expected_encoding_matches_monte_carlo():
    torch.manual_seed(3)
    lo=torch.rand(2,6)*.5; hi=lo+torch.rand(2,6)*.4
    f=torch.tensor([math.pi,2*math.pi])
    c,s=uniform_rope_moments(lo,hi,f)
    x=lo[None]+torch.rand(60000,2,6)*(hi-lo)[None]
    torch.testing.assert_close(c,(x[...,None]*f).cos().mean(0),atol=.007,rtol=.007)
    torch.testing.assert_close(s,(x[...,None]*f).sin().mean(0),atol=.007,rtol=.007)


def test_zero_uncertainty_equals_point_and_rotations_invert():
    x=torch.randn(2,3,4,48); coords=torch.randn(2,3,6); f=math.pi*2**torch.arange(4.)
    c,s=uniform_rope_moments(coords,coords,f)
    torch.testing.assert_close(c,torch.cos(coords[...,None]*f))
    y=rotate_channels(x,c,s,inverse=True)
    torch.testing.assert_close(rotate_channels(y,c,s),x,atol=2e-6,rtol=2e-6)


def test_expected_moment_attenuates_high_frequency():
    c,s=uniform_rope_moments(torch.full((1,6),-.25),torch.full((1,6),.25),torch.tensor([math.pi,4*math.pi]))
    amp=(c.square()+s.square()).sqrt()
    assert bool((amp[:, :, 1]<amp[:, :, 0]).all())


def test_query_frame_coordinates_are_rigid_transform_invariant():
    torch.manual_seed(1)
    center=torch.tensor([[[.01,.02,.5],[.05,-.03,.6]]])
    R=torch.eye(3)[None,None].expand(1,2,3,3).clone()
    points=center[:,:,None]+.02*torch.randn(1,2,5,3)
    cam=torch.zeros_like(center)
    q,l,u,m=ray_coordinates_in_grasp_frame(center,R,points-.002,points+.002,camera_origin=cam)
    angle=.4; A=torch.tensor([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1.]])
    shift=torch.tensor([.1,.2,.3])
    transform=lambda p:p@A.T+shift
    qq,ll,uu,mm=ray_coordinates_in_grasp_frame(transform(center),A[None,None]@R,
                        transform(points-.002),transform(points+.002),camera_origin=transform(cam))
    torch.testing.assert_close(l,ll,atol=5e-6,rtol=5e-6); torch.testing.assert_close(u,uu,atol=5e-6,rtol=5e-6)
    torch.testing.assert_close(q,qq); assert torch.equal(m,mm)


def group_inputs(B=1,M=4):
    torch.manual_seed(12)
    seed=torch.randn(B,8,M,requires_grad=True)
    features=torch.randn(B,8,32,32,requires_grad=True)
    depth=torch.full((B,1,32,32),.5,requires_grad=True)
    sigma=torch.full_like(depth,.02,requires_grad=True)
    inds=torch.tensor([[16*32+12,16*32+14,16*32+16,16*32+18]])[:,:M].expand(B,-1)
    center=torch.zeros(B,M,3); center[...,2]=.5; center.requires_grad_()
    R=torch.eye(3)[None,None].expand(B,M,3,3).clone().requires_grad_()
    return dict(seed_features=seed,token_sel_idx=inds,seed_xyz=center,top_view_rot=R,
                feat_map=features,depth_map=depth,camera_K=camera(B)),sigma


@pytest.mark.parametrize('encoding',['none','point','expected'])
def test_grouping_gradients_only_task_path(encoding):
    cfg=ModelConfig(use_rayrope=True,ray_encoding=encoding,group_chunk=2,ray_radius_px=8,checkpoint_chunks=False)
    m=GraspRayRoPEGrouping(cfg,8,16)
    inputs,sigma=group_inputs(); m.geometry_context={'sigma':sigma}
    out=m(**inputs); assert out.shape==(1,16,4) and torch.isfinite(out).all()
    (out*torch.randn_like(out)).sum().backward()
    assert inputs['feat_map'].grad is not None and inputs['seed_features'].grad is not None
    for name in ('depth_map','seed_xyz','top_view_rot'): assert inputs[name].grad is None
    assert sigma.grad is None


def test_checkpoint_chunks_match_unchunked_outputs_and_gradients():
    cfg=ModelConfig(use_rayrope=True,group_chunk=2,ray_radius_px=8,checkpoint_chunks=True)
    a=GraspRayRoPEGrouping(cfg,8,16)
    b=GraspRayRoPEGrouping(dataclasses.replace(cfg,group_chunk=16,checkpoint_chunks=False),8,16)
    b.load_state_dict(a.state_dict())
    ia,sa=group_inputs(); ib,sb=group_inputs()
    a.geometry_context={'sigma':sa}; b.geometry_context={'sigma':sb}
    oa=a(**ia); ob=b(**ib)
    torch.testing.assert_close(oa,ob,atol=2e-6,rtol=2e-6)
    oa.square().mean().backward(); ob.square().mean().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,atol=4e-6,rtol=4e-5)


def test_all_invalid_keys_are_finite_not_fake_valid_support():
    cfg=ModelConfig(use_rayrope=True,group_chunk=2,ray_radius_px=8,checkpoint_chunks=False)
    m=GraspRayRoPEGrouping(cfg,8,16)
    inputs,sigma=group_inputs(); inputs['depth_map']=torch.full_like(inputs['depth_map'],-1)
    m.geometry_context={'sigma':sigma}
    assert torch.isfinite(m(**inputs)).all()


def test_empty_object_filter_preserves_all_label_alignment():
    data={k:[[torch.tensor([[11]]),torch.tensor([[22]])]] for k in CPU_LABELS}
    data['grasp_points_list']=[[torch.empty(0,3),torch.ones(3,3)]]
    out=filter_empty_objects(data)
    assert len(out['object_poses_list'][0])==1
    for k in CPU_LABELS:
        if k!='grasp_points_list': assert out[k][0][0].item()==22
    assert len(data['object_poses_list'][0])==2


def test_main_import_config_is_not_triggered_by_cli_help():
    for script in ('train_moge_rayrope.py','inference_moge_rayrope.py','eval_moge_rayrope.py','compare_moge_rayrope.py'):
        r=subprocess.run([sys.executable,str(ROOT/script),'--help'],capture_output=True,text=True)
        assert r.returncode==0,r.stderr


@pytest.mark.parametrize('kwargs',[
    {'ray_dim':128},{'use_shape_tokens':True},{'uncertainty':'learned'},
    {'calibration_detach':False},{'ray_grid':4},{'fixed_halfwidth':float('nan')},
])
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError): ModelConfig(**kwargs)


def test_shell_syntax():
    for path in (ROOT/'scripts').glob('*.sh'): subprocess.run(['bash','-n',str(path)],check=True)


def test_no_accumulation_cli_and_baseline_default_off():
    import train_moge_rayrope
    p=train_moge_rayrope.parser(); a=p.parse_args(['--output-root','dummy'])
    assert a.use_moge==a.use_rayrope==0
    assert '--grad-accum' not in p.format_help()


def test_factorized_depth_tuple_and_gradient_contract_with_stub_dpt(monkeypatch):
    """Exercise actual adapter with a tiny DPT stub; not a CUDA backbone test."""
    from moge_rayrope.model import FactorizedMetricDepth
    class Backbone(nn.Module):
        def __init__(self): super().__init__(); self.proj=nn.Linear(3,384)
        def get_intermediate_layers(self,img,indices,return_class_token):
            tok=self.proj(img.mean((-1,-2)))[:,None].expand(-1,4,-1)
            return [(tok,tok.mean(1)) for _ in indices]
    class TinyDPT(nn.Module):
        def __init__(self,in_channels=384,features=64,out_dim=3,**kw):
            super().__init__(); self.latent=nn.Linear(in_channels,features); self.pred=nn.Linear(features,out_dim)
        def forward(self,feats,ph,pw):
            f=self.latent(feats[-1][0].mean(1))[:,:,None,None].expand(-1,-1,8,8)
            raw=self.pred(f[:,:,0,0])[:,:,None,None].expand(-1,-1,32,32)
            return f,raw
    class Pose(nn.Module):
        def __init__(self): super().__init__(); self.net=nn.Linear(4,384)
        def forward(self,features,pose):
            delta=self.net(pose)
            return [(x+delta[:,None],c+delta) for x,c in features],{}
    fake=types.ModuleType('models.dinov2_dpt'); fake.DPTHead=TinyDPT
    monkeypatch.setitem(sys.modules,'models.dinov2_dpt',fake)
    old=nn.Module(); old.depthnet=nn.Module(); old.depthnet.pretrained=Backbone()
    old.depthnet.depth_head=TinyDPT(out_dim=1); old.depthnet.encoder='vits'
    old.depthnet.intermediate_layer_idx={'vits':[1,2,3,4]}
    old.pose_aware_adapter=Pose(); old.stride=1
    cfg=ModelConfig(encoder='vits',use_moge=True)
    model=FactorizedMetricDepth(old,cfg).train()
    assert not model.depthnet.pretrained.training
    K=torch.tensor([[[350.,0.,223.5],[0.,350.,223.5],[0.,0.,1.]]])
    result=model(torch.rand(1,3,448,448),camera_pose_vec=torch.rand(1,4),camera_K=K,
                 return_feats=True,return_raw=True,return_pose_aux=True)
    assert len(result)==6 and result[0].shape==(1,1,448,448)
    assert result[3].shape==(1,1,448,448)
    shape_params=list(model.depthnet.depth_head.parameters())
    grads=torch.autograd.grad(result[0].mean(),shape_params,retain_graph=True,allow_unused=True)
    assert all(g is None or torch.count_nonzero(g)==0 for g in grads)
    shape_grad=torch.autograd.grad(model.last_geometry['points'].square().mean(),shape_params,allow_unused=True)
    assert any(g is not None and float(g.abs().sum())>0 for g in shape_grad)
    assert all(not p.requires_grad for p in model.depthnet.pretrained.parameters())


def test_frame_seed_restored_and_independent_of_access_order():
    import random
    from inference_moge_rayrope import SeededIndices
    class Base:
        def __getitem__(self,i): return (i,random.random(),np.random.rand(),float(torch.rand(())))
    ds=SeededIndices(Base(),[9,5])
    a=ds[0]; _=ds[1]; b=ds[0]
    assert a==b
    torch.manual_seed(17); expected=torch.rand(2)
    torch.manual_seed(17); first=torch.rand(()); _=ds[0]; second=torch.rand(())
    torch.testing.assert_close(torch.stack((first,second)),expected)


def test_rayrope_adapter_matches_native_decoder_channels(monkeypatch):
    from moge_rayrope.model import EconomicGraspMoGeRayRoPE
    native=nn.Module(); native.kview_grasp_module=nn.Module()
    native.kview_grasp_module.decoder=nn.Module()
    native.kview_grasp_module.decoder.input_proj=nn.Conv1d(128,128,1)
    fake=types.ModuleType('models.economicgrasp_bip3d')
    fake.economicgrasp_dpt=lambda **kwargs: native
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    model=EconomicGraspMoGeRayRoPE(ModelConfig(use_rayrope=True))
    group=model.base.kview_grasp_module.group
    output=group.out(torch.randn(1,3,group.out.in_features))
    assert model.base.kview_grasp_module.decoder.input_proj(output.transpose(1,2)).shape==(1,128,3)


def test_optional_shape_tokens_are_detached_and_shape_compatible():
    cfg=ModelConfig(use_moge=True,use_rayrope=True,use_shape_tokens=True,
                    group_chunk=2,ray_radius_px=8,checkpoint_chunks=False)
    model=GraspRayRoPEGrouping(cfg,8,16)
    inp,sigma=group_inputs(); shape=torch.randn(1,3,16,16,requires_grad=True)
    model.geometry_context={'sigma':sigma,'shape':shape}
    out=model(**inp)
    out.square().mean().backward()
    assert shape.grad is None
    assert inp['feat_map'].grad is not None
