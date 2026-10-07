"""CPU tests for physical coordinates, reader, detach, shapes and CLI safety.

These do not claim that the external GraspNet/CUDA full pipeline is installed.
"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.gripper_volume_reader import (GVARConfig, ROLE_NAMES, canonical_probes,
    project_probes, action_relative_geometry, GripperVolumeReader, ActionDepthAdapter, apply_depthwise_heads)

torch.set_num_threads(1)


def inputs(variant="volume", B=1, Q=2, A=3):
    torch.manual_seed(3)
    cfg = GVARConfig(variant=variant, reader_dim=16, reader_heads=4, action_chunk=7, reader_dropout=0., activation_checkpoint=False)
    reader = GripperVolumeReader(8, 16, cfg)
    q = torch.randn(B, Q, A, 4, 16, requires_grad=True)
    xyz = torch.zeros(B, Q*A, 3); xyz[..., 2] = .6
    xyz[..., 0] = torch.linspace(-.1, .1, Q*A)
    ctx = {"centers": xyz.requires_grad_(), "rotations": torch.eye(3).expand(B,Q*A,3,3).clone(),
           "K": torch.tensor([[120.,0.,31.5],[0.,120.,31.5],[0.,0.,1.]]).repeat(B,1,1),
           "pregeom": torch.randn(B,8,16,16,requires_grad=True), "depth": torch.full((B,1,64,64),.6,requires_grad=True),
           "image_hw": (64,64)}
    return reader, q, ctx


def test_probe_count_and_roles():
    p,r = canonical_probes(torch.tensor([.01,.04]), GVARConfig())
    assert p.shape == (2,36,3)
    assert torch.equal(torch.bincount(r), torch.full((6,),6))
    assert len(ROLE_NAMES) == 6


def test_insertion_moves_local_x_not_camera_z():
    p,_ = canonical_probes(torch.tensor([.01,.04]), GVARConfig())
    torch.testing.assert_close(p[1]-p[0], torch.tensor([.03,0.,0.]).expand(36,3))


def test_gripper_convention_matches_collision_regions():
    cfg=GVARConfig(); d=torch.tensor([.02]); p,r=canonical_probes(d,cfg); p=p[0]
    finger=p[r==3]
    assert torch.all(finger[:,0] > d-cfg.finger_length_m)
    assert torch.all(finger[:,0] < d)
    assert torch.all(finger[:,1].abs() > cfg.envelope_width_m/2)
    assert torch.all(finger[:,1].abs() < cfg.envelope_width_m/2+cfg.finger_thickness_m)
    assert torch.all(p[r==4,0] < d-cfg.finger_length_m)
    assert torch.all(p[r==5,0] < d-cfg.finger_length_m-cfg.finger_thickness_m)


def test_projection_uses_rotation_columns():
    # local x maps to camera y under Rz(90 deg)
    R=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    center=torch.tensor([[0.,0.,1.]])
    p=torch.tensor([[[.1,0.,0.]]]); K=torch.tensor([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
    x,uv,grid,v=project_probes(center,R,p,K,(101,101))
    torch.testing.assert_close(x,torch.tensor([[[0.,.1,1.]]]))
    torch.testing.assert_close(uv,torch.tensor([[[50.,60.]]]))
    assert v.all()


def test_behind_camera_and_invalid_projected_points_masked():
    p=torch.tensor([[[0.,0.,-2.],[1e5,0.,0.],[float('nan'),0.,0.]]])
    _,_,g,v=project_probes(torch.tensor([[0.,0.,1.]]),torch.eye(3)[None],p,torch.eye(3),(64,64))
    assert not v.any() and torch.isfinite(g).all()


@pytest.mark.parametrize("variant", ["volume_fixed","volume","volume_rel"])
def test_reader_shapes_and_detach(variant):
    reader,q,ctx=inputs(variant,B=2)
    out=reader(q,ctx)
    assert out.shape==q.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert q.grad is not None and ctx['pregeom'].grad is not None
    assert ctx['depth'].grad is None and ctx['centers'].grad is None
    assert all(p.grad is not None for p in reader.parameters()), 'DDP-unused parameter in executed reader'


def test_capacity_matched_volume_variants():
    counts=[]
    keys=[]
    for v in ('volume_fixed','volume','volume_rel'):
        reader,_,_=inputs(v); counts.append(sum(p.numel() for p in reader.parameters())); keys.append(list(reader.state_dict()))
    assert len(set(counts))==1 and keys[0]==keys[1]==keys[2]


def test_chunk_equivalence_eval():
    r,q,c=inputs(); r.eval(); a=r(q,c)
    r.cfg=replace(r.cfg,action_chunk=1000); b=r(q,c)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)


def test_checkpointed_reader_backward():
    r,q,c=inputs('volume_rel'); r.cfg=replace(r.cfg,activation_checkpoint=True)
    r.train(); r(q,c).square().mean().backward()
    assert c['pregeom'].grad is not None and c['depth'].grad is None


def test_all_invalid_is_identity():
    r,q,c=inputs(); c['centers']=torch.full_like(c['centers'],float('nan'))
    a=r(q,c); torch.testing.assert_close(a,q)
    assert float(r.last_debug['D: GVAR all invalid'])==1.


def test_fixed_control_has_different_grid_semantics():
    cfg=GVARConfig(); depths=torch.tensor([.01,.04]); p,_=canonical_probes(depths,cfg)
    fixed,_=canonical_probes(torch.full_like(depths,.025),cfg)
    assert not torch.equal(p[0],p[1]); torch.testing.assert_close(fixed[0],fixed[1])


def test_unknown_geometry_is_not_free_space():
    r,q,c=inputs('volume_rel'); centers=c['centers'][0,:1].detach(); R=c['rotations'][0,:1]; K=c['K'][0]
    p,_=canonical_probes(torch.tensor([.01]),r.cfg); x,uv,g,v=project_probes(centers,R,p,K,(64,64))
    geom=action_relative_geometry(torch.zeros(1,1,64,64),g,uv,x,centers,R,K,v,.06)
    assert torch.equal(geom[...,:5],torch.zeros_like(geom[...,:5]))
    assert torch.equal(geom[...,5].bool(),v)


def test_slot_queries_really_depend_on_insertion():
    a=ActionDepthAdapter(16,4,0.); x=torch.randn(1,2,3,16)
    y=a.finish(a.make_queries(x)); assert y.shape==(1,2,3,4,16)
    assert not torch.equal(y[:,:,:,0],y[:,:,:,1])


def test_original_depth_heads_shape_monotonic_and_grad():
    x=torch.randn(2,3,12,4,64,requires_grad=True); y=torch.randn_like(x,requires_grad=True)
    wh=nn.Conv1d(64,4,1); ch=nn.Conv1d(64,24,1)
    w,logit=apply_depthwise_heads(x,y,wh,ch,-4.)
    assert w.shape==(2,4,3,12) and logit.shape==(2,6,3,12,4)
    assert (logit[:,1:]-logit[:,:-1] >= -1e-7).all()
    (w.square().mean()+logit.square().mean()).backward()
    assert x.grad is not None and y.grad is not None
    assert wh.weight.grad.abs().sum()>0 and ch.weight.grad.abs().sum()>0


def test_depth_head_gather_matches_explicit_conv():
    x=torch.randn(1,2,3,4,64); h=nn.Conv1d(64,4,1); c=nn.Conv1d(64,24,1)
    w,l=apply_depthwise_heads(x,x,h,c,-4.)
    for d in range(4):
        explicit=h(x[:,:,:,d].reshape(1,6,64).transpose(1,2))[:,d].reshape(1,2,3)
        torch.testing.assert_close(w[:,d],explicit)


@pytest.mark.parametrize('script',['train_gvar.py','inference_gvar.py'])
def test_help_without_cuda_or_legacy_argparse(script):
    p=subprocess.run([sys.executable,str(ROOT/script),'--help'],capture_output=True,text=True,cwd=ROOT)
    assert p.returncode==0,p.stderr
    assert '--gvar_variant' in p.stdout


def test_parser_consumes_only_owned_flags(monkeypatch):
    from utils.gvar_runtime import parse_gvar_args
    monkeypatch.setattr(sys,'argv',['x','--gvar_variant','volume_rel','--gvar_action_chunk','64','--dataset_root','/data','--use_cdf'])
    a=parse_gvar_args(); assert a.gvar_action_chunk==64
    assert sys.argv==['x','--dataset_root','/data','--use_cdf']


def test_sampling_exact_frames():
    from utils.gvar_runtime import selected_frame_indices
    class Data:
        scenename=[f'scene_{s:04d}' for s in range(100,130) for a in range(256)]
        frameid=list(range(256))*30
    d=Data(); idx=selected_frame_indices(d)
    assert len(idx)==780 and [d.frameid[i] for i in idx[:26]]==list(range(0,256,10))
    with pytest.raises(ValueError): selected_frame_indices(d,.2)


def test_checkpoint_contract_rejects_old_and_wrong_variant():
    from utils.gvar_runtime import validate_checkpoint
    with pytest.raises(ValueError): validate_checkpoint({'model_state_dict':{}})
    ck={'gvar_contract_version':1,'gvar_config':GVARConfig().to_dict(),'detach_policy':dict(E=True,Q=True,C=True)}
    assert validate_checkpoint(ck).variant=='volume'
    with pytest.raises(ValueError): validate_checkpoint(ck,GVARConfig(variant='slot'))


def test_shell_dry_runs_do_not_start_experiments(tmp_path):
    import os
    env=dict(os.environ,DATASET_ROOT='/data/graspnet',GNTRANS_RGB_ROOT='/data/GN-Trans',CKPT='/not/a/checkpoint',DRY_RUN='1',OUTPUT_ROOT=str(tmp_path/'out'))
    for script in ['run_gvar_train.sh','run_gvar_eval.sh']:
        p=subprocess.run(['bash',str(ROOT/'scripts'/script)],env=env,capture_output=True,text=True)
        assert p.returncode==0,p.stderr
    assert not (tmp_path/'out').exists()
