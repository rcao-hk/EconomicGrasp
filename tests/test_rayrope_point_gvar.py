"""CPU-only consistency checks for the isolated Point+GVAR integration.

Full CUDA/DDP/GraspNet forward parity and official evaluation must run on the
robotics server; these fast tests do not claim a GraspNet AP result.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from rayrope_gvar.reader import GVARConfig, GripperVolumeReader
from rayrope_gvar.runtime import VERSION,VARIANTS,point_config,code_fingerprint
from compare_rayrope_gvar import scene_scores


def reader_fixture():
    cfg=GVARConfig(variant="volume_rel",reader_dim=16,reader_heads=4,
                   action_chunk=4,activation_checkpoint=False)
    reader=GripperVolumeReader(feat_dim=8,hidden_dim=32,cfg=cfg)
    B,Q,A,D=1,2,1,4
    q=torch.randn(B,Q,A,D,32,requires_grad=True)
    geom=torch.eye(3).view(1,1,3,3).expand(B,Q,3,3).clone()
    context={
        "pregeom":torch.randn(B,8,16,16,requires_grad=True),
        "centers":torch.tensor([[[0.,0.,.5],[.07,0.,.5]]],requires_grad=True),
        "rotations":geom,"K":torch.tensor([[[160.,0.,31.5],[0.,160.,31.5],[0.,0.,1.]]]),
        "depth":torch.full((1,1,64,64),.52,requires_grad=True),
        "image_hw":(64,64)
    }
    return reader,q,context


def test_point_config_is_moge_off_no_sigma():
    mc=point_config()
    assert mc.use_moge is False and mc.use_rayrope is True
    assert mc.ray_encoding=="point" and mc.uncertainty=="fixed"
    assert mc.group_chunk==32 and mc.seeds==1024
    assert len(VARIANTS)==2 and VERSION.startswith("rayrope_point_gvar_mixed_v2")


def test_reader_online_visual_gradients_and_numeric_geometry_detach():
    torch.manual_seed(7)
    reader,q,ctx=reader_fixture()
    result=reader(q,ctx)
    assert result.shape==q.shape and bool(torch.isfinite(result).all())
    result.square().mean().backward()
    assert ctx["pregeom"].grad is not None
    assert float(ctx["pregeom"].grad.abs().sum())>0
    assert ctx["depth"].grad is None and ctx["centers"].grad is None
    assert q.grad is not None and float(q.grad.abs().sum())>0
    assert reader.feature_projection.weight.grad is not None
    assert reader.attn.in_proj_weight.grad is not None


def test_reader_equal_shapes_and_state_dict_independent_of_action_chunk():
    torch.manual_seed(42)
    reader,q,ctx=reader_fixture()
    reader.eval()
    with torch.no_grad():
        first=reader(q,ctx)
        chunk_state=reader.state_dict()
        reader.cfg=GVARConfig(variant="volume_rel",reader_dim=16,reader_heads=4,
                            action_chunk=2,activation_checkpoint=False)
        second=reader(q,ctx)
        torch.testing.assert_close(first,second,atol=1e-5,rtol=1e-5)
        assert all(torch.equal(v,reader.state_dict()[k]) for k,v in chunk_state.items())


def test_scene_metric_not_frame_iid():
    a=np.full((30,26,50,6),.5,dtype=np.float32)
    a[0]+=0.1
    scores=scene_scores(a)
    assert scores.shape==(30,)
    assert float(scores[0])==pytest.approx(.6,abs=1e-5)
    assert scene_scores(a,.4).shape==(30,)
    with pytest.raises(ValueError):scene_scores(a[:29])


def test_new_entrypoint_bash_syntax():
    root=Path(__file__).resolve().parents[1]
    for name in ("run_rayrope_point_gvar_mixed_v2.sh",
                 "run_rayrope_point_gvar_suite.sh"):
        result=subprocess.run(["bash","-n",str(root/"scripts"/name)],
                              text=True,capture_output=True)
        assert result.returncode==0,result.stderr


def test_new_fingerprint_is_separate_from_old_moge_rayrope():
    from moge_rayrope.runtime import code_fingerprint as old_fingerprint
    old=old_fingerprint()
    new=code_fingerprint()
    assert len(old)==64 and len(new)==64 and old!=new


def test_cdf_gvar_decoder_tensor_contract_and_reader_gradients():
    """Exercise actual imported CDF decoder/reader path on synthetic CPU tensors."""
    from models.kview_query_transformer import CenterViewAngleCDFCandidateDecoder
    from rayrope_gvar.decoder import GripperVolumeCDFDecoder
    torch.manual_seed(17)
    Q,A,D=2,2,4
    original=CenterViewAngleCDFCandidateDecoder(
        num_angle=A,num_depth=D,in_dim=128,hidden_dim=32,
        num_layers=1,num_heads=4)
    cfg=GVARConfig(variant="volume_rel",reader_dim=16,reader_heads=4,
                   action_chunk=8,activation_checkpoint=False)
    decoder=GripperVolumeCDFDecoder(original,feat_dim=8,config=cfg)
    decoder.train()
    group=torch.randn(1,128,Q*A,requires_grad=True)
    depth=torch.full((1,1,64,64),.52,requires_grad=True)
    visual=torch.randn(1,8,16,16,requires_grad=True)
    context={"pregeom":visual,"centers":torch.tensor(
                  [[[0.,0.,.5],[.03,0.,.5],[.0,.02,.5],[-.03,0.,.5]]]),
             "rotations":torch.eye(3).reshape(1,1,3,3).expand(1,Q*A,3,3).clone(),
             "K":torch.tensor([[[160.,0.,31.5],[0.,160.,31.5],[0.,0.,1.]]]),
             "depth":depth,"image_hw":(64,64)}
    ep={"kview_angle_query_base_q":Q,"kview_angle_query_num_angle":A,
        "_gvar_context":context}
    out=decoder(group,ep)
    assert "_gvar_context" not in out
    w=out["grasp_width_pred_angle_depth"]
    c=out["grasp_cdf_pred_angle_depth"]
    assert w.shape==(1,D,Q,A) and c.shape==(1,6,Q,A,D)
    assert torch.isfinite(w).all() and torch.isfinite(c).all()
    assert (c[:,1:]-c[:,:-1] >= -1e-6).all()
    (w.mean()+c.mean()).backward()
    assert visual.grad is not None and float(visual.grad.abs().sum())>0
    assert group.grad is not None and float(group.grad.abs().sum())>0
    assert depth.grad is None
