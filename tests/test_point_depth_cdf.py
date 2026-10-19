"""CPU-level point-CDF contract tests; no dataset, CUDA or Minkowski needed.

Run on the configured robotarm environment:
  python -m pytest -q tests/test_point_depth_cdf.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.point_grasp_cdf_head import PointGraspCDFHead,decode_point_cdf


def _dummy(B=2,Q=8,A=12,D=4,T=6):
    centers=torch.zeros(B,Q,3)
    centers[...,2]=0.5
    views=torch.zeros_like(centers)
    views[...,2]=-1
    return {
        "xyz_graspable":centers,
        "grasp_top_view_xyz":views,
        "grasp_cdf_pred_angle_depth":torch.full((B,T,Q,A,D),-8.0),
        "grasp_width_pred_angle_depth":torch.full((B,D,Q,A),0.4),
    }


def test_cdf_head_shape_and_gradients():
    head=PointGraspCDFHead(in_channels=256,hidden_channels=64,
                           num_angle=12,num_depth=4,num_thresholds=6)
    x=torch.randn(2,256,8,requires_grad=True)
    output=head(x,{})
    assert output["grasp_cdf_pred_angle_depth"].shape==(2,6,8,12,4)
    assert output["grasp_width_pred_angle_depth"].shape==(2,4,8,12)
    y=output["grasp_cdf_pred_angle_depth"].square().mean()
    y=y+output["grasp_width_pred_angle_depth"].square().mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.cdf.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.width.parameters())


def test_decoder_joint_angle_depth_and_width_units():
    state=_dummy(B=2,Q=8)
    # High CDF utility at exactly one action among all 12x4.
    state["grasp_cdf_pred_angle_depth"][:,:, :,3,2]=8.0
    decoded=decode_point_cdf(state,num_angle=12,num_depth=4,max_width=0.1)
    assert len(decoded)==2
    for g in decoded:
        assert g.shape==(8,17)
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g[:,0],torch.sigmoid(torch.tensor(8.0)).expand(8))
        torch.testing.assert_close(g[:,1],torch.full((8,),0.048))
        torch.testing.assert_close(g[:,2],torch.full((8,),0.02))
        torch.testing.assert_close(g[:,3],torch.full((8,),0.03))
        torch.testing.assert_close(g[:,13:16],state["xyz_graspable"][0])
        assert (g[:,-1]==-1).all()


def test_decoder_invalid_shapes_and_values():
    state=_dummy(B=1,Q=3)
    state["grasp_width_pred_angle_depth"]=torch.zeros(1,3,3,12)
    with pytest.raises(ValueError,match="Width tensor shape mismatch"):
        decode_point_cdf(state)
    state=_dummy(B=1,Q=3)
    state["grasp_cdf_pred_angle_depth"][0,0,0,0,0]=float("nan")
    with pytest.raises(FloatingPointError,match="Non-finite"):
        decode_point_cdf(state)


def test_decoder_more_than_one_angle_depth_combination():
    state=_dummy(B=1,Q=2)
    state["grasp_cdf_pred_angle_depth"][:, :,0,1,0]=9.0
    state["grasp_cdf_pred_angle_depth"][:, :,1,11,3]=9.0
    decoded=decode_point_cdf(state)[0]
    torch.testing.assert_close(decoded[:,3],torch.tensor([0.01,0.04]))
    assert not torch.allclose(decoded[0,4:13],decoded[1,4:13])


def test_head_does_not_modify_legacy_outputs():
    head=PointGraspCDFHead()
    state={"grasp_angle_pred":"legacy"}
    out=head(torch.randn(1,256,2),state)
    assert out["grasp_angle_pred"]=="legacy"
