import sys,types
import numpy as np
import pytest
import torch
from torch import nn
from moge_rayrope.config import ModelConfig
from moge_rayrope.model import FactorizedMetricDepth
from moge_rayrope.metrics import EpochMetrics

def test_shape_head_keeps_gradient_on_negative_hidden_features(monkeypatch):
    class Head(nn.Module):
        def __init__(self,**kw):
            super().__init__(); self.scratch=nn.Module()
            self.scratch.output_conv2=nn.Sequential(nn.Identity(),nn.ReLU(),nn.Identity())
    fake=types.ModuleType('models.dinov2_dpt'); fake.DPTHead=Head
    monkeypatch.setitem(sys.modules,'models.dinov2_dpt',fake)
    old=nn.Module(); old.depthnet=nn.Module(); old.depthnet.pretrained=nn.Linear(1,1); old.stride=1
    model=FactorizedMetricDepth(old,ModelConfig(use_moge=True))
    x=torch.full((1,32,2,2),-1.,requires_grad=True)
    y=model.depthnet.depth_head.scratch.output_conv2(x); y.sum().backward()
    assert bool((y<0).all()) and bool((x.grad>0).all())

def test_nonfinite_metrics_name_the_bad_field():
    stat=EpochMetrics();stat.images=1;stat.v[8]=np.inf;stat.v[9]=1
    with pytest.raises(FloatingPointError,match='depth_mae_m=inf'): stat.report()
    stat.v[8]=0;stat.loss_sums['shape_gauge_mean']=np.inf
    with pytest.raises(FloatingPointError,match='scalars.shape_gauge_mean=inf'): stat.report()


def test_large_finite_pointmap_preserves_gauge_and_gradient():
    from moge_rayrope.geometry import image_rays,canonicalize_points
    K=torch.tensor([[[32.,0.,15.5],[0.,32.,15.5],[0.,0.,1.]]])
    rays=image_rays(K,(32,32)); z=torch.linspace(.3,.8,1024).reshape(1,1,32,32)
    p=rays*z; expected,_,_=canonicalize_points(p,rays)
    large=(p*1e20).requires_grad_()
    actual,shift,gauge=canonicalize_points(large,rays)
    assert torch.isfinite(gauge).all() and torch.isfinite(shift).all()
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
    actual.square().mean().backward()
    assert torch.isfinite(large.grad).all() and bool((large.grad!=0).any())

def test_zero_pointmap_floor_has_finite_backward():
    from moge_rayrope.geometry import image_rays,canonicalize_points
    K=torch.tensor([[[8.,0.,3.5],[0.,8.,3.5],[0.,0.,1.]]]); rays=image_rays(K,(8,8))
    p=torch.zeros_like(rays,requires_grad=True);c,_,g=canonicalize_points(p,rays)
    c.sum().backward();assert torch.isfinite(p.grad).all()
    torch.testing.assert_close(g,torch.full_like(g,.001))
