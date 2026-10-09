"""Unit controls for P1 paired-depth policy and confidence-loss gradients.

Run: python -m pytest -q tests/test_rayrope_mixed_p1.py
Full DDP/CUDA forward, GraspNet paths and AP require the robot server.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from moge_rayrope.config import ModelConfig
from moge_rayrope.objective import laplace_depth_objective


def test_laplace_detach_vs_joint_depth_gradient():
    gt = torch.tensor([[[[.5, .6]]]])
    valid = torch.ones_like(gt, dtype=torch.bool)
    h = torch.tensor([[[[.02, .03]]]], requires_grad=True)
    d = torch.tensor([[[[.55, .7]]]], requires_grad=True)
    loss = laplace_depth_objective(d, h, gt, valid, detach_mean=True)
    loss.backward()
    assert d.grad is None
    assert h.grad is not None and torch.isfinite(h.grad).all()
    h2 = h.detach().clone().requires_grad_()
    d2 = d.detach().clone().requires_grad_()
    coupled = laplace_depth_objective(
        d2, h2, gt, valid, detach_mean=False, valid_normalize=False)
    coupled.backward()
    assert torch.isfinite(d2.grad).all() and torch.any(d2.grad.abs() > 0)
    assert torch.isfinite(h2.grad).all()


def test_laplace_normalization_initial_scale():
    gt = torch.tensor([[[[.5, .7]]]])
    d = torch.tensor([[[[.6, .8]]]], requires_grad=True)
    h = torch.full_like(d, .02).detach()
    valid = torch.ones_like(gt, dtype=torch.bool)
    out = laplace_depth_objective(d, h, gt, valid, detach_mean=False,
                                  valid_normalize=False)
    out.backward()
    # At h=.02 the coupled depth derivative matches the original L1.
    assert torch.allclose(d.grad, torch.tensor([[[[.5, .5]]]]), atol=1e-6)


def test_modes_are_disjoint():
    ModelConfig(ray_encoding="point", uncertainty="fixed")
    ModelConfig(use_rayrope=True, ray_encoding="expected", uncertainty="learned",
                uncertainty_loss="laplace_decoupled")
    ModelConfig(use_rayrope=True, ray_encoding="expected", uncertainty="learned",
                uncertainty_loss="laplace_joint")
    with pytest.raises(ValueError):
        ModelConfig(ray_encoding="point", uncertainty="fixed",
                    uncertainty_loss="laplace_joint")
    with pytest.raises(ValueError):
        ModelConfig(use_rayrope=True, ray_encoding="point", uncertainty="learned")


def test_full_tsdf_real_vs_full_rendered_gntrans(tmp_path):
    from moge_rayrope.mixed import RealSenseFullTSDF, GNTransFullRendered
    raw = np.array([[200, 600], [800, 1000]], dtype=np.uint16)
    tsdf = np.array([[300, 500], [900, 1100]], dtype=np.uint16)
    label = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    target = tmp_path / "tsdf.png"
    Image.fromarray(tsdf).save(target)
    real = RealSenseFullTSDF.__new__(RealSenseFullTSDF)
    real.use_fuse_depth = True
    real.gt_factor_depth = None
    real.fusedepthpath = [str(target)]
    got_r = real.build_fused_gt_depth_m(raw, label, 0, 1000.)
    assert np.allclose(got_r, tsdf / 1000.)
    # Foreground TSDF must *not* silently change back to virtual GT.
    assert not np.isclose(got_r[0, 0], raw[0, 0]/1000.)
    gn = GNTransFullRendered.__new__(GNTransFullRendered)
    gn.use_fuse_depth = False
    gn.gt_factor_depth = None
    got_t = gn.build_fused_gt_depth_m(raw, label, 0, 1000.)
    assert np.allclose(got_t, raw / 1000.)
    gn.use_fuse_depth = True
    with pytest.raises(RuntimeError):
        gn.build_fused_gt_depth_m(raw, label, 0, 1000.)


def test_paired_indices_same_scene_frame():
    from moge_rayrope.mixed import paired_frame_indices
    class Fake:
        scenename = ["scene_0000"] * 20
        frameid = list(range(20))
    a, b, schedule = paired_frame_indices(Fake(), Fake(), .1)
    assert a == [0, 10] and b == [0, 10] and schedule == [[0, 0], [0, 10]]
    class Bad(Fake):
        frameid = list(range(1, 21))
    with pytest.raises(RuntimeError):
        paired_frame_indices(Fake(), Bad(), .1)
