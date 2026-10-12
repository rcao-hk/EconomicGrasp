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


def test_rendered_foreground_tsdf_background_and_full_rendered_gntrans(tmp_path):
    from moge_rayrope.mixed import (
        RealSenseRenderedForegroundTSDFBackground, GNTransFullRendered
    )
    rendered = np.array([[200, 600, 0], [800, 1000, 400]], dtype=np.uint16)
    tsdf = np.array([[300, 500, 700], [900, 1100, 0]], dtype=np.uint16)
    label = np.array([[1, 0, 1], [1, 0, 0]], dtype=np.uint8)
    target = tmp_path / "tsdf.png"
    Image.fromarray(tsdf).save(target)

    real = RealSenseRenderedForegroundTSDFBackground.__new__(
        RealSenseRenderedForegroundTSDFBackground)
    real.use_fuse_depth = True
    real.gt_factor_depth = None
    real.fusedepthpath = [str(target)]
    real.labelpath = ["label.png"]
    gt_real = real.build_fused_gt_depth_m(rendered, label, 0, 1000.)
    expected = np.array([[.2, .5, 0.], [.8, 1.1, 0.]], dtype=np.float32)
    assert np.allclose(gt_real, expected)
    # Full TSDF is forbidden in object pixels; rendered background is forbidden
    # in background pixels. Invalid source pixels must remain invalid.
    assert not np.isclose(gt_real[0, 0], tsdf[0, 0] / 1000.)
    assert not np.isclose(gt_real[0, 1], rendered[0, 1] / 1000.)
    assert gt_real[0, 2] == 0. and gt_real[1, 2] == 0.
    real.use_fuse_depth = False
    with pytest.raises(RuntimeError):
        real.build_fused_gt_depth_m(rendered, label, 0, 1000.)

    gn = GNTransFullRendered.__new__(GNTransFullRendered)
    gn.use_fuse_depth = False
    gn.gt_factor_depth = None
    gt_gn = gn.build_fused_gt_depth_m(rendered, label, 0, 1000.)
    assert np.allclose(gt_gn, rendered / 1000.)
    gn.use_fuse_depth = True
    with pytest.raises(RuntimeError):
        gn.build_fused_gt_depth_m(rendered, label, 0, 1000.)


def test_real_requires_both_rendered_and_tsdf_sources(tmp_path):
    from moge_rayrope.mixed import check_depth_sources
    source = tmp_path / "present.png"
    Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(source)
    missing = tmp_path / "missing_rendered_depth.png"

    class Fake:
        colorpath = [str(source)]
        gtdepthpath = [str(source)]
        fusedepthpath = [str(source)]
        labelpath = [str(source)]
        metapath = [str(source)]

    rs, gn = Fake(), Fake()
    check_depth_sources(rs, gn, [0], [0])
    rs.gtdepthpath = [str(missing)]
    with pytest.raises(FileNotFoundError, match="missing_rendered_depth"):
        check_depth_sources(rs, gn, [0], [])


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
