"""A2 RGB-to-point sampler/fuser contract tests (CPU; no MinkowskiEngine)."""

import pytest
import torch
from torch import nn

from models.point_dino_rgb_fusion import (
    MultiLevelDinoPointSampler,
    ResidualPointRGBFusion,
)


def test_four_level_feature_dimensions_and_gradients():
    torch.manual_seed(17)
    B, N, Hp, Wp, C = 2, 23, 4, 4, 12
    sampler = MultiLevelDinoPointSampler(
        feature_dim=C, levels=4, per_level_dim=5,
    )
    levels = [(torch.randn(B, Hp * Wp, C, requires_grad=True),
               torch.randn(B, C)) for _ in range(4)]
    idxs = torch.randint(0, (Hp * 14) * (Wp * 14), (B, N))
    output = sampler(levels, idxs, image_hw=(Hp * 14, Wp * 14))
    assert output.shape == (B, 20, N)
    output.square().mean().backward()
    # Freeze all DINO feature sources regardless of their incoming flags.
    assert all(feat[0].grad is None for feat in levels)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in sampler.parameters()
    )


def test_pixel_coordinate_to_patch_alignment_at_corners():
    sampler = MultiLevelDinoPointSampler(
        feature_dim=4, levels=4, per_level_dim=2,
    )
    for i in range(4):
        linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            linear.weight.zero_()
            linear.weight[0, 0] = 1.0
            linear.weight[1, 1] = 1.0
        sampler.projections[i] = linear
    # [top left, top right, bottom left, bottom right] in a 2x2 patch grid.
    patch = torch.tensor([[
        [1., 11., 0., 0.],
        [2., 12., 0., 0.],
        [3., 13., 0., 0.],
        [4., 14., 0., 0.],
    ]])
    # image H=W=28. Coordinates at corners should clamp to corresponding
    # edge patch without switching X/Y or swapping rows.
    idxs = torch.tensor([[0, 27, 27 * 28, 28 * 28 - 1]])
    out = sampler([patch] * 4, idxs, image_hw=(28, 28))
    assert out.shape == (1, 8, 4)
    expected = torch.tensor([
        [1., 2., 3., 4.],
        [11., 12., 13., 14.],
    ])
    for i in range(4):
        torch.testing.assert_close(out[0, 2*i:2*(i+1)], expected)


def test_point_fuser_is_exact_identity_at_initialization():
    torch.manual_seed(4)
    fuser = ResidualPointRGBFusion(
        point_dim=16, rgb_dim=8, hidden_dim=12,
    )
    geom = torch.randn(2, 16, 41, requires_grad=True)
    rgb = torch.randn(2, 8, 41, requires_grad=True)
    output = fuser(geom, rgb)
    torch.testing.assert_close(output, geom, atol=0, rtol=0)
    loss = output.square().mean()
    loss.backward()
    assert fuser.fusion[-1].weight.grad is not None
    assert fuser.fusion[-1].weight.grad.abs().sum() > 0
    assert rgb.grad is not None  # zero gradient on first step is expected
    assert torch.count_nonzero(rgb.grad) == 0


def test_sampler_feature_failure_modes():
    sampler = MultiLevelDinoPointSampler(
        feature_dim=8, levels=4, per_level_dim=2,
    )
    feat = [torch.randn(1, 4, 8)] * 4
    with pytest.raises(ValueError, match="four"):
        sampler(feat[:3], torch.tensor([[0]]), image_hw=(28, 28))
    with pytest.raises(ValueError, match="outside"):
        sampler(feat, torch.tensor([[28 * 28]]), image_hw=(28, 28))
    with pytest.raises(ValueError, match="expected"):
        sampler([torch.randn(1, 2, 8)] * 4,
                torch.tensor([[1]]), image_hw=(28, 28))
    with pytest.raises(TypeError, match="integer"):
        sampler(feat, torch.tensor([[0.0]]), image_hw=(28, 28))


def test_fuser_rejects_misaligned_point_counts():
    fusion = ResidualPointRGBFusion(point_dim=16, rgb_dim=8)
    with pytest.raises(ValueError, match="Expected"):
        fusion(torch.randn(1, 16, 10), torch.randn(1, 8, 9))
