from __future__ import annotations

import torch

from utils.p5_ops import (
    backproject_token_depth,
    build_gripper_keypoints_local,
    build_repair_targets,
    repair_loss_sums,
    structured_depth_corruption,
)


def test_zero_probability_corruption_is_exact_identity():
    depth = torch.full((2, 1, 8, 9), 0.5)
    out, diag = structured_depth_corruption(
        depth, min_depth=0.2, max_depth=1.0, probability=0.0,
        scene_bias_sigma_m=0.02, scale_sigma=0.03, region_sigma_m=0.02,
    )
    assert torch.equal(out, depth)
    assert diag["abs_mean_m"].item() == 0.0


def test_backprojection_and_gripper_keypoints_shapes():
    idx = torch.tensor([[4 * 10 + 5, 2 * 10 + 3]])
    z = torch.tensor([[0.5, 0.7]])
    K = torch.tensor([[[100.0, 0.0, 5.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]])
    xyz = backproject_token_depth(idx, z, K, 10)
    assert xyz.shape == (1, 2, 3)
    assert torch.allclose(xyz[..., 2], z)
    pts = build_gripper_keypoints_local(torch.tensor([[0.04, 0.08]]), torch.tensor([[0.02, 0.04]]))
    assert pts.shape == (1, 2, 11, 3)
    assert torch.equal(pts[:, :, 0], torch.zeros(1, 2, 3))


def test_repair_target_requires_same_action_positive_and_never_marks_unknown_negative():
    center = torch.zeros(1, 3, 3)
    nearest = center.clone()
    nearest[0, 0, 2] = 0.02   # nearby + positive
    nearest[0, 1, 2] = 0.02   # nearby but selected op has no positive label
    nearest[0, 2, 2] = 0.20   # positive label but outside repair radius
    R = torch.eye(3).view(1, 1, 3, 3).expand(1, 3, 3, 3).clone()
    bins = torch.zeros(1, 3, 2, 2, dtype=torch.long)
    bins[0, 0, 0, 0] = 1
    bins[0, 2, 0, 0] = 1
    angle = torch.zeros(1, 3, dtype=torch.long)
    depth = torch.zeros(1, 3, dtype=torch.long)
    t = build_repair_targets(
        proposal_center=center, rotation=R, nearest_label_center=nearest,
        cdf_bins=bins, angle_idx=angle, depth_idx=depth,
        num_thresholds=6, target_radius_m=0.06,
    )
    assert torch.equal(t["target_known"], torch.tensor([[True, False, False]]))
    assert t["same_action_positive"][0, 1].item() is False
    # Unknown samples do not receive a fabricated utility target in the loss;
    # they are excluded by target_known and only weak identity regularization applies.
    pred = torch.zeros_like(t["target_delta_local"], requires_grad=True)
    losses = repair_loss_sums(pred, t, beta_m=0.005, unknown_identity_weight=0.02)
    assert losses["repair"][1].item() == 1
    assert losses["unknown_identity"][1].item() == 2
    assert torch.isfinite(losses["repair"][0])


def test_corruption_is_spatially_correlated_and_bounded():
    torch.manual_seed(0)
    depth = torch.full((1, 1, 32, 32), 0.5)
    out, _ = structured_depth_corruption(
        depth, min_depth=0.2, max_depth=1.0, probability=1.0,
        scene_bias_sigma_m=0.01, scale_sigma=0.02, region_sigma_m=0.015, region_grid=5,
    )
    assert out.shape == depth.shape
    assert torch.all((out >= 0.2) & (out <= 1.0))
    assert not torch.equal(out, depth)
    # Smooth field should not behave like independent pixel noise.
    delta = out - depth
    adjacent = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
    shuffled = (delta.flatten()[1:] - delta.flatten()[:-1].flip(0)).abs().mean()
    assert adjacent < shuffled
