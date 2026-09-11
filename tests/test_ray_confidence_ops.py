from __future__ import annotations

import torch

from utils.ray_confidence_ops import (
    build_confidence_targets,
    build_ray_centers,
    confidence_loss_sums,
    raw_cdf_utility,
    select_raw_operation,
)


def test_zero_center_is_exact_identity():
    base = torch.tensor([[[0.1, -0.2, 0.50], [0.0, 0.0, 0.70]]], dtype=torch.float32)
    idx = torch.tensor([[10 * 20 + 11, 8 * 20 + 9]], dtype=torch.long)
    K = torch.tensor([[[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]])
    offsets = torch.tensor([-0.02, 0.0, 0.02])
    centers, valid, desc = build_ray_centers(base, idx, K, 20, offsets, 0.2, 1.0)
    assert centers.shape == (1, 2, 3, 3)
    assert valid.all()
    assert desc.shape == (1, 2, 3, 5)
    assert torch.equal(centers[:, :, 1], base)


def test_native_operation_selection_uses_only_zero_cdf():
    logits = torch.full((1, 3, 2, 2, 2), -4.0)
    logits[:, :, 0, 1, 0] = 4.0
    logits[:, :, 1, 0, 1] = 3.0
    a, d, score, utility = select_raw_operation(logits)
    assert utility.shape == (1, 2, 2, 2)
    assert torch.equal(a, torch.tensor([[1, 0]]))
    assert torch.equal(d, torch.tensor([[0, 1]]))
    assert torch.all(score > 0.9)
    assert torch.allclose(raw_cdf_utility(logits), utility)


def test_confidence_target_keeps_unlabelled_surface_unknown():
    bins = torch.zeros((1, 3, 2, 2), dtype=torch.long)
    valid = torch.zeros_like(bins, dtype=torch.bool)
    # q0: selected op valid and good; q1: known off surface; q2: on surface but selected op missing.
    bins[0, 0, 1, 0] = 1
    valid[0, 0, 1, 0] = True
    point_support = torch.tensor([[True, False, True]])
    point_known = torch.tensor([[True, True, True]])
    angle = torch.tensor([[1, 0, 0]])
    depth = torch.tensor([[0, 0, 0]])
    raw = torch.tensor([[0.8, 0.7, 0.9]])
    target = build_confidence_targets(
        bins, valid, point_support, point_known, angle, depth, raw, num_thresholds=3
    )
    assert torch.equal(target["target_known"], torch.tensor([[True, True, False]]))
    assert torch.allclose(target["target_score"][:, :2], torch.tensor([[1.0, 0.0]]))


def test_calibration_loss_has_gradients_only_through_confidence():
    raw = torch.tensor([[0.8, 0.7, 0.6]], requires_grad=False)
    logit = torch.zeros_like(raw, requires_grad=True)
    target = {
        "target_score": torch.tensor([[1.0, 0.0, 0.5]]),
        "target_known": torch.tensor([[True, True, True]]),
        "target_positive": torch.tensor([[True, False, True]]),
        "ideal_gate": torch.tensor([[1.0, 0.0, 0.8]]),
    }
    sums = confidence_loss_sums(raw, logit, target, rank_temperature=0.1)
    loss = sum(total / count.clamp_min(1) for total, count in sums.values())
    loss.backward()
    assert logit.grad is not None
    assert torch.isfinite(logit.grad).all()
