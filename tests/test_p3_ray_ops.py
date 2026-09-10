from __future__ import annotations

import torch

from utils.p3_ray_ops import (
    build_ray_centers,
    compact_cdf_utility,
    final_indices,
    loss_sums,
)


def test_zero_offset_center_is_exact_identity():
    base = torch.tensor([[[0.1, -0.2, 0.50], [0.0, 0.0, 0.70]]], dtype=torch.float32)
    idx = torch.tensor([[10 * 20 + 11, 8 * 20 + 9]], dtype=torch.long)
    K = torch.tensor([[[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]])
    offsets = torch.tensor([-0.02, 0.0, 0.02])
    centers, valid, desc = build_ray_centers(base, idx, K, 20, offsets, 0.2, 1.0)
    assert centers.shape == (1, 2, 3, 3)
    assert valid.all()
    assert desc.shape == (1, 2, 3, 5)
    assert torch.equal(centers[:, :, 1], base)
    assert torch.allclose(centers[:, :, 2, 2] - centers[:, :, 0, 2], torch.full((1, 2), 0.04))


def test_compact_cdf_utility_semantics():
    bins = torch.tensor([0, 1, 2, 3])
    utility = compact_cdf_utility(bins, 3)
    assert torch.allclose(utility, torch.tensor([0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]))


def _synthetic_endpoints():
    B, T, Q, Kd, A, D = 1, 3, 2, 3, 2, 2
    cdf = torch.full((B, T, Q, Kd, A, D), -4.0)
    # For q0 make k2,a1,d0 strongest; for q1 make k1,a0,d1 strongest.
    cdf[:, :, 0, 2, 1, 0] = 4.0
    cdf[:, :, 1, 1, 0, 1] = 3.0
    viability = torch.zeros(B, Q, Kd)
    viability[:, 0, 2] = 3.0
    viability[:, 1, 1] = 3.0
    bins = torch.zeros(B, Q, Kd, A, D, dtype=torch.long)
    valid = torch.zeros_like(bins, dtype=torch.bool)
    bins[:, 0, 2, 1, 0] = 1
    bins[:, 1, 1, 0, 1] = 2
    valid[:, 0, 2, 1, 0] = True
    valid[:, 1, 1, 0, 1] = True
    point = torch.zeros(B, Q, Kd, dtype=torch.bool)
    point[:, 0, 2] = True
    point[:, 1, 1] = True
    return {
        "p3_cdf_logits": cdf,
        "p3_width_pred": torch.zeros(B, D, Q, Kd, A),
        "p3_viability_logits": viability,
        "p3_cdf_bins": bins,
        "p3_cdf_valid": valid,
        "p3_width_label": torch.zeros(B, Q, Kd, A, D),
        "p3_width_valid": valid,
        "p3_point_support": point,
        "p3_point_known": torch.ones(B, Q, Kd, dtype=torch.bool),
        "p3_in_range": torch.ones(B, Q, Kd, dtype=torch.bool),
        "p3_offsets_m": torch.tensor([-0.01, 0.0, 0.01]),
    }


def test_joint_final_decode_is_over_k_angle_depth():
    ep = _synthetic_endpoints()
    k, a, d, score, utility = final_indices(ep, score_mode="joint")
    assert utility.shape == (1, 2, 3, 2, 2)
    assert torch.equal(k, torch.tensor([[2, 1]]))
    assert torch.equal(a, torch.tensor([[1, 0]]))
    assert torch.equal(d, torch.tensor([[0, 1]]))
    assert torch.all(score > 0)


def test_p3_loss_statistics_are_finite_and_masked():
    ep = _synthetic_endpoints()
    stats = loss_sums(ep)
    assert set(stats) == {"cdf", "width", "viability_pos", "viability_neg", "joint_pos", "joint_neg"}
    for total, count in stats.values():
        assert torch.isfinite(total)
        assert torch.isfinite(count)
        assert count >= 0
    assert stats["cdf"][1].item() == 2 * 3  # two valid KAD operations x T thresholds
    assert stats["viability_pos"][1].item() == 2
    assert stats["viability_neg"][1].item() == 4
