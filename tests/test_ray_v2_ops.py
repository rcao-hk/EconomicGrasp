from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_ops():
    path = Path(__file__).resolve().parents[1] / "models" / "ray_grasp_v2_ops.py"
    spec = importlib.util.spec_from_file_location("ray_grasp_v2_ops_standalone", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _part(k: int):
    B, M, C, A, D, T = 1, 2, 4, 2, 2, 3
    cdf = torch.zeros(B, T, M, A, D)
    cdf[:, :, :, 0, 0] = float(k) * 0.25
    bins = torch.zeros(B, M, A, D, dtype=torch.long)
    valid = torch.zeros_like(bins, dtype=torch.bool)
    valid[:, :, 0, 0] = True
    # k=1 is the strongest valid depth for both image rays.
    if k == 1:
        bins[:, :, 0, 0] = 1
    elif k == 0:
        bins[:, :, 0, 0] = 2
    return {
        "ray_context_feature": torch.full((B, M, C), float(k)),
        "ray_descriptor": torch.zeros(B, M, 5),
        "ray_support_logits": torch.zeros(B, M),
        "grasp_cdf_pred_angle_depth": cdf,
        "ray_in_range": torch.ones(B, M, dtype=torch.bool),
        "batch_grasp_cdf_bins_angle_depth": bins,
        "batch_grasp_cdf_valid_mask": valid,
        "batch_grasp_cdf_thresholds": torch.tensor([0.2, 0.4, 0.6]),
        "ray_support_target": torch.ones(B, M, dtype=torch.bool),
        "ray_support_known": torch.ones(B, M, dtype=torch.bool),
    }


def test_v2_input_target_and_listwise_contract():
    ops = _load_ops()
    parts = [_part(0), _part(1), _part(2)]
    features, in_range = ops.build_selector_inputs(parts)
    assert features.shape == (1, 2, 3, 13)  # C=4 plus nine scalar/ray features
    assert in_range.all()

    targets = ops.build_selector_targets(parts)
    assert targets["target_utility"].shape == (1, 2, 3)
    assert torch.equal(targets["target_utility"].argmax(-1), torch.ones(1, 2, dtype=torch.long))

    logits = torch.tensor([[[0.0, 2.0, -1.0], [0.0, 2.0, -1.0]]], requires_grad=True)
    sums = ops.selector_loss_sums(logits, targets, target_temperature=0.1)
    loss = sums["listwise"][0] / sums["listwise"][1].clamp_min(1)
    loss = loss + sums["calib_pos"][0] / sums["calib_pos"][1].clamp_min(1)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    metrics = ops.selector_metric_sums(logits.detach(), targets, torch.tensor([-0.01, 0.0, 0.01]))
    hit_sum, hit_count = metrics["v2_oracle_hit"]
    assert float(hit_sum) == float(hit_count) == 2.0
