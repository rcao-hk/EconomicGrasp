from __future__ import annotations

import torch

from utils.p5_v11_ops import (
    prepare_repair_to_set_targets,
    repair_to_set_loss_sums,
    repair_to_set_metric_sums,
)


def _base_targets():
    # q0 is already safe (3 mm), q1 is repairable (20 mm), q2 unknown.
    delta = torch.tensor([[[0.003, 0.0, 0.0], [0.020, 0.0, 0.0], [0.04, 0.0, 0.0]]])
    return {
        "target_delta_local": delta,
        "target_known": torch.tensor([[True, True, False]]),
        "target_distance_m": torch.tensor([[0.003, 0.020, 0.040]]),
        "target_utility": torch.tensor([[1.0, 2.0 / 3.0, 0.0]]),
    }


def test_safe_zone_has_identity_target_not_annotation_snap():
    t = prepare_repair_to_set_targets(_base_targets(), safe_radius_m=0.005)
    assert torch.equal(t["target_safe"], torch.tensor([[True, False, False]]))
    assert torch.equal(t["target_repairable"], torch.tensor([[False, True, False]]))
    assert torch.equal(t["target_unknown"], torch.tensor([[False, False, True]]))
    assert torch.allclose(t["target_delta_local"][0, 0], torch.zeros(3))
    assert torch.allclose(t["target_delta_local"][0, 1], torch.tensor([0.020, 0.0, 0.0]))
    # Raw target is retained for valid-set diagnostics.
    assert torch.allclose(t["raw_target_delta_local"][0, 0], torch.tensor([0.003, 0.0, 0.0]))


def test_loss_partitions_repair_safe_and_unknown():
    t = prepare_repair_to_set_targets(_base_targets(), safe_radius_m=0.005)
    pred = torch.zeros(1, 3, 3)
    sums = repair_to_set_loss_sums(pred, t, beta_m=0.005)
    assert sums["repair"][1].item() == 1
    assert sums["safe_identity"][1].item() == 1
    assert sums["unknown_identity"][1].item() == 1
    assert sums["safe_identity"][0].item() == 0.0
    assert sums["repair"][0].item() > 0.0


def test_set_metrics_reward_capture_without_moving_safe_query():
    t = prepare_repair_to_set_targets(_base_targets(), safe_radius_m=0.005)
    pred = torch.tensor([[[0.0, 0.0, 0.0], [0.018, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(1, 3, 3, 3).clone()
    proposal = torch.zeros(1, 3, 3)
    m = repair_to_set_metric_sums(pred, rotation, proposal, t)
    native_v = m["p5v11_native_set_violation_m"][0] / m["p5v11_native_set_violation_m"][1]
    repaired_v = m["p5v11_repaired_set_violation_m"][0] / m["p5v11_repaired_set_violation_m"][1]
    assert repaired_v < native_v
    noharm = m["p5v11_safe_noharm_ratio"][0] / m["p5v11_safe_noharm_ratio"][1]
    capture = m["p5v11_repairable_capture_ratio"][0] / m["p5v11_repairable_capture_ratio"][1]
    assert torch.allclose(noharm, torch.tensor(1.0))
    assert torch.allclose(capture, torch.tensor(1.0))
