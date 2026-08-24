#!/usr/bin/env python3
"""Synthetic contract test for Exact-Query Oracle-Selective CDF KD."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_OVERLAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_OVERLAY))

from models.oracle_selective_cdf_kd import (  # noqa: E402
    build_oracle_selective_cdf_gate,
    compute_oracle_selective_cdf_distillation_loss,
)


def _base_endpoints(student_logits: torch.Tensor):
    device = student_logits.device
    # B=1, T=2, Q=3, A=1, D=1.
    indices = torch.tensor([[10, 20, 30]], device=device)
    views = torch.tensor([[1, 2, 3]], device=device)
    return {
        "grasp_cdf_pred_angle_depth": student_logits,
        "batch_grasp_cdf_bins_angle_depth": torch.tensor(
            [[[[1]], [[1]], [[1]]]],
            device=device,
            dtype=torch.long,
        ),
        "batch_grasp_cdf_valid_mask": torch.tensor(
            [[[[1]], [[1]], [[1]]]],
            device=device,
            dtype=torch.bool,
        ),
        "kview_base_token_sel_idx": indices.clone(),
        "token_sel_idx": indices.clone(),
        "grasp_top_view_inds": views.clone(),
        "grasp_top_view_xyz": torch.ones(1, 3, 3, device=device),
    }


def _teacher_targets(teacher_logits: torch.Tensor):
    device = teacher_logits.device
    indices = torch.tensor([[10, 20, 30]], device=device)
    views = torch.tensor([[1, 2, 3]], device=device)
    return {
        "grasp_cdf_pred_angle_depth": teacher_logits.detach(),
        "batch_grasp_cdf_bins_angle_depth": torch.tensor(
            [[[[1]], [[1]], [[1]]]],
            device=device,
            dtype=torch.long,
        ),
        # Query 2 has no common support because teacher_valid=False.
        "batch_grasp_cdf_valid_mask": torch.tensor(
            [[[[1]], [[1]], [[0]]]],
            device=device,
            dtype=torch.bool,
        ),
        "kview_base_token_sel_idx": indices.clone(),
        "token_sel_idx": indices.clone(),
        "grasp_top_view_inds": views.clone(),
        "grasp_top_view_xyz": torch.ones(1, 3, 3, device=device),
    }


def test_selective_gate_and_gradient() -> None:
    # Target for bin=1 is [1,1].
    # q0: teacher is better; q1: student is better; q2: no common support.
    student_logits = torch.tensor(
        [[[[[-2.0]], [[2.0]], [[-1.0]]],
          [[[-2.0]], [[2.0]], [[-1.0]]]]],
        requires_grad=True,
    )  # [1,2,3,1,1]
    teacher_logits = torch.tensor(
        [[[[[2.0]], [[-2.0]], [[2.0]]],
          [[[2.0]], [[-2.0]], [[2.0]]]]],
    )

    student = _base_endpoints(student_logits)
    teacher = _teacher_targets(teacher_logits)

    gate = build_oracle_selective_cdf_gate(student, teacher)
    expected = torch.tensor([[True, False, False]])
    assert torch.equal(gate["teacher_better_bq"].cpu(), expected), gate[
        "teacher_better_bq"
    ]

    config = SimpleNamespace(
        overall_weight=1.0,
        cdf_weight=1.0,
        temperature=1.0,
    )
    loss, outputs = compute_oracle_selective_cdf_distillation_loss(
        student,
        teacher,
        config,
    )
    assert torch.isfinite(loss), loss
    assert loss.requires_grad
    loss.backward()

    grad = student_logits.grad
    assert grad is not None
    # Only q0 is selected; q1 and q2 must receive no KD gradient.
    assert float(grad[:, :, 0].abs().sum()) > 0.0
    assert float(grad[:, :, 1].abs().sum()) == 0.0
    assert float(grad[:, :, 2].abs().sum()) == 0.0
    assert abs(float(outputs["D: OracleSel selected query ratio"]) - 1.0 / 3.0) < 1e-6
    assert float(outputs["D: OracleSel exact seed ratio"]) == 1.0
    assert float(outputs["D: OracleSel exact pixel ratio"]) == 1.0
    assert float(outputs["D: OracleSel exact view ratio"]) == 1.0


def test_empty_gate_returns_differentiable_zero() -> None:
    student_logits = torch.full(
        (1, 2, 2, 1, 1),
        2.0,
        requires_grad=True,
    )
    teacher_logits = torch.full((1, 2, 2, 1, 1), -2.0)

    indices = torch.tensor([[4, 8]])
    views = torch.tensor([[7, 9]])
    student = {
        "grasp_cdf_pred_angle_depth": student_logits,
        "batch_grasp_cdf_bins_angle_depth": torch.ones(
            1, 2, 1, 1, dtype=torch.long
        ),
        "batch_grasp_cdf_valid_mask": torch.ones(
            1, 2, 1, 1, dtype=torch.bool
        ),
        "kview_base_token_sel_idx": indices.clone(),
        "token_sel_idx": indices.clone(),
        "grasp_top_view_inds": views.clone(),
        "grasp_top_view_xyz": torch.ones(1, 2, 3),
    }
    teacher = {
        "grasp_cdf_pred_angle_depth": teacher_logits,
        "batch_grasp_cdf_bins_angle_depth": torch.ones(
            1, 2, 1, 1, dtype=torch.long
        ),
        "batch_grasp_cdf_valid_mask": torch.ones(
            1, 2, 1, 1, dtype=torch.bool
        ),
        "kview_base_token_sel_idx": indices.clone(),
        "token_sel_idx": indices.clone(),
        "grasp_top_view_inds": views.clone(),
        "grasp_top_view_xyz": torch.ones(1, 2, 3),
    }
    config = SimpleNamespace(
        overall_weight=1.0,
        cdf_weight=1.0,
        temperature=1.0,
    )
    loss, outputs = compute_oracle_selective_cdf_distillation_loss(
        student,
        teacher,
        config,
    )
    assert loss.requires_grad
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert student_logits.grad is not None
    assert float(student_logits.grad.abs().sum()) == 0.0
    assert float(outputs["D: OracleSel selected query ratio"]) == 0.0


def test_exact_query_mismatch_fails() -> None:
    student_logits = torch.zeros(1, 2, 1, 1, 1, requires_grad=True)
    teacher_logits = torch.zeros(1, 2, 1, 1, 1)
    student = {
        "grasp_cdf_pred_angle_depth": student_logits,
        "batch_grasp_cdf_bins_angle_depth": torch.ones(
            1, 1, 1, 1, dtype=torch.long
        ),
        "batch_grasp_cdf_valid_mask": torch.ones(
            1, 1, 1, 1, dtype=torch.bool
        ),
        "kview_base_token_sel_idx": torch.tensor([[1]]),
        "token_sel_idx": torch.tensor([[1]]),
        "grasp_top_view_inds": torch.tensor([[2]]),
        "grasp_top_view_xyz": torch.ones(1, 1, 3),
    }
    teacher = {
        "grasp_cdf_pred_angle_depth": teacher_logits,
        "batch_grasp_cdf_bins_angle_depth": torch.ones(
            1, 1, 1, 1, dtype=torch.long
        ),
        "batch_grasp_cdf_valid_mask": torch.ones(
            1, 1, 1, 1, dtype=torch.bool
        ),
        "kview_base_token_sel_idx": torch.tensor([[1]]),
        "token_sel_idx": torch.tensor([[1]]),
        "grasp_top_view_inds": torch.tensor([[3]]),
        "grasp_top_view_xyz": torch.ones(1, 1, 3),
    }
    try:
        build_oracle_selective_cdf_gate(student, teacher)
    except RuntimeError as exc:
        assert "view" in str(exc).lower()
    else:
        raise AssertionError("Expected exact-view mismatch to fail.")


if __name__ == "__main__":
    test_selective_gate_and_gradient()
    test_empty_gate_returns_differentiable_zero()
    test_exact_query_mismatch_fails()
    print("Exact-Query Oracle-Selective CDF KD self-test passed.")
