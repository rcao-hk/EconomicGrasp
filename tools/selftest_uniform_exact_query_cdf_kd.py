#!/usr/bin/env python3
"""Synthetic contract test for exact-query uniform CDF-only KD."""

from __future__ import annotations

import types
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from models.uniform_exact_query_cdf_kd import (
    compute_uniform_exact_query_cdf_distillation_loss,
)


def make_endpoints():
    # B=1, T=2, Q=3, A=1, D=1.
    # q0 and q1 are common-valid. q2 is student-only valid.
    student_logits = torch.tensor(
        [[[[[-2.0]], [[2.0]], [[0.5]]], [[[ -2.0]], [[2.0]], [[0.5]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    # Teacher is better on q0 and worse on q1. Uniform control must still
    # distil both q0 and q1, while q2 must receive no KD gradient.
    teacher_logits = torch.tensor(
        [[[[[2.0]], [[-2.0]], [[-1.0]]], [[[2.0]], [[-2.0]], [[-1.0]]]]],
        dtype=torch.float32,
    )

    ids = torch.tensor([[4, 8, 12]], dtype=torch.long)
    views = torch.tensor([[10, 20, 30]], dtype=torch.long)
    bins = torch.tensor([[[[1]], [[1]], [[1]]]], dtype=torch.long)
    student_valid = torch.tensor([[[[1]], [[1]], [[1]]]], dtype=torch.bool)
    teacher_valid = torch.tensor([[[[1]], [[1]], [[0]]]], dtype=torch.bool)

    student = {
        "grasp_cdf_pred_angle_depth": student_logits,
        "batch_grasp_cdf_bins_angle_depth": bins,
        "batch_grasp_cdf_valid_mask": student_valid,
        "kview_base_token_sel_idx": ids,
        "token_sel_idx": ids.clone(),
        "grasp_top_view_inds": views,
    }
    teacher = {
        "grasp_cdf_pred_angle_depth": teacher_logits,
        "batch_grasp_cdf_bins_angle_depth": bins.clone(),
        "batch_grasp_cdf_valid_mask": teacher_valid,
        "kview_base_token_sel_idx": ids.clone(),
        "token_sel_idx": ids.clone(),
        "grasp_top_view_inds": views.clone(),
    }
    return student, teacher


def main():
    student, teacher = make_endpoints()
    config = types.SimpleNamespace(
        temperature=1.0,
        overall_weight=1.0,
        cdf_weight=1.0,
    )
    loss, out = compute_uniform_exact_query_cdf_distillation_loss(
        student,
        teacher,
        config,
    )
    assert torch.isfinite(loss), loss
    loss.backward()

    grad = student["grasp_cdf_pred_angle_depth"].grad
    assert grad is not None
    # q0 and q1 selected; q2 outside common-valid support.
    assert float(grad[:, :, 0].abs().sum()) > 0.0
    assert float(grad[:, :, 1].abs().sum()) > 0.0
    assert float(grad[:, :, 2].abs().sum()) == 0.0
    assert float(out["D: UniformCDF uses teacher-better gate"]) == 0.0
    assert float(out["D: UniformCDF uses all common queries"]) == 1.0
    assert abs(float(out["D: UniformCDF selected query ratio"]) - 2.0 / 3.0) < 1e-6

    # Exact-query mismatch must fail.
    student2, teacher2 = make_endpoints()
    teacher2["grasp_top_view_inds"] = teacher2["grasp_top_view_inds"].clone()
    teacher2["grasp_top_view_inds"][0, 1] += 1
    try:
        compute_uniform_exact_query_cdf_distillation_loss(
            student2,
            teacher2,
            config,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected exact-view mismatch to raise RuntimeError")

    print("Exact-Query Uniform CDF-only KD self-test passed.")


if __name__ == "__main__":
    main()
