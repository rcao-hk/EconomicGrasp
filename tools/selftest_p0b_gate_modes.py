#!/usr/bin/env python3
"""Small CPU self-test for balanced vs ordinary P0-B gate plumbing."""
import torch
import sys
from pathlib import Path
REPO_OVERLAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_OVERLAY))

from models.p0b_oracle_hybrid import build_p0b_cdf_variants

B, T, Q, A, D = 1, 3, 2, 2, 2
valid = torch.ones(B, Q, A, D, dtype=torch.bool)
bins = torch.tensor(
    [[[[0, 0], [0, 0]], [[1, 1], [1, 1]]]], dtype=torch.long
)
student = torch.zeros(B, T, Q, A, D)
teacher = torch.zeros_like(student)
student[:, :, 0] = 2.0
teacher[:, :, 0] = -2.0
student[:, :, 1] = 2.0
teacher[:, :, 1] = -2.0
s_ep = {
    "grasp_cdf_pred_angle_depth": student,
    "batch_grasp_cdf_bins_angle_depth": bins,
    "batch_grasp_cdf_valid_mask": valid,
}
t_ep = {
    "grasp_cdf_pred_angle_depth": teacher,
    "batch_grasp_cdf_valid_mask": valid,
}
for mode in ("balanced", "ordinary"):
    bundle = build_p0b_cdf_variants(s_ep, t_ep, gate_mode=mode)
    assert bundle.teacher_better_bq.tolist() == [[True, False]]
    if mode == "ordinary":
        assert float(bundle.diagnostics["gate_mode_ordinary"]) == 1.0
        assert float(bundle.diagnostics["gate_mode_balanced"]) == 0.0
    else:
        assert float(bundle.diagnostics["gate_mode_ordinary"]) == 0.0
        assert float(bundle.diagnostics["gate_mode_balanced"]) == 1.0
print("P0-B gate-mode self-test passed.")
