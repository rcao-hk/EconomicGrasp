import numpy as np
import torch

from utils.p5_separability import (
    FEATURE_NAMES, binary_auprc, binary_auroc, build_feature_views,
    gated_query_metrics,
)


def test_binary_metrics_perfect_order():
    y = np.array([0, 0, 1, 1], dtype=np.uint8)
    s = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    assert abs(binary_auroc(y, s) - 1.0) < 1e-8
    assert abs(binary_auprc(y, s) - 1.0) < 1e-8


def test_gated_metrics_need_oracle():
    arrays = {
        "native_violation_m": np.array([0.0, 0.0, 0.010, 0.020]),
        "repaired_violation_m": np.array([0.002, 0.000, 0.003, 0.025]),
        "native_safe": np.array([1, 1, 0, 0], dtype=np.uint8),
        "repaired_safe": np.array([0, 1, 1, 0], dtype=np.uint8),
        "safe": np.array([1, 1, 0, 0], dtype=np.uint8),
        "repairable": np.array([0, 0, 1, 1], dtype=np.uint8),
    }
    active = arrays["repairable"].astype(bool)
    m = gated_query_metrics(arrays, active)
    assert abs(m["active_ratio"] - 0.5) < 1e-8
    assert abs(m["safe_noharm"] - 1.0) < 1e-8
    assert abs(m["repairable_capture"] - 0.5) < 1e-8
    assert m["mean_gain_m"] > 0


def test_feature_views_shapes():
    B, Q, L, C = 1, 4, 11, 8
    kf = torch.randn(B, Q, L, 2 * C + 8)
    visible = torch.ones(B, Q, L, dtype=torch.bool)
    evidence = torch.randn(B, Q, 16)
    context = torch.randn(B, Q, 16)
    ep = {
        "p5_native_score": torch.rand(B, Q),
        "p5_proposal_center": torch.rand(B, Q, 3) + torch.tensor([0.0, 0.0, 0.3]),
        "p5_native_width_m": torch.rand(B, Q) * 0.08,
        "p5_native_depth_idx": torch.randint(0, 4, (B, Q)),
        "p5_native_angle_idx": torch.randint(0, 12, (B, Q)),
        "token_sel_idx": torch.tensor([[0, 4, 8, 12]]),
        "p5_depth_evidence": torch.ones(B, 1, 4, 4) * 0.5,
    }
    views = build_feature_views(
        keypoint_features=kf, visible=visible, evidence=evidence, context=context,
        end_points=ep, seed_feature_dim=C, min_depth=0.2, max_depth=1.0,
    )
    assert tuple(views) == FEATURE_NAMES
    assert views["F0"].shape == (B, Q, 14)
    assert views["F1"].shape == (B, Q, 14 + 3 + C)
    assert views["F2"].shape == (B, Q, 14 + C)
    assert views["F3"].shape == (B, Q, 14 + 16)
    assert views["F4"].shape == (B, Q, 14 + 16)
