import torch

from models.kview_query_transformer import (
    KViewQueryTransformerConfig,
    ViewConditionedAttentionGrouping as BaselineGrouping,
)
from models.robust_grasp_support import (
    RobustViewConditionedAttentionGrouping,
    configure_robust_support,
)


def _config():
    return KViewQueryTransformerConfig(
        patch_size=4,
        metric_radius=0.08,
        radius_px_min=1.0,
        radius_px_max=100.0,
        grouping_model_dim=32,
        grouping_num_heads=4,
        grouping_dropout=0.0,
        grouping_ffn_ratio=2.0,
    )


def _inputs(z=0.5):
    H = W = 64
    token_idx = torch.tensor([[32 * W + 32]], dtype=torch.long)
    seed_xyz = torch.tensor([[[0.0, 0.0, float(z)]]], dtype=torch.float32)
    rot = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    depth = torch.full((1, 1, H, W), float(z), dtype=torch.float32)
    K = torch.tensor(
        [[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return seed_xyz, token_idx, rot, depth, K, H, W


def _make(cls, mode, wide_scale=1.5, image_radius_px=16.0):
    configure_robust_support(
        mode=mode,
        wide_scale=wide_scale,
        image_radius_px=image_radius_px,
    )
    return cls(
        seed_feature_dim=16,
        feat_dim=8,
        out_dim=32,
        config=_config(),
    )


def _grid_call(module, z=0.5):
    seed_xyz, token_idx, rot, depth, K, H, W = _inputs(z=z)
    return module._make_view_conditioned_grid(
        seed_xyz=seed_xyz,
        token_sel_idx=token_idx,
        top_view_rot=rot,
        depth_map=depth,
        camera_K=K,
        H=H,
        W=W,
    )


def test_metric_mode_is_exact_baseline_grid():
    baseline = BaselineGrouping(
        seed_feature_dim=16,
        feat_dim=8,
        out_dim=32,
        config=_config(),
    )
    robust = _make(RobustViewConditionedAttentionGrouping, mode="metric")
    a = _grid_call(baseline, z=0.5)
    b = _grid_call(robust, z=0.5)
    for xa, xb in zip(a, b):
        assert torch.equal(xa, xb)


def test_wide_mode_preserves_token_count_and_expands_radius():
    metric = _make(RobustViewConditionedAttentionGrouping, mode="metric")
    wide = _make(
        RobustViewConditionedAttentionGrouping,
        mode="wide",
        wide_scale=1.5,
    )
    m = _grid_call(metric, z=0.5)
    w = _grid_call(wide, z=0.5)
    assert m[0].shape == w[0].shape
    assert m[3].shape[-2] == w[3].shape[-2] == 16
    assert torch.allclose(w[2], m[2] * 1.5, atol=1e-5, rtol=1e-5)


def test_dual_mode_keeps_budget_and_image_half_is_depth_radius_independent():
    dual = _make(
        RobustViewConditionedAttentionGrouping,
        mode="dual",
        image_radius_px=16.0,
    )
    a = _grid_call(dual, z=0.5)
    b = _grid_call(dual, z=1.0)
    patch_a = a[3]
    patch_b = b[3]
    mask = dual.robust_image_support_mask

    assert patch_a.shape[-2] == 16
    assert int(mask.sum().item()) == 8
    assert torch.allclose(
        patch_a[:, :, mask],
        patch_b[:, :, mask],
        atol=1e-6,
        rtol=0.0,
    )
    # At least part of the metric-conditioned half should move when z changes.
    assert not torch.allclose(
        patch_a[:, :, ~mask],
        patch_b[:, :, ~mask],
        atol=1e-6,
        rtol=0.0,
    )
