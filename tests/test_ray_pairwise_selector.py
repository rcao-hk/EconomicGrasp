from __future__ import annotations

import numpy as np
import torch

from utils.ray_pairwise_selector import (
    RayPairwiseSelector,
    compose_pairwise_features,
    extract_action_conditioned_features,
    listwise_exact_utility_loss,
    pairwise_feature_dim,
    select_with_native_fallback,
)


def test_extract_action_conditioned_features_uses_cdf_best_angle():
    # B=1,C=2,Q=1,A=3,D=2. Group is base-major angle order.
    grouped = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
    cdf = torch.full((1, 2, 1, 3, 2), -8.0)
    cdf[:, :, 0, 2, 1] = 8.0  # angle 2 should win
    ep = {"grasp_cdf_pred_angle_depth": cdf}
    selected, mean, angle = extract_action_conditioned_features(grouped, ep)
    assert angle.tolist() == [[2]]
    assert torch.allclose(selected[0, 0], torch.tensor([3.0, 30.0]))
    assert torch.allclose(mean[0, 0], torch.tensor([2.0, 20.0]))


def test_pairwise_feature_contract_and_native_residuals():
    K, N, C = 3, 2, 4
    selected = torch.arange(K * N * C, dtype=torch.float32).reshape(K, N, C)
    mean = selected + 100.0
    raw = torch.tensor([[0.2, 0.3], [0.5, 0.4], [0.1, 0.8]])
    offsets = torch.tensor([-10.0, 0.0, 10.0])
    zero = 1
    x = compose_pairwise_features(selected, mean, raw, offsets, zero)
    assert x.shape == (K, N, pairwise_feature_dim(C))
    # Native selected feature, candidate feature, selected residual.
    assert torch.equal(x[zero, :, :C], selected[zero])
    assert torch.equal(x[zero, :, C : 2 * C], selected[zero])
    assert torch.allclose(x[zero, :, 2 * C : 3 * C], torch.zeros(N, C))
    # Final normalized offset scalar is zero for native.
    assert torch.allclose(x[zero, :, -1], torch.zeros(N))


def test_native_fallback_threshold_prevents_weak_switch():
    pred = np.array([
        [-0.2, 0.05, 0.4],
        [0.0, 0.0, 0.0],   # native k=1
        [0.1, 0.2, -0.1],
    ])
    valid = np.ones_like(pred, dtype=bool)
    selected = select_with_native_fallback(pred, valid, zero_index=1, threshold=0.15)
    assert selected.tolist() == [1, 2, 0]


def test_listwise_loss_is_finite_and_differentiable():
    pred = torch.tensor([[0.0, 0.0], [0.2, -0.1], [-0.2, 0.4]], requires_grad=True)
    utility = torch.tensor([[0.5, 0.0], [1.0, 0.0], [0.0, 1.0]])
    valid = torch.ones_like(utility, dtype=torch.bool)
    loss = listwise_exact_utility_loss(pred, utility, valid, target_temperature=0.15)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_selector_output_shape():
    model = RayPairwiseSelector(feature_dim=24, hidden_dim=16, dropout=0.0)
    x = torch.randn(3, 5, 24)
    y = model(x)
    assert y.shape == (3, 5)
