import torch

from utils.ray_pairwise_feature_ablation import compose_feature_ablation, feature_ablation_dim
from utils.ray_pairwise_selector import compose_pairwise_features


def test_feature_ablation_dimensions_and_full_equivalence():
    K, N, C = 7, 5, 8
    torch.manual_seed(0)
    selected = torch.randn(K, N, C)
    mean = torch.randn(K, N, C)
    raw = torch.randn(K, N)
    offsets = torch.tensor([-40, -20, -10, 0, 10, 20, 40], dtype=torch.float32)
    zero = 3

    for mode in ("raw_offset", "selected_residual", "mean_residual", "selected_mean", "full"):
        x = compose_feature_ablation(selected, mean, raw, offsets, zero, mode)
        assert x.shape == (K, N, feature_ablation_dim(C, mode))

    full = compose_feature_ablation(selected, mean, raw, offsets, zero, "full")
    reference = compose_pairwise_features(selected, mean, raw, offsets, zero)
    assert torch.allclose(full, reference)


def test_raw_offset_block_contract():
    K, N, C = 3, 2, 4
    selected = torch.zeros(K, N, C)
    mean = torch.zeros_like(selected)
    raw = torch.tensor([[0.2, 0.4], [0.5, 0.1], [0.7, 0.9]])
    offsets = torch.tensor([-10.0, 0.0, 10.0])
    zero = 1
    x = compose_feature_ablation(selected, mean, raw, offsets, zero, "raw_offset")
    assert torch.allclose(x[:, :, 0], raw[zero : zero + 1].expand(K, N))
    assert torch.allclose(x[:, :, 1], raw)
    assert torch.allclose(x[:, :, 2], raw - raw[zero : zero + 1])
    assert torch.allclose(x[:, :, 3], torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]))
