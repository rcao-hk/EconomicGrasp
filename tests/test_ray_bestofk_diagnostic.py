from __future__ import annotations

import numpy as np
import torch

from utils.ray_bestofk_diagnostic import (
    build_ray_center_hypotheses,
    friction_utility,
    parse_offsets_mm,
    select_exact_oracle,
    select_raw_score,
)


def test_parse_offsets_requires_native_zero():
    assert parse_offsets_mm("-20,0,20") == (-20.0, 0.0, 20.0)
    try:
        parse_offsets_mm("-20,20")
    except ValueError:
        pass
    else:
        raise AssertionError("missing zero offset should fail")


def test_ray_hypotheses_stay_on_same_pixel_ray():
    native = torch.tensor([[[0.0, 0.0, 0.5], [0.25, 0.5, 0.5]]])
    idx = torch.tensor([[5, 14]], dtype=torch.long)  # (u,v)=(1,1),(2,3) in 4x4
    K = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    centers, valid = build_ray_center_hypotheses(
        native, idx, K, (4, 4), (-100.0, 0.0, 100.0), 0.2, 1.0
    )
    assert valid.all()
    assert torch.allclose(centers[1], native, atol=1e-6)
    assert torch.allclose(centers[0, 0, 0], torch.tensor([0.0, 0.0, 0.4]))
    # second ray: x/z=0.5, y/z=1.0 for every center
    ratio_x = centers[:, 0, 1, 0] / centers[:, 0, 1, 2]
    ratio_y = centers[:, 0, 1, 1] / centers[:, 0, 1, 2]
    assert torch.allclose(ratio_x, torch.full_like(ratio_x, 0.5))
    assert torch.allclose(ratio_y, torch.ones_like(ratio_y))


def test_raw_selection_respects_valid_mask():
    score = np.asarray([[0.1, 0.9], [0.8, 0.2], [0.7, 0.7]], dtype=np.float32)
    valid = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=bool)
    idx = select_raw_score(score, valid)
    assert idx.tolist() == [1, 2]


def test_exact_oracle_uses_utility_then_raw_score_tiebreak():
    utility = np.asarray([[0.2, 0.5], [0.8, 0.5], [0.8, 0.1]], dtype=np.float32)
    score = np.asarray([[0.9, 0.2], [0.3, 0.4], [0.7, 0.9]], dtype=np.float32)
    valid = np.ones_like(score, dtype=bool)
    idx = select_exact_oracle(utility, score, valid)
    assert idx.tolist() == [2, 1]


def test_friction_utility_is_monotonic_with_grasp_strength():
    f = np.asarray([-1.0, 1.2, 0.8, 0.4, 0.2], dtype=np.float32)
    u = friction_utility(f)
    assert np.all(np.diff(u) >= 0.0)
    assert u[0] == 0.0
    assert u[-1] == 1.0
