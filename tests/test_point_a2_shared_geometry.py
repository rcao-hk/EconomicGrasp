"""Lightweight tests of A1/A2 exact-depth audit (CPU only)."""

import pytest
import torch

from tools.verify_point_a2_shared_geometry import (
    compare_depth_states,
    depth_state,
)


def _state():
    return {
        "depth_net.depthnet.pretrained.layer1.weight": torch.arange(12., dtype=torch.float32).reshape(3, 4),
        "depth_net.pose_aware_adapter.scale.weight": torch.ones(2, 3),
        "grasp_net.grasp_head.weight": torch.randn(4, 4),
    }


def test_only_depth_state_compared():
    first = {"model_state_dict": _state()}
    second = {"model_state_dict": _state()}
    second["model_state_dict"]["grasp_net.grasp_head.weight"] = torch.randn(4, 4)
    assert compare_depth_states(depth_state(first), depth_state(second)) == 2


def test_changed_dpt_tensor_detected():
    first, second = _state(), _state()
    second["depth_net.pose_aware_adapter.scale.weight"] = torch.full((2, 3), 2.0)
    with pytest.raises(AssertionError, match="depth weights differ"):
        compare_depth_states(
            depth_state({"model_state_dict": first}),
            depth_state({"model_state_dict": second}),
        )


def test_missing_dpt_source_detected():
    with pytest.raises(ValueError, match="no depth_net"):
        depth_state({"model_state_dict": {"grasp.head": torch.ones(2)}})
