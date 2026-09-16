from __future__ import annotations

import pytest
import torch

from utils.cva_center_decoupling import (
    _native_grouping_graspness_map,
    assert_native_reread_equivalent,
    gather_reference_centers_from_depth,
    native_reread_equivalence_metrics,
    replace_decoded_translation,
)


def test_gather_reference_centers_uses_exact_selected_pixel_and_camera_ray():
    # 4x4 depth map, selected pixels (u,v)=(1,1) and (2,3).
    depth = torch.zeros(1, 4, 4)
    depth[0, 1, 1] = 0.5
    depth[0, 3, 2] = 0.8
    idx = torch.tensor([[5, 14]], dtype=torch.long)
    K = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    native = torch.tensor([[[0.0, 0.0, 0.6], [0.0, 0.0, 0.7]]])
    ep = {
        "kview_base_token_sel_idx": idx,
        "kview_base_xyz_graspable": native,
        "gt_depth_m": depth,
        "K": K,
    }
    xyz, valid, z = gather_reference_centers_from_depth(ep, min_depth=0.2, max_depth=1.0)
    assert valid.tolist() == [[True, True]]
    assert torch.allclose(z, torch.tensor([[0.5, 0.8]]))
    assert torch.allclose(xyz[0, 0], torch.tensor([0.0, 0.0, 0.5]))
    # u=2,v=3, fx=fy=2,cx=cy=1 => x=.4,y=.8,z=.8
    assert torch.allclose(xyz[0, 1], torch.tensor([0.4, 0.8, 0.8]), atol=1e-6)


def test_invalid_reference_depth_falls_back_to_native_z_on_same_ray():
    depth = torch.zeros(1, 2, 2)
    idx = torch.tensor([[3]], dtype=torch.long)  # u=1,v=1
    K = torch.eye(3).unsqueeze(0)
    native = torch.tensor([[[0.7, 0.7, 0.7]]])
    ep = {
        "kview_base_token_sel_idx": idx,
        "kview_base_xyz_graspable": native,
        "gt_depth_m": depth,
        "K": K,
    }
    xyz, valid, _ = gather_reference_centers_from_depth(ep, min_depth=0.2, max_depth=1.0)
    assert valid.tolist() == [[False]]
    assert torch.allclose(xyz[0, 0], native[0, 0])


def test_e01_translation_replacement_preserves_every_other_grasp_field():
    pred = torch.arange(34, dtype=torch.float32).reshape(2, 17)
    replacement = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    out = replace_decoded_translation([pred], replacement)[0]
    keep = list(range(13)) + [16]
    assert torch.equal(out[:, keep], pred[:, keep])
    assert torch.equal(out[:, 13:16], replacement[0])


def test_e11_can_be_formed_from_e10_by_translation_replacement():
    e10 = torch.randn(5, 17)
    ref = torch.randn(1, 5, 3)
    e11 = replace_decoded_translation([e10], ref)[0]
    keep = list(range(13)) + [16]
    assert torch.equal(e11[:, keep], e10[:, keep])
    assert torch.equal(e11[:, 13:16], ref[0])


def test_reread_uses_native_clipped_graspness_map_when_debug_tensor_exists():
    ep = {
        "dbg_grasp_sel": torch.tensor([[0.0, 0.2, 0.8, 1.0]]),
        # Deliberately inconsistent raw values: the reread must ignore these
        # when dbg_grasp_sel is available because native grouping saw the
        # clipped dbg_grasp_sel values.
        "graspness_score": torch.tensor([[[-2.0, 0.2, 0.8, 3.0]]]),
    }
    out = _native_grouping_graspness_map(ep, B=1, H=2, W=2)
    assert torch.equal(out.reshape(1, -1), ep["dbg_grasp_sel"])


def test_reread_fallback_clamps_raw_graspness_exactly_like_native_forward():
    ep = {
        "graspness_score": torch.tensor([[[-2.0, 0.2, 0.8, 3.0]]]),
    }
    out = _native_grouping_graspness_map(ep, B=1, H=2, W=2)
    expected = torch.tensor([[[[0.0, 0.2], [0.8, 1.0]]]])
    assert torch.equal(out, expected)


def test_noop_reread_equivalence_metrics_and_fail_fast():
    native = {
        "grasp_cdf_pred_angle_depth": torch.zeros(1, 6, 2, 12, 4),
        "grasp_width_pred_angle_depth": torch.zeros(1, 4, 2, 12),
    }
    replay = {
        "grasp_cdf_pred_angle_depth": native["grasp_cdf_pred_angle_depth"].clone(),
        "grasp_width_pred_angle_depth": native["grasp_width_pred_angle_depth"].clone(),
    }
    metrics = native_reread_equivalence_metrics(native, replay)
    assert metrics["noop_cdf_max_abs"] == 0.0
    assert metrics["noop_width_max_abs"] == 0.0
    assert_native_reread_equivalent(native, replay, atol=1e-6)

    replay["grasp_cdf_pred_angle_depth"][0, 0, 0, 0, 0] = 1e-2
    with pytest.raises(RuntimeError, match="no-op reread"):
        assert_native_reread_equivalent(native, replay, atol=1e-4)
