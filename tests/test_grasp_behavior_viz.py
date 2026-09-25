from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.grasp_behavior_viz import (
    BehaviorVizWriter,
    depth_to_points,
    feature_pca_rgb,
    parse_items,
    sparse_query_map,
)


def test_item_presets_and_iteration_schedule(tmp_path):
    core = parse_items("core")
    assert {"depth", "feature", "local", "grasps", "corruption_delta"} <= core
    writer = BehaviorVizWriter(tmp_path, items="feature,depth", every=5)
    assert writer.enabled("feature")
    assert not writer.enabled("grasps")
    assert writer.due(10)
    assert not writer.due(11)
    assert writer.due(None)


def test_depth_to_points_identity_intrinsics():
    depth = np.ones((2, 3), np.float32)
    K = np.eye(3, dtype=np.float32)
    points, colors = depth_to_points(depth, K, stride=1, max_depth=2.)
    assert colors is None
    expected = np.array([
        [0., 0., 1.], [1., 0., 1.], [2., 0., 1.],
        [0., 1., 1.], [1., 1., 1.], [2., 1., 1.],
    ], np.float32)
    np.testing.assert_allclose(points, expected)


def test_sparse_query_map_keeps_unobserved_pixels_nan():
    out = sparse_query_map(
        np.array([0, 3, 3]),
        np.array([.2, .4, .8], np.float32),
        (2, 2),
        reduce="max")
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(.2)
    assert np.isnan(out[0, 1])
    assert out[1, 1] == pytest.approx(.8)


def test_feature_pca_has_rgb_shape_and_finite_values():
    torch.manual_seed(3)
    feat = torch.randn(1, 8, 4, 5)
    rgb = feature_pca_rgb(feat)
    assert rgb.shape == (4, 5, 3)
    assert np.isfinite(rgb).all()
    assert float(rgb.min()) >= 0.
    assert float(rgb.max()) <= 1.


def test_visualization_cli_help_does_not_require_cuda():
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run(
        [sys.executable, str(root / "visualize_grasp_behavior.py"), "--help"],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "--frames" in p.stdout
    assert "--items" in p.stdout
    assert "--eval-cases" in p.stdout
