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
    save_candidate_latent_response,
    save_projected_query_overlay,
    write_multicloud_ply,
    _write_pointcloud_mesh_ply,
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
    assert "--dataset-root" in p.stdout
    assert "--stage1-checkpoint" in p.stdout
    assert "--dataset_root" not in p.stdout


def test_candidate_latent_response_writes_png(tmp_path):
    torch.manual_seed(7)
    latent = torch.randn(3, 5, 8)
    offsets = torch.tensor([-20., 0., 20.])
    score = torch.tensor([.9, .7, .5, .3, .1])
    path = tmp_path / "latent.png"
    save_candidate_latent_response(
        path, latent, offsets, score, zero_index=1, max_queries=4)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_merge_cli_runs_as_module():
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run(
        [sys.executable, "-m", "tools.merge_grasp_behavior_viz", "--help"],
        cwd=root, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "--root" in p.stdout


def test_projected_query_overlay_uses_visible_markers(tmp_path):
    rgb = np.zeros((32, 32, 3), np.float32)
    token = np.array([0, 31, 32 * 31 + 31], np.int64)
    values = np.array([0., .5, 1.], np.float32)
    path = tmp_path / "projected.png"
    save_projected_query_overlay(
        path, rgb, token, values, "projected points",
        cmap="viridis", point_size=36.)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_multicloud_ply_semantic_colors(tmp_path):
    path = tmp_path / "compare.ply"
    write_multicloud_ply(
        path,
        (
            ("active", np.array([[0., 0., 1.]], np.float32), (0, 255, 0)),
            ("predicted", np.array([[1., 0., 1.]], np.float32), (255, 0, 0)),
            ("rendered_gt", np.array([[2., 0., 1.]], np.float32), (0, 0, 255)),
        ))
    text = path.read_text()
    assert "comment active color=0,255,0" in text
    assert "comment predicted color=255,0,0" in text
    assert "comment rendered_gt color=0,0,255" in text
    assert "element vertex 3" in text


def test_combined_pointcloud_mesh_ply_has_faces(tmp_path):
    path = tmp_path / "scene_mesh.ply"
    scene = np.array([[0., 0., 1.], [0.1, 0., 1.]], np.float32)
    scene_color = np.array([[.5, .5, .5], [.5, .5, .5]], np.float32)
    vertices = np.array([
        [0., 0., .5], [0.01, 0., .5], [0., .01, .5]
    ], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    mesh_color = np.array([[1., 0., 0.]] * 3, np.float32)
    _write_pointcloud_mesh_ply(
        path, scene, scene_color, vertices, faces, mesh_color)
    text = path.read_text()
    assert "element vertex 5" in text
    assert "element face 1" in text
    # Face indices must be offset past the two scene-point vertices.
    assert "\n3 2 3 4\n" in text
