from pathlib import Path

import numpy as np
import torch

from tools.grasp_behavior_visualizer import (
    BehaviorVisConfig, BehaviorVisualizer, backproject_depth,
    feature_norm_map, feature_pca_rgb, grasp_skeleton_points,
)


def test_behavior_vis_interval_gate(tmp_path):
    cfg = BehaviorVisConfig.from_strings(
        str(tmp_path), items='rgb,depth', every=5, start=10, end=25)
    vis = BehaviorVisualizer(cfg)
    assert not vis.should_save(9)
    assert vis.should_save(10)
    assert not vis.should_save(11)
    assert vis.should_save(25)
    assert not vis.should_save(30)
    assert vis.wants('rgb') and not vis.wants('cdf')


def test_feature_visualization_shapes_are_finite():
    torch.manual_seed(0)
    feat = torch.randn(1, 8, 12, 10)
    norm = feature_norm_map(feat)
    pca = feature_pca_rgb(feat)
    assert norm.shape == (12, 10)
    assert pca.shape == (12, 10, 3)
    assert np.isfinite(norm).all()
    assert np.isfinite(pca).all()
    assert ((pca >= 0) & (pca <= 1)).all()


def test_backproject_depth_identity_camera():
    depth = np.ones((1, 1, 3, 3), np.float32)
    K = np.array([[[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]],
                 np.float32)
    pts, pix = backproject_depth(depth, K, stride=1, valid_range=(.1, 2.))
    assert pts.shape == (9, 3)
    center = pts[np.where((pix == [1, 1]).all(1))[0][0]]
    np.testing.assert_allclose(center, [0., 0., 1.], atol=1e-6)


def test_grasp_skeleton_respects_translation():
    a = np.zeros((1, 17), np.float32)
    a[:, 1] = .08
    a[:, 2] = .02
    a[:, 3] = .04
    a[:, 4:13] = np.eye(3, dtype=np.float32).reshape(1, 9)
    a[:, 13:16] = [0.1, -0.2, 0.5]
    skel = grasp_skeleton_points(a)
    assert skel.shape == (1, 7, 3)
    # Every point is expressed around the same translated grasp frame.
    assert np.allclose(skel[0, :, 2].mean(), .5, atol=.02)
