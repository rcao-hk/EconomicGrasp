"""CPU-only contracts for GraspClutter6D → CVA-CDF mixed-data conversion.

Run: python -m unittest discover -s tests -p 'test_gc6d_cdf_mix.py' -v
Real GC6D data-dependent integration checks are in check_gc6d_cdf_mix_data.py.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset.generate_economic_gc6d import (
    _build_cdf_depth_labels,
    _parse_cdf_thresholds,
    _gather_top_views_4d,
    write_csv,
)
from dataset.gc6d_dataset import GraspClutter6DMultiDataset
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter


class TestGC6DCDFEncoding(unittest.TestCase):
    def test_resumed_scene_stats_include_fresh_columns(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stats.csv'
            write_csv(path, [dict(scene_id=5), dict(scene_id=7, points=12)])
            with path.open() as reader:
                rows = list(csv.DictReader(reader))
            self.assertEqual(rows[0]['points'], '')
            self.assertEqual(rows[1]['points'], '12')

    def test_repeated_object_instances_keep_distinct_poses(self):
        dataset = GraspClutter6DMultiDataset.__new__(GraspClutter6DMultiDataset)
        frame = [dict(obj_id=69, cam_R_m2c=np.eye(3).flatten().tolist(),
                      cam_t_m2c=[x, 0, 500]) for x in (0, 100)]
        dataset._load_scene_gt = lambda scene: {'2': frame}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.npz'
            np.savez(path, obj_ids=np.array([69, 69]))
            with np.load(path) as cache:
                ids, poses = dataset._build_current_object_meta(5, 2, cache)
        self.assertEqual(ids.tolist(), [69, 69])
        np.testing.assert_allclose(poses[0, 3, :], [0., .1])

    def test_camera_json_is_camera_to_world(self):
        R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], np.float32)
        pose, up, _ = GraspClutter6DMultiDataset.build_camera_orientation_metadata(
            {'cam_R_w2c': R.flatten().tolist()})
        np.testing.assert_allclose(pose, -R[:, 2])
        np.testing.assert_allclose(up, R.T @ [0., 0., 1.])

    def test_binning_boundary_invalid_collision_and_width(self):
        thresholds = _parse_cdf_thresholds("0.2,0.4,0.6,0.8,1.0,1.2")
        raw = torch.tensor(
            [0.2, 0.4, 0.6, 1.2, 1.21, 0.0, -1.0, float("nan"), 0.2, 0.2],
            dtype=torch.float32,
        ).reshape(1, 1, 1, 10)
        widths = torch.full_like(raw, 0.06)
        widths[..., 9] = 0.11
        collisions = torch.zeros_like(raw, dtype=torch.bool)
        collisions[..., 8] = True
        bins, valid = _build_cdf_depth_labels(raw, widths, collisions, thresholds, 0.1)
        self.assertEqual(
            bins.flatten().tolist(),
            [1, 2, 3, 6, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(valid.flatten().tolist(), [True] * 4 + [False] * 6)

    def test_gather_full_depth_without_best_depth_collapse(self):
        dense = torch.arange(2 * 4 * 12 * 4).reshape(2, 4, 12, 4)
        views = torch.tensor([[3, 1], [0, 2]])
        selected = _gather_top_views_4d(dense, views)
        self.assertEqual(selected.shape, (2, 2, 12, 4))
        self.assertTrue(torch.equal(selected[0, 0], dense[0, 3]))
        self.assertTrue(torch.equal(selected[1, 1], dense[1, 2]))

    def test_camera_pose_matches_geometry_convention(self):
        identity = {"cam_R_w2c": np.eye(3, dtype=np.float32).reshape(-1).tolist()}
        v, up, tilt = GraspClutter6DMultiDataset.build_camera_orientation_metadata(identity)
        np.testing.assert_allclose(v, [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-6)
        self.assertAlmostEqual(float(tilt), 180.0, delta=1e-5)


class _FakeGC6DBase:
    extend_angle = True
    load_grasp_payload = False

    def __init__(self):
        self.scenename = ["000001"]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return {
            "object_poses_list": [
                np.concatenate((np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)), axis=1),
                np.concatenate((np.eye(3, dtype=np.float32), np.ones((3, 1), np.float32)), axis=1),
            ],
            "img": torch.ones(3, 448, 448),
        }


class TestAdapterContract(unittest.TestCase):
    def test_cdf_cache_keeps_per_object_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "economic_grasp_label_300views_extend_angle_cdf_depth"
            folder.mkdir()
            points = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)
            cdf = np.ones((2, 3, 12, 4), dtype=np.uint8)
            width = np.full((2, 3, 12, 4), 50, dtype=np.uint16)
            np.savez(
                folder / "000001_labels.npz",
                points=points,
                pointid=np.asarray([0, 1], np.uint8),
                vgraspness=np.ones((2, 300), np.float32),
                topview=np.zeros((2, 3), dtype=np.uint16),
                extend_angle=np.asarray([1], dtype=np.uint8),
                num_angle=np.asarray([12], dtype=np.uint8),
                num_depth=np.asarray([4], dtype=np.uint8),
                cdf_bins=cdf,
                cdf_thresholds=np.arange(0.2, 1.21, 0.2, dtype=np.float32),
                widths_depth_mm=width,
                width_valids_depth=np.ones_like(cdf, dtype=np.uint8),
            )
            adapter = CVAExtendedLabelAdapter(
                _FakeGC6DBase(), dataset_root=tmp,
                label_folder=folder.name, use_cdf=True, num_angle=12, num_depth=4
            )
            output = adapter[0]
            self.assertEqual(len(output["object_poses_list"]), 2)
            self.assertEqual(len(output["grasp_cdf_bins_list"]), 2)
            self.assertEqual(output["grasp_cdf_bins_list"][0].shape, (1, 3, 12, 4))
            self.assertEqual(output["grasp_widths_depth_list"][0].dtype, np.dtype("uint16"))
            self.assertTrue(np.isfinite(output["cdf_thresholds"]).all())


if __name__ == "__main__":
    unittest.main()
