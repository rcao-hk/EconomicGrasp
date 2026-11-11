"""CPU-only tests for candidate conflict graphs and exact-quality selection oracles."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from diagnostics.candidate_conflict.core import (
    build_conflict_edges, component_fixed_cardinality_oracle,
    connected_components, greedy_nms, official_ap_matrix, physical_ids,
    quality_bits, q_priority, scene_cluster_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]


def grasps(centers, scores=None, rotations=None):
    n = len(centers)
    g = np.zeros((n, 17), dtype=np.float32)
    g[:, 0] = np.asarray(scores if scores is not None else np.ones(n), dtype=np.float32)
    g[:, 1] = 0.06
    g[:, 2] = 0.02
    g[:, 3] = 0.02
    if rotations is None:
        rotations = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
    g[:, 4:13] = np.asarray(rotations, dtype=np.float32).reshape(n, 9)
    g[:, 13:16] = np.asarray(centers, dtype=np.float32)
    g[:, 16] = -1
    return g


class GraphTests(unittest.TestCase):
    def test_translation_and_rotation_define_edges(self):
        rot = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        a = np.deg2rad(45)
        rot[2] = np.array([[np.cos(a), -np.sin(a), 0],
                           [np.sin(a), np.cos(a), 0],
                           [0, 0, 1]], dtype=np.float32)
        g = grasps([[0, 0, 0], [0.01, 0, 0], [0.01, 0, 0]], rotations=rot)
        edges = build_conflict_edges(g, 0.03, np.deg2rad(30))
        self.assertEqual(edges.tolist(), [[0, 1]])

    def test_greedy_nms_and_suppressor(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        keep, suppressor, order = greedy_nms(np.array([0.9, 0.8, 0.7]), edges)
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(int(suppressor[1]), 0)
        self.assertEqual(int(suppressor[0]), -1)
        self.assertEqual(int(suppressor[2]), -1)

    def test_connected_components_include_isolated_nodes(self):
        labels = connected_components(5, np.array([[0, 1], [1, 2], [3, 4]]))
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[3])

    def test_boundary_translation_is_included(self):
        g = grasps([[0, 0, 0], [0.03, 0, 0]])
        edges = build_conflict_edges(g, 0.03, np.deg2rad(30))
        self.assertEqual(edges.shape, (1, 2))


class OracleTests(unittest.TestCase):
    def test_quality_priority_can_replace_nms_representative(self):
        edges = np.array([[0, 1]], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7])
        q = np.array([1, 6, 2])
        native, _, _ = greedy_nms(scores, edges)
        quality, _, _ = greedy_nms(q_priority(q, scores), edges)
        self.assertEqual(native.tolist(), [0, 2])
        self.assertEqual(quality.tolist(), [1, 2])

    def test_component_exact_fixed_cardinality(self):
        # Path 0--1--2, baseline chooses middle node (K=1); Q oracle chooses node 0.
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        comp = connected_components(3, edges)
        base = np.array([1], dtype=np.int64)
        selected, info = component_fixed_cardinality_oracle(
            3, edges, comp, base, np.array([6, 1, 3]), np.array([0.9, 0.8, 0.7]), 20
        )
        self.assertEqual(selected.tolist(), [0])
        self.assertTrue(info["is_globally_exact"])
        self.assertEqual(len(selected), len(base))

    def test_component_exact_preserves_two_survivors(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        comp = connected_components(3, edges)
        base = np.array([0, 2], dtype=np.int64)
        selected, info = component_fixed_cardinality_oracle(
            3, edges, comp, base, np.array([1, 6, 1]), np.array([0.9, 0.8, 0.7]), 20
        )
        self.assertEqual(selected.tolist(), [0, 2])
        self.assertEqual(len(selected), 2)
        self.assertTrue(info["same_survivor_count"])

    def test_large_component_falls_back_and_is_not_called_exact(self):
        n = 5
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        comp = connected_components(n, edges)
        base, _, _ = greedy_nms(np.arange(n, dtype=float)[::-1], edges)
        selected, info = component_fixed_cardinality_oracle(
            n, edges, comp, base, np.arange(n), np.arange(n, dtype=float), 3
        )
        self.assertEqual(selected.tolist(), base.tolist())
        self.assertFalse(info["is_globally_exact"])
        self.assertEqual(info["fallback_components"], 1)

    def test_quality_bits_are_monotone_and_invalid_is_zero(self):
        friction = np.array([0.2, 0.4, 1.2, -1.0], dtype=np.float32)
        invalid = np.array([False, False, False, False])
        bits = quality_bits(friction, invalid)
        self.assertEqual(bits.sum(axis=1).tolist(), [6, 5, 1, 0])
        self.assertTrue(np.all(np.diff(bits.astype(int), axis=1) >= 0))
        bits2 = quality_bits(np.array([0.2]), np.array([True]))
        self.assertEqual(bits2.sum(), 0)

    def test_physical_id_excludes_confidence_but_includes_width(self):
        g = grasps([[0, 0, 0]], [0.8])
        h = g.copy(); h[:, 0] = 0.2
        np.testing.assert_array_equal(physical_ids(g), physical_ids(h))
        h[:, 1] = 0.07
        self.assertNotEqual(physical_ids(g)[0], physical_ids(h)[0])


class APTests(unittest.TestCase):
    def test_official_ap_curve_denominators_and_friction(self):
        g = grasps([[0, 0, 0], [0.1, 0, 0]], [0.9, 0.8])
        assigned = np.array([0, 0])
        bad = np.array([False, False])
        friction = np.array([0.2, 1.2])
        ap = official_ap_matrix(
            g, assigned, bad, friction, np.array([0, 1]), g[:, 0]
        )
        self.assertAlmostEqual(ap[0, 0], 1.0)
        self.assertAlmostEqual(ap[1, 0], 0.5)
        self.assertAlmostEqual(ap[0, 5], 1.0)
        self.assertAlmostEqual(ap[1, 5], 1.0)
        self.assertAlmostEqual(ap[49, 5], 2.0 / 50.0)

    def test_per_object_top10_then_global_top50(self):
        n = 12
        g = grasps([[float(i), 0, 0] for i in range(n)], np.linspace(1, 0, n))
        obj = np.array([0] * 6 + [1] * 6)
        bad = np.zeros(n, dtype=bool)
        friction = np.full(n, 0.2)
        ap = official_ap_matrix(g, obj, bad, friction, np.arange(n), g[:, 0])
        # Only 12 candidates survive the per-object cap; AP denominator remains 50.
        self.assertAlmostEqual(ap[0, 0], 1.0)
        self.assertAlmostEqual(ap[11, 0], 1.0)
        self.assertAlmostEqual(ap[12, 0], 12.0 / 13.0)
        self.assertAlmostEqual(ap[49, 0], 12.0 / 50.0)

    def test_empty_candidate_ap_is_zero(self):
        g = np.zeros((0, 17), dtype=np.float32)
        ap = official_ap_matrix(g, np.zeros(0), np.zeros(0), np.zeros(0),
                                np.zeros(0, dtype=int), np.zeros(0))
        self.assertEqual(float(ap.sum()), 0.0)

    def test_scene_bootstrap_resamples_scenes(self):
        out = scene_cluster_bootstrap({1: 1.0, 2: 2.0, 3: 3.0}, 100, 0)
        self.assertEqual(out["n_scenes"], 3)
        self.assertAlmostEqual(out["mean"], 2.0)
        self.assertEqual(len(out["ci95"]), 2)


class CLITests(unittest.TestCase):
    def test_cli_help(self):
        for module in ("build", "label", "analyze"):
            result = subprocess.run(
                [sys.executable, "-m", f"diagnostics.candidate_conflict.{module}", "--help"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
