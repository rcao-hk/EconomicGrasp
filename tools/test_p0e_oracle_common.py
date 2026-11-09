#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from p0e_oracle_common import (
    FrameSpec,
    LocalPerturbation,
    apply_local_perturbation,
    build_local_perturbations,
    deduplicate_physical_actions,
    friction_to_utility,
    oracle_scores,
    pick_best_local_actions,
    resolve_prediction_root,
    select_local_base_indices,
    shard_frames,
    stack_local_lattice,
    verify_fixed_input_collision_policy,
)


def make_grasps(n: int) -> np.ndarray:
    array = np.zeros((n, 17), dtype=np.float32)
    array[:, 0] = np.linspace(1.0, 0.1, n, dtype=np.float32)
    array[:, 1] = 0.05
    array[:, 2] = 0.02
    array[:, 3] = 0.02
    array[:, 4:13] = np.eye(3, dtype=np.float32).reshape(1, 9)
    array[:, 13] = np.arange(n, dtype=np.float32) * 0.01
    array[:, 16] = -1
    return array


class P0ECommonTest(unittest.TestCase):
    def test_friction_utility_matches_threshold_cdf(self) -> None:
        friction = np.asarray([-1.0, 0.2, 0.4, 0.8, 1.2], dtype=np.float32)
        utility = friction_to_utility(friction)
        expected = np.asarray([0.0, 1.0, 5 / 6, 3 / 6, 1 / 6], dtype=np.float32)
        np.testing.assert_allclose(utility, expected, atol=1e-7)

    def test_oracle_tie_break_cannot_cross_utility_bin(self) -> None:
        friction = np.asarray([0.4, 0.4, 0.6], dtype=np.float32)
        original = np.asarray([0.1, 0.9, 1.0], dtype=np.float32)
        scores = oracle_scores(friction, original, tie_break_eps=1e-4).score
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[0], scores[2])

    def test_local_translation_is_gripper_frame(self) -> None:
        grasp = make_grasps(1)
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        grasp[0, 4:13] = rotation.reshape(-1)
        moved = apply_local_perturbation(
            grasp,
            LocalPerturbation("x", translation_local_m=(0.01, 0.0, 0.0)),
        )
        np.testing.assert_allclose(moved[0, 13:16], [0.0, 0.01, 0.0], atol=1e-7)

    def test_inplane_rotation_preserves_orthonormality(self) -> None:
        grasp = make_grasps(2)
        rotated = apply_local_perturbation(
            grasp,
            LocalPerturbation(
                "angle", rotation_local_rad=(math.radians(15.0), 0.0, 0.0)
            ),
        )
        matrices = rotated[:, 4:13].reshape(-1, 3, 3)
        for matrix in matrices:
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=6)

    def test_default_lattice_has_thirteen_identity_first_actions(self) -> None:
        perturbations = build_local_perturbations()
        self.assertEqual(len(perturbations), 13)
        self.assertTrue(perturbations[0].is_identity)

    def test_identity_wins_equal_utility(self) -> None:
        base = make_grasps(1)
        perturbations = build_local_perturbations(
            translation_mm=5,
            inplane_deg=0,
            depth_delta_m=0,
            width_delta_m=0,
        )
        lattice = stack_local_lattice(
            base,
            perturbations,
            min_width_m=0,
            max_width_m=0.1,
            min_depth_m=0,
            max_depth_m=0.1,
        )
        friction = np.full(lattice.shape[:2], 0.4, dtype=np.float32)
        best, _friction, _utility, ids = pick_best_local_actions(lattice, friction)
        self.assertEqual(int(ids[0]), 0)
        np.testing.assert_allclose(best, base)


    def test_prediction_root_resolver_supports_flat_split_and_p0b_student(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            camera = "realsense"

            flat = root / "flat"
            (flat / "scene_0100" / camera).mkdir(parents=True)
            resolved, layout = resolve_prediction_root(
                flat, split="test_seen", camera=camera, role="student"
            )
            self.assertEqual(resolved, flat.resolve())
            self.assertEqual(layout, "direct")

            nested = root / "nested"
            (nested / "test_seen" / "scene_0100" / camera).mkdir(parents=True)
            resolved, layout = resolve_prediction_root(
                nested, split="test_seen", camera=camera, role="student"
            )
            self.assertEqual(resolved, (nested / "test_seen").resolve())
            self.assertEqual(layout, "split_nested")

            p0b = root / "p0b"
            (p0b / "test_seen" / "student" / "scene_0100" / camera).mkdir(
                parents=True
            )
            resolved, layout = resolve_prediction_root(
                p0b, split="test_seen", camera=camera, role="student"
            )
            self.assertEqual(
                resolved, (p0b / "test_seen" / "student").resolve()
            )
            self.assertEqual(layout, "p0b_root_student")

    def test_prediction_root_resolver_rejects_p0b_teacher_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "teacher_full"
            (path / "scene_0100" / "realsense").mkdir(parents=True)
            with self.assertRaises(ValueError):
                resolve_prediction_root(
                    path,
                    split="test_seen",
                    camera="realsense",
                    role="teacher",
                )


    def test_fixed_collision_policy_verifies_p0b_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            split_root = Path(temp_dir) / "test_seen"
            student_root = split_root / "student"
            (student_root / "scene_0100" / "realsense").mkdir(parents=True)
            meta = split_root / "_p0b_meta"
            meta.mkdir()
            (meta / "worker_00_summary.json").write_text(
                '{"collision_thresh": 0.01}', encoding="utf-8"
            )
            result = verify_fixed_input_collision_policy(student_root)
            self.assertEqual(result["status"], "verified_from_p0b_metadata")
            self.assertEqual(result["worker_summaries"], 1)

            (meta / "worker_00_summary.json").write_text(
                '{"collision_thresh": 0.0}', encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                verify_fixed_input_collision_policy(student_root)


    def test_scene_sharding_keeps_each_scene_on_one_worker(self) -> None:
        frames = [
            FrameSpec(scene_id=scene, anno_id=anno, camera="realsense")
            for scene in range(100, 108)
            for anno in (0, 10, 20)
        ]
        owner = {}
        union = []
        for rank in range(4):
            shard, mode = shard_frames(
                frames, rank=rank, world_size=4, mode="auto"
            )
            self.assertEqual(mode, "scene")
            union.extend(frame.key for frame in shard)
            for frame in shard:
                previous = owner.setdefault(frame.scene_id, rank)
                self.assertEqual(previous, rank)
        self.assertEqual(sorted(union), sorted(frame.key for frame in frames))

    def test_auto_sharding_falls_back_to_frames_for_one_scene(self) -> None:
        frames = [
            FrameSpec(scene_id=100, anno_id=anno, camera="realsense")
            for anno in range(8)
        ]
        counts = []
        for rank in range(4):
            shard, mode = shard_frames(
                frames, rank=rank, world_size=4, mode="auto"
            )
            self.assertEqual(mode, "frame")
            counts.append(len(shard))
        self.assertEqual(counts, [2, 2, 2, 2])

    def test_physical_action_dedup_ignores_score_and_object_id(self) -> None:
        grasps = make_grasps(3)
        grasps[1, 1:16] = grasps[0, 1:16]
        grasps[1, 0] = 0.123
        grasps[1, 16] = 42
        unique, inverse = deduplicate_physical_actions(grasps)
        self.assertEqual(unique.shape[0], 2)
        self.assertEqual(int(inverse[0]), int(inverse[1]))
        labels = np.asarray([0.2, 0.8], dtype=np.float32)
        self.assertEqual(float(labels[inverse][0]), float(labels[inverse][1]))

    def test_per_object_selection(self) -> None:
        grasps = make_grasps(6)
        assigned = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        selected = select_local_base_indices(
            grasps,
            assigned,
            top_n_per_object=2,
            global_top_n=0,
        )
        self.assertEqual(set(selected.tolist()), {0, 1, 3, 4})


if __name__ == "__main__":
    unittest.main()
