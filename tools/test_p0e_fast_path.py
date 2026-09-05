#!/usr/bin/env python3
"""Dependency-light equivalence test for the v1.2 local-field fast path."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# run_p0e_oracles imports the real evaluator at module import time. Replace only
# that import for this synthetic equivalence test; _run_local_field accepts any
# object exposing evaluate(scene_id, anno_id, grasps).
fake_module = types.ModuleType("exact_action_graspnet_evaluator")
fake_module.ExactGraspNetActionEvaluator = object
sys.modules["exact_action_graspnet_evaluator"] = fake_module

spec = importlib.util.spec_from_file_location(
    "run_p0e_oracles_fast_test", REPO_ROOT / "run_p0e_oracles.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

from p0e_oracle_common import (  # noqa: E402
    FrameSpec,
    build_local_perturbations,
    pick_best_local_actions,
    select_local_base_indices,
    stack_local_lattice,
)


class FakeResult:
    def __init__(self, grasps: np.ndarray):
        g = np.asarray(grasps, np.float32)
        n = len(g)
        # Deterministic physical-action label; independent of score/object_id.
        x = g[:, 13]
        y = g[:, 14]
        depth = g[:, 3]
        width = g[:, 1]
        self.assigned_obj = (x >= 0.025).astype(np.int64)
        self.collision_or_empty = (np.abs(y) > 0.0045)
        self.pure_collision = self.collision_or_empty.copy()
        self.empty = np.zeros(n, dtype=bool)
        friction = np.full(n, 0.8, dtype=np.float32)
        friction[x > 0.015] = 0.4
        friction[depth > 0.025] = 0.2
        friction[width > 0.052] = 0.6
        friction[self.collision_or_empty] = -1.0
        self.friction = friction
        self.stats = {}


class FakeEvaluator:
    def __init__(self):
        self.calls = 0
        self.actions = 0

    def evaluate(self, _scene_id: int, _anno_id: int, grasps: np.ndarray):
        self.calls += 1
        self.actions += int(len(grasps))
        return FakeResult(grasps)


def make_grasps() -> np.ndarray:
    g = np.zeros((6, 17), dtype=np.float32)
    g[:, 0] = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], np.float32)
    g[:, 1] = 0.05
    g[:, 2] = 0.02
    g[:, 3] = 0.02
    g[:, 4:13] = np.eye(3, dtype=np.float32).reshape(1, 9)
    g[:, 13] = np.asarray([0.0, 0.01, 0.02, 0.03, 0.04, 0.05], np.float32)
    g[:, 16] = -1
    return g


def main() -> None:
    student = make_grasps()
    student_result = FakeResult(student)
    perturbations = build_local_perturbations()
    args = SimpleNamespace(
        local_top_n_per_object=3,
        local_global_top_n=0,
        min_width_m=0.0,
        max_width_m=0.10,
        min_depth_m=0.0,
        max_depth_m=0.10,
        tie_break_eps=1.0e-4,
    )
    frame = FrameSpec(100, 0, "realsense")
    evaluator = FakeEvaluator()
    fast_output, fast_meta, fast_summary = module._run_local_field(
        evaluator=evaluator,
        frame=frame,
        student_grasps=student,
        student_result=student_result,
        perturbations=perturbations,
        args=args,
    )

    selected = select_local_base_indices(
        student,
        student_result.assigned_obj,
        top_n_per_object=args.local_top_n_per_object,
        global_top_n=args.local_global_top_n,
    )
    lattice = stack_local_lattice(
        student[selected],
        perturbations,
        min_width_m=args.min_width_m,
        max_width_m=args.max_width_m,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    full = FakeResult(lattice.reshape(-1, 17))
    full_friction = full.friction.reshape(lattice.shape[:2])
    best_actions, best_friction, _best_utility, best_ids = pick_best_local_actions(
        lattice, full_friction
    )

    np.testing.assert_allclose(
        fast_meta["local_lattice_friction"], full_friction, atol=0.0, rtol=0.0
    )
    np.testing.assert_array_equal(fast_meta["local_best_perturb_id"], best_ids)
    np.testing.assert_allclose(fast_output[selected, 1:16], best_actions[:, 1:16])
    np.testing.assert_allclose(fast_meta["local_best_friction"], best_friction)
    logical = int(selected.size * len(perturbations))
    assert evaluator.actions < logical, (evaluator.actions, logical)
    assert int(fast_summary["local_identity_labels_reused"]) == int(selected.size)
    print(
        "PASS: local-field fast path is label/output equivalent; "
        f"logical_actions={logical}, evaluator_actions={evaluator.actions}"
    )


if __name__ == "__main__":
    main()
