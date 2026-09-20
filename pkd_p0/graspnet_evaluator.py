"""Exact GraspNet CAD/DexNet evaluator for P0-B candidate diagnostics.

This module is intentionally network-architecture agnostic.  It evaluates the
explicit [N,17] grasp rows dumped by ``p0_dump_current_candidates.py`` using
the same scene-model assignment, CAD/table collision check, and force-closure
friction sweep as the repository's exact-action evaluator.

The module lives under ``pkd_p0`` because ``p0_b_candidate_ranking.py`` imports
``pkd_p0.graspnet_evaluator``.  The original P0 overlay accidentally omitted
this file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from graspnetAPI import GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    collision_detection,
    compute_closest_points,
    create_table_points,
    get_grasp_score,
    transform_points,
    voxel_sample_points,
)
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import (
    GraspQualityConfigFactory,
)
from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D


@dataclass
class ExactActionEvalResult:
    assigned_obj: np.ndarray
    collision_or_empty: np.ndarray
    pure_collision: np.ndarray
    empty: np.ndarray
    friction: np.ndarray
    stats: Dict[str, Any]


class ExactGraspNetActionEvaluator:
    """Evaluate explicit grasp rows against GraspNet scene geometry.

    Parameters accept both the names used by P0-B and the names used by the
    earlier exact-action evaluator, so ``filtered_kwargs`` can pass them safely.
    ``reuse_contacts_binary`` changes only the number of quality evaluations;
    labels are verified against the official routine when ``fc_verify_n > 0``.
    """

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        camera: str = "realsense",
        split: str = "train",
        collision_chunk: int = 512,
        fc_mode: str = "reuse_contacts",
        fc_verify_n: int = 0,
        strict: bool = True,
        *,
        root: Optional[str] = None,
        chunk: Optional[int] = None,
        verify_n: Optional[int] = None,
        skip_force_closure: bool = False,
    ) -> None:
        if dataset_root is None:
            dataset_root = root
        if not dataset_root:
            raise ValueError("dataset_root/root is required")
        if chunk is not None:
            collision_chunk = int(chunk)
        if verify_n is not None:
            fc_verify_n = int(verify_n)
        self.skip_force_closure = bool(skip_force_closure)
        if self.skip_force_closure:
            raise ValueError(
                "P0-B candidate/ranking AP diagnostics require force-closure "
                "labels; do not pass --skip_force_closure 1."
            )

        self.eval = GraspNetEval(str(dataset_root), str(camera), split=str(split))
        self.config = get_config()
        self.collision_chunk = max(1, int(collision_chunk))
        self.fc_mode = str(fc_mode)
        self.fc_verify_n = max(0, int(fc_verify_n))
        self.strict = bool(strict)
        if self.fc_mode not in {
            "official",
            "reuse_contacts",
            "reuse_contacts_binary",
        }:
            raise ValueError(
                "fc_mode must be official, reuse_contacts, or "
                f"reuse_contacts_binary; got {self.fc_mode!r}."
            )

        self.table = create_table_points(
            1.0,
            1.0,
            0.05,
            dx=-0.5,
            dy=-0.5,
            dz=-0.05,
            grid_size=0.008,
        )
        self.scene_cache: Dict[int, Tuple[List[np.ndarray], List[Any]]] = {}
        # Preserve the official descending order exactly.
        self.fc_list = np.asarray(
            [1.2, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64
        )
        self.fc_ascending = self.fc_list[::-1].copy()
        self.fc_configs: Dict[float, Any] = {}
        for mu in self.fc_list:
            key = round(float(mu), 2)
            self.config["metrics"]["force_closure"]["friction_coef"] = key
            self.fc_configs[key] = GraspQualityConfigFactory.create_config(
                self.config["metrics"]["force_closure"]
            )

    def _scene_models(self, scene_id: int) -> Tuple[List[np.ndarray], List[Any]]:
        scene_id = int(scene_id)
        if scene_id not in self.scene_cache:
            models, dexmodels, _ = self.eval.get_scene_models(scene_id, ann_id=0)
            self.scene_cache[scene_id] = (
                [voxel_sample_points(model, 0.008) for model in models],
                dexmodels,
            )
        return self.scene_cache[scene_id]

    def _quality_with_contacts(
        self,
        grasp: Any,
        obj: Any,
        contacts: Sequence[Any],
        friction: float,
    ) -> bool:
        key = round(float(friction), 2)
        return bool(
            PointGraspMetrics3D.grasp_quality(
                grasp,
                obj,
                self.fc_configs[key],
                contacts=contacts,
            )
        )

    def _open_contacts(self, grasp: Any, obj: Any) -> Tuple[bool, Sequence[Any]]:
        base_cfg = self.fc_configs[round(float(self.fc_list[0]), 2)]
        return grasp.close_fingers(
            obj,
            check_approach=bool(getattr(base_cfg, "check_approach", False)),
            vis=False,
        )

    def _score_reuse_contacts(
        self,
        grasp: Any,
        obj: Any,
        *,
        binary: bool,
    ) -> Tuple[float, int, bool, bool]:
        found, contacts = self._open_contacts(grasp, obj)
        if not found:
            return -1.0, 0, False, False

        # Binary search is valid only for the normal two-contact force-closure
        # case. Fall back to the official-order sweep otherwise.
        use_binary = bool(binary and len(contacts) == 2)
        calls = 0
        if use_binary:
            lo, hi = 0, len(self.fc_ascending)
            while lo < hi:
                mid = (lo + hi) // 2
                calls += 1
                if self._quality_with_contacts(
                    grasp, obj, contacts, float(self.fc_ascending[mid])
                ):
                    hi = mid
                else:
                    lo = mid + 1
            if lo >= len(self.fc_ascending):
                return -1.0, calls, True, False
            return (
                round(float(self.fc_ascending[lo]), 2),
                calls,
                True,
                False,
            )

        previous_success = False
        score = -1.0
        for index, value in enumerate(self.fc_list):
            friction = round(float(value), 2)
            old_success = previous_success
            calls += 1
            previous_success = self._quality_with_contacts(
                grasp, obj, contacts, friction
            )
            if old_success and not previous_success:
                score = round(float(self.fc_list[index - 1]), 2)
                break
            if previous_success and index == len(self.fc_list) - 1:
                score = friction
                break
            if index == 0 and not previous_success:
                break
        return float(score), calls, True, bool(binary and not use_binary)

    def _score(self, grasp: Any, obj: Any) -> Tuple[float, int, bool, bool]:
        if self.fc_mode == "official":
            return (
                float(get_grasp_score(grasp, obj, self.fc_list, self.fc_configs)),
                -1,
                True,
                False,
            )
        return self._score_reuse_contacts(
            grasp,
            obj,
            binary=(self.fc_mode == "reuse_contacts_binary"),
        )

    def evaluate(
        self,
        scene_id: int,
        anno_id: int,
        grasps: np.ndarray,
    ) -> ExactActionEvalResult:
        grasps = np.asarray(grasps, dtype=np.float32)
        if grasps.ndim != 2 or grasps.shape[1] != 17:
            raise ValueError(f"grasps must be [N,17], got {grasps.shape}")
        n = int(grasps.shape[0])
        if n == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            empty_b = np.zeros(0, dtype=bool)
            empty_f = np.zeros(0, dtype=np.float32)
            return ExactActionEvalResult(
                empty_i,
                empty_b,
                empty_b,
                empty_b,
                empty_f,
                {
                    "collision_sec": 0.0,
                    "force_closure_sec": 0.0,
                    "fc_candidates": 0,
                    "fc_quality_calls": 0,
                    "fc_contacts_not_found": 0,
                    "fc_binary_fallbacks": 0,
                    "fc_verify_count": 0,
                    "fc_verify_mismatches": 0,
                    "fc_mode": self.fc_mode,
                },
            )

        models_obj, dexmodels = self._scene_models(int(scene_id))
        _, poses, camera_pose, align_mat = self.eval.get_model_poses(
            int(scene_id), int(anno_id)
        )
        models_cam = [
            transform_points(model, poses[obj_index])
            for obj_index, model in enumerate(models_obj)
        ]
        scene = np.concatenate(models_cam, axis=0)
        segmentation = np.concatenate(
            [
                np.full(len(model), obj_index, dtype=np.int64)
                for obj_index, model in enumerate(models_cam)
            ],
            axis=0,
        )
        nearest = compute_closest_points(grasps[:, 13:16], scene)
        assigned = segmentation[nearest]

        table_cam = transform_points(
            self.table,
            np.linalg.inv(np.matmul(align_mat, camera_pose)),
        )
        scene_with_table = np.concatenate([scene, table_cam], axis=0)

        collision_or_empty = np.zeros(n, dtype=bool)
        empty = np.zeros(n, dtype=bool)
        friction = np.full(n, -1.0, dtype=np.float32)
        collision_sec = 0.0
        force_closure_sec = 0.0
        fc_candidates = 0
        fc_quality_calls = 0
        contacts_not_found = 0
        binary_fallbacks = 0
        verify_count = 0
        verify_mismatches = 0

        for obj_index in range(len(models_cam)):
            object_rows = np.flatnonzero(assigned == obj_index)
            for start in range(0, len(object_rows), self.collision_chunk):
                ids = object_rows[start : start + self.collision_chunk]
                if len(ids) == 0:
                    continue
                chunk_grasps = grasps[ids]
                collision_start = time.perf_counter()
                collision_list, empty_list, dexgrasp_list = collision_detection(
                    [chunk_grasps],
                    [models_cam[obj_index]],
                    [dexmodels[obj_index]],
                    [poses[obj_index]],
                    scene_with_table,
                    outlier=0.05,
                    return_dexgrasps=True,
                )
                collision_sec += time.perf_counter() - collision_start
                collision_chunk = np.asarray(collision_list[0], dtype=bool)
                empty_chunk = np.asarray(empty_list[0], dtype=bool)
                collision_or_empty[ids] = collision_chunk
                empty[ids] = empty_chunk

                dexgrasps = dexgrasp_list[0]
                force_start = time.perf_counter()
                for local_index, global_index in enumerate(ids):
                    if collision_chunk[local_index] or dexgrasps[local_index] is None:
                        continue
                    fc_candidates += 1
                    score, calls, contacts_found, binary_fallback = self._score(
                        dexgrasps[local_index], dexmodels[obj_index]
                    )
                    friction[global_index] = float(score)
                    if calls > 0:
                        fc_quality_calls += int(calls)
                    contacts_not_found += int(not contacts_found)
                    binary_fallbacks += int(binary_fallback)

                    if self.fc_mode != "official" and verify_count < self.fc_verify_n:
                        official = float(
                            get_grasp_score(
                                dexgrasps[local_index],
                                dexmodels[obj_index],
                                self.fc_list,
                                self.fc_configs,
                            )
                        )
                        verify_count += 1
                        if not np.isclose(official, score, atol=1e-6, rtol=0.0):
                            verify_mismatches += 1
                            message = (
                                "Force-closure label mismatch: "
                                f"scene={scene_id}, anno={anno_id}, obj={obj_index}, "
                                f"candidate={int(global_index)}, optimized={score}, "
                                f"official={official}."
                            )
                            if self.strict:
                                raise RuntimeError(message)
                            print(f"[P0-B][FC][WARN] {message}", flush=True)
                force_closure_sec += time.perf_counter() - force_start

        return ExactActionEvalResult(
            assigned_obj=assigned.astype(np.int64),
            collision_or_empty=collision_or_empty,
            pure_collision=collision_or_empty & (~empty),
            empty=empty,
            friction=friction,
            stats={
                "collision_sec": float(collision_sec),
                "force_closure_sec": float(force_closure_sec),
                "fc_candidates": int(fc_candidates),
                "fc_quality_calls": int(fc_quality_calls),
                "fc_contacts_not_found": int(contacts_not_found),
                "fc_binary_fallbacks": int(binary_fallbacks),
                "fc_verify_count": int(verify_count),
                "fc_verify_mismatches": int(verify_mismatches),
                "fc_mode": self.fc_mode,
            },
        )


# Alternate name recognized by p0_b_candidate_ranking.py.
RawCandidateEvaluator = ExactGraspNetActionEvaluator
