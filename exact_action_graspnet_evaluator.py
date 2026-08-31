#!/usr/bin/env python3
"""Exact GraspNet CAD/DexNet evaluation for explicit grasp candidates.

The evaluator is independent of the network architecture. It assigns each raw
candidate to the nearest scene object, runs the official CAD/table collision
logic, and evaluates force closure at the official friction thresholds.

``reuse_contacts`` is a runtime optimization only: it acquires the two contacts
once and evaluates the same official force-closure configurations. Set
``fc_mode='official'`` for the stock implementation or use ``verify_n`` to
cross-check optimized labels on a smoke run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

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
    def __init__(
        self,
        dataset_root: str,
        camera: str,
        split: str = "train",
        collision_chunk: int = 512,
        fc_mode: str = "reuse_contacts",
        verify_n: int = 0,
        strict: bool = True,
    ) -> None:
        self.eval = GraspNetEval(dataset_root, camera, split=split)
        self.config = get_config()
        self.collision_chunk = max(1, int(collision_chunk))
        self.fc_mode = str(fc_mode)
        self.verify_n = max(0, int(verify_n))
        self.strict = bool(strict)
        if self.fc_mode not in {"official", "reuse_contacts"}:
            raise ValueError(
                f"fc_mode must be 'official' or 'reuse_contacts', got {self.fc_mode!r}."
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
        self.fc_list = np.asarray([1.2, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
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

    def _score_reuse_contacts(self, grasp: Any, obj: Any) -> Tuple[float, int, bool]:
        base_cfg = self.fc_configs[round(float(self.fc_list[0]), 2)]
        found, contacts = grasp.close_fingers(
            obj,
            check_approach=bool(getattr(base_cfg, "check_approach", False)),
            vis=False,
        )
        if not found:
            return -1.0, 0, False

        previous_success = False
        calls = 0
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
        return float(score), calls, True

    def _score(self, grasp: Any, obj: Any) -> Tuple[float, int, bool]:
        if self.fc_mode == "official":
            return (
                float(get_grasp_score(grasp, obj, self.fc_list, self.fc_configs)),
                -1,
                True,
            )
        return self._score_reuse_contacts(grasp, obj)

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
                    score, calls, contacts_found = self._score(
                        dexgrasps[local_index], dexmodels[obj_index]
                    )
                    friction[global_index] = float(score)
                    if calls > 0:
                        fc_quality_calls += int(calls)
                    contacts_not_found += int(not contacts_found)

                    if self.fc_mode != "official" and verify_count < self.verify_n:
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
                            print(f"[FC][WARN] {message}", flush=True)
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
                "fc_verify_count": int(verify_count),
                "fc_verify_mismatches": int(verify_mismatches),
                "fc_mode": self.fc_mode,
            },
        )
