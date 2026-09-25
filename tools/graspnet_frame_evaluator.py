"""Single-frame GraspNet evaluator wrapper for visualization/diagnostics.

This reproduces the core official eval_scene path for arbitrary annotation ids,
including frames that cannot be expressed by anno_sample_ratio (e.g. 128/255).
It keeps only one scene's CAD/DexNet cache alive at a time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class FrameEvalResult:
    grasps: np.ndarray
    friction_scores: np.ndarray
    collision: np.ndarray
    accuracy: np.ndarray

    def as_npz(self):
        return {
            'grasps': self.grasps,
            'friction_scores': self.friction_scores,
            'collision': self.collision,
            'accuracy': self.accuracy,
        }


class SelectedFrameGraspEvaluator:
    """Official-protocol exact evaluator for sparse selected frames."""

    def __init__(self, dataset_root: str, camera: str = 'realsense',
                 top_k: int = 50, max_width: float = .1,
                 voxel_size: float = .008):
        from graspnetAPI import GraspNetEval
        from graspnetAPI.utils.config import get_config
        from graspnetAPI.utils.eval_utils import create_table_points

        self.root = dataset_root
        self.camera = camera
        self.top_k = int(top_k)
        self.max_width = float(max_width)
        self.voxel_size = float(voxel_size)
        self.ge = GraspNetEval(dataset_root, camera, split='test')
        self.config = get_config()
        self.table = create_table_points(
            1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05,
            grid_size=self.voxel_size)
        self.scene_id: Optional[int] = None
        self.model_sampled_list = None
        self.dexmodel_list = None

    def _prepare_scene(self, scene_id: int):
        scene_id = int(scene_id)
        if self.scene_id == scene_id:
            return
        from graspnetAPI.utils.eval_utils import voxel_sample_points
        models, dexmodels, _ = self.ge.get_scene_models(scene_id, ann_id=0)
        self.model_sampled_list = [
            voxel_sample_points(m, self.voxel_size) for m in models
        ]
        self.dexmodel_list = dexmodels
        self.scene_id = scene_id

    def clear(self):
        self.scene_id = None
        self.model_sampled_list = None
        self.dexmodel_list = None

    def evaluate(self, scene_id: int, anno_id: int,
                 grasp_array: np.ndarray) -> FrameEvalResult:
        from graspnetAPI import GraspGroup
        from graspnetAPI.utils.eval_utils import (
            eval_grasp, transform_points,
        )

        self._prepare_scene(scene_id)
        gg_array = np.asarray(grasp_array, np.float32).copy()
        if gg_array.ndim != 2 or gg_array.shape[1] != 17:
            raise ValueError(f'Expected GraspNet grasp array [N,17], got {gg_array.shape}')
        gg_array[:, 1] = np.clip(gg_array[:, 1], 0., self.max_width)
        group = GraspGroup(gg_array)

        _, poses, camera_pose, align_mat = self.ge.get_model_poses(
            int(scene_id), int(anno_id))
        table_trans = transform_points(
            self.table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

        grasp_lists, score_lists, collision_lists = eval_grasp(
            group,
            self.model_sampled_list,
            self.dexmodel_list,
            poses,
            self.config,
            table=table_trans,
            voxel_size=self.voxel_size,
            TOP_K=self.top_k)

        triples = [
            (g, s, c) for g, s, c in
            zip(grasp_lists, score_lists, collision_lists)
            if len(g) != 0
        ]
        if not triples:
            return FrameEvalResult(
                grasps=np.empty((0, 17), np.float32),
                friction_scores=np.empty((0,), np.float32),
                collision=np.empty((0,), np.bool_),
                accuracy=np.zeros((self.top_k, 6), np.float32))

        grasps = np.concatenate([x[0] for x in triples], axis=0)
        scores = np.concatenate([x[1] for x in triples], axis=0).astype(np.float32)
        collision = np.concatenate([x[2] for x in triples], axis=0).astype(np.bool_)

        # Official eval_scene sorts the post-assignment/per-object-TOP_K list
        # globally by the model confidence stored in column 0.
        order = np.argsort(-grasps[:, 0], kind='stable')
        grasps, scores, collision = (
            grasps[order], scores[order], collision[order])

        frictions = np.asarray([.2, .4, .6, .8, 1.0, 1.2], np.float32)
        accuracy = np.zeros((self.top_k, len(frictions)), np.float32)
        for fi, fric in enumerate(frictions):
            success = (scores <= fric) & (scores > 0)
            prefix = np.cumsum(success.astype(np.float32))
            for k in range(self.top_k):
                if k < len(prefix):
                    accuracy[k, fi] = prefix[k] / float(k + 1)
                elif len(prefix):
                    accuracy[k, fi] = prefix[-1] / float(k + 1)
        return FrameEvalResult(
            grasps=grasps,
            friction_scores=scores,
            collision=collision,
            accuracy=accuracy)
