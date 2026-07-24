"""CDF-specific label adapter for the shared GraspNet dataset.

The shared ``dataset/graspnet_dataset.py`` must remain model-agnostic.  This
wrapper augments a base GraspNet sample with the compact depth-wise CDF and
width labels required by the CDF-only CVA trainer.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset


class CDFLabelAdapter(Dataset):
    """Attach CDF/width labels to a generic GraspNet dataset.

    The wrapped dataset remains responsible for RGB/depth/point-cloud inputs,
    scene metadata, object poses, objectness/graspness labels, and augmentation.
    This adapter only replaces the object-level grasp-label payload with the
    CDF-specific cache fields.

    Dense variable-size arrays stay as compact NumPy arrays on CPU.  The
    project collate function converts them to CPU tensors, and
    ``process_grasp_labels_cdf_width`` transfers only rows matched to the
    current queries.
    """

    _LEGACY_GRASP_KEYS = (
        "grasp_rotations_list",
        "grasp_depth_list",
        "grasp_widths_list",
        "grasp_scores_list",
        "grasp_collision_list",
        "grasp_cdf_bins_list",
        "grasp_widths_depth_list",
        "grasp_width_valids_depth_list",
        "cdf_thresholds",
    )

    _REQUIRED_CACHE_KEYS = (
        "points",
        "pointid",
        "vgraspness",
        "topview",
        "cdf_bins",
        "cdf_thresholds",
        "widths_depth_mm",
        "width_valids_depth",
    )

    def __init__(
        self,
        base_dataset: Dataset,
        dataset_root: str,
        label_folder: str = "economic_grasp_label_300views_extend_angle_cdf_depth",
    ) -> None:
        self.base_dataset = base_dataset
        self.dataset_root = os.path.abspath(dataset_root)
        self.label_folder = str(label_folder)
        self.label_root = os.path.join(self.dataset_root, self.label_folder)
        # A scene can contain objects for which the cache generator retained no
        # grasp points after filtering. Warn at most once per worker/process.
        self._warned_missing_object_scenes = set()

        if not os.path.isdir(self.label_root):
            raise FileNotFoundError(
                f"CDF label directory does not exist: {self.label_root}"
            )
        if not hasattr(base_dataset, "scenename"):
            raise AttributeError(
                "The wrapped dataset must expose scenename[index] so the "
                "adapter can locate scene-level CDF caches."
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        # Delegate public dataset metadata used by samplers/evaluation code.
        if name in {
            "base_dataset",
            "dataset_root",
            "label_folder",
            "label_root",
            "_warned_missing_object_scenes",
        }:
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def scene_list(self):
        if hasattr(self.base_dataset, "scene_list"):
            return self.base_dataset.scene_list()
        return self.base_dataset.scenename

    def _scene_name(self, index: int) -> str:
        scene = self.base_dataset.scenename[index]
        if isinstance(scene, bytes):
            scene = scene.decode("utf-8")
        return str(scene).strip()

    def _label_path(self, index: int) -> str:
        scene = self._scene_name(index)
        return os.path.join(self.label_root, f"{scene}_labels.npz")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base_dataset[index]
        if not isinstance(sample, dict):
            raise TypeError(
                f"Wrapped dataset must return dict samples, got {type(sample).__name__}."
            )
        # Avoid mutating a dict potentially cached/reused by the base dataset.
        sample = dict(sample)

        poses = sample.get("object_poses_list")
        if not isinstance(poses, (list, tuple)) or len(poses) == 0:
            raise KeyError(
                "CDFLabelAdapter requires object_poses_list from the shared "
                "training dataset. Ensure load_label=True."
            )
        num_objects = len(poses)

        label_path = self._label_path(index)
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"CDF label cache not found: {label_path}")

        with np.load(label_path, allow_pickle=False) as labels:
            missing = [key for key in self._REQUIRED_CACHE_KEYS if key not in labels]
            if missing:
                raise KeyError(
                    f"CDF cache {label_path} is missing required keys: {missing}"
                )

            points = labels["points"].astype(np.float32, copy=False)
            pointid = labels["pointid"].astype(np.int64, copy=False)
            view_graspness = labels["vgraspness"].astype(np.float32, copy=False)
            topview = labels["topview"].astype(np.int32, copy=False)
            cdf_bins = labels["cdf_bins"].astype(np.uint8, copy=False)
            thresholds = labels["cdf_thresholds"].astype(np.float32, copy=False)
            widths_mm = labels["widths_depth_mm"].astype(np.uint16, copy=False)
            width_valid = labels["width_valids_depth"].astype(np.uint8, copy=False)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be [P,3], got {points.shape} in {label_path}")
        if pointid.shape != (points.shape[0],):
            raise ValueError(
                f"pointid must be [P], got {pointid.shape} for P={points.shape[0]}"
            )
        if cdf_bins.ndim != 4:
            raise ValueError(
                f"cdf_bins must be [P,K,A,D], got {cdf_bins.shape} in {label_path}"
            )
        if cdf_bins.shape[0] != points.shape[0]:
            raise ValueError("cdf_bins and points have inconsistent first dimensions.")
        if widths_mm.shape != cdf_bins.shape or width_valid.shape != cdf_bins.shape:
            raise ValueError(
                "widths_depth_mm and width_valids_depth must match cdf_bins; "
                f"got cdf={cdf_bins.shape}, width={widths_mm.shape}, "
                f"valid={width_valid.shape}."
            )
        if topview.shape != cdf_bins.shape[:2]:
            raise ValueError(
                f"topview must be [P,K]={cdf_bins.shape[:2]}, got {topview.shape}."
            )
        if view_graspness.ndim != 2 or view_graspness.shape[0] != points.shape[0]:
            raise ValueError(
                f"vgraspness must be [P,V], got {view_graspness.shape}."
            )
        if thresholds.ndim != 1 or thresholds.size < 2:
            raise ValueError(
                f"cdf_thresholds must be [T>=2], got {thresholds.shape}."
            )
        if not np.all(np.isfinite(thresholds)) or not np.all(np.diff(thresholds) > 0):
            raise ValueError(f"Invalid CDF thresholds in {label_path}: {thresholds}")
        if np.any(pointid < 0) or np.any(pointid >= num_objects):
            bad = np.unique(pointid[(pointid < 0) | (pointid >= num_objects)])
            raise ValueError(
                f"Cache object indices {bad.tolist()} are incompatible with "
                f"{num_objects} object poses in {label_path}."
            )

        # The cache generator filters grasp points before saving. It is valid
        # for one scene object to lose every point (e.g. no retained grasp under
        # the configured quality/width constraints). Such an object has no
        # meaningful dense CDF supervision and must be omitted, not converted
        # into an all-zero negative object.
        active_object_ids = np.unique(pointid).astype(np.int64, copy=False)
        if active_object_ids.size == 0:
            raise RuntimeError(
                f"CDF cache contains no retained grasp points: {label_path}"
            )

        active_set = set(int(v) for v in active_object_ids.tolist())
        missing_object_ids = [
            obj_i for obj_i in range(num_objects) if obj_i not in active_set
        ]
        if missing_object_ids and label_path not in self._warned_missing_object_scenes:
            warnings.warn(
                "CDF cache has no retained grasp points for local object(s) "
                f"{missing_object_ids} in {label_path}. These objects are "
                "omitted from CDF/width grasp-head supervision; scene-level "
                "point/depth supervision is unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_missing_object_scenes.add(label_path)

        grasp_points_list = []
        view_graspness_list = []
        top_view_index_list = []
        cdf_list = []
        width_list = []
        width_valid_list = []
        active_poses = []

        # np.unique returns ascending ids, preserving the original local-object
        # order after missing objects are removed. Each list entry therefore
        # stays aligned with the corresponding filtered pose.
        for original_obj_i in active_object_ids.tolist():
            original_obj_i = int(original_obj_i)
            mask = pointid == original_obj_i
            num_obj_points = int(mask.sum())
            if num_obj_points <= 0:
                raise RuntimeError(
                    "Internal object filtering error for local object "
                    f"{original_obj_i} in {label_path}."
                )

            pose = poses[original_obj_i]
            pose_array = np.asarray(pose)
            if pose_array.shape != (3, 4):
                raise ValueError(
                    f"object_poses_list[{original_obj_i}] must be [3,4], "
                    f"got {pose_array.shape}."
                )

            active_poses.append(pose)
            grasp_points_list.append(points[mask])
            view_graspness_list.append(view_graspness[mask])
            top_view_index_list.append(topview[mask])
            cdf_list.append(cdf_bins[mask])
            width_list.append(widths_mm[mask])
            width_valid_list.append(width_valid[mask])

        # process_grasp_labels_cdf_width expects one pose for every object-wise
        # CDF payload. Filter the pose list using the same active-object order.
        sample["object_poses_list"] = active_poses

        # Remove only model-specific object-level grasp payloads. Scene/image
        # labels and object poses produced by the shared dataset remain intact.
        for key in self._LEGACY_GRASP_KEYS:
            sample.pop(key, None)

        sample["grasp_points_list"] = grasp_points_list
        sample["view_graspness_list"] = view_graspness_list
        sample["top_view_index_list"] = top_view_index_list
        sample["grasp_cdf_bins_list"] = cdf_list
        sample["grasp_widths_depth_list"] = width_list
        sample["grasp_width_valids_depth_list"] = width_valid_list
        sample["cdf_thresholds"] = thresholds
        return sample
