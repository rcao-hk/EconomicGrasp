"""Mode-aware extended-angle label adapter for the CVA Transformer.

Every Center-View-Angle (CVA) variant needs the extended per-view-per-angle
cache.  The common cache

    economic_grasp_label_300views_extend_angle_cdf_depth

is a strict superset containing both:

* legacy CVA labels [P,K,A]: depth / score / width / collision;
* CDF labels [P,K,A,D]: cdf_bins / depth-wise width / width validity.

The wrapped ``GraspNetMultiDataset`` supplies image/depth/point inputs and
per-frame object poses, but must not open an object-level grasp cache itself.
This adapter is the sole cache owner, preventing duplicate I/O and preserving
the compact uint16-millimetre width representation needed by the CDF matcher.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset, get_worker_info


class CVAExtendedLabelAdapter(Dataset):
    """Attach mode-specific labels from one common extended CVA cache."""

    _COMMON_KEYS = (
        "points",
        "pointid",
        "vgraspness",
        "topview",
        "extend_angle",
        "num_angle",
        "num_depth",
    )
    _LEGACY_KEYS = (
        "rotations",
        "depth",
        "scores",
        "widths",
        "collisions",
    )
    _CDF_KEYS = (
        "cdf_bins",
        "cdf_thresholds",
        "widths_depth_mm",
        "width_valids_depth",
    )
    _ALL_PAYLOAD_KEYS = (
        "grasp_points_list",
        "grasp_rotations_list",
        "grasp_depth_list",
        "grasp_widths_list",
        "grasp_scores_list",
        "view_graspness_list",
        "top_view_index_list",
        "grasp_collision_list",
        "grasp_cdf_bins_list",
        "grasp_widths_depth_list",
        "grasp_width_valids_depth_list",
        "cdf_thresholds",
    )

    def __init__(
        self,
        base_dataset: Dataset,
        dataset_root: str,
        use_cdf: bool = False,
        label_folder: str = (
            "economic_grasp_label_300views_extend_angle_cdf_depth"
        ),
        num_angle: int = 12,
        num_depth: int = 4,
    ) -> None:
        self.base_dataset = base_dataset
        self.dataset_root = os.path.abspath(dataset_root)
        self.use_cdf = bool(use_cdf)
        self.label_folder = str(label_folder)
        self.label_root = os.path.join(
            self.dataset_root,
            self.label_folder,
        )
        self.num_angle = int(num_angle)
        self.num_depth = int(num_depth)
        self._warned_missing_object_scenes = set()
        self._printed_width_unit = False

        if self.num_angle <= 0 or self.num_depth <= 0:
            raise ValueError(
                "num_angle and num_depth must be positive, got "
                f"{self.num_angle}/{self.num_depth}."
            )
        if not os.path.isdir(self.label_root):
            raise FileNotFoundError(
                f"Extended CVA label directory does not exist: "
                f"{self.label_root}"
            )
        if not hasattr(base_dataset, "scenename"):
            raise AttributeError(
                "The wrapped dataset must expose scenename[index]."
            )
        if bool(getattr(base_dataset, "load_grasp_payload", True)):
            raise ValueError(
                "CVAExtendedLabelAdapter must be the sole owner of the "
                "extended cache. Construct GraspNetMultiDataset with "
                "load_grasp_payload=False."
            )
        if not bool(getattr(base_dataset, "extend_angle", False)):
            raise ValueError(
                "The wrapped CVA base dataset must declare "
                "extend_angle=True."
            )
        self._validate_first_cache_schema()

    def _validate_first_cache_schema(self) -> None:
        """Fail before DataLoader workers start when cache routing is wrong."""
        if len(self.base_dataset) <= 0:
            raise RuntimeError("The wrapped CVA dataset is empty.")
        path = self._label_path(0)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Extended CVA cache not found: {path}"
            )
        required = self._COMMON_KEYS + (
            self._CDF_KEYS
            if self.use_cdf
            else self._LEGACY_KEYS
        )
        with np.load(path, allow_pickle=False) as labels:
            missing = [
                key for key in required
                if key not in labels
            ]
            if missing:
                mode = (
                    "CDF"
                    if self.use_cdf
                    else "legacy explicit-angle"
                )
                raise KeyError(
                    f"{mode} CVA cache {path} is missing keys: "
                    f"{missing}"
                )
            extend_marker = self._scalar_int(
                labels, "extend_angle", path
            )
            cache_num_angle = self._scalar_int(
                labels, "num_angle", path
            )
            cache_num_depth = self._scalar_int(
                labels, "num_depth", path
            )
        if extend_marker != 1:
            raise RuntimeError(
                f"Cache is not extended-angle: {path}"
            )
        if cache_num_angle != self.num_angle:
            raise RuntimeError(
                f"Cache num_angle={cache_num_angle}, model "
                f"num_angle={self.num_angle}: {path}"
            )
        if cache_num_depth != self.num_depth:
            raise RuntimeError(
                f"Cache num_depth={cache_num_depth}, model "
                f"num_depth={self.num_depth}: {path}"
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        if name in {
            "base_dataset",
            "dataset_root",
            "use_cdf",
            "label_folder",
            "label_root",
            "num_angle",
            "num_depth",
            "_warned_missing_object_scenes",
            "_printed_width_unit",
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
        return os.path.join(
            self.label_root,
            f"{self._scene_name(index)}_labels.npz",
        )

    @staticmethod
    def _scalar_int(labels, key: str, path: str) -> int:
        value = np.asarray(labels[key]).reshape(-1)
        if value.size != 1:
            raise ValueError(
                f"{key} must contain one scalar, got shape "
                f"{labels[key].shape} in {path}."
            )
        return int(value[0])

    @staticmethod
    def _require_shape(
        name: str,
        value: np.ndarray,
        expected,
        path: str,
    ) -> None:
        if tuple(value.shape) != tuple(expected):
            raise ValueError(
                f"{name} must be {tuple(expected)}, got "
                f"{tuple(value.shape)} in {path}."
            )

    @staticmethod
    def _rank0_worker0() -> bool:
        """Print once from rank 0 / DataLoader worker 0 only."""
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        return rank == 0 and worker_id == 0

    @staticmethod
    def _finite_stats(values: np.ndarray):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return (
            float(finite.min()),
            float(finite.mean()),
            float(finite.max()),
            int(finite.size),
        )

    def _maybe_print_width_unit(
        self,
        *,
        path: str,
        raw_width: np.ndarray,
        valid_mask: np.ndarray | None,
        key: str,
    ) -> None:
        """Report raw cache widths and the exact conversion used downstream."""
        if self._printed_width_unit or not self._rank0_worker0():
            return

        raw = np.asarray(raw_width)
        if valid_mask is not None:
            valid = np.asarray(valid_mask).astype(bool, copy=False)
            if valid.shape != raw.shape:
                raise ValueError(
                    f"{key} valid-mask shape {valid.shape} does not match "
                    f"width shape {raw.shape} in {path}."
                )
            selected = raw[valid]
        else:
            selected = raw.reshape(-1)

        # Zero is a valid encoded value for invalid/unavailable widths. Prefer
        # positive valid entries for the diagnostic, but fall back to all valid
        # entries when a cache slice contains only zeros.
        positive = selected[selected > 0]
        stats_source = positive if positive.size > 0 else selected
        stats = self._finite_stats(stats_source)

        integer_storage = np.issubdtype(raw.dtype, np.integer)
        if stats is None:
            raw_text = "empty"
            metre_text = "empty"
            status = "CHECK"
        else:
            raw_min, raw_mean, raw_max, count = stats
            raw_text = (
                f"min/mean/max={raw_min:.3f}/{raw_mean:.3f}/"
                f"{raw_max:.3f}, n={count}"
            )
            metre_text = (
                f"{raw_min * 1.0e-3:.6f}/"
                f"{raw_mean * 1.0e-3:.6f}/"
                f"{raw_max * 1.0e-3:.6f} m"
            )
            plausible_mm = raw_max <= 200.0 and raw_mean >= 0.5
            status = (
                "OK-mm"
                if integer_storage and plausible_mm
                else "CHECK"
            )

        if self.use_cdf:
            conversion = (
                "adapter keeps integer millimetres; "
                "process_grasp_labels_cdf_width multiplies by 1e-3 once"
            )
        else:
            conversion = (
                "adapter converts cache millimetres to metres with /1000 once"
            )

        print(
            "[CVA-WIDTH-CACHE] "
            f"mode={'CDF' if self.use_cdf else 'legacy'} "
            f"key={key} dtype={raw.dtype} declared_unit=millimetres "
            f"status={status} raw[{raw_text}] "
            f"as_metres[min/mean/max]={metre_text} "
            f"conversion='{conversion}' path={path}",
            flush=True,
        )
        self._printed_width_unit = True

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base_dataset[index]
        if not isinstance(sample, dict):
            raise TypeError(
                "Wrapped dataset must return a dict, got "
                f"{type(sample).__name__}."
            )
        sample = dict(sample)

        poses = sample.get("object_poses_list")
        if not isinstance(poses, (list, tuple)) or len(poses) == 0:
            raise KeyError(
                "CVAExtendedLabelAdapter requires object_poses_list "
                "from the per-frame meta file."
            )
        num_objects = len(poses)

        path = self._label_path(index)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Extended CVA cache not found: {path}"
            )

        required = self._COMMON_KEYS + (
            self._CDF_KEYS
            if self.use_cdf
            else self._LEGACY_KEYS
        )
        with np.load(path, allow_pickle=False) as labels:
            missing = [
                key for key in required
                if key not in labels
            ]
            if missing:
                mode = "CDF" if self.use_cdf else "legacy explicit-angle"
                raise KeyError(
                    f"{mode} CVA cache {path} is missing keys: "
                    f"{missing}"
                )

            extend_marker = self._scalar_int(
                labels, "extend_angle", path
            )
            cache_num_angle = self._scalar_int(
                labels, "num_angle", path
            )
            cache_num_depth = self._scalar_int(
                labels, "num_depth", path
            )
            if extend_marker != 1:
                raise RuntimeError(
                    f"Cache is not extended-angle: {path}"
                )
            if cache_num_angle != self.num_angle:
                raise RuntimeError(
                    f"Cache num_angle={cache_num_angle}, model "
                    f"num_angle={self.num_angle}: {path}"
                )
            if cache_num_depth != self.num_depth:
                raise RuntimeError(
                    f"Cache num_depth={cache_num_depth}, model "
                    f"num_depth={self.num_depth}: {path}"
                )

            points = labels["points"].astype(
                np.float32, copy=False
            )
            pointid = labels["pointid"].astype(
                np.int64, copy=False
            )
            view = labels["vgraspness"].astype(
                np.float32, copy=False
            )
            topview = labels["topview"].astype(
                np.int32, copy=False
            )

            if self.use_cdf:
                cdf = labels["cdf_bins"].astype(
                    np.uint8, copy=False
                )
                thresholds = labels[
                    "cdf_thresholds"
                ].astype(np.float32, copy=False)
                width_depth_raw = labels["widths_depth_mm"]
                width_depth_valid = labels[
                    "width_valids_depth"
                ].astype(np.uint8, copy=False)
                self._maybe_print_width_unit(
                    path=path,
                    raw_width=width_depth_raw,
                    valid_mask=width_depth_valid,
                    key="widths_depth_mm",
                )
                width_depth = width_depth_raw.astype(
                    np.uint16,
                    copy=False,
                )
            else:
                rotations = labels["rotations"].astype(
                    np.int32, copy=False
                )
                depth = labels["depth"].astype(
                    np.int32, copy=False
                )
                scores = (
                    labels["scores"].astype(np.float32)
                    / 10.0
                )
                widths_raw = labels["widths"]
                self._maybe_print_width_unit(
                    path=path,
                    raw_width=widths_raw,
                    valid_mask=None,
                    key="widths",
                )
                widths = (
                    widths_raw.astype(np.float32)
                    / 1000.0
                )
                collisions = labels[
                    "collisions"
                ].astype(np.float32, copy=False)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points must be [P,3], got {points.shape} "
                f"in {path}."
            )
        P = points.shape[0]
        if pointid.shape != (P,):
            raise ValueError(
                f"pointid must be [P], got {pointid.shape}."
            )
        if (
            view.ndim != 2
            or view.shape[0] != P
        ):
            raise ValueError(
                f"vgraspness must be [P,V], got {view.shape}."
            )
        if (
            topview.ndim != 2
            or topview.shape[0] != P
        ):
            raise ValueError(
                f"topview must be [P,K], got {topview.shape}."
            )
        K = topview.shape[1]

        if self.use_cdf:
            expected = (
                P,
                K,
                self.num_angle,
                self.num_depth,
            )
            self._require_shape(
                "cdf_bins", cdf, expected, path
            )
            self._require_shape(
                "widths_depth_mm",
                width_depth,
                expected,
                path,
            )
            self._require_shape(
                "width_valids_depth",
                width_depth_valid,
                expected,
                path,
            )
            if (
                thresholds.ndim != 1
                or thresholds.size < 2
                or not np.all(np.isfinite(thresholds))
                or not np.all(np.diff(thresholds) > 0)
            ):
                raise ValueError(
                    f"Invalid cdf_thresholds in {path}: "
                    f"{thresholds}"
                )
        else:
            expected = (P, K, self.num_angle)
            for name, value in (
                ("rotations", rotations),
                ("depth", depth),
                ("scores", scores),
                ("widths", widths),
                ("collisions", collisions),
            ):
                self._require_shape(
                    name, value, expected, path
                )

        if (
            np.any(pointid < 0)
            or np.any(pointid >= num_objects)
        ):
            bad = np.unique(
                pointid[
                    (pointid < 0)
                    | (pointid >= num_objects)
                ]
            )
            raise ValueError(
                f"Cache object ids {bad.tolist()} are "
                f"incompatible with {num_objects} poses in "
                f"{path}."
            )

        # The generator may remove every retained point for one object.  Such
        # an object is omitted from grasp-head supervision, while its scene
        # pixels still contribute to depth/objectness/graspness supervision.
        active_ids = np.unique(pointid).astype(
            np.int64, copy=False
        )
        if active_ids.size == 0:
            raise RuntimeError(
                f"No retained CVA points in {path}."
            )
        active_set = {
            int(value)
            for value in active_ids.tolist()
        }
        missing_objects = [
            obj_i
            for obj_i in range(num_objects)
            if obj_i not in active_set
        ]
        if (
            missing_objects
            and path
            not in self._warned_missing_object_scenes
        ):
            warnings.warn(
                "Extended CVA cache has no retained points "
                f"for local object(s) {missing_objects} in "
                f"{path}; those objects are omitted from "
                "grasp-head supervision.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_missing_object_scenes.add(path)

        for key in self._ALL_PAYLOAD_KEYS:
            sample.pop(key, None)

        active_poses = []
        point_list = []
        view_list = []
        topview_list = []
        legacy_lists = {
            "grasp_rotations_list": [],
            "grasp_depth_list": [],
            "grasp_scores_list": [],
            "grasp_widths_list": [],
            "grasp_collision_list": [],
        }
        cdf_lists = {
            "grasp_cdf_bins_list": [],
            "grasp_widths_depth_list": [],
            "grasp_width_valids_depth_list": [],
        }

        for obj_i in active_ids.tolist():
            obj_i = int(obj_i)
            mask = pointid == obj_i
            pose = np.asarray(poses[obj_i])
            if pose.shape != (3, 4):
                raise ValueError(
                    f"object_poses_list[{obj_i}] must be "
                    f"[3,4], got {pose.shape}."
                )
            active_poses.append(poses[obj_i])
            point_list.append(points[mask])
            view_list.append(view[mask])
            topview_list.append(topview[mask])

            if self.use_cdf:
                cdf_lists[
                    "grasp_cdf_bins_list"
                ].append(cdf[mask])
                cdf_lists[
                    "grasp_widths_depth_list"
                ].append(width_depth[mask])
                cdf_lists[
                    "grasp_width_valids_depth_list"
                ].append(width_depth_valid[mask])
            else:
                legacy_lists[
                    "grasp_rotations_list"
                ].append(rotations[mask])
                legacy_lists[
                    "grasp_depth_list"
                ].append(depth[mask])
                legacy_lists[
                    "grasp_scores_list"
                ].append(scores[mask])
                legacy_lists[
                    "grasp_widths_list"
                ].append(widths[mask])
                legacy_lists[
                    "grasp_collision_list"
                ].append(collisions[mask])

        sample["object_poses_list"] = active_poses
        sample["grasp_points_list"] = point_list
        sample["view_graspness_list"] = view_list
        sample["top_view_index_list"] = topview_list
        if self.use_cdf:
            sample.update(cdf_lists)
            sample["cdf_thresholds"] = thresholds
        else:
            sample.update(legacy_lists)
        return sample


# Backward-compatible import for older training scripts.
CDFLabelAdapter = CVAExtendedLabelAdapter
