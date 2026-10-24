"""Flexible reader for existing privileged-KD diagnosis caches.

The repository has evolved through several diagnostic revisions. This reader
accepts a user-supplied alias JSON and also recognizes the endpoint names used
by the current CDF/PKD implementation. It converts every sample into a compact,
row-aligned representation consumed by P0-C and P0-D.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .common import CDF_THRESHOLDS


DEFAULT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "student_logits": (
        "student_cdf_logits",
        "student_grasp_cdf_pred_angle_depth",
        "grasp_cdf_pred_angle_depth_student",
        "grasp_cdf_pred_angle_depth",
        "student/cdf_logits",
    ),
    "teacher_logits": (
        "teacher_cdf_logits",
        "teacher_grasp_cdf_pred_angle_depth",
        "grasp_cdf_pred_angle_depth_teacher",
        "teacher/cdf_logits",
    ),
    "gt_cdf": (
        "gt_cdf",
        "cdf_target",
        "cdf_targets",
        "grasp_cdf_label",
        "batch_grasp_cdf",
        "target_cdf",
    ),
    "friction": (
        "friction",
        "friction_label",
        "grasp_friction_label",
        "batch_grasp_score",
        "grasp_score_label",
    ),
    "valid_mask": (
        "valid_mask",
        "kd_valid_mask",
        "cdf_valid_mask",
        "query_valid_mask",
        "common_valid_mask",
    ),
    "common_valid": (
        "common_valid",
        "common_valid_mask",
        "teacher_student_common_valid",
    ),
    "center_z_error": (
        "center_z_error",
        "center_depth_error",
        "query_center_z_error",
        "depth_center_abs_error",
        "center_z_mae",
    ),
    "support_iou": (
        "support_iou",
        "valid_support_iou",
        "teacher_student_support_iou",
    ),
    "teacher_better": (
        "teacher_better",
        "teacher_better_mask",
        "cdf_teacher_better",
    ),
    "scene_id": ("scene_id", "scene_idx", "scene"),
    "anno_id": ("anno_id", "ann_id", "frame_id", "annotation_id"),
    "token_index": ("token_sel_idx", "token_index", "seed_index", "query_index"),
    "view_index": ("grasp_top_view_inds", "view_index", "selected_view_index"),
    "feature_seed_student": (
        "student_seed_feature",
        "seed_feature_student",
        "feature_seed_student",
    ),
    "feature_seed_teacher": (
        "teacher_seed_feature",
        "seed_feature_teacher",
        "feature_seed_teacher",
    ),
    "feature_pre_view_student": (
        "student_pre_view_feature",
        "pre_view_feature_student",
        "feature_pre_view_student",
    ),
    "feature_pre_view_teacher": (
        "teacher_pre_view_feature",
        "pre_view_feature_teacher",
        "feature_pre_view_teacher",
    ),
    "feature_selected_view_student": (
        "student_selected_view_feature",
        "selected_view_feature_student",
        "feature_selected_view_student",
    ),
    "feature_selected_view_teacher": (
        "teacher_selected_view_feature",
        "selected_view_feature_teacher",
        "feature_selected_view_teacher",
    ),
    "feature_local_student": (
        "student_local_feature",
        "student_cva_feature",
        "local_feature_student",
        "feature_local_student",
    ),
    "feature_local_teacher": (
        "teacher_local_feature",
        "teacher_cva_feature",
        "local_feature_teacher",
        "feature_local_teacher",
    ),
    "feature_pre_cdf_student": (
        "student_pre_cdf_feature",
        "student_cdf_head_input",
        "pre_cdf_feature_student",
        "feature_pre_cdf_student",
        "cdf_head_feature_student",
    ),
    "feature_pre_cdf_teacher": (
        "teacher_pre_cdf_feature",
        "teacher_cdf_head_input",
        "pre_cdf_feature_teacher",
        "feature_pre_cdf_teacher",
        "cdf_head_feature_teacher",
    ),
}


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def load_aliases(path: str = "") -> Dict[str, Tuple[str, ...]]:
    aliases = dict(DEFAULT_ALIASES)
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            custom = json.load(handle)
        if not isinstance(custom, Mapping):
            raise TypeError("mapping_json must contain an object")
        for canonical, values in custom.items():
            if isinstance(values, str):
                values = [values]
            aliases[str(canonical)] = tuple(str(value) for value in values)
    return aliases


def flatten_payload(value: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            out.update(flatten_payload(item, child))
    else:
        out[prefix] = value
    return out


def load_payload(path: Path) -> Dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {str(key): np.asarray(data[key]) for key in data.files}
    if path.suffix in {".pt", ".pth", ".tar"}:
        loaded = torch.load(path, map_location="cpu")
        return flatten_payload(loaded)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return flatten_payload(json.load(handle))
    raise ValueError(f"Unsupported paired-cache file {path}")


def discover_payload_files(root: str) -> List[Path]:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return [path]
    files: List[Path] = []
    for suffix in ("*.npz", "*.pt", "*.pth"):
        files.extend(path.rglob(suffix))
    return sorted(set(files))


def find_value(payload: Mapping[str, Any], aliases: Sequence[str]) -> Tuple[Optional[str], Optional[Any]]:
    for alias in aliases:
        if alias in payload:
            return alias, payload[alias]
    normalized = {normalize_key(key): key for key in payload}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key is not None:
            return key, payload[key]
    # Suffix match supports nested PyTorch dictionaries flattened with '/'.
    for alias in aliases:
        normalized_alias = normalize_key(alias)
        matches = [key for key in payload if normalize_key(key).endswith(normalized_alias)]
        if len(matches) == 1:
            return matches[0], payload[matches[0]]
    return None, None


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def move_threshold_last(array: np.ndarray, thresholds: int = 6) -> np.ndarray:
    value = np.asarray(array)
    axes = [axis for axis, size in enumerate(value.shape) if int(size) == int(thresholds)]
    if not axes:
        raise ValueError(f"Cannot find threshold axis of size {thresholds} in shape {value.shape}")
    # Current CDF tensors usually have a unique T=6 axis. Prefer the final one
    # if another dimension also happens to equal six.
    axis = value.ndim - 1 if value.shape[-1] == thresholds else axes[0]
    return np.moveaxis(value, axis, -1)


def broadcast_mask(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    while value.ndim > len(target_shape) and value.shape[-1] == 1:
        value = value[..., 0]
    try:
        return np.broadcast_to(value, target_shape).astype(bool, copy=False)
    except ValueError:
        # A query mask may omit angle/depth axes. Insert singleton dimensions
        # immediately before the threshold-free target tail.
        candidates = [value]
        for _ in range(len(target_shape) - value.ndim):
            candidates = [np.expand_dims(candidate, axis=-1) for candidate in candidates]
        for candidate in candidates:
            try:
                return np.broadcast_to(candidate, target_shape).astype(bool, copy=False)
            except ValueError:
                pass
    raise ValueError(f"Cannot broadcast mask shape {value.shape} to {tuple(target_shape)}")


def friction_to_cdf_numpy(friction: np.ndarray, thresholds: Sequence[float] = CDF_THRESHOLDS) -> np.ndarray:
    f = np.asarray(friction, dtype=np.float32)
    t = np.asarray(tuple(float(x) for x in thresholds), dtype=np.float32)
    return ((f[..., None] > 0.0) & (f[..., None] <= t)).astype(np.float32)


@dataclass
class StandardPairedRows:
    student_logits: np.ndarray
    teacher_logits: np.ndarray
    gt_cdf: np.ndarray
    valid_mask: np.ndarray
    scene_id: np.ndarray
    anno_id: np.ndarray
    center_z_error: Optional[np.ndarray]
    common_valid: Optional[np.ndarray]
    support_iou: Optional[np.ndarray]
    teacher_better: Optional[np.ndarray]
    metadata: Dict[str, Any]
    features: Dict[str, np.ndarray]

    @property
    def num_rows(self) -> int:
        return int(self.student_logits.shape[0])


def _scalar_or_rows(value: Optional[Any], rows: int, default: int = -1) -> np.ndarray:
    if value is None:
        return np.full(rows, default, dtype=np.int64)
    array = as_numpy(value)
    if array.size == 1:
        return np.full(rows, int(array.reshape(-1)[0]), dtype=np.int64)
    flat = array.reshape(-1)
    if len(flat) == rows:
        return flat.astype(np.int64)
    return np.full(rows, int(flat[0]), dtype=np.int64)


def standardize_payload(
    payload: Mapping[str, Any],
    *,
    aliases: Mapping[str, Sequence[str]],
    source: str,
) -> StandardPairedRows:
    found: Dict[str, Tuple[Optional[str], Optional[Any]]] = {
        canonical: find_value(payload, names) for canonical, names in aliases.items()
    }
    if found["student_logits"][1] is None or found["teacher_logits"][1] is None:
        raise KeyError(
            f"{source}: missing student/teacher CDF logits. Available keys: {sorted(payload)[:300]}"
        )
    student = move_threshold_last(as_numpy(found["student_logits"][1])).astype(np.float32)
    teacher = move_threshold_last(as_numpy(found["teacher_logits"][1])).astype(np.float32)
    if student.shape != teacher.shape:
        raise ValueError(f"{source}: student/teacher shapes differ: {student.shape} vs {teacher.shape}")

    gt_value = found["gt_cdf"][1]
    if gt_value is not None:
        gt = move_threshold_last(as_numpy(gt_value)).astype(np.float32)
        if gt.shape != student.shape:
            try:
                gt = np.broadcast_to(gt, student.shape).astype(np.float32)
            except ValueError as exc:
                raise ValueError(f"{source}: GT CDF shape {gt.shape} cannot match logits {student.shape}") from exc
    else:
        friction_value = found["friction"][1]
        if friction_value is None:
            raise KeyError(f"{source}: neither GT CDF nor friction labels were found")
        gt = friction_to_cdf_numpy(as_numpy(friction_value))
        if gt.shape != student.shape:
            try:
                gt = np.broadcast_to(gt, student.shape).astype(np.float32)
            except ValueError as exc:
                raise ValueError(f"{source}: friction-derived CDF {gt.shape} cannot match {student.shape}") from exc

    row_shape = student.shape[:-1]
    mask_value = found["valid_mask"][1]
    valid = np.ones(row_shape, dtype=bool) if mask_value is None else broadcast_mask(as_numpy(mask_value), row_shape)

    def optional_row_array(canonical: str, dtype: Any = np.float32) -> Optional[np.ndarray]:
        value = found[canonical][1]
        if value is None:
            return None
        array = as_numpy(value)
        try:
            array = np.broadcast_to(array, row_shape)
        except ValueError:
            while array.ndim < len(row_shape):
                array = np.expand_dims(array, axis=-1)
            array = np.broadcast_to(array, row_shape)
        return array.reshape(-1).astype(dtype)

    flattened_student = student.reshape(-1, student.shape[-1])
    flattened_teacher = teacher.reshape(-1, teacher.shape[-1])
    flattened_gt = gt.reshape(-1, gt.shape[-1])
    flattened_valid = valid.reshape(-1)
    rows = len(flattened_valid)

    # Features can be Q-level or Q*A-level. Preserve them and let P0-D align by
    # leading row count; the probe script reports incompatible layers rather
    # than silently repeating an ambiguous tensor.
    features: Dict[str, np.ndarray] = {}
    for canonical, (_, value) in found.items():
        if canonical.startswith("feature_") and value is not None:
            array = as_numpy(value).astype(np.float32)
            if array.ndim >= 2:
                features[canonical] = array

    return StandardPairedRows(
        student_logits=flattened_student,
        teacher_logits=flattened_teacher,
        gt_cdf=flattened_gt,
        valid_mask=flattened_valid,
        scene_id=_scalar_or_rows(found["scene_id"][1], rows),
        anno_id=_scalar_or_rows(found["anno_id"][1], rows),
        center_z_error=optional_row_array("center_z_error"),
        common_valid=optional_row_array("common_valid", np.bool_),
        support_iou=optional_row_array("support_iou"),
        teacher_better=optional_row_array("teacher_better", np.bool_),
        metadata={
            "source": source,
            "resolved_keys": {canonical: key for canonical, (key, value) in found.items() if value is not None},
            "original_logit_shape": list(student.shape),
        },
        features=features,
    )


def concatenate_rows(items: Sequence[StandardPairedRows]) -> StandardPairedRows:
    if not items:
        raise ValueError("No standardized paired rows")
    feature_names = set.intersection(*(set(item.features) for item in items)) if items else set()

    def concat_optional(name: str) -> Optional[np.ndarray]:
        values = [getattr(item, name) for item in items]
        return None if any(value is None for value in values) else np.concatenate(values, axis=0)

    features: Dict[str, np.ndarray] = {}
    for name in sorted(feature_names):
        arrays = [item.features[name] for item in items]
        # Only concatenate layers already row-aligned within each source.
        if all(array.ndim == 2 and array.shape[0] == item.num_rows for array, item in zip(arrays, items)):
            features[name] = np.concatenate(arrays, axis=0)

    return StandardPairedRows(
        student_logits=np.concatenate([item.student_logits for item in items], axis=0),
        teacher_logits=np.concatenate([item.teacher_logits for item in items], axis=0),
        gt_cdf=np.concatenate([item.gt_cdf for item in items], axis=0),
        valid_mask=np.concatenate([item.valid_mask for item in items], axis=0),
        scene_id=np.concatenate([item.scene_id for item in items], axis=0),
        anno_id=np.concatenate([item.anno_id for item in items], axis=0),
        center_z_error=concat_optional("center_z_error"),
        common_valid=concat_optional("common_valid"),
        support_iou=concat_optional("support_iou"),
        teacher_better=concat_optional("teacher_better"),
        metadata={"sources": [item.metadata for item in items]},
        features=features,
    )


def load_standard_rows(root: str, mapping_json: str = "", max_files: int = 0) -> StandardPairedRows:
    aliases = load_aliases(mapping_json)
    files = discover_payload_files(root)
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No NPZ/PT paired diagnostic files under {root}")
    try:
        progress_every = max(1, int(os.environ.get("PKD_P0_PROGRESS_EVERY", "50")))
    except ValueError:
        progress_every = 50
    print(
        f"[P0-C1][LOAD] root={Path(root).expanduser().resolve()} files={len(files)} "
        f"max_files={int(max_files)} progress_every={progress_every}",
        flush=True,
    )
    items: List[StandardPairedRows] = []
    errors: List[str] = []
    for file_index, path in enumerate(files, start=1):
        try:
            payload = load_payload(path)
            items.append(standardize_payload(payload, aliases=aliases, source=str(path)))
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
        if file_index == 1 or file_index % progress_every == 0 or file_index == len(files):
            print(
                f"[P0-C1][LOAD] {file_index}/{len(files)} "
                f"accepted={len(items)} rejected={len(errors)} file={path.name}",
                flush=True,
            )
    if not items:
        raise RuntimeError("No compatible paired-query cache file. Errors:\n" + "\n".join(errors[:30]))
    print(
        f"[P0-C1][CONCAT] concatenating accepted_files={len(items)}; "
        "this step may temporarily use substantial RAM",
        flush=True,
    )
    combined = concatenate_rows(items)
    combined.metadata["num_files_loaded"] = len(items)
    combined.metadata["num_files_rejected"] = len(errors)
    combined.metadata["rejected_examples"] = errors[:20]
    print(
        f"[P0-C1][READY] rows={combined.num_rows} "
        f"accepted_files={len(items)} rejected_files={len(errors)}",
        flush=True,
    )
    return combined
