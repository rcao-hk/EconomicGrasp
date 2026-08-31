#!/usr/bin/env python3
"""Strict loader and validator for current CVA-CDF exact-action caches."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from exact_action_cdf_common import (
    CACHE_SCHEMA_VERSION,
    FRICTION_THRESHOLDS,
    atomic_save_json,
)


REQUIRED_ARRAYS: Tuple[str, ...] = (
    "schema_version",
    "checkpoint_sha256",
    "cdf_increment_bias",
    "cdf_head_feature",
    "base_cdf_logits",
    "width_raw",
    "friction",
    "collision_or_empty",
    "pure_collision",
    "empty",
    "assigned_obj",
    "center_xyz",
    "view_xyz",
    "center_id",
    "angle_id",
    "view_id",
    "token_sel_idx",
    "scene_id",
    "anno_id",
    "dataset_idx",
    "num_angles",
    "num_depths",
    "num_thresholds",
    "feature_dim",
    "base_reconstruction_max_abs",
    "compact_replay_max_abs",
    "compact_replay_mean_abs",
)

_ALLOWED_FRICTION = np.asarray(
    [-1.0, *FRICTION_THRESHOLDS], dtype=np.float32
)


def _scalar_string(array: np.ndarray) -> str:
    value = np.asarray(array).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected one string value, got shape={np.asarray(array).shape}")
    return str(value[0])


def _scalar_int(array: np.ndarray) -> int:
    value = np.asarray(array).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected one integer value, got shape={np.asarray(array).shape}")
    return int(value[0])


def _scalar_float(array: np.ndarray) -> float:
    value = np.asarray(array).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected one float value, got shape={np.asarray(array).shape}")
    return float(value[0])


@dataclass(frozen=True)
class CacheFileMeta:
    path: str
    scene_id: int
    anno_id: int
    num_rows: int
    feature_dim: int
    num_angles: int
    num_depths: int
    num_thresholds: int
    checkpoint_sha256: str
    cdf_increment_bias: float


@dataclass(frozen=True)
class CacheInventory:
    cache_dir: str
    num_files: int
    num_rows: int
    num_actions: int
    scene_ids: Tuple[int, ...]
    checkpoint_sha256: str
    feature_dim: int
    num_angles: int
    num_depths: int
    num_thresholds: int
    cdf_increment_bias: float
    valid_ratio: float
    safe08_ratio: float
    collision_or_empty_ratio: float
    pure_collision_ratio: float
    empty_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "schema_version": CACHE_SCHEMA_VERSION,
            "num_files": self.num_files,
            "num_rows": self.num_rows,
            "num_actions": self.num_actions,
            "scene_ids": list(self.scene_ids),
            "checkpoint_sha256": self.checkpoint_sha256,
            "feature_dim": self.feature_dim,
            "num_angles": self.num_angles,
            "num_depths": self.num_depths,
            "num_thresholds": self.num_thresholds,
            "cdf_increment_bias": self.cdf_increment_bias,
            "valid_ratio": self.valid_ratio,
            "safe08_ratio": self.safe08_ratio,
            "collision_or_empty_ratio": self.collision_or_empty_ratio,
            "pure_collision_ratio": self.pure_collision_ratio,
            "empty_ratio": self.empty_ratio,
        }


def validate_cache_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    path: str = "<memory>",
    expected_checkpoint_sha256: Optional[str] = None,
    check_values: bool = True,
) -> CacheFileMeta:
    missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
    if missing:
        raise KeyError(f"{path}: missing current-cache arrays {missing}")

    schema = _scalar_string(arrays["schema_version"])
    if schema != CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: schema_version={schema!r}; expected {CACHE_SCHEMA_VERSION!r}. "
            "The old cva_joint_utility_v1 cache is intentionally unsupported."
        )
    checkpoint_sha = _scalar_string(arrays["checkpoint_sha256"])
    if expected_checkpoint_sha256 is not None and checkpoint_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"{path}: cache checkpoint SHA256={checkpoint_sha}, expected "
            f"{expected_checkpoint_sha256}."
        )

    feature = np.asarray(arrays["cdf_head_feature"])
    base_logits = np.asarray(arrays["base_cdf_logits"])
    width = np.asarray(arrays["width_raw"])
    friction = np.asarray(arrays["friction"])
    collision = np.asarray(arrays["collision_or_empty"])
    pure_collision = np.asarray(arrays["pure_collision"])
    empty = np.asarray(arrays["empty"])
    assigned = np.asarray(arrays["assigned_obj"])

    if feature.ndim != 2:
        raise ValueError(f"{path}: cdf_head_feature must be [R,C], got {feature.shape}")
    rows, feature_dim = feature.shape
    if rows <= 0 or feature_dim <= 0:
        raise ValueError(f"{path}: empty feature matrix {feature.shape}")
    if base_logits.ndim != 3 or base_logits.shape[0] != rows:
        raise ValueError(
            f"{path}: base_cdf_logits must be [R,D,T], got {base_logits.shape}"
        )
    num_depths, num_thresholds = base_logits.shape[1:]
    num_angles = _scalar_int(arrays["num_angles"])
    declared_depths = _scalar_int(arrays["num_depths"])
    declared_thresholds = _scalar_int(arrays["num_thresholds"])
    declared_feature_dim = _scalar_int(arrays["feature_dim"])
    if num_angles != 12:
        raise ValueError(f"{path}: current CVA-CDF cache requires A=12, got {num_angles}")
    if declared_depths != num_depths:
        raise ValueError(
            f"{path}: declared num_depths={declared_depths}, tensor has {num_depths}."
        )
    if declared_thresholds != num_thresholds:
        raise ValueError(
            f"{path}: declared num_thresholds={declared_thresholds}, "
            f"tensor has {num_thresholds}."
        )
    if declared_feature_dim != feature_dim:
        raise ValueError(
            f"{path}: declared feature_dim={declared_feature_dim}, tensor has "
            f"{feature_dim}."
        )
    expected_rd = (rows, num_depths)
    for key, array in {
        "width_raw": width,
        "friction": friction,
        "collision_or_empty": collision,
        "pure_collision": pure_collision,
        "empty": empty,
        "assigned_obj": assigned,
    }.items():
        if array.shape != expected_rd:
            raise ValueError(
                f"{path}: {key} shape={array.shape}; expected {expected_rd}."
            )

    for key in (
        "center_id",
        "angle_id",
        "view_id",
        "token_sel_idx",
    ):
        value = np.asarray(arrays[key])
        if value.shape != (rows,):
            raise ValueError(f"{path}: {key} must be [R], got {value.shape}")
    for key in ("center_xyz", "view_xyz"):
        value = np.asarray(arrays[key])
        if value.shape != (rows, 3):
            raise ValueError(f"{path}: {key} must be [R,3], got {value.shape}")

    if check_values:
        for key in (
            "cdf_head_feature",
            "base_cdf_logits",
            "width_raw",
            "center_xyz",
            "view_xyz",
        ):
            value = np.asarray(arrays[key])
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: {key} contains NaN/Inf")

        # Strict ordering/capture check based on the actual deployed Conv1d
        # output. It should remain near machine precision.
        reconstruction_error = _scalar_float(arrays["base_reconstruction_max_abs"])
        if not np.isfinite(reconstruction_error) or reconstruction_error > 2e-5:
            raise ValueError(
                f"{path}: invalid deployed-head reconstruction error "
                f"{reconstruction_error:.3e}."
            )

        # Compact row-wise replay is only a TF32 numerical diagnostic.
        compact_max = _scalar_float(arrays["compact_replay_max_abs"])
        compact_mean = _scalar_float(arrays["compact_replay_mean_abs"])
        if (
            not np.isfinite(compact_max)
            or not np.isfinite(compact_mean)
            or compact_max < 0.0
            or compact_mean < 0.0
            or compact_max > 5e-2
            or compact_mean > compact_max + 1e-12
        ):
            raise ValueError(
                f"{path}: invalid compact replay diagnostics: "
                f"max={compact_max:.3e}, mean={compact_mean:.3e}."
            )

        friction32 = friction.astype(np.float32)
        distance = np.abs(friction32[..., None] - _ALLOWED_FRICTION)
        if not (distance.min(axis=-1) < 5e-3).all():
            bad = friction32[distance.min(axis=-1) >= 5e-3]
            raise ValueError(
                f"{path}: unexpected friction values {np.unique(bad)[:12]}"
            )
        collision_b = collision.astype(bool)
        pure_b = pure_collision.astype(bool)
        empty_b = empty.astype(bool)
        if np.any(pure_b & empty_b):
            raise ValueError(f"{path}: pure_collision and empty overlap")
        if np.any(pure_b & (~collision_b)) or np.any(empty_b & (~collision_b)):
            raise ValueError(
                f"{path}: pure_collision/empty must be subsets of collision_or_empty"
            )
        if np.any((friction32 > 0.0) & collision_b):
            raise ValueError(f"{path}: colliding/empty action has positive friction")
        angle = np.asarray(arrays["angle_id"], dtype=np.int64)
        if np.any(angle < 0) or np.any(angle >= num_angles):
            raise ValueError(
                f"{path}: angle_id outside [0,{num_angles - 1}]"
            )
        view_norm = np.linalg.norm(np.asarray(arrays["view_xyz"], np.float32), axis=-1)
        if np.any(np.abs(view_norm - 1.0) > 5e-3):
            raise ValueError(f"{path}: view_xyz is not unit-normalized")

    return CacheFileMeta(
        path=os.path.abspath(path),
        scene_id=_scalar_int(arrays["scene_id"]),
        anno_id=_scalar_int(arrays["anno_id"]),
        num_rows=int(rows),
        feature_dim=int(feature_dim),
        num_angles=int(num_angles),
        num_depths=int(num_depths),
        num_thresholds=int(num_thresholds),
        checkpoint_sha256=checkpoint_sha,
        cdf_increment_bias=_scalar_float(arrays["cdf_increment_bias"]),
    )


def scan_cache(
    cache_dir: str,
    *,
    expected_checkpoint_sha256: Optional[str] = None,
    max_files: int = 0,
    strict: bool = True,
    check_values: bool = True,
) -> Tuple[List[CacheFileMeta], CacheInventory, List[Tuple[str, str]]]:
    cache_dir = os.path.abspath(cache_dir)
    files = sorted(glob.glob(os.path.join(cache_dir, "scene_*", "ann_*.npz")))
    if max_files > 0:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No current exact-action cache files under {cache_dir}")

    metadata: List[CacheFileMeta] = []
    failures: List[Tuple[str, str]] = []
    total_actions = 0
    valid = 0
    safe08 = 0
    collision = 0
    pure_collision = 0
    empty = 0

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            meta = validate_cache_arrays(
                arrays,
                path=path,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                check_values=check_values,
            )
            metadata.append(meta)
            friction = arrays["friction"].astype(np.float32)
            total_actions += int(friction.size)
            valid += int((friction > 0.0).sum())
            safe08 += int(((friction > 0.0) & (friction <= 0.8)).sum())
            collision += int(arrays["collision_or_empty"].astype(bool).sum())
            pure_collision += int(arrays["pure_collision"].astype(bool).sum())
            empty += int(arrays["empty"].astype(bool).sum())
        except Exception as exc:
            failures.append((path, repr(exc)))
            if strict:
                raise

    if not metadata:
        raise RuntimeError("No valid current exact-action cache file remains after validation.")

    checkpoints: Set[str] = {item.checkpoint_sha256 for item in metadata}
    feature_dims = {item.feature_dim for item in metadata}
    angle_dims = {item.num_angles for item in metadata}
    depth_dims = {item.num_depths for item in metadata}
    threshold_dims = {item.num_thresholds for item in metadata}
    increment_biases = {round(item.cdf_increment_bias, 8) for item in metadata}
    for name, values in {
        "checkpoint_sha256": checkpoints,
        "feature_dim": feature_dims,
        "num_angles": angle_dims,
        "num_depths": depth_dims,
        "num_thresholds": threshold_dims,
        "cdf_increment_bias": increment_biases,
    }.items():
        if len(values) != 1:
            raise RuntimeError(f"Cache mixes incompatible {name} values: {sorted(values)}")

    num_rows = sum(item.num_rows for item in metadata)
    inventory = CacheInventory(
        cache_dir=cache_dir,
        num_files=len(metadata),
        num_rows=num_rows,
        num_actions=total_actions,
        scene_ids=tuple(sorted({item.scene_id for item in metadata})),
        checkpoint_sha256=next(iter(checkpoints)),
        feature_dim=next(iter(feature_dims)),
        num_angles=next(iter(angle_dims)),
        num_depths=next(iter(depth_dims)),
        num_thresholds=next(iter(threshold_dims)),
        cdf_increment_bias=next(iter(increment_biases)),
        valid_ratio=valid / max(total_actions, 1),
        safe08_ratio=safe08 / max(total_actions, 1),
        collision_or_empty_ratio=collision / max(total_actions, 1),
        pure_collision_ratio=pure_collision / max(total_actions, 1),
        empty_ratio=empty / max(total_actions, 1),
    )
    return metadata, inventory, failures


class ExactActionCdfCacheDataset(Dataset):
    """One item is one cached frame; all selected center-angle rows are retained."""

    def __init__(
        self,
        cache_dir: str,
        *,
        split: str,
        val_scene_start: int = 100,
        expected_checkpoint_sha256: Optional[str] = None,
    ) -> None:
        metadata, inventory, _ = scan_cache(
            cache_dir,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            strict=True,
            check_values=True,
        )
        split = str(split).lower()
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train/val/all, got {split!r}")
        if split == "train":
            selected = [item for item in metadata if item.scene_id < int(val_scene_start)]
        elif split == "val":
            selected = [item for item in metadata if item.scene_id >= int(val_scene_start)]
        else:
            selected = metadata
        if not selected:
            raise RuntimeError(
                f"No cache files for split={split}, val_scene_start={val_scene_start}."
            )
        self.cache_dir = os.path.abspath(cache_dir)
        self.split = split
        self.val_scene_start = int(val_scene_start)
        self.metadata = selected
        self.inventory_all = inventory
        self.feature_dim = selected[0].feature_dim
        self.num_angles = selected[0].num_angles
        self.num_depths = selected[0].num_depths
        self.num_thresholds = selected[0].num_thresholds
        self.cdf_increment_bias = selected[0].cdf_increment_bias
        self.checkpoint_sha256 = selected[0].checkpoint_sha256

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[index]
        with np.load(meta.path, allow_pickle=False) as data:
            feature = torch.as_tensor(
                np.asarray(data["cdf_head_feature"]), dtype=torch.float32
            )
            base_logits = torch.as_tensor(
                np.asarray(data["base_cdf_logits"]), dtype=torch.float32
            )
            friction = torch.as_tensor(
                np.asarray(data["friction"]), dtype=torch.float32
            )
            center_id = torch.as_tensor(
                np.asarray(data["center_id"]), dtype=torch.long
            )
            angle_id = torch.as_tensor(
                np.asarray(data["angle_id"]), dtype=torch.long
            )
        return {
            "feature": feature,
            "base_cdf_logits": base_logits,
            "friction": friction,
            "center_id": center_id,
            "angle_id": angle_id,
            "scene_id": torch.tensor(meta.scene_id, dtype=torch.long),
            "anno_id": torch.tensor(meta.anno_id, dtype=torch.long),
        }


def collate_exact_action_cdf(samples: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty exact-action batch")
    output: Dict[str, torch.Tensor] = {}
    for key in ("feature", "base_cdf_logits", "friction", "center_id", "angle_id"):
        output[key] = torch.cat([sample[key] for sample in samples], dim=0)

    group_ids = []
    frame_ids = []
    for local_index, sample in enumerate(samples):
        rows = int(sample["feature"].shape[0])
        scene_id = int(sample["scene_id"])
        anno_id = int(sample["anno_id"])
        # Center ids are local to a frame; this composite remains exact in int64.
        group_ids.append(
            sample["center_id"].long()
            + int(anno_id) * 10_000
            + int(scene_id) * 10_000_000
        )
        frame_ids.append(torch.full((rows,), local_index, dtype=torch.long))
    output["center_group"] = torch.cat(group_ids, dim=0)
    output["frame_group"] = torch.cat(frame_ids, dim=0)
    output["scene_id"] = torch.stack([sample["scene_id"] for sample in samples])
    output["anno_id"] = torch.stack([sample["anno_id"] for sample in samples])
    return output


def save_inventory(inventory: CacheInventory, path: str) -> None:
    atomic_save_json(inventory.to_dict(), path)
