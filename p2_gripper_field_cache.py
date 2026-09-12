#!/usr/bin/env python3
"""Strict loader and validator for the scratch-MLP P2 representation cache."""
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
    CACHE_SCHEMA_VERSION as EXACT_ACTION_CACHE_SCHEMA_VERSION,
    FRICTION_THRESHOLDS,
    atomic_save_json,
)
from models.p2_gripper_cdf_field import (
    ACTION_POSE_DIM,
    P2_CACHE_SCHEMA_VERSION,
    P2FieldConfig,
    RAY_FEATURE_DIM,
    active_evidence_blocks,
    projected_feature_dim,
    validate_variant,
)


REQUIRED_ARRAYS: Tuple[str, ...] = (
    "schema_version",
    "source_exact_action_cache_schema_version",
    "source_base_checkpoint_sha256",
    "field_config_json",
    "field_config_sha256",
    "cdf_increment_bias",
    "cdf_head_feature",
    "base_cdf_logits",
    "action_pose_feature",
    "projected_field_feature",
    "ray_depth_feature",
    "friction",
    "collision_or_empty",
    "pure_collision",
    "empty",
    "assigned_obj",
    "center_xyz",
    "view_xyz",
    "width_raw",
    "center_id",
    "angle_id",
    "token_sel_idx",
    "scene_id",
    "anno_id",
    "dataset_idx",
    "feature_dim",
    "image_feature_dim",
    "num_depths",
    "num_thresholds",
    "action_pose_dim",
    "projected_feature_dim",
    "ray_feature_dim",
    "source_feature_max_abs",
    "source_center_max_abs",
    "source_view_max_abs",
    "source_width_max_abs",
    "base_endpoint_reconstruction_max_abs",
    "source_base_logits_max_abs",
    "field_valid_ratio",
    "field_depth_valid_ratio",
    "field_samples_per_action",
)

_ALLOWED_FRICTION = np.asarray([-1.0, *FRICTION_THRESHOLDS], dtype=np.float32)


def _scalar(array: np.ndarray, cast):
    value = np.asarray(array).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected scalar, got shape={np.asarray(array).shape}")
    return cast(value[0])


@dataclass(frozen=True)
class P2CacheFileMeta:
    path: str
    scene_id: int
    anno_id: int
    dataset_idx: int
    num_rows: int
    feature_dim: int
    image_feature_dim: int
    num_depths: int
    num_thresholds: int
    source_base_checkpoint_sha256: str
    field_config_sha256: str
    cdf_increment_bias: float


@dataclass(frozen=True)
class P2CacheInventory:
    cache_dir: str
    num_files: int
    num_rows: int
    num_actions: int
    scene_ids: Tuple[int, ...]
    feature_dim: int
    image_feature_dim: int
    num_depths: int
    num_thresholds: int
    projected_feature_dim: int
    ray_feature_dim: int
    source_base_checkpoint_sha256: str
    field_config_sha256: str
    field_config: Mapping[str, Any]
    cdf_increment_bias: float
    valid_ratio: float
    safe08_ratio: float
    collision_or_empty_ratio: float
    pure_collision_ratio: float
    empty_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "schema_version": P2_CACHE_SCHEMA_VERSION,
            "num_files": self.num_files,
            "num_rows": self.num_rows,
            "num_actions": self.num_actions,
            "scene_ids": list(self.scene_ids),
            "feature_dim": self.feature_dim,
            "image_feature_dim": self.image_feature_dim,
            "num_depths": self.num_depths,
            "num_thresholds": self.num_thresholds,
            "projected_feature_dim": self.projected_feature_dim,
            "ray_feature_dim": self.ray_feature_dim,
            "source_base_checkpoint_sha256": self.source_base_checkpoint_sha256,
            "field_config_sha256": self.field_config_sha256,
            "field_config": dict(self.field_config),
            "cdf_increment_bias": self.cdf_increment_bias,
            "valid_ratio": self.valid_ratio,
            "safe08_ratio": self.safe08_ratio,
            "collision_or_empty_ratio": self.collision_or_empty_ratio,
            "pure_collision_ratio": self.pure_collision_ratio,
            "empty_ratio": self.empty_ratio,
        }


def validate_p2_cache_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    path: str = "<memory>",
    expected_source_base_checkpoint_sha256: Optional[str] = None,
    expected_field_config_sha256: Optional[str] = None,
    check_values: bool = True,
) -> P2CacheFileMeta:
    missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
    if missing:
        raise KeyError(f"{path}: missing P2 cache arrays {missing}")

    schema = _scalar(arrays["schema_version"], str)
    if schema != P2_CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: schema={schema!r}, expected={P2_CACHE_SCHEMA_VERSION!r}. "
            "The residual-on-P1 P2 cache is intentionally incompatible."
        )
    source_schema = _scalar(arrays["source_exact_action_cache_schema_version"], str)
    if source_schema != EXACT_ACTION_CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: source exact-action schema={source_schema!r}, "
            f"expected={EXACT_ACTION_CACHE_SCHEMA_VERSION!r}"
        )
    base_sha = _scalar(arrays["source_base_checkpoint_sha256"], str)
    config_sha = _scalar(arrays["field_config_sha256"], str)
    if (
        expected_source_base_checkpoint_sha256
        and base_sha != expected_source_base_checkpoint_sha256
    ):
        raise RuntimeError(f"{path}: source Base checkpoint SHA mismatch")
    if expected_field_config_sha256 and config_sha != expected_field_config_sha256:
        raise RuntimeError(f"{path}: field config SHA mismatch")

    try:
        config_dict = json.loads(_scalar(arrays["field_config_json"], str))
        config = P2FieldConfig(**config_dict)
    except Exception as exc:
        raise ValueError(f"{path}: invalid field_config_json: {exc}") from exc
    if config.sha256() != config_sha:
        raise RuntimeError(f"{path}: field config content/hash mismatch")

    feature = np.asarray(arrays["cdf_head_feature"])
    base_logits = np.asarray(arrays["base_cdf_logits"])
    action = np.asarray(arrays["action_pose_feature"])
    projected = np.asarray(arrays["projected_field_feature"])
    ray = np.asarray(arrays["ray_depth_feature"])
    friction = np.asarray(arrays["friction"])
    if feature.ndim != 2:
        raise ValueError(f"{path}: cdf_head_feature must be [R,C], got {feature.shape}")
    rows, feature_dim = feature.shape
    if base_logits.ndim != 3 or base_logits.shape[0] != rows:
        raise ValueError(f"{path}: base_cdf_logits must be [R,D,T], got {base_logits.shape}")
    num_depths, num_thresholds = base_logits.shape[1:]
    if friction.shape != (rows, num_depths):
        raise ValueError(f"{path}: friction shape={friction.shape}")
    if action.shape != (rows, num_depths, ACTION_POSE_DIM):
        raise ValueError(f"{path}: action_pose_feature shape={action.shape}")

    image_feature_dim = _scalar(arrays["image_feature_dim"], int)
    expected_projected = projected_feature_dim(image_feature_dim)
    if projected.shape != (rows, num_depths, expected_projected):
        raise ValueError(f"{path}: projected_field_feature shape={projected.shape}")
    if ray.shape != (rows, num_depths, RAY_FEATURE_DIM):
        raise ValueError(f"{path}: ray_depth_feature shape={ray.shape}")

    declared = {
        "feature_dim": (feature_dim, _scalar(arrays["feature_dim"], int)),
        "num_depths": (num_depths, _scalar(arrays["num_depths"], int)),
        "num_thresholds": (num_thresholds, _scalar(arrays["num_thresholds"], int)),
        "action_pose_dim": (ACTION_POSE_DIM, _scalar(arrays["action_pose_dim"], int)),
        "projected_feature_dim": (
            expected_projected,
            _scalar(arrays["projected_feature_dim"], int),
        ),
        "ray_feature_dim": (RAY_FEATURE_DIM, _scalar(arrays["ray_feature_dim"], int)),
    }
    for name, (actual, stored) in declared.items():
        if actual != stored:
            raise ValueError(f"{path}: {name} actual={actual}, declared={stored}")

    for key in ("collision_or_empty", "pure_collision", "empty", "assigned_obj"):
        if np.asarray(arrays[key]).shape != (rows, num_depths):
            raise ValueError(f"{path}: {key} has incompatible shape")
    for key in ("center_id", "angle_id", "token_sel_idx"):
        if np.asarray(arrays[key]).shape != (rows,):
            raise ValueError(f"{path}: {key} must be [R]")
    if np.asarray(arrays["center_xyz"]).shape != (rows, 3):
        raise ValueError(f"{path}: center_xyz must be [R,3]")
    if np.asarray(arrays["view_xyz"]).shape != (rows, 3):
        raise ValueError(f"{path}: view_xyz must be [R,3]")
    if np.asarray(arrays["width_raw"]).shape != (rows, num_depths):
        raise ValueError(f"{path}: width_raw must be [R,D]")

    if check_values:
        for key in (
            "cdf_head_feature",
            "base_cdf_logits",
            "action_pose_feature",
            "projected_field_feature",
            "ray_depth_feature",
            "friction",
            "center_xyz",
            "view_xyz",
            "width_raw",
        ):
            if not np.isfinite(np.asarray(arrays[key])).all():
                raise ValueError(f"{path}: {key} contains NaN/Inf")
        view_norm = np.linalg.norm(np.asarray(arrays["view_xyz"], np.float32), axis=-1)
        if np.any(np.abs(view_norm - 1.0) > 5e-3):
            raise ValueError(f"{path}: view_xyz is not unit-normalized")
        friction32 = friction.astype(np.float32)
        distance = np.abs(friction32[..., None] - _ALLOWED_FRICTION)
        if not (distance.min(axis=-1) < 5e-3).all():
            bad = friction32[distance.min(axis=-1) >= 5e-3]
            raise ValueError(f"{path}: unexpected friction values {np.unique(bad)[:12]}")
        collision = np.asarray(arrays["collision_or_empty"]).astype(bool)
        pure = np.asarray(arrays["pure_collision"]).astype(bool)
        empty = np.asarray(arrays["empty"]).astype(bool)
        if np.any(pure & empty):
            raise ValueError(f"{path}: pure_collision and empty overlap")
        if np.any(pure & ~collision) or np.any(empty & ~collision):
            raise ValueError(f"{path}: pure_collision/empty must be subsets of collision")
        if np.any((friction32 > 0.0) & collision):
            raise ValueError(f"{path}: colliding/empty action has positive friction")

        strict_limits = {
            "source_feature_max_abs": 5e-4,
            "source_center_max_abs": 5e-5,
            "source_view_max_abs": 5e-5,
            "source_width_max_abs": 5e-5,
            "base_endpoint_reconstruction_max_abs": 2e-5,
            "source_base_logits_max_abs": 5e-4,
        }
        for key, limit in strict_limits.items():
            value = _scalar(arrays[key], float)
            if not np.isfinite(value) or value > limit:
                raise ValueError(f"{path}: {key}={value:.3e}, limit={limit:.3e}")

    return P2CacheFileMeta(
        path=os.path.abspath(path),
        scene_id=_scalar(arrays["scene_id"], int),
        anno_id=_scalar(arrays["anno_id"], int),
        dataset_idx=_scalar(arrays["dataset_idx"], int),
        num_rows=int(rows),
        feature_dim=int(feature_dim),
        image_feature_dim=int(image_feature_dim),
        num_depths=int(num_depths),
        num_thresholds=int(num_thresholds),
        source_base_checkpoint_sha256=base_sha,
        field_config_sha256=config_sha,
        cdf_increment_bias=_scalar(arrays["cdf_increment_bias"], float),
    )


def scan_p2_cache(
    cache_dir: str,
    *,
    expected_source_base_checkpoint_sha256: Optional[str] = None,
    expected_field_config_sha256: Optional[str] = None,
    max_files: int = 0,
    strict: bool = True,
    check_values: bool = True,
) -> Tuple[List[P2CacheFileMeta], P2CacheInventory, List[Tuple[str, str]]]:
    cache_dir = os.path.abspath(cache_dir)
    files = sorted(glob.glob(os.path.join(cache_dir, "scene_*", "ann_*.npz")))
    if max_files > 0:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No P2 scratch cache files under {cache_dir}")

    metadata: List[P2CacheFileMeta] = []
    failures: List[Tuple[str, str]] = []
    total_actions = valid = safe08 = collision = pure_collision = empty = 0
    field_config_dict: Optional[Mapping[str, Any]] = None

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            meta = validate_p2_cache_arrays(
                arrays,
                path=path,
                expected_source_base_checkpoint_sha256=(
                    expected_source_base_checkpoint_sha256
                ),
                expected_field_config_sha256=expected_field_config_sha256,
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
            current_config = json.loads(_scalar(arrays["field_config_json"], str))
            if field_config_dict is None:
                field_config_dict = current_config
            elif current_config != field_config_dict:
                raise RuntimeError("P2 cache mixes field configurations")
        except Exception as exc:
            failures.append((path, repr(exc)))
            if strict:
                raise

    if not metadata:
        raise RuntimeError("No valid P2 scratch cache remains after validation")

    fields: Dict[str, Set[Any]] = {
        "feature_dim": {m.feature_dim for m in metadata},
        "image_feature_dim": {m.image_feature_dim for m in metadata},
        "num_depths": {m.num_depths for m in metadata},
        "num_thresholds": {m.num_thresholds for m in metadata},
        "source_base_checkpoint_sha256": {
            m.source_base_checkpoint_sha256 for m in metadata
        },
        "field_config_sha256": {m.field_config_sha256 for m in metadata},
        "cdf_increment_bias": {round(m.cdf_increment_bias, 8) for m in metadata},
    }
    for name, values in fields.items():
        if len(values) != 1:
            raise RuntimeError(f"P2 cache mixes incompatible {name}: {sorted(values)}")

    assert field_config_dict is not None
    image_dim = next(iter(fields["image_feature_dim"]))
    inventory = P2CacheInventory(
        cache_dir=cache_dir,
        num_files=len(metadata),
        num_rows=sum(m.num_rows for m in metadata),
        num_actions=total_actions,
        scene_ids=tuple(sorted({m.scene_id for m in metadata})),
        feature_dim=next(iter(fields["feature_dim"])),
        image_feature_dim=image_dim,
        num_depths=next(iter(fields["num_depths"])),
        num_thresholds=next(iter(fields["num_thresholds"])),
        projected_feature_dim=projected_feature_dim(image_dim),
        ray_feature_dim=RAY_FEATURE_DIM,
        source_base_checkpoint_sha256=next(
            iter(fields["source_base_checkpoint_sha256"])
        ),
        field_config_sha256=next(iter(fields["field_config_sha256"])),
        field_config=field_config_dict,
        cdf_increment_bias=next(iter(fields["cdf_increment_bias"])),
        valid_ratio=valid / max(total_actions, 1),
        safe08_ratio=safe08 / max(total_actions, 1),
        collision_or_empty_ratio=collision / max(total_actions, 1),
        pure_collision_ratio=pure_collision / max(total_actions, 1),
        empty_ratio=empty / max(total_actions, 1),
    )
    return metadata, inventory, failures


class P2GripperFieldCacheDataset(Dataset):
    """One item is one cached frame; actions remain grouped as [R,D]."""

    def __init__(
        self,
        cache_dir: str,
        *,
        split: str,
        variant: str,
        val_scene_start: int = 90,
        expected_source_base_checkpoint_sha256: Optional[str] = None,
        expected_field_config_sha256: Optional[str] = None,
        metadata_override: Optional[Sequence[P2CacheFileMeta]] = None,
        inventory_override: Optional[P2CacheInventory] = None,
    ) -> None:
        variant = validate_variant(variant)
        if metadata_override is None or inventory_override is None:
            metadata, inventory, _ = scan_p2_cache(
                cache_dir,
                expected_source_base_checkpoint_sha256=(
                    expected_source_base_checkpoint_sha256
                ),
                expected_field_config_sha256=expected_field_config_sha256,
                strict=True,
                check_values=True,
            )
        else:
            metadata = list(metadata_override)
            inventory = inventory_override
        split = str(split).lower()
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train/val/all, got {split!r}")
        if split == "train":
            selected = [m for m in metadata if m.scene_id < int(val_scene_start)]
        elif split == "val":
            selected = [m for m in metadata if m.scene_id >= int(val_scene_start)]
        else:
            selected = list(metadata)
        if not selected:
            raise RuntimeError(
                f"No P2 cache files for split={split}, val_scene_start={val_scene_start}"
            )
        self.cache_dir = os.path.abspath(cache_dir)
        self.variant = variant
        self.active_blocks = set(active_evidence_blocks(variant))
        self.metadata = selected
        self.inventory = inventory

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[int(index)]
        with np.load(meta.path, allow_pickle=False) as data:
            result: Dict[str, torch.Tensor] = {
                "base_feature": torch.as_tensor(
                    np.asarray(data["cdf_head_feature"]), dtype=torch.float32
                ),
                "base_cdf_logits": torch.as_tensor(
                    np.asarray(data["base_cdf_logits"]), dtype=torch.float32
                ),
                "friction": torch.as_tensor(
                    np.asarray(data["friction"]), dtype=torch.float32
                ),
                "center_id": torch.as_tensor(
                    np.asarray(data["center_id"]), dtype=torch.long
                ),
                "angle_id": torch.as_tensor(
                    np.asarray(data["angle_id"]), dtype=torch.long
                ),
                "scene_id": torch.tensor(meta.scene_id, dtype=torch.long),
                "anno_id": torch.tensor(meta.anno_id, dtype=torch.long),
            }
            if "pose" in self.active_blocks:
                result["action_pose_feature"] = torch.as_tensor(
                    np.asarray(data["action_pose_feature"]), dtype=torch.float32
                )
            if "projected" in self.active_blocks:
                result["projected_field_feature"] = torch.as_tensor(
                    np.asarray(data["projected_field_feature"]), dtype=torch.float32
                )
            if "ray_depth" in self.active_blocks:
                result["ray_depth_feature"] = torch.as_tensor(
                    np.asarray(data["ray_depth_feature"]), dtype=torch.float32
                )
        return result


def collate_p2_gripper_field(
    samples: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty P2 batch")
    keys = set(samples[0].keys())
    for sample in samples[1:]:
        if set(sample.keys()) != keys:
            raise RuntimeError("P2 samples in one batch have different feature blocks")
    output: Dict[str, torch.Tensor] = {}
    row_keys = keys - {"scene_id", "anno_id"}
    for key in sorted(row_keys):
        output[key] = torch.cat([sample[key] for sample in samples], dim=0)

    group_ids = []
    frame_ids = []
    for local_index, sample in enumerate(samples):
        rows = int(sample["base_feature"].shape[0])
        scene_id = int(sample["scene_id"])
        anno_id = int(sample["anno_id"])
        group_ids.append(
            sample["center_id"].long()
            + anno_id * 10_000
            + scene_id * 10_000_000
        )
        frame_ids.append(torch.full((rows,), local_index, dtype=torch.long))
    output["center_group"] = torch.cat(group_ids, dim=0)
    output["frame_group"] = torch.cat(frame_ids, dim=0)
    output["scene_id"] = torch.stack([sample["scene_id"] for sample in samples])
    output["anno_id"] = torch.stack([sample["anno_id"] for sample in samples])
    return output


def save_inventory(inventory: P2CacheInventory, path: str) -> None:
    atomic_save_json(inventory.to_dict(), path)
