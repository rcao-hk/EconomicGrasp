"""Dataset for exact-evaluator CVA candidate caches."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_ARRAYS = (
    "angle_feature",
    "score_logits",
    "depth_logits",
    "width_raw",
    "angle_id",
    "depth_id",
    "view_rank",
    "view_score",
    "center_xyz",
    "view_xyz",
    "friction",
    "pure_collision",
    "empty",
    "center_id",
    "legacy_score",
    "legacy_quality",
    "assigned_obj",
)


@dataclass
class CacheSamplingConfig:
    max_candidates_per_sample: int = 2048
    keep_hard_negatives: int = 512
    keep_positives: int = 512
    keep_per_center: int = 1
    seed: int = 0


class CVAJointUtilityCacheDataset(Dataset):
    """One item corresponds to all retained candidates from one RGB-D frame."""

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        val_scene_start: int = 90,
        scene_ids: Optional[Sequence[int]] = None,
        sampling: Optional[CacheSamplingConfig] = None,
        strict: bool = True,
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.split = str(split)
        self.val_scene_start = int(val_scene_start)
        self.sampling = sampling or CacheSamplingConfig()
        self.strict = bool(strict)
        allowed = None if scene_ids is None else {int(x) for x in scene_ids}

        files = sorted(glob.glob(os.path.join(self.cache_dir, "**", "*.npz"), recursive=True))
        selected: List[str] = []
        metadata: List[Dict[str, int]] = []
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as data:
                    scene_id = int(np.asarray(data["scene_id"]).reshape(-1)[0])
                    anno_id = int(np.asarray(data["anno_id"]).reshape(-1)[0])
            except Exception:
                if self.strict:
                    raise
                continue
            if allowed is not None and scene_id not in allowed:
                continue
            if self.split == "train" and scene_id >= self.val_scene_start:
                continue
            if self.split in {"val", "valid", "validation"} and scene_id < self.val_scene_start:
                continue
            if self.split not in {"train", "val", "valid", "validation", "all"}:
                raise ValueError(f"Unsupported split={self.split}")
            selected.append(path)
            metadata.append({"scene_id": scene_id, "anno_id": anno_id})
        if not selected:
            raise RuntimeError(
                f"No cache files selected under {self.cache_dir!r} for split={self.split}; "
                f"val_scene_start={self.val_scene_start}"
            )
        self.files = selected
        self.metadata = metadata

        # Infer schema once and fail early if the cache is incompatible.
        with np.load(self.files[0], allow_pickle=False) as first:
            missing = [k for k in REQUIRED_ARRAYS if k not in first]
            if missing:
                raise KeyError(f"Cache {self.files[0]} is missing arrays: {missing}")
            self.schema = {
                "angle_feature_dim": int(first["angle_feature"].shape[-1]),
                "score_logit_dim": int(first["score_logits"].shape[-1]),
                "depth_logit_dim": int(first["depth_logits"].shape[-1]),
                "num_angles": int(np.asarray(first.get("num_angles", 12)).reshape(-1)[0]),
                "num_depths": int(np.asarray(first.get("num_depths", 4)).reshape(-1)[0]),
                "max_view_rank": int(np.asarray(first.get("top_views", 4)).reshape(-1)[0]),
            }

    def __len__(self) -> int:
        return len(self.files)

    def _subsample_indices(self, data: Mapping[str, np.ndarray], index: int) -> np.ndarray:
        n = int(data["friction"].shape[0])
        limit = int(self.sampling.max_candidates_per_sample)
        if limit <= 0 or n <= limit:
            return np.arange(n, dtype=np.int64)

        rng = np.random.default_rng(int(self.sampling.seed) + int(index) * 1000003)
        friction = np.asarray(data["friction"], dtype=np.float32)
        utility = ((friction[:, None] > 0.0) & (
            friction[:, None] <= np.asarray([0.2,0.4,0.6,0.8,1.0,1.2], dtype=np.float32)[None]
        )).mean(axis=1)
        legacy = np.asarray(data["legacy_score"], dtype=np.float32)
        center = np.asarray(data["center_id"], dtype=np.int64)

        chosen: List[int] = []
        chosen_set = set()

        def add(ids: Iterable[int]) -> None:
            for value in ids:
                i = int(value)
                if i not in chosen_set and len(chosen) < limit:
                    chosen.append(i)
                    chosen_set.add(i)

        # Preserve at least a small amount of center coverage before focusing
        # on hard candidates. This is important for within-center ranking loss.
        per_center = max(0, int(self.sampling.keep_per_center))
        if per_center:
            for cid in np.unique(center):
                ids = np.flatnonzero(center == cid)
                order = ids[np.argsort(-legacy[ids], kind="stable")]
                add(order[:per_center])
                if len(chosen) >= limit:
                    break

        invalid = np.flatnonzero(utility <= 0.0)
        hard_neg = invalid[np.argsort(-legacy[invalid], kind="stable")]
        add(hard_neg[: max(0, int(self.sampling.keep_hard_negatives))])

        positive = np.flatnonzero(utility > 0.0)
        # Highest evaluator utility first, then model confidence to retain both
        # strong targets and difficult positive examples.
        pos_order = positive[np.lexsort((-legacy[positive], -utility[positive]))]
        add(pos_order[: max(0, int(self.sampling.keep_positives))])

        remaining = np.asarray([i for i in range(n) if i not in chosen_set], dtype=np.int64)
        if len(chosen) < limit and remaining.size:
            take = min(limit - len(chosen), remaining.size)
            add(rng.choice(remaining, size=take, replace=False))
        return np.asarray(chosen, dtype=np.int64)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        ids = self._subsample_indices(arrays, index)

        def take(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(arrays[name][ids], dtype=dtype)

        out: Dict[str, torch.Tensor] = {
            "angle_feature": take("angle_feature", torch.float32),
            "score_logits": take("score_logits", torch.float32),
            "depth_logits": take("depth_logits", torch.float32),
            "width_raw": take("width_raw", torch.float32),
            "legacy_collision_logit": take("legacy_collision_logit", torch.float32)
                if "legacy_collision_logit" in arrays else torch.zeros(len(ids), dtype=torch.float32),
            "angle_id": take("angle_id", torch.long),
            "depth_id": take("depth_id", torch.long),
            "view_rank": take("view_rank", torch.long),
            "view_score": take("view_score", torch.float32),
            "center_xyz": take("center_xyz", torch.float32),
            "view_xyz": take("view_xyz", torch.float32),
            "friction": take("friction", torch.float32),
            "pure_collision": take("pure_collision", torch.float32),
            "empty": take("empty", torch.float32),
            "center_id": take("center_id", torch.long),
            "legacy_score": take("legacy_score", torch.float32),
            "legacy_quality": take("legacy_quality", torch.float32),
            "assigned_obj": take("assigned_obj", torch.long),
            "scene_id": torch.tensor(int(arrays["scene_id"].reshape(-1)[0]), dtype=torch.long),
            "anno_id": torch.tensor(int(arrays["anno_id"].reshape(-1)[0]), dtype=torch.long),
            "dataset_index": torch.tensor(index, dtype=torch.long),
        }
        return out


def collate_joint_utility_cache(samples: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    candidate_keys = [
        "angle_feature", "score_logits", "depth_logits", "width_raw",
        "legacy_collision_logit", "angle_id", "depth_id", "view_rank",
        "view_score", "center_xyz", "view_xyz", "friction",
        "pure_collision", "empty", "center_id", "legacy_score", "legacy_quality",
        "assigned_obj",
    ]
    out = {key: torch.cat([s[key] for s in samples], dim=0) for key in candidate_keys}

    sample_groups = []
    center_groups = []
    object_groups = []
    scene_per_candidate = []
    anno_per_candidate = []
    for local_sample, sample in enumerate(samples):
        n = int(sample["friction"].numel())
        sample_groups.append(torch.full((n,), local_sample, dtype=torch.long))
        # center ids are local to a frame; combine with local sample id.
        center_groups.append(sample["center_id"].long() + local_sample * 100000)
        object_groups.append(sample["assigned_obj"].long() + local_sample * 100000)
        scene_per_candidate.append(torch.full((n,), int(sample["scene_id"]), dtype=torch.long))
        anno_per_candidate.append(torch.full((n,), int(sample["anno_id"]), dtype=torch.long))
    out["sample_group"] = torch.cat(sample_groups, dim=0)
    out["center_group"] = torch.cat(center_groups, dim=0)
    out["object_group"] = torch.cat(object_groups, dim=0)
    out["scene_id_per_candidate"] = torch.cat(scene_per_candidate, dim=0)
    out["anno_id_per_candidate"] = torch.cat(anno_per_candidate, dim=0)
    out["scene_id"] = torch.stack([s["scene_id"] for s in samples])
    out["anno_id"] = torch.stack([s["anno_id"] for s in samples])
    out["num_samples"] = torch.tensor(len(samples), dtype=torch.long)
    return out


def save_cache_inventory(dataset: CVAJointUtilityCacheDataset, path: str) -> None:
    payload = {
        "cache_dir": dataset.cache_dir,
        "split": dataset.split,
        "num_files": len(dataset),
        "schema": dataset.schema,
        "scene_ids": sorted({x["scene_id"] for x in dataset.metadata}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
