#!/usr/bin/env python3
"""Train RGB-only EconomicGrasp CVA-CDF on a deterministic GraspNet subset.

This is the equal-exposure control for the GN-Trans mixed-data experiment.
The model, losses, optimizer, A1 view protocol, and validation logic are inherited
from ``train_cva_ddp.py``. Only the train/eval datasets are replaced by
per-scene deterministic subsets.

For ``--train_fraction 0.2`` and 256 frames/scene, stride 5 keeps
0,5,10,...,255 = 52 frames/scene. Across 100 training scenes this is exactly
5200 images/epoch, matching the 10% GraspNet + 10% GN-Trans mixed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset


def _parse_subset_flags():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--train_fraction", type=float, default=0.2)
    p.add_argument("--eval_fraction", type=float, default=0.1)
    args, remaining = p.parse_known_args()
    for name in ("train_fraction", "eval_fraction"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name} must lie in (0,1], got {value}.")
    sys.argv = [sys.argv[0], *remaining]
    return args


SUBSET = _parse_subset_flags()

# utils.arguments parses argv at import time, so subset-only arguments must be
# removed before importing the canonical trainer.
import train_cva_ddp as base
from utils.arguments import cfgs
from dataset.graspnet_dataset import collate_fn


SUBSET_CONTRACT_VERSION = 1
BASE_MAIN_SHA = "ffb73e2cb4d8a05069907dcec53a40fd365a515a"


def _scene_stride_indices(dataset, fraction: float) -> List[int]:
    """Deterministically sample each scene before constructing the DataLoader."""
    if not hasattr(dataset, "scenename") or not hasattr(dataset, "frameid"):
        raise AttributeError("Dataset must expose scenename and frameid.")
    if len(dataset.scenename) != len(dataset) or len(dataset.frameid) != len(dataset):
        raise RuntimeError("Dataset scene/frame metadata length mismatch.")
    if fraction >= 1.0:
        return list(range(len(dataset)))

    stride = max(1, int(round(1.0 / float(fraction))))
    groups: Dict[str, List[int]] = {}
    for idx, scene in enumerate(dataset.scenename):
        groups.setdefault(str(scene), []).append(idx)

    indices: List[int] = []
    for scene in sorted(groups):
        rows = sorted(groups[scene], key=lambda i: int(dataset.frameid[i]))
        indices.extend(rows[::stride])
    return indices


def _fingerprint(indices: List[int]) -> str:
    arr = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


class GraspNetSubsetTrainer(base.Trainer):
    def __init__(self):
        super().__init__()
        if not self.use_cdf:
            raise RuntimeError("GraspNet subset control requires --use_cdf.")
        if not bool(getattr(cfgs, "extend_angle", False)):
            raise RuntimeError("GraspNet subset control requires --extend_angle.")
        if bool(getattr(cfgs, "use_obs_depth", False)):
            raise RuntimeError("This control is RGB-only; remove --use_obs_depth.")

        full_train = self.TRAIN_DATASET
        full_seen = self.TEST_DATASET
        train_idx = _scene_stride_indices(full_train, SUBSET.train_fraction)
        seen_idx = _scene_stride_indices(full_seen, SUBSET.eval_fraction)

        self.TRAIN_DATASET = Subset(full_train, train_idx)
        self.TEST_DATASET = Subset(full_seen, seen_idx)

        self.subset_protocol = {
            "contract_version": SUBSET_CONTRACT_VERSION,
            "base_main_sha": BASE_MAIN_SHA,
            "domain": "graspnet_only",
            "train_fraction": float(SUBSET.train_fraction),
            "eval_fraction": float(SUBSET.eval_fraction),
            "train_count": len(train_idx),
            "seen_val_count": len(seen_idx),
            "train_index_sha256": _fingerprint(train_idx),
            "seen_val_index_sha256": _fingerprint(seen_idx),
            "sampling": "per-scene deterministic stride",
            "model_intervention": "none; equal-exposure data control",
        }

        # For the intended 20% control, fail loudly if the canonical GraspNet
        # 100-scene x 256-frame contract has changed unexpectedly.
        if abs(float(SUBSET.train_fraction) - 0.2) < 1e-12 and len(train_idx) != 5200:
            raise RuntimeError(
                f"20% GraspNet control expected 5200 train frames, got {len(train_idx)}."
            )

        self.train_sampler = DistributedSampler(
            self.TRAIN_DATASET,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=False,
            seed=int(getattr(cfgs, "seed", 0)),
        ) if self.distributed else None
        self.test_sampler = DistributedSampler(
            self.TEST_DATASET,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        ) if self.distributed else None

        self.TRAIN_DATALOADER = DataLoader(
            self.TRAIN_DATASET,
            batch_size=cfgs.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=cfgs.num_workers,
            worker_init_fn=base.my_worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=(cfgs.num_workers > 0),
        )
        eval_workers = max(int(getattr(cfgs, "eval_num_workers", 1)), 0)
        self.TEST_DATALOADER = DataLoader(
            self.TEST_DATASET,
            batch_size=cfgs.batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=eval_workers,
            worker_init_fn=base.my_worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

        if self.main:
            self.log_string("[GRASPNET-SUBSET] " + json.dumps(self.subset_protocol, sort_keys=True))
            Path(cfgs.log_dir, "graspnet_subset_protocol.json").write_text(
                json.dumps(self.subset_protocol, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def save_checkpoint(self, epoch, save_interval=False):
        if not self.main:
            return
        save_dict = {
            "epoch": epoch + 1,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": self.unwrap_model().state_dict(),
            "graspnet_subset_contract_version": SUBSET_CONTRACT_VERSION,
            "graspnet_subset_protocol": self.subset_protocol,
            "use_cdf": bool(self.use_cdf),
            "pose_depth_mode": str(getattr(cfgs, "pose_depth_mode", "none")),
            "use_fuse_depth": bool(getattr(cfgs, "use_fuse_depth", False)),
        }
        if save_interval:
            torch.save(
                save_dict,
                os.path.join(cfgs.log_dir, f"checkpoint_{epoch}.tar"),
            )
        torch.save(save_dict, os.path.join(cfgs.log_dir, "checkpoint.tar"))


def main():
    trainer = GraspNetSubsetTrainer()
    try:
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
