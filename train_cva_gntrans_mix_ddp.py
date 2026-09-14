#!/usr/bin/env python3
"""Train RGB-only EconomicGrasp CVA-CDF on paired GraspNet + GN-Trans RGB.

This experiment changes the training data distribution only.  The network,
CVA/CDF losses, A1 view protocol, optimizer and decoder are inherited from
``train_cva_ddp.py``.

Sampling contract
-----------------
For a fraction f<1, each domain is sampled *independently per scene* with
stride round(1/f).  With f=0.1 and 256 frames/scene this keeps frames
0,10,...,250 (26 frames/scene).  GraspNet and GN-Trans use the same scene/frame
indices, so the training set contains paired clean/material-shifted RGB views
of the same scene geometry.  The two 10% subsets are concatenated, giving equal
domain weight because their sizes are identical.

GN-Trans is used as an RGB training domain, not as an observed-depth input.
The existing GraspNetTransDataset supplies rendered GT metric depth for depth
supervision and virtual graspness labels; inference remains RGB-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset


def _parse_mix_flags():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gntrans_rgb_root", required=True)
    p.add_argument("--mix_train_fraction", type=float, default=0.1)
    p.add_argument("--mix_eval_fraction", type=float, default=0.1)
    p.add_argument(
        "--mix_use_fused_background",
        action="store_true",
        help=(
            "For GN-Trans GT-depth supervision, use the same GraspNet TSDF "
            "background as --use_fuse_depth. Object depth stays rendered GT."
        ),
    )
    args, remaining = p.parse_known_args()
    for name in ("mix_train_fraction", "mix_eval_fraction"):
        x = float(getattr(args, name))
        if not 0.0 < x <= 1.0:
            raise ValueError(f"--{name} must lie in (0,1], got {x}.")
    sys.argv = [sys.argv[0], *remaining]
    return args


MIX = _parse_mix_flags()

# Import only after stripping mix-only CLI arguments.  The repository-wide
# utils.arguments parser is executed by train_cva_ddp at import time.
import train_cva_ddp as base
from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetTransDataset, collate_fn
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter


MIX_CONTRACT_VERSION = 1
MIX_BASE_MAIN_SHA = "ffb73e2cb4d8a05069907dcec53a40fd365a515a"


def _scene_stride_indices(dataset, fraction: float) -> List[int]:
    """Per-scene deterministic stride sampling; never sample after concat."""
    if not hasattr(dataset, "scenename") or not hasattr(dataset, "frameid"):
        raise AttributeError("Mixed CVA datasets must expose scenename and frameid.")
    n = len(dataset)
    if len(dataset.scenename) != n or len(dataset.frameid) != n:
        raise RuntimeError("Dataset scene/frame metadata length mismatch.")
    if fraction >= 1.0:
        return list(range(n))
    stride = max(1, int(round(1.0 / float(fraction))))
    groups: Dict[str, List[int]] = {}
    for idx, scene in enumerate(dataset.scenename):
        groups.setdefault(str(scene), []).append(idx)
    indices: List[int] = []
    for scene in sorted(groups):
        rows = groups[scene]
        rows = sorted(rows, key=lambda i: int(dataset.frameid[i]))
        indices.extend(rows[::stride])
    return indices


def _fingerprint(indices: List[int]) -> str:
    arr = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _assert_paired(original, trans, a: List[int], b: List[int], tag: str) -> None:
    if len(a) != len(b):
        raise RuntimeError(f"{tag}: original/trans sample counts differ: {len(a)} vs {len(b)}")
    for ia, ib in zip(a, b):
        sa, sb = str(original.scenename[ia]), str(trans.scenename[ib])
        fa, fb = int(original.frameid[ia]), int(trans.frameid[ib])
        if sa != sb or fa != fb:
            raise RuntimeError(
                f"{tag}: pair mismatch original=({sa},{fa}) trans=({sb},{fb})."
            )


def _label_folder() -> str:
    return str(
        getattr(cfgs, "cva_label_folder", "")
        or getattr(cfgs, "cdf_label_folder", "")
        or os.environ.get(
            "CVA_LABEL_FOLDER",
            os.environ.get(
                "CDF_LABEL_FOLDER",
                "economic_grasp_label_300views_extend_angle_cdf_depth",
            ),
        )
    )


def _build_trans_adapter(split: str, use_cdf: bool):
    if (getattr(cfgs, "graspness_mode", None) or "scene") != "scene":
        raise ValueError(
            "GraspNetTransDataset currently provides virtual scene-graspness; "
            "use --graspness_mode scene for this controlled experiment."
        )
    trans = GraspNetTransDataset(
        cfgs.dataset_root,
        MIX.gntrans_rgb_root,
        camera=cfgs.camera,
        split=split,
        num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size,
        remove_outlier=True,
        augment=False,
        load_label=True,
        use_gt_depth=True,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        depth_strides=1,
    )
    # The existing GN-Trans loader predates the single-owner CVA adapter.
    # Switch it to the same contract used by GraspNetMultiDataset without
    # changing its default behavior elsewhere in the repository.
    trans.extend_angle = True
    trans.load_grasp_payload = False

    # Optional background target parity with the canonical Stage-1 protocol.
    # Scene geometry/camera are unchanged by material rendering, so the original
    # GraspNet fused background is valid for the paired GN-Trans RGB image.
    if MIX.mix_use_fused_background:
        trans.use_fuse_depth = True
        trans.fusedepthpath = [
            os.path.join(
                cfgs.dataset_root,
                "tsdf_depth",
                str(scene),
                cfgs.camera,
                f"{int(frame):04d}_depth.png",
            )
            for scene, frame in zip(trans.scenename, trans.frameid)
        ]
    else:
        trans.use_fuse_depth = False

    return CVAExtendedLabelAdapter(
        trans,
        dataset_root=cfgs.dataset_root,
        use_cdf=bool(use_cdf),
        label_folder=_label_folder(),
        num_angle=cfgs.num_angle,
        num_depth=cfgs.num_depth,
    )


class GNTransMixedTrainer(base.Trainer):
    def __init__(self):
        super().__init__()
        if not self.use_cdf:
            raise RuntimeError("GN-Trans mixed experiment requires --use_cdf.")
        if not bool(getattr(cfgs, "extend_angle", False)):
            raise RuntimeError("GN-Trans mixed experiment requires --extend_angle.")
        if bool(getattr(cfgs, "use_obs_depth", False)):
            raise RuntimeError("This experiment is RGB-only; remove --use_obs_depth.")

        original_train_full = self.TRAIN_DATASET
        original_seen_full = self.TEST_DATASET
        trans_train_full = _build_trans_adapter("train", self.use_cdf)
        trans_seen_full = _build_trans_adapter("test_seen", self.use_cdf)

        orig_train_idx = _scene_stride_indices(
            original_train_full, MIX.mix_train_fraction
        )
        trans_train_idx = _scene_stride_indices(
            trans_train_full, MIX.mix_train_fraction
        )
        orig_seen_idx = _scene_stride_indices(
            original_seen_full, MIX.mix_eval_fraction
        )
        trans_seen_idx = _scene_stride_indices(
            trans_seen_full, MIX.mix_eval_fraction
        )
        _assert_paired(
            original_train_full, trans_train_full,
            orig_train_idx, trans_train_idx, "train"
        )
        _assert_paired(
            original_seen_full, trans_seen_full,
            orig_seen_idx, trans_seen_idx, "test_seen"
        )

        self.ORIG_TRAIN = Subset(original_train_full, orig_train_idx)
        self.TRANS_TRAIN = Subset(trans_train_full, trans_train_idx)
        self.ORIG_VAL = Subset(original_seen_full, orig_seen_idx)
        self.TRANS_VAL = Subset(trans_seen_full, trans_seen_idx)
        self.TRAIN_DATASET = ConcatDataset([self.ORIG_TRAIN, self.TRANS_TRAIN])
        self.TEST_DATASET = ConcatDataset([self.ORIG_VAL, self.TRANS_VAL])

        self.mix_protocol = {
            "contract_version": MIX_CONTRACT_VERSION,
            "base_main_sha": MIX_BASE_MAIN_SHA,
            "gntrans_rgb_root": str(Path(MIX.gntrans_rgb_root).resolve()),
            "train_fraction_each_domain": float(MIX.mix_train_fraction),
            "eval_fraction_each_domain": float(MIX.mix_eval_fraction),
            "paired_scene_frame_sampling": True,
            "original_train_count": len(self.ORIG_TRAIN),
            "gntrans_train_count": len(self.TRANS_TRAIN),
            "mixed_train_count": len(self.TRAIN_DATASET),
            "original_seen_count": len(self.ORIG_VAL),
            "gntrans_seen_count": len(self.TRANS_VAL),
            "mixed_seen_count": len(self.TEST_DATASET),
            "original_train_index_sha256": _fingerprint(orig_train_idx),
            "gntrans_train_index_sha256": _fingerprint(trans_train_idx),
            "original_seen_index_sha256": _fingerprint(orig_seen_idx),
            "gntrans_seen_index_sha256": _fingerprint(trans_seen_idx),
            "gntrans_object_depth_supervision": "rendered_gt",
            "gntrans_observed_depth_network_input": False,
            "gntrans_fused_background": bool(MIX.mix_use_fused_background),
            "model_intervention": "none; data-distribution-only experiment",
        }

        # Replace the full-data samplers/loaders created by BaseTrainer.
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
            self.log_string("[GNTRANS-MIX] " + json.dumps(self.mix_protocol, sort_keys=True))
            Path(cfgs.log_dir, "gntrans_mix_protocol.json").write_text(
                json.dumps(self.mix_protocol, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def save_checkpoint(self, epoch, save_interval=False):
        if not self.main:
            return
        save_dict = {
            "epoch": epoch + 1,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": self.unwrap_model().state_dict(),
            "gntrans_mix_contract_version": MIX_CONTRACT_VERSION,
            "gntrans_mix_protocol": self.mix_protocol,
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
    trainer = GNTransMixedTrainer()
    try:
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
