#!/usr/bin/env python3
"""RGB-only CVA-CDF training on GraspNet 10% + GN-Trans 10% + filtered GC6D.

This extends the main-branch GN-Trans mixed trainer without changing the
network, losses, pose-depth protocol or GraspNet evaluation split.

The GC6D scene list must be a previously audited no-GraspNet-test-overlap
subset. A strict expected-count contract prevents accidental use of all GC6D.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler


def _parse_gc6d_flags():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gc6d_root", default=os.environ.get("GC6D_ROOT"))
    parser.add_argument("--gc6d_camera", default="realsense-d435")
    parser.add_argument("--gc6d_scene_ids_file", default=None)
    parser.add_argument(
        "--gc6d_cdf_label_folder",
        default="economic_grasp_label_300views_extend_angle_cdf_depth",
    )
    parser.add_argument("--gc6d_max_ann", type=int, default=13)
    parser.add_argument("--gc6d_expected_scenes", type=int, default=232)
    parser.add_argument("--gc6d_expected_frames", type=int, default=3016)
    parser.add_argument("--gc6d_workspace_pose_mode", default="json")
    parser.add_argument("--gc6d_graspness_mode", choices=("scene", "instance"), default="scene")
    args, other = parser.parse_known_args()
    sys.argv = [sys.argv[0], *other]
    if not args.gc6d_root:
        raise ValueError("Set --gc6d_root or GC6D_ROOT.")
    if not args.gc6d_scene_ids_file:
        args.gc6d_scene_ids_file = os.path.join(
            args.gc6d_root, "split_info", "no_gn_test_overlap",
            "grasp_train_scene_ids_no_graspnet_test_overlap.json",
        )
    return args


GC6D_FLAGS = _parse_gc6d_flags()

# Both parsers must remove their own flags before train_cva_ddp imports cfgs.
import train_cva_gntrans_mix_ddp as pair
from dataset.gc6d_dataset import GraspClutter6DMultiDataset, read_split_scene_ids
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
from dataset.graspnet_dataset import collate_fn
from utils.arguments import cfgs


TRIPLE_MIX_CONTRACT_VERSION = 1


def _filtered_scene_ids(root: Path, source: Path):
    if not source.is_file():
        raise FileNotFoundError(
            f"Audited GC6D scene-list JSON not found: {source}. "
            "Run the existing dataset/gc6d_filter.py workflow first."
        )
    with source.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, list):
        raise ValueError(f"GC6D filtered scene list must be a JSON array: {source}")
    ids = [int(x) for x in data]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate GC6D scene ID(s) in {source}.")
    ids = sorted(ids)
    if len(ids) != int(GC6D_FLAGS.gc6d_expected_scenes):
        raise ValueError(
            f"Expected {GC6D_FLAGS.gc6d_expected_scenes} filtered scenes, "
            f"found {len(ids)} in {source}. Do not train on the unfiltered set."
        )
    official = set(read_split_scene_ids(root, "train"))
    extra = sorted(set(ids) - official)
    if extra:
        raise RuntimeError(
            f"GC6D filtered scene list contains {len(extra)} non-training scenes: "
            f"{extra[:15]}"
        )
    return ids


class GraspNetGNTransGC6DMixedTrainer(pair.GNTransMixedTrainer):
    def __init__(self):
        super().__init__()
        root = Path(GC6D_FLAGS.gc6d_root).expanduser().resolve()
        split_path = Path(GC6D_FLAGS.gc6d_scene_ids_file).expanduser().resolve()
        ids = _filtered_scene_ids(root, split_path)

        label_folder = GC6D_FLAGS.gc6d_cdf_label_folder
        if os.path.isabs(label_folder) or ".." in Path(label_folder).parts:
            raise ValueError("--gc6d_cdf_label_folder must be relative to --gc6d_root.")
        base_dataset = GraspClutter6DMultiDataset(
            root=str(root),
            camera=GC6D_FLAGS.gc6d_camera,
            split="train",
            scene_ids=ids,
            max_ann=GC6D_FLAGS.gc6d_max_ann,
            num_points=cfgs.num_point,
            voxel_size=cfgs.voxel_size,
            augment=False,
            remove_outlier=None,
            mask_mode="workspace_depth",
            workspace_pose_mode=GC6D_FLAGS.gc6d_workspace_pose_mode,
            load_label=True,
            load_grasp_payload=False,
            use_gt_depth=False,
            # GC6D virtual depth is already fused; GraspNet TSDF option
            # must not be blindly forwarded to a different data layout.
            use_fuse_depth=False,
            graspness_mode=GC6D_FLAGS.gc6d_graspness_mode,
            extend_angle=True,
            economic_label_root=str(root / label_folder),
            scene_name_format="plain6",
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            depth_strides=1,
            strict_label_check=True,
        )
        if len(base_dataset) != int(GC6D_FLAGS.gc6d_expected_frames):
            raise ValueError(
                f"GC6D expected {GC6D_FLAGS.gc6d_expected_frames} training frames "
                f"but indexed {len(base_dataset)}. Check camera/max_ann/scene list."
            )
        self.GC6D_TRAIN = CVAExtendedLabelAdapter(
            base_dataset,
            dataset_root=str(root),
            use_cdf=self.use_cdf,
            label_folder=label_folder,
            num_angle=cfgs.num_angle,
            num_depth=cfgs.num_depth,
        )
        if not len(self.ORIG_TRAIN) or not len(self.TRANS_TRAIN):
            raise RuntimeError("GraspNet/GN-Trans selection must be nonempty.")

        # Natural approximately balanced exposure: 2600 / 2600 / 3016.
        self.TRAIN_DATASET = ConcatDataset(
            [self.ORIG_TRAIN, self.TRANS_TRAIN, self.GC6D_TRAIN]
        )
        self.train_sampler = (
            DistributedSampler(
                self.TRAIN_DATASET,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
                seed=int(getattr(cfgs, "seed", 0)),
            ) if self.distributed else None
        )
        self.TRAIN_DATALOADER = DataLoader(
            self.TRAIN_DATASET,
            batch_size=cfgs.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=cfgs.num_workers,
            worker_init_fn=pair.base.my_worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=(cfgs.num_workers > 0),
        )
        # Validation stays original+GN-Trans seen 10%, as in the parent.
        # Full per-domain seen/similar/novel AP is handled by the eval launcher.

        self.mix_protocol.update({
            "grasp_design": self.unwrap_model().grasp_design,
            "gc6d_graspness_mode": GC6D_FLAGS.gc6d_graspness_mode,
            "contract_version": TRIPLE_MIX_CONTRACT_VERSION,
            "sources": ["graspnet", "gntrans", "graspclutter6d_filtered"],
            "gc6d_root": str(root),
            "gc6d_scene_ids_file": str(split_path),
            "gc6d_scene_ids_sha256": hashlib.sha256(
                ",".join(map(str, ids)).encode("ascii")
            ).hexdigest(),
            "gc6d_scene_ids_file_sha256": hashlib.sha256(
                split_path.read_bytes()
            ).hexdigest(),
            "gc6d_camera": base_dataset.camera,
            "gc6d_cdf_label_folder": label_folder,
            "gc6d_scene_count": len(ids),
            "gc6d_frame_count": len(self.GC6D_TRAIN),
            "gc6d_max_ann": GC6D_FLAGS.gc6d_max_ann,
            "gc6d_uses_fused_background": False,
            "graspnet_train_count": len(self.ORIG_TRAIN),
            "gntrans_train_count": len(self.TRANS_TRAIN),
            "mixed_train_count": len(self.TRAIN_DATASET),
            "source_weights": "natural-frame-count; not oversampled",
            "training_input": "RGB only; other modalities supervision only",
            "in_training_validation": "original+gntrans test_seen subsets",
            "filter_provenance": "audited scene-ID file; raw-file and canonical-ID SHA256 recorded",
            "model_intervention": self.unwrap_model().grasp_design,
        })
        if self.main:
            self.log_string("[GNTRANS-GC6D-MIX] " + json.dumps(self.mix_protocol, sort_keys=True))
            protocol_json = json.dumps(self.mix_protocol, indent=2, sort_keys=True)
            (Path(cfgs.log_dir) / "gntrans_gc6d_mix_protocol.json").write_text(
                protocol_json, encoding="utf-8",
            )
            # The inherited two-source trainer has already written this path;
            # refresh it so there is no stale 5200-sample manifest next to
            # the actual three-source 8216-sample run.
            (Path(cfgs.log_dir) / "gntrans_mix_protocol.json").write_text(
                protocol_json, encoding="utf-8",
            )


def main():
    trainer = GraspNetGNTransGC6DMixedTrainer()
    try:
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
