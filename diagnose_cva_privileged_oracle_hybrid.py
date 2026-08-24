#!/usr/bin/env python3
"""Checkpoint-only Priority-1 oracle-hybrid PKD diagnosis.

The script imports the current GitHub ``train_cva_distill_ddp.py`` and reuses
its Stage-0/1/2 model, dataset, checkpoint, DDP, exact-view diagnostic, and
logging contracts.  It only adds:

* selectable GraspNet test split;
* oracle teacher-better/common-valid CDF hybrid metrics.

No optimizer step is executed.
"""

from __future__ import annotations

import os
from typing import Any, Dict

# The base trainer consumes its own distillation flags at import time. Priority
# 1 configuration is kept in environment variables so no unknown CLI option can
# leak into ``utils.arguments``.
DIAG_SPLIT = os.environ.get("PRIORITY12_DIAG_SPLIT", "test_seen").strip()
ORACLE_MARGIN = float(
    os.environ.get("PRIORITY12_ORACLE_TEACHER_BETTER_MARGIN", "0.0")
)
DIAGNOSE_EPOCH = int(os.environ.get("DIAGNOSE_EPOCH", "0"))

if DIAG_SPLIT not in {"test_seen", "test_similar", "test_novel"}:
    raise ValueError(
        "PRIORITY12_DIAG_SPLIT must be test_seen, test_similar, or "
        f"test_novel; got {DIAG_SPLIT!r}."
    )
if ORACLE_MARGIN < 0.0:
    raise ValueError(
        "PRIORITY12_ORACLE_TEACHER_BETTER_MARGIN must be non-negative."
    )

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import train_cva_distill_ddp as base
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.privileged_kd_priority12 import (
    compute_oracle_teacher_better_hybrid_diagnostics,
)


def _build_diagnostic_test_loader(trainer: Any, split: str) -> None:
    """Replace the base trainer's fixed test_seen loader with ``split``."""
    cfgs = base.cfgs
    test_base_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        camera=cfgs.camera,
        split=split,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        voxel_size=cfgs.voxel_size,
        use_gt_depth=False,
        use_fuse_depth=cfgs.use_fuse_depth,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        depth_strides=1,
        extend_angle=True,
        load_grasp_payload=False,
    )
    cva_label_folder = str(
        getattr(cfgs, "cva_label_folder", "")
        or os.environ.get(
            "CVA_LABEL_FOLDER",
            os.environ.get(
                "CDF_LABEL_FOLDER",
                "economic_grasp_label_300views_extend_angle_cdf_depth",
            ),
        )
    )
    test_dataset = CVAExtendedLabelAdapter(
        test_base_dataset,
        dataset_root=cfgs.dataset_root,
        use_cdf=trainer.use_cdf,
        label_folder=cva_label_folder,
        num_angle=cfgs.num_angle,
        num_depth=cfgs.num_depth,
    )
    if trainer.distributed:
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=trainer.world_size,
            rank=trainer.rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        test_sampler = None

    eval_num_workers = max(int(getattr(cfgs, "eval_num_workers", 1)), 0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=eval_num_workers,
        worker_init_fn=base.my_worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )

    trainer.TEST_DATASET = test_dataset
    trainer.test_sampler = test_sampler
    trainer.TEST_DATALOADER = test_loader


def _install_oracle_metric_wrapper() -> None:
    """Extend the base paired diagnostics without changing their behavior."""
    original = base.compute_privileged_kd_diagnostics

    def _combined(
        student_end_points: Dict[str, Any],
        teacher_end_points: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = original(student_end_points, teacher_end_points)
        out.update(
            compute_oracle_teacher_better_hybrid_diagnostics(
                student_end_points,
                teacher_end_points,
                teacher_better_margin=ORACLE_MARGIN,
            )
        )
        return out

    base.compute_privileged_kd_diagnostics = _combined


def main() -> None:
    trainer = base.Trainer()
    try:
        if trainer.distill_stage != 2 or trainer.teacher is None:
            raise RuntimeError(
                "Priority-1 diagnosis requires --distill_stage 2, a Stage-0 "
                "--teacher_checkpoint, and a Stage-1/2 student checkpoint."
            )
        if trainer.distributed and not dist.is_initialized():
            raise RuntimeError("Distributed trainer was created without DDP init.")

        _build_diagnostic_test_loader(trainer, DIAG_SPLIT)
        _install_oracle_metric_wrapper()
        trainer.log_string(
            "[PRIORITY12][P1] oracle-hybrid diagnosis: "
            f"split={DIAG_SPLIT}, margin={ORACLE_MARGIN:.6g}, "
            f"paired_batches={trainer.kd_diag_eval_batches}"
        )
        trainer.evaluate_one_epoch(DIAGNOSE_EPOCH)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
