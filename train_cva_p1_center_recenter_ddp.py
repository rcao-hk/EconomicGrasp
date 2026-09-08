#!/usr/bin/env python3
"""Train P1 sparse grasp-query center recentering on top of a Stage-1 RGB model.

This entry point deliberately reuses the current CVA-CDF Stage-1 trainer and
changes only:
  1) the student model class, adding the P1 sparse z-residual head;
  2) an additional direct center-recentering loss at student-selected seeds;
  3) optional deterministic scene-stratified train/eval subsampling;
  4) checkpoint metadata for strict P1 inference/resume.

No teacher and no Dex-Net/GraspNet evaluator is invoked during training.

Recommended first causal run:
    --p1_train_mode center_only
This freezes every Stage-1 parameter, trains only the zero-initialized residual
head from GT depth at the selected image-FPS seeds, and stops grasp-loss
gradients at the corrected center. ``head_grasp`` and ``joint`` are follow-up
optimization variants, not the primary mechanism control.

For fast mechanism validation, use for example:
    --p1_train_sample_interval 0.1
    --p1_eval_sample_interval 0.1
Sampling is deterministic and stratified within every scene rather than taking
only the first fraction of scenes.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import torch


def _parse_p1_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--p1_train_mode",
        choices=("center_only", "head_grasp", "joint"),
        default="center_only",
    )
    parser.add_argument("--p1_center_loss_weight", type=float, default=1.0)
    parser.add_argument("--p1_center_beta_m", type=float, default=0.01)
    parser.add_argument("--p1_max_residual_m", type=float, default=0.08)
    parser.add_argument("--p1_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--p1_train_sample_interval",
        type=float,
        default=1.0,
        help=(
            "Fraction of frames retained independently inside every training "
            "scene. 1.0 keeps the full training set; 0.1 keeps about 1/10 of "
            "the frames in every scene using deterministic even sampling."
        ),
    )
    parser.add_argument(
        "--p1_eval_sample_interval",
        type=float,
        default=1.0,
        help=(
            "Fraction of frames retained independently inside every validation "
            "scene. 1.0 keeps the full validation set; 0.1 keeps about 1/10 of "
            "the frames in every scene using deterministic even sampling."
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


P1_ARGS = _parse_p1_args()

# Import only after consuming P1-specific flags. The base trainer then consumes
# distillation flags and finally lets utils.arguments parse the ordinary config.
import train_cva_distill_ddp as base

from models.p1_grasp_center_recenter import (
    P1_CENTER_RECENTER_CONTRACT_VERSION,
    compute_p1_center_recenter_loss,
    economicgrasp_dpt_p1_center_recenter,
    p1_train_mode_flags,
)


if float(P1_ARGS.p1_center_loss_weight) <= 0.0:
    raise ValueError("--p1_center_loss_weight must be positive.")
if float(P1_ARGS.p1_center_beta_m) <= 0.0:
    raise ValueError("--p1_center_beta_m must be positive.")
if int(P1_ARGS.p1_hidden_dim) <= 0:
    raise ValueError("--p1_hidden_dim must be positive.")
for _name in ("p1_train_sample_interval", "p1_eval_sample_interval"):
    _value = float(getattr(P1_ARGS, _name))
    if not (0.0 < _value <= 1.0):
        raise ValueError(f"--{_name} must lie in (0, 1], got {_value}.")

_MODE_FLAGS = p1_train_mode_flags(P1_ARGS.p1_train_mode)


class _ConfiguredP1Student(economicgrasp_dpt_p1_center_recenter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args,
            p1_hidden_dim=int(P1_ARGS.p1_hidden_dim),
            p1_max_residual_m=float(P1_ARGS.p1_max_residual_m),
            p1_compute_gt_targets=True,
            p1_force_zero_residual=False,
            **_MODE_FLAGS,
            **kwargs,
        )


# Patch the globals resolved by base.Trainer.__init__ and its train/eval loops.
# Stage 0/2 are explicitly rejected below, so only the student constructor is
# replaced.
base.economicgrasp_dpt_student = _ConfiguredP1Student
_BASE_GRASP_LOSS = base.get_loss_economicgrasp


def _get_p1_loss(end_points, use_cdf: bool = False):
    base_loss, end_points = _BASE_GRASP_LOSS(
        end_points,
        use_cdf=use_cdf,
    )
    center_loss, end_points = compute_p1_center_recenter_loss(
        end_points,
        beta_m=float(P1_ARGS.p1_center_beta_m),
    )
    weighted_center = float(P1_ARGS.p1_center_loss_weight) * center_loss
    total = base_loss + weighted_center

    end_points["A: P1 Base CVA Loss"] = base_loss
    end_points["A: P1 Center Loss Weighted"] = weighted_center
    end_points["A: P1 Total Loss"] = total
    end_points["A: Overall Loss"] = total
    return total, end_points


base.get_loss_economicgrasp = _get_p1_loss


def _read_checkpoint_metadata(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "P1 requires --checkpoint_path pointing to a Stage-1 initialization "
            f"or a resumable P1 checkpoint; got {path!r}."
        )
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "P1 requires a full EconomicGrasp training checkpoint with metadata."
        )
    return checkpoint


def _validate_initial_checkpoint(checkpoint: Dict[str, Any]) -> None:
    resume = bool(getattr(base.cfgs, "resume", False))
    path = str(base.CHECKPOINT_PATH or "")

    if resume:
        if not bool(checkpoint.get("p1_center_recenter", False)):
            raise RuntimeError(
                "--resume requires a P1 checkpoint, not a Stage-1 baseline."
            )
        if int(checkpoint.get("p1_contract_version", -1)) != (
            P1_CENTER_RECENTER_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "P1 resume contract mismatch: expected "
                f"{P1_CENTER_RECENTER_CONTRACT_VERSION}, got "
                f"{checkpoint.get('p1_contract_version')!r}."
            )
        for key, expected in (
            ("p1_train_mode", str(P1_ARGS.p1_train_mode)),
            ("p1_hidden_dim", int(P1_ARGS.p1_hidden_dim)),
        ):
            if checkpoint.get(key) != expected:
                raise RuntimeError(
                    f"P1 resume mismatch for {key}: checkpoint="
                    f"{checkpoint.get(key)!r}, current={expected!r}."
                )
        for key, expected, default in (
            ("p1_max_residual_m", float(P1_ARGS.p1_max_residual_m), float("nan")),
            ("p1_center_loss_weight", float(P1_ARGS.p1_center_loss_weight), float("nan")),
            ("p1_center_beta_m", float(P1_ARGS.p1_center_beta_m), float("nan")),
            # Older P1 checkpoints predate explicit sampling and therefore
            # correspond to the full-dataset value 1.0.
            ("p1_train_sample_interval", float(P1_ARGS.p1_train_sample_interval), 1.0),
            ("p1_eval_sample_interval", float(P1_ARGS.p1_eval_sample_interval), 1.0),
        ):
            saved = float(checkpoint.get(key, default))
            if abs(saved - expected) > 1.0e-12:
                raise RuntimeError(
                    f"P1 resume mismatch for {key}: checkpoint={saved}, "
                    f"current={expected}."
                )
    else:
        # Primary P1 protocol starts from the controlled Stage-1 RGB baseline.
        if bool(checkpoint.get("p1_center_recenter", False)):
            raise RuntimeError(
                "Fresh P1 initialization must use a Stage-1 baseline checkpoint. "
                "Use --resume to continue an existing P1 run."
            )
        if int(checkpoint.get("distill_stage", -1)) != 1:
            raise RuntimeError(
                "Fresh P1 must initialize from distill_stage=1; got "
                f"{checkpoint.get('distill_stage')!r} from {path}."
            )
        if int(checkpoint.get("distill_contract_version", -1)) != (
            base.DISTILL_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "Stage-1 initialization uses an incompatible distillation "
                "contract."
            )
        if str(checkpoint.get("seed_selection_mode", "")) != "image_fps":
            raise RuntimeError(
                "P1 requires the controlled Stage-1 image-FPS checkpoint."
            )
        if str(checkpoint.get("geometry_depth_source", "")) != "pred":
            raise RuntimeError(
                "P1 must initialize from an RGB predicted-depth Stage-1 model."
            )

    if int(checkpoint.get("distill_stage", -1)) != 1:
        raise RuntimeError("P1 checkpoints retain distill_stage=1 metadata.")
    if bool(checkpoint.get("use_fuse_depth", False)) != bool(
        base.cfgs.use_fuse_depth
    ):
        raise RuntimeError(
            "P1 --use_fuse_depth must match the initialization/resume checkpoint."
        )
    saved_pose = str(checkpoint.get("pose_depth_mode", ""))
    current_pose = str(getattr(base.cfgs, "pose_depth_mode", "none") or "none")
    if saved_pose != current_pose:
        raise RuntimeError(
            "P1 pose_depth_mode mismatch: checkpoint="
            f"{saved_pose!r}, current={current_pose!r}."
        )


def _normalize_scene_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _scene_stratified_indices(
    dataset,
    interval: float,
) -> Tuple[List[int], Dict[str, Tuple[int, int]]]:
    """Select an even deterministic fraction independently inside every scene.

    This intentionally uses the dataset's actual ``scenename`` sequence instead
    of assuming 256 frames/scene. Therefore it remains correct for incomplete or
    filtered scenes as long as the wrapped dataset preserves one scene name per
    sample.
    """
    interval = float(interval)
    if not (0.0 < interval <= 1.0):
        raise ValueError(f"sample interval must be in (0,1], got {interval}.")
    total = len(dataset)
    if total <= 0:
        raise RuntimeError("Cannot subsample an empty dataset.")

    scene_names = getattr(dataset, "scenename", None)
    if scene_names is None:
        raise AttributeError(
            "P1 scene-stratified sampling requires dataset.scenename."
        )
    if len(scene_names) != total:
        raise RuntimeError(
            "dataset.scenename length does not match dataset length: "
            f"{len(scene_names)} vs {total}."
        )

    groups: Dict[str, List[int]] = {}
    for index, scene in enumerate(scene_names):
        name = _normalize_scene_name(scene)
        groups.setdefault(name, []).append(index)

    selected: List[int] = []
    counts: Dict[str, Tuple[int, int]] = {}
    for scene, rows in groups.items():
        n = len(rows)
        if interval >= 1.0:
            chosen = list(rows)
        else:
            keep = max(1, int(round(float(n) * interval)))
            keep = min(keep, n)
            # Even deterministic positions in [0,n). Because keep <= n,
            # floor(j*n/keep) is strictly increasing and creates no duplicates.
            positions = [(j * n) // keep for j in range(keep)]
            chosen = [rows[pos] for pos in positions]
        selected.extend(chosen)
        counts[scene] = (n, len(chosen))

    selected.sort()
    if len(selected) <= 0:
        raise RuntimeError("Scene-stratified sampling selected no frames.")
    return selected, counts


class P1CenterRecenterTrainer(base.Trainer):
    def __init__(self) -> None:
        if int(base.DISTILL_ARGS.distill_stage) != 1:
            raise RuntimeError(
                "P1 is a Stage-1 RGB-only extension. Launch with --distill_stage 1."
            )
        if str(base.DISTILL_ARGS.teacher_checkpoint).strip():
            raise RuntimeError("P1 training does not use a teacher checkpoint.")
        checkpoint = _read_checkpoint_metadata(str(base.CHECKPOINT_PATH or ""))
        _validate_initial_checkpoint(checkpoint)
        del checkpoint

        super().__init__()

        # Apply P1-only data reduction after the base trainer has constructed
        # the canonical CVA datasets. Model/checkpoint construction is therefore
        # unchanged; only the train/eval iterators are replaced.
        self._apply_p1_sample_intervals()

        model = self.unwrap_model()
        if not isinstance(model, economicgrasp_dpt_p1_center_recenter):
            raise RuntimeError(
                "P1 trainer did not instantiate the configured recenter model."
            )
        if self.teacher is not None:
            raise RuntimeError("P1 must not instantiate a privileged teacher.")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        head_trainable = sum(
            p.numel()
            for p in model.p1_center_recenter_head.parameters()
            if p.requires_grad
        )
        if trainable <= 0 or head_trainable <= 0:
            raise RuntimeError("P1 recenter head has no trainable parameters.")
        if bool(_MODE_FLAGS["p1_freeze_base"]) and trainable != head_trainable:
            raise RuntimeError(
                "P1 frozen-base mode exposed trainable parameters outside the "
                "recenter head."
            )

        self.log_string(
            "-> P1 grasp-query center recentering enabled: "
            f"mode={P1_ARGS.p1_train_mode}, max_residual="
            f"{float(P1_ARGS.p1_max_residual_m):.4f}m, hidden="
            f"{int(P1_ARGS.p1_hidden_dim)}, center_loss_weight="
            f"{float(P1_ARGS.p1_center_loss_weight):g}, beta="
            f"{float(P1_ARGS.p1_center_beta_m):.4f}m"
        )
        self.log_string(
            "-> P1 trainability: "
            f"trainable_params={trainable}, head_trainable={head_trainable}, "
            f"base_frozen={int(_MODE_FLAGS['p1_freeze_base'])}, "
            "DexNet_evaluator=0"
        )

    def _apply_p1_sample_intervals(self) -> None:
        train_interval = float(P1_ARGS.p1_train_sample_interval)
        eval_interval = float(P1_ARGS.p1_eval_sample_interval)

        train_full_dataset = self.TRAIN_DATASET
        eval_full_dataset = self.TEST_DATASET
        self.p1_train_full_size = len(train_full_dataset)
        self.p1_eval_full_size = len(eval_full_dataset)

        train_indices, train_counts = _scene_stratified_indices(
            train_full_dataset, train_interval
        )
        eval_indices, eval_counts = _scene_stratified_indices(
            eval_full_dataset, eval_interval
        )

        if len(train_indices) < self.p1_train_full_size:
            self.TRAIN_DATASET = torch.utils.data.Subset(
                train_full_dataset, train_indices
            )
        if len(eval_indices) < self.p1_eval_full_size:
            self.TEST_DATASET = torch.utils.data.Subset(
                eval_full_dataset, eval_indices
            )

        self.p1_train_selected_size = len(self.TRAIN_DATASET)
        self.p1_eval_selected_size = len(self.TEST_DATASET)
        self.p1_train_scene_count = len(train_counts)
        self.p1_eval_scene_count = len(eval_counts)

        # Rebuild samplers/loaders on the selected subsets. DistributedSampler
        # retains epoch-wise shuffling for training while validation remains
        # deterministic. In single-GPU mode, DataLoader shuffle matches base.
        self.train_sampler = (
            base.DistributedSampler(
                self.TRAIN_DATASET,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
            if self.distributed
            else None
        )
        self.test_sampler = (
            base.DistributedSampler(
                self.TEST_DATASET,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            if self.distributed
            else None
        )

        self.TRAIN_DATALOADER = base.DataLoader(
            self.TRAIN_DATASET,
            batch_size=base.cfgs.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=base.cfgs.num_workers,
            worker_init_fn=base.my_worker_init_fn,
            collate_fn=base.collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=(base.cfgs.num_workers > 0),
        )
        eval_num_workers = max(
            int(getattr(base.cfgs, "eval_num_workers", 1)), 0
        )
        self.TEST_DATALOADER = base.DataLoader(
            self.TEST_DATASET,
            batch_size=base.cfgs.batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=eval_num_workers,
            worker_init_fn=base.my_worker_init_fn,
            collate_fn=base.collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

        train_per_scene_selected = [selected for _, selected in train_counts.values()]
        eval_per_scene_selected = [selected for _, selected in eval_counts.values()]
        self.log_string(
            "-> P1 scene-stratified sampling: "
            f"train_interval={train_interval:g} "
            f"frames={self.p1_train_selected_size}/{self.p1_train_full_size} "
            f"scenes={self.p1_train_scene_count} "
            f"per_scene_selected[min/max]="
            f"{min(train_per_scene_selected)}/{max(train_per_scene_selected)}; "
            f"eval_interval={eval_interval:g} "
            f"frames={self.p1_eval_selected_size}/{self.p1_eval_full_size} "
            f"scenes={self.p1_eval_scene_count} "
            f"per_scene_selected[min/max]="
            f"{min(eval_per_scene_selected)}/{max(eval_per_scene_selected)}"
        )

    def _checkpoint_payload(
        self,
        epoch: int,
        *,
        include_optimizer: bool,
        train_loss: float | None = None,
        eval_loss: float | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "epoch": int(epoch) + 1,
            "model_state_dict": self.unwrap_model().state_dict(),
            "distill_stage": 1,
            "teacher_checkpoint": "",
            "distillation_config": self.distill_config.to_dict(),
            "distill_contract_version": base.DISTILL_CONTRACT_VERSION,
            "seed_selection_mode": "image_fps",
            "geometry_depth_source": "pred",
            "teacher_geometry_depth_source": None,
            "depth_head_executed": True,
            "pose_depth_mode": self.train_pose_depth_mode,
            "camera_pose_key": str(
                getattr(base.cfgs, "camera_pose_key", "camera_pose_vec")
            ),
            "camera_gravity_key": str(
                getattr(base.cfgs, "camera_gravity_key", "camera_gravity_vec")
            ),
            "pose_hidden_dim": int(getattr(base.cfgs, "pose_hidden_dim", 64)),
            "ray_gravity_hidden_dim": int(
                getattr(base.cfgs, "ray_gravity_hidden_dim", 64)
            ),
            "ray_gravity_mid_dim": int(
                getattr(base.cfgs, "ray_gravity_mid_dim", 32)
            ),
            "use_fuse_depth": bool(base.cfgs.use_fuse_depth),
            "legacy_dataset_use_gt_depth": False,
            "stage2_shared_teacher_image_fps": False,
            "stage2_seed_source": None,
            "stage2_teacher_reuses_student_image_fps": False,
            # P1 contract.
            "experiment": "p1_grasp_query_center_recenter_v1",
            "p1_center_recenter": True,
            "p1_contract_version": P1_CENTER_RECENTER_CONTRACT_VERSION,
            "p1_train_mode": str(P1_ARGS.p1_train_mode),
            "p1_center_loss_weight": float(P1_ARGS.p1_center_loss_weight),
            "p1_center_beta_m": float(P1_ARGS.p1_center_beta_m),
            "p1_max_residual_m": float(P1_ARGS.p1_max_residual_m),
            "p1_hidden_dim": int(P1_ARGS.p1_hidden_dim),
            "p1_freeze_base": bool(_MODE_FLAGS["p1_freeze_base"]),
            "p1_detach_head_features": bool(
                _MODE_FLAGS["p1_detach_head_features"]
            ),
            "p1_detach_corrected_for_downstream": bool(
                _MODE_FLAGS["p1_detach_corrected_for_downstream"]
            ),
            "p1_inference_requires_gt_depth": False,
            "p1_training_uses_gt_depth_at_selected_queries": True,
            "p1_uses_dexnet_evaluator": False,
            "p1_initialization_checkpoint": str(base.CHECKPOINT_PATH or ""),
            # Dataset-reduction protocol.
            "p1_sampling_protocol": "deterministic_scene_stratified_even_v1",
            "p1_train_sample_interval": float(P1_ARGS.p1_train_sample_interval),
            "p1_eval_sample_interval": float(P1_ARGS.p1_eval_sample_interval),
            "p1_train_full_size": int(self.p1_train_full_size),
            "p1_train_selected_size": int(self.p1_train_selected_size),
            "p1_train_scene_count": int(self.p1_train_scene_count),
            "p1_eval_full_size": int(self.p1_eval_full_size),
            "p1_eval_selected_size": int(self.p1_eval_selected_size),
            "p1_eval_scene_count": int(self.p1_eval_scene_count),
        }
        if include_optimizer:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if train_loss is not None:
            payload["train_loss"] = float(train_loss)
        if eval_loss is not None:
            payload["eval_loss"] = float(eval_loss)
        return payload

    def save_best_state_dict(self, epoch, train_loss, eval_loss):
        if not self.main:
            return
        ckpt_name = f"epoch_{epoch}_train_{train_loss}_val_{eval_loss}"
        torch.save(
            self._checkpoint_payload(
                epoch,
                include_optimizer=False,
                train_loss=train_loss,
                eval_loss=eval_loss,
            ),
            os.path.join(base.cfgs.log_dir, ckpt_name + ".tar"),
        )

    def save_checkpoint(self, epoch, save_interval=False):
        if not self.main:
            return
        payload = self._checkpoint_payload(epoch, include_optimizer=True)
        if save_interval:
            torch.save(
                payload,
                os.path.join(base.cfgs.log_dir, f"checkpoint_{epoch}.tar"),
            )
        torch.save(
            payload,
            os.path.join(base.cfgs.log_dir, "checkpoint.tar"),
        )


def main() -> None:
    trainer = P1CenterRecenterTrainer()
    try:
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
