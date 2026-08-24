"""Stage-2 Exact-Query Oracle-Selective CDF Distillation.

This entry point reuses the current ``train_cva_distill_ddp.py`` trainer and
changes only the Stage-2 teacher-query protocol and CDF KD term:

* trainable RGB student starts from the repository's normal task initialization;
* frozen Stage-0 clean-geometry teacher reuses the student's exact image-FPS
  seeds and exact selected views;
* teacher and student are compared with ordinary per-query CDF BCE on the
  student query's GT and common-valid support;
* CDF soft-target KD is applied only to teacher-better queries.

No balanced CDF loss, feature KD, view/width/depth KD, margin, warm-up, or new
inference component is introduced.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import torch

# Importing the base trainer intentionally consumes the existing distillation
# and repository arguments. This companion entry point introduces no new CLI
# surface; selecting this file selects the oracle-selective protocol.
import train_cva_distill_ddp as base

from models.oracle_selective_cdf_kd import (
    compute_oracle_selective_cdf_distillation_loss,
    extract_oracle_selective_teacher_targets,
)


# The base loop resolves these symbols at runtime. Replacing only these two
# functions leaves data loading, supervised losses, optimization, checkpointing,
# and evaluation unchanged.
base.extract_distillation_targets = extract_oracle_selective_teacher_targets
base.compute_output_distillation_loss = (
    compute_oracle_selective_cdf_distillation_loss
)


class OracleSelectiveCDFTrainer(base.Trainer):
    """Minimal Stage-2 trainer with exact-view teacher and selective CDF KD."""

    def __init__(self) -> None:
        self._latest_student_query: dict[str, torch.Tensor] | None = None
        self._student_hook_handle = None
        self._teacher_pre_hook_handle = None

        super().__init__()

        if self.distill_stage != 2:
            raise RuntimeError(
                "train_cva_distill_oracle_selective_ddp.py is Stage-2 only. "
                "Launch it with --distill_stage 2."
            )
        if self.teacher is None:
            raise RuntimeError(
                "Oracle-selective CDF KD requires a frozen Stage-0 teacher."
            )

        # The requested experiment trains Stage 2 from the normal fresh task
        # initialization. Reject both checkpoint initialization and resume so a
        # launcher cannot silently turn this into Stage-1 fine-tuning.
        checkpoint_path = str(
            getattr(base.cfgs, "checkpoint_path", "") or ""
        ).strip()
        if checkpoint_path:
            raise RuntimeError(
                "Oracle-selective Stage 2 must start from fresh task "
                "initialization. Do not pass --checkpoint_path."
            )
        if bool(getattr(base.cfgs, "resume", False)):
            raise RuntimeError(
                "Oracle-selective Stage 2 does not resume a student checkpoint."
            )
        if self.start_epoch != 0:
            raise RuntimeError(
                f"Fresh Stage-2 training must start at epoch 0, got "
                f"{self.start_epoch}."
            )

        # Enforce the single-factor method: only ordinary CDF KD is active.
        expected_zero = {
            "objectness_weight": self.distill_config.objectness_weight,
            "graspness_weight": self.distill_config.graspness_weight,
            "depth_weight": self.distill_config.depth_weight,
            "view_weight": self.distill_config.view_weight,
            "width_weight": self.distill_config.width_weight,
        }
        nonzero = {
            key: float(value)
            for key, value in expected_zero.items()
            if abs(float(value)) > 1.0e-12
        }
        if nonzero:
            raise RuntimeError(
                "Exact-Query Oracle-Selective CDF Distillation permits only "
                f"CDF KD. Set all other KD weights to zero; got {nonzero}."
            )
        if float(self.distill_config.cdf_weight) <= 0.0:
            raise RuntimeError("--kd_cdf_weight must be positive.")
        if float(self.distill_config.overall_weight) <= 0.0:
            raise RuntimeError("--distill_weight must be positive.")
        if abs(float(self.distill_config.temperature) - 1.0) > 1.0e-12:
            raise RuntimeError(
                "The minimal protocol fixes --kd_temperature 1.0."
            )

        # Existing paired diagnostics contain balanced-BCE analyses. They are
        # intentionally disabled here so the training protocol and logs focus
        # on the ordinary-BCE oracle gate only.
        self.kd_diag_interval_steps = 0
        self.kd_diag_eval_batches = -1
        self.kd_diag_grad_conflict = False

        # Cache the latest student endpoints, then inject the student's exact
        # query contract into every frozen-teacher forward performed by the base
        # Stage-2 loop.
        self._student_hook_handle = self.unwrap_model().register_forward_hook(
            self._cache_student_end_points
        )
        self._teacher_pre_hook_handle = self.teacher.register_forward_pre_hook(
            self._prepare_exact_query_teacher_input
        )

        self.log_string(
            "-> Exact-Query Oracle-Selective CDF Distillation enabled: "
            "ordinary per-query BCE gate, common-valid support, CDF-only KD"
        )
        self.log_string(
            "-> Stage-2 student initialization=fresh; teacher seed/view source="
            "student; margin=0; temperature=1; balanced_loss=0"
        )
        self._write_protocol_metadata()

    def _cache_student_end_points(self, module, args, output) -> None:
        del module, args
        if not isinstance(output, Mapping):
            raise TypeError(
                "Student forward must return an endpoint mapping for "
                "oracle-selective KD."
            )
        required = (
            "kview_base_token_sel_idx",
            "token_sel_idx",
            "grasp_top_view_inds",
        )
        missing = [key for key in required if key not in output]
        if missing:
            raise KeyError(
                "Student forward is missing endpoint(s) required by exact-query "
                "oracle-selective KD: " + ", ".join(missing)
            )

        # Keep only detached integer query ids. Retaining the complete endpoint
        # dictionary here would keep the previous student autograd graph alive
        # until the next forward and unnecessarily increase peak memory.
        self._latest_student_query = {
            key: output[key].detach().long()
            for key in required
        }

    def _prepare_exact_query_teacher_input(self, module, args) -> None:
        del module
        if not args or not isinstance(args[0], dict):
            raise TypeError(
                "Frozen teacher forward must receive one endpoint dictionary."
            )
        if self._latest_student_query is None:
            raise RuntimeError(
                "Teacher forward occurred before a student forward; exact-query "
                "overrides are unavailable."
            )

        teacher_input = args[0]
        student = self._latest_student_query
        student_seed = student["kview_base_token_sel_idx"].detach().long()
        student_query = student["token_sel_idx"].detach().long()
        student_view = student["grasp_top_view_inds"].detach().long()
        if (
            student_seed.dim() != 2
            or student_query.dim() != 2
            or student_view.dim() != 2
        ):
            raise RuntimeError(
                "Exact-query Stage 2 requires [B,M] seed/query/view indices; "
                f"got seed={tuple(student_seed.shape)}, "
                f"query={tuple(student_query.shape)}, "
                f"view={tuple(student_view.shape)}."
            )
        if not (
            student_seed.shape
            == student_query.shape
            == student_view.shape
        ):
            raise RuntimeError(
                "This minimal variant supports one selected view per image-FPS "
                "seed (Q=M). Top-K training requires a separate selector-level "
                f"override; got seed={tuple(student_seed.shape)}, "
                f"query={tuple(student_query.shape)}, "
                f"view={tuple(student_view.shape)}."
            )
        if not torch.equal(student_query, student_seed):
            raise RuntimeError(
                "Exact-query Stage 2 requires token_sel_idx to equal the "
                "image-FPS base indices for every query."
            )
        teacher_input["image_fps_seed_idx_override"] = student_seed
        teacher_input["oracle_view_inds_override"] = student_view

        # A frozen eval-mode teacher normally skips label matching. The
        # teacher-better gate requires its valid mask, so request the ordinary
        # CDF label matcher without changing teacher training/eval mode.
        teacher_input["cva_force_process_grasp_labels"] = True
        teacher_input["cva_compute_diagnostics"] = False
        teacher_input["geometry_compute_diagnostics"] = False
        teacher_input["cva_export_angle_feature"] = False

    def _write_protocol_metadata(self) -> None:
        if not self.main:
            return
        metadata = {
            "protocol": "exact-query-oracle-selective-cdf-kd-v1",
            "distill_stage": 2,
            "student_initialization": "fresh_task_initialization",
            "teacher_checkpoint": str(base.DISTILL_ARGS.teacher_checkpoint),
            "teacher_geometry": "gt",
            "student_geometry": "pred",
            "teacher_seed_override": "student_kview_base_token_sel_idx",
            "teacher_view_override": "student_grasp_top_view_inds",
            "force_teacher_label_processing": True,
            "gate_target": "student_query_gt_cdf",
            "gate_loss": "ordinary_per_query_bce",
            "gate_margin": 0.0,
            "common_valid_support": True,
            "kd_target": "teacher_soft_cdf",
            "kd_loss": "ordinary_bce_with_logits",
            "temperature": 1.0,
            "balanced_supervised_cdf": False,
            "balanced_kd": False,
            "kd_weights": {
                "overall": float(self.distill_config.overall_weight),
                "objectness": float(self.distill_config.objectness_weight),
                "graspness": float(self.distill_config.graspness_weight),
                "depth": float(self.distill_config.depth_weight),
                "view": float(self.distill_config.view_weight),
                "cdf": float(self.distill_config.cdf_weight),
                "width": float(self.distill_config.width_weight),
            },
            "seed": int(getattr(base.cfgs, "seed", 0)),
            "use_fuse_depth": bool(base.cfgs.use_fuse_depth),
            "pose_depth_mode": str(self.train_pose_depth_mode),
            "graspness_mode": str(base.cfgs.graspness_mode),
        }
        path = os.path.join(
            base.cfgs.log_dir,
            "oracle_selective_cdf_protocol.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        self.log_string(f"-> protocol metadata: {path}")

    def close(self) -> None:
        if self._student_hook_handle is not None:
            self._student_hook_handle.remove()
            self._student_hook_handle = None
        if self._teacher_pre_hook_handle is not None:
            self._teacher_pre_hook_handle.remove()
            self._teacher_pre_hook_handle = None
        super().close()


def main() -> None:
    if bool(base.DISTILL_ARGS.diagnose_only):
        raise RuntimeError(
            "This entry point is a training experiment, not diagnose-only."
        )

    trainer = OracleSelectiveCDFTrainer()
    try:
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
