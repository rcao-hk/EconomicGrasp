#!/usr/bin/env python3
"""Priority-2 exact-student-view, View-only Stage-2 distillation.

This entry point reuses the current GitHub Stage-2 trainer unchanged and
installs a runtime adapter on its frozen teacher.  The adapter validates exact
student seed/pixel/view correspondence and preserves the teacher's unmodified
dense view field for View KD.  All other KD branches must be disabled by the
launcher.
"""

from __future__ import annotations

import math

import train_cva_distill_ddp as base
from models.privileged_kd_priority12 import (
    install_exact_student_view_teacher_forward,
)


def _require_close(name: str, value: float, expected: float, tol: float = 1e-12) -> None:
    if not math.isfinite(float(value)) or abs(float(value) - float(expected)) > tol:
        raise RuntimeError(
            f"Priority-2 View-only protocol requires {name}={expected}, "
            f"got {value}."
        )


def _validate_view_only_protocol(trainer: base.Trainer) -> None:
    if trainer.distill_stage != 2 or trainer.teacher is None:
        raise RuntimeError(
            "Priority-2 requires --distill_stage 2 and a valid Stage-0 teacher."
        )
    cfg = trainer.distill_config
    _require_close("KD_OBJECTNESS_WEIGHT", cfg.objectness_weight, 0.0)
    _require_close("KD_GRASPNESS_WEIGHT", cfg.graspness_weight, 0.0)
    _require_close("KD_DEPTH_WEIGHT", cfg.depth_weight, 0.0)
    _require_close("KD_CDF_WEIGHT", cfg.cdf_weight, 0.0)
    _require_close("KD_WIDTH_WEIGHT", cfg.width_weight, 0.0)
    if not math.isfinite(float(cfg.view_weight)) or float(cfg.view_weight) <= 0.0:
        raise RuntimeError(
            "Priority-2 View-only KD requires KD_VIEW_WEIGHT > 0, got "
            f"{cfg.view_weight}."
        )


def main() -> None:
    trainer = base.Trainer()
    try:
        _validate_view_only_protocol(trainer)
        install_exact_student_view_teacher_forward(
            trainer,
            preserve_raw_view_field=True,
            print_once=True,
        )
        trainer.log_string(
            "[PRIORITY12][P2] exact student-view teacher forward enabled; "
            "KD branches: view=on, objectness/graspness/depth/CDF/width=off."
        )
        trainer.log_string(
            "[PRIORITY12][P2] the frozen teacher executes two sequential "
            "passes per student batch: seed-aligned raw-view field, then "
            "exact-view downstream query."
        )
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
