"""Exact-query uniform CDF-only distillation control.

This module is the one-factor control for
``Exact-Query Oracle-Selective CDF Distillation``.

Both variants use the same:
  * fresh RGB student initialization;
  * frozen Stage-0 clean-geometry teacher;
  * student-selected image-FPS seeds;
  * exact student-selected teacher views;
  * teacher label matching;
  * student/teacher common-valid CDF support;
  * ordinary (unbalanced) soft-target CDF BCE.

The only difference is query selection:
  * oracle-selective: distil only common-valid queries where teacher BCE is
    lower than student BCE on the student query's GT CDF;
  * this control: distil every common-valid query.

No feature, view, width, depth, objectness, or graspness KD is introduced.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F

from .oracle_selective_cdf_kd import (
    build_oracle_selective_cdf_gate,
    extract_oracle_selective_teacher_targets,
)


# Re-export the exact same teacher target extractor used by the selective run.
extract_uniform_exact_query_teacher_targets = (
    extract_oracle_selective_teacher_targets
)


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise KeyError(
            f"Exact-query uniform CDF KD requires tensor endpoint {key!r}."
        )
    return value


def _zero_from_student(student_end_points: Mapping[str, Any]) -> torch.Tensor:
    return _require_tensor(
        student_end_points,
        "grasp_cdf_pred_angle_depth",
    ).sum() * 0.0


def _masked_scalar_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=torch.bool)
    if bool(mask.any()):
        return value[mask].float().mean()
    return zero.detach()


def compute_uniform_exact_query_cdf_distillation_loss(
    student_end_points: Dict[str, Any],
    teacher_targets: Mapping[str, torch.Tensor],
    config: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Distil teacher CDF on every exact-query common-valid query.

    The function intentionally has the same call signature as the repository's
    ``compute_output_distillation_loss`` and the oracle-selective replacement.
    It reuses ``build_oracle_selective_cdf_gate`` only to guarantee identical:

      * exact seed/pixel/view assertions;
      * student-GT CDF target construction;
      * ordinary per-query BCE diagnostics;
      * common-valid support construction.

    The teacher-better result is logged but never used by the loss. The selected
    query mask is exactly ``common_query_bq``.
    """
    zero = _zero_from_student(student_end_points)

    temperature = float(getattr(config, "temperature", 1.0))
    if abs(temperature - 1.0) > 1.0e-12:
        raise ValueError(
            "Minimal exact-query uniform CDF KD fixes temperature=1.0; got "
            f"{temperature}."
        )

    gate = build_oracle_selective_cdf_gate(
        student_end_points,
        teacher_targets,
    )

    student_logits = _require_tensor(
        student_end_points,
        "grasp_cdf_pred_angle_depth",
    )
    teacher_logits = _require_tensor(
        teacher_targets,
        "grasp_cdf_pred_angle_depth",
    ).to(student_logits).detach()
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Exact-query uniform CDF shape mismatch: "
            f"student={tuple(student_logits.shape)}, "
            f"teacher={tuple(teacher_logits.shape)}."
        )

    teacher_prob = torch.sigmoid(teacher_logits)
    kd_map = F.binary_cross_entropy_with_logits(
        student_logits,
        teacher_prob,
        reduction="none",
    )

    # This is the only substantive difference from oracle-selective KD:
    # select every common-valid query instead of teacher-better queries only.
    selected_query = gate["common_query_bq"]
    common_valid = gate["common_valid_bqad"]
    selected_mask = (
        selected_query[:, None, :, None, None]
        & common_valid[:, None, :, :, :]
    ).expand_as(kd_map)

    teacher_entropy_map = F.binary_cross_entropy_with_logits(
        teacher_logits,
        teacher_prob,
        reduction="none",
    )
    if bool(selected_mask.any()):
        cdf_kd_loss = kd_map[selected_mask].mean()
        cdf_teacher_entropy = teacher_entropy_map[selected_mask].mean()
        cdf_excess = (cdf_kd_loss - cdf_teacher_entropy).clamp_min(0.0)
    else:
        # DDP-safe differentiable zero for batches without common-valid queries.
        cdf_kd_loss = zero
        cdf_teacher_entropy = zero.detach()
        cdf_excess = zero.detach()

    overall_weight = float(getattr(config, "overall_weight", 1.0))
    cdf_weight = float(getattr(config, "cdf_weight", 1.0))
    total = overall_weight * cdf_weight * cdf_kd_loss

    # Preserve the base trainer's scalar-key contract.
    student_end_points["B: KD Objectness Loss"] = zero
    student_end_points["B: KD Graspness Loss"] = zero
    student_end_points["B: KD Depth Loss"] = zero
    student_end_points["B: KD View Loss"] = zero
    student_end_points["B: KD CDF Loss"] = cdf_kd_loss
    student_end_points["B: KD CDF Excess"] = cdf_excess
    student_end_points["B: KD CDF Teacher Entropy"] = cdf_teacher_entropy
    student_end_points["B: KD Width Loss"] = zero
    student_end_points["A: Distill Loss"] = total

    with torch.no_grad():
        common_query = gate["common_query_bq"]
        teacher_better = gate["teacher_better_bq"]
        student_bce = gate["student_bce_bq"]
        teacher_bce = gate["teacher_bce_bq"]
        selected_count_per_sample = selected_query.float().sum(dim=1)

        student_end_points["D: UniformCDF exact seed ratio"] = gate[
            "exact_seed_ratio"
        ].detach().reshape(())
        student_end_points["D: UniformCDF exact pixel ratio"] = gate[
            "exact_pixel_ratio"
        ].detach().reshape(())
        student_end_points["D: UniformCDF exact view ratio"] = gate[
            "exact_view_ratio"
        ].detach().reshape(())
        student_end_points["D: UniformCDF student valid query ratio"] = (
            gate["student_valid_bqad"].any(dim=-1).any(dim=-1).float().mean()
        ).reshape(())
        student_end_points["D: UniformCDF teacher valid query ratio"] = (
            gate["teacher_valid_bqad"].any(dim=-1).any(dim=-1).float().mean()
        ).reshape(())
        student_end_points["D: UniformCDF common query ratio"] = (
            common_query.float().mean().reshape(())
        )
        student_end_points["D: UniformCDF selected query ratio"] = (
            selected_query.float().mean().reshape(())
        )
        student_end_points["D: UniformCDF selected query count"] = (
            selected_count_per_sample.mean().reshape(())
        )
        student_end_points["D: UniformCDF selected element ratio"] = (
            selected_mask.float().mean().reshape(())
        )
        student_end_points[
            "D: UniformCDF teacher better among common"
        ] = _masked_scalar_mean(
            teacher_better.float(),
            common_query,
            zero,
        ).reshape(())
        student_end_points["D: UniformCDF student BCE common"] = (
            _masked_scalar_mean(student_bce, common_query, zero).reshape(())
        )
        student_end_points["D: UniformCDF teacher BCE common"] = (
            _masked_scalar_mean(teacher_bce, common_query, zero).reshape(())
        )
        student_end_points["D: UniformCDF teacher-student BCE delta"] = (
            _masked_scalar_mean(
                teacher_bce - student_bce,
                common_query,
                zero,
            ).reshape(())
        )
        student_end_points["D: UniformCDF common operation ratio"] = (
            common_valid.float().mean().reshape(())
        )
        student_end_points["D: UniformCDF teacher probability selected"] = (
            _masked_scalar_mean(
                teacher_prob,
                selected_mask,
                zero,
            ).reshape(())
        )
        student_end_points["D: UniformCDF ordinary BCE"] = (
            zero.detach().new_tensor(1.0).reshape(())
        )
        student_end_points["D: UniformCDF balanced loss"] = (
            zero.detach().new_tensor(0.0).reshape(())
        )
        student_end_points["D: UniformCDF uses teacher-better gate"] = (
            zero.detach().new_tensor(0.0).reshape(())
        )
        student_end_points["D: UniformCDF uses all common queries"] = (
            zero.detach().new_tensor(1.0).reshape(())
        )

    return total, student_end_points
