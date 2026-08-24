"""Exact-query oracle-selective CDF distillation.

This module implements the smallest training-time change needed to test whether
negative transfer in Stage 2 is caused by teacher-worse grasp queries.

Protocol
--------
1. The RGB student selects image-FPS seeds and grasp views normally.
2. The frozen clean-geometry teacher is evaluated at those exact seeds/views.
3. Teacher and student are compared against the *student query's* GT CDF using
   ordinary (unbalanced) per-query BCE on their common-valid support.
4. Soft CDF distillation is applied only where the teacher has lower BCE.

No feature KD, view KD, width KD, depth KD, loss balancing, margin, warm-up,
learned selector, or inference-time component is introduced here.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F


_TEACHER_TARGET_KEYS = (
    "grasp_cdf_pred_angle_depth",
    "batch_grasp_cdf_bins_angle_depth",
    "batch_grasp_cdf_valid_mask",
    "kview_base_token_sel_idx",
    "token_sel_idx",
    "grasp_top_view_inds",
    "grasp_top_view_xyz",
)


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise KeyError(
            f"Exact-query oracle-selective CDF KD requires tensor endpoint "
            f"{key!r}."
        )
    return value


def _zero_from_student(student_end_points: Mapping[str, Any]) -> torch.Tensor:
    cdf = _require_tensor(
        student_end_points,
        "grasp_cdf_pred_angle_depth",
    )
    return cdf.sum() * 0.0


def extract_oracle_selective_teacher_targets(
    teacher_end_points: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Detach exact-view teacher outputs and matched CDF labels.

    The teacher forward must be run with both:
      * image_fps_seed_idx_override = student image-FPS seed indices;
      * oracle_view_inds_override   = student selected view indices;

    and with cva_force_process_grasp_labels=True so the teacher valid mask is
    available for common-support filtering.
    """
    targets: Dict[str, torch.Tensor] = {}
    missing = []
    for key in _TEACHER_TARGET_KEYS:
        value = teacher_end_points.get(key)
        if value is None:
            missing.append(key)
            continue
        if not torch.is_tensor(value):
            raise TypeError(
                f"Teacher endpoint {key!r} must be a tensor, got "
                f"{type(value).__name__}."
            )
        targets[key] = value.detach()
    if missing:
        raise KeyError(
            "Exact-query teacher forward is missing required endpoint(s): "
            + ", ".join(missing)
        )
    return targets


def _cdf_logits_to_bqadt(logits_btqad: torch.Tensor) -> torch.Tensor:
    if logits_btqad.dim() != 5:
        raise ValueError(
            "CDF logits must be [B,T,Q,A,D], got "
            f"{tuple(logits_btqad.shape)}."
        )
    return logits_btqad.permute(0, 2, 3, 4, 1).contiguous()


def _cdf_bins_to_target(
    bins_bqad: torch.Tensor,
    num_thresholds: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand compact CDF bins [B,Q,A,D] to [B,Q,A,D,T]."""
    if bins_bqad.dim() != 4:
        raise ValueError(
            "CDF bins must be [B,Q,A,D], got "
            f"{tuple(bins_bqad.shape)}."
        )
    bins = bins_bqad.long()
    threshold_ids = torch.arange(
        int(num_thresholds),
        device=bins.device,
        dtype=bins.dtype,
    )
    return (
        (bins.unsqueeze(-1) > 0)
        & (threshold_ids >= bins.unsqueeze(-1) - 1)
    ).to(dtype=dtype)


def _per_query_ordinary_bce(
    logits_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ordinary BCE averaged independently for every [B,Q] query.

    There is no positive/negative or threshold balancing. This intentionally
    matches the repository's current supervised CDF setting, which calls
    compute_cva_cdf_loss(..., balanced=False).
    """
    if logits_bqadt.shape != target_bqadt.shape:
        raise ValueError(
            "CDF logits/target shapes differ: "
            f"{tuple(logits_bqadt.shape)} vs {tuple(target_bqadt.shape)}."
        )
    if valid_bqad.shape != logits_bqadt.shape[:-1]:
        raise ValueError(
            "CDF valid mask must match [B,Q,A,D]: "
            f"valid={tuple(valid_bqad.shape)}, "
            f"logits={tuple(logits_bqadt.shape)}."
        )

    loss_map = F.binary_cross_entropy_with_logits(
        logits_bqadt,
        target_bqadt,
        reduction="none",
    )
    element_mask = valid_bqad.bool().unsqueeze(-1).expand_as(loss_map)
    element_count = element_mask.sum(dim=(-1, -2, -3)).to(loss_map.dtype)
    loss_sum = (loss_map * element_mask.to(loss_map.dtype)).sum(
        dim=(-1, -2, -3)
    )
    query_valid = element_count > 0
    per_query = loss_sum / element_count.clamp_min(1.0)
    return per_query, query_valid


def _assert_exact_queries(
    student_end_points: Mapping[str, Any],
    teacher_targets: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Fail fast unless teacher and student use identical seed/pixel/view ids."""
    specs = (
        ("seed", "kview_base_token_sel_idx"),
        ("pixel", "token_sel_idx"),
        ("view", "grasp_top_view_inds"),
    )
    ratios: Dict[str, torch.Tensor] = {}
    for name, key in specs:
        student_value = _require_tensor(student_end_points, key).long()
        teacher_value = _require_tensor(teacher_targets, key).to(
            device=student_value.device,
            dtype=torch.long,
        )
        if student_value.shape != teacher_value.shape:
            raise RuntimeError(
                f"Exact-query {name} shape mismatch for {key}: "
                f"student={tuple(student_value.shape)}, "
                f"teacher={tuple(teacher_value.shape)}."
            )
        equal = student_value == teacher_value
        ratio = equal.float().mean().reshape(())
        ratios[name] = ratio
        if not bool(equal.all()):
            mismatch = float((~equal).float().mean().detach().item())
            raise RuntimeError(
                f"Exact-query oracle-selective KD requires identical {name} "
                f"indices, but {100.0 * mismatch:.4f}% differ."
            )
    return ratios


@torch.no_grad()
def build_oracle_selective_cdf_gate(
    student_end_points: Mapping[str, Any],
    teacher_targets: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Build the GT-based teacher-better query gate with ordinary BCE.

    Teacher and student are both evaluated against the student query's GT CDF.
    A query is selected only when:
      1. student and teacher have non-empty common-valid support; and
      2. the teacher's ordinary per-query BCE is strictly lower.
    """
    exact_ratios = _assert_exact_queries(
        student_end_points,
        teacher_targets,
    )

    student_logits = _cdf_logits_to_bqadt(
        _require_tensor(
            student_end_points,
            "grasp_cdf_pred_angle_depth",
        ).detach().float()
    )
    teacher_logits = _cdf_logits_to_bqadt(
        _require_tensor(
            teacher_targets,
            "grasp_cdf_pred_angle_depth",
        ).to(student_logits).detach().float()
    )
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Exact-query student/teacher CDF shapes differ: "
            f"{tuple(student_logits.shape)} vs "
            f"{tuple(teacher_logits.shape)}."
        )

    num_thresholds = int(student_logits.shape[-1])
    student_bins = _require_tensor(
        student_end_points,
        "batch_grasp_cdf_bins_angle_depth",
    ).to(device=student_logits.device, dtype=torch.long)
    student_valid = _require_tensor(
        student_end_points,
        "batch_grasp_cdf_valid_mask",
    ).to(device=student_logits.device, dtype=torch.bool)
    teacher_valid = _require_tensor(
        teacher_targets,
        "batch_grasp_cdf_valid_mask",
    ).to(device=student_logits.device, dtype=torch.bool)

    if not (
        student_bins.shape
        == student_valid.shape
        == teacher_valid.shape
        == student_logits.shape[:-1]
    ):
        raise ValueError(
            "Student bins/valid and teacher valid must all be [B,Q,A,D]: "
            f"bins={tuple(student_bins.shape)}, "
            f"student_valid={tuple(student_valid.shape)}, "
            f"teacher_valid={tuple(teacher_valid.shape)}, "
            f"cdf={tuple(student_logits.shape)}."
        )

    student_target = _cdf_bins_to_target(
        student_bins,
        num_thresholds,
        dtype=student_logits.dtype,
    )
    common_valid = student_valid & teacher_valid

    student_bce, student_query_valid = _per_query_ordinary_bce(
        student_logits,
        student_target,
        common_valid,
    )
    teacher_bce, teacher_query_valid = _per_query_ordinary_bce(
        teacher_logits,
        student_target,
        common_valid,
    )
    common_query = student_query_valid & teacher_query_valid
    teacher_better = common_query & (teacher_bce < student_bce)

    return {
        "student_target_bqadt": student_target,
        "student_valid_bqad": student_valid,
        "teacher_valid_bqad": teacher_valid,
        "common_valid_bqad": common_valid,
        "common_query_bq": common_query,
        "teacher_better_bq": teacher_better,
        "student_bce_bq": student_bce,
        "teacher_bce_bq": teacher_bce,
        "exact_seed_ratio": exact_ratios["seed"],
        "exact_pixel_ratio": exact_ratios["pixel"],
        "exact_view_ratio": exact_ratios["view"],
    }


def _masked_scalar_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=torch.bool)
    if bool(mask.any()):
        return value[mask].float().mean()
    return zero.detach()


def compute_oracle_selective_cdf_distillation_loss(
    student_end_points: Dict[str, Any],
    teacher_targets: Mapping[str, torch.Tensor],
    config: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute exact-query, teacher-better, ordinary soft CDF distillation.

    This function intentionally has the same call signature as the repository's
    compute_output_distillation_loss so it can replace only the Stage-2 KD term
    while leaving the supervised training loop unchanged.
    """
    zero = _zero_from_student(student_end_points)

    temperature = float(getattr(config, "temperature", 1.0))
    if abs(temperature - 1.0) > 1.0e-12:
        raise ValueError(
            "Minimal oracle-selective CDF KD fixes temperature=1.0; got "
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
            "Selective KD CDF shape mismatch: "
            f"student={tuple(student_logits.shape)}, "
            f"teacher={tuple(teacher_logits.shape)}."
        )

    teacher_prob = torch.sigmoid(teacher_logits)
    kd_map = F.binary_cross_entropy_with_logits(
        student_logits,
        teacher_prob,
        reduction="none",
    )

    # [B,Q] and [B,Q,A,D] -> [B,T,Q,A,D].
    selected_query = gate["teacher_better_bq"]
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
        # Preserve a valid differentiable scalar on every DDP rank.
        cdf_kd_loss = zero
        cdf_teacher_entropy = zero.detach()
        cdf_excess = zero.detach()

    overall_weight = float(getattr(config, "overall_weight", 1.0))
    cdf_weight = float(getattr(config, "cdf_weight", 1.0))
    total = overall_weight * cdf_weight * cdf_kd_loss

    # Keep the repository's scalar key contract while explicitly zeroing every
    # branch that this minimal variant does not use.
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
        selected_count_per_sample = teacher_better.float().sum(dim=1)
        selected_advantage = student_bce - teacher_bce

        student_end_points["D: OracleSel exact seed ratio"] = gate[
            "exact_seed_ratio"
        ].detach().reshape(())
        student_end_points["D: OracleSel exact pixel ratio"] = gate[
            "exact_pixel_ratio"
        ].detach().reshape(())
        student_end_points["D: OracleSel exact view ratio"] = gate[
            "exact_view_ratio"
        ].detach().reshape(())
        student_end_points["D: OracleSel student valid query ratio"] = (
            gate["student_valid_bqad"].any(dim=-1).any(dim=-1).float().mean()
        ).reshape(())
        student_end_points["D: OracleSel teacher valid query ratio"] = (
            gate["teacher_valid_bqad"].any(dim=-1).any(dim=-1).float().mean()
        ).reshape(())
        student_end_points["D: OracleSel common query ratio"] = (
            common_query.float().mean().reshape(())
        )
        student_end_points["D: OracleSel selected query ratio"] = (
            teacher_better.float().mean().reshape(())
        )
        student_end_points[
            "D: OracleSel teacher better among common"
        ] = _masked_scalar_mean(
            teacher_better.float(),
            common_query,
            zero,
        ).reshape(())
        student_end_points["D: OracleSel selected query count"] = (
            selected_count_per_sample.mean().reshape(())
        )
        student_end_points["D: OracleSel selected element ratio"] = (
            selected_mask.float().mean().reshape(())
        )
        student_end_points["D: OracleSel student BCE common"] = (
            _masked_scalar_mean(student_bce, common_query, zero).reshape(())
        )
        student_end_points["D: OracleSel teacher BCE common"] = (
            _masked_scalar_mean(teacher_bce, common_query, zero).reshape(())
        )
        student_end_points["D: OracleSel student BCE selected"] = (
            _masked_scalar_mean(student_bce, teacher_better, zero).reshape(())
        )
        student_end_points["D: OracleSel teacher BCE selected"] = (
            _masked_scalar_mean(teacher_bce, teacher_better, zero).reshape(())
        )
        student_end_points[
            "D: OracleSel BCE advantage selected"
        ] = _masked_scalar_mean(
            selected_advantage,
            teacher_better,
            zero,
        ).reshape(())
        student_end_points["D: OracleSel common operation ratio"] = (
            common_valid.float().mean().reshape(())
        )
        student_end_points["D: OracleSel teacher probability selected"] = (
            _masked_scalar_mean(
                teacher_prob,
                selected_mask,
                zero,
            ).reshape(())
        )
        student_end_points["D: OracleSel ordinary BCE gate"] = (
            zero.detach().new_tensor(1.0).reshape(())
        )
        student_end_points["D: OracleSel balanced gate"] = (
            zero.detach().new_tensor(0.0).reshape(())
        )

    return total, student_end_points
