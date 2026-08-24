"""Utilities for P0-B official-AP oracle-hybrid CDF evaluation.

The P0-B protocol compares four CDF variants while preserving the deployed
RGB student's query centers, selected views, and depth-wise width predictions:

* ``student``: the unmodified student CDF.
* ``teacher_full``: the exact-query teacher CDF on every angle/depth bin.
* ``teacher_common``: teacher CDF only on student/teacher common-valid bins.
* ``oracle_hybrid``: teacher CDF only for teacher-better queries and only on
  common-valid bins.

The oracle gate exactly matches the Priority-1 local diagnosis: teacher and
student are evaluated against the *student* CDF target with threshold-balanced
BCE on common-valid support.  This module is intentionally independent of the
training code so it can be used in checkpoint-only inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import torch
import torch.nn.functional as F


P0B_VARIANTS: Tuple[str, ...] = (
    "student",
    "teacher_full",
    "teacher_common",
    "oracle_hybrid",
)


@dataclass(frozen=True)
class P0BVariantBundle:
    """Constructed CDF variants and their exact oracle-gate state."""

    logits_btqad: Dict[str, torch.Tensor]
    common_valid_bqad: torch.Tensor
    gate_valid_bq: torch.Tensor
    teacher_better_bq: torch.Tensor
    student_common_loss_bq: torch.Tensor
    teacher_common_loss_bq: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def _require_tensor(end_points: Mapping[str, Any], key: str) -> torch.Tensor:
    value = end_points.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Required P0-B endpoint {key!r} is missing or is not a tensor.")
    return value


def _cdf_btqad_to_bqadt(logits: torch.Tensor) -> torch.Tensor:
    """Convert CDF logits [B,T,Q,A,D] to [B,Q,A,D,T]."""
    if logits.dim() != 5:
        raise ValueError(
            "CDF logits must be [B,T,Q,A,D], got "
            f"{tuple(logits.shape)}."
        )
    return logits.permute(0, 2, 3, 4, 1).contiguous()


def _cdf_bqadt_to_btqad(logits: torch.Tensor) -> torch.Tensor:
    """Convert CDF logits [B,Q,A,D,T] to [B,T,Q,A,D]."""
    if logits.dim() != 5:
        raise ValueError(
            "CDF logits must be [B,Q,A,D,T], got "
            f"{tuple(logits.shape)}."
        )
    return logits.permute(0, 4, 1, 2, 3).contiguous()


def _cdf_bins_to_target(
    bins_bqad: torch.Tensor,
    num_thresholds: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand compact CDF bins [B,Q,A,D] to cumulative targets [B,Q,A,D,T]."""
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


def _per_query_threshold_balanced_bce(
    logits_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Priority-1 threshold-balanced BCE decomposed per query.

    For each CDF threshold, positive and negative elements receive equal total
    weight when both classes exist.  The resulting globally shared weights are
    then normalized within each query so teacher and student are compared under
    the same task-consistent objective.
    """
    if logits_bqadt.shape != target_bqadt.shape:
        raise ValueError(
            "Balanced BCE logits/target shape mismatch: "
            f"{tuple(logits_bqadt.shape)} vs {tuple(target_bqadt.shape)}."
        )
    if valid_bqad.shape != logits_bqadt.shape[:-1]:
        raise ValueError(
            "Balanced BCE valid shape mismatch: "
            f"valid={tuple(valid_bqad.shape)}, logits={tuple(logits_bqadt.shape)}."
        )

    loss = F.binary_cross_entropy_with_logits(
        logits_bqadt,
        target_bqadt,
        reduction="none",
    )
    valid = valid_bqad.bool()
    weights = torch.zeros_like(loss)
    num_thresholds = int(loss.shape[-1])

    for threshold_idx in range(num_thresholds):
        target_t = target_bqadt[..., threshold_idx]
        pos = valid & (target_t > 0.5)
        neg = valid & (~pos)
        has_pos = bool(pos.any())
        has_neg = bool(neg.any())
        if has_pos and has_neg:
            weights[..., threshold_idx][pos] = 0.5 / pos.sum().float()
            weights[..., threshold_idx][neg] = 0.5 / neg.sum().float()
        elif has_pos:
            weights[..., threshold_idx][pos] = 1.0 / pos.sum().float()
        elif has_neg:
            weights[..., threshold_idx][neg] = 1.0 / neg.sum().float()

    weights = weights / max(num_thresholds, 1)
    query_weight = weights.sum(dim=(-1, -2, -3))
    query_loss = (loss * weights).sum(dim=(-1, -2, -3))
    query_valid = query_weight > 0
    query_loss = query_loss / query_weight.clamp_min(1.0e-12)
    return query_loss, query_valid


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=torch.bool)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    if bool(mask.any()):
        return value[mask].float().mean().reshape(())
    return zero.reshape(())


def extract_student_query_contract(
    student_end_points: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Extract and validate the single-view exact-query student contract."""
    required = (
        "kview_base_token_sel_idx",
        "token_sel_idx",
        "grasp_top_view_inds",
    )
    contract: Dict[str, torch.Tensor] = {}
    for key in required:
        value = _require_tensor(student_end_points, key)
        if value.dim() != 2:
            raise ValueError(
                f"student_end_points[{key!r}] must be rank 2, got "
                f"{tuple(value.shape)}."
            )
        contract[key] = value.detach().long()

    base = contract["kview_base_token_sel_idx"]
    query = contract["token_sel_idx"]
    view = contract["grasp_top_view_inds"]
    if query.shape != base.shape or view.shape != base.shape:
        raise RuntimeError(
            "P0-B currently supports one selected view per image-FPS seed "
            f"(Q=M). Got base={tuple(base.shape)}, query={tuple(query.shape)}, "
            f"view={tuple(view.shape)}. Disable Top-K view inference."
        )
    if not torch.equal(query, base):
        raise RuntimeError(
            "P0-B exact-query mode requires token_sel_idx to equal "
            "kview_base_token_sel_idx for every query."
        )
    return contract


def build_exact_teacher_input(
    pristine_batch: Mapping[str, Any],
    student_end_points: Mapping[str, Any],
    *,
    force_process_grasp_labels: bool = True,
) -> Dict[str, Any]:
    """Build a clean-depth teacher input on the student's exact seed/view queries."""
    contract = extract_student_query_contract(student_end_points)
    teacher_input: MutableMapping[str, Any] = dict(pristine_batch)
    teacher_input.pop("image_fps_seed_idx_override", None)
    teacher_input.pop("oracle_view_inds_override", None)
    teacher_input["image_fps_seed_idx_override"] = contract[
        "kview_base_token_sel_idx"
    ]
    teacher_input["oracle_view_inds_override"] = contract[
        "grasp_top_view_inds"
    ]
    teacher_input["cva_force_process_grasp_labels"] = bool(
        force_process_grasp_labels
    )
    teacher_input["cva_compute_diagnostics"] = False
    teacher_input["geometry_compute_diagnostics"] = False
    teacher_input["cva_export_angle_feature"] = False
    return dict(teacher_input)


def assert_exact_teacher_output(
    student_end_points: Mapping[str, Any],
    teacher_end_points: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Fail unless teacher and student use identical base seeds, pixels, and views."""
    contract = extract_student_query_contract(student_end_points)
    ratios: Dict[str, torch.Tensor] = {}
    for key, label in (
        ("kview_base_token_sel_idx", "base seed"),
        ("token_sel_idx", "query pixel"),
        ("grasp_top_view_inds", "selected view"),
    ):
        student = contract[key]
        teacher = _require_tensor(teacher_end_points, key).to(
            device=student.device,
            dtype=student.dtype,
        )
        if teacher.shape != student.shape:
            raise RuntimeError(
                f"P0-B exact teacher/student {label} shape mismatch: "
                f"{tuple(student.shape)} vs {tuple(teacher.shape)}."
            )
        equal = student == teacher
        if not bool(equal.all()):
            mismatch = float((~equal).float().mean().item())
            raise RuntimeError(
                f"P0-B exact teacher/student {label} mismatch at "
                f"{100.0 * mismatch:.4f}% of entries."
            )
        ratios[key] = equal.float().mean().reshape(())
    return ratios


def _selected_target_utility(
    logits_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = torch.sigmoid(logits_bqadt)
    predicted_utility = probability.mean(dim=-1)
    target_utility = target_bqadt.mean(dim=-1)
    valid = valid_bqad.bool()
    query_valid = valid.flatten(2).any(dim=-1)
    selected_index = predicted_utility.masked_fill(~valid, -1.0e6).flatten(2).argmax(-1)
    selected_target = target_utility.flatten(2).gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    oracle_target = target_utility.masked_fill(~valid, -1.0).flatten(2).max(-1).values
    regret = (oracle_target.clamp_min(0.0) - selected_target).clamp_min(0.0)
    return selected_target, regret, query_valid


@torch.no_grad()
def build_p0b_cdf_variants(
    student_end_points: Mapping[str, Any],
    exact_view_teacher_end_points: Mapping[str, Any],
    *,
    teacher_better_margin: float = 0.0,
) -> P0BVariantBundle:
    """Construct the four P0-B CDF variants using the Priority-1 oracle gate."""
    margin = float(teacher_better_margin)
    if margin < 0.0:
        raise ValueError("teacher_better_margin must be non-negative.")

    s_btqad = _require_tensor(
        student_end_points, "grasp_cdf_pred_angle_depth"
    ).float()
    t_btqad = _require_tensor(
        exact_view_teacher_end_points, "grasp_cdf_pred_angle_depth"
    ).to(s_btqad).float()
    if s_btqad.shape != t_btqad.shape:
        raise ValueError(
            "P0-B exact-view student/teacher CDF shapes differ: "
            f"{tuple(s_btqad.shape)} vs {tuple(t_btqad.shape)}."
        )

    s_logits = _cdf_btqad_to_bqadt(s_btqad)
    t_logits = _cdf_btqad_to_bqadt(t_btqad)
    expected_bqad = s_logits.shape[:-1]

    student_bins = _require_tensor(
        student_end_points, "batch_grasp_cdf_bins_angle_depth"
    ).long().to(s_logits.device)
    student_valid = _require_tensor(
        student_end_points, "batch_grasp_cdf_valid_mask"
    ).bool().to(s_logits.device)
    teacher_valid = _require_tensor(
        exact_view_teacher_end_points, "batch_grasp_cdf_valid_mask"
    ).bool().to(s_logits.device)

    if student_bins.shape != expected_bqad:
        raise ValueError(
            "Student CDF bin shape mismatch: "
            f"got {tuple(student_bins.shape)}, expected {tuple(expected_bqad)}."
        )
    if student_valid.shape != expected_bqad or teacher_valid.shape != expected_bqad:
        raise ValueError(
            "Student/teacher CDF valid shape mismatch: "
            f"student={tuple(student_valid.shape)}, "
            f"teacher={tuple(teacher_valid.shape)}, "
            f"expected={tuple(expected_bqad)}."
        )

    student_target = _cdf_bins_to_target(
        student_bins,
        num_thresholds=int(s_logits.shape[-1]),
        dtype=s_logits.dtype,
    )
    common_valid = student_valid & teacher_valid

    s_common_loss, s_common_query = _per_query_threshold_balanced_bce(
        s_logits,
        student_target,
        common_valid,
    )
    t_common_loss, t_common_query = _per_query_threshold_balanced_bce(
        t_logits,
        student_target,
        common_valid,
    )
    gate_valid = s_common_query & t_common_query
    teacher_better = gate_valid & (
        t_common_loss + margin < s_common_loss
    )

    common_element_mask = common_valid.unsqueeze(-1)
    teacher_better_element_mask = (
        teacher_better.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        & common_element_mask
    )

    teacher_common = torch.where(common_element_mask, t_logits, s_logits)
    oracle_hybrid = torch.where(
        teacher_better_element_mask,
        t_logits,
        s_logits,
    )

    variants_bqadt = {
        "student": s_logits,
        "teacher_full": t_logits,
        "teacher_common": teacher_common,
        "oracle_hybrid": oracle_hybrid,
    }
    variants_btqad = {
        name: _cdf_bqadt_to_btqad(logits)
        for name, logits in variants_bqadt.items()
    }

    zero = s_logits.new_zeros(())
    diagnostics: Dict[str, torch.Tensor] = {
        "common_valid_element_ratio": common_valid.float().mean().reshape(()),
        "gate_valid_query_ratio": gate_valid.float().mean().reshape(()),
        "teacher_better_query_ratio": teacher_better.float().mean().reshape(()),
        "teacher_better_among_gate_valid": _masked_mean(
            teacher_better.float(), gate_valid, zero
        ),
        "teacher_common_bce_advantage": _masked_mean(
            s_common_loss - t_common_loss, gate_valid, zero
        ),
        "oracle_replaced_element_ratio": teacher_better_element_mask.float().mean().reshape(()),
        "teacher_better_margin": zero.new_tensor(margin).reshape(()),
    }

    for name, logits in variants_bqadt.items():
        balanced_loss, balanced_valid = _per_query_threshold_balanced_bce(
            logits,
            student_target,
            student_valid,
        )
        selected_target, regret, query_valid = _selected_target_utility(
            logits,
            student_target,
            student_valid,
        )
        diagnostics[f"{name}_balanced_bce"] = _masked_mean(
            balanced_loss, balanced_valid, zero
        )
        diagnostics[f"{name}_selected_target_utility"] = _masked_mean(
            selected_target, query_valid, zero
        )
        diagnostics[f"{name}_selection_regret"] = _masked_mean(
            regret, query_valid, zero
        )

    return P0BVariantBundle(
        logits_btqad=variants_btqad,
        common_valid_bqad=common_valid,
        gate_valid_bq=gate_valid,
        teacher_better_bq=teacher_better,
        student_common_loss_bq=s_common_loss,
        teacher_common_loss_bq=t_common_loss,
        diagnostics=diagnostics,
    )


def make_variant_end_points(
    student_end_points: Mapping[str, Any],
    variant_logits_btqad: torch.Tensor,
) -> Dict[str, Any]:
    """Shallow-copy student endpoints and replace only the CDF logits."""
    original = _require_tensor(
        student_end_points, "grasp_cdf_pred_angle_depth"
    )
    logits = variant_logits_btqad.to(
        device=original.device,
        dtype=original.dtype,
    )
    if logits.shape != original.shape:
        raise ValueError(
            "P0-B variant CDF shape mismatch: "
            f"{tuple(logits.shape)} vs {tuple(original.shape)}."
        )
    variant_end_points = dict(student_end_points)
    variant_end_points["grasp_cdf_pred_angle_depth"] = logits
    return variant_end_points
