"""Pure tensor diagnostics for P0 pose-space CDF transport.

P0 asks whether the teacher/student physical-center mismatch can be absorbed by
EconomicGrasp's discrete grasp-depth axis without invoking the GraspNet/Dex-Net
evaluator.  The diagnostic is deliberately read-only: it never changes model
outputs, labels, or checkpoints.

The sign convention follows ``process_grasp_labels_depth_cls_compensated`` in
``utils/label_generation.py``.  For a student center P, a matched clean point Q,
and approach axis a, ``delta = dot(Q - P, a)`` and the transported grasp depth is
``d_student = d_teacher + delta``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class P0TransportConfig:
    point_match_thresh_m: float = 0.005
    tolerated_parallel_m: float = 0.03
    depth_start_m: float = 0.01
    depth_interval_m: float = 0.01
    num_depth: int = 4
    eps: float = 1.0e-8

    def validate(self) -> None:
        if self.point_match_thresh_m <= 0:
            raise ValueError("point_match_thresh_m must be positive")
        if self.tolerated_parallel_m <= 0:
            raise ValueError("tolerated_parallel_m must be positive")
        if self.depth_interval_m <= 0:
            raise ValueError("depth_interval_m must be positive")
        if self.num_depth <= 0:
            raise ValueError("num_depth must be positive")


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"P0 diagnostic requires tensor endpoint {key!r}.")
    return value


def _get_center(mapping: Mapping[str, Any]) -> torch.Tensor:
    for key in ("token_sel_xyz", "xyz_graspable"):
        value = mapping.get(key)
        if torch.is_tensor(value):
            if value.dim() != 3 or value.shape[-1] != 3:
                raise ValueError(f"{key} must be [B,Q,3], got {tuple(value.shape)}")
            return value
    raise KeyError("P0 diagnostic requires token_sel_xyz or xyz_graspable.")


def _cdf_logits_to_bqadt(logits_btqad: torch.Tensor) -> torch.Tensor:
    if logits_btqad.dim() != 5:
        raise ValueError(
            "grasp_cdf_pred_angle_depth must be [B,T,Q,A,D], got "
            f"{tuple(logits_btqad.shape)}"
        )
    return logits_btqad.permute(0, 2, 3, 4, 1).contiguous()


def cdf_bins_to_target(
    bins_bqad: torch.Tensor,
    num_thresholds: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand compact CDF bins [B,Q,A,D] into binary [B,Q,A,D,T]."""
    if bins_bqad.dim() != 4:
        raise ValueError(f"CDF bins must be [B,Q,A,D], got {tuple(bins_bqad.shape)}")
    bins = bins_bqad.long()
    threshold_ids = torch.arange(
        int(num_thresholds), device=bins.device, dtype=bins.dtype
    )
    return (
        (bins.unsqueeze(-1) > 0)
        & (threshold_ids >= bins.unsqueeze(-1) - 1)
    ).to(dtype=dtype)


def nearest_depth_shift_bins(delta_m_bq: torch.Tensor, interval_m: float) -> torch.Tensor:
    """Legacy-compatible nearest-bin shift using floor(x + 0.5)."""
    return torch.floor(delta_m_bq / float(interval_m) + 0.5).long()


def shift_depth_axis_nearest(
    values_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
    shift_bins_bq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transport independent depth-conditioned values with one signed shift/query.

    Source depth ``d`` maps to destination ``d + shift``.  Out-of-range source
    candidates are discarded rather than clipped.  This is appropriate because
    the D dimension enumerates independent grasp candidates; it is not a
    probability simplex over depth.
    """
    if values_bqadt.dim() != 5:
        raise ValueError(f"values_bqadt must be [B,Q,A,D,T], got {tuple(values_bqadt.shape)}")
    if valid_bqad.shape != values_bqadt.shape[:-1]:
        raise ValueError(
            f"valid_bqad shape {tuple(valid_bqad.shape)} != {tuple(values_bqadt.shape[:-1])}"
        )
    if shift_bins_bq.shape != values_bqadt.shape[:2]:
        raise ValueError(
            f"shift_bins_bq shape {tuple(shift_bins_bq.shape)} != BQ {tuple(values_bqadt.shape[:2])}"
        )

    B, Q, A, D, T = values_bqadt.shape
    out = torch.zeros_like(values_bqadt)
    out_valid = torch.zeros_like(valid_bqad, dtype=torch.bool)

    for dst in range(D):
        src = dst - shift_bins_bq
        in_range = (src >= 0) & (src < D)
        src_safe = src.clamp(0, D - 1)
        gather_values = src_safe[:, :, None, None, None].expand(B, Q, A, 1, T)
        gathered = torch.gather(values_bqadt, dim=3, index=gather_values).squeeze(3)
        gather_valid = src_safe[:, :, None, None].expand(B, Q, A, 1)
        gathered_valid = torch.gather(valid_bqad, dim=3, index=gather_valid).squeeze(3)
        dst_valid = gathered_valid & in_range[:, :, None]
        out[:, :, :, dst, :] = torch.where(
            dst_valid.unsqueeze(-1), gathered, torch.zeros_like(gathered)
        )
        out_valid[:, :, :, dst] = dst_valid
    return out, out_valid


def _project_error(error_bq3: torch.Tensor, axis_bq3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    parallel = (error_bq3 * axis_bq3).sum(dim=-1)
    lateral_vec = error_bq3 - parallel.unsqueeze(-1) * axis_bq3
    lateral = torch.linalg.norm(lateral_vec, dim=-1)
    return parallel, lateral


def _query_valid(valid_bqad: torch.Tensor) -> torch.Tensor:
    return valid_bqad.bool().any(dim=-1).any(dim=-1)


def _sum_count(value: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    value = value.float()
    if mask is None:
        mask = torch.ones_like(value, dtype=torch.bool)
    else:
        mask = mask.to(device=value.device, dtype=torch.bool)
        if mask.shape != value.shape:
            raise ValueError(f"mask/value shape mismatch: {tuple(mask.shape)} vs {tuple(value.shape)}")
    finite = torch.isfinite(value) & mask
    return value[finite].sum(), finite.float().sum()


def _ratio_sum_count(mask: torch.Tensor, denom: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = mask.bool()
    if denom is None:
        denom = torch.ones_like(mask, dtype=torch.bool)
    else:
        denom = denom.to(device=mask.device, dtype=torch.bool)
        if denom.shape != mask.shape:
            raise ValueError(f"ratio mask/denom mismatch: {tuple(mask.shape)} vs {tuple(denom.shape)}")
    return (mask & denom).float().sum(), denom.float().sum()


def _support_iou_sum_count(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a, b = a.bool(), b.bool()
    return (a & b).float().sum(), (a | b).float().sum()


def _masked_bce_per_query(
    prob_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prob_bqadt.shape != target_bqadt.shape:
        raise ValueError("prob and target must have identical [B,Q,A,D,T] shape")
    if valid_bqad.shape != prob_bqadt.shape[:-1]:
        raise ValueError("valid mask must have [B,Q,A,D] shape")
    prob = prob_bqadt.clamp(float(eps), 1.0 - float(eps))
    bce = F.binary_cross_entropy(prob, target_bqadt, reduction="none")
    mask = valid_bqad.unsqueeze(-1).expand_as(bce)
    count = mask.sum(dim=(-1, -2, -3)).float()
    total = (bce * mask.to(bce.dtype)).sum(dim=(-1, -2, -3))
    valid_query = count > 0
    return total / count.clamp_min(1.0), valid_query


def _assert_exact_query_contract(
    student: Mapping[str, Any], teacher: Mapping[str, Any]
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, key in (
        ("seed", "kview_base_token_sel_idx"),
        ("pixel", "token_sel_idx"),
        ("view", "grasp_top_view_inds"),
    ):
        s = _require_tensor(student, key).long()
        t = _require_tensor(teacher, key).to(device=s.device, dtype=torch.long)
        if s.shape != t.shape:
            raise RuntimeError(
                f"P0 exact-query {name} shape mismatch: student={tuple(s.shape)} teacher={tuple(t.shape)}"
            )
        equal = s == t
        out[f"exact_{name}_ratio"] = equal.float().mean()
        if not bool(equal.all()):
            raise RuntimeError(
                f"P0 requires exact student/teacher {name}; mismatch ratio="
                f"{float((~equal).float().mean().item()):.6f}"
            )
    return out


@torch.no_grad()
def compute_p0_pose_space_transport_diagnostics(
    student: Mapping[str, Any],
    teacher: Mapping[str, Any],
    config: P0TransportConfig,
) -> Dict[str, Any]:
    """Compute P0 geometry/support/CDF diagnostics for one paired batch."""
    config.validate()
    exact = _assert_exact_query_contract(student, teacher)

    center_s = _get_center(student).float()
    center_t = _get_center(teacher).to(center_s).float()
    if center_s.shape != center_t.shape:
        raise ValueError("student/teacher center shapes differ")
    B, Q, _ = center_s.shape

    view_xyz = _require_tensor(student, "grasp_top_view_xyz").to(center_s).float()
    view_rot = _require_tensor(student, "grasp_top_view_rot").to(center_s).float()
    if view_xyz.shape != center_s.shape or view_rot.shape != (B, Q, 3, 3):
        raise ValueError("unexpected selected-view endpoint shape")
    # EconomicGrasp decodes ``approaching = -grasp_top_view_xyz``; the first
    # column of grasp_top_view_rot is constructed from that same approach.
    axis = F.normalize(view_rot[..., :, 0], dim=-1)
    axis_from_view = F.normalize(-view_xyz, dim=-1)
    axis_error = torch.linalg.norm(axis - axis_from_view, dim=-1)
    if float(axis_error.max().item()) > 2.0e-4:
        raise RuntimeError(
            "grasp_top_view_rot[:, :, :, 0] is inconsistent with -grasp_top_view_xyz; "
            f"max difference={float(axis_error.max().item()):.3e}"
        )

    student_valid_bqad = _require_tensor(student, "batch_grasp_cdf_valid_mask").bool()
    teacher_valid_bqad = _require_tensor(teacher, "batch_grasp_cdf_valid_mask").to(
        device=center_s.device, dtype=torch.bool
    )
    if student_valid_bqad.shape != teacher_valid_bqad.shape:
        raise ValueError("student/teacher CDF valid-mask shapes differ")
    if student_valid_bqad.shape[:2] != (B, Q):
        raise ValueError("CDF valid masks do not match paired query dimensions")
    A, D = student_valid_bqad.shape[-2:]
    if D != int(config.num_depth):
        raise ValueError(f"config.num_depth={config.num_depth} but CDF D={D}")

    student_query_valid = _query_valid(student_valid_bqad)
    teacher_query_valid = _query_valid(teacher_valid_bqad)
    common_query = student_query_valid & teacher_query_valid

    # 1) Pure teacher/student center mismatch under the same pixel + view.
    center_error_vec = center_t - center_s
    center_error = torch.linalg.norm(center_error_vec, dim=-1)
    center_parallel, center_lateral = _project_error(center_error_vec, axis)
    center_shift_bins = nearest_depth_shift_bins(
        center_parallel, config.depth_interval_m
    )
    center_parallel_quantized = center_shift_bins.float() * float(config.depth_interval_m)
    center_parallel_residual = center_parallel - center_parallel_quantized
    center_action_residual = torch.sqrt(
        center_lateral.square() + center_parallel_residual.square()
    )

    depth_ids = torch.arange(D, device=center_s.device).view(1, 1, D)
    dst_ids = depth_ids + center_shift_bins.unsqueeze(-1)
    center_depth_retained_bqd = (dst_ids >= 0) & (dst_ids < D)
    center_depth_retained_fraction = center_depth_retained_bqd.float().mean(dim=-1)
    center_any_depth_retained = center_depth_retained_bqd.any(dim=-1)

    # 2) Existing CDF label matcher geometry: student center -> nearest clean
    # CAD-derived grasp point. This matches the legacy compensation helper.
    matched_point_s = _require_tensor(student, "batch_grasp_point").to(center_s).float()
    matched_point_t = _require_tensor(teacher, "batch_grasp_point").to(center_s).float()
    if matched_point_s.shape != center_s.shape or matched_point_t.shape != center_s.shape:
        raise ValueError("batch_grasp_point must match [B,Q,3]")

    label_error_vec = matched_point_s - center_s
    label_point_dist = torch.linalg.norm(label_error_vec, dim=-1)
    label_parallel, label_lateral = _project_error(label_error_vec, axis)
    label_shift_bins = nearest_depth_shift_bins(label_parallel, config.depth_interval_m)
    label_parallel_residual = (
        label_parallel - label_shift_bins.float() * float(config.depth_interval_m)
    )
    label_action_residual = torch.sqrt(
        label_lateral.square() + label_parallel_residual.square()
    )
    label_comp_continuous = (
        (label_lateral < float(config.point_match_thresh_m))
        & (label_parallel.abs() < float(config.tolerated_parallel_m))
    )
    label_dst_ids = depth_ids + label_shift_bins.unsqueeze(-1)
    label_depth_retained_bqd = (label_dst_ids >= 0) & (label_dst_ids < D)
    label_any_depth_retained = label_depth_retained_bqd.any(dim=-1)
    label_comp_nearest = (
        label_comp_continuous
        & label_any_depth_retained
        & (label_action_residual < float(config.point_match_thresh_m))
    )

    # Conservative recovery: only count queries for which the clean-depth
    # teacher confirms that this exact selected view has normal label support.
    recovered_continuous = (~student_query_valid) & teacher_query_valid & label_comp_continuous
    recovered_nearest = (~student_query_valid) & teacher_query_valid & label_comp_nearest
    support_continuous = student_query_valid | recovered_continuous
    support_nearest = student_query_valid | recovered_nearest

    # 3) Transport the teacher CDF prediction/GT target along the depth axis.
    # This is evaluated only where the existing student target is valid;
    # recovered queries have no current hard target and are not used for BCE.
    student_logits = _cdf_logits_to_bqadt(
        _require_tensor(student, "grasp_cdf_pred_angle_depth").float()
    )
    teacher_logits = _cdf_logits_to_bqadt(
        _require_tensor(teacher, "grasp_cdf_pred_angle_depth").to(student_logits).float()
    )
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher CDF logit shapes differ")
    T = student_logits.shape[-1]

    student_bins = _require_tensor(student, "batch_grasp_cdf_bins_angle_depth").long()
    teacher_bins = _require_tensor(teacher, "batch_grasp_cdf_bins_angle_depth").to(
        device=center_s.device, dtype=torch.long
    )
    if student_bins.shape != student_valid_bqad.shape or teacher_bins.shape != teacher_valid_bqad.shape:
        raise ValueError("CDF compact-bin shape does not match valid mask")
    student_target = cdf_bins_to_target(student_bins, T, dtype=student_logits.dtype)
    teacher_target = cdf_bins_to_target(teacher_bins, T, dtype=student_logits.dtype)
    teacher_prob = torch.sigmoid(teacher_logits)

    teacher_prob_trans, teacher_valid_trans = shift_depth_axis_nearest(
        teacher_prob, teacher_valid_bqad, center_shift_bins
    )
    teacher_target_trans, teacher_gt_valid_trans = shift_depth_axis_nearest(
        teacher_target, teacher_valid_bqad, center_shift_bins
    )
    if not torch.equal(teacher_valid_trans, teacher_gt_valid_trans):
        raise RuntimeError("transported teacher prediction/GT masks disagree")

    paired_dest_valid = student_valid_bqad & teacher_valid_trans
    before_bce_q, before_bce_valid = _masked_bce_per_query(
        teacher_prob, student_target, paired_dest_valid, config.eps
    )
    after_bce_q, after_bce_valid = _masked_bce_per_query(
        teacher_prob_trans, student_target, paired_dest_valid, config.eps
    )
    paired_bce_query = before_bce_valid & after_bce_valid
    bce_improved = paired_bce_query & (after_bce_q < before_bce_q)

    gt_abs_before = (teacher_target - student_target).abs()
    gt_abs_after = (teacher_target_trans - student_target).abs()
    target_element_mask = paired_dest_valid.unsqueeze(-1).expand_as(gt_abs_before)
    target_exact_before = (gt_abs_before < 0.5) & target_element_mask
    target_exact_after = (gt_abs_after < 0.5) & target_element_mask

    common_valid_bqad = student_valid_bqad & teacher_valid_bqad
    common_element_mask = common_valid_bqad.unsqueeze(-1).expand_as(student_target)
    common_exact = ((student_target - teacher_target).abs() < 0.5) & common_element_mask

    matched_point_same = torch.linalg.norm(matched_point_s - matched_point_t, dim=-1) < 1.0e-6

    stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    stats["exact_seed_ratio"] = _sum_count(exact["exact_seed_ratio"].reshape(1))
    stats["exact_pixel_ratio"] = _sum_count(exact["exact_pixel_ratio"].reshape(1))
    stats["exact_view_ratio"] = _sum_count(exact["exact_view_ratio"].reshape(1))
    stats["approach_axis_consistency"] = _sum_count(axis_error)
    stats["matched_gt_point_same_ratio"] = _ratio_sum_count(matched_point_same)

    stats["student_valid_query_ratio"] = _ratio_sum_count(student_query_valid)
    stats["teacher_valid_query_ratio"] = _ratio_sum_count(teacher_query_valid)
    stats["common_valid_query_ratio"] = _ratio_sum_count(common_query)
    stats["support_iou_before"] = _support_iou_sum_count(student_query_valid, teacher_query_valid)
    stats["support_iou_after_continuous"] = _support_iou_sum_count(support_continuous, teacher_query_valid)
    stats["support_iou_after_nearest"] = _support_iou_sum_count(support_nearest, teacher_query_valid)
    stats["teacher_support_recall_before"] = _ratio_sum_count(student_query_valid, teacher_query_valid)
    stats["teacher_support_recall_after_continuous"] = _ratio_sum_count(support_continuous, teacher_query_valid)
    stats["teacher_support_recall_after_nearest"] = _ratio_sum_count(support_nearest, teacher_query_valid)
    stats["p0_recovered_continuous_ratio_all"] = _ratio_sum_count(recovered_continuous)
    stats["p0_recovered_nearest_ratio_all"] = _ratio_sum_count(recovered_nearest)
    recoverable_denom = (~student_query_valid) & teacher_query_valid
    stats["p0_recovered_continuous_fraction_of_missing_teacher_support"] = _ratio_sum_count(
        recovered_continuous, recoverable_denom
    )
    stats["p0_recovered_nearest_fraction_of_missing_teacher_support"] = _ratio_sum_count(
        recovered_nearest, recoverable_denom
    )

    stats["center_error_mean_m"] = _sum_count(center_error)
    stats["center_parallel_abs_mean_m"] = _sum_count(center_parallel.abs())
    stats["center_lateral_mean_m"] = _sum_count(center_lateral)
    stats["center_nearest_action_residual_mean_m"] = _sum_count(center_action_residual)
    stats["center_continuous_transportable_5mm"] = _ratio_sum_count(
        center_lateral < float(config.point_match_thresh_m)
    )
    stats["center_nearest_transportable_5mm"] = _ratio_sum_count(
        (center_action_residual < float(config.point_match_thresh_m)) & center_any_depth_retained
    )
    stats["center_depth_retained_fraction"] = _sum_count(center_depth_retained_fraction)
    stats["center_shift_abs_bins_mean"] = _sum_count(center_shift_bins.abs().float())

    stats["label_point_dist_mean_m"] = _sum_count(label_point_dist)
    stats["label_parallel_abs_mean_m"] = _sum_count(label_parallel.abs())
    stats["label_lateral_mean_m"] = _sum_count(label_lateral)
    stats["label_nearest_action_residual_mean_m"] = _sum_count(label_action_residual)
    stats["label_comp_continuous_ratio"] = _ratio_sum_count(label_comp_continuous)
    stats["label_comp_nearest_ratio"] = _ratio_sum_count(label_comp_nearest)
    stats["label_shift_abs_bins_mean"] = _sum_count(label_shift_bins.abs().float())

    stats["common_gt_cdf_element_exact_before"] = (
        common_exact.float().sum(), common_element_mask.float().sum()
    )
    stats["paired_gt_cdf_element_exact_before"] = (
        target_exact_before.float().sum(), target_element_mask.float().sum()
    )
    stats["paired_gt_cdf_element_exact_after_nearest"] = (
        target_exact_after.float().sum(), target_element_mask.float().sum()
    )
    stats["teacher_cdf_bce_before_transport"] = _sum_count(before_bce_q, paired_bce_query)
    stats["teacher_cdf_bce_after_nearest_transport"] = _sum_count(after_bce_q, paired_bce_query)
    stats["teacher_cdf_transport_improved_query_ratio"] = _ratio_sum_count(
        bce_improved, paired_bce_query
    )
    stats["teacher_cdf_transport_paired_query_ratio"] = _ratio_sum_count(paired_bce_query)

    per_query: Dict[str, torch.Tensor] = {
        "student_valid": student_query_valid,
        "teacher_valid": teacher_query_valid,
        "common_valid": common_query,
        "center_error_m": center_error,
        "center_parallel_m": center_parallel,
        "center_lateral_m": center_lateral,
        "center_shift_bins": center_shift_bins,
        "center_action_residual_m": center_action_residual,
        "center_depth_retained_fraction": center_depth_retained_fraction,
        "label_point_dist_m": label_point_dist,
        "label_parallel_m": label_parallel,
        "label_lateral_m": label_lateral,
        "label_shift_bins": label_shift_bins,
        "label_action_residual_m": label_action_residual,
        "label_comp_continuous": label_comp_continuous,
        "label_comp_nearest": label_comp_nearest,
        "recovered_continuous": recovered_continuous,
        "recovered_nearest": recovered_nearest,
        "matched_gt_point_same": matched_point_same,
        "teacher_cdf_bce_before": before_bce_q,
        "teacher_cdf_bce_after": after_bce_q,
        "teacher_cdf_bce_paired": paired_bce_query,
        "teacher_cdf_bce_improved": bce_improved,
    }

    return {
        "stats": stats,
        "per_query": per_query,
        "center_shift_bins_bq": center_shift_bins,
        "label_shift_bins_bq": label_shift_bins,
    }
