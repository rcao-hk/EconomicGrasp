"""Diagnostics for privileged clean-depth knowledge distillation.

These functions are observational: they do not change model predictions or the
training objective.  They answer three questions that ordinary KD losses cannot:

1. Is the clean-depth teacher actually closer to grasp GT than the RGB student
   on the same student-selected image query?
2. Do the teacher and student correspond to the same physical neighborhood, or
   does metric-depth error change the matched CDF/width labels?
3. Do the supervised and KD objectives push the student outputs in compatible
   directions?

The expensive paired diagnostics are intended to run every few thousand
optimizer steps and on a configurable subset of validation batches.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _zero_like(end_points: Mapping[str, Any]) -> torch.Tensor:
    for key in (
        "grasp_cdf_pred_angle_depth",
        "view_score",
        "grasp_width_pred_angle_depth",
    ):
        value = end_points.get(key)
        if torch.is_tensor(value):
            return value.detach().sum() * 0.0
    return torch.tensor(0.0)


def _safe_scalar_mean(
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    value = value.float()
    if mask is None:
        return value.mean() if value.numel() else zero
    mask = mask.to(device=value.device, dtype=torch.bool)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    if bool(mask.any()):
        return value[mask].mean()
    return zero


def _safe_quantile(
    value: torch.Tensor,
    q: float,
    mask: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    value = value.float()
    if mask is not None:
        mask = mask.to(device=value.device, dtype=torch.bool)
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value)
        value = value[mask]
    else:
        value = value.reshape(-1)
    if value.numel() == 0:
        return zero
    return torch.quantile(value, float(q))


def _cdf_bins_to_target(
    bins: torch.Tensor,
    num_thresholds: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand compact CDF bins [... ] to cumulative targets [..., T]."""
    bins = bins.long()
    ids = torch.arange(
        int(num_thresholds),
        device=bins.device,
        dtype=bins.dtype,
    )
    return (
        (bins.unsqueeze(-1) > 0)
        & (ids >= bins.unsqueeze(-1) - 1)
    ).to(dtype=dtype)


def _cdf_logits_to_qadt(logits_btqad: torch.Tensor) -> torch.Tensor:
    if logits_btqad.dim() != 5:
        raise ValueError(
            "CDF logits must be [B,T,Q,A,D], got "
            f"{tuple(logits_btqad.shape)}"
        )
    return logits_btqad.permute(0, 2, 3, 4, 1).contiguous()


def _per_query_bce(
    logits_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-query unbalanced BCE and the query-valid mask.

    The same target is used when comparing teacher and student, so class
    imbalance does not bias the *difference* between the two losses.
    """
    loss = F.binary_cross_entropy_with_logits(
        logits_bqadt,
        target_bqadt,
        reduction="none",
    )
    mask = valid_bqad.bool().unsqueeze(-1).expand_as(loss)
    count = mask.sum(dim=(-1, -2, -3)).float()
    loss_sum = (loss * mask.to(loss.dtype)).sum(dim=(-1, -2, -3))
    query_valid = count > 0
    per_query = loss_sum / count.clamp_min(1.0)
    return per_query, query_valid


def _per_query_balanced_bce(
    logits_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose the repository's threshold-balanced CDF loss per query.

    The ordinary CDF objective balances positive and negative candidates at
    each threshold across the whole batch.  This function constructs the same
    global element weights, then normalizes each query contribution by the
    weight mass assigned to that query.  Comparing teacher and student with
    these shared weights answers whether the teacher is better under the
    *actual supervised CDF objective*, not only unbalanced BCE.
    """
    loss = F.binary_cross_entropy_with_logits(
        logits_bqadt,
        target_bqadt,
        reduction="none",
    )
    valid = valid_bqad.bool()
    weights = torch.zeros_like(loss)
    num_thresholds = int(loss.shape[-1])
    for t in range(num_thresholds):
        target_t = target_bqadt[..., t]
        pos = valid & (target_t > 0.5)
        neg = valid & (~pos)
        has_pos = bool(pos.any())
        has_neg = bool(neg.any())
        if has_pos and has_neg:
            weights[..., t][pos] = 0.5 / pos.sum().float()
            weights[..., t][neg] = 0.5 / neg.sum().float()
        elif has_pos:
            weights[..., t][pos] = 1.0 / pos.sum().float()
        elif has_neg:
            weights[..., t][neg] = 1.0 / neg.sum().float()
    weights = weights / max(num_thresholds, 1)
    query_weight = weights.sum(dim=(-1, -2, -3))
    query_loss = (loss * weights).sum(dim=(-1, -2, -3))
    query_valid = query_weight > 0
    query_loss = query_loss / query_weight.clamp_min(1.0e-12)
    return query_loss, query_valid


def _selected_target_utility(
    logits_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT utility selected by each prediction for every center-view query."""
    prob = torch.sigmoid(logits_bqadt)
    pred_utility = prob.mean(dim=-1)          # [B,Q,A,D]
    target_utility = target_bqadt.mean(dim=-1)
    valid = valid_bqad.bool()
    query_valid = valid.any(dim=-1).any(dim=-1)
    masked = pred_utility.masked_fill(~valid, -1.0e6)
    selected = masked.flatten(2).argmax(dim=-1)
    selected_target = target_utility.flatten(2).gather(
        -1, selected.unsqueeze(-1)
    ).squeeze(-1)
    return selected_target, query_valid


def _width_prediction_qad(
    width_bdqa: torch.Tensor,
) -> torch.Tensor:
    if width_bdqa.dim() != 4:
        raise ValueError(
            "Width prediction must be [B,D,Q,A], got "
            f"{tuple(width_bdqa.shape)}"
        )
    return width_bdqa.permute(0, 2, 3, 1).contiguous()


def _require_tensor(
    end_points: Mapping[str, Any],
    key: str,
) -> torch.Tensor:
    value = end_points.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Required diagnostic endpoint {key!r} is missing.")
    return value


@torch.no_grad()
def compute_privileged_kd_diagnostics(
    student_end_points: Mapping[str, Any],
    exact_view_teacher_end_points: Mapping[str, Any],
    *,
    center_thresholds_m: Sequence[float] = (0.005, 0.010, 0.020),
    prefix: str = "D: PrivKD ",
) -> Dict[str, torch.Tensor]:
    """Compare teacher and student on exact same image seeds and view anchors.

    ``exact_view_teacher_end_points`` must come from a teacher forward with:
      * ``image_fps_seed_idx_override`` = student base seed indices
      * ``oracle_view_inds_override``   = student selected view indices

    Both endpoint dictionaries must then pass through the ordinary supervised
    loss once so their matched CDF/width GT tensors are available.
    """
    zero = _zero_like(student_end_points)
    out: Dict[str, torch.Tensor] = {}

    s_seed = _require_tensor(student_end_points, "kview_base_token_sel_idx").long()
    t_seed = _require_tensor(
        exact_view_teacher_end_points, "kview_base_token_sel_idx"
    ).to(device=s_seed.device, dtype=torch.long)
    seed_equal = s_seed == t_seed
    out[prefix + "seed exact ratio"] = seed_equal.float().mean().reshape(())

    s_view_idx = _require_tensor(student_end_points, "grasp_top_view_inds").long()
    t_view_idx = _require_tensor(
        exact_view_teacher_end_points, "grasp_top_view_inds"
    ).to(device=s_view_idx.device, dtype=torch.long)
    view_equal = s_view_idx == t_view_idx
    out[prefix + "view exact ratio"] = view_equal.float().mean().reshape(())

    s_center = _require_tensor(student_end_points, "xyz_graspable").float()
    t_center = _require_tensor(
        exact_view_teacher_end_points, "xyz_graspable"
    ).to(s_center).float()
    if s_center.shape != t_center.shape:
        raise ValueError(
            "Teacher/student center shapes differ: "
            f"{tuple(s_center.shape)} vs {tuple(t_center.shape)}"
        )
    center_delta = torch.linalg.vector_norm(s_center - t_center, dim=-1)
    z_delta = (s_center[..., 2] - t_center[..., 2]).abs()
    finite_center = torch.isfinite(center_delta) & torch.isfinite(z_delta)
    out[prefix + "center xyz MAE"] = _safe_scalar_mean(
        center_delta, finite_center, zero
    )
    out[prefix + "center xyz p90"] = _safe_quantile(
        center_delta, 0.90, finite_center, zero
    )
    out[prefix + "center z MAE"] = _safe_scalar_mean(
        z_delta, finite_center, zero
    )
    out[prefix + "center z p50"] = _safe_quantile(
        z_delta, 0.50, finite_center, zero
    )
    out[prefix + "center z p90"] = _safe_quantile(
        z_delta, 0.90, finite_center, zero
    )
    for threshold in center_thresholds_m:
        tag = int(round(float(threshold) * 1000.0))
        out[prefix + f"center z <{tag}mm"] = (
            (finite_center & (z_delta < float(threshold))).float().mean().reshape(())
        )

    s_logits = _cdf_logits_to_qadt(
        _require_tensor(
            student_end_points, "grasp_cdf_pred_angle_depth"
        ).float()
    )
    t_logits = _cdf_logits_to_qadt(
        _require_tensor(
            exact_view_teacher_end_points,
            "grasp_cdf_pred_angle_depth",
        ).to(s_logits).float()
    )
    if s_logits.shape != t_logits.shape:
        raise ValueError(
            "Exact-view CDF shapes differ: "
            f"{tuple(s_logits.shape)} vs {tuple(t_logits.shape)}"
        )
    num_thresholds = int(s_logits.shape[-1])

    s_bins = _require_tensor(
        student_end_points, "batch_grasp_cdf_bins_angle_depth"
    ).long().to(s_logits.device)
    t_bins = _require_tensor(
        exact_view_teacher_end_points,
        "batch_grasp_cdf_bins_angle_depth",
    ).long().to(s_logits.device)
    s_valid = _require_tensor(
        student_end_points, "batch_grasp_cdf_valid_mask"
    ).bool().to(s_logits.device)
    t_valid = _require_tensor(
        exact_view_teacher_end_points,
        "batch_grasp_cdf_valid_mask",
    ).bool().to(s_logits.device)
    if not (
        s_bins.shape == t_bins.shape == s_valid.shape == t_valid.shape
    ):
        raise ValueError(
            "Teacher/student CDF label shapes differ: "
            f"s_bins={tuple(s_bins.shape)}, t_bins={tuple(t_bins.shape)}, "
            f"s_valid={tuple(s_valid.shape)}, t_valid={tuple(t_valid.shape)}"
        )

    s_target = _cdf_bins_to_target(s_bins, num_thresholds, s_logits.dtype)
    t_target = _cdf_bins_to_target(t_bins, num_thresholds, s_logits.dtype)
    valid_union = s_valid | t_valid
    valid_common = s_valid & t_valid
    valid_intersection = valid_common.float().sum()
    valid_union_count = valid_union.float().sum().clamp_min(1.0)
    out[prefix + "CDF valid IoU"] = (
        valid_intersection / valid_union_count
    ).reshape(())
    out[prefix + "CDF valid common ratio"] = valid_common.float().mean().reshape(())
    out[prefix + "CDF label bin exact"] = _safe_scalar_mean(
        (s_bins == t_bins).float(), valid_common, zero
    )
    out[prefix + "CDF target disagreement"] = _safe_scalar_mean(
        (s_target - t_target).abs(),
        valid_common.unsqueeze(-1),
        zero,
    )
    s_target_utility = s_target.mean(dim=-1)
    t_target_utility = t_target.mean(dim=-1)
    out[prefix + "CDF GT utility drift"] = _safe_scalar_mean(
        (s_target_utility - t_target_utility).abs(),
        valid_common,
        zero,
    )

    teacher_prob = torch.sigmoid(t_logits)
    out[prefix + "CDF teacher-soft vs student-GT MAE"] = _safe_scalar_mean(
        (teacher_prob - s_target).abs(),
        s_valid.unsqueeze(-1),
        zero,
    )
    out[prefix + "CDF teacher-hard vs student-GT disagree"] = _safe_scalar_mean(
        ((teacher_prob >= 0.5) != (s_target >= 0.5)).float(),
        s_valid.unsqueeze(-1),
        zero,
    )

    s_loss_on_s_gt, s_query_valid = _per_query_bce(
        s_logits, s_target, s_valid
    )
    t_loss_on_s_gt, t_query_valid = _per_query_bce(
        t_logits, s_target, s_valid
    )
    common_query = s_query_valid & t_query_valid
    loss_delta = t_loss_on_s_gt - s_loss_on_s_gt
    out[prefix + "teacher-student BCE delta on student GT"] = _safe_scalar_mean(
        loss_delta, common_query, zero
    )
    out[prefix + "teacher better ratio on student GT"] = _safe_scalar_mean(
        (loss_delta < 0).float(), common_query, zero
    )

    s_balanced_on_s_gt, s_balanced_valid = _per_query_balanced_bce(
        s_logits, s_target, s_valid
    )
    t_balanced_on_s_gt, t_balanced_valid = _per_query_balanced_bce(
        t_logits, s_target, s_valid
    )
    balanced_common = s_balanced_valid & t_balanced_valid
    balanced_delta = t_balanced_on_s_gt - s_balanced_on_s_gt
    out[prefix + "teacher-student balanced BCE delta"] = _safe_scalar_mean(
        balanced_delta, balanced_common, zero
    )
    out[prefix + "teacher better ratio balanced"] = _safe_scalar_mean(
        (balanced_delta < 0).float(), balanced_common, zero
    )

    s_selected, s_sel_valid = _selected_target_utility(
        s_logits, s_target, s_valid
    )
    t_selected, t_sel_valid = _selected_target_utility(
        t_logits, s_target, s_valid
    )
    sel_valid = s_sel_valid & t_sel_valid
    selected_advantage = t_selected - s_selected
    out[prefix + "teacher selected-utility advantage"] = _safe_scalar_mean(
        selected_advantage, sel_valid, zero
    )
    out[prefix + "teacher selection better ratio"] = _safe_scalar_mean(
        (selected_advantage > 0).float(), sel_valid, zero
    )

    cdf_prob_mae = (torch.sigmoid(s_logits) - teacher_prob).abs()
    out[prefix + "CDF probability MAE"] = _safe_scalar_mean(
        cdf_prob_mae, s_valid.unsqueeze(-1), zero
    )

    # Whether teacher advantage concentrates in physically aligned queries.
    query_z = z_delta
    for threshold in center_thresholds_m:
        tag = int(round(float(threshold) * 1000.0))
        z_mask = common_query & (query_z < float(threshold))
        out[prefix + f"teacher better ratio z<{tag}mm"] = _safe_scalar_mean(
            (loss_delta < 0).float(), z_mask, zero
        )
        out[prefix + f"teacher BCE advantage z<{tag}mm"] = -_safe_scalar_mean(
            loss_delta, z_mask, zero
        )
        balanced_z_mask = balanced_common & (query_z < float(threshold))
        out[prefix + f"teacher better balanced z<{tag}mm"] = _safe_scalar_mean(
            (balanced_delta < 0).float(), balanced_z_mask, zero
        )
        out[prefix + f"balanced BCE advantage z<{tag}mm"] = -_safe_scalar_mean(
            balanced_delta, balanced_z_mask, zero
        )
        sel_z_mask = sel_valid & (query_z < float(threshold))
        out[prefix + f"selection advantage z<{tag}mm"] = _safe_scalar_mean(
            selected_advantage, sel_z_mask, zero
        )

    # Width target/prediction diagnostics.
    s_width = _width_prediction_qad(
        _require_tensor(
            student_end_points, "grasp_width_pred_angle_depth"
        ).float()
    )
    t_width = _width_prediction_qad(
        _require_tensor(
            exact_view_teacher_end_points,
            "grasp_width_pred_angle_depth",
        ).to(s_width).float()
    )
    s_width_gt = _require_tensor(
        student_end_points, "batch_grasp_width_angle_depth"
    ).to(s_width).float() * 10.0
    t_width_gt = _require_tensor(
        exact_view_teacher_end_points,
        "batch_grasp_width_angle_depth",
    ).to(s_width).float() * 10.0
    s_width_valid = _require_tensor(
        student_end_points,
        "batch_grasp_width_valid_mask_angle_depth",
    ).bool().to(s_width.device)
    t_width_valid = _require_tensor(
        exact_view_teacher_end_points,
        "batch_grasp_width_valid_mask_angle_depth",
    ).bool().to(s_width.device)
    width_common = s_width_valid & t_width_valid
    out[prefix + "width GT drift x10"] = _safe_scalar_mean(
        (s_width_gt - t_width_gt).abs(), width_common, zero
    )
    s_width_err = (s_width - s_width_gt).abs()
    t_width_err_on_s_gt = (t_width - s_width_gt).abs()
    width_delta = t_width_err_on_s_gt - s_width_err
    out[prefix + "teacher-student width error delta"] = _safe_scalar_mean(
        width_delta, s_width_valid, zero
    )
    out[prefix + "teacher width better ratio"] = _safe_scalar_mean(
        (width_delta < 0).float(), s_width_valid, zero
    )

    # Copy the teacher's ordinary supervised components for direct comparison.
    for src_key, dst_suffix in (
        ("B: CDF Loss", "teacher supervised CDF loss"),
        ("B: View Loss", "teacher supervised view loss"),
        ("B: Width Depth Loss", "teacher supervised width loss"),
        ("D: CDF Selection Regret", "teacher CDF selection regret"),
        ("D: CDF Selected Target Utility", "teacher selected target utility"),
    ):
        value = exact_view_teacher_end_points.get(src_key)
        if torch.is_tensor(value) and value.numel() == 1:
            out[prefix + dst_suffix] = value.detach().reshape(())

    return out


def _gradient_pair_stats(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    output: torch.Tensor,
    *,
    zero: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Output-space gradient cosine, conflict ratio, and norm statistics."""
    if (
        not torch.is_tensor(loss_a)
        or not torch.is_tensor(loss_b)
        or not output.requires_grad
        or not loss_a.requires_grad
        or not loss_b.requires_grad
    ):
        return zero, zero, zero, zero
    grad_a = torch.autograd.grad(
        loss_a,
        output,
        retain_graph=True,
        allow_unused=True,
    )[0]
    grad_b = torch.autograd.grad(
        loss_b,
        output,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad_a is None or grad_b is None:
        return zero, zero, zero, zero
    a = grad_a.detach().float().reshape(-1)
    b = grad_b.detach().float().reshape(-1)
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    cosine = (a * b).sum() / (norm_a * norm_b).clamp_min(1.0e-12)
    active = (a.abs() > 1.0e-12) & (b.abs() > 1.0e-12)
    if bool(active.any()):
        conflict = ((a[active] * b[active]) < 0).float().mean()
    else:
        conflict = zero
    norm_ratio = norm_b / norm_a.clamp_min(1.0e-12)
    active_ratio = active.float().mean()
    return cosine, conflict, norm_ratio, active_ratio


def compute_output_gradient_conflict(
    student_end_points: Mapping[str, Any],
    *,
    prefix: str = "D: KDDiag ",
) -> Dict[str, torch.Tensor]:
    """Measure supervised-vs-KD conflicts directly at student output tensors.

    Computing gradients with respect to output tensors avoids DDP parameter-hook
    side effects and is safe to run before the ordinary ``loss.backward()``.
    """
    zero = _zero_like(student_end_points)
    out: Dict[str, torch.Tensor] = {}
    specs = (
        (
            "CDF",
            "B: CDF Loss",
            "B: KD CDF Loss",
            "grasp_cdf_pred_angle_depth",
        ),
        (
            "View",
            "B: View Loss",
            "B: KD View Loss",
            "view_score",
        ),
        (
            "Width",
            "B: Width Depth Loss",
            "B: KD Width Loss",
            "grasp_width_pred_angle_depth",
        ),
    )
    for name, supervised_key, kd_key, output_key in specs:
        supervised = student_end_points.get(supervised_key)
        kd = student_end_points.get(kd_key)
        output = student_end_points.get(output_key)
        if not (
            torch.is_tensor(supervised)
            and torch.is_tensor(kd)
            and torch.is_tensor(output)
        ):
            continue
        cosine, conflict, norm_ratio, active_ratio = _gradient_pair_stats(
            supervised,
            kd,
            output,
            zero=zero,
        )
        out[prefix + f"{name} grad cosine"] = cosine.reshape(())
        out[prefix + f"{name} grad sign-conflict"] = conflict.reshape(())
        out[prefix + f"{name} KD/sup grad norm"] = norm_ratio.reshape(())
        out[prefix + f"{name} active-grad ratio"] = active_ratio.reshape(())
    return out
