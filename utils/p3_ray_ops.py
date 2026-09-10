"""Pure tensor operations for P3 ray-evidence aggregation.

No repository model or global argparse modules are imported here.  P3 retains K
metric center hypotheses through representation formation and performs only the
unavoidable final grasp decode over (ray depth, angle, insertion depth).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

DEFAULT_P3_OFFSETS_MM = (-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0)


def parse_offsets(value: str | Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in (value.split(",") if isinstance(value, str) else value))
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("P3 ray offsets must be a non-empty finite list in millimetres.")
    if len(set(values)) != len(values):
        raise ValueError("P3 ray offsets must be unique.")
    if values.count(0.0) != 1:
        raise ValueError("P3 requires exactly one zero-offset center.")
    if max(abs(x) for x in values) > 250.0:
        raise ValueError("P3 ray offsets exceed 250 mm; check units.")
    return values


def backproject_uvz(uv: torch.Tensor, z: torch.Tensor, camera_k: torch.Tensor) -> torch.Tensor:
    """Backproject camera-z depth. uv=[B,Q,2], z=[B,Q], K=[B,3,3]."""
    if uv.dim() != 3 or uv.shape[-1] != 2 or z.shape != uv.shape[:2]:
        raise ValueError("Expected uv [B,Q,2] and z [B,Q].")
    if camera_k.shape != (uv.shape[0], 3, 3):
        raise ValueError("Expected camera intrinsics [B,3,3].")
    camera_k = camera_k.to(z)
    fx = camera_k[:, 0, 0].unsqueeze(1)
    fy = camera_k[:, 1, 1].unsqueeze(1)
    if not bool(((fx > 0) & (fy > 0)).all()):
        raise ValueError("Camera focal lengths must be positive.")
    cx = camera_k[:, 0, 2].unsqueeze(1)
    cy = camera_k[:, 1, 2].unsqueeze(1)
    x = (uv[..., 0].to(z) - cx) / fx * z
    y = (uv[..., 1].to(z) - cy) / fy * z
    return torch.stack((x, y, z), dim=-1)


def build_ray_centers(
    base_xyz: torch.Tensor,
    token_idx: torch.Tensor,
    camera_k: torch.Tensor,
    image_width: int,
    offsets_m: torch.Tensor,
    min_depth: float,
    max_depth: float,
):
    """Return centers [B,Q,K,3], in_range [B,Q,K], descriptor [B,Q,K,5].

    The offsets perturb camera z, not Euclidean ray length.  The zero-offset
    center is copied from the base query exactly to make the identity control
    strict.  Out-of-range hypotheses are numerically clamped but always masked.
    """
    if base_xyz.dim() != 3 or base_xyz.shape[-1] != 3:
        raise ValueError("base_xyz must be [B,Q,3].")
    if token_idx.shape != base_xyz.shape[:2] or offsets_m.dim() != 1:
        raise ValueError("token_idx/offset shape mismatch.")
    if not bool(torch.isfinite(base_xyz).all()):
        raise FloatingPointError("P3 received a non-finite base center.")
    B, Q, _ = base_xyz.shape
    z0 = base_xyz[..., 2].detach()
    u = (token_idx % int(image_width)).to(z0)
    v = (token_idx // int(image_width)).to(z0)
    uv = torch.stack((u, v), dim=-1)
    ray_at_one = backproject_uvz(uv, torch.ones_like(z0), camera_k)
    offsets = offsets_m.to(z0).view(1, 1, -1)
    z = z0.unsqueeze(-1) + offsets
    in_range = torch.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    safe_z = torch.nan_to_num(z, nan=float(min_depth), posinf=float(max_depth), neginf=float(min_depth))
    safe_z = safe_z.clamp(float(min_depth), float(max_depth))
    centers = ray_at_one.unsqueeze(2) * safe_z.unsqueeze(-1)
    zero_idx = torch.nonzero(offsets_m == 0, as_tuple=False).flatten()
    if zero_idx.numel() != 1:
        raise ValueError("P3 needs one zero-offset hypothesis.")
    centers = centers.clone()
    centers[:, :, int(zero_idx.item())] = base_xyz.detach()

    z_span = max(float(max_depth) - float(min_depth), 1e-6)
    off_scale = max(float(offsets_m.abs().max().item()), 0.01)
    rx = ray_at_one[..., 0].unsqueeze(-1).expand(B, Q, offsets_m.numel())
    ry = ray_at_one[..., 1].unsqueeze(-1).expand_as(rx)
    base_norm = (2.0 * (z0 - float(min_depth)) / z_span - 1.0).unsqueeze(-1).expand_as(rx)
    candidate_norm = 2.0 * (safe_z - float(min_depth)) / z_span - 1.0
    offset_norm = offsets.expand_as(rx) / off_scale
    descriptor = torch.stack((rx, ry, base_norm, candidate_norm, offset_norm), dim=-1)
    if not bool(torch.isfinite(descriptor).all()):
        raise FloatingPointError("Non-finite P3 ray descriptor.")
    return centers.contiguous(), in_range.contiguous(), descriptor.contiguous()


def compact_cdf_utility(bins: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    """Convert compact CDF onset bins 0..T to mean success utility in [0,1]."""
    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive.")
    bins = bins.long()
    if bool(((bins < 0) | (bins > int(num_thresholds))).any()):
        raise ValueError("Compact CDF bins must lie in [0,T].")
    return torch.where(
        bins > 0,
        (float(num_thresholds) - bins.float() + 1.0) / float(num_thresholds),
        torch.zeros_like(bins, dtype=torch.float32),
    )


def predicted_raw_utility(cdf_logits: torch.Tensor) -> torch.Tensor:
    """CDF mean utility [B,Q,K,A,D] from logits [B,T,Q,K,A,D]."""
    if cdf_logits.dim() != 6:
        raise ValueError("P3 CDF logits must be [B,T,Q,K,A,D].")
    return torch.sigmoid(cdf_logits.float()).mean(dim=1)


def predicted_joint_utility(cdf_logits: torch.Tensor, viability_logits: torch.Tensor) -> torch.Tensor:
    raw = predicted_raw_utility(cdf_logits)
    if viability_logits.shape != raw.shape[:3]:
        raise ValueError("P3 viability logits must be [B,Q,K].")
    return raw * torch.sigmoid(viability_logits.float()).unsqueeze(-1).unsqueeze(-1)


def _sum_count(value: torch.Tensor, mask: torch.Tensor):
    mask = mask.bool()
    if mask.shape != value.shape:
        raise ValueError(f"Mask/value shape mismatch: {tuple(mask.shape)} vs {tuple(value.shape)}")
    return value.masked_select(mask).sum(), mask.sum().to(value.dtype)


def loss_sums(end_points: Mapping[str, torch.Tensor]):
    """Sufficient statistics for the P3 objective.

    CDF/width preserve the original label masks.  The viability term is balanced
    between 5-mm label-support positives and negatives.  The joint utility loss
    directly calibrates the score used for the final KxAxD decode: valid CDF
    operations use their compact target utility, while hypotheses outside the
    5-mm label domain are known zero targets.  Selected-view-but-unlabelled
    surface hypotheses remain unknown rather than being forced negative.
    """
    cdf = end_points["p3_cdf_logits"].float()                    # B,T,Q,K,A,D
    width = end_points["p3_width_pred"] .float()                # B,D,Q,K,A
    viability = end_points["p3_viability_logits"].float()       # B,Q,K
    bins = end_points["p3_cdf_bins"].long()                     # B,Q,K,A,D
    cdf_valid = end_points["p3_cdf_valid"].bool()
    width_label = end_points["p3_width_label"].float()           # B,Q,K,A,D metres
    width_valid = end_points["p3_width_valid"].bool()
    point_support = end_points["p3_point_support"].bool()        # B,Q,K
    point_known = end_points["p3_point_known"].bool()
    in_range = end_points["p3_in_range"].bool()
    if cdf.dim() != 6:
        raise ValueError("p3_cdf_logits must be [B,T,Q,K,A,D].")
    B, T, Q, K, A, D = cdf.shape
    if bins.shape != (B, Q, K, A, D) or cdf_valid.shape != bins.shape:
        raise ValueError("P3 CDF labels must be [B,Q,K,A,D].")
    if width.shape != (B, D, Q, K, A):
        raise ValueError("p3_width_pred must be [B,D,Q,K,A].")
    if width_label.shape != bins.shape or width_valid.shape != bins.shape:
        raise ValueError("P3 width labels/masks must match [B,Q,K,A,D].")
    if viability.shape != (B, Q, K) or point_support.shape != viability.shape:
        raise ValueError("P3 viability/support shape mismatch.")
    if point_known.shape != viability.shape or in_range.shape != viability.shape:
        raise ValueError("P3 known/range shape mismatch.")

    threshold_ids = torch.arange(T, device=cdf.device, dtype=bins.dtype)
    target_cdf = ((bins.unsqueeze(-1) > 0) &
                  (threshold_ids.view(1, 1, 1, 1, 1, T) >= bins.unsqueeze(-1) - 1)).to(cdf)
    logits_qkadt = cdf.permute(0, 2, 3, 4, 5, 1).contiguous()
    cdf_map = F.binary_cross_entropy_with_logits(logits_qkadt, target_cdf, reduction="none")
    cdf_mask = cdf_valid.unsqueeze(-1).expand_as(cdf_map) & in_range[..., None, None, None]

    width_qkad = width.permute(0, 2, 3, 4, 1).contiguous()
    width_map = F.smooth_l1_loss(width_qkad, width_label * 10.0, reduction="none")
    width_mask = width_valid & cdf_valid & in_range[..., None, None]

    viability_target = point_support.to(viability)
    viability_map = F.binary_cross_entropy_with_logits(viability, viability_target, reduction="none")
    viability_known = point_known & in_range
    viability_pos = viability_known & point_support
    viability_neg = viability_known & (~point_support)

    target_utility = compact_cdf_utility(bins, T).to(cdf)
    target_utility = target_utility.masked_fill(~cdf_valid, 0.0)
    pred_joint = predicted_joint_utility(cdf, viability)
    # Known joint targets: evaluator-labelled operations OR centers known to be
    # outside the 5-mm grasp-label domain.  Missing selected-view labels at a
    # geometrically supported center stay unknown.
    off_surface = viability_known & (~point_support)
    joint_known = cdf_valid | off_surface[..., None, None]
    joint_known &= in_range[..., None, None]
    joint_target = torch.where(cdf_valid, target_utility, torch.zeros_like(target_utility))
    joint_map = F.binary_cross_entropy(
        pred_joint.clamp(1e-6, 1.0 - 1e-6), joint_target, reduction="none"
    )
    joint_pos = joint_known & (joint_target > 0)
    joint_neg = joint_known & (~joint_pos)

    return {
        "cdf": _sum_count(cdf_map, cdf_mask),
        "width": _sum_count(width_map, width_mask),
        "viability_pos": _sum_count(viability_map, viability_pos),
        "viability_neg": _sum_count(viability_map, viability_neg),
        "joint_pos": _sum_count(joint_map, joint_pos),
        "joint_neg": _sum_count(joint_map, joint_neg),
    }


def final_indices(end_points: Mapping[str, torch.Tensor], score_mode: str = "joint",
                  force_zero: bool = False):
    """Return final k/a/d and per-ray score.  No pre-representation selection."""
    cdf = end_points["p3_cdf_logits"]
    in_range = end_points["p3_in_range"].bool()
    offsets = end_points["p3_offsets_m"]
    utility = (
        predicted_joint_utility(cdf, end_points["p3_viability_logits"])
        if score_mode == "joint"
        else predicted_raw_utility(cdf)
    )
    if score_mode not in ("joint", "raw"):
        raise ValueError("P3 score_mode must be joint or raw.")
    utility = utility.masked_fill(~in_range[..., None, None], -1.0)
    B, Q, K, A, D = utility.shape
    if force_zero:
        zero = torch.nonzero(offsets == 0, as_tuple=False).flatten()
        if zero.numel() != 1:
            raise ValueError("P3 zero control requires one zero offset.")
        k = torch.full((B, Q), int(zero.item()), device=utility.device, dtype=torch.long)
        sub = utility[:, :, int(zero.item())].reshape(B, Q, A * D)
        op = sub.argmax(-1)
        score = sub.gather(-1, op.unsqueeze(-1)).squeeze(-1)
        a = torch.div(op, D, rounding_mode="floor")
        d = torch.remainder(op, D)
        return k, a, d, score, utility
    flat = utility.reshape(B, Q, K * A * D)
    idx = flat.argmax(-1)
    score = flat.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    per_k = A * D
    k = torch.div(idx, per_k, rounding_mode="floor")
    rem = torch.remainder(idx, per_k)
    a = torch.div(rem, D, rounding_mode="floor")
    d = torch.remainder(rem, D)
    return k, a, d, score, utility


@torch.no_grad()
def metric_sums(end_points: Mapping[str, torch.Tensor]):
    """Coverage and final-decode diagnostics; not an analytic AP oracle."""
    offsets = end_points["p3_offsets_m"]
    point = end_points["p3_point_support"].bool()
    known = end_points["p3_point_known"].bool() & end_points["p3_in_range"].bool()
    cdf_valid = end_points["p3_cdf_valid"].bool().any(-1).any(-1)
    bins = end_points["p3_cdf_bins"].long()
    T = int(end_points["p3_cdf_logits"].shape[1])
    gt_util = compact_cdf_utility(bins, T).to(end_points["p3_cdf_logits"])
    gt_util = gt_util.masked_fill(~end_points["p3_cdf_valid"].bool(), 0.0)
    target_best_kad = gt_util.reshape(*gt_util.shape[:3], -1).max(-1).values
    target_best_kad *= point.float()
    positive_ray = target_best_kad.max(-1).values > 0
    any_known = known.any(-1)
    zero = int(torch.nonzero(offsets == 0, as_tuple=False).flatten()[0].item())

    def pair(v, m):
        v = v.to(end_points["p3_cdf_logits"].dtype)
        m = m.bool()
        return v.masked_select(m).sum(), m.sum().to(v.dtype)

    result = {
        "p3_base_point_support": pair(point[..., zero].float(), known[..., zero]),
        "p3_any_point_support": pair((point & known).any(-1).float(), any_known),
        "p3_any_cdf_support": pair((cdf_valid & known).any(-1).float(), any_known),
    }
    for mode in ("joint", "raw"):
        k, a, d, score, _ = final_indices(end_points, score_mode=mode, force_zero=False)
        selected_point = point.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        selected_cdf = cdf_valid.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        selected_gt = target_best_kad.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        oracle_gt = target_best_kad.max(-1).values
        signed = offsets.to(score)[k]
        prefix = f"p3_{mode}"
        result.update({
            f"{prefix}_selected_point_support": pair(selected_point.float(), any_known),
            f"{prefix}_selected_cdf_support": pair(selected_cdf.float(), any_known),
            f"{prefix}_selected_nonzero": pair((k != zero).float(), any_known),
            f"{prefix}_offset_abs_m": pair(signed.abs(), any_known),
            f"{prefix}_offset_signed_m": pair(signed, any_known),
            f"{prefix}_selected_target_utility": pair(selected_gt, positive_ray),
            f"{prefix}_oracle_target_utility": pair(oracle_gt, positive_ray),
            f"{prefix}_selection_regret": pair((oracle_gt - selected_gt).clamp_min(0), positive_ray),
        })
        for i in range(offsets.numel()):
            result[f"{prefix}_selected_k{i}"] = pair((k == i).float(), any_known)
    return result
