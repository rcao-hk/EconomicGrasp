"""Pure tensor operations for ray-conditioned grasp confidence calibration.

This experiment keeps the Stage-1 grasp pose exactly unchanged. Multi-depth
local evidence is used only to predict a scalar confidence gate per native image
ray. The final calibrated ranking score is

    s_cal = s_raw * sigmoid(confidence_logit).

No depth/center/view/angle/insertion-depth decision is changed by this module.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

DEFAULT_RC_OFFSETS_MM = (-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0)


def parse_offsets(value: str | Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in (value.split(",") if isinstance(value, str) else value))
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Ray-confidence offsets must be a non-empty finite list in millimetres.")
    if len(set(values)) != len(values) or values.count(0.0) != 1:
        raise ValueError("Ray-confidence offsets must be unique and contain exactly one zero.")
    if max(abs(x) for x in values) > 250.0:
        raise ValueError("Ray-confidence offsets exceed 250 mm; check units.")
    return values


def backproject_uvz(uv: torch.Tensor, z: torch.Tensor, camera_k: torch.Tensor) -> torch.Tensor:
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
    """Build camera-z hypotheses without changing the exact zero-offset center.

    Returns centers [B,Q,K,3], in_range [B,Q,K], descriptor [B,Q,K,5].
    """
    if base_xyz.dim() != 3 or base_xyz.shape[-1] != 3:
        raise ValueError("base_xyz must be [B,Q,3].")
    if token_idx.shape != base_xyz.shape[:2] or offsets_m.dim() != 1:
        raise ValueError("token_idx/offset shape mismatch.")
    if not bool(torch.isfinite(base_xyz).all()):
        raise FloatingPointError("Non-finite base query center.")
    B, Q, _ = base_xyz.shape
    z0 = base_xyz[..., 2].detach()
    u = (token_idx % int(image_width)).to(z0)
    v = (token_idx // int(image_width)).to(z0)
    uv = torch.stack((u, v), dim=-1)
    ray = backproject_uvz(uv, torch.ones_like(z0), camera_k)
    offsets = offsets_m.to(z0).view(1, 1, -1)
    z = z0.unsqueeze(-1) + offsets
    in_range = torch.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    safe_z = torch.nan_to_num(z, nan=float(min_depth), posinf=float(max_depth), neginf=float(min_depth))
    safe_z = safe_z.clamp(float(min_depth), float(max_depth))
    centers = ray.unsqueeze(2) * safe_z.unsqueeze(-1)
    zero = torch.nonzero(offsets_m == 0, as_tuple=False).flatten()
    if zero.numel() != 1:
        raise ValueError("Exactly one zero offset is required.")
    centers = centers.clone()
    centers[:, :, int(zero.item())] = base_xyz.detach()

    z_span = max(float(max_depth) - float(min_depth), 1e-6)
    off_scale = max(float(offsets_m.abs().max().item()), 0.01)
    rx = ray[..., 0].unsqueeze(-1).expand(B, Q, offsets_m.numel())
    ry = ray[..., 1].unsqueeze(-1).expand_as(rx)
    base_norm = (2.0 * (z0 - float(min_depth)) / z_span - 1.0).unsqueeze(-1).expand_as(rx)
    cand_norm = 2.0 * (safe_z - float(min_depth)) / z_span - 1.0
    off_norm = offsets.expand_as(rx) / off_scale
    descriptor = torch.stack((rx, ry, base_norm, cand_norm, off_norm), dim=-1)
    if not bool(torch.isfinite(descriptor).all()):
        raise FloatingPointError("Non-finite ray-confidence descriptor.")
    return centers.contiguous(), in_range.contiguous(), descriptor.contiguous()


def raw_cdf_utility(cdf_logits: torch.Tensor) -> torch.Tensor:
    """Return mean-threshold CDF utility [B,Q,A,D] from [B,T,Q,A,D]."""
    if cdf_logits.dim() != 5:
        raise ValueError("CDF logits must be [B,T,Q,A,D].")
    return torch.sigmoid(cdf_logits.float()).mean(dim=1)


def select_raw_operation(cdf_logits: torch.Tensor):
    """Select the native Stage-1 angle/insertion-depth operation by raw CDF."""
    utility = raw_cdf_utility(cdf_logits)
    B, Q, A, D = utility.shape
    flat = utility.reshape(B, Q, A * D)
    op = flat.argmax(dim=-1)
    score = flat.gather(-1, op.unsqueeze(-1)).squeeze(-1)
    angle = torch.div(op, D, rounding_mode="floor")
    depth = torch.remainder(op, D)
    return angle, depth, score, utility


def compact_cdf_utility(bins: torch.Tensor, num_thresholds: int) -> torch.Tensor:
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


def build_confidence_targets(
    cdf_bins: torch.Tensor,
    cdf_valid: torch.Tensor,
    point_support: torch.Tensor,
    point_known: torch.Tensor,
    angle_idx: torch.Tensor,
    depth_idx: torch.Tensor,
    raw_score: torch.Tensor,
    num_thresholds: int,
):
    """Build GT reliability for the *unchanged* Stage-1 decoded operation.

    A center known to lie outside the 5-mm grasp-label domain is a known zero.
    At a geometrically supported center, the selected operation is supervised
    only when its CDF label is valid. Missing selected-view/action labels remain
    unknown rather than being forced negative.
    """
    if cdf_bins.dim() != 4 or cdf_valid.shape != cdf_bins.shape:
        raise ValueError("CDF bins/valid must be [B,Q,A,D].")
    B, Q, A, D = cdf_bins.shape
    if angle_idx.shape != (B, Q) or depth_idx.shape != (B, Q) or raw_score.shape != (B, Q):
        raise ValueError("Selected operation/raw score must be [B,Q].")
    if point_support.shape != (B, Q) or point_known.shape != (B, Q):
        raise ValueError("Point support/known must be [B,Q].")
    b = torch.arange(B, device=cdf_bins.device).view(B, 1).expand(B, Q)
    q = torch.arange(Q, device=cdf_bins.device).view(1, Q).expand(B, Q)
    a = angle_idx.long().clamp(0, A - 1)
    d = depth_idx.long().clamp(0, D - 1)
    selected_bin = cdf_bins[b, q, a, d]
    selected_valid = cdf_valid[b, q, a, d].bool()
    selected_utility = compact_cdf_utility(selected_bin, num_thresholds).to(raw_score)

    known_off_surface = point_known.bool() & (~point_support.bool())
    known_labeled = point_known.bool() & point_support.bool() & selected_valid
    known = known_off_surface | known_labeled
    target = torch.where(known_labeled, selected_utility, torch.zeros_like(selected_utility))
    target = target.clamp(0.0, 1.0)
    raw = raw_score.detach().clamp(1e-6, 1.0)
    ideal_gate = torch.where(known, (target / raw).clamp(0.0, 1.0), torch.ones_like(target))
    return {
        "target_score": target,
        "target_known": known,
        "target_positive": known & (target > 0),
        "ideal_gate": ideal_gate,
        "selected_label_valid": selected_valid,
    }


def confidence_loss_sums(
    raw_score: torch.Tensor,
    confidence_logit: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    rank_temperature: float = 0.10,
):
    """Balanced score calibration plus frame-wise listwise ranking loss."""
    if raw_score.shape != confidence_logit.shape:
        raise ValueError("raw_score/confidence_logit shape mismatch.")
    if rank_temperature <= 0:
        raise ValueError("rank_temperature must be positive.")
    target = targets["target_score"].to(raw_score).clamp(0.0, 1.0)
    known = targets["target_known"].to(raw_score.device).bool()
    if target.shape != raw_score.shape or known.shape != raw_score.shape:
        raise ValueError("Confidence targets must match [B,Q].")
    gate = torch.sigmoid(confidence_logit.float())
    calibrated = (raw_score.detach().float() * gate).clamp(1e-6, 1.0 - 1e-6)
    calibration = F.binary_cross_entropy(calibrated, target, reduction="none")
    positive = known & (target > 0)
    negative = known & (~positive)

    # Listwise ranking among known native Stage-1 grasps in each frame. Using
    # logit(calibrated score) gives the softmax enough dynamic range while still
    # preserving the deployable multiplicative score definition.
    pred_logit = torch.logit(calibrated, eps=1e-6) / float(rank_temperature)
    teacher_logit = target / float(rank_temperature)
    mask_value = -1.0e4
    pred_logit = pred_logit.masked_fill(~known, mask_value)
    teacher_logit = teacher_logit.masked_fill(~known, mask_value)
    valid_frame = (known.sum(dim=-1) >= 2) & positive.any(dim=-1)
    teacher_prob = torch.softmax(teacher_logit, dim=-1)
    rank_map = -(teacher_prob * torch.log_softmax(pred_logit, dim=-1)).sum(dim=-1)

    def pair(value, mask):
        mask = mask.bool()
        return value.masked_select(mask).sum(), mask.sum().to(value.dtype)

    return {
        "calib_pos": pair(calibration, positive),
        "calib_neg": pair(calibration, negative),
        "rank": pair(rank_map, valid_frame),
    }


@torch.no_grad()
def confidence_metric_sums(
    raw_score: torch.Tensor,
    confidence_logit: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
):
    target = targets["target_score"].to(raw_score)
    known = targets["target_known"].to(raw_score.device).bool()
    positive = targets["target_positive"].to(raw_score.device).bool()
    gate = torch.sigmoid(confidence_logit.float())
    calibrated = raw_score.float() * gate
    B, Q = raw_score.shape

    raw_bce = F.binary_cross_entropy(raw_score.float().clamp(1e-6, 1 - 1e-6), target, reduction="none")
    cal_bce = F.binary_cross_entropy(calibrated.clamp(1e-6, 1 - 1e-6), target, reduction="none")

    def pair(value, mask):
        mask = mask.bool()
        value = value.to(raw_score)
        return value.masked_select(mask).sum(), mask.sum().to(value.dtype)

    result = {
        "rc_known_ratio": pair(known.float(), torch.ones_like(known)),
        "rc_positive_ratio": pair(positive.float(), known),
        "rc_gate_mean": pair(gate, torch.ones_like(known)),
        "rc_gate_positive": pair(gate, positive),
        "rc_gate_negative": pair(gate, known & (~positive)),
        "rc_raw_bce": pair(raw_bce, known),
        "rc_calibrated_bce": pair(cal_bce, known),
        "rc_score_suppression": pair((raw_score.float() - calibrated).clamp_min(0.0), torch.ones_like(known)),
    }

    # Ranking proxies: mean GT utility among the top-r predicted known grasps.
    for name, score in (("raw", raw_score.float()), ("cal", calibrated)):
        masked = score.masked_fill(~known, -1.0)
        for kk in (1, 10, 50):
            k = min(int(kk), Q)
            idx = torch.topk(masked, k=k, dim=-1).indices
            gathered_target = torch.gather(target, 1, idx)
            gathered_known = torch.gather(known, 1, idx)
            # Per-frame mean target over known retrieved items, then aggregate frames.
            count = gathered_known.sum(dim=-1)
            frame_valid = count > 0
            frame_mean = (gathered_target * gathered_known.to(target)).sum(dim=-1) / count.clamp_min(1).to(target)
            result[f"rc_{name}_top{kk}_target"] = pair(frame_mean, frame_valid)
    return result
