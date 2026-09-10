"""Tensor operations for P2-v2 ray-wise relational depth selection.

P2-v2 does not change the finite ray grid or the frozen P2-v1 per-depth grasp
predictions. It learns to compare K hypotheses *within the same image ray*.
The supervision is derived only from the existing compact CDF labels and the
5-mm geometric label-support signal already computed by P2-v1.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def _require(part: Mapping, key: str) -> torch.Tensor:
    value = part.get(key, None)
    if not torch.is_tensor(value):
        raise KeyError(f"P2-v2 requires tensor endpoint {key!r}.")
    return value


def build_selector_inputs(parts: Sequence[Mapping]):
    """Build [B,M,K,C+9] relational tokens from frozen P2-v1 outputs.

    Per-depth token = frozen local CVA grouping feature + five ray descriptors
    + raw CDF best/mean/top1-top2 margin + frozen v1 support logit.  No GT field
    enters this function, so the exact same selector inputs exist at inference.
    """
    if not parts:
        raise ValueError("P2-v2 requires at least one ray hypothesis.")
    context, desc, support, best, mean, margin, valid = [], [], [], [], [], [], []
    reference = None
    for ki, part in enumerate(parts):
        ctx = _require(part, "ray_context_feature").detach().float()  # B,M,C
        ray_desc = _require(part, "ray_descriptor").detach().float()  # B,M,5
        sup = _require(part, "ray_support_logits").detach().float()   # B,M
        cdf = _require(part, "grasp_cdf_pred_angle_depth").detach().float()
        in_range = _require(part, "ray_in_range").detach().bool()
        if ctx.dim() != 3 or ray_desc.shape != (*ctx.shape[:2], 5):
            raise ValueError(f"Malformed P2-v2 context/descriptor at k={ki}.")
        if sup.shape != ctx.shape[:2] or in_range.shape != ctx.shape[:2]:
            raise ValueError(f"Malformed support/range mask at k={ki}.")
        if cdf.dim() != 5 or cdf.shape[0] != ctx.shape[0] or cdf.shape[2] != ctx.shape[1]:
            raise ValueError(f"Malformed CDF tensor at k={ki}: {tuple(cdf.shape)}")
        shape = (ctx.shape[0], ctx.shape[1], ctx.shape[2])
        if reference is None:
            reference = shape
        elif shape != reference:
            raise ValueError("P2-v2 hypotheses do not share B/M/context dimensions.")
        utility = torch.sigmoid(cdf).mean(dim=1)  # B,M,A,D
        flat = utility.flatten(-2)
        top = torch.topk(flat, k=min(2, flat.shape[-1]), dim=-1).values
        raw_best = top[..., 0]
        raw_margin = top[..., 0] - (top[..., 1] if top.shape[-1] > 1 else top[..., 0])
        context.append(ctx)
        desc.append(ray_desc)
        support.append(sup.unsqueeze(-1))
        best.append(raw_best.unsqueeze(-1))
        mean.append(utility.mean(dim=(-1, -2)).unsqueeze(-1))
        margin.append(raw_margin.unsqueeze(-1))
        valid.append(in_range)

    context = torch.stack(context, dim=2)  # B,M,K,C
    extras = torch.cat((torch.stack(desc, dim=2),
                        torch.stack(best, dim=2),
                        torch.stack(mean, dim=2),
                        torch.stack(margin, dim=2),
                        torch.stack(support, dim=2)), dim=-1)
    in_range = torch.stack(valid, dim=2)
    if not bool(in_range.any(dim=-1).all()):
        raise RuntimeError("Every image ray must contain at least one in-range depth hypothesis.")
    features = torch.cat((context, extras), dim=-1)
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("Non-finite P2-v2 selector input.")
    return features, in_range


def build_selector_targets(parts: Sequence[Mapping]):
    """Return contextual depth target utility and supervision masks [B,M,K].

    For one depth k, GT utility is the best evaluator-aligned CDF utility over
    (angle, gripper-insertion-depth), multiplied by geometric label support.
    Off-support hypotheses are zero for *ray-depth selection*; their CDF head is
    still not trained as a negative, preserving P2-v1's label semantics.
    """
    targets, knowns, supports, cdf_supports = [], [], [], []
    for ki, part in enumerate(parts):
        bins = _require(part, "batch_grasp_cdf_bins_angle_depth").long()  # B,M,A,D
        valid = _require(part, "batch_grasp_cdf_valid_mask").bool()
        thresholds = _require(part, "batch_grasp_cdf_thresholds")
        in_range = _require(part, "ray_in_range").bool()
        support = _require(part, "ray_support_target").bool()
        known = _require(part, "ray_support_known").bool() & in_range
        if bins.shape != valid.shape or bins.shape[:2] != in_range.shape:
            raise ValueError(f"Malformed P2-v2 labels at k={ki}.")
        T = int(thresholds.numel())
        if T <= 0 or bool(((bins < 0) | (bins > T)).any()):
            raise ValueError("Invalid compact CDF bins/thresholds for P2-v2.")
        # Compact bin b means thresholds b-1,...,T-1 are successful.
        utility = torch.where(
            bins > 0,
            (float(T) - bins.float() + 1.0) / float(T),
            torch.zeros_like(bins, dtype=torch.float32),
        )
        utility = utility.masked_fill(~valid, 0.0)
        best = utility.flatten(-2).max(dim=-1).values
        target = best * support.float()
        targets.append(target)
        knowns.append(known)
        supports.append(support & known)
        cdf_supports.append(valid.any(dim=-1).any(dim=-1) & in_range)
    return {
        "target_utility": torch.stack(targets, dim=2),
        "known": torch.stack(knowns, dim=2),
        "support": torch.stack(supports, dim=2),
        "cdf_support": torch.stack(cdf_supports, dim=2),
    }


def selector_loss_sums(logits: torch.Tensor, targets: Mapping[str, torch.Tensor],
                       target_temperature: float = 0.1):
    """Global-normalizable listwise + balanced absolute-calibration statistics."""
    if target_temperature <= 0:
        raise ValueError("P2-v2 target temperature must be positive.")
    target = targets["target_utility"].to(logits).clamp(0.0, 1.0)
    known = targets["known"].to(logits.device).bool()
    if logits.shape != target.shape or known.shape != logits.shape:
        raise ValueError("P2-v2 selector logits/targets must share [B,M,K].")
    if not bool(known.any(dim=-1).all()):
        raise RuntimeError("Every ray needs at least one known depth target.")

    masked_logits = logits.masked_fill(~known, -1.0e4)
    positive_ray = known.any(dim=-1) & (target.masked_fill(~known, 0.0).max(dim=-1).values > 0)
    teacher_logits = (target / float(target_temperature)).masked_fill(~known, -1.0e4)
    teacher_prob = torch.softmax(teacher_logits, dim=-1)
    log_prob = torch.log_softmax(masked_logits, dim=-1)
    listwise_map = -(teacher_prob * log_prob).sum(dim=-1)
    listwise = (listwise_map[positive_ray].sum(), positive_ray.sum().to(logits.dtype))

    # Absolute contextual utility is needed for optional cross-ray ranking.
    # Balance positive-vs-zero depths so the many off-support hypotheses cannot
    # make a trivial all-zero selector look good.
    calib = F.binary_cross_entropy_with_logits(masked_logits, target, reduction="none")
    pos = known & (target > 0)
    neg = known & (~pos)
    return {
        "listwise": listwise,
        "calib_pos": (calib.masked_select(pos).sum(), pos.sum().to(logits.dtype)),
        "calib_neg": (calib.masked_select(neg).sum(), neg.sum().to(logits.dtype)),
    }


def relational_indices(logits: torch.Tensor, in_range: torch.Tensor) -> torch.Tensor:
    if logits.shape != in_range.shape:
        raise ValueError("P2-v2 logits/range mask shape mismatch.")
    if not bool(in_range.any(dim=-1).all()):
        raise RuntimeError("No valid depth hypothesis for at least one ray.")
    return logits.masked_fill(~in_range, -1.0e4).argmax(dim=-1)


@torch.no_grad()
def selector_metric_sums(logits: torch.Tensor, targets: Mapping[str, torch.Tensor],
                         offsets_m: torch.Tensor):
    target = targets["target_utility"].to(logits)
    known = targets["known"].to(logits.device).bool()
    support = targets["support"].to(logits.device).bool()
    cdf_support = targets["cdf_support"].to(logits.device).bool()
    if logits.shape != target.shape or logits.shape[-1] != offsets_m.numel():
        raise ValueError("P2-v2 metric shape/grid mismatch.")
    k = relational_indices(logits, known)
    selected_target = target.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    oracle_target = target.masked_fill(~known, -1.0).max(dim=-1).values.clamp_min(0.0)
    positive_ray = oracle_target > 0
    any_known = known.any(dim=-1)
    selected_support = support.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    selected_cdf = cdf_support.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    offset = offsets_m.to(logits)[k]
    zero = int((offsets_m == 0).nonzero(as_tuple=False)[0].item())
    prob = torch.softmax(logits.masked_fill(~known, -1.0e4), dim=-1)
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / max(float(torch.log(torch.tensor(float(logits.shape[-1])))), 1e-6)

    def pair(value, mask):
        value = value.to(logits.dtype)
        mask = mask.bool()
        return value.masked_select(mask).sum(), mask.sum().to(logits.dtype)

    result = {
        "v2_positive_ray_ratio": pair(positive_ray.float(), any_known),
        "v2_selected_target_utility": pair(selected_target, positive_ray),
        "v2_oracle_target_utility": pair(oracle_target, positive_ray),
        "v2_selection_regret": pair((oracle_target - selected_target).clamp_min(0.0), positive_ray),
        "v2_oracle_hit": pair((selected_target >= oracle_target - 1e-6).float(), positive_ray),
        "v2_selected_label_point_support": pair(selected_support.float(), any_known),
        "v2_selected_cdf_label_support": pair(selected_cdf.float(), any_known),
        "v2_selected_nonzero": pair((k != zero).float(), any_known),
        "v2_selected_offset_abs_m": pair(offset.abs(), any_known),
        "v2_selected_offset_signed_m": pair(offset, any_known),
        "v2_selector_entropy": pair(entropy, any_known),
        "v2_zero_probability": pair(prob[..., zero], any_known),
    }
    for i in range(offsets_m.numel()):
        result[f"v2_selected_k{i}"] = pair((k == i).float(), any_known)
    return result
