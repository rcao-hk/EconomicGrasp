"""P5-v1.1 repair-to-valid-set objectives and diagnostics.

P5-v1.1 changes the *training semantics*, not the P5 inference architecture.
The nearest annotated positive grasp remains the reference geometry, but a
proposal already within ``safe_radius_m`` of that reference is treated as a
valid-set member and receives an identity target rather than a point-regression
target.  Queries without coherent positive supervision remain unknown.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def prepare_repair_to_set_targets(
    base_targets: Mapping[str, torch.Tensor],
    *,
    safe_radius_m: float = 0.005,
):
    """Convert P5 point-repair targets into valid-set targets.

    ``base_targets`` are the outputs already produced by P5-v1.  Their
    ``target_known`` mask means: nearest annotation is inside the repair radius
    and the frozen Stage-1 (view, angle, insertion-depth) operation is positive
    at that annotation.  P5-v1.1 keeps that coherent mask, but partitions it:

      safe       : distance <= safe radius -> target residual is exactly zero
      repairable : safe radius < distance <= repair radius -> repair to label
      unknown    : no coherent positive repair target -> weak identity only

    Importantly, unknown is never converted into a grasp-quality negative.
    """
    safe_radius_m = float(safe_radius_m)
    if safe_radius_m < 0:
        raise ValueError("P5-v1.1 safe_radius_m must be non-negative.")
    raw_delta = base_targets["target_delta_local"]
    known = base_targets["target_known"].bool()
    distance = base_targets["target_distance_m"].to(raw_delta)
    if raw_delta.dim() != 3 or raw_delta.shape[-1] != 3:
        raise ValueError("P5-v1.1 target_delta_local must be [B,Q,3].")
    if known.shape != raw_delta.shape[:2] or distance.shape != known.shape:
        raise ValueError("P5-v1.1 target masks/distances must be [B,Q].")

    safe = known & (distance <= safe_radius_m)
    repairable = known & (~safe)
    unknown = ~known
    target_delta = torch.where(
        safe.unsqueeze(-1), torch.zeros_like(raw_delta), raw_delta
    )
    return {
        "target_delta_local": target_delta.detach(),
        "raw_target_delta_local": raw_delta.detach(),
        "target_known": known.detach(),
        "target_safe": safe.detach(),
        "target_repairable": repairable.detach(),
        "target_unknown": unknown.detach(),
        "target_distance_m": distance.detach(),
        "target_utility": base_targets["target_utility"].detach(),
        "safe_radius_m": raw_delta.new_tensor(safe_radius_m),
    }


def repair_to_set_loss_sums(
    pred_delta_local: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    *,
    beta_m: float = 0.005,
):
    """Return unweighted sufficient statistics for P5-v1.1.

    Weighting is intentionally left to the trainer so checkpoint metadata can
    state the exact optimization objective.
    """
    if beta_m <= 0:
        raise ValueError("P5-v1.1 beta_m must be positive.")
    target = targets["target_delta_local"].to(pred_delta_local)
    if pred_delta_local.shape != target.shape:
        raise ValueError("P5-v1.1 prediction/target shape mismatch.")
    safe = targets["target_safe"].bool()
    repairable = targets["target_repairable"].bool()
    unknown = targets["target_unknown"].bool()

    repair_map = F.smooth_l1_loss(
        pred_delta_local, target, beta=float(beta_m), reduction="none"
    ).sum(-1)
    identity_map = F.smooth_l1_loss(
        pred_delta_local, torch.zeros_like(pred_delta_local),
        beta=float(beta_m), reduction="none"
    ).sum(-1)

    def pair(value, mask):
        mask = mask.bool()
        return value.masked_select(mask).sum(), mask.sum().to(value.dtype)

    return {
        "repair": pair(repair_map, repairable),
        "safe_identity": pair(identity_map, safe),
        "unknown_identity": pair(identity_map, unknown),
    }


@torch.no_grad()
def repair_to_set_metric_sums(
    pred_delta_local: torch.Tensor,
    rotation: torch.Tensor,
    proposal_center: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
):
    """Mechanism metrics for a set-valued center target.

    The annotation center is retained only for measuring distance.  The task
    target is the ball of radius ``safe_radius_m`` around that center, so the
    primary error is distance *outside* that valid set rather than point error.
    """
    raw_target = targets["raw_target_delta_local"].to(pred_delta_local)
    known = targets["target_known"].bool()
    safe = targets["target_safe"].bool()
    repairable = targets["target_repairable"].bool()
    unknown = targets["target_unknown"].bool()
    radius = float(targets["safe_radius_m"].detach().item())
    if rotation.shape != (*proposal_center.shape[:2], 3, 3):
        raise ValueError("P5-v1.1 rotation shape mismatch.")

    target_center = proposal_center + torch.matmul(
        rotation, raw_target.unsqueeze(-1)
    ).squeeze(-1)
    repaired_center = proposal_center + torch.matmul(
        rotation, pred_delta_local.unsqueeze(-1)
    ).squeeze(-1)
    native_dist = (proposal_center - target_center).norm(dim=-1)
    repaired_dist = (repaired_center - target_center).norm(dim=-1)
    native_violation = (native_dist - radius).clamp_min(0.0)
    repaired_violation = (repaired_dist - radius).clamp_min(0.0)
    improvement = native_violation - repaired_violation
    pred_abs = pred_delta_local.norm(dim=-1)

    def pair(value, mask):
        mask = mask.bool()
        value = value.to(pred_delta_local)
        return value.masked_select(mask).sum(), mask.sum().to(value.dtype)

    all_q = torch.ones_like(known)
    result = {
        "p5v11_target_known_ratio": pair(known.float(), all_q),
        "p5v11_safe_ratio": pair(safe.float(), known),
        "p5v11_repairable_ratio": pair(repairable.float(), known),
        "p5v11_native_label_dist_m": pair(native_dist, known),
        "p5v11_repaired_label_dist_m": pair(repaired_dist, known),
        "p5v11_native_set_violation_m": pair(native_violation, known),
        "p5v11_repaired_set_violation_m": pair(repaired_violation, known),
        "p5v11_set_improvement_m": pair(improvement, known),
        "p5v11_native_within_safe": pair((native_dist <= radius).float(), known),
        "p5v11_repaired_within_safe": pair((repaired_dist <= radius).float(), known),
        "p5v11_safe_noharm_ratio": pair((repaired_dist <= radius).float(), safe),
        "p5v11_repairable_capture_ratio": pair((repaired_dist <= radius).float(), repairable),
        "p5v11_pred_delta_safe_m": pair(pred_abs, safe),
        "p5v11_pred_delta_repairable_m": pair(pred_abs, repairable),
        "p5v11_pred_delta_unknown_m": pair(pred_abs, unknown),
        "p5v11_pred_delta_all_m": pair(pred_abs, all_q),
    }
    return result
