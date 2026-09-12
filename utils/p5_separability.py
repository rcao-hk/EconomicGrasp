"""Utilities for the P5 safe-vs-repairable evidence diagnostic.

This diagnostic is intentionally post-hoc: the trained P5-v1.1 model is frozen.
It asks whether the evidence already available to P5 can distinguish proposals
that should be left unchanged from proposals that need translation repair, and
whether a perfect/learned activation gate would make the existing repair vector
useful.

Unknown queries are excluded from the binary separability task.  They are not
converted into negatives.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping

import numpy as np
import torch


FEATURE_NAMES = ("F0", "F1", "F2", "F3", "F4")


def _masked_pool(x: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Visible-token mean pool for x [B,Q,L,C]."""
    if x.dim() != 4 or visible.shape != x.shape[:3]:
        raise ValueError("Expected x [B,Q,L,C] and visible [B,Q,L].")
    w = visible.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)


def _residual_stats(residual: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Return six per-query signed/absolute residual statistics."""
    if residual.shape != visible.shape:
        raise ValueError("Residual/visibility shape mismatch.")
    w = visible.to(residual.dtype)
    n = w.sum(dim=-1).clamp_min(1.0)
    signed_mean = (residual * w).sum(dim=-1) / n
    abs_r = residual.abs()
    abs_mean = (abs_r * w).sum(dim=-1) / n
    centered = (residual - signed_mean.unsqueeze(-1)) * w
    std = torch.sqrt((centered.square().sum(dim=-1) / n).clamp_min(0.0))
    masked_abs = abs_r.masked_fill(~visible.bool(), -1.0)
    abs_max = masked_abs.max(dim=-1).values.clamp_min(0.0)
    pos_frac = ((residual > 0) & visible.bool()).to(residual.dtype).sum(dim=-1) / n
    neg_frac = ((residual < 0) & visible.bool()).to(residual.dtype).sum(dim=-1) / n
    return torch.stack((signed_mean, abs_mean, abs_max, std, pos_frac, neg_frac), dim=-1)


def build_feature_views(
    *,
    keypoint_features: torch.Tensor,
    visible: torch.Tensor,
    evidence: torch.Tensor,
    context: torch.Tensor,
    end_points: Mapping[str, torch.Tensor],
    seed_feature_dim: int,
    min_depth: float,
    max_depth: float,
) -> Dict[str, torch.Tensor]:
    """Build the five frozen feature levels used by the diagnostic.

    P5 keypoint layout is:
      RGB(3), pre-geometry(C), geometry-conditioned(C), signed depth residual(1),
      gripper-local xyz(3), visibility(1).

    F0: scalar/geometry sanity baseline.
    F1: F0 + pooled RGB + pooled pre-geometry feature.
    F2: F0 + pooled geometry-conditioned feature.
    F3: F0 + P5 gripper evidence latent.
    F4: F0 + P5 cross-ray context latent.
    """
    C = int(seed_feature_dim)
    if keypoint_features.dim() != 4 or keypoint_features.shape[-1] != 2 * C + 8:
        raise ValueError(
            f"P5 diagnostic expected keypoint dim={2*C+8}, got {tuple(keypoint_features.shape)}"
        )
    B, Q, L, _ = keypoint_features.shape
    if visible.shape != (B, Q, L):
        raise ValueError("P5 diagnostic visibility shape mismatch.")
    if evidence.shape[:2] != (B, Q) or context.shape != evidence.shape:
        raise ValueError("P5 diagnostic latent shape mismatch.")

    rgb = keypoint_features[..., :3].float()
    pre = keypoint_features[..., 3:3 + C].float()
    geom = keypoint_features[..., 3 + C:3 + 2 * C].float()
    residual = keypoint_features[..., 3 + 2 * C].float()

    rgb_pool = _masked_pool(rgb, visible)
    pre_pool = _masked_pool(pre, visible)
    geom_pool = _masked_pool(geom, visible)
    rstats = _residual_stats(residual, visible)
    visibility = visible.float().mean(dim=-1, keepdim=True)

    raw_score = end_points["p5_native_score"].float()
    z = end_points["p5_proposal_center"][..., 2].float()
    width = end_points["p5_native_width_m"].float()
    depth_idx = end_points["p5_native_depth_idx"].float()
    angle_idx = end_points["p5_native_angle_idx"].float()
    token_idx = end_points.get("kview_base_token_sel_idx", end_points.get("token_sel_idx"))
    if token_idx is None or token_idx.shape != (B, Q):
        raise ValueError("P5 diagnostic cannot recover native token indices.")
    depth_map = end_points["p5_depth_evidence"]
    H, W = depth_map.shape[-2:]
    u_norm = (token_idx % W).float() / max(float(W - 1), 1.0) * 2.0 - 1.0
    v_norm = (token_idx // W).float() / max(float(H - 1), 1.0) * 2.0 - 1.0
    z_norm = 2.0 * (z - float(min_depth)) / max(float(max_depth - min_depth), 1e-6) - 1.0
    insertion = (depth_idx + 1.0) * 0.01
    cdf = end_points.get("grasp_cdf_pred_angle_depth")
    num_angle = int(cdf.shape[-2]) if torch.is_tensor(cdf) and cdf.dim() == 5 else 12
    angle_norm = angle_idx / max(float(num_angle - 1), 1.0) * 2.0 - 1.0

    f0 = torch.cat((
        raw_score.unsqueeze(-1), z_norm.unsqueeze(-1), width.unsqueeze(-1),
        insertion.unsqueeze(-1), angle_norm.unsqueeze(-1),
        u_norm.unsqueeze(-1), v_norm.unsqueeze(-1), visibility, rstats,
    ), dim=-1)
    return {
        "F0": f0.detach(),
        "F1": torch.cat((f0, rgb_pool, pre_pool), dim=-1).detach(),
        "F2": torch.cat((f0, geom_pool), dim=-1).detach(),
        "F3": torch.cat((f0, evidence.float()), dim=-1).detach(),
        "F4": torch.cat((f0, context.float()), dim=-1).detach(),
    }


@torch.no_grad()
def build_query_outcomes(
    end_points: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
):
    """Return query labels and the causal gain of the current P5 repair vector."""
    proposal = end_points["p5_proposal_center"].float()
    rotation = end_points["p5_rotation"].float()
    pred = end_points["p5_delta_local"].float()
    raw_target = targets["raw_target_delta_local"].to(pred)
    radius = float(targets["safe_radius_m"].detach().item())
    known = targets["target_known"].bool()
    safe = targets["target_safe"].bool()
    repairable = targets["target_repairable"].bool()

    target_center = proposal + torch.matmul(rotation, raw_target.unsqueeze(-1)).squeeze(-1)
    repaired_center = proposal + torch.matmul(rotation, pred.unsqueeze(-1)).squeeze(-1)
    native_dist = (proposal - target_center).norm(dim=-1)
    repaired_dist = (repaired_center - target_center).norm(dim=-1)
    native_violation = (native_dist - radius).clamp_min(0.0)
    repaired_violation = (repaired_dist - radius).clamp_min(0.0)
    gain = native_violation - repaired_violation
    return {
        "known": known,
        "safe": safe,
        "repairable": repairable,
        "need_repair": repairable,
        "beneficial": known & (gain > 1e-6),
        "gain_m": gain,
        "native_violation_m": native_violation,
        "repaired_violation_m": repaired_violation,
        "native_safe": known & (native_dist <= radius),
        "repaired_safe": known & (repaired_dist <= radius),
        "pred_delta_abs_m": pred.norm(dim=-1),
    }


def gated_query_metrics(arrays: Mapping[str, np.ndarray], active: np.ndarray) -> Dict[str, float]:
    """Evaluate a query gate without re-running the network.

    If active[q] is true, use P5 repaired outcome; otherwise use native outcome.
    Only coherent known queries are present in the exported diagnostic shards.
    """
    active = np.asarray(active, dtype=bool)
    nviol = np.asarray(arrays["native_violation_m"], dtype=np.float64)
    rviol = np.asarray(arrays["repaired_violation_m"], dtype=np.float64)
    nsafe = np.asarray(arrays["native_safe"], dtype=bool)
    rsafe = np.asarray(arrays["repaired_safe"], dtype=bool)
    safe_cls = np.asarray(arrays["safe"], dtype=bool)
    repair_cls = np.asarray(arrays["repairable"], dtype=bool)
    if active.shape != nviol.shape:
        raise ValueError("Gate/activity shape mismatch.")
    viol = np.where(active, rviol, nviol)
    within = np.where(active, rsafe, nsafe)
    def mean_mask(x, m):
        m = np.asarray(m, dtype=bool)
        return float(np.mean(np.asarray(x)[m])) if np.any(m) else float("nan")
    return {
        "active_ratio": float(active.mean()) if active.size else float("nan"),
        "set_violation_m": float(viol.mean()) if viol.size else float("nan"),
        "within_safe": float(within.mean()) if within.size else float("nan"),
        "safe_noharm": mean_mask(within.astype(np.float32), safe_cls),
        "repairable_capture": mean_mask(within.astype(np.float32), repair_cls),
        "mean_gain_m": float((nviol - viol).mean()) if viol.size else float("nan"),
    }


def binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC via rank statistics, tie-aware without sklearn."""
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(score, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def binary_auprc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision under descending score ordering."""
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(score, dtype=np.float64)
    npos = int((y == 1).sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy == 1)
    precision = tp / np.arange(1, len(yy) + 1)
    return float(precision[yy == 1].sum() / npos)


def classification_summary(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(score, dtype=np.float64)
    out = {
        "n": int(len(y)),
        "positive_ratio": float(y.mean()) if len(y) else float("nan"),
        "auroc": binary_auroc(y, s),
        "auprc": binary_auprc(y, s),
    }
    order = np.argsort(-s, kind="mergesort")
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(math.ceil(frac * len(y)))) if len(y) else 0
        out[f"precision_at_{int(frac*100)}pct"] = (
            float(y[order[:k]].mean()) if k else float("nan")
        )
    neg = y == 0
    pos = y == 1
    for fpr in (0.01, 0.02, 0.05):
        if not np.any(neg) or not np.any(pos):
            out[f"recall_at_safe_fpr_{int(fpr*100)}pct"] = float("nan")
            continue
        neg_s = np.sort(s[neg])[::-1]
        k = max(1, int(math.ceil(fpr * len(neg_s))))
        threshold = neg_s[min(k - 1, len(neg_s) - 1)]
        out[f"recall_at_safe_fpr_{int(fpr*100)}pct"] = float((s[pos] >= threshold).mean())
    return out


def threshold_for_safe_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(score, dtype=np.float64)
    neg_s = np.sort(s[y == 0])[::-1]
    if len(neg_s) == 0:
        return float("inf")
    k = max(1, int(math.ceil(float(fpr) * len(neg_s))))
    return float(neg_s[min(k - 1, len(neg_s) - 1)])
