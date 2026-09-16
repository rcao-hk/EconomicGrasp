"""Utilities for exact-action relative-advantage center selection.

The selector is intentionally lightweight.  It operates on frozen Stage-1
center-conditioned local features and predicts whether an alternative physical
center is better than the native zero-offset center.  No grasp-network weight is
updated by these helpers.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RayPairwiseSelector(nn.Module):
    """Small MLP that predicts exact-utility advantage over the native center."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        # Start close to the conservative native fallback: predicted advantage 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def extract_action_conditioned_features(
    grouped: torch.Tensor,
    decode_end_points: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract selected-angle and angle-mean frozen local features.

    Parameters
    ----------
    grouped:
        Frozen local grouping output in base-major angle order [B,C,Q*A].
    decode_end_points:
        Endpoint dictionary containing CDF logits [B,T,Q,A,D].

    Returns
    -------
    selected : [B,Q,C]
        Grouping feature at the angle selected by the frozen CDF utility.
    mean : [B,Q,C]
        Mean grouping feature across all in-plane angles.
    best_angle : [B,Q]
        CDF-selected in-plane-angle index.
    """
    cdf = decode_end_points.get("grasp_cdf_pred_angle_depth", None)
    if not torch.is_tensor(cdf) or cdf.dim() != 5:
        raise ValueError("Expected grasp_cdf_pred_angle_depth [B,T,Q,A,D].")
    if grouped.dim() != 3:
        raise ValueError(f"grouped must be [B,C,Q*A], got {tuple(grouped.shape)}")
    B, T, Q, A, D = cdf.shape
    if grouped.shape[0] != B or grouped.shape[-1] != Q * A:
        raise ValueError(
            f"Grouped/CDF shape mismatch: grouped={tuple(grouped.shape)}, cdf={tuple(cdf.shape)}"
        )
    C = grouped.shape[1]
    # The deployed CDF score is the mean threshold success probability.
    utility = torch.sigmoid(cdf.float()).mean(dim=1)  # [B,Q,A,D]
    best_flat = utility.reshape(B, Q, A * D).argmax(dim=-1)
    best_angle = torch.div(best_flat, D, rounding_mode="floor")

    g = grouped.reshape(B, C, Q, A).permute(0, 2, 3, 1).contiguous()  # B,Q,A,C
    gather_idx = best_angle[..., None, None].expand(B, Q, 1, C)
    selected = torch.gather(g, dim=2, index=gather_idx).squeeze(2)
    mean = g.mean(dim=2)
    return selected.contiguous(), mean.contiguous(), best_angle.contiguous()


def compose_pairwise_features(
    selected_knc: torch.Tensor,
    mean_knc: torch.Tensor,
    raw_score_kn: torch.Tensor,
    offsets_mm: torch.Tensor,
    zero_index: int,
) -> torch.Tensor:
    """Build [K,N,F] candidate-vs-native selector features.

    Feature blocks are deliberately interpretable:
      native selected-angle feature,
      candidate selected-angle feature,
      selected-angle residual,
      native angle-mean feature,
      candidate-minus-native angle-mean residual,
      native raw CDF score,
      candidate raw CDF score,
      raw-score difference,
      normalized camera-z offset.
    """
    if selected_knc.dim() != 3 or mean_knc.shape != selected_knc.shape:
        raise ValueError("selected_knc and mean_knc must share [K,N,C].")
    K, N, C = selected_knc.shape
    if raw_score_kn.shape != (K, N):
        raise ValueError("raw_score_kn must be [K,N].")
    if offsets_mm.numel() != K:
        raise ValueError("offsets_mm length must equal K.")
    z = int(zero_index)
    if z < 0 or z >= K:
        raise ValueError("zero_index out of range.")

    native_sel = selected_knc[z : z + 1].expand(K, N, C)
    native_mean = mean_knc[z : z + 1].expand(K, N, C)
    raw0 = raw_score_kn[z : z + 1].expand(K, N)
    scale = offsets_mm.abs().max().clamp_min(1.0)
    offset = (offsets_mm.to(selected_knc).view(K, 1).expand(K, N) / scale)

    scalars = torch.stack((raw0, raw_score_kn, raw_score_kn - raw0, offset), dim=-1)
    return torch.cat(
        (
            native_sel,
            selected_knc,
            selected_knc - native_sel,
            native_mean,
            mean_knc - native_mean,
            scalars,
        ),
        dim=-1,
    ).contiguous()


def pairwise_feature_dim(group_feature_dim: int) -> int:
    return 5 * int(group_feature_dim) + 4


def balanced_sign_bce(pred: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    """Balanced beneficial-vs-harmful BCE; exact ties are ignored."""
    informative = target_delta.abs() > 1.0e-8
    if not bool(informative.any()):
        return pred.new_zeros(())
    p = pred[informative]
    y = (target_delta[informative] > 0).to(p.dtype)
    pos = y > 0.5
    neg = ~pos
    terms = []
    if bool(pos.any()):
        terms.append(F.binary_cross_entropy_with_logits(p[pos], y[pos]))
    if bool(neg.any()):
        terms.append(F.binary_cross_entropy_with_logits(p[neg], y[neg]))
    return torch.stack(terms).mean() if terms else pred.new_zeros(())


def listwise_exact_utility_loss(
    pred_kn: torch.Tensor,
    utility_kn: torch.Tensor,
    valid_kn: torch.Tensor,
    target_temperature: float = 0.15,
) -> torch.Tensor:
    """Listwise CE over K only for queries whose exact utility varies with center."""
    if not (pred_kn.shape == utility_kn.shape == valid_kn.shape) or pred_kn.dim() != 2:
        raise ValueError("pred_kn, utility_kn and valid_kn must share [K,N].")
    masked_u = utility_kn.masked_fill(~valid_kn, -1.0e9)
    masked_p = pred_kn.masked_fill(~valid_kn, -1.0e9)
    u_max = masked_u.max(dim=0).values
    u_min = utility_kn.masked_fill(~valid_kn, 1.0e9).min(dim=0).values
    informative = (u_max - u_min) > 1.0e-8
    if not bool(informative.any()):
        return pred_kn.new_zeros(())
    temp = max(float(target_temperature), 1.0e-4)
    target = torch.softmax(masked_u[:, informative] / temp, dim=0)
    logp = torch.log_softmax(masked_p[:, informative] / temp, dim=0)
    return -(target * logp).sum(dim=0).mean()


def select_with_native_fallback(
    pred_delta_kn: np.ndarray,
    valid_kn: np.ndarray,
    zero_index: int,
    threshold: float,
) -> np.ndarray:
    """Choose an alternative only when predicted advantage exceeds threshold."""
    p = np.asarray(pred_delta_kn, dtype=np.float64)
    v = np.asarray(valid_kn, dtype=bool)
    if p.shape != v.shape or p.ndim != 2:
        raise ValueError("pred_delta_kn and valid_kn must share [K,N].")
    K, N = p.shape
    z = int(zero_index)
    if z < 0 or z >= K or not np.all(v[z]):
        raise ValueError("Native zero-offset candidate must be valid for every query.")
    score = np.where(v, p, -np.inf)
    score[z] = -np.inf  # alternatives only
    best_k = np.argmax(score, axis=0)
    best = score[best_k, np.arange(N)]
    return np.where(best > float(threshold), best_k, z).astype(np.int64)
