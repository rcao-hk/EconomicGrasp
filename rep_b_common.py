"""Shared utilities for Rep-B fixed-action experiments."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rep_a_common import cdf_targets, tensors


def rep_b_tensors(d: dict, device):
    out = tensors(d, device)
    out["valid"] = torch.from_numpy(np.asarray(d["valid"], dtype=np.bool_).copy()).to(device)
    out["offsets_mm"] = torch.from_numpy(np.asarray(d["offsets_mm"], dtype=np.float32).copy()).to(device)
    out["zero_index"] = int(np.asarray(d["zero_index"]).reshape(-1)[0])
    return out


def pairwise_ranking_loss(
    logits: torch.Tensor,
    utility_np,
    valid: torch.Tensor,
    *,
    min_gap: float = 1e-4,
    temperature: float = 0.1,
):
    """Within-ray pairwise utility ranking loss.

    This is optional and disabled in the formal first Rep-B protocol by default.
    It is provided for a later controlled loss ablation without changing code.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    pred = logits.sigmoid().mean(-1)  # [K,Q]
    utility = torch.as_tensor(utility_np, dtype=pred.dtype, device=pred.device)
    k = pred.shape[0]
    tri = torch.triu(torch.ones((k, k), dtype=torch.bool, device=pred.device), diagonal=1)
    du = utility[:, None, :] - utility[None, :, :]
    dp = pred[:, None, :] - pred[None, :, :]
    pair_valid = tri[..., None] & valid[:, None, :] & valid[None, :, :] & (du.abs() > min_gap)
    if not bool(pair_valid.any()):
        return pred.sum() * 0.0
    sign = du.sign()
    return F.softplus(-(sign * dp) / temperature)[pair_valid].mean()


def cdf_bce_loss(logits: torch.Tensor, friction_np, valid: torch.Tensor):
    target = torch.from_numpy(cdf_targets(friction_np)).to(logits.device)
    if target.shape != logits.shape:
        raise ValueError(f"CDF target/logit mismatch: {target.shape} vs {logits.shape}")
    if not bool(valid.any()):
        raise ValueError("No valid Rep-B actions")
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid])
