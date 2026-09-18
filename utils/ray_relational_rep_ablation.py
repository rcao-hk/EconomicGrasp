"""Representation ablations for fully trainable cross-center relational selectors.

Only the input evidence changes across variants. The complete selector
(input projection, Transformer, move gate, center selector, and delta head)
is trained from scratch for every variant.

Modes
-----
G0_current_full:
    Exact current relational token. This is the controlled reproduction.
G1_no_abs_raw:
    G0 without absolute raw CDF score s_k; keeps score residual s_k-s_0.
G2_residual_only:
    Only cross-center residual evidence plus ray offset/native identity.
    Removes absolute local features and absolute raw confidence.
G3_residual_profile:
    G2 plus compact ray-global profile statistics derived only from residuals.
G4_mean_profile:
    G2 plus channel-wise mean/std profile of angle-mean feature residuals.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.ray_relational_selective import compose_relational_tokens

REPRESENTATION_MODES: Tuple[str, ...] = (
    "G0_current_full",
    "G1_no_abs_raw",
    "G2_residual_only",
    "G3_residual_profile",
    "G4_mean_profile",
)


def _validate(
    selected_knc: torch.Tensor,
    mean_knc: torch.Tensor,
    raw_score_kn: torch.Tensor,
    offsets_mm: torch.Tensor,
    zero_index: int,
    valid_kn: torch.Tensor | None,
):
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
    if valid_kn is None:
        valid = torch.ones((K, N), dtype=torch.bool, device=selected_knc.device)
    else:
        if valid_kn.shape != (K, N):
            raise ValueError("valid_kn must be [K,N].")
        valid = valid_kn.to(device=selected_knc.device, dtype=torch.bool)
    if not bool(valid[z].all()):
        raise ValueError("Native center must be valid for every query.")
    return K, N, C, z, valid


def _masked_mean_std(x_knf: torch.Tensor, valid_kn: torch.Tensor):
    """Masked K-wise mean/std for [K,N,F], returning [N,F] each."""
    w = valid_kn.to(x_knf.dtype).unsqueeze(-1)
    count = w.sum(dim=0).clamp_min(1.0)
    mean = (x_knf * w).sum(dim=0) / count
    var = (((x_knf - mean.unsqueeze(0)) ** 2) * w).sum(dim=0) / count
    return mean, var.clamp_min(0.0).sqrt()


def _masked_scalar_stats(x_kn: torch.Tensor, valid_kn: torch.Tensor):
    """Return masked mean/std/min/max [N,4] for scalar profile [K,N]."""
    w = valid_kn.to(x_kn.dtype)
    count = w.sum(dim=0).clamp_min(1.0)
    mean = (x_kn * w).sum(dim=0) / count
    var = (((x_kn - mean.unsqueeze(0)) ** 2) * w).sum(dim=0) / count
    inf = torch.full_like(x_kn, float("inf"))
    ninf = torch.full_like(x_kn, float("-inf"))
    xmin = torch.where(valid_kn, x_kn, inf).min(dim=0).values
    xmax = torch.where(valid_kn, x_kn, ninf).max(dim=0).values
    return torch.stack((mean, var.clamp_min(0.0).sqrt(), xmin, xmax), dim=-1)


def _masked_side_mean(x_kn: torch.Tensor, mask_kn: torch.Tensor):
    w = mask_kn.to(x_kn.dtype)
    count = w.sum(dim=0)
    mean = (x_kn * w).sum(dim=0) / count.clamp_min(1.0)
    return torch.where(count > 0, mean, torch.zeros_like(mean))


def _ray_profile_stats(
    dsel_knc: torch.Tensor,
    dmean_knc: torch.Tensor,
    draw_kn: torch.Tensor,
    offsets_norm_kn: torch.Tensor,
    valid_kn: torch.Tensor,
):
    """Compact [N,13] residual-profile summary.

    Components:
      raw residual mean/std/min/max                             4
      selected-residual L2 norm mean/std/max                   3
      mean-residual L2 norm mean/std/max                       3
      positive-minus-negative side asymmetry for the above     3
    """
    C = dsel_knc.shape[-1]
    sel_norm = dsel_knc.square().mean(dim=-1).sqrt()
    mean_norm = dmean_knc.square().mean(dim=-1).sqrt()

    raw_stats = _masked_scalar_stats(draw_kn, valid_kn)
    sel_stats4 = _masked_scalar_stats(sel_norm, valid_kn)
    mean_stats4 = _masked_scalar_stats(mean_norm, valid_kn)
    sel_stats = sel_stats4[:, (0, 1, 3)]
    mean_stats = mean_stats4[:, (0, 1, 3)]

    pos = valid_kn & (offsets_norm_kn > 0)
    neg = valid_kn & (offsets_norm_kn < 0)
    raw_asym = _masked_side_mean(draw_kn, pos) - _masked_side_mean(draw_kn, neg)
    sel_asym = _masked_side_mean(sel_norm, pos) - _masked_side_mean(sel_norm, neg)
    mean_asym = _masked_side_mean(mean_norm, pos) - _masked_side_mean(mean_norm, neg)
    asym = torch.stack((raw_asym, sel_asym, mean_asym), dim=-1)
    return torch.cat((raw_stats, sel_stats, mean_stats, asym), dim=-1)


def compose_relational_ablation_tokens(
    selected_knc: torch.Tensor,
    mean_knc: torch.Tensor,
    raw_score_kn: torch.Tensor,
    offsets_mm: torch.Tensor,
    zero_index: int,
    mode: str,
    valid_kn: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose one ablation token tensor [N,K,F]."""
    if mode not in REPRESENTATION_MODES:
        raise ValueError(f"Unknown representation mode {mode!r}; choose from {REPRESENTATION_MODES}")

    K, N, C, z, valid = _validate(
        selected_knc, mean_knc, raw_score_kn, offsets_mm, zero_index, valid_kn
    )

    if mode == "G0_current_full":
        # Fail-safe controlled baseline: use the exact pre-existing composer.
        return compose_relational_tokens(
            selected_knc, mean_knc, raw_score_kn, offsets_mm, z
        )

    sel0 = selected_knc[z : z + 1].expand(K, N, C)
    mean0 = mean_knc[z : z + 1].expand(K, N, C)
    raw0 = raw_score_kn[z : z + 1].expand(K, N)
    dsel = selected_knc - sel0
    dmean = mean_knc - mean0
    draw = raw_score_kn - raw0

    scale = offsets_mm.abs().max().clamp_min(1.0)
    offset = offsets_mm.to(selected_knc).view(K, 1).expand(K, N) / scale
    native = torch.zeros((K, N), dtype=selected_knc.dtype, device=selected_knc.device)
    native[z] = 1.0
    structural = torch.stack((draw, offset, native), dim=-1)  # K,N,3

    if mode == "G1_no_abs_raw":
        token_knf = torch.cat(
            (selected_knc, dsel, mean_knc, dmean, structural),
            dim=-1,
        )
    elif mode == "G2_residual_only":
        token_knf = torch.cat((dsel, dmean, structural), dim=-1)
    elif mode == "G3_residual_profile":
        profile_nf = _ray_profile_stats(dsel, dmean, draw, offset, valid)
        profile_knf = profile_nf.unsqueeze(0).expand(K, N, profile_nf.shape[-1])
        token_knf = torch.cat((dsel, dmean, structural, profile_knf), dim=-1)
    else:  # G4_mean_profile
        prof_mean, prof_std = _masked_mean_std(dmean, valid)
        profile_nf = torch.cat((prof_mean, prof_std), dim=-1)  # N,2C
        profile_knf = profile_nf.unsqueeze(0).expand(K, N, 2 * C)
        token_knf = torch.cat((dsel, dmean, structural, profile_knf), dim=-1)

    return token_knf.permute(1, 0, 2).contiguous()


def relational_ablation_token_dim(group_feature_dim: int, mode: str) -> int:
    c = int(group_feature_dim)
    dims: Dict[str, int] = {
        "G0_current_full": 4 * c + 4,
        "G1_no_abs_raw": 4 * c + 3,
        "G2_residual_only": 2 * c + 3,
        "G3_residual_profile": 2 * c + 3 + 13,
        "G4_mean_profile": 4 * c + 3,
    }
    if mode not in dims:
        raise ValueError(f"Unknown representation mode {mode!r}; choose from {REPRESENTATION_MODES}")
    return dims[mode]
