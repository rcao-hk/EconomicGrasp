"""Feature subsets for cache-only pairwise center-selector ablations.

All variants use the same 3-linear-layer ``RayPairwiseSelector``.  Only the
input evidence changes.  This keeps optimization/target semantics fixed while
asking which frozen Stage-1 evidence transfers across scenes.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

FEATURE_MODES: Tuple[str, ...] = (
    "raw_offset",
    "selected_residual",
    "mean_residual",
    "selected_mean",
    "full",
)


def compose_feature_ablation(
    selected_knc: torch.Tensor,
    mean_knc: torch.Tensor,
    raw_score_kn: torch.Tensor,
    offsets_mm: torch.Tensor,
    zero_index: int,
    mode: str,
) -> torch.Tensor:
    """Compose one controlled selector input [K,N,F].

    Modes
    -----
    raw_offset:
        [score0, scorek, scorek-score0, normalized_offset]
    selected_residual:
        [F0_sel, Fk_sel, Fk_sel-F0_sel]
    mean_residual:
        [F0_mean, Fk_mean-F0_mean]
    selected_mean:
        [F0_sel, Fk_sel, Fk_sel-F0_sel, F0_mean, Fk_mean-F0_mean]
    full:
        selected_mean plus the four raw/offset scalars (current selector input).
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode {mode!r}; choose from {FEATURE_MODES}")
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
    offset = offsets_mm.to(selected_knc).view(K, 1).expand(K, N) / scale
    scalars = torch.stack((raw0, raw_score_kn, raw_score_kn - raw0, offset), dim=-1)

    selected_blocks = (native_sel, selected_knc, selected_knc - native_sel)
    mean_blocks = (native_mean, mean_knc - native_mean)

    if mode == "raw_offset":
        blocks = (scalars,)
    elif mode == "selected_residual":
        blocks = selected_blocks
    elif mode == "mean_residual":
        blocks = mean_blocks
    elif mode == "selected_mean":
        blocks = selected_blocks + mean_blocks
    else:  # full
        blocks = selected_blocks + mean_blocks + (scalars,)
    return torch.cat(blocks, dim=-1).contiguous()


def feature_ablation_dim(group_feature_dim: int, mode: str) -> int:
    c = int(group_feature_dim)
    dims: Dict[str, int] = {
        "raw_offset": 4,
        "selected_residual": 3 * c,
        "mean_residual": 2 * c,
        "selected_mean": 5 * c,
        "full": 5 * c + 4,
    }
    if mode not in dims:
        raise ValueError(f"Unknown feature mode {mode!r}; choose from {FEATURE_MODES}")
    return dims[mode]
