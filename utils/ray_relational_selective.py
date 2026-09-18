"""Cross-center relational selective correction for ray-depth grasp hypotheses.

This module is deliberately downstream of the frozen Stage-1 K-center reread.
It treats the K centers on one camera ray as one ordered set rather than scoring
each center independently.

The decision is factorized into:
  1) move / keep-native gate;
  2) conditional alternative-center ranking.

Exact-action supervision is used only from the previously mined cache.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compose_relational_tokens(
    selected_knc: torch.Tensor,
    mean_knc: torch.Tensor,
    raw_score_kn: torch.Tensor,
    offsets_mm: torch.Tensor,
    zero_index: int,
) -> torch.Tensor:
    """Build relational ray tokens [N,K,F].

    Per-center token:
      [F_sel_k,
       F_sel_k - F_sel_0,
       F_mean_k,
       F_mean_k - F_mean_0,
       raw_score_k,
       raw_score_k - raw_score_0,
       normalized_offset,
       native_flag]

    Absolute local evidence lets the encoder judge physical plausibility at each
    center. Relative blocks expose displacement-conditioned changes explicitly.
    The ordered K-set is then contextualized jointly by the Transformer.
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

    sel0 = selected_knc[z : z + 1].expand(K, N, C)
    mean0 = mean_knc[z : z + 1].expand(K, N, C)
    raw0 = raw_score_kn[z : z + 1].expand(K, N)
    scale = offsets_mm.abs().max().clamp_min(1.0)
    offset = offsets_mm.to(selected_knc).view(K, 1).expand(K, N) / scale
    native = torch.zeros((K, N), dtype=selected_knc.dtype, device=selected_knc.device)
    native[z] = 1.0
    scalars = torch.stack((raw_score_kn, raw_score_kn - raw0, offset, native), dim=-1)

    token_knf = torch.cat(
        (
            selected_knc,
            selected_knc - sel0,
            mean_knc,
            mean_knc - mean0,
            scalars,
        ),
        dim=-1,
    )
    return token_knf.permute(1, 0, 2).contiguous()


def relational_token_dim(group_feature_dim: int) -> int:
    return 4 * int(group_feature_dim) + 4


class CrossCenterRelationalSelective(nn.Module):
    """Joint K-center encoder with a move gate and conditional center selector."""

    def __init__(
        self,
        token_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.token_dim = int(token_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)

        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.selector_head = nn.Linear(self.d_model, 1)
        self.delta_head = nn.Linear(self.d_model, 1)
        self.gate_head = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        # Conservative initialization: no initial preference to move or rank.
        nn.init.zeros_(self.selector_head.weight)
        nn.init.zeros_(self.selector_head.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def forward(
        self,
        tokens_nkf: torch.Tensor,
        valid_nk: torch.Tensor,
        zero_index: int,
    ) -> Dict[str, torch.Tensor]:
        if tokens_nkf.dim() != 3:
            raise ValueError("tokens_nkf must be [N,K,F].")
        N, K, Fdim = tokens_nkf.shape
        if Fdim != self.token_dim:
            raise ValueError(f"Token dim mismatch: got {Fdim}, expected {self.token_dim}.")
        if valid_nk.shape != (N, K):
            raise ValueError("valid_nk must be [N,K].")
        z = int(zero_index)
        if z < 0 or z >= K:
            raise ValueError("zero_index out of range.")
        if not bool(valid_nk[:, z].all()):
            raise ValueError("Native center must be valid for every query.")

        x = self.input_proj(tokens_nkf)
        h = self.encoder(x, src_key_padding_mask=~valid_nk)
        selector_logits = self.selector_head(h).squeeze(-1)
        delta_pred = self.delta_head(h).squeeze(-1)

        valid_f = valid_nk.to(h.dtype).unsqueeze(-1)
        pooled = (h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        native_h = h[:, z]
        gate_logit = self.gate_head(torch.cat((native_h, pooled), dim=-1)).squeeze(-1)

        return {
            "context": h,
            "gate_logit": gate_logit,
            "selector_logits": selector_logits,
            "delta_pred": delta_pred,
        }


def balanced_move_bce(gate_logit: torch.Tensor, move_target: torch.Tensor) -> torch.Tensor:
    """Class-balanced BCE over ray queries."""
    if gate_logit.shape != move_target.shape:
        raise ValueError("gate_logit and move_target must share shape.")
    y = move_target.to(gate_logit.dtype)
    pos = y > 0.5
    neg = ~pos
    terms = []
    if bool(pos.any()):
        terms.append(F.binary_cross_entropy_with_logits(gate_logit[pos], y[pos]))
    if bool(neg.any()):
        terms.append(F.binary_cross_entropy_with_logits(gate_logit[neg], y[neg]))
    if not terms:
        return gate_logit.sum() * 0.0
    return torch.stack(terms).mean()


def relational_exact_action_losses(
    outputs: Dict[str, torch.Tensor],
    utility_kn: torch.Tensor,
    valid_kn: torch.Tensor,
    zero_index: int,
    *,
    selector_temperature: float = 0.15,
    delta_beta: float = 0.1,
    eps: float = 1.0e-8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute move-gate, conditional K-ranking and relative-utility losses.

    The gate target is strict: move only when some valid non-native action has
    exact utility greater than the native action. Selector loss is applied only
    to such move-positive queries. This prevents exact-utility ties from teaching
    gratuitous center changes.
    """
    if utility_kn.dim() != 2 or valid_kn.shape != utility_kn.shape:
        raise ValueError("utility_kn and valid_kn must share [K,N].")
    K, N = utility_kn.shape
    z = int(zero_index)
    gate = outputs["gate_logit"]
    logits_nk = outputs["selector_logits"]
    delta_nk = outputs["delta_pred"]
    if gate.shape != (N,) or logits_nk.shape != (N, K) or delta_nk.shape != (N, K):
        raise ValueError("Relational output shape mismatch.")

    utility_nk = utility_kn.transpose(0, 1).contiguous()
    valid_nk = valid_kn.transpose(0, 1).contiguous()
    native_u = utility_nk[:, z]
    alt_valid = valid_nk.clone()
    alt_valid[:, z] = False
    masked_alt_u = utility_nk.masked_fill(~alt_valid, -1.0e9)
    best_alt_u = masked_alt_u.max(dim=1).values
    move_target = best_alt_u > (native_u + float(eps))

    gate_loss = balanced_move_bce(gate, move_target)

    # Conditional listwise selector: only queries for which moving is physically
    # beneficial. Native is excluded by construction.
    if bool(move_target.any()):
        temp = max(float(selector_temperature), 1.0e-4)
        sel_logits = logits_nk[move_target].masked_fill(~alt_valid[move_target], -1.0e9)
        target_logits = utility_nk[move_target].masked_fill(~alt_valid[move_target], -1.0e9)
        target = torch.softmax(target_logits / temp, dim=1)
        logp = torch.log_softmax(sel_logits / temp, dim=1)
        selector_loss = -(target * logp).sum(dim=1).mean()
    else:
        selector_loss = logits_nk.sum() * 0.0

    target_delta = utility_nk - native_u[:, None]
    reg_mask = alt_valid
    if bool(reg_mask.any()):
        delta_loss = F.smooth_l1_loss(
            delta_nk[reg_mask],
            target_delta[reg_mask],
            beta=float(delta_beta),
        )
    else:
        delta_loss = delta_nk.sum() * 0.0

    losses = {
        "gate": gate_loss,
        "selector": selector_loss,
        "delta": delta_loss,
    }
    targets = {
        "move_target": move_target,
        "target_delta_nk": target_delta,
        "best_alt_utility": best_alt_u,
        "native_utility": native_u,
    }
    return losses, targets


@torch.no_grad()
def select_relational_correction(
    gate_logit_n: torch.Tensor | np.ndarray,
    selector_logits_nk: torch.Tensor | np.ndarray,
    valid_nk: torch.Tensor | np.ndarray,
    zero_index: int,
    move_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select native or one alternative using gate + conditional K ranking.

    Returns
    -------
    selected_k : [N]
    best_alt_k : [N]
    move_prob : [N]
    """
    if torch.is_tensor(gate_logit_n):
        gate = gate_logit_n.detach().float().cpu().numpy()
    else:
        gate = np.asarray(gate_logit_n, dtype=np.float64)
    if torch.is_tensor(selector_logits_nk):
        logits = selector_logits_nk.detach().float().cpu().numpy()
    else:
        logits = np.asarray(selector_logits_nk, dtype=np.float64)
    if torch.is_tensor(valid_nk):
        valid = valid_nk.detach().cpu().numpy().astype(bool)
    else:
        valid = np.asarray(valid_nk, dtype=bool)

    if logits.ndim != 2 or valid.shape != logits.shape:
        raise ValueError("selector_logits_nk and valid_nk must share [N,K].")
    N, K = logits.shape
    if gate.shape != (N,):
        raise ValueError("gate_logit_n must be [N].")
    z = int(zero_index)
    if z < 0 or z >= K or not np.all(valid[:, z]):
        raise ValueError("Native center must be valid.")

    alt_valid = valid.copy()
    alt_valid[:, z] = False
    has_alt = alt_valid.any(axis=1)
    alt_score = np.where(alt_valid, logits, -np.inf)
    best_alt = np.argmax(alt_score, axis=1).astype(np.int64)
    best_alt = np.where(has_alt, best_alt, z).astype(np.int64)
    move_prob = 1.0 / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
    do_move = (move_prob > float(move_threshold)) & has_alt
    selected = np.where(do_move, best_alt, z).astype(np.int64)
    return selected, best_alt, move_prob.astype(np.float32)
