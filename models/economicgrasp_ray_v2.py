"""P2-v2: relational cross-depth selection over the frozen P2-v1 ray field.

The finite ray grid, RGB/DPT frontend, image-FPS queries, ViewNet, local CVA
features and per-depth CDF/width/support predictions are inherited from a trained
P2-v1 checkpoint and frozen.  A tiny Transformer compares the K depth hypotheses
belonging to the same image ray before selecting one physical grasp center.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .economicgrasp_ray import RayConditionedGrasp
from .ray_grasp_ops import select_hypotheses
from .ray_grasp_v2_ops import build_selector_inputs, build_selector_targets, relational_indices

RAY_V2_CONTRACT_VERSION = 2


class CrossDepthRaySelector(nn.Module):
    """Relational scorer over K fixed physical-center hypotheses of one ray."""

    def __init__(self, context_dim: int, hidden: int = 128, layers: int = 2,
                 heads: int = 4, dropout: float = 0.10, zero_bias: float = 0.5):
        super().__init__()
        if context_dim <= 0 or hidden <= 0 or layers <= 0 or heads <= 0:
            raise ValueError("P2-v2 selector dimensions/layers/heads must be positive.")
        if hidden % heads != 0:
            raise ValueError("P2-v2 hidden dimension must be divisible by heads.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("P2-v2 dropout must lie in [0,1).")
        self.context_dim = int(context_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.input_dim = self.context_dim + 9

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=self.heads,
            dim_feedforward=2 * self.hidden,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.layers)
        self.output_norm = nn.LayerNorm(self.hidden)
        self.score_head = nn.Linear(self.hidden, 1)
        # Exact conservative start: the learned branch initially ties all K,
        # while a small explicit zero-depth bias makes the native center win.
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        self.zero_logit_bias = nn.Parameter(torch.tensor(float(zero_bias)))

    def forward(self, parts, offsets_m: torch.Tensor):
        features, in_range = build_selector_inputs(parts)
        B, M, K, F = features.shape
        if F != self.input_dim or K != offsets_m.numel():
            raise RuntimeError(
                f"P2-v2 selector input/grid mismatch: F={F}/{self.input_dim}, K={K}/{offsets_m.numel()}."
            )
        x = self.input_proj(features).reshape(B * M, K, self.hidden)
        pad = (~in_range).reshape(B * M, K)
        x = self.encoder(x, src_key_padding_mask=pad)
        logits = self.score_head(self.output_norm(x)).squeeze(-1).reshape(B, M, K)
        zero_idx = torch.where(offsets_m == 0)[0]
        if zero_idx.numel() != 1:
            raise RuntimeError("P2-v2 requires exactly one zero-offset hypothesis.")
        zero_mask = torch.zeros(K, device=logits.device, dtype=logits.dtype)
        zero_mask[int(zero_idx.item())] = 1.0
        logits = logits + self.zero_logit_bias * zero_mask.view(1, 1, K)
        logits = logits.masked_fill(~in_range, -1.0e4)
        probability = torch.softmax(logits, dim=-1)
        if not bool(torch.isfinite(probability).all()):
            raise FloatingPointError("Non-finite P2-v2 selector probability.")
        return logits, probability, in_range


class RayConditionedGraspV2(RayConditionedGrasp):
    """Frozen trained P2-v1 + trainable ray-wise relational selector."""

    def __init__(self, *args: Any, v2_hidden: int = 128, v2_layers: int = 2,
                 v2_heads: int = 4, v2_dropout: float = 0.10,
                 v2_zero_bias: float = 0.5, **kwargs: Any):
        super().__init__(*args, **kwargs)
        context_dim = int(
            self.kview_grasp_module.core.decoder.base.input_proj.in_channels
        )
        self.ray_cross_depth_selector = CrossDepthRaySelector(
            context_dim=context_dim,
            hidden=v2_hidden,
            layers=v2_layers,
            heads=v2_heads,
            dropout=v2_dropout,
            zero_bias=v2_zero_bias,
        )
        self.v2_hidden = int(v2_hidden)
        self.v2_layers = int(v2_layers)
        self.v2_heads = int(v2_heads)
        self.v2_dropout = float(v2_dropout)

        # Freeze every P2-v1 parameter. Only relational comparison is learned.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ray_cross_depth_selector.parameters():
            parameter.requires_grad_(True)
        self.train(False)

    def train(self, mode: bool = True):
        # Parent construction calls self.train(False) before v2 selector exists,
        # so this method must tolerate that phase.  Frozen P2-v1 stays eval even
        # while the new selector uses Transformer dropout during training.
        nn.Module.train(self, False)
        self.training = bool(mode)
        if hasattr(self, "ray_cross_depth_selector"):
            self.ray_cross_depth_selector.train(mode)
        return self

    def forward(self, batch, with_labels: bool = False):
        out = super().forward(batch, with_labels=with_labels)
        logits, probability, in_range = self.ray_cross_depth_selector(
            out["ray_hypotheses"], out["ray_offsets_m"]
        )
        out["ray_v2_selector_logits"] = logits
        out["ray_v2_selector_probability"] = probability
        out["ray_v2_in_range"] = in_range
        if with_labels:
            targets = build_selector_targets(out["ray_hypotheses"])
            out["ray_v2_target_utility"] = targets["target_utility"]
            out["ray_v2_target_known"] = targets["known"]
            out["ray_v2_target_support"] = targets["support"]
            out["ray_v2_target_cdf_support"] = targets["cdf_support"]
        return out

    def load_v1(self, state) -> None:
        clean = {key.removeprefix("module."): value for key, value in state.items()}
        result = self.load_state_dict(clean, strict=False)
        missing = [key for key in result.missing_keys
                   if not key.startswith("ray_cross_depth_selector.")]
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"P2-v1 -> v2 initialization mismatch: missing={missing}, unexpected={unexpected}"
            )


def _targets_from_endpoints(end_points):
    return {
        "target_utility": end_points["ray_v2_target_utility"],
        "known": end_points["ray_v2_target_known"],
        "support": end_points["ray_v2_target_support"],
        "cdf_support": end_points["ray_v2_target_cdf_support"],
    }


@torch.no_grad()
def decode_ray_grasps_v2(end_points, selection: str = "relational",
                         final_score: str = "raw"):
    """Decode one grasp per image ray while keeping the candidate budget fixed.

    selection:
      relational -- P2-v2 Transformer chooses physical center k.
      zero       -- native deterministic center control.
      raw        -- P2-v1 raw-CDF cross-depth selection control.
      supported  -- P2-v1 support*CDF cross-depth control.

    final_score:
      raw        -- preserve selected candidate's original CDF score for
                    cross-image-ray ranking (recommended first P2-v2 test).
      contextual -- use sigmoid(v2 contextual utility logit).
      product    -- multiply raw CDF score by contextual utility.
    """
    from models.economicgrasp_bip3d import pred_decode_center_view_angle

    parts = end_points["ray_hypotheses"]
    offsets = end_points["ray_offsets_m"]
    logits = end_points["ray_v2_selector_logits"]
    in_range = end_points["ray_v2_in_range"].bool()
    if selection == "relational":
        k = relational_indices(logits, in_range)
    elif selection == "zero":
        zero = torch.where(offsets == 0)[0]
        if zero.numel() != 1:
            raise RuntimeError("Exactly one zero offset is required.")
        k = torch.full_like(logits[..., 0], int(zero.item()), dtype=torch.long)
    elif selection in ("raw", "supported"):
        k, _, _ = select_hypotheses(
            parts, offsets, supported=(selection == "supported"), selection="best"
        )
    else:
        raise ValueError("selection must be relational, zero, raw or supported.")

    decoded = [pred_decode_center_view_angle(part, use_cdf=True) for part in parts]
    outputs = []
    selected_context = torch.sigmoid(logits).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    for bi in range(k.shape[0]):
        all_grasps = torch.stack([grasp[bi] for grasp in decoded], dim=1)  # M,K,17
        if all_grasps.shape[:2] != (k.shape[1], len(parts)):
            raise RuntimeError("P2-v2 decoder changed per-ray candidate order/count.")
        row = torch.arange(k.shape[1], device=k.device)
        pred = all_grasps[row, k[bi]].clone()
        raw_score = pred[:, 0].clone()
        contextual = selected_context[bi]
        if final_score == "raw":
            score = raw_score
        elif final_score == "contextual":
            score = contextual
        elif final_score == "product":
            score = raw_score * contextual
        else:
            raise ValueError("final_score must be raw, contextual or product.")
        pred[:, 0] = score
        outputs.append(pred)
    return outputs, k, selected_context
