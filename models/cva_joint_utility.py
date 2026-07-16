"""Evaluator-aligned joint utility scorer for first-generation CVA Transformer.

The scorer operates on an explicit center-view-angle-depth candidate rather
than on the angle candidate after depth has already been collapsed.  It is
intended for a two-stage training protocol:

1. Freeze the existing EconomicGrasp-DPT + CVA checkpoint and mine candidate
   features together with exact GraspNet CAD/DexNet evaluator outcomes.
2. Train this lightweight scorer on the mined cache.  The primary output is a
   six-threshold cumulative success distribution aligned with the benchmark's
   friction thresholds {0.2, ..., 1.2}.

The six output probabilities are monotonic by construction.  Their mean is the
candidate's expected benchmark utility and can be used directly for
cross-view/angle/depth comparison before candidate collapse and NMS.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FRICTION_THRESHOLDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


@dataclass
class JointUtilityScorerConfig:
    angle_feature_dim: int = 256
    score_logit_dim: int = 6
    depth_logit_dim: int = 5  # four physical depths + one dummy class
    num_angles: int = 12
    num_depths: int = 4
    max_view_rank: int = 8
    hidden_dim: int = 256
    aux_hidden_dim: int = 128
    num_residual_blocks: int = 3
    dropout: float = 0.10
    use_legacy_collision: bool = True
    use_geometry: bool = True
    use_view_score: bool = True
    monotonic_cdf: bool = True
    monotonic_increment_bias: float = -4.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)


class EvaluatorAlignedJointUtilityScorer(nn.Module):
    """Score explicit CVA candidates against GraspNet evaluator utility.

    Required flat candidate batch fields
    ------------------------------------
    angle_feature:
        [N, C] hidden feature exposed after CVA angle self-attention.
    score_logits:
        [N, 6] legacy friction-quality logits for the angle candidate.
    depth_logits:
        [N, D+1] legacy depth logits for the angle candidate.
    width_raw:
        [N] or [N,1], legacy width output in the repository's x10 scale.
    angle_id, depth_id, view_rank:
        [N] integer candidate indices.

    Optional fields
    ---------------
    legacy_collision_logit:
        [N] angle-level collision logit.  It is treated only as input evidence,
        not as the final validity probability.
    view_score:
        [N] original (unpinned) ViewNet score for the candidate view.
    center_xyz, view_xyz:
        [N,3] metric center and view direction.

    Returns
    -------
    cdf_logits:
        [N,6] logits for P(success at friction threshold <= mu_j).
    cdf_prob:
        [N,6], monotonic non-decreasing probabilities.
    utility:
        [N], mean CDF probability, exactly matching the evaluator's average
        friction utility in expectation.
    collision_logit, empty_logit:
        [N] auxiliary explicit invalidity predictions.
    candidate_feature:
        [N,H] fused feature, useful for diagnostics.
    """

    def __init__(self, config: Optional[JointUtilityScorerConfig] = None) -> None:
        super().__init__()
        self.config = config or JointUtilityScorerConfig()
        c = self.config

        self.angle_norm = nn.LayerNorm(c.angle_feature_dim)
        self.angle_proj = nn.Sequential(
            nn.Linear(c.angle_feature_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        emb_dim = max(16, c.hidden_dim // 8)
        self.angle_embed = nn.Embedding(c.num_angles, emb_dim)
        self.depth_embed = nn.Embedding(c.num_depths, emb_dim)
        self.view_rank_embed = nn.Embedding(c.max_view_rank, emb_dim)

        # Continuous auxiliary feature layout.  Keeping this construction
        # explicit makes cache compatibility and checkpoint inspection simple.
        aux_dim = c.score_logit_dim + c.depth_logit_dim
        aux_dim += 1  # selected-depth probability
        aux_dim += 1  # expected legacy quality
        aux_dim += 1  # width raw
        aux_dim += 2  # sin/cos angle
        aux_dim += 1  # normalized depth id
        aux_dim += 1 if c.use_legacy_collision else 0
        aux_dim += 1 if c.use_view_score else 0
        aux_dim += 6 if c.use_geometry else 0  # center xyz + view xyz
        self.aux_dim = int(aux_dim)

        self.aux_encoder = nn.Sequential(
            nn.LayerNorm(self.aux_dim),
            nn.Linear(self.aux_dim, c.aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.aux_hidden_dim, c.hidden_dim),
            nn.GELU(),
        )

        fusion_dim = c.hidden_dim + c.hidden_dim + emb_dim * 3
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(c.hidden_dim, c.dropout) for _ in range(c.num_residual_blocks)]
        )
        self.final_norm = nn.LayerNorm(c.hidden_dim)

        self.cdf_head = nn.Linear(c.hidden_dim, len(FRICTION_THRESHOLDS))
        self.collision_head = nn.Linear(c.hidden_dim, 1)
        self.empty_head = nn.Linear(c.hidden_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.angle_embed.weight, std=0.02)
        nn.init.normal_(self.depth_embed.weight, std=0.02)
        nn.init.normal_(self.view_rank_embed.weight, std=0.02)
        # Start near uncertainty rather than a confident all-safe prior.
        nn.init.zeros_(self.cdf_head.bias)
        nn.init.zeros_(self.collision_head.bias)
        nn.init.zeros_(self.empty_head.bias)

    @staticmethod
    def _as_column(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)[:, :1]

    def _build_aux(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        score_logits = batch["score_logits"].float()
        depth_logits = batch["depth_logits"].float()
        if score_logits.shape[-1] != self.config.score_logit_dim:
            raise ValueError(
                f"score_logits last dim={score_logits.shape[-1]}, expected "
                f"{self.config.score_logit_dim}"
            )
        if depth_logits.shape[-1] != self.config.depth_logit_dim:
            raise ValueError(
                f"depth_logits last dim={depth_logits.shape[-1]}, expected "
                f"{self.config.depth_logit_dim}"
            )

        depth_id = batch["depth_id"].long().clamp(0, self.config.num_depths - 1)
        depth_prob = F.softmax(depth_logits, dim=-1)
        selected_depth_prob = depth_prob.gather(1, depth_id[:, None])

        score_prob = F.softmax(score_logits, dim=-1)
        bins = torch.linspace(
            0.0,
            1.0,
            self.config.score_logit_dim,
            device=score_logits.device,
            dtype=score_logits.dtype,
        )
        expected_quality = (score_prob * bins[None]).sum(dim=-1, keepdim=True)

        angle_id = batch["angle_id"].long().clamp(0, self.config.num_angles - 1)
        angle_rad = angle_id.float() * (torch.pi / float(self.config.num_angles))
        angle_geom = torch.stack([torch.sin(angle_rad), torch.cos(angle_rad)], dim=-1)
        depth_norm = depth_id.float().div(max(self.config.num_depths - 1, 1)).unsqueeze(-1)

        fields = [
            score_logits,
            depth_logits,
            selected_depth_prob,
            expected_quality,
            self._as_column(batch["width_raw"].float()),
            angle_geom,
            depth_norm,
        ]
        if self.config.use_legacy_collision:
            value = batch.get("legacy_collision_logit")
            if value is None:
                value = torch.zeros_like(batch["width_raw"], dtype=torch.float32)
            fields.append(self._as_column(value.float()))
        if self.config.use_view_score:
            value = batch.get("view_score")
            if value is None:
                value = torch.zeros_like(batch["width_raw"], dtype=torch.float32)
            fields.append(self._as_column(value.float()))
        if self.config.use_geometry:
            fields.extend([batch["center_xyz"].float(), batch["view_xyz"].float()])

        aux = torch.cat(fields, dim=-1)
        if aux.shape[-1] != self.aux_dim:
            raise RuntimeError(f"Constructed aux dim={aux.shape[-1]}, expected {self.aux_dim}")
        return aux

    def _monotonic_logits(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.config.monotonic_cdf:
            return raw
        base = raw[:, :1]
        increments = F.softplus(
            raw[:, 1:] + float(self.config.monotonic_increment_bias)
        )
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)

    def forward_flat(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        angle_feature = batch["angle_feature"].float()
        if angle_feature.dim() != 2:
            raise ValueError(f"angle_feature must be [N,C], got {tuple(angle_feature.shape)}")
        if angle_feature.shape[-1] != self.config.angle_feature_dim:
            raise ValueError(
                f"angle_feature dim={angle_feature.shape[-1]}, expected "
                f"{self.config.angle_feature_dim}"
            )

        angle_id = batch["angle_id"].long().clamp(0, self.config.num_angles - 1)
        depth_id = batch["depth_id"].long().clamp(0, self.config.num_depths - 1)
        view_rank = batch["view_rank"].long().clamp(0, self.config.max_view_rank - 1)

        visual = self.angle_proj(self.angle_norm(angle_feature))
        aux = self.aux_encoder(self._build_aux(batch))
        emb = torch.cat(
            [
                self.angle_embed(angle_id),
                self.depth_embed(depth_id),
                self.view_rank_embed(view_rank),
            ],
            dim=-1,
        )
        x = self.fusion(torch.cat([visual, aux, emb], dim=-1))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        raw_cdf = self.cdf_head(x)
        cdf_logits = self._monotonic_logits(raw_cdf)
        cdf_prob = torch.sigmoid(cdf_logits)
        utility = cdf_prob.mean(dim=-1)
        collision_logit = self.collision_head(x).squeeze(-1)
        empty_logit = self.empty_head(x).squeeze(-1)
        return {
            "cdf_logits": cdf_logits,
            "cdf_prob": cdf_prob,
            "utility": utility,
            "collision_logit": collision_logit,
            "empty_logit": empty_logit,
            "candidate_feature": x,
        }

    def forward_lattice(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Score a lattice whose candidate axes precede the final feature axis.

        Example shapes:
            angle_feature [B,Q,Kv,Ka,Kd,C]
            score_logits  [B,Q,Kv,Ka,Kd,6]
            scalar ids    [B,Q,Kv,Ka,Kd]

        The returned tensors restore the same leading lattice dimensions.
        """
        leading = batch["angle_feature"].shape[:-1]
        flat: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if not torch.is_tensor(value):
                continue
            if key in {"angle_feature", "score_logits", "depth_logits", "center_xyz", "view_xyz"}:
                flat[key] = value.reshape(-1, value.shape[-1])
            else:
                flat[key] = value.reshape(-1)
        out = self.forward_flat(flat)
        restored: Dict[str, torch.Tensor] = {}
        for key, value in out.items():
            if key in {"cdf_logits", "cdf_prob", "candidate_feature"}:
                restored[key] = value.reshape(*leading, value.shape[-1])
            else:
                restored[key] = value.reshape(*leading)
        return restored

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward_flat(batch)


def load_joint_utility_scorer(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> EvaluatorAlignedJointUtilityScorer:
    """Load a scorer checkpoint produced by train_cva_joint_utility_ddp.py."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = checkpoint.get("scorer_config")
    if cfg_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path!r} has no scorer_config")
    model = EvaluatorAlignedJointUtilityScorer(JointUtilityScorerConfig(**cfg_dict))
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=strict)
    if device is not None:
        model.to(device)
    return model.eval()
