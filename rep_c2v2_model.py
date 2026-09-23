"""Rep-C2-v2 verifier with independent pre-enhancer RGB evidence.

All variants instantiate exactly the same modules:
  score    : A1 score/profile evidence only; RGB branch is zeroed.
  rgb      : score + paired candidate-conditioned RGB evidence.
  rgb_only : paired RGB evidence only; score branch is zeroed.

The RGB branch uses Rep-A's IndependentImageReader on the pre-enhancer image
feature map. It receives the physical grasp hypothesis and intrinsics but no
predicted depth map, sensor depth, rendered depth, CAD model or exact labels.
"""
from __future__ import annotations

import torch
from torch import nn

from rep_a_model import IndependentImageReader
from rep_c2v2_common import VARIANTS


SCORE_DIM = 25


def score_features(probabilities, offsets_mm, native_score):
    """Build [Q,25] native-vs-proposal score evidence."""
    p = probabilities.float()
    if p.ndim != 3 or p.shape[0] != 2 or p.shape[-1] != 6:
        raise ValueError(f"probabilities must be [2,Q,6], got {p.shape}")
    q = p.shape[1]
    off = offsets_mm.float().reshape(q) / 40.
    ns = native_score.float().reshape(q)
    p0, pc = p[0], p[1]
    u0, uc = p0.mean(-1), pc.mean(-1)
    x = torch.cat([
        p0, pc, pc-p0,
        u0[:,None], uc[:,None], (uc-u0)[:,None],
        off[:,None], off.abs()[:,None], off.square()[:,None],
        ns[:,None],
    ], -1)
    if x.shape[-1] != SCORE_DIM:
        raise RuntimeError(f"score feature dim {x.shape[-1]} != {SCORE_DIM}")
    return x


class RepC2V2Verifier(nn.Module):
    def __init__(
        self,
        image_channels: int,
        dim: int = 128,
        heads: int = 4,
        dropout: float = .1,
        variant: str = "rgb",
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Rep-C2-v2 variant {variant}")
        self.variant = variant
        self.dim = int(dim)
        self.rgb_reader = IndependentImageReader(
            int(image_channels), dim=self.dim, heads=heads, dropout=dropout
        )
        self.score_embed = nn.Sequential(
            nn.Linear(SCORE_DIM, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.rgb_pair_embed = nn.Sequential(
            nn.Linear(3*self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(2*self.dim),
            nn.Linear(2*self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(self.dim, 3)
        self.delta_head = nn.Linear(self.dim, 1)

    def forward(
        self,
        image_feature,
        K,
        actions,
        probabilities,
        offsets_mm,
        native_score,
        image_hw,
    ):
        """Forward one compact native/proposal batch.

        image_feature: [C,Hf,Wf]
        K:             [3,3]
        actions:       [2,Q,17]
        probabilities: [2,Q,6]
        offsets_mm:    [Q]
        native_score:  [Q]
        """
        a = actions.float()
        if a.ndim != 3 or a.shape[0] != 2 or a.shape[-1] != 17:
            raise ValueError(f"actions must be [2,Q,17], got {a.shape}")
        q = a.shape[1]
        score = self.score_embed(
            score_features(probabilities, offsets_mm, native_score)
        )

        flat = a.reshape(2*q,17)
        rgb = self.rgb_reader(
            image_feature.float()[None],
            flat,
            K.float()[None],
            image_hw,
        ).reshape(2,q,self.dim)
        rgb_pair = self.rgb_pair_embed(
            torch.cat((rgb[0], rgb[1], rgb[1]-rgb[0]), -1)
        )

        if self.variant == "score":
            rgb_pair = rgb_pair * 0.
        elif self.variant == "rgb_only":
            score = score * 0.

        h = self.fuse(torch.cat((score, rgb_pair), -1))
        return {
            "class_logits": self.classifier(h),
            "delta": self.delta_head(h).squeeze(-1),
            "embedding": h,
        }


def paired_initial_state(image_channels, dim=128, heads=4, dropout=.1):
    """Create one shared initialization used by all evidence variants."""
    m = RepC2V2Verifier(
        image_channels=image_channels,
        dim=dim,
        heads=heads,
        dropout=dropout,
        variant="rgb",
    )
    return {k:v.detach().clone() for k,v in m.state_dict().items()}
