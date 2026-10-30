"""A2: frozen-DINO multi-level RGB evidence sampled at point pixels.

The four raw DINOv2 patch-token feature levels are the *same upstream visual
source* consumed by the image-centric proposal_head; they are NOT the trained
image-centric proposal/enhancer output.  All DINO/pose-aware depth weights can
be strictly frozen from the common checkpoint, while this small projection and
point fusion module learn only from the grasp objective.

The operation is point-indexed, not grasp-action-conditioned:
    RGB -> frozen DINO levels -> small trainable per-level projections
        -> bilinear pixel-coordinate sampling -> [B,128,N] point features
        -> point backbone fusion -> original point grouping/CDF head.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiLevelDinoPointSampler(nn.Module):
    """Project four (B, Hpatch*Wpatch, 768) maps and sample by crop pixels.

    Projection is applied at the 32x32 patch grid *before* gathering, avoiding
    a dense 448x448x768 expansion or 20k-by-768 per-point projections.
    """

    def __init__(self, feature_dim: int = 768, levels: int = 4,
                 per_level_dim: int = 32):
        super().__init__()
        if levels != 4:
            raise ValueError("A2 uses the four same DINO intermediate levels as A3")
        self.feature_dim = int(feature_dim)
        self.levels = int(levels)
        self.per_level_dim = int(per_level_dim)
        self.output_dim = self.levels * self.per_level_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.per_level_dim),
            )
            for _ in range(self.levels)
        ])

    @staticmethod
    def _patch_tokens(level):
        if isinstance(level, (tuple, list)):
            if not level:
                raise ValueError("Empty DINO feature level")
            level = level[0]
        if not torch.is_tensor(level) or level.ndim != 3:
            raise TypeError("DINO intermediate patch tokens must be [B,N,C]")
        return level

    def forward(self, dino_levels, img_idxs: torch.Tensor, *,
                image_hw=(448, 448)) -> torch.Tensor:
        if not isinstance(dino_levels, (tuple, list)) or len(dino_levels) != 4:
            raise ValueError("Expected exactly four raw DINO intermediate feature levels")
        H, W = map(int, image_hw)
        if H <= 0 or W <= 0 or H % 14 or W % 14:
            raise ValueError("DINOv2 patch14 requires image dimensions divisible by 14")
        Hp, Wp = H // 14, W // 14
        if not torch.is_tensor(img_idxs) or img_idxs.ndim != 2:
            raise ValueError("img_idxs must be [B,N]")
        if img_idxs.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError("img_idxs must contain integer pixel indices")
        B, N = img_idxs.shape
        if not bool(((img_idxs >= 0) & (img_idxs < H * W)).all()):
            raise ValueError("img_idxs outside cropped image; do not silently clamp")
        u = torch.remainder(img_idxs, W).to(dtype=torch.float32)
        v = torch.div(img_idxs, W, rounding_mode="floor").to(dtype=torch.float32)
        # align_corners=False: normalized coordinates refer to pixel centers.
        grid = torch.stack((
            2.0 * (u + 0.5) / float(W) - 1.0,
            2.0 * (v + 0.5) / float(H) - 1.0,
        ), dim=-1).unsqueeze(1)  # [B,1,N,2], 32x32 bilinear feature sampling

        sampled = []
        for idx, (level, projector) in enumerate(
            zip(dino_levels, self.projections)
        ):
            patches = self._patch_tokens(level)
            if patches.shape != (B, Hp * Wp, self.feature_dim):
                raise ValueError(
                    f"DINO level {idx}: expected "
                    f"[{B},{Hp * Wp},{self.feature_dim}], "
                    f"got {tuple(patches.shape)}"
                )
            if patches.device != img_idxs.device:
                raise RuntimeError("DINO features and pixel indices must be on same device")
            # Frozen upstream tokens; A2 learns only the light point adapter.
            patches = patches.detach()
            projected = projector(patches)
            fmap = projected.transpose(1, 2).reshape(
                B, self.per_level_dim, Hp, Wp
            )
            point_features = F.grid_sample(
                fmap, grid, mode="bilinear", padding_mode="border",
                align_corners=False,
            ).squeeze(2)  # [B,per_level_dim,N]
            sampled.append(point_features)
        out = torch.cat(sampled, dim=1).contiguous()
        if out.shape != (B, self.output_dim, N):
            raise AssertionError("Unexpected A2 per-point RGB feature shape")
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError("Nonfinite A2 sampled RGB features")
        return out


class ResidualPointRGBFusion(nn.Module):
    """Fuse sampled RGB into point encoder outputs before graspness/view/FPS.

    This preserves the exact geometry-only Minkowski backbone and point
    selection/grouping. Zero-initialize the final residual transform so A2
    initially has the same point features as A1 (not a changed random trunk).
    """

    def __init__(self, point_dim: int = 512,
                 rgb_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.rgb_dim = int(rgb_dim)
        self.point_dim = int(point_dim)
        self.norm = nn.LayerNorm(self.rgb_dim)
        self.fusion = nn.Sequential(
            nn.Conv1d(self.point_dim + self.rgb_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.point_dim, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, points: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError("Point tensor should be [B,C,N]")
        B, C, N = points.shape
        if C != self.point_dim or rgb.shape != (B, self.rgb_dim, N):
            raise ValueError(
                f"Expected point [B,{self.point_dim},N], "
                f"RGB [B,{self.rgb_dim},N]; got {tuple(points.shape)} and "
                f"{tuple(rgb.shape)}"
            )
        if rgb.device != points.device:
            raise RuntimeError("RGB and point feature tensors are on different devices")
        normalized = self.norm(rgb.transpose(1, 2)).transpose(1, 2)
        delta = self.fusion(torch.cat((points, normalized), dim=1))
        return points + delta
