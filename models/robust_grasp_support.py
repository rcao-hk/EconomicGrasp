"""Robust grasp-evidence support variants for the CVA local grouping.

This experiment deliberately changes *where* the existing CVA grouping reads
local evidence while keeping the network heads, supervision, decoder, and token
budget unchanged.

Modes
-----
metric
    Exact baseline behavior from ViewConditionedAttentionGrouping.
wide
    Same metric-conditioned, gripper-aligned support and same token count, but
    expand the projected patch around the selected image query by ``wide_scale``.
    This is the receptive-field control.
dual
    Keep the total patch-token count fixed. Half of the sampling positions use
    the original metric-conditioned, gripper-aligned support; the other half use
    a fixed-radius image-space support centered at the same image query. The
    image-space half does not use the predicted metric depth to set its radius.

No new trainable parameters are introduced. A checkpoint therefore has the same
parameter schema as the baseline CVA-CDF model; the support mode is a runtime
architecture protocol and must be supplied consistently at training/inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch

import models.kview_query_transformer as kqt


@dataclass(frozen=True)
class RobustSupportSpec:
    mode: str = "metric"
    wide_scale: float = 1.5
    image_radius_px: float = 32.0

    def validated(self) -> "RobustSupportSpec":
        mode = str(self.mode).lower()
        if mode not in {"metric", "wide", "dual"}:
            raise ValueError(
                f"robust support mode must be metric/wide/dual, got {self.mode!r}."
            )
        if float(self.wide_scale) <= 0:
            raise ValueError("wide_scale must be > 0.")
        if float(self.image_radius_px) <= 0:
            raise ValueError("image_radius_px must be > 0.")
        return RobustSupportSpec(
            mode=mode,
            wide_scale=float(self.wide_scale),
            image_radius_px=float(self.image_radius_px),
        )


_ACTIVE_SPEC = RobustSupportSpec().validated()
_BASE_GROUPING = kqt.ViewConditionedAttentionGrouping


def configure_robust_support(
    mode: str,
    wide_scale: float = 1.5,
    image_radius_px: float = 32.0,
) -> RobustSupportSpec:
    """Set the process-local support protocol used by subsequently built models."""
    global _ACTIVE_SPEC
    _ACTIVE_SPEC = RobustSupportSpec(
        mode=mode,
        wide_scale=wide_scale,
        image_radius_px=image_radius_px,
    ).validated()
    return _ACTIVE_SPEC


def current_robust_support() -> RobustSupportSpec:
    return _ACTIVE_SPEC


class RobustViewConditionedAttentionGrouping(_BASE_GROUPING):
    """Drop-in grouping with metric/wide/dual support and unchanged parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec = current_robust_support()
        self.robust_support_mode = spec.mode
        self.robust_wide_scale = float(spec.wide_scale)
        self.robust_image_radius_px = float(spec.image_radius_px)

        # Checkerboard partition keeps the total number of patch tokens exactly
        # unchanged. For the default even patch size, the split is exactly 50/50.
        S = int(self.config.patch_size)
        yy, xx = torch.meshgrid(
            torch.arange(S, dtype=torch.long),
            torch.arange(S, dtype=torch.long),
            indexing="ij",
        )
        image_mask = ((xx + yy) % 2 == 1).reshape(-1)
        self.register_buffer(
            "robust_image_support_mask",
            image_mask.bool(),
            persistent=False,
        )

    @staticmethod
    def _normalize_uv_grid(patch_uv: torch.Tensor, H: int, W: int):
        valid = (
            torch.isfinite(patch_uv).all(dim=-1)
            & (patch_uv[..., 0] >= 0.0)
            & (patch_uv[..., 0] <= float(W - 1))
            & (patch_uv[..., 1] >= 0.0)
            & (patch_uv[..., 1] <= float(H - 1))
        )
        x_norm = patch_uv[..., 0] / max(float(W - 1), 1.0) * 2.0 - 1.0
        y_norm = patch_uv[..., 1] / max(float(H - 1), 1.0) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).clamp(-2.0, 2.0).contiguous()
        return grid, valid

    def _make_view_conditioned_grid(
        self,
        seed_xyz: torch.Tensor,
        token_sel_idx: torch.Tensor,
        top_view_rot: torch.Tensor,
        depth_map: torch.Tensor,
        camera_K: torch.Tensor,
        H: int,
        W: int,
    ):
        base = super()._make_view_conditioned_grid(
            seed_xyz=seed_xyz,
            token_sel_idx=token_sel_idx,
            top_view_rot=top_view_rot,
            depth_map=depth_map,
            camera_K=camera_K,
            H=H,
            W=W,
        )
        (
            grid,
            valid,
            radius_dbg,
            patch_uv,
            center_uv,
            center_proj_uv,
            center_proj_err,
            vec_y,
            vec_z,
        ) = base

        mode = self.robust_support_mode
        if mode == "metric":
            # Exact baseline path: do not recompute coordinates.
            return base

        if mode == "wide":
            scale = float(self.robust_wide_scale)
            patch_uv = center_uv.unsqueeze(2) + scale * (
                patch_uv - center_uv.unsqueeze(2)
            )
            grid, valid = self._normalize_uv_grid(patch_uv, H=H, W=W)
            return (
                grid,
                valid,
                radius_dbg * scale,
                patch_uv,
                center_uv,
                center_proj_uv,
                center_proj_err,
                vec_y * scale,
                vec_z * scale,
            )

        if mode == "dual":
            # Fixed-radius, image-axis-aligned support. Its radius is independent
            # of predicted metric depth; the center remains the selected image
            # query so the proposal itself is unchanged.
            offsets = self.unit_offsets.to(
                device=patch_uv.device,
                dtype=patch_uv.dtype,
            )
            image_patch_uv = (
                center_uv.unsqueeze(2)
                + offsets.view(1, 1, -1, 2)
                * float(self.robust_image_radius_px)
            )
            image_mask = self.robust_image_support_mask.to(
                device=patch_uv.device
            ).view(1, 1, -1, 1)
            patch_uv = torch.where(image_mask, image_patch_uv, patch_uv)
            grid, valid = self._normalize_uv_grid(patch_uv, H=H, W=W)

            image_ratio = float(
                self.robust_image_support_mask.float().mean().item()
            )
            radius_dbg = (
                radius_dbg * (1.0 - image_ratio)
                + float(self.robust_image_radius_px) * image_ratio
            )
            return (
                grid,
                valid,
                radius_dbg,
                patch_uv,
                center_uv,
                center_proj_uv,
                center_proj_err,
                vec_y,
                vec_z,
            )

        raise AssertionError(mode)

    def forward(self, *args, **kwargs):
        end_points: Dict[str, Any] | None = kwargs.get("end_points", None)
        if end_points is None and len(args) >= 10:
            # Base signature ends with end_points.
            end_points = args[9]
        if isinstance(end_points, dict):
            mode_id = {"metric": 0.0, "wide": 1.0, "dual": 2.0}[
                self.robust_support_mode
            ]
            device = None
            seed_features = kwargs.get("seed_features", args[0] if args else None)
            if torch.is_tensor(seed_features):
                device = seed_features.device
                dtype = seed_features.dtype
            else:
                dtype = torch.float32
            end_points["robust_support_mode"] = self.robust_support_mode
            end_points["D: KVCA support mode id"] = torch.tensor(
                mode_id, device=device, dtype=dtype
            ).reshape(())
            end_points["D: KVCA image support ratio"] = torch.tensor(
                float(self.robust_image_support_mask.float().mean().item())
                if self.robust_support_mode == "dual"
                else 0.0,
                device=device,
                dtype=dtype,
            ).reshape(())
            end_points["D: KVCA image radius px"] = torch.tensor(
                float(self.robust_image_radius_px)
                if self.robust_support_mode == "dual"
                else 0.0,
                device=device,
                dtype=dtype,
            ).reshape(())
            end_points["D: KVCA wide scale"] = torch.tensor(
                float(self.robust_wide_scale)
                if self.robust_support_mode == "wide"
                else 1.0,
                device=device,
                dtype=dtype,
            ).reshape(())
        return super().forward(*args, **kwargs)


def install_robust_support_grouping(
    mode: str,
    wide_scale: float = 1.5,
    image_radius_px: float = 32.0,
) -> RobustSupportSpec:
    """Configure and install the drop-in grouping before model construction."""
    spec = configure_robust_support(
        mode=mode,
        wide_scale=wide_scale,
        image_radius_px=image_radius_px,
    )
    kqt.ViewConditionedAttentionGrouping = RobustViewConditionedAttentionGrouping
    return spec
