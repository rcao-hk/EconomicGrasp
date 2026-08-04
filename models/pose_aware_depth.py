"""Camera-pose conditioning modules for DAV2/DPT metric depth.

Supported modes
---------------
``none``
    Do not condition the DPT depth decoder.

``global_film``
    Use the original 3D camera-view vector to predict one channel-wise FiLM
    transform per DINO feature level.

``ray_gravity_film``
    Build a dense ray--gravity field from crop-aware intrinsics and the table
    normal/world-up direction expressed in the camera frame.  A small CNN then
    predicts spatially varying FiLM transforms for DINO patch tokens.  Class
    tokens are conditioned by a global pooled geometry feature.

All output FiLM projections are zero-initialized.  Consequently, enabling a
conditioning module preserves the unconditioned DPT behavior exactly at
initialization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


POSE_DEPTH_MODES = (
    "none",
    "global_film",
    "ray_gravity_film",
)


def _as_feature_tensor(feature: Any) -> torch.Tensor:
    if isinstance(feature, (tuple, list)):
        if len(feature) < 1:
            raise ValueError("Received an empty DINO feature tuple/list.")
        feature = feature[0]
    if not torch.is_tensor(feature):
        raise TypeError(f"Unsupported DINO feature type: {type(feature)!r}.")
    return feature


def _normalize_b3_vector(
    value: torch.Tensor,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    if value is None:
        raise KeyError(f"{name} is required with shape (B,3).")
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    if value.dim() == 1:
        value = value.unsqueeze(0)
    while value.dim() > 2:
        singleton_dims = [
            dim
            for dim in range(1, value.dim() - 1)
            if value.size(dim) == 1
        ]
        if not singleton_dims:
            break
        value = value.squeeze(singleton_dims[0])

    expected = (batch_size, 3)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"Expected {name} shape {expected}, got {tuple(value.shape)}."
        )

    value = torch.nan_to_num(
        value.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norm = value.norm(dim=-1, keepdim=True)
    if bool((norm < 1e-6).any().item()):
        bad = torch.nonzero(norm.squeeze(-1) < 1e-6).flatten().tolist()
        raise ValueError(
            f"{name} must be finite and non-zero for every sample; "
            f"invalid batch indices: {bad}."
        )
    return value / norm.clamp_min(1e-6)


def _validate_camera_k(
    camera_k: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if camera_k is None:
        raise KeyError(
            "ray_gravity_film requires crop-aware camera intrinsics K "
            "with shape (B,3,3)."
        )
    if not torch.is_tensor(camera_k):
        camera_k = torch.as_tensor(camera_k)
    if camera_k.dim() == 2:
        camera_k = camera_k.unsqueeze(0)
    expected = (batch_size, 3, 3)
    if tuple(camera_k.shape) != expected:
        raise ValueError(
            f"Expected camera intrinsics shape {expected}, got "
            f"{tuple(camera_k.shape)}."
        )
    camera_k = torch.nan_to_num(
        camera_k.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    fx = camera_k[:, 0, 0]
    fy = camera_k[:, 1, 1]
    if bool(((fx.abs() < 1e-6) | (fy.abs() < 1e-6)).any().item()):
        raise ValueError("Camera intrinsics contain zero/invalid focal lengths.")
    return camera_k


class PoseAwareDPTFiLM(nn.Module):
    """Original global, channel-wise pose FiLM for multi-level DINO features.

    The parameter/module names intentionally match the previous implementation
    so existing ``global_film`` checkpoints can still be loaded strictly.
    """

    def __init__(
        self,
        feat_dim: int,
        num_levels: int = 4,
        pose_dim: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_levels = int(num_levels)
        self.pose_dim = int(pose_dim)
        self.hidden_dim = int(hidden_dim)

        if self.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {self.feat_dim}.")
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {self.num_levels}.")
        if self.pose_dim != 3:
            raise ValueError(
                "global_film requires a 3D camera-pose vector, got "
                f"pose_dim={self.pose_dim}."
            )

        self.pose_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.level_film = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, 2 * self.feat_dim)
                for _ in range(self.num_levels)
            ]
        )

        for projection in self.level_film:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _apply_film(
        feature: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        gamma = gamma.to(device=feature.device, dtype=feature.dtype)
        beta = beta.to(device=feature.device, dtype=feature.dtype)

        if feature.dim() == 2:
            return feature * (1.0 + gamma) + beta
        if feature.dim() == 3:
            return feature * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        if feature.dim() == 4:
            return (
                feature * (1.0 + gamma[:, :, None, None])
                + beta[:, :, None, None]
            )
        raise ValueError(
            f"Unsupported DINO feature shape for FiLM: {tuple(feature.shape)}."
        )

    def forward(
        self,
        features: Sequence[Any],
        camera_pose_vec: torch.Tensor,
    ) -> Tuple[List[Any], Dict[str, torch.Tensor]]:
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} DINO feature levels, "
                f"got {len(features)}."
            )

        first = _as_feature_tensor(features[0])
        batch_size = int(first.shape[0])
        pose_unit = _normalize_b3_vector(
            camera_pose_vec,
            batch_size,
            "camera_pose_vec",
        ).to(device=first.device)

        adapter_dtype = self.pose_encoder[0].weight.dtype
        pose_latent = self.pose_encoder(pose_unit.to(dtype=adapter_dtype))

        conditioned: List[Any] = []
        gamma_level_means = []
        beta_level_means = []
        gamma_level_max = []
        beta_level_max = []

        for level_idx, feature in enumerate(features):
            film = self.level_film[level_idx](pose_latent)
            gamma, beta = film.chunk(2, dim=-1)
            gamma_level_means.append(gamma.detach().abs().mean())
            beta_level_means.append(beta.detach().abs().mean())
            gamma_level_max.append(gamma.detach().abs().amax())
            beta_level_max.append(beta.detach().abs().amax())

            if isinstance(feature, tuple):
                patch_tokens, cls_token = feature
                conditioned.append(
                    (
                        self._apply_film(patch_tokens, gamma, beta),
                        self._apply_film(cls_token, gamma, beta),
                    )
                )
            elif isinstance(feature, list):
                if len(feature) != 2:
                    raise ValueError(
                        "Expected DINO list feature [patch_tokens, cls_token], "
                        f"got length {len(feature)}."
                    )
                patch_tokens, cls_token = feature
                conditioned.append(
                    [
                        self._apply_film(patch_tokens, gamma, beta),
                        self._apply_film(cls_token, gamma, beta),
                    ]
                )
            else:
                conditioned.append(self._apply_film(feature, gamma, beta))

        gamma_levels = torch.stack(gamma_level_means)
        beta_levels = torch.stack(beta_level_means)
        aux = {
            "camera_pose_unit": pose_unit.detach(),
            "pose_depth_gamma_abs_mean": gamma_levels.mean(),
            "pose_depth_beta_abs_mean": beta_levels.mean(),
            "pose_depth_gamma_abs_mean_levels": gamma_levels,
            "pose_depth_beta_abs_mean_levels": beta_levels,
            "pose_depth_gamma_abs_max_levels": torch.stack(gamma_level_max),
            "pose_depth_beta_abs_max_levels": torch.stack(beta_level_max),
            "pose_depth_gamma_spatial_std": gamma_levels.new_zeros(()),
            "pose_depth_beta_spatial_std": beta_levels.new_zeros(()),
            # Legacy aliases used by existing logs/checkpoints.
            "pose_film_gamma_abs_mean": gamma_levels.mean(),
            "pose_film_beta_abs_mean": beta_levels.mean(),
            "pose_film_gamma_abs_mean_levels": gamma_levels,
            "pose_film_beta_abs_mean_levels": beta_levels,
        }
        return conditioned, aux


class RayGravityDenseDPTFiLM(nn.Module):
    """Spatially varying DPT FiLM driven by camera rays and world-up.

    ``camera_gravity_vec`` follows this convention:

    * coordinate frame: camera frame;
    * direction: table normal / world-up expressed in the camera frame;
    * shape: ``(B,3)`` and unit normalized.

    For each DINO patch center, the module builds a 10-channel field:

    ``[ray(3), gravity(3), -ray·gravity(1), tangent-gravity(3)]``.
    """

    def __init__(
        self,
        feat_dim: int,
        num_levels: int = 4,
        geometry_hidden_dim: int = 64,
        geometry_mid_dim: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_levels = int(num_levels)
        self.geometry_hidden_dim = int(geometry_hidden_dim)
        self.geometry_mid_dim = int(geometry_mid_dim)
        self.eps = float(eps)

        if self.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {self.feat_dim}.")
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {self.num_levels}.")
        if self.geometry_hidden_dim <= 0 or self.geometry_mid_dim <= 0:
            raise ValueError("Geometry encoder dimensions must be positive.")

        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(10, self.geometry_mid_dim, 3, padding=1, bias=False),
            nn.GroupNorm(self._num_groups(self.geometry_mid_dim), self.geometry_mid_dim),
            nn.GELU(),
            nn.Conv2d(
                self.geometry_mid_dim,
                self.geometry_hidden_dim,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._num_groups(self.geometry_hidden_dim),
                self.geometry_hidden_dim,
            ),
            nn.GELU(),
        )

        self.patch_film = nn.ModuleList(
            [
                nn.Conv2d(
                    self.geometry_hidden_dim,
                    2 * self.feat_dim,
                    kernel_size=1,
                    bias=True,
                )
                for _ in range(self.num_levels)
            ]
        )
        self.cls_film = nn.ModuleList(
            [
                nn.Linear(
                    self.geometry_hidden_dim,
                    2 * self.feat_dim,
                    bias=True,
                )
                for _ in range(self.num_levels)
            ]
        )

        # Exact identity at initialization.
        for head in list(self.patch_film) + list(self.cls_film):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _num_groups(channels: int, max_groups: int = 8) -> int:
        groups = min(int(max_groups), int(channels))
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return groups

    def _build_ray_gravity_field(
        self,
        camera_k: torch.Tensor,
        gravity_unit: torch.Tensor,
        image_hw: Tuple[int, int],
        patch_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(camera_k.shape[0])
        image_h, image_w = map(int, image_hw)
        patch_h, patch_w = map(int, patch_hw)
        if image_h <= 0 or image_w <= 0 or patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"Invalid image/patch shapes: image_hw={image_hw}, "
                f"patch_hw={patch_hw}."
            )

        device = camera_k.device
        # Patch centers in the crop-resized image coordinate system.
        u = (
            (torch.arange(patch_w, device=device, dtype=torch.float32) + 0.5)
            * (float(image_w) / float(patch_w))
            - 0.5
        )
        v = (
            (torch.arange(patch_h, device=device, dtype=torch.float32) + 0.5)
            * (float(image_h) / float(patch_h))
            - 0.5
        )
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        fx = camera_k[:, 0, 0].view(batch_size, 1, 1)
        fy = camera_k[:, 1, 1].view(batch_size, 1, 1)
        cx = camera_k[:, 0, 2].view(batch_size, 1, 1)
        cy = camera_k[:, 1, 2].view(batch_size, 1, 1)

        ray_x = (uu.unsqueeze(0) - cx) / fx
        ray_y = (vv.unsqueeze(0) - cy) / fy
        ray_z = torch.ones_like(ray_x)
        ray = torch.stack([ray_x, ray_y, ray_z], dim=1)
        ray = ray / ray.norm(dim=1, keepdim=True).clamp_min(self.eps)

        gravity = gravity_unit[:, :, None, None].expand(
            -1,
            -1,
            patch_h,
            patch_w,
        )
        ray_dot_gravity = (ray * gravity).sum(dim=1, keepdim=True)
        downward_alignment = -ray_dot_gravity

        tangent = gravity - ray_dot_gravity * ray
        tangent_norm = tangent.norm(dim=1, keepdim=True)
        tangent = tangent / tangent_norm.clamp_min(self.eps)
        tangent = torch.where(
            tangent_norm > self.eps,
            tangent,
            torch.zeros_like(tangent),
        )

        field = torch.cat(
            [
                ray,
                gravity,
                downward_alignment,
                tangent,
            ],
            dim=1,
        )
        if field.shape != (batch_size, 10, patch_h, patch_w):
            raise RuntimeError(
                f"Unexpected ray-gravity field shape: {tuple(field.shape)}."
            )
        return field, downward_alignment

    def _apply_dense_patch_film(
        self,
        feature: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        patch_hw: Tuple[int, int],
    ) -> torch.Tensor:
        patch_h, patch_w = patch_hw
        gamma = gamma.to(device=feature.device, dtype=feature.dtype)
        beta = beta.to(device=feature.device, dtype=feature.dtype)

        if feature.dim() == 3:
            batch_size, num_tokens, channels = feature.shape
            if channels != self.feat_dim:
                raise ValueError(
                    f"Expected patch-token dim {self.feat_dim}, got {channels}."
                )
            if num_tokens != patch_h * patch_w:
                raise ValueError(
                    f"Expected {patch_h * patch_w} patch tokens, got "
                    f"{num_tokens}."
                )
            feature_map = feature.transpose(1, 2).reshape(
                batch_size,
                channels,
                patch_h,
                patch_w,
            )
            output_map = feature_map * (1.0 + gamma) + beta
            return output_map.flatten(2).transpose(1, 2).contiguous()

        if feature.dim() == 4:
            if feature.shape[1] != self.feat_dim:
                raise ValueError(
                    f"Expected feature-map channels {self.feat_dim}, got "
                    f"{feature.shape[1]}."
                )
            if feature.shape[-2:] != (patch_h, patch_w):
                raise ValueError(
                    f"Expected feature-map spatial shape {(patch_h, patch_w)}, "
                    f"got {tuple(feature.shape[-2:])}."
                )
            return feature * (1.0 + gamma) + beta

        raise ValueError(
            "Dense ray-gravity FiLM expects patch tokens (B,N,C) or a "
            f"feature map (B,C,H,W), got {tuple(feature.shape)}."
        )

    @staticmethod
    def _apply_cls_film(
        cls_token: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if cls_token.dim() != 2:
            raise ValueError(
                f"Expected class token shape (B,C), got {tuple(cls_token.shape)}."
            )
        gamma = gamma.to(device=cls_token.device, dtype=cls_token.dtype)
        beta = beta.to(device=cls_token.device, dtype=cls_token.dtype)
        return cls_token * (1.0 + gamma) + beta

    def forward(
        self,
        features: Sequence[Any],
        camera_k: torch.Tensor,
        camera_gravity_vec: torch.Tensor,
        image_hw: Tuple[int, int],
        patch_hw: Tuple[int, int],
    ) -> Tuple[List[Any], Dict[str, torch.Tensor]]:
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} DINO feature levels, "
                f"got {len(features)}."
            )

        first = _as_feature_tensor(features[0])
        batch_size = int(first.shape[0])
        device = first.device
        camera_k = _validate_camera_k(camera_k, batch_size).to(device=device)
        gravity_unit = _normalize_b3_vector(
            camera_gravity_vec,
            batch_size,
            "camera_gravity_vec",
        ).to(device=device)

        field, alignment = self._build_ray_gravity_field(
            camera_k,
            gravity_unit,
            image_hw=image_hw,
            patch_hw=patch_hw,
        )
        encoder_dtype = self.geometry_encoder[0].weight.dtype
        geometry_feature = self.geometry_encoder(field.to(dtype=encoder_dtype))
        geometry_global = geometry_feature.mean(dim=(-2, -1))

        conditioned: List[Any] = []
        gamma_abs_mean_levels = []
        beta_abs_mean_levels = []
        gamma_spatial_std_levels = []
        beta_spatial_std_levels = []
        gamma_abs_max_levels = []
        beta_abs_max_levels = []

        for level_idx, feature in enumerate(features):
            patch_params = self.patch_film[level_idx](geometry_feature)
            gamma_patch, beta_patch = patch_params.chunk(2, dim=1)
            cls_params = self.cls_film[level_idx](geometry_global)
            gamma_cls, beta_cls = cls_params.chunk(2, dim=-1)

            gamma_abs_mean_levels.append(gamma_patch.detach().abs().mean())
            beta_abs_mean_levels.append(beta_patch.detach().abs().mean())
            gamma_spatial_std_levels.append(
                gamma_patch.detach().float().std(
                    dim=(-2, -1),
                    unbiased=False,
                ).mean()
            )
            beta_spatial_std_levels.append(
                beta_patch.detach().float().std(
                    dim=(-2, -1),
                    unbiased=False,
                ).mean()
            )
            gamma_abs_max_levels.append(gamma_patch.detach().abs().amax())
            beta_abs_max_levels.append(beta_patch.detach().abs().amax())

            if isinstance(feature, tuple):
                if len(feature) != 2:
                    raise ValueError(
                        f"Expected (patch_tokens, cls_token), got {len(feature)} items."
                    )
                patch_tokens, cls_token = feature
                conditioned.append(
                    (
                        self._apply_dense_patch_film(
                            patch_tokens,
                            gamma_patch,
                            beta_patch,
                            patch_hw,
                        ),
                        self._apply_cls_film(
                            cls_token,
                            gamma_cls,
                            beta_cls,
                        ),
                    )
                )
            elif isinstance(feature, list):
                if len(feature) != 2:
                    raise ValueError(
                        f"Expected [patch_tokens, cls_token], got {len(feature)} items."
                    )
                patch_tokens, cls_token = feature
                conditioned.append(
                    [
                        self._apply_dense_patch_film(
                            patch_tokens,
                            gamma_patch,
                            beta_patch,
                            patch_hw,
                        ),
                        self._apply_cls_film(
                            cls_token,
                            gamma_cls,
                            beta_cls,
                        ),
                    ]
                )
            else:
                conditioned.append(
                    self._apply_dense_patch_film(
                        feature,
                        gamma_patch,
                        beta_patch,
                        patch_hw,
                    )
                )

        gamma_means = torch.stack(gamma_abs_mean_levels)
        beta_means = torch.stack(beta_abs_mean_levels)
        gamma_spatial = torch.stack(gamma_spatial_std_levels)
        beta_spatial = torch.stack(beta_spatial_std_levels)

        aux = {
            "camera_gravity_unit": gravity_unit.detach(),
            "ray_gravity_alignment_mean": alignment.detach().mean(),
            "ray_gravity_alignment_min": alignment.detach().amin(),
            "ray_gravity_alignment_max": alignment.detach().amax(),
            "ray_gravity_alignment_map": alignment.detach(),
            "pose_depth_gamma_abs_mean": gamma_means.mean(),
            "pose_depth_beta_abs_mean": beta_means.mean(),
            "pose_depth_gamma_abs_mean_levels": gamma_means,
            "pose_depth_beta_abs_mean_levels": beta_means,
            "pose_depth_gamma_spatial_std": gamma_spatial.mean(),
            "pose_depth_beta_spatial_std": beta_spatial.mean(),
            "pose_depth_gamma_spatial_std_levels": gamma_spatial,
            "pose_depth_beta_spatial_std_levels": beta_spatial,
            "pose_depth_gamma_abs_max_levels": torch.stack(
                gamma_abs_max_levels
            ),
            "pose_depth_beta_abs_max_levels": torch.stack(
                beta_abs_max_levels
            ),
            # Legacy logger aliases.
            "pose_film_gamma_abs_mean": gamma_means.mean(),
            "pose_film_beta_abs_mean": beta_means.mean(),
            "pose_film_gamma_abs_mean_levels": gamma_means,
            "pose_film_beta_abs_mean_levels": beta_means,
        }
        return conditioned, aux
