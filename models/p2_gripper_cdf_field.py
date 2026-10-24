#!/usr/bin/env python3
"""Capacity-matched scratch CDF predictors for the P2 representation study.

The module is deliberately independent of EconomicGrasp internals. Callers
provide the frozen Stage-1 pre-CDF feature and, depending on the variant, exact
action-pose, projected gripper-region, and signed ray-depth evidence.

Four cumulative variants are supported:

``p2_0``
    Pre-CDF feature only. This is the nonlinear-capacity control.
``p2_a``
    ``p2_0`` + explicit exact-action pose descriptors for all depth anchors.
``p2_b``
    ``p2_a`` + region-pooled DPT features sampled at projected gripper support
    points.
``p2_c``
    ``p2_b`` + region-wise signed ray/depth residual statistics.

Every variant instantiates the same fixed-input, three-linear-layer MLP. Evidence
blocks not used by a variant are hard-masked to zero, so the nominal architecture
and parameter count are identical. The predictor is randomly initialized and
trained from scratch with exact-action CDF BCE; it never adds a residual to P1 or
to the original Stage-1 CDF output.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


P2_VARIANTS: Tuple[str, ...] = ("p2_0", "p2_a", "p2_b", "p2_c")
P2_FIELD_VERSION = "p2_gripper_conditioned_cdf_field_v2_scratch_mlp"
P2_CACHE_SCHEMA_VERSION = "p2_gripper_conditioned_cdf_cache_v2_scratch_mlp"
P2_PROBE_VERSION = "p2_gripper_conditioned_cdf_probe_v2_scratch_mlp"
REGION_NAMES: Tuple[str, ...] = (
    "left_finger",
    "right_finger",
    "closing",
    "palm",
    "approach",
)
RAY_STATS: Tuple[str, ...] = (
    "mean",
    "mean_abs",
    "min",
    "max",
    "near_ratio",
    "front_ratio",
    "behind_ratio",
    "depth_valid_ratio",
)


@dataclass(frozen=True)
class P2FieldConfig:
    """Non-learned physical sampling contract shared by cache and inference."""

    image_height: int = 448
    image_width: int = 448
    grasp_height_m: float = 0.02
    finger_half_thickness_m: float = 0.005
    max_grasp_width_m: float = 0.10
    min_metric_depth_m: float = 0.20
    max_metric_depth_m: float = 1.00
    center_xy_scale_m: float = 0.50
    residual_tau_m: float = 0.02
    surface_tau_m: float = 0.01
    align_corners: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


ACTION_POSE_DIM = 12
NUM_REGIONS = len(REGION_NAMES)
RAY_FEATURE_DIM = NUM_REGIONS * len(RAY_STATS)


def projected_feature_dim(image_feature_dim: int) -> int:
    return NUM_REGIONS * int(image_feature_dim) + NUM_REGIONS


def validate_variant(variant: str) -> str:
    aliases = {
        "p2_0": "p2_0",
        "p2-0": "p2_0",
        "base_feature": "p2_0",
        "p2_a": "p2_a",
        "p2-a": "p2_a",
        "pose": "p2_a",
        "p2_b": "p2_b",
        "p2-b": "p2_b",
        "projected": "p2_b",
        "p2_c": "p2_c",
        "p2-c": "p2_c",
        "ray_depth": "p2_c",
    }
    key = str(variant).strip().lower()
    if key not in aliases:
        raise ValueError(
            f"variant must be one of {P2_VARIANTS} (or a documented alias), "
            f"got {variant!r}"
        )
    return aliases[key]


def active_evidence_blocks(variant: str) -> Tuple[str, ...]:
    variant = validate_variant(variant)
    if variant == "p2_0":
        return ("base",)
    if variant == "p2_a":
        return ("base", "pose")
    if variant == "p2_b":
        return ("base", "pose", "projected")
    return ("base", "pose", "projected", "ray_depth")


def monotonic_cdf_logits_from_raw(
    raw: torch.Tensor,
    increment_bias: float,
) -> torch.Tensor:
    """Map raw [...,T] outputs to monotonically increasing CDF logits."""
    if raw.shape[-1] < 2:
        raise ValueError(f"raw must have T>=2, got {tuple(raw.shape)}")
    base = raw[..., :1]
    increments = F.softplus(raw[..., 1:] + float(increment_bias))
    return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


def build_action_pose_feature(
    center_xyz: torch.Tensor,
    view_xyz: torch.Tensor,
    angle_rad: torch.Tensor,
    grasp_depth_m: torch.Tensor,
    grasp_width_m: torch.Tensor,
    config: P2FieldConfig,
) -> torch.Tensor:
    """Encode an exact parallel-jaw action as a compact metric descriptor.

    Inputs have a common leading shape ``[...]``.  The angle uses a doubled
    phase because a parallel-jaw gripper is pi-periodic in its in-plane angle.
    Camera pose is intentionally not injected here; the current Stage-1 model
    already uses it only in the metric-depth branch.
    """
    if center_xyz.shape[-1] != 3 or view_xyz.shape[-1] != 3:
        raise ValueError("center_xyz and view_xyz must end in dimension 3")
    lead = center_xyz.shape[:-1]
    for name, tensor in {
        "view_xyz": view_xyz,
        "angle_rad": angle_rad,
        "grasp_depth_m": grasp_depth_m,
        "grasp_width_m": grasp_width_m,
    }.items():
        expected = lead + ((3,) if name == "view_xyz" else ())
        if tensor.shape != expected:
            raise ValueError(f"{name} shape={tuple(tensor.shape)}, expected={expected}")

    center = torch.nan_to_num(center_xyz.float(), nan=0.0, posinf=0.0, neginf=0.0)
    view = F.normalize(
        torch.nan_to_num(view_xyz.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    z = center[..., 2].clamp_min(1e-4)
    ray_xy = center[..., :2] / z.unsqueeze(-1)
    center_norm = torch.stack(
        [
            center[..., 0] / max(float(config.center_xy_scale_m), 1e-6),
            center[..., 1] / max(float(config.center_xy_scale_m), 1e-6),
            (
                center[..., 2]
                - 0.5 * (float(config.min_metric_depth_m) + float(config.max_metric_depth_m))
            )
            / max(
                0.5 * (
                    float(config.max_metric_depth_m)
                    - float(config.min_metric_depth_m)
                ),
                1e-6,
            ),
        ],
        dim=-1,
    )
    angle = angle_rad.float()
    phase = torch.stack([torch.cos(2.0 * angle), torch.sin(2.0 * angle)], dim=-1)
    depth_norm = grasp_depth_m.float().unsqueeze(-1) / 0.04
    width_norm = grasp_width_m.float().unsqueeze(-1) / max(
        float(config.max_grasp_width_m), 1e-6
    )
    feature = torch.cat(
        [center_norm, ray_xy, view, phase, depth_norm, width_norm], dim=-1
    )
    if feature.shape[-1] != ACTION_POSE_DIM:
        raise RuntimeError(
            f"Action pose feature has {feature.shape[-1]} channels, expected {ACTION_POSE_DIM}"
        )
    return torch.nan_to_num(feature, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-8.0, 8.0)


class GripperFieldSampler(nn.Module):
    """Project deterministic gripper-region samples into DPT/depth maps."""

    def __init__(self, config: Optional[P2FieldConfig] = None) -> None:
        super().__init__()
        self.config = config or P2FieldConfig()
        dx, y_scale, y_offset, z_value, region = self._build_templates(self.config)
        self.register_buffer("template_dx", dx, persistent=False)
        self.register_buffer("template_y_scale", y_scale, persistent=False)
        self.register_buffer("template_y_offset", y_offset, persistent=False)
        self.register_buffer("template_z", z_value, persistent=False)
        self.register_buffer("template_region", region, persistent=False)

    @staticmethod
    def _build_templates(config: P2FieldConfig):
        dx = []
        ys = []
        yo = []
        zz = []
        rr = []

        # Finger bodies: three positions along the finger, sampled at the two
        # visible height offsets.  The lateral offset is half the finger width.
        for region, sign in ((0, -1.0), (1, 1.0)):
            for x_delta in (-0.050, -0.025, -0.005):
                for z in (-0.006, 0.006):
                    dx.append(x_delta)
                    ys.append(0.5 * sign)
                    yo.append(float(config.finger_half_thickness_m) * sign)
                    zz.append(z)
                    rr.append(region)

        # Closing volume: samples between both fingers.
        for x_delta in (-0.045, -0.025, -0.005):
            for y_scale in (-0.25, 0.0, 0.25):
                dx.append(x_delta)
                ys.append(y_scale)
                yo.append(0.0)
                zz.append(0.0)
                rr.append(2)

        # Palm/back plate.
        for y_scale in (-0.50, -0.25, 0.0, 0.25, 0.50):
            dx.append(-0.065)
            ys.append(y_scale)
            yo.append(0.0)
            zz.append(0.0)
            rr.append(3)

        # Swept approach corridor behind the palm.
        for x_delta in (-0.075, -0.090, -0.105):
            for y_scale in (-0.25, 0.25):
                dx.append(x_delta)
                ys.append(y_scale)
                yo.append(0.0)
                zz.append(0.0)
                rr.append(4)

        return (
            torch.tensor(dx, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32),
            torch.tensor(yo, dtype=torch.float32),
            torch.tensor(zz, dtype=torch.float32),
            torch.tensor(rr, dtype=torch.long),
        )

    @property
    def num_samples(self) -> int:
        return int(self.template_region.numel())

    @staticmethod
    def _project_xyz(
        xyz: torch.Tensor,
        K: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if K.shape != (3, 3):
            raise ValueError(f"K must be [3,3], got {tuple(K.shape)}")
        z = xyz[..., 2]
        safe_z = z.clamp_min(eps)
        u = K[0, 0] * xyz[..., 0] / safe_z + K[0, 2]
        v = K[1, 1] * xyz[..., 1] / safe_z + K[1, 2]
        return torch.stack([u, v], dim=-1), z

    @staticmethod
    def _sample_map(
        feature_map: torch.Tensor,
        uv: torch.Tensor,
        image_hw: Tuple[int, int],
        align_corners: bool,
    ) -> torch.Tensor:
        """Sample one map [1,C,Hf,Wf] at uv [N,S,2] in image coordinates."""
        if feature_map.dim() != 4 or feature_map.shape[0] != 1:
            raise ValueError(
                f"feature_map must be [1,C,H,W], got {tuple(feature_map.shape)}"
            )
        n, s = uv.shape[:2]
        h_src, w_src = image_hw
        h_map, w_map = feature_map.shape[-2:]
        # Convert source-image coordinates to map coordinates first.  The
        # current DPT map is 448x448, but this keeps the contract explicit.
        u_map = uv[..., 0]
        v_map = uv[..., 1]
        if w_src > 1:
            u_map = u_map * float(w_map - 1) / float(w_src - 1)
        if h_src > 1:
            v_map = v_map * float(h_map - 1) / float(h_src - 1)
        gx = 2.0 * u_map / max(float(w_map - 1), 1.0) - 1.0
        gy = 2.0 * v_map / max(float(h_map - 1), 1.0) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, n * s, 1, 2)
        sampled = F.grid_sample(
            feature_map,
            grid.to(feature_map.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=bool(align_corners),
        )
        return (
            sampled.squeeze(0)
            .squeeze(-1)
            .transpose(0, 1)
            .contiguous()
            .view(n, s, feature_map.shape[1])
        )

    def _local_samples(
        self,
        grasp_width_m: torch.Tensor,
        grasp_depth_m: torch.Tensor,
    ) -> torch.Tensor:
        width = grasp_width_m.float().reshape(-1, 1)
        depth = grasp_depth_m.float().reshape(-1, 1)
        x = depth + self.template_dx.to(width)
        y = width * self.template_y_scale.to(width) + self.template_y_offset.to(width)
        z = self.template_z.to(width).expand(width.shape[0], -1)
        return torch.stack([x, y, z], dim=-1)

    @torch.no_grad()
    def forward(
        self,
        image_feature_map: torch.Tensor,
        depth_map: torch.Tensor,
        K: torch.Tensor,
        center_xyz: torch.Tensor,
        rotation: torch.Tensor,
        grasp_width_m: torch.Tensor,
        grasp_depth_m: torch.Tensor,
        *,
        action_chunk: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Return projected-image and signed-ray/depth field descriptors.

        All action tensors use shape [N,...].  The dense maps and intrinsics are
        for one frame; P2 cache/inference intentionally runs with batch size 1.
        """
        if image_feature_map.dim() != 4 or image_feature_map.shape[0] != 1:
            raise ValueError("P2 field sampling supports one frame at a time")
        if depth_map.dim() != 4 or depth_map.shape[:2] != (1, 1):
            raise ValueError(f"depth_map must be [1,1,H,W], got {tuple(depth_map.shape)}")
        if K.dim() == 3:
            if K.shape[0] != 1:
                raise ValueError("P2 field sampling supports one K matrix")
            K = K[0]
        if center_xyz.dim() != 2 or center_xyz.shape[-1] != 3:
            raise ValueError("center_xyz must be [N,3]")
        n = int(center_xyz.shape[0])
        if rotation.shape != (n, 3, 3):
            raise ValueError(f"rotation must be [{n},3,3], got {tuple(rotation.shape)}")
        for name, value in {
            "grasp_width_m": grasp_width_m,
            "grasp_depth_m": grasp_depth_m,
        }.items():
            if value.shape != (n,):
                raise ValueError(f"{name} must be [{n}], got {tuple(value.shape)}")
        if n == 0:
            image_dim = int(image_feature_map.shape[1])
            return (
                image_feature_map.new_zeros((0, projected_feature_dim(image_dim))),
                image_feature_map.new_zeros((0, RAY_FEATURE_DIM)),
                {"num_actions": 0.0, "valid_ratio": 0.0, "depth_valid_ratio": 0.0},
            )

        h, w = depth_map.shape[-2:]
        if (h, w) != (int(self.config.image_height), int(self.config.image_width)):
            raise ValueError(
                f"Depth image shape {(h,w)} differs from field config "
                f"{(self.config.image_height,self.config.image_width)}"
            )
        image_feature_map = torch.nan_to_num(
            image_feature_map.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        depth_map = torch.nan_to_num(
            depth_map.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        K = K.to(device=center_xyz.device, dtype=torch.float32)
        image_feature_map = image_feature_map.to(center_xyz.device)
        depth_map = depth_map.to(center_xyz.device)

        projected_chunks = []
        ray_chunks = []
        total_valid = 0.0
        total_depth_valid = 0.0
        total_samples = 0
        chunk_size = max(1, int(action_chunk))
        region_ids = self.template_region.to(center_xyz.device)
        region_masks = [region_ids == region for region in range(NUM_REGIONS)]
        surface_threshold = math.tanh(
            float(self.config.surface_tau_m)
            / max(float(self.config.residual_tau_m), 1e-6)
        )

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            center = center_xyz[start:stop].float()
            rot = rotation[start:stop].float()
            width_c = grasp_width_m[start:stop].float()
            depth_c = grasp_depth_m[start:stop].float()
            local = self._local_samples(width_c, depth_c).to(center.device)
            camera = center[:, None, :] + torch.einsum("nij,nsj->nsi", rot, local)
            uv, sample_z = self._project_xyz(camera, K)
            valid = (
                (sample_z > 1e-6)
                & (uv[..., 0] >= 0.0)
                & (uv[..., 0] <= float(w - 1))
                & (uv[..., 1] >= 0.0)
                & (uv[..., 1] <= float(h - 1))
            )
            sampled_img = self._sample_map(
                image_feature_map,
                uv,
                (h, w),
                bool(self.config.align_corners),
            )
            sampled_depth = self._sample_map(
                depth_map,
                uv,
                (h, w),
                bool(self.config.align_corners),
            ).squeeze(-1)
            depth_valid = (
                valid
                & torch.isfinite(sampled_depth)
                & (sampled_depth > float(self.config.min_metric_depth_m))
                & (sampled_depth < float(self.config.max_metric_depth_m))
            )
            raw_residual = sampled_depth - sample_z
            residual = torch.tanh(
                raw_residual / max(float(self.config.residual_tau_m), 1e-6)
            )

            image_parts = []
            valid_parts = []
            ray_parts = []
            for region in range(NUM_REGIONS):
                region_mask_1d = region_masks[region]
                count_region = max(int(region_mask_1d.sum().item()), 1)
                mask = valid[:, region_mask_1d]
                mask_f = mask.float().unsqueeze(-1)
                denom = mask_f.sum(dim=1).clamp_min(1.0)
                region_img = sampled_img[:, region_mask_1d]
                region_mean = (region_img * mask_f).sum(dim=1) / denom
                image_parts.append(region_mean)
                valid_parts.append(mask.float().sum(dim=1, keepdim=True) / float(count_region))

                dmask = depth_valid[:, region_mask_1d]
                values = residual[:, region_mask_1d]
                dmask_f = dmask.float()
                ddenom = dmask_f.sum(dim=1).clamp_min(1.0)
                mean = (values * dmask_f).sum(dim=1) / ddenom
                mean_abs = (values.abs() * dmask_f).sum(dim=1) / ddenom
                inf = torch.full_like(values, float("inf"))
                ninf = torch.full_like(values, float("-inf"))
                minimum = torch.where(dmask, values, inf).amin(dim=1)
                maximum = torch.where(dmask, values, ninf).amax(dim=1)
                any_valid = dmask.any(dim=1)
                minimum = torch.where(any_valid, minimum, torch.zeros_like(minimum))
                maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
                near = ((values.abs() <= surface_threshold) & dmask).float().sum(dim=1) / ddenom
                front = ((values > surface_threshold) & dmask).float().sum(dim=1) / ddenom
                behind = ((values < -surface_threshold) & dmask).float().sum(dim=1) / ddenom
                depth_valid_ratio = dmask_f.sum(dim=1) / float(count_region)
                ray_parts.extend(
                    [
                        mean,
                        mean_abs,
                        minimum,
                        maximum,
                        near,
                        front,
                        behind,
                        depth_valid_ratio,
                    ]
                )

            projected = torch.cat(
                [torch.cat(image_parts, dim=-1), torch.cat(valid_parts, dim=-1)],
                dim=-1,
            )
            ray = torch.stack(ray_parts, dim=-1)
            expected_projected = projected_feature_dim(int(image_feature_map.shape[1]))
            if projected.shape[-1] != expected_projected:
                raise RuntimeError(
                    f"Projected feature dim={projected.shape[-1]}, expected={expected_projected}"
                )
            if ray.shape[-1] != RAY_FEATURE_DIM:
                raise RuntimeError(
                    f"Ray feature dim={ray.shape[-1]}, expected={RAY_FEATURE_DIM}"
                )
            projected_chunks.append(projected)
            ray_chunks.append(ray)
            total_valid += float(valid.float().sum().item())
            total_depth_valid += float(depth_valid.float().sum().item())
            total_samples += int(valid.numel())

        projected_all = torch.cat(projected_chunks, dim=0)
        ray_all = torch.cat(ray_chunks, dim=0)
        diagnostics = {
            "num_actions": float(n),
            "samples_per_action": float(self.num_samples),
            "valid_ratio": total_valid / max(total_samples, 1),
            "depth_valid_ratio": total_depth_valid / max(total_samples, 1),
        }
        return projected_all, ray_all, diagnostics



class P2ScratchCdfMLP(nn.Module):
    """Capacity-matched three-layer CDF MLP trained from scratch.

    The predictor operates on one angle-conditioned row at a time and emits the
    complete ``D x T`` CDF lattice for that row. This preserves the original
    decoder's depth-specific outputs without leaking a P1 prediction into P2-0.

    All four variants instantiate the same feature blocks, LayerNorm modules,
    fixed total input width, and three Linear layers. Inactive evidence blocks
    are replaced by exact zeros *after* normalization, so they cannot encode a
    learned constant or affect the output.
    """

    def __init__(
        self,
        *,
        variant: str,
        base_feature_dim: int,
        image_feature_dim: int,
        num_depths: int = 4,
        num_thresholds: int = 6,
        hidden_dim: int = 256,
        increment_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.variant = validate_variant(variant)
        self.base_feature_dim = int(base_feature_dim)
        self.image_feature_dim = int(image_feature_dim)
        self.num_depths = int(num_depths)
        self.num_thresholds = int(num_thresholds)
        self.hidden_dim = int(hidden_dim)
        self.increment_bias = float(increment_bias)
        if min(
            self.base_feature_dim,
            self.image_feature_dim,
            self.num_depths,
            self.num_thresholds,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("All P2 Scratch-MLP dimensions must be positive")

        self.action_pose_dim = self.num_depths * ACTION_POSE_DIM
        self.projected_per_action_dim = projected_feature_dim(self.image_feature_dim)
        self.projected_dim = self.num_depths * self.projected_per_action_dim
        self.ray_per_action_dim = RAY_FEATURE_DIM
        self.ray_dim = self.num_depths * self.ray_per_action_dim
        self.total_input_dim = (
            self.base_feature_dim
            + self.action_pose_dim
            + self.projected_dim
            + self.ray_dim
        )

        # Every variant owns the exact same normalization modules and MLP.
        self.base_norm = nn.LayerNorm(self.base_feature_dim)
        self.pose_norm = nn.LayerNorm(self.action_pose_dim)
        self.projected_norm = nn.LayerNorm(self.projected_dim)
        self.ray_norm = nn.LayerNorm(self.ray_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.total_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.num_depths * self.num_thresholds,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Random scratch initialization, independent of the Stage-1 CDF head."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def contract(self) -> Dict[str, object]:
        return {
            "field_version": P2_FIELD_VERSION,
            "variant": self.variant,
            "active_evidence_blocks": list(active_evidence_blocks(self.variant)),
            "base_feature_dim": self.base_feature_dim,
            "image_feature_dim": self.image_feature_dim,
            "num_depths": self.num_depths,
            "num_thresholds": self.num_thresholds,
            "action_pose_dim_per_action": ACTION_POSE_DIM,
            "action_pose_dim_flat": self.action_pose_dim,
            "projected_dim_per_action": self.projected_per_action_dim,
            "projected_dim_flat": self.projected_dim,
            "ray_dim_per_action": self.ray_per_action_dim,
            "ray_dim_flat": self.ray_dim,
            "total_input_dim": self.total_input_dim,
            "hidden_dim": self.hidden_dim,
            "increment_bias": self.increment_bias,
            "architecture": "LayerNorm-per-block_then_Linear-GELU-Linear-GELU-Linear",
            "num_linear_layers": 3,
            "scratch_initialization": "xavier_uniform",
            "uses_p1_prediction": False,
            "uses_residual_on_stage1_or_p1": False,
            "capacity_matched_fixed_input_layout": True,
        }

    @staticmethod
    def _zeros(
        reference: torch.Tensor,
        rows: int,
        width: int,
    ) -> torch.Tensor:
        return reference.new_zeros((rows, width))

    @staticmethod
    def _flatten_block(
        value: torch.Tensor,
        *,
        rows: int,
        depths: int,
        width_per_depth: int,
        name: str,
    ) -> torch.Tensor:
        expected = (rows, depths, width_per_depth)
        if value.shape != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")
        return value.reshape(rows, depths * width_per_depth)

    def build_input(
        self,
        base_feature: torch.Tensor,
        action_pose_feature: Optional[torch.Tensor] = None,
        projected_field_feature: Optional[torch.Tensor] = None,
        ray_depth_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_feature.dim() != 2 or base_feature.shape[-1] != self.base_feature_dim:
            raise ValueError(
                f"base_feature must be [R,{self.base_feature_dim}], got "
                f"{tuple(base_feature.shape)}"
            )
        rows = int(base_feature.shape[0])
        active = set(active_evidence_blocks(self.variant))
        base = self.base_norm(base_feature.float())

        if "pose" in active:
            if action_pose_feature is None:
                raise ValueError(f"{self.variant} requires action_pose_feature")
            pose = self._flatten_block(
                action_pose_feature.float(),
                rows=rows,
                depths=self.num_depths,
                width_per_depth=ACTION_POSE_DIM,
                name="action_pose_feature",
            )
            pose = self.pose_norm(pose)
        else:
            pose = self._zeros(base, rows, self.action_pose_dim)

        if "projected" in active:
            if projected_field_feature is None:
                raise ValueError(f"{self.variant} requires projected_field_feature")
            projected = self._flatten_block(
                projected_field_feature.float(),
                rows=rows,
                depths=self.num_depths,
                width_per_depth=self.projected_per_action_dim,
                name="projected_field_feature",
            )
            projected = self.projected_norm(projected)
        else:
            projected = self._zeros(base, rows, self.projected_dim)

        if "ray_depth" in active:
            if ray_depth_feature is None:
                raise ValueError(f"{self.variant} requires ray_depth_feature")
            ray = self._flatten_block(
                ray_depth_feature.float(),
                rows=rows,
                depths=self.num_depths,
                width_per_depth=self.ray_per_action_dim,
                name="ray_depth_feature",
            )
            ray = self.ray_norm(ray)
        else:
            ray = self._zeros(base, rows, self.ray_dim)

        combined = torch.cat([base, pose, projected, ray], dim=-1)
        if combined.shape != (rows, self.total_input_dim):
            raise RuntimeError(
                f"P2 combined input has shape {tuple(combined.shape)}, expected "
                f"{(rows, self.total_input_dim)}"
            )
        return torch.nan_to_num(combined, nan=0.0, posinf=8.0, neginf=-8.0)

    def forward(
        self,
        base_feature: torch.Tensor,
        action_pose_feature: Optional[torch.Tensor] = None,
        projected_field_feature: Optional[torch.Tensor] = None,
        ray_depth_feature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = self.build_input(
            base_feature,
            action_pose_feature,
            projected_field_feature,
            ray_depth_feature,
        )
        raw = self.mlp(combined).view(
            -1,
            self.num_depths,
            self.num_thresholds,
        )
        logits = monotonic_cdf_logits_from_raw(raw, self.increment_bias)
        return logits, raw


def checkpoint_predictor_contract(payload: Mapping[str, object]) -> Mapping[str, object]:
    probe = payload.get("p2_gripper_cdf_probe")
    if not isinstance(probe, Mapping):
        raise RuntimeError("Checkpoint has no p2_gripper_cdf_probe metadata")
    if str(probe.get("version", "")) != P2_PROBE_VERSION:
        raise RuntimeError(
            f"Unexpected P2 probe version {probe.get('version')!r}; "
            f"expected {P2_PROBE_VERSION!r}"
        )
    return probe
