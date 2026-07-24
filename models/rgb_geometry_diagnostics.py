from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rank0_only() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


class RGBGeometryDiagnostics(nn.Module):
    """Task-local geometry diagnostics for RGB-only grasp synthesis.

    This module does not change model outputs or gradients. It measures the
    geometry actually used by the grasp pipeline at selected query pixels:

      1. center depth / XYZ error;
      2. center normal angular error;
      3. selected-view/angle-aligned local surface-patch error;
      4. correlations between geometry error and CDF score/target/regret;
      5. geometry quality among the highest-ranked candidates.

    The local patch is aligned with the CDF-selected grasp rotation. This is
    more task-relevant than whole-image depth MAE, while remaining lightweight
    enough to run periodically during training or validation.
    """

    def __init__(
        self,
        *,
        num_angle: int,
        num_depth: int,
        batch_viewpoint_params_to_matrix_fn,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        patch_size: int = 5,
        metric_radius: float = 0.04,
        radius_px_min: float = 4.0,
        radius_px_max: float = 32.0,
        normal_pixel_offset: float = 1.0,
        topk: int = 50,
        high_center_error_m: float = 0.02,
        high_patch_error_m: float = 0.02,
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        vis_num_queries: int = 256,
        vis_num_cases: int = 4,
        save_npz: bool = True,
    ):
        super().__init__()
        self.num_angle = int(num_angle)
        self.num_depth = int(num_depth)
        self.batch_viewpoint_params_to_matrix_fn = (
            batch_viewpoint_params_to_matrix_fn
        )
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.patch_size = int(patch_size)
        self.metric_radius = float(metric_radius)
        self.radius_px_min = float(radius_px_min)
        self.radius_px_max = float(radius_px_max)
        self.normal_pixel_offset = float(normal_pixel_offset)
        self.topk = max(int(topk), 1)
        self.high_center_error_m = float(high_center_error_m)
        self.high_patch_error_m = float(high_patch_error_m)
        self.vis_dir = vis_dir
        self.vis_every = max(int(vis_every), 1)
        self.vis_num_queries = max(int(vis_num_queries), 1)
        self.vis_num_cases = max(int(vis_num_cases), 1)
        self.save_npz = bool(save_npz)

        if self.patch_size < 3 or self.patch_size % 2 != 1:
            raise ValueError("geometry diagnostic patch_size must be odd and >=3")
        if self.metric_radius <= 0:
            raise ValueError("metric_radius must be positive")
        if self.radius_px_min <= 0 or self.radius_px_max < self.radius_px_min:
            raise ValueError("Require 0 < radius_px_min <= radius_px_max")
        if self.batch_viewpoint_params_to_matrix_fn is None:
            raise ValueError(
                "batch_viewpoint_params_to_matrix_fn is required"
            )

        lin = torch.linspace(-1.0, 1.0, self.patch_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        offsets = torch.stack(
            [xx.reshape(-1), yy.reshape(-1)],
            dim=-1,
        )
        self.register_buffer(
            "unit_patch_offsets",
            offsets.float(),
            persistent=False,
        )

        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

    @staticmethod
    def _flag(value: Any) -> bool:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    "diagnostic flag tensor must be scalar, got "
                    f"{tuple(value.shape)}"
                )
            return bool(value.detach().item())
        return bool(value)

    @staticmethod
    def _normalize_depth(
        depth: torch.Tensor,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4:
            depth = depth[:, :1]
        else:
            raise ValueError(
                f"depth must be [B,H,W] or [B,1,H,W], got {tuple(depth.shape)}"
            )
        depth = torch.nan_to_num(
            depth.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if target_hw is not None and depth.shape[-2:] != target_hw:
            depth = F.interpolate(
                depth,
                size=target_hw,
                mode="nearest",
            )
        return depth

    @staticmethod
    def _idx_to_uv(idx: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx % W).float()
        v = (idx // W).float()
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _backproject(
        uv: torch.Tensor,
        z: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        # uv: [B,...,2], z: [B,...], K: [B,3,3]
        extra = [1] * (uv.dim() - 2)
        fx = K[:, 0, 0].view(K.shape[0], *extra)
        fy = K[:, 1, 1].view(K.shape[0], *extra)
        cx = K[:, 0, 2].view(K.shape[0], *extra)
        cy = K[:, 1, 2].view(K.shape[0], *extra)
        x = (uv[..., 0] - cx) / fx.clamp_min(1e-6) * z
        y = (uv[..., 1] - cy) / fy.clamp_min(1e-6) * z
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _project(xyz: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        extra = [1] * (xyz.dim() - 2)
        fx = K[:, 0, 0].view(K.shape[0], *extra)
        fy = K[:, 1, 1].view(K.shape[0], *extra)
        cx = K[:, 0, 2].view(K.shape[0], *extra)
        cy = K[:, 1, 2].view(K.shape[0], *extra)
        z = xyz[..., 2].clamp_min(1e-6)
        u = fx * xyz[..., 0] / z + cx
        v = fy * xyz[..., 1] / z + cy
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _sample_depth(
        depth: torch.Tensor,
        uv: torch.Tensor,
    ) -> torch.Tensor:
        # depth [B,1,H,W], uv [B,Q,S,2] -> [B,Q,S]
        B, _, H, W = depth.shape
        gx = uv[..., 0] / max(W - 1, 1) * 2.0 - 1.0
        gy = uv[..., 1] / max(H - 1, 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        sampled = F.grid_sample(
            depth,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled[:, 0]

    @staticmethod
    def _valid_depth(
        depth: torch.Tensor,
        min_depth: float,
        max_depth: float,
    ) -> torch.Tensor:
        return (
            torch.isfinite(depth)
            & (depth > float(min_depth))
            & (depth < float(max_depth))
        )

    @staticmethod
    def _masked_mean_per_query(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # value/mask [B,Q,S]
        denom = mask.float().sum(dim=-1)
        mean = (
            (value * mask.float()).sum(dim=-1)
            / denom.clamp_min(1.0)
        )
        valid_q = denom > 0
        return mean, valid_q

    @staticmethod
    def _safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return x.new_zeros(())
        return x[mask].float().mean().reshape(())

    @staticmethod
    def _safe_quantile(
        x: torch.Tensor,
        mask: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        if not bool(mask.any()):
            return x.new_zeros(())
        return torch.quantile(x[mask].float(), float(q)).reshape(())

    @staticmethod
    def _pearson(
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        finite = mask & torch.isfinite(x) & torch.isfinite(y)
        if int(finite.sum().item()) < 2:
            return x.new_zeros(())
        xx = x[finite].float()
        yy = y[finite].float()
        xx = xx - xx.mean()
        yy = yy - yy.mean()
        denom = torch.sqrt(
            xx.square().sum().clamp_min(1e-12)
            * yy.square().sum().clamp_min(1e-12)
        )
        return ((xx * yy).sum() / denom).reshape(())

    @staticmethod
    def _cdf_target_from_bins(
        bins: torch.Tensor,
        num_thresholds: int,
    ) -> torch.Tensor:
        ids = torch.arange(
            num_thresholds,
            device=bins.device,
            dtype=bins.dtype,
        )
        return (
            (bins.unsqueeze(-1) > 0)
            & (ids >= bins.unsqueeze(-1) - 1)
        ).float()

    @staticmethod
    def _make_canvas(img: torch.Tensor) -> np.ndarray:
        x = img.detach().float().cpu()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        x = x[:3]
        x = x - x.amin()
        x = x / (x.amax() + 1e-6)
        return x.permute(1, 2, 0).numpy()

    @staticmethod
    def _clamp_projected_axis(
        vec: torch.Tensor,
        fallback: torch.Tensor,
        radius_px: torch.Tensor,
        radius_px_min: float,
        radius_px_max: float,
    ) -> torch.Tensor:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        invalid = (~torch.isfinite(norm)) | (norm < 1e-6)
        vec = torch.where(invalid, fallback, vec)
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
        target = norm.clamp(
            float(radius_px_min),
            float(radius_px_max),
        )
        target = torch.where(
            invalid,
            radius_px.unsqueeze(-1),
            target,
        )
        return vec / norm * target

    def _candidate_rotation_and_utility(
        self,
        end_points: Dict[str, Any],
    ):
        required = (
            "grasp_cdf_pred_angle_depth",
            "grasp_top_view_xyz",
        )
        missing = [key for key in required if key not in end_points]
        if missing:
            raise KeyError(
                "RGB geometry diagnostics require endpoint(s): "
                + ", ".join(missing)
            )

        logits = end_points["grasp_cdf_pred_angle_depth"].detach().float()
        views = end_points["grasp_top_view_xyz"].detach().float()
        if logits.dim() != 5:
            raise ValueError(
                "grasp_cdf_pred_angle_depth must be [B,T,Q,A,D], got "
                f"{tuple(logits.shape)}"
            )
        B, T, Q, A, D = logits.shape
        if A != self.num_angle or D != self.num_depth:
            raise ValueError(
                f"CDF A/D mismatch: expected {self.num_angle}/{self.num_depth}, "
                f"got {A}/{D}"
            )
        if views.shape != (B, Q, 3):
            raise ValueError(
                "grasp_top_view_xyz must be [B,Q,3], got "
                f"{tuple(views.shape)}"
            )

        utility = torch.sigmoid(logits).mean(dim=1)  # [B,Q,A,D]
        flat = utility.reshape(B, Q, A * D)
        joint_idx = flat.argmax(dim=-1)
        angle_idx = torch.div(
            joint_idx,
            D,
            rounding_mode="floor",
        )
        depth_idx = torch.remainder(joint_idx, D)
        selected_utility = torch.gather(
            flat,
            dim=-1,
            index=joint_idx.unsqueeze(-1),
        ).squeeze(-1)

        angle_rad = angle_idx.float() * (math.pi / float(A))
        rotation = self.batch_viewpoint_params_to_matrix_fn(
            -views.reshape(-1, 3),
            angle_rad.reshape(-1),
        ).view(B, Q, 3, 3)

        return (
            logits,
            utility,
            joint_idx,
            angle_idx,
            depth_idx,
            selected_utility,
            rotation,
        )

    def _make_candidate_patch_uv(
        self,
        center_xyz: torch.Tensor,
        center_uv: torch.Tensor,
        rotation: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        B, Q, _ = center_xyz.shape
        axis_y = rotation[..., :, 1]
        axis_z = rotation[..., :, 2]

        uv_y = self._project(
            center_xyz + axis_y * self.metric_radius,
            K,
        )
        uv_z = self._project(
            center_xyz + axis_z * self.metric_radius,
            K,
        )
        vec_y = uv_y - center_uv
        vec_z = uv_z - center_uv

        fmean = 0.5 * (
            K[:, 0, 0].view(B, 1)
            + K[:, 1, 1].view(B, 1)
        )
        radius_px = (
            fmean
            * self.metric_radius
            / center_xyz[..., 2].clamp_min(1e-6)
        ).clamp(self.radius_px_min, self.radius_px_max)

        fallback_y = torch.zeros_like(vec_y)
        fallback_z = torch.zeros_like(vec_z)
        fallback_y[..., 0] = radius_px
        fallback_z[..., 1] = radius_px
        vec_y = self._clamp_projected_axis(
            vec_y,
            fallback_y,
            radius_px,
            self.radius_px_min,
            self.radius_px_max,
        )
        vec_z = self._clamp_projected_axis(
            vec_z,
            fallback_z,
            radius_px,
            self.radius_px_min,
            self.radius_px_max,
        )

        offsets = self.unit_patch_offsets.to(
            device=center_xyz.device,
            dtype=center_xyz.dtype,
        )
        return (
            center_uv.unsqueeze(2)
            + offsets[:, 0].view(1, 1, -1, 1) * vec_y.unsqueeze(2)
            + offsets[:, 1].view(1, 1, -1, 1) * vec_z.unsqueeze(2)
        )

    def _normal_error(
        self,
        center_uv: torch.Tensor,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        K: torch.Tensor,
    ):
        B, Q, _ = center_uv.shape
        o = float(self.normal_pixel_offset)
        delta = center_uv.new_tensor(
            [[-o, 0.0], [o, 0.0], [0.0, -o], [0.0, o]]
        )
        uv = center_uv.unsqueeze(2) + delta.view(1, 1, 4, 2)

        pred_z = self._sample_depth(pred_depth, uv)
        gt_z = self._sample_depth(gt_depth, uv)
        _, _, H, W = pred_depth.shape
        uv_valid = (
            (uv[..., 0] >= 0)
            & (uv[..., 0] <= W - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= H - 1)
        )
        pred_valid = uv_valid & self._valid_depth(
            pred_z, self.min_depth, self.max_depth
        )
        gt_valid = uv_valid & self._valid_depth(
            gt_z, self.min_depth, self.max_depth
        )
        normal_valid = pred_valid.all(dim=-1) & gt_valid.all(dim=-1)

        pred_xyz = self._backproject(uv, pred_z, K)
        gt_xyz = self._backproject(uv, gt_z, K)

        pred_dx = pred_xyz[:, :, 1] - pred_xyz[:, :, 0]
        pred_dy = pred_xyz[:, :, 3] - pred_xyz[:, :, 2]
        gt_dx = gt_xyz[:, :, 1] - gt_xyz[:, :, 0]
        gt_dy = gt_xyz[:, :, 3] - gt_xyz[:, :, 2]

        pred_n = F.normalize(
            torch.cross(pred_dx, pred_dy, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        gt_n = F.normalize(
            torch.cross(gt_dx, gt_dy, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        cos = (pred_n * gt_n).sum(dim=-1).abs().clamp(0.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos.clamp(max=1.0 - 1e-7)))
        return angle, normal_valid

    @torch.no_grad()
    def forward(
        self,
        *,
        end_points: Dict[str, Any],
        depth_pred: torch.Tensor,
        gt_depth: Optional[torch.Tensor],
        K: torch.Tensor,
        img: Optional[torch.Tensor],
        step: int,
        modality: str,
    ) -> Dict[str, Any]:
        run_scalar = self._flag(
            end_points.get("cva_compute_diagnostics", False)
        )
        run_vis = (
            self.vis_dir is not None
            and int(step) % self.vis_every == 0
            and _rank0_only()
        )
        if not run_scalar and not run_vis:
            return end_points
        if gt_depth is None:
            # Geometry error cannot be defined without GT depth.
            end_points["D: RGBGeom GT available"] = depth_pred.new_zeros(())
            return end_points

        required = (
            "token_sel_idx",
            "xyz_graspable",
        )
        missing = [key for key in required if key not in end_points]
        if missing:
            raise KeyError(
                "RGB geometry diagnostics require endpoint(s): "
                + ", ".join(missing)
            )

        pred = self._normalize_depth(depth_pred)
        B, _, H, W = pred.shape
        gt = self._normalize_depth(gt_depth, target_hw=(H, W)).to(pred)
        K = K.detach().float().to(pred.device)

        token_idx = end_points["token_sel_idx"].detach().long()
        if token_idx.dim() != 2 or token_idx.shape[0] != B:
            raise ValueError(
                "token_sel_idx must be [B,Q], got "
                f"{tuple(token_idx.shape)}"
            )
        Q = token_idx.shape[1]
        if bool((token_idx < 0).any()) or bool((token_idx >= H * W).any()):
            raise ValueError("token_sel_idx contains out-of-range pixels")

        (
            _logits,
            _utility,
            joint_idx,
            angle_idx,
            depth_idx,
            selected_utility,
            rotation,
        ) = self._candidate_rotation_and_utility(end_points)
        if selected_utility.shape != (B, Q):
            raise ValueError(
                "CDF query count does not match token_sel_idx: "
                f"CDF={tuple(selected_utility.shape)}, token={tuple(token_idx.shape)}"
            )

        center_uv = self._idx_to_uv(token_idx, H, W).to(pred.device)
        pred_center_z = pred[:, 0].reshape(B, -1).gather(1, token_idx)
        gt_center_z = gt[:, 0].reshape(B, -1).gather(1, token_idx)
        center_valid = (
            self._valid_depth(
                pred_center_z, self.min_depth, self.max_depth
            )
            & self._valid_depth(
                gt_center_z, self.min_depth, self.max_depth
            )
        )
        pred_center_xyz = self._backproject(
            center_uv,
            pred_center_z,
            K,
        )
        gt_center_xyz = self._backproject(
            center_uv,
            gt_center_z,
            K,
        )
        center_z_error = (pred_center_z - gt_center_z).abs()
        center_xyz_error = torch.linalg.norm(
            pred_center_xyz - gt_center_xyz,
            dim=-1,
        )

        model_center = end_points["xyz_graspable"].detach().float()
        if model_center.shape != (B, Q, 3):
            raise ValueError(
                "xyz_graspable must be [B,Q,3], got "
                f"{tuple(model_center.shape)}"
            )
        model_vs_pred = torch.linalg.norm(
            model_center - pred_center_xyz,
            dim=-1,
        )
        model_vs_gt = torch.linalg.norm(
            model_center - gt_center_xyz,
            dim=-1,
        )

        patch_uv = self._make_candidate_patch_uv(
            pred_center_xyz,
            center_uv,
            rotation,
            K,
        )
        S = patch_uv.shape[2]
        uv_valid = (
            (patch_uv[..., 0] >= 0)
            & (patch_uv[..., 0] <= W - 1)
            & (patch_uv[..., 1] >= 0)
            & (patch_uv[..., 1] <= H - 1)
        )
        pred_patch = self._sample_depth(pred, patch_uv)
        gt_patch = self._sample_depth(gt, patch_uv)
        patch_valid = (
            uv_valid
            & self._valid_depth(
                pred_patch, self.min_depth, self.max_depth
            )
            & self._valid_depth(
                gt_patch, self.min_depth, self.max_depth
            )
            & center_valid.unsqueeze(-1)
        )

        patch_abs_error, patch_query_valid = self._masked_mean_per_query(
            (pred_patch - gt_patch).abs(),
            patch_valid,
        )
        pred_shape = pred_patch - pred_center_z.unsqueeze(-1)
        gt_shape = gt_patch - gt_center_z.unsqueeze(-1)
        patch_shape_error, patch_shape_valid = self._masked_mean_per_query(
            (pred_shape - gt_shape).abs(),
            patch_valid,
        )
        surface_order_agree, order_valid = self._masked_mean_per_query(
            (
                torch.sign(pred_shape)
                == torch.sign(gt_shape)
            ).float(),
            patch_valid,
        )

        normal_error, normal_valid = self._normal_error(
            center_uv,
            pred,
            gt,
            K,
        )

        # Candidate target utility and selected-operation regret.
        selected_target = None
        oracle_target = None
        regret = None
        selected_label_valid = None
        bins = end_points.get(
            "batch_grasp_cdf_bins_angle_depth", None
        )
        candidate_valid = end_points.get(
            "batch_grasp_cdf_valid_mask", None
        )
        if (
            torch.is_tensor(bins)
            and torch.is_tensor(candidate_valid)
            and tuple(bins.shape[:2]) == (B, Q)
            and candidate_valid.shape == bins.shape
        ):
            bins = bins.detach().long().to(pred.device)
            candidate_valid = candidate_valid.detach().bool().to(pred.device)
            T = int(
                end_points["grasp_cdf_pred_angle_depth"].shape[1]
            )
            target = self._cdf_target_from_bins(bins, T)
            target_utility = target.mean(dim=-1)
            flat_target = target_utility.reshape(
                B, Q, self.num_angle * self.num_depth
            )
            flat_valid = candidate_valid.reshape(
                B, Q, self.num_angle * self.num_depth
            )
            selected_target = torch.gather(
                flat_target,
                dim=-1,
                index=joint_idx.unsqueeze(-1),
            ).squeeze(-1)
            selected_label_valid = torch.gather(
                flat_valid,
                dim=-1,
                index=joint_idx.unsqueeze(-1),
            ).squeeze(-1)
            oracle_target = flat_target.masked_fill(
                ~flat_valid,
                -1.0,
            ).max(dim=-1).values.clamp_min(0.0)
            regret = (oracle_target - selected_target).clamp_min(0.0)

        p = "D: RGBGeom"
        end_points[f"{p} GT available"] = pred.new_ones(())
        end_points[f"{p} center valid ratio"] = (
            center_valid.float().mean().reshape(())
        )
        end_points[f"{p} center z MAE"] = self._safe_mean(
            center_z_error, center_valid
        )
        end_points[f"{p} center z p50"] = self._safe_quantile(
            center_z_error, center_valid, 0.50
        )
        end_points[f"{p} center z p90"] = self._safe_quantile(
            center_z_error, center_valid, 0.90
        )
        end_points[f"{p} center xyz MAE"] = self._safe_mean(
            center_xyz_error, center_valid
        )
        end_points[f"{p} center xyz p90"] = self._safe_quantile(
            center_xyz_error, center_valid, 0.90
        )
        end_points[f"{p} model-pred center delta"] = self._safe_mean(
            model_vs_pred, center_valid
        )
        end_points[f"{p} model-GT center delta"] = self._safe_mean(
            model_vs_gt, center_valid
        )
        end_points[f"{p} model closer GT ratio"] = self._safe_mean(
            (model_vs_gt < model_vs_pred).float(),
            center_valid,
        )

        end_points[f"{p} patch valid ratio"] = (
            patch_valid.float().mean().reshape(())
        )
        end_points[f"{p} patch query valid ratio"] = (
            patch_query_valid.float().mean().reshape(())
        )
        end_points[f"{p} patch depth MAE"] = self._safe_mean(
            patch_abs_error, patch_query_valid
        )
        end_points[f"{p} patch depth p90"] = self._safe_quantile(
            patch_abs_error, patch_query_valid, 0.90
        )
        end_points[f"{p} patch shape MAE"] = self._safe_mean(
            patch_shape_error, patch_shape_valid
        )
        end_points[f"{p} patch shape p90"] = self._safe_quantile(
            patch_shape_error, patch_shape_valid, 0.90
        )
        end_points[f"{p} surface order agreement"] = self._safe_mean(
            surface_order_agree, order_valid
        )
        end_points[f"{p} normal valid ratio"] = (
            normal_valid.float().mean().reshape(())
        )
        end_points[f"{p} normal angle mean"] = self._safe_mean(
            normal_error, normal_valid
        )
        end_points[f"{p} normal angle p90"] = self._safe_quantile(
            normal_error, normal_valid, 0.90
        )

        end_points[f"{p} score x center zerr"] = self._pearson(
            selected_utility,
            center_z_error,
            center_valid,
        )
        end_points[f"{p} score x patch shapeerr"] = self._pearson(
            selected_utility,
            patch_shape_error,
            patch_shape_valid,
        )
        end_points[f"{p} highscore high centererr ratio"] = self._safe_mean(
            (
                (selected_utility >= 0.9)
                & (center_z_error >= self.high_center_error_m)
            ).float(),
            center_valid,
        )
        end_points[f"{p} highscore high patcherr ratio"] = self._safe_mean(
            (
                (selected_utility >= 0.9)
                & (patch_shape_error >= self.high_patch_error_m)
            ).float(),
            patch_shape_valid,
        )

        if selected_target is not None:
            label_mask = selected_label_valid & center_valid
            patch_label_mask = selected_label_valid & patch_shape_valid
            normal_label_mask = selected_label_valid & normal_valid
            end_points[f"{p} selected label valid ratio"] = (
                selected_label_valid.float().mean().reshape(())
            )
            end_points[f"{p} target x center zerr"] = self._pearson(
                selected_target,
                center_z_error,
                label_mask,
            )
            end_points[f"{p} target x center xyzerr"] = self._pearson(
                selected_target,
                center_xyz_error,
                label_mask,
            )
            end_points[f"{p} target x patch shapeerr"] = self._pearson(
                selected_target,
                patch_shape_error,
                patch_label_mask,
            )
            end_points[f"{p} target x normalerr"] = self._pearson(
                selected_target,
                normal_error,
                normal_label_mask,
            )
            end_points[f"{p} regret x center zerr"] = self._pearson(
                regret,
                center_z_error,
                label_mask,
            )
            end_points[f"{p} regret x patch shapeerr"] = self._pearson(
                regret,
                patch_shape_error,
                patch_label_mask,
            )

        # Geometry quality among candidates that would dominate ranking.
        for topk in (10, self.topk):
            topk = max(int(topk), 1)
            z_values = []
            patch_values = []
            normal_values = []
            for batch_i in range(B):
                rank_pool = torch.where(center_valid[batch_i])[0]
                if rank_pool.numel() == 0:
                    continue
                k_eff = min(topk, int(rank_pool.numel()))
                top_local = torch.topk(
                    selected_utility[batch_i, rank_pool],
                    k=k_eff,
                    largest=True,
                ).indices
                q_idx = rank_pool[top_local]
                z_values.append(center_z_error[batch_i, q_idx].mean())
                q_patch = q_idx[patch_shape_valid[batch_i, q_idx]]
                if q_patch.numel() > 0:
                    patch_values.append(
                        patch_shape_error[batch_i, q_patch].mean()
                    )
                q_normal = q_idx[normal_valid[batch_i, q_idx]]
                if q_normal.numel() > 0:
                    normal_values.append(
                        normal_error[batch_i, q_normal].mean()
                    )
            if z_values:
                end_points[f"{p} top{topk} center z MAE"] = (
                    torch.stack(z_values).mean().reshape(())
                )
            if patch_values:
                end_points[f"{p} top{topk} patch shape MAE"] = (
                    torch.stack(patch_values).mean().reshape(())
                )
            if normal_values:
                end_points[f"{p} top{topk} normal angle"] = (
                    torch.stack(normal_values).mean().reshape(())
                )

        # Detached arrays for ad-hoc analysis; first batch only and bounded.
        n_export = min(Q, self.vis_num_queries)
        q_export = torch.topk(
            selected_utility[0],
            k=n_export,
            largest=True,
        ).indices
        end_points["rgbgeom_debug_query_idx0"] = q_export.detach().cpu()
        end_points["rgbgeom_debug_center_z_error0"] = (
            center_z_error[0, q_export].detach().cpu()
        )
        end_points["rgbgeom_debug_patch_shape_error0"] = (
            patch_shape_error[0, q_export].detach().cpu()
        )
        end_points["rgbgeom_debug_normal_error0"] = (
            normal_error[0, q_export].detach().cpu()
        )
        end_points["rgbgeom_debug_selected_utility0"] = (
            selected_utility[0, q_export].detach().cpu()
        )
        if selected_target is not None:
            end_points["rgbgeom_debug_selected_target0"] = (
                selected_target[0, q_export].detach().cpu()
            )
            end_points["rgbgeom_debug_regret0"] = (
                regret[0, q_export].detach().cpu()
            )

        if run_vis:
            self._save_visualization(
                end_points=end_points,
                img=img,
                modality=modality,
                step=step,
                center_uv=center_uv,
                selected_utility=selected_utility,
                selected_target=selected_target,
                center_z_error=center_z_error,
                patch_shape_error=patch_shape_error,
                normal_error=normal_error,
                center_valid=center_valid,
                patch_shape_valid=patch_shape_valid,
                normal_valid=normal_valid,
                pred_shape=pred_shape,
                gt_shape=gt_shape,
                patch_valid=patch_valid,
                angle_idx=angle_idx,
                depth_idx=depth_idx,
            )
        return end_points

    @torch.no_grad()
    def _save_visualization(
        self,
        *,
        end_points: Dict[str, Any],
        img: Optional[torch.Tensor],
        modality: str,
        step: int,
        center_uv: torch.Tensor,
        selected_utility: torch.Tensor,
        selected_target: Optional[torch.Tensor],
        center_z_error: torch.Tensor,
        patch_shape_error: torch.Tensor,
        normal_error: torch.Tensor,
        center_valid: torch.Tensor,
        patch_shape_valid: torch.Tensor,
        normal_valid: torch.Tensor,
        pred_shape: torch.Tensor,
        gt_shape: torch.Tensor,
        patch_valid: torch.Tensor,
        angle_idx: torch.Tensor,
        depth_idx: torch.Tensor,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "RGB geometry visualization requires Matplotlib"
            ) from exc

        prefix = f"rgbgeom_{str(modality)}_it{int(step):06d}"
        b = 0
        Q = selected_utility.shape[1]
        nvis = min(Q, self.vis_num_queries)
        q_sel = torch.topk(
            selected_utility[b],
            k=nvis,
            largest=True,
        ).indices

        if img is not None and torch.is_tensor(img):
            canvas = self._make_canvas(img[b])
            uv = center_uv[b, q_sel].detach().cpu().numpy()

            for values, valid, name, label in [
                (
                    center_z_error[b, q_sel],
                    center_valid[b, q_sel],
                    "center_z_error_overlay",
                    "center |z_pred-z_gt| (m)",
                ),
                (
                    patch_shape_error[b, q_sel],
                    patch_shape_valid[b, q_sel],
                    "patch_shape_error_overlay",
                    "candidate-aligned local shape error (m)",
                ),
                (
                    normal_error[b, q_sel],
                    normal_valid[b, q_sel],
                    "normal_error_overlay",
                    "normal angular error (deg)",
                ),
            ]:
                vals = values.detach().float().cpu().numpy()
                vm = valid.detach().cpu().numpy().astype(bool)
                fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=160)
                ax.imshow(canvas)
                if vm.any():
                    sc = ax.scatter(
                        uv[vm, 0],
                        uv[vm, 1],
                        c=vals[vm],
                        s=12,
                        cmap="magma",
                    )
                    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=label)
                if (~vm).any():
                    ax.scatter(
                        uv[~vm, 0],
                        uv[~vm, 1],
                        c="gray",
                        s=10,
                        marker="x",
                    )
                ax.axis("off")
                ax.set_title(f"{modality}: {label}")
                fig.tight_layout(pad=0.1)
                fig.savefig(
                    os.path.join(self.vis_dir, f"{prefix}_{name}.png")
                )
                plt.close(fig)

        # Geometry versus CDF score/target.
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), dpi=160)
        score = selected_utility[b]
        pairs = [
            (center_z_error[b], center_valid[b], "center z error (m)"),
            (
                patch_shape_error[b],
                patch_shape_valid[b],
                "local shape error (m)",
            ),
            (normal_error[b], normal_valid[b], "normal error (deg)"),
        ]
        for ax, (err, mask, ylabel) in zip(axes.reshape(-1)[:3], pairs):
            m = mask & torch.isfinite(err) & torch.isfinite(score)
            if bool(m.any()):
                ax.scatter(
                    score[m].cpu().numpy(),
                    err[m].cpu().numpy(),
                    s=5,
                    alpha=0.25,
                )
            ax.set_xlabel("predicted CDF utility")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
        ax = axes.reshape(-1)[3]
        if selected_target is not None:
            m = patch_shape_valid[b] & torch.isfinite(selected_target[b])
            if bool(m.any()):
                ax.scatter(
                    selected_target[b, m].cpu().numpy(),
                    patch_shape_error[b, m].cpu().numpy(),
                    s=5,
                    alpha=0.25,
                )
            ax.set_xlabel("selected target utility")
            ax.set_ylabel("local shape error (m)")
        else:
            ax.axis("off")
        ax.grid(alpha=0.2)
        fig.suptitle(f"{modality}: task-local geometry diagnostics")
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.vis_dir, f"{prefix}_scatter.png")
        )
        plt.close(fig)

        # High-score/high-geometry-error cases: predicted vs GT centered patches.
        hard_score = (
            selected_utility[b]
            + patch_shape_error[b]
            / max(self.high_patch_error_m, 1e-6)
        ).masked_fill(~patch_shape_valid[b], -1.0)
        ncase = min(self.vis_num_cases, int(patch_shape_valid[b].sum().item()))
        if ncase > 0:
            case_idx = torch.topk(
                hard_score,
                k=ncase,
                largest=True,
            ).indices
            fig, axes = plt.subplots(
                ncase,
                3,
                figsize=(9.0, 2.8 * ncase),
                dpi=160,
                squeeze=False,
            )
            P = self.patch_size
            for row, q in enumerate(case_idx.tolist()):
                pm = patch_valid[b, q].view(P, P).cpu().numpy()
                pred_np = pred_shape[b, q].view(P, P).cpu().numpy()
                gt_np = gt_shape[b, q].view(P, P).cpu().numpy()
                diff = np.abs(pred_np - gt_np)
                for col, (arr, title) in enumerate(
                    [
                        (np.where(pm, pred_np, np.nan), "pred centered depth"),
                        (np.where(pm, gt_np, np.nan), "GT centered depth"),
                        (np.where(pm, diff, np.nan), "absolute shape error"),
                    ]
                ):
                    im = axes[row, col].imshow(
                        arr,
                        cmap="coolwarm" if col < 2 else "magma",
                    )
                    axes[row, col].set_title(
                        f"{title}\nq={q}, a={int(angle_idx[b,q])}, "
                        f"d={int(depth_idx[b,q])+1}cm, "
                        f"U={float(selected_utility[b,q]):.2f}"
                    )
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    fig.colorbar(im, ax=axes[row, col], fraction=0.046)
            fig.tight_layout()
            fig.savefig(
                os.path.join(self.vis_dir, f"{prefix}_hard_cases.png")
            )
            plt.close(fig)

        if self.save_npz:
            dump = {
                "query_idx": q_sel.cpu().numpy().astype(np.int32),
                "center_uv": center_uv[b, q_sel].cpu().numpy(),
                "selected_utility": selected_utility[b, q_sel].cpu().numpy(),
                "center_z_error": center_z_error[b, q_sel].cpu().numpy(),
                "patch_shape_error": patch_shape_error[b, q_sel].cpu().numpy(),
                "normal_error_deg": normal_error[b, q_sel].cpu().numpy(),
                "angle_idx": angle_idx[b, q_sel].cpu().numpy().astype(np.int16),
                "depth_idx": depth_idx[b, q_sel].cpu().numpy().astype(np.int16),
            }
            if selected_target is not None:
                dump["selected_target_utility"] = (
                    selected_target[b, q_sel].cpu().numpy()
                )
            np.savez_compressed(
                os.path.join(self.vis_dir, f"{prefix}.npz"),
                **dump,
            )
