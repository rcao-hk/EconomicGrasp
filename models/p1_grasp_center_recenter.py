"""P1 grasp-query center recentering for RGB-only EconomicGrasp-DPT CVA-CDF.

P1 keeps the Stage-1 dense metric-depth map unchanged and corrects only the
selected sparse grasp-query centers.  For each student-owned image-FPS seed,
a small RGB-feature-conditioned head predicts a scalar z-depth residual.  The
corrected z is backprojected through the same pixel and camera intrinsics and is
then consumed by ViewNet/CVA local analysis and the standard decoder.

Training privilege is limited to ``gt_depth_m`` at the selected pixels.  The
ground-truth depth is used only to supervise the residual head; it is never an
input to the correction prediction.  At inference, the P1 model can run after
``gt_depth_m`` is removed from the batch.

Three trainer modes are supported by the entry point:
  * center_only: freeze the Stage-1 model and stop grasp-loss gradients at the
    corrected center.  Only the direct center loss trains the new head.
  * head_grasp: freeze the Stage-1 model but allow existing grasp losses to
    update the new head through the corrected center.
  * joint: jointly fine-tune Stage-1 and the new head.

The last linear layer is zero-initialized, so a newly attached P1 head initially
reproduces the Stage-1 physical centers exactly.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .economicgrasp_dpt_distill import economicgrasp_dpt_student


P1_CENTER_RECENTER_CONTRACT_VERSION = 1


class economicgrasp_dpt_p1_center_recenter(economicgrasp_dpt_student):
    """Stage-1 RGB student plus a sparse grasp-query z-recentering head."""

    def __init__(
        self,
        *args: Any,
        p1_hidden_dim: int = 128,
        p1_max_residual_m: float = 0.08,
        p1_freeze_base: bool = True,
        p1_detach_head_features: bool = True,
        p1_detach_corrected_for_downstream: bool = True,
        p1_compute_gt_targets: bool | None = None,
        p1_force_zero_residual: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if str(getattr(self, "geometry_depth_source", "")) != "pred":
            raise RuntimeError(
                "P1 center recentering is defined only for the RGB predicted-depth "
                "student path."
            )
        self.p1_hidden_dim = int(p1_hidden_dim)
        self.p1_max_residual_m = float(p1_max_residual_m)
        self.p1_freeze_base = bool(p1_freeze_base)
        self.p1_detach_head_features = bool(p1_detach_head_features)
        self.p1_detach_corrected_for_downstream = bool(
            p1_detach_corrected_for_downstream
        )
        self.p1_compute_gt_targets = (
            bool(self.is_training)
            if p1_compute_gt_targets is None
            else bool(p1_compute_gt_targets)
        )
        self.p1_force_zero_residual = bool(p1_force_zero_residual)

        if self.p1_hidden_dim <= 0:
            raise ValueError("p1_hidden_dim must be positive.")
        if not (0.0 < self.p1_max_residual_m <= 0.25):
            raise ValueError(
                "p1_max_residual_m must be in (0, 0.25] metres; got "
                f"{self.p1_max_residual_m}."
            )

        in_dim = int(self.seed_feature_dim) + 3
        self.p1_center_recenter_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.p1_hidden_dim),
            nn.GELU(),
            nn.Linear(self.p1_hidden_dim, self.p1_hidden_dim),
            nn.GELU(),
            nn.Linear(self.p1_hidden_dim, 1),
        )
        # Exact Stage-1 behavior at attachment time.
        nn.init.zeros_(self.p1_center_recenter_head[-1].weight)
        nn.init.zeros_(self.p1_center_recenter_head[-1].bias)

        if self.p1_freeze_base:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.p1_center_recenter_head.parameters():
                parameter.requires_grad_(True)

    @staticmethod
    def _token_uv(
        token_sel_idx: torch.Tensor,
        height: int,
        width: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if token_sel_idx.dim() != 2:
            raise ValueError(
                f"token_sel_idx must be [B,M], got {tuple(token_sel_idx.shape)}"
            )
        if bool(
            ((token_sel_idx < 0) | (token_sel_idx >= int(height * width))).any()
        ):
            raise ValueError("token_sel_idx contains an out-of-range pixel index.")
        u = (token_sel_idx % int(width)).to(dtype=dtype)
        v = (token_sel_idx // int(width)).to(dtype=dtype)
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _gather_depth(
        depth_b1hw: torch.Tensor,
        token_sel_idx: torch.Tensor,
    ) -> torch.Tensor:
        if depth_b1hw.dim() == 3:
            depth_b1hw = depth_b1hw.unsqueeze(1)
        if depth_b1hw.dim() != 4 or depth_b1hw.shape[1] < 1:
            raise ValueError(
                "Depth map must be [B,H,W] or [B,1,H,W], got "
                f"{tuple(depth_b1hw.shape)}"
            )
        B = depth_b1hw.shape[0]
        flat = depth_b1hw[:, 0].reshape(B, -1)
        return torch.gather(flat, 1, token_sel_idx.long())

    def _build_recenter_input(
        self,
        seed_features_bcm: torch.Tensor,
        base_z_bm: torch.Tensor,
        uv_bm2: torch.Tensor,
        camera_K_b33: torch.Tensor,
    ) -> torch.Tensor:
        B, C, M = seed_features_bcm.shape
        if C != int(self.seed_feature_dim):
            raise ValueError(
                f"P1 seed feature dim mismatch: got C={C}, "
                f"expected {self.seed_feature_dim}."
            )
        if base_z_bm.shape != (B, M):
            raise ValueError("base_z_bm must match [B,M].")
        if uv_bm2.shape != (B, M, 2):
            raise ValueError("uv_bm2 must match [B,M,2].")
        if camera_K_b33.shape != (B, 3, 3):
            raise ValueError(
                f"camera_K must be [B,3,3], got {tuple(camera_K_b33.shape)}"
            )

        features = seed_features_bcm.transpose(1, 2).contiguous()
        if self.p1_detach_head_features:
            features = features.detach()

        K = camera_K_b33.to(device=features.device, dtype=features.dtype)
        uv = uv_bm2.to(device=features.device, dtype=features.dtype)
        z = base_z_bm.to(device=features.device, dtype=features.dtype)

        fx = K[:, 0, 0].unsqueeze(1).clamp_min(1.0e-6)
        fy = K[:, 1, 1].unsqueeze(1).clamp_min(1.0e-6)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)
        ray_x = (uv[..., 0] - cx) / fx
        ray_y = (uv[..., 1] - cy) / fy

        depth_span = max(float(self.max_depth - self.min_depth), 1.0e-6)
        z_norm = 2.0 * (z - float(self.min_depth)) / depth_span - 1.0
        geom = torch.stack([z_norm, ray_x, ray_y], dim=-1)
        return torch.cat([features, geom], dim=-1)

    def _add_gt_center_targets(
        self,
        *,
        end_points: Dict[str, Any],
        token_sel_idx: torch.Tensor,
        base_z_bm: torch.Tensor,
        corrected_z_bm: torch.Tensor,
        applied_residual_bm: torch.Tensor,
        uv_bm2: torch.Tensor,
        camera_K: torch.Tensor,
    ) -> None:
        gt = end_points.get("gt_depth_m", None)
        if gt is None:
            raise KeyError(
                "P1 training/GT diagnostics require end_points['gt_depth_m']."
            )
        if not torch.is_tensor(gt):
            raise TypeError("gt_depth_m must be a tensor.")
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 4:
            gt = gt[:, :1]
        else:
            raise ValueError(
                f"gt_depth_m must be [B,H,W] or [B,1,H,W], got {tuple(gt.shape)}"
            )

        _, _, H, W = gt.shape
        expected_hw = tuple(end_points["depth_map_used_for_geometry"].shape[-2:])
        if (H, W) != expected_hw:
            raise RuntimeError(
                "P1 refuses to resample GT depth for query supervision because "
                "that would change boundary targets: "
                f"gt={(H, W)}, geometry={expected_hw}."
            )

        gt_z = self._gather_depth(
            gt.to(device=base_z_bm.device, dtype=base_z_bm.dtype),
            token_sel_idx,
        )
        valid = (
            torch.isfinite(gt_z)
            & (gt_z >= float(self.min_depth))
            & (gt_z <= float(self.max_depth))
            & torch.isfinite(base_z_bm)
        )
        target_residual = gt_z - base_z_bm

        gt_xyz = self._backproject_uvz(
            uv_bm2.to(dtype=base_z_bm.dtype),
            gt_z.unsqueeze(-1),
            camera_K.to(dtype=base_z_bm.dtype),
        )
        corrected_xyz = self._backproject_uvz(
            uv_bm2.to(dtype=base_z_bm.dtype),
            corrected_z_bm.unsqueeze(-1),
            camera_K.to(dtype=base_z_bm.dtype),
        )
        base_xyz = self._backproject_uvz(
            uv_bm2.to(dtype=base_z_bm.dtype),
            base_z_bm.unsqueeze(-1),
            camera_K.to(dtype=base_z_bm.dtype),
        )

        end_points["p1_center_depth_gt"] = gt_z
        end_points["p1_center_residual_target"] = target_residual
        end_points["p1_center_residual_valid_mask"] = valid
        end_points["p1_center_xyz_gt"] = gt_xyz

        with torch.no_grad():
            zero = base_z_bm.detach().new_zeros(())
            end_points["D: P1 center GT valid ratio"] = valid.float().mean()
            if bool(valid.any()):
                end_points["D: P1 center base z MAE"] = (
                    (base_z_bm - gt_z).abs()[valid].mean()
                )
                end_points["D: P1 center corrected z MAE"] = (
                    (corrected_z_bm - gt_z).abs()[valid].mean()
                )
                end_points["D: P1 center base xyz MAE"] = torch.linalg.norm(
                    base_xyz - gt_xyz, dim=-1
                )[valid].mean()
                end_points["D: P1 center corrected xyz MAE"] = torch.linalg.norm(
                    corrected_xyz - gt_xyz, dim=-1
                )[valid].mean()
                end_points["D: P1 target residual abs mean"] = (
                    target_residual.abs()[valid].mean()
                )
                end_points["D: P1 target within residual range"] = (
                    target_residual.abs()[valid]
                    <= float(self.p1_max_residual_m)
                ).float().mean()
                end_points["D: P1 residual sign agreement"] = (
                    (
                        torch.sign(applied_residual_bm[valid])
                        == torch.sign(target_residual[valid])
                    )
                    | (target_residual[valid].abs() < 1.0e-6)
                ).float().mean()
            else:
                end_points["D: P1 center base z MAE"] = zero
                end_points["D: P1 center corrected z MAE"] = zero
                end_points["D: P1 center base xyz MAE"] = zero
                end_points["D: P1 center corrected xyz MAE"] = zero
                end_points["D: P1 target residual abs mean"] = zero
                end_points["D: P1 target within residual range"] = zero
                end_points["D: P1 residual sign agreement"] = zero

    def _select_graspable_seed_queries(
        self,
        feat_grid: torch.Tensor,
        depth_map: torch.Tensor,
        camera_K: torch.Tensor,
        graspable_mask: torch.Tensor,
        valid_tok: torch.Tensor,
        grasp_score: torch.Tensor,
        end_points: dict,
    ):
        (
            seed_features,
            seed_xyz_base,
            token_sel_idx,
            xyz_all_pred,
            uv_all,
            graspable_num_batch,
        ) = super()._select_graspable_seed_queries(
            feat_grid=feat_grid,
            depth_map=depth_map,
            camera_K=camera_K,
            graspable_mask=graspable_mask,
            valid_tok=valid_tok,
            grasp_score=grasp_score,
            end_points=end_points,
        )

        if seed_xyz_base.dim() != 3 or seed_xyz_base.shape[-1] != 3:
            raise ValueError(
                "Base sparse-query selector returned invalid center shape: "
                f"{tuple(seed_xyz_base.shape)}"
            )
        B, M, _ = seed_xyz_base.shape
        H, W = int(depth_map.shape[-2]), int(depth_map.shape[-1])
        uv = self._token_uv(
            token_sel_idx,
            H,
            W,
            dtype=seed_xyz_base.dtype,
        ).to(seed_xyz_base.device)

        # The current EconomicGrasp sparse backprojection already detaches the
        # dense metric depth before forming query centers.  Preserve that
        # contract: P1 learns an additional sparse residual rather than turning
        # the full DPT map into a target of the center loss.
        base_z = seed_xyz_base[..., 2].detach()
        recenter_input = self._build_recenter_input(
            seed_features,
            base_z,
            uv,
            camera_K,
        )
        raw_residual = self.p1_center_recenter_head(recenter_input).squeeze(-1)
        residual = float(self.p1_max_residual_m) * torch.tanh(raw_residual)
        if self.p1_force_zero_residual:
            residual = residual * 0.0

        corrected_z_unclamped = base_z + residual
        corrected_z = corrected_z_unclamped.clamp(
            min=float(self.min_depth),
            max=float(self.max_depth),
        )
        applied_residual = corrected_z - base_z
        corrected_xyz = self._backproject_uvz(
            uv,
            corrected_z.unsqueeze(-1),
            camera_K.to(device=uv.device, dtype=uv.dtype),
        )

        # Endpoints retain both physical query sets for diagnostics.  The
        # returned center controls every downstream view/CVA/decode operation.
        end_points["p1_center_xyz_base"] = seed_xyz_base
        end_points["p1_center_xyz_corrected"] = corrected_xyz
        end_points["p1_center_depth_base"] = base_z
        end_points["p1_center_depth_corrected"] = corrected_z
        end_points["p1_center_residual_raw"] = raw_residual
        end_points["p1_center_residual_pred"] = residual
        end_points["p1_center_residual_applied"] = applied_residual
        end_points["p1_center_token_sel_idx"] = token_sel_idx

        with torch.no_grad():
            end_points["D: P1 enabled"] = corrected_z.new_tensor(1.0).reshape(())
            end_points["D: P1 force zero residual"] = corrected_z.new_tensor(
                float(self.p1_force_zero_residual)
            ).reshape(())
            end_points["D: P1 residual abs mean"] = applied_residual.abs().mean()
            end_points["D: P1 residual abs max"] = applied_residual.abs().max()
            end_points["D: P1 residual positive ratio"] = (
                applied_residual > 0
            ).float().mean()
            end_points["D: P1 residual clipped ratio"] = (
                (corrected_z != corrected_z_unclamped)
            ).float().mean()
            end_points["D: P1 detach corrected downstream"] = corrected_z.new_tensor(
                float(self.p1_detach_corrected_for_downstream)
            ).reshape(())
            end_points["D: P1 base frozen"] = corrected_z.new_tensor(
                float(self.p1_freeze_base)
            ).reshape(())

        if self.p1_compute_gt_targets:
            self._add_gt_center_targets(
                end_points=end_points,
                token_sel_idx=token_sel_idx,
                base_z_bm=base_z,
                corrected_z_bm=corrected_z,
                applied_residual_bm=applied_residual,
                uv_bm2=uv,
                camera_K=camera_K,
            )

        downstream_xyz = corrected_xyz
        if self.p1_detach_corrected_for_downstream:
            downstream_xyz = downstream_xyz.detach()

        return (
            seed_features,
            downstream_xyz.contiguous(),
            token_sel_idx,
            xyz_all_pred,
            uv_all,
            graspable_num_batch,
        )


def compute_p1_center_recenter_loss(
    end_points: Dict[str, Any],
    *,
    beta_m: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Direct Smooth-L1 supervision of the sparse z-depth correction."""
    pred = end_points.get("p1_center_residual_applied", None)
    target = end_points.get("p1_center_residual_target", None)
    valid = end_points.get("p1_center_residual_valid_mask", None)
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise KeyError(
            "P1 center loss requires p1_center_residual_applied and "
            "p1_center_residual_target."
        )
    if not torch.is_tensor(valid):
        raise KeyError("P1 center loss requires p1_center_residual_valid_mask.")
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError(
            "P1 residual pred/target/valid shapes must match: "
            f"pred={tuple(pred.shape)}, target={tuple(target.shape)}, "
            f"valid={tuple(valid.shape)}"
        )
    if beta_m <= 0:
        raise ValueError("P1 Smooth-L1 beta must be positive.")

    valid = valid.to(device=pred.device, dtype=torch.bool)
    loss_map = F.smooth_l1_loss(
        pred,
        target.to(pred),
        reduction="none",
        beta=float(beta_m),
    )
    if bool(valid.any()):
        loss = loss_map[valid].mean()
    else:
        loss = pred.sum() * 0.0

    end_points["B: P1 Center Recenter Loss"] = loss
    with torch.no_grad():
        zero = loss.detach() * 0.0
        end_points["D: P1 center loss valid ratio"] = valid.float().mean()
        if bool(valid.any()):
            error = pred.detach() - target.to(pred).detach()
            end_points["D: P1 center residual MAE"] = error.abs()[valid].mean()
            end_points["D: P1 center residual RMSE"] = torch.sqrt(
                error.square()[valid].mean()
            )
        else:
            end_points["D: P1 center residual MAE"] = zero
            end_points["D: P1 center residual RMSE"] = zero
    return loss, end_points


def p1_train_mode_flags(mode: str) -> Dict[str, bool]:
    """Canonical one-factor training modes used by the P1 trainer."""
    mode = str(mode).strip().lower()
    if mode == "center_only":
        return {
            "p1_freeze_base": True,
            "p1_detach_head_features": True,
            "p1_detach_corrected_for_downstream": True,
        }
    if mode == "head_grasp":
        return {
            "p1_freeze_base": True,
            "p1_detach_head_features": True,
            "p1_detach_corrected_for_downstream": False,
        }
    if mode == "joint":
        return {
            "p1_freeze_base": False,
            "p1_detach_head_features": False,
            "p1_detach_corrected_for_downstream": False,
        }
    raise ValueError(
        "p1_train_mode must be one of center_only/head_grasp/joint; "
        f"got {mode!r}."
    )
