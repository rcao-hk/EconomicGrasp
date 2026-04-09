import os
import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arguments import cfgs

try:
    import open3d as o3d
except Exception:
    o3d = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Assumptions / dependencies from your existing c2.3.py namespace:
#   - cfgs
#   - furthest_point_sample, gather_operation
#   - gather_depth_by_img_idxs
#   - DINOv2DepthDistributionNet
#   - TokGraspableHead2D
#   - ViewNet
#   - Cylinder_Grouping_Global_Interaction
#   - Grasp_Head_Local_Interaction
#   - process_grasp_labels
# -----------------------------------------------------------------------------
from .economicgrasp_depth import DINOv2DepthDistributionNet
from models.bip3d.models.modules.resnet import ResNet
from models.bip3d.models.modules.channel_mapper import ChannelMapper
from .economicgrasp_depth_c1 import TokGraspableHead2D
from models.modules_economicgrasp import ViewNet, Cylinder_Grouping_Global_Interaction, Grasp_Head_Local_Interaction
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from utils.label_generation import process_grasp_labels, batch_viewpoint_params_to_matrix

def gather_depth_by_img_idxs(depth_map_1hw: torch.Tensor, img_idxs: torch.Tensor):
    # depth_map_1hw: (B,1,H,W) or (B,H,W)
    if depth_map_1hw.dim() == 3:
        depth_flat = depth_map_1hw.reshape(depth_map_1hw.size(0), -1)
    else:
        depth_flat = depth_map_1hw[:, 0].reshape(depth_map_1hw.size(0), -1)
    return depth_flat.gather(1, img_idxs)


# ========================= single-scale BIP3D-style enhancer =========================
class FFN(nn.Module):
    def __init__(self, embed_dims: int, feedforward_channels: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, feedforward_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SingleScaleDepthFusionSpatialEnhancer(nn.Module):
    """
    Single-scale BIP3D-style spatial enhancer.

    Difference from official BIP3D:
      - depth_prob is supplied externally (from DINOv2DepthDistributionNet)
    Everything else follows the same idea:
      - generate frustum 3D points
      - pts_fc to embed them
      - weighted sum by depth_prob
      - fuse [feature_2d, optional feature_3d, pts_feature]
      - residual + LayerNorm
    """

    def __init__(
        self,
        embed_dims: int,
        feature_3d_dim: int = 32,
        ff_dim: int = 1024,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        num_depth: int = 256,
        with_feature_3d: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.feature_3d_dim = int(feature_3d_dim)
        self.ff_dim = int(ff_dim)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.num_depth = int(num_depth)
        self.with_feature_3d = bool(with_feature_3d)
        self.eps = float(eps)

        self.pts_fc = nn.Linear(3, self.feature_3d_dim)

        fusion_dim = self.embed_dims + self.feature_3d_dim
        if self.with_feature_3d:
            fusion_dim += self.feature_3d_dim
        self.fusion_fc = nn.Sequential(
            FFN(fusion_dim, feedforward_channels=self.ff_dim),
            nn.Linear(fusion_dim, self.embed_dims),
        )
        self.fusion_norm = nn.LayerNorm(self.embed_dims)

        bins = torch.linspace(self.min_depth, self.max_depth, self.num_depth)
        self.register_buffer("depth_bins", bins, persistent=False)

    @staticmethod
    def _token_centers(H: int, W: int, stride: int, device, dtype):
        j = torch.arange(W, device=device, dtype=dtype)
        i = torch.arange(H, device=device, dtype=dtype)
        u = (j + 0.5) * stride - 0.5
        v = (i + 0.5) * stride - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        return uu.reshape(-1), vv.reshape(-1)

    def get_pts(self, H: int, W: int, K: torch.Tensor, stride: int, device, dtype) -> torch.Tensor:
        B = K.shape[0]
        u, v = self._token_centers(H, W, stride, device, dtype)
        N = u.numel()
        u = u.view(1, N, 1).expand(B, -1, -1)
        v = v.view(1, N, 1).expand(B, -1, -1)

        z = self.depth_bins.to(device=device, dtype=dtype).view(1, 1, self.num_depth)
        z = z.expand(B, N, -1)

        fx = K[:, 0, 0].view(B, 1, 1).to(dtype)
        fy = K[:, 1, 1].view(B, 1, 1).to(dtype)
        cx = K[:, 0, 2].view(B, 1, 1).to(dtype)
        cy = K[:, 1, 2].view(B, 1, 1).to(dtype)

        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pts = torch.stack([x, y, z], dim=-1)  # (B,N,D,3)
        return pts

    def forward(
        self,
        feat_2d: torch.Tensor,
        depth_prob: torch.Tensor,
        K: torch.Tensor,
        feature_3d: Optional[torch.Tensor] = None,
        stride: int = 1,
    ) -> torch.Tensor:
        B, C, H, W = feat_2d.shape
        assert C == self.embed_dims, f"embed_dims mismatch: got {C}, expect {self.embed_dims}"

        feature_2d_flat = feat_2d.flatten(start_dim=-2).transpose(-1, -2)  # (B,N,C)

        if depth_prob.dim() == 4:
            if depth_prob.shape[-2:] != (H, W):
                depth_prob = F.interpolate(depth_prob, size=(H, W), mode="bilinear", align_corners=False)
            depth_prob = depth_prob.flatten(start_dim=-2).transpose(-1, -2)  # (B,N,D)
        elif depth_prob.dim() == 3:
            assert depth_prob.shape[1] == H * W
        else:
            raise ValueError(f"Unsupported depth_prob shape: {tuple(depth_prob.shape)}")

        D = depth_prob.shape[-1]
        if D != self.num_depth:
            raise ValueError(f"depth_prob last dim {D} != configured num_depth {self.num_depth}")

        depth_prob = torch.nan_to_num(depth_prob, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(self.eps)
        depth_prob = depth_prob / depth_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        pts = self.get_pts(H, W, K, stride=stride, device=feat_2d.device, dtype=feat_2d.dtype)  # (B,N,D,3)
        pts_feature = self.pts_fc(pts)  # (B,N,D,C3)
        pts_feature = (depth_prob.unsqueeze(-1) * pts_feature).sum(dim=-2)  # (B,N,C3)

        fused_inputs = [feature_2d_flat]
        if self.with_feature_3d:
            if feature_3d is None:
                raise ValueError("feature_3d is required when with_feature_3d=True")
            if feature_3d.shape[-2:] != (H, W):
                feature_3d = F.interpolate(feature_3d, size=(H, W), mode="bilinear", align_corners=False)
            feature_3d_flat = feature_3d.flatten(start_dim=-2).transpose(-1, -2)
            if feature_3d_flat.shape[-1] != self.feature_3d_dim:
                raise ValueError(
                    f"feature_3d channel {feature_3d_flat.shape[-1]} != feature_3d_dim {self.feature_3d_dim}"
                )
            fused_inputs.append(feature_3d_flat)

        fused_inputs.append(pts_feature)
        fused = torch.cat(fused_inputs, dim=-1)
        out = self.fusion_fc(fused) + feature_2d_flat
        out = self.fusion_norm(out)
        out = out.transpose(-1, -2).reshape(B, C, H, W).contiguous()
        return out


# ========================= complete economicgrasp_bip3d =========================
class economicgrasp_bip3d(nn.Module):
    """
    economicgrasp_bip3d

    Frontend:
      RGB -> DINOv2DepthDistributionNet (unchanged)
      optional depth -> ResNet34 + ChannelMapper
      single-scale BIP3D-style spatial enhancer using external depth_prob

    Backend:
      reuse the economicgrasp_c2_3 token graspable head / Top-M selection /
      ViewNet / Cylinder grouping / Grasp head pipeline.

    Notes:
      - This class is intended to be pasted into your c2.3.py, below the shared
        helper definitions, so it can directly reuse project functions/classes.
      - `depth_prob_pred` keeps the existing shape expected by your loss code:
        (B,1,N,D)
      - Explicit modality flag: input_modality in {"rgb", "rgbd"}
    """

    def __init__(
        self,
        cylinder_radius: float = 0.05,
        seed_feat_dim: int = 512,
        is_training: bool = True,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        tok_feat_dim: int = 128,
        feature_3d_dim: int = 32,
        input_modality: str = "rgb",
        depth_feature_level: int = 0,
        depth_branch_in_channels: int = 1,
        use_gt_xyz_for_train: bool = False,
        use_input_depth_for_xyz: bool = False,
        detach_prob_for_enhancer: bool = False,
        detach_depth_feat_for_enhancer: bool = False,
        topk_use_objectness: bool = True,
        vis_dir: str = 'bip3d',
        vis_every: int = 1000,
        debug_print_every: int = 50,
        vis_token_every: Optional[int] = None,
        vis_token_maxB: int = 1,
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.input_modality = str(input_modality).lower()
        if self.input_modality not in ["rgb", "rgbd"]:
            raise ValueError(f"input_modality must be 'rgb' or 'rgbd', got {input_modality}")
        self.use_depth_branch = (self.input_modality == "rgbd")
        self.depth_feature_level = int(depth_feature_level)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.use_input_depth_for_xyz = bool(use_input_depth_for_xyz)
        self.detach_prob_for_enhancer = bool(detach_prob_for_enhancer)
        self.detach_depth_feat_for_enhancer = bool(detach_depth_feat_for_enhancer)
        self.topk_use_objectness = bool(topk_use_objectness)

        # sel_proj removed: downstream seed feature dim must match token feature dim
        self.seed_feature_dim = int(tok_feat_dim)
        self.num_depth = int(getattr(cfgs, 'num_depth', 4))
        self.num_angle = int(getattr(cfgs, 'num_angle', 12))
        self.M_points = int(getattr(cfgs, 'm_point', 1024))
        self.num_view = int(getattr(cfgs, 'num_view', 300))
        self.bin_num = int(bin_num)

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        # -----------------------------
        # RGB image encoder + depth distribution head (unchanged)
        # -----------------------------
        self.depth_net = DINOv2DepthDistributionNet(
            encoder="vitb",
            stride=1,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            bin_num=self.bin_num,
            freeze_backbone=True,
        )

        # -----------------------------
        # optional depth branch: ResNet34 + ChannelMapper
        # -----------------------------
        if self.use_depth_branch:
            self.backbone_3d = ResNet(
                depth=34,
                in_channels=depth_branch_in_channels,
                base_channels=4,
                num_stages=4,
                out_indices=(1, 2, 3),
                bn_eval=True,
                with_cp=True,
                style="pytorch",
            )
            self.neck_3d = ChannelMapper(
                in_channels=[8, 16, 32],
                kernel_size=1,
                out_channels=feature_3d_dim,
                act_cfg=None,
                bias=True,
                norm_cfg=dict(type=nn.GroupNorm, num_groups=4),
                num_outs=4,
            )
        else:
            self.backbone_3d = None
            self.neck_3d = None

        # -----------------------------
        # single-scale BIP3D-style enhancer
        # -----------------------------
        self.enhancer = SingleScaleDepthFusionSpatialEnhancer(
            embed_dims=tok_feat_dim,
            feature_3d_dim=feature_3d_dim,
            ff_dim=1024,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            num_depth=self.bin_num,
            with_feature_3d=self.use_depth_branch,
        )

        # lightweight 2D graspable head
        self.graspable_2d = TokGraspableHead2D(in_dim=tok_feat_dim)

        # reuse original heads
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

        # vis / debug
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.vis_token_every = int(vis_every if vis_token_every is None else vis_token_every)
        self.vis_token_maxB = int(vis_token_maxB)
        self.debug_print_every = int(debug_print_every)
        self.debug_first_only = False
        self._debug_has_done = False
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

    # ========================= helpers =========================
    @staticmethod
    def _backproject_uvz(uv_b_n2, z_b_n1, K_b_33):
        fx = K_b_33[:, 0, 0].unsqueeze(1)
        fy = K_b_33[:, 1, 1].unsqueeze(1)
        cx = K_b_33[:, 0, 2].unsqueeze(1)
        cy = K_b_33[:, 1, 2].unsqueeze(1)
        u = uv_b_n2[..., 0]
        v = uv_b_n2[..., 1]
        z = z_b_n1.squeeze(-1)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _save_map_png(arr2d, out_path, vmin=None, vmax=None, cmap="Spectral", title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6, 6))
        if vmin is None:
            vmin = float(np.nanmin(arr2d))
        if vmax is None:
            vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    @staticmethod
    def _save_overlay_points(img_448, pts_uv, out_path, radius=1, color=(0, 0, 255)):
        import cv2
        x = img_448.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        x_bgr = x[..., ::-1].copy()

        pts = pts_uv.detach().cpu().numpy()
        H, W = x_bgr.shape[:2]
        for (u, v) in pts:
            uu = int(round(float(u)))
            vv = int(round(float(v)))
            if 0 <= uu < W and 0 <= vv < H:
                cv2.circle(x_bgr, (uu, vv), radius, color, thickness=-1)
        cv2.imwrite(out_path, x_bgr)

    @torch.no_grad()
    def _save_pred_gt_cloud_ply(self, cloud_pred: torch.Tensor, cloud_gt: torch.Tensor, end_points: dict):
        if o3d is None or self.vis_dir is None:
            return
        p = cloud_pred[0].detach().float().cpu().numpy()
        g = cloud_gt[0].detach().float().cpu().numpy()

        def _valid(x):
            m = np.isfinite(x).all(axis=1)
            m &= (x[:, 2] > 0)
            return x[m]

        p = _valid(p)
        g = _valid(g)
        if p.shape[0] == 0 or g.shape[0] == 0:
            return

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32)
        p_col[:, 0] = 1.0
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32)
        g_col[:, 2] = 1.0
        pts = np.concatenate([p, g], axis=0)
        cols = np.concatenate([p_col, g_col], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        tag = end_points.get("vis_tag", None)
        if tag is None:
            scene = end_points.get("scene", "scene")
            frame = end_points.get("frame", "frame")
            tag = f"{scene}_{frame}"

        out_path = os.path.join(self.vis_dir, f"tok_pred_gt_xyz_{tag}_iter{self._vis_iter:06d}.ply")
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

    @staticmethod
    def _get_depth_mu_entropy(depth_prob_maps: torch.Tensor, depth_bins: torch.Tensor, eps: float = 1e-6):
        prob = torch.nan_to_num(depth_prob_maps, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
        prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(eps)
        z = depth_bins.view(1, -1, 1, 1).to(prob)
        mu = (prob * z).sum(dim=1)
        ent = -(prob * prob.log()).sum(dim=1)
        return mu, ent

    @staticmethod
    def _pick_depth_feature_level(feature_3d_levels, level: int, target_hw: Tuple[int, int]):
        if feature_3d_levels is None:
            return None
        assert isinstance(feature_3d_levels, (list, tuple)) and len(feature_3d_levels) > 0
        level = max(0, min(level, len(feature_3d_levels) - 1))
        feat = feature_3d_levels[level]
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return feat

    @staticmethod
    def _get_depth_input(end_points: dict) -> Optional[torch.Tensor]:
        cand_keys = [
            "depth", "depths", "depth_input", "depth_map", "depth_img", "depths_1hw",
            "depth_map_input", "sensor_depth", "gt_depth_m",
        ]
        for k in cand_keys:
            if k in end_points and torch.is_tensor(end_points[k]):
                d = end_points[k]
                if d.dim() == 3:
                    d = d.unsqueeze(1)
                elif d.dim() == 4:
                    d = d[:, :1]
                return d
        return None

    def _run_depth_branch(self, depth_input: Optional[torch.Tensor], target_hw: Tuple[int, int]):
        if (not self.use_depth_branch) or (depth_input is None):
            return None, None
        depth_input = torch.nan_to_num(depth_input, nan=0.0, posinf=0.0, neginf=0.0)
        feat_3d = self.backbone_3d(depth_input)
        if not isinstance(feat_3d, (list, tuple)):
            feat_3d = (feat_3d,)
        feat_3d = self.neck_3d(feat_3d)
        feat_3d_single = self._pick_depth_feature_level(feat_3d, self.depth_feature_level, target_hw)
        return feat_3d, feat_3d_single

    # ========================= forward =========================
    def forward(self, end_points: dict):
        img = end_points["img"]
        K = end_points["K"]
        B, _, H, W = img.shape
        assert (H, W) == (448, 448), f"economicgrasp_bip3d expects 448x448, got {(H, W)}"
        Ntok = H * W
        M = int(self.M_points)

        # ------------------------------------------------------------------
        # 1) RGB image encoder + external depth distribution
        # ------------------------------------------------------------------
        depth_map_pred_448, depth_tok_dbg, img_feat, depth_prob_448, depth_logits_448, prob_tok_flat = \
            self.depth_net(img, return_prob=True, return_tok_prob=True)

        depth_map_pred = torch.nan_to_num(depth_map_pred_448, nan=0.0, posinf=0.0, neginf=0.0)
        depth_map_pred = depth_map_pred.clamp(min=self.min_depth, max=self.max_depth)

        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

        end_points["depth_map_pred"] = depth_map_pred
        end_points["depth_tok_dbg"] = depth_tok_dbg
        end_points["depth_tok_pred"] = depth_map_pred
        end_points["depth_prob_maps"] = depth_prob_448
        end_points["depth_logits_448"] = depth_logits_448
        end_points["depth_prob_pred"] = prob_tok_flat  # (B,1,N,D), expected by existing loss
        end_points["img_feat_dpt"] = img_feat

        # ------------------------------------------------------------------
        # 2) optional depth branch (RGB-D)
        # ------------------------------------------------------------------
        depth_input = self._get_depth_input(end_points)
        feat_3d_levels, feat_3d_single = self._run_depth_branch(depth_input, target_hw=(H, W))
        if self.use_depth_branch:
            if depth_input is None:
                raise ValueError(
                    "economicgrasp_bip3d is in RGB-D mode (input_modality='rgbd'), but no depth input was found in end_points."
                )
            end_points["has_depth_branch"] = torch.tensor(1.0, device=img.device)
            end_points["depth_input_used"] = depth_input
            end_points["depth_feat_3d_levels"] = feat_3d_levels
            end_points["depth_feat_3d_single"] = feat_3d_single
        else:
            end_points["has_depth_branch"] = torch.tensor(0.0, device=img.device)

        # ------------------------------------------------------------------
        # 3) single-scale BIP3D-style enhancer
        # ------------------------------------------------------------------
        depth_prob_for_enh = depth_prob_448.detach() if self.detach_prob_for_enhancer else depth_prob_448
        if feat_3d_single is not None and self.detach_depth_feat_for_enhancer:
            feat_3d_single = feat_3d_single.detach()

        feat_grid_enh = self.enhancer(
            feat_2d=img_feat,
            depth_prob=depth_prob_for_enh,
            K=K,
            feature_3d=feat_3d_single,
            stride=1,
        )
        end_points["img_feat_enh"] = feat_grid_enh

        # ------------------------------------------------------------------
        # 4) full-res token graspable head
        # ------------------------------------------------------------------
        end_points = self.graspable_2d(feat_grid_enh, end_points)
        objectness_score = end_points["objectness_score"]            # (B,2,Ntok)
        graspness_score = end_points["graspness_score"].squeeze(1)   # (B,Ntok)
        objectness_pred = torch.argmax(objectness_score, dim=1)       # (B,Ntok)

        if "token_valid_mask" in end_points:
            valid_tok = end_points["token_valid_mask"].bool()
            if valid_tok.shape[1] != Ntok:
                raise ValueError(
                    f"economicgrasp_bip3d expects token_valid_mask with Ntok={Ntok}, got {tuple(valid_tok.shape)}"
                )
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        grasp_raw = graspness_score
        grasp_sel = grasp_raw.clamp(0.0, 1.0)

        mask_obj_pred = valid_tok & (objectness_pred == 1)
        mask_thr_pred = mask_obj_pred & (grasp_sel > float(cfgs.graspness_threshold))

        end_points["dbg_grasp_raw"] = grasp_raw.detach()
        end_points["dbg_grasp_sel"] = grasp_sel.detach()
        end_points["dbg_mask_obj"] = mask_obj_pred.detach()
        end_points["dbg_mask_pred"] = mask_thr_pred.detach()
        end_points["dbg_objectness_pred"] = objectness_pred.detach()

        with torch.no_grad():
            end_points["D: PredCand#(thr)"] = mask_thr_pred.float().sum(dim=1).mean().reshape(())
            end_points["D: PredObj#"] = mask_obj_pred.float().sum(dim=1).mean().reshape(())
            end_points["D: GraspRaw min"] = grasp_raw.min().reshape(())
            end_points["D: GraspRaw max"] = grasp_raw.max().reshape(())
            end_points["D: GraspSel mean"] = grasp_sel.mean().reshape(())
            end_points["D: DepthProb Sum"] = prob_tok_flat.sum(dim=-1).mean().reshape(())
            end_points["D: DepthProb Valid Ratio"] = torch.ones((), device=img.device, dtype=torch.float32)

        # ------------------------------------------------------------------
        # 5) xyz for all pixels
        # ------------------------------------------------------------------
        flat_all = torch.arange(H * W, device=img.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        u_all = (flat_all % W).float()
        v_all = (flat_all // W).float()
        uv_all = torch.stack([u_all, v_all], dim=-1)

        # which z to use for geometry / sampling
        z_map_for_xyz = depth_map_pred
        if self.use_input_depth_for_xyz and depth_input is not None:
            z_map_for_xyz = torch.nan_to_num(depth_input[:, :1], nan=0.0, posinf=0.0, neginf=0.0)
            z_map_for_xyz = z_map_for_xyz.clamp(min=1e-6)
            end_points["depth_map_used_for_xyz"] = z_map_for_xyz
        else:
            end_points["depth_map_used_for_xyz"] = depth_map_pred

        z_all_pred = z_map_for_xyz.view(B, -1, 1).contiguous()
        z_all_pred = torch.nan_to_num(z_all_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
        xyz_all_pred = self._backproject_uvz(uv_all, z_all_pred.detach(), K)

        # GT xyz for label matching if enabled
        use_gt_xyz = self.is_training and self.use_gt_xyz_for_train and ("gt_depth_m" in end_points)
        if use_gt_xyz:
            gt_depth = end_points["gt_depth_m"]
            if gt_depth.dim() == 3:
                gt_depth = gt_depth.unsqueeze(1)
            elif gt_depth.dim() == 4:
                gt_depth = gt_depth[:, :1]
            z_all_gt = gt_depth.view(B, -1, 1).contiguous()
            z_all_gt = torch.nan_to_num(z_all_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            xyz_all_match = self._backproject_uvz(uv_all, z_all_gt, K)
        else:
            xyz_all_match = xyz_all_pred

        # ------------------------------------------------------------------
        # 6) graspable sampling: FPS on graspable-mask (no sel_proj)
        # ------------------------------------------------------------------
        seed_features_flipped = feat_grid_enh.view(B, feat_grid_enh.shape[1], -1).contiguous()  # (B,C,Ntok)
        seed_xyz = xyz_all_match.contiguous()  # (B,Ntok,3)
        graspable_mask = mask_thr_pred

        seed_features_graspable = []
        seed_xyz_graspable = []
        sel_idx_list = []
        graspable_num_batch = torch.zeros((), device=img.device, dtype=torch.float32)

        for i in range(B):
            cur_mask = graspable_mask[i]
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
            graspable_num_batch = graspable_num_batch + cur_mask.float().sum()

            # robust fallback when predicted graspable points are insufficient
            if cur_idx.numel() == 0:
                cur_idx = torch.randint(0, Ntok, (self.M_points,), device=img.device, dtype=torch.long)
            elif cur_idx.numel() < self.M_points:
                rep = torch.randint(0, cur_idx.numel(), (self.M_points - cur_idx.numel(),), device=img.device)
                cur_idx = torch.cat([cur_idx, cur_idx[rep]], dim=0)

            cur_feat = seed_features_flipped[i][:, cur_idx].transpose(0, 1).contiguous()  # (Ng,C)
            cur_seed_xyz = seed_xyz[i][cur_idx]  # (Ng,3)

            cur_seed_xyz_b = cur_seed_xyz.unsqueeze(0).contiguous()
            fps_idxs = furthest_point_sample(cur_seed_xyz_b, self.M_points)

            cur_seed_xyz_flipped = cur_seed_xyz_b.transpose(1, 2).contiguous()
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()

            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()  # (C,M)

            cur_sel_idx = cur_idx.index_select(0, fps_idxs.squeeze(0).long()).contiguous()

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            sel_idx_list.append(cur_sel_idx)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0).contiguous()
        seed_features_graspable = torch.stack(seed_features_graspable, 0).contiguous()  # (B,C,M)

        end_points["token_sel_idx"] = torch.stack(sel_idx_list, dim=0).contiguous()
        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["token_sel_xyz"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = (graspable_num_batch / float(B)).detach().reshape(())

        # ------------------------------------------------------------------
        # 7) debug prints
        # ------------------------------------------------------------------
        do_print = (self._vis_iter % self.debug_print_every == 0) or bool(end_points.get("force_vis", False))
        if do_print:
            with torch.no_grad():
                depth_mu_map, depth_ent_map = self._get_depth_mu_entropy(
                    depth_prob_448, self.enhancer.depth_bins, eps=self.enhancer.eps
                )
                msg = (
                    f"[bip3d] iter={self._vis_iter} "
                    f"valid={valid_tok.float().sum(1).mean().item():.1f} "
                    f"obj_pred1={mask_obj_pred.float().sum(1).mean().item():.1f} "
                    f"cand_thr={mask_thr_pred.float().sum(1).mean().item():.1f} "
                    f"gr_raw[min,max]=({grasp_raw.min().item():.3f},{grasp_raw.max().item():.3f}) "
                    f"gr_sel[p10,p50,p90]=({torch.quantile(grasp_sel, 0.1).item():.3f},"
                    f"{torch.quantile(grasp_sel, 0.5).item():.3f},"
                    f"{torch.quantile(grasp_sel, 0.9).item():.3f}) "
                    f"zmu[min,p50,max]=({depth_mu_map.min().item():.3f},"
                    f"{torch.quantile(depth_mu_map.flatten(),0.5).item():.3f},"
                    f"{depth_mu_map.max().item():.3f}) "
                    f"dent[p50,p90]=({torch.quantile(depth_ent_map.flatten(),0.5).item():.3f},"
                    f"{torch.quantile(depth_ent_map.flatten(),0.9).item():.3f}) "
                    f"modality={self.input_modality} | depth_branch={int(self.use_depth_branch and feat_3d_single is not None)}"
                )

                if self.use_depth_branch and feat_3d_single is not None:
                    msg += f" | feat3d_norm={feat_3d_single.norm(dim=1).mean().item():.3f}"

                if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                    gt_obj = end_points["objectness_label_tok"].long()
                    gt_gra = end_points["graspness_label_tok"].float()
                    if gt_obj.shape[1] == Ntok and gt_gra.shape[1] == Ntok:
                        gt_valid = (gt_obj != -1) & valid_tok
                        gt_obj1 = (gt_obj == 1) & gt_valid
                        if gt_obj1.any():
                            g = gt_gra[gt_obj1]
                            msg += (
                                f" | GT_gra[obj] p50={torch.quantile(g,0.5).item():.3f}"
                                f" p90={torch.quantile(g,0.9).item():.3f} mean={g.mean().item():.3f}"
                            )
                        sel = end_points["token_sel_idx"]
                        gt_pos = gt_obj1 & (gt_gra > 0.2)
                        cover, sel_gtg = [], []
                        for bb in range(B):
                            cover.append(gt_pos[bb].gather(0, sel[bb]).float().mean())
                            sel_gtg.append(gt_gra[bb].gather(0, sel[bb]).mean())
                        msg += f" | sel_gtpos_ratio={torch.stack(cover).mean().item():.3f}"
                        msg += f" | sel_GTg_mean={torch.stack(sel_gtg).mean().item():.3f}"
                print(msg)

        # ------------------------------------------------------------------
        # 8) token/depth visualization
        # ------------------------------------------------------------------
        do_tok_vis = (self.vis_dir is not None) and (
            (self._vis_iter % self.vis_token_every) == 0 or bool(end_points.get("force_vis", False))
        )
        if do_tok_vis:
            tag = end_points.get("vis_tag", f"iter{self._vis_iter:06d}")
            out_dir = os.path.join(self.vis_dir, f"tokdbg_{tag}")
            os.makedirs(out_dir, exist_ok=True)

            depth_mu_map, depth_ent_map = self._get_depth_mu_entropy(
                depth_prob_448, self.enhancer.depth_bins, eps=self.enhancer.eps
            )

            for b in range(min(B, self.vis_token_maxB)):
                pred_obj_map = end_points["dbg_objectness_pred"][b].view(H, W).float()
                pred_gra_map = end_points["dbg_grasp_sel"][b].view(H, W)
                cand_map = end_points["dbg_mask_pred"][b].float().view(H, W)

                self._save_map_png(pred_obj_map, os.path.join(out_dir, f"b{b}_pred_objectness.png"),
                                   vmin=0, vmax=1, cmap="viridis", title="Pred obj")
                self._save_map_png(pred_gra_map, os.path.join(out_dir, f"b{b}_pred_graspness.png"),
                                   vmin=0, vmax=1, cmap="Spectral", title="Pred graspness")
                self._save_map_png(cand_map, os.path.join(out_dir, f"b{b}_pred_candidate_mask.png"),
                                   vmin=0, vmax=1, cmap="gray", title="Pred cand mask")
                self._save_map_png(depth_mu_map[b], os.path.join(out_dir, f"b{b}_depth_mu.png"),
                                   vmin=self.min_depth, vmax=self.max_depth, cmap="viridis", title="Depth mean")
                self._save_map_png(depth_ent_map[b], os.path.join(out_dir, f"b{b}_depth_entropy.png"),
                                   cmap="magma", title="Depth entropy")
                self._save_map_png(depth_map_pred[b, 0], os.path.join(out_dir, f"b{b}_depth_pred_map.png"),
                                   vmin=self.min_depth, vmax=self.max_depth, cmap="viridis", title="Depth pred")

                if self.use_depth_branch and feat_3d_single is not None:
                    feat3d_norm = feat_3d_single[b].norm(dim=0)
                    self._save_map_png(feat3d_norm, os.path.join(out_dir, f"b{b}_depth_feat_norm.png"),
                                       cmap="magma", title="Depth feature norm")

                if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                    gt_obj = end_points["objectness_label_tok"]
                    gt_gra = end_points["graspness_label_tok"]
                    if gt_obj.shape[1] == Ntok and gt_gra.shape[1] == Ntok:
                        self._save_map_png(gt_obj[b].view(H, W).float(), os.path.join(out_dir, f"b{b}_gt_objectness.png"),
                                           vmin=-1, vmax=1, cmap="viridis", title="GT obj (-1 invalid)")
                        self._save_map_png(gt_gra[b].view(H, W).float(), os.path.join(out_dir, f"b{b}_gt_graspness.png"),
                                           vmin=0, vmax=1, cmap="Spectral", title="GT graspness")

                sel = end_points["token_sel_idx"][b]
                sel_mask = torch.zeros((Ntok,), device=img.device)
                sel_mask[sel] = 1.0
                self._save_map_png(sel_mask.view(H, W), os.path.join(out_dir, f"b{b}_selected_tokens.png"),
                                   vmin=0, vmax=1, cmap="gray", title="Selected Top-M")
                xs = (sel % W).float()
                ys = (sel // W).float()
                pts_uv = torch.stack([xs, ys], dim=-1)
                self._save_overlay_points(img[b], pts_uv, os.path.join(out_dir, f"b{b}_overlay_selected.png"),
                                          radius=1, color=(0, 0, 255))

        # ------------------------------------------------------------------
        # 9) point cloud vis on sampled img_idxs
        # ------------------------------------------------------------------
        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0) or bool(end_points.get("force_vis", False))
            if do_vis and ("gt_depth_m" in end_points) and ("img_idxs" in end_points):
                gt_depth = end_points["gt_depth_m"]
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]

                img_idxs_vis = end_points["img_idxs"].long().clamp(0, H * W - 1)
                z_pred = gather_depth_by_img_idxs(end_points["depth_map_used_for_xyz"], img_idxs_vis)
                z_gt = gather_depth_by_img_idxs(gt_depth, img_idxs_vis)

                z_pred = torch.nan_to_num(z_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
                z_gt = torch.nan_to_num(z_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

                u_vis = (img_idxs_vis % W).float()
                v_vis = (img_idxs_vis // W).float()
                uv_vis = torch.stack([u_vis, v_vis], dim=-1)

                xyz_pred = self._backproject_uvz(uv_vis, z_pred, K)
                xyz_gt = self._backproject_uvz(uv_vis, z_gt, K)
                self._save_pred_gt_cloud_ply(xyz_pred, xyz_gt, end_points)
                print("[vis] point cloud: red=used depth for xyz, blue=GT depth")

        # ------------------------------------------------------------------
        # 10) view + labels + grouping + head
        # ------------------------------------------------------------------
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        group_features = self.cy_group(seed_xyz_graspable, seed_features_graspable, grasp_top_views_rot)
        end_points = self.grasp_head(group_features, end_points)

        self._vis_iter += 1
        return end_points
