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
from models.modules_economicgrasp import AttentionModule

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
    Memory-efficient single-scale BIP3D-style spatial enhancer.

    Compared with the official BIP3D implementation:
      - depth_prob is supplied externally
      - we avoid explicitly materializing (B, N, D, 3) frustum points
      - because pts_fc is linear in official BIP3D, using E[pts] first and then
        applying pts_fc is mathematically equivalent to applying pts_fc to all pts
        and then taking the expectation.

    Official BIP3D behavior retained:
      - build 3D-aware feature from depth distribution
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

        # same role as official BIP3D: embed 3D points
        self.pts_fc = nn.Linear(3, self.feature_3d_dim)

        fusion_dim = self.embed_dims + self.feature_3d_dim
        if self.with_feature_3d:
            fusion_dim += self.feature_3d_dim

        self.fusion_fc = nn.Sequential(
            FFN(embed_dims=fusion_dim, feedforward_channels=self.ff_dim),
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
        return uu.reshape(-1), vv.reshape(-1)  # (N,), (N,)

    def forward(
        self,
        feat_2d: torch.Tensor,
        depth_prob: torch.Tensor,
        K: torch.Tensor,
        feature_3d: Optional[torch.Tensor] = None,
        stride: int = 1,
    ) -> torch.Tensor:
        """
        Args:
          feat_2d:
            (B, C, H, W)
          depth_prob:
            either (B, D, H, W) or (B, N, D)
          K:
            (B, 3, 3)
          feature_3d:
            optional extra feature map, (B, feature_3d_dim, H, W)
          stride:
            pixel stride of feat_2d wrt 448 image
        Returns:
          out:
            (B, C, H, W)
        """
        B, C, H, W = feat_2d.shape
        assert C == self.embed_dims, f"embed_dims mismatch: got {C}, expect {self.embed_dims}"

        # (B,N,C)
        feature_2d_flat = feat_2d.flatten(start_dim=-2).transpose(-1, -2)

        # depth_prob -> (B,N,D)
        if depth_prob.dim() == 4:
            if depth_prob.shape[-2:] != (H, W):
                depth_prob = F.interpolate(
                    depth_prob, size=(H, W), mode="bilinear", align_corners=False
                )
            depth_prob = depth_prob.flatten(start_dim=-2).transpose(-1, -2)
        elif depth_prob.dim() == 3:
            assert depth_prob.shape[1] == H * W, \
                f"depth_prob second dim {depth_prob.shape[1]} != H*W {H*W}"
        else:
            raise ValueError(f"Unsupported depth_prob shape: {tuple(depth_prob.shape)}")

        D = depth_prob.shape[-1]
        if D != self.num_depth:
            raise ValueError(f"depth_prob last dim {D} != configured num_depth {self.num_depth}")

        depth_prob = torch.nan_to_num(depth_prob, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(self.eps)
        depth_prob = depth_prob / depth_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # ------------------------------------------------------------------
        #   do NOT build (B,N,D,3) pts explicitly
        #   compute first moment z_mean, then build E[(x,y,z)] directly
        # ------------------------------------------------------------------
        N = H * W
        u, v = self._token_centers(H, W, stride, feat_2d.device, feat_2d.dtype)
        u = u.view(1, N, 1).expand(B, -1, -1)  # (B,N,1)
        v = v.view(1, N, 1).expand(B, -1, -1)  # (B,N,1)

        bins = self.depth_bins.to(device=feat_2d.device, dtype=feat_2d.dtype).view(1, 1, self.num_depth)
        z_mean = (depth_prob * bins).sum(dim=-1, keepdim=True)  # (B,N,1)

        fx = K[:, 0, 0].view(B, 1, 1).to(feat_2d.dtype)
        fy = K[:, 1, 1].view(B, 1, 1).to(feat_2d.dtype)
        cx = K[:, 0, 2].view(B, 1, 1).to(feat_2d.dtype)
        cy = K[:, 1, 2].view(B, 1, 1).to(feat_2d.dtype)

        x_mean = (u - cx) / fx * z_mean
        y_mean = (v - cy) / fy * z_mean
        pts_mean = torch.cat([x_mean, y_mean, z_mean], dim=-1)   # (B,N,3)

        pts_feature = self.pts_fc(pts_mean)                      # (B,N,C3)

        fused_inputs = [feature_2d_flat]

        if self.with_feature_3d:
            if feature_3d is None:
                raise ValueError("feature_3d is required when with_feature_3d=True")
            if feature_3d.shape[-2:] != (H, W):
                feature_3d = F.interpolate(
                    feature_3d, size=(H, W), mode="bilinear", align_corners=False
                )
            feature_3d_flat = feature_3d.flatten(start_dim=-2).transpose(-1, -2)
            if feature_3d_flat.shape[-1] != self.feature_3d_dim:
                raise ValueError(
                    f"feature_3d channel {feature_3d_flat.shape[-1]} "
                    f"!= feature_3d_dim {self.feature_3d_dim}"
                )
            fused_inputs.append(feature_3d_flat)

        fused_inputs.append(pts_feature)
        fused = torch.cat(fused_inputs, dim=-1)                  # (B,N,fusion_dim)
        out = self.fusion_fc(fused) + feature_2d_flat
        out = self.fusion_norm(out)
        out = out.transpose(-1, -2).reshape(B, C, H, W).contiguous()
        return out


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
        detach_prob_for_enhancer: bool = True,
        detach_depth_feat_for_enhancer: bool = False,
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
        # self.enhancer = SingleScaleDepthFusionSpatialEnhancer(
        #     embed_dims=tok_feat_dim,
        #     feature_3d_dim=feature_3d_dim,
        #     ff_dim=256,
        #     min_depth=self.min_depth,
        #     max_depth=self.max_depth,
        #     num_depth=self.bin_num,
        #     with_feature_3d=self.use_depth_branch,
        # )

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
        # Prefer real sensor depth / explicit depth input
        cand_keys = [
            "depth",            # recommended: dataloader output, (B,1,H,W) or (B,H,W)
        ]

        # GT depth should be last-resort fallback only
        fallback_keys = [
            "gt_depth_m",
        ]

        for k in cand_keys + fallback_keys:
            if k not in end_points:
                continue
            d = end_points[k]
            if not torch.is_tensor(d):
                continue

            # normalize to (B,1,H,W)
            if d.dim() == 2:
                # unlikely after collate, but keep robust
                d = d.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
            elif d.dim() == 3:
                # could be (B,H,W) or (1,H,W)
                d = d.unsqueeze(1)                # -> (B,1,H,W)
            elif d.dim() == 4:
                d = d[:, :1]                      # keep single channel
            else:
                continue

            d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).float()
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

        # depth_map_pred = torch.nan_to_num(depth_map_pred_448, nan=0.0, posinf=0.0, neginf=0.0)
        # depth_map_pred = depth_map_pred.clamp(min=self.min_depth, max=self.max_depth)
        depth_map_pred = depth_map_pred_448
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
        # depth_prob_for_enh = depth_prob_448.detach() if self.detach_prob_for_enhancer else depth_prob_448
        # img_feat = img_feat.detach() if self.detach_prob_for_enhancer else img_feat
        # if feat_3d_single is not None and self.detach_depth_feat_for_enhancer:
        #     feat_3d_single = feat_3d_single.detach()

        # feat_grid_enh = self.enhancer(
        #     feat_2d=img_feat,
        #     depth_prob=depth_prob_for_enh,
        #     K=K,
        #     feature_3d=feat_3d_single,
        #     stride=1,
        # )
        feat_grid_enh = img_feat.detach()
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
                # depth_mu_map, depth_ent_map = self._get_depth_mu_entropy(
                #     depth_prob_448, self.enhancer.depth_bins, eps=self.enhancer.eps
                # )
                msg = (
                    f"[bip3d] iter={self._vis_iter} "
                    f"valid={valid_tok.float().sum(1).mean().item():.1f} "
                    f"obj_pred1={mask_obj_pred.float().sum(1).mean().item():.1f} "
                    f"cand_thr={mask_thr_pred.float().sum(1).mean().item():.1f} "
                    f"gr_raw[min,max]=({grasp_raw.min().item():.3f},{grasp_raw.max().item():.3f}) "
                    f"gr_sel[p10,p50,p90]=({torch.quantile(grasp_sel, 0.1).item():.3f},"
                    f"{torch.quantile(grasp_sel, 0.5).item():.3f},"
                    f"{torch.quantile(grasp_sel, 0.9).item():.3f}) "
                    # f"zmu[min,p50,max]=({depth_mu_map.min().item():.3f},"
                    # f"{torch.quantile(depth_mu_map.flatten(),0.5).item():.3f},"
                    # f"{depth_mu_map.max().item():.3f}) "
                    # f"dent[p50,p90]=({torch.quantile(depth_ent_map.flatten(),0.5).item():.3f},"
                    # f"{torch.quantile(depth_ent_map.flatten(),0.9).item():.3f}) "
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

            # depth_mu_map, depth_ent_map = self._get_depth_mu_entropy(
            #     depth_prob_448, self.enhancer.depth_bins, eps=self.enhancer.eps
            # )

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
                # self._save_map_png(depth_mu_map[b], os.path.join(out_dir, f"b{b}_depth_mu.png"),
                #                    vmin=self.min_depth, vmax=self.max_depth, cmap="viridis", title="Depth mean")
                # self._save_map_png(depth_ent_map[b], os.path.join(out_dir, f"b{b}_depth_entropy.png"),
                #                    cmap="magma", title="Depth entropy")
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


# 按你项目真实路径改
# from models.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
from utils.label_generation import generate_grasp_views
class GeometryAwareDenseFieldViewNet(nn.Module):
    """
    Simplified geometry-aware dense-field ViewNet
    only uses:
      - seed feature
      - ray
      - normal proxy
      - uncertainty
    outputs:
      - view_score: (B,M,V)
      - grasp_top_view_inds: (B,M)
      - grasp_top_view_xyz: (B,M,3)
      - grasp_top_view_rot: (B,M,3,3)
      - res_feat: (B,C,M)
    """
    def __init__(
        self,
        num_view: int,
        seed_feature_dim: int = 128,
        hidden_dim: int = 128,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        view_dirs: torch.Tensor = None,
        vis_dir: str = None,
        vis_every: int = 500,
    ):
        super().__init__()
        self.num_view = int(num_view)
        self.seed_feature_dim = int(seed_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)

        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        if view_dirs is None:
            view_dirs = generate_grasp_views(self.num_view)  # must match label order
        if not torch.is_tensor(view_dirs):
            view_dirs = torch.tensor(view_dirs, dtype=torch.float32)
        view_dirs = F.normalize(view_dirs.float(), dim=-1)
        self.register_buffer("view_dirs", view_dirs)  # (V,3)

        # for uncertainty from depth distribution
        bin_centers = torch.linspace(self.min_depth, self.max_depth, self.bin_num, dtype=torch.float32)
        self.register_buffer("bin_centers", bin_centers.view(1, self.bin_num, 1, 1))

        # query from seed feature + weak geometry
        # [seed_feat(C), ray(3), normal(3), uncert(1)]
        q_in_dim = self.seed_feature_dim + 3 + 3 + 1
        self.query_mlp = nn.Sequential(
            nn.Linear(q_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # anchor embedding from view direction
        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # explicit geometry compatibility bias
        # [ray_dot, abs(normal_dot), uncert]
        self.bias_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # residual feature to feed back into main path
        self.res_mlp = nn.Sequential(
            nn.Linear(hidden_dim, seed_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(seed_feature_dim, seed_feature_dim),
        )

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap='viridis', title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6, 6))
        if vmin is None:
            vmin = float(np.nanmin(arr2d))
        if vmax is None:
            vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _to_rgb_np(self, img_chw: torch.Tensor):
        x = img_chw.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = x.permute(1, 2, 0).numpy()
        return x

    def _save_seed_scatter_overlay(
        self,
        img_chw: torch.Tensor,
        token_sel_idx: torch.Tensor,
        values: torch.Tensor,
        out_path: str,
        H: int,
        W: int,
        cmap: str = 'viridis',
        title: str = None,
        vmin=None,
        vmax=None,
        dot_size: int = 10,
    ):
        img_np = self._to_rgb_np(img_chw)
        idx = token_sel_idx.detach().cpu()
        vals = values.detach().float().cpu()

        u = (idx % W).numpy()
        v = (idx // W).numpy()

        if vmin is None:
            vmin = float(vals.min().item())
        if vmax is None:
            vmax = float(vals.max().item())

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.scatter(u, v, c=vals.numpy(), s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_seed_view_id_overlay(
        self,
        img_chw: torch.Tensor,
        token_sel_idx: torch.Tensor,
        view_ids: torch.Tensor,
        out_path: str,
        H: int,
        W: int,
        title: str = None,
        dot_size: int = 10,
    ):
        img_np = self._to_rgb_np(img_chw)
        idx = token_sel_idx.detach().cpu()
        vids = view_ids.detach().float().cpu()

        u = (idx % W).numpy()
        v = (idx // W).numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.scatter(u, v, c=vids.numpy(), s=dot_size, cmap='hsv', vmin=0, vmax=max(self.num_view - 1, 1))
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def _maybe_visualize(
        self,
        end_points: dict,
        token_sel_idx: torch.Tensor,      # (B,M)
        uncert_map: torch.Tensor,         # (B,1,H,W)
        top_view_inds: torch.Tensor,      # (B,M)
        top1_score: torch.Tensor,         # (B,M)
        top1_margin: torch.Tensor,        # (B,M)
        normal_align: torch.Tensor,       # (B,M)
    ):
        if self.vis_dir is None:
            return
        if self._vis_iter % self.vis_every != 0:
            return
        if 'img' not in end_points:
            return

        try:
            img = end_points['img']
            B, _, H, W = img.shape
            b0 = 0

            prefix = f'viewnet_it{self._vis_iter:06d}'

            # 1) dense uncertainty
            self._save_map_png(
                uncert_map[b0, 0],
                os.path.join(self.vis_dir, f'{prefix}_uncert_map.png'),
                vmin=0.0,
                vmax=1.0,
                cmap='magma',
                title='depth uncertainty'
            )

            # 2) top-1 score on selected seeds
            self._save_seed_scatter_overlay(
                img[b0],
                token_sel_idx[b0],
                top1_score[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top1_score.png'),
                H=H,
                W=W,
                cmap='viridis',
                title='seed top1 view score',
                vmin=0.0,
                vmax=1.0,
            )

            # 3) top1-top2 margin
            self._save_seed_scatter_overlay(
                img[b0],
                token_sel_idx[b0],
                top1_margin[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top1_margin.png'),
                H=H,
                W=W,
                cmap='plasma',
                title='seed top1-top2 margin',
                vmin=0.0,
                vmax=float(max(top1_margin[b0].max().item(), 1e-6)),
            )

            # 4) |normal dot top-view|
            self._save_seed_scatter_overlay(
                img[b0],
                token_sel_idx[b0],
                normal_align[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_normal_align.png'),
                H=H,
                W=W,
                cmap='coolwarm',
                title='|normal · top_view|',
                vmin=0.0,
                vmax=1.0,
            )

            # 5) top-view id
            self._save_seed_view_id_overlay(
                img[b0],
                token_sel_idx[b0],
                top_view_inds[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top_view_id.png'),
                H=H,
                W=W,
                title='predicted top view id',
            )
        except Exception as e:
            print(f"[GeometryAwareDenseFieldViewNet vis] failed at iter {self._vis_iter}: {repr(e)}")
        
    @staticmethod
    def _gather_bchw(feat_bchw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
        # feat_bchw: (B,C,H,W), idx_bm: (B,M) flattened index
        B, C, H, W = feat_bchw.shape
        feat_flat = feat_bchw.view(B, C, H * W)
        idx = idx_bm.unsqueeze(1).expand(-1, C, -1)
        return torch.gather(feat_flat, 2, idx)  # (B,C,M)

    @staticmethod
    def _ray_dirs_from_idx(idx_bm: torch.Tensor, K_b33: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()

        fx = K_b33[:, 0, 0].unsqueeze(1)
        fy = K_b33[:, 1, 1].unsqueeze(1)
        cx = K_b33[:, 0, 2].unsqueeze(1)
        cy = K_b33[:, 1, 2].unsqueeze(1)

        x = (u - cx) / fx
        y = (v - cy) / fy
        z = torch.ones_like(x)
        rays = torch.stack([x, y, z], dim=-1)
        rays = F.normalize(rays, dim=-1)
        return rays  # (B,M,3)

    @staticmethod
    def _backproject_depth(depth_b1hw: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth_b1hw.shape
        device = depth_b1hw.device
        dtype = depth_b1hw.dtype

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        xs = xs.view(1, 1, H, W).expand(B, -1, -1, -1)
        ys = ys.view(1, 1, H, W).expand(B, -1, -1, -1)

        fx = K_b33[:, 0, 0].view(B, 1, 1, 1)
        fy = K_b33[:, 1, 1].view(B, 1, 1, 1)
        cx = K_b33[:, 0, 2].view(B, 1, 1, 1)
        cy = K_b33[:, 1, 2].view(B, 1, 1, 1)

        z = depth_b1hw
        x = (xs - cx) / fx * z
        y = (ys - cy) / fy * z
        xyz = torch.cat([x, y, z], dim=1)
        return xyz  # (B,3,H,W)

    @staticmethod
    def _normal_map_from_xyz(xyz_b3hw: torch.Tensor) -> torch.Tensor:
        dx = xyz_b3hw[:, :, :, 2:] - xyz_b3hw[:, :, :, :-2]      # (B,3,H,W-2)
        dy = xyz_b3hw[:, :, 2:, :] - xyz_b3hw[:, :, :-2, :]      # (B,3,H-2,W)

        dx = dx[:, :, 1:-1, :]                                   # (B,3,H-2,W-2)
        dy = dy[:, :, :, 1:-1]                                   # (B,3,H-2,W-2)

        dxp = dx.permute(0, 2, 3, 1).contiguous()
        dyp = dy.permute(0, 2, 3, 1).contiguous()

        n = torch.cross(dxp, dyp, dim=-1)                        # (B,H-2,W-2,3)
        n = F.normalize(n, dim=-1, eps=1e-6)
        n = n.permute(0, 3, 1, 2).contiguous()                  # (B,3,H-2,W-2)
        n = F.pad(n, (1, 1, 1, 1), mode='replicate')
        return n  # (B,3,H,W)

    def _depth_uncertainty(self, depth_prob_bdhw: torch.Tensor) -> torch.Tensor:
        prob = depth_prob_bdhw.clamp_min(1e-8)
        entropy = -(prob * prob.log()).sum(dim=1, keepdim=True) / math.log(self.bin_num)
        return entropy.clamp(0.0, 1.0)  # (B,1,H,W)

    def forward(
        self,
        seed_features: torch.Tensor,   # (B,C,M)
        token_sel_idx: torch.Tensor,   # (B,M)
        K: torch.Tensor,               # (B,3,3)
        depth_map: torch.Tensor,       # (B,1,H,W)
        depth_prob: torch.Tensor,      # (B,D,H,W)
        end_points: dict,
    ):
        B, C, M = seed_features.shape
        _, _, H, W = depth_map.shape
        device = seed_features.device

        # weak geometry: detached
        depth_map_det = depth_map.detach()
        depth_prob_det = depth_prob.detach()

        xyz_map = self._backproject_depth(depth_map_det, K)      # only for normal estimation
        normal_map = self._normal_map_from_xyz(xyz_map)
        uncert_map = self._depth_uncertainty(depth_prob_det)

        # per-seed weak geometry
        ray_bm3 = self._ray_dirs_from_idx(token_sel_idx, K, H, W)              # (B,M,3)
        normal_b3m = self._gather_bchw(normal_map, token_sel_idx)              # (B,3,M)
        uncert_b1m = self._gather_bchw(uncert_map, token_sel_idx)              # (B,1,M)

        normal_bm3 = normal_b3m.transpose(1, 2).contiguous()                   # (B,M,3)
        uncert_bm1 = uncert_b1m.transpose(1, 2).contiguous()                   # (B,M,1)

        # confidence-gated normal, but uncertainty itself remains explicit input
        normal_bm3 = normal_bm3 * (1.0 - uncert_bm1)

        seed_feat_bmC = seed_features.transpose(1, 2).contiguous()             # (B,M,C)

        query_in = torch.cat([
            seed_feat_bmC,
            ray_bm3,
            normal_bm3,
            uncert_bm1,
        ], dim=-1)                                                             # (B,M,C+7)

        q = self.query_mlp(query_in)                                           # (B,M,Hid)
        k = self.anchor_mlp(self.view_dirs)                                    # (V,Hid)

        compat = torch.einsum('bmh,vh->bmv', q, k) / math.sqrt(self.hidden_dim)

        ray_dot = torch.einsum('bmc,vc->bmv', ray_bm3, self.view_dirs)
        normal_dot = torch.einsum('bmc,vc->bmv', normal_bm3, self.view_dirs).abs()

        scalar_in = torch.stack([
            ray_dot,
            normal_dot,
            uncert_bm1.expand(-1, -1, self.num_view),
        ], dim=-1)                                                             # (B,M,V,3)

        bias = self.bias_mlp(scalar_in).squeeze(-1)                            # (B,M,V)

        view_score = compat + bias
        res_feat = self.res_mlp(q).transpose(1, 2).contiguous()                # (B,C,M)

        top_view_inds = torch.argmax(view_score, dim=-1)                       # (B,M)
        top_view_xyz = self.view_dirs.index_select(
            0, top_view_inds.reshape(-1)
        ).view(B, M, 3)

        zero_angle = torch.zeros(B * M, device=device, dtype=top_view_xyz.dtype)
        grasp_top_view_rot = batch_viewpoint_params_to_matrix(
            -top_view_xyz.view(-1, 3), zero_angle
        ).view(B, M, 3, 3)

        # ---------- debug / visualization stats ----------
        top2_vals, _ = torch.topk(view_score, k=min(2, self.num_view), dim=-1)
        top1_score = top2_vals[..., 0]                                        # (B,M)
        if self.num_view >= 2:
            top1_margin = top2_vals[..., 0] - top2_vals[..., 1]               # (B,M)
        else:
            top1_margin = torch.zeros_like(top1_score)

        normal_align = (normal_bm3 * top_view_xyz).sum(dim=-1).abs().clamp(0.0, 1.0)  # (B,M)

        self._maybe_visualize(
            end_points=end_points,
            token_sel_idx=token_sel_idx,
            uncert_map=uncert_map,
            top_view_inds=top_view_inds,
            top1_score=top1_score,
            top1_margin=top1_margin,
            normal_align=normal_align,
        )
        
        end_points['view_score'] = view_score                     # (B,M,300)
        end_points['grasp_top_view_inds'] = top_view_inds        # (B,M)
        end_points['grasp_top_view_xyz'] = top_view_xyz          # (B,M,3)
        end_points['grasp_top_view_rot'] = grasp_top_view_rot    # (B,M,3,3)

        # debug
        end_points['view_geom_uncert'] = uncert_bm1.squeeze(-1)  # (B,M)
        self._vis_iter += 1
        return end_points, res_feat
    

# class GeometryAwareDenseFieldAttnViewNet(nn.Module):
#     """
#     Drop-in replacement for GeometryAwareDenseFieldViewNet.

#     Core idea:
#       - keep weak geometry extraction exactly late and detached
#       - keep explicit geometric bias branch
#       - replace static q·k matching with:
#             anchor-query cross-attention over {appearance token, geometry token}

#     Inputs:
#       - seed_features: (B,C,M)
#       - token_sel_idx: (B,M)
#       - K:            (B,3,3)
#       - depth_map:    (B,1,H,W)
#       - depth_prob:   (B,D,H,W)
#       - end_points:   dict

#     Outputs:
#       - end_points['view_score']           : (B,M,V)
#       - end_points['grasp_top_view_inds']  : (B,M)
#       - end_points['grasp_top_view_xyz']   : (B,M,3)
#       - end_points['grasp_top_view_rot']   : (B,M,3,3)
#       - end_points['view_geom_uncert']     : (B,M)
#       - end_points['view_attn_top_geo']    : (B,M)
#       - end_points['view_attn_top_app']    : (B,M)
#       - end_points['view_attn_top_entropy']: (B,M)
#       - res_feat                           : (B,C,M)
#     """
#     def __init__(
#         self,
#         num_view: int,
#         seed_feature_dim: int = 128,
#         hidden_dim: int = 128,
#         min_depth: float = 0.2,
#         max_depth: float = 1.0,
#         bin_num: int = 256,
#         view_dirs: torch.Tensor = None,
#         vis_dir: str = None,
#         vis_every: int = 500,
#         num_heads: int = 4,
#         attn_dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.num_view = int(num_view)
#         self.seed_feature_dim = int(seed_feature_dim)
#         self.hidden_dim = int(hidden_dim)
#         self.min_depth = float(min_depth)
#         self.max_depth = float(max_depth)
#         self.bin_num = int(bin_num)
#         self.num_heads = int(num_heads)
#         self.attn_dropout = float(attn_dropout)

#         if self.hidden_dim % self.num_heads != 0:
#             raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}")

#         self.vis_dir = vis_dir
#         self.vis_every = int(vis_every)
#         self._vis_iter = 0
#         if self.vis_dir is not None:
#             os.makedirs(self.vis_dir, exist_ok=True)

#         view_dirs = F.normalize(view_dirs.float(), dim=-1)
#         self.register_buffer("view_dirs", view_dirs)  # (V,3)

#         # for uncertainty from depth distribution
#         bin_centers = torch.linspace(self.min_depth, self.max_depth, self.bin_num, dtype=torch.float32)
#         self.register_buffer("bin_centers", bin_centers.view(1, self.bin_num, 1, 1))

#         # per-seed conditioning query
#         # [seed_feat(C), ray(3), normal(3), uncert(1)]
#         q_in_dim = self.seed_feature_dim + 3 + 3 + 1
#         self.query_mlp = nn.Sequential(
#             nn.Linear(q_in_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#         )

#         # split the two evidence sources explicitly
#         self.app_mlp = nn.Sequential(
#             nn.Linear(self.seed_feature_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
#         self.geo_mlp = nn.Sequential(
#             nn.Linear(3 + 3 + 1, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#         # anchor embedding from view direction
#         self.anchor_mlp = nn.Sequential(
#             nn.Linear(3, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#         # anchor-query cross-attention:
#         # queries = V anchor tokens
#         # keys/values = 2 evidence tokens {appearance, geometry}
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=self.num_heads,
#             dropout=self.attn_dropout,
#             batch_first=True,
#         )

#         # compat score after attention
#         self.score_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, 1),
#         )

#         # explicit geometry compatibility bias
#         # [ray_dot, abs(normal_dot), uncert]
#         self.bias_mlp = nn.Sequential(
#             nn.Linear(3, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim // 2, 1),
#         )

#         # residual feature back to main path
#         self.res_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, seed_feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(seed_feature_dim, seed_feature_dim),
#         )

#     # -------------------------------------------------------------------------
#     # visualization helpers
#     # -------------------------------------------------------------------------
#     def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap='viridis', title=None):
#         if torch.is_tensor(arr2d):
#             arr2d = arr2d.detach().float().cpu().numpy()

#         arr2d = np.asarray(arr2d)
#         finite = np.isfinite(arr2d)
#         if finite.sum() == 0:
#             return

#         if vmin is None:
#             vmin = float(np.nanmin(arr2d))
#         if vmax is None:
#             vmax = float(np.nanmax(arr2d))
#         if not np.isfinite(vmin):
#             vmin = 0.0
#         if not np.isfinite(vmax) or vmax <= vmin:
#             vmax = vmin + 1e-6

#         plt.figure(figsize=(6, 6))
#         plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
#         plt.axis('off')
#         if title is not None:
#             plt.title(title)
#         plt.tight_layout(pad=0)
#         plt.savefig(out_path, dpi=150)
#         plt.close()

#     def _save_curve_png(self, arr1d, out_path, title=None, xlabel='anchor id', ylabel='value'):
#         if torch.is_tensor(arr1d):
#             arr1d = arr1d.detach().float().cpu().numpy()

#         arr1d = np.asarray(arr1d)
#         finite = np.isfinite(arr1d)
#         if finite.sum() == 0:
#             return

#         x = np.arange(arr1d.shape[0])
#         plt.figure(figsize=(8, 3))
#         plt.plot(x, arr1d, linewidth=1.2)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         if title is not None:
#             plt.title(title)
#         plt.tight_layout()
#         plt.savefig(out_path, dpi=150)
#         plt.close()

#     def _to_rgb_np(self, img_chw: torch.Tensor):
#         x = img_chw.detach().float().cpu()
#         x = x - x.min()
#         x = x / (x.max() + 1e-6)
#         x = x.permute(1, 2, 0).numpy()
#         return x

#     def _save_seed_scatter_overlay(
#         self,
#         img_chw: torch.Tensor,
#         token_sel_idx: torch.Tensor,
#         values: torch.Tensor,
#         out_path: str,
#         H: int,
#         W: int,
#         cmap: str = 'viridis',
#         title: str = None,
#         vmin=None,
#         vmax=None,
#         dot_size: int = 10,
#     ):
#         img_np = self._to_rgb_np(img_chw)
#         idx = token_sel_idx.detach().cpu()
#         vals = values.detach().float().cpu()

#         u = (idx % W).numpy()
#         v = (idx // W).numpy()
#         vals_np = vals.numpy()

#         finite = np.isfinite(vals_np)
#         if finite.sum() == 0:
#             return

#         u = u[finite]
#         v = v[finite]
#         vals_np = vals_np[finite]

#         if vmin is None:
#             vmin = float(np.min(vals_np))
#         if vmax is None:
#             vmax = float(np.max(vals_np))
#         if not np.isfinite(vmin):
#             vmin = 0.0
#         if not np.isfinite(vmax) or vmax <= vmin:
#             vmax = vmin + 1e-6

#         plt.figure(figsize=(6, 6))
#         plt.imshow(img_np)
#         plt.scatter(u, v, c=vals_np, s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
#         plt.axis('off')
#         if title is not None:
#             plt.title(title)
#         plt.tight_layout(pad=0)
#         plt.savefig(out_path, dpi=150)
#         plt.close()

#     def _save_seed_view_id_overlay(
#         self,
#         img_chw: torch.Tensor,
#         token_sel_idx: torch.Tensor,
#         view_ids: torch.Tensor,
#         out_path: str,
#         H: int,
#         W: int,
#         title: str = None,
#         dot_size: int = 10,
#     ):
#         img_np = self._to_rgb_np(img_chw)
#         idx = token_sel_idx.detach().cpu()
#         vids = torch.nan_to_num(view_ids.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

#         u = (idx % W).numpy()
#         v = (idx // W).numpy()
#         vids_np = vids.numpy()

#         plt.figure(figsize=(6, 6))
#         plt.imshow(img_np)
#         plt.scatter(u, v, c=vids_np, s=dot_size, cmap='hsv', vmin=0, vmax=max(self.num_view - 1, 1))
#         plt.axis('off')
#         if title is not None:
#             plt.title(title)
#         plt.tight_layout(pad=0)
#         plt.savefig(out_path, dpi=150)
#         plt.close()

#     @torch.no_grad()
#     def _maybe_visualize(
#         self,
#         end_points: dict,
#         token_sel_idx: torch.Tensor,        # (B,M)
#         uncert_map: torch.Tensor,           # (B,1,H,W)
#         top_view_inds: torch.Tensor,        # (B,M)
#         top1_score: torch.Tensor,           # (B,M)
#         top1_margin: torch.Tensor,          # (B,M)
#         normal_align: torch.Tensor,         # (B,M)
#         top_attn_app: torch.Tensor,         # (B,M)
#         top_attn_geo: torch.Tensor,         # (B,M)
#         top_attn_entropy: torch.Tensor,     # (B,M)
#         mean_geo_attn: torch.Tensor,        # (B,M)
#         anchor_geo_curve: torch.Tensor,     # (V,)
#     ):
#         if self.vis_dir is None:
#             return
#         if self._vis_iter % self.vis_every != 0:
#             return
#         if 'img' not in end_points:
#             return

#         try:
#             img = end_points['img']
#             B, _, H, W = img.shape
#             b0 = 0

#             prefix = f'viewnet_it{self._vis_iter:06d}'

#             # base diagnostics
#             self._save_map_png(
#                 uncert_map[b0, 0],
#                 os.path.join(self.vis_dir, f'{prefix}_uncert_map.png'),
#                 vmin=0.0,
#                 vmax=1.0,
#                 cmap='magma',
#                 title='depth uncertainty'
#             )

#             self._save_seed_scatter_overlay(
#                 img[b0], token_sel_idx[b0], top1_score[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_top1_score.png'),
#                 H=H, W=W, cmap='viridis',
#                 title='seed top1 view score'
#             )

#             self._save_seed_scatter_overlay(
#                 img[b0], token_sel_idx[b0], top1_margin[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_top1_margin.png'),
#                 H=H, W=W, cmap='plasma',
#                 title='seed top1-top2 margin',
#                 vmin=0.0,
#             )

#             self._save_seed_scatter_overlay(
#                 img[b0], token_sel_idx[b0], normal_align[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_normal_align.png'),
#                 H=H, W=W, cmap='coolwarm',
#                 title='|normal · top_view|',
#                 vmin=0.0, vmax=1.0,
#             )

#             self._save_seed_view_id_overlay(
#                 img[b0], token_sel_idx[b0], top_view_inds[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_top_view_id.png'),
#                 H=H, W=W,
#                 title='predicted top view id'
#             )

#             # attention-specific diagnostics
#             self._save_seed_scatter_overlay(
#                 img[b0], token_sel_idx[b0], top_attn_geo[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_top_attn_geo.png'),
#                 H=H, W=W, cmap='viridis',
#                 title='top-view attention to geometry token',
#                 vmin=0.0, vmax=1.0,
#             )

#             self._save_seed_scatter_overlay(
#                 img[b0], token_sel_idx[b0], top_attn_app[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_top_attn_app.png'),
#                 H=H, W=W, cmap='viridis',
#                 title='top-view attention to appearance token',
#                 vmin=0.0, vmax=1.0,
#             )

#             self._save_seed_scatter_overlay(
#                 img[b0], token_sel_idx[b0], top_attn_entropy[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_top_attn_entropy.png'),
#                 H=H, W=W, cmap='plasma',
#                 title='top-view attention entropy',
#                 vmin=0.0, vmax=1.0,
#             )

#             self._save_seed_scatter_overlay(
#                 img[b0], token_sel_idx[b0], mean_geo_attn[b0],
#                 os.path.join(self.vis_dir, f'{prefix}_seed_mean_geo_attn.png'),
#                 H=H, W=W, cmap='viridis',
#                 title='mean geometry attention over anchors',
#                 vmin=0.0, vmax=1.0,
#             )

#             self._save_curve_png(
#                 anchor_geo_curve,
#                 os.path.join(self.vis_dir, f'{prefix}_anchor_geo_attn_curve.png'),
#                 title='mean geometry attention over anchor ids',
#                 xlabel='view anchor id',
#                 ylabel='mean geo attention',
#             )

#         except Exception as e:
#             print(f"[GeometryAwareDenseFieldAttnViewNet vis] failed at iter {self._vis_iter}: {repr(e)}")

#     # -------------------------------------------------------------------------
#     # geometry helpers
#     # -------------------------------------------------------------------------
#     @staticmethod
#     def _gather_bchw(feat_bchw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
#         # feat_bchw: (B,C,H,W), idx_bm: (B,M) flattened index
#         B, C, H, W = feat_bchw.shape
#         feat_flat = feat_bchw.view(B, C, H * W)
#         idx = idx_bm.unsqueeze(1).expand(-1, C, -1)
#         return torch.gather(feat_flat, 2, idx)  # (B,C,M)

#     @staticmethod
#     def _ray_dirs_from_idx(idx_bm: torch.Tensor, K_b33: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         u = (idx_bm % W).float()
#         v = (idx_bm // W).float()

#         fx = K_b33[:, 0, 0].unsqueeze(1)
#         fy = K_b33[:, 1, 1].unsqueeze(1)
#         cx = K_b33[:, 0, 2].unsqueeze(1)
#         cy = K_b33[:, 1, 2].unsqueeze(1)

#         x = (u - cx) / fx
#         y = (v - cy) / fy
#         z = torch.ones_like(x)
#         rays = torch.stack([x, y, z], dim=-1)
#         rays = F.normalize(rays, dim=-1)
#         return rays  # (B,M,3)

#     @staticmethod
#     def _backproject_depth(depth_b1hw: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
#         B, _, H, W = depth_b1hw.shape
#         device = depth_b1hw.device
#         dtype = depth_b1hw.dtype

#         ys, xs = torch.meshgrid(
#             torch.arange(H, device=device, dtype=dtype),
#             torch.arange(W, device=device, dtype=dtype),
#             indexing='ij'
#         )
#         xs = xs.view(1, 1, H, W).expand(B, -1, -1, -1)
#         ys = ys.view(1, 1, H, W).expand(B, -1, -1, -1)

#         fx = K_b33[:, 0, 0].view(B, 1, 1, 1)
#         fy = K_b33[:, 1, 1].view(B, 1, 1, 1)
#         cx = K_b33[:, 0, 2].view(B, 1, 1, 1)
#         cy = K_b33[:, 1, 2].view(B, 1, 1, 1)

#         z = depth_b1hw
#         x = (xs - cx) / fx * z
#         y = (ys - cy) / fy * z
#         xyz = torch.cat([x, y, z], dim=1)
#         return xyz  # (B,3,H,W)

#     @staticmethod
#     def _normal_map_from_xyz(xyz_b3hw: torch.Tensor) -> torch.Tensor:
#         dx = xyz_b3hw[:, :, :, 2:] - xyz_b3hw[:, :, :, :-2]      # (B,3,H,W-2)
#         dy = xyz_b3hw[:, :, 2:, :] - xyz_b3hw[:, :, :-2, :]      # (B,3,H-2,W)

#         dx = dx[:, :, 1:-1, :]                                   # (B,3,H-2,W-2)
#         dy = dy[:, :, :, 1:-1]                                   # (B,3,H-2,W-2)

#         dxp = dx.permute(0, 2, 3, 1).contiguous()
#         dyp = dy.permute(0, 2, 3, 1).contiguous()

#         n = torch.cross(dxp, dyp, dim=-1)                        # (B,H-2,W-2,3)
#         n = F.normalize(n, dim=-1, eps=1e-6)
#         n = n.permute(0, 3, 1, 2).contiguous()                  # (B,3,H-2,W-2)
#         n = F.pad(n, (1, 1, 1, 1), mode='replicate')
#         return n  # (B,3,H,W)

#     def _depth_uncertainty(self, depth_prob_bdhw: torch.Tensor) -> torch.Tensor:
#         prob = depth_prob_bdhw.clamp_min(1e-8)
#         entropy = -(prob * prob.log()).sum(dim=1, keepdim=True) / math.log(self.bin_num)
#         return entropy.clamp(0.0, 1.0)  # (B,1,H,W)

#     # -------------------------------------------------------------------------
#     # forward
#     # -------------------------------------------------------------------------
#     def forward(
#         self,
#         seed_features: torch.Tensor,   # (B,C,M)
#         token_sel_idx: torch.Tensor,   # (B,M)
#         K: torch.Tensor,               # (B,3,3)
#         depth_map: torch.Tensor,       # (B,1,H,W)
#         depth_prob: torch.Tensor,      # (B,D,H,W)
#         end_points: dict,
#     ):
#         B, C, M = seed_features.shape
#         _, _, H, W = depth_map.shape
#         device = seed_features.device
#         V = self.num_view
#         Hdim = self.hidden_dim

#         # weak geometry: detached
#         depth_map_det = depth_map.detach()
#         depth_prob_det = depth_prob.detach()

#         xyz_map = self._backproject_depth(depth_map_det, K)      # only for normal estimation
#         normal_map = self._normal_map_from_xyz(xyz_map)
#         uncert_map = self._depth_uncertainty(depth_prob_det)

#         # per-seed weak geometry
#         ray_bm3 = self._ray_dirs_from_idx(token_sel_idx, K, H, W)          # (B,M,3)
#         normal_b3m = self._gather_bchw(normal_map, token_sel_idx)          # (B,3,M)
#         uncert_b1m = self._gather_bchw(uncert_map, token_sel_idx)          # (B,1,M)

#         normal_bm3 = normal_b3m.transpose(1, 2).contiguous()               # (B,M,3)
#         uncert_bm1 = uncert_b1m.transpose(1, 2).contiguous()               # (B,M,1)

#         # sanitize
#         normal_bm3 = torch.nan_to_num(normal_bm3, nan=0.0, posinf=0.0, neginf=0.0)
#         uncert_bm1 = torch.nan_to_num(uncert_bm1, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)
#         ray_bm3 = torch.nan_to_num(ray_bm3, nan=0.0, posinf=0.0, neginf=0.0)

#         # confidence-gated normal, uncertainty still remains explicit input
#         normal_bm3 = normal_bm3 * (1.0 - uncert_bm1)

#         seed_feat_bmC = seed_features.transpose(1, 2).contiguous()         # (B,M,C)

#         # ------------------------------------------------------------------
#         # seed-level conditioning
#         # ------------------------------------------------------------------
#         query_in = torch.cat([
#             seed_feat_bmC,
#             ray_bm3,
#             normal_bm3,
#             uncert_bm1,
#         ], dim=-1)                                                         # (B,M,C+7)
#         q_seed = self.query_mlp(query_in)                                  # (B,M,H)

#         # two evidence tokens
#         app_tok = self.app_mlp(seed_feat_bmC)                              # (B,M,H)
#         geo_in = torch.cat([ray_bm3, normal_bm3, uncert_bm1], dim=-1)     # (B,M,7)
#         geo_tok = self.geo_mlp(geo_in)                                     # (B,M,H)
#         geo_tok = geo_tok * (1.0 - uncert_bm1)                             # uncertainty gate

#         # anchor tokens
#         anchor_tok = self.anchor_mlp(self.view_dirs)                       # (V,H)
#         anchor_tok = anchor_tok.view(1, 1, V, Hdim).expand(B, M, V, Hdim) # (B,M,V,H)

#         # seed-conditioned anchor queries
#         anchor_query = anchor_tok + q_seed.unsqueeze(2)                    # (B,M,V,H)

#         # cross-attention over the tiny evidence set {appearance, geometry}
#         q_attn = anchor_query.reshape(B * M, V, Hdim)                      # (BM,V,H)
#         kv = torch.stack([app_tok, geo_tok], dim=2).reshape(B * M, 2, Hdim)  # (BM,2,H)

#         attn_out, attn_w = self.cross_attn(
#             q_attn,
#             kv,
#             kv,
#             need_weights=True,
#             average_attn_weights=False,
#         )
#         # attn_out: (BM,V,H)
#         # attn_w:   (BM,num_heads,V,2)

#         attn_out = attn_out.view(B, M, V, Hdim)
#         attn_w = attn_w.view(B, M, self.num_heads, V, 2).mean(dim=2)       # (B,M,V,2)

#         # compat from attended anchor representation
#         compat_in = torch.cat([attn_out, anchor_query], dim=-1)            # (B,M,V,2H)
#         compat = self.score_mlp(compat_in).squeeze(-1)                     # (B,M,V)

#         # explicit geometric bias stays as an independent branch
#         ray_dot = torch.einsum('bmc,vc->bmv', ray_bm3, self.view_dirs)
#         normal_dot = torch.einsum('bmc,vc->bmv', normal_bm3, self.view_dirs).abs()

#         scalar_in = torch.stack([
#             ray_dot,
#             normal_dot,
#             uncert_bm1.expand(-1, -1, V),
#         ], dim=-1)                                                         # (B,M,V,3)

#         bias = self.bias_mlp(scalar_in).squeeze(-1)                        # (B,M,V)

#         view_score = compat + bias

#         # seed residual back to main path
#         seed_ctx = q_seed + attn_out.mean(dim=2)                           # (B,M,H)
#         res_feat = self.res_mlp(seed_ctx).transpose(1, 2).contiguous()     # (B,C,M)

#         # top view
#         top_view_inds = torch.argmax(view_score, dim=-1)                   # (B,M)
#         top_view_xyz = self.view_dirs.index_select(
#             0, top_view_inds.reshape(-1)
#         ).view(B, M, 3)

#         zero_angle = torch.zeros(B * M, device=device, dtype=top_view_xyz.dtype)
#         grasp_top_view_rot = batch_viewpoint_params_to_matrix(
#             -top_view_xyz.view(-1, 3), zero_angle
#         ).view(B, M, 3, 3)

#         # ------------------------------------------------------------------
#         # attention diagnostics
#         # ------------------------------------------------------------------
#         gather_idx = top_view_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)  # (B,M,1,2)
#         top_attn = torch.gather(attn_w, 2, gather_idx).squeeze(2)                     # (B,M,2)
#         top_attn = torch.nan_to_num(top_attn, nan=0.5, posinf=0.5, neginf=0.5)

#         top_attn_app = top_attn[..., 0]                                               # (B,M)
#         top_attn_geo = top_attn[..., 1]                                               # (B,M)

#         p = top_attn.clamp_min(1e-8)
#         top_attn_entropy = -(p * p.log()).sum(dim=-1) / math.log(2.0)                # (B,M)
#         top_attn_entropy = torch.nan_to_num(top_attn_entropy, nan=0.0, posinf=0.0, neginf=0.0)

#         mean_geo_attn = attn_w[..., 1].mean(dim=-1)                                   # (B,M)
#         mean_geo_attn = torch.nan_to_num(mean_geo_attn, nan=0.0, posinf=0.0, neginf=0.0)

#         anchor_geo_curve = attn_w[..., 1].mean(dim=(0, 1))                            # (V,)
#         anchor_geo_curve = torch.nan_to_num(anchor_geo_curve, nan=0.0, posinf=0.0, neginf=0.0)

#         # ------------------------------------------------------------------
#         # existing diagnostics
#         # ------------------------------------------------------------------
#         top2_vals, _ = torch.topk(view_score, k=min(2, V), dim=-1)
#         top1_score = top2_vals[..., 0]
#         if V >= 2:
#             top1_margin = top2_vals[..., 0] - top2_vals[..., 1]
#         else:
#             top1_margin = torch.zeros_like(top1_score)

#         top1_score = torch.nan_to_num(top1_score, nan=0.0, posinf=0.0, neginf=0.0)
#         top1_margin = torch.nan_to_num(top1_margin, nan=0.0, posinf=0.0, neginf=0.0)

#         normal_align = (normal_bm3 * top_view_xyz).sum(dim=-1).abs().clamp(0.0, 1.0)
#         normal_align = torch.nan_to_num(normal_align, nan=0.0, posinf=0.0, neginf=0.0)

#         uncert_map = torch.nan_to_num(uncert_map, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)

#         self._maybe_visualize(
#             end_points=end_points,
#             token_sel_idx=token_sel_idx,
#             uncert_map=uncert_map,
#             top_view_inds=top_view_inds,
#             top1_score=top1_score,
#             top1_margin=top1_margin,
#             normal_align=normal_align,
#             top_attn_app=top_attn_app,
#             top_attn_geo=top_attn_geo,
#             top_attn_entropy=top_attn_entropy,
#             mean_geo_attn=mean_geo_attn,
#             anchor_geo_curve=anchor_geo_curve,
#         )

#         # outputs
#         end_points['view_score'] = view_score                       # (B,M,V)
#         end_points['grasp_top_view_inds'] = top_view_inds          # (B,M)
#         end_points['grasp_top_view_xyz'] = top_view_xyz            # (B,M,3)
#         end_points['grasp_top_view_rot'] = grasp_top_view_rot      # (B,M,3,3)

#         end_points['view_geom_uncert'] = uncert_bm1.squeeze(-1)    # (B,M)
#         end_points['view_attn_top_geo'] = top_attn_geo             # (B,M)
#         end_points['view_attn_top_app'] = top_attn_app             # (B,M)
#         end_points['view_attn_top_entropy'] = top_attn_entropy     # (B,M)
#         end_points['view_attn_mean_geo'] = mean_geo_attn           # (B,M)

#         self._vis_iter += 1
#         return end_points, res_feat


class GeometryAwareDenseFieldAttnViewNet(nn.Module):
    """
    Drop-in replacement for GeometryAwareDenseFieldViewNet.

    Core idea:
      - keep weak geometry extraction exactly late and detached
      - replace static q·k matching with:
            anchor-query cross-attention over {appearance token, geometry token}

    Inputs:
      - seed_features: (B,C,M)
      - token_sel_idx: (B,M)
      - K:            (B,3,3)
      - depth_map:    (B,1,H,W)
      - depth_prob:   (B,D,H,W)
      - end_points:   dict

    Outputs:
      - end_points['view_score']           : (B,M,V)
      - end_points['grasp_top_view_inds']  : (B,M)
      - end_points['grasp_top_view_xyz']   : (B,M,3)
      - end_points['grasp_top_view_rot']   : (B,M,3,3)
      - end_points['view_geom_uncert']     : (B,M)
      - end_points['view_attn_top_geo']    : (B,M)
      - end_points['view_attn_top_app']    : (B,M)
      - end_points['view_attn_top_entropy']: (B,M)
      - res_feat                           : (B,C,M)
    """
    def __init__(
        self,
        num_view: int,
        seed_feature_dim: int = 128,
        hidden_dim: int = 128,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        view_dirs: torch.Tensor = None,
        vis_dir: str = None,
        vis_every: int = 500,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_view = int(num_view)
        self.seed_feature_dim = int(seed_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)
        self.num_heads = int(num_heads)
        self.attn_dropout = float(attn_dropout)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}")

        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        view_dirs = F.normalize(view_dirs.float(), dim=-1)
        self.register_buffer("view_dirs", view_dirs)  # (V,3)

        # split the two evidence sources explicitly
        self.app_mlp = nn.Sequential(
            nn.Linear(self.seed_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(3 + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # anchor embedding from view direction
        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # anchor-query cross-attention:
        # queries = V anchor tokens
        # keys/values = 2 evidence tokens {appearance, geometry}
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

        # compat score after attention
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # residual feature back to main path
        self.res_mlp = nn.Sequential(
            nn.Linear(hidden_dim, seed_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(seed_feature_dim, seed_feature_dim),
        )

    # -------------------------------------------------------------------------
    # visualization helpers
    # -------------------------------------------------------------------------
    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap='viridis', title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()

        arr2d = np.asarray(arr2d)
        finite = np.isfinite(arr2d)
        if finite.sum() == 0:
            return

        if vmin is None:
            vmin = float(np.nanmin(arr2d))
        if vmax is None:
            vmax = float(np.nanmax(arr2d))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6

        plt.figure(figsize=(6, 6))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_curve_png(self, arr1d, out_path, title=None, xlabel='anchor id', ylabel='value'):
        if torch.is_tensor(arr1d):
            arr1d = arr1d.detach().float().cpu().numpy()

        arr1d = np.asarray(arr1d)
        finite = np.isfinite(arr1d)
        if finite.sum() == 0:
            return

        x = np.arange(arr1d.shape[0])
        plt.figure(figsize=(8, 3))
        plt.plot(x, arr1d, linewidth=1.2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _to_rgb_np(self, img_chw: torch.Tensor):
        x = img_chw.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = x.permute(1, 2, 0).numpy()
        return x

    def _save_seed_scatter_overlay(
        self,
        img_chw: torch.Tensor,
        token_sel_idx: torch.Tensor,
        values: torch.Tensor,
        out_path: str,
        H: int,
        W: int,
        cmap: str = 'viridis',
        title: str = None,
        vmin=None,
        vmax=None,
        dot_size: int = 10,
    ):
        img_np = self._to_rgb_np(img_chw)
        idx = token_sel_idx.detach().cpu()
        vals = values.detach().float().cpu()

        u = (idx % W).numpy()
        v = (idx // W).numpy()
        vals_np = vals.numpy()

        finite = np.isfinite(vals_np)
        if finite.sum() == 0:
            return

        u = u[finite]
        v = v[finite]
        vals_np = vals_np[finite]

        if vmin is None:
            vmin = float(np.min(vals_np))
        if vmax is None:
            vmax = float(np.max(vals_np))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.scatter(u, v, c=vals_np, s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_seed_view_id_overlay(
        self,
        img_chw: torch.Tensor,
        token_sel_idx: torch.Tensor,
        view_ids: torch.Tensor,
        out_path: str,
        H: int,
        W: int,
        title: str = None,
        dot_size: int = 10,
    ):
        img_np = self._to_rgb_np(img_chw)
        idx = token_sel_idx.detach().cpu()
        vids = torch.nan_to_num(view_ids.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

        u = (idx % W).numpy()
        v = (idx // W).numpy()
        vids_np = vids.numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.scatter(u, v, c=vids_np, s=dot_size, cmap='hsv', vmin=0, vmax=max(self.num_view - 1, 1))
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def _maybe_visualize(
        self,
        end_points: dict,
        token_sel_idx: torch.Tensor,        # (B,M)
        uncert_map: torch.Tensor,           # (B,1,H,W)
        top_view_inds: torch.Tensor,        # (B,M)
        top1_score: torch.Tensor,           # (B,M)
        top1_margin: torch.Tensor,          # (B,M)
        normal_align: torch.Tensor,         # (B,M)
        top_attn_app: torch.Tensor,         # (B,M)
        top_attn_geo: torch.Tensor,         # (B,M)
        top_attn_geo_eff: torch.Tensor,
        top_attn_entropy: torch.Tensor,     # (B,M)
        mean_geo_attn: torch.Tensor,        # (B,M)
        anchor_geo_curve: torch.Tensor,     # (V,)
    ):
        if self.vis_dir is None:
            return
        if self._vis_iter % self.vis_every != 0:
            return
        if 'img' not in end_points:
            return

        try:
            img = end_points['img']
            B, _, H, W = img.shape
            b0 = 0

            prefix = f'viewnet_it{self._vis_iter:06d}'

            # base diagnostics
            self._save_map_png(
                uncert_map[b0, 0],
                os.path.join(self.vis_dir, f'{prefix}_uncert_map.png'),
                vmin=0.0,
                vmax=1.0,
                cmap='magma',
                title='depth uncertainty'
            )

            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], top1_score[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top1_score.png'),
                H=H, W=W, cmap='viridis',
                title='seed top1 view score'
            )

            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], top1_margin[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top1_margin.png'),
                H=H, W=W, cmap='plasma',
                title='seed top1-top2 margin',
                vmin=0.0,
            )

            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], normal_align[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_normal_align.png'),
                H=H, W=W, cmap='coolwarm',
                title='|normal · top_view|',
                vmin=0.0, vmax=1.0,
            )

            self._save_seed_view_id_overlay(
                img[b0], token_sel_idx[b0], top_view_inds[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top_view_id.png'),
                H=H, W=W,
                title='predicted top view id'
            )

            # attention-specific diagnostics
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], top_attn_geo[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top_attn_geo.png'),
                H=H, W=W, cmap='viridis',
                title='top-view attention to geometry token',
                vmin=0.0, vmax=1.0,
            )
            
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], top_attn_geo_eff[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top_attn_geo_eff.png'),
                H=H, W=W, cmap='viridis',
                title='effective geometry attention',
                vmin=0.0, vmax=1.0,
            )
            
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], top_attn_app[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top_attn_app.png'),
                H=H, W=W, cmap='viridis',
                title='top-view attention to appearance token',
                vmin=0.0, vmax=1.0,
            )

            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], top_attn_entropy[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_top_attn_entropy.png'),
                H=H, W=W, cmap='plasma',
                title='top-view attention entropy',
                vmin=0.0, vmax=1.0,
            )

            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], mean_geo_attn[b0],
                os.path.join(self.vis_dir, f'{prefix}_seed_mean_geo_attn.png'),
                H=H, W=W, cmap='viridis',
                title='mean geometry attention over anchors',
                vmin=0.0, vmax=1.0,
            )

            self._save_curve_png(
                anchor_geo_curve,
                os.path.join(self.vis_dir, f'{prefix}_anchor_geo_attn_curve.png'),
                title='mean geometry attention over anchor ids',
                xlabel='view anchor id',
                ylabel='mean geo attention',
            )

        except Exception as e:
            print(f"[GeometryAwareDenseFieldAttnViewNet vis] failed at iter {self._vis_iter}: {repr(e)}")

    # -------------------------------------------------------------------------
    # geometry helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _gather_bchw(feat_bchw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
        # feat_bchw: (B,C,H,W), idx_bm: (B,M) flattened index
        B, C, H, W = feat_bchw.shape
        feat_flat = feat_bchw.view(B, C, H * W)
        idx = idx_bm.unsqueeze(1).expand(-1, C, -1)
        return torch.gather(feat_flat, 2, idx)  # (B,C,M)

    @staticmethod
    def _ray_dirs_from_idx(idx_bm: torch.Tensor, K_b33: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()

        fx = K_b33[:, 0, 0].unsqueeze(1)
        fy = K_b33[:, 1, 1].unsqueeze(1)
        cx = K_b33[:, 0, 2].unsqueeze(1)
        cy = K_b33[:, 1, 2].unsqueeze(1)

        x = (u - cx) / fx
        y = (v - cy) / fy
        z = torch.ones_like(x)
        rays = torch.stack([x, y, z], dim=-1)
        rays = F.normalize(rays, dim=-1)
        return rays  # (B,M,3)

    @staticmethod
    def _backproject_depth(depth_b1hw: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth_b1hw.shape
        device = depth_b1hw.device
        dtype = depth_b1hw.dtype

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        xs = xs.view(1, 1, H, W).expand(B, -1, -1, -1)
        ys = ys.view(1, 1, H, W).expand(B, -1, -1, -1)

        fx = K_b33[:, 0, 0].view(B, 1, 1, 1)
        fy = K_b33[:, 1, 1].view(B, 1, 1, 1)
        cx = K_b33[:, 0, 2].view(B, 1, 1, 1)
        cy = K_b33[:, 1, 2].view(B, 1, 1, 1)

        z = depth_b1hw
        x = (xs - cx) / fx * z
        y = (ys - cy) / fy * z
        xyz = torch.cat([x, y, z], dim=1)
        return xyz  # (B,3,H,W)

    @staticmethod
    def _normal_map_from_xyz(xyz_b3hw: torch.Tensor) -> torch.Tensor:
        dx = xyz_b3hw[:, :, :, 2:] - xyz_b3hw[:, :, :, :-2]      # (B,3,H,W-2)
        dy = xyz_b3hw[:, :, 2:, :] - xyz_b3hw[:, :, :-2, :]      # (B,3,H-2,W)

        dx = dx[:, :, 1:-1, :]                                   # (B,3,H-2,W-2)
        dy = dy[:, :, :, 1:-1]                                   # (B,3,H-2,W-2)

        dxp = dx.permute(0, 2, 3, 1).contiguous()
        dyp = dy.permute(0, 2, 3, 1).contiguous()

        n = torch.cross(dxp, dyp, dim=-1)                        # (B,H-2,W-2,3)
        n = F.normalize(n, dim=-1, eps=1e-6)
        n = n.permute(0, 3, 1, 2).contiguous()                  # (B,3,H-2,W-2)
        n = F.pad(n, (1, 1, 1, 1), mode='replicate')
        return n  # (B,3,H,W)

    def _depth_uncertainty(self, depth_prob_bdhw: torch.Tensor) -> torch.Tensor:
        prob = depth_prob_bdhw.clamp_min(1e-8)
        entropy = -(prob * prob.log()).sum(dim=1, keepdim=True) / math.log(self.bin_num)
        return entropy.clamp(0.0, 1.0)  # (B,1,H,W)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        seed_features: torch.Tensor,   # (B,C,M)
        token_sel_idx: torch.Tensor,   # (B,M)
        K: torch.Tensor,               # (B,3,3)
        depth_map: torch.Tensor,       # (B,1,H,W)
        depth_prob: torch.Tensor,      # (B,D,H,W)
        end_points: dict,
    ):
        B, C, M = seed_features.shape
        _, _, H, W = depth_map.shape
        device = seed_features.device
        V = self.num_view
        Hdim = self.hidden_dim

        # weak geometry: detached
        depth_map_det = depth_map.detach()
        depth_prob_det = depth_prob.detach()

        xyz_map = self._backproject_depth(depth_map_det, K)      # only for normal estimation
        normal_map = self._normal_map_from_xyz(xyz_map)
        uncert_map = self._depth_uncertainty(depth_prob_det)

        # per-seed weak geometry
        ray_bm3 = self._ray_dirs_from_idx(token_sel_idx, K, H, W)          # (B,M,3)
        normal_b3m = self._gather_bchw(normal_map, token_sel_idx)          # (B,3,M)
        uncert_b1m = self._gather_bchw(uncert_map, token_sel_idx)          # (B,1,M)

        normal_bm3 = normal_b3m.transpose(1, 2).contiguous()               # (B,M,3)
        uncert_bm1 = uncert_b1m.transpose(1, 2).contiguous()               # (B,M,1)

        # sanitize
        normal_bm3 = torch.nan_to_num(normal_bm3, nan=0.0, posinf=0.0, neginf=0.0)
        uncert_bm1 = torch.nan_to_num(uncert_bm1, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)
        ray_bm3 = torch.nan_to_num(ray_bm3, nan=0.0, posinf=0.0, neginf=0.0)

        seed_feat_bmC = seed_features.transpose(1, 2).contiguous()         # (B,M,C)

        # two evidence tokens
        app_tok = self.app_mlp(seed_feat_bmC)                              # (B,M,H)
        geo_tok = self.geo_mlp(torch.cat([ray_bm3, normal_bm3], dim=-1))   # (B,M,H)
        geo_tok = geo_tok * (1.0 - uncert_bm1)                             # uncertainty gate
        q_seed = app_tok + geo_tok
    
        # anchor tokens
        anchor_tok = self.anchor_mlp(self.view_dirs)                       # (V,H)
        anchor_tok = anchor_tok.view(1, 1, V, Hdim).expand(B, M, V, Hdim) # (B,M,V,H)

        # seed-conditioned anchor queries
        anchor_query = anchor_tok + q_seed.unsqueeze(2)                    # (B,M,V,H)

        # cross-attention over the tiny evidence set {appearance, geometry}
        q_attn = anchor_query.reshape(B * M, V, Hdim)                      # (BM,V,H)
        kv = torch.stack([app_tok, geo_tok], dim=2).reshape(B * M, 2, Hdim)  # (BM,2,H)

        attn_out, attn_w = self.cross_attn(
            q_attn,
            kv,
            kv,
            need_weights=True,
            average_attn_weights=False,
        )
        # attn_out: (BM,V,H)
        # attn_w:   (BM,num_heads,V,2)

        attn_out = attn_out.view(B, M, V, Hdim)
        attn_w = attn_w.view(B, M, self.num_heads, V, 2).mean(dim=2)       # (B,M,V,2)

        # compat from attended anchor representation
        ray_dot = torch.einsum('bmc,vc->bmv', ray_bm3, self.view_dirs)
        normal_dot = torch.einsum('bmc,vc->bmv', normal_bm3, self.view_dirs).abs()

        score_in = torch.cat([
            attn_out,
            anchor_query,
            ray_dot.unsqueeze(-1),
            normal_dot.unsqueeze(-1),
        ], dim=-1)  # (B,M,V,2H+2)

        view_score = self.score_mlp(score_in).squeeze(-1)  # (B,M,V)

        # seed residual back to main path
        res_feat = self.res_mlp(q_seed).transpose(1, 2).contiguous()     # (B,C,M)

        # top view
        top_view_inds = torch.argmax(view_score, dim=-1)                   # (B,M)
        top_view_xyz = self.view_dirs.index_select(
            0, top_view_inds.reshape(-1)
        ).view(B, M, 3)

        zero_angle = torch.zeros(B * M, device=device, dtype=top_view_xyz.dtype)
        grasp_top_view_rot = batch_viewpoint_params_to_matrix(
            -top_view_xyz.view(-1, 3), zero_angle
        ).view(B, M, 3, 3)

        # ------------------------------------------------------------------
        # attention diagnostics
        # ------------------------------------------------------------------
        gather_idx = top_view_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)  # (B,M,1,2)
        top_attn = torch.gather(attn_w, 2, gather_idx).squeeze(2)                     # (B,M,2)
        top_attn = torch.nan_to_num(top_attn, nan=0.5, posinf=0.5, neginf=0.5)

        top_attn_app = top_attn[..., 0]                                               # (B,M)
        top_attn_geo = top_attn[..., 1]                                               # (B,M)
        top_attn_geo_eff = top_attn_geo * (1.0 - uncert_bm1.squeeze(-1))
        
        p = top_attn.clamp_min(1e-8)
        top_attn_entropy = -(p * p.log()).sum(dim=-1) / math.log(2.0)                # (B,M)
        top_attn_entropy = torch.nan_to_num(top_attn_entropy, nan=0.0, posinf=0.0, neginf=0.0)

        mean_geo_attn = attn_w[..., 1].mean(dim=-1)                                   # (B,M)
        mean_geo_attn = torch.nan_to_num(mean_geo_attn, nan=0.0, posinf=0.0, neginf=0.0)

        anchor_geo_curve = attn_w[..., 1].mean(dim=(0, 1))                            # (V,)
        anchor_geo_curve = torch.nan_to_num(anchor_geo_curve, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------------
        # existing diagnostics
        # ------------------------------------------------------------------
        top2_vals, _ = torch.topk(view_score, k=min(2, V), dim=-1)
        top1_score = top2_vals[..., 0]
        if V >= 2:
            top1_margin = top2_vals[..., 0] - top2_vals[..., 1]
        else:
            top1_margin = torch.zeros_like(top1_score)

        top1_score = torch.nan_to_num(top1_score, nan=0.0, posinf=0.0, neginf=0.0)
        top1_margin = torch.nan_to_num(top1_margin, nan=0.0, posinf=0.0, neginf=0.0)

        normal_align = (normal_bm3 * top_view_xyz).sum(dim=-1).abs().clamp(0.0, 1.0)
        normal_align = torch.nan_to_num(normal_align, nan=0.0, posinf=0.0, neginf=0.0)

        uncert_map = torch.nan_to_num(uncert_map, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)

        self._maybe_visualize(
            end_points=end_points,
            token_sel_idx=token_sel_idx,
            uncert_map=uncert_map,
            top_view_inds=top_view_inds,
            top1_score=top1_score,
            top1_margin=top1_margin,
            normal_align=normal_align,
            top_attn_app=top_attn_app,
            top_attn_geo=top_attn_geo,
            top_attn_geo_eff=top_attn_geo_eff,
            top_attn_entropy=top_attn_entropy,
            mean_geo_attn=mean_geo_attn,
            anchor_geo_curve=anchor_geo_curve,
        )

        # outputs
        end_points['view_score'] = view_score                       # (B,M,V)
        end_points['grasp_top_view_inds'] = top_view_inds          # (B,M)
        end_points['grasp_top_view_xyz'] = top_view_xyz            # (B,M,3)
        end_points['grasp_top_view_rot'] = grasp_top_view_rot      # (B,M,3,3)

        end_points['view_geom_uncert'] = uncert_bm1.squeeze(-1)    # (B,M)

        end_points['view_attn_top_geo_eff'] = top_attn_geo_eff
        end_points['view_attn_top_geo'] = top_attn_geo             # (B,M)
        end_points['view_attn_top_app'] = top_attn_app             # (B,M)
        end_points['view_attn_top_entropy'] = top_attn_entropy     # (B,M)
        end_points['view_attn_mean_geo'] = mean_geo_attn           # (B,M)

        self._vis_iter += 1
        return end_points, res_feat


class ProjectedViewGrouping(nn.Module):
    """
    Projected View Grouping with built-in visualization.

    This is a deterministic 2Dized replacement for point-cloud cylinder grouping.
    Given selected seed pixels and a selected view rotation, it:
      1) gathers the predicted seed depth,
      2) backprojects each seed to a camera-frame 3D point,
      3) lays a small view-aligned tube around the seed in the selected view frame,
      4) projects tube samples back to the dense 2D feature map,
      5) aggregates sampled 2D features plus a simple depth residual.

    Output:
      grouped feature of shape (B, 256, M), matching Grasp_Head_Local_Interaction.

    Visualization files, when vis_dir is not None:
      - *_overlay_signed.png: projected samples over RGB/depth, colored by raw dz.
      - *_overlay_valid.png: valid/invalid projected samples over RGB/depth.
      - *_grid.png: 4 x 9 layout heatmaps for valid, raw dz, tanh dz, sample z.
      - *_pool_winner.png: max-pool winner histogram in the same 4 x 9 layout.
      - *.npz: raw uv/residual/valid/sample xyz dump for quick offline inspection.
      - *.ply: optional 3D seed/sample cloud if save_ply=True.
    """
    def __init__(
        self,
        seed_feature_dim: int,
        out_dim: int = 256,
        num_axial: int = 4,
        num_radial: int = 8,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        residual_tau: float = 0.05,
        offset_embed_dim: int = 32,
        detach_depth: bool = True,
        use_local_attention: bool = True,
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        vis_num_seeds: int = 4,
        vis_seed_mode: str = "valid_first",
        vis_dpi: int = 160,
        save_npz: bool = True,
        save_ply: bool = False,
    ):
        super().__init__()
        self.seed_feature_dim = int(seed_feature_dim)
        self.out_dim = int(out_dim)
        self.num_axial = int(num_axial)
        self.num_radial = int(num_radial)
        self.radius = float(radius)
        self.hmin = float(hmin)
        self.hmax = float(hmax)
        self.residual_tau = float(residual_tau)
        self.detach_depth = bool(detach_depth)
        self.use_local_attention = bool(use_local_attention)

        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.vis_num_seeds = int(vis_num_seeds)
        self.vis_seed_mode = str(vis_seed_mode)
        self.vis_dpi = int(vis_dpi)
        self.save_npz = bool(save_npz)
        self.save_ply = bool(save_ply)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        offsets = self._make_view_tube_offsets(
            num_axial=self.num_axial,
            num_radial=self.num_radial,
            radius=self.radius,
            hmin=self.hmin,
            hmax=self.hmax,
        )
        self.register_buffer('canonical_offsets', offsets, persistent=False)  # (S,3), meters

        offset_scale = max(abs(self.hmin), abs(self.hmax), self.radius, 1e-6)
        self.register_buffer('offset_scale', torch.tensor(float(offset_scale)), persistent=False)

        self.offset_mlp = nn.Sequential(
            nn.Linear(3, offset_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(offset_embed_dim, offset_embed_dim),
            nn.ReLU(inplace=True),
        )

        sample_in_dim = self.seed_feature_dim + offset_embed_dim + 2  # sampled feat + residual + valid
        self.sample_mlp = nn.Sequential(
            nn.Linear(sample_in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

        if self.use_local_attention:
            self.local_interaction_module = AttentionModule(dim=out_dim, n_head=1, msa_dropout=0.05)
        else:
            self.local_interaction_module = None

        self.seed_proj = nn.Sequential(
            nn.Linear(self.seed_feature_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _make_view_tube_offsets(
        num_axial: int,
        num_radial: int,
        radius: float,
        hmin: float,
        hmax: float,
    ) -> torch.Tensor:
        """Build a simple view-aligned tube: center line + one radial ring per axial slice."""
        xs = torch.linspace(float(hmin), float(hmax), int(num_axial), dtype=torch.float32)
        theta = torch.linspace(0.0, 2.0 * math.pi, int(num_radial) + 1, dtype=torch.float32)[:-1]
        ring_y = float(radius) * torch.cos(theta)
        ring_z = float(radius) * torch.sin(theta)

        offsets = []
        for x in xs:
            offsets.append(torch.tensor([x.item(), 0.0, 0.0], dtype=torch.float32))
            for y, z in zip(ring_y, ring_z):
                offsets.append(torch.tensor([x.item(), y.item(), z.item()], dtype=torch.float32))
        return torch.stack(offsets, dim=0).contiguous()  # (S,3)

    @staticmethod
    def _idx_to_uv(idx_bm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()
        return torch.stack([u, v], dim=-1)  # (B,M,2), order: u,x then v,y

    @staticmethod
    def _gather_depth(depth_b1hw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth_b1hw.shape
        flat = depth_b1hw[:, 0].reshape(B, H * W)
        z = torch.gather(flat, 1, idx_bm).unsqueeze(-1)
        return z  # (B,M,1)

    @staticmethod
    def _backproject_uvz(uv_bm2: torch.Tensor, z_bm1: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
        fx = K_b33[:, 0, 0].unsqueeze(1)
        fy = K_b33[:, 1, 1].unsqueeze(1)
        cx = K_b33[:, 0, 2].unsqueeze(1)
        cy = K_b33[:, 1, 2].unsqueeze(1)
        u = uv_bm2[..., 0]
        v = uv_bm2[..., 1]
        z = z_bm1.squeeze(-1)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)  # (B,M,3)

    @staticmethod
    def _project_xyz(xyz_bms3: torch.Tensor, K_b33: torch.Tensor, eps: float = 1e-6):
        fx = K_b33[:, 0, 0].view(-1, 1, 1)
        fy = K_b33[:, 1, 1].view(-1, 1, 1)
        cx = K_b33[:, 0, 2].view(-1, 1, 1)
        cy = K_b33[:, 1, 2].view(-1, 1, 1)
        x = xyz_bms3[..., 0]
        y = xyz_bms3[..., 1]
        z = xyz_bms3[..., 2].clamp_min(eps)
        u = fx * x / z + cx
        v = fy * y / z + cy
        uv = torch.stack([u, v], dim=-1)
        return uv, z  # (B,M,S,2), (B,M,S)

    @staticmethod
    def _grid_sample_bms(feat_bchw: torch.Tensor, uv_bms2: torch.Tensor, src_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Sample B,C,Hf,Wf feature map at uv coordinates defined in the source
        image/depth coordinate system src_hw. Returns (B,M,S,C).
        """
        B, C, Hf, Wf = feat_bchw.shape
        Hsrc, Wsrc = src_hw
        M, S = uv_bms2.shape[1], uv_bms2.shape[2]

        if Wsrc > 1:
            u_feat = uv_bms2[..., 0] * (float(Wf - 1) / float(Wsrc - 1))
        else:
            u_feat = uv_bms2[..., 0]
        if Hsrc > 1:
            v_feat = uv_bms2[..., 1] * (float(Hf - 1) / float(Hsrc - 1))
        else:
            v_feat = uv_bms2[..., 1]

        if Wf > 1:
            gx = 2.0 * u_feat / float(Wf - 1) - 1.0
        else:
            gx = torch.zeros_like(u_feat)
        if Hf > 1:
            gy = 2.0 * v_feat / float(Hf - 1) - 1.0
        else:
            gy = torch.zeros_like(v_feat)

        grid = torch.stack([gx, gy], dim=-1).view(B, M * S, 1, 2)
        sampled = F.grid_sample(
            feat_bchw,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )  # (B,C,M*S,1)
        sampled = sampled.squeeze(-1).view(B, C, M, S).permute(0, 2, 3, 1).contiguous()
        return sampled  # (B,M,S,C)

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
        return x

    @staticmethod
    def _normalize_img_chw(img_chw: torch.Tensor) -> np.ndarray:
        x = img_chw.detach().float().cpu()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            arr = x[0].numpy()
            lo, hi = np.nanpercentile(arr, [1, 99])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return np.repeat(arr[..., None], 3, axis=-1)
        x = x[:3]
        x = x - x.amin(dim=(1, 2), keepdim=True)
        x = x / (x.amax(dim=(1, 2), keepdim=True) + 1e-6)
        return x.permute(1, 2, 0).numpy()

    def _background_np(self, end_points: Optional[dict], depth_map: torch.Tensor, b: int) -> Tuple[np.ndarray, str]:
        if end_points is not None and 'img' in end_points and torch.is_tensor(end_points['img']):
            img = end_points['img']
            if img.dim() == 4 and img.size(0) > b:
                return self._normalize_img_chw(img[b]), 'rgb'
        return self._normalize_img_chw(depth_map[b, 0]), 'depth'

    def _reshape_layout(self, values_1d: np.ndarray) -> np.ndarray:
        return values_1d.reshape(self.num_axial, self.num_radial + 1)

    def _choose_vis_seed_indices(
        self,
        end_points: Optional[dict],
        valid: torch.Tensor,          # (B,M,S)
        raw_residual: torch.Tensor,   # (B,M,S)
        b: int,
    ) -> List[int]:
        M = valid.shape[1]
        device = valid.device
        candidates = None

        if end_points is not None and 'batch_valid_mask' in end_points:
            mask = end_points['batch_valid_mask']
            if torch.is_tensor(mask) and mask.dim() == 2 and mask.size(0) > b and mask.size(1) == M:
                idx = torch.nonzero(mask[b].bool(), as_tuple=False).flatten()
                if idx.numel() > 0:
                    candidates = idx

        if candidates is None:
            candidates = torch.arange(M, device=device)

        if self.vis_seed_mode == 'high_residual':
            score = raw_residual[b].detach().abs().mean(dim=-1)
            order = torch.argsort(score[candidates], descending=True)
            candidates = candidates[order]
        elif self.vis_seed_mode == 'low_valid':
            score = valid[b].detach().float().mean(dim=-1)
            order = torch.argsort(score[candidates], descending=False)
            candidates = candidates[order]
        elif self.vis_seed_mode == 'random':
            perm = torch.randperm(candidates.numel(), device=device)
            candidates = candidates[perm]

        return candidates[:max(self.vis_num_seeds, 0)].detach().cpu().tolist()

    def _save_overlay(
        self,
        bg: np.ndarray,
        seed_uv: np.ndarray,        # (2,)
        sample_uv: np.ndarray,      # (S,2)
        values: np.ndarray,         # (S,)
        valid: np.ndarray,          # (S,)
        out_path: str,
        title: str,
        mode: str = 'signed_residual',
    ):
        plt.figure(figsize=(7, 7))
        plt.imshow(bg)
        plt.scatter([seed_uv[0]], [seed_uv[1]], s=90, marker='*', c='yellow', edgecolors='black', linewidths=0.7)

        if mode == 'valid':
            vvals = valid.astype(np.float32)
            plt.scatter(sample_uv[:, 0], sample_uv[:, 1], c=vvals, s=42, cmap='RdYlGn', vmin=0.0, vmax=1.0,
                        edgecolors='black', linewidths=0.35)
            cb = plt.colorbar(fraction=0.046, pad=0.04)
            cb.set_label('valid')
        else:
            rmax = float(np.nanpercentile(np.abs(values[valid.astype(bool)]), 95)) if valid.any() else float(np.nanmax(np.abs(values)) + 1e-6)
            rmax = max(rmax, self.residual_tau, 1e-6)
            plt.scatter(sample_uv[valid.astype(bool), 0], sample_uv[valid.astype(bool), 1],
                        c=values[valid.astype(bool)], s=42, cmap='coolwarm', vmin=-rmax, vmax=rmax,
                        edgecolors='black', linewidths=0.35)
            if (~valid.astype(bool)).any():
                plt.scatter(sample_uv[~valid.astype(bool), 0], sample_uv[~valid.astype(bool), 1],
                            s=45, marker='x', c='black', linewidths=1.0)
            cb = plt.colorbar(fraction=0.046, pad=0.04)
            cb.set_label('raw dz = D(uv) - z_sample')

        plt.xlim(0, bg.shape[1] - 1)
        plt.ylim(bg.shape[0] - 1, 0)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout(pad=0.15)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_grid_summary(
        self,
        valid_1d: np.ndarray,
        raw_residual_1d: np.ndarray,
        residual_tanh_1d: np.ndarray,
        sample_z_1d: np.ndarray,
        out_path: str,
        title: str,
    ):
        valid_grid = self._reshape_layout(valid_1d.astype(np.float32))
        raw_grid = self._reshape_layout(raw_residual_1d)
        tanh_grid = self._reshape_layout(residual_tanh_1d)
        z_grid = self._reshape_layout(sample_z_1d)

        fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
        im0 = axes[0].imshow(valid_grid, vmin=0.0, vmax=1.0, cmap='gray')
        axes[0].set_title('valid')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        rmax = max(float(np.nanpercentile(np.abs(raw_residual_1d), 95)), self.residual_tau, 1e-6)
        im1 = axes[1].imshow(raw_grid, vmin=-rmax, vmax=rmax, cmap='coolwarm')
        axes[1].set_title('raw dz')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(tanh_grid, vmin=-1.0, vmax=1.0, cmap='coolwarm')
        axes[2].set_title('tanh(dz/tau)')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(z_grid, cmap='viridis')
        axes[3].set_title('sample z')
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel('0=center, 1..radial')
            ax.set_ylabel('axial slice')
            ax.set_xticks(range(self.num_radial + 1))
            ax.set_yticks(range(self.num_axial))

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.vis_dpi)
        plt.close(fig)

    def _save_pool_winner_grid(
        self,
        winner_idx_1d: np.ndarray,   # (out_dim,)
        valid_1d: np.ndarray,        # (S,)
        out_path: str,
        title: str,
    ):
        S = self.canonical_offsets.shape[0]
        hist = np.bincount(winner_idx_1d.astype(np.int64), minlength=S).astype(np.float32)
        hist = hist / max(float(hist.sum()), 1.0)
        hist_grid = self._reshape_layout(hist)
        valid_grid = self._reshape_layout(valid_1d.astype(np.float32))

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        im0 = axes[0].imshow(hist_grid, cmap='magma')
        axes[0].set_title('max-pool winner freq')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(valid_grid, vmin=0.0, vmax=1.0, cmap='gray')
        axes[1].set_title('valid mask')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.set_xlabel('0=center, 1..radial')
            ax.set_ylabel('axial slice')
            ax.set_xticks(range(self.num_radial + 1))
            ax.set_yticks(range(self.num_axial))
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.vis_dpi)
        plt.close(fig)

    @staticmethod
    def _write_ply_ascii(points: np.ndarray, colors: np.ndarray, out_path: str):
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        with open(out_path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {points.shape[0]}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write('end_header\n')
            for p, c in zip(points, colors):
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n')

    @torch.no_grad()
    def _maybe_visualize(
        self,
        end_points: Optional[dict],
        depth_map: torch.Tensor,       # (B,1,H,W)
        seed_uv: torch.Tensor,         # (B,M,2)
        seed_xyz: torch.Tensor,        # (B,M,3)
        sample_uv: torch.Tensor,       # (B,M,S,2)
        sample_xyz: torch.Tensor,      # (B,M,S,3)
        sample_z: torch.Tensor,        # (B,M,S)
        valid: torch.Tensor,           # (B,M,S)
        raw_residual: torch.Tensor,    # (B,M,S)
        residual_tanh: torch.Tensor,   # (B,M,S)
        winner_idx: torch.Tensor,      # (B,M,out_dim)
        token_sel_idx: torch.Tensor,   # (B,M)
    ):
        if self.vis_dir is None:
            self._vis_iter += 1
            return
        if self.vis_every <= 0 or (self._vis_iter % self.vis_every) != 0:
            self._vis_iter += 1
            return

        try:
            b = 0
            bg, bg_name = self._background_np(end_points, depth_map, b)
            seed_ids = self._choose_vis_seed_indices(end_points, valid, raw_residual, b)
            prefix = f'pvg_it{self._vis_iter:06d}_b{b}'

            top_view_inds = None
            if end_points is not None and 'grasp_top_view_inds' in end_points and torch.is_tensor(end_points['grasp_top_view_inds']):
                top_view_inds = end_points['grasp_top_view_inds']

            for local_rank, m in enumerate(seed_ids):
                seed_uv_np = self._to_numpy(seed_uv[b, m])
                sample_uv_np = self._to_numpy(sample_uv[b, m])
                valid_np = self._to_numpy(valid[b, m]).astype(bool)
                raw_np = self._to_numpy(raw_residual[b, m])
                tanh_np = self._to_numpy(residual_tanh[b, m])
                sample_z_np = self._to_numpy(sample_z[b, m])
                winner_np = winner_idx[b, m].detach().cpu().numpy()

                valid_ratio = float(valid_np.mean())
                raw_abs = float(np.mean(np.abs(raw_np[valid_np]))) if valid_np.any() else float(np.mean(np.abs(raw_np)))
                view_text = ''
                if top_view_inds is not None and top_view_inds.dim() == 2 and top_view_inds.size(1) > m:
                    view_text = f', pred_view={int(top_view_inds[b, m].detach().cpu())}'
                title = f'seed m={m}, pix=({seed_uv_np[0]:.1f},{seed_uv_np[1]:.1f}), valid={valid_ratio:.2f}, |dz|={raw_abs:.4f}{view_text}'
                stem = f'{prefix}_seed{local_rank:02d}_m{m:04d}'

                self._save_overlay(
                    bg=bg,
                    seed_uv=seed_uv_np,
                    sample_uv=sample_uv_np,
                    values=raw_np,
                    valid=valid_np,
                    out_path=os.path.join(self.vis_dir, f'{stem}_overlay_signed.png'),
                    title=f'{title} [{bg_name}]',
                    mode='signed_residual',
                )
                self._save_overlay(
                    bg=bg,
                    seed_uv=seed_uv_np,
                    sample_uv=sample_uv_np,
                    values=raw_np,
                    valid=valid_np,
                    out_path=os.path.join(self.vis_dir, f'{stem}_overlay_valid.png'),
                    title=f'{title} [{bg_name}]',
                    mode='valid',
                )
                self._save_grid_summary(
                    valid_1d=valid_np,
                    raw_residual_1d=raw_np,
                    residual_tanh_1d=tanh_np,
                    sample_z_1d=sample_z_np,
                    out_path=os.path.join(self.vis_dir, f'{stem}_grid.png'),
                    title=title,
                )
                self._save_pool_winner_grid(
                    winner_idx_1d=winner_np,
                    valid_1d=valid_np,
                    out_path=os.path.join(self.vis_dir, f'{stem}_pool_winner.png'),
                    title=title,
                )

                if self.save_npz:
                    np.savez_compressed(
                        os.path.join(self.vis_dir, f'{stem}.npz'),
                        seed_uv=seed_uv_np,
                        seed_xyz=self._to_numpy(seed_xyz[b, m]),
                        sample_uv=sample_uv_np,
                        sample_xyz=self._to_numpy(sample_xyz[b, m]),
                        sample_z=sample_z_np,
                        valid=valid_np,
                        raw_residual=raw_np,
                        residual_tanh=tanh_np,
                        winner_idx=winner_np,
                        token_idx=int(token_sel_idx[b, m].detach().cpu()),
                    )

                if self.save_ply:
                    pts = np.concatenate([
                        self._to_numpy(seed_xyz[b, m]).reshape(1, 3),
                        self._to_numpy(sample_xyz[b, m]),
                    ], axis=0)
                    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
                    colors[0] = np.array([255, 255, 0], dtype=np.uint8)
                    sample_colors = np.zeros((valid_np.shape[0], 3), dtype=np.uint8)
                    sample_colors[valid_np & (raw_np >= 0.0)] = np.array([220, 60, 60], dtype=np.uint8)
                    sample_colors[valid_np & (raw_np < 0.0)] = np.array([60, 90, 220], dtype=np.uint8)
                    sample_colors[~valid_np] = np.array([20, 20, 20], dtype=np.uint8)
                    colors[1:] = sample_colors
                    self._write_ply_ascii(pts, colors, os.path.join(self.vis_dir, f'{stem}.ply'))
        except Exception as e:
            print(f'[ProjectedViewGrouping vis] failed at iter {self._vis_iter}: {repr(e)}')
        finally:
            self._vis_iter += 1

    @staticmethod
    def _safe_p90(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return torch.quantile(x, 0.90)

    def forward(
        self,
        seed_features: torch.Tensor,       # (B,C,M)
        token_sel_idx: torch.Tensor,       # (B,M), flattened index in depth/image map
        top_view_rot: torch.Tensor,        # (B,M,3,3)
        feat_map: torch.Tensor,            # (B,C,Hf,Wf), dense DPT feature map
        depth_map: torch.Tensor,           # (B,1,H,W), predicted depth map
        K: torch.Tensor,                   # (B,3,3), intrinsics for depth/image resolution
        end_points: Optional[dict] = None,
    ) -> torch.Tensor:
        B, C, M = seed_features.shape
        _, _, Hd, Wd = depth_map.shape
        device = seed_features.device
        dtype = seed_features.dtype

        if self.detach_depth:
            depth_for_geom = depth_map.detach()
        else:
            depth_for_geom = depth_map
        depth_for_geom = torch.nan_to_num(depth_for_geom, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

        # 1) seed uv and seed 3D position from predicted depth
        seed_uv = self._idx_to_uv(token_sel_idx, Hd, Wd).to(device=device, dtype=depth_for_geom.dtype)  # (B,M,2)
        seed_z = self._gather_depth(depth_for_geom, token_sel_idx).clamp_min(1e-6)                      # (B,M,1)
        seed_xyz = self._backproject_uvz(seed_uv, seed_z, K.to(depth_for_geom.dtype))                   # (B,M,3)

        # 2) view-aligned local samples in camera frame
        offsets = self.canonical_offsets.to(device=device, dtype=top_view_rot.dtype)                    # (S,3)
        S = offsets.shape[0]
        local_xyz = torch.einsum('bmij,sj->bmsi', top_view_rot.to(offsets.dtype), offsets)              # (B,M,S,3)
        sample_xyz = seed_xyz.to(local_xyz.dtype).unsqueeze(2) + local_xyz                              # (B,M,S,3)

        # 3) project samples back to the 2D feature/depth maps
        sample_uv, sample_z = self._project_xyz(sample_xyz, K.to(sample_xyz.dtype))                     # (B,M,S,2), (B,M,S)

        valid = (
            (sample_z > 1e-6) &
            (sample_uv[..., 0] >= 0.0) & (sample_uv[..., 0] <= float(Wd - 1)) &
            (sample_uv[..., 1] >= 0.0) & (sample_uv[..., 1] <= float(Hd - 1))
        )                                                                                               # (B,M,S)

        # 4) bilinear sampling from image feature and predicted depth
        sampled_feat = self._grid_sample_bms(feat_map, sample_uv.to(feat_map.dtype), src_hw=(Hd, Wd))   # (B,M,S,C)
        sampled_depth = self._grid_sample_bms(depth_for_geom, sample_uv.to(depth_for_geom.dtype), src_hw=(Hd, Wd))  # (B,M,S,1)

        # 5) simple local 3D evidence: depth residual D(projected_uv) - z_sample
        raw_residual = sampled_depth.squeeze(-1).to(sample_z.dtype) - sample_z                          # (B,M,S)
        residual_tanh = torch.tanh(raw_residual / max(self.residual_tau, 1e-6))                         # (B,M,S)
        valid_f = valid.to(dtype).unsqueeze(-1)                                                         # (B,M,S,1)

        # Canonical offset embedding, shared across batch/seeds.
        offset_norm = (self.canonical_offsets / self.offset_scale.clamp_min(1e-6)).to(device=device, dtype=dtype)
        offset_emb = self.offset_mlp(offset_norm).view(1, 1, S, -1).expand(B, M, S, -1)                 # (B,M,S,E)

        # Sample token: local image feature + canonical position + soft occupancy cue.
        h = torch.cat([
            sampled_feat.to(dtype),
            offset_emb,
            residual_tanh.unsqueeze(-1).to(dtype),
            valid_f,
        ], dim=-1)                                                                                      # (B,M,S,C+E+2)

        sample_tokens = self.sample_mlp(h).view(B * M, S, self.out_dim)                                 # (BM,S,256)
        if self.local_interaction_module is not None:
            sample_tokens = self.local_interaction_module(sample_tokens, sample_tokens, sample_tokens, mask=None)

        sample_tokens = sample_tokens.view(B, M, S, self.out_dim)
        valid_any = valid.any(dim=-1, keepdim=True)                                                     # (B,M,1)
        masked_tokens = sample_tokens.masked_fill(~valid.unsqueeze(-1), -1e4)
        pooled, winner_idx = masked_tokens.max(dim=2)                                                   # (B,M,256), (B,M,256)
        pooled = torch.where(valid_any, pooled, torch.zeros_like(pooled))

        seed_ctx = self.seed_proj(seed_features.transpose(1, 2).contiguous())                           # (B,M,256)
        grouped = self.fuse_mlp(torch.cat([seed_ctx, pooled], dim=-1)) + seed_ctx                       # (B,M,256)
        grouped = grouped.transpose(1, 2).contiguous()                                                  # (B,256,M)

        if end_points is not None:
            with torch.no_grad():
                valid_count = valid.float().sum().clamp_min(1.0)
                raw_abs_valid = raw_residual.detach().abs()[valid]
                end_points['D: PVG valid ratio'] = valid.float().mean().reshape(())
                end_points['D: PVG |depth residual|'] = residual_tanh.detach().abs().mean().reshape(())
                end_points['D: PVG samples'] = torch.tensor(float(S), device=device)
                if raw_abs_valid.numel() > 0:
                    end_points['D: PVG raw |dz| mean'] = raw_abs_valid.mean().reshape(())
                    end_points['D: PVG raw |dz| median'] = raw_abs_valid.median().reshape(())
                    end_points['D: PVG raw |dz| p90'] = self._safe_p90(raw_abs_valid).reshape(())
                else:
                    zero = torch.zeros((), device=device, dtype=dtype)
                    end_points['D: PVG raw |dz| mean'] = zero
                    end_points['D: PVG raw |dz| median'] = zero
                    end_points['D: PVG raw |dz| p90'] = zero

                tau = max(self.residual_tau, 1e-6)
                end_points['D: PVG surface ratio'] = ((raw_residual.detach().abs() < tau) & valid).float().sum() / valid_count
                end_points['D: PVG front ratio'] = ((raw_residual.detach() > tau) & valid).float().sum() / valid_count
                end_points['D: PVG behind ratio'] = ((raw_residual.detach() < -tau) & valid).float().sum() / valid_count

        self._maybe_visualize(
            end_points=end_points,
            depth_map=depth_for_geom,
            seed_uv=seed_uv,
            seed_xyz=seed_xyz,
            sample_uv=sample_uv,
            sample_xyz=sample_xyz,
            sample_z=sample_z,
            valid=valid,
            raw_residual=raw_residual,
            residual_tanh=residual_tanh,
            winner_idx=winner_idx,
            token_sel_idx=token_sel_idx,
        )

        return grouped



class ProjectedSurfaceCylinderGrouping(nn.Module):
    """
    Projected Surface Cylinder Grouping (PSCG).

    This is a 2.5D / image-centric counterpart of EconomicGrasp's 3D cylinder
    grouping.  Unlike ProjectedViewGrouping, it does not place artificial 3D
    volume probes and then ask a depth residual whether they lie on a surface.
    Instead, it:

      1) takes a local 2D candidate window around each selected seed pixel;
      2) samples predicted depth at these candidate pixels;
      3) backprojects candidate pixels into visible surface points;
      4) transforms candidate surface points into the selected-view local frame;
      5) softly ranks/selects points by view-aligned cylinder membership;
      6) aggregates their dense 2D features + normalized local coordinates.

    Inputs are drop-in compatible with the previous PVG module:
        seed_features : (B,C,M)
        token_sel_idx : (B,M), flattened seed pixel indices in depth map resolution
        top_view_rot  : (B,M,3,3)
        feat_map      : (B,C,Hf,Wf)
        depth_map     : (B,1,H,W)
        K             : (B,3,3)

    Output:
        grouped feature: (B, out_dim, M), usually (B,256,M).

    Visualization, when vis_dir is provided:
        *_overlay_topk.png       top selected surface candidates over RGB/depth
        *_overlay_candidates.png local candidate window and hard cylinder candidates
        *_candidate_grid.png     soft cylinder weight / hard mask / rho / h / valid
        *_topk_summary.png       top-K weight, rho, h, valid sorted by soft weight
        *_pool_winner.png        max-pool winner histogram over selected surface points
        *.npz                    raw arrays for offline analysis
        *.ply                    optional seed + top-K surface point cloud
    """
    def __init__(
        self,
        seed_feature_dim: int,
        out_dim: int = 256,
        nsample: int = 32,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        grid_size: int = 21,
        dynamic_window: bool = True,
        window_radius_px: float = 32.0,
        window_radius_px_min: float = 8.0,
        window_radius_px_max: float = 48.0,
        window_radius_factor: float = 1.5,
        h_soft_tau: Optional[float] = None,
        min_group_weight: float = 1e-4,
        coord_embed_dim: int = 32,
        detach_depth: bool = True,
        use_local_attention: bool = True,
        residual_fusion: bool = True,
        init_group_scale: float = 0.1,
        debug_print_every: int = 50,
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        vis_num_seeds: int = 4,
        vis_seed_mode: str = "valid_first",
        vis_dpi: int = 160,
        save_npz: bool = True,
        save_ply: bool = False,
    ):
        super().__init__()
        self.seed_feature_dim = int(seed_feature_dim)
        self.out_dim = int(out_dim)
        self.nsample = int(nsample)
        self.radius = float(radius)
        self.hmin = float(hmin)
        self.hmax = float(hmax)
        self.grid_size = int(grid_size)
        assert self.grid_size >= 3 and self.grid_size % 2 == 1, "grid_size should be an odd integer >= 3"
        self.dynamic_window = bool(dynamic_window)
        self.window_radius_px = float(window_radius_px)
        self.window_radius_px_min = float(window_radius_px_min)
        self.window_radius_px_max = float(window_radius_px_max)
        self.window_radius_factor = float(window_radius_factor)
        self.h_soft_tau = float(h_soft_tau) if h_soft_tau is not None else max((self.hmax - self.hmin) * 0.15, 0.005)
        self.min_group_weight = float(min_group_weight)
        self.detach_depth = bool(detach_depth)
        self.use_local_attention = bool(use_local_attention)
        self.residual_fusion = bool(residual_fusion)
        self.debug_print_every = int(debug_print_every)

        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.vis_num_seeds = int(vis_num_seeds)
        self.vis_seed_mode = str(vis_seed_mode)
        self.vis_dpi = int(vis_dpi)
        self.save_npz = bool(save_npz)
        self.save_ply = bool(save_ply)
        self._iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        # Unit square candidate grid in image space, centered at seed pixel.
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.grid_size, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, self.grid_size, dtype=torch.float32),
            indexing="ij",
        )
        base_grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).contiguous()  # (L,2), u/v order
        self.register_buffer("base_grid_uv", base_grid, persistent=False)

        # Local coordinate embedding.  Coordinates are in selected-view frame.
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, coord_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_embed_dim, coord_embed_dim),
            nn.ReLU(inplace=True),
        )

        # token: sampled image feature + local-coordinate embedding + [soft_weight, hard_mask, valid]
        sample_in_dim = self.seed_feature_dim + coord_embed_dim + 3
        self.sample_mlp = nn.Sequential(
            nn.Linear(sample_in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

        if self.use_local_attention:
            self.local_interaction_module = AttentionModule(dim=out_dim, n_head=1, msa_dropout=0.05)
        else:
            self.local_interaction_module = None

        self.seed_proj = nn.Sequential(
            nn.Linear(self.seed_feature_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.group_delta = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.group_scale = nn.Parameter(torch.tensor(float(init_group_scale), dtype=torch.float32))

    @staticmethod
    def _idx_to_uv(idx_bm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()
        return torch.stack([u, v], dim=-1)  # (B,M,2)

    @staticmethod
    def _gather_depth(depth_b1hw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth_b1hw.shape
        flat = depth_b1hw[:, 0].reshape(B, H * W)
        return torch.gather(flat, 1, idx_bm).unsqueeze(-1)  # (B,M,1)

    @staticmethod
    def _backproject_uvz_bm(uv_bm2: torch.Tensor, z_bm1: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
        fx = K_b33[:, 0, 0].unsqueeze(1)
        fy = K_b33[:, 1, 1].unsqueeze(1)
        cx = K_b33[:, 0, 2].unsqueeze(1)
        cy = K_b33[:, 1, 2].unsqueeze(1)
        u = uv_bm2[..., 0]
        v = uv_bm2[..., 1]
        z = z_bm1.squeeze(-1)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)  # (B,M,3)

    @staticmethod
    def _backproject_uvz_bml(uv_bml2: torch.Tensor, z_bml1: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
        fx = K_b33[:, 0, 0].view(-1, 1, 1)
        fy = K_b33[:, 1, 1].view(-1, 1, 1)
        cx = K_b33[:, 0, 2].view(-1, 1, 1)
        cy = K_b33[:, 1, 2].view(-1, 1, 1)
        u = uv_bml2[..., 0]
        v = uv_bml2[..., 1]
        z = z_bml1.squeeze(-1)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)  # (B,M,L,3)

    @staticmethod
    def _grid_sample_bms(feat_bchw: torch.Tensor, uv_bms2: torch.Tensor, src_hw: Tuple[int, int]) -> torch.Tensor:
        """Sample B,C,Hf,Wf feature map at uv defined in source image/depth coordinates. Returns (B,M,S,C)."""
        B, C, Hf, Wf = feat_bchw.shape
        Hsrc, Wsrc = src_hw
        M, S = uv_bms2.shape[1], uv_bms2.shape[2]

        if Wsrc > 1:
            u_feat = uv_bms2[..., 0] * (float(Wf - 1) / float(Wsrc - 1))
        else:
            u_feat = uv_bms2[..., 0]
        if Hsrc > 1:
            v_feat = uv_bms2[..., 1] * (float(Hf - 1) / float(Hsrc - 1))
        else:
            v_feat = uv_bms2[..., 1]

        gx = 2.0 * u_feat / float(max(Wf - 1, 1)) - 1.0 if Wf > 1 else torch.zeros_like(u_feat)
        gy = 2.0 * v_feat / float(max(Hf - 1, 1)) - 1.0 if Hf > 1 else torch.zeros_like(v_feat)
        grid = torch.stack([gx, gy], dim=-1).view(B, M * S, 1, 2)
        sampled = F.grid_sample(feat_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled.squeeze(-1).view(B, C, M, S).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _gather_bml(x: torch.Tensor, idx_bmk: torch.Tensor) -> torch.Tensor:
        """Gather along dim=2 for tensors shaped (B,M,L,...) using idx (B,M,K)."""
        if x.dim() == 3:
            return torch.gather(x, 2, idx_bmk)
        idx = idx_bmk.unsqueeze(-1).expand(*idx_bmk.shape, x.shape[-1])
        return torch.gather(x, 2, idx)

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().float().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _normalize_img_chw(img_chw: torch.Tensor) -> np.ndarray:
        x = img_chw.detach().float().cpu()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            arr = x[0].numpy()
            lo, hi = np.nanpercentile(arr, [1, 99])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return np.repeat(arr[..., None], 3, axis=-1)
        x = x[:3]
        x = x - x.amin(dim=(1, 2), keepdim=True)
        x = x / (x.amax(dim=(1, 2), keepdim=True) + 1e-6)
        return x.permute(1, 2, 0).numpy()

    def _background_np(self, end_points: Optional[dict], depth_map: torch.Tensor, b: int) -> Tuple[np.ndarray, str]:
        if end_points is not None and "img" in end_points and torch.is_tensor(end_points["img"]):
            img = end_points["img"]
            if img.dim() == 4 and img.size(0) > b:
                return self._normalize_img_chw(img[b]), "rgb"
        return self._normalize_img_chw(depth_map[b, 0]), "depth"

    def _choose_vis_seed_indices(
        self,
        end_points: Optional[dict],
        top_valid: torch.Tensor,        # (B,M,K)
        top_weight: torch.Tensor,       # (B,M,K)
        hard_ratio: torch.Tensor,       # (B,M)
        b: int,
    ) -> List[int]:
        M = top_valid.shape[1]
        device = top_valid.device
        candidates = None
        if end_points is not None and "batch_valid_mask" in end_points:
            mask = end_points["batch_valid_mask"]
            if torch.is_tensor(mask) and mask.dim() == 2 and mask.size(0) > b and mask.size(1) == M:
                idx = torch.nonzero(mask[b].bool(), as_tuple=False).flatten()
                if idx.numel() > 0:
                    candidates = idx
        if candidates is None:
            candidates = torch.arange(M, device=device)

        if self.vis_seed_mode == "low_valid":
            score = top_valid[b].float().mean(dim=-1)
            candidates = candidates[torch.argsort(score[candidates], descending=False)]
        elif self.vis_seed_mode == "low_cyl":
            candidates = candidates[torch.argsort(hard_ratio[b, candidates], descending=False)]
        elif self.vis_seed_mode == "high_weight":
            score = top_weight[b].mean(dim=-1)
            candidates = candidates[torch.argsort(score[candidates], descending=True)]
        elif self.vis_seed_mode == "random":
            candidates = candidates[torch.randperm(candidates.numel(), device=device)]
        return candidates[:max(self.vis_num_seeds, 0)].detach().cpu().tolist()

    def _save_overlay_candidates(
        self,
        bg: np.ndarray,
        seed_uv: np.ndarray,
        cand_uv: np.ndarray,
        hard_mask: np.ndarray,
        pixel_valid: np.ndarray,
        top_uv: np.ndarray,
        top_weight: np.ndarray,
        top_valid: np.ndarray,
        out_path: str,
        title: str,
    ):
        plt.figure(figsize=(7, 7))
        plt.imshow(bg)
        valid_c = pixel_valid.astype(bool)
        plt.scatter(cand_uv[valid_c, 0], cand_uv[valid_c, 1], s=8, c="lightgray", alpha=0.45, linewidths=0)
        if (~valid_c).any():
            plt.scatter(cand_uv[~valid_c, 0], cand_uv[~valid_c, 1], s=8, c="black", alpha=0.25, marker="x")
        hm = hard_mask.astype(bool) & valid_c
        if hm.any():
            plt.scatter(cand_uv[hm, 0], cand_uv[hm, 1], s=14, c="lime", alpha=0.75, linewidths=0)
        tv = top_valid.astype(bool)
        if tv.any():
            plt.scatter(top_uv[tv, 0], top_uv[tv, 1], c=top_weight[tv], s=54, cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(top_weight)), 1e-6), edgecolors="black", linewidths=0.35)
            cb = plt.colorbar(fraction=0.046, pad=0.04)
            cb.set_label("top-K soft cylinder weight")
        if (~tv).any():
            plt.scatter(top_uv[~tv, 0], top_uv[~tv, 1], s=48, marker="x", c="red", linewidths=1.0)
        plt.scatter([seed_uv[0]], [seed_uv[1]], s=100, marker="*", c="yellow", edgecolors="black", linewidths=0.7)
        plt.xlim(0, bg.shape[1] - 1)
        plt.ylim(bg.shape[0] - 1, 0)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout(pad=0.15)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_overlay_topk(
        self,
        bg: np.ndarray,
        seed_uv: np.ndarray,
        top_uv: np.ndarray,
        top_weight: np.ndarray,
        top_valid: np.ndarray,
        top_hard: np.ndarray,
        out_path: str,
        title: str,
    ):
        plt.figure(figsize=(7, 7))
        plt.imshow(bg)
        tv = top_valid.astype(bool)
        if tv.any():
            plt.scatter(top_uv[tv, 0], top_uv[tv, 1], c=top_weight[tv], s=58, cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(top_weight)), 1e-6), edgecolors="black", linewidths=0.35)
            cb = plt.colorbar(fraction=0.046, pad=0.04)
            cb.set_label("soft cylinder weight")
        hard = top_hard.astype(bool) & tv
        if hard.any():
            plt.scatter(top_uv[hard, 0], top_uv[hard, 1], s=90, facecolors="none", edgecolors="lime", linewidths=1.2)
        if (~tv).any():
            plt.scatter(top_uv[~tv, 0], top_uv[~tv, 1], s=48, marker="x", c="red", linewidths=1.0)
        plt.scatter([seed_uv[0]], [seed_uv[1]], s=100, marker="*", c="yellow", edgecolors="black", linewidths=0.7)
        plt.xlim(0, bg.shape[1] - 1)
        plt.ylim(bg.shape[0] - 1, 0)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout(pad=0.15)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_candidate_grid(
        self,
        weight: np.ndarray,
        hard: np.ndarray,
        rho: np.ndarray,
        h: np.ndarray,
        valid: np.ndarray,
        out_path: str,
        title: str,
    ):
        G = self.grid_size
        weight_g = weight.reshape(G, G)
        hard_g = hard.reshape(G, G).astype(np.float32)
        rho_g = rho.reshape(G, G)
        h_g = h.reshape(G, G)
        valid_g = valid.reshape(G, G).astype(np.float32)
        fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.3))
        im0 = axes[0].imshow(weight_g, vmin=0.0, vmax=max(float(np.nanmax(weight_g)), 1e-6), cmap="viridis")
        axes[0].set_title("soft weight")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(hard_g, vmin=0.0, vmax=1.0, cmap="gray")
        axes[1].set_title("hard cylinder")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(rho_g / max(self.radius, 1e-6), cmap="magma")
        axes[2].set_title("rho / radius")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        im3 = axes[3].imshow(h_g, vmin=self.hmin, vmax=self.hmax, cmap="coolwarm")
        axes[3].set_title("h along view")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        im4 = axes[4].imshow(valid_g, vmin=0.0, vmax=1.0, cmap="gray")
        axes[4].set_title("pixel/depth valid")
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.vis_dpi)
        plt.close(fig)

    def _save_topk_summary(
        self,
        top_weight: np.ndarray,
        top_rho: np.ndarray,
        top_h: np.ndarray,
        top_valid: np.ndarray,
        top_hard: np.ndarray,
        out_path: str,
        title: str,
    ):
        rows = np.stack([
            top_weight,
            top_rho / max(self.radius, 1e-6),
            top_h,
            top_valid.astype(np.float32),
            top_hard.astype(np.float32),
        ], axis=0)
        fig, axes = plt.subplots(5, 1, figsize=(8.5, 6.0), sharex=True)
        names = ["soft weight", "rho/radius", "h", "valid", "hard cyl"]
        cmaps = ["viridis", "magma", "coolwarm", "gray", "gray"]
        for i, ax in enumerate(axes):
            data = rows[i:i+1]
            if i == 2:
                im = ax.imshow(data, aspect="auto", cmap=cmaps[i], vmin=self.hmin, vmax=self.hmax)
            elif i in (3, 4):
                im = ax.imshow(data, aspect="auto", cmap=cmaps[i], vmin=0.0, vmax=1.0)
            else:
                im = ax.imshow(data, aspect="auto", cmap=cmaps[i])
            ax.set_yticks([])
            ax.set_ylabel(names[i], rotation=0, ha="right", va="center")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        axes[-1].set_xlabel("top-K rank by soft cylinder weight")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.vis_dpi)
        plt.close(fig)

    def _save_pool_winner(
        self,
        winner_idx: np.ndarray,
        top_valid: np.ndarray,
        out_path: str,
        title: str,
    ):
        K = top_valid.shape[0]
        hist = np.bincount(winner_idx.astype(np.int64), minlength=K).astype(np.float32)
        hist = hist / max(float(hist.sum()), 1.0)
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.2), sharex=True)
        axes[0].bar(np.arange(K), hist)
        axes[0].set_ylabel("winner freq")
        axes[0].set_title("max-pool winner over output channels")
        axes[1].bar(np.arange(K), top_valid.astype(np.float32))
        axes[1].set_ylabel("valid")
        axes[1].set_xlabel("top-K index")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.vis_dpi)
        plt.close(fig)

    @staticmethod
    def _write_ply_ascii(points: np.ndarray, colors: np.ndarray, out_path: str):
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        with open(out_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(points, colors):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    @torch.no_grad()
    def _maybe_visualize(
        self,
        end_points: Optional[dict],
        depth_map: torch.Tensor,
        seed_uv: torch.Tensor,
        seed_xyz: torch.Tensor,
        cand_uv: torch.Tensor,
        cand_xyz: torch.Tensor,
        cand_valid: torch.Tensor,
        cand_weight: torch.Tensor,
        cand_hard: torch.Tensor,
        cand_rho: torch.Tensor,
        cand_h: torch.Tensor,
        top_uv: torch.Tensor,
        top_xyz: torch.Tensor,
        top_valid: torch.Tensor,
        top_weight: torch.Tensor,
        top_hard: torch.Tensor,
        top_rho: torch.Tensor,
        top_h: torch.Tensor,
        winner_idx: torch.Tensor,
        token_sel_idx: torch.Tensor,
        window_radius_px: torch.Tensor,
    ):
        if self.vis_dir is None or self.vis_every <= 0 or (self._iter % self.vis_every) != 0:
            return
        try:
            b = 0
            bg, bg_name = self._background_np(end_points, depth_map, b)
            hard_ratio = cand_hard.float().mean(dim=-1)
            seed_ids = self._choose_vis_seed_indices(end_points, top_valid, top_weight, hard_ratio, b)
            prefix = f"pscg_it{self._iter:06d}_b{b}"
            view_ids = None
            if end_points is not None and "grasp_top_view_inds" in end_points and torch.is_tensor(end_points["grasp_top_view_inds"]):
                view_ids = end_points["grasp_top_view_inds"]

            for local_rank, m in enumerate(seed_ids):
                seed_uv_np = self._to_numpy(seed_uv[b, m])
                cand_uv_np = self._to_numpy(cand_uv[b, m])
                cand_valid_np = self._to_numpy(cand_valid[b, m]).astype(bool)
                cand_weight_np = self._to_numpy(cand_weight[b, m])
                cand_hard_np = self._to_numpy(cand_hard[b, m]).astype(bool)
                cand_rho_np = self._to_numpy(cand_rho[b, m])
                cand_h_np = self._to_numpy(cand_h[b, m])
                top_uv_np = self._to_numpy(top_uv[b, m])
                top_valid_np = self._to_numpy(top_valid[b, m]).astype(bool)
                top_weight_np = self._to_numpy(top_weight[b, m])
                top_hard_np = self._to_numpy(top_hard[b, m]).astype(bool)
                top_rho_np = self._to_numpy(top_rho[b, m])
                top_h_np = self._to_numpy(top_h[b, m])
                winner_np = winner_idx[b, m].detach().cpu().numpy()
                wpx = float(window_radius_px[b, m].detach().cpu())
                view_text = ""
                if view_ids is not None and view_ids.dim() == 2 and view_ids.size(1) > m:
                    view_text = f", view={int(view_ids[b, m].detach().cpu())}"
                title = (
                    f"seed m={m}, pix=({seed_uv_np[0]:.1f},{seed_uv_np[1]:.1f}), "
                    f"top_valid={top_valid_np.mean():.2f}, cand_hard={cand_hard_np.mean():.2f}, "
                    f"wpx={wpx:.1f}{view_text}"
                )
                stem = f"{prefix}_seed{local_rank:02d}_m{m:04d}"

                self._save_overlay_candidates(
                    bg, seed_uv_np, cand_uv_np, cand_hard_np, cand_valid_np,
                    top_uv_np, top_weight_np, top_valid_np,
                    os.path.join(self.vis_dir, f"{stem}_overlay_candidates.png"),
                    f"{title} [{bg_name}]",
                )
                self._save_overlay_topk(
                    bg, seed_uv_np, top_uv_np, top_weight_np, top_valid_np, top_hard_np,
                    os.path.join(self.vis_dir, f"{stem}_overlay_topk.png"),
                    f"{title} [{bg_name}]",
                )
                self._save_candidate_grid(
                    cand_weight_np, cand_hard_np, cand_rho_np, cand_h_np, cand_valid_np,
                    os.path.join(self.vis_dir, f"{stem}_candidate_grid.png"),
                    title,
                )
                self._save_topk_summary(
                    top_weight_np, top_rho_np, top_h_np, top_valid_np, top_hard_np,
                    os.path.join(self.vis_dir, f"{stem}_topk_summary.png"),
                    title,
                )
                self._save_pool_winner(
                    winner_np, top_valid_np,
                    os.path.join(self.vis_dir, f"{stem}_pool_winner.png"),
                    title,
                )
                if self.save_npz:
                    np.savez_compressed(
                        os.path.join(self.vis_dir, f"{stem}.npz"),
                        seed_uv=seed_uv_np,
                        seed_xyz=self._to_numpy(seed_xyz[b, m]),
                        cand_uv=cand_uv_np,
                        cand_xyz=self._to_numpy(cand_xyz[b, m]),
                        cand_valid=cand_valid_np,
                        cand_weight=cand_weight_np,
                        cand_hard=cand_hard_np,
                        cand_rho=cand_rho_np,
                        cand_h=cand_h_np,
                        top_uv=top_uv_np,
                        top_xyz=self._to_numpy(top_xyz[b, m]),
                        top_valid=top_valid_np,
                        top_weight=top_weight_np,
                        top_hard=top_hard_np,
                        top_rho=top_rho_np,
                        top_h=top_h_np,
                        winner_idx=winner_np,
                        window_radius_px=wpx,
                        token_idx=int(token_sel_idx[b, m].detach().cpu()),
                    )
                if self.save_ply:
                    pts = np.concatenate([
                        self._to_numpy(seed_xyz[b, m]).reshape(1, 3),
                        self._to_numpy(top_xyz[b, m]),
                    ], axis=0)
                    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
                    colors[0] = np.array([255, 255, 0], dtype=np.uint8)
                    sc = np.zeros((top_valid_np.shape[0], 3), dtype=np.uint8)
                    sc[top_valid_np & top_hard_np] = np.array([40, 220, 60], dtype=np.uint8)
                    sc[top_valid_np & ~top_hard_np] = np.array([220, 140, 40], dtype=np.uint8)
                    sc[~top_valid_np] = np.array([20, 20, 20], dtype=np.uint8)
                    colors[1:] = sc
                    self._write_ply_ascii(pts, colors, os.path.join(self.vis_dir, f"{stem}.ply"))
        except Exception as e:
            print(f"[ProjectedSurfaceCylinderGrouping vis] failed at iter {self._iter}: {repr(e)}")

    @staticmethod
    def _safe_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return torch.quantile(x, q)

    def _maybe_print(self, end_points: Optional[dict]):
        if self.debug_print_every <= 0 or (self._iter % self.debug_print_every) != 0:
            return
        if end_points is None:
            return
        keys = [
            "D: PSCG cand valid ratio",
            "D: PSCG cand hard cyl ratio",
            "D: PSCG top valid ratio",
            "D: PSCG empty group ratio",
            "D: PSCG top weight mean",
            "D: PSCG top rho/r mean",
            "D: PSCG top h mean",
            "D: PSCG window px mean",
            "D: PSCG group scale",
        ]
        parts = []
        for k in keys:
            v = end_points.get(k, None)
            if torch.is_tensor(v):
                parts.append(f"{k.replace('D: PSCG ', '')}={float(v.detach().reshape(()).cpu()):.4f}")
        if parts:
            print(f"[ProjectedSurfaceCylinderGrouping] it={self._iter} " + " ".join(parts))

    def forward(
        self,
        seed_features: torch.Tensor,       # (B,C,M)
        token_sel_idx: torch.Tensor,       # (B,M)
        top_view_rot: torch.Tensor,        # (B,M,3,3)
        feat_map: torch.Tensor,            # (B,C,Hf,Wf)
        depth_map: torch.Tensor,           # (B,1,H,W)
        K: torch.Tensor,                   # (B,3,3)
        end_points: Optional[dict] = None,
    ) -> torch.Tensor:
        B, C, M = seed_features.shape
        _, _, Hd, Wd = depth_map.shape
        device = seed_features.device
        dtype = seed_features.dtype
        L = self.base_grid_uv.shape[0]
        Ksel = min(self.nsample, L)

        depth_for_geom = depth_map.detach() if self.detach_depth else depth_map
        depth_for_geom = torch.nan_to_num(depth_for_geom, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
        K_geom = K.to(device=device, dtype=depth_for_geom.dtype)

        # 1) seed surface point.
        seed_uv = self._idx_to_uv(token_sel_idx, Hd, Wd).to(device=device, dtype=depth_for_geom.dtype)  # (B,M,2)
        seed_z = self._gather_depth(depth_for_geom, token_sel_idx).clamp_min(1e-6)                      # (B,M,1)
        seed_xyz = self._backproject_uvz_bm(seed_uv, seed_z, K_geom)                                    # (B,M,3)

        # 2) local 2D candidate pixels around each seed.
        if self.dynamic_window:
            fx = K_geom[:, 0, 0].unsqueeze(1)
            fy = K_geom[:, 1, 1].unsqueeze(1)
            fmean = 0.5 * (fx + fy)
            metric_span = math.sqrt(self.radius ** 2 + max(abs(self.hmin), abs(self.hmax)) ** 2)
            window_radius_px = fmean * (metric_span * self.window_radius_factor) / seed_z.squeeze(-1).clamp_min(1e-6)
            window_radius_px = window_radius_px.clamp(self.window_radius_px_min, self.window_radius_px_max)
        else:
            window_radius_px = torch.full((B, M), self.window_radius_px, device=device, dtype=depth_for_geom.dtype)

        base_grid = self.base_grid_uv.to(device=device, dtype=depth_for_geom.dtype)                     # (L,2)
        cand_uv = seed_uv.unsqueeze(2) + base_grid.view(1, 1, L, 2) * window_radius_px.unsqueeze(-1).unsqueeze(-1)
        cand_pixel_valid = (
            (cand_uv[..., 0] >= 0.0) & (cand_uv[..., 0] <= float(Wd - 1)) &
            (cand_uv[..., 1] >= 0.0) & (cand_uv[..., 1] <= float(Hd - 1))
        )

        # 3) visible surface candidates from depth map.
        cand_depth = self._grid_sample_bms(depth_for_geom, cand_uv, src_hw=(Hd, Wd))                    # (B,M,L,1)
        cand_depth_valid = cand_depth.squeeze(-1) > 1e-6
        cand_valid = cand_pixel_valid & cand_depth_valid
        cand_xyz = self._backproject_uvz_bml(cand_uv, cand_depth, K_geom)                               # (B,M,L,3)

        # 4) local view-frame coordinates, mimicking 3D CylinderQueryAndGroup logic.
        delta = cand_xyz - seed_xyz.unsqueeze(2)                                                        # (B,M,L,3)
        local = torch.einsum("bmij,bmlj->bmli", top_view_rot.to(delta.dtype).transpose(-1, -2), delta)  # (B,M,L,3)
        h = local[..., 0]
        y = local[..., 1]
        z = local[..., 2]
        rho = torch.sqrt(y * y + z * z + 1e-12)
        hard_cyl = cand_valid & (rho <= self.radius) & (h >= self.hmin) & (h <= self.hmax)

        # 5) soft cylinder membership, used to select top-K surface points.
        w_rad = torch.exp(-torch.square(rho / max(self.radius, 1e-6)))
        w_h1 = torch.sigmoid((h - self.hmin) / max(self.h_soft_tau, 1e-6))
        w_h2 = torch.sigmoid((self.hmax - h) / max(self.h_soft_tau, 1e-6))
        soft_weight = w_rad * w_h1 * w_h2 * cand_valid.to(w_rad.dtype)
        soft_weight = torch.nan_to_num(soft_weight, nan=0.0, posinf=0.0, neginf=0.0)

        top_weight, top_idx = torch.topk(soft_weight, k=Ksel, dim=2, largest=True, sorted=True)          # (B,M,K)
        top_uv = self._gather_bml(cand_uv, top_idx)                                                     # (B,M,K,2)
        top_xyz = self._gather_bml(cand_xyz, top_idx)                                                   # (B,M,K,3)
        top_local = self._gather_bml(local, top_idx)                                                     # (B,M,K,3)
        top_rho = self._gather_bml(rho, top_idx)                                                        # (B,M,K)
        top_h = self._gather_bml(h, top_idx)                                                            # (B,M,K)
        top_hard = self._gather_bml(hard_cyl.float(), top_idx).bool()                                   # (B,M,K)
        top_valid = self._gather_bml(cand_valid.float(), top_idx).bool() & (top_weight > self.min_group_weight)

        # 6) sample dense image feature only for selected surface candidates.
        sampled_feat = self._grid_sample_bms(feat_map, top_uv.to(feat_map.dtype), src_hw=(Hd, Wd))       # (B,M,K,C)
        top_valid_f = top_valid.to(dtype).unsqueeze(-1)
        top_hard_f = top_hard.to(dtype).unsqueeze(-1)
        top_weight_f = top_weight.to(dtype).unsqueeze(-1)

        # normalized local coordinates: [h, y, z] in selected-view frame.
        h_scale = max(abs(self.hmin), abs(self.hmax), (self.hmax - self.hmin) * 0.5, 1e-6)
        local_norm = torch.stack([
            top_local[..., 0] / h_scale,
            top_local[..., 1] / max(self.radius, 1e-6),
            top_local[..., 2] / max(self.radius, 1e-6),
        ], dim=-1).to(dtype)
        local_emb = self.coord_mlp(local_norm)

        sample_in = torch.cat([
            sampled_feat.to(dtype),
            local_emb,
            top_weight_f,
            top_hard_f,
            top_valid_f,
        ], dim=-1)
        sample_tokens = self.sample_mlp(sample_in).view(B * M, Ksel, self.out_dim)

        if self.local_interaction_module is not None:
            valid_flat = top_valid.view(B * M, Ksel)
            # key-valid attention mask.  Invalid tokens cannot be attended to; all-empty groups are kept unmasked to avoid NaNs.
            attn_mask = valid_flat[:, None, :].expand(-1, Ksel, -1).clone()
            empty_flat = ~valid_flat.any(dim=1)
            if empty_flat.any():
                attn_mask[empty_flat] = True
            sample_tokens = self.local_interaction_module(sample_tokens, sample_tokens, sample_tokens, mask=attn_mask)

        sample_tokens = sample_tokens.view(B, M, Ksel, self.out_dim)
        valid_any = top_valid.any(dim=-1, keepdim=True)                                                  # (B,M,1)
        masked_tokens = sample_tokens.masked_fill(~top_valid.unsqueeze(-1), -1e4)
        pooled, winner_idx = masked_tokens.max(dim=2)                                                    # (B,M,out_dim)
        pooled = torch.where(valid_any, pooled, torch.zeros_like(pooled))

        seed_ctx = self.seed_proj(seed_features.transpose(1, 2).contiguous())                            # (B,M,out_dim)
        delta_feat = self.group_delta(torch.cat([seed_ctx, pooled], dim=-1))
        if self.residual_fusion:
            grouped = seed_ctx + self.group_scale.to(seed_ctx.dtype) * delta_feat
        else:
            grouped = delta_feat + seed_ctx
        grouped = grouped.transpose(1, 2).contiguous()                                                   # (B,out_dim,M)

        if end_points is not None:
            with torch.no_grad():
                valid_count = top_valid.float().sum().clamp_min(1.0)
                hard_count = hard_cyl.float().sum().clamp_min(1.0)
                top_rho_valid = top_rho.detach()[top_valid]
                top_h_valid = top_h.detach()[top_valid]
                top_weight_valid = top_weight.detach()[top_valid]
                zero = torch.zeros((), device=device, dtype=dtype)

                end_points["D: PSCG candidates"] = torch.tensor(float(L), device=device)
                end_points["D: PSCG samples"] = torch.tensor(float(Ksel), device=device)
                end_points["D: PSCG window px mean"] = window_radius_px.detach().mean().reshape(())
                end_points["D: PSCG window px max"] = window_radius_px.detach().max().reshape(())
                end_points["D: PSCG cand pixel valid ratio"] = cand_pixel_valid.float().mean().reshape(())
                end_points["D: PSCG cand depth valid ratio"] = cand_depth_valid.float().mean().reshape(())
                end_points["D: PSCG cand valid ratio"] = cand_valid.float().mean().reshape(())
                end_points["D: PSCG cand hard cyl ratio"] = hard_cyl.float().mean().reshape(())
                end_points["D: PSCG top valid ratio"] = top_valid.float().mean().reshape(())
                end_points["D: PSCG empty group ratio"] = (~top_valid.any(dim=-1)).float().mean().reshape(())
                end_points["D: PSCG group scale"] = self.group_scale.detach().reshape(())
                if top_weight_valid.numel() > 0:
                    end_points["D: PSCG top weight mean"] = top_weight_valid.mean().reshape(())
                    end_points["D: PSCG top weight median"] = top_weight_valid.median().reshape(())
                    end_points["D: PSCG top weight p90"] = self._safe_quantile(top_weight_valid, 0.90).reshape(())
                else:
                    end_points["D: PSCG top weight mean"] = zero
                    end_points["D: PSCG top weight median"] = zero
                    end_points["D: PSCG top weight p90"] = zero
                if top_rho_valid.numel() > 0:
                    end_points["D: PSCG top rho mean"] = top_rho_valid.mean().reshape(())
                    end_points["D: PSCG top rho/r mean"] = (top_rho_valid / max(self.radius, 1e-6)).mean().reshape(())
                    end_points["D: PSCG top h mean"] = top_h_valid.mean().reshape(())
                    end_points["D: PSCG top |h| mean"] = top_h_valid.abs().mean().reshape(())
                else:
                    end_points["D: PSCG top rho mean"] = zero
                    end_points["D: PSCG top rho/r mean"] = zero
                    end_points["D: PSCG top h mean"] = zero
                    end_points["D: PSCG top |h| mean"] = zero

        self._maybe_visualize(
            end_points=end_points,
            depth_map=depth_for_geom,
            seed_uv=seed_uv,
            seed_xyz=seed_xyz,
            cand_uv=cand_uv,
            cand_xyz=cand_xyz,
            cand_valid=cand_valid,
            cand_weight=soft_weight,
            cand_hard=hard_cyl,
            cand_rho=rho,
            cand_h=h,
            top_uv=top_uv,
            top_xyz=top_xyz,
            top_valid=top_valid,
            top_weight=top_weight,
            top_hard=top_hard,
            top_rho=top_rho,
            top_h=top_h,
            winner_idx=winner_idx,
            token_sel_idx=token_sel_idx,
            window_radius_px=window_radius_px,
        )
        self._maybe_print(end_points)
        self._iter += 1
        return grouped



class ViewConditionedDeformable2DGrouping(nn.Module):
    """
    Minimal view-conditioned deformable 2D grouping.

    Differences from ProjectedSurfaceCylinderGrouping:
      - Does NOT backproject local pixels into pseudo surface points.
      - Does NOT use hard/soft cylinder membership.
      - Samples dense 2D feature map directly using learned offsets.
      - selected view only modulates offsets/attention as a condition.

    Inputs:
        seed_features: (B, C_seed, M)
        token_sel_idx: (B, M), flattened pixel indices in feat_map resolution
        top_view_rot:  (B, M, 3, 3), selected view rotation from ViewNet / label matching
        feat_map:      (B, C_feat, H, W), dense DPT feature map
        end_points:    optional dict for debug scalars / visualization

    Output:
        operation_feature: (B, out_dim, M), default out_dim=256.
    """

    def __init__(
        self,
        seed_feature_dim: int = 128,
        feat_dim: int = 128,
        out_dim: int = 256,
        hidden_dim: Optional[int] = None,
        num_samples: int = 32,
        num_heads: int = 4,
        base_radius_px: float = 24.0,
        offset_scale_px: float = 16.0,
        dropout: float = 0.05,
        zero_init_offsets: bool = True,
        use_context_skip: bool = True,
        debug_print_every: int = 50,
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        vis_num_seeds: int = 4,
        vis_seed_mode: str = "first",  # first | high_entropy | high_offset
        save_npz: bool = True,
    ):
        super().__init__()
        self.seed_feature_dim = int(seed_feature_dim)
        self.feat_dim = int(feat_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim or out_dim)
        self.num_samples = int(num_samples)
        self.num_heads = int(num_heads)
        assert self.out_dim % self.num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = self.out_dim // self.num_heads
        self.base_radius_px = float(base_radius_px)
        self.offset_scale_px = float(offset_scale_px)
        self.use_context_skip = bool(use_context_skip)
        self.debug_print_every = int(debug_print_every)
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.vis_num_seeds = int(vis_num_seeds)
        self.vis_seed_mode = str(vis_seed_mode)
        self.save_npz = bool(save_npz)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        # Unit disk base offsets. Learnable offsets start as perturbations of this pattern.
        base_offsets = self._make_base_offsets(self.num_samples)  # (K,2), in unit disk
        self.register_buffer("base_offsets", base_offsets)

        # Context = seed feature + selected view rotation(9) + normalized seed uv(2).
        context_dim = self.seed_feature_dim + 9 + 2
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.offset_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_samples * 2),
        )

        # Optional per-sample attention prior from seed/view context.
        self.attn_prior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_samples),
        )

        # sample token = sampled 2D feature + normalized offset + valid flag.
        sample_in_dim = self.feat_dim + 2 + 1
        self.sample_proj = nn.Sequential(
            nn.Linear(sample_in_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.sample_norm = nn.LayerNorm(self.out_dim)

        # Multi-head deformable attention: one query per seed attends to K sampled tokens.
        self.q_proj = nn.Linear(self.hidden_dim, self.out_dim)
        self.k_proj = nn.Linear(self.out_dim, self.out_dim)
        self.v_proj = nn.Linear(self.out_dim, self.out_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.out_norm = nn.LayerNorm(self.out_dim)

        if self.use_context_skip:
            self.context_skip = nn.Linear(self.hidden_dim, self.out_dim)
        else:
            self.context_skip = None

        # Start from the fixed local pattern. This makes the first iteration close to
        # ordinary 2D local pooling, then learns deformation if helpful.
        if zero_init_offsets:
            nn.init.zeros_(self.offset_mlp[-1].weight)
            nn.init.zeros_(self.offset_mlp[-1].bias)
            nn.init.zeros_(self.attn_prior_mlp[-1].weight)
            nn.init.zeros_(self.attn_prior_mlp[-1].bias)

    @staticmethod
    def _make_base_offsets(num_samples: int) -> torch.Tensor:
        """Sunflower pattern in unit disk, first point at center."""
        K = int(num_samples)
        offsets = np.zeros((K, 2), dtype=np.float32)
        if K <= 1:
            return torch.from_numpy(offsets)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        offsets[0] = 0.0
        for i in range(1, K):
            r = math.sqrt(i / max(K - 1, 1))
            theta = i * golden
            offsets[i, 0] = r * math.cos(theta)
            offsets[i, 1] = r * math.sin(theta)
        return torch.from_numpy(offsets)

    @staticmethod
    def _idx_to_uv(idx_bm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()
        return torch.stack([u, v], dim=-1)  # (B,M,2), pixel coords

    @staticmethod
    def _pixel_to_grid(uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Pixel uv -> grid_sample normalized xy, align_corners=True."""
        u = uv[..., 0]
        v = uv[..., 1]
        x = 2.0 * u / max(W - 1, 1) - 1.0
        y = 2.0 * v / max(H - 1, 1) - 1.0
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _valid_uv(uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = uv[..., 0]
        v = uv[..., 1]
        return (u >= 0.0) & (u <= (W - 1)) & (v >= 0.0) & (v <= (H - 1))

    @staticmethod
    def _to_rgb_np(img_chw: torch.Tensor) -> np.ndarray:
        x = img_chw.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x.permute(1, 2, 0).numpy()

    def _sample_feat_map(self, feat_map: torch.Tensor, sample_uv: torch.Tensor) -> torch.Tensor:
        """
        feat_map:  (B,C,H,W)
        sample_uv: (B,M,K,2) pixel coords
        return:    (B,M,K,C)
        """
        B, C, H, W = feat_map.shape
        _, M, K, _ = sample_uv.shape
        grid = self._pixel_to_grid(sample_uv, H, W).view(B, M * K, 1, 2)
        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B,C,M*K,1)
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous().view(B, M, K, C)
        return sampled

    @torch.no_grad()
    def _maybe_visualize(
        self,
        end_points: Optional[dict],
        seed_uv: torch.Tensor,       # (B,M,2)
        sample_uv: torch.Tensor,     # (B,M,K,2)
        offsets_px: torch.Tensor,    # (B,M,K,2)
        valid: torch.Tensor,         # (B,M,K)
        attn_mean: torch.Tensor,     # (B,M,K)
        attn_entropy: torch.Tensor,  # (B,M)
        offset_norm: torch.Tensor,   # (B,M,K)
    ):
        if self.vis_dir is None:
            return
        if self._vis_iter % self.vis_every != 0:
            return
        if end_points is None or "img" not in end_points:
            return

        try:
            img = end_points["img"]
            B, _, H, W = img.shape
            b = 0
            img_np = self._to_rgb_np(img[b])
            prefix = f"d2dgroup_it{self._vis_iter:06d}"

            # Seed-level maps: attention entropy / mean offset norm.
            seed_idx = end_points.get("token_sel_idx", None)
            if torch.is_tensor(seed_idx):
                idx = seed_idx[b].detach().cpu()
            else:
                idx = (seed_uv[b, :, 1].long() * W + seed_uv[b, :, 0].long()).detach().cpu()
            u = (idx % W).numpy()
            v = (idx // W).numpy()

            ent = attn_entropy[b].detach().float().cpu().numpy()
            offm = offset_norm[b].mean(dim=-1).detach().float().cpu().numpy()

            for values, name, cmap in [
                (ent, "attn_entropy", "magma"),
                (offm, "mean_offset_norm", "viridis"),
            ]:
                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                plt.scatter(u, v, c=values, s=8, cmap=cmap)
                plt.axis("off")
                plt.title(name)
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(self.vis_dir, f"{prefix}_seed_{name}.png"), dpi=150)
                plt.close()

            # Choose seeds for detailed visualization.
            M = seed_uv.shape[1]
            nvis = min(self.vis_num_seeds, M)
            if self.vis_seed_mode == "high_entropy":
                order = torch.argsort(attn_entropy[b], descending=True)[:nvis]
            elif self.vis_seed_mode == "high_offset":
                order = torch.argsort(offset_norm[b].mean(dim=-1), descending=True)[:nvis]
            else:
                order = torch.arange(nvis, device=seed_uv.device)

            for rank, m_idx_t in enumerate(order):
                m_idx = int(m_idx_t.item())
                su = seed_uv[b, m_idx].detach().cpu().numpy()
                pts = sample_uv[b, m_idx].detach().cpu().numpy()
                att = attn_mean[b, m_idx].detach().float().cpu().numpy()
                val = valid[b, m_idx].detach().cpu().numpy().astype(bool)
                off = offsets_px[b, m_idx].detach().float().cpu().numpy()

                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                plt.scatter(pts[~val, 0], pts[~val, 1], c="red", s=18, marker="x", label="invalid")
                sc = plt.scatter(pts[val, 0], pts[val, 1], c=att[val], s=28, cmap="viridis", vmin=0.0, vmax=max(float(att.max()), 1e-6), label="sample")
                plt.scatter([su[0]], [su[1]], c="yellow", s=80, marker="*", edgecolors="black", linewidths=0.6, label="seed")
                plt.colorbar(sc, fraction=0.046, pad=0.04, label="mean attention")
                plt.axis("off")
                plt.title(f"seed {m_idx} deformable samples")
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(self.vis_dir, f"{prefix}_seed{rank:02d}_m{m_idx:04d}_overlay.png"), dpi=150)
                plt.close()

                # Attention/offset summary for this seed.
                x = np.arange(att.shape[0])
                fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
                axes[0].bar(x, att)
                axes[0].set_ylabel("attn")
                axes[1].bar(x, np.linalg.norm(off, axis=-1))
                axes[1].set_ylabel("|offset| px")
                axes[2].bar(x, val.astype(np.float32))
                axes[2].set_ylabel("valid")
                axes[2].set_xlabel("sample index")
                fig.tight_layout()
                fig.savefig(os.path.join(self.vis_dir, f"{prefix}_seed{rank:02d}_m{m_idx:04d}_bars.png"), dpi=150)
                plt.close(fig)

                if self.save_npz:
                    np.savez_compressed(
                        os.path.join(self.vis_dir, f"{prefix}_seed{rank:02d}_m{m_idx:04d}.npz"),
                        seed_uv=su,
                        sample_uv=pts,
                        offsets_px=off,
                        attention=att,
                        valid=val,
                    )
        except Exception as e:
            print(f"[ViewConditionedDeformable2DGrouping vis] failed at iter {self._vis_iter}: {repr(e)}")

    def forward(
        self,
        seed_features: torch.Tensor,       # (B,C_seed,M)
        token_sel_idx: torch.Tensor,       # (B,M)
        top_view_rot: torch.Tensor,        # (B,M,3,3)
        feat_map: torch.Tensor,            # (B,C_feat,H,W)
        end_points: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, C_seed, M = seed_features.shape
        Bf, C_feat, H, W = feat_map.shape
        assert B == Bf, f"batch mismatch: seed {B}, feat {Bf}"
        assert C_feat == self.feat_dim, f"feat_dim mismatch: expected {self.feat_dim}, got {C_feat}"
        K = self.num_samples
        device = seed_features.device
        dtype = seed_features.dtype

        seed_feat_bm = seed_features.transpose(1, 2).contiguous()  # (B,M,C)
        seed_uv = self._idx_to_uv(token_sel_idx.long(), H, W)       # (B,M,2)
        seed_uv_norm = torch.stack([
            2.0 * seed_uv[..., 0] / max(W - 1, 1) - 1.0,
            2.0 * seed_uv[..., 1] / max(H - 1, 1) - 1.0,
        ], dim=-1)

        view_flat = top_view_rot.reshape(B, M, 9).to(dtype)
        context_in = torch.cat([seed_feat_bm, view_flat, seed_uv_norm.to(dtype)], dim=-1)
        context = self.context_mlp(context_in)  # (B,M,Hid)

        base_offsets_px = self.base_offsets.to(device=device, dtype=dtype) * self.base_radius_px  # (K,2)
        delta = self.offset_mlp(context).view(B, M, K, 2)
        delta = torch.tanh(delta) * self.offset_scale_px
        offsets_px = base_offsets_px.view(1, 1, K, 2) + delta

        sample_uv = seed_uv.unsqueeze(2).to(dtype) + offsets_px  # (B,M,K,2)
        valid = self._valid_uv(sample_uv, H, W)                  # (B,M,K)

        sampled_feat = self._sample_feat_map(feat_map, sample_uv)  # (B,M,K,C_feat)

        offset_normed = offsets_px / max(self.base_radius_px + self.offset_scale_px, 1e-6)
        valid_f = valid.to(dtype).unsqueeze(-1)
        sample_in = torch.cat([sampled_feat, offset_normed, valid_f], dim=-1)
        sample_tokens = self.sample_proj(sample_in)
        sample_tokens = self.sample_norm(sample_tokens)

        q = self.q_proj(context).view(B, M, self.num_heads, self.head_dim)          # (B,M,Hd,D)
        k = self.k_proj(sample_tokens).view(B, M, K, self.num_heads, self.head_dim) # (B,M,K,Hd,D)
        v = self.v_proj(sample_tokens).view(B, M, K, self.num_heads, self.head_dim)

        # (B,M,Hd,K)
        attn_logits = (q.unsqueeze(2) * k).sum(dim=-1).permute(0, 1, 3, 2)
        attn_logits = attn_logits / math.sqrt(self.head_dim)
        attn_prior = self.attn_prior_mlp(context).unsqueeze(2)  # (B,M,1,K)
        attn_logits = attn_logits + attn_prior

        # Mask invalid sampled positions. Seed center is usually valid; still guard all-invalid.
        valid_h = valid.unsqueeze(2)  # (B,M,1,K)
        all_invalid = ~valid.any(dim=-1, keepdim=True)  # (B,M,1)
        if all_invalid.any():
            # Make sample 0 valid for all-invalid groups to avoid NaN softmax.
            valid = valid.clone()
            valid[..., 0] = torch.where(all_invalid.squeeze(-1), torch.ones_like(valid[..., 0]), valid[..., 0])
            valid_h = valid.unsqueeze(2)
        attn_logits = attn_logits.masked_fill(~valid_h, -1e4)
        attn = torch.softmax(attn_logits, dim=-1)  # (B,M,Hd,K)

        # Weighted sum over samples.
        v_h = v.permute(0, 1, 3, 2, 4)  # (B,M,Hd,K,D)
        out = (attn.unsqueeze(-1) * v_h).sum(dim=3).reshape(B, M, self.out_dim)
        if self.context_skip is not None:
            out = out + self.context_skip(context)
        out = self.out_norm(out)
        out = self.out_proj(out)
        grouped_feature = out.transpose(1, 2).contiguous()  # (B,out_dim,M)

        # Debug scalars.
        with torch.no_grad():
            attn_mean = attn.mean(dim=2)  # (B,M,K)
            attn_entropy = -(attn_mean.clamp_min(1e-8) * attn_mean.clamp_min(1e-8).log()).sum(dim=-1)
            attn_entropy = attn_entropy / math.log(max(K, 2))
            attn_max = attn_mean.max(dim=-1).values
            offset_norm = offsets_px.norm(dim=-1)
            valid_ratio = valid.float().mean()
            mean_offset_norm = offset_norm.mean()
            max_offset_norm = offset_norm.max()
            sampled_feat_norm = sampled_feat.norm(dim=-1).mean()
            op_norm = grouped_feature.transpose(1, 2).norm(dim=-1).mean()
            seed_norm = seed_feat_bm.norm(dim=-1).mean().clamp_min(1e-8)
            op_seed_ratio = op_norm / seed_norm

            if end_points is not None:
                end_points["D: D2DG samples"] = torch.tensor(float(K), device=device)
                end_points["D: D2DG valid ratio"] = valid_ratio.reshape(())
                end_points["D: D2DG attn entropy"] = attn_entropy.mean().reshape(())
                end_points["D: D2DG attn max"] = attn_max.mean().reshape(())
                end_points["D: D2DG offset norm mean"] = mean_offset_norm.reshape(())
                end_points["D: D2DG offset norm max"] = max_offset_norm.reshape(())
                end_points["D: D2DG sampled feat norm"] = sampled_feat_norm.reshape(())
                end_points["D: D2DG op feat norm"] = op_norm.reshape(())
                end_points["D: D2DG op/seed norm"] = op_seed_ratio.reshape(())
                end_points["D: D2DG base radius px"] = torch.tensor(self.base_radius_px, device=device)
                end_points["D: D2DG offset scale px"] = torch.tensor(self.offset_scale_px, device=device)

                # Useful tensors for ad-hoc debugging. Detached to avoid memory surprises.
                end_points["d2dg_sample_uv"] = sample_uv.detach()
                end_points["d2dg_offsets_px"] = offsets_px.detach()
                end_points["d2dg_attn_mean"] = attn_mean.detach()
                end_points["d2dg_valid"] = valid.detach()

            if self.debug_print_every > 0 and (self._vis_iter % self.debug_print_every == 0):
                print(
                    f"[ViewConditionedDeformable2DGrouping] it={self._vis_iter} "
                    f"valid={float(valid_ratio.item()):.3f} "
                    f"attn_entropy={float(attn_entropy.mean().item()):.3f} "
                    f"attn_max={float(attn_max.mean().item()):.3f} "
                    f"offset_mean={float(mean_offset_norm.item()):.2f}px "
                    f"offset_max={float(max_offset_norm.item()):.2f}px "
                    f"op/seed={float(op_seed_ratio.item()):.3f}"
                )

            self._maybe_visualize(
                end_points=end_points,
                seed_uv=seed_uv,
                sample_uv=sample_uv,
                offsets_px=offsets_px,
                valid=valid,
                attn_mean=attn_mean,
                attn_entropy=attn_entropy,
                offset_norm=offset_norm,
            )

        self._vis_iter += 1
        return grouped_feature


class DepthDisLocalAttnGrouping(nn.Module):
    """
    Depth-Distribution Local Attention Grouping (DDLA).

    For each selected seed pixel, DDLA builds local tokens from:
      local 2D pixels around seed  x  top-k depth hypotheses per pixel.
    patch_stride controls the pixel spacing between neighboring local samples,
    so patch_size=7, patch_stride=3 covers roughly a 19x19 px region with only 49 pixels.

    A token contains:
      - dense 2D feature F(u_i)
      - 2D relative offset from seed
      - soft depth probability p(z_k|u_i)
      - camera backprojected relative 3D coordinate under depth hypothesis z_k
      - selected-view local coordinate R_view^T (X_i,k - X_seed)
      - ray/view compatibility

    A seed-view query then attends to all local ray/depth tokens.
    """

    def __init__(
        self,
        seed_feature_dim: int = 128,
        feat_dim: int = 128,
        out_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        patch_size: int = 5,
        patch_stride: float = 1.0,
        adaptive_radius: bool = False,
        physical_radius: float = 0.04,
        patch_radius_px_min: float = 6.0,
        patch_radius_px_max: float = 32.0,
        detach_depth_prob: bool = True,
        depth_topk: int = 4,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        xyz_norm_scale: float = 0.10,
        depth_rel_norm_scale: float = 0.10,
        min_depth_prob: float = 1e-5,
        dropout: float = 0.05,
        use_seed_skip: bool = True,
        debug_print_every: int = 50,
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        vis_num_seeds: int = 4,
        vis_seed_mode: str = "first",  # first | high_entropy | high_offset | low_valid
        save_npz: bool = True,
    ):
        super().__init__()
        self.seed_feature_dim = int(seed_feature_dim)
        self.feat_dim = int(feat_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = self.hidden_dim // self.num_heads

        self.patch_size = int(patch_size)
        if self.patch_size % 2 != 1:
            raise ValueError("patch_size should be odd")
        self.patch_stride = float(patch_stride)
        if self.patch_stride <= 0:
            raise ValueError("patch_stride should be > 0")
        self.adaptive_radius = bool(adaptive_radius)
        self.physical_radius = float(physical_radius)
        self.patch_radius_px_min = float(patch_radius_px_min)
        self.patch_radius_px_max = float(patch_radius_px_max)
        if self.patch_radius_px_min <= 0 or self.patch_radius_px_max < self.patch_radius_px_min:
            raise ValueError("Require 0 < patch_radius_px_min <= patch_radius_px_max")
        self.detach_depth_prob = bool(detach_depth_prob)
        self.patch_radius_px = float(self.patch_size // 2) * self.patch_stride
        self.depth_topk = int(depth_topk)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)
        self.xyz_norm_scale = float(xyz_norm_scale)
        self.depth_rel_norm_scale = float(depth_rel_norm_scale)
        self.min_depth_prob = float(min_depth_prob)
        self.use_seed_skip = bool(use_seed_skip)

        self.debug_print_every = int(debug_print_every)
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.vis_num_seeds = int(vis_num_seeds)
        self.vis_seed_mode = str(vis_seed_mode)
        self.save_npz = bool(save_npz)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        # Fixed square patch offsets in pixel units, order: row-major.
        r = self.patch_size // 2
        ys, xs = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=torch.float32),
            torch.arange(-r, r + 1, dtype=torch.float32),
            indexing="ij",
        )
        patch_offsets_unit = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (L,2) as integer grid offsets
        self.register_buffer("patch_offsets_unit", patch_offsets_unit)
        self.num_local_pixels = int(patch_offsets_unit.shape[0])
        self.num_tokens = self.num_local_pixels * self.depth_topk

        # Default depth bin centers. If input D != bin_num, forward regenerates centers on the fly.
        bin_centers = torch.linspace(self.min_depth, self.max_depth, self.bin_num, dtype=torch.float32)
        self.register_buffer("bin_centers", bin_centers)

        # Query from seed feature + selected view axis + seed ray + seed depth statistics.
        query_in_dim = self.seed_feature_dim + 3 + 3 + 2  # seed_feat, view_dir, seed_ray, z_norm+entropy
        self.query_mlp = nn.Sequential(
            nn.Linear(query_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Token from sampled feat + 2D offset + local/view-frame xyz + depth/prob/ray cues.
        # [feat(C), offset2d(2), local_xyz(3), rel_depth(1), prob(1), ray_dot_view(1), valid(1)]
        token_in_dim = self.feat_dim + 2 + 3 + 1 + 1 + 1 + 1
        self.token_mlp = nn.Sequential(
            nn.Linear(token_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.context_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.seed_proj = nn.Linear(self.seed_feature_dim, self.out_dim)
        self.out_norm = nn.LayerNorm(self.out_dim)
        self.out_ffn = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, self.out_dim),
        )

    # ------------------------------------------------------------------
    # Geometry / sampling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _idx_to_uv(idx_bm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()
        return torch.stack([u, v], dim=-1)  # (B,M,2)

    @staticmethod
    def _ray_dirs_from_uv(uv_bm2: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
        fx = K_b33[:, 0, 0].unsqueeze(1)
        fy = K_b33[:, 1, 1].unsqueeze(1)
        cx = K_b33[:, 0, 2].unsqueeze(1)
        cy = K_b33[:, 1, 2].unsqueeze(1)
        u = uv_bm2[..., 0]
        v = uv_bm2[..., 1]
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = torch.ones_like(x)
        rays = torch.stack([x, y, z], dim=-1)
        return F.normalize(rays, dim=-1, eps=1e-6)

    @staticmethod
    def _backproject_uvz(uv: torch.Tensor, z: torch.Tensor, K_b33: torch.Tensor) -> torch.Tensor:
        """Backproject arbitrary uv/z.

        uv: (B,M,...,2), z: (B,M,...), K: (B,3,3)
        return: (B,M,...,3)
        """
        # Broadcast K to uv dims.
        expand_shape = [K_b33.shape[0]] + [1] * (uv.dim() - 2)
        fx = K_b33[:, 0, 0].view(*expand_shape)
        fy = K_b33[:, 1, 1].view(*expand_shape)
        cx = K_b33[:, 0, 2].view(*expand_shape)
        cy = K_b33[:, 1, 2].view(*expand_shape)
        u = uv[..., 0]
        v = uv[..., 1]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _sample_bchw_at_uv(feat_bchw: torch.Tensor, uv_bml2: torch.Tensor) -> torch.Tensor:
        """Bilinear sample feature map at absolute pixel coordinates.

        feat_bchw: (B,C,H,W)
        uv_bml2:  (B,M,L,2), coordinates in pixel units (u,v)
        return:    (B,M,L,C)
        """
        B, C, H, W = feat_bchw.shape
        _, M, L, _ = uv_bml2.shape
        x = uv_bml2[..., 0]
        y = uv_bml2[..., 1]
        gx = x / max(W - 1, 1) * 2.0 - 1.0
        gy = y / max(H - 1, 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(B, M * L, 1, 2)
        sampled = F.grid_sample(
            feat_bchw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B,C,M*L,1)
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous().view(B, M, L, C)
        return sampled

    def _prepare_depth_prob(self, depth_prob: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Return depth_prob as normalized (B,D,H,W)."""
        if depth_prob.dim() == 4:
            # (B,D,H,W)
            if depth_prob.shape[-2:] == (H, W) and depth_prob.shape[1] != 1:
                prob = depth_prob
            # (B,1,N,D)
            elif depth_prob.shape[1] == 1 and depth_prob.shape[2] == H * W:
                prob = depth_prob.squeeze(1).view(depth_prob.shape[0], H, W, depth_prob.shape[3]).permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"Unsupported depth_prob shape {tuple(depth_prob.shape)} for H,W={H},{W}")
        elif depth_prob.dim() == 3 and depth_prob.shape[1] == H * W:
            # (B,N,D)
            prob = depth_prob.view(depth_prob.shape[0], H, W, depth_prob.shape[2]).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported depth_prob shape {tuple(depth_prob.shape)}")

        prob = torch.nan_to_num(prob.float(), nan=0.0, posinf=0.0, neginf=0.0)
        # If looks like logits or non-normalized values, normalize robustly.
        if prob.min().item() < -1e-4 or prob.max().item() > 1.0 + 1e-4:
            prob = torch.softmax(prob, dim=1)
        else:
            prob = prob.clamp_min(1e-8)
            prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return prob

    def _get_bin_centers(self, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if D == self.bin_num:
            return self.bin_centers.to(device=device, dtype=dtype)
        return torch.linspace(self.min_depth, self.max_depth, D, device=device, dtype=dtype)

    @staticmethod
    def _safe_entropy(prob: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        p = prob.clamp_min(eps)
        h = -(p * p.log()).sum(dim=dim)
        denom = math.log(max(prob.shape[dim], 2))
        return h / denom

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_rgb_np(img_chw: torch.Tensor) -> np.ndarray:
        x = img_chw.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x.permute(1, 2, 0).numpy()

    def _save_seed_scatter_overlay(
        self,
        img_chw: torch.Tensor,
        token_sel_idx: torch.Tensor,
        values: torch.Tensor,
        out_path: str,
        H: int,
        W: int,
        cmap: str = "viridis",
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        dot_size: int = 10,
    ) -> None:
        img_np = self._to_rgb_np(img_chw)
        idx = token_sel_idx.detach().cpu()
        vals = values.detach().float().cpu()
        u = (idx % W).numpy()
        v = (idx // W).numpy()
        if vmin is None:
            vmin = float(vals.min().item())
        if vmax is None:
            vmax = float(vals.max().item())
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.scatter(u, v, c=vals.numpy(), s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _select_vis_seed_indices(
        self,
        ent_bm: torch.Tensor,
        weighted_offset_norm_bm: torch.Tensor,
        seed_valid_ratio_bm: torch.Tensor,
        b: int,
        M: int,
    ) -> torch.Tensor:
        n = min(self.vis_num_seeds, M)
        mode = self.vis_seed_mode
        if mode == "high_entropy":
            return torch.topk(ent_bm[b], k=n, largest=True).indices
        if mode == "high_offset":
            return torch.topk(weighted_offset_norm_bm[b], k=n, largest=True).indices
        if mode == "low_valid":
            return torch.topk(seed_valid_ratio_bm[b], k=n, largest=False).indices
        return torch.arange(n, device=ent_bm.device)

    @torch.no_grad()
    def _maybe_visualize(
        self,
        end_points: Optional[dict],
        token_sel_idx: torch.Tensor,
        cand_uv: torch.Tensor,             # (B,M,L,2)
        pixel_attn: torch.Tensor,          # (B,M,L)
        token_attn: torch.Tensor,          # (B,M,L,Kz)
        top_prob: torch.Tensor,            # (B,M,L,Kz)
        top_z: torch.Tensor,               # (B,M,L,Kz)
        pix_valid: torch.Tensor,           # (B,M,L)
        token_valid: torch.Tensor,         # (B,M,L,Kz)
        seed_attn_entropy: torch.Tensor,   # (B,M)
        weighted_offset_norm: torch.Tensor,# (B,M)
        seed_valid_ratio: torch.Tensor,    # (B,M)
        patch_radius_px_bm: torch.Tensor,  # (B,M)
        patch_stride_bm: torch.Tensor,     # (B,M)
    ) -> None:
        if self.vis_dir is None or end_points is None:
            return
        if self._vis_iter % self.vis_every != 0:
            return
        if "img" not in end_points:
            return

        try:
            img = end_points["img"]
            B, _, H, W = img.shape
            b0 = 0
            prefix = f"ddla_it{self._vis_iter:06d}"

            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], seed_attn_entropy[b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_attn_entropy.png"),
                H=H, W=W, cmap="magma", title="DDLA attention entropy", vmin=0.0, vmax=1.0,
            )
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], weighted_offset_norm[b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_weighted_offset_norm.png"),
                H=H, W=W, cmap="viridis", title="DDLA weighted 2D offset norm",
            )
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], seed_valid_ratio[b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_valid_ratio.png"),
                H=H, W=W, cmap="plasma", title="DDLA token valid ratio", vmin=0.0, vmax=1.0,
            )

            M = token_sel_idx.shape[1]
            seed_ids = self._select_vis_seed_indices(seed_attn_entropy, weighted_offset_norm, seed_valid_ratio, b0, M)
            img_np = self._to_rgb_np(img[b0])
            patch_hw = (self.patch_size, self.patch_size)

            for rank, m_idx_t in enumerate(seed_ids):
                m_idx = int(m_idx_t.item())
                seed_uv = self._idx_to_uv(token_sel_idx[b0:b0 + 1, m_idx:m_idx + 1], H, W)[0, 0].detach().cpu().numpy()
                uv = cand_uv[b0, m_idx].detach().cpu().numpy()             # (L,2)
                pattn = pixel_attn[b0, m_idx].detach().float().cpu().numpy()# (L,)
                pvalid = pix_valid[b0, m_idx].detach().cpu().numpy().astype(bool)
                pprob = top_prob[b0, m_idx].detach().float().cpu().numpy()  # (L,Kz)
                tz = top_z[b0, m_idx].detach().float().cpu().numpy()        # (L,Kz)
                tattn = token_attn[b0, m_idx].detach().float().cpu().numpy()# (L,Kz)
                tvalid = token_valid[b0, m_idx].detach().cpu().numpy().astype(bool)

                # Overlay: local pixels colored by depth-aggregated attention.
                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                if np.any(pvalid):
                    plt.scatter(uv[pvalid, 0], uv[pvalid, 1], c=pattn[pvalid], s=35,
                                cmap="magma", vmin=0.0, vmax=max(float(pattn.max()), 1e-8), edgecolors="white", linewidths=0.4)
                if np.any(~pvalid):
                    plt.scatter(uv[~pvalid, 0], uv[~pvalid, 1], c="lightgray", marker="x", s=20)
                plt.scatter([seed_uv[0]], [seed_uv[1]], c="yellow", marker="*", s=120, edgecolors="black", linewidths=0.8)
                plt.axis("off")
                plt.title(f"DDLA seed {m_idx}: pixel attention")
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(self.vis_dir, f"{prefix}_seed{rank:02d}_overlay.png"), dpi=150)
                plt.close()

                # Heatmaps: pixel attention, top-k prob mass, valid, expected sampled depth.
                attn_grid = pattn.reshape(patch_hw)
                prob_mass_grid = pprob.sum(axis=-1).reshape(patch_hw)
                valid_grid = pvalid.astype(np.float32).reshape(patch_hw)
                z_exp = (pprob * tz).sum(axis=-1) / np.maximum(pprob.sum(axis=-1), 1e-8)
                z_grid = z_exp.reshape(patch_hw)

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                ims = []
                ims.append(axes[0].imshow(attn_grid, cmap="magma")); axes[0].set_title("pixel attn")
                ims.append(axes[1].imshow(prob_mass_grid, cmap="viridis", vmin=0.0, vmax=1.0)); axes[1].set_title("topK prob mass")
                ims.append(axes[2].imshow(valid_grid, cmap="gray", vmin=0.0, vmax=1.0)); axes[2].set_title("pixel valid")
                ims.append(axes[3].imshow(z_grid, cmap="plasma", vmin=self.min_depth, vmax=self.max_depth)); axes[3].set_title("E[z] topK")
                for ax, im in zip(axes, ims):
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(os.path.join(self.vis_dir, f"{prefix}_seed{rank:02d}_patch_grid.png"), dpi=150)
                plt.close()

                # Token bars: flattened token attention/prob/depth/valid.
                flat_attn = tattn.reshape(-1)
                flat_prob = pprob.reshape(-1)
                flat_z = tz.reshape(-1)
                flat_valid = tvalid.reshape(-1).astype(np.float32)
                x = np.arange(flat_attn.shape[0])
                fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
                axes[0].bar(x, flat_attn); axes[0].set_ylabel("attn")
                axes[1].bar(x, flat_prob); axes[1].set_ylabel("p(z|u)")
                axes[2].bar(x, flat_z); axes[2].set_ylabel("z")
                axes[3].bar(x, flat_valid); axes[3].set_ylabel("valid")
                axes[3].set_xlabel("flattened pixel-depth token")
                fig.suptitle(f"DDLA seed {m_idx}: token stats")
                plt.tight_layout()
                plt.savefig(os.path.join(self.vis_dir, f"{prefix}_seed{rank:02d}_token_bars.png"), dpi=150)
                plt.close()

                if self.save_npz:
                    np.savez_compressed(
                        os.path.join(self.vis_dir, f"{prefix}_seed{rank:02d}.npz"),
                        seed_index=np.array([m_idx], dtype=np.int64),
                        seed_uv=seed_uv.astype(np.float32),
                        patch_stride=np.array([float(patch_stride_bm[b0, m_idx].item())], dtype=np.float32),
                        patch_radius_px=np.array([float(patch_radius_px_bm[b0, m_idx].item())], dtype=np.float32),
                        cand_uv=uv.astype(np.float32),
                        pixel_attn=pattn.astype(np.float32),
                        token_attn=tattn.astype(np.float32),
                        top_prob=pprob.astype(np.float32),
                        top_z=tz.astype(np.float32),
                        pix_valid=pvalid.astype(np.bool_),
                        token_valid=tvalid.astype(np.bool_),
                    )
        except Exception as e:
            print(f"[DepthDisLocalAttnGrouping vis] failed at iter {self._vis_iter}: {repr(e)}")

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(
        self,
        seed_features: torch.Tensor,       # (B,Cs,M)
        token_sel_idx: torch.Tensor,       # (B,M)
        top_view_rot: torch.Tensor,        # (B,M,3,3)
        feat_map: torch.Tensor,            # (B,Cf,H,W)
        depth_prob: torch.Tensor,          # (B,D,H,W) or (B,1,N,D)
        K: torch.Tensor,                   # (B,3,3)
        end_points: Optional[dict] = None,
    ) -> torch.Tensor:
        B, Cs, M = seed_features.shape
        Bf, Cf, H, W = feat_map.shape
        if Bf != B:
            raise ValueError(f"Batch mismatch: seed B={B}, feat B={Bf}")
        if Cf != self.feat_dim:
            raise ValueError(f"feat_dim mismatch: expected {self.feat_dim}, got {Cf}")
        if Cs != self.seed_feature_dim:
            raise ValueError(f"seed_feature_dim mismatch: expected {self.seed_feature_dim}, got {Cs}")

        device = feat_map.device
        dtype = feat_map.dtype

        # Critical: by default DDLA treats depth distribution as a geometry prior,
        # not as a target to be optimized by grasp losses. This prevents grasp/operation
        # losses from collapsing the depth branch through this grouping module.
        depth_src = depth_prob.detach() if self.detach_depth_prob else depth_prob
        prob_map = self._prepare_depth_prob(depth_src, H, W).to(device=device, dtype=dtype)  # (B,D,H,W)
        if self.detach_depth_prob:
            prob_map = prob_map.detach()

        D = prob_map.shape[1]
        if self.depth_topk > D:
            raise ValueError(f"depth_topk={self.depth_topk} > D={D}")
        bin_centers = self._get_bin_centers(D, device, dtype)  # (D,)

        # Seed uv and seed depth statistics. Compute this before building the patch
        # so adaptive radius can be defined by a physical support size.
        seed_uv = self._idx_to_uv(token_sel_idx.long(), H, W)  # (B,M,2)
        seed_prob = self._sample_bchw_at_uv(prob_map, seed_uv[:, :, None, :])[:, :, 0, :]  # (B,M,D)
        seed_prob = seed_prob.clamp_min(1e-8)
        seed_prob = seed_prob / seed_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        seed_z_mean = (seed_prob * bin_centers.view(1, 1, D)).sum(dim=-1)  # (B,M)
        seed_entropy = self._safe_entropy(seed_prob, dim=-1)              # (B,M)
        seed_ray = self._ray_dirs_from_uv(seed_uv, K)                     # (B,M,3)
        seed_xyz = self._backproject_uvz(seed_uv, seed_z_mean, K)         # (B,M,3)

        # Build local patch uv. Non-adaptive mode uses a fixed pixel stride.
        # Adaptive mode keeps token count fixed but scales the support in pixels
        # according to a physical radius: r_px = fx * physical_radius / z_seed.
        r_grid = max(self.patch_size // 2, 1)
        if self.adaptive_radius:
            fx = K[:, 0, 0].view(B, 1).to(device=device, dtype=dtype)
            raw_radius_px = fx * float(self.physical_radius) / seed_z_mean.clamp_min(1e-6)
            patch_radius_px_bm = raw_radius_px.clamp(self.patch_radius_px_min, self.patch_radius_px_max)
            patch_stride_bm = patch_radius_px_bm / float(r_grid)
        else:
            patch_radius_px_bm = torch.full((B, M), float(self.patch_radius_px), device=device, dtype=dtype)
            patch_stride_bm = torch.full((B, M), float(self.patch_stride), device=device, dtype=dtype)

        offsets_unit = self.patch_offsets_unit.to(device=device, dtype=dtype)  # (L,2), integer grid offsets
        L = offsets_unit.shape[0]
        offsets_bml2 = offsets_unit.view(1, 1, L, 2) * patch_stride_bm[:, :, None, None]
        cand_uv = seed_uv[:, :, None, :] + offsets_bml2  # (B,M,L,2)
        pix_valid = (
            (cand_uv[..., 0] >= 0.0) & (cand_uv[..., 0] <= float(W - 1)) &
            (cand_uv[..., 1] >= 0.0) & (cand_uv[..., 1] <= float(H - 1))
        )  # (B,M,L)

        sampled_feat = self._sample_bchw_at_uv(feat_map, cand_uv)       # (B,M,L,Cf)
        sampled_prob = self._sample_bchw_at_uv(prob_map, cand_uv)       # (B,M,L,D)
        sampled_prob = sampled_prob.clamp_min(1e-8)
        sampled_prob = sampled_prob / sampled_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        top_prob, top_idx = torch.topk(sampled_prob, k=self.depth_topk, dim=-1)  # (B,M,L,Kz)
        top_z = bin_centers.index_select(0, top_idx.reshape(-1)).view(B, M, L, self.depth_topk)  # (B,M,L,Kz)
        top_prob_mass = top_prob.sum(dim=-1)  # (B,M,L)

        # Candidate ray/depth hypotheses.
        cand_uv_z = cand_uv[:, :, :, None, :].expand(-1, -1, -1, self.depth_topk, -1)  # (B,M,L,Kz,2)
        cand_xyz = self._backproject_uvz(cand_uv_z, top_z, K)                         # (B,M,L,Kz,3)
        rel_xyz = cand_xyz - seed_xyz[:, :, None, None, :]                            # (B,M,L,Kz,3)

        # Transform relative xyz into selected-view local coordinates.
        R = top_view_rot.float().to(device=device)                                    # (B,M,3,3)
        rel_local = torch.einsum("bmji,bmlkj->bmlki", R, rel_xyz.float()).to(dtype)   # R^T * rel if R columns are basis
        rel_local_norm = torch.tanh(rel_local / max(self.xyz_norm_scale, 1e-6))

        view_dir = F.normalize(R[..., :, 0].to(dtype), dim=-1, eps=1e-6)              # (B,M,3), approach axis convention
        cand_ray = self._ray_dirs_from_uv(cand_uv.view(B, M * L, 2), K).view(B, M, L, 3)
        ray_dot_view = (cand_ray[:, :, :, None, :] * view_dir[:, :, None, None, :]).sum(dim=-1).clamp(-1.0, 1.0)  # (B,M,L,1)
        ray_dot_view = ray_dot_view.expand(-1, -1, -1, self.depth_topk)              # (B,M,L,Kz)

        # Token valid mask. If a seed has no valid token, force center/top-1 valid as fallback.
        token_valid = pix_valid[:, :, :, None] & (top_prob > self.min_depth_prob)     # (B,M,L,Kz)
        token_valid_flat = token_valid.view(B, M, self.num_tokens)
        no_valid = ~token_valid_flat.any(dim=-1)
        if no_valid.any():
            token_valid_flat = token_valid_flat.clone()
            token_valid_flat[no_valid, 0] = True
            token_valid = token_valid_flat.view(B, M, L, self.depth_topk)

        # Build token input.
        sampled_feat_k = sampled_feat[:, :, :, None, :].expand(-1, -1, -1, self.depth_topk, -1)
        offset_norm = offsets_bml2[:, :, :, None, :].expand(B, M, -1, self.depth_topk, -1) / patch_radius_px_bm[:, :, None, None, None].clamp_min(1.0)
        rel_depth_norm = torch.tanh((top_z - seed_z_mean[:, :, None, None]) / max(self.depth_rel_norm_scale, 1e-6)).unsqueeze(-1)
        prob_feat = top_prob.unsqueeze(-1)
        ray_feat = ray_dot_view.unsqueeze(-1)
        valid_feat = token_valid.float().unsqueeze(-1).to(dtype)

        token_in = torch.cat([
            sampled_feat_k,
            offset_norm.to(dtype),
            rel_local_norm.to(dtype),
            rel_depth_norm.to(dtype),
            prob_feat.to(dtype),
            ray_feat.to(dtype),
            valid_feat,
        ], dim=-1)  # (B,M,L,Kz,Ctok)
        token_in = token_in.view(B, M, self.num_tokens, -1)
        token_hidden = self.token_mlp(token_in)  # (B,M,T,Hid)

        # Query.
        seed_feat_bmC = seed_features.transpose(1, 2).contiguous()  # (B,M,Cs)
        seed_z_norm = ((seed_z_mean - self.min_depth) / max(self.max_depth - self.min_depth, 1e-6)).clamp(0.0, 1.0)
        query_in = torch.cat([
            seed_feat_bmC,
            view_dir.to(dtype),
            seed_ray.to(dtype),
            seed_z_norm.unsqueeze(-1).to(dtype),
            seed_entropy.unsqueeze(-1).to(dtype),
        ], dim=-1)
        query_hidden = self.query_mlp(query_in)  # (B,M,Hid)

        # Multi-head single-query cross-attention.
        q = self.q_proj(query_hidden).view(B, M, self.num_heads, self.head_dim)       # (B,M,H,Dh)
        k = self.k_proj(token_hidden).view(B, M, self.num_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(token_hidden).view(B, M, self.num_tokens, self.num_heads, self.head_dim)
        scores = torch.einsum("bmhd,bmthd->bmht", q, k) / math.sqrt(float(self.head_dim))
        mask = token_valid_flat[:, :, None, :]  # (B,M,1,T)
        scores = scores.masked_fill(~mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        # Zero invalid and renormalize defensively.
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.einsum("bmht,bmthd->bmhd", attn, v).reshape(B, M, self.hidden_dim)

        context_out = self.context_proj(context)  # (B,M,out_dim)
        if self.use_seed_skip:
            op = context_out + self.seed_proj(seed_feat_bmC)
            op = self.out_norm(op)
            op = op + self.out_ffn(op)
        else:
            op = context_out
        group_features = op.transpose(1, 2).contiguous()  # (B,out_dim,M)

        # ------------------------------------------------------------------
        # Debug scalars
        # ------------------------------------------------------------------
        with torch.no_grad():
            attn_mean = attn.mean(dim=2)  # (B,M,T)
            entropy_raw = -(attn_mean.clamp_min(1e-8) * attn_mean.clamp_min(1e-8).log()).sum(dim=-1)
            attn_entropy = entropy_raw / math.log(max(self.num_tokens, 2))
            effective_tokens = torch.exp(entropy_raw)
            attn_max = attn_mean.max(dim=-1).values

            offset_norm_px = offsets_bml2.norm(dim=-1)[:, :, :, None].expand(B, M, L, self.depth_topk).reshape(B, M, self.num_tokens)
            weighted_offset_norm = (attn_mean * offset_norm_px).sum(dim=-1)
            token_rel_norm = rel_local.norm(dim=-1).reshape(B, M, self.num_tokens)
            weighted_rel_norm = (attn_mean * token_rel_norm).sum(dim=-1)
            weighted_depth_rel = (attn_mean * (top_z - seed_z_mean[:, :, None, None]).abs().reshape(B, M, self.num_tokens)).sum(dim=-1)
            seed_valid_ratio = token_valid_flat.float().mean(dim=-1)

            seed_norm = seed_feat_bmC.norm(dim=-1).mean()
            op_norm = group_features.transpose(1, 2).norm(dim=-1).mean()
            op_seed_ratio = op_norm / seed_norm.clamp_min(1e-6)

            if end_points is not None:
                end_points["D: DDLA patch size"] = torch.tensor(float(self.patch_size), device=device)
                end_points["D: DDLA depth detached"] = torch.tensor(float(self.detach_depth_prob), device=device)
                end_points["D: DDLA adaptive radius"] = torch.tensor(float(self.adaptive_radius), device=device)
                end_points["D: DDLA physical radius m"] = torch.tensor(float(self.physical_radius), device=device)
                end_points["D: DDLA patch stride"] = patch_stride_bm.mean().reshape(())
                end_points["D: DDLA patch radius px"] = patch_radius_px_bm.mean().reshape(())
                end_points["D: DDLA patch radius min"] = patch_radius_px_bm.min().reshape(())
                end_points["D: DDLA patch radius max"] = patch_radius_px_bm.max().reshape(())
                end_points["D: DDLA local pixels"] = torch.tensor(float(self.num_local_pixels), device=device)
                end_points["D: DDLA depth topk"] = torch.tensor(float(self.depth_topk), device=device)
                end_points["D: DDLA tokens"] = torch.tensor(float(self.num_tokens), device=device)
                end_points["D: DDLA pixel valid ratio"] = pix_valid.float().mean().reshape(())
                end_points["D: DDLA token valid ratio"] = token_valid_flat.float().mean().reshape(())
                end_points["D: DDLA topk prob mass"] = top_prob_mass.mean().reshape(())
                end_points["D: DDLA top prob mean"] = top_prob.mean().reshape(())
                end_points["D: DDLA seed depth mean"] = seed_z_mean.mean().reshape(())
                end_points["D: DDLA seed entropy"] = seed_entropy.mean().reshape(())
                end_points["D: DDLA attn entropy"] = attn_entropy.mean().reshape(())
                end_points["D: DDLA effective tokens"] = effective_tokens.mean().reshape(())
                end_points["D: DDLA attn max"] = attn_max.mean().reshape(())
                end_points["D: DDLA weighted offset px"] = weighted_offset_norm.mean().reshape(())
                end_points["D: DDLA weighted rel xyz"] = weighted_rel_norm.mean().reshape(())
                end_points["D: DDLA weighted |dz|"] = weighted_depth_rel.mean().reshape(())
                end_points["D: DDLA op feat norm"] = op_norm.reshape(())
                end_points["D: DDLA seed feat norm"] = seed_norm.reshape(())
                end_points["D: DDLA op/seed norm"] = op_seed_ratio.reshape(())

            if (self.debug_print_every > 0) and (self._vis_iter % self.debug_print_every == 0):
                print(
                    f"[DepthDisLocalAttnGrouping] it={self._vis_iter} "
                    f"detachD={int(self.detach_depth_prob)} adaptR={int(self.adaptive_radius)} "
                    f"stride={patch_stride_bm.mean().item():.1f} rad={patch_radius_px_bm.mean().item():.1f}px "
                    f"pix_valid={pix_valid.float().mean().item():.3f} "
                    f"tok_valid={token_valid_flat.float().mean().item():.3f} "
                    f"topk_mass={top_prob_mass.mean().item():.3f} "
                    f"seed_H={seed_entropy.mean().item():.3f} "
                    f"attn_H={attn_entropy.mean().item():.3f} "
                    f"effT={effective_tokens.mean().item():.1f}/{self.num_tokens} "
                    f"attn_max={attn_max.mean().item():.3f} "
                    f"w_off={weighted_offset_norm.mean().item():.2f}px "
                    f"w_rel={weighted_rel_norm.mean().item():.4f}m "
                    f"w_dz={weighted_depth_rel.mean().item():.4f}m "
                    f"op/seed={op_seed_ratio.item():.3f}"
                )

            if self.vis_dir is not None:
                pixel_attn = attn_mean.view(B, M, L, self.depth_topk).sum(dim=-1)  # (B,M,L)
                self._maybe_visualize(
                    end_points=end_points,
                    token_sel_idx=token_sel_idx,
                    cand_uv=cand_uv,
                    pixel_attn=pixel_attn,
                    token_attn=attn_mean.view(B, M, L, self.depth_topk),
                    top_prob=top_prob,
                    top_z=top_z,
                    pix_valid=pix_valid,
                    token_valid=token_valid,
                    seed_attn_entropy=attn_entropy,
                    weighted_offset_norm=weighted_offset_norm,
                    seed_valid_ratio=seed_valid_ratio,
                    patch_radius_px_bm=patch_radius_px_bm,
                    patch_stride_bm=patch_stride_bm,
                )

        self._vis_iter += 1
        return group_features

 
def get_model(cfgs):
    return economicgrasp_dpt(
        is_training=False,
        vis_dir=getattr(cfgs, "save_dir", None),
        vis_every=500,
    )


class MetricRegionCropGrouping(nn.Module):
    """
    E3GNet-style depth-adaptive local region crop for economicgrasp_dpt.

    Changes in this version:
      1) Removed all chunk-based computation. The whole B x M seed set is processed once.
      2) Added use_view_conditioned_pool flag:
           - False: valid-aware max pooling over ROI patch features.
           - True:  seed+view conditioned attention pooling over ROI patch features.
      3) Concatenated valid mask into patch_encoder input.
      4) Removed group_delta/residual_fusion. Final grouped feature is a simple weighted sum:
           grouped = norm(seed_scale * seed_ctx + roi_scale * pooled + view_scale * view_ctx)
      5) Added debug scalars in end_points and optional rank-0 print logs.
      6) Visualization panel now includes pool attention map when view-conditioned pooling is enabled.

    Input:
      seed_features:      (B,C,M)
      token_sel_idx:      (B,M), flattened index on source image/depth map
      seed_xyz:           (B,M,3), used for seed depth / radius scale
      top_view_rot:       (B,M,3,3), selected/GT view rotation used as conditioning
      feat_map:           (B,C,Hf,Wf), dense DPT/DINO feature map
      depth_map:          (B,1,H,W)
      depth_prob:         (B,D,H,W), optional, used only for uncertainty visualization/debug
      objectness_logits:  (B,2,H,W), optional, used only for visualization/debug
      graspness_map:      (B,1,H,W), optional, used only for visualization/debug
      K:                  (B,3,3), resized/cropped image intrinsics

    Output:
      grouped feature:    (B,256,M), compatible with Grasp_Head_Local_Interaction.
    """

    def __init__(
        self,
        seed_feature_dim: int = 128,
        feat_dim: int = 128,
        out_dim: int = 256,
        hidden_dim: int = 128,
        patch_size: int = 12,
        metric_radius: float = 0.08,
        radius_px_min: float = 8.0,
        radius_px_max: float = 64.0,
        train_scale_min: float = 0.80,
        train_scale_max: float = 1.25,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        depth_norm_scale: float = 0.08,
        detach_depth: bool = True,
        detach_aux_maps: bool = True,
        # pooling / fusion
        use_view_conditioned_pool: bool = False,
        # visualization / debug
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        vis_num_seeds: int = 4,
        vis_seed_mode: str = "first",  # first | large_radius | high_uncert | random
        vis_dpi: int = 150,
        save_npz: bool = True,
        debug_print_every: int = 0,
        debug_rank0_only: bool = True,
    ):
        super().__init__()
        self.seed_feature_dim = int(seed_feature_dim)
        self.feat_dim = int(feat_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.metric_radius = float(metric_radius)
        self.radius_px_min = float(radius_px_min)
        self.radius_px_max = float(radius_px_max)
        self.train_scale_min = float(train_scale_min)
        self.train_scale_max = float(train_scale_max)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.depth_norm_scale = float(depth_norm_scale)
        self.detach_depth = bool(detach_depth)
        self.detach_aux_maps = bool(detach_aux_maps)
        self.use_view_conditioned_pool = bool(use_view_conditioned_pool)

        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.vis_num_seeds = int(vis_num_seeds)
        self.vis_seed_mode = str(vis_seed_mode)
        self.vis_dpi = int(vis_dpi)
        self.save_npz = bool(save_npz)
        self.debug_print_every = int(debug_print_every)
        self.debug_rank0_only = bool(debug_rank0_only)
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        P = self.patch_size
        lin = torch.linspace(-1.0, 1.0, P, dtype=torch.float32)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        base_xy = torch.stack([xx, yy], dim=0)  # (2,P,P), x/u first, y/v second
        self.register_buffer("base_xy", base_xy, persistent=False)

        # aux channels used by the train path:
        #   local x/y (2), depth residual (1), valid mask (1)
        aux_dim = 4
        in_dim = self.feat_dim + aux_dim

        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True),
        )

        self.seed_proj = nn.Sequential(
            nn.Linear(seed_feature_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.view_mlp = nn.Sequential(
            nn.Linear(9, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

        if self.use_view_conditioned_pool:
            self.view_pool_score = nn.Sequential(
                nn.Conv2d(out_dim * 2, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
            )

            self.view_pool_norm = nn.LayerNorm(out_dim)

        self._iter = 0

    # ------------------------------------------------------------------
    # geometry / sampling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _idx_to_uv(idx_bm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()
        return torch.stack([u, v], dim=-1)  # (B,M,2)

    @staticmethod
    def _gather_depth(depth_b1hw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth_b1hw.shape
        flat = depth_b1hw[:, 0].reshape(B, H * W)
        return torch.gather(flat, 1, idx_bm).unsqueeze(-1)  # (B,M,1)

    @staticmethod
    def _sample_bmpp(
        map_bchw: torch.Tensor,
        grid_u: torch.Tensor,
        grid_v: torch.Tensor,
        src_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        map_bchw: (B,C,Hm,Wm)
        grid_u/v: (B,M,P,P), coordinates in src_hw image coordinate system.
        return:   (B,C,M,P,P)
        """
        B, C, Hm, Wm = map_bchw.shape
        Hsrc, Wsrc = src_hw
        M, P = grid_u.shape[1], grid_u.shape[2]

        if Wsrc > 1:
            u_m = grid_u * (float(Wm - 1) / float(Wsrc - 1))
        else:
            u_m = grid_u
        if Hsrc > 1:
            v_m = grid_v * (float(Hm - 1) / float(Hsrc - 1))
        else:
            v_m = grid_v

        gx = 2.0 * u_m / float(max(Wm - 1, 1)) - 1.0 if Wm > 1 else torch.zeros_like(u_m)
        gy = 2.0 * v_m / float(max(Hm - 1, 1)) - 1.0 if Hm > 1 else torch.zeros_like(v_m)

        grid = torch.stack([gx, gy], dim=-1).view(B, M * P, P, 2)
        sampled = F.grid_sample(
            map_bchw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B,C,M*P,P)
        return sampled.view(B, C, M, P, P).contiguous()

    @staticmethod
    def _sample_one_chw(
        map_chw: torch.Tensor,
        grid_u_pp: torch.Tensor,
        grid_v_pp: torch.Tensor,
        src_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Sample one CHW map on one P x P grid. Returns C x P x P."""
        if map_chw.dim() == 2:
            map_chw = map_chw.unsqueeze(0)
        map_bchw = map_chw.unsqueeze(0)
        grid_u = grid_u_pp.view(1, 1, *grid_u_pp.shape)
        grid_v = grid_v_pp.view(1, 1, *grid_v_pp.shape)
        return MetricRegionCropGrouping._sample_bmpp(map_bchw, grid_u, grid_v, src_hw=src_hw)[0, :, 0]

    @staticmethod
    def _depth_uncertainty(depth_prob_bdhw: torch.Tensor) -> torch.Tensor:
        D = max(int(depth_prob_bdhw.shape[1]), 2)
        prob = depth_prob_bdhw.clamp_min(1e-8)
        entropy = -(prob * prob.log()).sum(dim=1, keepdim=True) / math.log(D)
        return entropy.clamp(0.0, 1.0)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.float()
        mask_f = mask.float()
        return (x * mask_f).sum() / (mask_f.sum() + eps)

    def _make_objectness_prob(self, objectness_logits: Optional[torch.Tensor], depth_map: torch.Tensor) -> torch.Tensor:
        if objectness_logits is None:
            return torch.zeros_like(depth_map)
        return F.softmax(objectness_logits, dim=1)[:, 1:2]

    # ------------------------------------------------------------------
    # pooling
    # ------------------------------------------------------------------
    def _valid_aware_max_pool(
        self,
        patch_feat: torch.Tensor,    # (B*M,C,P,P)
        valid_flat: torch.Tensor,    # (B*M,1,P,P), bool
        B: int,
        M: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        patch_masked = patch_feat.masked_fill(~valid_flat, -1e4)
        pooled = patch_masked.flatten(2).max(dim=-1).values  # (BM,C)
        valid_tok = valid_flat.flatten(2).squeeze(1).bool()  # (BM,S)
        empty = ~valid_tok.any(dim=-1)                       # (BM,)
        pooled = torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)

        # For logging only: max pooling is equivalent to hard one-hot selection per channel,
        # so we report approximate token entropy as zero and max probability as one.
        entropy = torch.zeros((), device=patch_feat.device)
        maxprob = torch.ones((), device=patch_feat.device)
        return pooled.view(B, M, self.out_dim), None, entropy, maxprob

    def _view_conditioned_pool(
        self,
        patch_feat: torch.Tensor,  # (B*M, C, P, P)
        valid_flat: torch.Tensor,  # (B*M, 1, P, P), bool
        view_ctx: torch.Tensor,    # (B, M, C)
        B: int,
        M: int,
    ):
        BM, C, P, _ = patch_feat.shape
        S = P * P

        # Broadcast selected-view feature to every ROI cell.
        view_bmcpp = view_ctx.reshape(BM, C, 1, 1).to(patch_feat.dtype)
        view_bmcpp = view_bmcpp.expand(-1, -1, P, P)

        # Spatial logits from [patch feature, view feature].
        score_in = torch.cat([patch_feat, view_bmcpp], dim=1)  # (BM,2C,P,P)
        logits = self.view_pool_score(score_in).reshape(BM, S) # (BM,S)

        valid_tok = valid_flat.reshape(BM, S).bool()
        logits = logits.masked_fill(~valid_tok, -1e4)

        attn = F.softmax(logits, dim=-1)
        attn = attn * valid_tok.float()
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        tokens = patch_feat.flatten(2).transpose(1, 2).contiguous()  # (BM,S,C)

        # Use original patch_feat as value, no value projection.
        pooled = (attn.unsqueeze(-1) * tokens).sum(dim=1)  # (BM,C)

        empty = ~valid_tok.any(dim=-1)
        pooled = torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)

        pooled = self.view_pool_norm(pooled)
        pooled = pooled.view(B, M, C)

        # Debug stats.
        p = attn.clamp_min(1e-8)
        pool_entropy = -(p * p.log()).sum(dim=-1) / math.log(max(S, 2))
        pool_maxprob = attn.max(dim=-1).values

        pool_attn_map = attn.view(B, M, P, P)

        return pooled, pool_attn_map, pool_entropy.view(B, M), pool_maxprob.view(B, M)

    # ------------------------------------------------------------------
    # visualization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().float().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _normalize_img_chw(img_chw: torch.Tensor) -> np.ndarray:
        x = img_chw.detach().float().cpu()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            arr = x[0].numpy()
            finite = np.isfinite(arr)
            if finite.sum() == 0:
                return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
            lo, hi = np.nanpercentile(arr[finite], [1, 99])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return np.repeat(arr[..., None], 3, axis=-1)
        x = x[:3]
        x = x - x.amin(dim=(1, 2), keepdim=True)
        x = x / (x.amax(dim=(1, 2), keepdim=True) + 1e-6)
        return x.permute(1, 2, 0).numpy()

    def _background_np(self, end_points: Optional[dict], depth_map: torch.Tensor, b: int) -> Tuple[np.ndarray, str]:
        if end_points is not None and "img" in end_points and torch.is_tensor(end_points["img"]):
            img = end_points["img"]
            if img.dim() == 4 and img.size(0) > b:
                return self._normalize_img_chw(img[b]), "rgb"
        return self._normalize_img_chw(depth_map[b, 0]), "depth"

    def _save_seed_radius_overlay(
        self,
        bg: np.ndarray,
        uv_bm2: torch.Tensor,
        radius_px_bm: torch.Tensor,
        out_path: str,
        H: int,
        W: int,
        selected: Optional[List[int]] = None,
    ):
        uv = self._to_numpy(uv_bm2)
        r = self._to_numpy(radius_px_bm)
        finite = np.isfinite(r) & np.isfinite(uv).all(axis=-1)
        if finite.sum() == 0:
            return
        plt.figure(figsize=(7, 7))
        plt.imshow(bg)
        sc = plt.scatter(uv[finite, 0], uv[finite, 1], c=r[finite], s=8, cmap="viridis")
        plt.colorbar(sc, fraction=0.046, pad=0.04, label="crop radius (px)")
        if selected is not None:
            for m in selected:
                if 0 <= m < uv.shape[0]:
                    plt.scatter([uv[m, 0]], [uv[m, 1]], s=90, marker="*", c="yellow", edgecolors="black", linewidths=0.6)
        plt.xlim(0, W - 1)
        plt.ylim(H - 1, 0)
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_crop_overlay(
        self,
        bg: np.ndarray,
        seed_uv: np.ndarray,
        grid_u: np.ndarray,
        grid_v: np.ndarray,
        valid: np.ndarray,
        out_path: str,
        title: str,
    ):
        plt.figure(figsize=(7, 7))
        plt.imshow(bg)
        boundary_u = np.concatenate([grid_u[0, :], grid_u[:, -1], grid_u[-1, ::-1], grid_u[::-1, 0]])
        boundary_v = np.concatenate([grid_v[0, :], grid_v[:, -1], grid_v[-1, ::-1], grid_v[::-1, 0]])
        plt.plot(boundary_u, boundary_v, color="yellow", linewidth=1.2)
        step = max(grid_u.shape[0] // 6, 1)
        gu = grid_u[::step, ::step].reshape(-1)
        gv = grid_v[::step, ::step].reshape(-1)
        vv = valid[::step, ::step].reshape(-1).astype(np.float32)
        plt.scatter(gu, gv, c=vv, s=18, cmap="RdYlGn", vmin=0.0, vmax=1.0, edgecolors="black", linewidths=0.25)
        plt.scatter([seed_uv[0]], [seed_uv[1]], s=100, marker="*", c="cyan", edgecolors="black", linewidths=0.7)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_roi_panel(
        self,
        roi_rgb: np.ndarray,
        depth_rel: np.ndarray,
        uncert: np.ndarray,
        obj: np.ndarray,
        grasp: np.ndarray,
        valid: np.ndarray,
        feat_norm: np.ndarray,
        pool_attn: Optional[np.ndarray],
        out_path: str,
        title: str,
    ):
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        ax = axes.reshape(-1)
        ax[0].imshow(roi_rgb)
        ax[0].set_title("RGB crop")

        im1 = ax[1].imshow(depth_rel, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax[1].set_title("tanh((D-z)/scale)")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(uncert, vmin=0.0, vmax=1.0, cmap="magma")
        ax[2].set_title("depth uncertainty")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        im3 = ax[3].imshow(obj, vmin=0.0, vmax=1.0, cmap="viridis")
        ax[3].set_title("objectness prob")
        fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        im4 = ax[4].imshow(grasp, vmin=0.0, vmax=1.0, cmap="viridis")
        ax[4].set_title("graspness")
        fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

        im5 = ax[5].imshow(valid.astype(np.float32), vmin=0.0, vmax=1.0, cmap="gray")
        ax[5].set_title("valid mask")
        fig.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

        finite = np.isfinite(feat_norm)
        if finite.sum() > 0:
            vmin = float(np.nanpercentile(feat_norm[finite], 1))
            vmax = float(np.nanpercentile(feat_norm[finite], 99))
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        im6 = ax[6].imshow(feat_norm, vmin=vmin, vmax=vmax, cmap="plasma")
        ax[6].set_title("DPT feat norm")
        fig.colorbar(im6, ax=ax[6], fraction=0.046, pad=0.04)

        if pool_attn is not None:
            attn = pool_attn.astype(np.float32)
            im7 = ax[7].imshow(attn, vmin=0.0, vmax=max(float(np.nanmax(attn)), 1e-6), cmap="inferno")
            ax[7].set_title("view-pool attn")
            fig.colorbar(im7, ax=ax[7], fraction=0.046, pad=0.04)
        else:
            ax[7].axis("off")
            ax[7].set_title("max pool")

        for a in ax[:7]:
            a.set_xticks([])
            a.set_yticks([])
        if pool_attn is not None:
            ax[7].set_xticks([])
            ax[7].set_yticks([])
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _choose_vis_seed_indices(
        self,
        token_sel_idx_bm: torch.Tensor,
        radius_px_bm: torch.Tensor,
        uncert_map: torch.Tensor,
        b: int,
    ) -> List[int]:
        M = token_sel_idx_bm.shape[0]
        device = token_sel_idx_bm.device
        if self.vis_seed_mode == "random":
            order = torch.randperm(M, device=device)
        elif self.vis_seed_mode == "large_radius":
            order = torch.argsort(radius_px_bm.detach(), descending=True)
        elif self.vis_seed_mode == "high_uncert":
            flat_uncert = uncert_map[b, 0].reshape(-1)
            score = torch.gather(flat_uncert, 0, token_sel_idx_bm.long()).detach()
            order = torch.argsort(score, descending=True)
        else:
            order = torch.arange(M, device=device)
        return order[:max(self.vis_num_seeds, 0)].detach().cpu().tolist()

    @torch.no_grad()
    def _maybe_visualize(
        self,
        end_points: Optional[dict],
        token_sel_idx: torch.Tensor,       # (B,M)
        seed_xyz: Optional[torch.Tensor],  # (B,M,3)
        radius_px_vis: torch.Tensor,       # (B,M)
        feat_map: torch.Tensor,
        depth_map: torch.Tensor,
        uncert_map: torch.Tensor,
        obj_prob: torch.Tensor,
        graspness_map: torch.Tensor,
        pool_attn_map: Optional[torch.Tensor],  # (B,M,P,P) or None
    ):
        if self.vis_dir is None or self.vis_every <= 0 or (self._iter % self.vis_every) != 0:
            return
        try:
            B, _, H, W = depth_map.shape
            b = 0
            bg, bg_name = self._background_np(end_points, depth_map, b)
            uv = self._idx_to_uv(token_sel_idx, H, W)
            seed_ids = self._choose_vis_seed_indices(
                token_sel_idx_bm=token_sel_idx[b],
                radius_px_bm=radius_px_vis[b],
                uncert_map=uncert_map,
                b=b,
            )

            prefix = f"lrc_it{self._iter:06d}_b{b}"
            self._save_seed_radius_overlay(
                bg=bg,
                uv_bm2=uv[b],
                radius_px_bm=radius_px_vis[b],
                out_path=os.path.join(self.vis_dir, f"{prefix}_seed_radius_overlay.png"),
                H=H,
                W=W,
                selected=seed_ids,
            )

            img = None
            if end_points is not None and "img" in end_points and torch.is_tensor(end_points["img"]):
                img = end_points["img"]

            feat_norm_map = feat_map.detach().float().norm(dim=1, keepdim=True)
            base_xy = self.base_xy.to(device=depth_map.device, dtype=depth_map.dtype)

            for local_rank, m in enumerate(seed_ids):
                seed_uv = uv[b, m].detach().float()
                r = float(radius_px_vis[b, m].detach().float().cpu())
                z = float(seed_xyz[b, m, 2].detach().float().cpu()) if seed_xyz is not None else float("nan")
                rx = torch.tensor(r, device=depth_map.device, dtype=depth_map.dtype)
                ry = torch.tensor(r, device=depth_map.device, dtype=depth_map.dtype)
                grid_u = seed_uv[0].to(depth_map.dtype) + rx * base_xy[0]
                grid_v = seed_uv[1].to(depth_map.dtype) + ry * base_xy[1]
                valid = (grid_u >= 0.0) & (grid_u <= float(W - 1)) & (grid_v >= 0.0) & (grid_v <= float(H - 1))

                roi_depth = self._sample_one_chw(depth_map[b], grid_u, grid_v, src_hw=(H, W))[0]
                z_ref = seed_xyz[b, m, 2].detach()
                depth_rel = torch.tanh((roi_depth - z_ref) / max(self.depth_norm_scale, 1e-6))
                roi_uncert = self._sample_one_chw(uncert_map[b], grid_u, grid_v, src_hw=(H, W))[0]
                roi_obj = self._sample_one_chw(obj_prob[b], grid_u, grid_v, src_hw=(H, W))[0]
                roi_grasp = self._sample_one_chw(graspness_map[b], grid_u, grid_v, src_hw=(H, W))[0]
                roi_feat_norm = self._sample_one_chw(feat_norm_map[b], grid_u, grid_v, src_hw=(H, W))[0]

                if img is not None:
                    roi_img_chw = self._sample_one_chw(img[b].detach(), grid_u, grid_v, src_hw=(H, W))
                    roi_rgb = self._normalize_img_chw(roi_img_chw)
                else:
                    roi_rgb = self._normalize_img_chw(roi_depth)

                pool_attn_np = None
                if pool_attn_map is not None:
                    pool_attn_np = self._to_numpy(pool_attn_map[b, m])

                grid_u_np = self._to_numpy(grid_u)
                grid_v_np = self._to_numpy(grid_v)
                valid_np = self._to_numpy(valid).astype(bool)
                seed_uv_np = self._to_numpy(seed_uv)
                depth_rel_np = self._to_numpy(depth_rel)
                uncert_np = self._to_numpy(roi_uncert)
                obj_np = self._to_numpy(roi_obj)
                grasp_np = self._to_numpy(roi_grasp)
                feat_norm_np = self._to_numpy(roi_feat_norm)

                view_text = ""
                if end_points is not None and "grasp_top_view_inds" in end_points and torch.is_tensor(end_points["grasp_top_view_inds"]):
                    vinds = end_points["grasp_top_view_inds"]
                    if vinds.dim() == 2 and vinds.size(0) > b and vinds.size(1) > m:
                        view_text = f", view={int(vinds[b, m].detach().cpu())}"

                pool_text = "view-attn" if pool_attn_np is not None else "max"
                title = (
                    f"seed m={m}, pix=({seed_uv_np[0]:.1f},{seed_uv_np[1]:.1f}), "
                    f"r={r:.1f}px, z={z:.3f}, valid={valid_np.mean():.2f}, pool={pool_text}{view_text}"
                )
                stem = f"{prefix}_seed{local_rank:02d}_m{m:04d}"
                self._save_crop_overlay(
                    bg=bg,
                    seed_uv=seed_uv_np,
                    grid_u=grid_u_np,
                    grid_v=grid_v_np,
                    valid=valid_np,
                    out_path=os.path.join(self.vis_dir, f"{stem}_overlay_box.png"),
                    title=f"{title} [{bg_name}]",
                )
                self._save_roi_panel(
                    roi_rgb=roi_rgb,
                    depth_rel=depth_rel_np,
                    uncert=uncert_np,
                    obj=obj_np,
                    grasp=grasp_np,
                    valid=valid_np,
                    feat_norm=feat_norm_np,
                    pool_attn=pool_attn_np,
                    out_path=os.path.join(self.vis_dir, f"{stem}_roi_panel.png"),
                    title=title,
                )
                if self.save_npz:
                    npz_kwargs = dict(
                        seed_uv=seed_uv_np,
                        radius_px=np.asarray(r, dtype=np.float32),
                        seed_z=np.asarray(z, dtype=np.float32),
                        grid_u=grid_u_np,
                        grid_v=grid_v_np,
                        valid=valid_np.astype(np.float32),
                        depth_rel=depth_rel_np,
                        uncert=uncert_np,
                        obj=obj_np,
                        grasp=grasp_np,
                        feat_norm=feat_norm_np,
                    )
                    if pool_attn_np is not None:
                        npz_kwargs["pool_attn"] = pool_attn_np
                    np.savez_compressed(os.path.join(self.vis_dir, f"{stem}.npz"), **npz_kwargs)
        except Exception as e:
            print(f"[MetricRegionCropGrouping vis] failed at iter={self._iter}: {repr(e)}")

    # ------------------------------------------------------------------
    # debug helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_rank0() -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _maybe_debug_print(self, stats: dict):
        if self.debug_print_every <= 0:
            return
        if self._iter % self.debug_print_every != 0:
            return
        if self.debug_rank0_only and not self._is_rank0():
            return
        try:
            msg = (
                f"[MetricRegionCropGrouping] it={self._iter} "
                f"view_pool={int(self.use_view_conditioned_pool)} "
                f"valid={float(stats['valid_ratio']):.3f} "
                f"empty={float(stats['empty_ratio']):.3f} "
                f"r_mean={float(stats['radius_px_mean']):.1f} "
                f"r_min={float(stats['radius_px_min']):.1f} "
                f"r_max={float(stats['radius_px_max']):.1f} "
                f"dz_abs={float(stats['depth_abs_mean']):.3f} "
                f"pool_ent={float(stats['pool_entropy']):.3f} "
                f"pool_maxp={float(stats['pool_maxprob']):.3f} "
            )
            print(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        seed_features: torch.Tensor,       # (B,C,M)
        token_sel_idx: torch.Tensor,       # (B,M)
        seed_xyz: torch.Tensor,            # (B,M,3)
        top_view_rot: torch.Tensor,        # (B,M,3,3)
        feat_map: torch.Tensor,            # (B,C,Hf,Wf)
        depth_map: torch.Tensor,           # (B,1,H,W)
        depth_prob: torch.Tensor = None,   # (B,D,H,W)
        objectness_logits: torch.Tensor = None,
        graspness_map: torch.Tensor = None,
        K: torch.Tensor = None,
        end_points: dict = None,
    ) -> torch.Tensor:
        B, Cseed, M = seed_features.shape
        _, Cf, _, _ = feat_map.shape
        _, _, H, W = depth_map.shape
        device = seed_features.device
        dtype = feat_map.dtype
        P = self.patch_size

        if K is None:
            raise ValueError("MetricRegionCropGrouping requires K with shape (B,3,3).")
        if Cf != self.feat_dim:
            raise ValueError(f"MetricRegionCropGrouping expected feat_dim={self.feat_dim}, got feat_map channels={Cf}.")

        # Aux maps are detached by default so local grasp loss does not destabilize depth/proposal heads.
        depth_for_aux = depth_map.detach() if self.detach_depth else depth_map
        if depth_prob is not None:
            depth_prob_for_aux = depth_prob.detach() if self.detach_depth else depth_prob
            uncert_map = self._depth_uncertainty(depth_prob_for_aux)
        else:
            uncert_map = torch.zeros_like(depth_for_aux)

        obj_prob = self._make_objectness_prob(objectness_logits, depth_for_aux)
        if graspness_map is None:
            graspness_map = torch.zeros_like(depth_for_aux)
        if self.detach_aux_maps:
            obj_prob = obj_prob.detach()
            graspness_map = graspness_map.detach()
            uncert_map = uncert_map.detach()

        uv = self._idx_to_uv(token_sel_idx, H, W)  # (B,M,2)
        if seed_xyz is not None:
            z_seed_all = seed_xyz[..., 2:3]
        else:
            z_seed_all = self._gather_depth(depth_for_aux, token_sel_idx)
        z_seed_all = torch.nan_to_num(z_seed_all, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-4)
        z_seed_for_radius = z_seed_all.detach()

        fx = K[:, 0, 0].view(B, 1)
        fy = K[:, 1, 1].view(B, 1)

        z_base = z_seed_for_radius[..., 0].clamp_min(1e-4)
        rx = fx.to(dtype) * self.metric_radius / z_base.to(dtype)
        ry = fy.to(dtype) * self.metric_radius / z_base.to(dtype)
        if self.training and self.train_scale_max > self.train_scale_min:
            scale = torch.empty(B, M, device=device, dtype=dtype).uniform_(self.train_scale_min, self.train_scale_max)
            rx = rx * scale
            ry = ry * scale
        rx = rx.clamp(self.radius_px_min, self.radius_px_max)
        ry = ry.clamp(self.radius_px_min, self.radius_px_max)

        with torch.no_grad():
            rx_vis = (fx * self.metric_radius / z_base).clamp(self.radius_px_min, self.radius_px_max)
            ry_vis = (fy * self.metric_radius / z_base).clamp(self.radius_px_min, self.radius_px_max)
            radius_px_vis = ((rx_vis + ry_vis) * 0.5).detach()

        base_xy = self.base_xy.to(device=device, dtype=dtype)  # (2,P,P)
        local_xy = base_xy.view(1, 2, 1, P, P)                 # (1,2,1,P,P)
        uv_d = uv.to(dtype)

        grid_u = uv_d[..., 0].view(B, M, 1, 1) + rx.view(B, M, 1, 1) * base_xy[0].view(1, 1, P, P)
        grid_v = uv_d[..., 1].view(B, M, 1, 1) + ry.view(B, M, 1, 1) * base_xy[1].view(1, 1, P, P)

        valid = (
            (grid_u >= 0.0) & (grid_u <= float(W - 1)) &
            (grid_v >= 0.0) & (grid_v <= float(H - 1))
        )  # (B,M,P,P)
        valid_f = valid.to(dtype).unsqueeze(1)  # (B,1,M,P,P)

        roi_feat = self._sample_bmpp(feat_map, grid_u, grid_v, src_hw=(H, W)).to(dtype)          # (B,C,M,P,P)
        roi_depth = self._sample_bmpp(depth_for_aux, grid_u, grid_v, src_hw=(H, W)).to(dtype)    # (B,1,M,P,P)

        z_c_b1 = z_seed_all[..., 0].to(dtype).view(B, 1, M, 1, 1)
        depth_residual = torch.tanh((roi_depth - z_c_b1) / max(self.depth_norm_scale, 1e-6))
        local_xy_b = local_xy.expand(B, -1, M, -1, -1)

        # Mask invalid cells before conv and also expose the valid mask as an input channel.
        roi_feat = roi_feat * valid_f
        local_xy_b = local_xy_b * valid_f
        depth_residual = depth_residual * valid_f

        roi = torch.cat([
            roi_feat,
            local_xy_b,
            depth_residual,
            valid_f,
        ], dim=1)  # (B,C+4,M,P,P)
        roi = torch.nan_to_num(roi, nan=0.0, posinf=0.0, neginf=0.0)
        roi = roi.permute(0, 2, 1, 3, 4).reshape(B * M, Cf + 4, P, P).contiguous()

        patch_feat = self.patch_encoder(roi)  # (B*M,256,P,P)
        valid_flat = valid.reshape(B * M, 1, P, P)

        seed_ctx = self.seed_proj(seed_features.transpose(1, 2).contiguous())  # (B,M,256)
        view_ctx = self.view_mlp(top_view_rot.reshape(B, M, 9).to(seed_ctx.dtype))  # (B,M,256)

        if self.use_view_conditioned_pool:
            pooled, pool_attn_map, pool_entropy, pool_maxprob = self._view_conditioned_pool(
                patch_feat=patch_feat,
                valid_flat=valid_flat,
                view_ctx=view_ctx,
                B=B,
                M=M,
            )
        else:
            pooled, pool_attn_map, pool_entropy, pool_maxprob = self._valid_aware_max_pool(
                patch_feat=patch_feat,
                valid_flat=valid_flat,
                B=B,
                M=M,
            )

        grouped = self.output_mlp(
            torch.cat([seed_ctx, pooled.to(seed_ctx.dtype), view_ctx], dim=-1)
        ).transpose(1, 2).contiguous()
            
        # Debug stats.
        with torch.no_grad():
            valid_ratio = valid.float().mean()
            empty_ratio = (~valid.flatten(2).any(dim=-1)).float().mean()
            radius_train = ((rx.float() + ry.float()) * 0.5)
            depth_abs = self._masked_mean(depth_residual.detach().abs().squeeze(1), valid)

            stats = {
                "valid_ratio": valid_ratio.detach(),
                "empty_ratio": empty_ratio.detach(),
                "radius_px_mean": radius_train.mean().detach(),
                "radius_px_min": radius_train.min().detach(),
                "radius_px_max": radius_train.max().detach(),
                "depth_abs_mean": depth_abs.detach(),
                "pool_entropy": pool_entropy.detach(),
                "pool_maxprob": pool_maxprob.detach(),
            }

        if end_points is not None:
            end_points["D: LR use view pool"] = torch.tensor(float(self.use_view_conditioned_pool), device=device).reshape(())
            end_points["D: LR valid ratio"] = stats["valid_ratio"].reshape(())
            end_points["D: LR empty ratio"] = stats["empty_ratio"].reshape(())
            end_points["D: LR radius px mean"] = stats["radius_px_mean"].reshape(())
            end_points["D: LR radius px min"] = stats["radius_px_min"].reshape(())
            end_points["D: LR radius px max"] = stats["radius_px_max"].reshape(())
            end_points["D: LR depth rel abs mean"] = stats["depth_abs_mean"].reshape(())
            end_points["D: LR pool entropy"] = stats["pool_entropy"].reshape(())
            end_points["D: LR pool maxprob"] = stats["pool_maxprob"].reshape(())
            
        self._maybe_debug_print(stats)

        self._maybe_visualize(
            end_points=end_points,
            token_sel_idx=token_sel_idx,
            seed_xyz=seed_xyz if seed_xyz is not None else None,
            radius_px_vis=radius_px_vis,
            feat_map=feat_map,
            depth_map=depth_for_aux,
            uncert_map=uncert_map,
            obj_prob=obj_prob,
            graspness_map=graspness_map,
            pool_attn_map=pool_attn_map,
        )

        self._iter += 1
        return grouped


from models.dinov2_dpt import DPTHead
from models.grasp_spatial_enhancer import GraspSpatialEnhancer
class economicgrasp_dpt(nn.Module):
    """
    economicgrasp_dpt:
      - no enhancer
      - DINOv2DepthDistributionNet produces depth distribution + returns raw feats
      - a new DPTHead(out_dim=3) predicts objectness(2) + graspness(1) from the same feats
      - path_1 from the proposal DPTHead is used as dense seed feature map
      - seed selection / ViewNet / grouping / grasp head reuse EconomicGrasp pipeline
    """
    def __init__(
        self,
        encoder: str = 'vitb',
        tok_feat_dim: int = 128,
        cylinder_radius: float = 0.05,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        freeze_backbone: bool = True,
        use_gt_xyz_for_train: bool = False,
        is_training: bool = True,
        use_obs_depth: bool = False,
        vis_dir: Optional[str] = 'vis_dpt',
        vis_every: int = 500,
        debug_print_every: int = 50,
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.seed_feature_dim = int(tok_feat_dim)
        self.num_depth = int(cfgs.num_depth)
        self.num_angle = int(cfgs.num_angle)
        self.M_points = int(cfgs.m_point)
        self.num_view = int(cfgs.num_view)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)
        self.use_obs_depth = bool(use_obs_depth)
        
        self.stride = 1
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.debug_print_every = int(debug_print_every)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        self.depth_net = DINOv2DepthDistributionNet(
            encoder=encoder,
            stride=self.stride,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            bin_num=self.bin_num,
            freeze_backbone=freeze_backbone,
        )

        model_configs = {
            'vits': {'embed_dim': 384, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'embed_dim': 768, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'embed_dim': 1024, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'embed_dim': 1536, 'out_channels': [1536, 1536, 1536, 1536]},
        }
        cfg = model_configs[encoder]

        # One DPT head predicts [objectness_logit_0, objectness_logit_1, graspness]
        self.proposal_head = DPTHead(
            in_channels=cfg['embed_dim'],
            features=tok_feat_dim,
            use_bn=False,
            out_channels=cfg['out_channels'],
            out_dim=3,
            use_clstoken=True,
        )

        self.spatial_enhancer = GraspSpatialEnhancer(
            embed_dims=tok_feat_dim,
            feature_3d_dim=32,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            num_depth=self.bin_num,
            detach_depth_prob=True,      # 第一轮建议 True，避免破坏 depth_net
            use_prob_embedding=False,    # 第一轮建议 False，先用 depth moments + IPE
            use_post_norm=False,         # 第一轮建议 False，保持 path_1 分布
            use_sensor_depth_feat=self.use_obs_depth,
            vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'spatial_enhancer'),
            vis_every=self.vis_every,
            vis_rank0_only=True,
            save_vis_npz=True,
            )
        
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        # self.view = GeometryAwareDenseFieldViewNet(
        #     num_view=self.num_view,
        #     seed_feature_dim=self.seed_feature_dim,
        #     hidden_dim=self.seed_feature_dim,
        #     min_depth=self.min_depth,
        #     max_depth=self.max_depth,
        #     bin_num=self.bin_num,
        #     view_dirs=generate_grasp_views(self.num_view),
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'geom_viewnet'),
        #     vis_every=self.vis_every,
        # )
        # self.view = GeometryAwareDenseFieldAttnViewNet(
        #     num_view=self.num_view,
        #     seed_feature_dim=self.seed_feature_dim,
        #     hidden_dim=self.seed_feature_dim,
        #     min_depth=self.min_depth,
        #     max_depth=self.max_depth,
        #     bin_num=self.bin_num,
        #     view_dirs=generate_grasp_views(self.num_view),
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'geom_attn_viewnet'),
        #     vis_every=self.vis_every,
        #     num_heads=4,
        #     attn_dropout=0.01
        # )
        # self.cy_group = Cylinder_Grouping_Global_Interaction(
        #     nsample=16,
        #     cylinder_radius=cylinder_radius,
        #     seed_feature_dim=self.seed_feature_dim,
        # )
        # self.pv_group = ProjectedViewGrouping(
        #     seed_feature_dim=self.seed_feature_dim,
        #     out_dim=256,
        #     num_axial=4,
        #     num_radial=8,
        #     radius=cylinder_radius,
        #     hmin=-0.02,
        #     hmax=0.04,
        #     residual_tau=0.02,
        #     detach_depth=True,
        #     use_local_attention=True,
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'projected_view_group'),
        #     vis_every=self.vis_every,
        #     vis_num_seeds=4,
        #     vis_seed_mode='valid_first',
        #     save_npz=True,
        #     save_ply=True,
        # )
        # self.pv_group = ProjectedSurfaceCylinderGrouping(
        #     seed_feature_dim=self.seed_feature_dim,
        #     out_dim=256,
        #     nsample=32,
        #     radius=cylinder_radius,
        #     hmin=-0.02,
        #     hmax=0.04,
        #     grid_size=21,
        #     dynamic_window=True,
        #     window_radius_px_min=8.0,
        #     window_radius_px_max=48.0,
        #     window_radius_factor=1.5,
        #     h_soft_tau=None,
        #     min_group_weight=1e-4,
        #     detach_depth=True,
        #     use_local_attention=True,
        #     residual_fusion=True,
        #     init_group_scale=0.1,
        #     debug_print_every=self.debug_print_every,
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'projected_surface_cylinder_group'),
        #     vis_every=self.vis_every,
        #     vis_num_seeds=4,
        #     vis_seed_mode='valid_first',
        #     save_npz=True,
        #     save_ply=False,
        # )
        # self.def2d_group = ViewConditionedDeformable2DGrouping(
        #     seed_feature_dim=self.seed_feature_dim,
        #     feat_dim=self.seed_feature_dim,
        #     out_dim=256,
        #     num_samples=32,
        #     num_heads=4,
        #     base_radius_px=24.0,
        #     offset_scale_px=16.0,
        #     dropout=0.05,
        #     zero_init_offsets=True,
        #     use_context_skip=True,
        #     debug_print_every=self.debug_print_every,
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'deform2d_group'),
        #     vis_every=self.vis_every,
        #     vis_num_seeds=4,
        #     vis_seed_mode='first',  # first | high_entropy | high_offset
        #     save_npz=True,
        # )
        # self.ddla_group = DepthDisLocalAttnGrouping(
        #     seed_feature_dim=self.seed_feature_dim,
        #     feat_dim=self.seed_feature_dim,
        #     out_dim=256,
        #     hidden_dim=128,
        #     num_heads=4,
        #     patch_size=7,
        #     patch_stride=3.0,          # adaptive_radius=False 时使用
        #     adaptive_radius=True,
        #     physical_radius=0.04,
        #     patch_radius_px_min=6.0,
        #     patch_radius_px_max=32.0,
        #     detach_depth_prob=True,
        #     depth_topk=4,
        #     min_depth=self.min_depth,
        #     max_depth=self.max_depth,
        #     bin_num=self.bin_num,
        #     xyz_norm_scale=0.10,
        #     depth_rel_norm_scale=0.10,
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'ddla_group'),
        #     vis_every=self.vis_every,
        #     debug_print_every=self.debug_print_every,
        # )
        self.local_region_group = MetricRegionCropGrouping(
            seed_feature_dim=self.seed_feature_dim,
            feat_dim=self.seed_feature_dim,
            out_dim=256,
            hidden_dim=128,

            patch_size=12,
            metric_radius=0.08,
            radius_px_min=8.0,
            radius_px_max=64.0,

            train_scale_min=0.80,
            train_scale_max=1.25,

            min_depth=self.min_depth,
            max_depth=self.max_depth,
            depth_norm_scale=0.08,

            detach_depth=True,
            detach_aux_maps=True,

            use_view_conditioned_pool=True,

            vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'local_region_crop'),
            vis_every=self.vis_every,
            vis_num_seeds=4,
            vis_seed_mode='first',
            save_npz=True,
        )

        self.grasp_head = Grasp_Head_Local_Interaction(
            num_angle=self.num_angle,
            num_depth=self.num_depth,
        )

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

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap='Spectral', title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6, 6))
        if vmin is None:
            vmin = float(np.nanmin(arr2d))
        if vmax is None:
            vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_points(self, img_448, pts_uv, out_path, radius=1, color=(0, 0, 255)):
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
        if self.vis_dir is None:
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

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0
        pts = np.concatenate([p, g], axis=0)
        cols = np.concatenate([p_col, g_col], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        scene = int(end_points.get('scene_idx', -1)[0].item()) if torch.is_tensor(end_points.get('scene_idx', None)) else int(end_points.get('scene_idx', -1))
        anno = int(end_points.get('anno_idx', -1)[0].item()) if torch.is_tensor(end_points.get('anno_idx', None)) else int(end_points.get('anno_idx', -1))
        out_path = os.path.join(self.vis_dir, f'dpt_pred_gt_xyz_scene{scene:04d}_anno{anno:04d}_iter{self._vis_iter:06d}.ply')
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

    def forward(self, end_points: dict):
        img = end_points['img']
        K = end_points['K']
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)
        Ntok = H * W
        M = self.M_points

        depth_448, depth_tok, _, depth_prob_448, depth_logits_448, depth_prob_pred, feats = self.depth_net(
            img,
            return_prob=True,
            return_tok_prob=True,
            return_feats=True,
        )

        patch_h, patch_w = H // 14, W // 14
        proposal_path1, proposal_logits_448 = self.proposal_head(feats, patch_h, patch_w)

        # ------------------------------------------------------------------
        # Grasp Spatial Enhancer
        # ------------------------------------------------------------------
        # proposal_path1_enh, spatial_aux = self.spatial_enhancer(
        #     feat_2d=proposal_path1,       # (B,C,Hf,Wf)
        #     depth_prob=depth_prob_448,    # (B,D,448,448)
        #     K=K,                          # K must match resized/cropped 448x448 image
        #     image_hw=(H, W),              # usually (448,448)
        #     return_maps=False,
        #     img=end_points.get("img", img) if isinstance(end_points, dict) else img,
        #     vis_prefix=None,
        # )

        proposal_path1_enh, spatial_aux = self.spatial_enhancer(
            feat_2d=proposal_path1,
            depth_prob=depth_prob_448,       # IPE 仍来自预测 depth_prob
            K=K,
            image_hw=(H, W),
            sensor_depth=end_points.get("sensor_depth_m", None) if self.use_obs_depth else None,
            return_maps=False,
            img=end_points.get("img", img),
        )

        for k, v in spatial_aux.items():
            end_points[k] = v

        feat_grid = F.interpolate(proposal_path1_enh, size=(H, W), mode='bilinear', align_corners=False)
        # feat_grid = F.interpolate(proposal_path1, size=(H, W), mode='bilinear', align_corners=False)

        objectness_logits_448 = proposal_logits_448[:, :2, :, :]
        graspness_logits_448 = proposal_logits_448[:, 2:3, :, :]

        end_points['img_feat_dpt'] = feat_grid
        end_points['depth_map_pred'] = depth_448
        end_points['depth_tok_pred'] = depth_tok
        end_points['depth_prob_pred'] = depth_prob_pred
        end_points['depth_prob_448'] = depth_prob_448
        end_points['depth_logits_448'] = depth_logits_448

        objectness_score = objectness_logits_448.view(B, 2, -1).contiguous()
        graspness_score = graspness_logits_448.view(B, 1, -1).contiguous()
        end_points['objectness_score'] = objectness_score
        end_points['graspness_score'] = graspness_score

        objectness_pred = torch.argmax(objectness_score, dim=1)
        grasp_raw = graspness_score.squeeze(1)
        grasp_sel = grasp_raw.clamp(0.0, 1.0)

        if 'token_valid_mask' in end_points:
            valid_tok = end_points['token_valid_mask'].bool()
            if valid_tok.shape[1] != Ntok:
                raise ValueError(f'Expected token_valid_mask with {Ntok}, got {tuple(valid_tok.shape)}')
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        mask_obj_pred = valid_tok & (objectness_pred == 1)
        mask_thr_pred = mask_obj_pred & (grasp_sel > float(cfgs.graspness_threshold))

        end_points['dbg_grasp_raw'] = grasp_raw.detach()
        end_points['dbg_grasp_sel'] = grasp_sel.detach()
        end_points['dbg_mask_obj'] = mask_obj_pred.detach()
        end_points['dbg_mask_pred'] = mask_thr_pred.detach()
        end_points['dbg_objectness_pred'] = objectness_pred.detach()
        end_points['D: PredCand#(thr)'] = mask_thr_pred.float().sum(dim=1).mean().reshape(())
        end_points['D: PredObj#'] = mask_obj_pred.float().sum(dim=1).mean().reshape(())
        end_points['D: GraspRaw min'] = grasp_raw.min().reshape(())
        end_points['D: GraspRaw max'] = grasp_raw.max().reshape(())
        end_points['D: GraspSel mean'] = grasp_sel.mean().reshape(())

        flat_all = torch.arange(H * W, device=img.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        u_all = (flat_all % W).float()
        v_all = (flat_all // W).float()
        uv_all = torch.stack([u_all, v_all], dim=-1)

        z_all_pred = depth_448.view(B, -1, 1).contiguous()
        z_all_pred = torch.nan_to_num(z_all_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
        xyz_all_pred = self._backproject_uvz(uv_all, z_all_pred.detach(), K)

        use_gt_xyz = self.is_training and self.use_gt_xyz_for_train and ('gt_depth_m' in end_points)
        if use_gt_xyz:
            gt_depth = end_points['gt_depth_m']
            if gt_depth.dim() == 3:
                gt_depth = gt_depth.unsqueeze(1)
            elif gt_depth.dim() == 4:
                gt_depth = gt_depth[:, :1]
            z_all_gt = gt_depth.view(B, -1, 1).contiguous()
            z_all_gt = torch.nan_to_num(z_all_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            xyz_all_match = self._backproject_uvz(uv_all, z_all_gt, K)
        else:
            xyz_all_match = xyz_all_pred

        seed_features_flipped = feat_grid.view(B, feat_grid.shape[1], -1).contiguous()  # (B,C,N)
        seed_xyz = xyz_all_match
        graspable_mask = mask_thr_pred

        seed_features_graspable = []
        seed_xyz_graspable = []
        token_sel_idx = []
        graspable_num_batch = 0.0
        for i in range(B):
            cur_mask = graspable_mask[i]
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
            graspable_num_batch += float(cur_mask.sum().item())

            if cur_idx.numel() == 0:
                cur_idx = torch.arange(Ntok, device=img.device)

            cur_feat = seed_features_flipped[i][:, cur_idx]   # (C,Ng)
            cur_seed_xyz = seed_xyz[i][cur_idx]               # (Ng,3)

            if cur_seed_xyz.shape[0] >= M:
                cur_seed_xyz_ = cur_seed_xyz.unsqueeze(0).contiguous()
                fps_idxs = furthest_point_sample(cur_seed_xyz_, M)
                cur_seed_xyz = gather_operation(cur_seed_xyz_.transpose(1, 2).contiguous(), fps_idxs).transpose(1, 2).squeeze(0).contiguous()
                cur_feat = gather_operation(cur_feat.unsqueeze(0).contiguous(), fps_idxs).squeeze(0).contiguous()
                cur_idx_sel = cur_idx[fps_idxs.squeeze(0).long()]
            else:
                rep = torch.randint(0, cur_seed_xyz.shape[0], (M,), device=img.device)
                cur_seed_xyz = cur_seed_xyz[rep]
                cur_feat = cur_feat[:, rep]
                cur_idx_sel = cur_idx[rep]

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            token_sel_idx.append(cur_idx_sel)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        seed_features_graspable = torch.stack(seed_features_graspable, 0)  # (B,C,M)
        token_sel_idx = torch.stack(token_sel_idx, 0)

        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['token_sel_idx'] = token_sel_idx
        end_points['token_sel_xyz'] = seed_xyz_graspable
        end_points['D: Graspable Points'] = torch.tensor(graspable_num_batch / float(B), device=img.device)

        if (self.vis_dir is not None) and (self._vis_iter % self.vis_every == 0):
            try:
                self._save_map_png(grasp_sel[0].view(H, W), os.path.join(self.vis_dir, f'dpt_grasp_map_it{self._vis_iter:06d}.png'), cmap='viridis')
                self._save_map_png(objectness_pred[0].view(H, W).float(), os.path.join(self.vis_dir, f'dpt_objectness_it{self._vis_iter:06d}.png'), cmap='gray')
                self._save_map_png(depth_448[0, 0], os.path.join(self.vis_dir, f'dpt_depth_it{self._vis_iter:06d}.png'), cmap='magma')
                pts_uv = torch.stack([(token_sel_idx[0] % W).float(), (token_sel_idx[0] // W).float()], dim=-1)
                self._save_overlay_points(img[0], pts_uv, os.path.join(self.vis_dir, f'dpt_seed_overlay_it{self._vis_iter:06d}.png'))
                if 'gt_depth_m' in end_points:
                    gt_depth = end_points['gt_depth_m']
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    z_all_gt = gt_depth.view(B, -1, 1).contiguous().clamp_min(1e-6)
                    xyz_all_gt = self._backproject_uvz(uv_all, z_all_gt, K)
                    self._save_pred_gt_cloud_ply(xyz_all_pred, xyz_all_gt, end_points)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 10) view + labels + grouping + head
        # ------------------------------------------------------------------
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        # end_points, res_feat = self.view(
        #     seed_features=seed_features_graspable,   # (B,C,M)
        #     token_sel_idx=token_sel_idx,             # (B,M)
        #     K=K,
        #     depth_map=depth_448,                     # (B,1,448,448)
        #     depth_prob=depth_prob_448,               # (B,D,448,448)
        #     end_points=end_points,
        # )
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        # group_features = self.cy_group(seed_xyz_graspable, seed_features_graspable, grasp_top_views_rot)
        # group_features = self.pv_group(
        #     seed_features=seed_features_graspable,
        #     token_sel_idx=token_sel_idx,
        #     top_view_rot=grasp_top_views_rot,
        #     feat_map=feat_grid,
        #     depth_map=depth_448,
        #     K=K,
        #     end_points=end_points,
        # )
        # group_features = self.def2d_group(
        #     seed_features=seed_features_graspable,   # (B,C,M)
        #     token_sel_idx=token_sel_idx,             # (B,M)
        #     top_view_rot=grasp_top_views_rot,        # (B,M,3,3)
        #     feat_map=feat_grid,                      # (B,C,H,W)
        #     end_points=end_points,
        # )
        # group_features = self.ddla_group(
        #     seed_features=seed_features_graspable,   # (B,C,M)
        #     token_sel_idx=token_sel_idx,             # (B,M)
        #     top_view_rot=grasp_top_views_rot,        # (B,M,3,3)
        #     feat_map=feat_grid,                      # (B,C,H,W)
        #     depth_prob=depth_prob_448,               # (B,D,H,W)
        #     K=K,                                     # (B,3,3)
        #     end_points=end_points,
        # )
        group_features = self.local_region_group(
            seed_features=seed_features_graspable,
            token_sel_idx=token_sel_idx,
            seed_xyz=seed_xyz_graspable,
            top_view_rot=grasp_top_views_rot,
            feat_map=feat_grid,
            depth_map=depth_448,
            depth_prob=depth_prob_448,
            objectness_logits=objectness_logits_448,
            graspness_map=grasp_sel.view(B, 1, H, W).contiguous(),
            K=K,
            end_points=end_points,
        )
        
        end_points = self.grasp_head(group_features, end_points)
        if (self._vis_iter % self.debug_print_every == 0):
            with torch.no_grad():
                print(
                    f"[economicgrasp_dpt] it={self._vis_iter} "
                    f"graspable={end_points['D: Graspable Points'].item():.1f} "
                    f"cand={end_points['D: PredCand#(thr)'].item():.1f} "
                    f"obj={end_points['D: PredObj#'].item():.1f} "
                    f"grasp_mean={end_points['D: GraspSel mean'].item():.4f}"
                )

        self._vis_iter += 1
        return end_points
    

class Direct2DOperationFeature(nn.Module):
    """
    No-group operation feature generator.

    It directly predicts the operation feature consumed by Grasp_Head_Local_Interaction
    from per-seed 2D feature plus view/ray/depth-distribution statistics.

    Inputs:
      seed_features: (B,C,M)
      token_sel_idx: (B,M), flattened 448x448 pixel index
      top_view_rot:  (B,M,3,3), only used as fallback to infer view direction
      top_view_xyz:  (B,M,3), preferred selected view direction if available
      depth_prob:    (B,D,H,W), detached by default
      K:             (B,3,3)
    Output:
      operation_feature: (B,out_dim,M), default out_dim=256
    """
    def __init__(
        self,
        seed_feature_dim: int = 128,
        out_dim: int = 256,
        hidden_dim: int = 256,
        num_depth_bins: int = 256,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        depth_topk: int = 4,
        detach_depth_prob: bool = True,
        use_depth_stats: bool = True,
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        debug_print_every: int = 50,
    ):
        super().__init__()
        self.seed_feature_dim = int(seed_feature_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_depth_bins = int(num_depth_bins)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.depth_topk = int(depth_topk)
        self.detach_depth_prob = bool(detach_depth_prob)
        self.use_depth_stats = bool(use_depth_stats)
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.debug_print_every = int(debug_print_every)
        self._iter = 0

        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        # Geometry/stat feature layout:
        # view_xyz(3), ray(3), ray_dot_view(1),
        # depth_mean_norm(1), depth_std_norm(1), depth_entropy(1), topk_mass(1), top_prob(1)
        self.geom_dim = 3 + 3 + 1 + 5
        in_dim = self.seed_feature_dim + self.geom_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        # A light residual projection from seed feature helps keep the feature scale sane.
        # This is still no grouping; it only stabilizes the direct feature representation.
        self.seed_proj = nn.Sequential(
            nn.Conv1d(seed_feature_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
        )

        bin_centers = torch.linspace(self.min_depth, self.max_depth, self.num_depth_bins, dtype=torch.float32)
        self.register_buffer("bin_centers", bin_centers.view(1, 1, self.num_depth_bins))  # (1,1,D)

    @staticmethod
    def _idx_to_uv(idx_bm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()
        return torch.stack([u, v], dim=-1)  # (B,M,2)

    @staticmethod
    def _ray_dirs_from_idx(idx_bm: torch.Tensor, K_b33: torch.Tensor, H: int, W: int) -> torch.Tensor:
        u = (idx_bm % W).float()
        v = (idx_bm // W).float()
        fx = K_b33[:, 0, 0].unsqueeze(1)
        fy = K_b33[:, 1, 1].unsqueeze(1)
        cx = K_b33[:, 0, 2].unsqueeze(1)
        cy = K_b33[:, 1, 2].unsqueeze(1)
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = torch.ones_like(x)
        rays = torch.stack([x, y, z], dim=-1)
        return F.normalize(rays, dim=-1, eps=1e-6)

    @staticmethod
    def _gather_prob_at_idx(prob_bdhw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
        # prob_bdhw: (B,D,H,W) -> (B,M,D)
        B, D, H, W = prob_bdhw.shape
        flat = prob_bdhw.view(B, D, H * W)
        idx = idx_bm.long().unsqueeze(1).expand(-1, D, -1)
        gathered = torch.gather(flat, dim=2, index=idx)  # (B,D,M)
        return gathered.transpose(1, 2).contiguous()     # (B,M,D)

    def _prepare_depth_prob(self, depth_prob: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if depth_prob.dim() != 4:
            raise ValueError(f"Expected depth_prob (B,D,H,W), got {tuple(depth_prob.shape)}")
        if depth_prob.shape[-2:] != (H, W):
            depth_prob = F.interpolate(depth_prob, size=(H, W), mode="bilinear", align_corners=False)
        prob = torch.nan_to_num(depth_prob.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-8)
        prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return prob

    def _depth_stats(self, depth_prob: torch.Tensor, token_sel_idx: torch.Tensor, H: int, W: int):
        # Returns normalized seed-level depth stats: mean, std, entropy, topk_mass, top_prob.
        prob = depth_prob.detach() if self.detach_depth_prob else depth_prob
        prob = self._prepare_depth_prob(prob, H, W)
        seed_prob = self._gather_prob_at_idx(prob, token_sel_idx)  # (B,M,D)
        B, M, D = seed_prob.shape

        if self.bin_centers.shape[-1] != D:
            bin_centers = torch.linspace(self.min_depth, self.max_depth, D, dtype=seed_prob.dtype, device=seed_prob.device)
            bin_centers = bin_centers.view(1, 1, D)
        else:
            bin_centers = self.bin_centers.to(device=seed_prob.device, dtype=seed_prob.dtype)

        mean_z = (seed_prob * bin_centers).sum(dim=-1)  # (B,M)
        var_z = (seed_prob * (bin_centers - mean_z.unsqueeze(-1)).pow(2)).sum(dim=-1)
        std_z = var_z.clamp_min(0.0).sqrt()
        entropy = -(seed_prob * seed_prob.clamp_min(1e-8).log()).sum(dim=-1) / math.log(max(D, 2))
        topk = min(max(self.depth_topk, 1), D)
        top_prob, _ = torch.topk(seed_prob, k=topk, dim=-1)
        topk_mass = top_prob.sum(dim=-1)
        top_prob_max = top_prob[..., 0]

        depth_range = max(self.max_depth - self.min_depth, 1e-6)
        mean_norm = (mean_z - self.min_depth) / depth_range
        std_norm = std_z / depth_range
        stats = torch.stack([mean_norm, std_norm, entropy, topk_mass, top_prob_max], dim=-1)
        raw = {
            "mean_z": mean_z,
            "std_z": std_z,
            "entropy": entropy,
            "topk_mass": topk_mass,
            "top_prob": top_prob_max,
        }
        return stats, raw

    @staticmethod
    def _to_rgb_np(img_chw: torch.Tensor):
        x = img_chw.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x.permute(1, 2, 0).numpy()

    def _save_seed_scatter_overlay(
        self,
        img_chw: torch.Tensor,
        token_sel_idx: torch.Tensor,
        values: torch.Tensor,
        out_path: str,
        H: int,
        W: int,
        cmap: str = "viridis",
        title: Optional[str] = None,
        vmin=None,
        vmax=None,
        dot_size: int = 10,
    ):
        img_np = self._to_rgb_np(img_chw)
        idx = token_sel_idx.detach().cpu()
        vals = values.detach().float().cpu()
        u = (idx % W).numpy()
        v = (idx // W).numpy()
        if vmin is None:
            vmin = float(vals.min().item())
        if vmax is None:
            vmax = float(vals.max().item())
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.scatter(u, v, c=vals.numpy(), s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def _maybe_visualize(self, end_points, token_sel_idx, depth_raw, ray_dot, op_norm):
        if self.vis_dir is None or self._iter % self.vis_every != 0:
            return
        if "img" not in end_points:
            return
        try:
            img = end_points["img"]
            B, _, H, W = img.shape
            b0 = 0
            prefix = f"direct2d_it{self._iter:06d}"
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], depth_raw["mean_z"][b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_depth_mean.png"), H, W,
                cmap="magma", title="seed depth mean", vmin=self.min_depth, vmax=self.max_depth)
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], depth_raw["entropy"][b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_depth_entropy.png"), H, W,
                cmap="viridis", title="seed depth entropy", vmin=0.0, vmax=1.0)
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], depth_raw["topk_mass"][b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_topk_mass.png"), H, W,
                cmap="plasma", title="seed depth top-k mass", vmin=0.0, vmax=1.0)
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], ray_dot[b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_ray_dot_view.png"), H, W,
                cmap="coolwarm", title="ray dot selected view", vmin=-1.0, vmax=1.0)
            self._save_seed_scatter_overlay(
                img[b0], token_sel_idx[b0], op_norm[b0],
                os.path.join(self.vis_dir, f"{prefix}_seed_op_norm.png"), H, W,
                cmap="viridis", title="operation feature norm")
        except Exception as e:
            print(f"[Direct2DOperationFeature vis] failed at iter {self._iter}: {repr(e)}")

    def forward(
        self,
        seed_features: torch.Tensor,
        token_sel_idx: torch.Tensor,
        top_view_rot: torch.Tensor,
        top_view_xyz: Optional[torch.Tensor],
        depth_prob: torch.Tensor,
        K: torch.Tensor,
        end_points: Optional[dict] = None,
    ) -> torch.Tensor:
        B, C, M = seed_features.shape
        _, _, H, W = depth_prob.shape
        dtype = seed_features.dtype
        device = seed_features.device

        seed_bmc = seed_features.transpose(1, 2).contiguous()  # (B,M,C)

        rays = self._ray_dirs_from_idx(token_sel_idx, K, H, W).to(dtype=dtype)  # (B,M,3)
        if top_view_xyz is not None and torch.is_tensor(top_view_xyz):
            view_dir = F.normalize(top_view_xyz.to(device=device, dtype=dtype), dim=-1, eps=1e-6)
        else:
            # batch_viewpoint_params_to_matrix is called with -top_view_xyz as axis_x in many pipelines,
            # so top_view_xyz is approximately -R[:,0]. This fallback recovers the same convention.
            view_dir = F.normalize(-top_view_rot[..., 0].to(dtype=dtype), dim=-1, eps=1e-6)

        ray_dot = (rays * view_dir).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (B,M,1)

        if self.use_depth_stats:
            depth_stats, depth_raw = self._depth_stats(depth_prob, token_sel_idx, H, W)
            depth_stats = depth_stats.to(device=device, dtype=dtype)
        else:
            depth_stats = torch.zeros((B, M, 5), device=device, dtype=dtype)
            z = torch.zeros((B, M), device=device, dtype=dtype)
            depth_raw = {"mean_z": z, "std_z": z, "entropy": z, "topk_mass": z, "top_prob": z}

        geom = torch.cat([view_dir, rays, ray_dot, depth_stats], dim=-1)  # (B,M,12)
        op_in = torch.cat([seed_bmc, geom], dim=-1)
        op = self.mlp(op_in).transpose(1, 2).contiguous()                 # (B,out_dim,M)
        op = op + self.seed_proj(seed_features)

        with torch.no_grad():
            seed_norm = seed_features.norm(dim=1).mean()
            op_norm_seed = op.norm(dim=1)  # (B,M)
            op_norm = op_norm_seed.mean()
            if end_points is not None:
                end_points["D: Direct2D depth detached"] = torch.tensor(float(self.detach_depth_prob), device=device)
                end_points["D: Direct2D use depth stats"] = torch.tensor(float(self.use_depth_stats), device=device)
                end_points["D: Direct2D seed depth mean"] = depth_raw["mean_z"].mean().reshape(())
                end_points["D: Direct2D seed depth std"] = depth_raw["std_z"].mean().reshape(())
                end_points["D: Direct2D seed entropy"] = depth_raw["entropy"].mean().reshape(())
                end_points["D: Direct2D topk mass"] = depth_raw["topk_mass"].mean().reshape(())
                end_points["D: Direct2D top prob"] = depth_raw["top_prob"].mean().reshape(())
                end_points["D: Direct2D ray dot view"] = ray_dot.mean().reshape(())
                end_points["D: Direct2D op feat norm"] = op_norm.reshape(())
                end_points["D: Direct2D seed feat norm"] = seed_norm.reshape(())
                end_points["D: Direct2D op/seed norm"] = (op_norm / seed_norm.clamp_min(1e-6)).reshape(())

            if (self._iter % self.debug_print_every == 0):
                print(
                    f"[Direct2DOperationFeature] it={self._iter} "
                    f"detachD={int(self.detach_depth_prob)} useD={int(self.use_depth_stats)} "
                    f"z={depth_raw['mean_z'].mean().item():.4f} "
                    f"H={depth_raw['entropy'].mean().item():.4f} "
                    f"topk={depth_raw['topk_mass'].mean().item():.4f} "
                    f"raydot={ray_dot.mean().item():.4f} "
                    f"op/seed={(op_norm / seed_norm.clamp_min(1e-6)).item():.3f}"
                )

            if end_points is not None:
                self._maybe_visualize(end_points, token_sel_idx, depth_raw, ray_dot.squeeze(-1), op_norm_seed)

        self._iter += 1
        return op


class economicgrasp_dpt_direct(nn.Module):
    """
    economicgrasp_dpt_direct:
      - same DPT proposal/depth front-end as economicgrasp_dpt
      - same seed selection and GeometryAwareDenseFieldAttnViewNet supervision
      - NO explicit grouping module
      - operation feature is predicted directly from seed 2D feature + selected view + ray + detached depth stats
      - output is fed to the original Grasp_Head_Local_Interaction
    """
    def __init__(
        self,
        encoder: str = 'vitb',
        tok_feat_dim: int = 128,
        cylinder_radius: float = 0.05,  # kept for CLI compatibility, unused
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        freeze_backbone: bool = True,
        use_gt_xyz_for_train: bool = False,
        is_training: bool = True,
        vis_dir: Optional[str] = 'vis_dpt_direct',
        vis_every: int = 500,
        debug_print_every: int = 50,
        direct_hidden_dim: int = 256,
        direct_depth_topk: int = 4,
        direct_detach_depth_prob: bool = True,
        direct_use_depth_stats: bool = True,
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.seed_feature_dim = int(tok_feat_dim)
        self.num_depth = int(cfgs.num_depth)
        self.num_angle = int(cfgs.num_angle)
        self.M_points = int(cfgs.m_point)
        self.num_view = int(cfgs.num_view)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)
        self.stride = 1
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.debug_print_every = int(debug_print_every)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        self.depth_net = DINOv2DepthDistributionNet(
            encoder=encoder,
            stride=self.stride,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            bin_num=self.bin_num,
            freeze_backbone=freeze_backbone,
        )

        model_configs = {
            'vits': {'embed_dim': 384, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'embed_dim': 768, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'embed_dim': 1024, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'embed_dim': 1536, 'out_channels': [1536, 1536, 1536, 1536]},
        }
        cfg = model_configs[encoder]

        self.proposal_head = DPTHead(
            in_channels=cfg['embed_dim'],
            features=tok_feat_dim,
            use_bn=False,
            out_channels=cfg['out_channels'],
            out_dim=3,
            use_clstoken=True,
        )

        self.view = GeometryAwareDenseFieldAttnViewNet(
            num_view=self.num_view,
            seed_feature_dim=self.seed_feature_dim,
            hidden_dim=self.seed_feature_dim,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            bin_num=self.bin_num,
            view_dirs=generate_grasp_views(self.num_view),
            vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'geom_attn_viewnet'),
            vis_every=self.vis_every,
            num_heads=4,
            attn_dropout=0.01
        )

        self.direct_op = Direct2DOperationFeature(
            seed_feature_dim=self.seed_feature_dim,
            out_dim=256,
            hidden_dim=direct_hidden_dim,
            num_depth_bins=self.bin_num,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            depth_topk=direct_depth_topk,
            detach_depth_prob=direct_detach_depth_prob,
            use_depth_stats=direct_use_depth_stats,
            vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'direct2d_op'),
            vis_every=self.vis_every,
            debug_print_every=self.debug_print_every,
        )

        self.grasp_head = Grasp_Head_Local_Interaction(
            num_angle=self.num_angle,
            num_depth=self.num_depth,
        )

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

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap='Spectral', title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6, 6))
        if vmin is None:
            vmin = float(np.nanmin(arr2d))
        if vmax is None:
            vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_points(self, img_448, pts_uv, out_path, radius=1, color=(0, 0, 255)):
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
        if self.vis_dir is None or o3d is None:
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
        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0
        pts = np.concatenate([p, g], axis=0)
        cols = np.concatenate([p_col, g_col], axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        scene = int(end_points.get('scene_idx', -1)[0].item()) if torch.is_tensor(end_points.get('scene_idx', None)) else int(end_points.get('scene_idx', -1))
        anno = int(end_points.get('anno_idx', -1)[0].item()) if torch.is_tensor(end_points.get('anno_idx', None)) else int(end_points.get('anno_idx', -1))
        out_path = os.path.join(self.vis_dir, f'dpt_direct_pred_gt_xyz_scene{scene:04d}_anno{anno:04d}_iter{self._vis_iter:06d}.ply')
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

    def forward(self, end_points: dict):
        img = end_points['img']
        K = end_points['K']
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)
        Ntok = H * W
        M = self.M_points

        depth_448, depth_tok, _, depth_prob_448, depth_logits_448, depth_prob_pred, feats = self.depth_net(
            img,
            return_prob=True,
            return_tok_prob=True,
            return_feats=True,
        )

        patch_h, patch_w = H // 14, W // 14
        proposal_path1, proposal_logits_448 = self.proposal_head(feats, patch_h, patch_w)
        objectness_logits_448 = proposal_logits_448[:, :2, :, :]
        graspness_logits_448 = proposal_logits_448[:, 2:3, :, :]
        feat_grid = F.interpolate(proposal_path1, size=(H, W), mode='bilinear', align_corners=False)

        end_points['img_feat_dpt'] = feat_grid
        end_points['depth_map_pred'] = depth_448
        end_points['depth_tok_pred'] = depth_tok
        end_points['depth_prob_pred'] = depth_prob_pred
        end_points['depth_prob_448'] = depth_prob_448
        end_points['depth_logits_448'] = depth_logits_448

        objectness_score = objectness_logits_448.view(B, 2, -1).contiguous()
        graspness_score = graspness_logits_448.view(B, 1, -1).contiguous()
        end_points['objectness_score'] = objectness_score
        end_points['graspness_score'] = graspness_score

        objectness_pred = torch.argmax(objectness_score, dim=1)
        grasp_raw = graspness_score.squeeze(1)
        grasp_sel = grasp_raw.clamp(0.0, 1.0)

        if 'token_valid_mask' in end_points:
            valid_tok = end_points['token_valid_mask'].bool()
            if valid_tok.shape[1] != Ntok:
                raise ValueError(f'Expected token_valid_mask with {Ntok}, got {tuple(valid_tok.shape)}')
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        mask_obj_pred = valid_tok & (objectness_pred == 1)
        mask_thr_pred = mask_obj_pred & (grasp_sel > float(cfgs.graspness_threshold))

        end_points['dbg_grasp_raw'] = grasp_raw.detach()
        end_points['dbg_grasp_sel'] = grasp_sel.detach()
        end_points['dbg_mask_obj'] = mask_obj_pred.detach()
        end_points['dbg_mask_pred'] = mask_thr_pred.detach()
        end_points['dbg_objectness_pred'] = objectness_pred.detach()
        end_points['D: PredCand#(thr)'] = mask_thr_pred.float().sum(dim=1).mean().reshape(())
        end_points['D: PredObj#'] = mask_obj_pred.float().sum(dim=1).mean().reshape(())
        end_points['D: GraspRaw min'] = grasp_raw.min().reshape(())
        end_points['D: GraspRaw max'] = grasp_raw.max().reshape(())
        end_points['D: GraspSel mean'] = grasp_sel.mean().reshape(())

        flat_all = torch.arange(H * W, device=img.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        u_all = (flat_all % W).float()
        v_all = (flat_all // W).float()
        uv_all = torch.stack([u_all, v_all], dim=-1)

        z_all_pred = depth_448.view(B, -1, 1).contiguous()
        z_all_pred = torch.nan_to_num(z_all_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
        xyz_all_pred = self._backproject_uvz(uv_all, z_all_pred.detach(), K)

        use_gt_xyz = self.is_training and self.use_gt_xyz_for_train and ('gt_depth_m' in end_points)
        if use_gt_xyz:
            gt_depth = end_points['gt_depth_m']
            if gt_depth.dim() == 3:
                gt_depth = gt_depth.unsqueeze(1)
            elif gt_depth.dim() == 4:
                gt_depth = gt_depth[:, :1]
            z_all_gt = gt_depth.view(B, -1, 1).contiguous()
            z_all_gt = torch.nan_to_num(z_all_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            xyz_all_match = self._backproject_uvz(uv_all, z_all_gt, K)
        else:
            xyz_all_match = xyz_all_pred

        seed_features_flipped = feat_grid.view(B, feat_grid.shape[1], -1).contiguous()
        seed_xyz = xyz_all_match
        graspable_mask = mask_thr_pred

        seed_features_graspable = []
        seed_xyz_graspable = []
        token_sel_idx = []
        graspable_num_batch = 0.0
        for i in range(B):
            cur_mask = graspable_mask[i]
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
            graspable_num_batch += float(cur_mask.sum().item())
            if cur_idx.numel() == 0:
                cur_idx = torch.arange(Ntok, device=img.device)

            cur_feat = seed_features_flipped[i][:, cur_idx]
            cur_seed_xyz = seed_xyz[i][cur_idx]

            if cur_seed_xyz.shape[0] >= M:
                cur_seed_xyz_ = cur_seed_xyz.unsqueeze(0).contiguous()
                fps_idxs = furthest_point_sample(cur_seed_xyz_, M)
                cur_seed_xyz = gather_operation(cur_seed_xyz_.transpose(1, 2).contiguous(), fps_idxs).transpose(1, 2).squeeze(0).contiguous()
                cur_feat = gather_operation(cur_feat.unsqueeze(0).contiguous(), fps_idxs).squeeze(0).contiguous()
                cur_idx_sel = cur_idx[fps_idxs.squeeze(0).long()]
            else:
                rep = torch.randint(0, cur_seed_xyz.shape[0], (M,), device=img.device)
                cur_seed_xyz = cur_seed_xyz[rep]
                cur_feat = cur_feat[:, rep]
                cur_idx_sel = cur_idx[rep]

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            token_sel_idx.append(cur_idx_sel)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        seed_features_graspable = torch.stack(seed_features_graspable, 0)
        token_sel_idx = torch.stack(token_sel_idx, 0)

        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['token_sel_idx'] = token_sel_idx
        end_points['token_sel_xyz'] = seed_xyz_graspable
        end_points['D: Graspable Points'] = torch.tensor(graspable_num_batch / float(B), device=img.device)

        if (self.vis_dir is not None) and (self._vis_iter % self.vis_every == 0):
            try:
                self._save_map_png(grasp_sel[0].view(H, W), os.path.join(self.vis_dir, f'dpt_direct_grasp_map_it{self._vis_iter:06d}.png'), cmap='viridis')
                self._save_map_png(objectness_pred[0].view(H, W).float(), os.path.join(self.vis_dir, f'dpt_direct_objectness_it{self._vis_iter:06d}.png'), cmap='gray')
                self._save_map_png(depth_448[0, 0], os.path.join(self.vis_dir, f'dpt_direct_depth_it{self._vis_iter:06d}.png'), cmap='magma')
                pts_uv = torch.stack([(token_sel_idx[0] % W).float(), (token_sel_idx[0] // W).float()], dim=-1)
                self._save_overlay_points(img[0], pts_uv, os.path.join(self.vis_dir, f'dpt_direct_seed_overlay_it{self._vis_iter:06d}.png'))
                if 'gt_depth_m' in end_points:
                    gt_depth = end_points['gt_depth_m']
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    z_all_gt = gt_depth.view(B, -1, 1).contiguous().clamp_min(1e-6)
                    xyz_all_gt = self._backproject_uvz(uv_all, z_all_gt, K)
                    self._save_pred_gt_cloud_ply(xyz_all_pred, xyz_all_gt, end_points)
            except Exception:
                pass

        # 10) view + labels + direct operation feature + head
        end_points, res_feat = self.view(
            seed_features=seed_features_graspable,
            token_sel_idx=token_sel_idx,
            K=K,
            depth_map=depth_448,
            depth_prob=depth_prob_448,
            end_points=end_points,
        )
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        # Condition Direct2D on the same selected/matched view rotation used by operation labels.
        # Passing top_view_xyz=None recovers view direction from grasp_top_views_rot and avoids
        # mismatch with process_grasp_labels outputs during training.
        operation_feature = self.direct_op(
            seed_features=seed_features_graspable,
            token_sel_idx=token_sel_idx,
            top_view_rot=grasp_top_views_rot,
            top_view_xyz=end_points.get("grasp_top_view_xyz", None),
            depth_prob=depth_prob_448,
            K=K,
            end_points=end_points,
        )

        end_points = self.grasp_head(operation_feature, end_points)

        if (self._vis_iter % self.debug_print_every == 0):
            with torch.no_grad():
                print(
                    f"[economicgrasp_dpt_direct] it={self._vis_iter} "
                    f"graspable={end_points['D: Graspable Points'].item():.1f} "
                    f"cand={end_points['D: PredCand#(thr)'].item():.1f} "
                    f"obj={end_points['D: PredObj#'].item():.1f} "
                    f"grasp_mean={end_points['D: GraspSel mean'].item():.4f}"
                )

        self._vis_iter += 1
        return end_points