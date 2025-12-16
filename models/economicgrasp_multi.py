import torch
import torch.nn as nn

from models.bip3d.modules import ChannelMapper, ResNet, SwinTransformer
from models.modules_economicgrasp import (
    Cylinder_Grouping_Global_Interaction,
    Grasp_Head_Local_Interaction,
    GraspableNet,
    ViewNet,
)
from utils.arguments import cfgs


class economicgrasp_multi_bip3d(nn.Module):
    """Point-centric EconomicGrasp variant using BIP3D encoders.

    The model extracts multi-scale image and depth features using a Swin-Tiny
    style encoder and a ResNet-34 depth encoder, aligns channels via
    ``ChannelMapper`` and fuses them per-scale. The highest-resolution fused
    feature map is sampled using ``img_idxs`` (flattened pixel indices) to build
    per-point descriptors that flow through the original graspable/view/grouping
    heads.
    """

    def __init__(
        self,
        embed_dims: int = 128,
        cylinder_radius: float = 0.05,
        seed_feat_dim: int = 512,
        is_training: bool = True,
        with_depth: bool = True,
    ):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points = cfgs.m_point
        self.num_view = cfgs.num_view

        # --- image encoder ---
        self.img_backbone = SwinTransformer(
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            out_indices=(1, 2, 3),
        )
        self.img_neck = ChannelMapper(
            in_channels=[192, 384, 768],
            kernel_size=1,
            out_channels=embed_dims,
            norm_cfg=dict(type=nn.GroupNorm, num_groups=32),
            num_outs=4,
        )

        # --- depth encoder ---
        self.with_depth = with_depth
        if with_depth:
            self.depth_backbone = ResNet(
                depth=34,
                in_channels=1,
                base_channels=4,
                num_stages=4,
                out_indices=(1, 2, 3),
            )
            self.depth_neck = ChannelMapper(
                in_channels=[8, 16, 32],
                kernel_size=1,
                out_channels=32,
                norm_cfg=dict(type=nn.GroupNorm, num_groups=4),
                num_outs=4,
            )
        else:
            self.depth_backbone = None
            self.depth_neck = None

        # fusion per scale
        fusion_in_dim = embed_dims + (32 if with_depth else 0)
        self.fusion = nn.ModuleList(
            [nn.Conv2d(fusion_in_dim, embed_dims, kernel_size=1) for _ in range(4)]
        )

        # projection to point-wise seed features
        self.seed_proj = nn.Linear(embed_dims, self.seed_feature_dim)

        # downstream grasp heads (unchanged)
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

    def _gather_point_features(self, fused_feat: torch.Tensor, img_idxs: torch.Tensor):
        # fused_feat: (B,C,H,W); img_idxs: (B,N)
        B, C, H, W = fused_feat.shape
        fused_flat = fused_feat.view(B, C, -1)
        idx = img_idxs.long().clamp(0, H * W - 1)
        idx = idx.unsqueeze(1).expand(-1, C, -1)
        gathered = torch.gather(fused_flat, 2, idx)
        return gathered.transpose(1, 2).contiguous()  # (B,N,C)

    def forward(self, end_points):
        seed_xyz = end_points["point_clouds"]  # (B,N,3)
        B, point_num, _ = seed_xyz.shape

        imgs = end_points["imgs"]  # (B,3,H,W)
        img_feats = self.img_neck(self.img_backbone(imgs))

        depth_feats = None
        if self.with_depth:
            depth = end_points["depths"]  # (B,1,H,W)
            depth_feats = self.depth_neck(self.depth_backbone(depth))

        fused_feats = []
        for i, img_feat in enumerate(img_feats):
            feat = img_feat
            if depth_feats is not None:
                depth_feat = depth_feats[min(i, len(depth_feats) - 1)]
                if depth_feat.shape[-2:] != img_feat.shape[-2:]:
                    depth_feat = nn.functional.interpolate(
                        depth_feat, size=img_feat.shape[-2:], mode="bilinear", align_corners=False
                    )
                feat = torch.cat([img_feat, depth_feat], dim=1)
            fused_feats.append(self.fusion[i](feat))

        fused_feat = fused_feats[-1]
        img_idxs = end_points["img_idxs"]
        point_img_feat = self._gather_point_features(fused_feat, img_idxs)
        seed_features = self.seed_proj(point_img_feat).transpose(1, 2)  # (B, seed_feat_dim, N)

        # downstream identical to single-modality pipeline
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # (B,N,C)

        objectness_score = end_points["objectness_score"]
        graspness_score = end_points["graspness_score"].squeeze(1)

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = objectness_pred == 1
        graspness_mask = graspness_score > cfgs.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.0

        for i in range(B):
            cur_mask = graspable_mask[i]
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
            graspable_num_batch += cur_idx.numel()

            if cur_idx.numel() == 0:
                ridx = torch.randint(0, point_num, (self.M_points,), device=seed_xyz.device)
                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()
                cur_feat_mc = seed_features_flipped[i].index_select(0, ridx).contiguous()
                cur_feat = cur_feat_mc.transpose(0, 1).contiguous()
                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            if cur_idx.numel() < self.M_points:
                rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=seed_xyz.device)
                ridx = cur_idx[rep]
                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()
                cur_feat_mc = seed_features_flipped[i].index_select(0, ridx).contiguous()
                cur_feat = cur_feat_mc.transpose(0, 1).contiguous()
                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            xyz_in = seed_xyz[i].index_select(0, cur_idx).unsqueeze(0).contiguous()
            fps_idxs = end_points.get("fps_idx_override")
            if fps_idxs is None:
                from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

                fps_idxs = furthest_point_sample(xyz_in, self.M_points)
                fps_idxs = fps_idxs.to(device=xyz_in.device, dtype=torch.int32).contiguous()
                feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()
                cur_seed_xyz = gather_operation(xyz_in.transpose(1, 2).contiguous(), fps_idxs)
                cur_seed_xyz = cur_seed_xyz.transpose(1, 2).squeeze(0).contiguous()
                cur_feat = gather_operation(feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs)
                cur_feat = cur_feat.squeeze(0).contiguous()
            else:
                cur_seed_xyz = xyz_in.squeeze(0)[: self.M_points]
                cur_feat = seed_features_flipped[i].index_select(0, cur_idx)[: self.M_points].transpose(0, 1)

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        seed_features_graspable = torch.stack(seed_features_graspable, 0)
        end_points["xyz_graspable"] = seed_xyz_graspable.contiguous()
        end_points["D: Graspable Points"] = (
            torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32) / float(B)
        ).detach().reshape(())

        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            from utils.label_generation import process_grasp_labels

            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        group_features = self.cy_group(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.grasp_head(group_features, end_points)
        return end_points
