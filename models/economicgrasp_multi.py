import torch
import torch.nn as nn

from models.bip3d.modules import ChannelMapper, ResNet, SwinTransformer
from models.bip3d.spatial_enhancer import BatchDepthProbGTGenerator, DepthFusionSpatialEnhancer
from models.modules_economicgrasp import (
    Cylinder_Grouping_Global_Interaction,
    Grasp_Head_Local_Interaction,
    GraspableNet,
    ViewNet,
)
# from utils.arguments import cfgs
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
def _ensure_bvchw(x: torch.Tensor, is_img: bool) -> torch.Tensor:
    """
    Ensure tensor has shape (B, V, C, H, W).
    imgs: could be (B,3,H,W) or (B,V,3,H,W)
    depths: could be (B,1,H,W) or (B,V,1,H,W)
    """
    if x.dim() == 4:
        # (B,C,H,W) -> (B,1,C,H,W)
        return x.unsqueeze(1)
    if x.dim() == 5:
        return x
    raise ValueError(f"Expect 4D/5D tensor for {'img' if is_img else 'depth'}, got {x.shape}")


def _flat_to_uv(flat: torch.Tensor, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    # flat: (B,N) or (B,V,N)
    u = flat % W
    v = torch.div(flat, W, rounding_mode="floor")
    return u, v


def _gather_points_from_fmap_single_view(
    fmap: torch.Tensor,          # (B, C, Hf, Wf)
    img_idxs: torch.Tensor,      # (B, N)  flatten in (H,W)
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Return per-point feature: (B, N, C) by integer gather (nearest neighbor) on feature map.
    """
    B, C, Hf, Wf = fmap.shape
    img_idxs = img_idxs.long()

    # (B,N) -> (u,v) in original image
    u, v = _flat_to_uv(img_idxs, W)

    # map to feature map coords
    uf = torch.clamp((u.float() * (Wf / float(W))).floor().long(), 0, Wf - 1)
    vf = torch.clamp((v.float() * (Hf / float(H))).floor().long(), 0, Hf - 1)
    idx_f = vf * Wf + uf  # (B,N) flatten in Hf*Wf

    fmap_flat = fmap.view(B, C, -1)  # (B,C,Hf*Wf)
    idx_f = idx_f.clamp(0, Hf * Wf - 1)
    idx_f = idx_f.unsqueeze(1).expand(-1, C, -1)  # (B,C,N)
    feat = torch.gather(fmap_flat, 2, idx_f)      # (B,C,N)
    return feat.transpose(1, 2).contiguous()      # (B,N,C)


class economicgrasp_multi_bip3d(nn.Module):
    """
    Image-center (BIP3D-style) multi-modal feature extractor + Point-center EconomicGrasp heads.

    Inputs (end_points dict):
      - point_clouds: (B,N,3)
      - imgs or img:  (B,3,H,W) or (B,V,3,H,W)
      - depths:       (B,1,H,W) or (B,V,1,H,W)   (optional if with_depth=False)
      - img_idxs:     (B,N) flatten indices in H*W (aligned with imgs input size)
      - image_wh:     (B,2) or (2,) or (1,2)  (optional; else infer from imgs)
      - projection_mat:(B,V,4,4) or (B,4,4) or (4,4) (optional; else identity)

    Notes:
      - 默认只用 view=0 来给点取特征（GraspNet 单视角场景就是 V=1）。
      - 若未来要多视角，需要额外提供每个点对应的 view id / 或 img_idxs 做成 (B,V,N)。
    """

    def __init__(
        self,
        cylinder_radius: float = 0.05,
        seed_feat_dim: int = 512,
        embed_dims: int = 128,
        with_depth: bool = True,
        depth_feat_dim: int = 32,
        is_training: bool = True,
        num_view: int = 300,
        num_angle: int = 12,
        num_depth: int = 4,
        m_point: int = 1024,
        graspness_threshold: float = 0.1,
        # spatial enhancer
        use_spatial_enhancer: bool = True,
        spatial_num_depth: int = 64,
        spatial_min_depth: float = 0.25,
        spatial_max_depth: float = 10.0,
        loss_depth_weight: float = -1.0,  # <=0 表示不算 depth loss
        point_level: int = 0,             # 用 neck 输出哪一层做 per-point gather：0=28x28, 1=14x14, 2=7x7, 3=4x4
    ):
        super().__init__()
        self.is_training = is_training

        self.seed_feature_dim = seed_feat_dim
        self.embed_dims = embed_dims
        self.with_depth = with_depth
        self.depth_feat_dim = depth_feat_dim

        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.M_points = m_point
        self.graspness_threshold = graspness_threshold
        self.point_level = point_level

        # -------- BIP3D image encoder + neck --------
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

        # -------- depth encoder + neck --------
        if self.with_depth:
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
                out_channels=depth_feat_dim,
                norm_cfg=dict(type=nn.GroupNorm, num_groups=4),
                num_outs=4,
            )
        else:
            self.depth_backbone = None
            self.depth_neck = None

        # -------- spatial enhancer (DepthFusionSpatialEnhancer) --------
        self.use_spatial_enhancer = use_spatial_enhancer and self.with_depth
        if self.use_spatial_enhancer:
            self.spatial_enhancer = DepthFusionSpatialEnhancer(
                embed_dims=embed_dims,
                feature_3d_dim=depth_feat_dim,
                num_depth_layers=2,
                min_depth=spatial_min_depth,
                max_depth=spatial_max_depth,
                num_depth=spatial_num_depth,
                with_feature_3d=True,
                loss_depth_weight=loss_depth_weight,
            )
            # 训练时如果要 depth_prob_loss，必须生成 depth_prob_gt
            if loss_depth_weight > 0:
                # stride 只是给 gt generator 用来 avg_pool；224->(28,14,7,4) 对应 stride 约 (8,16,32,56)
                self.depth_prob_gt_gen = BatchDepthProbGTGenerator(
                    stride=[8, 16, 32, 56],
                    min_depth=spatial_min_depth,
                    max_depth=spatial_max_depth,
                    num_depth=spatial_num_depth,
                    origin_stride=1,
                    input_key="depths",
                    output_key="depth_prob_gt",
                    max_valid_depth=None,
                    valid_threshold=-1,
                )
            else:
                self.depth_prob_gt_gen = None
        else:
            self.spatial_enhancer = None
            self.depth_prob_gt_gen = None

        # -------- per-point projection (replace Minkowski sparse conv) --------
        # (B,N,embed_dims) -> (B,seed_feat_dim,N)
        self.point_proj = nn.Sequential(
            nn.Conv1d(embed_dims, seed_feat_dim, 1, bias=False),
            nn.BatchNorm1d(seed_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(seed_feat_dim, seed_feat_dim, 1, bias=False),
            nn.BatchNorm1d(seed_feat_dim),
            nn.ReLU(inplace=True),
        )

        # -------- EconomicGrasp heads (unchanged) --------
        self.graspable = GraspableNet(seed_feature_dim=seed_feat_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=seed_feat_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=seed_feat_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

    def _extract_feat_2d_3d(self, imgs: torch.Tensor, depths: torch.Tensor | None):
        """
        imgs: (B,V,3,H,W)
        depths:(B,V,1,H,W) or None
        returns feature_maps(list[Tensor]), feature_3d(list[Tensor]|None)
          feature_maps: list of 4 levels, each (B,V,embed_dims,Hl,Wl)
          feature_3d:   list of 4 levels, each (B,V,depth_feat_dim,Hl,Wl)
        """
        B, V, _, H, W = imgs.shape
        imgs_flat = imgs.flatten(0, 1)  # (B*V,3,H,W)

        img_feats = self.img_backbone(imgs_flat)          # tuple/list of 3
        img_feats = self.img_neck(list(img_feats))        # list of 4
        img_feats = [f.unflatten(0, (B, V)) for f in img_feats]  # (B,V,C,H,W)

        if self.with_depth and depths is not None:
            depths_flat = depths.flatten(0, 1)            # (B*V,1,H,W)
            dep_feats = self.depth_backbone(depths_flat)  # tuple/list of 3
            dep_feats = self.depth_neck(list(dep_feats))  # list of 4
            dep_feats = [f.unflatten(0, (B, V)) for f in dep_feats]
        else:
            dep_feats = None

        return img_feats, dep_feats

    def _prepare_camera_inputs(self, end_points: dict, B: int, V: int, H: int, W: int, device):
        # image_wh
        if "image_wh" in end_points:
            image_wh = end_points["image_wh"]
            if torch.is_tensor(image_wh):
                image_wh = image_wh.to(device)
                if image_wh.numel() == 2:
                    image_wh = image_wh.reshape(1, 2).repeat(B, 1)
                elif image_wh.shape[-1] == 2 and image_wh.dim() == 2:
                    pass
                else:
                    image_wh = image_wh.reshape(B, 2)
            else:
                image_wh = torch.tensor([W, H], device=device, dtype=torch.float32).reshape(1, 2).repeat(B, 1)
        else:
            image_wh = torch.tensor([W, H], device=device, dtype=torch.float32).reshape(1, 2).repeat(B, 1)

        # projection_mat
        if "projection_mat" in end_points:
            P = end_points["projection_mat"]
            if not torch.is_tensor(P):
                P = torch.as_tensor(P, dtype=torch.float32, device=device)
            else:
                P = P.to(device)

            if P.dim() == 2 and P.shape == (4, 4):
                P = P.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)  # (B,V,4,4)
            elif P.dim() == 3 and P.shape[-2:] == (4, 4):
                # (B,4,4) -> (B,V,4,4)
                P = P.unsqueeze(1).repeat(1, V, 1, 1)
            elif P.dim() == 4:
                # (B,V,4,4)
                pass
            else:
                raise ValueError(f"projection_mat shape not supported: {P.shape}")
        else:
            P = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)

        return image_wh, P

    def forward(self, end_points: dict):
        # -------- inputs --------
        xyz = end_points["point_clouds"]  # (B,N,3)
        B, N, _ = xyz.shape
        device = xyz.device

        imgs = end_points.get("imgs", end_points.get("img", None))
        if imgs is None:
            raise KeyError("end_points must have 'imgs' (preferred) or 'img'")
        imgs = _ensure_bvchw(imgs, is_img=True).to(device)

        depths = end_points.get("depths", None)
        if depths is not None:
            depths = _ensure_bvchw(depths, is_img=False).to(device)

        # assume point cloud aligned to view0
        _, V, _, H, W = imgs.shape

        img_idxs = end_points["img_idxs"].to(device)
        if img_idxs.dim() == 3:
            # (B,V,N) -> take view0 by default
            img_idxs = img_idxs[:, 0, :]
        elif img_idxs.dim() != 2:
            raise ValueError(f"img_idxs should be (B,N) or (B,V,N), got {img_idxs.shape}")

        # -------- BIP3D feature extraction --------
        feat_maps, feat_3d = self._extract_feat_2d_3d(imgs, depths)

        # -------- spatial enhancer (optional) --------
        if self.use_spatial_enhancer and self.spatial_enhancer is not None:
            image_wh, proj = self._prepare_camera_inputs(end_points, B, V, H, W, device)

            # training: generate depth_prob_gt if needed
            inputs_for_spatial = {
                "image_wh": image_wh,
                "projection_mat": proj,
            }
            if self.training and self.depth_prob_gt_gen is not None:
                # generator reads from end_points["depths"] in (B,V,1,H,W)
                tmp = {"depths": depths}
                tmp = self.depth_prob_gt_gen(tmp)
                inputs_for_spatial["depth_prob_gt"] = tmp["depth_prob_gt"]

            feat_maps, depth_prob, loss_depth = self.spatial_enhancer(
                feature_maps=feat_maps,
                feature_3d=feat_3d,
                inputs=inputs_for_spatial,
            )
            end_points["depth_prob"] = depth_prob
            if loss_depth is not None:
                end_points["loss/depth_prob_loss"] = loss_depth

        # -------- gather per-point feature from chosen level (view0) --------
        lvl = int(self.point_level)
        lvl = max(0, min(lvl, len(feat_maps) - 1))
        fmap = feat_maps[lvl][:, 0]  # (B,C,Hl,Wl) take view0
        pt_feat = _gather_points_from_fmap_single_view(fmap, img_idxs, H=H, W=W)  # (B,N,C)

        # -------- project to seed features (replace sparse conv) --------
        seed_features = self.point_proj(pt_feat.transpose(1, 2).contiguous())  # (B,seed_feat_dim,N)

        # -------- Graspable head --------
        end_points = self.graspable(seed_features, end_points)

        seed_features_flipped = seed_features.transpose(1, 2)  # (B,N,C)

        objectness_score = end_points["objectness_score"]           # (B,2,N)
        graspness_score = end_points["graspness_score"].squeeze(1)  # (B,N)

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > float(self.graspness_threshold)
        graspable_mask = objectness_mask & graspness_mask

        # -------- FPS downsample to M graspable points (robust) --------
        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.0

        for i in range(B):
            cur_mask = graspable_mask[i]
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
            graspable_num_batch += float(cur_idx.numel())

            # Ng == 0: random sample
            if cur_idx.numel() == 0:
                ridx = torch.randint(0, N, (self.M_points,), device=device)
                cur_xyz = xyz[i].index_select(0, ridx).contiguous()                # (M,3)
                cur_feat_mc = seed_features_flipped[i].index_select(0, ridx)      # (M,C)
                cur_feat = cur_feat_mc.transpose(0, 1).contiguous()               # (C,M)
                seed_xyz_graspable.append(cur_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            # Ng < M: sample with replacement (no FPS)
            if cur_idx.numel() < self.M_points:
                rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=device)
                ridx = cur_idx[rep]
                cur_xyz = xyz[i].index_select(0, ridx).contiguous()
                cur_feat_mc = seed_features_flipped[i].index_select(0, ridx).contiguous()
                cur_feat = cur_feat_mc.transpose(0, 1).contiguous()
                seed_xyz_graspable.append(cur_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            # Ng >= M: FPS + gather_operation
            xyz_in = xyz[i].index_select(0, cur_idx).unsqueeze(0).contiguous()  # (1,Ng,3)
            fps = furthest_point_sample(xyz_in, self.M_points)                 # (1,M)
            fps = fps.to(dtype=torch.int32, device=device).contiguous()

            cur_xyz = gather_operation(xyz_in.transpose(1, 2).contiguous(), fps) \
                        .transpose(1, 2).squeeze(0).contiguous()              # (M,3)

            feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()  # (Ng,C)
            cur_feat = gather_operation(feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps) \
                        .squeeze(0).contiguous()                                      # (C,M)

            seed_xyz_graspable.append(cur_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0).contiguous()            # (B,M,3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0).contiguous()  # (B,C,M)

        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = torch.as_tensor(
            graspable_num_batch / float(B), device=device, dtype=torch.float32
        ).reshape(())

        # -------- View selection --------
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        # -------- Labels (training) --------
        if self.is_training and self.training:
            # 你原来的 EconomicGrasp process_grasp_labels 直接用
            from models.economicgrasp import process_grasp_labels
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        # -------- Grouping + Grasp head --------
        group_features = self.cy_group(
            seed_xyz_graspable.contiguous(),
            seed_features_graspable.contiguous(),
            grasp_top_views_rot,
        )
        end_points = self.grasp_head(group_features, end_points)
        return end_points