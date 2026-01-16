import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.backbone import TDUnet
from models.modules_economicgrasp import GraspableNet, ViewNet, Cylinder_Grouping_Global_Interaction, Grasp_Head_Local_Interaction
from utils.label_generation import process_grasp_labels, batch_viewpoint_params_to_matrix
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from utils.arguments import cfgs


class economicgrasp(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True, voxel_size=0.005):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points = cfgs.m_point
        self.num_view = cfgs.num_view
        self.voxel_size = voxel_size

        # Backbone
        self.backbone = TDUnet(in_channels=3, out_channels=self.seed_feature_dim, D=3)

        # Objectness and graspness
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)

        # View Selection
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)

        # Cylinder Grouping
        self.cy_group = Cylinder_Grouping_Global_Interaction(nsample=16, cylinder_radius=cylinder_radius,
                                                            seed_feature_dim=self.seed_feature_dim)

        # Depth and Score searching
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, [B, point_num (15000)， 3]
        B, point_num, _ = seed_xyz.shape  # batch _size

        # Generate input to meet the Minkowski Engine
        coordinates_batch, features_batch = ME.utils.sparse_collate(
                                             [coord for coord in end_points['coordinates_for_voxel']],
                                             [feat for feat in np.ones_like(seed_xyz.cpu()).astype(np.float32)])
        coordinates_batch, features_batch, _, end_points['quantize2original'] = \
            ME.utils.sparse_quantize(coordinates_batch, features_batch, return_index=True, return_inverse=True)

        coordinates_batch = coordinates_batch.cuda()
        features_batch = features_batch.cuda()

        end_points['coors'] = coordinates_batch  # [points of the whole scenes after quantize, 3(coors) + 1(index)]
        end_points['feats'] = features_batch  # [points of the whole scenes after quantize, 3 (input feature dim)]
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)

        # Minkowski Backbone
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        # [B (batch size), 512 (feature dim), 20000 (points in a scene)]

        # Generate the masks of the objectness and the graspness
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)
        objectness_score = end_points['objectness_score']
        # [B (batch size), 2 (object classification), 20000 (points in a scene)]
        graspness_score = end_points['graspness_score'].squeeze(1)  # [B (batch size), 20000 (points in a scene)]
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > cfgs.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        # Generate the downsample point (1024 per scene) using the furthest point sampling
        # seed_features_graspable = []
        # seed_xyz_graspable = []
        # graspable_num_batch = 0.
        # for i in range(B):
        #     cur_mask = graspable_mask[i]
        #     graspable_num_batch += cur_mask.sum()
        #     cur_feat = seed_features_flipped[i][cur_mask]
        #     cur_seed_xyz = seed_xyz[i][cur_mask]

        #     cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
        #     fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
        #     cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
        #     cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()
        #     cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
        #     cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()

        #     seed_features_graspable.append(cur_feat)
        #     seed_xyz_graspable.append(cur_seed_xyz)
        # seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        # # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        # seed_features_graspable = torch.stack(seed_features_graspable)
        # end_points['xyz_graspable'] = seed_xyz_graspable
        # end_points['D: Graspable Points'] = graspable_num_batch / B

        # -------- FPS downsample (robust, always return (C,M) and (M,3)) --------
        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.

        for i in range(B):
            cur_mask = graspable_mask[i]  # (N,)
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)  # (Ng,)
            graspable_num_batch += cur_idx.numel()

            # ========== Case 1: Ng == 0 -> random sample from all points ==========
            if cur_idx.numel() == 0:
                ridx = torch.randint(0, point_num, (self.M_points,), device=seed_xyz.device)

                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            # ========== Case 2: 0 < Ng < M -> pad with replacement (no FPS/gather) ==========
            if cur_idx.numel() < self.M_points:
                rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=seed_xyz.device)
                ridx = cur_idx[rep]  # (M,)

                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            # ========== Case 3: Ng >= M -> FPS + gather_operation ==========
            xyz_in = seed_xyz[i].index_select(0, cur_idx).unsqueeze(0).contiguous()        # (1, Ng, 3)
            fps_idxs = furthest_point_sample(xyz_in, self.M_points)                        # (1, M)
            fps_idxs = fps_idxs.to(device=xyz_in.device, dtype=torch.int32).contiguous()   # ✅ must be int32

            # (optional) debug safety check
            # N = xyz_in.size(1)
            # if int(fps_idxs.max()) >= N or int(fps_idxs.min()) < 0:
            #     raise RuntimeError(f"FPS idx out of range: [{fps_idxs.min().item()}, {fps_idxs.max().item()}], N={N}")

            cur_seed_xyz = gather_operation(xyz_in.transpose(1, 2).contiguous(), fps_idxs) \
                            .transpose(1, 2).squeeze(0).contiguous()                      # (M, 3)

            feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()       # (Ng, C)
            cur_feat = gather_operation(feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs).squeeze(0).contiguous()                                          # ✅ (C, M)

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)            # (B, M, 3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0)  # ✅ (B, C, M)

        end_points['xyz_graspable'] = seed_xyz_graspable.contiguous()
        end_points['D: Graspable Points'] = (
            torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32)
            / float(B)
        ).detach().reshape(())

        # Select the view for each point
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = seed_features_graspable + res_feat
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]

        # Generate the labels
        if self.is_training:
            # generate the scene-level grasp labels from the object-level grasp label and the object poses
            # map the scene sampled points to the labeled object points
            # (note that the labeled object points and the sampled points
            # may not 100% match due to the sampling and argumentation)
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        # Cylinder grouping
        group_features = self.cy_group(seed_xyz_graspable.contiguous(),
                                   seed_features_graspable.contiguous(),
                                   grasp_top_views_rot)

        # Width and score predicting
        end_points = self.grasp_head(group_features, end_points)

        return end_points

from models.pspnet import PSPNet
class economicgrasp_multi(nn.Module):
    """Point-center multi-modal EconomicGrasp.

    This variant keeps the original point-centric grasp head while replacing the
    feature extractor with an image-guided pathway:

    1. A PSPNet backbone extracts dense 2D features from the RGB image.
    2. Each 3D point gathers its aligned pixel feature via ``img_idxs``
       (precomputed during data loading) to build per-point image descriptors.
    3. The Minkowski U-Net consumes these point-wise image features to produce
       sparse 3D features, which flow through the existing graspable/view/GR
       heads without structural changes.

    The design remains robust to missing graspable points by padding/FPS logic
    in the forward pass, making it suitable for rapid experimentation with
    multi-modal cues while preserving the point-center inference interface.
    """

    def __init__(self,
                 cylinder_radius=0.05,
                 seed_feat_dim=512,
                 img_feat_dim=64,
                 is_training=True,
                 voxel_size=0.005):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.img_feat_dim = img_feat_dim

        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points = cfgs.m_point
        self.num_view = cfgs.num_view
        self.voxel_size = voxel_size

        # --- image backbone (same as your mmgnet) ---
        self.img_backbone = PSPNet(
            sizes=(1, 2, 3, 6),
            psp_size=512,
            deep_features_size=img_feat_dim,
            backend='resnet34'
        )

        # --- early fusion: point backbone consumes img features ---
        self.backbone = TDUnet(in_channels=img_feat_dim, out_channels=self.seed_feature_dim, D=3)

        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, np.math.sqrt(2. / n))
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # def forward(self, end_points):
    #     seed_xyz = end_points['point_clouds']  # (B,N,3)
    #     B, point_num, _ = seed_xyz.shape

    #     # -------- Early fusion: build per-point image features --------
    #     img = end_points['img']               # (B,3,H,W)
    #     img_idxs = end_points['img_idxs']     # (B,N) flattened indices
    #     point_img_feat = self._gather_point_img_features(img, img_idxs)  # (B,N,C=img_feat_dim)

    #     # -------- Minkowski input: coords + feats(=image features) --------
    #     # coords: end_points['coordinates_for_voxel'] 是 list[B]，每个 (N,4) 或 (N,3+batch)
    #     coordinates_batch, features_batch = ME.utils.sparse_collate(
    #         coords=[coord for coord in end_points['coordinates_for_voxel']],
    #         feats=[feat for feat in point_img_feat],
    #         dtype=torch.float32
    #     )

    #     # sparse quantize + inverse map
    #     coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
    #         coordinates_batch,
    #         features_batch,
    #         return_index=True,
    #         return_inverse=True,
    #         device=seed_xyz.device
    #     )
    #     end_points['quantize2original'] = quantize2original

    #     mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)

    #     # -------- Backbone --------
    #     seed_features = self.backbone(mink_input).F
    #     seed_features = seed_features[quantize2original].view(B, point_num, -1).transpose(1, 2)
    #     # seed_features: (B, seed_feat_dim, N)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # (B,N,3)
        B, point_num, _ = seed_xyz.shape

        # -------- image -> per-point features --------
        img = end_points['img']         # (B,3,448,448)
        img_idxs = end_points['img_idxs']  # (B,N) flatten idx in 448*448

        img_feat = self.img_backbone(img)   # (B,C,H,W) 期望 H=W=448（与你 mmgnet 一致）
        Bf, C, Hf, Wf = img_feat.shape
        img_feat = img_feat.view(Bf, C, -1)  # (B,C,Hf*Wf)

        img_idxs = img_idxs.long().clamp(0, Hf * Wf - 1)
        img_idxs = img_idxs.unsqueeze(1).expand(-1, C, -1)        # (B,C,N)
        point_img_feat = torch.gather(img_feat, 2, img_idxs)      # (B,C,N)
        point_img_feat = point_img_feat.transpose(1, 2).contiguous()  # (B,N,C)

        # -------- Minkowski input: coords + feats(image) --------
        coordinates_batch, features_batch = ME.utils.sparse_collate(
            coords=[coord for coord in end_points['coordinates_for_voxel']],
            feats=[point_img_feat[i] for i in range(B)],
            dtype=torch.float32
        )

        coordinates_batch, features_batch, _, end_points['quantize2original'] = ME.utils.sparse_quantize(
            coordinates_batch, features_batch,
            return_index=True, return_inverse=True,
            device=seed_xyz.device
        )

        mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)

        # -------- Minkowski backbone --------
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

        # -------- Graspable mask --------
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # (B,N,C)

        objectness_score = end_points['objectness_score']       # (B,2,N)
        graspness_score = end_points['graspness_score'].squeeze(1)  # (B,N)

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > cfgs.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        # -------- FPS downsample (robust, always return (C,M) and (M,3)) --------
        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.

        for i in range(B):
            cur_mask = graspable_mask[i]  # (N,)
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)  # (Ng,)
            graspable_num_batch += cur_idx.numel()

            # ========== Case 1: Ng == 0 -> random sample from all points ==========
            if cur_idx.numel() == 0:
                ridx = torch.randint(0, point_num, (self.M_points,), device=seed_xyz.device)

                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            # ========== Case 2: 0 < Ng < M -> pad with replacement (no FPS/gather) ==========
            if cur_idx.numel() < self.M_points:
                rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=seed_xyz.device)
                ridx = cur_idx[rep]  # (M,)

                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            # ========== Case 3: Ng >= M -> FPS + gather_operation ==========
            xyz_in = seed_xyz[i].index_select(0, cur_idx).unsqueeze(0).contiguous()        # (1, Ng, 3)
            fps_idxs = furthest_point_sample(xyz_in, self.M_points)                        # (1, M)
            fps_idxs = fps_idxs.to(device=xyz_in.device, dtype=torch.int32).contiguous()   # ✅ must be int32

            # (optional) debug safety check
            # N = xyz_in.size(1)
            # if int(fps_idxs.max()) >= N or int(fps_idxs.min()) < 0:
            #     raise RuntimeError(f"FPS idx out of range: [{fps_idxs.min().item()}, {fps_idxs.max().item()}], N={N}")

            cur_seed_xyz = gather_operation(xyz_in.transpose(1, 2).contiguous(), fps_idxs) \
                            .transpose(1, 2).squeeze(0).contiguous()                      # (M, 3)

            feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()       # (Ng, C)
            cur_feat = gather_operation(feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs).squeeze(0).contiguous()                                          # ✅ (C, M)

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)            # (B, M, 3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0)  # ✅ (B, C, M)
        # 保证 xyz_graspable 是 tensor (B, M, 3)
        end_points['xyz_graspable'] = seed_xyz_graspable.contiguous()

        # 保证统计量是 0-d torch Tensor（而不是 python float）
        end_points['D: Graspable Points'] = (
            torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32)
            / float(B)
        ).detach().reshape(())

        # -------- View selection --------
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        # -------- Label processing --------
        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        # -------- Grouping + Head --------
        group_features = self.cy_group(
            seed_xyz_graspable.contiguous(),
            seed_features_graspable.contiguous(),
            grasp_top_views_rot
        )
        end_points = self.grasp_head(group_features, end_points)
        return end_points

# score cls
def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        # composite score estimation
        grasp_score_prob = end_points['grasp_score_pred'][i].float()
        score = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1]).view(-1, 1).expand(-1, grasp_score_prob.shape[1]).to(grasp_score_prob)
        score = torch.sum(score * grasp_score_prob, dim=0)
        grasp_score = score.view(-1, 1)

        grasp_angle_pred = end_points['grasp_angle_pred'][i].float()
        grasp_angle, grasp_angle_indxs = torch.max(grasp_angle_pred.squeeze(0), 0)
        grasp_angle = grasp_angle_indxs * np.pi / 12

        grasp_depth_pred = end_points['grasp_depth_pred'][i].float()
        grasp_depth, grasp_depth_indxs = torch.max(grasp_depth_pred.squeeze(0), 0)
        grasp_depth = (grasp_depth_indxs + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)

        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = torch.clamp(grasp_width, min=0., max=cfgs.grasp_max_width)
        grasp_width = grasp_width.view(-1, 1)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(cfgs.m_point, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds