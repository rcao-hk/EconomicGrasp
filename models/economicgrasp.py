import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.backbone import TDUnet
from models.modules_economicgrasp import GraspableNet, ViewNet, Cylinder_Grouping_Global_Interaction, Grasp_Head_Local_Interaction
from utils.label_generation import process_grasp_labels, batch_viewpoint_params_to_matrix
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from utils.arguments import cfgs
import os
import json


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
        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = torch.stack(seed_features_graspable)
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['D: Graspable Points'] = graspable_num_batch / B

        # -------- FPS downsample (robust, always return (C,M) and (M,3)) --------
        # seed_features_graspable = []
        # seed_xyz_graspable = []
        # graspable_num_batch = 0.

        # for i in range(B):
        #     cur_mask = graspable_mask[i]  # (N,)
        #     cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)  # (Ng,)
        #     graspable_num_batch += cur_idx.numel()

        #     # ========== Case 1: Ng == 0 -> random sample from all points ==========
        #     if cur_idx.numel() == 0:
        #         ridx = torch.randint(0, point_num, (self.M_points,), device=seed_xyz.device)

        #         cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
        #         cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
        #         cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

        #         seed_xyz_graspable.append(cur_seed_xyz)
        #         seed_features_graspable.append(cur_feat)
        #         continue

        #     # ========== Case 2: 0 < Ng < M -> pad with replacement (no FPS/gather) ==========
        #     if cur_idx.numel() < self.M_points:
        #         rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=seed_xyz.device)
        #         ridx = cur_idx[rep]  # (M,)

        #         cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
        #         cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
        #         cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

        #         seed_xyz_graspable.append(cur_seed_xyz)
        #         seed_features_graspable.append(cur_feat)
        #         continue

        #     # ========== Case 3: Ng >= M -> FPS + gather_operation ==========
        #     xyz_in = seed_xyz[i].index_select(0, cur_idx).unsqueeze(0).contiguous()        # (1, Ng, 3)
        #     fps_idxs = furthest_point_sample(xyz_in, self.M_points)                        # (1, M)
        #     fps_idxs = fps_idxs.to(device=xyz_in.device, dtype=torch.int32).contiguous()   # ✅ must be int32

        #     # (optional) debug safety check
        #     # N = xyz_in.size(1)
        #     # if int(fps_idxs.max()) >= N or int(fps_idxs.min()) < 0:
        #     #     raise RuntimeError(f"FPS idx out of range: [{fps_idxs.min().item()}, {fps_idxs.max().item()}], N={N}")

        #     cur_seed_xyz = gather_operation(xyz_in.transpose(1, 2).contiguous(), fps_idxs) \
        #                     .transpose(1, 2).squeeze(0).contiguous()                      # (M, 3)

        #     feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()       # (Ng, C)
        #     cur_feat = gather_operation(feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs).squeeze(0).contiguous()                                          # ✅ (C, M)

        #     seed_xyz_graspable.append(cur_seed_xyz)
        #     seed_features_graspable.append(cur_feat)

        # seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)            # (B, M, 3)
        # seed_features_graspable = torch.stack(seed_features_graspable, 0)  # ✅ (B, C, M)

        # end_points['xyz_graspable'] = seed_xyz_graspable.contiguous()
        # end_points['D: Graspable Points'] = (
        #     torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32)
        #     / float(B)
        # ).detach().reshape(())

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


class TDUnet_InterFuse(TDUnet):
    def __init__(self, in_channels_3d, out_channels, img_dim, D=3):
        super().__init__(in_channels=in_channels_3d, out_channels=out_channels, D=D)

        # encoder各层做 sparse concat，再用1x1 sparse conv压回原通道
        self.fuse_p1  = ME.MinkowskiConvolution(self.INIT_DIM + img_dim, self.INIT_DIM, kernel_size=1, dimension=D)   # 32+64 -> 32
        self.fuse_p2  = ME.MinkowskiConvolution(self.PLANES[0] + img_dim, self.PLANES[0], kernel_size=1, dimension=D) # 32+64 -> 32
        self.fuse_p4  = ME.MinkowskiConvolution(self.PLANES[1] + img_dim, self.PLANES[1], kernel_size=1, dimension=D) # 64+64 -> 64
        self.fuse_p8  = ME.MinkowskiConvolution(self.PLANES[2] + img_dim, self.PLANES[2], kernel_size=1, dimension=D) # 128+64 -> 128
        self.fuse_p16 = ME.MinkowskiConvolution(self.PLANES[3] + img_dim, self.PLANES[3], kernel_size=1, dimension=D) # 256+64 -> 256

    @staticmethod
    def _scatter_mean(feats: torch.Tensor, idx: torch.Tensor, M: int) -> torch.Tensor:
        # feats: (K,C), idx: (K,) -> (M,C)
        C = feats.shape[1]
        out = feats.new_zeros((M, C))
        cnt = feats.new_zeros((M, 1))
        out.index_add_(0, idx, feats)
        cnt.index_add_(0, idx, torch.ones((feats.shape[0], 1), device=feats.device, dtype=feats.dtype))
        return out / cnt.clamp_min_(1.0)

    @staticmethod
    def _make_bn_coords_from_coors(coors_list, stride: int, device) -> torch.Tensor:
        """
        coors_list: list(B) of (N,3) voxel coords, no batch col
        return: (B*N,4) with batch index, snapped to stride grid
        """
        coords4 = []
        for b, c in enumerate(coors_list):
            if not torch.is_tensor(c):
                c = torch.as_tensor(c, device=device)

            c = torch.floor(c).long() if c.dtype.is_floating_point else c.long()

            if stride > 1:
                c = torch.div(c, stride, rounding_mode='floor') * stride

            bcol = torch.full((c.shape[0], 1), b, device=device, dtype=torch.long)
            coords4.append(torch.cat([bcol, c], dim=1))
        return torch.cat(coords4, dim=0)

    @staticmethod
    def _pack_keys(coords4: torch.Tensor, shift_xyz: torch.Tensor, mx: int, my: int, mz: int) -> torch.Tensor:
        b = coords4[:, 0]
        xyz = coords4[:, 1:] + shift_xyz.view(1, 3)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return (((b * mx + x) * my + y) * mz + z)

    def _build_img_sparse_like(self, target: ME.SparseTensor, pfeat_BNC: torch.Tensor, coors_list, stride: int):
        """
        target: 当前3D sparse tensor
        pfeat_BNC: (B,N,C) 对应尺度的 per-point 2D feat
        coors_list: list(B), each (N,3), 与 pfeat 的 N 一一对应
        """
        device = target.F.device
        B, N, C = pfeat_BNC.shape
        feats_bnC = pfeat_BNC.reshape(B * N, C)

        tgt_coords = target.C.to(device).long()   # (M,4)
        M = tgt_coords.shape[0]

        pts_coords = self._make_bn_coords_from_coors(coors_list, stride=stride, device=device)  # (B*N,4)

        xyz_all = torch.cat([pts_coords[:, 1:], tgt_coords[:, 1:]], dim=0)
        min_xyz = xyz_all.min(dim=0).values
        shift = (-min_xyz).clamp_min(0).long()

        pts_xyz = pts_coords[:, 1:] + shift.view(1, 3)
        tgt_xyz = tgt_coords[:, 1:] + shift.view(1, 3)
        max_xyz = torch.max(pts_xyz.max(dim=0).values, tgt_xyz.max(dim=0).values).long()

        mx = int(max_xyz[0].item()) + 1
        my = int(max_xyz[1].item()) + 1
        mz = int(max_xyz[2].item()) + 1
        if mx * my * mz > 2**62:
            raise RuntimeError(f"[InterFuse] key packing overflow risk: mx*my*mz={mx*my*mz}")

        pts_key = self._pack_keys(pts_coords, shift, mx, my, mz)
        tgt_key = self._pack_keys(tgt_coords, shift, mx, my, mz)

        tgt_key_sorted, order = torch.sort(tgt_key)
        pos = torch.searchsorted(tgt_key_sorted, pts_key)
        valid = (pos < M) & (tgt_key_sorted[pos] == pts_key)

        if not torch.all(valid):
            bad = int((~valid).sum().item())
            raise RuntimeError(
                f"[InterFuse] {bad} points cannot map to target coords at stride={stride}. "
                f"Check coordinates_for_voxel / quantization consistency."
            )

        tgt_idx = order[pos]
        img_feat_u = self._scatter_mean(feats_bnC, tgt_idx, M)

        return ME.SparseTensor(
            features=img_feat_u,
            coordinate_map_key=target.coordinate_map_key,
            coordinate_manager=target.coordinate_manager
        )

    def forward(self, x_sparse: ME.SparseTensor, pfeat: dict, coors_list):
        # stride = 1
        out = self.conv0p1s1(x_sparse)
        out = self.bn0(out)
        out = self.relu(out)

        img1 = self._build_img_sparse_like(out, pfeat['p1'], coors_list, stride=1)
        out = self.fuse_p1(ME.cat(out, img1))
        out_p1 = out

        # stride = 2
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.block1(out)

        img2 = self._build_img_sparse_like(out, pfeat['p2'], coors_list, stride=2)
        out = self.fuse_p2(ME.cat(out, img2))
        out_b1p2 = out

        # stride = 4
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.block2(out)

        img4 = self._build_img_sparse_like(out, pfeat['p4'], coors_list, stride=4)
        out = self.fuse_p4(ME.cat(out, img4))
        out_b2p4 = out

        # stride = 8
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.block3(out)

        img8 = self._build_img_sparse_like(out, pfeat['p8'], coors_list, stride=8)
        out = self.fuse_p8(ME.cat(out, img8))
        out_b3p8 = out

        # stride = 16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        img16 = self._build_img_sparse_like(out, pfeat['p16'], coors_list, stride=16)
        out = self.fuse_p16(ME.cat(out, img16))

        # decoder unchanged
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)
    
    
from models.pspnet import PSPNet
# class economicgrasp_multi(nn.Module):
#     """Point-center multi-modal EconomicGrasp.

#     This variant keeps the original point-centric grasp head while replacing the
#     feature extractor with an image-guided pathway:

#     1. A PSPNet backbone extracts dense 2D features from the RGB image.
#     2. Each 3D point gathers its aligned pixel feature via ``img_idxs``
#        (precomputed during data loading) to build per-point image descriptors.
#     3. The Minkowski U-Net consumes these point-wise image features to produce
#        sparse 3D features, which flow through the existing graspable/view/GR
#        heads without structural changes.

#     The design remains robust to missing graspable points by padding/FPS logic
#     in the forward pass, making it suitable for rapid experimentation with
#     multi-modal cues while preserving the point-center inference interface.
#     """

#     def __init__(self,
#                  cylinder_radius=0.05,
#                  seed_feat_dim=512,
#                  img_feat_dim=64,
#                  is_training=True,
#                  voxel_size=0.005):
#         super().__init__()
#         self.is_training = is_training
#         self.seed_feature_dim = seed_feat_dim
#         self.img_feat_dim = img_feat_dim

#         self.num_depth = cfgs.num_depth
#         self.num_angle = cfgs.num_angle
#         self.M_points = cfgs.m_point
#         self.num_view = cfgs.num_view
#         self.voxel_size = voxel_size

#         # --- image backbone (same as your mmgnet) ---
#         self.img_backbone = PSPNet(
#             sizes=(1, 2, 3, 6),
#             psp_size=512,
#             deep_features_size=img_feat_dim,
#             backend='resnet34'
#         )

#         # --- early fusion: point backbone consumes img features ---
#         self.backbone = TDUnet(in_channels=img_feat_dim, out_channels=self.seed_feature_dim, D=3)

#         self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
#         self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
#         self.cy_group = Cylinder_Grouping_Global_Interaction(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
#         self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)
#         self._init_weights()

#     def _init_weights(self):
#         for name, module in self.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
#                 module.weight.data.normal_(0, np.math.sqrt(2. / n))
#             elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
#                 nn.init.constant_(module.weight, 1)
#                 nn.init.constant_(module.bias, 0)
#             elif isinstance(module, (nn.Linear, nn.Conv1d)):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)

#     # def forward(self, end_points):
#     #     seed_xyz = end_points['point_clouds']  # (B,N,3)
#     #     B, point_num, _ = seed_xyz.shape

#     #     # -------- Early fusion: build per-point image features --------
#     #     img = end_points['img']               # (B,3,H,W)
#     #     img_idxs = end_points['img_idxs']     # (B,N) flattened indices
#     #     point_img_feat = self._gather_point_img_features(img, img_idxs)  # (B,N,C=img_feat_dim)

#     #     # -------- Minkowski input: coords + feats(=image features) --------
#     #     # coords: end_points['coordinates_for_voxel'] 是 list[B]，每个 (N,4) 或 (N,3+batch)
#     #     coordinates_batch, features_batch = ME.utils.sparse_collate(
#     #         coords=[coord for coord in end_points['coordinates_for_voxel']],
#     #         feats=[feat for feat in point_img_feat],
#     #         dtype=torch.float32
#     #     )

#     #     # sparse quantize + inverse map
#     #     coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
#     #         coordinates_batch,
#     #         features_batch,
#     #         return_index=True,
#     #         return_inverse=True,
#     #         device=seed_xyz.device
#     #     )
#     #     end_points['quantize2original'] = quantize2original

#     #     mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)

#     #     # -------- Backbone --------
#     #     seed_features = self.backbone(mink_input).F
#     #     seed_features = seed_features[quantize2original].view(B, point_num, -1).transpose(1, 2)
#     #     # seed_features: (B, seed_feat_dim, N)

#     def forward(self, end_points):
#         seed_xyz = end_points['point_clouds']  # (B,N,3)
#         B, point_num, _ = seed_xyz.shape

#         # -------- image -> per-point features --------
#         img = end_points['img']         # (B,3,448,448)
#         img_idxs = end_points['img_idxs']  # (B,N) flatten idx in 448*448

#         img_feat = self.img_backbone(img)   # (B,C,H,W) 期望 H=W=448（与你 mmgnet 一致）
#         Bf, C, Hf, Wf = img_feat.shape
#         img_feat = img_feat.view(Bf, C, -1)  # (B,C,Hf*Wf)

#         img_idxs = img_idxs.long().clamp(0, Hf * Wf - 1)
#         img_idxs = img_idxs.unsqueeze(1).expand(-1, C, -1)        # (B,C,N)
#         point_img_feat = torch.gather(img_feat, 2, img_idxs)      # (B,C,N)
#         point_img_feat = point_img_feat.transpose(1, 2).contiguous()  # (B,N,C)

#         # -------- Minkowski input: coords + feats(image) --------
#         coordinates_batch, features_batch = ME.utils.sparse_collate(
#             coords=[coord for coord in end_points['coordinates_for_voxel']],
#             feats=[point_img_feat[i] for i in range(B)],
#             dtype=torch.float32
#         )

#         coordinates_batch, features_batch, _, end_points['quantize2original'] = ME.utils.sparse_quantize(
#             coordinates_batch, features_batch,
#             return_index=True, return_inverse=True,
#             device=seed_xyz.device
#         )

#         mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)

#         # -------- Minkowski backbone --------
#         seed_features = self.backbone(mink_input).F
#         seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

#         # -------- Graspable mask --------
#         end_points = self.graspable(seed_features, end_points)
#         seed_features_flipped = seed_features.transpose(1, 2)  # (B,N,C)

#         objectness_score = end_points['objectness_score']       # (B,2,N)
#         graspness_score = end_points['graspness_score'].squeeze(1)  # (B,N)

#         objectness_pred = torch.argmax(objectness_score, 1)
#         objectness_mask = (objectness_pred == 1)
#         graspness_mask = graspness_score > cfgs.graspness_threshold
#         graspable_mask = objectness_mask & graspness_mask

#         # Generate the downsample point (1024 per scene) using the furthest point sampling
#         seed_features_graspable = []
#         seed_xyz_graspable = []
#         graspable_num_batch = 0.
#         for i in range(B):
#             cur_mask = graspable_mask[i]
#             graspable_num_batch += cur_mask.sum()
#             cur_feat = seed_features_flipped[i][cur_mask]
#             cur_seed_xyz = seed_xyz[i][cur_mask]

#             cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
#             fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
#             cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
#             cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()
#             cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
#             cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()

#             seed_features_graspable.append(cur_feat)
#             seed_xyz_graspable.append(cur_seed_xyz)
#         seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
#         # [B (batch size), 512 (feature dim), 1024 (points after sample)]
#         seed_features_graspable = torch.stack(seed_features_graspable)
#         end_points['xyz_graspable'] = seed_xyz_graspable
#         end_points['D: Graspable Points'] = graspable_num_batch / B

#         # # -------- FPS downsample (robust, always return (C,M) and (M,3)) --------
#         # seed_features_graspable = []
#         # seed_xyz_graspable = []
#         # graspable_num_batch = 0.

#         # for i in range(B):
#         #     cur_mask = graspable_mask[i]  # (N,)
#         #     cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)  # (Ng,)
#         #     graspable_num_batch += cur_idx.numel()

#         #     # ========== Case 1: Ng == 0 -> random sample from all points ==========
#         #     if cur_idx.numel() == 0:
#         #         ridx = torch.randint(0, point_num, (self.M_points,), device=seed_xyz.device)

#         #         cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
#         #         cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
#         #         cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

#         #         seed_xyz_graspable.append(cur_seed_xyz)
#         #         seed_features_graspable.append(cur_feat)
#         #         continue

#         #     # ========== Case 2: 0 < Ng < M -> pad with replacement (no FPS/gather) ==========
#         #     if cur_idx.numel() < self.M_points:
#         #         rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=seed_xyz.device)
#         #         ridx = cur_idx[rep]  # (M,)

#         #         cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()              # (M, 3)
#         #         cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()# (M, C)
#         #         cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()                   # ✅ (C, M)

#         #         seed_xyz_graspable.append(cur_seed_xyz)
#         #         seed_features_graspable.append(cur_feat)
#         #         continue

#         #     # ========== Case 3: Ng >= M -> FPS + gather_operation ==========
#         #     xyz_in = seed_xyz[i].index_select(0, cur_idx).unsqueeze(0).contiguous()        # (1, Ng, 3)
#         #     fps_idxs = furthest_point_sample(xyz_in, self.M_points)                        # (1, M)
#         #     fps_idxs = fps_idxs.to(device=xyz_in.device, dtype=torch.int32).contiguous()   # ✅ must be int32

#         #     # (optional) debug safety check
#         #     # N = xyz_in.size(1)
#         #     # if int(fps_idxs.max()) >= N or int(fps_idxs.min()) < 0:
#         #     #     raise RuntimeError(f"FPS idx out of range: [{fps_idxs.min().item()}, {fps_idxs.max().item()}], N={N}")

#         #     cur_seed_xyz = gather_operation(xyz_in.transpose(1, 2).contiguous(), fps_idxs) \
#         #                     .transpose(1, 2).squeeze(0).contiguous()                      # (M, 3)

#         #     feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()       # (Ng, C)
#         #     cur_feat = gather_operation(feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs).squeeze(0).contiguous()                                          # ✅ (C, M)

#         #     seed_xyz_graspable.append(cur_seed_xyz)
#         #     seed_features_graspable.append(cur_feat)

#         # seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)            # (B, M, 3)
#         # seed_features_graspable = torch.stack(seed_features_graspable, 0)  # ✅ (B, C, M)
#         # # 保证 xyz_graspable 是 tensor (B, M, 3)
#         # end_points['xyz_graspable'] = seed_xyz_graspable.contiguous()

#         # # 保证统计量是 0-d torch Tensor（而不是 python float）
#         # end_points['D: Graspable Points'] = (
#         #     torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32)
#         #     / float(B)
#         # ).detach().reshape(())

#         # -------- View selection --------
#         end_points, res_feat = self.view(seed_features_graspable, end_points)
#         seed_features_graspable = seed_features_graspable + res_feat

#         # -------- Label processing --------
#         if self.is_training:
#             grasp_top_views_rot, end_points = process_grasp_labels(end_points)
#         else:
#             grasp_top_views_rot = end_points['grasp_top_view_rot']

#         # -------- Grouping + Head --------
#         group_features = self.cy_group(
#             seed_xyz_graspable.contiguous(),
#             seed_features_graspable.contiguous(),
#             grasp_top_views_rot
#         )
#         end_points = self.grasp_head(group_features, end_points)
#         return end_points


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _denorm_img(img_t: torch.Tensor) -> torch.Tensor:
    """
    img_t: (3,H,W), ImageNet normalized tensor
    return: (3,H,W) in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(3, 1, 1)
    out = img_t * std + mean
    return out.clamp(0, 1)

def _imgidx_to_xy(img_idxs: torch.Tensor, W: int):
    ys = torch.div(img_idxs, W, rounding_mode='floor')
    xs = img_idxs - ys * W
    return xs, ys

def _density_map(xs: np.ndarray, ys: np.ndarray, H: int, W: int) -> np.ndarray:
    den = np.zeros((H, W), dtype=np.float32)
    np.add.at(den, (ys, xs), 1.0)
    return den

def _avg_map(xs: np.ndarray, ys: np.ndarray, vals: np.ndarray, H: int, W: int) -> np.ndarray:
    s = np.zeros((H, W), dtype=np.float32)
    c = np.zeros((H, W), dtype=np.float32)
    np.add.at(s, (ys, xs), vals)
    np.add.at(c, (ys, xs), 1.0)
    return s / np.maximum(c, 1.0)

def _write_ply_xyz_rgb(path: str, xyz: np.ndarray, rgb: np.ndarray = None):
    """
    xyz: (N,3), rgb: (N,3) in [0,1] or [0,255]
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)

    with open(path, "w") as f:
        if rgb is None:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in xyz:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
                
                
class economicgrasp_multi(nn.Module):
    """
    Support two fusion modes:
      - early
      - intermediate
    """
    def __init__(self,
                 cylinder_radius=0.05,
                 seed_feat_dim=512,
                 img_feat_dim=64,
                 is_training=True,
                 voxel_size=0.005,
                 fuse_type='early',
                 pc_in_dim=3,
                 vis_dir=None,
                 vis_every=200,):
        super().__init__()
        assert fuse_type in ['early', 'intermediate']
        self.fuse_type = fuse_type
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.img_feat_dim = img_feat_dim
        self.pc_in_dim = pc_in_dim

        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points = cfgs.m_point
        self.num_view = cfgs.num_view
        self.voxel_size = voxel_size

        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self.max_points_2d = 30000
        self.max_points_ply = 50000
        self._vis_iter = 0

        if self.fuse_type == 'early':
            # 单尺度输出即可
            self.img_backbone = PSPNet(
                sizes=(1, 2, 3, 6),
                psp_size=512,
                deep_features_size=img_feat_dim,
                backend='resnet34'
            )
            self.backbone = TDUnet(
                in_channels=img_feat_dim,
                out_channels=self.seed_feature_dim,
                D=3
            )
            print('economicgrasp_multi: early fusion')

        elif self.fuse_type == 'intermediate':
            # intermediate 需要多尺度 pyramid 输出
            self.img_backbone = PSPNet(
                sizes=(1, 2, 3, 6),
                psp_size=512,
                backend='resnet34',
                deep_features_size=img_feat_dim,
                out_dim=img_feat_dim,
                return_pyramid=True,
                pretrained=True
            )
            self.backbone = TDUnet_InterFuse(
                in_channels_3d=pc_in_dim,
                out_channels=self.seed_feature_dim,
                img_dim=img_feat_dim,
                D=3
            )
            print('economicgrasp_multi: intermediate fusion')

        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16,
            cylinder_radius=cylinder_radius,
            seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(
            num_angle=self.num_angle,
            num_depth=self.num_depth
        )

        # self._init_weights()

    def enable_vis(self, vis_dir: str, vis_every: int = 200,
                max_points_2d: int = 30000, max_points_ply: int = 50000):
        self.vis_dir = str(vis_dir)
        self.vis_every = int(vis_every)
        self.max_points_2d = int(max_points_2d)
        self.max_points_ply = int(max_points_ply)
        self._vis_iter = 0
        _ensure_dir(self.vis_dir)
    
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

    def _gather_2d_to_points(self, feat2d: torch.Tensor, img_idxs: torch.Tensor, base_hw=(448, 448)):
        """
        feat2d: (B,C,Hf,Wf)
        img_idxs: (B,N), flatten idx in base_hw
        return: (B,N,C)
        """
        Hb, Wb = base_hw
        B, C, Hf, Wf = feat2d.shape

        ys = torch.div(img_idxs, Wb, rounding_mode='floor')
        xs = img_idxs - ys * Wb

        yf = torch.clamp((ys.float() * (Hf / Hb)).long(), 0, Hf - 1)
        xf = torch.clamp((xs.float() * (Wf / Wb)).long(), 0, Wf - 1)

        flat_f = (yf * Wf + xf)
        feat_flat = feat2d.view(B, C, -1)
        gather_idx = flat_f.unsqueeze(1).expand(-1, C, -1)
        out = torch.gather(feat_flat, 2, gather_idx)
        return out.transpose(1, 2).contiguous()

    @torch.no_grad()
    def _maybe_save_vis(
        self,
        end_points: dict,
        *,
        xyz_full: torch.Tensor,              # (B,N,3)
        img: torch.Tensor = None,            # (B,3,H,W)
        img_idxs: torch.Tensor = None,       # (B,N)
        objectness_prob: torch.Tensor = None,# (B,N)
        graspness_score: torch.Tensor = None,# (B,N)
        graspable_mask: torch.Tensor = None, # (B,N) bool
        xyz_graspable: torch.Tensor = None,  # (B,M,3)
        img_idxs_graspable: torch.Tensor = None, # (B,M)
        top_view_xyz: torch.Tensor = None,   # (B,M,3), optional
        stats: dict = None,
    ):
        if self.vis_dir is None:
            return

        do_vis = (self._vis_iter % self.vis_every == 0) or bool(end_points.get("force_vis", False))
        if not do_vis:
            return

        b = 0  # only save batch0
        scene_val = end_points.get("scene_idx", end_points.get("scene", "scene"))
        frame_val = end_points.get("frameid", end_points.get("frame_idx", "frame"))

        if torch.is_tensor(scene_val):
            tag_scene = str(int(scene_val[b].item())) if scene_val.numel() > 1 else str(int(scene_val.item()))
        elif isinstance(scene_val, (list, tuple)):
            tag_scene = str(scene_val[b])
        else:
            tag_scene = str(scene_val)

        if torch.is_tensor(frame_val):
            tag_frame = str(int(frame_val[b].item())) if frame_val.numel() > 1 else str(int(frame_val.item()))
        elif isinstance(frame_val, (list, tuple)):
            tag_frame = str(frame_val[b])
        else:
            tag_frame = str(frame_val)

        prefix = f"{tag_scene}_{tag_frame}_it{self._vis_iter:06d}"
        out_dir = os.path.join(self.vis_dir, prefix)
        _ensure_dir(out_dir)

        xyz_full_b = xyz_full[b].detach().cpu()            # (N,3)
        N = xyz_full_b.shape[0]
        img_b = _denorm_img(img[b]).detach().cpu() if img is not None else None
        img_idxs_b = img_idxs[b].detach().cpu() if img_idxs is not None else None
        objectness_prob_b = objectness_prob[b].detach().cpu() if objectness_prob is not None else None
        graspness_score_b = graspness_score[b].detach().cpu() if graspness_score is not None else None
        graspable_mask_b = graspable_mask[b].detach().cpu() if graspable_mask is not None else None
        xyz_graspable_b = xyz_graspable[b].detach().cpu() if xyz_graspable is not None else None
        img_idxs_graspable_b = img_idxs_graspable[b].detach().cpu() if img_idxs_graspable is not None else None
        top_view_xyz_b = top_view_xyz[b].detach().cpu() if top_view_xyz is not None else None

        # ---------- dump debug ----------
        dump = {
            "xyz_full": xyz_full_b.to(torch.float16),
            "img_idxs": (img_idxs_b.to(torch.int32) if img_idxs_b is not None else None),
            "objectness_prob": (objectness_prob_b.to(torch.float16) if objectness_prob_b is not None else None),
            "graspness_score": (graspness_score_b.to(torch.float16) if graspness_score_b is not None else None),
            "graspable_mask": (graspable_mask_b.to(torch.uint8) if graspable_mask_b is not None else None),
            "xyz_graspable": (xyz_graspable_b.to(torch.float16) if xyz_graspable_b is not None else None),
            "img_idxs_graspable": (img_idxs_graspable_b.to(torch.int32) if img_idxs_graspable_b is not None else None),
            "top_view_xyz": (top_view_xyz_b.to(torch.float16) if top_view_xyz_b is not None else None),
            "stats": stats or {},
        }
        torch.save(dump, os.path.join(out_dir, "debug.pt"))

        # ---------- save PLY ----------
        if N > self.max_points_ply:
            idx = torch.linspace(0, N - 1, steps=self.max_points_ply).long()
            xyz_ply = xyz_full_b[idx].numpy()
        else:
            xyz_ply = xyz_full_b.numpy()
        _write_ply_xyz_rgb(os.path.join(out_dir, "cloud_full.ply"), xyz_ply)

        if graspable_mask_b is not None and graspable_mask_b.any():
            xyz_g = xyz_full_b[graspable_mask_b].numpy()
            _write_ply_xyz_rgb(os.path.join(out_dir, "cloud_graspable.ply"), xyz_g)

        if xyz_graspable_b is not None:
            _write_ply_xyz_rgb(os.path.join(out_dir, "cloud_seed.ply"), xyz_graspable_b.numpy())

        # ---------- save top-view dir as npy ----------
        if top_view_xyz_b is not None:
            np.save(os.path.join(out_dir, "top_view_xyz.npy"), top_view_xyz_b.numpy())

        # ---------- 2D plots ----------
        try:
            import matplotlib.pyplot as plt
        except Exception:
            with open(os.path.join(out_dir, "note.txt"), "w") as f:
                f.write("matplotlib not available; skipped 2D images.\n")
            return

        if img_b is not None and img_idxs_b is not None:
            H, W = img_b.shape[1], img_b.shape[2]
            xs, ys = _imgidx_to_xy(img_idxs_b.long(), W=W)

            nn = xs.numel()
            step = max(1, nn // self.max_points_2d)
            xs_np = xs[::step].numpy()
            ys_np = ys[::step].numpy()
            img_np = img_b.permute(1, 2, 0).numpy()

            # rgb
            plt.figure(figsize=(6, 6))
            plt.imshow(img_np)
            plt.axis("off")
            plt.savefig(os.path.join(out_dir, "rgb.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # graspness scatter
            if graspness_score_b is not None:
                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                plt.scatter(xs_np, ys_np, s=3, c=graspness_score_b[::step].numpy(), cmap="jet", vmin=0, vmax=1)
                plt.colorbar(fraction=0.046)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, "rgb_graspness.png"), dpi=200, bbox_inches="tight")
                plt.close()

            # objectness prob scatter
            if objectness_prob_b is not None:
                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                plt.scatter(xs_np, ys_np, s=3, c=objectness_prob_b[::step].numpy(), cmap="jet", vmin=0, vmax=1)
                plt.colorbar(fraction=0.046)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, "rgb_objectness_prob.png"), dpi=200, bbox_inches="tight")
                plt.close()

            # graspable mask scatter
            if graspable_mask_b is not None:
                mask_np = graspable_mask_b[::step].numpy().astype(bool)
                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                plt.scatter(xs_np[~mask_np], ys_np[~mask_np], s=2, alpha=0.3)
                plt.scatter(xs_np[mask_np], ys_np[mask_np], s=4)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, "rgb_graspable_mask.png"), dpi=200, bbox_inches="tight")
                plt.close()

            # sampled seed overlay
            if img_idxs_graspable_b is not None:
                xs_s, ys_s = _imgidx_to_xy(img_idxs_graspable_b.long(), W=W)
                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                plt.scatter(xs_np, ys_np, s=2, alpha=0.25)
                plt.scatter(xs_s.numpy(), ys_s.numpy(), s=10)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, "rgb_seed.png"), dpi=200, bbox_inches="tight")
                plt.close()

            # maps
            if graspness_score_b is not None:
                grasp_map = _avg_map(xs.numpy(), ys.numpy(), graspness_score_b.numpy().astype(np.float32), H=H, W=W)
                plt.figure(figsize=(6, 6))
                plt.imshow(grasp_map, cmap="jet", vmin=0, vmax=1)
                plt.colorbar(fraction=0.046)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, "graspness_map.png"), dpi=200, bbox_inches="tight")
                plt.close()

            if objectness_prob_b is not None:
                obj_map = _avg_map(xs.numpy(), ys.numpy(), objectness_prob_b.numpy().astype(np.float32), H=H, W=W)
                plt.figure(figsize=(6, 6))
                plt.imshow(obj_map, cmap="jet", vmin=0, vmax=1)
                plt.colorbar(fraction=0.046)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, "objectness_prob_map.png"), dpi=200, bbox_inches="tight")
                plt.close()

            if graspable_mask_b is not None:
                mask_map = _avg_map(xs.numpy(), ys.numpy(), graspable_mask_b.numpy().astype(np.float32), H=H, W=W)
                plt.figure(figsize=(6, 6))
                plt.imshow(mask_map, cmap="gray", vmin=0, vmax=1)
                plt.colorbar(fraction=0.046)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, "graspable_mask_map.png"), dpi=200, bbox_inches="tight")
                plt.close()

            den = _density_map(xs.numpy(), ys.numpy(), H=H, W=W)
            plt.figure(figsize=(6, 6))
            plt.imshow(np.log1p(den), cmap="magma")
            plt.axis("off")
            plt.savefig(os.path.join(out_dir, "density_all.png"), dpi=200, bbox_inches="tight")
            plt.close()

        with open(os.path.join(out_dir, "stats.json"), "w") as f:
            json.dump(stats or {}, f, indent=2)
        
    def _sample_graspable_points(self, seed_xyz, seed_features, graspable_mask, img_idxs=None):
        """
        EcoGrasp original strategy + optional img_idxs tracing.

        seed_xyz:       (B,N,3)
        seed_features:  (B,C,N)
        graspable_mask: (B,N)
        img_idxs:       (B,N), optional

        return:
            seed_xyz_graspable:      (B,M,3)
            seed_features_graspable: (B,C,M)
            mean_graspable_num:      scalar
            seed_img_idxs_graspable: (B,M) or None
        """
        B, _, _ = seed_xyz.shape
        seed_features_flipped = seed_features.transpose(1, 2).contiguous()  # (B,N,C)

        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_img_idxs_graspable = [] if img_idxs is not None else None
        graspable_num_batch = 0.

        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()

            cur_feat = seed_features_flipped[i][cur_mask]   # (Ng,C)
            cur_seed_xyz = seed_xyz[i][cur_mask]            # (Ng,3)
            cur_img_idxs = img_idxs[i][cur_mask] if img_idxs is not None else None  # (Ng,)

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)        # (1,Ng,3)
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)  # (1,M)
            fps_idxs_1d = fps_idxs.squeeze(0).long()

            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()   # (1,3,Ng)
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()  # (M,3)

            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # (1,C,Ng)
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()  # (C,M)

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)

            if cur_img_idxs is not None:
                seed_img_idxs_graspable.append(cur_img_idxs.index_select(0, fps_idxs_1d).contiguous())

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)             # (B,M,3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0)   # (B,C,M)
        mean_graspable_num = graspable_num_batch / B

        if seed_img_idxs_graspable is not None:
            seed_img_idxs_graspable = torch.stack(seed_img_idxs_graspable, 0)  # (B,M)
        return seed_xyz_graspable, seed_features_graspable, mean_graspable_num, seed_img_idxs_graspable

    # def _sample_graspable_points(self, seed_xyz, seed_features, graspable_mask, graspness_score):
    #     """
    #     seed_xyz:      (B,N,3)
    #     seed_features: (B,C,N)
    #     graspable_mask:(B,N)
    #     graspness_score:(B,N)
    #     return:
    #         seed_xyz_graspable: (B,M,3)
    #         seed_features_graspable: (B,C,M)
    #         mean_graspable_num: scalar
    #     """
    #     B, N, _ = seed_xyz.shape
    #     device = seed_xyz.device
    #     seed_features_flipped = seed_features.transpose(1, 2).contiguous()  # (B,N,C)

    #     seed_xyz_graspable = []
    #     seed_features_graspable = []
    #     graspable_num_batch = 0.

    #     for i in range(B):
    #         cur_idx = torch.nonzero(graspable_mask[i], as_tuple=False).squeeze(1)
    #         graspable_num_batch += cur_idx.numel()

    #         # fallback: 没有 graspable 点时，用 graspness topk
    #         if cur_idx.numel() == 0:
    #             k = min(self.M_points, N)
    #             cur_idx = torch.topk(graspness_score[i], k=k, largest=True).indices

    #         cur_seed_xyz = seed_xyz[i].index_select(0, cur_idx).contiguous()       # (Ng,3)
    #         cur_feat_mc = seed_features_flipped[i].index_select(0, cur_idx).contiguous()  # (Ng,C)

    #         if cur_idx.numel() >= self.M_points:
    #             fps_idxs = furthest_point_sample(cur_seed_xyz.unsqueeze(0), self.M_points)
    #             cur_seed_xyz = gather_operation(
    #                 cur_seed_xyz.unsqueeze(0).transpose(1, 2).contiguous(),
    #                 fps_idxs
    #             ).transpose(1, 2).squeeze(0).contiguous()
    #             cur_feat = gather_operation(
    #                 cur_feat_mc.unsqueeze(0).transpose(1, 2).contiguous(),
    #                 fps_idxs
    #             ).squeeze(0).contiguous()   # (C,M)
    #         else:
    #             rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=device)
    #             ridx = cur_idx[rep]
    #             cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()
    #             cur_feat = seed_features_flipped[i].index_select(0, ridx).transpose(0, 1).contiguous()

    #         seed_xyz_graspable.append(cur_seed_xyz)
    #         seed_features_graspable.append(cur_feat)

    #     seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)            # (B,M,3)
    #     seed_features_graspable = torch.stack(seed_features_graspable, 0)  # (B,C,M)
    #     mean_graspable_num = graspable_num_batch / B

    #     return seed_xyz_graspable, seed_features_graspable, mean_graspable_num

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']   # (B,N,3)
        B, point_num, _ = seed_xyz.shape
        device = seed_xyz.device

        img = end_points['img']                 # (B,3,H,W)
        img_idxs = end_points['img_idxs'].long()

        # -------------------------------------------------
        # 1) build sparse input features according to fuse_type
        # -------------------------------------------------
        if self.fuse_type == 'early':
            img_feat = self.img_backbone(img)   # (B,C,Hf,Wf)
            Bf, C, Hf, Wf = img_feat.shape
            img_feat = img_feat.view(Bf, C, -1)

            gather_idx = img_idxs.clamp(0, Hf * Wf - 1).unsqueeze(1).expand(-1, C, -1)
            point_img_feat = torch.gather(img_feat, 2, gather_idx).transpose(1, 2).contiguous()  # (B,N,C)

            sparse_feats_list = [point_img_feat[i] for i in range(B)]
            coords_list = end_points['coordinates_for_voxel']

            coordinates_batch, features_batch = ME.utils.sparse_collate(
                coords=[coord for coord in coords_list],
                feats=sparse_feats_list,
                dtype=torch.float32
            )

            coordinates_batch, features_batch, _, end_points['quantize2original'] = ME.utils.sparse_quantize(
                coordinates_batch,
                features_batch,
                return_index=True,
                return_inverse=True,
                device=device
            )

            mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)
            seed_features = self.backbone(mink_input).F

        elif self.fuse_type == 'intermediate':
            H0, W0 = img.shape[-2], img.shape[-1]

            # PSPNet must support return_pyramid=True
            pyr = self.img_backbone(img, return_pyramid=True)

            pfeat = {
                k: self._gather_2d_to_points(pyr[k], img_idxs, base_hw=(H0, W0))
                for k in ['p1', 'p2', 'p4', 'p8', 'p16']
            }

            coords_list = end_points['coordinates_for_voxel']
            input_feats = [
                torch.ones_like(seed_xyz[i], dtype=torch.float32)
                for i in range(B)
            ]  # each: (N,3)

            coordinates_batch, features_batch = ME.utils.sparse_collate(
                coords=[coord for coord in coords_list],
                feats=[feat for feat in input_feats],
                dtype=torch.float32
            )

            coordinates_batch, features_batch, _, end_points['quantize2original'] = ME.utils.sparse_quantize(
                coordinates_batch,
                features_batch,
                return_index=True,
                return_inverse=True,
                device=device
            )

            mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)
            seed_features = self.backbone(mink_input, pfeat, coords_list).F

        # restore to dense point order: (B,C,N)
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2).contiguous()

        # -------------------------------------------------
        # 2) original EcoGrasp heads
        # -------------------------------------------------
        end_points = self.graspable(seed_features, end_points)

        objectness_score = end_points['objectness_score']              # (B,2,N)
        graspness_score = end_points['graspness_score'].squeeze(1)    # (B,N)
        objectness_prob = torch.softmax(objectness_score, dim=1)[:, 1]  # (B,N)
        
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > cfgs.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        seed_xyz_graspable, seed_features_graspable, graspable_num, seed_img_idxs_graspable = self._sample_graspable_points(
            seed_xyz=seed_xyz,
            seed_features=seed_features,
            graspable_mask=graspable_mask,
            img_idxs=img_idxs if 'img_idxs' in end_points else None
        )

        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['D: Graspable Points'] = graspable_num
        if seed_img_idxs_graspable is not None:
            end_points['img_idxs_graspable'] = seed_img_idxs_graspable

        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        self._maybe_save_vis(
            end_points,
            xyz_full=seed_xyz,
            img=img if 'img' in end_points else None,
            img_idxs=img_idxs if 'img_idxs' in end_points else None,
            objectness_prob=objectness_prob,
            graspness_score=graspness_score,
            graspable_mask=graspable_mask,
            xyz_graspable=seed_xyz_graspable,
            img_idxs_graspable=seed_img_idxs_graspable,
            top_view_xyz=end_points.get('grasp_top_view_xyz', None),
            stats={
                "fuse_type": getattr(self, "fuse_type", "unknown"),
                "mean_objectness_prob": float(objectness_prob.mean().item()),
                "mean_graspness": float(graspness_score.mean().item()),
                "mean_graspable_points": float(graspable_num.item() if torch.is_tensor(graspable_num) else graspable_num),
                "num_total_points": int(seed_xyz.shape[1]),
                "num_sampled_points": int(self.M_points),
            }
        )
        self._vis_iter += 1

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