import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME

from models.pspnet import PSPNet
from utils.arguments import cfgs

# your original EconomicGrasp (point-center) implementation
# (keep your existing file/class; here we just import it)
from models.economicgrasp import economicgrasp as EconomicGrasp3D


# -------------------------
# utils: gather depth @ sampled pixels and backproject to 3D
# -------------------------
@torch.no_grad()
def depth_to_cloud_by_img_idxs(depth_map_448: torch.Tensor,
                               K_448: torch.Tensor,
                               img_idxs: torch.Tensor) -> torch.Tensor:
    """
    depth_map_448: (B,1,448,448) float32 meters
    K_448:         (B,3,3) float32
    img_idxs:      (B,N) flatten indices in 448*448 (from dataloader resized_idxs)
    return:
      cloud: (B,N,3) in camera frame
    """
    assert depth_map_448.ndim == 4 and depth_map_448.shape[2:] == (448, 448)
    B, _, H, W = depth_map_448.shape
    assert K_448.shape == (B, 3, 3)
    assert img_idxs.shape[0] == B

    idx = img_idxs.long().clamp(0, H * W - 1)           # (B,N)
    depth_flat = depth_map_448[:, 0].reshape(B, -1)     # (B,H*W)
    z = depth_flat.gather(1, idx).float()               # (B,N)

    u = (idx % W).float()                               # (B,N)
    v = (idx // W).float()                              # (B,N)

    fx = K_448[:, 0, 0].unsqueeze(1)
    fy = K_448[:, 1, 1].unsqueeze(1)
    cx = K_448[:, 0, 2].unsqueeze(1)
    cy = K_448[:, 1, 2].unsqueeze(1)

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    cloud = torch.stack([x, y, z], dim=-1)              # (B,N,3)
    return cloud


def _make_coords_for_me(cloud: torch.Tensor, voxel_size: float):
    """
    cloud: (B,N,3) float tensor (cuda or cpu)
    return:
      coords_list: list of np.ndarray, each (N,3) float32 in "voxel coordinates"
                  (keep consistent with your dataset: cloud/voxel_size)
    """
    B = cloud.shape[0]
    coords_list = []
    for b in range(B):
        coor = (cloud[b].detach().cpu().numpy().astype(np.float32) / float(voxel_size))
        coords_list.append(coor)
    return coords_list


# -------------------------
# RGB -> depth distribution (stride=2 => 224x224 tokens)
# -------------------------
class RGBDepthDistributionNet(nn.Module):
    """
    RGB-only depth distribution head:
      img (B,3,448,448)
        -> PSPNet feat (B,C,448,448)
        -> avgpool(stride) -> (B,C,448/stride,448/stride)
        -> 1x1 conv -> logits (B,D,Hr,Wr)
        -> softmax -> depth_prob_map (B,D,Hr,Wr)
        -> expected depth -> depth_map_448 (B,1,448,448)

    Returns:
      depth_prob_pred: (B,1,N,D)  where N=Hr*Wr, D=bin_num
      spatial_shape:   (Hr,Wr)
      depth_map_pred:  (B,1,448,448)
    """

    def __init__(self,
                 img_feat_dim=64,
                 stride=2,
                 min_depth=0.2,
                 max_depth=2.0,
                 bin_num=256,
                 psp_backend="resnet34"):
        super().__init__()
        assert stride in [1, 2, 4, 8], "keep it simple for now"
        self.stride = int(stride)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)

        # PSPNet output keeps 448x448 resolution in your implementation
        self.img_backbone = PSPNet(
            sizes=(1, 2, 3, 6),
            psp_size=512,
            deep_features_size=img_feat_dim,
            backend=psp_backend
        )

        self.depth_logits = nn.Conv2d(img_feat_dim, self.bin_num, kernel_size=1, bias=True)

        # register depth anchors as buffer
        anchors = torch.linspace(self.min_depth, self.max_depth, self.bin_num).view(1, self.bin_num, 1, 1)
        self.register_buffer("depth_anchors", anchors, persistent=False)

    def forward(self, img: torch.Tensor):
        """
        img: (B,3,448,448)
        """
        B, _, H, W = img.shape
        assert (H, W) == (448, 448), "expect resized input 448x448"

        feat = self.img_backbone(img)  # (B,C,448,448)
        if self.stride > 1:
            feat_s = F.avg_pool2d(feat, kernel_size=self.stride, stride=self.stride)  # (B,C,Hr,Wr)
        else:
            feat_s = feat

        Hr, Wr = feat_s.shape[2], feat_s.shape[3]

        logits = self.depth_logits(feat_s)              # (B,D,Hr,Wr)
        prob = logits.softmax(dim=1)                    # (B,D,Hr,Wr)

        # expected depth at token resolution
        anchors = self.depth_anchors.to(prob.dtype)     # (1,D,1,1)
        depth_tok = (prob * anchors).sum(dim=1, keepdim=True)  # (B,1,Hr,Wr)

        # upsample to 448x448 for backprojection
        if (Hr, Wr) != (448, 448):
            depth_448 = F.interpolate(depth_tok, size=(448, 448), mode="bilinear", align_corners=True)
        else:
            depth_448 = depth_tok

        # token layout: (B,1,N,D)
        depth_prob_pred = prob.permute(0, 2, 3, 1).contiguous().view(B, 1, Hr * Wr, self.bin_num)

        return depth_prob_pred, (Hr, Wr), depth_448


# -------------------------
# Wrapper: RGB -> depth -> backproject @ sampled pixels -> EcoGrasp3D
# -------------------------
class EconomicGrasp_RGBDepthProb(nn.Module):
    """
    RGB-only baseline (risk-reduced alignment):
      - dataloader provides img_idxs (B,20000): sampled pixels in 448x448
      - model predicts depth_map_pred_448
      - gather depth at img_idxs and backproject with K -> point_clouds (B,20000,3)
      - run original point-center EcoGrasp3D + its original supervision (process_grasp_labels)

    Required keys in end_points:
      - img:      (B,3,448,448)
      - K:        (B,3,3) resized intrinsics for 448x448
      - img_idxs: (B,20000) flatten idx in 448*448 (from dataloader)

    Plus all original EcoGrasp label keys for training:
      object_poses_list, grasp_points_list, grasp_rotations_list, grasp_depth_list,
      grasp_scores_list, grasp_widths_list, view_graspness_list, top_view_index_list, ...
      + point-level labels (objectness_label, graspness_label) if you keep those losses.
    """

    def __init__(self,
                 cylinder_radius=0.05,
                 seed_feat_dim=512,
                 voxel_size=0.005,
                 num_points=20000,
                 is_training=True,
                 img_feat_dim=64,
                 depth_stride=2,     # <-- your expectation: 224x224 tokens
                 min_depth=0.2,
                 max_depth=2.0,
                 bin_num=256):
        super().__init__()
        self.is_training = is_training
        self.voxel_size = float(voxel_size)
        self.num_points = int(num_points)

        self.depth_net = RGBDepthDistributionNet(
            img_feat_dim=img_feat_dim,
            stride=depth_stride,
            min_depth=min_depth,
            max_depth=max_depth,
            bin_num=bin_num,
            psp_backend="resnet34"
        )

        self.grasp_net = EconomicGrasp3D(
            cylinder_radius=cylinder_radius,
            seed_feat_dim=seed_feat_dim,
            is_training=is_training,
            voxel_size=voxel_size
        )

    def forward(self, end_points: dict):
        img = end_points["img"]          # (B,3,448,448)
        K = end_points["K"]              # (B,3,3)
        img_idxs = end_points["img_idxs"]  # (B,20000) flatten idx in 448*448

        # ---- 1) RGB -> depth distribution + depth map (448) ----
        depth_prob_pred, spatial_shape, depth_map_pred = self.depth_net(img)
        end_points["depth_prob_pred"] = depth_prob_pred          # (B,1,Ntok,D)
        end_points["depth_spatial_shape"] = spatial_shape        # (Hr,Wr)
        end_points["depth_map_pred"] = depth_map_pred            # (B,1,448,448)

        # ---- 2) backproject ONLY at sampled pixels (aligned to labels) ----
        # cloud: (B,20000,3)
        cloud = depth_to_cloud_by_img_idxs(depth_map_pred, K, img_idxs)
        end_points["point_clouds"] = cloud

        # ---- 3) ME voxel coords ----
        end_points["coordinates_for_voxel"] = _make_coords_for_me(cloud, self.voxel_size)

        # ---- 4) run original EcoGrasp (point-center head + original supervision) ----
        end_points = self.grasp_net(end_points)
        return end_points
