import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME

from models.pspnet import PSPNet
from utils.arguments import cfgs

# your original EconomicGrasp (point-center) implementation
# (keep your existing file/class; here we just import it)
from utils.label_generation import process_grasp_labels, batch_viewpoint_params_to_matrix
from models.economicgrasp import economicgrasp as EconomicGrasp3D
import open3d as o3d
import os


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
    
    
# from models.dinov2_dpt import DA2
# class DINOv2DepthDistributionNet(nn.Module):
#     """
#     使用 DA2 的 DPTHead 直接输出 bin logits（out_dim=bin_num），不额外加 conv。

#     forward(img, return_prob=False):
#       return_prob=False:
#         depth_448: (B,1,448,448)  E[z] in meters
#         depth_tok: (B,1,448/stride,448/stride) debug
#         img_feat:  (B,Cf,Hf,Wf)  path_1 feature (通常 patch_h*14, patch_w*14 -> 448)
#       return_prob=True:
#         + depth_prob_448:   (B,bin_num,448,448)
#         + depth_logits_448: (B,bin_num,448,448)
#     """
#     def __init__(self,
#                  encoder="vitb",
#                  stride=2,
#                  min_depth=0.2,
#                  max_depth=2.0,
#                  bin_num=256,
#                  freeze_backbone=True,
#                  eps=1e-6):
#         super().__init__()
#         assert stride in [1, 2, 4]
#         self.stride = int(stride)
#         self.min_depth = float(min_depth)
#         self.max_depth = float(max_depth)
#         self.bin_num = int(bin_num)
#         self.freeze_backbone_flag = bool(freeze_backbone)
#         self.eps = float(eps)

#         model_configs = {
#             "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
#             "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
#             "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
#             "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
#         }
#         assert encoder in model_configs

#         # 关键：out_dim=bin_num
#         self.depthnet = DA2(**{
#             **model_configs[encoder],
#             "out_dim": self.bin_num
#         })

#         # 只加载 backbone（与你现有一致）
#         ckpt = torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu")
#         self.depthnet.load_state_dict({k: v for k, v in ckpt.items() if "pretrained" in k}, strict=False)

#         if self.freeze_backbone_flag:
#             self._freeze_backbone()

#         # bin centers
#         bins = torch.linspace(self.min_depth, self.max_depth, self.bin_num)
#         self.register_buffer("depth_bins", bins, persistent=False)

#     def _freeze_backbone(self):
#         for p in self.depthnet.pretrained.parameters():
#             p.requires_grad_(False)

#     def train(self, mode=True):
#         super().train(mode)
#         if mode and self.freeze_backbone_flag:
#             self._freeze_backbone()
#         return self

#     def forward(
#         self,
#         img: torch.Tensor,
#         return_prob: bool = False,
#         return_tok_prob: bool = False,
#     ):
#         B, _, H, W = img.shape
#         assert (H, W) == (448, 448)

#         patch_h, patch_w = H // 14, W // 14
#         feats = self.depthnet.pretrained.get_intermediate_layers(
#             img,
#             self.depthnet.intermediate_layer_idx[self.depthnet.encoder],
#             return_class_token=True
#         )

#         # (B,Cf,?,?) and (B,D,448,448)
#         img_feat, depth_logits_448 = self.depthnet.depth_head(feats, patch_h, patch_w)

#         # logits -> prob on 448
#         depth_prob_448 = torch.softmax(depth_logits_448, dim=1)  # (B,D,448,448)
#         depth_prob_448 = torch.nan_to_num(depth_prob_448, nan=0.0, posinf=0.0, neginf=0.0)

#         # E[z] on 448
#         z = self.depth_bins.view(1, self.bin_num, 1, 1).to(depth_prob_448)
#         depth_448 = (depth_prob_448 * z).sum(dim=1, keepdim=True).clamp(self.min_depth, self.max_depth)

#         # token debug depth (E[z])
#         if self.stride > 1:
#             depth_tok = F.interpolate(depth_448, size=(H // self.stride, W // self.stride), mode="nearest")
#         else:
#             depth_tok = depth_448

#         # align img_feat to 448 for later gather
#         if img_feat.shape[-2:] != (H, W):
#             img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

#         if not return_prob:
#             return depth_448, depth_tok, img_feat

#         # ---------- optional: token/patch prob for BIP3D-style loss ----------
#         if return_tok_prob:
#             # IMPORTANT: must match your GT builder's patch size.
#             # If your build_depth_prob_gt uses s=2, keep stride=2 here.
#             s = self.stride  # typically 2 -> 224x224
#             prob_tok = F.avg_pool2d(depth_prob_448, kernel_size=s, stride=s)  # (B,D,Ht,Wt)
#             prob_tok = torch.nan_to_num(prob_tok, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(self.eps)
#             prob_tok = prob_tok / prob_tok.sum(dim=1, keepdim=True).clamp_min(self.eps)  # renorm sum_D=1

#             # (B,1,Nfeat,D)
#             prob_tok_flat = prob_tok.permute(0, 2, 3, 1).reshape(B, -1, self.bin_num).unsqueeze(1).contiguous()

#             return depth_448, depth_tok, img_feat, depth_prob_448, depth_logits_448, prob_tok_flat

#         return depth_448, depth_tok, img_feat, depth_prob_448, depth_logits_448
    

from models.dinov2_dpt import DA2
class DINOv2DepthDistributionNet(nn.Module):
    """
    Depth distribution head based on Depth-Anything-V2 DINOv2 backbone.

    Added behavior versus the previous version:
      - can optionally return raw DINO feats so another DPTHead can reuse them.
    """
    def __init__(
        self,
        encoder: str = "vitb",
        stride: int = 2,
        min_depth: float = 0.2,
        max_depth: float = 2.0,
        bin_num: int = 256,
        freeze_backbone: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert stride in [1, 2, 4]
        self.stride = int(stride)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)
        self.freeze_backbone_flag = bool(freeze_backbone)
        self.eps = float(eps)

        model_configs = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        assert encoder in model_configs

        self.depthnet = DA2(**{**model_configs[encoder], "out_dim": self.bin_num})

        ckpt = torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu")
        self.depthnet.load_state_dict({k: v for k, v in ckpt.items() if "pretrained" in k}, strict=False)

        if self.freeze_backbone_flag:
            self._freeze_backbone()

        bins = torch.linspace(self.min_depth, self.max_depth, self.bin_num)
        self.register_buffer("depth_bins", bins, persistent=False)

    def _freeze_backbone(self):
        for p in self.depthnet.pretrained.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone_flag:
            self._freeze_backbone()
        return self

    def forward(
        self,
        img: torch.Tensor,
        return_prob: bool = False,
        return_tok_prob: bool = False,
        return_feats: bool = False,
    ):
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)

        patch_h, patch_w = H // 14, W // 14
        feats = self.depthnet.pretrained.get_intermediate_layers(
            img,
            self.depthnet.intermediate_layer_idx[self.depthnet.encoder],
            return_class_token=True,
        )

        img_feat, depth_logits_448 = self.depthnet.depth_head(feats, patch_h, patch_w)

        depth_prob_448 = torch.softmax(depth_logits_448, dim=1)
        depth_prob_448 = torch.nan_to_num(depth_prob_448, nan=0.0, posinf=0.0, neginf=0.0)

        z = self.depth_bins.view(1, self.bin_num, 1, 1).to(depth_prob_448)
        depth_448 = (depth_prob_448 * z).sum(dim=1, keepdim=True).clamp(self.min_depth, self.max_depth)

        if self.stride > 1:
            depth_tok = F.interpolate(depth_448, size=(H // self.stride, W // self.stride), mode="nearest")
        else:
            depth_tok = depth_448

        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

        if not return_prob:
            if return_feats:
                return depth_448, depth_tok, img_feat, feats
            return depth_448, depth_tok, img_feat

        outputs = [depth_448, depth_tok, img_feat, depth_prob_448, depth_logits_448]

        if return_tok_prob:
            s = self.stride
            prob_tok = F.avg_pool2d(depth_prob_448, kernel_size=s, stride=s)
            prob_tok = torch.nan_to_num(prob_tok, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(self.eps)
            prob_tok = prob_tok / prob_tok.sum(dim=1, keepdim=True).clamp_min(self.eps)
            prob_tok_flat = prob_tok.permute(0, 2, 3, 1).reshape(B, -1, self.bin_num).unsqueeze(1).contiguous()
            outputs.append(prob_tok_flat)

        if return_feats:
            outputs.append(feats)

        return tuple(outputs)


class DINOv2DepthRegressionNet(nn.Module):
    """
    RGB -> depth_reg_448 (B,1,448,448)
    可选：输出 depth_reg_tok (B,1,224,224) 便于 debug
    """
    def __init__(self,
                 encoder="vitb",
                 stride=2,              # 你希望 token=224x224 用来监控的话保留
                 min_depth=0.2,
                 max_depth=1.0,
                 freeze_backbone=True):
        super().__init__()
        assert stride in [1, 2, 4]
        self.stride = int(stride)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.freeze_backbone_flag = bool(freeze_backbone)

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.depthnet = DA2(**{**model_configs[encoder], 'max_depth': self.max_depth}, out_dim=1)

        self.depthnet.load_state_dict({k: v for k, v in torch.load('checkpoints/depth_anything_v2_{}.pth'.format(encoder), map_location='cpu').items() if 'pretrained' in k}, strict=False)

        if self.freeze_backbone_flag:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.depthnet.pretrained.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone_flag:
            self._freeze_backbone()
        return self

    def forward(self, img, return_feats: bool = False, return_raw: bool = False):
        """
        return:
        depth_448: bounded direct depth, used when use_obs_depth=False
        depth_tok: bounded direct depth token map
        img_feat:  dense feature from DA2 depth head
        raw_448:   raw 1-channel output, used as residual when use_obs_depth=True
        feats:     raw DINO intermediate features for proposal_head
        """
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)

        patch_h, patch_w = H // 14, W // 14
        feats = self.depthnet.pretrained.get_intermediate_layers(
            img,
            self.depthnet.intermediate_layer_idx[self.depthnet.encoder],
            return_class_token=True
        )

        img_feat, raw_448 = self.depthnet.depth_head(feats, patch_h, patch_w)  # (B,1,448,448)

        # RGB mode uses this as direct metric depth.
        depth_448 = F.sigmoid(raw_448) * self.max_depth
        # depth_448 = self.min_depth + F.sigmoid(raw_448) * (self.max_depth - self.min_depth)
        # depth_448 = depth_448.clamp(self.min_depth, self.max_depth)

        if self.stride > 1:
            depth_tok = F.interpolate(depth_448, size=(H // self.stride, W // self.stride), mode='nearest')
        else:
            depth_tok = depth_448

        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(
                img_feat,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        if return_feats and return_raw:
            return depth_448, depth_tok, img_feat, raw_448, feats
        if return_feats:
            return depth_448, depth_tok, img_feat, feats
        if return_raw:
            return depth_448, depth_tok, img_feat, raw_448

        return depth_448, depth_tok, img_feat
    
from models.grasp_spatial_enhancer import MultiBasicEncoder
class ObsDepthAdapter(nn.Module):
    """
    Observed depth encoder using MultiBasicEncoder.

    Input:
        obs_depth: (B,1,H,W), meter
    Output:
        obs_feat:  (B,C,Hf,Wf)
        obs_valid: (B,1,H,W), valid mask at original obs-depth resolution
    """
    def __init__(
        self,
        out_dim: int = 128,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        norm_fn: str = "group",
        downsample: int = 2,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        self.encoder = MultiBasicEncoder(
            input_dim=2,
            output_dim=[(out_dim, out_dim, out_dim)],
            norm_fn=norm_fn,
            dropout=0.0,
            downsample=downsample,
        )

    def forward(self, obs_depth: torch.Tensor, out_hw):
        if obs_depth.dim() == 3:
            obs_depth = obs_depth.unsqueeze(1)
        assert obs_depth.dim() == 4 and obs_depth.size(1) == 1

        obs_mask = (
            torch.isfinite(obs_depth)
            & (obs_depth > self.min_depth)
        ).float()

        z = torch.nan_to_num(obs_depth, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat([z, obs_mask], dim=1)  # (B,2,H,W)
        
        outputs04, outputs08, outputs16 = self.encoder(x, num_layers=3)

        # 最大空间分辨率 feature map
        obs_feat = outputs04[0]

        if obs_feat.shape[-2:] != tuple(out_hw):
            obs_feat = F.interpolate(
                obs_feat,
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )

        return obs_feat, obs_mask
    

def _make_gn(num_channels: int, max_groups: int = 8):
    g = min(max_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class DepthRefine(nn.Module):
    """
    Minimal depth fusion.

    final_depth = C * net_depth + (1 - C) * obs_depth

    C is the confidence of network-predicted depth.
    """
    def __init__(
        self,
        rgb_feat_dim: int,
        obs_feat_dim: int = None,
        hidden_dim: int = None,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        norm_fn: str = "group",
        downsample: int = 2,
    ):
        super().__init__()
        self.rgb_feat_dim = int(rgb_feat_dim)
        self.hidden_dim = int(hidden_dim or rgb_feat_dim)
        self.obs_feat_dim = int(obs_feat_dim or self.hidden_dim)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        self.obs_encoder = ObsDepthAdapter(
            out_dim=self.obs_feat_dim,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            norm_fn=norm_fn,
            downsample=downsample,
        )

        self.rgb_proj = (
            nn.Identity()
            if self.rgb_feat_dim == self.hidden_dim
            else nn.Conv2d(self.rgb_feat_dim, self.hidden_dim, kernel_size=1)
        )

        self.obs_proj = (
            nn.Identity()
            if self.obs_feat_dim == self.hidden_dim
            else nn.Conv2d(self.obs_feat_dim, self.hidden_dim, kernel_size=1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_gn(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_gn(self.hidden_dim),
            nn.GELU(),
        )

        self.conf_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        rgb_feat: torch.Tensor,   # (B,C,Hf,Wf), e.g. depth_img_feat
        net_depth: torch.Tensor,  # (B,1,H,W), network predicted absolute depth
        obs_depth: torch.Tensor,  # (B,1,H,W), observed depth
    ):  

        obs_depth = obs_depth.to(device=net_depth.device, dtype=net_depth.dtype)

        if net_depth.dim() == 3:
            net_depth = net_depth.unsqueeze(1)
        net_depth = net_depth[:, :1]

        B, _, H, W = net_depth.shape

        if obs_depth.shape[-2:] != (H, W):
            obs_depth = F.interpolate(obs_depth, size=(H, W), mode="nearest")

        obs_feat, obs_valid = self.obs_encoder(obs_depth, out_hw=rgb_feat.shape[-2:])

        rgb_feat = self.rgb_proj(rgb_feat)
        obs_feat = self.obs_proj(obs_feat)

        fused_feat = self.fuse(torch.cat([rgb_feat, obs_feat], dim=1))

        conf_feat = torch.sigmoid(self.conf_head(fused_feat))

        if conf_feat.shape[-2:] != (H, W):
            conf = F.interpolate(
                conf_feat,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        else:
            conf = conf_feat

        obs_valid = obs_valid.to(device=net_depth.device, dtype=net_depth.dtype)
        if obs_valid.shape[-2:] != (H, W):
            obs_valid = F.interpolate(obs_valid, size=(H, W), mode="nearest")

        obs_depth_clean = torch.nan_to_num(obs_depth, nan=0.0, posinf=0.0, neginf=0.0)
        final_depth = conf * net_depth + (1.0 - conf) * obs_depth_clean

        aux = {
            "depth_confidence": conf,
            "obs_depth_valid": obs_valid,
            "obs_depth_feat_norm": obs_feat.detach().norm(dim=1, keepdim=True),
            "rgbd_fused_depth_feat": fused_feat,
        }
        return final_depth, aux
    
# -------------------------
# Wrapper: RGB -> depth -> backproject @ sampled pixels -> EcoGrasp3D
# -------------------------
class economicgrasp_depth_baseline(nn.Module):
    """
    RGB-only baseline:
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
                 depth_stride=2,     # <-- your expectation: 224x224 tokens
                 min_depth=0.2,
                 max_depth=2.0,
                 use_obs_depth=False,
                 vis_dir=None,
                 vis_every=100):
        super().__init__()
        self.is_training = is_training
        self.voxel_size = float(voxel_size)
        self.num_points = int(num_points)
        self.use_obs_depth = bool(use_obs_depth)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.encoder = 'vitb'
        # self.depth_net = RGBDepthDistributionNet(
        #     img_feat_dim=img_feat_dim,
        #     stride=depth_stride,
        #     min_depth=min_depth,
        #     max_depth=max_depth,
        #     bin_num=bin_num,
        #     psp_backend="resnet34"
        # )
        # self.depth_net = DINOv2DepthDistributionNet(
        #     encoder="vitb",
        #     stride=depth_stride,
        #     min_depth=min_depth,
        #     max_depth=max_depth,
        #     bin_num=bin_num,
        #     freeze_backbone=True,   # ✅ 训练时冻结 ViT backbone
        # )
        self.depth_net = DINOv2DepthRegressionNet(
            encoder=self.encoder,
            stride=depth_stride,     # 2 -> 224x224 token debug
            min_depth=min_depth,
            max_depth=max_depth,
            freeze_backbone=True
        )

        if self.use_obs_depth:
            depth_feat_dim_map = {
                "vits": 64,
                "vitb": 128,
                "vitl": 256,
                "vitg": 384,
            }
            self.depth_feat_dim = depth_feat_dim_map[self.encoder]
            self.depth_refine = DepthRefine(
                rgb_feat_dim=self.depth_feat_dim,
                hidden_dim=self.depth_feat_dim,
                min_depth=min_depth,
                max_depth=max_depth,
                downsample=depth_stride,
            )
        else:
            self.depth_refine = None

        self.grasp_net = EconomicGrasp3D(
            cylinder_radius=cylinder_radius,
            seed_feat_dim=seed_feat_dim,
            is_training=is_training,
            voxel_size=voxel_size
        )

        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
        self.debug_every = 50
        self._dbg_iter = 0

    @torch.no_grad()
    def _save_map_png(
        self,
        x,
        out_path,
        vmin=None,
        vmax=None,
        cmap="Spectral",
        title=None,
    ):
        if self.vis_dir is None:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if x is None:
            return

        if torch.is_tensor(x):
            arr = x.detach().float().cpu().numpy()
        else:
            arr = np.asarray(x)

        arr = np.squeeze(arr)

        # If accidentally given BxCxHxW, take first map.
        while arr.ndim > 2:
            arr = arr[0]

        if arr.ndim != 2:
            return

        finite = np.isfinite(arr)

        if not finite.any():
            return

        if vmin is None:
            vmin = float(np.percentile(arr[finite], 1))
        if vmax is None:
            vmax = float(np.percentile(arr[finite], 99))

        if vmax <= vmin + 1e-6:
            vmax = vmin + 1e-6

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        plt.figure(figsize=(6, 6))
        im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()


    def _read_first_int(self, end_points, key, default=-1):
        if key not in end_points:
            return default
        x = end_points[key]
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return int(x.reshape(-1)[0].detach().cpu().item())
        try:
            return int(x)
        except Exception:
            return default

    def _force_vis_flag(self, end_points):
        force_vis = end_points.get("force_vis", False)
        if torch.is_tensor(force_vis):
            if force_vis.numel() == 0:
                return False
            return bool(force_vis.reshape(-1)[0].detach().cpu().item())
        return bool(force_vis)

    @torch.no_grad()
    def _save_depth_maps_vis(
        self,
        end_points,
        img,
        img_idxs,
        depth_net_pred,
        depth_map_pred,
        depth_tok=None,
        raw_depth=None,
        obs_depth=None,
        gt_depth=None,
        depth_confidence=None,
        depth_correction=None,
    ):
        if self.vis_dir is None:
            return

        scene = self._read_first_int(end_points, "scene_idx", -1)
        anno = self._read_first_int(end_points, "anno_idx", -1)

        # fallback for older dataset keys
        if scene < 0:
            scene = self._read_first_int(end_points, "scene", -1)
        if anno < 0:
            anno = self._read_first_int(end_points, "frame", -1)

        prefix = os.path.join(
            self.vis_dir,
            f"scene{scene:04d}_anno{anno:04d}_it{self._vis_iter:06d}",
        )

        B, _, H, W = img.shape

        def _b1hw(x, name):
            if x is None:
                return None
            if x.dim() == 3:
                x = x.unsqueeze(1)
            elif x.dim() == 4:
                x = x[:, :1]
            else:
                raise ValueError(f"Unexpected {name} shape: {tuple(x.shape)}")
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode="nearest")
            return x

        depth_net_pred = _b1hw(depth_net_pred, "depth_net_pred")
        depth_map_pred = _b1hw(depth_map_pred, "depth_map_pred")
        obs_depth = _b1hw(obs_depth, "obs_depth") if obs_depth is not None else None
        gt_depth = _b1hw(gt_depth, "gt_depth") if gt_depth is not None else None
        raw_depth = _b1hw(raw_depth, "raw_depth") if raw_depth is not None else None
        depth_confidence = _b1hw(depth_confidence, "depth_confidence") if depth_confidence is not None else None
        depth_correction = _b1hw(depth_correction, "depth_correction") if depth_correction is not None else None

        # ------------------------------------------------------------
        # 1. Main depth maps
        # ------------------------------------------------------------
        self._save_map_png(
            depth_net_pred[0, 0],
            prefix + "_depth_net_pred.png",
            vmin=self.min_depth,
            vmax=self.max_depth,
            cmap="Spectral",
        )

        self._save_map_png(
            depth_map_pred[0, 0],
            prefix + "_depth_final.png",
            vmin=self.min_depth,
            vmax=self.max_depth,
            cmap="Spectral",
        )

        if depth_tok is not None:
            self._save_map_png(
                depth_tok[0, 0] if depth_tok.dim() == 4 else depth_tok[0],
                prefix + "_depth_tok.png",
                vmin=self.min_depth,
                vmax=self.max_depth,
                cmap="Spectral",
            )

        # raw_depth is not metric; use adaptive vmin/vmax.
        if raw_depth is not None:
            self._save_map_png(
                raw_depth[0, 0],
                prefix + "_depth_raw.png",
                vmin=None,
                vmax=None,
                cmap="coolwarm",
            )

        if obs_depth is not None:
            self._save_map_png(
                obs_depth[0, 0],
                prefix + "_obs_depth.png",
                vmin=self.min_depth,
                vmax=self.max_depth,
                cmap="Spectral",
            )

        if depth_correction is not None:
            corr0 = depth_correction[0, 0]
            finite = torch.isfinite(corr0)
            if bool(finite.any()):
                abs_p = torch.quantile(corr0[finite].abs(), 0.99).detach()
                vmax_corr = float(abs_p.cpu().item())
                vmax_corr = max(vmax_corr, 1e-3)
            else:
                vmax_corr = 0.05

            self._save_map_png(
                corr0,
                prefix + "_depth_correction.png",
                vmin=-vmax_corr,
                vmax=vmax_corr,
                cmap="coolwarm",
            )

        if depth_confidence is not None:
            self._save_map_png(
                depth_confidence[0, 0],
                prefix + "_depth_confidence.png",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )

        # ------------------------------------------------------------
        # 2. GT and error maps
        # ------------------------------------------------------------
        if gt_depth is not None:
            self._save_map_png(
                gt_depth[0, 0],
                prefix + "_gt_depth.png",
                vmin=self.min_depth,
                vmax=self.max_depth,
                cmap="Spectral",
            )

            valid_gt = (
                torch.isfinite(gt_depth)
                & (gt_depth >= self.min_depth)
                & (gt_depth <= self.max_depth)
            )

            err_final = (depth_map_pred - gt_depth).abs()
            err_final = torch.where(valid_gt, err_final, torch.full_like(err_final, float("nan")))

            err_net = (depth_net_pred - gt_depth).abs()
            err_net = torch.where(valid_gt, err_net, torch.full_like(err_net, float("nan")))

            self._save_map_png(
                err_final[0, 0],
                prefix + "_err_final_gt.png",
                vmin=0.0,
                vmax=0.10,
                cmap="magma",
            )

            self._save_map_png(
                err_net[0, 0],
                prefix + "_err_net_gt.png",
                vmin=0.0,
                vmax=0.10,
                cmap="magma",
            )

            if obs_depth is not None:
                err_obs = (obs_depth - gt_depth).abs()
                err_obs = torch.where(valid_gt, err_obs, torch.full_like(err_obs, float("nan")))

                self._save_map_png(
                    err_obs[0, 0],
                    prefix + "_err_obs_gt.png",
                    vmin=0.0,
                    vmax=0.10,
                    cmap="magma",
                )

        # ------------------------------------------------------------
        # 3. img_idxs sampling maps
        # ------------------------------------------------------------
        if img_idxs is not None:
            idx0 = img_idxs[0].detach().long().clamp(0, H * W - 1)

            sampled_mask = torch.zeros((H * W,), device=img.device, dtype=torch.float32)
            sampled_mask[idx0] = 1.0

            self._save_map_png(
                sampled_mask.view(H, W),
                prefix + "_sampled_mask.png",
                vmin=0.0,
                vmax=1.0,
                cmap="gray",
            )

            sparse_final = torch.full((H * W,), float("nan"), device=img.device, dtype=depth_map_pred.dtype)
            final_flat = depth_map_pred[0, 0].reshape(-1)
            sparse_final[idx0] = final_flat[idx0]

            self._save_map_png(
                sparse_final.view(H, W),
                prefix + "_sampled_final_depth.png",
                vmin=self.min_depth,
                vmax=self.max_depth,
                cmap="Spectral",
            )
        
    def _as_b1hw(self, x, hw, device, dtype, name="depth"):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x[:, :1]
        else:
            raise ValueError(f"Unexpected {name} shape: {tuple(x.shape)}")

        x = x.to(device=device, dtype=dtype)

        if x.shape[-2:] != hw:
            x = F.interpolate(x, size=hw, mode="nearest")

        return x.contiguous()
     
    @torch.no_grad()
    def _save_pred_gt_cloud_ply(
        self,
        cloud_pred: torch.Tensor,
        cloud_gt: torch.Tensor,
        end_points: dict,
        cloud_obs: torch.Tensor = None,
    ):
        """
        pred=red, gt=blue, obs=green(optional)
        """
        if o3d is None:
            return

        p = cloud_pred[0].detach().float().cpu().numpy()
        g = cloud_gt[0].detach().float().cpu().numpy()
        o = None if cloud_obs is None else cloud_obs[0].detach().float().cpu().numpy()

        def _valid(x):
            m = np.isfinite(x).all(axis=1)
            m &= (x[:, 2] > 0)
            return x[m]

        p = _valid(p)
        g = _valid(g)
        if o is not None:
            o = _valid(o)

        if p.shape[0] == 0 or g.shape[0] == 0:
            return

        pts_list = []
        col_list = []

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32)
        p_col[:, 0] = 1.0  # red
        pts_list.append(p)
        col_list.append(p_col)

        g_col = np.zeros((g.shape[0], 3), dtype=np.float32)
        g_col[:, 2] = 1.0  # blue
        pts_list.append(g)
        col_list.append(g_col)

        if o is not None and o.shape[0] > 0:
            o_col = np.zeros((o.shape[0], 3), dtype=np.float32)
            o_col[:, 1] = 1.0  # green
            pts_list.append(o)
            col_list.append(o_col)

        pts = np.concatenate(pts_list, axis=0)
        cols = np.concatenate(col_list, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        tag = end_points.get("vis_tag", None)
        if tag is None:
            scene = end_points.get("scene", "scene")
            frame = end_points.get("frame", "frame")
            tag = f"{scene}_{frame}"

        out_path = os.path.join(
            self.vis_dir,
            f"pred_gt_obs_cloud_{tag}_iter{self._vis_iter:06d}.ply"
        )
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

    def forward(self, end_points: dict):
        img = end_points["img"]          # (B,3,448,448)
        K = end_points["K"]              # (B,3,3)
        img_idxs = end_points["img_idxs"]  # (B,20000) flatten idx in 448*448

        # ------------------------------------------------------------
        # 1) RGB depth prediction
        # ------------------------------------------------------------
        depth_net_pred, depth_tok, img_feat, depth_net_raw = self.depth_net(img, return_raw=True)

        # Ensure shape: (B,1,448,448)
        depth_net_pred = self._as_b1hw(
            depth_net_pred,
            hw=img.shape[-2:],
            device=img.device,
            dtype=img.dtype,
            name="depth_net_pred",
        )

        depth_net_raw = self._as_b1hw(
            depth_net_raw,
            hw=img.shape[-2:],
            device=img.device,
            dtype=img.dtype,
            name="depth_net_raw",
        )
        
        obs_depth_448 = None
        depth_confidence = None
        depth_refined_correction = torch.zeros_like(depth_net_pred)

        # ------------------------------------------------------------
        # 2) Optional obs-depth branch
        # ------------------------------------------------------------
        if not self.use_obs_depth:
            depth_map_pred = depth_net_pred
        else:
            obs_depth_448 = end_points.get("sensor_depth_m", None)

            if obs_depth_448 is None:
                raise ValueError(
                    "use_obs_depth=True requires end_points['sensor_depth_m'] "
                )

            obs_depth_448 = self._as_b1hw(
                obs_depth_448,
                hw=img.shape[-2:],
                device=img.device,
                dtype=depth_net_pred.dtype,
                name="sensor_depth_m",
            )

            depth_map_pred, fusion_aux = self.depth_refine(
                rgb_feat=img_feat,
                net_depth=depth_net_raw,
                obs_depth=obs_depth_448,
            )

            depth_map_pred = self._as_b1hw(
                depth_map_pred,
                hw=img.shape[-2:],
                device=img.device,
                dtype=depth_net_pred.dtype,
                name="depth_map_pred_refined",
            )

            depth_confidence = fusion_aux.get("depth_confidence", None)
            depth_refined_correction = depth_map_pred - obs_depth_448

        # ------------------------------------------------------------
        # 3) Store depth outputs
        # depth_map_pred is final depth used for depth loss and backprojection.
        # depth_net_pred keeps raw RGB-only depth for debugging.
        # ------------------------------------------------------------
        end_points["depth_map_pred"] = depth_map_pred
        end_points["depth_net_pred"] = depth_net_pred
        # end_points["depth_tok_pred"] = depth_tok
        end_points["img_feat_dpt"] = img_feat
        end_points["depth_refined_correction"] = depth_refined_correction

        if self.use_obs_depth:
            end_points["obs_depth_m_used"] = obs_depth_448
            end_points["sensor_depth_m_used"] = obs_depth_448
            if depth_confidence is not None:
                end_points["depth_confidence_pred"] = depth_confidence

        # -------- depth classification (whole-image, patch-level loss) --------
        # depth_map_pred, depth_tok, img_feat, depth_prob_448, depth_logits_448 = self.depth_net(img, return_prob=True)

        # end_points["depth_map_pred"] = depth_map_pred      # (B,1,448,448)  E[z] for backprojection
        # end_points["depth_tok_pred"] = depth_tok           # (B,1,224,224)  debug
        # end_points["img_feat_dpt"]   = img_feat            # (B,C,448,448)

        # # ---- build token/patch distribution prediction for loss (no img_idxs) ----
        # # depth_prob_gt is (B,1,Nfeat,256) where Nfeat=224*224 (2x2 patches)
        # B, D, H, W = depth_prob_448.shape

        # # pool per-pixel prob -> per-2x2-patch prob : (B,256,224,224)
        # prob_tok = F.avg_pool2d(depth_prob_448, kernel_size=2, stride=2)

        # # reshape to (B,1,Nfeat,D)
        # prob_tok = prob_tok.permute(0, 2, 3, 1).reshape(B, -1, D).unsqueeze(1).contiguous()  # (B,1,224*224,256)

        # eps = 1e-6
        # end_points["depth_prob_logits"] = prob_tok.clamp_min(eps).log()  # (B,1,Nfeat,256)

        # ---- 2) backproject ONLY at sampled pixels (aligned to labels) ----
        # cloud: (B,20000,3)
        cloud = depth_to_cloud_by_img_idxs(depth_map_pred.detach(), K, img_idxs)
        end_points["point_clouds"] = cloud

        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0)
            # 也可以手动触发：end_points["force_vis"]=True
            do_vis = do_vis or bool(end_points.get("force_vis", False))
            if do_vis and ("gt_depth_m" in end_points):
                gt_depth_map = self._as_b1hw(
                    end_points["gt_depth_m"],
                    hw=img.shape[-2:],
                    device=img.device,
                    dtype=depth_map_pred.dtype,
                    name="gt_depth_m",
                )
                cloud_gt = depth_to_cloud_by_img_idxs(gt_depth_map, K, img_idxs)
                cloud_obs = None
                if self.use_obs_depth and obs_depth_448 is not None:
                    cloud_obs = depth_to_cloud_by_img_idxs(obs_depth_448, K, img_idxs)

                self._save_pred_gt_cloud_ply(
                    cloud_pred=cloud,
                    cloud_gt=cloud_gt,
                    end_points=end_points,
                    cloud_obs=cloud_obs,
                )
                
                self._save_depth_maps_vis(
                    end_points=end_points,
                    img=img,
                    img_idxs=img_idxs,
                    depth_net_pred=depth_net_pred,
                    depth_map_pred=depth_map_pred,
                    depth_tok=None,
                    raw_depth=depth_net_raw,
                    obs_depth=obs_depth_448,
                    gt_depth=end_points.get("gt_depth_m", None),
                    depth_confidence=None,
                    depth_correction=depth_refined_correction,
                )
                
            self._vis_iter += 1

        # ---- 3) ME voxel coords ----
        end_points["coordinates_for_voxel"] = _make_coords_for_me(cloud, self.voxel_size)

        # ---- 4) run original EcoGrasp (point-center head + original supervision) ----
        end_points = self.grasp_net(end_points)
        return end_points

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