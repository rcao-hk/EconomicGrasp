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


def _tstats(name, t: torch.Tensor):
    td = t.detach()
    numel = td.numel()

    tmin = float(td.min().item()) if numel else float("nan")
    tmax = float(td.max().item()) if numel else float("nan")

    # mean/std 统一用 float 计算，避免 Long 报错
    tf = td.float() if (numel and not td.is_floating_point()) else td
    mean = float(tf.mean().item()) if numel else float("nan")
    std  = float(tf.std().item()) if numel else float("nan")

    return (f"{name}: shape={tuple(td.shape)} dtype={str(td.dtype).replace('torch.','')} dev={td.device} "
            f"min={tmin:.6f} max={tmax:.6f} mean={mean:.6f} std={std:.6f}")


def _dist_stats(name, logits_bdhw: torch.Tensor, anchors_1d: torch.Tensor, sample_xy=((0,0),)):
    """
    logits_bdhw: (B,D,H,W)
    anchors_1d:  (D,)
    """
    with torch.no_grad():
        B, D, H, W = logits_bdhw.shape
        prob = logits_bdhw.softmax(dim=1)
        # 1) 空间变化：logits 在 H,W 上的 std（对每个 bin 统计，再平均）
        spatial_std = logits_bdhw.std(dim=(2,3)).mean().item()
        # 2) “bin 维度”的尖锐程度：max prob 平均、entropy 平均
        pmax = prob.max(dim=1).values.mean().item()
        ent = (-(prob.clamp_min(1e-9) * prob.clamp_min(1e-9).log()).sum(dim=1)).mean().item()  # (B,H,W) -> mean
        # 3) 期望深度范围
        anchors = anchors_1d.view(1, D, 1, 1).to(prob)
        depth_exp = (prob * anchors).sum(dim=1)  # (B,H,W)
        dmin = depth_exp.min().item()
        dmax = depth_exp.max().item()
        dstd = depth_exp.std().item()

        msg = (f"[DBG][{name}] logits_spatial_std(mean over D)={spatial_std:.6f} | "
               f"mean_maxprob={pmax:.6f} | mean_entropy={ent:.6f} (max~{np.log(D):.6f}) | "
               f"E[depth] min={dmin:.6f} max={dmax:.6f} std={dstd:.6f}")
        print(msg)

        # 4) 打印几个像素点 top-k
        for (yy, xx) in sample_xy:
            yy = int(np.clip(yy, 0, H-1))
            xx = int(np.clip(xx, 0, W-1))
            topv, topi = prob[0, :, yy, xx].topk(5)
            a = anchors_1d[topi.cpu()].numpy()
            print(f"[DBG][{name}] (y={yy},x={xx}) top5 bins={topi.tolist()} "
                  f"top5 p={topv.tolist()} top5 depth={a.tolist()}")
            

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


from models.dinov2_dpt import DA2
class DINOv2DepthDistributionNet(nn.Module):
    """
    使用 DA2 的 DPTHead 直接输出 bin logits（out_dim=bin_num），不额外加 conv。

    forward(img, return_prob=False):
      return_prob=False:
        depth_448: (B,1,448,448)  E[z] in meters
        depth_tok: (B,1,448/stride,448/stride) debug
        img_feat:  (B,Cf,Hf,Wf)  path_1 feature (通常 patch_h*14, patch_w*14 -> 448)
      return_prob=True:
        + depth_prob_448:   (B,bin_num,448,448)
        + depth_logits_448: (B,bin_num,448,448)
    """
    def __init__(self,
                 encoder="vitb",
                 stride=2,
                 min_depth=0.2,
                 max_depth=2.0,
                 bin_num=256,
                 freeze_backbone=True,
                 eps=1e-6):
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

        # 关键：out_dim=bin_num
        self.depthnet = DA2(**{
            **model_configs[encoder],
            "out_dim": self.bin_num
        })

        # 只加载 backbone（与你现有一致）
        ckpt = torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu")
        self.depthnet.load_state_dict({k: v for k, v in ckpt.items() if "pretrained" in k}, strict=False)

        if self.freeze_backbone_flag:
            self._freeze_backbone()

        # bin centers
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

    def forward(self, img: torch.Tensor, return_prob: bool = False):
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)

        patch_h, patch_w = H // 14, W // 14
        feats = self.depthnet.pretrained.get_intermediate_layers(
            img,
            self.depthnet.intermediate_layer_idx[self.depthnet.encoder],
            return_class_token=True
        )

        # 直接用 depth_head：out 是 (B,bin_num,448,448)
        img_feat, depth_logits_448 = self.depthnet.depth_head(feats, patch_h, patch_w)

        # logits -> prob
        depth_prob_448 = torch.softmax(depth_logits_448, dim=1)  # (B,K,448,448)

        # E[z]
        z = self.depth_bins.view(1, self.bin_num, 1, 1).to(depth_prob_448)
        depth_448 = (depth_prob_448 * z).sum(dim=1, keepdim=True)
        depth_448 = depth_448.clamp(self.min_depth, self.max_depth)

        # token debug
        if self.stride > 1:
            depth_tok = F.interpolate(depth_448, size=(H // self.stride, W // self.stride), mode="nearest")
        else:
            depth_tok = depth_448

        # 你后续 gather 希望 feature 对齐 448：如果 img_feat 不是 448，就插值一下
        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

        if return_prob:
            return depth_448, depth_tok, img_feat, depth_prob_448, depth_logits_448
        else:
            return depth_448, depth_tok, img_feat
    
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

    def forward(self, img):
        """
        img: (B,3,448,448) (你 dataloader 已经 Normalize 了)
        return:
          depth_reg_448: (B,1,448,448) meters in [min_depth, max_depth]
          depth_reg_tok: (B,1,Hr,Wr) (Hr=Wr=224 when stride=2)
          img_feat:      (B,C,?,?) 你 DPTHead 的 path_1
        """
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)

        patch_h, patch_w = H // 14, W // 14
        feats = self.depthnet.pretrained.get_intermediate_layers(
            img,
            self.depthnet.intermediate_layer_idx[self.depthnet.encoder],
            return_class_token=True
        )
        img_feat, depth_logits_448 = self.depthnet.depth_head(feats, patch_h, patch_w)   # (B,1,448,448)

        # ✅ 把输出约束到 [min_depth, max_depth]，避免爆炸 / 无意义尺度
        depth_448 = F.sigmoid(depth_logits_448) * self.max_depth
        # depth_448 = depth_448 * self.max_depth

        # token 级别
        if self.stride > 1:
            depth_tok = F.interpolate(depth_448, size=(H // self.stride, W // self.stride), mode='nearest')
        else:
            depth_tok = depth_448

        return depth_448, depth_tok, img_feat
    

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
            encoder="vitb",
            stride=depth_stride,     # 2 -> 224x224 token debug
            min_depth=min_depth,
            max_depth=max_depth,
            freeze_backbone=True
        )

        self.grasp_net = EconomicGrasp3D(
            cylinder_radius=cylinder_radius,
            seed_feat_dim=seed_feat_dim,
            is_training=is_training,
            voxel_size=voxel_size
        )

        self.vis_dir = os.path.join('vis', 'reg')  # e.g. "vis_cloud"
        self.vis_every = 100
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
        self.debug_every = 50
        self._dbg_iter = 0
                
    @torch.no_grad()
    def _save_pred_gt_cloud_ply(self, cloud_pred: torch.Tensor, cloud_gt: torch.Tensor, end_points: dict):
        """
        cloud_pred/cloud_gt: (B,20000,3) float32
        save only batch[0] as ply, pred=red, gt=blue
        """
        if o3d is None:
            return

        # pick first scene in batch
        p = cloud_pred[0].detach().float().cpu().numpy()
        g = cloud_gt[0].detach().float().cpu().numpy()

        # filter invalid
        def _valid(x):
            m = np.isfinite(x).all(axis=1)
            m &= (x[:, 2] > 0)  # z>0
            return x[m]

        p = _valid(p)
        g = _valid(g)

        if p.shape[0] == 0 or g.shape[0] == 0:
            return

        # colors
        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0  # red
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0  # blue

        pts = np.concatenate([p, g], axis=0)
        cols = np.concatenate([p_col, g_col], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        # name tag (optional)
        tag = end_points.get("vis_tag", None)
        if tag is None:
            # 尽量用 scene/frame 命名（若你 end_points 里有的话）
            scene = end_points.get("scene", "scene")
            frame = end_points.get("frame", "frame")
            tag = f"{scene}_{frame}"

        out_path = os.path.join(self.vis_dir, f"pred_gt_cloud_{tag}_iter{self._vis_iter:06d}.ply")
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

    def forward(self, end_points: dict):
        img = end_points["img"]          # (B,3,448,448)
        K = end_points["K"]              # (B,3,3)
        img_idxs = end_points["img_idxs"]  # (B,20000) flatten idx in 448*448

        depth_map_pred, depth_tok, img_feat = self.depth_net(img)
        end_points["depth_map_pred"] = depth_map_pred      # (B,1,448,448)
        end_points["depth_tok_pred"] = depth_tok           # (B,1,224,224) debug 用
        end_points["img_feat_dpt"] = img_feat

        # -------- depth classification (whole-image, patch-level loss) --------
        # depth_map_pred, depth_tok, img_feat, depth_prob_448, depth_logits_448 = self.depth_net(img, return_prob=True)

        # end_points["depth_map_pred"] = depth_map_pred      # (B,1,448,448)  E[z] for backprojection
        # end_points["depth_tok_pred"] = depth_tok           # (B,1,224,224)  debug
        # end_points["img_feat_dpt"]   = img_feat            # (B,C,448,448)

        # ---- build token/patch distribution prediction for loss (no img_idxs) ----
        # depth_prob_gt is (B,1,Nfeat,256) where Nfeat=224*224 (2x2 patches)
        # B, D, H, W = depth_prob_448.shape

        # # pool per-pixel prob -> per-2x2-patch prob : (B,256,224,224)
        # prob_tok = F.avg_pool2d(depth_prob_448, kernel_size=2, stride=2)

        # # reshape to (B,1,Nfeat,D)
        # prob_tok = prob_tok.permute(0, 2, 3, 1).reshape(B, -1, D).unsqueeze(1).contiguous()  # (B,1,224*224,256)

        # eps = 1e-6
        # end_points["depth_prob_logits"] = prob_tok.clamp_min(eps).log()  # (B,1,Nfeat,256)

        # if hasattr(self.depth_net, "depth_anchors"):
        #     end_points["depth_anchors_1d"] = self.depth_net.depth_anchors.view(-1).detach()
        # # ------------------ DEBUG BLOCK (dataset-key aligned) ------------------
        # with torch.no_grad():
        #     print("[DBG] ===== DepthHead (dataset-key aligned) =====")
        #     print(_tstats("img", img))
        #     print(_tstats("K", K))
        #     print(_tstats("img_idxs", img_idxs))

        #     # extra keys from dataloader (may exist only in train)
        #     if "sampled_masked_idxs" in end_points:
        #         print(_tstats("sampled_masked_idxs", end_points["sampled_masked_idxs"]))  # (B,20000) or (20000,)
        #     if "pix_flat" in end_points:
        #         print(_tstats("pix_flat(orig H*W)", end_points["pix_flat"]))              # (B,20000) or (20000,)

        #     Hr, Wr = spatial_shape
        #     print(f"[DBG] spatial_shape(Hr,Wr)=({Hr},{Wr})  (expect 224x224 when stride=2)")

        #     # anchors sanity
        #     if hasattr(self.depth_net, "depth_anchors"):
        #         a = self.depth_net.depth_anchors.detach().view(-1).cpu()
        #         print(f"[DBG] anchors: D={a.numel()} first={float(a[0]):.6f} last={float(a[-1]):.6f} mean={float(a.mean()):.6f}")

        #     print(_tstats("depth_logits_tok", depth_logits_tok))  # (B,256,224,224)
        #     print(_tstats("depth_map_pred", depth_map_pred))      # (B,1,448,448)

        #     # -------- compare with GT depth (virtual) --------
        #     if "gt_depth_m" in end_points:
        #         gt_depth_m = end_points["gt_depth_m"]  # from dataset: (B,448,448) or (448,448)
        #         if gt_depth_m.ndim == 2:
        #             gt_depth_m = gt_depth_m.unsqueeze(0)  # (1,448,448)
        #         # unify shape to (B,1,448,448) for stat
        #         if gt_depth_m.ndim == 3:
        #             gt_depth_m_ = gt_depth_m.unsqueeze(1)
        #         else:
        #             gt_depth_m_ = gt_depth_m
        #         print(_tstats("gt_depth_m", gt_depth_m_))

        #         # sample gt depth at the same img_idxs to compare z distribution directly
        #         B0 = img.shape[0]
        #         gt_flat = gt_depth_m_.view(B0, -1)  # (B,448*448)
        #         z_gt = torch.gather(gt_flat, 1, img_idxs)  # (B,20000)
        #         z_pred = torch.gather(depth_map_pred.view(B0, -1), 1, img_idxs)  # (B,20000)
        #         print(_tstats("z_gt@img_idxs", z_gt[0]))
        #         print(_tstats("z_pred@img_idxs", z_pred[0]))
        #         print(f"[DBG] z_gt std={float(z_gt[0].std().item()):.6e} | z_pred std={float(z_pred[0].std().item()):.6e}")

        #         # quick correlation check (batch0)
        #         zg = z_gt[0].float()
        #         zp = z_pred[0].float()
        #         mask = (zg > 0) & torch.isfinite(zg) & torch.isfinite(zp)
        #         if mask.sum() > 16:
        #             zg0 = zg[mask] - zg[mask].mean()
        #             zp0 = zp[mask] - zp[mask].mean()
        #             corr = (zg0 * zp0).mean() / (zg0.std() * zp0.std() + 1e-12)
        #             print(f"[DBG] corr(z_pred, z_gt) (batch0 valid) = {float(corr.item()):.4f} (expect >0 if learning)")

        #     # -------- distribution diagnostics (pred logits) --------
        #     if hasattr(self.depth_net, "depth_anchors"):
        #         anchors_1d = self.depth_net.depth_anchors.view(-1).detach().cpu()
        #         _dist_stats(
        #             "pred",
        #             depth_logits_tok,  # (B,D,Hr,Wr)
        #             anchors_1d,
        #             sample_xy=((Hr//2, Wr//2), (Hr//4, Wr//4), (3*Hr//4, 3*Wr//4)),
        #         )

        #     # -------- compare pred distribution with GT depth_prob_gt (soft label) --------
        #     if "depth_prob_gt" in end_points:
        #         gt_prob = end_points["depth_prob_gt"]           # dataset: (B,1,Nfeat,D) or (1,Nfeat,D)
        #         gt_w    = end_points.get("depth_prob_weight")   # dataset: (B,1,Nfeat) or (1,Nfeat)

        #         if torch.is_tensor(gt_prob):
        #             print(_tstats("depth_prob_gt", gt_prob))
        #         if torch.is_tensor(gt_w):
        #             print(_tstats("depth_prob_weight", gt_w))

        #         # reshape gt_prob -> (B,D,Hr,Wr) for local inspection, assuming stride=2 => Nfeat=Hr*Wr
        #         try:
        #             if gt_prob.ndim == 3:
        #                 gt_prob_ = gt_prob.unsqueeze(0)  # (1,1,Nfeat,D) ? 这里可能不对，下面再判断
        #             else:
        #                 gt_prob_ = gt_prob

        #             # handle (B,1,Nfeat,D)
        #             if gt_prob_.ndim == 4 and gt_prob_.shape[1] == 1:
        #                 B0 = gt_prob_.shape[0]
        #                 Nfeat = gt_prob_.shape[2]
        #                 D = gt_prob_.shape[3]
        #                 if Nfeat == Hr * Wr and D == depth_logits_tok.shape[1]:
        #                     gt_prob_hw = gt_prob_[0, 0].view(Hr, Wr, D).permute(2, 0, 1).unsqueeze(0)  # (1,D,Hr,Wr)
        #                     # 直接看 GT 分布尖锐程度：maxprob / entropy
        #                     anchors_1d = self.depth_net.depth_anchors.view(-1).detach().cpu() if hasattr(self.depth_net, "depth_anchors") else torch.arange(D)
        #                     _dist_stats("gt_prob", gt_prob_hw.log().clamp_min(-50.0), anchors_1d, sample_xy=((Hr//2, Wr//2),))
        #         except Exception as e:
        #             print(f"[DBG][WARN] depth_prob_gt reshape failed: {repr(e)}")

        #     # -------- verify backproject inputs (batch0) --------
        #     # confirm img_idxs within [0, 448*448-1]
        #     idx0 = img_idxs[0]
        #     print(f"[DBG] img_idxs[0]: min={int(idx0.min().item())} max={int(idx0.max().item())} (expect within [0, {448*448-1}])")

        # ---------------- END DEBUG BLOCK ------------------

        # ---- 2) backproject ONLY at sampled pixels (aligned to labels) ----
        # cloud: (B,20000,3)
        cloud = depth_to_cloud_by_img_idxs(depth_map_pred, K, img_idxs)
        end_points["point_clouds"] = cloud

        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0)
            # 也可以手动触发：end_points["force_vis"]=True
            do_vis = do_vis or bool(end_points.get("force_vis", False))
            if do_vis and ("gt_depth_m" in end_points):
                gt_depth_map = end_points["gt_depth_m"]  # (B,1,448,448)
                # print('gt_depth_map shape:', gt_depth_map.shape)
                print("cloud min/max:", cloud.min().item(), cloud.max().item())
                cloud_gt = depth_to_cloud_by_img_idxs(gt_depth_map.unsqueeze(1), K, img_idxs)  # (B,20000,3)
                self._save_pred_gt_cloud_ply(cloud, cloud_gt, end_points)

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