import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules_economicgrasp import GraspableNet, ViewNet, Cylinder_Grouping_Global_Interaction, Grasp_Head_Local_Interaction
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from utils.arguments import cfgs
from utils.label_generation import process_grasp_labels, batch_viewpoint_params_to_matrix
import numpy as np
import os
import open3d as o3d


# ========= util =========
def img_idxs_to_uv(img_idxs: torch.Tensor, W: int):
    u = (img_idxs % W).float()
    v = (img_idxs // W).float()
    return u, v

def gather_depth_by_img_idxs(depth_map_1hw: torch.Tensor, img_idxs: torch.Tensor):
    # depth_map_1hw: (B,1,H,W) or (B,H,W)
    if depth_map_1hw.dim() == 3:
        depth_flat = depth_map_1hw.reshape(depth_map_1hw.size(0), -1)
    else:
        depth_flat = depth_map_1hw[:, 0].reshape(depth_map_1hw.size(0), -1)
    return depth_flat.gather(1, img_idxs)

def backproject_uvz(u, v, z, K):
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)


from .economicgrasp_depth import DINOv2DepthRegressionNet
class economicgrasp_c1(nn.Module):
    """
    C1 image-centric EcoGrasp (DepthAnything feature as image feature):
      - RGB -> depth_net -> (depth_448, depth_tok, img_feat=DPT path_1)
      - gather per-point img_feat & predicted depth by img_idxs
      - backproject(pred depth) -> xyz (camera coords)   <-- grasp branch always uses pred xyz
      - geometry PE (xyz + 1/z) + fuse -> seed_features (B,512,N)
      - reuse original graspable/view/cylinder/head + process_grasp_labels
      - depth regression supervised by GT depth (full map)
    """
    def __init__(
        self,
        cylinder_radius=0.05,
        seed_feat_dim=512,
        img_feat_dim=None,          # None: fuse 用 LazyLinear 自动适配 depth_net 的 Cf
        is_training=True,
        pe_dim=64,
        depth_stride=2,
        min_depth=0.2,
        max_depth=1.0,
    ):
        super().__init__()
        self.is_training = bool(is_training)

        self.seed_feature_dim = int(seed_feat_dim)
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points  = cfgs.m_point
        self.num_view  = cfgs.num_view

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        self.depth_net = DINOv2DepthRegressionNet(
            encoder="vitb",
            stride=depth_stride,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            freeze_backbone=True
        )

        # -------- per-point img feature projection (optional) --------
        self.img_feat_dim = img_feat_dim
        if img_feat_dim is None:
            self.img_proj = nn.Identity()
        else:
            # 用 LazyLinear 让 Cf 自动推断（不依赖你手写 Cf=128）
            self.img_proj = nn.LazyLinear(int(img_feat_dim))

        # -------- geometry PE: [x,y,z,1/z] --------
        self.pe_mlp = nn.Sequential(
            nn.Linear(4, pe_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pe_dim, pe_dim),
            nn.ReLU(inplace=True),
        )

        # -------- fuse: [img_feat + pe] -> 512 --------
        if img_feat_dim is None:
            # 输入维度未知，用 LazyLinear 自动适配 (Cf + pe_dim)
            self.fuse = nn.Sequential(
                nn.LazyLinear(self.seed_feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.seed_feature_dim, self.seed_feature_dim),
            )
        else:
            self.fuse = nn.Sequential(
                nn.Linear(int(img_feat_dim) + pe_dim, self.seed_feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.seed_feature_dim, self.seed_feature_dim),
            )

        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

        # self._init_weights(skip_modules=(self.depth_net,))
        
        self.vis_dir = os.path.join('vis', 'c1_detach')  # e.g. "vis_cloud"
        self.vis_every = 1000
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
        self.debug_every = 50
        self._dbg_iter = 0

    @torch.no_grad()
    def _save_pred_gt_cloud_ply(self, cloud_pred: torch.Tensor, cloud_gt: torch.Tensor, end_points: dict):
        """
        cloud_pred/cloud_gt: (B,N,3) float
        save only batch[0], pred=red, gt=blue
        """
        if o3d is None or self.vis_dir is None:
            return

        p = cloud_pred[0].detach().float().cpu().numpy()
        g = cloud_gt[0].detach().float().cpu().numpy()

        def _valid(x):
            m = np.isfinite(x).all(axis=1)
            m &= (x[:, 2] > 0)  # z>0
            return x[m]

        p = _valid(p)
        g = _valid(g)
        if p.shape[0] == 0 or g.shape[0] == 0:
            return

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0  # red
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0  # blue

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

        out_path = os.path.join(self.vis_dir, f"pred_gt_xyz_{tag}_iter{self._vis_iter:06d}.ply")
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    
    
    def _init_weights(self, skip_modules=()):
        for module in self.modules():
            if any(module is m for m in skip_modules):
                continue
            # LazyLinear 在首次 forward 前参数未 materialize，别强行 init
            if isinstance(module, nn.LazyLinear):
                continue
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, np.sqrt(2. / n))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, end_points: dict):
        img = end_points["img"]            # (B,3,448,448)
        K   = end_points["K"]              # (B,3,3)
        img_idxs = end_points["img_idxs"]  # (B,N) flatten idx in 448*448

        B, _, H, W = img.shape
        assert (H, W) == (448, 448)

        # -------- 1) depth_net: depth + img_feat --------
        depth_map_pred, depth_tok, img_feat = self.depth_net(img)  # depth_map_pred: (B,1,448,448)
        end_points["depth_map_pred"] = depth_map_pred
        end_points["depth_tok_pred"] = depth_tok

        # -------- 2) align img_feat to 448 & gather per-point img_feat --------
        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

        Cf = img_feat.shape[1]
        img_feat_flat = img_feat.view(B, Cf, -1)  # (B,Cf,HW)
        idx = img_idxs.long().clamp(0, H * W - 1).unsqueeze(1).expand(-1, Cf, -1)
        point_img_feat = torch.gather(img_feat_flat, 2, idx).transpose(1, 2).contiguous()  # (B,N,Cf)

        # optional projection
        point_img_feat = self.img_proj(point_img_feat)  # (B,N,C') if configured

        # -------- 3) ALWAYS use predicted depth to backproject xyz for grasp branch --------
        z = gather_depth_by_img_idxs(depth_map_pred, img_idxs)  # (B,N,1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

        # stop-grad to depth via geometry path
        z_sg = z.detach()
        u, v = img_idxs_to_uv(img_idxs, W) # (B,N,1)
        xyz = backproject_uvz(u, v, z_sg, K)   # xyz is now disconnected from depth_map_pred
        end_points["point_clouds"] = xyz
        
        # -------- vis: save pred xyz vs gt xyz (same img_idxs) --------
        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0)
            do_vis = do_vis or bool(end_points.get("force_vis", False))

            if do_vis and ("gt_depth_m" in end_points):
                gt_depth = end_points["gt_depth_m"]
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)          # (B,1,H,W)
                elif gt_depth.dim() == 4:
                    pass                                      # already (B,1,H,W) or (B,C,H,W)
                else:
                    gt_depth = None

                if gt_depth is not None:
                    # 用同一套 (u,v) 和 img_idxs 对齐采样
                    z_gt = gather_depth_by_img_idxs(gt_depth, img_idxs)           # (B,N,1)
                    z_gt = torch.nan_to_num(z_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
                    xyz_gt = backproject_uvz(u, v, z_gt, K)                       # (B,N,3)
                    self._save_pred_gt_cloud_ply(xyz, xyz_gt, end_points)

            self._vis_iter += 1
            
        # -------- 4) geometry PE + fuse -> seed_features --------
        invz = (1.0 / xyz[..., 2:3]).clamp_max(1e6)
        pe = self.pe_mlp(torch.cat([xyz, invz], dim=-1))                 # (B,N,pe_dim)

        fused = self.fuse(torch.cat([point_img_feat, pe], dim=-1))       # (B,N,512)
        seed_features = fused.transpose(1, 2).contiguous()               # (B,512,N)

        # -------- 5) graspable mask --------
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)            # (B,N,512)

        objectness_score = end_points["objectness_score"]                # (B,2,N)
        graspness_score  = end_points["graspness_score"].squeeze(1)      # (B,N)

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask  = graspness_score > cfgs.graspness_threshold
        graspable_mask  = objectness_mask & graspness_mask

        # -------- 6) FPS downsample (robust) --------
        seed_xyz = xyz
        point_num = seed_xyz.size(1)

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
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()
                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            if cur_idx.numel() < self.M_points:
                rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=seed_xyz.device)
                ridx = cur_idx[rep]
                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()
                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                continue

            xyz_in = seed_xyz[i].index_select(0, cur_idx).unsqueeze(0).contiguous()  # (1,Ng,3)
            fps_idxs = furthest_point_sample(xyz_in, self.M_points).to(torch.int32).contiguous()

            cur_seed_xyz = gather_operation(xyz_in.transpose(1, 2).contiguous(), fps_idxs) \
                .transpose(1, 2).squeeze(0).contiguous()  # (M,3)

            feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()  # (Ng,512)
            cur_feat = gather_operation(feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs) \
                .squeeze(0).contiguous()  # (512,M)

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0).contiguous()            # (B,M,3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0).contiguous()  # (B,512,M)

        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = (
            torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32) / float(B)
        ).detach().reshape(())

        # -------- 7) view selection --------
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        # -------- 8) label processing (still uses xyz_graspable) --------
        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        # -------- 9) grouping + head --------
        group_features = self.cy_group(seed_xyz_graspable, seed_features_graspable, grasp_top_views_rot)
        end_points = self.grasp_head(group_features, end_points)

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