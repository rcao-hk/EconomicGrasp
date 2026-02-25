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


class DepthFusionSpatialEnhancer(nn.Module):
    """
    BIP3D-style spatial enhancer (single stride=2 feature map).
    - Use depth distribution to build an implicit geometry embedding (IPE)
    - Fuse IPE back to 2D token features

    Notes:
      * Default uses mean depth (BIP3D linear pts_fc equivalence).
      * Optionally append var/entropy so distribution shape matters.
    """
    def __init__(
        self,
        feature_3d_dim=32,
        use_var=True,
        use_entropy=True,
        ff_dim=1024,
        eps=1e-6,
    ):
        super().__init__()
        self.feature_3d_dim = int(feature_3d_dim)
        self.use_var = bool(use_var)
        self.use_entropy = bool(use_entropy)
        self.ff_dim = int(ff_dim)
        self.eps = float(eps)

        # geometry embedding: input is [x,y,z] (+ var + entropy)
        in_dim = 3
        if self.use_var:
            in_dim += 1
        if self.use_entropy:
            in_dim += 1

        self.geom_fc = nn.Sequential(
            nn.Linear(in_dim, self.feature_3d_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_3d_dim, self.feature_3d_dim),
        )

        # fusion layers are built lazily because embed_dims (C) may be unknown beforehand
        self._built = False
        self.fusion_fc1 = None
        self.fusion_fc2 = None
        self.norm = None

    def _build(self, embed_dims: int, device):
        fusion_dim = embed_dims + self.feature_3d_dim
        self.fusion_fc1 = nn.Linear(fusion_dim, self.ff_dim).to(device)
        self.fusion_fc2 = nn.Linear(self.ff_dim, embed_dims).to(device)
        self.norm = nn.LayerNorm(embed_dims).to(device)
        self._built = True

    @staticmethod
    def _token_centers(Ht, Wt, stride, device, dtype):
        # center of each stride x stride patch in 448x448 coordinates
        # u = (j + 0.5)*s - 0.5 ; v similarly
        j = torch.arange(Wt, device=device, dtype=dtype)
        i = torch.arange(Ht, device=device, dtype=dtype)
        u = (j + 0.5) * stride - 0.5
        v = (i + 0.5) * stride - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # (Ht,Wt)
        return uu.reshape(-1), vv.reshape(-1)          # (Nfeat,)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        # 如果 ckpt 里有 fusion 权重但当前还没 build，就先 build 再加载
        key_w2 = prefix + "fusion_fc2.weight"
        if (not getattr(self, "_built", False)) and (key_w2 in state_dict):
            w2 = state_dict[key_w2]  # (embed_dims, ff_dim)
            embed_dims = int(w2.shape[0])
            # fusion_fc1.weight: (ff_dim, fusion_dim)
            w1 = state_dict[prefix + "fusion_fc1.weight"]
            self.ff_dim = int(w1.shape[0])
            self._build(embed_dims, device=w2.device)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )
        
    def forward(self, feat_tok: torch.Tensor, prob_tok: torch.Tensor, K: torch.Tensor, stride: int = 2):
        """
        feat_tok: (B,C,224,224)
        prob_tok: (B,D,224,224)
        K:        (B,3,3) intrinsics for 448
        """
        B, C, Ht, Wt = feat_tok.shape
        Bd, D, Ht2, Wt2 = prob_tok.shape
        assert (B == Bd) and (Ht == Ht2) and (Wt == Wt2)

        if not self._built:
            self._build(C, feat_tok.device)

        # (B,Nfeat,C)
        feat = feat_tok.permute(0, 2, 3, 1).reshape(B, -1, C)

        # (B,Nfeat,D)
        prob = prob_tok.permute(0, 2, 3, 1).reshape(B, -1, D).clamp_min(self.eps)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # depth bins (must be provided by caller via buffer ideally; here we infer from prob only -> caller should pass)
        # We'll require caller to attach `self.depth_bins` dynamically for clean usage.
        assert hasattr(self, "depth_bins"), "Please set enhancer.depth_bins = torch.linspace(min_d,max_d,D) as buffer."
        z_bins = self.depth_bins.to(device=prob.device, dtype=prob.dtype).view(1, 1, D)  # (1,1,D)

        # mean depth: (B,Nfeat,1)
        z_mean = (prob * z_bins).sum(dim=-1, keepdim=True)

        feats_geom = []

        # build ray directions from token centers
        uu, vv = self._token_centers(Ht, Wt, stride, prob.device, prob.dtype)  # (Nfeat,)
        uu = uu.view(1, -1, 1).expand(B, -1, -1)
        vv = vv.view(1, -1, 1).expand(B, -1, -1)

        fx = K[:, 0, 0].view(B, 1, 1).to(prob.dtype)
        fy = K[:, 1, 1].view(B, 1, 1).to(prob.dtype)
        cx = K[:, 0, 2].view(B, 1, 1).to(prob.dtype)
        cy = K[:, 1, 2].view(B, 1, 1).to(prob.dtype)

        x_dir = (uu - cx) / fx
        y_dir = (vv - cy) / fy

        X = x_dir * z_mean
        Y = y_dir * z_mean
        Z = z_mean
        feats_geom.append(torch.cat([X, Y, Z], dim=-1))  # (B,Nfeat,3)

        if self.use_var:
            z2 = (prob * (z_bins ** 2)).sum(dim=-1, keepdim=True)
            z_var = (z2 - z_mean ** 2).clamp_min(0.0)
            feats_geom.append(z_var)

        if self.use_entropy:
            ent = -(prob * prob.log()).sum(dim=-1, keepdim=True)
            feats_geom.append(ent)

        geom_in = torch.cat(feats_geom, dim=-1)            # (B,Nfeat,in_dim)
        geom_feat = self.geom_fc(geom_in)                  # (B,Nfeat,C3d)

        fused = torch.cat([feat, geom_feat], dim=-1)       # (B,Nfeat,C+C3d)
        out = F.relu(self.fusion_fc1(fused), inplace=True)
        out = self.fusion_fc2(out)                         # (B,Nfeat,C)

        out = self.norm(out + feat)                        # residual + LN
        out = out.view(B, Ht, Wt, C).permute(0, 3, 1, 2).contiguous()
        return out


from .economicgrasp_depth import DINOv2DepthDistributionNet
class economicgrasp_c2(nn.Module):
    """
    C2-A: BIP3D-style spatial enhancer
      - depth distribution supervision (depth_prob_gt)
      - use depth_prob to build IPE and enhance 2D token features
      - xyz for grasp branch still uses detached expected depth
    """
    def __init__(
        self,
        cylinder_radius=0.05,
        seed_feat_dim=512,
        is_training=True,
        depth_stride=2,       # token stride=2 -> 224x224
        min_depth=0.2,
        max_depth=1.0,
        bin_num=256,
        feature_3d_dim=32,
        use_var=True,
        use_entropy=True,
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
        self.bin_num   = int(bin_num)

        # ---- your distribution depth net (assumed available) ----
        # should return: depth_map_pred(448), depth_tok(224), img_feat(path_1,448),
        #               depth_prob_448(B,D,448,448), depth_logits_448(B,D,448,448)
        self.depth_net = DINOv2DepthDistributionNet(
            encoder="vitb",
            stride=depth_stride,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            bin_num=self.bin_num,
            freeze_backbone=True,
        )

        # ---- spatial enhancer (token-level) ----
        self.enhancer = DepthFusionSpatialEnhancer(
            feature_3d_dim=feature_3d_dim,
            use_var=use_var,
            use_entropy=use_entropy,
        )
        # set bins once
        self.enhancer.register_buffer(
            "depth_bins",
            torch.linspace(self.min_depth, self.max_depth, self.bin_num),
            persistent=False
        )
        
        # ---- point feature -> seed feature ----
        self.seed_proj = nn.Sequential(
            nn.LazyLinear(self.seed_feature_dim),
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
        
        self.vis_dir = os.path.join('vis', 'c2')  # e.g. "vis_cloud"
        self.vis_every = 1000
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
        self.debug_every = 50
        self._dbg_iter = 0

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
        
    def forward(self, end_points: dict):
        img = end_points["img"]            # (B,3,448,448)
        K   = end_points["K"]              # (B,3,3)
        img_idxs = end_points["img_idxs"]  # (B,N) in 448*448

        B, _, H, W = img.shape
        assert (H, W) == (448, 448)
        s = 2
        Ht, Wt = H // s, W // s
        Nfeat = Ht * Wt

        # ---------- depth inference (classification) ----------
        depth_map_pred_448, depth_tok_dbg, img_feat, depth_prob_448, depth_logits_448 = \
            self.depth_net(img, return_prob=True, return_tok_prob=False)

        end_points["img_feat_dpt"] = img_feat  # (B,C,448,448)

        # --- token prob from pooled logits (BIP3D-like) ---
        s = 2  # must match build_depth_prob_gt patch size
        logits_tok = F.avg_pool2d(depth_logits_448, kernel_size=s, stride=s)  # (B,D,224,224)
        prob_tok_map = torch.softmax(logits_tok, dim=1)                       # (B,D,224,224)

        # (B,1,Nfeat,D) for BCE loss
        prob_tok_flat = prob_tok_map.permute(0, 2, 3, 1).reshape(B, 1, -1, self.bin_num).contiguous()
        end_points["depth_prob_pred"] = prob_tok_flat
        end_points.pop("depth_prob_logits", None)

        # --- IMPORTANT: make depth_map_pred consistent with token distribution ---
        z_bins = self.depth_net.depth_bins.view(1, self.bin_num, 1, 1).to(prob_tok_map)  # (1,D,1,1)

        depth_tok = (prob_tok_map * z_bins).sum(dim=1, keepdim=True)               # (B,1,224,224)
        depth_map_pred = F.interpolate(depth_tok, size=(H, W), mode="nearest")      # (B,1,448,448)

        end_points["depth_tok_pred"] = depth_tok
        end_points["depth_map_pred"] = depth_map_pred

        feat_tok = F.avg_pool2d(img_feat, kernel_size=2, stride=2)  # (B,C,224,224)
        feat_tok_enh = self.enhancer(feat_tok, prob_tok_map, K, stride=2)

        # ---------- 4) gather per-point enhanced token feature ----------
        # map 448 pixel idx -> token idx
        u = (img_idxs % W).long().clamp(0, W - 1)
        v = (img_idxs // W).long().clamp(0, H - 1)
        tu = (u // s).clamp(0, Wt - 1)
        tv = (v // s).clamp(0, Ht - 1)
        tid = (tv * Wt + tu).long()  # (B,N) in [0,Nfeat)

        Ctok = feat_tok_enh.shape[1]
        feat_flat = feat_tok_enh.view(B, Ctok, -1)  # (B,Ctok,Nfeat)
        tid_expand = tid.unsqueeze(1).expand(-1, Ctok, -1)
        point_img_feat = feat_flat.gather(2, tid_expand).transpose(1, 2).contiguous()  # (B,N,Ctok)

        # ---------- 5) xyz for grasp branch (detached expected depth) ----------
        z = gather_depth_by_img_idxs(depth_map_pred, img_idxs)  # (B,N,1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
        z_sg = z.detach()

        u1, v1 = img_idxs_to_uv(img_idxs, W)          # (B,N,1) float
        xyz = backproject_uvz(u1, v1, z_sg, K)        # (B,N,3)
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
                    xyz_gt = backproject_uvz(u1, v1, z_gt, K)                       # (B,N,3)
                    self._save_pred_gt_cloud_ply(xyz, xyz_gt, end_points)

            self._vis_iter += 1
            
        # ---------- 6) point feature -> seed feature ----------
        fused = self.seed_proj(point_img_feat)                # (B,N,512)
        seed_features = fused.transpose(1, 2).contiguous()     # (B,512,N)

        # ---------- 7) same as C1 from here ----------
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # (B,N,512)

        objectness_score = end_points["objectness_score"]
        graspness_score  = end_points["graspness_score"].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        graspable_mask  = (objectness_pred == 1) & (graspness_score > cfgs.graspness_threshold)

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