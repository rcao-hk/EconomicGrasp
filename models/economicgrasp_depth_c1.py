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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        vis_dir='c1',
        vis_every=1000
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
        
        self.vis_dir = vis_dir  # e.g. "vis_cloud"
        self.vis_every = vis_every
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
    
    def _to_vis_bgr(self, img_chw: torch.Tensor):
        """
        img_chw: (3,H,W), torch tensor
        return: uint8 BGR image for cv2
        """
        import cv2
        img = img_chw.detach().cpu().float().permute(1, 2, 0).numpy()  # HWC

        # heuristic de-normalization
        # if image looks normalized by ImageNet stats
        if img.min() < -0.05 or img.max() > 1.05:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = img * std + mean

        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _save_overlay_selected(
        self,
        img: torch.Tensor,                    # (B,3,H,W)
        all_graspable_img_idxs: list,        # len=B, each (Ng,)
        selected_img_idxs: list,             # len=B, each (M,)
        end_points: dict,
        prefix: str = "overlay_selected",
    ):
        """
        可视化:
        green  = all graspable points before FPS
        red    = final selected M points after fallback/repeat/FPS
        """
        import cv2
        if self.vis_dir is None:
            return

        os.makedirs(self.vis_dir, exist_ok=True)
        B, _, H, W = img.shape

        for b in range(B):
            canvas = self._to_vis_bgr(img[b])
            overlay = canvas.copy()

            # ---- all graspable: green small points ----
            if all_graspable_img_idxs[b] is not None and all_graspable_img_idxs[b].numel() > 0:
                idx_all = all_graspable_img_idxs[b].detach().cpu().long().numpy()
                u_all = idx_all % W
                v_all = idx_all // W
                valid = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
                u_all, v_all = u_all[valid], v_all[valid]

                for uu, vv in zip(u_all, v_all):
                    cv2.circle(overlay, (int(uu), int(vv)), 1, (0, 255, 0), -1)

            # blend graspable overlay
            canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

            # ---- selected M points: red larger points ----
            if selected_img_idxs[b] is not None and selected_img_idxs[b].numel() > 0:
                idx_sel = selected_img_idxs[b].detach().cpu().long().numpy()
                u_sel = idx_sel % W
                v_sel = idx_sel // W
                valid = (u_sel >= 0) & (u_sel < W) & (v_sel >= 0) & (v_sel < H)
                u_sel, v_sel = u_sel[valid], v_sel[valid]

                for k, (uu, vv) in enumerate(zip(u_sel, v_sel)):
                    cv2.circle(canvas, (int(uu), int(vv)), 3, (0, 0, 255), -1)
                    # 如果你想标编号，取消下面这行注释
                    # cv2.putText(canvas, str(k), (int(uu)+2, int(vv)-2),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)

            save_path = os.path.join(
                self.vis_dir,
                f"{prefix}_iter{self._vis_iter:06d}",
            )
            os.makedirs(save_path, exist_ok=True) 
            cv2.imwrite(os.path.join(save_path, "overlay_selected.png"), canvas)
        
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

        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0)
            do_vis = do_vis or bool(end_points.get("force_vis", False))
            
        # -------- vis: save pred xyz vs gt xyz (same img_idxs) --------
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

        # for visualization / debugging
        all_graspable_img_idxs = []         # len=B, each (Ng,)
        selected_img_idxs_graspable = []    # len=B, each (M,)

        for i in range(B):
            cur_mask = graspable_mask[i]
            cur_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
            graspable_num_batch += cur_idx.numel()

            if do_vis:
                if cur_idx.numel() > 0:
                    cur_all_img_idxs = img_idxs[i].index_select(0, cur_idx).contiguous()
                else:
                    cur_all_img_idxs = torch.empty((0,), dtype=img_idxs.dtype, device=img_idxs.device)
                all_graspable_img_idxs.append(cur_all_img_idxs.detach().cpu())
                
            if cur_idx.numel() == 0:
                ridx = torch.randint(0, point_num, (self.M_points,), device=seed_xyz.device)
                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()
                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                if do_vis:
                    cur_sel_img_idxs = img_idxs[i].index_select(0, ridx).contiguous()
                    selected_img_idxs_graspable.append(cur_sel_img_idxs.detach().cpu())
                continue

            if cur_idx.numel() < self.M_points:
                rep = torch.randint(0, cur_idx.numel(), (self.M_points,), device=seed_xyz.device)
                ridx = cur_idx[rep]
                cur_seed_xyz = seed_xyz[i].index_select(0, ridx).contiguous()
                cur_feat_mc  = seed_features_flipped[i].index_select(0, ridx).contiguous()
                cur_feat     = cur_feat_mc.transpose(0, 1).contiguous()
                seed_xyz_graspable.append(cur_seed_xyz)
                seed_features_graspable.append(cur_feat)
                if do_vis:
                    cur_sel_img_idxs = img_idxs[i].index_select(0, ridx).contiguous()
                    selected_img_idxs_graspable.append(cur_sel_img_idxs.detach().cpu())
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

            if do_vis:
                # fps_idxs is index inside cur_idx, convert back to original point index
                fps_idxs_long = fps_idxs.squeeze(0).long()      # (M,)
                ridx = cur_idx.index_select(0, fps_idxs_long)   # (M,)
                cur_sel_img_idxs = img_idxs[i].index_select(0, ridx).contiguous()
                selected_img_idxs_graspable.append(cur_sel_img_idxs.detach().cpu())
                
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0).contiguous()            # (B,M,3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0).contiguous()  # (B,512,M)

        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = (
            torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32) / float(B)
        ).detach().reshape(())

        if do_vis and len(selected_img_idxs_graspable) == B:
            end_points["selected_img_idxs_graspable"] = torch.stack(
                [x.to(img_idxs.device) for x in selected_img_idxs_graspable], dim=0
            )  # (B,M)

            self._save_overlay_selected(
                img=img,
                all_graspable_img_idxs=all_graspable_img_idxs,
                selected_img_idxs=selected_img_idxs_graspable,
                end_points=end_points,
                prefix="c1dbg"
            )

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

        if self.vis_dir is not None:
            self._vis_iter += 1
            
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
    

class economicgrasp_c2_1(nn.Module):
    """
    C2-A (Stage-1 token seeds):
      - depth distribution supervision (depth_prob_gt) at token level (224x224)
      - BIP3D-style spatial enhancer: (feat_tok, prob_tok_map, K) -> feat_tok_enh
      - token-grid seeds: run GraspableNet on ALL tokens, select Top-M tokens
      - xyz_graspable comes from token centers backprojected with z_tok (train: optional GT z_tok)
      - reuse original ViewNet / process_grasp_labels / Cylinder grouping / Grasp head
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
        use_gt_xyz_for_train=False,   # ✅ Stage-1 stability: label matching uses GT token xyz when available
        topk_use_objectness=True,    # ✅ if False: select by graspness only
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.topk_use_objectness = bool(topk_use_objectness)

        self.seed_feature_dim = int(seed_feat_dim)
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points  = cfgs.m_point
        self.num_view  = cfgs.num_view

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num   = int(bin_num)

        self.converage_ratio = 0.3
        # ---- depth distribution net ----
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
        self.enhancer.register_buffer(
            "depth_bins",
            torch.linspace(self.min_depth, self.max_depth, self.bin_num),
            persistent=False
        )

        # ---- token feature -> seed feature (512) ----
        # self.seed_proj = nn.Sequential(
        #     nn.LazyLinear(self.seed_feature_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.seed_feature_dim, self.seed_feature_dim),
        # )
        tok_feat_dim = 128
        self.seed_proj = nn.Sequential(
            nn.Conv2d(tok_feat_dim, self.seed_feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.seed_feature_dim, self.seed_feature_dim, kernel_size=1),
        )
        
        # reuse original heads
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

        # vis
        self.vis_dir = os.path.join('vis', 'c2.1_dbg')
        self.vis_every = 200
        self._vis_iter = 0
        self.vis_token_every = 200        # 每多少 iter 输出一次 token 可视化
        self.vis_token_maxB = 1           # 只可视化 batch 前几个
        self.debug_print_every = 50       # 每多少 iter 打印一次统计
        self.debug_first_only = False     # True: 只输出第一次
        self._debug_has_done = False
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _make_token_uv_grid(Ht, Wt, s, device, dtype):
        """
        token center coords in 448x448 pixel space:
          u = (tu+0.5)*s - 0.5
          v = (tv+0.5)*s - 0.5
        return uv: (Ntok,2)
        """
        tu = torch.arange(Wt, device=device, dtype=dtype)
        tv = torch.arange(Ht, device=device, dtype=dtype)
        u = (tu + 0.5) * s - 0.5
        v = (tv + 0.5) * s - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # (Ht,Wt)
        uv = torch.stack([uu, vv], dim=-1).reshape(-1, 2)  # (Ntok,2)
        return uv

    @staticmethod
    def _backproject_uvz(uv_b_n2, z_b_n1, K_b_33):
        """
        uv_b_n2: (B,N,2) in pixels (448-coord)
        z_b_n1:  (B,N,1) depth in meters
        K_b_33:  (B,3,3) intrinsics for 448
        return xyz: (B,N,3)
        """
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
    def _fps_indices(xyz: torch.Tensor, M: int, start: torch.Tensor = None):
        """
        xyz: (K,3) float tensor on GPU
        returns idx: (M,) long tensor, indices in [0,K)
        Notes: O(K*M) loop, but K ~ 8k, M ~ 1k => ok.
        """
        device = xyz.device
        K = xyz.shape[0]
        if K == 0:
            return torch.zeros((M,), device=device, dtype=torch.long)

        M = min(M, K)
        centroids = torch.empty((M,), device=device, dtype=torch.long)
        dist = torch.full((K,), 1e10, device=device, dtype=xyz.dtype)

        if start is None:
            farthest = torch.randint(0, K, (1,), device=device, dtype=torch.long)
        else:
            farthest = start.view(1).to(device=device, dtype=torch.long).clamp_(0, K - 1)

        for i in range(M):
            centroids[i] = farthest[0]
            centroid = xyz[farthest]  # (1,3)
            d = ((xyz - centroid) ** 2).sum(dim=-1)  # (K,)
            dist = torch.minimum(dist, d)
            farthest = torch.argmax(dist).view(1)

        # 如果需要严格 M（当 K<M 时），在外部补齐
        return centroids

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

        p = _valid(p); g = _valid(g)
        if p.shape[0] == 0 or g.shape[0] == 0:
            return

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0
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

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap="Spectral", title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6,6))
        if vmin is None: vmin = float(np.nanmin(arr2d))
        if vmax is None: vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_points(self, img_448, pts_uv, out_path, radius=1, color=(0,0,255)):
        import cv2
        x = img_448.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = (x.permute(1,2,0).numpy() * 255.0).astype(np.uint8)  # RGB
        x_bgr = x[..., ::-1].copy()

        pts = pts_uv.detach().cpu().numpy()
        H, W = x_bgr.shape[:2]
        for (u,v) in pts:
            uu = int(round(float(u))); vv = int(round(float(v)))
            if 0 <= uu < W and 0 <= vv < H:
                cv2.circle(x_bgr, (uu,vv), radius, color, thickness=-1)
        cv2.imwrite(out_path, x_bgr)
        
    # ----------------------------
    # forward (Stage-1 token seeds)
    # ----------------------------
    def forward(self, end_points: dict):
        img = end_points["img"]   # (B,3,448,448)
        K   = end_points["K"]     # (B,3,3)

        B, _, H, W = img.shape
        assert (H, W) == (448, 448)
        s = 2
        Ht, Wt = H // s, W // s
        Ntok = Ht * Wt
        M = int(self.M_points)

        # respect both flags
        # is_train = bool(self.is_training) and bool(self.training)
        # self.view.is_training = is_train

        # ---------- 1) depth distribution ----------
        depth_map_pred_448, depth_tok_dbg, img_feat, depth_prob_448, depth_logits_448 = \
            self.depth_net(img, return_prob=True, return_tok_prob=False)
        end_points["img_feat_dpt"] = img_feat  # (B,C,448,448)

        # --- token prob from pooled logits (BIP3D-like) ---
        logits_tok = F.avg_pool2d(depth_logits_448, kernel_size=s, stride=s)   # (B,D,224,224)
        prob_tok_map = torch.softmax(logits_tok, dim=1)                        # (B,D,224,224)

        # (B,1,Ntok,D) for depth_prob loss
        end_points["depth_prob_pred"] = prob_tok_map.permute(0, 2, 3, 1).reshape(B, 1, Ntok, self.bin_num).contiguous()
        end_points.pop("depth_prob_logits", None)

        # --- depth_tok from token distribution (consistent) ---
        z_bins = self.depth_net.depth_bins.view(1, self.bin_num, 1, 1).to(prob_tok_map)  # (1,D,1,1)
        depth_tok = (prob_tok_map * z_bins).sum(dim=1, keepdim=True)                      # (B,1,224,224)
        depth_map_pred = F.interpolate(depth_tok, size=(H, W), mode="nearest")            # (B,1,448,448)
        end_points["depth_tok_pred"] = depth_tok
        end_points["depth_map_pred"] = depth_map_pred

        # ---------- 2) token feature + spatial enhancer ----------
        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)
        feat_tok = F.avg_pool2d(img_feat, kernel_size=s, stride=s)  # (B,C,224,224)

        feat_tok_enh = self.enhancer(feat_tok, prob_tok_map, K, stride=s)  # (B,C,224,224)

        # ---------- 3) build token seeds (ALL tokens) ----------
        seed_map = self.seed_proj(feat_tok_enh)                           # (B,Ntok,512)
        seed_features_all = seed_map.view(B, self.seed_feature_dim, -1).contiguous()  # (B,512,Ntok)

        # token-level graspable prediction (reuse GraspableNet)
        end_points = self.graspable(seed_features_all, end_points)
        objectness_score = end_points["objectness_score"]                  # (B,2,Ntok)
        graspness_score  = end_points["graspness_score"].squeeze(1)         # (B,Ntok)

        objectness_pred = torch.argmax(objectness_score, 1)                # (B,Ntok)

        # token validity mask (training can use GT validity if provided)
        if "token_valid_mask" in end_points:
            valid_tok = end_points["token_valid_mask"].bool()              # (B,Ntok)
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        if self.topk_use_objectness:
            mask_tok = valid_tok & (objectness_pred == 1) & (graspness_score > cfgs.graspness_threshold)
        else:
            mask_tok = valid_tok & (graspness_score > cfgs.graspness_threshold)

        # -----------------------------
        # Debug-friendly: bounded graspness for selection
        # -----------------------------
        grasp_raw = graspness_score  # (B,Ntok), raw regression
        grasp_sel = grasp_raw  # 用于选点/可视化（不改 loss）

        mask_obj_pred = valid_tok & (objectness_pred == 1)
        mask_thr_pred = mask_obj_pred & (grasp_sel > cfgs.graspness_threshold)

        # 你当前 mask_tok 等价于 mask_thr_pred（只是用 raw graspness_score），这里建议统一用 grasp_sel
        mask_tok = mask_thr_pred

        # 存下来，后面可视化/打印不要依赖 locals()
        end_points["dbg_grasp_raw"] = grasp_raw.detach()
        end_points["dbg_grasp_sel"] = grasp_sel.detach()
        end_points["dbg_mask_obj"] = mask_obj_pred.detach()
        end_points["dbg_mask_pred"] = mask_thr_pred.detach()
        end_points["dbg_objectness_pred"] = objectness_pred.detach()

        with torch.no_grad():
            end_points["D: PredCand#(thr)"] = mask_thr_pred.float().sum(dim=1).mean()
            end_points["D: PredObj#"] = mask_obj_pred.float().sum(dim=1).mean()
            end_points["D: GraspRaw min"] = grasp_raw.min()
            end_points["D: GraspRaw max"] = grasp_raw.max()
            end_points["D: GraspSel mean"] = grasp_sel.mean()
    
        # ---------- 4) xyz for tokens (Fix-A: use token-center pixel depth) ----------
        # token center pixel (in 448 grid): (yc=2*tv+1, xc=2*tu+1)
        # build center pixel flat indices: (Ntok,)
        tv = torch.arange(Ht, device=img.device, dtype=torch.long)  # [0..223]
        tu = torch.arange(Wt, device=img.device, dtype=torch.long)  # [0..223]
        tvv, tuu = torch.meshgrid(tv, tu, indexing="ij")            # (Ht,Wt)

        yc = (tvv * s + (s // 2)).clamp(0, H - 1)  # s=2 -> +1
        xc = (tuu * s + (s // 2)).clamp(0, W - 1)  # s=2 -> +1
        center_pix_flat = (yc * W + xc).reshape(-1)                 # (Ntok,)

        # expand to batch: (B,Ntok)
        center_pix_flat_b = center_pix_flat.unsqueeze(0).expand(B, -1).contiguous()

        # gather depth at token centers from predicted depth_map_pred (448)
        # ensure depth_map_pred is (B,1,448,448)
        z_center_pred = gather_depth_by_img_idxs(depth_map_pred, center_pix_flat_b)  # (B,Ntok,1)
        z_center_pred = torch.nan_to_num(z_center_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

        # build uv (float) at token center pixels for backproject
        u_center, v_center = img_idxs_to_uv(center_pix_flat_b, W)  # (B,Ntok,1) float

        # xyz from pred (detach for stability, same spirit as C1)
        xyz_tok_pred = backproject_uvz(u_center, v_center, z_center_pred.detach(), K)  # (B,Ntok,3)

        # GT xyz for label matching (optional but recommended for training stability)
        use_gt_xyz = self.is_training and self.use_gt_xyz_for_train and ("gt_depth_m" in end_points)
        if use_gt_xyz:
            gt_depth = end_points["gt_depth_m"]  # (B,448,448) or (B,1,448,448)
            if torch.is_tensor(gt_depth):
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)   # (B,1,448,448)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]
            else:
                gt_depth = torch.as_tensor(gt_depth, device=img.device, dtype=torch.float32)
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]

            z_center_gt = gather_depth_by_img_idxs(gt_depth, center_pix_flat_b)  # (B,Ntok,1)
            z_center_gt = torch.nan_to_num(z_center_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

            xyz_tok_match = backproject_uvz(u_center, v_center, z_center_gt, K)  # (B,Ntok,3)
        else:
            xyz_tok_match = xyz_tok_pred

        # ---------- 5) select Top-M tokens and form xyz_graspable / seed_features_graspable ----------
        seed_features_flipped = seed_features_all.transpose(1, 2).contiguous()  # (B,Ntok,512)

        sel_xyz = []
        sel_feat = []
        sel_idx_list = []
        graspable_num_batch = 0.0
        
        xyz_for_fps = xyz_tok_match
        preselect_mult = 8
        preselect_mult_cov = 16
        M_high = int(round(M * (1.0 - self.converage_ratio)))
        M_cov  = M - M_high

        for b in range(B):
            # --------- pools ----------
            # coverage pool: 所有 obj=1 token
            idx_obj = torch.nonzero(valid_tok[b] & (objectness_pred[b] == 1), as_tuple=False).squeeze(1)

            # quality pool: obj=1 且 grasp高
            idx_hi = torch.nonzero(valid_tok[b] & (objectness_pred[b] == 1) & (grasp_sel[b] > float(cfgs.graspness_threshold)), as_tuple=False).squeeze(1)

            # --------- fallback: 如果 obj 都没有，直接随机 ---------
            if idx_obj.numel() == 0:
                ridx = torch.randint(0, Ntok, (M,), device=img.device)
                sel_idx_list.append(ridx)
                sel_xyz.append(xyz_tok_match[b].index_select(0, ridx).contiguous())  # 你原来用 match 给 head
                sel_feat.append(seed_features_flipped[b].index_select(0, ridx).t().contiguous())
                continue

            # --------- 1) quality FPS ---------
            ridx_parts = []

            if (M_high > 0) and (idx_hi.numel() > 0):
                # topK by score in idx_hi
                scores_hi = grasp_sel[b].index_select(0, idx_hi)
                K1 = min(idx_hi.numel(), preselect_mult * M_high)
                top = torch.topk(scores_hi, k=K1, largest=True).indices
                cand_hi = idx_hi[top]  # (K1,)

                xyz_hi = xyz_for_fps[b].index_select(0, cand_hi)  # (K1,3)

                # start from highest score (in cand_hi)
                start = torch.argmax(grasp_sel[b].index_select(0, cand_hi))
                fps_local = self._fps_indices(xyz_hi, M_high, start=start)
                sel_hi = cand_hi.index_select(0, fps_local)
                ridx_parts.append(sel_hi)
            else:
                # quality pool 不够：退化成从 obj pool 里拿
                pass

            # --------- 2) coverage FPS ---------
            if M_cov > 0:
                # 可选：用更松的 graspness 过滤一下 coverage（防止全是背景边缘）
                idx_cov = torch.nonzero(valid_tok[b] & (objectness_pred[b] == 1), as_tuple=False).squeeze(1)
                if idx_cov.numel() == 0:
                    idx_cov = idx_obj

                # 从 coverage pool 抽 K2 个候选（随机 or topK）
                K2 = min(idx_cov.numel(), preselect_mult_cov * M_cov)
                if idx_cov.numel() > K2:
                    # 随机抽样更利于覆盖（比 topK 更分散）
                    perm = torch.randperm(idx_cov.numel(), device=img.device)[:K2]
                    cand_cov = idx_cov[perm]
                else:
                    cand_cov = idx_cov

                xyz_cov = xyz_for_fps[b].index_select(0, cand_cov)  # (K2,3)

                # start：选一个和已选点尽量远的（如果已有 sel_hi）
                if len(ridx_parts) > 0:
                    sel0 = ridx_parts[0]
                    center = xyz_for_fps[b].index_select(0, sel0).mean(dim=0, keepdim=True)  # (1,3)
                    d = ((xyz_cov - center) ** 2).sum(dim=-1)
                    start = torch.argmax(d)
                else:
                    start = torch.argmax(grasp_sel[b].index_select(0, cand_cov))

                fps_local = self._fps_indices(xyz_cov, M_cov, start=start)
                sel_cov = cand_cov.index_select(0, fps_local)
                ridx_parts.append(sel_cov)

            # --------- merge / pad / trim ---------
            ridx = torch.cat(ridx_parts, dim=0)
            # 去重（可选但推荐）
            ridx = torch.unique(ridx, sorted=False)

            if ridx.numel() < M:
                # 用 obj pool 补齐（随机）
                perm = torch.randint(0, idx_obj.numel(), (M - ridx.numel(),), device=img.device)
                ridx = torch.cat([ridx, idx_obj[perm]], dim=0)
            elif ridx.numel() > M:
                # 保留前 M 个（也可按 grasp_sel 再排序）
                ridx = ridx[:M]

            # --------- store ---------
            sel_idx_list.append(ridx)
            sel_xyz.append(xyz_tok_match[b].index_select(0, ridx).contiguous())  # 你现在用于 supervision/matching 的 xyz
            sel_feat.append(seed_features_flipped[b].index_select(0, ridx).t().contiguous())

        seed_xyz_graspable = torch.stack(sel_xyz, 0).contiguous()              # (B,M,3)
        seed_features_graspable = torch.stack(sel_feat, 0).contiguous()        # (B,512,M)

        end_points["token_sel_idx"] = torch.stack(sel_idx_list, dim=0).contiguous()  # (B,M)
        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = (
            torch.as_tensor(graspable_num_batch, device=img.device, dtype=torch.float32) / float(B)
        ).detach().reshape(())

        # optional: store for debugging
        end_points["token_sel_xyz"] = seed_xyz_graspable

        do_print = (self._vis_iter % self.debug_print_every == 0) or bool(end_points.get("force_vis", False))
        if do_print:
            with torch.no_grad():
                tok_valid = end_points.get("token_valid_mask", None)
                if tok_valid is None:
                    tok_valid = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

                pred_obj = end_points["dbg_objectness_pred"]   # (B,Ntok)
                grasp_raw = end_points["dbg_grasp_raw"]        # (B,Ntok)
                grasp_sel = end_points["dbg_grasp_sel"]        # (B,Ntok)
                mask_obj = end_points["dbg_mask_obj"]          # (B,Ntok)
                mask_thr = end_points["dbg_mask_pred"]         # (B,Ntok)

                msg = (f"[dbg] iter={self._vis_iter} "
                    f"valid={tok_valid.float().sum(1).mean().item():.1f} "
                    f"obj_pred1={mask_obj.float().sum(1).mean().item():.1f} "
                    f"cand_thr={mask_thr.float().sum(1).mean().item():.1f} "
                    f"gr_raw[min,max]=({grasp_raw.min().item():.3f},{grasp_raw.max().item():.3f}) "
                    f"gr_sel[p10,p50,p90]=({torch.quantile(grasp_sel,0.1).item():.3f},"
                    f"{torch.quantile(grasp_sel,0.5).item():.3f},"
                    f"{torch.quantile(grasp_sel,0.9).item():.3f})")

                # 如果有 GT token labels，就输出 GT 分布 + 选点覆盖率
                if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                    gt_obj = end_points["objectness_label_tok"].long()   # (B,Ntok)
                    gt_gra = end_points["graspness_label_tok"].float()   # (B,Ntok)

                    gt_valid = (gt_obj != -1) & tok_valid
                    gt_obj1 = (gt_obj == 1) & gt_valid
                    # 看看 GT graspness 在 object tokens 上是什么分布（帮你设 threshold）
                    if gt_obj1.any():
                        g = gt_gra[gt_obj1]
                        msg += (f" | GT_gra[obj] p50={torch.quantile(g,0.5).item():.3f}"
                                f" p90={torch.quantile(g,0.9).item():.3f}"
                                f" mean={g.mean().item():.3f}")

                    # sel 覆盖 GT 正样本比例
                    if "token_sel_idx" in end_points:
                        sel = end_points["token_sel_idx"]  # (B,M)
                        gt_pos = gt_obj1 & (gt_gra > 0.2)  # 0.2 可调
                        cover = []
                        for b in range(B):
                            cover.append(gt_pos[b].gather(0, sel[b]).float().mean())
                        cover = torch.stack(cover).mean().item()
                        msg += f" | sel_gtpos_ratio={cover:.3f}"

                        # 再看一下：你选中的 token，它们的 GT graspness 均值是多少
                        sel_gtg = []
                        for b in range(B):
                            sel_gtg.append(gt_gra[b].gather(0, sel[b]))
                        sel_gtg = torch.stack(sel_gtg).mean().item()
                        msg += f" | sel_GTg_mean={sel_gtg:.3f}"

                print(msg)

        do_vis = (self.vis_dir is not None) and ((self._vis_iter % self.vis_token_every) == 0 or bool(end_points.get("force_vis", False)))
        if do_vis:
            tag = end_points.get("vis_tag", f"iter{self._vis_iter:06d}")
            out_dir = os.path.join(self.vis_dir, f"tokdbg_{tag}")
            os.makedirs(out_dir, exist_ok=True)

            # 只看 b=0
            b = 0
            pred_obj = end_points["dbg_objectness_pred"][b].view(Ht, Wt).float()
            pred_gra = end_points["dbg_grasp_sel"][b].view(Ht, Wt)          # 0..1
            cand     = end_points["dbg_mask_pred"][b].float().view(Ht, Wt)  # 0/1

            self._save_map_png(pred_obj, os.path.join(out_dir, "pred_objectness.png"), vmin=0, vmax=1, cmap="viridis", title="Pred obj")
            self._save_map_png(pred_gra, os.path.join(out_dir, "pred_graspness.png"), vmin=0, vmax=1, cmap="Spectral", title="Pred graspness(sel)")
            self._save_map_png(cand,     os.path.join(out_dir, "pred_candidate_mask.png"), vmin=0, vmax=1, cmap="gray", title="Pred cand mask")

            # GT maps（如果有）
            if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                gt_obj = end_points["objectness_label_tok"][b].view(Ht, Wt).float()
                gt_gra = end_points["graspness_label_tok"][b].view(Ht, Wt).float()
                self._save_map_png(gt_obj, os.path.join(out_dir, "gt_objectness.png"), vmin=-1, vmax=1, cmap="viridis", title="GT obj (-1 invalid)")
                self._save_map_png(gt_gra, os.path.join(out_dir, "gt_graspness.png"), vmin=0, vmax=1, cmap="Spectral", title="GT graspness")

            # selected tokens map + overlay
            if "token_sel_idx" in end_points:
                sel = end_points["token_sel_idx"][b]  # (M,)
                sel_mask = torch.zeros((Ntok,), device=img.device)
                sel_mask[sel] = 1.0
                self._save_map_png(sel_mask.view(Ht, Wt), os.path.join(out_dir, "selected_tokens.png"),
                                vmin=0, vmax=1, cmap="gray", title="Selected Top-M")

                # overlay on RGB (need u_center/v_center exist in your code)
                pts_uv = torch.cat([
                    u_center[b].squeeze(-1)[sel].unsqueeze(-1),
                    v_center[b].squeeze(-1)[sel].unsqueeze(-1),
                ], dim=-1)
                self._save_overlay_points(img[b], pts_uv, os.path.join(out_dir, "overlay_selected.png"),
                                        radius=1, color=(0,0,255))
        
        # -------- vis: red=pred depth cloud, blue=gt depth cloud --------
        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0)
            do_vis = do_vis or bool(end_points.get("force_vis", False))

            if do_vis and ("gt_depth_m" in end_points) and ("img_idxs" in end_points):
                gt_depth = end_points["gt_depth_m"]  # expected (B,448,448) or (B,1,448,448)

                # shape normalize -> (B,1,448,448)
                if torch.is_tensor(gt_depth):
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.dim() == 4:
                        gt_depth = gt_depth[:, :1]
                else:
                    # rare: numpy slipped in
                    gt_depth = torch.as_tensor(gt_depth, device=img.device, dtype=torch.float32)
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.dim() == 4:
                        gt_depth = gt_depth[:, :1]

                # ensure img_idxs is long
                img_idxs_vis = end_points["img_idxs"].long().clamp(0, H * W - 1)

                # gather z at sampled pixels
                z_pred = gather_depth_by_img_idxs(depth_map_pred, img_idxs_vis)  # (B,N,1)
                z_gt   = gather_depth_by_img_idxs(gt_depth,       img_idxs_vis)  # (B,N,1)

                z_pred = torch.nan_to_num(z_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
                z_gt   = torch.nan_to_num(z_gt,   nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

                # backproject (same u,v)
                u1, v1 = img_idxs_to_uv(img_idxs_vis, W)  # (B,N,1) float
                xyz_pred = backproject_uvz(u1, v1, z_pred, K)  # (B,N,3)  red
                xyz_gt   = backproject_uvz(u1, v1, z_gt,   K)  # (B,N,3)  blue

                # save ply (your helper already colors pred=red, gt=blue)
                self._save_pred_gt_cloud_ply(xyz_pred, xyz_gt, end_points)

                # optional print once per save
                print("[vis] point cloud: red=pred depth, blue=GT depth")

            self._vis_iter += 1

        # ---------- 6) view selection + labels + grouping + head (unchanged) ----------
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        group_features = self.cy_group(seed_xyz_graspable, seed_features_graspable, grasp_top_views_rot)
        end_points = self.grasp_head(group_features, end_points)

        return end_points


class RegressionSpatialEnhancer(nn.Module):
    """
    Regression-based spatial enhancer.
    Input:
      feat_tok : (B,C,Ht,Wt)
      depth_tok: (B,1,Ht,Wt)  continuous metric depth
      K        : (B,3,3)      intrinsics for 448x448
    Output:
      feat_tok_enh: (B,C,Ht,Wt)
    """
    def __init__(
        self,
        tok_feat_dim=128,
        feature_3d_dim=32,
        use_inv_depth=True,
        use_depth_grad=True,
        eps=1e-6,
    ):
        super().__init__()
        self.use_inv_depth = bool(use_inv_depth)
        self.use_depth_grad = bool(use_depth_grad)
        self.eps = float(eps)

        geo_in_dim = 3  # x,y,z
        if self.use_inv_depth:
            geo_in_dim += 1
        if self.use_depth_grad:
            geo_in_dim += 2

        self.geo_proj = nn.Sequential(
            nn.Conv2d(geo_in_dim, feature_3d_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_3d_dim, feature_3d_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(tok_feat_dim + feature_3d_dim, tok_feat_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(tok_feat_dim, tok_feat_dim, kernel_size=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(tok_feat_dim + feature_3d_dim, tok_feat_dim, kernel_size=1),
            nn.Sigmoid()
        )

    @staticmethod
    def _make_token_uv_grid(Ht, Wt, stride, device, dtype):
        tu = torch.arange(Wt, device=device, dtype=dtype)
        tv = torch.arange(Ht, device=device, dtype=dtype)
        u = (tu + 0.5) * stride - 0.5
        v = (tv + 0.5) * stride - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        return uu[None, None], vv[None, None]   # (1,1,Ht,Wt), (1,1,Ht,Wt)

    def forward(self, feat_tok, depth_tok, K, stride=2):
        """
        feat_tok : (B,C,Ht,Wt)
        depth_tok: (B,1,Ht,Wt)
        K       : (B,3,3)
        """
        B, C, Ht, Wt = feat_tok.shape
        device, dtype = feat_tok.device, feat_tok.dtype

        z = torch.nan_to_num(depth_tok, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(self.eps)  # (B,1,Ht,Wt)

        uu, vv = self._make_token_uv_grid(Ht, Wt, stride, device, dtype)
        uu = uu.expand(B, -1, -1, -1)  # (B,1,Ht,Wt)
        vv = vv.expand(B, -1, -1, -1)

        fx = K[:, 0, 0].view(B, 1, 1, 1).to(dtype)
        fy = K[:, 1, 1].view(B, 1, 1, 1).to(dtype)
        cx = K[:, 0, 2].view(B, 1, 1, 1).to(dtype)
        cy = K[:, 1, 2].view(B, 1, 1, 1).to(dtype)

        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z

        geo_list = [x, y, z]

        if self.use_inv_depth:
            inv_z = 1.0 / z.clamp_min(self.eps)
            geo_list.append(inv_z)

        if self.use_depth_grad:
            dzdx = torch.zeros_like(z)
            dzdy = torch.zeros_like(z)
            dzdx[:, :, :, 1:-1] = 0.5 * (z[:, :, :, 2:] - z[:, :, :, :-2])
            dzdy[:, :, 1:-1, :] = 0.5 * (z[:, :, 2:, :] - z[:, :, :-2, :])
            geo_list.extend([dzdx, dzdy])

        geo = torch.cat(geo_list, dim=1)  # (B,Cg,Ht,Wt)
        geo_feat = self.geo_proj(geo)

        fuse_in = torch.cat([feat_tok, geo_feat], dim=1)
        delta = self.fuse(fuse_in)
        gate = self.gate(fuse_in)

        feat_tok_enh = feat_tok + gate * delta
        return feat_tok_enh


class economicgrasp_c2_2(nn.Module):
    """
    RGB-only regression version of c2_1:
      - depth regression supervision at full-map level
      - regression-based spatial enhancer: (feat_tok, depth_tok, K) -> feat_tok_enh
      - token-grid seeds + graspable/objectness on all tokens
      - FPS + coverage-aware token selection
      - reuse original ViewNet / process_grasp_labels / Cylinder grouping / Grasp head
    """
    def __init__(
        self,
        cylinder_radius=0.05,
        seed_feat_dim=512,
        is_training=True,
        depth_stride=2,
        min_depth=0.2,
        max_depth=1.0,
        tok_feat_dim=128,
        feature_3d_dim=32,
        use_inv_depth=False,
        use_depth_grad=False,
        use_gt_xyz_for_train=False,
        topk_use_objectness=True,
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.topk_use_objectness = bool(topk_use_objectness)

        self.seed_feature_dim = int(seed_feat_dim)
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points  = cfgs.m_point
        self.num_view  = cfgs.num_view

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.converage_ratio = 0.3
        self.depth_stride = int(depth_stride)

        # ------------------------------------------------------------------
        # depth regression net
        # 这里直接替换成你 C1 里已经在用的 regression depth net 即可
        # 期望接口:
        #   depth_map_pred_448, img_feat = self.depth_net(img, return_feat=True)
        # 其中:
        #   depth_map_pred_448: (B,1,448,448)
        #   img_feat          : (B,C,448,448), C=tok_feat_dim
        # ------------------------------------------------------------------
        self.depth_net = DINOv2DepthRegressionNet(
            encoder="vitb",
            stride=depth_stride,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            freeze_backbone=True,
        )

        # regression-based spatial enhancer
        self.enhancer = RegressionSpatialEnhancer(
            tok_feat_dim=tok_feat_dim,
            feature_3d_dim=feature_3d_dim,
            use_inv_depth=use_inv_depth,
            use_depth_grad=use_depth_grad,
        )

        # token feature -> seed feature
        self.seed_proj = nn.Sequential(
            nn.Conv2d(tok_feat_dim, self.seed_feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.seed_feature_dim, self.seed_feature_dim, kernel_size=1),
        )

        # reuse original heads
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

        # vis/debug
        self.vis_dir = os.path.join('vis', 'c2_2')
        self.vis_every = 200
        self._vis_iter = 0
        self.vis_token_every = 200
        self.vis_token_maxB = 1
        self.debug_print_every = 50
        self.debug_first_only = False
        self._debug_has_done = False
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

    @staticmethod
    def _make_token_uv_grid(Ht, Wt, s, device, dtype):
        tu = torch.arange(Wt, device=device, dtype=dtype)
        tv = torch.arange(Ht, device=device, dtype=dtype)
        u = (tu + 0.5) * s - 0.5
        v = (tv + 0.5) * s - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack([uu, vv], dim=-1).reshape(-1, 2)
        return uv

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
    def _fps_indices(xyz: torch.Tensor, M: int, start: torch.Tensor = None):
        device = xyz.device
        K = xyz.shape[0]
        if K == 0:
            return torch.zeros((M,), device=device, dtype=torch.long)

        M = min(M, K)
        centroids = torch.empty((M,), device=device, dtype=torch.long)
        dist = torch.full((K,), 1e10, device=device, dtype=xyz.dtype)

        if start is None:
            farthest = torch.randint(0, K, (1,), device=device, dtype=torch.long)
        else:
            farthest = start.view(1).to(device=device, dtype=torch.long).clamp_(0, K - 1)

        for i in range(M):
            centroids[i] = farthest[0]
            centroid = xyz[farthest]
            d = ((xyz - centroid) ** 2).sum(dim=-1)
            dist = torch.minimum(dist, d)
            farthest = torch.argmax(dist).view(1)
        return centroids

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

        p = _valid(p); g = _valid(g)
        if p.shape[0] == 0 or g.shape[0] == 0:
            return

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0
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

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap="Spectral", title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6,6))
        if vmin is None: vmin = float(np.nanmin(arr2d))
        if vmax is None: vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_points(self, img_448, pts_uv, out_path, radius=1, color=(0,0,255)):
        import cv2
        x = img_448.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = (x.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
        x_bgr = x[..., ::-1].copy()

        pts = pts_uv.detach().cpu().numpy()
        H, W = x_bgr.shape[:2]
        for (u,v) in pts:
            uu = int(round(float(u))); vv = int(round(float(v)))
            if 0 <= uu < W and 0 <= vv < H:
                cv2.circle(x_bgr, (uu,vv), radius, color, thickness=-1)
        cv2.imwrite(out_path, x_bgr)

    def forward(self, end_points: dict):
        img = end_points["img"]   # (B,3,448,448)
        K   = end_points["K"]     # (B,3,3)

        B, _, H, W = img.shape
        assert (H, W) == (448, 448)
        s = self.depth_stride
        Ht, Wt = H // s, W // s
        Ntok = Ht * Wt
        M = int(self.M_points)

        # ---------- 1) depth regression ----------
        depth_map_pred_448, depth_tok_dbg, img_feat = self.depth_net(img)
        # depth_map_pred_448: (B,1,448,448)
        # depth_tok_dbg     : (B,1,224,224) 仅用于 debug
        # img_feat          : (B,128,448,448)

        depth_map_pred = torch.nan_to_num(
            depth_map_pred_448, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=self.min_depth, max=self.max_depth)

        end_points["img_feat_dpt"] = img_feat
        end_points["depth_map_pred"] = depth_map_pred
        end_points["depth_tok_dbg"] = depth_tok_dbg

        # 用 avg_pool 得 token depth，和 feat_tok 的 pooling 一致
        depth_tok = F.avg_pool2d(depth_map_pred, kernel_size=s, stride=s)   # (B,1,224,224)
        end_points["depth_tok_pred"] = depth_tok

        # ---------- 2) token feature + regression spatial enhancer ----------
        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

        feat_tok = F.avg_pool2d(img_feat, kernel_size=s, stride=s)  # (B,128,224,224)
        feat_tok_enh = self.enhancer(feat_tok, depth_tok.detach(), K, stride=s)

        # ------------------------------------------------------------
        # 3) build token seeds
        # ------------------------------------------------------------
        seed_map = self.seed_proj(feat_tok_enh)  # (B,512,Ht,Wt)
        seed_features_all = seed_map.view(B, self.seed_feature_dim, -1).contiguous()  # (B,512,Ntok)

        end_points = self.graspable(seed_features_all, end_points)
        objectness_score = end_points["objectness_score"]            # (B,2,Ntok)
        graspness_score  = end_points["graspness_score"].squeeze(1) # (B,Ntok)
        objectness_pred  = torch.argmax(objectness_score, 1)         # (B,Ntok)

        if "token_valid_mask" in end_points:
            valid_tok = end_points["token_valid_mask"].bool()
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        # debug-friendly
        grasp_raw = graspness_score
        grasp_sel = grasp_raw
        mask_obj_pred = valid_tok & (objectness_pred == 1)
        mask_thr_pred = mask_obj_pred & (grasp_sel > cfgs.graspness_threshold)
        mask_tok = mask_thr_pred

        end_points["dbg_grasp_raw"] = grasp_raw.detach()
        end_points["dbg_grasp_sel"] = grasp_sel.detach()
        end_points["dbg_mask_obj"] = mask_obj_pred.detach()
        end_points["dbg_mask_pred"] = mask_thr_pred.detach()
        end_points["dbg_objectness_pred"] = objectness_pred.detach()

        with torch.no_grad():
            end_points["D: PredCand#(thr)"] = mask_thr_pred.float().sum(dim=1).mean()
            end_points["D: PredObj#"] = mask_obj_pred.float().sum(dim=1).mean()
            end_points["D: GraspRaw min"] = grasp_raw.min()
            end_points["D: GraspRaw max"] = grasp_raw.max()
            end_points["D: GraspSel mean"] = grasp_sel.mean()

        # ------------------------------------------------------------
        # 4) xyz for tokens (same as c2_1)
        # ------------------------------------------------------------
        tv = torch.arange(Ht, device=img.device, dtype=torch.long)
        tu = torch.arange(Wt, device=img.device, dtype=torch.long)
        tvv, tuu = torch.meshgrid(tv, tu, indexing="ij")

        yc = (tvv * s + (s // 2)).clamp(0, H - 1)
        xc = (tuu * s + (s // 2)).clamp(0, W - 1)
        center_pix_flat = (yc * W + xc).reshape(-1)
        center_pix_flat_b = center_pix_flat.unsqueeze(0).expand(B, -1).contiguous()

        z_center_pred = gather_depth_by_img_idxs(depth_map_pred, center_pix_flat_b)
        z_center_pred = torch.nan_to_num(z_center_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

        u_center, v_center = img_idxs_to_uv(center_pix_flat_b, W)
        xyz_tok_pred = backproject_uvz(u_center, v_center, z_center_pred.detach(), K)

        use_gt_xyz = self.is_training and self.use_gt_xyz_for_train and ("gt_depth_m" in end_points)
        if use_gt_xyz:
            gt_depth = end_points["gt_depth_m"]
            if torch.is_tensor(gt_depth):
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]
            else:
                gt_depth = torch.as_tensor(gt_depth, device=img.device, dtype=torch.float32)
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]

            z_center_gt = gather_depth_by_img_idxs(gt_depth, center_pix_flat_b)
            z_center_gt = torch.nan_to_num(z_center_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            xyz_tok_match = backproject_uvz(u_center, v_center, z_center_gt, K)
        else:
            xyz_tok_match = xyz_tok_pred

        # ------------------------------------------------------------
        # 5) select Top-M tokens (same FPS + coverage logic)
        # ------------------------------------------------------------
        seed_features_flipped = seed_features_all.transpose(1, 2).contiguous()  # (B,Ntok,512)

        sel_xyz = []
        sel_feat = []
        sel_idx_list = []
        graspable_num_batch = 0.0

        xyz_for_fps = xyz_tok_match
        preselect_mult = 8
        preselect_mult_cov = 16
        M_high = int(round(M * (1.0 - self.converage_ratio)))
        M_cov  = M - M_high

        for b in range(B):
            idx_obj = torch.nonzero(valid_tok[b] & (objectness_pred[b] == 1), as_tuple=False).squeeze(1)
            idx_hi = torch.nonzero(valid_tok[b] & (objectness_pred[b] == 1) &
                                   (grasp_sel[b] > float(cfgs.graspness_threshold)),
                                   as_tuple=False).squeeze(1)

            if idx_obj.numel() == 0:
                ridx = torch.randint(0, Ntok, (M,), device=img.device)
                sel_idx_list.append(ridx)
                sel_xyz.append(xyz_tok_match[b].index_select(0, ridx).contiguous())
                sel_feat.append(seed_features_flipped[b].index_select(0, ridx).t().contiguous())
                continue

            ridx_parts = []

            # quality FPS
            if (M_high > 0) and (idx_hi.numel() > 0):
                scores_hi = grasp_sel[b].index_select(0, idx_hi)
                K1 = min(idx_hi.numel(), preselect_mult * M_high)
                top = torch.topk(scores_hi, k=K1, largest=True).indices
                cand_hi = idx_hi[top]

                xyz_hi = xyz_for_fps[b].index_select(0, cand_hi)
                start = torch.argmax(grasp_sel[b].index_select(0, cand_hi))
                fps_local = self._fps_indices(xyz_hi, M_high, start=start)
                sel_hi = cand_hi.index_select(0, fps_local)
                ridx_parts.append(sel_hi)

            # coverage FPS
            if M_cov > 0:
                idx_cov = torch.nonzero(valid_tok[b] & (objectness_pred[b] == 1),
                                        as_tuple=False).squeeze(1)
                if idx_cov.numel() == 0:
                    idx_cov = idx_obj

                K2 = min(idx_cov.numel(), preselect_mult_cov * M_cov)
                if idx_cov.numel() > K2:
                    perm = torch.randperm(idx_cov.numel(), device=img.device)[:K2]
                    cand_cov = idx_cov[perm]
                else:
                    cand_cov = idx_cov

                xyz_cov = xyz_for_fps[b].index_select(0, cand_cov)

                if len(ridx_parts) > 0:
                    sel0 = ridx_parts[0]
                    center = xyz_for_fps[b].index_select(0, sel0).mean(dim=0, keepdim=True)
                    d = ((xyz_cov - center) ** 2).sum(dim=-1)
                    start = torch.argmax(d)
                else:
                    start = torch.argmax(grasp_sel[b].index_select(0, cand_cov))

                fps_local = self._fps_indices(xyz_cov, M_cov, start=start)
                sel_cov = cand_cov.index_select(0, fps_local)
                ridx_parts.append(sel_cov)

            ridx = torch.cat(ridx_parts, dim=0)
            ridx = torch.unique(ridx, sorted=False)

            if ridx.numel() < M:
                perm = torch.randint(0, idx_obj.numel(), (M - ridx.numel(),), device=img.device)
                ridx = torch.cat([ridx, idx_obj[perm]], dim=0)
            elif ridx.numel() > M:
                ridx = ridx[:M]

            sel_idx_list.append(ridx)
            sel_xyz.append(xyz_tok_match[b].index_select(0, ridx).contiguous())
            sel_feat.append(seed_features_flipped[b].index_select(0, ridx).t().contiguous())

        seed_xyz_graspable = torch.stack(sel_xyz, 0).contiguous()
        seed_features_graspable = torch.stack(sel_feat, 0).contiguous()

        end_points["token_sel_idx"] = torch.stack(sel_idx_list, dim=0).contiguous()
        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = (
            torch.as_tensor(graspable_num_batch, device=img.device, dtype=torch.float32) / float(B)
        ).detach().reshape(())
        end_points["token_sel_xyz"] = seed_xyz_graspable

        # ------------------------------------------------------------
        # 5.5) debug print / vis
        # ------------------------------------------------------------
        do_print = (self._vis_iter % self.debug_print_every == 0) or bool(end_points.get("force_vis", False))
        if do_print:
            with torch.no_grad():
                tok_valid = end_points.get("token_valid_mask", None)
                if tok_valid is None:
                    tok_valid = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

                pred_obj = end_points["dbg_objectness_pred"]
                grasp_raw = end_points["dbg_grasp_raw"]
                grasp_sel = end_points["dbg_grasp_sel"]
                mask_obj = end_points["dbg_mask_obj"]
                mask_thr = end_points["dbg_mask_pred"]

                msg = (f"[dbg] iter={self._vis_iter} "
                       f"valid={tok_valid.float().sum(1).mean().item():.1f} "
                       f"obj_pred1={mask_obj.float().sum(1).mean().item():.1f} "
                       f"cand_thr={mask_thr.float().sum(1).mean().item():.1f} "
                       f"gr_raw[min,max]=({grasp_raw.min().item():.3f},{grasp_raw.max().item():.3f}) "
                       f"gr_sel[p10,p50,p90]=({torch.quantile(grasp_sel,0.1).item():.3f},"
                       f"{torch.quantile(grasp_sel,0.5).item():.3f},"
                       f"{torch.quantile(grasp_sel,0.9).item():.3f})")

                if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                    gt_obj = end_points["objectness_label_tok"].long()
                    gt_gra = end_points["graspness_label_tok"].float()

                    gt_valid = (gt_obj != -1) & tok_valid
                    gt_obj1 = (gt_obj == 1) & gt_valid

                    if gt_obj1.any():
                        g = gt_gra[gt_obj1]
                        msg += (f" | GT_gra[obj] p50={torch.quantile(g,0.5).item():.3f}"
                                f" p90={torch.quantile(g,0.9).item():.3f}"
                                f" mean={g.mean().item():.3f}")

                    if "token_sel_idx" in end_points:
                        sel = end_points["token_sel_idx"]
                        gt_pos = gt_obj1 & (gt_gra > 0.2)
                        cover = []
                        for bb in range(B):
                            cover.append(gt_pos[bb].gather(0, sel[bb]).float().mean())
                        cover = torch.stack(cover).mean().item()
                        msg += f" | sel_gtpos_ratio={cover:.3f}"

                        sel_gtg = []
                        for bb in range(B):
                            sel_gtg.append(gt_gra[bb].gather(0, sel[bb]))
                        sel_gtg = torch.stack(sel_gtg).mean().item()
                        msg += f" | sel_GTg_mean={sel_gtg:.3f}"

                print(msg)

        do_tok_vis = (self.vis_dir is not None) and ((self._vis_iter % self.vis_token_every) == 0 or bool(end_points.get("force_vis", False)))
        if do_tok_vis:
            tag = end_points.get("vis_tag", f"iter{self._vis_iter:06d}")
            out_dir = os.path.join(self.vis_dir, f"tokdbg_{tag}")
            os.makedirs(out_dir, exist_ok=True)

            b = 0
            pred_obj = end_points["dbg_objectness_pred"][b].view(Ht, Wt).float()
            pred_gra = end_points["dbg_grasp_sel"][b].view(Ht, Wt)
            cand     = end_points["dbg_mask_pred"][b].float().view(Ht, Wt)

            self._save_map_png(pred_obj, os.path.join(out_dir, "pred_objectness.png"),
                               vmin=0, vmax=1, cmap="viridis", title="Pred obj")
            self._save_map_png(pred_gra, os.path.join(out_dir, "pred_graspness.png"),
                               vmin=0, vmax=1, cmap="Spectral", title="Pred graspness")
            self._save_map_png(cand, os.path.join(out_dir, "pred_candidate_mask.png"),
                               vmin=0, vmax=1, cmap="gray", title="Pred cand mask")

            if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                gt_obj = end_points["objectness_label_tok"][b].view(Ht, Wt).float()
                gt_gra = end_points["graspness_label_tok"][b].view(Ht, Wt).float()
                self._save_map_png(gt_obj, os.path.join(out_dir, "gt_objectness.png"),
                                   vmin=-1, vmax=1, cmap="viridis", title="GT obj (-1 invalid)")
                self._save_map_png(gt_gra, os.path.join(out_dir, "gt_graspness.png"),
                                   vmin=0, vmax=1, cmap="Spectral", title="GT graspness")

            if "token_sel_idx" in end_points:
                sel = end_points["token_sel_idx"][b]
                sel_mask = torch.zeros((Ntok,), device=img.device)
                sel_mask[sel] = 1.0
                self._save_map_png(sel_mask.view(Ht, Wt), os.path.join(out_dir, "selected_tokens.png"),
                                   vmin=0, vmax=1, cmap="gray", title="Selected Top-M")

                pts_uv = torch.cat([
                    u_center[b].squeeze(-1)[sel].unsqueeze(-1),
                    v_center[b].squeeze(-1)[sel].unsqueeze(-1),
                ], dim=-1)
                self._save_overlay_points(img[b], pts_uv, os.path.join(out_dir, "overlay_selected.png"),
                                          radius=1, color=(0,0,255))

        # point cloud vis
        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0) or bool(end_points.get("force_vis", False))
            if do_vis and ("gt_depth_m" in end_points) and ("img_idxs" in end_points):
                gt_depth = end_points["gt_depth_m"]
                if torch.is_tensor(gt_depth):
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.dim() == 4:
                        gt_depth = gt_depth[:, :1]
                else:
                    gt_depth = torch.as_tensor(gt_depth, device=img.device, dtype=torch.float32)
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.dim() == 4:
                        gt_depth = gt_depth[:, :1]

                img_idxs_vis = end_points["img_idxs"].long().clamp(0, H * W - 1)
                z_pred = gather_depth_by_img_idxs(depth_map_pred, img_idxs_vis)
                z_gt   = gather_depth_by_img_idxs(gt_depth, img_idxs_vis)

                z_pred = torch.nan_to_num(z_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
                z_gt   = torch.nan_to_num(z_gt,   nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

                u1, v1 = img_idxs_to_uv(img_idxs_vis, W)
                xyz_pred = backproject_uvz(u1, v1, z_pred, K)
                xyz_gt   = backproject_uvz(u1, v1, z_gt,   K)
                self._save_pred_gt_cloud_ply(xyz_pred, xyz_gt, end_points)
                print("[vis] point cloud: red=pred depth, blue=GT depth")

            self._vis_iter += 1

        # ------------------------------------------------------------
        # 6) view + labels + grouping + head
        # ------------------------------------------------------------
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        group_features = self.cy_group(seed_xyz_graspable, seed_features_graspable, grasp_top_views_rot)
        end_points = self.grasp_head(group_features, end_points)

        return end_points
    

class TokGraspableHead2D(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.head = nn.Conv2d(in_dim, 3, kernel_size=1)

    def forward(self, feat_grid, end_points):
        """
        feat_grid: (B,C,H,W)
        output:
          objectness_score: (B,2,HW)
          graspness_score : (B,1,HW)
        """
        x = self.head(feat_grid)  # (B,3,H,W)
        B, _, H, W = x.shape
        x = x.view(B, 3, H * W)
        end_points["objectness_score"] = x[:, :2, :]
        end_points["graspness_score"] = x[:, 2:3, :]
        return end_points
    
    
class economicgrasp_c2_3(nn.Module):
    """
    C2.3:
      - full-resolution alignment (stride=1)
      - no avg_pool on img_feat / depth_map_pred
      - enhancer on 448x448
      - lightweight 2D graspable head on all pixels
      - project only selected Top-M tokens to 512
      - reuse ViewNet / process_grasp_labels / Cylinder grouping / Grasp head
    """
    def __init__(
        self,
        cylinder_radius=0.05,
        seed_feat_dim=512,
        is_training=True,
        min_depth=0.2,
        max_depth=1.0,
        tok_feat_dim=128,
        feature_3d_dim=32,
        use_inv_depth=True,
        use_depth_grad=True,
        use_gt_xyz_for_train=False,
        topk_use_objectness=True,
        detach_depth_for_enhancer=True,
        vis_dir='c2.3',
        vis_every=1000
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.topk_use_objectness = bool(topk_use_objectness)
        self.detach_depth_for_enhancer = bool(detach_depth_for_enhancer)

        self.seed_feature_dim = int(seed_feat_dim)
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points  = cfgs.m_point
        self.num_view  = cfgs.num_view

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.converage_ratio = 0.3

        # depth regression net
        self.depth_net = DINOv2DepthRegressionNet(
            encoder="vitb",
            stride=1,   # 这里固定 1，只是 debug tok 用不到了
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            freeze_backbone=True,
        )

        # full-res regression spatial enhancer
        self.enhancer = RegressionSpatialEnhancer(
            tok_feat_dim=tok_feat_dim,
            feature_3d_dim=feature_3d_dim,
            use_inv_depth=use_inv_depth,
            use_depth_grad=use_depth_grad,
        )

        # full-res lightweight token head
        self.graspable_2d = TokGraspableHead2D(in_dim=tok_feat_dim)

        # selected tokens -> 512
        self.sel_proj = nn.Sequential(
            nn.Linear(tok_feat_dim, self.seed_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.seed_feature_dim, self.seed_feature_dim),
        )

        # reuse original heads
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

        # vis/debug (可直接沿用你 c2_2 的 helper)
        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self._vis_iter = 0
        self.vis_token_every = vis_every
        self.vis_token_maxB = 1
        self.debug_print_every = 50
        self.debug_first_only = False
        self._debug_has_done = False
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

    # ---------- reuse these helpers from c2_2 ----------
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
    def _fps_indices(xyz: torch.Tensor, M: int, start: torch.Tensor = None):
        device = xyz.device
        K = xyz.shape[0]
        if K == 0:
            return torch.zeros((M,), device=device, dtype=torch.long)

        M = min(M, K)
        centroids = torch.empty((M,), device=device, dtype=torch.long)
        dist = torch.full((K,), 1e10, device=device, dtype=xyz.dtype)

        if start is None:
            farthest = torch.randint(0, K, (1,), device=device, dtype=torch.long)
        else:
            farthest = start.view(1).to(device=device, dtype=torch.long).clamp_(0, K - 1)

        for i in range(M):
            centroids[i] = farthest[0]
            centroid = xyz[farthest]
            d = ((xyz - centroid) ** 2).sum(dim=-1)
            dist = torch.minimum(dist, d)
            farthest = torch.argmax(dist).view(1)
        return centroids

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap="Spectral", title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6, 6))
        if vmin is None: vmin = float(np.nanmin(arr2d))
        if vmax is None: vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_points(self, img_448, pts_uv, out_path, radius=1, color=(0,0,255)):
        import cv2
        x = img_448.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = (x.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
        x_bgr = x[..., ::-1].copy()

        pts = pts_uv.detach().cpu().numpy()
        H, W = x_bgr.shape[:2]
        for (u,v) in pts:
            uu = int(round(float(u))); vv = int(round(float(v)))
            if 0 <= uu < W and 0 <= vv < H:
                cv2.circle(x_bgr, (uu,vv), radius, color, thickness=-1)
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

        p = _valid(p); g = _valid(g)
        if p.shape[0] == 0 or g.shape[0] == 0:
            return

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0
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

    def forward(self, end_points: dict):
        img = end_points["img"]   # (B,3,448,448)
        K   = end_points["K"]     # (B,3,3)

        B, _, H, W = img.shape
        assert (H, W) == (448, 448)
        Ntok = H * W
        M = int(self.M_points)

        # ============================================================
        # 1) depth regression
        # ============================================================
        # depth_map_pred_448: (B,1,448,448)
        # depth_tok_dbg     : (B,1,448,448) if stride=1
        # img_feat          : (B,C,448,448), C=128 for vitb
        depth_map_pred_448, depth_tok_dbg, img_feat = self.depth_net(img)

        depth_map_pred = torch.nan_to_num(
            depth_map_pred_448, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=self.min_depth, max=self.max_depth)

        end_points["depth_map_pred"] = depth_map_pred
        end_points["depth_tok_dbg"] = depth_tok_dbg
        end_points["depth_tok_pred"] = depth_map_pred   # full-res for C2.3
        end_points["img_feat_dpt"] = img_feat

        # ============================================================
        # 2) full-resolution enhancer (no avg_pool)
        # ============================================================
        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

        # 可选：像 c2_2_detach 那样切断 grasp->depth 的直接梯度
        if getattr(self, "detach_depth_for_enhancer", True):
            depth_for_enh = depth_map_pred.detach()
        else:
            depth_for_enh = depth_map_pred

        # feat_grid_enh: (B,C,H,W)
        feat_grid_enh = self.enhancer(img_feat, depth_for_enh, K, stride=1)

        # ============================================================
        # 3) full-res token graspable head
        # ============================================================
        end_points = self.graspable_2d(feat_grid_enh, end_points)
        objectness_score = end_points["objectness_score"]            # (B,2,Ntok)
        graspness_score  = end_points["graspness_score"].squeeze(1) # (B,Ntok)
        objectness_pred  = torch.argmax(objectness_score, dim=1)    # (B,Ntok)

        # token_valid_mask should be full-res: (B,Ntok)
        if "token_valid_mask" in end_points:
            valid_tok = end_points["token_valid_mask"].bool()
            if valid_tok.shape[1] != Ntok:
                raise ValueError(
                    f"C2.3 expects token_valid_mask with Ntok={Ntok}, got {tuple(valid_tok.shape)}. "
                    f"Please change dataloader token labels to full-resolution 448x448 flatten."
                )
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        # 用于筛选/可视化的 graspness，建议限制到 [0,1]
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

        # ============================================================
        # 4) xyz for all full-res pixels
        # ============================================================
        # flat_all: (B,Ntok)
        flat_all = torch.arange(H * W, device=img.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()

        # full-res pixel (u,v) from flat index
        # x = idx % W, y = idx // W
        u_all = (flat_all % W).float()   # (B,Ntok)
        v_all = (flat_all // W).float()  # (B,Ntok)
        uv_all = torch.stack([u_all, v_all], dim=-1)  # (B,Ntok,2)

        # pred z for all pixels
        z_all_pred = depth_map_pred.view(B, -1, 1).contiguous()  # (B,Ntok,1)
        z_all_pred = torch.nan_to_num(z_all_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

        # xyz from pred
        xyz_all_pred = self._backproject_uvz(uv_all, z_all_pred.detach(), K)  # (B,Ntok,3)

        # GT xyz for label matching if enabled
        use_gt_xyz = self.is_training and self.use_gt_xyz_for_train and ("gt_depth_m" in end_points)
        if use_gt_xyz:
            gt_depth = end_points["gt_depth_m"]
            if torch.is_tensor(gt_depth):
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)   # (B,1,H,W)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]
            else:
                gt_depth = torch.as_tensor(gt_depth, device=img.device, dtype=torch.float32)
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]

            z_all_gt = gt_depth.view(B, -1, 1).contiguous()  # (B,Ntok,1)
            z_all_gt = torch.nan_to_num(z_all_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            xyz_all_match = self._backproject_uvz(uv_all, z_all_gt, K)
        else:
            xyz_all_match = xyz_all_pred

        # ============================================================
        # 5) Top-M selection with FPS + coverage
        # ============================================================
        # feat_flat: (B,Ntok,C)
        feat_flat = feat_grid_enh.view(B, feat_grid_enh.shape[1], -1).transpose(1, 2).contiguous()

        sel_xyz = []
        sel_feat = []
        sel_idx_list = []

        graspable_num_batch = 0.0

        # --------------------------------------------------------
        # C1-style sampler for C2.3
        # candidate = objectness & graspness
        # selection = one-shot FPS on 3D xyz of all graspable candidates
        # --------------------------------------------------------
        for b in range(B):
            idx_graspable = torch.nonzero(
                valid_tok[b]
                & (objectness_pred[b] == 1)
                & (grasp_sel[b] > float(cfgs.graspness_threshold)),
                as_tuple=False
            ).squeeze(1)

            graspable_num_batch += idx_graspable.numel()

            # fallback: no graspable candidates
            if idx_graspable.numel() == 0:
                ridx = torch.randint(0, Ntok, (M,), device=img.device)
                sel_idx_list.append(ridx)

                cur_xyz = xyz_all_match[b].index_select(0, ridx).contiguous()
                cur_feat_low = feat_flat[b].index_select(0, ridx).contiguous()
                cur_feat = self.sel_proj(cur_feat_low).transpose(0, 1).contiguous()

                sel_xyz.append(cur_xyz)
                sel_feat.append(cur_feat)
                continue

            # not enough candidates: repeat sampling like C1
            if idx_graspable.numel() < M:
                rep = torch.randint(0, idx_graspable.numel(), (M,), device=img.device)
                ridx = idx_graspable[rep]
                sel_idx_list.append(ridx)

                cur_xyz = xyz_all_match[b].index_select(0, ridx).contiguous()
                cur_feat_low = feat_flat[b].index_select(0, ridx).contiguous()
                cur_feat = self.sel_proj(cur_feat_low).transpose(0, 1).contiguous()

                sel_xyz.append(cur_xyz)
                sel_feat.append(cur_feat)
                continue

            # enough candidates: one-shot FPS in 3D
            xyz_in = xyz_all_match[b].index_select(0, idx_graspable).unsqueeze(0).contiguous()  # (1,Ng,3)
            fps_idxs = furthest_point_sample(xyz_in, M).to(torch.int32).contiguous()             # (1,M)

            cur_xyz = gather_operation(
                xyz_in.transpose(1, 2).contiguous(), fps_idxs
            ).transpose(1, 2).squeeze(0).contiguous()  # (M,3)

            feat_in = feat_flat[b].index_select(0, idx_graspable).contiguous()  # (Ng,C)
            cur_feat_low = gather_operation(
                feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs
            ).squeeze(0).transpose(0, 1).contiguous()  # (M,C)

            cur_feat = self.sel_proj(cur_feat_low).transpose(0, 1).contiguous()  # (512,M)

            # convert local fps index back to original token index
            ridx = idx_graspable.index_select(0, fps_idxs.squeeze(0).long())
            sel_idx_list.append(ridx)

            sel_xyz.append(cur_xyz)
            sel_feat.append(cur_feat)
    
        seed_xyz_graspable = torch.stack(sel_xyz, 0).contiguous()           # (B,M,3)
        seed_features_graspable = torch.stack(sel_feat, 0).contiguous()     # (B,512,M)

        end_points["token_sel_idx"] = torch.stack(sel_idx_list, dim=0).contiguous()  # (B,M)
        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["token_sel_xyz"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = (
            torch.as_tensor(graspable_num_batch, device=img.device, dtype=torch.float32) / float(B)
        ).detach().reshape(())

        # ============================================================
        # 5.5) debug print
        # ============================================================
        do_print = (self._vis_iter % self.debug_print_every == 0) or bool(end_points.get("force_vis", False))
        if do_print:
            with torch.no_grad():
                tok_valid = valid_tok

                pred_obj = end_points["dbg_objectness_pred"]
                grasp_raw_ = end_points["dbg_grasp_raw"]
                grasp_sel_ = end_points["dbg_grasp_sel"]
                mask_obj = end_points["dbg_mask_obj"]
                mask_thr = end_points["dbg_mask_pred"]

                msg = (
                    f"[dbg] iter={self._vis_iter} "
                    f"valid={tok_valid.float().sum(1).mean().item():.1f} "
                    f"obj_pred1={mask_obj.float().sum(1).mean().item():.1f} "
                    f"cand_thr={mask_thr.float().sum(1).mean().item():.1f} "
                    f"gr_raw[min,max]=({grasp_raw_.min().item():.3f},{grasp_raw_.max().item():.3f}) "
                    f"gr_sel[p10,p50,p90]=("
                    f"{torch.quantile(grasp_sel_, 0.1).item():.3f},"
                    f"{torch.quantile(grasp_sel_, 0.5).item():.3f},"
                    f"{torch.quantile(grasp_sel_, 0.9).item():.3f})"
                )

                if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                    gt_obj = end_points["objectness_label_tok"].long()
                    gt_gra = end_points["graspness_label_tok"].float()

                    if gt_obj.shape[1] == Ntok and gt_gra.shape[1] == Ntok:
                        gt_valid = (gt_obj != -1) & tok_valid
                        gt_obj1 = (gt_obj == 1) & gt_valid

                        if gt_obj1.any():
                            g = gt_gra[gt_obj1]
                            msg += (
                                f" | GT_gra[obj] p50={torch.quantile(g,0.5).item():.3f}"
                                f" p90={torch.quantile(g,0.9).item():.3f}"
                                f" mean={g.mean().item():.3f}"
                            )

                        sel = end_points["token_sel_idx"]  # (B,M)
                        gt_pos = gt_obj1 & (gt_gra > 0.2)
                        cover = []
                        sel_gtg = []
                        for bb in range(B):
                            cover.append(gt_pos[bb].gather(0, sel[bb]).float().mean())
                            sel_gtg.append(gt_gra[bb].gather(0, sel[bb]).mean())
                        msg += f" | sel_gtpos_ratio={torch.stack(cover).mean().item():.3f}"
                        msg += f" | sel_GTg_mean={torch.stack(sel_gtg).mean().item():.3f}"
                    else:
                        msg += f" | WARNING: full-res GT token labels expected Ntok={Ntok}, got obj={tuple(gt_obj.shape)}, gra={tuple(gt_gra.shape)}"

                print(msg)

        # ============================================================
        # 5.6) token vis
        # ============================================================
        do_tok_vis = (self.vis_dir is not None) and (
            (self._vis_iter % self.vis_token_every) == 0 or bool(end_points.get("force_vis", False))
        )
        if do_tok_vis:
            tag = end_points.get("vis_tag", f"iter{self._vis_iter:06d}")
            out_dir = os.path.join(self.vis_dir, f"tokdbg_{tag}")
            os.makedirs(out_dir, exist_ok=True)

            b = 0
            pred_obj_map = end_points["dbg_objectness_pred"][b].view(H, W).float()
            pred_gra_map = end_points["dbg_grasp_sel"][b].view(H, W)
            cand_map     = end_points["dbg_mask_pred"][b].float().view(H, W)

            self._save_map_png(pred_obj_map, os.path.join(out_dir, "pred_objectness.png"),
                            vmin=0, vmax=1, cmap="viridis", title="Pred obj")
            self._save_map_png(pred_gra_map, os.path.join(out_dir, "pred_graspness.png"),
                            vmin=0, vmax=1, cmap="Spectral", title="Pred graspness")
            self._save_map_png(cand_map, os.path.join(out_dir, "pred_candidate_mask.png"),
                            vmin=0, vmax=1, cmap="gray", title="Pred cand mask")

            if ("objectness_label_tok" in end_points) and ("graspness_label_tok" in end_points):
                gt_obj = end_points["objectness_label_tok"]
                gt_gra = end_points["graspness_label_tok"]
                if gt_obj.shape[1] == Ntok and gt_gra.shape[1] == Ntok:
                    gt_obj_map = gt_obj[b].view(H, W).float()
                    gt_gra_map = gt_gra[b].view(H, W).float()
                    self._save_map_png(gt_obj_map, os.path.join(out_dir, "gt_objectness.png"),
                                    vmin=-1, vmax=1, cmap="viridis", title="GT obj (-1 invalid)")
                    self._save_map_png(gt_gra_map, os.path.join(out_dir, "gt_graspness.png"),
                                    vmin=0, vmax=1, cmap="Spectral", title="GT graspness")

            if "token_sel_idx" in end_points:
                sel = end_points["token_sel_idx"][b]  # (M,)
                sel_mask = torch.zeros((Ntok,), device=img.device)
                sel_mask[sel] = 1.0
                self._save_map_png(sel_mask.view(H, W), os.path.join(out_dir, "selected_tokens.png"),
                                vmin=0, vmax=1, cmap="gray", title="Selected Top-M")

                xs = (sel % W).float()
                ys = (sel // W).float()
                pts_uv = torch.stack([xs, ys], dim=-1)  # (M,2)
                self._save_overlay_points(img[b], pts_uv, os.path.join(out_dir, "overlay_selected.png"),
                                        radius=1, color=(0,0,255))

        # ============================================================
        # 5.7) point cloud vis on sampled img_idxs
        # ============================================================
        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0) or bool(end_points.get("force_vis", False))
            if do_vis and ("gt_depth_m" in end_points) and ("img_idxs" in end_points):
                gt_depth = end_points["gt_depth_m"]
                if torch.is_tensor(gt_depth):
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.dim() == 4:
                        gt_depth = gt_depth[:, :1]
                else:
                    gt_depth = torch.as_tensor(gt_depth, device=img.device, dtype=torch.float32)
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.dim() == 4:
                        gt_depth = gt_depth[:, :1]

                img_idxs_vis = end_points["img_idxs"].long().clamp(0, H * W - 1)  # (B,Ns)
                z_pred = gather_depth_by_img_idxs(depth_map_pred, img_idxs_vis)    # (B,Ns,1)
                z_gt   = gather_depth_by_img_idxs(gt_depth, img_idxs_vis)          # (B,Ns,1)

                z_pred = torch.nan_to_num(z_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
                z_gt   = torch.nan_to_num(z_gt,   nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

                u_vis = (img_idxs_vis % W).float()
                v_vis = (img_idxs_vis // W).float()
                uv_vis = torch.stack([u_vis, v_vis], dim=-1)  # (B,Ns,2)

                xyz_pred = self._backproject_uvz(uv_vis, z_pred, K)
                xyz_gt   = self._backproject_uvz(uv_vis, z_gt, K)
                self._save_pred_gt_cloud_ply(xyz_pred, xyz_gt, end_points)
                print("[vis] point cloud: red=pred depth, blue=GT depth")

            self._vis_iter += 1

        # ============================================================
        # 6) view + labels + grouping + head
        # ============================================================
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        group_features = self.cy_group(seed_xyz_graspable, seed_features_graspable, grasp_top_views_rot)
        end_points = self.grasp_head(group_features, end_points)

        return end_points
    

class economicgrasp_c2_4(nn.Module):
    """
    C2.4:
      - full-resolution regression depth
      - full-resolution regression spatial enhancer
      - point-aligned gather by img_idxs (same spirit as C1_detach)
      - reuse C1 point-wise graspable/FPS/view/group/head

    Compared with C2.3:
      * remove dense full-res token head
      * remove token-level selection
      * keep image-centric frontend, switch back to point-centric backend
    """
    def __init__(
        self,
        cylinder_radius=0.05,
        seed_feat_dim=512,
        pe_dim=64,
        is_training=True,
        min_depth=0.2,
        max_depth=1.0,
        tok_feat_dim=128,
        feature_3d_dim=32,
        use_inv_depth=True,
        use_depth_grad=True,
        use_gt_xyz_for_train=False,
        detach_depth_for_enhancer=True,
        freeze_backbone=True,
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.detach_depth_for_enhancer = bool(detach_depth_for_enhancer)

        self.seed_feature_dim = int(seed_feat_dim)
        self.pe_dim = int(pe_dim)
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points  = cfgs.m_point
        self.num_view  = cfgs.num_view

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        # ------------------------------------------------------------
        # depth regression net (same family as C2.3)
        # ------------------------------------------------------------
        self.depth_net = DINOv2DepthRegressionNet(
            encoder="vitb",
            stride=1,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            freeze_backbone=freeze_backbone,
        )

        # ------------------------------------------------------------
        # full-res regression enhancer (same spirit as C2.3)
        # ------------------------------------------------------------
        self.enhancer = RegressionSpatialEnhancer(
            tok_feat_dim=tok_feat_dim,
            feature_3d_dim=feature_3d_dim,
            use_inv_depth=use_inv_depth,
            use_depth_grad=use_depth_grad,
        )

        # ------------------------------------------------------------
        # point-wise projection/fusion (same spirit as C1_detach)
        # ------------------------------------------------------------
        self.img_proj = nn.Identity()

        self.pe_mlp = nn.Sequential(
            nn.Linear(4, self.pe_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pe_dim, self.pe_dim),
        )

        # point_img_feat dim may vary, so use LazyLinear
        self.fuse = nn.Sequential(
            nn.LazyLinear(self.seed_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.seed_feature_dim, self.seed_feature_dim),
        )

        # ------------------------------------------------------------
        # reuse original heads (same as C1)
        # ------------------------------------------------------------
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim
        )
        self.grasp_head = Grasp_Head_Local_Interaction(
            num_angle=self.num_angle,
            num_depth=self.num_depth
        )

        # ------------------------------------------------------------
        # vis/debug
        # ------------------------------------------------------------
        self.vis_dir = os.path.join("vis", "c2_4_dbg")
        self.vis_every = 200
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

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

        p_col = np.zeros((p.shape[0], 3), dtype=np.float32); p_col[:, 0] = 1.0
        g_col = np.zeros((g.shape[0], 3), dtype=np.float32); g_col[:, 2] = 1.0

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
        img_idxs = end_points["img_idxs"]  # (B,N) flatten idx in 448*448

        B, _, H, W = img.shape
        assert (H, W) == (448, 448)

        # ============================================================
        # 1) depth regression
        # ============================================================
        # depth_map_pred_448: (B,1,448,448)
        # depth_tok_dbg     : (B,1,448,448) because stride=1
        # img_feat          : (B,128,448,448)
        depth_map_pred_448, depth_tok_dbg, img_feat = self.depth_net(img)

        depth_map_pred = torch.nan_to_num(
            depth_map_pred_448, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=self.min_depth, max=self.max_depth)

        end_points["depth_map_pred"] = depth_map_pred
        end_points["depth_tok_dbg"] = depth_tok_dbg
        end_points["depth_tok_pred"] = depth_map_pred
        end_points["img_feat_dpt"] = img_feat

        # ============================================================
        # 2) full-resolution enhancer
        # ============================================================
        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode="bilinear", align_corners=False)

        depth_for_enh = depth_map_pred.detach() if self.detach_depth_for_enhancer else depth_map_pred
        feat_grid_enh = self.enhancer(img_feat, depth_for_enh, K, stride=1)   # (B,C,448,448)

        # ============================================================
        # 3) point-wise gather from enhanced feature map
        # ============================================================
        Cf = feat_grid_enh.shape[1]
        feat_flat = feat_grid_enh.view(B, Cf, -1)                              # (B,Cf,HW)
        idx = img_idxs.long().clamp(0, H * W - 1).unsqueeze(1).expand(-1, Cf, -1)
        point_img_feat = torch.gather(feat_flat, 2, idx).transpose(1, 2).contiguous()  # (B,N,Cf)

        point_img_feat = self.img_proj(point_img_feat)  # (B,N,C') or identity

        # ============================================================
        # 4) point-wise xyz from predicted depth (same style as C1_detach)
        # ============================================================
        z = gather_depth_by_img_idxs(depth_map_pred, img_idxs)  # (B,N,1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

        # detach geometry path like C1_detach
        z_sg = z.detach()

        u, v = img_idxs_to_uv(img_idxs, W)   # typically (B,N,1)
        xyz = backproject_uvz(u, v, z_sg, K) # (B,N,3)
        end_points["point_clouds"] = xyz

        # optional vis
        if self.vis_dir is not None:
            do_vis = (self._vis_iter % self.vis_every == 0) or bool(end_points.get("force_vis", False))

            if do_vis and ("gt_depth_m" in end_points):
                gt_depth = end_points["gt_depth_m"]
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]

                z_gt = gather_depth_by_img_idxs(gt_depth, img_idxs)
                z_gt = torch.nan_to_num(z_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
                xyz_gt = backproject_uvz(u, v, z_gt, K)
                self._save_pred_gt_cloud_ply(xyz, xyz_gt, end_points)

            self._vis_iter += 1

        # ============================================================
        # 5) geometry PE + fuse -> seed_features (same as C1)
        # ============================================================
        invz = (1.0 / xyz[..., 2:3]).clamp_max(1e6)
        pe = self.pe_mlp(torch.cat([xyz, invz], dim=-1))              # (B,N,pe_dim)

        fused = self.fuse(torch.cat([point_img_feat, pe], dim=-1))    # (B,N,512)
        seed_features = fused.transpose(1, 2).contiguous()            # (B,512,N)

        # ============================================================
        # 6) point-wise graspable mask (same as C1)
        # ============================================================
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2).contiguous()  # (B,N,512)

        objectness_score = end_points["objectness_score"]                  # (B,2,N)
        graspness_score  = end_points["graspness_score"].squeeze(1)        # (B,N)

        objectness_pred = torch.argmax(objectness_score, dim=1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask  = graspness_score > cfgs.graspness_threshold
        graspable_mask  = objectness_mask & graspness_mask

        # ============================================================
        # 7) FPS downsample on graspable points (same as C1)
        # ============================================================
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

            cur_seed_xyz = gather_operation(
                xyz_in.transpose(1, 2).contiguous(), fps_idxs
            ).transpose(1, 2).squeeze(0).contiguous()  # (M,3)

            feat_in = seed_features_flipped[i].index_select(0, cur_idx).contiguous()  # (Ng,512)
            cur_feat = gather_operation(
                feat_in.unsqueeze(0).transpose(1, 2).contiguous(), fps_idxs
            ).squeeze(0).contiguous()  # (512,M)

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0).contiguous()            # (B,M,3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0).contiguous()  # (B,512,M)

        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["D: Graspable Points"] = (
            torch.as_tensor(graspable_num_batch, device=seed_xyz.device, dtype=torch.float32) / float(B)
        ).detach().reshape(())

        # ============================================================
        # 8) view selection
        # ============================================================
        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        # ============================================================
        # 9) label processing
        # ============================================================
        if self.is_training:
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        # ============================================================
        # 10) grouping + head
        # ============================================================
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


def pred_decode_c2_1(end_points):
    """
    Decode predictions for economicgrasp_c2_1 (token seeds).
    Required keys:
      - xyz_graspable:      (B,M,3)
      - grasp_score_pred:   (B,6,M) logits OR probs (we assume logits and softmax)
      - grasp_angle_pred:   (B,num_angle+1,M) logits
      - grasp_depth_pred:   (B,num_depth+1,M) logits
      - grasp_width_pred:   (B,1,M) regression
      - grasp_top_view_xyz: (B,M,3) from ViewNet (eval mode)
    Output:
      list length B, each tensor: (M, 1+1+1+1+9+3+1) = (M,17)
      [score, width, height, depth, rot(9), center(3), obj_id]
    """
    grasp_preds = []

    xyz = end_points["xyz_graspable"].float()   # (B,M,3)
    B, M, _ = xyz.shape

    # constants
    score_bins = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1.0], device=xyz.device, dtype=xyz.dtype).view(6, 1)

    for i in range(B):
        grasp_center = xyz[i]  # (M,3)

        # --- score: (6,M) -> softmax -> expected value ---
        score_logits = end_points["grasp_score_pred"][i].float()  # (6,M)
        score_prob = F.softmax(score_logits, dim=0)               # (6,M)
        grasp_score = (score_bins * score_prob).sum(dim=0, keepdim=True).transpose(0, 1)  # (M,1)

        # --- angle: (A+1,M) -> take max over first A bins ---
        angle_logits = end_points["grasp_angle_pred"][i].float()  # (A+1,M)
        A1 = angle_logits.shape[0]
        A = A1 - 1
        angle_logits_valid = angle_logits[:A, :]                  # drop last "invalid" bin
        _, angle_idx = torch.max(angle_logits_valid, dim=0)       # (M,)
        grasp_angle = angle_idx.to(xyz.dtype) * (np.pi / 12.0)    # (M,)

        # --- depth: (D+1,M) -> take max over first D bins ---
        depth_logits = end_points["grasp_depth_pred"][i].float()  # (D+1,M)
        D1 = depth_logits.shape[0]
        D = D1 - 1
        depth_logits_valid = depth_logits[:D, :]                  # drop last "invalid" bin
        _, depth_idx = torch.max(depth_logits_valid, dim=0)       # (M,)
        grasp_depth = ((depth_idx + 1).to(xyz.dtype) * 0.01).view(-1, 1)  # (M,1)

        # --- width: (1,M) -> (M,1) ---
        grasp_width = (1.2 * end_points["grasp_width_pred"][i].float() / 10.0)  # (1,M) or (M,) depending
        if grasp_width.dim() == 2:
            grasp_width = grasp_width.squeeze(0)
        grasp_width = torch.clamp(grasp_width, min=0.0, max=cfgs.grasp_max_width).view(-1, 1)  # (M,1)

        # --- approaching dir from ViewNet ---
        # ViewNet in eval should provide grasp_top_view_xyz
        approaching = -end_points["grasp_top_view_xyz"][i].float()  # (M,3)
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)  # (M,3,3)
        grasp_rot = grasp_rot.view(M, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)

        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth,
                       grasp_rot, grasp_center, obj_ids], dim=-1)
        )

    return grasp_preds