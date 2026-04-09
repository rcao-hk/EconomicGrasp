import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules_economicgrasp import GraspableNet, ViewNet, Cylinder_Grouping_Global_Interaction, Grasp_Head_Local_Interaction
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from utils.arguments import cfgs
import numpy as np
import os
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .economicgrasp_depth import DINOv2DepthRegressionNet
from .economicgrasp_depth_c1 import RegressionSpatialEnhancer, TokGraspableHead2D

from utils.loss_utils import (
    batch_viewpoint_params_to_matrix,
    transform_point_cloud,
    generate_grasp_views,
)

def gather_depth_by_img_idxs(depth_map_1hw: torch.Tensor, img_idxs: torch.Tensor):
    if depth_map_1hw.dim() == 3:
        depth_flat = depth_map_1hw.reshape(depth_map_1hw.size(0), -1)
    else:
        depth_flat = depth_map_1hw[:, 0].reshape(depth_map_1hw.size(0), -1)
    return depth_flat.gather(1, img_idxs)


def build_top_view_rot_from_inds(top_view_inds: torch.Tensor, num_view: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      top_view_inds: (B, M) long
    Returns:
      vp_xyz: (B, M, 3)
      vp_rot: (B, M, 3, 3)
    """
    B, M = top_view_inds.shape
    device = top_view_inds.device
    template_views = generate_grasp_views(num_view).to(device)  # (V,3)
    vp_xyz = template_views[top_view_inds.view(-1)].view(B, M, 3)
    batch_angle = torch.zeros(B * M, dtype=vp_xyz.dtype, device=device)
    vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz.view(-1, 3), batch_angle).view(B, M, 3, 3)
    return vp_xyz, vp_rot


class economicgrasp_query(nn.Module):
    """
    Standalone v0:
      - c2.3 front-end kept
      - sampled query points kept
      - existing ViewNet / grouping / grasp head kept
      - no decoder, no memory
      - train supervision uses scene-level GT set + Hungarian matching
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
        use_inv_depth=False,
        use_depth_grad=False,
        use_gt_xyz_for_train=False,
        topk_use_objectness=True,
        detach_depth_for_enhancer=True,
        vis_dir=None,
        vis_every=1000,
        debug_print_every=50,
        vis_token_maxB=1,
        save_pointcloud_vis=True,
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.topk_use_objectness = bool(topk_use_objectness)
        self.detach_depth_for_enhancer = bool(detach_depth_for_enhancer)
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.debug_print_every = int(debug_print_every)
        self.vis_token_maxB = int(vis_token_maxB)
        self.save_pointcloud_vis = bool(save_pointcloud_vis)
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

        self.seed_feature_dim = int(tok_feat_dim)
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points = cfgs.m_point
        # self.M_points = 256
        self.num_view = cfgs.num_view
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        self.depth_net = DINOv2DepthRegressionNet(
            encoder='vitb',
            stride=1,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            freeze_backbone=True,
        )
        self.enhancer = RegressionSpatialEnhancer(
            tok_feat_dim=tok_feat_dim,
            feature_3d_dim=feature_3d_dim,
            use_inv_depth=use_inv_depth,
            use_depth_grad=use_depth_grad,
        )
        self.graspable_2d = TokGraspableHead2D(in_dim=tok_feat_dim)
        # self.sel_proj = nn.Sequential(
        #     nn.Conv1d(tok_feat_dim, self.seed_feature_dim, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(self.seed_feature_dim, self.seed_feature_dim, 1),
        # )
        self.view = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim)
        self.cy_group = Cylinder_Grouping_Global_Interaction(
            nsample=16,
            cylinder_radius=cylinder_radius,
            seed_feature_dim=self.seed_feature_dim,
        )
        self.grasp_head = Grasp_Head_Local_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap="Spectral", title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        arr2d = np.asarray(arr2d)
        plt.figure(figsize=(6, 6))
        if vmin is None:
            vmin = float(np.nanmin(arr2d)) if np.isfinite(arr2d).any() else 0.0
        if vmax is None:
            vmax = float(np.nanmax(arr2d)) if np.isfinite(arr2d).any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_points(self, img_448, pts_uv, out_path, values=None, title=None, cmap='viridis', s=8):
        x = img_448.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = x.permute(1, 2, 0).numpy()
        pts = pts_uv.detach().float().cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(x)
        if pts.shape[0] > 0:
            if values is None:
                plt.scatter(pts[:, 0], pts[:, 1], s=s, c='r')
            else:
                val = values.detach().float().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
                sc = plt.scatter(pts[:, 0], pts[:, 1], s=s, c=val, cmap=cmap)
                plt.colorbar(sc, fraction=0.046, pad=0.04)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_binary_points(self, img_448, pts_uv, mask01, out_path, title=None, s=10):
        x = img_448.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = x.permute(1, 2, 0).numpy()
        pts = pts_uv.detach().float().cpu().numpy()
        mask01 = mask01.detach().bool().cpu().numpy() if torch.is_tensor(mask01) else np.asarray(mask01).astype(bool)
        plt.figure(figsize=(6, 6))
        plt.imshow(x)
        if pts.shape[0] > 0:
            if (~mask01).any():
                plt.scatter(pts[~mask01, 0], pts[~mask01, 1], s=s, c='r', label='0')
            if mask01.any():
                plt.scatter(pts[mask01, 0], pts[mask01, 1], s=s, c='lime', label='1')
            plt.legend(loc='upper right')
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def _save_pred_gt_cloud_ply(self, cloud_pred: torch.Tensor, cloud_gt: torch.Tensor, out_path: str):
        if (o3d is None) or (not self.save_pointcloud_vis):
            return
        p = cloud_pred.detach().float().cpu().numpy()
        g = cloud_gt.detach().float().cpu().numpy()

        def _valid(x):
            m = np.isfinite(x).all(axis=1)
            m &= (x[:, 2] > 0)
            return x[m]

        p = _valid(p)
        g = _valid(g)
        if p.shape[0] == 0 and g.shape[0] == 0:
            return

        pts_list, col_list = [], []
        if p.shape[0] > 0:
            pts_list.append(p)
            c = np.zeros((p.shape[0], 3), dtype=np.float32)
            c[:, 0] = 1.0
            col_list.append(c)
        if g.shape[0] > 0:
            pts_list.append(g)
            c = np.zeros((g.shape[0], 3), dtype=np.float32)
            c[:, 2] = 1.0
            col_list.append(c)
        pts = np.concatenate(pts_list, axis=0)
        cols = np.concatenate(col_list, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

    @torch.no_grad()
    def _save_query_gt_cloud_ply(self, query_xyz: torch.Tensor, gt_center: torch.Tensor, out_path: str):
        self._save_pred_gt_cloud_ply(query_xyz, gt_center, out_path)

    def _print_debug(self, end_points: Dict):
        if self.debug_print_every <= 0:
            return
        if (self._vis_iter % self.debug_print_every) != 0 and not bool(end_points.get('force_vis', False)):
            return
        grasp_raw = end_points.get('dbg_grasp_raw', None)
        grasp_sel = end_points.get('dbg_grasp_sel', None)
        mask_obj = end_points.get('dbg_mask_obj', None)
        mask_thr = end_points.get('dbg_mask_pred', None)
        valid_tok = end_points.get('token_valid_mask', None)
        msg = f"[querydbg] iter={self._vis_iter}"
        if valid_tok is not None:
            msg += f" valid={valid_tok.float().sum(1).mean().item():.1f}"
        if mask_obj is not None:
            msg += f" obj_pred1={mask_obj.float().sum(1).mean().item():.1f}"
        if mask_thr is not None:
            msg += f" cand_thr={mask_thr.float().sum(1).mean().item():.1f}"
        if grasp_raw is not None:
            msg += f" gr_raw[min,max]=({grasp_raw.min().item():.3f},{grasp_raw.max().item():.3f})"
        if grasp_sel is not None:
            q10 = torch.quantile(grasp_sel, 0.1).item()
            q50 = torch.quantile(grasp_sel, 0.5).item()
            q90 = torch.quantile(grasp_sel, 0.9).item()
            msg += f" gr_sel[p10,p50,p90]=({q10:.3f},{q50:.3f},{q90:.3f})"
        if 'D: Num GT' in end_points:
            msg += f" num_gt={float(end_points['D: Num GT']):.2f}"
        print(msg)

    @torch.no_grad()
    def _write_vis_bundle(self, end_points: Dict, vis_iter: int, force: bool = False):
        if self.vis_dir is None:
            return
        do_vis = force or bool(end_points.get('force_vis', False)) or ((vis_iter % max(self.vis_every, 1)) == 0)
        if not do_vis:
            return

        img = end_points['img']
        B, _, H, W = img.shape
        tag = end_points.get('vis_tag', f'iter{vis_iter:06d}')
        out_dir = os.path.join(self.vis_dir, f'querydbg_{tag}')
        os.makedirs(out_dir, exist_ok=True)

        maxB = min(B, max(self.vis_token_maxB, 1))
        for b in range(maxB):
            sub = os.path.join(out_dir, f'b{b}')
            os.makedirs(sub, exist_ok=True)
            pred_obj_map = end_points['dbg_objectness_pred'][b].view(H, W).float()
            pred_gra_map = end_points['dbg_grasp_sel'][b].view(H, W)
            cand_map = end_points['dbg_mask_pred'][b].float().view(H, W)
            self._save_map_png(pred_obj_map, os.path.join(sub, 'pred_objectness.png'), vmin=0, vmax=1, cmap='viridis', title='Pred obj')
            self._save_map_png(pred_gra_map, os.path.join(sub, 'pred_graspness.png'), vmin=0, vmax=1, cmap='Spectral', title='Pred graspness')
            self._save_map_png(cand_map, os.path.join(sub, 'pred_candidate_mask.png'), vmin=0, vmax=1, cmap='gray', title='Pred cand mask')

            if ('objectness_label_tok' in end_points) and ('graspness_label_tok' in end_points):
                gt_obj = end_points['objectness_label_tok']
                gt_gra = end_points['graspness_label_tok']
                if gt_obj.shape[1] == H * W and gt_gra.shape[1] == H * W:
                    self._save_map_png(gt_obj[b].view(H, W).float(), os.path.join(sub, 'gt_objectness.png'), vmin=-1, vmax=1, cmap='viridis', title='GT obj (-1 invalid)')
                    self._save_map_png(gt_gra[b].view(H, W).float(), os.path.join(sub, 'gt_graspness.png'), vmin=0, vmax=1, cmap='Spectral', title='GT graspness')

            if 'token_sel_idx' in end_points:
                sel = end_points['token_sel_idx'][b]
                xs = (sel % W).float()
                ys = (sel // W).float()
                pts_uv = torch.stack([xs, ys], dim=-1)
                sel_mask = torch.zeros((H * W,), device=img.device)
                sel_mask[sel] = 1.0
                self._save_map_png(sel_mask.view(H, W), os.path.join(sub, 'selected_tokens.png'), vmin=0, vmax=1, cmap='gray', title='Selected queries')
                self._save_overlay_points(img[b], pts_uv, os.path.join(sub, 'overlay_selected.png'), title='Selected queries')

                if 'query_view_conf' in end_points:
                    self._save_overlay_points(img[b], pts_uv, os.path.join(sub, 'overlay_view_conf.png'), values=end_points['query_view_conf'][b], title='Query view confidence')
                if 'query_match_mask' in end_points:
                    self._save_overlay_binary_points(img[b], pts_uv, end_points['query_match_mask'][b] > 0, os.path.join(sub, 'overlay_matched_queries.png'), title='Matched queries')
                if 'query_score_expectation' in end_points:
                    self._save_overlay_points(img[b], pts_uv, os.path.join(sub, 'overlay_score_expectation.png'), values=end_points['query_score_expectation'][b], title='Score expectation')

                if ('batch_gt_grasp_set' in end_points) and (o3d is not None) and self.save_pointcloud_vis:
                    gt = end_points['batch_gt_grasp_set'][b]['center']
                    q = end_points['xyz_graspable'][b]
                    self._save_query_gt_cloud_ply(q, gt, os.path.join(sub, 'query_vs_gt_centers.ply'))

            if self.save_pointcloud_vis and ('gt_depth_m' in end_points) and ('img_idxs' in end_points):
                gt_depth = end_points['gt_depth_m']
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.dim() == 4:
                    gt_depth = gt_depth[:, :1]
                img_idxs_vis = end_points['img_idxs'][b:b+1].long().clamp(0, H * W - 1)
                z_pred = gather_depth_by_img_idxs(end_points['depth_map_pred'][b:b+1], img_idxs_vis).unsqueeze(-1)
                z_gt = gather_depth_by_img_idxs(gt_depth[b:b+1], img_idxs_vis).unsqueeze(-1)
                u_vis = (img_idxs_vis % W).float()
                v_vis = (img_idxs_vis // W).float()
                uv_vis = torch.stack([u_vis, v_vis], dim=-1)
                xyz_pred = self._backproject_uvz(uv_vis, z_pred, end_points['K'][b:b+1]).squeeze(0)
                xyz_gt = self._backproject_uvz(uv_vis, z_gt, end_points['K'][b:b+1]).squeeze(0)
                self._save_pred_gt_cloud_ply(xyz_pred, xyz_gt, os.path.join(sub, 'pred_vs_gt_depth_cloud.ply'))

    @torch.no_grad()
    def visualize_debug(self, end_points: Dict, force: bool = False, advance_iter: bool = False):
        """
        Post-loss debug writer. By default it writes into the same iteration folder
        created by forward-time auto visualization and does NOT advance the internal
        counter. If you call it every iteration after loss, files are only saved every
        `vis_every` steps, consistent with c2.3-style behavior.
        """
        vis_iter = max(self._vis_iter - 1, 0) if not advance_iter else self._vis_iter
        self._print_debug(end_points)
        self._write_vis_bundle(end_points, vis_iter=vis_iter, force=force)
        if advance_iter:
            self._vis_iter += 1

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

    def _select_queries(
        self,
        feat_grid_enh: torch.Tensor,
        depth_map_pred: torch.Tensor,
        K: torch.Tensor,
        end_points: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img = end_points['img']
        B, _, H, W = img.shape
        Ntok = H * W
        M = int(self.M_points)

        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, dim=1)

        if 'token_valid_mask' in end_points:
            valid_tok = end_points['token_valid_mask'].bool()
            if valid_tok.shape[1] != Ntok:
                raise ValueError(f'Expected token_valid_mask with Ntok={Ntok}, got {tuple(valid_tok.shape)}')
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        grasp_raw = graspness_score
        grasp_sel = grasp_raw.clamp(0.0, 1.0)
        mask_obj_pred = valid_tok & (objectness_pred == 1)
        mask_thr_pred = mask_obj_pred & (grasp_sel > float(cfgs.graspness_threshold))
        end_points['dbg_grasp_raw'] = grasp_raw.detach()
        end_points['dbg_grasp_sel'] = grasp_sel.detach()
        end_points['dbg_mask_obj'] = mask_obj_pred.detach()
        end_points['dbg_mask_pred'] = mask_thr_pred.detach()
        end_points['dbg_objectness_pred'] = objectness_pred.detach()

        flat_all = torch.arange(H * W, device=img.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        u_all = (flat_all % W).float()
        v_all = (flat_all // W).float()
        uv_all = torch.stack([u_all, v_all], dim=-1)
        z_all_pred = depth_map_pred.view(B, -1, 1).contiguous()
        z_all_pred = torch.nan_to_num(z_all_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
        xyz_all_pred = self._backproject_uvz(uv_all, z_all_pred.detach(), K)

        use_gt_xyz = self.is_training and self.use_gt_xyz_for_train and ('gt_depth_m' in end_points)
        if use_gt_xyz:
            gt_depth = end_points['gt_depth_m']
            if gt_depth.dim() == 3:
                gt_depth = gt_depth.unsqueeze(1)
            elif gt_depth.dim() == 4:
                gt_depth = gt_depth[:, :1]
            z_all_gt = gt_depth.view(B, -1, 1).contiguous()
            z_all_gt = torch.nan_to_num(z_all_gt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            xyz_all_match = self._backproject_uvz(uv_all, z_all_gt, K)
        else:
            xyz_all_match = xyz_all_pred

        # feat_grid_enh: (B,C,H,W)
        seed_features_flipped = feat_grid_enh.view(B, feat_grid_enh.shape[1], -1).transpose(1, 2).contiguous()  # (B,Ntok,C)
        seed_xyz = xyz_all_match  # (B,Ntok,3)

        graspable_mask = mask_thr_pred   # (B,Ntok)
        seed_features_graspable = []
        seed_xyz_graspable = []
        sel_idx_list = []
        graspable_num_batch = 0.0

        for i in range(B):
            cur_mask = graspable_mask[i]
            idx_graspable = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
            graspable_num_batch += cur_mask.sum().float()

            # ---- case 1: no graspable points ----
            if idx_graspable.numel() == 0:
                sel_idx = torch.randint(0, Ntok, (M,), device=img.device)   # (M,)
                cur_feat = seed_features_flipped[i].index_select(0, sel_idx)   # (M,C)
                cur_seed_xyz = seed_xyz[i].index_select(0, sel_idx)            # (M,3)
                cur_feat = cur_feat.transpose(0, 1).contiguous()               # (C,M)

            # ---- case 2: not enough graspable points ----
            elif idx_graspable.numel() < M:
                rep = torch.randint(0, idx_graspable.numel(), (M,), device=img.device)
                sel_idx = idx_graspable.index_select(0, rep)                   # (M,)
                cur_feat = seed_features_flipped[i].index_select(0, sel_idx)   # (M,C)
                cur_seed_xyz = seed_xyz[i].index_select(0, sel_idx)            # (M,3)
                cur_feat = cur_feat.transpose(0, 1).contiguous()               # (C,M)

            # ---- case 3: enough graspable points, do FPS ----
            else:
                cur_feat_all = seed_features_flipped[i].index_select(0, idx_graspable)   # (Ng,C)
                cur_seed_xyz_all = seed_xyz[i].index_select(0, idx_graspable)             # (Ng,3)

                cur_seed_xyz_in = cur_seed_xyz_all.unsqueeze(0).contiguous()               # (1,Ng,3)
                fps_idxs = furthest_point_sample(cur_seed_xyz_in, M).long()                # (1,M)

                cur_seed_xyz_flipped = cur_seed_xyz_in.transpose(1, 2).contiguous()        # (1,3,Ng)
                cur_seed_xyz = gather_operation(
                    cur_seed_xyz_flipped, fps_idxs.int()
                ).transpose(1, 2).squeeze(0).contiguous()                                  # (M,3)

                cur_feat_flipped = cur_feat_all.unsqueeze(0).transpose(1, 2).contiguous()  # (1,C,Ng)
                cur_feat = gather_operation(
                    cur_feat_flipped, fps_idxs.int()
                ).squeeze(0).contiguous()                                                  # (C,M)

                sel_idx = idx_graspable.index_select(0, fps_idxs.squeeze(0))               # (M,)

            seed_features_graspable.append(cur_feat)      # (C,M)
            seed_xyz_graspable.append(cur_seed_xyz)       # (M,3)
            sel_idx_list.append(sel_idx)                  # (M,)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0).contiguous()          # (B,M,3)
        seed_features_graspable = torch.stack(seed_features_graspable, 0).contiguous()# (B,C,M)
        token_sel_idx = torch.stack(sel_idx_list, dim=0).contiguous()                 # (B,M)

        end_points['token_sel_idx'] = token_sel_idx
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['token_sel_xyz'] = seed_xyz_graspable
        end_points['D: Graspable Points'] = graspable_num_batch / float(B)

        return seed_xyz_graspable, seed_features_graspable, token_sel_idx, xyz_all_pred, xyz_all_match

    def forward(self, end_points: Dict) -> Dict:
        img = end_points['img']
        K = end_points['K']
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)

        depth_map_pred_448, depth_tok_dbg, img_feat = self.depth_net(img)
        depth_map_pred = torch.nan_to_num(depth_map_pred_448, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            min=self.min_depth, max=self.max_depth
        )
        end_points['depth_map_pred'] = depth_map_pred
        end_points['depth_tok_dbg'] = depth_tok_dbg
        end_points['depth_tok_pred'] = depth_map_pred
        end_points['img_feat_dpt'] = img_feat

        if img_feat.shape[-2:] != (H, W):
            img_feat = F.interpolate(img_feat, size=(H, W), mode='bilinear', align_corners=False)
        depth_for_enh = depth_map_pred.detach() if self.detach_depth_for_enhancer else depth_map_pred
        feat_grid_enh = self.enhancer(img_feat, depth_for_enh, K, stride=1)

        end_points = self.graspable_2d(feat_grid_enh, end_points)
        seed_xyz_graspable, seed_features_graspable, token_sel_idx, xyz_all_pred, xyz_all_match = self._select_queries(
            feat_grid_enh=feat_grid_enh,
            depth_map_pred=depth_map_pred,
            K=K,
            end_points=end_points,
        )
        end_points['xyz_all_pred'] = xyz_all_pred
        end_points['xyz_all_match'] = xyz_all_match

        end_points, res_feat = self.view(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        top_view_inds = torch.argmax(end_points['view_score'], dim=2)  # (B,M)
        vp_xyz, grasp_top_views_rot = build_top_view_rot_from_inds(top_view_inds, self.num_view)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = grasp_top_views_rot

        group_features = self.cy_group(seed_xyz_graspable, seed_features_graspable, grasp_top_views_rot)
        end_points = self.grasp_head(group_features, end_points)
        end_points['group_features_query'] = group_features

        with torch.no_grad():
            view_prob = torch.softmax(end_points['view_score'], dim=2)
            end_points['query_view_conf'] = view_prob.max(dim=2).values
            score_prob = torch.softmax(end_points['grasp_score_pred'], dim=1)
            score_bins = torch.arange(score_prob.shape[1], device=score_prob.device, dtype=score_prob.dtype).view(1, -1, 1)
            end_points['query_score_expectation'] = (score_prob * score_bins).sum(dim=1)

        # c2.3-style periodic visualization: auto-save every fixed number of iters.
        self._print_debug(end_points)
        self._write_vis_bundle(end_points, vis_iter=self._vis_iter, force=False)
        self._vis_iter += 1
        return end_points


def pred_decode_query(
    end_points,
):
    """
    Decode predictions for economicgrasp_query.

    Required keys:
      - xyz_graspable:        (B,M,3)
      - grasp_score_pred:     (B,6,M) logits
      - grasp_angle_pred:     (B,num_angle+1,M) logits
      - grasp_depth_pred:     (B,num_depth+1,M) logits
      - grasp_width_pred:     (B,1,M) regression
      - grasp_top_view_xyz:   (B,M,3)


    Output:
      list length B, each tensor: (K,17) or (M,17)
      [score, width, height, depth, rot(9), center(3), obj_id]
    """
    
    grasp_preds = []

    xyz = end_points["xyz_graspable"].float()   # (B,M,3)
    B, M, _ = xyz.shape
    device = xyz.device
    dtype = xyz.dtype

    # score bins for 6-way score classification
    score_bins = torch.tensor(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        device=device, dtype=dtype
    ).view(6, 1)

    for i in range(B):
        grasp_center = xyz[i]  # (M,3)

        # -----------------------------
        # score: logits -> expected value
        # -----------------------------
        score_logits = end_points["grasp_score_pred"][i].float()  # (6,M)
        score_prob = F.softmax(score_logits, dim=0)               # (6,M)
        grasp_score = (score_bins * score_prob).sum(dim=0)        # (M,)

        grasp_score = grasp_score.view(-1, 1)  # (M,1)

        # -----------------------------
        # angle: drop last invalid bin
        # -----------------------------
        angle_logits = end_points["grasp_angle_pred"][i].float()  # (A+1,M)
        A1 = angle_logits.shape[0]
        A = A1 - 1
        angle_logits_valid = angle_logits[:A, :]
        _, angle_idx = torch.max(angle_logits_valid, dim=0)       # (M,)
        grasp_angle = angle_idx.to(dtype) * (np.pi / float(A))    # (M,)

        # -----------------------------
        # depth: drop last invalid bin
        # keep same convention as your c2.3 decode: (idx+1)*0.01
        # -----------------------------
        depth_logits = end_points["grasp_depth_pred"][i].float()  # (D+1,M)
        D1 = depth_logits.shape[0]
        D = D1 - 1
        depth_logits_valid = depth_logits[:D, :]
        _, depth_idx = torch.max(depth_logits_valid, dim=0)       # (M,)
        grasp_depth = ((depth_idx + 1).to(dtype) * 0.01).view(-1, 1)  # (M,1)

        # -----------------------------
        # width
        # keep same convention as your c2.3 decode
        # -----------------------------
        grasp_width = 1.2 * end_points["grasp_width_pred"][i].float() / 10.0
        if grasp_width.dim() == 2:
            grasp_width = grasp_width.squeeze(0)
        grasp_width = torch.clamp(
            grasp_width, min=0.0, max=cfgs.grasp_max_width
        ).view(-1, 1)  # (M,1)

        # -----------------------------
        # approaching dir from predicted top view
        # -----------------------------
        approaching = -end_points["grasp_top_view_xyz"][i].float()  # (M,3)
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)  # (M,3,3)
        grasp_rot = grasp_rot.view(M, 9)

        grasp_height = 0.02 * torch.ones((M, 1), device=device, dtype=dtype)
        obj_ids = -1.0 * torch.ones((M, 1), device=device, dtype=dtype)

        pred = torch.cat(
            [grasp_score, grasp_width, grasp_height, grasp_depth,
             grasp_rot, grasp_center, obj_ids],
            dim=-1
        )  # (M,17)

        grasp_preds.append(pred)

    return grasp_preds