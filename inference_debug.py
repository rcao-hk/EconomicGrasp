#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from graspnetAPI import GraspGroup
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn
from torch.utils.data import DataLoader, Subset
import json

# ---- adjust these imports if your repo path differs ----
from models.economicgrasp_2d import (
    economicgrasp_c3,
    generate_grasp_views,
    batch_viewpoint_params_to_matrix,
)
from utils.label_generation_2d import process_grasp_labels, process_view_labels_dense

# =========================================================
# user controls
# =========================================================
ORACLE_MODE = getattr(cfgs, 'oracle_mode', 'A1')         # choices: "A1", "A2", "A6"
SAMPLE_INTERVAL = getattr(cfgs, "sample_interval", 1)        # e.g. 1 = every frame; 4 = 1/4 frames
TEST_SPLIT = "test_seen"   # fixed for this debug script
USE_FPS_TOKENS = True
PRED_TOPK = getattr(cfgs, "m_point", 256)

# save under a separate subdir to avoid overwriting normal inference
# SAVE_ROOT = os.path.join(cfgs.save_dir, f"{TEST_SPLIT}_{ORACLE_MODE}_si{SAMPLE_INTERVAL}")
# os.makedirs(SAVE_ROOT, exist_ok=True)

# =========================================================
# extra debug controls
# =========================================================
SAVE_NOCOLLISION_NPY = bool(getattr(cfgs, "save_nocollision_npy", True))
NOCOLLISION_SUFFIX = getattr(cfgs, "nocollision_suffix", "_nocollision")
DEBUG_LOG_DIR = os.path.join(
    cfgs.save_dir,
    f"_oracle_logs_{TEST_SPLIT}_{ORACLE_MODE}_si{SAMPLE_INTERVAL}"
)
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

F2_SELECT_BY = getattr(cfgs, "f2_select_by", "score")   # "score" or "fps"
TOKEN_MODE_TOPK = int(getattr(cfgs, "token_mode_topk", 8))


def append_jsonl(path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    
# =========================================================
# helpers
# =========================================================
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_sampled_indices(dataset, sample_interval=1):
    """
    Per-scene sampling by anno/frame id.
    Keep samples whose anno_idx % sample_interval == 0.
    """
    if sample_interval <= 1:
        return list(range(len(dataset)))

    sampled = []
    for idx in range(len(dataset)):
        anno_idx = int(dataset.frameid[idx])
        if (anno_idx % sample_interval) == 0:
            sampled.append(idx)
    return sampled


def move_batch_to_device(batch_data, device):
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        elif 'graph' in key:
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    return batch_data


def fps_indices_3d(xyz: torch.Tensor, M: int, start_idx: torch.Tensor = None):
    """
    xyz: (K,3)
    return: (min(M,K),) long

    Pure PyTorch FPS for debug/oracle selection.
    """
    device = xyz.device
    K = xyz.shape[0]
    if K == 0:
        return torch.zeros((0,), device=device, dtype=torch.long)

    M = min(int(M), int(K))
    centroids = torch.empty((M,), device=device, dtype=torch.long)
    dist = torch.full((K,), 1e10, device=device, dtype=xyz.dtype)

    if start_idx is None:
        farthest = torch.randint(0, K, (1,), device=device, dtype=torch.long)
    else:
        farthest = start_idx.view(1).to(device=device, dtype=torch.long).clamp_(0, K - 1)

    for i in range(M):
        centroids[i] = farthest[0]
        centroid = xyz[farthest]                         # (1,3)
        d = ((xyz - centroid) ** 2).sum(dim=-1)         # (K,)
        dist = torch.minimum(dist, d)
        farthest = torch.argmax(dist).view(1)

    return centroids


def infer_hw(end_points):
    if "depth_map_pred" in end_points:
        d = end_points["depth_map_pred"]
        if d.dim() == 4:
            _, _, H, W = d.shape
        elif d.dim() == 3:
            _, H, W = d.shape
        else:
            raise ValueError(f"Unsupported depth_map_pred shape: {tuple(d.shape)}")
    elif "img" in end_points:
        _, _, H, W = end_points["img"].shape
    else:
        raise ValueError("Cannot infer H,W from end_points.")
    return H, W


def to_int_scalar(x):
    if torch.is_tensor(x):
        return int(x.reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        return int(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


def build_gt_oracle_labels(end_points):
    """
    A6:  GT view + GT angle
    A7:  GT depthoff
    A8:  GT width
    A9:  GT z(scene depth) + GT view + GT angle + GT depthoff + GT width

    E1:  current selected tokens, geometry predicted, final score = GT score
    E2:  GT-selected tokens + GT score + A9 geometry
    E2B: GT-selected tokens + GT score + A9 geometry + FPS on A9 centers
    E3:  GT-selected tokens + GT score + GT center xyz + GT view/angle/depth/width
    E3B: same as E3, but token selection by FPS on GT center xyz
    """
    if ORACLE_MODE not in ["A6", "A7", "A8", "A9", "E1", "E2", "E2B", "E3", "E3B"]:
        return end_points

    with torch.no_grad():
        end_points = process_grasp_labels(
            end_points,
            s_tok=1,
            depth_min=float(cfgs.min_depth),
            depth_max=float(cfgs.max_depth),
            max_width=float(getattr(cfgs, "grasp_max_width", 0.1)),
            min_score=1e-6,
            vis_thresh=float(getattr(cfgs, "label_vis_thresh", 0.01)),
        )

        if ORACLE_MODE in ["A6", "A9", "E2", "E2B", "E3", "E3B"]:
            end_points = process_view_labels_dense(
                end_points,
                view_stride=int(getattr(cfgs, "view_stride", 4)),
                depth_min=float(cfgs.min_depth),
                depth_max=float(cfgs.max_depth),
                vis_thresh=float(getattr(cfgs, "label_vis_thresh", 0.01)),
            )

    return end_points

def pred_decode_c3_oracle(
    end_points,
    mode="A1",
    pred_topk=256,
    use_fps_tokens=True,
    graspness_threshold=None,
    width_scale=None,
    width_expand_ratio=1.0,
    depth_bin_size=0.01,
    grasp_height=0.02,
    view_stride=None,
):
    """
    A1: predicted geometry, final score = obj_prob * graspness
    A2: predicted angle + heuristic view, final score = obj_prob * graspness
    A3: GT z, others predicted, final score = obj_prob * graspness
    A6: GT view + GT angle, others predicted, final score = obj_prob * graspness
    A7: GT depthoff, others predicted, final score = obj_prob * graspness
    A8: GT width, others predicted, final score = obj_prob * graspness
    A9: GT z + GT view + GT angle + GT depthoff + GT width, final score = obj_prob * graspness

    E1: current selected tokens, geometry predicted, final score = GT score label
    E2: GT-selected tokens + GT score + A9 geometry
    E3: GT-selected tokens + GT score + GT center xyz + GT view/angle/depth/width
    E3B: same as E3, but token selection by FPS on GT center xyz
    """
    assert mode in [
        "A1", "A2", "A3", "A6", "A7", "A8", "A9",
        "E1", "E2", "E2B", "E3", "E3B",
    ]

    device = end_points["objectness_score"].device

    if graspness_threshold is None:
        graspness_threshold = float(getattr(cfgs, "graspness_threshold", 0.0))
    if width_scale is None:
        width_scale = float(getattr(cfgs, "width_scale", 10.0))
    if view_stride is None:
        view_stride = int(getattr(cfgs, "view_stride", 4))

    num_angle = int(getattr(cfgs, "num_angle", 12))
    num_view = int(getattr(cfgs, "num_view", 300))

    H, W = infer_hw(end_points)

    objectness_score = end_points["objectness_score"].float()                  # (B,2,Ntok)
    graspness_score = end_points["graspness_score"].float().squeeze(1)         # (B,Ntok)
    grasp_z_pred = end_points["grasp_z_pred"].float().squeeze(1)               # (B,Ntok)
    grasp_angle_pred = end_points["grasp_angle_pred_tok"].float()              # (B,A,Ntok)
    grasp_depthoff_pred = end_points["grasp_depthoff_pred_tok"].float()        # (B,D,Ntok)
    grasp_width_pred = end_points["grasp_width_pred_tok"].float().squeeze(1)   # (B,Ntok)
    K_all = end_points["K"].float()                                            # (B,3,3)

    B, _, Ntok = objectness_score.shape
    assert Ntok == H * W, f"Ntok={Ntok} but H*W={H*W}"

    if "token_valid_mask" in end_points:
        token_valid_mask = end_points["token_valid_mask"].bool()
    else:
        token_valid_mask = torch.ones((B, Ntok), dtype=torch.bool, device=device)

    obj_prob = F.softmax(objectness_score, dim=1)[:, 1, :]                     # (B,Ntok)
    cand_score_all = obj_prob * graspness_score                                # (B,Ntok)

    obj_pred = torch.argmax(objectness_score, dim=1)
    cand_mask = token_valid_mask & (obj_pred == 1) & (graspness_score > graspness_threshold)

    template_views = generate_grasp_views(num_view).to(device).float()

    # --------------------------------------------------
    # predicted dense view for A1/A3/A7/A8/E1
    # --------------------------------------------------
    if mode in ["A1", "A3", "A7", "A8", "E1"]:
        view_score_map = end_points["view_score_map"].float()                  # (B,V,Hv,Wv)
        Bv, V, Hv, Wv = view_score_map.shape
        assert Bv == B and V == num_view

    # --------------------------------------------------
    # GT score label needed by E1/E2/E3/E3B
    # --------------------------------------------------
    if mode in ["E1", "E2", "E2B", "E3", "E3B"]:
        assert "grasp_score_label_tok" in end_points
        gt_score_label = end_points["grasp_score_label_tok"].float().clamp(min=0.0, max=1.0)

    # --------------------------------------------------
    # GT center xyz needed by E3/E3B
    # --------------------------------------------------
    if mode in ["E3", "E3B"]:
        assert "grasp_center_xyz_label_tok" in end_points
        gt_center_xyz = end_points["grasp_center_xyz_label_tok"].float()       # (B,Ntok,3)

    # --------------------------------------------------
    # GT view/angle needed by A6/A9/E2/E3/E3B
    # --------------------------------------------------
    if mode in ["A6", "A9", "E2", "E2B", "E3", "E3B"]:
        assert "view_graspness_label_tok" in end_points
        assert "view_valid_mask_tok" in end_points
        assert "grasp_angle_label_tok" in end_points

        gt_view_label = end_points["view_graspness_label_tok"].float()         # (B,Nv,V)
        gt_view_valid = end_points["view_valid_mask_tok"].bool()               # (B,Nv)
        Hv, Wv = end_points["view_grid_hw"]

        if isinstance(Hv, torch.Tensor):
            Hv = int(Hv.item())
        if isinstance(Wv, torch.Tensor):
            Wv = int(Wv.item())

        gt_angle_label = end_points["grasp_angle_label_tok"].long()            # (B,Ntok)

    # --------------------------------------------------
    # GT depthoff needed by A7/A9/E2/E3/E3B
    # --------------------------------------------------
    if mode in ["A7", "A9", "E2", "E2B", "E3", "E3B"]:
        assert "grasp_depthoff_label_tok" in end_points
        gt_depthoff_label = end_points["grasp_depthoff_label_tok"].long()      # (B,Ntok)

    # --------------------------------------------------
    # GT width needed by A8/A9/E2/E3/E3B
    # --------------------------------------------------
    if mode in ["A8", "A9", "E2", "E2B", "E3", "E3B"]:
        assert "grasp_width_label_tok" in end_points
        gt_width_label = end_points["grasp_width_label_tok"].float()           # (B,Ntok)

    # --------------------------------------------------
    # GT depth needed by A3/A9/E2
    # --------------------------------------------------
    if mode in ["A3", "A9", "E2", "E2B"]:
        assert "gt_depth_m" in end_points
        gt_depth = end_points["gt_depth_m"]
        if gt_depth.dim() == 4:
            gt_depth = gt_depth[:, 0]
        gt_depth = gt_depth.contiguous().view(gt_depth.shape[0], -1)           # (B,Ntok)

    # --------------------------------------------------
    # GT graspable needed by E2/E3/E3B
    # --------------------------------------------------
    if mode in ["E2", "E2B", "E3", "E3B"]:
        assert "graspable_label_tok" in end_points
        gt_graspable = end_points["graspable_label_tok"].bool()                # (B,Ntok)


    grasp_preds = []

    for i in range(B):
        score_i = cand_score_all[i]
        mask_i = cand_mask[i]
        K_i = K_all[i]

        # --------------------------------------------------
        # 1) token selection
        # --------------------------------------------------
        if mode in ["E2", "E2B", "E3", "E3B"]:
            gt_valid_mask = gt_graspable[i] & token_valid_mask[i]

            gt_valid_mask = gt_valid_mask & (end_points["grasp_angle_label_tok"][i] >= 0)
            gt_valid_mask = gt_valid_mask & (end_points["grasp_depthoff_label_tok"][i] >= 0)
            gt_valid_mask = gt_valid_mask & torch.isfinite(end_points["grasp_width_label_tok"][i])
            gt_valid_mask = gt_valid_mask & (end_points["grasp_width_label_tok"][i] >= 0)

            if mode == "E2B":
                gt_valid_mask = gt_valid_mask & (gt_depth[i] > 0)
                
            if mode in ["E3", "E3B"]:
                gt_valid_mask = gt_valid_mask & torch.isfinite(gt_center_xyz[i]).all(dim=-1)
                gt_valid_mask = gt_valid_mask & (gt_center_xyz[i, :, 2] > 0)

            tu_all = torch.arange(W, device=device).view(1, W).expand(H, W).reshape(-1)
            tv_all = torch.arange(H, device=device).view(H, 1).expand(H, W).reshape(-1)
            tu_all = torch.clamp((tu_all.float() / float(view_stride)).floor().long(), 0, Wv - 1)
            tv_all = torch.clamp((tv_all.float() / float(view_stride)).floor().long(), 0, Hv - 1)
            low_idx_all = tv_all * Wv + tu_all
            gt_view_valid_all = gt_view_valid[i].gather(0, low_idx_all)
            gt_valid_mask = gt_valid_mask & gt_view_valid_all

            idx_valid = torch.nonzero(gt_valid_mask, as_tuple=False).squeeze(1)

            if idx_valid.numel() == 0:
                grasp_preds.append(torch.zeros((0, 17), device=device, dtype=torch.float32))
                continue

            if mode in ["E2", "E3"]:
                gt_score_valid = gt_score_label[i].gather(0, idx_valid)
                Ksel = min(pred_topk, idx_valid.numel())
                rank = torch.topk(gt_score_valid, k=Ksel, dim=0).indices
                sel_idx = idx_valid.index_select(0, rank)

            elif mode == "E2B":
                # FPS on A9 centers:
                # use token pixel + GT depth backprojection, NOT GT center xyz
                z_valid = gt_depth[i].gather(0, idx_valid).clamp_min(1e-6)

                u_valid = (idx_valid % W).float()
                v_valid = (idx_valid // W).float()

                fx, fy = K_i[0, 0], K_i[1, 1]
                cx, cy = K_i[0, 2], K_i[1, 2]

                x_valid = (u_valid - cx) / fx * z_valid
                y_valid = (v_valid - cy) / fy * z_valid
                xyz_valid = torch.stack([x_valid, y_valid, z_valid], dim=-1)

                score_valid = gt_score_label[i].gather(0, idx_valid)
                start_idx = torch.argmax(score_valid).view(1)
                fps_local = fps_indices_3d(xyz_valid, pred_topk, start_idx=start_idx)
                sel_idx = idx_valid.index_select(0, fps_local)

            elif mode == "E3B":
                xyz_valid = gt_center_xyz[i].index_select(0, idx_valid)
                score_valid = gt_score_label[i].gather(0, idx_valid)

                start_idx = torch.argmax(score_valid).view(1)
                fps_local = fps_indices_3d(xyz_valid, pred_topk, start_idx=start_idx)
                sel_idx = idx_valid.index_select(0, fps_local)

        else:
            sel_idx = None
            if use_fps_tokens and ("token_sel_idx_fps" in end_points):
                sel_idx = end_points["token_sel_idx_fps"][i].long()
            elif "token_sel_idx_score" in end_points:
                sel_idx = end_points["token_sel_idx_score"][i].long()
            elif "token_sel_idx" in end_points:
                sel_idx = end_points["token_sel_idx"][i].long()
            else:
                if mask_i.any():
                    score_valid = score_i.clone()
                    score_valid[~mask_i] = -1e9
                    Ksel = min(pred_topk, int(mask_i.sum().item()))
                    sel_idx = torch.topk(score_valid, k=Ksel, dim=0).indices
                else:
                    score_valid = score_i.clone()
                    score_valid[~token_valid_mask[i]] = -1e9
                    Ksel = min(pred_topk, int(token_valid_mask[i].sum().item()))
                    sel_idx = torch.topk(score_valid, k=Ksel, dim=0).indices

            sel_idx = torch.unique(sel_idx)
            sel_idx = sel_idx[token_valid_mask[i].gather(0, sel_idx)]

            if sel_idx.numel() == 0:
                grasp_preds.append(torch.zeros((0, 17), device=device, dtype=torch.float32))
                continue

            if sel_idx.numel() > pred_topk:
                sel_scores = score_i.gather(0, sel_idx)
                rank = torch.topk(sel_scores, k=pred_topk, dim=0).indices
                sel_idx = sel_idx.index_select(0, rank)

        # --------------------------------------------------
        # 2) center (z)
        # --------------------------------------------------
        u = (sel_idx % W).float()
        v = (sel_idx // W).float()

        if mode in ["E3", "E3B"]:
            grasp_center = gt_center_xyz[i].index_select(0, sel_idx)
            z = grasp_center[:, 2].clamp_min(1e-6)
        else:
            if mode in ["A3", "A9", "E2", "E2B"]:
                z = gt_depth[i].gather(0, sel_idx).clamp_min(1e-6)
            else:
                z = grasp_z_pred[i].gather(0, sel_idx).clamp_min(1e-6)

            fx, fy = K_i[0, 0], K_i[1, 1]
            cx, cy = K_i[0, 2], K_i[1, 2]
            x = (u - cx) / fx * z
            y = (v - cy) / fy * z
            grasp_center = torch.stack([x, y, z], dim=-1)

        # --------------------------------------------------
        # 3) final score
        # --------------------------------------------------
        if mode in ["E1", "E2", "E2B", "E3", "E3B"]:
            grasp_score = gt_score_label[i].gather(0, sel_idx).view(-1, 1)
        else:
            grasp_score = score_i.gather(0, sel_idx).view(-1, 1)

        # --------------------------------------------------
        # 4) width
        # --------------------------------------------------
        if mode in ["A8", "A9", "E2", "E2B", "E3", "E3B"]:
            grasp_width = gt_width_label[i].gather(0, sel_idx)
        else:
            grasp_width = grasp_width_pred[i].gather(0, sel_idx) / float(width_scale)
            grasp_width = width_expand_ratio * grasp_width

        grasp_width = torch.clamp(
            grasp_width,
            min=0.0,
            max=float(getattr(cfgs, "grasp_max_width", 0.1))
        ).view(-1, 1)

        # --------------------------------------------------
        # 5) depthoff
        # --------------------------------------------------
        if mode in ["A7", "A9", "E2", "E2B", "E3", "E3B"]:
            gt_depth_idx = gt_depthoff_label[i].gather(0, sel_idx)
            valid_depth = gt_depth_idx >= 0
            if valid_depth.any():
                sel_idx = sel_idx[valid_depth]
                grasp_center = grasp_center[valid_depth]
                grasp_score = grasp_score[valid_depth]
                grasp_width = grasp_width[valid_depth]
                u = u[valid_depth]
                v = v[valid_depth]
                gt_depth_idx = gt_depth_idx[valid_depth]
            else:
                grasp_preds.append(torch.zeros((0, 17), device=device, dtype=torch.float32))
                continue

            grasp_depth = ((gt_depth_idx + 1).to(grasp_center.dtype) * depth_bin_size).view(-1, 1)
        else:
            depth_logits_sel = grasp_depthoff_pred[i].index_select(1, sel_idx)
            depth_idx = torch.argmax(depth_logits_sel, dim=0)
            grasp_depth = ((depth_idx + 1).to(grasp_center.dtype) * depth_bin_size).view(-1, 1)

        # --------------------------------------------------
        # 6) angle / view
        # --------------------------------------------------
        if mode in ["A1", "A2", "A3", "A7", "A8", "E1"]:
            angle_logits_sel = grasp_angle_pred[i].index_select(1, sel_idx)
            angle_idx = torch.argmax(angle_logits_sel, dim=0)
            grasp_angle = angle_idx.to(grasp_center.dtype) * (np.pi / float(num_angle))

        if mode in ["A1", "A3", "A7", "A8", "E1"]:
            tu = torch.clamp((u / float(view_stride)).floor().long(), 0, Wv - 1)
            tv = torch.clamp((v / float(view_stride)).floor().long(), 0, Hv - 1)
            view_logits_sel = view_score_map[i, :, tv, tu]
            top_view_idx = torch.argmax(view_logits_sel, dim=0)
            view_xyz = template_views.index_select(0, top_view_idx.long())
            approaching = -view_xyz
            approaching = F.normalize(approaching, dim=-1)

        elif mode == "A2":
            grasp_angle = angle_idx.to(grasp_center.dtype) * (np.pi / float(num_angle))
            approaching = F.normalize(grasp_center, dim=-1)

        elif mode in ["A6", "A9", "E2", "E2B", "E3", "E3B"]:
            gt_angle_sel = gt_angle_label[i].gather(0, sel_idx)

            tu = torch.clamp((u / float(view_stride)).floor().long(), 0, Wv - 1)
            tv = torch.clamp((v / float(view_stride)).floor().long(), 0, Hv - 1)
            low_idx = tv * Wv + tu

            gt_view_valid_sel = gt_view_valid[i].gather(0, low_idx)
            gt_view_logits_sel = gt_view_label[i].index_select(0, low_idx)      # (M,V)
            gt_top_view_idx = torch.argmax(gt_view_logits_sel, dim=1)

            valid_oracle = (gt_angle_sel >= 0) & gt_view_valid_sel
            if valid_oracle.any():
                grasp_center = grasp_center[valid_oracle]
                grasp_score = grasp_score[valid_oracle]
                grasp_width = grasp_width[valid_oracle]
                grasp_depth = grasp_depth[valid_oracle]
                gt_angle_sel = gt_angle_sel[valid_oracle]
                gt_top_view_idx = gt_top_view_idx[valid_oracle]
            else:
                grasp_preds.append(torch.zeros((0, 17), device=device, dtype=torch.float32))
                continue

            grasp_angle = gt_angle_sel.to(grasp_center.dtype) * (np.pi / float(num_angle))
            view_xyz = template_views.index_select(0, gt_top_view_idx.long())
            approaching = -view_xyz
            approaching = F.normalize(approaching, dim=-1)

        # --------------------------------------------------
        # 7) compose rotation
        # --------------------------------------------------
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(grasp_rot.shape[0], 9)

        # --------------------------------------------------
        # 8) assemble
        # --------------------------------------------------
        M = grasp_rot.shape[0]
        grasp_height_t = float(grasp_height) * torch.ones((M, 1), device=device, dtype=grasp_center.dtype)
        obj_ids = -1.0 * torch.ones((M, 1), device=device, dtype=grasp_center.dtype)

        grasp_pred = torch.cat(
            [
                grasp_score,
                grasp_width,
                grasp_height_t,
                grasp_depth,
                grasp_rot,
                grasp_center,
                obj_ids,
            ],
            dim=-1
        )

        rank = torch.argsort(grasp_pred[:, 0], descending=True)
        grasp_pred = grasp_pred.index_select(0, rank)
        grasp_preds.append(grasp_pred)

    return grasp_preds


# =========================================================
# dataset / model
# =========================================================
if cfgs.multi_modal:
    TEST_DATASET = GraspNetMultiDataset(
        cfgs.dataset_root,
        split='{}'.format(cfgs.test_mode),
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=True
    )
else:
    TEST_DATASET = GraspNetDataset(
        cfgs.dataset_root,
        split='{}'.format(cfgs.test_mode),
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=True
    )

sampled_indices = build_sampled_indices(TEST_DATASET, SAMPLE_INTERVAL)
TEST_SUBSET = Subset(TEST_DATASET, sampled_indices)

print(f"Total samples in split: {len(TEST_DATASET)}")
print(f"Sample interval: {SAMPLE_INTERVAL}")
print(f"Sampled samples: {len(TEST_SUBSET)}")

TEST_DATALOADER = DataLoader(
    TEST_SUBSET,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn
)

net = economicgrasp_c3(
    min_depth=cfgs.min_depth,
    max_depth=cfgs.max_depth,
    is_training=False,
    vis_dir=None,
    vis_every=100,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

checkpoint = torch.load(cfgs.checkpoint_path)
try:
    net.load_state_dict(checkpoint["model_state_dict"])
except Exception:
    net.load_state_dict(checkpoint)

print(f"-> loaded checkpoint {cfgs.checkpoint_path}")
print(f"-> oracle mode: {ORACLE_MODE}")
print(f"-> split: {TEST_SPLIT}")
print(f"-> sample_interval: {SAMPLE_INTERVAL}")
# print(f"-> save root: {SAVE_ROOT}")


# =========================================================
# inference
# =========================================================
def inference():
    batch_interval = 20
    net.eval()
    tic = time.time()

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):

        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            elif key in ['scene_idx', 'anno_idx', 'dataset_idx']:
                # keep on cpu / scalar-friendly
                pass
            else:
                batch_data[key] = batch_data[key].to(device)

        with torch.no_grad():
            end_points = net(batch_data)
            end_points = build_gt_oracle_labels(end_points)
            grasp_preds = pred_decode_c3_oracle(end_points, mode=ORACLE_MODE, use_fps_tokens=USE_FPS_TOKENS)

        # batch_size=1
        preds = grasp_preds[0].detach().cpu().numpy()

        # raw / no-collision predictions
        gg_raw = GraspGroup(preds.copy())
        num_raw = len(gg_raw)

        scene_idx = to_int_scalar(batch_data['scene_idx'])
        anno_idx = to_int_scalar(batch_data['anno_idx'])
        dataset_idx = to_int_scalar(batch_data['dataset_idx'])

        # save no-collision version if needed
        scene_name = f"scene_{scene_idx:04d}"

        if SAVE_NOCOLLISION_NPY:
            save_dir_nc = os.path.join(
                cfgs.save_dir + NOCOLLISION_SUFFIX,
                scene_name,
                cfgs.camera
            )
            os.makedirs(save_dir_nc, exist_ok=True)
            save_path_nc = os.path.join(save_dir_nc, f"{anno_idx:04d}.npy")
            gg_raw.save_npy(save_path_nc)

        # collision detection
        num_collision_removed = 0
        collision_ratio = 0.0

        gg = gg_raw
        if cfgs.collision_thresh > 0 and num_raw > 0:
            cloud, _ = TEST_DATASET.get_data(dataset_idx, return_raw_cloud=True)
            mfcdetector = ModelFreeCollisionDetectorTorch(
                cloud.reshape(-1, 3),
                voxel_size=cfgs.collision_voxel_size
            )
            collision_mask = mfcdetector.detect(
                gg_raw,
                approach_dist=0.05,
                collision_thresh=cfgs.collision_thresh
            )
            collision_mask = collision_mask.detach().cpu().numpy()

            num_collision_removed = int(collision_mask.sum())
            collision_ratio = float(num_collision_removed / max(1, num_raw))
            gg = gg_raw[~collision_mask]

        num_after_collision = len(gg)

        # jsonl logging
        log_record = {
            "oracle_mode": ORACLE_MODE,
            "scene_idx": int(scene_idx),
            "anno_idx": int(anno_idx),
            "dataset_idx": int(dataset_idx),
            "num_pred_raw": int(num_raw),
            "num_collision_removed": int(num_collision_removed),
            "num_pred_after_collision": int(num_after_collision),
            "collision_ratio": float(collision_ratio),
            "collision_thresh": float(cfgs.collision_thresh),
            "collision_voxel_size": float(cfgs.collision_voxel_size),
        }

        append_jsonl(
            os.path.join(DEBUG_LOG_DIR, "collision_stats.jsonl"),
            log_record
        )

        if batch_idx == 0 and ORACLE_MODE == "E3":
            print("[dbg] has grasp_center_xyz_label_tok:", "grasp_center_xyz_label_tok" in end_points)
            if "grasp_center_xyz_label_tok" in end_points:
                c = end_points["grasp_center_xyz_label_tok"]
                print("[dbg] grasp_center_xyz_label_tok shape:", tuple(c.shape))
                print("[dbg] center z valid mean:", (c[..., 2] > 0).float().mean().item())
                
        # save by scene_idx / anno_idx
        scene_name = f"scene_{scene_idx:04d}"
        save_dir = os.path.join(cfgs.save_dir, scene_name, cfgs.camera)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{anno_idx:04d}.npy")
        gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print(
                f"Eval batch: {batch_idx}, "
                f"scene={scene_idx:04d}, anno={anno_idx:04d}, "
                f"time: {(toc - tic) / batch_interval:.6f}s"
            )
            tic = time.time()

if __name__ == "__main__":
    inference()