
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arguments import cfgs
from scipy.optimize import linear_sum_assignment
from pytorch3d.ops.knn import knn_points
from torch_linear_assignment import batch_linear_assignment

from utils.loss_utils import (
    batch_viewpoint_params_to_matrix,
    transform_point_cloud,
    generate_grasp_views,
)

def _build_scene_view_id_map(pose: torch.Tensor, num_view: int) -> torch.Tensor:
    templates = generate_grasp_views(num_view).to(pose.device)
    views_trans = transform_point_cloud(templates, pose[:3, :3], '3x3')
    _, nn_inds, _ = knn_points(views_trans.unsqueeze(0), templates.unsqueeze(0), K=1)
    return nn_inds.squeeze(0).squeeze(-1).long()

def build_scene_gt_grasp_set(
    end_points: Dict,
    score_threshold: float = 0.0,
    width_scale: float = 10.0,
) -> List[Dict[str, torch.Tensor]]:
    """
    For each GT grasp point, keep only the best valid grasp.
    Labels are converted to scene frame.
    """
    batch_size = len(end_points['object_poses_list'])
    gt_sets: List[Dict[str, torch.Tensor]] = []

    for b in range(batch_size):
        poses = end_points['object_poses_list'][b]
        centers_all, view_all = [], []
        angle_all, depth_all = [], []
        width_all, score_cls_all, score_val_all = [], [], []

        for obj_idx, pose in enumerate(poses):
            pose = pose.to(end_points['img'].device)
            grasp_points = end_points['grasp_points_list'][b][obj_idx].to(pose.device)
            grasp_rotations = end_points['grasp_rotations_list'][b][obj_idx].long().to(pose.device)
            grasp_depth = end_points['grasp_depth_list'][b][obj_idx].long().to(pose.device)
            grasp_scores = end_points['grasp_scores_list'][b][obj_idx].float().to(pose.device)
            grasp_widths = end_points['grasp_widths_list'][b][obj_idx].float().to(pose.device)
            top_view_index = end_points['top_view_index_list'][b][obj_idx].long().to(pose.device)

            if grasp_points.numel() == 0:
                continue

            grasp_points_scene = transform_point_cloud(grasp_points, pose, '3x4')
            view_id_map = _build_scene_view_id_map(pose, cfgs.num_view)
            top_view_index_clamped = top_view_index.clamp(min=0, max=cfgs.num_view - 1)
            scene_view_ids = view_id_map[top_view_index_clamped]

            valid = (top_view_index >= 0) & (grasp_scores > score_threshold)
            if not valid.any():
                continue

            masked_scores = torch.where(valid, grasp_scores, torch.full_like(grasp_scores, -1e9))
            best_k = masked_scores.argmax(dim=1)
            best_valid = valid.any(dim=1)
            if not best_valid.any():
                continue

            ar = torch.arange(grasp_points.shape[0], device=pose.device)
            centers_all.append(grasp_points_scene[best_valid])
            view_all.append(scene_view_ids[ar, best_k][best_valid])
            angle_all.append(grasp_rotations[ar, best_k][best_valid])
            depth_all.append(grasp_depth[ar, best_k][best_valid])
            width_all.append((grasp_widths[ar, best_k][best_valid] * width_scale).float())
            score_val = grasp_scores[ar, best_k][best_valid].float()
            score_val_all.append(score_val)
            score_cls_all.append(((score_val * 10.0) / 2.0).long().clamp(0, 5))

        if len(centers_all) == 0:
            dev = end_points['img'].device
            gt_sets.append({
                'center': torch.empty((0, 3), device=dev, dtype=torch.float32),
                'view': torch.empty((0,), device=dev, dtype=torch.long),
                'angle': torch.empty((0,), device=dev, dtype=torch.long),
                'depth': torch.empty((0,), device=dev, dtype=torch.long),
                'width': torch.empty((0,), device=dev, dtype=torch.float32),
                'score_cls': torch.empty((0,), device=dev, dtype=torch.long),
                'score_val': torch.empty((0,), device=dev, dtype=torch.float32),
            })
        else:
            gt_sets.append({
                'center': torch.cat(centers_all, dim=0),
                'view': torch.cat(view_all, dim=0),
                'angle': torch.cat(angle_all, dim=0),
                'depth': torch.cat(depth_all, dim=0),
                'width': torch.cat(width_all, dim=0),
                'score_cls': torch.cat(score_cls_all, dim=0),
                'score_val': torch.cat(score_val_all, dim=0),
            })

    return gt_sets


# -----------------------------------------------------------------------------
# Hungarian matching
# -----------------------------------------------------------------------------
@torch.no_grad()
def hungarian_match_single_grasp(
    end_points: Dict,
    gt_sets: Optional[List[Dict[str, torch.Tensor]]] = None,
    center_radius: float = 0.03,
    lambda_center: float = 5.0,
    lambda_view: float = 1.0,
    lambda_angle: float = 1.0,
    lambda_depth: float = 1.0,
    lambda_width: float = 0.5,
    lambda_score: float = 0.5,
    max_gt: int = 256,
    invalid_cost: float = 1e6,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    GPU batched matching with torch-linear-assignment.

    Notes
    -----
    1) We prune GTs by center radius before matching.
    2) We cap GT count by max_gt (recommended <= num_query).
    3) We pad GT columns across the batch, then filter padded assignments out.
    4) Returned gt_sets are the *pruned/capped* GT sets aligned with gt_idx.
    """

    def _empty_like_gt(gt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        n0 = gt["center"].shape[0]
        for k, v in gt.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == n0:
                out[k] = v[:0]
            else:
                out[k] = v
        out["orig_idx"] = torch.empty((0,), device=gt["center"].device, dtype=torch.long)
        return out

    def _slice_gt(gt: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        if idx.numel() == 0:
            return _empty_like_gt(gt)
        out = {}
        n0 = gt["center"].shape[0]
        for k, v in gt.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == n0:
                out[k] = v.index_select(0, idx)
            else:
                out[k] = v
        out["orig_idx"] = idx
        return out

    if gt_sets is None:
        gt_sets = build_scene_gt_grasp_set(end_points)

    pred_center = end_points["xyz_graspable"].detach()  # (B,Q,3)
    view_logp   = F.log_softmax(end_points["view_score"].detach(), dim=-1)  # (B,Q,V)
    angle_logp  = F.log_softmax(end_points["grasp_angle_pred"].detach().transpose(1, 2), dim=-1)  # (B,Q,A)
    depth_logp  = F.log_softmax(end_points["grasp_depth_pred"].detach().transpose(1, 2), dim=-1)  # (B,Q,D)
    score_logp  = F.log_softmax(end_points["grasp_score_pred"].detach().transpose(1, 2), dim=-1)  # (B,Q,S)
    pred_width  = end_points["grasp_width_pred"].detach().squeeze(1)  # (B,Q)

    B, Q, _ = pred_center.shape
    dev = pred_center.device
    max_gt = min(int(max_gt), int(Q))  # important: keep rows >= cols behavior

    # ------------------------------------------------------------------
    # 1) prune/cap GT per sample
    # ------------------------------------------------------------------
    pruned_gt_sets: List[Dict[str, torch.Tensor]] = []
    raw_gt_counts = []
    pruned_gt_counts = []

    for b in range(B):
        gt = gt_sets[b]
        G = int(gt["center"].shape[0])
        raw_gt_counts.append(float(G))

        if G == 0 or Q == 0:
            pruned_gt_sets.append(_empty_like_gt(gt))
            pruned_gt_counts.append(0.0)
            continue

        # center-based pruning
        center_dist_bg = torch.cdist(pred_center[b], gt["center"], p=2)  # (Q,G)
        min_d = center_dist_bg.min(dim=0).values  # (G,)

        if center_radius is not None and center_radius > 0:
            keep = min_d <= center_radius
        else:
            keep = torch.ones_like(min_d, dtype=torch.bool)

        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)

        # fallback: if nothing survives radius pruning, keep a tiny nearest subset
        if keep_idx.numel() == 0:
            k = min(max_gt, G)
            keep_idx = torch.topk(min_d, k=k, largest=False).indices

        # cap GT size
        if keep_idx.numel() > max_gt:
            local_d = min_d.index_select(0, keep_idx)
            top_local = torch.topk(local_d, k=max_gt, largest=False).indices
            keep_idx = keep_idx.index_select(0, top_local)

        keep_idx = keep_idx.long().to(gt["center"].device)
        gt_sub = _slice_gt(gt, keep_idx)

        pruned_gt_sets.append(gt_sub)
        pruned_gt_counts.append(float(gt_sub["center"].shape[0]))

    Gmax = max((int(gt["center"].shape[0]) for gt in pruned_gt_sets), default=0)

    # no GT at all after pruning
    if Gmax == 0 or Q == 0:
        batch_matches = [{
            "query_idx": torch.empty((0,), device=dev, dtype=torch.long),
            "gt_idx": torch.empty((0,), device=dev, dtype=torch.long),
        } for _ in range(B)]
        end_points["D: Num GT Raw"] = torch.tensor(raw_gt_counts, device=dev).mean() if len(raw_gt_counts) > 0 else torch.zeros((), device=dev)
        end_points["D: Num GT Pruned"] = torch.tensor(pruned_gt_counts, device=dev).mean() if len(pruned_gt_counts) > 0 else torch.zeros((), device=dev)
        end_points["D: Num Matched"] = torch.zeros((), device=dev)
        return batch_matches, pruned_gt_sets

    # ------------------------------------------------------------------
    # 2) pad GT to batch them
    # ------------------------------------------------------------------
    gt_center_pad = pred_center.new_zeros((B, Gmax, 3))
    gt_view_pad   = torch.zeros((B, Gmax), device=dev, dtype=torch.long)
    gt_angle_pad  = torch.zeros((B, Gmax), device=dev, dtype=torch.long)
    gt_depth_pad  = torch.zeros((B, Gmax), device=dev, dtype=torch.long)
    gt_score_pad  = torch.zeros((B, Gmax), device=dev, dtype=torch.long)
    gt_width_pad  = pred_width.new_zeros((B, Gmax))
    gt_valid_pad  = torch.zeros((B, Gmax), device=dev, dtype=torch.bool)

    for b in range(B):
        gt = pruned_gt_sets[b]
        g = int(gt["center"].shape[0])
        if g == 0:
            continue
        gt_center_pad[b, :g] = gt["center"].to(dev)
        gt_view_pad[b, :g]   = gt["view"].to(dev)
        gt_angle_pad[b, :g]  = gt["angle"].to(dev)
        gt_depth_pad[b, :g]  = gt["depth"].to(dev)
        gt_score_pad[b, :g]  = gt["score_cls"].to(dev)
        gt_width_pad[b, :g]  = gt["width"].to(dev)
        gt_valid_pad[b, :g]  = True

    # ------------------------------------------------------------------
    # 3) build batched cost on GPU
    # ------------------------------------------------------------------
    center_dist = torch.cdist(pred_center, gt_center_pad, p=2)  # (B,Q,G)

    gather_view_idx  = gt_view_pad.unsqueeze(1).expand(B, Q, Gmax)
    gather_angle_idx = gt_angle_pad.unsqueeze(1).expand(B, Q, Gmax)
    gather_depth_idx = gt_depth_pad.unsqueeze(1).expand(B, Q, Gmax)
    gather_score_idx = gt_score_pad.unsqueeze(1).expand(B, Q, Gmax)

    view_cost  = -torch.gather(view_logp,  2, gather_view_idx)
    angle_cost = -torch.gather(angle_logp, 2, gather_angle_idx)
    depth_cost = -torch.gather(depth_logp, 2, gather_depth_idx)
    score_cost = -torch.gather(score_logp, 2, gather_score_idx)
    width_cost = (pred_width.unsqueeze(-1) - gt_width_pad.unsqueeze(1)).abs()

    total_cost = (
        lambda_center * center_dist
        + lambda_view * view_cost
        + lambda_angle * angle_cost
        + lambda_depth * depth_cost
        + lambda_width * width_cost
        + lambda_score * score_cost
    )

    # mask padded GT columns
    total_cost = total_cost.masked_fill(~gt_valid_pad.unsqueeze(1), float(invalid_cost))

    # optional hard radius gate
    if center_radius is not None and center_radius > 0:
        total_cost = total_cost + (center_dist > center_radius).to(total_cost.dtype) * float(invalid_cost)

    # ------------------------------------------------------------------
    # 4) solve batched LAP on GPU
    # assignment: (B,Q), each row = matched GT col or -1
    # ------------------------------------------------------------------
    assignment = batch_linear_assignment(total_cost)

    # ------------------------------------------------------------------
    # 5) parse assignments and filter invalid/padded cols
    # ------------------------------------------------------------------
    batch_matches: List[Dict[str, torch.Tensor]] = []
    matched_counts = []

    for b in range(B):
        a = assignment[b].long()  # (Q,)
        if a.numel() == 0:
            batch_matches.append({
                "query_idx": torch.empty((0,), device=dev, dtype=torch.long),
                "gt_idx": torch.empty((0,), device=dev, dtype=torch.long),
            })
            matched_counts.append(torch.tensor(0.0, device=dev))
            continue

        q_idx = torch.nonzero(a >= 0, as_tuple=False).squeeze(1)
        if q_idx.numel() == 0:
            batch_matches.append({
                "query_idx": torch.empty((0,), device=dev, dtype=torch.long),
                "gt_idx": torch.empty((0,), device=dev, dtype=torch.long),
            })
            matched_counts.append(torch.tensor(0.0, device=dev))
            continue

        g_idx = a.index_select(0, q_idx)  # local indices into pruned_gt_sets[b]

        keep = gt_valid_pad[b].index_select(0, g_idx)
        if center_radius is not None and center_radius > 0:
            keep = keep & (center_dist[b, q_idx, g_idx] <= center_radius)

        q_idx = q_idx[keep]
        g_idx = g_idx[keep]

        batch_matches.append({
            "query_idx": q_idx,
            "gt_idx": g_idx,
        })
        matched_counts.append(torch.tensor(float(q_idx.numel()), device=dev))

    end_points["D: Num GT Raw"] = torch.tensor(raw_gt_counts, device=dev).mean() if len(raw_gt_counts) > 0 else torch.zeros((), device=dev)
    end_points["D: Num GT Pruned"] = torch.tensor(pruned_gt_counts, device=dev).mean() if len(pruned_gt_counts) > 0 else torch.zeros((), device=dev)
    end_points["D: Num Matched"] = torch.stack(matched_counts).mean() if len(matched_counts) > 0 else torch.zeros((), device=dev)

    return batch_matches, pruned_gt_sets


def compute_query_presence_loss(end_points: Dict, batch_matches: List[Dict[str, torch.Tensor]]):
    logits = end_points['query_presence_logits']
    B, _, M = logits.shape
    target = torch.zeros((B, M), device=logits.device, dtype=torch.long)
    for b, match in enumerate(batch_matches):
        if match['query_idx'].numel() > 0:
            target[b, match['query_idx']] = 1
    loss = F.cross_entropy(logits, target, reduction='mean')
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == target).float().mean()
    end_points['query_presence_target'] = target
    end_points['B: Presence Loss'] = loss
    end_points['D: Presence Acc'] = acc
    return loss, end_points


def compute_matched_query_losses(
    end_points: Dict,
    batch_matches: List[Dict[str, torch.Tensor]],
    gt_sets: List[Dict[str, torch.Tensor]],
):
    view_logits = end_points['view_score']
    angle_logits = end_points['grasp_angle_pred'].transpose(1, 2)
    depth_logits = end_points['grasp_depth_pred'].transpose(1, 2)
    score_logits = end_points['grasp_score_pred'].transpose(1, 2)
    width_pred = end_points['grasp_width_pred'].squeeze(1)

    view_losses, angle_losses, depth_losses = [], [], []
    score_losses, width_losses = [], []
    matched_counts = []

    for b, match in enumerate(batch_matches):
        q_idx = match['query_idx']
        g_idx = match['gt_idx']
        matched_counts.append(torch.tensor(float(q_idx.numel()), device=view_logits.device))
        if q_idx.numel() == 0:
            continue
        gt = gt_sets[b]
        view_losses.append(F.cross_entropy(view_logits[b, q_idx], gt['view'][g_idx], reduction='mean'))
        angle_losses.append(F.cross_entropy(angle_logits[b, q_idx], gt['angle'][g_idx], reduction='mean'))
        depth_losses.append(F.cross_entropy(depth_logits[b, q_idx], gt['depth'][g_idx], reduction='mean'))
        score_losses.append(F.cross_entropy(score_logits[b, q_idx], gt['score_cls'][g_idx], reduction='mean'))
        width_losses.append(F.smooth_l1_loss(width_pred[b, q_idx], gt['width'][g_idx], reduction='mean'))

    def _safe_mean(loss_list, ref_tensor):
        if len(loss_list) == 0:
            return torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)
        return torch.stack(loss_list).mean()

    view_loss = _safe_mean(view_losses, view_logits)
    angle_loss = _safe_mean(angle_losses, view_logits)
    depth_loss = _safe_mean(depth_losses, view_logits)
    score_loss = _safe_mean(score_losses, view_logits)
    width_loss = _safe_mean(width_losses, view_logits)
    matched_mean = torch.stack(matched_counts).mean() if len(matched_counts) > 0 else torch.zeros((), device=view_logits.device)

    end_points['B: View Loss'] = view_loss
    end_points['B: Angle Loss'] = angle_loss
    end_points['B: Depth Loss'] = depth_loss
    end_points['B: Score Loss'] = score_loss
    end_points['B: Width Loss'] = width_loss
    end_points['D: Matched Queries'] = matched_mean
    return {
        'view_loss': view_loss,
        'angle_loss': angle_loss,
        'depth_loss': depth_loss,
        'score_loss': score_loss,
        'width_loss': width_loss,
    }, end_points
    

def get_loss_query(
    end_points: Dict,
    center_radius: float = 0.03,
    lambda_center: float = 5.0,
    lambda_view_cost: float = 1.0,
    lambda_angle_cost: float = 1.0,
    lambda_depth_cost: float = 1.0,
    lambda_width_cost: float = 0.5,
    lambda_score_cost: float = 0.5,
):
    """
    Simplified no-presence loss:
      dense losses + Hungarian matching + matched-query grasp losses
    """
    depth_reg_loss, end_points = compute_depth_reg_loss(end_points)
    objectness_loss, end_points = compute_objectness_loss_tok(end_points)
    graspness_loss, end_points = compute_graspness_loss_tok(end_points)

    batch_matches, gt_sets = hungarian_match_single_grasp(
        end_points=end_points,
        gt_sets=None,
        center_radius=center_radius,
        lambda_center=lambda_center,
        lambda_view=lambda_view_cost,
        lambda_angle=lambda_angle_cost,
        lambda_depth=lambda_depth_cost,
        lambda_width=lambda_width_cost,
        lambda_score=lambda_score_cost,
    )
    end_points['batch_gt_grasp_set'] = gt_sets
    end_points['batch_query_match'] = batch_matches

    B, M = end_points['xyz_graspable'].shape[:2]
    dev = end_points['xyz_graspable'].device
    match_mask = torch.zeros((B, M), device=dev, dtype=torch.bool)
    match_gt_idx = torch.full((B, M), -1, device=dev, dtype=torch.long)
    gt_counts = []
    matched_counts = []
    for b, match in enumerate(batch_matches):
        q_idx = match['query_idx']
        g_idx = match['gt_idx']
        if q_idx.numel() > 0:
            match_mask[b, q_idx] = True
            match_gt_idx[b, q_idx] = g_idx
        gt_counts.append(torch.tensor(float(gt_sets[b]['center'].shape[0]), device=dev))
        matched_counts.append(torch.tensor(float(q_idx.numel()), device=dev))

    end_points['query_match_mask'] = match_mask
    end_points['query_match_gt_idx'] = match_gt_idx
    end_points['D: Num GT'] = torch.stack(gt_counts).mean() if len(gt_counts) > 0 else torch.zeros((), device=dev)
    end_points['D: Num Matched'] = torch.stack(matched_counts).mean() if len(matched_counts) > 0 else torch.zeros((), device=dev)

    matched_losses, end_points = compute_matched_query_losses(end_points, batch_matches, gt_sets)

    obj_loss = cfgs.objectness_loss_weight * objectness_loss
    dense_graspness_loss = cfgs.graspness_loss_weight * graspness_loss
    query_grasp_loss = (
        cfgs.view_loss_weight * matched_losses['view_loss']
        + cfgs.angle_loss_weight * matched_losses['angle_loss']
        + cfgs.depth_loss_weight * matched_losses['depth_loss']
        + cfgs.score_loss_weight * matched_losses['score_loss']
        + cfgs.width_loss_weight * matched_losses['width_loss']
    )
    depth_reg_loss_w = cfgs.depth_prob_loss_weight * depth_reg_loss

    loss = obj_loss + dense_graspness_loss + query_grasp_loss + depth_reg_loss_w

    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Dense Graspness Loss'] = graspness_loss
    end_points['A: Grasp Loss'] = query_grasp_loss
    end_points['A: DepthReg Loss'] = depth_reg_loss
    end_points['A: Overall Loss'] = loss
    return loss, end_points


def compute_objectness_loss_tok(end_points):
    # objectness_score: (B,2,Ntok)
    # objectness_label_tok: (B,Ntok) with -1 for invalid
    score = end_points["objectness_score"]
    label = end_points["objectness_label_tok"].long()

    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
    loss = criterion(score, label)
    end_points["B: Objectness Loss"] = loss

    with torch.no_grad():
        valid = (label != -1)
        if valid.any():
            pred = torch.argmax(score, 1)
            end_points["D: Objectness Acc"] = (pred[valid] == label[valid]).float().mean()
        else:
            end_points["D: Objectness Acc"] = torch.zeros((), device=score.device)
    return loss, end_points


def compute_graspness_loss_tok(end_points):
    pred = end_points["graspness_score"].squeeze(1)  # (B,Ntok)
    gt   = end_points["graspness_label_tok"]         # (B,Ntok) or (B,Ntok,1)
    if gt.dim() == 3:
        gt = gt.squeeze(-1)

    obj = end_points["objectness_label_tok"].long()  # (B,Ntok), -1 invalid
    valid_tok = (obj != -1)
    obj_tok = (obj == 1)

    mask = valid_tok & obj_tok
    if "token_valid_mask" in end_points:
        mask = mask & end_points["token_valid_mask"].bool()

    criterion = nn.SmoothL1Loss(reduction="none")
    loss_map = criterion(pred, gt.to(pred))

    loss = loss_map[mask].mean() if mask.any() else 0.0 * pred.sum()
    end_points["B: Graspness Loss"] = loss
    return loss, end_points


def compute_depth_reg_loss(
    end_points,
    depth_min: float = 0.2,
    depth_max: float = 1.0,
):
    """
    Full-map depth regression loss.

    Required:
      - depth_map_pred: (B,1,448,448) or (B,448,448)
      - gt_depth_m:     (B,1,448,448) or (B,448,448)

    Valid mask:
      gt in (0, +inf) AND gt within [depth_min, depth_max]
      (只用 GT 做 mask，避免 pred 参与 mask 导致训练不稳定)

    Output:
      - end_points["B: DepthReg Loss"]
      - some debug stats
    """
    if ("depth_map_pred" not in end_points) or ("gt_depth_m" not in end_points):
        dev = None
        for v in end_points.values():
            if torch.is_tensor(v):
                dev = v.device
                break
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = torch.zeros((), device=dev)
        end_points["B: DepthReg Loss"] = loss
        return loss, end_points

    pred = end_points["depth_map_pred"]
    gt   = end_points["gt_depth_m"]
    
    if pred.dim() == 3:  # (B,H,W) -> (B,1,H,W)
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    # valid mask from GT range
    valid = (gt > 0) & (gt >= depth_min) & (gt <= depth_max)  # bool (B,1,H,W)
    m = valid.to(dtype=pred.dtype)

    # huber on full map
    # loss_map = F.smooth_l1_loss(pred, gt, reduction="none", beta=huber_beta)  # (B,1,H,W)
    loss_map = F.l1_loss(pred, gt, reduction="none")  # (B,1,H,W)
    loss = (loss_map * m).mean()

    end_points["B: DepthReg Loss"] = loss

    # ---- debug stats ----
    with torch.no_grad():
        end_points["D: depth_reg_valid_pix"] = m.sum() / pred.shape[0]
        # std over valid pixels (avoid constant-plane collapse)
        if valid.any():
            pred_valid = pred[valid]
            gt_valid = gt[valid]
            end_points["D: z_pred_std(valid)"] = pred_valid.std()
            end_points["D: z_gt_std(valid)"] = gt_valid.std()
            end_points["D: z_pred_mean(valid)"] = pred_valid.mean()
            end_points["D: z_gt_mean(valid)"] = gt_valid.mean()
            abs_err = (pred - gt).abs()
            end_points["D: Depth MAE"] = abs_err[valid].mean() if valid.any() else torch.zeros((), device=pred.device)
        else:
            end_points["D: z_pred_std(valid)"] = torch.zeros((), device=pred.device)
            end_points["D: z_gt_std(valid)"] = torch.zeros((), device=pred.device)

    return loss, end_points


def compute_depth_prob_loss(end_points, eps: float = 1e-6):
    if ("depth_prob_pred" not in end_points) or ("depth_prob_gt" not in end_points):
        dev = next((v.device for v in end_points.values() if torch.is_tensor(v)),
                   torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        loss = torch.zeros((), device=dev)
        end_points["B: DepthProb Loss"] = loss
        return loss, end_points

    gt_pack = end_points["depth_prob_gt"]
    if isinstance(gt_pack, (list, tuple)):
        gt, weight = gt_pack
    else:
        gt, weight = gt_pack, None

    pred = end_points["depth_prob_pred"]   # e.g. (B,1,N,D) or (B,V,N,D)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=eps, max=1.0 - eps)
    gt = torch.nan_to_num(gt.to(pred), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

    # BCE over depth bins
    loss_map = F.binary_cross_entropy(pred, gt, reduction="none").mean(dim=-1)  # (...,N)

    if weight is not None:
        weight = weight.to(loss_map)
        loss_map = loss_map * weight
        loss = loss_map.mean()
        valid_ratio = (weight > 0).float().mean()
    else:
        valid = (gt.sum(dim=-1) > 0)
        loss = loss_map[valid].mean() if valid.any() else torch.zeros((), device=pred.device, dtype=pred.dtype)
        valid_ratio = valid.float().mean()

    end_points["B: DepthProb Loss"] = loss
    with torch.no_grad():
        end_points["D: DepthProb Sum"] = pred.sum(dim=-1).mean()
        end_points["D: DepthProb Valid Ratio"] = valid_ratio
    return loss, end_points