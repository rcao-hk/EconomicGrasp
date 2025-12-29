import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arguments import cfgs


def get_loss(end_points):
    # ---------- EcoGrasp original losses ----------
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points  = compute_graspness_loss(end_points)
    view_loss, end_points       = compute_view_graspness_loss(end_points)

    angle_loss, end_points = compute_angle_loss(end_points)
    depth_loss, end_points = compute_depth_loss(end_points)
    score_loss, end_points = compute_score_loss_cls(end_points)
    width_loss, end_points = compute_width_loss(end_points)

    # ---------- NEW: depth distribution BCE loss ----------
    depth_prob_loss, end_points = compute_depth_prob_loss(end_points)

    loss = (
        cfgs.objectness_loss_weight * objectness_loss +
        cfgs.graspness_loss_weight  * graspness_loss  +
        cfgs.view_loss_weight       * view_loss       +
        cfgs.angle_loss_weight      * angle_loss      +
        cfgs.depth_loss_weight      * depth_loss      +
        cfgs.score_loss_weight      * score_loss      +
        cfgs.width_loss_weight      * width_loss      +
        cfgs.depth_prob_loss_weight * depth_prob_loss
    )

    end_points['A: Overall Loss'] = loss
    return loss, end_points


# --------------------------
# NEW: depth distribution loss
# --------------------------
def compute_depth_prob_loss(end_points):
    """
    Required keys:
      - depth_prob_pred: (B,1,N,D)  probability after softmax, in [0,1]
      - depth_prob_gt:   (B,1,N,D) OR [gt, weight], where weight: (B,1,N)
    """
    if 'depth_prob_pred' not in end_points or 'depth_prob_gt' not in end_points:
        # allow running without depth head
        loss = torch.zeros((), device=next(iter(end_points.values())).device)
        end_points['B: DepthProb Loss'] = loss
        return loss, end_points

    pred = end_points['depth_prob_pred']
    gt_pack = end_points['depth_prob_gt']

    if isinstance(gt_pack, (list, tuple)):
        gt, weight = gt_pack
    else:
        gt, weight = gt_pack, None

    # safety: clamp to avoid log(0) instabilities (optional but helps)
    pred = pred.clamp(min=1e-6, max=1.0 - 1e-6)

    # BCE per-bin, then mean over depth bins -> (B,1,N)
    loss_map = F.binary_cross_entropy(pred, gt, reduction='none').mean(dim=-1)

    if weight is not None:
        # weight is typically valid_ratio in [0,1], shape (B,1,N)
        loss_map = loss_map * weight

    loss = loss_map.mean()
    end_points['B: DepthProb Loss'] = loss

    # (optional) monitoring: mean prob mass (should be ~1 after softmax)
    with torch.no_grad():
        end_points['D: DepthProb Sum'] = pred.sum(dim=-1).mean()

    return loss, end_points


# --------------------------
# EcoGrasp original losses
# --------------------------
def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']  # (B,2,N)
    objectness_label = end_points['objectness_label']  # (B,N)
    loss = criterion(objectness_score, objectness_label)
    end_points['B: Objectness Loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['D: Objectness Acc'] = (objectness_pred == objectness_label.long()).float().mean()
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)   # (B,N)
    graspness_label = end_points['graspness_label'].squeeze(-1)  # (B,N) or (B,N,1)

    loss_mask = end_points['objectness_label'].bool()            # (B,N)
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean() if loss.numel() > 0 else 0.0 * graspness_score.sum()

    end_points['B: Graspness Loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']                        # (B,M,300)
    view_label = end_points['batch_grasp_view_graspness']        # (B,M,300)
    loss = criterion(view_score, view_label)
    end_points['B: View Loss'] = loss
    return loss, end_points


def compute_angle_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_pred = end_points['grasp_angle_pred']            # (B, num_angle+1, M)
    grasp_angle_label = end_points['batch_grasp_rotations'].long()# (B,M)
    valid_mask = end_points['batch_valid_mask']                  # (B,M)

    loss = criterion(grasp_angle_pred, grasp_angle_label)        # (B,M)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
        acc = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()
        acc = (torch.argmax(grasp_angle_pred, 1) == grasp_angle_label)[valid_mask].float().mean()

    end_points['B: Angle Loss'] = loss
    end_points['D: Angle Acc'] = acc
    return loss, end_points


def compute_depth_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_depth_pred = end_points['grasp_depth_pred']            # (B, num_depth+1, M)
    grasp_depth_label = end_points['batch_grasp_depth'].long()   # (B,M)
    valid_mask = end_points['batch_valid_mask']                  # (B,M)

    loss = criterion(grasp_depth_pred, grasp_depth_label)        # (B,M)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
        acc = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()
        acc = (torch.argmax(grasp_depth_pred, 1) == grasp_depth_label)[valid_mask].float().mean()

    end_points['B: Depth Loss'] = loss
    end_points['D: Depth Acc'] = acc
    return loss, end_points


def compute_score_loss_cls(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_score_pred = end_points['grasp_score_pred']            # (B,6,M) or (B,6,M) (cls logits)
    grasp_score_label = (end_points['batch_grasp_score'] * 10 / 2).long()  # (B,M)
    valid_mask = end_points['batch_valid_mask']                  # (B,M)

    loss = criterion(grasp_score_pred.squeeze(1), grasp_score_label)  # (B,M)  (squeeze(1) keeps compat)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
        acc = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()
        acc = (torch.argmax(grasp_score_pred, 1) == grasp_score_label)[valid_mask].float().mean()

    end_points['B: Score Loss'] = loss
    end_points['D: Score Acc'] = acc
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']            # (B,1,M) or (B,M)
    grasp_width_label = end_points['batch_grasp_width'] * 10     # (B,M)
    valid_mask = end_points['batch_valid_mask']                  # (B,M)

    loss = criterion(grasp_width_pred.squeeze(1), grasp_width_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()

    end_points['B: Width Loss'] = loss
    return loss, end_points