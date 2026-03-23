import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arguments import cfgs


def get_loss(end_points):
    # ---------- NEW: depth distribution BCE loss ----------
    # depth_prob_loss, end_points = compute_depth_prob_loss(end_points)
    # depth_exp_l1_loss, end_points = compute_depth_exp_l1_loss(end_points)
    depth_reg_loss, end_points = compute_depth_reg_loss(end_points)

    # ---------- EcoGrasp original losses ----------
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points  = compute_graspness_loss(end_points)
    view_loss, end_points       = compute_view_graspness_loss(end_points)

    angle_loss, end_points = compute_angle_loss(end_points)
    depth_loss, end_points = compute_depth_loss(end_points)
    score_loss, end_points = compute_score_loss_cls(end_points)
    width_loss, end_points = compute_width_loss(end_points)

    obj_loss = cfgs.objectness_loss_weight * objectness_loss
    grasp_loss = (cfgs.graspness_loss_weight  * graspness_loss  +
        cfgs.view_loss_weight       * view_loss       +
        cfgs.angle_loss_weight      * angle_loss      +
        cfgs.depth_loss_weight      * depth_loss      +
        cfgs.score_loss_weight      * score_loss      +
        cfgs.width_loss_weight      * width_loss )
    
    depth_reg_loss = cfgs.depth_prob_loss_weight * depth_reg_loss
    loss = obj_loss + grasp_loss + depth_reg_loss

    # depth_prob_loss = cfgs.depth_prob_loss_weight * depth_prob_loss
    # loss = obj_loss + grasp_loss + depth_prob_loss
    
    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Grasp Loss'] = grasp_loss
    end_points['A: DepthReg Loss'] = depth_reg_loss
    end_points['A: Overall Loss'] = loss
    return loss, end_points


def get_loss_c2_1(end_points):
    depth_prob_loss, end_points = compute_depth_prob_loss(end_points)

    # choose token-level or point-level supervision
    objectness_loss, end_points = compute_objectness_loss_tok(end_points)
    graspness_loss, end_points  = compute_graspness_loss_tok(end_points)

    # EcoGrasp head losses (on selected M seeds)
    view_loss, end_points  = compute_view_graspness_loss(end_points)
    angle_loss, end_points = compute_angle_loss(end_points)
    depth_loss, end_points = compute_depth_loss(end_points)
    score_loss, end_points = compute_score_loss_cls(end_points)
    width_loss, end_points = compute_width_loss(end_points)

    obj_loss = cfgs.objectness_loss_weight * objectness_loss
    grasp_loss = (
        cfgs.graspness_loss_weight * graspness_loss +
        cfgs.view_loss_weight      * view_loss +
        cfgs.angle_loss_weight     * angle_loss +
        cfgs.depth_loss_weight     * depth_loss +
        cfgs.score_loss_weight     * score_loss +
        cfgs.width_loss_weight     * width_loss
    )

    depth_prob_loss = cfgs.depth_prob_loss_weight * depth_prob_loss
    loss = obj_loss + grasp_loss + depth_prob_loss

    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Grasp Loss'] = grasp_loss
    end_points['A: DepthReg Loss'] = depth_prob_loss
    end_points['A: Overall Loss'] = loss
    return loss, end_points


def get_loss_c2_2(end_points):
    depth_reg_loss, end_points = compute_depth_reg_loss(end_points)

    # choose token-level or point-level supervision
    objectness_loss, end_points = compute_objectness_loss_tok(end_points)
    graspness_loss, end_points  = compute_graspness_loss_tok(end_points)

    # EcoGrasp head losses (on selected M seeds)
    view_loss, end_points  = compute_view_graspness_loss(end_points)
    angle_loss, end_points = compute_angle_loss(end_points)
    depth_loss, end_points = compute_depth_loss(end_points)
    score_loss, end_points = compute_score_loss_cls(end_points)
    width_loss, end_points = compute_width_loss(end_points)

    obj_loss = cfgs.objectness_loss_weight * objectness_loss
    grasp_loss = (
        cfgs.graspness_loss_weight * graspness_loss +
        cfgs.view_loss_weight      * view_loss +
        cfgs.angle_loss_weight     * angle_loss +
        cfgs.depth_loss_weight     * depth_loss +
        cfgs.score_loss_weight     * score_loss +
        cfgs.width_loss_weight     * width_loss
    )

    depth_reg_loss = cfgs.depth_prob_loss_weight * depth_reg_loss
    loss = obj_loss + grasp_loss + depth_reg_loss

    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Grasp Loss'] = grasp_loss
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
    gt = gt_pack[0] if isinstance(gt_pack, (list, tuple)) else gt_pack  # drop weight

    pred = end_points["depth_prob_pred"]  # (B,1,N,D)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=eps, max=1.0 - eps)
    gt   = torch.nan_to_num(gt.to(pred), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

    loss_map = F.binary_cross_entropy(pred, gt, reduction="none").mean(dim=-1)  # (B,1,N)

    valid = (gt.sum(dim=-1) > 0)  # (B,1,N)
    loss = loss_map[valid].mean() if valid.any() else torch.zeros((), device=pred.device, dtype=pred.dtype)

    end_points["B: DepthProb Loss"] = loss
    with torch.no_grad():
        end_points["D: DepthProb Sum"] = pred.sum(dim=-1).mean()
        end_points["D: DepthProb Valid Ratio"] = valid.float().mean()
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