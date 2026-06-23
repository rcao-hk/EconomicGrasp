import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arguments import cfgs


def get_loss(end_points):
    depth_reg_loss, end_points = compute_depth_reg_loss(end_points)

    # choose token-level or point-level supervision
    objectness_loss, end_points = compute_objectness_loss_tok(end_points)
    graspness_loss, end_points  = compute_graspness_loss_tok(end_points)

    # EcoGrasp head losses (on selected M seeds)
    view_loss, end_points  = compute_view_graspness_loss(end_points)
    # angle_loss, end_points = compute_angle_loss(end_points)
    # depth_loss, end_points = compute_depth_loss(end_points)
    # score_loss, end_points = compute_score_loss_cls(end_points)
    # width_loss, end_points = compute_width_loss(end_points)
    score_loss, end_points = compute_cva_score_loss(end_points)
    depth_loss, end_points = compute_cva_depth_loss(end_points)
    width_loss, end_points = compute_cva_width_loss(end_points)
    collision_loss, end_points = compute_collision_loss(end_points, True)
    
    obj_loss = cfgs.objectness_loss_weight * objectness_loss
    grasp_loss = (
        cfgs.graspness_loss_weight * graspness_loss +
        cfgs.view_loss_weight      * view_loss +
        # cfgs.angle_loss_weight     * angle_loss +
        cfgs.depth_loss_weight     * depth_loss +
        cfgs.score_loss_weight     * score_loss +
        cfgs.width_loss_weight     * width_loss
    )

    depth_reg_loss = cfgs.depth_prob_loss_weight * depth_reg_loss
    collision_loss = cfgs.collision_loss_weight  * collision_loss
    grasp_loss = grasp_loss + collision_loss
    loss = obj_loss + grasp_loss + depth_reg_loss

    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Grasp Loss'] = grasp_loss
    end_points['A: Collision Loss'] = collision_loss
    end_points['A: DepthReg Loss'] = depth_reg_loss
    end_points['A: Overall Loss'] = loss
    return loss, end_points


def _score_to_cls(score: torch.Tensor) -> torch.Tensor:
    # Match current EconomicGrasp score classification:
    # grasp_score_label = (batch_grasp_score * 10 / 2).long()
    return (score.float() * 10.0 / 2.0).long().clamp(0, 5)


def compute_cva_score_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction="none")
    pred = end_points["grasp_score_pred_angle"]       # [B,6,Q,A]
    label = _score_to_cls(end_points["batch_grasp_score_angle"])  # [B,Q,A]
    valid = end_points["batch_grasp_angle_valid_mask"].bool()     # [B,Q,A]

    loss_map = criterion(pred, label)  # [B,Q,A]
    if valid.sum() == 0:
        loss = 0.0 * loss_map.sum()
        acc = 0.0 * loss_map.sum()
    else:
        loss = loss_map[valid].mean()
        acc = (torch.argmax(pred, dim=1) == label)[valid].float().mean()

    end_points["B: Score Loss"] = loss
    end_points["D: Score Acc"] = acc

    with torch.no_grad():
        score_prob = F.softmax(pred, dim=1)
        bins = torch.tensor([0, .2, .4, .6, .8, 1.0], device=pred.device, dtype=pred.dtype).view(1, 6, 1, 1)
        score_exp = (score_prob * bins).sum(dim=1)  # [B,Q,A]
        if valid.any():
            end_points["D: CVA score label raw min"] = end_points["batch_grasp_score_angle"].float()[valid].min()
            end_points["D: CVA score label raw max"] = end_points["batch_grasp_score_angle"].float()[valid].max()
            end_points["D: CVA score expected valid"] = score_exp[valid].mean()
            end_points["D: CVA score label valid"] = end_points["batch_grasp_score_angle"].float()[valid].mean()
        pos = end_points.get("batch_grasp_angle_pos_mask", None)
        if torch.is_tensor(pos):
            pos = pos.bool()
            neg = valid & (~pos)
            if pos.any():
                end_points["D: CVA score expected pos"] = score_exp[pos].mean()
            if neg.any():
                end_points["D: CVA score expected neg"] = score_exp[neg].mean()
        
        label_score = end_points["batch_grasp_score_angle"].float()  # [B,Q,A]
        valid_q = end_points["batch_valid_mask"].bool()              # [B,Q], if available

        pred_angle = score_exp.argmax(dim=-1)       # [B,Q]
        label_angle = label_score.argmax(dim=-1)    # [B,Q]

        # only evaluate where point-view has at least one positive angle
        has_pos = end_points["batch_grasp_angle_pos_mask"].bool().any(dim=-1)

        m = has_pos
        if m.any():
            end_points["D: CVA selected angle acc"] = (pred_angle[m] == label_angle[m]).float().mean()
            end_points["D: CVA selected angle bin0 ratio"] = (pred_angle[m] == 0).float().mean()
            end_points["D: CVA label best angle bin0 ratio"] = (label_angle[m] == 0).float().mean()

            pred_sel_score = torch.gather(score_exp, -1, pred_angle.unsqueeze(-1)).squeeze(-1)
            label_best_score = torch.gather(label_score, -1, label_angle.unsqueeze(-1)).squeeze(-1)
            end_points["D: CVA selected score expected"] = pred_sel_score[m].mean()
            end_points["D: CVA label best score"] = label_best_score[m].mean()
    
    return loss, end_points


def compute_cva_depth_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction="none")
    pred = end_points["grasp_depth_pred_angle"]       # [B,D+1,Q,A]
    label = end_points["batch_grasp_depth_angle"].long()  # [B,Q,A]
    pos = end_points["batch_grasp_angle_pos_mask"].bool() # [B,Q,A]

    loss_map = criterion(pred, label)
    if pos.sum() == 0:
        loss = 0.0 * loss_map.sum()
        acc = 0.0 * loss_map.sum()
    else:
        loss = loss_map[pos].mean()
        acc = (torch.argmax(pred, dim=1) == label)[pos].float().mean()

    end_points["B: Depth Loss"] = loss
    end_points["D: Depth Acc"] = acc
    with torch.no_grad():
        pred_idx = torch.argmax(pred, dim=1)
        if pos.any():
            end_points["D: CVA pred depth01 pos"] = (pred_idx[pos] <= 1).float().mean()
            end_points["D: CVA label depth01 pos"] = (label[pos] <= 1).float().mean()
    return loss, end_points


def compute_cva_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction="none")
    pred = end_points["grasp_width_pred_angle"].squeeze(1)  # [B,Q,A]
    label = end_points["batch_grasp_width_angle"].float() * 10.0
    pos = end_points["batch_grasp_angle_pos_mask"].bool()

    loss_map = criterion(pred, label)
    if pos.sum() == 0:
        loss = 0.0 * loss_map.sum()
    else:
        loss = loss_map[pos].mean()

    end_points["B: Width Loss"] = loss
    with torch.no_grad():
        valid = end_points.get("batch_grasp_angle_valid_mask", None)
        if torch.is_tensor(valid) and valid.bool().any():
            end_points["D: Width Loss allangle-valid"] = loss_map[valid.bool()].mean()
    return loss, end_points


def compute_cva_candidate_rank_loss(end_points, margin: float = 0.1, weight: float = 0.0):
    """Optional score ranking: best positive angle should beat best negative angle.

    Set weight=0 to disable.  This is useful if score argmax over angles is used
    at inference and you want an explicit positive-vs-negative angle ordering.
    """
    if weight <= 0:
        dev = end_points["grasp_score_pred_angle"].device
        loss = torch.zeros((), device=dev)
        end_points["B: CVA Rank Loss"] = loss
        return loss, end_points

    score_logits = end_points["grasp_score_pred_angle"]  # [B,6,Q,A]
    valid = end_points["batch_grasp_angle_valid_mask"].bool()
    pos = end_points["batch_grasp_angle_pos_mask"].bool()
    neg = valid & (~pos)

    bins = torch.tensor([0, .2, .4, .6, .8, 1.0], device=score_logits.device, dtype=score_logits.dtype).view(1, 6, 1, 1)
    score_exp = (F.softmax(score_logits, dim=1) * bins).sum(dim=1)  # [B,Q,A]

    pos_score = score_exp.masked_fill(~pos, -1e6).max(dim=-1).values
    neg_score = score_exp.masked_fill(~neg, -1e6).max(dim=-1).values
    has_pos = pos.any(dim=-1)
    has_neg = neg.any(dim=-1)
    m = has_pos & has_neg
    if m.sum() == 0:
        loss = 0.0 * score_exp.sum()
    else:
        loss = F.relu(float(margin) - pos_score[m] + neg_score[m]).mean() * float(weight)

    end_points["B: CVA Rank Loss"] = loss
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
            end_points["D: Depth MAE"] = abs_err[valid].mean()
        else:
            zero = pred.sum() * 0.0
            end_points["D: z_pred_std(valid)"] = zero
            end_points["D: z_gt_std(valid)"] = zero
            end_points["D: z_pred_mean(valid)"] = zero
            end_points["D: z_gt_mean(valid)"] = zero
            end_points["D: Depth MAE"] = zero

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
    criterion = nn.CrossEntropyLoss(reduction="none")

    grasp_angle_pred = end_points["grasp_angle_pred"]             # [B, A+1, Q]
    grasp_angle_label = end_points["batch_grasp_rotations"].long() # [B, Q]
    valid_mask = end_points["batch_valid_mask"].bool()             # [B, Q]

    pose_mask = compute_pose_loss_mask(end_points, valid_mask)

    loss_map = criterion(grasp_angle_pred, grasp_angle_label)      # [B, Q]
    pred = torch.argmax(grasp_angle_pred, dim=1)

    if pose_mask.sum() == 0:
        loss = 0.0 * loss_map.sum()
        acc = 0.0 * loss_map.sum()
    else:
        loss = loss_map[pose_mask].mean()
        acc = (pred == grasp_angle_label)[pose_mask].float().mean()

    with torch.no_grad():
        if valid_mask.any():
            acc_all = (pred == grasp_angle_label)[valid_mask].float().mean()
        else:
            acc_all = 0.0 * loss_map.sum()

    end_points["B: Angle Loss"] = loss
    end_points["D: Angle Acc"] = acc
    end_points["D: Angle Acc allvalid"] = acc_all
    return loss, end_points


def compute_depth_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction="none")

    grasp_depth_pred = end_points["grasp_depth_pred"]          # [B, D+1, Q]
    grasp_depth_label = end_points["batch_grasp_depth"].long()  # [B, Q]
    valid_mask = end_points["batch_valid_mask"].bool()          # [B, Q]

    pose_mask = compute_pose_loss_mask(end_points, valid_mask)

    loss_map = criterion(grasp_depth_pred, grasp_depth_label)   # [B, Q]
    pred = torch.argmax(grasp_depth_pred, dim=1)

    if pose_mask.sum() == 0:
        loss = 0.0 * loss_map.sum()
        acc = 0.0 * loss_map.sum()
    else:
        loss = loss_map[pose_mask].mean()
        acc = (pred == grasp_depth_label)[pose_mask].float().mean()

    with torch.no_grad():
        if valid_mask.any():
            acc_all = (pred == grasp_depth_label)[valid_mask].float().mean()
        else:
            acc_all = 0.0 * loss_map.sum()

    end_points["B: Depth Loss"] = loss
    end_points["D: Depth Acc"] = acc
    end_points["D: Depth Acc allvalid"] = acc_all
    return loss, end_points

def compute_score_loss_cls(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_score_pred = end_points['grasp_score_pred']            # (B,6,M) or (B,6,M) (cls logits)
    grasp_score_label = (end_points['batch_grasp_score'] * 10 / 2).long()  # (B,M)
    valid_mask = end_points['batch_valid_mask']                  # (B,M)

    loss = criterion(grasp_score_pred, grasp_score_label)  # (B,M)  (squeeze(1) keeps compat)
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
    criterion = nn.SmoothL1Loss(reduction="none")

    grasp_width_pred = end_points["grasp_width_pred"]          # [B,1,Q] or [B,Q]
    grasp_width_label = end_points["batch_grasp_width"] * 10   # [B,Q]
    valid_mask = end_points["batch_valid_mask"].bool()         # [B,Q]

    pose_mask = compute_pose_loss_mask(end_points, valid_mask)

    pred = grasp_width_pred.squeeze(1)
    loss_map = criterion(pred, grasp_width_label)              # [B,Q]

    if pose_mask.sum() == 0:
        loss = 0.0 * loss_map.sum()
    else:
        loss = loss_map[pose_mask].mean()

    with torch.no_grad():
        if valid_mask.any():
            width_l1_all = loss_map[valid_mask].mean()
        else:
            width_l1_all = 0.0 * loss_map.sum()

    end_points["B: Width Loss"] = loss
    end_points["D: Width Loss allvalid"] = width_l1_all
    return loss, end_points


def compute_pose_loss_mask(end_points, valid_mask):
    """
    Pose mask for angle/depth/width only.

    Required when cfgs.use_pose_loss_mask=True:
      - end_points["kview_query_selected_gt"]: [B, Q]

    No fallback by design.
    """
    valid_mask = valid_mask.bool()

    if not cfgs.use_pose_loss_mask:
        end_points["D: PoseMask enabled"] = torch.zeros((), device=valid_mask.device)
        end_points["D: PoseMask ratio"] = valid_mask.float().mean()
        end_points["D: PoseMask valid ratio"] = valid_mask.float().mean()
        return valid_mask

    selected_gt = end_points["kview_query_selected_gt"].to(
        device=valid_mask.device,
        dtype=torch.float32,
    )

    if selected_gt.shape != valid_mask.shape:
        raise RuntimeError(
            f"kview_query_selected_gt shape {tuple(selected_gt.shape)} "
            f"does not match batch_valid_mask shape {tuple(valid_mask.shape)}"
        )

    thresh = float(cfgs.pose_loss_view_gt_thresh)
    pose_mask = valid_mask & (selected_gt > thresh)

    with torch.no_grad():
        end_points["D: PoseMask enabled"] = torch.ones((), device=valid_mask.device)
        end_points["D: PoseMask thresh"] = torch.tensor(thresh, device=valid_mask.device)
        end_points["D: PoseMask valid ratio"] = valid_mask.float().mean()
        end_points["D: PoseMask ratio"] = pose_mask.float().mean()

        if valid_mask.any():
            end_points["D: PoseMask selectedGT valid mean"] = selected_gt[valid_mask].mean()
            end_points["D: PoseMask selectedGT>thr valid"] = (
                selected_gt[valid_mask] > thresh
            ).float().mean()
        else:
            z = selected_gt.sum() * 0.0
            end_points["D: PoseMask selectedGT valid mean"] = z
            end_points["D: PoseMask selectedGT>thr valid"] = z

        if pose_mask.any():
            end_points["D: PoseMask selectedGT pose mean"] = selected_gt[pose_mask].mean()
            end_points["D: PoseMask#"] = pose_mask.float().sum() / max(float(valid_mask.shape[0]), 1.0)
        else:
            z = selected_gt.sum() * 0.0
            end_points["D: PoseMask selectedGT pose mean"] = z
            end_points["D: PoseMask#"] = z

    return pose_mask


def compute_collision_loss(end_points, balanced=True):
    """
    Strict selected collision loss.

    Required:
      - grasp_collision_pred: [B,1,M]
      - batch_grasp_collision: [B,M]
      - batch_valid_mask: [B,M]
    """
    if "grasp_collision_pred" not in end_points:
        dev = end_points["batch_valid_mask"].device
        loss = torch.zeros((), device=dev)
        end_points["B: Collision Loss"] = loss
        return loss, end_points

    pred = end_points["grasp_collision_pred"]
    if pred.dim() == 3:
        pred = pred.squeeze(1)  # [B,M]

    label = end_points["batch_grasp_collision"].to(device=pred.device, dtype=pred.dtype)
    valid_mask = end_points["batch_valid_mask"].bool().to(pred.device)

    loss_map = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

    if valid_mask.sum() == 0:
        loss = 0.0 * loss_map.sum()
        acc = 0.0 * loss_map.sum()
        pos_ratio = 0.0 * loss_map.sum()
        pred_pos_ratio = 0.0 * loss_map.sum()
        precision = 0.0 * loss_map.sum()
        recall = 0.0 * loss_map.sum()
    else:
        if balanced:
            pos_mask = valid_mask & (label > 0.5)
            neg_mask = valid_mask & (label <= 0.5)

            if pos_mask.any() and neg_mask.any():
                loss = 0.5 * (loss_map[pos_mask].mean() + loss_map[neg_mask].mean())
            else:
                loss = loss_map[valid_mask].mean()
        else:
            loss = loss_map[valid_mask].mean()

        with torch.no_grad():
            prob = torch.sigmoid(pred)
            pred_label = prob > 0.5
            label_bool = label > 0.5

            acc = (pred_label[valid_mask] == label_bool[valid_mask]).float().mean()
            pos_ratio = label_bool[valid_mask].float().mean()
            pred_pos_ratio = pred_label[valid_mask].float().mean()

            tp = (pred_label & label_bool & valid_mask).float().sum()
            fp = (pred_label & (~label_bool) & valid_mask).float().sum()
            fn = ((~pred_label) & label_bool & valid_mask).float().sum()

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)

    end_points["B: Collision Loss"] = loss
    end_points["D: Collision Label PosRatio"] = pos_ratio
    end_points["D: Collision Pred PosRatio"] = pred_pos_ratio
    end_points["D: Collision Acc"] = acc
    end_points["D: Collision Precision"] = precision
    end_points["D: Collision Recall"] = recall

    return loss, end_points