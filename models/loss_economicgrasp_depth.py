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
    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Grasp Loss'] = grasp_loss
    end_points['A: DepthReg Loss'] = depth_reg_loss
    end_points['A: Overall Loss'] = loss
    return loss, end_points


# --------------------------
# depth distribution loss
# --------------------------
# def _infer_anchors_from_pred(pred_D: int,
#                             device,
#                             dtype,
#                             min_depth: float = 0.2,
#                             max_depth: float = 1.0):
#     anchors = torch.linspace(min_depth, max_depth, pred_D, device=device, dtype=dtype)
#     return anchors  # (D,)

# def compute_depth_exp_l1_loss(end_points,
#                               min_depth: float = 0.2,
#                               max_depth: float = 1.0,
#                               eps: float = 1e-6):
#     """
#     Expectation L1 Loss:
#       pred_prob -> E[d]_pred, gt_prob -> E[d]_gt, then L1(E[d]_pred, E[d]_gt)

#     Required keys (any one of pred sources):
#       - depth_logits_tok:   (B,D,Hr,Wr)   preferred (stable)
#         OR depth_prob_logits:(B,1,N,D)
#         OR depth_prob_pred: (B,1,N,D)

#     GT keys:
#       - depth_prob_gt:      (B,1,N,D) or [gt, weight]
#       - depth_prob_weight:  (B,1,N) optional (you also have it as a separate key)

#     Outputs:
#       - end_points['B: DepthExpL1 Loss']
#       - monitors: pred/gt exp stats
#     """
#     # --------- presence check ----------
#     has_pred = ('depth_logits_tok' in end_points) or ('depth_prob_logits' in end_points) or ('depth_prob_pred' in end_points)
#     if (not has_pred) or ('depth_prob_gt' not in end_points):
#         dev = None
#         for v in end_points.values():
#             if torch.is_tensor(v):
#                 dev = v.device
#                 break
#         if dev is None:
#             dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         loss = torch.zeros((), device=dev)
#         end_points['B: DepthExpL1 Loss'] = loss
#         return loss, end_points

#     # --------- load gt prob + weight ----------
#     gt_pack = end_points['depth_prob_gt']
#     if isinstance(gt_pack, (list, tuple)):
#         gt_prob, weight = gt_pack
#     else:
#         gt_prob = gt_pack
#         weight = end_points.get('depth_prob_weight', None)

#     # --------- build pred_prob in shape (B,1,N,D) ----------
#     if 'depth_logits_tok' in end_points:
#         # (B,D,Hr,Wr) -> prob -> (B,1,N,D)
#         logits = end_points['depth_logits_tok']                    # (B,D,Hr,Wr)
#         pred_prob = F.softmax(logits, dim=1)                       # (B,D,Hr,Wr)
#         B, D, Hr, Wr = pred_prob.shape
#         pred_prob = pred_prob.permute(0, 2, 3, 1).contiguous()     # (B,Hr,Wr,D)
#         pred_prob = pred_prob.view(B, 1, Hr * Wr, D)               # (B,1,N,D)
#     elif 'depth_prob_logits' in end_points:
#         logits = end_points['depth_prob_logits']                  # (B,1,N,D)
#         pred_prob = F.softmax(logits, dim=-1)
#         B, _, N, D = pred_prob.shape
#     else:
#         pred_prob = end_points['depth_prob_pred']                 # (B,1,N,D) already prob
#         B, _, N, D = pred_prob.shape
#         # 防止数值问题
#         pred_prob = pred_prob.clamp(min=eps)

#     # --------- normalize gt prob to a valid distribution ----------
#     gt_prob = gt_prob.to(device=pred_prob.device, dtype=pred_prob.dtype)
#     gt_sum = gt_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
#     gt_prob = gt_prob / gt_sum                                    # (B,1,N,D)

#     # --------- anchors ----------
#     anchors = end_points.get('depth_anchors_1d', None)
#     if anchors is None:
#         anchors = _infer_anchors_from_pred(D, pred_prob.device, pred_prob.dtype, min_depth, max_depth)  # (D,)
#     else:
#         anchors = anchors.to(device=pred_prob.device, dtype=pred_prob.dtype).view(-1)                   # (D,)
#         assert anchors.numel() == D, f"anchors D mismatch: {anchors.numel()} vs pred D {D}"
#     anchors = anchors.view(1, 1, 1, D)  # broadcast

#     # --------- expectation ----------
#     exp_pred = (pred_prob * anchors).sum(dim=-1)                  # (B,1,N)
#     exp_gt   = (gt_prob   * anchors).sum(dim=-1)                  # (B,1,N)

#     # --------- L1 with optional weight ----------
#     l1 = (exp_pred - exp_gt).abs()                                # (B,1,N)

#     if weight is not None:
#         weight = weight.to(device=l1.device, dtype=l1.dtype)       # (B,1,N) or broadcastable
#         l1 = l1 * weight

#     loss = l1.mean()
#     end_points['B: DepthExpL1 Loss'] = loss

#     # monitors (helpful for your “plane” debug)
#     with torch.no_grad():
#         end_points['D: DepthExp Pred Std'] = exp_pred.std()
#         end_points['D: DepthExp GT Std'] = exp_gt.std()
#         end_points['D: DepthProb Sum'] = pred_prob.sum(dim=-1).mean()

#     return loss, end_points

# def compute_depth_reg_loss(end_points, huber_beta=0.02):
#     """
#     Required:
#       - depth_map_pred: (B,1,448,448)
#       - gt_depth_m:     (B,1,448,448)  (你的 dataloader 已给)
#       - img_idxs:       (B,20000)
#     Optional:
#       - objectness_label: (B,20000) in {0,1}  (推荐用来压制桌面主导)
#     """
#     if ("depth_map_pred" not in end_points) or ("gt_depth_m" not in end_points) or ("img_idxs" not in end_points):
#         dev = next(v.device for v in end_points.values() if torch.is_tensor(v))
#         loss = torch.zeros((), device=dev)
#         end_points["B: DepthReg Loss"] = loss
#         return loss, end_points

#     pred = end_points["depth_map_pred"]   # (B,1,448,448)
#     gt   = end_points["gt_depth_m"]       # (B,1,448,448)
#     idx  = end_points["img_idxs"]         # (B,N)

#     if gt.dim() == 3:   # (B,448,448) -> (B,1,448,448)
#         gt = gt.unsqueeze(1)

#     B, _, H, W = pred.shape
#     pred_flat = pred.view(B, -1)
#     gt_flat   = gt.view(B, -1)

#     z_pred = pred_flat.gather(1, idx)
#     z_gt   = gt_flat.gather(1, idx)

#     m = (z_gt > 0).to(z_pred.dtype)

#     # ✅ 建议：只对物体点监督深度，避免“桌面平面”把网络拉到常数解
#     # if "objectness_label" in end_points:
#     #     m = m * end_points["objectness_label"].to(z_pred.dtype)

#     loss_map = F.smooth_l1_loss(z_pred, z_gt, reduction="none", beta=huber_beta)
#     loss = (loss_map * m).sum() / (m.sum().clamp_min(1.0))

#     end_points["B: DepthReg Loss"] = loss
#     with torch.no_grad():
#         end_points["D: z_pred_std@img_idxs"] = z_pred.std()
#         end_points["D: z_gt_std@img_idxs"] = z_gt.std()
#         end_points["D: depth_reg_pts"] = m.sum() / B
#     return loss, end_points


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
        else:
            end_points["D: z_pred_std(valid)"] = torch.zeros((), device=pred.device)
            end_points["D: z_gt_std(valid)"] = torch.zeros((), device=pred.device)

    return loss, end_points


def compute_depth_exp_l1_loss(end_points):
    """
    Use expectation depth_map_pred (B,1,448,448) supervised by gt_depth_m (B,1,448,448),
    only on sampled pixels img_idxs (B,Np). Optionally weight by objectness_label (B,Np).
    """
    if ("depth_map_pred" not in end_points) or ("gt_depth_m" not in end_points) or ("img_idxs" not in end_points):
        dev = next(v.device for v in end_points.values() if torch.is_tensor(v))
        loss = torch.zeros((), device=dev)
        end_points["B: DepthExpL1 Loss"] = loss
        return loss, end_points

    pred = end_points["depth_map_pred"]   # (B,1,448,448)
    gt   = end_points["gt_depth_m"]       # (B,1,448,448)
    img_idxs = end_points["img_idxs"]     # (B,Np)

    B, _, H, W = pred.shape
    pred_flat = pred.view(B, -1)
    gt_flat   = gt.view(B, -1)

    z_pred = pred_flat.gather(1, img_idxs)
    z_gt   = gt_flat.gather(1, img_idxs)

    # valid mask from gt depth
    m = (z_gt > 0).to(z_pred.dtype)

    # # ✅ 强烈建议：只在“物体点”监督 depth（否则桌面主导 -> 常数平面）
    # if "objectness_label" in end_points:
    #     # objectness_label: (B,Np) with {0,1}
    #     m = m * end_points["objectness_label"].to(z_pred.dtype)

    # SmoothL1 (Huber)
    # loss_map = F.smooth_l1_loss(z_pred, z_gt, reduction="none", beta=huber_beta)
    loss_map = F.l1_loss(z_pred, z_gt, reduction="none")
    loss = (loss_map * m).sum() / (m.sum().clamp_min(1.0))

    end_points["B: DepthExpL1 Loss"] = loss
    with torch.no_grad():
        end_points["D: z_pred_std@img_idxs"] = z_pred.std()
        end_points["D: z_gt_std@img_idxs"] = z_gt.std()
        end_points["D: depth_supervised_pts"] = m.sum()/ B

    return loss, end_points


def compute_depth_prob_loss(end_points, eps: float = 1e-6):
    """
    Soft-label CE (recommended for your change-B):

    Required keys:
      - depth_prob_pred: (B,1,N,D) probability after softmax, in (0,1)
        OR optionally depth_prob_logits: (B,1,N,D) logits (preferred if you have it)
      - depth_prob_gt:   (B,1,N,D) OR [gt, weight], where weight: (B,1,N)

    Output:
      - end_points['B: DepthProb Loss']
      - end_points['D: DepthProb Sum']  (monitor)
    """
    if ('depth_prob_pred' not in end_points and 'depth_prob_logits' not in end_points) or ('depth_prob_gt' not in end_points):
        # allow running without depth head
        dev = None
        for v in end_points.values():
            if torch.is_tensor(v):
                dev = v.device
                break
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = torch.zeros((), device=dev)
        end_points['B: DepthProb Loss'] = loss
        return loss, end_points

    gt_pack = end_points['depth_prob_gt']
    if isinstance(gt_pack, (list, tuple)):
        gt, weight = gt_pack
    else:
        gt, weight = gt_pack, None

    # ---- get log-prob ----
    if 'depth_prob_logits' in end_points:
        logits = end_points['depth_prob_logits']  # (B,1,N,D)
        logp = F.log_softmax(logits, dim=-1)
        pred = logp.exp()
    else:
        pred = end_points['depth_prob_pred']      # (B,1,N,D)
        pred = pred.clamp(min=eps, max=1.0)       # avoid log(0)
        logp = pred.log()

    # ---- safety normalize gt to a proper distribution ----
    gt = gt.to(dtype=logp.dtype, device=logp.device)
    gt_sum = gt.sum(dim=-1, keepdim=True).clamp_min(eps)
    gt = gt / gt_sum

    # ---- soft cross entropy: sum over D, NOT mean ----
    loss_map = -(gt * logp).sum(dim=-1)          # (B,1,N)

    if weight is not None:
        weight = weight.to(device=loss_map.device, dtype=loss_map.dtype)
        loss_map = loss_map * weight

    loss = loss_map.mean()
    end_points['B: DepthProb Loss'] = loss

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