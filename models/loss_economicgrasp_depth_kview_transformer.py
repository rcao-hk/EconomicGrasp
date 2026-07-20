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
    # Embedded evaluator-aligned heads.  The score loss is CDF-only; an
    # independent collision head is intentionally disabled because collision,
    # empty, and force-closure failures are all represented by an all-zero CDF.
    score_loss, end_points = compute_cva_cdf_loss(end_points, balanced=True)
    depth_loss, end_points = compute_cva_depth_loss(end_points)
    width_loss, end_points = compute_cva_width_depth_loss(end_points)
    collision_loss = 0.0 * score_loss
    
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
    loss = obj_loss + grasp_loss + depth_reg_loss

    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Grasp Loss'] = grasp_loss
    end_points['A: Collision Loss'] = collision_loss
    end_points['A: DepthReg Loss'] = depth_reg_loss
    end_points['A: Overall Loss'] = loss
    return loss, end_points


def get_loss_fullrot(end_points):
    """Loss for GeometryAwareDenseFieldRotNet + FullRotation CVA decoder.

    The current model has two grasp stages:
      1. dense coarse full-rotation proposal field [B,M,V,A];
      2. sparse fixed-rotation queries [B,Q], Q=M*L, predicting score/depth/width.

    There is no separate view loss or angle CE: view and in-plane angle are
    already part of the full-rotation proposal/query state.
    """
    depth_reg_loss, end_points = compute_depth_reg_loss(end_points)

    objectness_loss, end_points = compute_objectness_loss_tok(end_points)
    graspness_loss, end_points = compute_graspness_loss_tok(end_points)

    # Dense coarse rotation-field supervision. This is intentionally separate
    # from the sparse final-query score loss.
    rotation_loss, end_points = compute_rotation_proposal_loss(end_points, False, cfgs.rotation_reg_weight, cfgs.rotation_listwise_weight)

    # Final sparse full-rotation query losses. Each query already has a fixed
    # view + in-plane angle, so no angle loss is used.
    score_loss, end_points = compute_fullrot_score_loss(end_points)
    depth_loss, end_points = compute_fullrot_depth_loss(end_points)
    width_loss, end_points = compute_fullrot_width_loss(end_points)
    # collision_loss, end_points = compute_collision_loss(end_points, balanced=True)

    obj_loss = cfgs.objectness_loss_weight * objectness_loss
    grasp_loss = (
        cfgs.graspness_loss_weight * graspness_loss
        + rotation_loss
        + cfgs.depth_loss_weight * depth_loss
        + cfgs.score_loss_weight * score_loss
        + cfgs.width_loss_weight * width_loss
    )

    depth_reg_loss = cfgs.depth_prob_loss_weight * depth_reg_loss
    # collision_loss = cfgs.collision_loss_weight * collision_loss
    # grasp_loss = grasp_loss + collision_loss
    loss = obj_loss + grasp_loss + depth_reg_loss

    end_points['A: Objectness Loss'] = objectness_loss
    end_points['A: Rotation Proposal Loss'] = rotation_loss
    end_points['A: Grasp Loss'] = grasp_loss
    # end_points['A: Collision Loss'] = collision_loss
    end_points['A: DepthReg Loss'] = depth_reg_loss
    end_points['A: Overall Loss'] = loss
    return loss, end_points


# def compute_rotation_proposal_loss(
#     end_points,
#     balanced=False,
#     positive_threshold=0.0,
# ):
#     """Balanced Smooth-L1 loss for the dense coarse rotation field.

#     Required labels:
#         batch_grasp_rotation_score:      [B,M,V,A]
#         batch_grasp_rotation_valid_mask: [B,M,V,A]
#     """
#     pred = end_points["rotation_score"]
#     label = end_points["batch_grasp_rotation_score"].to(
#         device=pred.device,
#         dtype=pred.dtype,
#     )
#     valid = end_points["batch_grasp_rotation_valid_mask"].bool().to(pred.device)
#     if pred.shape != label.shape or pred.shape != valid.shape:
#         raise RuntimeError(
#             f"rotation proposal shapes must match, got pred={tuple(pred.shape)}, "
#             f"label={tuple(label.shape)}, valid={tuple(valid.shape)}"
#         )

#     loss_map = F.smooth_l1_loss(pred, label, reduction="none")
#     if not bool(valid.any()):
#         loss = 0.0 * loss_map.sum()
#     elif balanced:
#         pos = valid & (label > float(positive_threshold))
#         neg = valid & (~pos)
#         if bool(pos.any()) and bool(neg.any()):
#             loss = 0.5 * (loss_map[pos].mean() + loss_map[neg].mean())
#         else:
#             loss = loss_map[valid].mean()
#     else:
#         loss = loss_map[valid].mean()

#     end_points["B: Rotation Proposal Loss"] = loss
#     with torch.no_grad():
#         label_flat = label.flatten(2)
#         valid_flat = valid.flatten(2)
#         proposal = end_points["rot_proposal_inds"].long()
#         has_pos = ((label_flat > float(positive_threshold)) & valid_flat).any(dim=-1)
#         best = label_flat.masked_fill(~valid_flat, -1.0).argmax(dim=-1)
#         if bool(has_pos.any()):
#             recall = (proposal == best.unsqueeze(-1)).any(dim=-1)
#             end_points["D: RotNet best recall@L"] = recall[has_pos].float().mean()
#         else:
#             end_points["D: RotNet best recall@L"] = loss.detach() * 0.0

#         proposal_label = torch.gather(label_flat, dim=-1, index=proposal)
#         proposal_valid = torch.gather(valid_flat, dim=-1, index=proposal)
#         end_points["D: RotNet label positive ratio"] = (
#             (proposal_label > float(positive_threshold)) & proposal_valid
#         ).float().mean()
#         if bool(proposal_valid.any()):
#             end_points["D: RotNet proposal label"] = proposal_label[
#                 proposal_valid
#             ].mean()
#         else:
#             end_points["D: RotNet proposal label"] = loss.detach() * 0.0
#     return loss, end_points

def compute_rotation_proposal_loss(
    end_points,
    balanced=False,
    reg_weight=1.0,
    listwise_weight=1.0,
):
    """Simple dense RotNet proposal loss with only two weights.

    Loss = reg_weight * balanced_smooth_l1 + listwise_weight * listwise_softmax.

    Required:
        rotation_score:                    [B,M,V,A]
        batch_grasp_rotation_score:        [B,M,V,A], quality in [0,1]
        batch_grasp_rotation_valid_mask:   [B,M,V,A]
        rot_proposal_inds:                 [B,M,L]

    Design choices:
      - positive = valid & label > 0
      - negatives = valid & label == 0
      - SmoothL1 is balanced between positives and negatives.
      - Listwise target is label-normalized over positive rotations only;
        negatives remain in the softmax denominator and are thus pushed down.
      - No temperature, margin, hard-k, or positive-threshold hyperparameter.
    """
    pred = end_points["rotation_score"]
    label = end_points["batch_grasp_rotation_score"].to(
        device=pred.device,
        dtype=pred.dtype,
    )
    valid = end_points["batch_grasp_rotation_valid_mask"].bool().to(pred.device)
    if pred.shape != label.shape or pred.shape != valid.shape:
        raise RuntimeError(
            f"rotation proposal shapes must match, got pred={tuple(pred.shape)}, "
            f"label={tuple(label.shape)}, valid={tuple(valid.shape)}"
        )

    B, M, V, A = pred.shape
    R = V * A
    pred_flat = pred.reshape(B, M, R)
    label_flat = label.reshape(B, M, R)
    valid_flat = valid.reshape(B, M, R)
    pos_flat = valid_flat & (label_flat > 0)
    neg_flat = valid_flat & (~pos_flat)

    # 1) Balanced pointwise regression preserves the dense quality landscape.
    reg_map = F.smooth_l1_loss(pred_flat, label_flat, reduction="none")
    if balanced and pos_flat.any() and neg_flat.any():
        reg_loss = (
            0.5 * reg_map[pos_flat].mean()
            + 0.5 * reg_map[neg_flat].mean()
        )
    elif valid_flat.any():
        reg_loss = reg_map[valid_flat].mean()
    else:
        reg_loss = 0.0 * reg_map.sum()

    # 2) Listwise softmax directly optimizes relative ranking within each seed.
    # Target distribution uses only positive rotations but negatives are in the
    # prediction denominator, so high-scoring negatives are penalized.
    has_pos = pos_flat.any(dim=-1)  # [B,M]
    if bool(has_pos.any()):
        pred_logits = pred_flat.masked_fill(~valid_flat, -1e4)
        pred_logprob = F.log_softmax(pred_logits[has_pos], dim=-1)

        target = torch.where(pos_flat, label_flat.clamp_min(0.0), torch.zeros_like(label_flat))
        target_sum = target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        target_prob = target / target_sum
        listwise_loss = -(target_prob[has_pos] * pred_logprob).sum(dim=-1).mean()
    else:
        listwise_loss = 0.0 * pred_flat.sum()

    loss = float(reg_weight) * reg_loss + float(listwise_weight) * listwise_loss
    end_points["B: Rotation Proposal Loss"] = loss

    with torch.no_grad():
        proposal = end_points["rot_proposal_inds"].long().to(pred.device)  # [B,M,L]
        valid_label = label_flat.masked_fill(~valid_flat, -1.0)
        best_label, best_idx = valid_label.max(dim=-1)  # [B,M]
        pred_top_idx = pred_flat.masked_fill(~valid_flat, -1e4).argmax(dim=-1)
        pred_top_label = torch.gather(label_flat, dim=-1, index=pred_top_idx.unsqueeze(-1)).squeeze(-1)
        proposal_label = torch.gather(label_flat, dim=-1, index=proposal.clamp(0, R - 1))
        proposal_valid = torch.gather(valid_flat, dim=-1, index=proposal.clamp(0, R - 1))
        proposal_pos = proposal_valid & (proposal_label > 0)

        z = loss.detach() * 0.0
        end_points["D: RotNet reg loss"] = reg_loss.detach()
        end_points["D: RotNet listwise loss"] = listwise_loss.detach()
        end_points["D: RotNet loss reg weight"] = pred.new_tensor(float(reg_weight))
        end_points["D: RotNet loss listwise weight"] = pred.new_tensor(float(listwise_weight))

        # Dense-field label statistics.  These are renamed to avoid confusion
        # with proposal statistics from RotNet selected queries.
        end_points["D: RotNet dense valid ratio"] = valid_flat.float().mean()
        end_points["D: RotNet dense positive ratio"] = pos_flat.float().mean()
        end_points["D: RotNet seeds with positive"] = has_pos.float().mean()
        end_points["D: RotNet balanced reg"] = float(balanced)
        
        if bool(has_pos.any()):
            recall = (proposal == best_idx.unsqueeze(-1)).any(dim=-1)
            end_points["D: RotNet best recall@L"] = recall[has_pos].float().mean()
            end_points["D: RotNet top1 exact acc"] = (pred_top_idx[has_pos] == best_idx[has_pos]).float().mean()
            end_points["D: RotNet top1 label"] = pred_top_label[has_pos].mean()
            end_points["D: RotNet best label"] = best_label[has_pos].mean()
            end_points["D: RotNet top1 label ratio"] = (
                pred_top_label[has_pos] / best_label[has_pos].clamp_min(1e-6)
            ).mean()
        else:
            end_points["D: RotNet best recall@L"] = z
            end_points["D: RotNet top1 exact acc"] = z
            end_points["D: RotNet top1 label"] = z
            end_points["D: RotNet best label"] = z
            end_points["D: RotNet top1 label ratio"] = z

        # Proposal statistics; this replaces the old misleading
        # "RotNet label positive ratio" key.
        end_points["D: RotNet proposal positive ratio"] = proposal_pos.float().mean()
        if bool(proposal_valid.any()):
            end_points["D: RotNet proposal label"] = proposal_label[proposal_valid].mean()
        else:
            end_points["D: RotNet proposal label"] = z
    return loss, end_points

def _fullrot_required_mask(end_points, key):
    if key not in end_points:
        raise KeyError(f"Full-rotation decoder training requires end_points['{key}']")
    return end_points[key].bool()


def compute_fullrot_score_loss(end_points, balanced=False):
    """Score CE for sparse fixed-rotation queries.

    Required:
      grasp_score_pred:                    [B,6,Q]
      batch_grasp_score:                  [B,Q], score in [0,1]
      batch_rotation_query_valid_mask:    [B,Q]
      batch_rotation_query_pos_mask:      [B,Q]

    Positive/negative balancing is useful because RotNet proposals can contain
    many zero-score rotations, especially early in training.
    """
    pred = end_points['grasp_score_pred']
    label_score = end_points['batch_grasp_score'].to(pred.device).float()
    valid = _fullrot_required_mask(
        end_points, 'batch_rotation_query_valid_mask'
    ).to(pred.device)
    pos = _fullrot_required_mask(
        end_points, 'batch_rotation_query_pos_mask'
    ).to(pred.device) & valid
    neg = valid & (~pos)

    if pred.dim() != 3 or pred.shape[1] != 6:
        raise RuntimeError(
            f'grasp_score_pred must be [B,6,Q], got {tuple(pred.shape)}'
        )
    if label_score.shape != pred.shape[:1] + pred.shape[2:]:
        raise RuntimeError(
            f'batch_grasp_score shape {tuple(label_score.shape)} does not '
            f'match score prediction [B,Q]={pred.shape[:1] + pred.shape[2:]}'
        )
    if valid.shape != label_score.shape or pos.shape != label_score.shape:
        raise RuntimeError(
            'Full-rotation score labels/masks must all be [B,Q], got '
            f'label={tuple(label_score.shape)}, valid={tuple(valid.shape)}, '
            f'pos={tuple(pos.shape)}'
        )

    label = _score_to_cls(label_score)
    loss_map = F.cross_entropy(pred, label, reduction='none')

    if not bool(valid.any()):
        loss = 0.0 * loss_map.sum()
        acc = 0.0 * loss_map.sum()
    elif balanced and bool(pos.any()) and bool(neg.any()):
        loss = 0.5 * (loss_map[pos].mean() + loss_map[neg].mean())
        with torch.no_grad():
            pred_cls = pred.argmax(dim=1)
            acc = (pred_cls[valid] == label[valid]).float().mean()
    else:
        loss = loss_map[valid].mean()
        with torch.no_grad():
            pred_cls = pred.argmax(dim=1)
            acc = (pred_cls[valid] == label[valid]).float().mean()

    end_points['B: Score Loss'] = loss
    end_points['D: Score Acc'] = acc

    with torch.no_grad():
        score_prob = F.softmax(pred, dim=1)
        bins = torch.tensor(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            device=pred.device,
            dtype=pred.dtype,
        ).view(1, 6, 1)
        score_expected = (score_prob * bins).sum(dim=1)
        end_points['D: FullRot query valid ratio'] = valid.float().mean()
        end_points['D: FullRot query pos ratio'] = pos.float().mean()
        if bool(valid.any()):
            end_points['D: FullRot score label min'] = label_score[valid].min()
            end_points['D: FullRot score label max'] = label_score[valid].max()
            end_points['D: FullRot score expected valid'] = score_expected[valid].mean()
        if bool(pos.any()):
            end_points['D: FullRot score expected pos'] = score_expected[pos].mean()
        else:
            end_points['D: FullRot score expected pos'] = 0.0 * score_expected.sum()
        if bool(neg.any()):
            end_points['D: FullRot score expected neg'] = score_expected[neg].mean()
        else:
            end_points['D: FullRot score expected neg'] = 0.0 * score_expected.sum()

        # Final local-decoder ranking among L proposals that share one seed.
        if (
            'fullrot_query_base_m' in end_points
            and 'fullrot_query_num_proposals' in end_points
        ):
            M = int(end_points['fullrot_query_base_m'])
            L = int(end_points['fullrot_query_num_proposals'])
            B, Q = score_expected.shape
            if Q == M * L:
                score_bml = score_expected.view(B, M, L)
                label_bml = label_score.view(B, M, L)
                valid_bml = valid.view(B, M, L)
                pred_rank = score_bml.argmax(dim=-1)
                label_rank = label_bml.masked_fill(~valid_bml, -1.0).argmax(dim=-1)
                has_valid = valid_bml.any(dim=-1)
                if bool(has_valid.any()):
                    end_points['D: FullRot selected proposal acc'] = (
                        pred_rank[has_valid] == label_rank[has_valid]
                    ).float().mean()
                else:
                    end_points['D: FullRot selected proposal acc'] = 0.0 * score_expected.sum()

    return loss, end_points


def compute_fullrot_depth_loss(end_points):
    """Depth CE only for positive fixed-rotation queries."""
    pred = end_points['grasp_depth_pred']
    label = end_points['batch_grasp_depth'].long().to(pred.device)
    valid = _fullrot_required_mask(
        end_points, 'batch_rotation_query_valid_mask'
    ).to(pred.device)
    pos = _fullrot_required_mask(
        end_points, 'batch_rotation_query_pos_mask'
    ).to(pred.device) & valid

    if pred.dim() != 3:
        raise RuntimeError(
            f'grasp_depth_pred must be [B,D+1,Q], got {tuple(pred.shape)}'
        )
    if label.shape != pred.shape[:1] + pred.shape[2:]:
        raise RuntimeError(
            f'batch_grasp_depth shape {tuple(label.shape)} does not match '
            f'depth prediction [B,Q]={pred.shape[:1] + pred.shape[2:]}'
        )

    loss_map = F.cross_entropy(pred, label, reduction='none')
    if not bool(pos.any()):
        loss = 0.0 * loss_map.sum()
        acc = 0.0 * loss_map.sum()
    else:
        loss = loss_map[pos].mean()
        with torch.no_grad():
            pred_idx = pred.argmax(dim=1)
            acc = (pred_idx[pos] == label[pos]).float().mean()

    end_points['B: Depth Loss'] = loss
    end_points['D: Depth Acc'] = acc
    with torch.no_grad():
        pred_idx = pred[:, :-1, :].argmax(dim=1)
        if bool(pos.any()):
            end_points['D: FullRot pred depth01 pos'] = (
                pred_idx[pos] <= 1
            ).float().mean()
            end_points['D: FullRot label depth01 pos'] = (
                label[pos] <= 1
            ).float().mean()
    return loss, end_points


def compute_fullrot_width_loss(end_points):
    """Width Smooth-L1 only for positive fixed-rotation queries."""
    pred = end_points['grasp_width_pred'].squeeze(1)
    label = end_points['batch_grasp_width'].to(pred.device).float() * 10.0
    valid = _fullrot_required_mask(
        end_points, 'batch_rotation_query_valid_mask'
    ).to(pred.device)
    pos = _fullrot_required_mask(
        end_points, 'batch_rotation_query_pos_mask'
    ).to(pred.device) & valid

    if pred.shape != label.shape:
        raise RuntimeError(
            f'width prediction/label mismatch: pred={tuple(pred.shape)}, '
            f'label={tuple(label.shape)}'
        )

    loss_map = F.smooth_l1_loss(pred, label, reduction='none')
    if not bool(pos.any()):
        loss = 0.0 * loss_map.sum()
    else:
        loss = loss_map[pos].mean()

    end_points['B: Width Loss'] = loss
    with torch.no_grad():
        if bool(valid.any()):
            end_points['D: Width Loss fullrot-valid'] = loss_map[valid].mean()
    return loss, end_points


def _score_to_cls(score: torch.Tensor) -> torch.Tensor:
    # Match current EconomicGrasp score classification:
    # grasp_score_label = (batch_grasp_score * 10 / 2).long()
    return (score.float() * 10.0 / 2.0).long().clamp(0, 5)


def _cdf_bins_to_target(
    bins: torch.Tensor,
    num_thresholds: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand compact CDF bins [..] to binary cumulative targets [..,T]."""
    if bins.min() < 0 or bins.max() > num_thresholds:
        raise ValueError(
            f"CDF bins must be in [0,{num_thresholds}], got "
            f"min={int(bins.min())}, max={int(bins.max())}"
        )
    ids = torch.arange(
        num_thresholds,
        device=bins.device,
        dtype=bins.dtype,
    )
    return (
        (bins.unsqueeze(-1) > 0)
        & (ids >= bins.unsqueeze(-1) - 1)
    ).to(dtype=dtype)


def compute_cva_cdf_loss(end_points, balanced: bool = True):
    """CDF-only loss for complete center-view-angle-depth candidates.

    Required:
      grasp_cdf_pred_angle_depth:          [B,T,Q,A,D]
      batch_grasp_cdf_bins_angle_depth:   [B,Q,A,D], values 0..T
      batch_grasp_cdf_valid_mask:          [B,Q,A,D]

    Each threshold is balanced independently between positive and negative
    candidates.  This matches the validated external J0 scorer objective while
    preserving an absolute probability scale across views/centers/objects.
    """
    pred = end_points["grasp_cdf_pred_angle_depth"]
    if pred.dim() != 5:
        raise ValueError(
            "grasp_cdf_pred_angle_depth must be [B,T,Q,A,D], got "
            f"{tuple(pred.shape)}"
        )
    B, T, Q, A, D = pred.shape
    bins = end_points["batch_grasp_cdf_bins_angle_depth"].long().to(pred.device)
    valid = end_points["batch_grasp_cdf_valid_mask"].bool().to(pred.device)
    expected = (B, Q, A, D)
    if bins.shape != expected or valid.shape != expected:
        raise ValueError(
            f"CDF label shapes must be {expected}, got bins={tuple(bins.shape)}, "
            f"valid={tuple(valid.shape)}"
        )

    thresholds = end_points.get("batch_grasp_cdf_thresholds", None)
    if torch.is_tensor(thresholds) and thresholds.numel() != T:
        raise ValueError(
            f"Model predicts T={T} thresholds but label cache has "
            f"{thresholds.numel()}."
        )

    target = _cdf_bins_to_target(bins, T, pred.dtype)  # [B,Q,A,D,T]
    logits = pred.permute(0, 2, 3, 4, 1).contiguous() # [B,Q,A,D,T]
    loss_map = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )

    pieces = []
    for t in range(T):
        loss_t = loss_map[..., t]
        target_t = target[..., t]
        if not bool(valid.any()):
            pieces.append(0.0 * loss_t.sum())
            continue
        if balanced:
            pos = valid & (target_t > 0.5)
            neg = valid & (~pos)
            sub = []
            if bool(pos.any()):
                sub.append(loss_t[pos].mean())
            if bool(neg.any()):
                sub.append(loss_t[neg].mean())
            pieces.append(
                torch.stack(sub).mean() if sub else 0.0 * loss_t.sum()
            )
        else:
            pieces.append(loss_t[valid].mean())
    loss = torch.stack(pieces).mean()

    end_points["B: Score Loss"] = loss
    end_points["B: CDF Loss"] = loss

    run_diag = end_points.get("cva_compute_diagnostics", True)
    if torch.is_tensor(run_diag):
        run_diag = bool(run_diag.detach().item())
    else:
        run_diag = bool(run_diag)

    # These metrics allocate sigmoid probabilities, utility maps, binary maps,
    # and flattened selection tensors.  They are observational only; computing
    # them every N steps is sufficient and does not change the loss/gradient.
    if run_diag:
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            utility = prob.mean(dim=-1)          # [B,Q,A,D]
            target_utility = target.mean(dim=-1)
            pred_binary = prob >= 0.5
            target_binary = target >= 0.5

            if bool(valid.any()):
                end_points["D: CDF Acc"] = (
                    pred_binary[valid] == target_binary[valid]
                ).float().mean()
                end_points["D: CDF Utility Pred"] = utility[valid].mean()
                end_points["D: CDF Utility Target"] = target_utility[valid].mean()
                end_points["D: CDF Utility MAE"] = (
                    utility[valid] - target_utility[valid]
                ).abs().mean()
                end_points["D: CDF Positive Ratio"] = (
                    bins[valid] > 0
                ).float().mean()

                # Candidate-selection diagnostic: target utility of the model's
                # best angle-depth candidate for every matched center-view query.
                valid_q = valid.any(dim=-1).any(dim=-1)  # [B,Q]
                masked_utility = utility.masked_fill(~valid, -1e6)
                selected = masked_utility.reshape(B, Q, A * D).argmax(dim=-1)
                selected_target = target_utility.reshape(B, Q, A * D).gather(
                    -1, selected.unsqueeze(-1)
                ).squeeze(-1)
                oracle_target = target_utility.masked_fill(~valid, -1.0)
                oracle_target = oracle_target.reshape(B, Q, A * D).max(dim=-1).values
                if bool(valid_q.any()):
                    end_points["D: CDF Selected Target Utility"] = (
                        selected_target[valid_q].mean()
                    )
                    end_points["D: CDF Oracle Target Utility"] = (
                        oracle_target[valid_q].mean()
                    )
                    end_points["D: CDF Selection Regret"] = (
                        oracle_target[valid_q] - selected_target[valid_q]
                    ).mean()
            else:
                z = loss.detach() * 0.0
                end_points["D: CDF Acc"] = z
                end_points["D: CDF Utility Pred"] = z
                end_points["D: CDF Utility Target"] = z
                end_points["D: CDF Utility MAE"] = z
                end_points["D: CDF Positive Ratio"] = z

            monotonic_violation = (
                prob[..., 1:] + 1e-7 < prob[..., :-1]
            ).float().mean()
            end_points["D: CDF Monotonic Violation"] = monotonic_violation

    return loss, end_points


def compute_cva_width_depth_loss(end_points):
    """Depth-wise width regression for all evaluator-valid candidates.

    Required:
      grasp_width_pred_angle_depth:               [B,D,Q,A]
      batch_grasp_width_angle_depth:              [B,Q,A,D], meters
      batch_grasp_width_valid_mask_angle_depth:   [B,Q,A,D]
    """
    pred = end_points["grasp_width_pred_angle_depth"]
    if pred.dim() != 4:
        raise ValueError(
            "grasp_width_pred_angle_depth must be [B,D,Q,A], got "
            f"{tuple(pred.shape)}"
        )
    pred = pred.permute(0, 2, 3, 1).contiguous()  # [B,Q,A,D]
    label = end_points["batch_grasp_width_angle_depth"].to(
        device=pred.device, dtype=pred.dtype
    ) * 10.0
    valid = end_points[
        "batch_grasp_width_valid_mask_angle_depth"
    ].bool().to(pred.device)

    if pred.shape != label.shape or pred.shape != valid.shape:
        raise ValueError(
            f"Depth-wise width shapes must match, pred={tuple(pred.shape)}, "
            f"label={tuple(label.shape)}, valid={tuple(valid.shape)}"
        )

    loss_map = F.smooth_l1_loss(pred, label, reduction="none")
    if not bool(valid.any()):
        loss = 0.0 * loss_map.sum()
    else:
        loss = loss_map[valid].mean()

    end_points["B: Width Loss"] = loss
    end_points["B: Width Depth Loss"] = loss
    with torch.no_grad():
        if bool(valid.any()):
            end_points["D: Width Depth MAE x10"] = (
                pred[valid] - label[valid]
            ).abs().mean()
            end_points["D: Width Depth Label mean"] = label[valid].mean() / 10.0
            end_points["D: Width Depth Pred mean"] = pred[valid].mean() / 10.0
            end_points["D: Width Depth valid ratio"] = valid.float().mean()
    return loss, end_points


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


def compute_cva_collision_loss(end_points, balanced=False):
    """
    Angle-level CVA collision loss.

    Required:
      - grasp_collision_pred_angle:   [B,1,Q,A]
      - batch_grasp_collision_angle:  [B,Q,A]
      - batch_grasp_angle_valid_mask: [B,Q,A]

    Label convention:
      - 1: collision / invalid grasp
      - 0: non-collision / valid grasp

    Args:
        end_points: dict
        balanced: if True, average positive and negative losses equally.

    Returns:
        loss, end_points
    """
    required_keys = [
        "grasp_collision_pred_angle",
        "batch_grasp_collision_angle",
        "batch_grasp_angle_valid_mask",
    ]
    for k in required_keys:
        if k not in end_points:
            raise KeyError(f"compute_cva_collision_loss requires end_points['{k}'].")

    pred = end_points["grasp_collision_pred_angle"]  # [B,1,Q,A]
    if pred.dim() != 4 or pred.shape[1] != 1:
        raise ValueError(
            "grasp_collision_pred_angle must be [B,1,Q,A], "
            f"got {tuple(pred.shape)}"
        )

    pred = pred.squeeze(1)  # [B,Q,A]

    label = end_points["batch_grasp_collision_angle"].to(
        device=pred.device,
        dtype=pred.dtype,
    )  # [B,Q,A]

    valid_mask = end_points["batch_grasp_angle_valid_mask"].bool().to(pred.device)

    if label.shape != pred.shape:
        raise ValueError(
            "batch_grasp_collision_angle shape must match pred [B,Q,A], "
            f"got label={tuple(label.shape)}, pred={tuple(pred.shape)}"
        )

    if valid_mask.shape != pred.shape:
        raise ValueError(
            "batch_grasp_angle_valid_mask shape must match pred [B,Q,A], "
            f"got valid_mask={tuple(valid_mask.shape)}, pred={tuple(pred.shape)}"
        )

    loss_map = F.binary_cross_entropy_with_logits(
        pred,
        label,
        reduction="none",
    )  # [B,Q,A]

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
                loss = 0.5 * (
                    loss_map[pos_mask].mean()
                    + loss_map[neg_mask].mean()
                )
            else:
                loss = loss_map[valid_mask].mean()
        else:
            loss = loss_map[valid_mask].mean()

        with torch.no_grad():
            prob = torch.sigmoid(pred)
            pred_label = prob > 0.5
            label_bool = label > 0.5

            acc = (
                pred_label[valid_mask]
                == label_bool[valid_mask]
            ).float().mean()

            pos_ratio = label_bool[valid_mask].float().mean()
            pred_pos_ratio = pred_label[valid_mask].float().mean()

            tp = (pred_label & label_bool & valid_mask).float().sum()
            fp = (pred_label & (~label_bool) & valid_mask).float().sum()
            fn = ((~pred_label) & label_bool & valid_mask).float().sum()

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)

    end_points["B: CVA Collision Loss"] = loss

    end_points["D: CVA Collision Label PosRatio"] = pos_ratio
    end_points["D: CVA Collision Pred PosRatio"] = pred_pos_ratio
    end_points["D: CVA Collision Acc"] = acc
    end_points["D: CVA Collision Precision"] = precision
    end_points["D: CVA Collision Recall"] = recall

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


def compute_collision_loss(end_points, balanced=False):
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