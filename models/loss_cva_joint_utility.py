"""Training objective for the evaluator-aligned joint utility scorer.

The primary label is the exact GraspNet evaluator friction outcome for each
mined center-view-angle-depth-width candidate.  A valid candidate with minimum
friction coefficient ``mu`` is converted to six cumulative success targets:

    y_j = 1[mu > 0 and mu <= threshold_j]

for thresholds {0.2, 0.4, ..., 1.2}.  Invalid candidates (collision, empty, or
force-closure failure) receive all-zero targets.  The mean of these six targets
is exactly the candidate's benchmark utility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from .cva_joint_utility import FRICTION_THRESHOLDS


@dataclass
class JointUtilityLossConfig:
    cdf_weight: float = 1.0
    utility_weight: float = 0.25
    collision_weight: float = 0.15
    empty_weight: float = 0.10
    center_rank_weight: float = 0.50
    object_rank_weight: float = 0.35
    sample_rank_weight: float = 0.25
    listwise_weight: float = 0.15
    balanced_cdf: bool = True
    balanced_aux: bool = True
    hard_negative_weight: float = 1.5
    hard_negative_quality_threshold: float = 0.75
    rank_margin: float = 0.10
    rank_target_gap: float = 1.0 / 6.0
    listwise_temperature: float = 0.20
    utility_beta: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def friction_to_cdf_targets(friction: torch.Tensor) -> torch.Tensor:
    """Convert minimum friction coefficient to cumulative success targets.

    Parameters
    ----------
    friction:
        [N] exact evaluator output. Values <= 0 are invalid.

    Returns
    -------
    [N,6] float tensor, non-decreasing along the threshold dimension.
    """
    threshold = friction.new_tensor(FRICTION_THRESHOLDS).view(1, -1)
    f = friction.view(-1, 1)
    return ((f > 0.0) & (f <= threshold)).to(friction.dtype)


def friction_to_utility(friction: torch.Tensor) -> torch.Tensor:
    return friction_to_cdf_targets(friction).mean(dim=-1)


def _balanced_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    balanced: bool = True,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    if not balanced:
        return loss.mean()
    pos = target > 0.5
    neg = ~pos
    pieces = []
    if bool(pos.any()):
        pieces.append(loss[pos].mean())
    if bool(neg.any()):
        pieces.append(loss[neg].mean())
    return torch.stack(pieces).mean() if pieces else 0.0 * logits.sum()


def _balanced_cdf_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_weight: Optional[torch.Tensor],
    balanced: bool,
) -> torch.Tensor:
    pieces = []
    for j in range(logits.shape[-1]):
        pieces.append(
            _balanced_binary_loss(
                logits[:, j],
                target[:, j],
                weight=candidate_weight,
                balanced=balanced,
            )
        )
    return torch.stack(pieces).mean()


def _hard_negative_candidate_weight(
    target_utility: torch.Tensor,
    legacy_score: Optional[torch.Tensor],
    config: JointUtilityLossConfig,
) -> torch.Tensor:
    weight = torch.ones_like(target_utility)
    if legacy_score is None or config.hard_negative_weight <= 1.0:
        return weight
    hard = (target_utility <= 0.0) & (
        legacy_score >= float(config.hard_negative_quality_threshold)
    )
    return torch.where(
        hard,
        torch.full_like(weight, float(config.hard_negative_weight)),
        weight,
    )


def _group_hard_pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor,
    margin: float,
    target_gap: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Best-target vs hardest-predicted lower-utility candidate per group."""
    losses = []
    correct = []
    used = 0
    for gid in torch.unique(group):
        mask = group == gid
        if int(mask.sum()) < 2:
            continue
        p = pred[mask]
        t = target[mask]
        best_value, best_idx = t.max(dim=0)
        negatives = t <= (best_value - float(target_gap))
        if not bool(negatives.any()) or float(best_value) <= 0.0:
            continue
        good_pred = p[best_idx]
        bad_pred = p.masked_fill(~negatives, -1e6).max()
        losses.append(F.softplus(float(margin) + bad_pred - good_pred))
        correct.append((good_pred > bad_pred).float())
        used += 1
    if not losses:
        zero = 0.0 * pred.sum()
        return zero, zero.detach(), pred.new_tensor(0.0)
    return torch.stack(losses).mean(), torch.stack(correct).mean(), pred.new_tensor(float(used))


def _sample_listwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_group: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses = []
    temp = max(float(temperature), 1e-4)
    for gid in torch.unique(sample_group):
        mask = sample_group == gid
        if int(mask.sum()) < 2:
            continue
        p = pred[mask]
        t = target[mask]
        positive = t > 0.0
        if not bool(positive.any()):
            continue
        # The target distribution is defined only over evaluator-valid
        # candidates, while log-softmax is computed over the complete frame.
        # Invalid candidates therefore compete in the denominator but receive
        # zero target mass. Multiple equivalent valid grasps remain soft.
        target_dist = F.softmax(t[positive] / temp, dim=0).detach()
        log_pred = F.log_softmax(p / temp, dim=0)
        losses.append(-(target_dist * log_pred[positive]).sum())
    return torch.stack(losses).mean() if losses else 0.0 * pred.sum()


@torch.no_grad()
def _topk_metrics(
    pred: torch.Tensor,
    target_utility: torch.Tensor,
    collision: torch.Tensor,
    empty: torch.Tensor,
    sample_group: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    sums: Dict[str, list] = {
        "top1_utility": [],
        "top1_invalid": [],
        "top1_collision": [],
        "top1_empty": [],
        "top10_utility": [],
        "top10_invalid": [],
        "top10_collision": [],
        "top10_empty": [],
        "oracle_top1_utility": [],
    }
    for gid in torch.unique(sample_group):
        mask = sample_group == gid
        p = pred[mask]
        u = target_utility[mask]
        c = collision[mask]
        e = empty[mask]
        if p.numel() == 0:
            continue
        order = torch.argsort(p, descending=True)
        top1 = order[:1]
        top10 = order[: min(10, order.numel())]
        sums["top1_utility"].append(u[top1].mean())
        sums["top1_invalid"].append((u[top1] <= 0).float().mean())
        sums["top1_collision"].append(c[top1].float().mean())
        sums["top1_empty"].append(e[top1].float().mean())
        sums["top10_utility"].append(u[top10].mean())
        sums["top10_invalid"].append((u[top10] <= 0).float().mean())
        sums["top10_collision"].append(c[top10].float().mean())
        sums["top10_empty"].append(e[top10].float().mean())
        sums["oracle_top1_utility"].append(u.max())
    return {
        key: torch.stack(values).mean() if values else pred.new_tensor(float("nan"))
        for key, values in sums.items()
    }



@torch.no_grad()
def _evaluator_proxy_metrics(
    pred: torch.Tensor,
    target_utility: torch.Tensor,
    collision: torch.Tensor,
    empty: torch.Tensor,
    sample_group: torch.Tensor,
    object_group: torch.Tensor,
    per_object_k: int = 10,
    global_k: int = 50,
) -> Dict[str, torch.Tensor]:
    """Evaluator-shaped metric using exact utility and predicted ordering.

    The proxy reproduces the score-dependent per-object Top-10 admission and
    global Top-50 ordering.  Given target utility averaged over friction
    thresholds, the harmonic rank weights exactly reproduce mean
    Precision@1..K in expectation. Missing ranks contribute zero.
    """
    device = pred.device
    harmonic = torch.cumsum(
        1.0 / torch.arange(global_k, 0, -1, device=device, dtype=pred.dtype),
        dim=0,
    ).flip(0) / float(global_k)
    # The expression above is not the desired ordering; construct directly for
    # clarity: w_r = 1/K * sum_{k=r}^K 1/k.
    inv = 1.0 / torch.arange(1, global_k + 1, device=device, dtype=pred.dtype)
    harmonic = torch.flip(torch.cumsum(torch.flip(inv, dims=[0]), dim=0), dims=[0]) / float(global_k)

    ap_values = []
    oracle_values = []
    invalid_values = []
    collision_values = []
    empty_values = []
    for sid in torch.unique(sample_group):
        sample_mask = sample_group == sid
        p = pred[sample_mask]
        t = target_utility[sample_mask]
        c = collision[sample_mask]
        e = empty[sample_mask]
        o = object_group[sample_mask]
        if p.numel() == 0:
            continue

        def build_pool(score: torch.Tensor) -> torch.Tensor:
            selected = []
            for oid in torch.unique(o):
                ids = torch.nonzero(o == oid, as_tuple=False).flatten()
                k = min(int(per_object_k), int(ids.numel()))
                if k:
                    selected.append(ids[torch.topk(score[ids], k=k, largest=True, sorted=False).indices])
            if not selected:
                return torch.zeros(0, dtype=torch.long, device=device)
            pool = torch.cat(selected, dim=0)
            k = min(int(global_k), int(pool.numel()))
            return pool[torch.topk(score[pool], k=k, largest=True, sorted=True).indices]

        pred_order = build_pool(p)
        oracle_order = build_pool(t)
        if pred_order.numel():
            w = harmonic[: pred_order.numel()]
            ap_values.append((w * t[pred_order]).sum())
            invalid_values.append((t[pred_order] <= 0).float().mean())
            collision_values.append(c[pred_order].float().mean())
            empty_values.append(e[pred_order].float().mean())
        if oracle_order.numel():
            w = harmonic[: oracle_order.numel()]
            oracle_values.append((w * t[oracle_order]).sum())

    zero = pred.new_tensor(float('nan'))
    mean = lambda values: torch.stack(values).mean() if values else zero
    ap = mean(ap_values)
    oracle = mean(oracle_values)
    return {
        'evaluator_proxy_ap': ap,
        'evaluator_proxy_oracle_ap': oracle,
        'evaluator_proxy_gap': oracle - ap,
        'evaluator_proxy_top50_invalid': mean(invalid_values),
        'evaluator_proxy_top50_collision': mean(collision_values),
        'evaluator_proxy_top50_empty': mean(empty_values),
    }

def compute_joint_utility_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: Optional[JointUtilityLossConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the complete evaluator-aligned scorer objective."""
    cfg = config or JointUtilityLossConfig()
    friction = batch["friction"].float()
    cdf_target = friction_to_cdf_targets(friction)
    utility_target = cdf_target.mean(dim=-1)
    collision_target = batch["pure_collision"].float()
    empty_target = batch["empty"].float()

    hard_negative_score = batch.get("legacy_quality", batch.get("legacy_score"))
    if hard_negative_score is not None:
        hard_negative_score = hard_negative_score.float()
    candidate_weight = _hard_negative_candidate_weight(
        utility_target, hard_negative_score, cfg
    )

    cdf_loss = _balanced_cdf_loss(
        outputs["cdf_logits"], cdf_target, candidate_weight, cfg.balanced_cdf
    )
    utility_loss = F.smooth_l1_loss(
        outputs["utility"], utility_target, beta=float(cfg.utility_beta)
    )
    collision_loss = _balanced_binary_loss(
        outputs["collision_logit"],
        collision_target,
        candidate_weight,
        balanced=cfg.balanced_aux,
    )
    empty_loss = _balanced_binary_loss(
        outputs["empty_logit"],
        empty_target,
        candidate_weight,
        balanced=cfg.balanced_aux,
    )

    center_rank, center_pair_acc, center_groups = _group_hard_pairwise_rank_loss(
        outputs["utility"],
        utility_target,
        batch["center_group"].long(),
        cfg.rank_margin,
        cfg.rank_target_gap,
    )
    object_rank, object_pair_acc, object_groups = _group_hard_pairwise_rank_loss(
        outputs["utility"],
        utility_target,
        batch["object_group"].long(),
        cfg.rank_margin,
        cfg.rank_target_gap,
    )
    sample_rank, sample_pair_acc, sample_groups = _group_hard_pairwise_rank_loss(
        outputs["utility"],
        utility_target,
        batch["sample_group"].long(),
        cfg.rank_margin,
        cfg.rank_target_gap,
    )
    listwise = _sample_listwise_loss(
        outputs["utility"],
        utility_target,
        batch["sample_group"].long(),
        cfg.listwise_temperature,
    )

    total = (
        cfg.cdf_weight * cdf_loss
        + cfg.utility_weight * utility_loss
        + cfg.collision_weight * collision_loss
        + cfg.empty_weight * empty_loss
        + cfg.center_rank_weight * center_rank
        + cfg.object_rank_weight * object_rank
        + cfg.sample_rank_weight * sample_rank
        + cfg.listwise_weight * listwise
    )

    with torch.no_grad():
        mae = (outputs["utility"] - utility_target).abs().mean()
        monotonic_violation = (
            outputs["cdf_prob"][:, :-1] > outputs["cdf_prob"][:, 1:] + 1e-6
        ).float().mean()
        topk = _topk_metrics(
            outputs["utility"],
            utility_target,
            collision_target,
            empty_target,
            batch["sample_group"].long(),
        )
        evaluator_proxy = _evaluator_proxy_metrics(
            outputs["utility"],
            utility_target,
            collision_target,
            empty_target,
            batch["sample_group"].long(),
            batch["object_group"].long(),
        )

    metrics: Dict[str, torch.Tensor] = {
        "loss/total": total.detach(),
        "loss/cdf": cdf_loss.detach(),
        "loss/utility": utility_loss.detach(),
        "loss/collision": collision_loss.detach(),
        "loss/empty": empty_loss.detach(),
        "loss/center_rank": center_rank.detach(),
        "loss/object_rank": object_rank.detach(),
        "loss/sample_rank": sample_rank.detach(),
        "loss/listwise": listwise.detach(),
        "metric/utility_mae": mae,
        "metric/target_utility_mean": utility_target.mean(),
        "metric/pred_utility_mean": outputs["utility"].mean(),
        "metric/valid_ratio": (utility_target > 0).float().mean(),
        "metric/safe08_ratio": (utility_target >= 0.5).float().mean(),
        "metric/center_pair_acc": center_pair_acc,
        "metric/object_pair_acc": object_pair_acc,
        "metric/sample_pair_acc": sample_pair_acc,
        "metric/center_rank_groups": center_groups,
        "metric/object_rank_groups": object_groups,
        "metric/sample_rank_groups": sample_groups,
        "metric/monotonic_violation": monotonic_violation,
        **{f"metric/{k}": v for k, v in topk.items()},
        **{f"metric/{k}": v for k, v in evaluator_proxy.items()},
    }
    return total, metrics
