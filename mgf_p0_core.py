"""Numerical contracts for online frozen-source P0-1/P0-2 (no CUDA extensions)."""
from __future__ import annotations
import hashlib
import math
from typing import Mapping
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'mgf_online_p0_v1'
VARIANTS = ('base', 'feature_only', 'full', 'cva')
THRESHOLDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def fingerprint(module: nn.Module) -> str:
    """Include frozen parameters AND persistent buffers, not just requires_grad."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        t = value.detach().contiguous().cpu()
        h.update(f'{name}:{t.dtype}:{tuple(t.shape)}'.encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    # EconomicGrasp also uses explicit flags, independently of Module.training.
    for sub in module.modules():
        if hasattr(sub, 'is_training'):
            sub.is_training = False
    return module


def assert_frozen(module: nn.Module) -> None:
    if any(p.requires_grad or p.grad is not None for p in module.parameters()):
        raise RuntimeError('Frozen source has a trainable parameter or accumulated gradient')
    if any(m.training or bool(getattr(m, 'is_training', False)) for m in module.modules()):
        raise RuntimeError('Frozen source left deterministic eval mode')


def cdf_targets(bins: torch.Tensor, thresholds: int = 6) -> torch.Tensor:
    if bins.dtype.is_floating_point or bool(((bins < 0) | (bins > thresholds)).any()):
        raise ValueError('CDF bins must be integers in [0,T]')
    t = torch.arange(1, thresholds + 1, device=bins.device)
    return ((bins[..., None] > 0) & (bins[..., None] <= t)).float()


def _distributed_mean(numerator, count, distributed=True):
    """DDP averages gradients: compensate to obtain a true global masked mean."""
    denominator = count.detach().to(numerator).clone()
    world = 1
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(denominator)
    return numerator * world / denominator.clamp_min(1)


def score_loss(logits, bins, valid, ranking_weight=0.1, temperature=0.1, distributed=True):
    """Unbalanced threshold BCE + informative-query listwise KL; no geometry loss."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not math.isfinite(ranking_weight) or ranking_weight < 0:
        raise ValueError('ranking_weight must be finite and nonnegative')
    if logits.ndim != 5 or bins.shape != logits.shape[:1] + logits.shape[2:] or valid.shape != bins.shape:
        raise ValueError('Expected logits [B,T,Q,A,D] and labels/mask [B,Q,A,D]')
    target = cdf_targets(bins, logits.shape[1])
    pred = logits.float().movedim(1, -1)
    valid = valid.bool()
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none').mean(-1)
    bce = _distributed_mean((bce * valid).sum(), valid.sum(), distributed)
    pu, tu = pred.sigmoid().mean(-1).flatten(-2), target.mean(-1).flatten(-2)
    mask = valid.flatten(-2)
    informative = (mask.sum(-1) >= 2) & (
        tu.masked_fill(~mask, -1).max(-1).values - tu.masked_fill(~mask, 2).min(-1).values > 1e-6)
    lp = F.log_softmax((pu / temperature).masked_fill(~mask, -1e4), -1)
    lq = F.log_softmax((tu / temperature).masked_fill(~mask, -1e4), -1)
    kl = (lq.exp() * (lq - lp)).sum(-1)
    rank = _distributed_mean((kl * informative).sum(), informative.sum(), distributed)
    total = bce + ranking_weight * rank
    return total, {'bce': bce.detach(), 'ranking': rank.detach(),
                   'valid_candidates': valid.sum().detach(),
                   'informative_queries': informative.sum().detach()}


class Metrics:
    """Pooled sufficient statistics, NOT an average of per-batch AUROC or ratios."""
    def __init__(self):
        self.values = np.zeros(13 + 128, dtype=np.float64)

    @torch.no_grad()
    def update(self, logits, bins, valid):
        pred = logits.float().movedim(1, -1).sigmoid()
        target = cdf_targets(bins, logits.shape[1])
        valid = valid.bool()
        u, y = pred.mean(-1), target.mean(-1)
        p, t = u[valid], y[valid]
        pos = (bins[valid] > 0)
        idx = (p.clamp(0, 1) * 64).long().clamp_max(63)
        ph = torch.bincount(idx[pos], minlength=64).double()
        nh = torch.bincount(idx[~pos], minlength=64).double()
        uf, yf, m = u.flatten(-2), y.flatten(-2), valid.flatten(-2)
        oracle = yf.masked_fill(~m, -1).max(-1).values
        informative = (m.sum(-1) >= 2) & (oracle - yf.masked_fill(~m, 2).min(-1).values > 1e-6)
        chosen = uf.masked_fill(~m, -1).argmax(-1)
        selected = yf.gather(-1, chosen[..., None])[..., 0]
        zero = p.sum() * 0
        vals = [valid.sum(), pos.sum(), p.sum(), t.sum(), p[pos].sum(), p[~pos].sum(),
                (p-t).abs().sum(), informative.sum(),
                (oracle-selected)[informative].sum(),
                ((oracle-selected).abs()[informative] < 1e-6).sum(),
                F.binary_cross_entropy(pred[valid].clamp(1e-7, 1-1e-7), target[valid], reduction='sum')
                if p.numel() else zero,
                torch.as_tensor(m.shape[0]*m.shape[1], device=p.device),
                selected[informative].sum()]
        self.values += torch.cat((torch.stack([x.double() for x in vals]), ph, nh)).cpu().numpy()

    def synchronize(self, device):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor(self.values, device=device)
            torch.distributed.all_reduce(t)
            self.values = t.cpu().numpy()

    def report(self):
        v = self.values
        n, p, q = v[0], v[1], v[7]
        def div(a,b): return float(a/b) if b else None
        ph, nh = v[13:77], v[77:141]
        auc = div((ph*(np.cumsum(nh)-nh+0.5*nh)).sum(), p*(n-p))
        tp, fp = np.cumsum(ph[::-1]), np.cumsum(nh[::-1])
        ap = div((ph[::-1]*tp/np.maximum(tp+fp, 1)).sum(), p)
        return {'valid_candidates': int(n), 'positive_fraction': div(p,n),
                'utility_pred_mean': div(v[2],n), 'utility_target_mean': div(v[3],n),
                'pred_pos_mean': div(v[4],p), 'pred_neg_mean': div(v[5],n-p),
                'utility_mae': div(v[6],n), 'informative_queries': int(q),
                'informative_query_fraction': div(q,v[11]),
                'selection_regret': div(v[8],q), 'top1_best_hit': div(v[9],q),
                'selected_target_utility': div(v[12],q), 'bce': div(v[10],6*n),
                'auroc64': auc, 'auprc64': ap,
                'auprc_lift64': div(ap,p/n) if ap is not None and n else None,
                'ranking_metric_defined': bool(p and n-p)}


def perturb_actions(actions, kind, amount, max_width=0.1):
    """Camera-Z same-ray translation, local approach-axis roll, or metric width."""
    if not math.isfinite(float(amount)):
        raise ValueError('Nonfinite intervention')
    if actions.shape[-1] != 17 or not torch.isfinite(actions).all():
        raise ValueError('Actions must be finite [...,17]')
    out = actions.clone()
    if kind == 'ray_z':
        z = actions[..., 15]
        if bool((z <= 0).any()):
            raise ValueError('Nonpositive input camera-Z')
        out[..., 13:16] = actions[..., 13:16] * ((z+amount)/z)[..., None]
    elif kind == 'roll':
        c, s = math.cos(amount), math.sin(amount)
        rx = actions.new_tensor([[1,0,0], [0,c,-s], [0,s,c]])
        out[..., 4:13] = (actions[..., 4:13].reshape(*actions.shape[:-1],3,3) @ rx).flatten(-2)
    elif kind == 'width':
        out[..., 1] += amount
    else:
        raise ValueError(f'Unknown action intervention: {kind}')
    valid = (out[..., 1] > 0) & (out[..., 1] <= max_width) & (out[..., 15] > 0)
    # Invalid actions are excluded and counted. Never silently clamp/deduplicate.
    return out, valid


def exact_targets(result, n):
    f = np.asarray(result.friction, dtype=np.float64)
    invalid = np.asarray(result.collision_or_empty, dtype=bool)
    if f.shape != (n,) or invalid.shape != (n,) or not np.isfinite(f).all():
        raise RuntimeError('Exact evaluator returned invalid or unaligned rows')
    return ((f[:,None] > 0) & (f[:,None] <= np.asarray(THRESHOLDS)[None]+1e-6)
            & ~invalid[:,None]).astype(np.float32)


def comparison_metrics(scores, targets, valid, reference_scores=None):
    """Exact-label audit statistics. Scores/targets [Q,C]; validity same shape."""
    s, y, m = np.asarray(scores), np.asarray(targets), np.asarray(valid, bool)
    if s.shape != y.shape or y.shape != m.shape or s.ndim != 2:
        raise ValueError('Audit requires aligned [query,candidate] arrays')
    if not np.isfinite(s[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError('Nonfinite valid audit score/target')
    usable = m.any(-1)
    oracle = np.where(m, y, -np.inf).max(-1)
    idx = np.where(m, s, -np.inf).argmax(-1)
    selected = y[np.arange(len(y)), idx]
    def mean(a): return float(a.mean()) if a.size else None
    out = {'queries': int(usable.sum()), 'candidates': int(m.sum()),
           'selected_exact_utility': mean(selected[usable]),
           'oracle_exact_utility': mean(oracle[usable]),
           'selection_regret': mean((oracle-selected)[usable]),
           'top1_best_hit': mean((np.abs(oracle-selected)[usable]<1e-6).astype(float)),
           'utility_mae': mean(np.abs(s-y)[m])}
    if reference_scores is not None:
        rs = np.asarray(reference_scores)
        if rs.shape != s.shape: raise ValueError('Reference score shape differs')
        ridx = np.where(m, rs, -np.inf).argmax(-1)
        out.update(utility_shift=mean(np.abs(s-rs)[m]),
                   top1_changed=mean((idx[usable]!=ridx[usable]).astype(float)))
    return out
