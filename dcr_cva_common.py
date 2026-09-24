"""Decoupled center selection / query ranking: pure, CPU-testable contracts.

Stage-1 scores are an ordering prior, NOT ground-truth success probabilities
for translated actions. The residual is trained on *cross-query* comparisons
of the actual selected actions. It never changes which center is executed.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from e1e2_common import select_centers

DCR_VERSION = 'decoupled_center_ranking_v1'
METHODS = ('native', 'local', 'stage1', 'anchored')


def checked_native_score(score: torch.Tensor) -> torch.Tensor:
    s = score.detach().float()
    if s.ndim != 1 or not bool(torch.isfinite(s).all()) or bool(((s < 0) | (s > 1)).any()):
        raise ValueError('Frozen Stage-1 score must be finite [Q] in [0,1]')
    return s


def anchor_logit(score: torch.Tensor) -> torch.Tensor:
    return torch.logit(checked_native_score(score).clamp(1e-6, 1-1e-6))


def anchored_score(score: torch.Tensor, residual: torch.Tensor, strength: float = 1.) -> torch.Tensor:
    """Bounded log-odds update; zero residual/strength reproduces Stage-1 exactly.

    Odds form avoids a logit/sigmoid round-trip for the no-update case. Scores
    at the closed interval endpoints remain endpoints. Residual bounds are
    supplied by RankResidualHead, not by this function.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('rank strength must be in [0,1]')
    s = checked_native_score(score)
    if residual.shape[-1:] != s.shape or not bool(torch.isfinite(residual).all()):
        raise ValueError('Residual must be finite [...,Q]')
    if strength == 0:
        return s.expand_as(residual)
    d = residual.float()*strength
    return s / (s + (1-s)*torch.exp(-d))


class RankResidualHead(nn.Module):
    """One zero-initialized, bounded ranking head; no ranking gradient upstream.

    [native latent, candidate latent, difference, 3x six-threshold CDF profiles,
     camera-z offset, |offset|, native score] -> residual log-odds.
    No case identity, sensor depth, CAD label, or selected GT utility is input.
    """
    def __init__(self, feature_dim: int, hidden: int = 128, bound: float = .5):
        super().__init__()
        if min(feature_dim, hidden) < 1 or not math.isfinite(bound) or not 0 < bound <= 4:
            raise ValueError('Invalid rank-head dimensions/bound')
        self.feature_dim, self.bound = int(feature_dim), float(bound)
        self.net = nn.Sequential(nn.LayerNorm(3*feature_dim+21),
                                 nn.Linear(3*feature_dim+21, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, cdf_logits, native_score, offsets_mm, zero):
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError('Expected candidate features [C,Q,H]')
        c, q, _ = features.shape
        if cdf_logits.shape != (c,q,6) or offsets_mm.shape != (c,):
            raise ValueError('Feature/CDF/offset contract mismatch')
        if not 0 <= zero < c:
            raise ValueError('Invalid native center index')
        h = features.detach().float()
        p = cdf_logits.detach().float().sigmoid()
        h0, p0 = h[zero:zero+1].expand_as(h), p[zero:zero+1].expand_as(p)
        s = checked_native_score(native_score)
        if len(s) != q:
            raise ValueError('Native score/query mismatch')
        off = offsets_mm.detach().float().view(c,1,1).expand(c,q,1)/40.
        x = torch.cat((h0,h,h-h0,p0,p,p-p0,off,off.abs(),s[None,:,None].expand(c,q,1)), -1)
        residual = self.bound*torch.tanh(self.net(x).squeeze(-1))
        # No physical correction => no score correction. This is a structural
        # anchor, stronger than asking a penalty to learn native identity.
        mask = torch.ones(c,1,device=residual.device,dtype=residual.dtype)
        mask[zero] = 0
        return residual*mask


def pairwise_ranking_loss(logits: torch.Tensor, utility: torch.Tensor):
    """All unequal-utility pairs of DISTINCT queries in ONE scene/frame.

    Utility-gap-weighted logistic ranking, no cross-center pairs and no pairs
    across unrelated frames. Equal utilities carry no ranking label.
    """
    if logits.ndim != 1 or utility.shape != logits.shape:
        raise ValueError('Expected ranking logits and true utility [Q]')
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(utility).all()):
        raise ValueError('Nonfinite rank input/label')
    y = utility.detach().float()
    i,j = torch.triu_indices(len(y),len(y),offset=1,device=y.device)
    gap = y[i]-y[j]
    keep = gap.abs() > 1e-7
    i,j,gap = i[keep],j[keep],gap[keep]
    if not len(gap):
        return logits.sum()*0., 0
    losses = F.softplus(-gap.sign()*(logits[i]-logits[j]))
    return (losses*gap.abs()).sum()/gap.abs().sum(), int(len(gap))


def selected_rank_loss(residual, native_score, selected, target, anchor_weight=.1):
    """Hard choice is made by the correction path; rank loss cannot alter it."""
    if anchor_weight < 0 or not math.isfinite(anchor_weight):
        raise ValueError('Invalid anchor penalty')
    q = torch.arange(len(selected),device=residual.device)
    sel = selected.detach()
    r = residual[sel,q]
    y = target.detach().float().mean(-1)[sel,q]
    pair, n = pairwise_ranking_loss(anchor_logit(native_score)+r,y)
    # Shrink actual log-odds updates, not logits/utility of the correction path.
    anchor = r.square().mean()
    loss = pair + anchor_weight*anchor
    return loss, {'rank_pair':float(pair.detach()), 'rank_anchor':float(anchor.detach()),
                  'rank_pairs':n, 'rank_abs_residual':float(r.detach().abs().mean())}


def make_outputs(cdf_logits, residual, bundle, zero, rank_strength=1.):
    """All three corrected dumps have IDENTICAL physical poses and row order."""
    utility = cdf_logits.float().sigmoid().mean(-1)
    sel = select_centers(utility,bundle['valid'],zero)
    q = torch.arange(len(sel),device=sel.device)
    native = bundle['actions'][zero].detach().clone()
    s0 = checked_native_score(native[:,0])
    physical = bundle['actions'][sel,q].detach().clone()
    score = {'local':utility[sel,q], 'stage1':s0,
             'anchored':anchored_score(s0,residual[sel,q],rank_strength)}
    outputs = {'native':native}
    for name,s in score.items():
        out = physical.clone(); out[:,0] = s; outputs[name] = out
    return outputs,sel


def rank_metrics(score, utility):
    """Diagnostics on sampled queries, NOT official AP or a success guarantee."""
    s = score.detach().float(); y = utility.detach().float()
    i,j = torch.triu_indices(len(y),len(y),offset=1,device=y.device)
    gap = y[i]-y[j]; mask = gap.abs()>1e-7
    products = (s[i]-s[j])*gap
    correct = (products[mask]>0).float()+.5*(products[mask]==0).float()
    result = {'pair_concordance':float(correct.mean()) if len(correct) else None,
              'rank_pairs':int(mask.sum())}
    order = torch.argsort(s,descending=True,stable=True)
    for k in (10,50):
        result[f'top{k}_utility'] = float(y[order[:min(k,len(y))]].mean())
    return result


def code_fingerprint():
    """New training signatures track implementation as well as CLI settings."""
    from pathlib import Path
    from e1e2_common import digest, file_sha
    root = Path(__file__).resolve().parent
    files = ('dcr_cva_common.py','models/economicgrasp_cva_dcr.py',
             'models/economicgrasp_cva_centers.py','train_dcr_cva.py')
    return digest({p:file_sha(root/p) for p in files})
