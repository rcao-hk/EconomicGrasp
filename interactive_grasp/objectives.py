"""Shared loss for all four variants. Missing labels are not negative labels."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def sparse_quality_loss(logit, count, label_mask, action_valid, *, pair_seed=0,
                        max_pairs=2048, mse_weight=10.0, pair_weight=0.5):
    if logit.ndim != 2 or any(x.shape != logit.shape for x in (count,label_mask,action_valid)):
        raise ValueError("Loss requires [B,N] tensors")
    if count.is_floating_point() or label_mask.dtype != torch.bool or action_valid.dtype != torch.bool:
        raise ValueError("Integer count and boolean masks required")
    if max_pairs < 1 or min(mse_weight, pair_weight) < 0:
        raise ValueError("Invalid loss weights/pair budget")
    if (label_mask & ~action_valid).any():
        raise ValueError("A padded action cannot be labelled")
    losses=[]; mses=[]; pairs=[]; eligible=used=labelled=0
    for b in range(len(logit)):
        mask=label_mask[b] & action_valid[b]
        if not mask.any():
            raise ValueError("Training/validation frame has no supervised actions")
        y=count[b,mask]; z=logit[b,mask]
        if ((y<0)|(y>6)).any() or not torch.isfinite(z).all():
            raise ValueError("Invalid supervised quality/logit")
        point=F.mse_loss(z.sigmoid(), y.to(z.dtype)/6)
        ycpu=y.detach().cpu()
        ij=torch.triu_indices(len(y),len(y),offset=1)
        ij=ij[:,ycpu[ij[0]]!=ycpu[ij[1]]]
        eligible+=ij.shape[1]
        if ij.shape[1]>max_pairs:
            rng=torch.Generator().manual_seed(pair_seed+104729*b)
            ij=ij[:,torch.randperm(ij.shape[1],generator=rng)[:max_pairs]]
        used+=ij.shape[1];labelled+=len(y)
        if ij.shape[1]:
            ij=ij.to(z.device)
            sign=torch.sign(y[ij[0]].to(z.dtype)-y[ij[1]].to(z.dtype))
            pair=F.softplus(-sign*(z[ij[0]]-z[ij[1]])).mean()
        else:
            pair=z.sum()*0.0
        losses.append(mse_weight*point+pair_weight*pair)
        mses.append(point.detach());pairs.append(pair.detach())
    return torch.stack(losses).mean(), {
        "mse":torch.stack(mses).mean().item(), "pair":torch.stack(pairs).mean().item(),
        "labelled":labelled,"eligible_pairs":eligible,"used_pairs":used}


def pair_changes(base, updated, count, mask):
    """Exact comparisons only on supplied labelled actions, not global ranks/AP."""
    base, updated, y = [np.asarray(v)[mask] for v in (base,updated,count)]
    i,j=np.triu_indices(len(y),1)
    valid=y[i]!=y[j];i,j=i[valid],j[valid]
    if not len(i):
        return dict(comparable_pairs=0,repair=0,damage=0,base_accuracy=None,accuracy=None)
    sign=np.sign(y[i]-y[j]);a=np.sign(base[i]-base[j])*sign;b=np.sign(updated[i]-updated[j])*sign
    return dict(comparable_pairs=len(i),repair=int(np.sum((a<0)&(b>0))),
                damage=int(np.sum((a>0)&(b<0))),
                base_accuracy=float(np.mean(np.where(a==0,.5,a>0))),
                accuracy=float(np.mean(np.where(b==0,.5,b>0))))
