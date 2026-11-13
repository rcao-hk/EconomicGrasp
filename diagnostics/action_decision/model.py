"""Capacity-matched frozen-feature quality probes and selection-aware losses."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diagnostics.candidate_conflict.core import build_conflict_edges

MODES=("base_geometry","score64","reader256","reader_delta","score_reader")
OBJECTIVES=("mse","rare","decision")
INPUT_DIM=1+4+15+64+256+256


def action_geometry(grasps: np.ndarray) -> np.ndarray:
    """Only executed physical 17D action, not evaluator labels or score."""
    g=np.asarray(grasps,dtype=np.float32)
    if g.ndim!=2 or g.shape[1]!=17 or not np.isfinite(g).all():
        raise ValueError("Need finite [N,17] grasps")
    x=np.concatenate((
        g[:,1:2]/0.1,
        g[:,2:3]/0.02,
        g[:,3:4]/0.04,
        g[:,4:13],
        g[:,13:16]/0.5,
    ),axis=1)
    if x.shape!=(len(g),15):raise RuntimeError("Physical geometry feature size")
    return x


def make_input(source: dict, added: dict, device, *, require_reader=True):
    """Verify paired exact-action cache and reader capture before tensorization."""
    n=len(source["base_logit"])
    for k in ("features","base_logit","depth","count","family","grasps","physical_id"):
        if k not in source:raise ValueError(f"Missing original cache key {k}")
    if added is not None:
        for k in ("reader","reader_delta","depth","family","physical_id"):
            if k not in added:raise ValueError(f"Missing GVAR reader cache key {k}")
        if (added["reader"].shape!=(n,256) or added["reader_delta"].shape!=(n,256) or
            not np.array_equal(added["physical_id"],source["physical_id"]) or
            not np.array_equal(added["depth"],source["depth"]) or
            not np.array_equal(added["family"],source["family"])):
            raise RuntimeError("Reader/label physical action identity mismatch")
    elif require_reader:
        raise ValueError("Missing actual GVAR reader feature cache")
    if (source["features"].shape!=(n,64) or
        source["count"].shape!=(n,) or source["grasps"].shape!=(n,17)):
        raise ValueError("Invalid quality cache shape")
    reader=np.zeros((n,256),np.float32) if added is None else added["reader"].astype(np.float32)
    delta=np.zeros((n,256),np.float32) if added is None else added["reader_delta"].astype(np.float32)
    result=dict(
        base=torch.as_tensor(np.asarray(source["base_logit"],dtype=np.float32),device=device),
        score64=torch.as_tensor(np.asarray(source["features"],dtype=np.float32),device=device),
        reader=torch.as_tensor(reader,device=device),
        reader_delta=torch.as_tensor(delta,device=device),
        geometry=torch.as_tensor(action_geometry(source["grasps"]),device=device),
        depth=torch.as_tensor(np.asarray(source["depth"],dtype=np.int64),device=device),
        count=torch.as_tensor(np.asarray(source["count"],dtype=np.int64),device=device),
        family=torch.as_tensor(np.asarray(source["family"],dtype=np.int64),device=device),
        grasps=np.asarray(source["grasps"],dtype=np.float32),
    )
    if any(not torch.isfinite(result[k]).all() for k in
           ("base","score64","reader","reader_delta","geometry")):
        raise ValueError("Nonfinite probe input")
    if not torch.all((result["count"]>=0)&(result["count"]<=6)):
        raise ValueError("Exact six-mu count outside 0..6")
    return result


class ActionQualityProbe(nn.Module):
    """Every feature mode uses identical parameter shapes (masked inputs).

    Residual initialized exactly to zero so epoch0 returns the frozen score.
    P0 isolates representation informativeness under controlled capacity.
    """
    def __init__(self, mode: str, hidden: int=64):
        super().__init__()
        if mode not in MODES:raise ValueError(f"Unknown feature mode {mode}")
        self.mode=mode
        self.norm64=nn.LayerNorm(64)
        self.norm256=nn.LayerNorm(256)
        self.norm_delta=nn.LayerNorm(256)
        self.network=nn.Sequential(nn.Linear(INPUT_DIM,hidden),
                                   nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: dict):
        base=x["base"].float().reshape(-1)
        n=len(base)
        g=x["geometry"].float()
        depth=x["depth"].long()
        if g.shape!=(n,15) or depth.shape!=(n,) or base.shape!=(n,):
            raise ValueError("Probe batch shape mismatch")
        zeros=lambda dim:torch.zeros(n,dim,device=base.device,dtype=base.dtype)
        score=self.norm64(x["score64"].float()) if self.mode in ("score64","score_reader") else zeros(64)
        reader=self.norm256(x["reader"].float()) if self.mode in ("reader256","score_reader") else zeros(256)
        delta=self.norm_delta(x["reader_delta"].float()) if self.mode=="reader_delta" else zeros(256)
        features=torch.cat((base[:,None],
            F.one_hot(depth,4).to(base.dtype),g,score,reader,delta),dim=-1)
        if features.shape!=(n,INPUT_DIM):raise RuntimeError("Feature schema mismatch")
        return base+self.network(features).reshape(-1)


def _valid_pairs(count: torch.Tensor):
    y=count.detach().cpu().numpy().astype(np.int64)
    rows,cols=np.triu_indices(len(y),k=1)
    good=y[rows]!=y[cols]
    return np.stack((rows[good],cols[good]),axis=1)


def _rank_pair_loss(z,count,pairs):
    if len(pairs)==0:return z.sum()*0
    idx=torch.as_tensor(pairs,dtype=torch.long,device=z.device)
    sign=torch.sign((count[idx[:,0]]-count[idx[:,1]]).to(z.dtype))
    if torch.any(sign==0):
        raise RuntimeError("Quality-tie pairs must be excluded")
    return F.softplus(-sign*(z[idx[:,0]]-z[idx[:,1]])).mean()


def choose_pairs(source: dict, objective: str, budget: int, seed: int,
                 top_k: int=16):
    """RARE: random all-image non-tie pairs.
    Decision: source-score top-query conflicts first, source-score inversions
    second, then representative other pairs, all from THE SAME labelled pool.
    No test labels, no dynamic model-generated graph or success-score leakage.
    """
    if objective not in OBJECTIVES or budget<1 or top_k<1:
        raise ValueError("Bad pair objective/budget")
    count=np.asarray(source["count"],dtype=np.int64)
    all_pairs=_valid_pairs(torch.as_tensor(count))
    if objective=="mse":
        return np.zeros((0,2),np.int64),dict(eligible=len(all_pairs),used=0,
                                            conflict=0,hard=0)
    rng=np.random.default_rng(seed)
    if objective=="rare":
        if len(all_pairs)>budget:
            all_pairs=all_pairs[rng.choice(len(all_pairs),budget,replace=False)]
        return all_pairs,dict(eligible=len(_valid_pairs(torch.as_tensor(count))),
                              used=len(all_pairs),conflict=0,hard=0)
    # Use one frozen native-depth winner per labelled query. Its orientation,
    # center and width are *physical*; NMS conflicts depend on center and R.
    family=np.asarray(source["family"],dtype=np.int64)
    base=np.asarray(source["base_logit"],dtype=np.float64)
    unique=np.unique(family)
    winners=[]
    for f in unique:
        ix=np.flatnonzero(family==f)
        winners.append(int(ix[np.argmax(base[ix])]))
    winners=np.asarray(winners,dtype=np.int64)
    order=np.argsort(base[winners])[::-1]
    top=winners[order[:min(top_k,len(winners))]]
    edges=build_conflict_edges(np.asarray(source["grasps"])[winners])
    conflict=[tuple(sorted((int(winners[i]),int(winners[j]))))
              for i,j in edges if count[winners[i]]!=count[winners[j]]
              and (winners[i] in top or winners[j] in top)]
    hard=[]
    for i in top:
        for j in winners:
            if i==j or count[i]==count[j]:continue
            if (count[i]-count[j])*(base[i]-base[j])<0:
                hard.append(tuple(sorted((int(i),int(j)))))
    special=sorted(set(conflict)) + sorted(set(hard)-set(conflict))
    selected=special[:budget]
    if len(selected)<budget:
        have=set(selected)
        remaining=[tuple(p) for p in all_pairs if tuple(p) not in have]
        need=min(budget-len(selected),len(remaining))
        if need: selected.extend([remaining[int(i)] for i in
                                   rng.choice(len(remaining),need,replace=False)])
    used=np.asarray(selected,dtype=np.int64).reshape(-1,2)
    return used,dict(eligible=len(all_pairs),used=len(used),
                     conflict=min(len(conflict),len(used)),
                     hard=len(set(used.flatten().tolist()) & set(top.tolist())),
                     initial_conflict=len(conflict))


def quality_losses(z: torch.Tensor, data: dict, objective: str,
                   budget: int, seed: int, top_k: int=16):
    if z.ndim!=1 or z.shape!=data["count"].shape or not torch.isfinite(z).all():
        raise ValueError("Bad quality predictions")
    q=data["count"].float()/6.0
    mse=F.mse_loss(z.sigmoid(),q)
    pairs,stats=choose_pairs(
        {"count":data["count"].detach().cpu().numpy(),
         "family":data["family"].detach().cpu().numpy(),
         "base_logit":data["base"].detach().cpu().numpy(),
         "grasps":data["grasps"]},objective,budget,seed,top_k)
    pair=_rank_pair_loss(z,data["count"],pairs)
    return mse,pair,stats
