"""Rep-C2-v2 decision diagnostic utilities.

This module tests whether the already-trained verifier contains useful decision
signals that were lost by reducing it to one global P(beneficial) threshold.

Decision rules:
  p_beneficial : original C2-v2 policy score.
  pred_delta   : auxiliary predicted utility difference.
  class_margin : P(beneficial) - P(harmful).
  fused_delta  : class-direction confidence times signed predicted magnitude.
  query_ridge  : lightweight ridge calibration using query-local signals only.
  context_ridge: same calibration plus frame-level A1 correction context.

The ridge calibrators use no injected-error label/case identity as an input.
They are fit on one half of Seen scenes. Thresholds are selected on the other
half and then frozen for Similar/Novel.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from rep_c2v2_common import full_query_gate_metrics


RULES = (
    "p_beneficial",
    "pred_delta",
    "class_margin",
    "fused_delta",
    "query_ridge",
    "context_ridge",
)

QUERY_FEATURE_NAMES = (
    "pred_delta",
    "p_beneficial",
    "p_harmful",
    "class_margin",
    "a1_margin",
    "offset_norm",
    "abs_offset_norm",
)

FRAME_FEATURE_NAMES = (
    "frame_proposal_rate",
    "frame_mean_offset_norm",
    "frame_mean_abs_offset_norm",
    "frame_positive_offset_fraction",
    "frame_negative_offset_fraction",
    "frame_mean_a1_margin",
    "frame_std_a1_margin",
    "frame_mean_abs_a1_margin",
)

CONTEXT_FEATURE_NAMES = QUERY_FEATURE_NAMES + FRAME_FEATURE_NAMES


@torch.no_grad()
def predict_verifier_outputs(model, ex, cache_frame, device, chunk=128):
    """Return [N,3] class probabilities and [N] predicted delta."""
    n=int(np.asarray(ex["actions"]).shape[1])
    if n==0:
        return np.empty((0,3),np.float32),np.empty(0,np.float32)
    hw=cache_frame["depth"].shape[-2:]
    image=torch.from_numpy(np.asarray(cache_frame["image_feature"]).copy()).to(device).float()
    K=torch.from_numpy(np.asarray(cache_frame["K"]).copy()).to(device).float()
    probs=[]; deltas=[]
    for start in range(0,n,int(chunk)):
        sl=slice(start,min(start+int(chunk),n))
        actions=torch.from_numpy(np.asarray(ex["actions"][:,sl]).copy()).to(device).float()
        vp=torch.from_numpy(np.asarray(ex["probabilities"][:,sl]).copy()).to(device).float()
        offsets=torch.from_numpy(np.asarray(ex["offsets_mm"][sl]).copy()).to(device).float()
        native=torch.from_numpy(np.asarray(ex["original_native_score"][sl]).copy()).to(device).float()
        out=model(image,K,actions,vp,offsets,native,hw)
        probs.append(out["class_logits"].softmax(-1).cpu().numpy())
        deltas.append(out["delta"].cpu().numpy())
    return np.concatenate(probs).astype(np.float32),np.concatenate(deltas).astype(np.float32)


def frame_context(ex, total_queries):
    """Inference-available frame-level A1 proposal statistics."""
    n=len(ex["offsets_mm"])
    q=max(1,int(total_queries))
    if n==0:
        return np.zeros(len(FRAME_FEATURE_NAMES),np.float32)
    offsets=np.asarray(ex["offsets_mm"],np.float64)/40.
    margin=(
        np.asarray(ex["probabilities"][1],np.float64).mean(-1)
        - np.asarray(ex["probabilities"][0],np.float64).mean(-1)
    )
    return np.asarray([
        n/q,
        offsets.mean(),
        np.abs(offsets).mean(),
        (offsets>0).mean(),
        (offsets<0).mean(),
        margin.mean(),
        margin.std(),
        np.abs(margin).mean(),
    ],np.float32)


def query_features(ex, class_prob, pred_delta):
    """Per-proposal deployable signals; no exact label or error-case input."""
    cp=np.asarray(class_prob,np.float32)
    pd=np.asarray(pred_delta,np.float32)
    n=len(pd)
    if cp.shape!=(n,3):
        raise ValueError(f"class_prob must be [N,3], got {cp.shape}")
    p_h=cp[:,0]
    p_b=cp[:,2]
    margin=(
        np.asarray(ex["probabilities"][1],np.float32).mean(-1)
        - np.asarray(ex["probabilities"][0],np.float32).mean(-1)
    )
    off=np.asarray(ex["offsets_mm"],np.float32)/40.
    return np.stack([
        pd,
        p_b,
        p_h,
        p_b-p_h,
        margin,
        off,
        np.abs(off),
    ],-1).astype(np.float32)


def context_features(ex, class_prob, pred_delta, total_queries):
    qf=query_features(ex,class_prob,pred_delta)
    fc=frame_context(ex,total_queries)
    return np.concatenate((qf,np.repeat(fc[None],len(qf),axis=0)),-1).astype(np.float32)


def raw_rule_scores(class_prob, pred_delta):
    cp=np.asarray(class_prob,np.float32)
    pd=np.asarray(pred_delta,np.float32)
    if cp.shape!=(len(pd),3):
        raise ValueError("class-prob/delta shape mismatch")
    p_h=cp[:,0]; p_b=cp[:,2]
    # A deliberately simple confidence-weighted signed utility heuristic.
    fused=p_b*np.maximum(pd,0.) - p_h*np.maximum(-pd,0.)
    return {
        "p_beneficial":p_b.astype(np.float32),
        "pred_delta":pd.astype(np.float32),
        "class_margin":(p_b-p_h).astype(np.float32),
        "fused_delta":fused.astype(np.float32),
    }


@dataclass
class RidgeCalibrator:
    feature_names: tuple[str,...]
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    ridge: float
    gain_weight: float

    def predict(self,x):
        x=np.asarray(x,np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names):
            raise ValueError(
                f"Expected [N,{len(self.feature_names)}], got {x.shape}"
            )
        z=(x-self.mean)/self.scale
        return (z@self.weight+self.bias).astype(np.float32)

    def to_json(self):
        return {
            "feature_names":list(self.feature_names),
            "mean":self.mean.tolist(),
            "scale":self.scale.tolist(),
            "weight":self.weight.tolist(),
            "bias":float(self.bias),
            "ridge":float(self.ridge),
            "gain_weight":float(self.gain_weight),
        }

    @classmethod
    def from_json(cls,d):
        return cls(
            feature_names=tuple(d["feature_names"]),
            mean=np.asarray(d["mean"],np.float64),
            scale=np.asarray(d["scale"],np.float64),
            weight=np.asarray(d["weight"],np.float64),
            bias=float(d["bias"]),
            ridge=float(d["ridge"]),
            gain_weight=float(d["gain_weight"]),
        )


def fit_ridge(x,y,feature_names,ridge=1e-2,gain_weight=4.):
    """Weighted ridge regression to true delta utility.

    Weighting by |delta| makes the diagnostic focus on utility-bearing decisions
    without using category balancing or injected-error labels.
    """
    x=np.asarray(x,np.float64)
    y=np.asarray(y,np.float64).reshape(-1)
    if x.ndim!=2 or len(x)!=len(y) or x.shape[1]!=len(feature_names):
        raise ValueError("Bad ridge training shapes")
    if len(y)==0:
        raise ValueError("Cannot fit ridge on zero examples")
    mean=x.mean(0)
    scale=x.std(0)
    scale=np.where(scale<1e-6,1.,scale)
    z=(x-mean)/scale
    design=np.concatenate((z,np.ones((len(z),1),np.float64)),-1)
    sw=np.sqrt(1.+float(gain_weight)*np.abs(y))
    a=design*sw[:,None]
    b=y*sw
    reg=np.eye(design.shape[1],dtype=np.float64)*float(ridge)
    reg[-1,-1]=0.  # do not regularize intercept
    theta=np.linalg.solve(a.T@a+reg,a.T@b)
    return RidgeCalibrator(
        tuple(feature_names),mean,scale,theta[:-1],float(theta[-1]),
        float(ridge),float(gain_weight),
    )


def threshold_grid(scores,n=201):
    """Quantile threshold grid including accept-all and reject-all endpoints."""
    s=np.asarray(scores,np.float64)
    s=s[np.isfinite(s)]
    if not len(s):
        raise ValueError("No finite decision scores")
    lo=float(s.min()); hi=float(s.max())
    span=max(abs(hi-lo),1.)
    eps=span*1e-7
    qs=np.quantile(s,np.linspace(0,1,max(3,int(n))))
    return np.unique(np.concatenate(([lo-eps],qs,[hi+eps])))


def _case_metric(records,threshold):
    if not records:
        raise ValueError("No records")
    score=np.concatenate([r["score"] for r in records])
    delta=np.concatenate([r["delta"] for r in records])
    total_q=sum(int(r["total_queries"]) for r in records)
    return full_query_gate_metrics(score,delta,threshold,total_q)


def select_global_threshold(records,n_grid=201):
    """Select one threshold by macro verifier increment over calibration cases."""
    if not records:
        raise ValueError("No threshold-calibration records")
    scores=np.concatenate([r["score"] for r in records])
    cases=sorted({r["case"] for r in records})
    rows=[]
    for thr in threshold_grid(scores,n_grid):
        by_case={
            c:_case_metric([r for r in records if r["case"]==c],thr)
            for c in cases
        }
        recoveries=[
            m.oracle_gap_recovery for m in by_case.values()
            if m.oracle_gap_recovery is not None
        ]
        row={
            "threshold":float(thr),
            "macro_full_query_utility_gain":float(np.mean([m.verified_gain for m in by_case.values()])),
            "macro_a1_fixed0_gain":float(np.mean([m.a1_fixed0_gain for m in by_case.values()])),
            "macro_oracle_accept_gain":float(np.mean([m.oracle_accept_gain for m in by_case.values()])),
            "macro_verifier_increment":float(np.mean([m.verifier_increment for m in by_case.values()])),
            "macro_oracle_gap_recovery":float(np.mean(recoveries)) if recoveries else 0.,
            "macro_beneficial_retention":float(np.mean([m.beneficial_retention for m in by_case.values()])),
            "macro_harmful_rejection":float(np.mean([m.harmful_rejection for m in by_case.values()])),
            "macro_accept_precision":float(np.mean([m.accept_precision for m in by_case.values()])),
            "cases":{c:vars(m) for c,m in by_case.items()},
        }
        rows.append(row)
    best=max(rows,key=lambda x:(
        x["macro_verifier_increment"],
        x["macro_oracle_gap_recovery"],
        x["macro_beneficial_retention"],
        x["macro_harmful_rejection"],
    ))
    return best,rows


def select_case_thresholds(records,n_grid=201):
    """Privileged diagnostic: separate threshold for each synthetic error case."""
    result={}
    sweeps={}
    for case in sorted({r["case"] for r in records}):
        rr=[r for r in records if r["case"]==case]
        grid=threshold_grid(np.concatenate([r["score"] for r in rr]),n_grid)
        rows=[]
        for thr in grid:
            m=_case_metric(rr,thr)
            rows.append(vars(m))
        result[case]=max(rows,key=lambda x:(
            x["verifier_increment"],
            -1 if x["oracle_gap_recovery"] is None else x["oracle_gap_recovery"],
            x["beneficial_retention"],
            x["harmful_rejection"],
        ))
        sweeps[case]=rows
    return result,sweeps


def split_seen_scenes(scene_ids,fraction=.5,seed=2029):
    scenes=np.unique(np.asarray(scene_ids,np.int64))
    if len(scenes)<2:
        raise ValueError("Need at least two Seen scenes for calibration split")
    rng=np.random.default_rng(int(seed))
    scenes=scenes.copy(); rng.shuffle(scenes)
    n=int(round(len(scenes)*float(fraction)))
    n=min(max(1,n),len(scenes)-1)
    return np.sort(scenes[:n]),np.sort(scenes[n:])


def save_calibration(path,payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False))


def load_calibration(path):
    return json.loads(Path(path).read_text())
