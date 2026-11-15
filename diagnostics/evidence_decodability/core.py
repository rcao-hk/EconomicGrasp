"""Matched-capacity P1 action-quality/failure decodability probes.

Evidence originates from the frozen 50%+50% RGB-predicted-depth GVAR model.
CAD/GT action labels are consumed only by objectives/metrics, never forward().
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diagnostics.decision_supervision.core import BASE_SHA, physical_ids

SCHEMA = "physical-evidence-decoding-v1"
ROLES = ("left_contact", "right_contact", "closing", "finger_body", "palm", "approach")
MODES = ("score64", "pre_reader", "post_reader", "reader_delta",
         "role_tokens", "pred_geometry", "score_plus_roles",
         "score_plus_geometry", "all", "reference_geometry")
TASKS = ("quality", "failure")
FAILURE_NAMES = ("success", "collision", "empty", "other_zero_quality")
MU = np.asarray([.2, .4, .6, .8, 1., 1.2], dtype=np.float64)

FEATURE_FIELDS = (
    "score64", "pre_reader", "post_reader", "reader_delta",
    "role_tokens", "role_mass", "role_valid", "pred_geometry",
)
LABEL_FIELDS = ("quality_count", "quality_bits", "pure_collision", "empty")
ROW_FIELDS = ("query_id", "depth_id", "native_score", "base_logit", "grasps",
              "physical_id", "source_new", "sensor_valid", "nms_role")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)


def failure_class(count, pure_collision, empty):
    """Mutually exclusive classes; prioritize collision over empty."""
    count = np.asarray(count, dtype=np.int64)
    collision = np.asarray(pure_collision, dtype=bool)
    empty = np.asarray(empty, dtype=bool)
    if not (count.shape == collision.shape == empty.shape):
        raise ValueError("Failure label shape mismatch")
    if np.any((count < 0) | (count > 6)):
        raise ValueError("Quality count must lie in [0,6]")
    if np.any((count > 0) & (collision | empty)):
        raise ValueError("Positive-quality action cannot have collision/empty flag")
    label = np.zeros(count.shape, dtype=np.int64)
    negative = count == 0
    label[negative] = 3
    label[negative & empty] = 2
    label[negative & collision] = 1
    return label


def failure_metrics(logits, ground_truth):
    logits = np.asarray(logits, dtype=np.float64)
    true = np.asarray(ground_truth, dtype=np.int64).reshape(-1)
    if logits.shape != (len(true), 4) or not np.isfinite(logits).all():
        raise ValueError("Invalid four-class logits")
    prediction = logits.argmax(axis=1)
    confusion = np.zeros((4,4), dtype=np.int64)
    for y, p in zip(true, prediction):
        if not 0 <= y < 4:
            raise ValueError("Invalid class label")
        confusion[int(y), int(p)] += 1
    recalls = []
    f1s = []
    for i in range(4):
        tp = int(confusion[i,i])
        fn = int(confusion[i].sum()-tp)
        fp = int(confusion[:,i].sum()-tp)
        recalls.append(tp/(tp+fn) if tp+fn else None)
        f1s.append(2*tp/(2*tp+fn+fp) if (2*tp+fn+fp) else None)
    return dict(confusion=confusion.tolist(), recall=recalls, f1=f1s,
                balanced_accuracy=float(np.mean([x for x in recalls if x is not None])),
                macro_f1=float(np.mean([x for x in f1s if x is not None])),
                samples=int(len(true)))


def check_bank(x, *, require_labels=True):
    needed = FEATURE_FIELDS + ROW_FIELDS + (LABEL_FIELDS if require_labels else ())
    if any(k not in x for k in needed):
        raise ValueError(f"Missing bank fields: {[k for k in needed if k not in x]}")
    n = len(x["query_id"])
    if not n:
        raise ValueError("Empty action evidence bank")
    if np.asarray(x["query_id"]).shape != (n,) or np.asarray(x["depth_id"]).shape != (n,):
        raise ValueError("Action identity IDs must be one-dimensional")
    if np.asarray(x["grasps"]).shape != (n,17):
        raise ValueError("Grasps shape mismatch")
    if np.asarray(x["score64"]).shape != (n,64):
        raise ValueError("Expected frozen 64D score feature")
    H = int(x["pre_reader"].shape[-1])
    if H < 16 or any(np.asarray(x[k]).shape != (n,H)
                     for k in ("pre_reader","post_reader","reader_delta")):
        raise ValueError("Reader feature shapes incompatible")
    if np.asarray(x["role_tokens"]).shape != (n,6,64):
        raise ValueError("Expected six pooled 64D role tokens")
    for k in ("role_mass","role_valid"):
        if np.asarray(x[k]).shape != (n,6):
            raise ValueError(f"Expected 6 role scalars: {k}")
    if np.asarray(x["pred_geometry"]).shape != (n,6,6):
        raise ValueError("Expected six roles x six relative geometry values")
    if "reference_geometry" in x and np.asarray(x["reference_geometry"]).shape != (n,6,6):
        raise ValueError("Invalid privileged reference geometry")
    for k in ("native_score","base_logit","source_new","sensor_valid","nms_role","physical_id"):
        if np.asarray(x[k]).shape != (n,):
            raise ValueError(f"Invalid one-dimensional field {k}")
    if not np.array_equal(physical_ids(x["grasps"]), x["physical_id"].astype(str)):
        raise ValueError("Executed physical-action identity changed")
    if len(np.unique(x["query_id"])) != n:
        raise ValueError("More than one executed depth per query in native-only bank")
    if np.any((x["query_id"]<0)|(x["query_id"]>=1024)|
              (x["depth_id"]<0)|(x["depth_id"]>3)):
        raise ValueError("Query/depth identifiers out of range")
    if not np.isfinite(x["native_score"]).all() or not np.isfinite(x["base_logit"]).all():
        raise ValueError("Nonfinite native scores/logits")
    for k in FEATURE_FIELDS + ("grasps",):
        if not np.isfinite(x[k]).all():
            raise ValueError(f"Nonfinite evidence: {k}")
    if "reference_geometry" in x and not np.isfinite(x["reference_geometry"]).all():
        raise ValueError("Nonfinite reference geometry")
    if require_labels:
        q = np.asarray(x["quality_count"])
        bits = np.asarray(x["quality_bits"])
        if q.shape != (n,) or bits.shape != (n,6):
            raise ValueError("Quality labels not aligned to executed actions")
        if not np.isin(bits,[0,1]).all() or not np.array_equal(bits.sum(1),q):
            raise ValueError("Six-friction success bits/count mismatch")
        if not np.all(np.diff(bits.astype(int),axis=1)>=0):
            raise ValueError("Friction success must be monotone")
        for k in ("pure_collision","empty"):
            if np.asarray(x[k]).shape != (n,):
                raise ValueError(f"Malformed failure metadata: {k}")
        failure_class(q,x["pure_collision"],x["empty"])
    return H


def feature_dim(hidden_dim):
    return 64 + 3*hidden_dim + (6*64) + 6 + 6 + 6*6 + 15 + 4 + 1


def enabled_groups(mode):
    mapping = {
        "score64": {"score64"},
        "pre_reader": {"pre_reader"},
        "post_reader": {"post_reader"},
        "reader_delta": {"reader_delta"},
        "role_tokens": {"role_tokens","role_mass","role_valid"},
        "pred_geometry": {"pred_geometry"},
        "score_plus_roles": {"score64","role_tokens","role_mass","role_valid"},
        "score_plus_geometry": {"score64","pred_geometry"},
        "all": {"score64","pre_reader","post_reader","reader_delta",
                "role_tokens","role_mass","role_valid","pred_geometry"},
        "reference_geometry": {"reference_geometry"},
    }
    if mode not in mapping:
        raise ValueError(f"Unknown evidence mode: {mode}")
    return mapping[mode]


class EvidenceProbe(nn.Module):
    """Identical parameter count per task across evidence modes (masked inputs).

    Scalar quality is a residual over the original CDF-expected utility logit.
    Failure classification is an independent four-way diagnostic task.
    """
    def __init__(self, hidden_dim, mode, task="quality", width=64):
        super().__init__()
        if mode not in MODES or task not in TASKS or hidden_dim < 16 or width < 1:
            raise ValueError("Invalid probe mode/task/hidden size")
        self.mode = mode
        self.task = task
        self.hidden_dim = int(hidden_dim)
        self.width = int(width)
        self.net = nn.Sequential(nn.Linear(feature_dim(hidden_dim),width),
                                 nn.GELU(),nn.Linear(width,1 if task=="quality" else 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        enabled=enabled_groups(self.mode)
        size=len(x["query_id"])
        device=next(self.parameters()).device
        dtype=torch.float32
        def convert(key):
            return torch.as_tensor(x[key],dtype=dtype,device=device).reshape(size,-1)
        keys = ("score64","pre_reader","post_reader","reader_delta",
                "role_tokens","role_mass","role_valid","pred_geometry")
        groups=[]
        for key in keys:
            data=convert(key)
            if key not in enabled:
                data=torch.zeros_like(data)
            groups.append(data)
        if self.mode=="reference_geometry":
            if "reference_geometry" not in x:
                raise ValueError("Privileged reference geometry was not explicitly provided")
            groups[-1]=convert("reference_geometry")
        grasp=convert("grasps")[:,1:16].clone()
        grasp[:,0] /= .1
        grasp[:,1] /= .02
        grasp[:,2] /= .04
        depth=torch.as_tensor(x["depth_id"],dtype=torch.long,device=device)
        depth_oh=F.one_hot(depth,num_classes=4).to(dtype)
        logit=torch.as_tensor(x["base_logit"],device=device,dtype=dtype).reshape(-1,1)
        if logit.shape!=(size,1):
            raise ValueError("Base score logit missing")
        input_tensor=torch.cat([*groups,grasp,depth_oh,logit],dim=-1)
        if input_tensor.shape[1] != feature_dim(self.hidden_dim):
            raise RuntimeError("Matched-capacity input contract broken")
        output=self.net(input_tensor)
        return output[:,0]+logit[:,0] if self.task=="quality" else output


def quality_stats(scores, q):
    scores=np.asarray(scores,dtype=np.float64).reshape(-1)
    q=np.asarray(q,dtype=np.float64).reshape(-1)
    if scores.shape!=q.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid scores/qualities")
    mse=float(np.mean((scores-q)**2))
    good=np.triu_indices(len(scores),k=1)
    i,j=good
    valid=q[i]!=q[j]
    if not valid.any():
        agree=None
    else:
        diff=scores[i[valid]]-scores[j[valid]]
        truth=q[i[valid]]-q[j[valid]]
        agree=float(np.mean(np.where(abs(diff)<=1e-9,.5,((diff*truth)>0).astype(float))))
    return dict(mse=mse,pair_agreement=agree,
                top10_quality=float(q[np.argsort(-scores)[:10]].mean()),
                top10_zero=float((q[np.argsort(-scores)[:10]]==0).mean()),
                mean_quality=float(q.mean()),samples=int(len(scores)))
