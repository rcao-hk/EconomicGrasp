"""Frozen-backbone quality probes with matched-capacity modality interventions.

P0 compares the information carried by native score, executed action geometry,
and 64D post-GVAR score features. No test labels may enter these probes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rare_grasp.core import physical_ids, ranking_loss, validate_cache

SCHEMA = "quality-decision-p0p2-v1"
MODES = ("score", "score_geometry", "score_feature", "score_feature_geometry",
         "score_feature_permuted")
MU = np.array([.2, .4, .6, .8, 1., 1.2], dtype=np.float64)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def action_geometry(grasp):
    """Physical geometry only: 0.1-m width, 0.02-m height, 0.04-m insertion,
    SO(3) rotation and camera-frame center. Never use evaluator/GT fields.
    """
    if grasp.ndim != 2 or grasp.shape[1] != 17:
        raise ValueError("Expected [N,17] executed grasps")
    g = grasp[:, 1:16].clone()
    g[:, 0] /= .1
    g[:, 1] /= .02
    g[:, 2] /= .04
    return g


class QualityProbe(nn.Module):
    """Same architecture and number of trainable parameters in every P0 arm.

    Output: residual over original CDF-expected-quality logit. Initial head is
    exactly native scoring; changing modes only masks input groups.
    """
    def __init__(self, mode: str, hidden: int = 64):
        super().__init__()
        if mode not in MODES or hidden < 1:
            raise ValueError(f"Invalid probe mode: {mode}")
        self.mode = mode
        self.hidden = int(hidden)
        self.feature_norm = nn.LayerNorm(64)
        self.net = nn.Sequential(nn.Linear(64 + 15 + 4 + 1, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat, base_logit, grasps, depth, permutation=None):
        if feat.ndim != 2 or feat.shape[1] != 64:
            raise ValueError("Expected [N,64] frozen action features")
        n = feat.shape[0]
        if base_logit.shape != (n,) or grasps.shape != (n, 17) or depth.shape != (n,):
            raise ValueError("Batch shape disagreement")
        if not torch.isfinite(feat).all() or not torch.isfinite(base_logit).all():
            raise ValueError("Nonfinite inputs")
        if ((depth < 0) | (depth > 3)).any():
            raise ValueError("Invalid insertion depth IDs")
        features = self.feature_norm(feat)
        geom = action_geometry(grasps)
        if self.mode == "score_feature_permuted":
            if permutation is None or permutation.shape != (n,):
                raise ValueError("Permuted negative control requires deterministic permutation")
            features = features.index_select(0, permutation)
        elif self.mode not in ("score_feature", "score_feature_geometry"):
            features = torch.zeros_like(features)
        if self.mode not in ("score_geometry", "score_feature_geometry"):
            geom = torch.zeros_like(geom)
        inputs = torch.cat([features, geom,
                            F.one_hot(depth.long(), num_classes=4).to(feat.dtype),
                            base_logit[:, None]], dim=-1)
        return base_logit + self.net(inputs).squeeze(-1)


def frame_arrays(x: dict):
    validate_cache(x)
    return {k: x[k] for k in
            ("features", "base_logit", "grasps", "depth", "count", "family")}


def predictor(model, arrays, device, seed=0):
    x = frame_arrays(arrays)
    feat = torch.as_tensor(x["features"], device=device, dtype=torch.float32)
    base = torch.as_tensor(x["base_logit"], device=device, dtype=torch.float32)
    grasp = torch.as_tensor(x["grasps"], device=device, dtype=torch.float32)
    depth = torch.as_tensor(x["depth"], device=device, dtype=torch.long)
    if model.mode == "score_feature_permuted":
        rng = np.random.default_rng(seed)
        perm = torch.as_tensor(rng.permutation(len(feat)), device=device, dtype=torch.long)
    else:
        perm = None
    return model(feat, base, grasp, depth, permutation=perm)


def objective(z, counts, family, pair_weight=0., pair_scope="image",
              max_pairs=2048, seed=0):
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite scorer logits")
    target = counts.to(z.dtype) / 6
    abs_loss = F.mse_loss(z.sigmoid(), target)
    if pair_weight:
        _, rank_loss, audit = ranking_loss(
            z, counts, family, scope=pair_scope,
            max_pairs=max_pairs, seed=seed
        )
    else:
        rank_loss = z.sum() * 0
        audit = dict(eligible_pairs=None, used_pairs=0)
    return 10. * abs_loss + pair_weight * rank_loss, dict(
        quality_mse=float(abs_loss.detach()), pair=float(rank_loss.detach()),
        **audit
    )


def frame_metrics(scores, arrays):
    validate_cache(arrays)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(arrays["count"], dtype=np.float64).reshape(-1) / 6
    family = np.asarray(arrays["family"]).reshape(-1)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Predictions mismatch")
    groups = []
    for f in np.unique(family):
        idx = np.flatnonzero(family == f)
        if len(idx) != 4:
            raise ValueError("Incomplete four-depth family")
        groups.append((scores[idx], y[idx]))
    selected = np.mean([v[np.argmax(s)] for s, v in groups])
    oracle = np.mean([np.max(v) for _, v in groups])
    consistent = []
    for s, v in groups:
        for i in range(4):
            for j in range(i + 1, 4):
                if v[i] == v[j]:
                    continue
                ds = s[i] - s[j]
                consistent.append(.5 if abs(ds) < 1e-9 else float(ds * (v[i] - v[j]) > 0))
    return dict(mse=float(np.mean((scores-y)**2)),
                selected_q=float(selected), oracle_q=float(oracle),
                regret=float(oracle-selected),
                pairwise=float(np.mean(consistent)) if consistent else None,
                pairwise_n=len(consistent), actions=len(scores),
                positive_rate=float((y>0).mean()), mean_q=float(y.mean()))


def scene_equal_aggregate(rows):
    """Equal domain->scene->frame weight; no frame-level IID CIs."""
    if not rows:
        raise ValueError("Empty validation rows")
    result={}
    for field in ("mse", "selected_q", "oracle_q", "regret", "pairwise"):
        domains = sorted(set(r["domain"] for r in rows))
        dvals=[]
        for domain in domains:
            scenes = sorted({r["scene"] for r in rows if r["domain"] == domain})
            sval=[]
            for scene in scenes:
                values=[r[field] for r in rows
                        if r["domain"] == domain and r["scene"] == scene
                        and r[field] is not None]
                if values:
                    sval.append(float(np.mean(values)))
            if sval:
                dvals.append(float(np.mean(sval)))
        result[field] = float(np.mean(dvals)) if dvals else None
    result["scenes"] = len({r["scene"] for r in rows})
    result["frames"] = len(rows)
    return result
