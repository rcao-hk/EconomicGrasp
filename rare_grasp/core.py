"""Pure CPU/GPU components for an executed-action RARE ranking experiment.

RARE §3.1: quality MSE on sigmoid(z) + logistic pair loss on raw z. Quality
is six-friction success count / 6, not 3D IoU. No ties, margin, direction guard,
GT geometry input, or changes to the frozen action generator.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

BASE_SHA = '7a956d37aaa427a32e598f21fdc6b910ed3c2704'
SCHEMA = 'rare-executed-grasp-v1'
MU = np.array([.2, .4, .6, .8, 1., 1.2], dtype=np.float64)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def frame_key(r: dict) -> str:
    return f"{r['domain']}_scene{int(r['scene']):04d}_frame{int(r['anno']):04d}"


def frame_seed(r: dict, seed: int) -> int:
    return int(digest([frame_key(r), int(seed)])[:8], 16)


def mix_frame_stride(train_fraction: float) -> int:
    """Allow exactly the preregistered 10% or 50% training-frame protocols."""
    fraction = float(train_fraction)
    if fraction == 0.1:
        return 10
    if fraction == 0.5:
        return 2
    raise ValueError(f'Only mix training fractions 0.1 and 0.5 are supported, got {train_fraction}')


def make_protocol(train_fraction: float = 0.1) -> dict:
    """Paired, per-scene RGB training views; unchanged 10% evaluation protocol.

    10% is 26 frames/scene (0,10,...,250); 50% is 128 (0,2,...,254).
    The 40 validation images are held-out *frames* from training scenes,
    NOT independent scenes, and do not overlap either training frame set.
    Without an argument, returns the original mix10 manifest byte-equivalent
    after JSON serialization to preserve its legacy contract fingerprint.
    """
    stride = mix_frame_stride(train_fraction)
    count_per_domain = 100 * len(range(0, 256, stride))

    def row(domain, scene, anno, partition):
        split = 'train' if scene < 100 else ('test_seen' if scene < 130 else
                                           'test_similar' if scene < 160 else 'test_novel')
        return dict(domain=domain, scene=scene, anno=anno, split=split, partition=partition)

    train = [row(d, s, a, 'train') for d in ('original', 'gntrans')
             for s in range(100) for a in range(0, 256, stride)]
    val = [row(d, s, a, 'validation') for d in ('original', 'gntrans')
           for s in range(10) for a in (5, 135)]
    test = [row(d, s, a, 'test') for d in ('original', 'gntrans')
            for s in range(100, 190) for a in range(0, 256, 10)]
    obj = dict(schema=SCHEMA, base_main=BASE_SHA, train=train, validation=val, test=test,
               training_count_by_domain={'original': count_per_domain,
                                         'gntrans': count_per_domain},
               validation_note='40 frame-held-out train-domain views; NOT scene-independent',
               primary_selection='fixed final epoch; no test AP checkpoint selection')
    if stride == 2:
        # New fields only in mix50: old mix10 JSON and SHA remain unchanged.
        obj['mix_training_fraction'] = 0.5
        obj['test_frame_fraction'] = 0.1
        obj['protocol_variant'] = 'rare-mix50-train-test10-v1'
    obj['protocol_sha256'] = digest(obj)
    return obj


def verify_protocol(p: dict) -> None:
    if not isinstance(p, dict):
        raise ValueError('Protocol must be a JSON object')
    fraction = p.get('mix_training_fraction', 0.1)
    want = make_protocol(fraction)
    if p != want:
        raise ValueError(f'Frame manifest differs from preregistered {fraction}+{fraction} protocol')


def success_bits(friction: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Round evaluator coefficient to 2 decimals, avoiding float32 .2 > .2."""
    f = np.round(np.asarray(friction, dtype=np.float64), 2)
    if not np.isfinite(f).all() or np.asarray(invalid).shape != f.shape:
        raise ValueError('Invalid evaluator labels')
    return ((f[..., None] > 0) & (f[..., None] <= MU) &
            (~np.asarray(invalid, dtype=bool)[..., None])).astype(np.uint8)


def physical_ids(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype='<f4')
    if g.ndim != 2 or g.shape[1] != 17 or not np.isfinite(g).all():
        raise ValueError('Expected finite N x 17 physical grasps')
    return np.array([hashlib.sha256(x[1:16].tobytes()).hexdigest() for x in g])


class ResidualRanker(nn.Module):
    """Identical scalar scorer for MSE and RARE, zero-init delta at epoch 0.

    This is a NEW scalar ranker, not the original six-threshold CDF head.
    Frozen CDF expected utility provides z0; f is the native score-branch
    feature. Depth one-hot is needed because Point shares f across depths.
    """
    def __init__(self, feature_dim: int = 64, hidden: int = 64, depths: int = 4):
        super().__init__()
        self.feature_dim, self.hidden, self.depths = feature_dim, hidden, depths
        self.norm = nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(nn.Linear(feature_dim + depths + 1, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, base_logit: torch.Tensor,
                depth_ids: torch.Tensor) -> torch.Tensor:
        if features.shape[:-1] != base_logit.shape or depth_ids.shape != base_logit.shape:
            raise ValueError('Ranker feature/logit/depth shape mismatch')
        if features.shape[-1] != self.feature_dim or (depth_ids < 0).any() or (depth_ids >= self.depths).any():
            raise ValueError('Ranker dimensions or depth IDs out of range')
        onehot = F.one_hot(depth_ids.long(), self.depths).to(features.dtype)
        x = torch.cat([self.norm(features), onehot, base_logit[..., None]], dim=-1)
        return base_logit + self.net(x).squeeze(-1)


def ranking_loss(z: torch.Tensor, count: torch.Tensor, family: torch.Tensor, *,
                 scope: str = 'image', max_pairs: int = 2048,
                 seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One image only. Uniform pair sample without replacement; ties excluded.

    The caller averages images, not all eligible pairs pooled across images.
    Thus an image/domain with more valid pairs cannot silently dominate.
    """
    if scope not in ('image', 'family') or max_pairs < 1:
        raise ValueError('Bad pairing scope/budget')
    if z.ndim != 1 or count.shape != z.shape or family.shape != z.shape:
        raise ValueError('Pass a single image with aligned one-dimensional fields')
    if count.is_floating_point() or ((count < 0) | (count > 6)).any():
        raise ValueError('Quality must be integer success count 0..6')
    if z.numel() == 0 or not torch.isfinite(z).all():
        raise ValueError('Empty or nonfinite scores')
    q = count.to(z.dtype) / 6
    point = F.mse_loss(torch.sigmoid(z), q)
    # Pair RNG is independent of model init and DataLoader RNG.
    y = count.detach().cpu(); fam = family.detach().cpu()
    ij = torch.triu_indices(len(y), len(y), offset=1)
    eligible = y[ij[0]] != y[ij[1]]
    if scope == 'family':
        eligible &= fam[ij[0]] == fam[ij[1]]
    ij = ij[:, eligible]; available = ij.shape[1]
    if available > max_pairs:
        gen = torch.Generator().manual_seed(seed)
        ij = ij[:, torch.randperm(available, generator=gen)[:max_pairs]]
    if ij.shape[1] == 0:
        pair = z.sum() * 0.0  # differentiable zero; do NOT invent tie preferences
    else:
        ij = ij.to(z.device)
        sign = torch.sign(count[ij[0]].to(z.dtype) - count[ij[1]].to(z.dtype))
        pair = F.softplus(-sign * (z[ij[0]] - z[ij[1]])).mean()
    return point, pair, {'eligible_pairs': available, 'used_pairs': ij.shape[1],
                         'ties_excluded': True, 'scope': scope}


def select_grasps(bank: np.ndarray, scores: np.ndarray, policy: str) -> np.ndarray:
    """bank[Q,D,17], scores[Q,D]; angle/view/width remain base-frozen.

    both: new depth and new query score; depth_only: new depth, old priority;
    rank_only: old depth, new max priority (controlled hybrid, not calibrated);
    native: original. Always returns the EXECUTED depth's original width.
    """
    bank = np.asarray(bank, dtype=np.float32); scores = np.asarray(scores)
    if bank.ndim != 3 or bank.shape[-1] != 17 or scores.shape != bank.shape[:2]:
        raise ValueError('Bad bank/scoring shape')
    if not np.isfinite(bank).all() or not np.isfinite(scores).all():
        raise ValueError('Nonfinite action bank or scores')
    if policy not in ('native', 'both', 'depth_only', 'rank_only'):
        raise ValueError('Unknown decode policy')
    base = bank[..., 0]; old_d = base.argmax(1); new_d = scores.argmax(1)
    d = new_d if policy in ('both', 'depth_only') else old_d
    result = bank[np.arange(len(bank)), d].copy()
    result[:, 0] = scores.max(1) if policy in ('both', 'rank_only') else base.max(1)
    return result


def candidate_metrics(scores: np.ndarray, count: np.ndarray) -> dict:
    """Fixed-action metrics, explicitly NOT official AP."""
    scores = np.asarray(scores); q = np.asarray(count, dtype=float)/6
    if scores.shape != q.shape or scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError('Expected family x four insertion depths')
    d = scores.argmax(1); chosen = q[np.arange(len(q)), d]; oracle = q.max(1)
    order = np.argsort(-scores, axis=1, kind='stable')
    oracle2 = np.take_along_axis(q, order[:, :2], axis=1).max(1)
    agreements = []
    for s, v in zip(scores, q):
        good = []
        for i in range(4):
            for j in range(i+1, 4):
                if v[i] == v[j]: continue
                ds = s[i]-s[j]
                good.append(.5 if abs(ds) <= 1e-8 else float(ds*(v[i]-v[j]) > 0))
        if good: agreements.append(float(np.mean(good)))
    return dict(selected_q=float(chosen.mean()), oracle_q=float(oracle.mean()),
                regret=float((oracle-chosen).mean()), oracle_at2=float(oracle2.mean()),
                comparable_families=len(agreements),
                depth_pair_agreement=float(np.mean(agreements)) if agreements else None)


def validate_cache(a: dict) -> None:
    fields = ('features', 'base_logit', 'count', 'family', 'depth', 'grasps', 'physical_id')
    if any(k not in a for k in fields): raise ValueError('Missing cache fields')
    n = len(a['base_logit'])
    if a['features'].shape != (n, 64) or a['grasps'].shape != (n, 17):
        raise ValueError('Invalid cache tensor shape')
    if any(np.asarray(a[k]).shape != (n,) for k in ('count','family','depth','physical_id')):
        raise ValueError('Invalid cache vector shape')
    if not np.isfinite(a['features']).all() or not np.isfinite(a['base_logit']).all():
        raise ValueError('Nonfinite cached features')
    if np.asarray(a['count']).dtype.kind not in 'iu' or ((a['count'] < 0) | (a['count'] > 6)).any():
        raise ValueError('Invalid quality count')
    if 'bits' in a:
        bits=np.asarray(a['bits'])
        if bits.shape!=(n,6) or not np.isin(bits,[0,1]).all() or not np.array_equal(bits.sum(-1),a['count']):
            raise ValueError('Six friction bits do not match quality count')
        if np.any(np.diff(bits.astype(int),axis=-1)<0):
            raise ValueError('Force-closure success bits must be monotone in mu')
    if not np.array_equal(physical_ids(a['grasps']), a['physical_id']):
        raise ValueError('Physical action identity mismatch')
    if len(np.unique(a['family'])) * 4 != n:
        raise ValueError('Incomplete four-depth families')
    for f in np.unique(a['family']):
        ix = np.flatnonzero(a['family'] == f)
        if not np.array_equal(a['depth'][ix], np.arange(4)):
            raise ValueError('Depth bins reordered or incomplete')
        if not np.allclose(a['grasps'][ix, 3], (.01,.02,.03,.04), atol=1e-7):
            raise ValueError('Metric insertion mismatch')
        if not np.allclose(a['grasps'][ix,4:16], a['grasps'][ix[0],4:16], atol=1e-6):
            raise ValueError('Family center/rotation must stay fixed')
