"""Fixed-action next-round controls. No CAD inference and no production edits.

QualityProbe state keys match the audited P0/P1 5,633-parameter model. Pair
supervision uses real executed-action labels, never a preferred depth direction.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'decision-supervision-v1'
BASE_SHA = '2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675'
MODES = ('score', 'score_geometry', 'score_feature', 'score_feature_geometry', 'score_feature_permuted')
ARMS = {'A': ('decision', 'random'), 'B': ('stratified', 'random'),
        'C': ('decision', 'targeted'), 'D': ('stratified', 'targeted')}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def physical_ids(g):
    """Legacy ID encoding only; original float64 geometry is never recast for eval."""
    return np.array([hashlib.sha256(v[1:16].tobytes()).hexdigest()
                     for v in np.asarray(g, dtype='<f4')])


def validate(x):
    required = ('features', 'base_logit', 'grasps', 'family', 'depth', 'count', 'bits', 'physical_id')
    if any(k not in x for k in required):
        raise ValueError('Missing exact-action cache field')
    n = len(x['base_logit'])
    if not n or x['features'].shape != (n, 64) or x['grasps'].shape != (n, 17):
        raise ValueError('Expected nonempty 64D features and 17D grasps')
    for k in ('family', 'depth', 'count', 'physical_id'):
        if np.asarray(x[k]).shape != (n,):
            raise ValueError(f'Invalid vector shape: {k}')
    for k in ('features', 'base_logit', 'grasps'):
        if not np.isfinite(x[k]).all():
            raise ValueError(f'Nonfinite {k}')
    for k in ('family', 'depth', 'count'):
        if np.asarray(x[k]).dtype.kind not in 'iu':
            raise ValueError(f'{k} must be integers')
    bits = np.asarray(x['bits'])
    if (bits.shape != (n, 6) or not np.isin(bits, [0, 1]).all() or
        np.any(np.diff(bits.astype(int), axis=1) < 0) or
        not np.array_equal(bits.sum(1), x['count'])):
        raise ValueError('Six-friction quality bits/count mismatch')
    if np.any((x['grasps'][:, 1] < 0) | (x['grasps'][:, 1] > .1 + 1e-6)):
        raise ValueError('Invalid executed width')
    if not np.array_equal(physical_ids(x['grasps']), np.asarray(x['physical_id']).astype(str)):
        raise ValueError('Physical identity mismatch')
    for f in np.unique(x['family']):
        ix = np.flatnonzero(x['family'] == f)
        if len(ix) != 4 or not np.array_equal(x['depth'][ix], np.arange(4)):
            raise ValueError('Four-depth family incomplete or reordered')
        if not np.allclose(x['grasps'][ix, 3], [.01, .02, .03, .04], atol=1e-7, rtol=0):
            raise ValueError('Insertion depth differs from depth ID')
        if not np.allclose(x['grasps'][ix, 4:16], x['grasps'][ix[0], 4:16], atol=1e-6, rtol=0):
            raise ValueError('Family center/rotation changed')
    if 'edges' in x:
        e = np.asarray(x['edges'])
        if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in 'iu':
            raise ValueError('Expected [M,2] integer conflict edges')
        if len(e) and (e.min() < 0 or e.max() >= n or np.any(e[:, 0] >= e[:, 1])):
            raise ValueError('Conflict endpoints invalid/not canonical')
        if len(e) and not np.isin(e, native_rows(x)).all():
            raise ValueError('Target edges must connect actual native executed depths')
        if len(np.unique(e, axis=0)) != len(e):
            raise ValueError('Duplicate conflict edges')


class QualityProbe(nn.Module):
    """Numerically/state-dict compatible with quality_decision.core.QualityProbe."""
    def __init__(self, mode='score_feature_geometry', hidden=64):
        super().__init__()
        if mode not in MODES or hidden < 1:
            raise ValueError('Unknown probe mode/hidden size')
        self.mode, self.hidden = mode, hidden
        self.feature_norm = nn.LayerNorm(64)
        self.net = nn.Sequential(nn.Linear(84, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat, base_logit, grasps, depth, permutation=None):
        if feat.shape != (len(base_logit), 64) or grasps.shape != (len(base_logit), 17):
            raise ValueError('Probe input shape mismatch')
        features = self.feature_norm(feat)
        geom = grasps[:, 1:16].clone()
        geom[:, 0] /= .1; geom[:, 1] /= .02; geom[:, 2] /= .04
        if self.mode == 'score_feature_permuted':
            if permutation is None: raise ValueError('Permutation required')
            features = features[permutation]
        elif self.mode not in ('score_feature', 'score_feature_geometry'):
            features = torch.zeros_like(features)
        if self.mode not in ('score_geometry', 'score_feature_geometry'):
            geom = torch.zeros_like(geom)
        x = torch.cat([features, geom, F.one_hot(depth.long(), 4).to(feat.dtype), base_logit[:, None]], -1)
        return base_logit + self.net(x).squeeze(-1)


def predict(model, x, device, seed=0):
    # Prediction reads ONLY evidence/actions, never count/bits/edges/assigned objects.
    tensors = [torch.as_tensor(x[k], device=device, dtype=torch.float32)
               for k in ('features', 'base_logit', 'grasps')]
    d = torch.as_tensor(x['depth'], device=device, dtype=torch.long)
    perm = None
    if model.mode == 'score_feature_permuted':
        perm = torch.as_tensor(np.random.default_rng(seed).permutation(len(d)), device=device)
    return model(*tensors, d, permutation=perm)


def native_rows(x):
    return np.array([ix[np.argmax(x['grasps'][ix, 0])]
                     for f in np.unique(x['family'])
                     for ix in [np.flatnonzero(x['family'] == f)]], dtype=np.int64)


def subset(x, families):
    families = np.asarray(families, dtype=np.int64)
    if len(families) != len(np.unique(families)) or not np.isin(families, x['family']).all():
        raise ValueError('Unknown/duplicate family')
    ix = np.concatenate([np.flatnonzero(x['family'] == f) for f in np.sort(families)])
    keys = ('features', 'base_logit', 'grasps', 'family', 'depth', 'count', 'bits', 'physical_id')
    y = {k: x[k][ix] for k in keys}
    remap = np.full(len(x['family']), -1, dtype=np.int64); remap[ix] = np.arange(len(ix))
    e = np.asarray(x.get('edges', np.empty((0, 2), np.int64)))
    e = remap[e]
    y['edges'] = e[np.all(e >= 0, axis=1)]
    validate(y)
    return y


def stratify(x, seed, hard_quantile=.5):
    """Training-only family resampling from SAME paid-for U∪D label pool.

    Raw arm is old Decision. Retain 32 deterministic U∩D anchors; change only
    other 32. Seek 16 high-native-score q=0, 8 count1..3, 8 count4..6 families.
    Fallback is random without replacement and explicitly counted, never fake.
    """
    u, d = np.asarray(x['uniform_families']), np.asarray(x['decision_families'])
    if len(u) != 64 or len(d) != 64: raise ValueError('Expected 64-family source arms')
    shared = np.intersect1d(u, d)
    if len(shared) < 32: raise ValueError('Source arms share fewer than 32 anchors')
    rng = np.random.default_rng(seed)
    anchors = rng.choice(shared, 32, replace=False)
    nr = native_rows(x); fam = x['family'][nr]; score = x['grasps'][nr, 0]; q = x['count'][nr]
    valid = np.asarray(x['native_sensor_valid'], dtype=bool)
    if valid.shape != (len(fam),): raise ValueError('Native collision-pass mask missing')
    cutoff = float(np.quantile(score[valid], hard_quantile)) if valid.any() else None
    unused = ~np.isin(fam, anchors)
    masks = [valid & unused & (q == 0) & (score >= cutoff if cutoff is not None else False),
             valid & unused & (q >= 1) & (q <= 3), valid & unused & (q >= 4)]
    picked = list(map(int, anchors)); counts = []
    for mask, quota in zip(masks, (16, 8, 8)):
        options = np.flatnonzero(mask)
        # High-score zero-success first; other strata random at fixed seed.
        order = options[np.lexsort((fam[options], -score[options]))] if quota == 16 else rng.permutation(options)
        use = list(map(int, fam[order[:quota]])); picked.extend(use); counts.append(len(use))
    missing = 64 - len(picked)
    remaining = fam[~np.isin(fam, picked)]
    if len(remaining) < missing: raise ValueError('Insufficient family reservoir')
    picked.extend(map(int, rng.choice(remaining, missing, replace=False)))
    selected = np.sort(np.asarray(picked, np.int64))
    if len(np.unique(selected)) != 64: raise RuntimeError('Sampler duplicated families')
    return selected, dict(shared_anchors=32, hard_zero_selected=counts[0],
                         middle_selected=counts[1], high_selected=counts[2],
                         fallback_families=missing, hard_score_cutoff=cutoff,
                         distinct_changes_from_decision=int(np.sum(~np.isin(selected, d))))


def pair_batch(count, edges, mode, seed, budget=2048, target_cap=256):
    """No replacement, no quality ties; actual exposure/weights are auditable.

    Targeted loss allocates half its pair mass to unique covered conflict edges
    and half to random other pairs. Sparse edges are not duplicated to fill a
    quota. If none are eligible, use exactly the random control for this frame.
    """
    count = np.asarray(count)
    if mode not in ('random', 'targeted') or budget < 2 or target_cap < 1:
        raise ValueError('Invalid pair policy')
    ij = np.stack(np.triu_indices(len(count), 1), axis=1)
    ij = ij[count[ij[:, 0]] != count[ij[:, 1]]]
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[count[e[:, 0]] != count[e[:, 1]]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    code = lambda a: a[:, 0] * len(count) + a[:, 1]
    ec = code(e); ic = code(ij)
    rng = np.random.default_rng(seed)
    n = min(budget, len(ij)); targeted = 0
    if mode == 'targeted' and len(e) and n > 1:
        targeted = min(len(e), target_cap, n - 1)
        front = e[rng.choice(len(e), targeted, replace=False)]
        rest = ij[~np.isin(ic, code(front))]
        back = rest[rng.choice(len(rest), n - targeted, replace=False)]
        pairs = np.concatenate([front, back])
        weights = np.r_[np.full(targeted, .5 / targeted), np.full(n - targeted, .5 / (n - targeted))]
    else:
        pairs = ij[rng.choice(len(ij), n, replace=False)] if n else np.empty((0, 2), np.int64)
        weights = np.full(n, 1 / n) if n else np.empty(0)
    audit = dict(eligible_pairs=len(ij), covered_non_tie_conflict_edges=len(e),
                 sampled_pairs=n, targeted_group_pairs=targeted,
                 actual_sampled_conflict_edges=int(np.isin(code(pairs), ec).sum()),
                 max_pair_weight=float(weights.max()) if n else 0.,
                 conflict_loss_mass=float(weights[np.isin(code(pairs), ec)].sum()) if n else 0.,
                 fallback_to_random=bool(mode == 'targeted' and targeted == 0),
                 pairs_sha256=hashlib.sha256(pairs.tobytes()).hexdigest())
    return pairs, weights, audit


def loss(z, x, mode, seed):
    pairs, weights, audit = pair_batch(x['count'], x['edges'], mode, seed)
    q = torch.as_tensor(x['count'], device=z.device, dtype=z.dtype) / 6
    mse = F.mse_loss(z.sigmoid(), q)
    if len(pairs):
        ij = torch.as_tensor(pairs, device=z.device)
        signs = torch.sign(q[ij[:, 0]] - q[ij[:, 1]])
        per = F.softplus(-signs * (z[ij[:, 0]] - z[ij[:, 1]]))
        relative = (per * torch.as_tensor(weights, device=z.device, dtype=z.dtype)).sum()
    else:
        relative = z.sum() * 0
    return 10 * mse + .5 * relative, {**audit, 'mse': float(mse.detach()), 'pair_loss': float(relative.detach())}


def metrics(scores, x):
    """Sampled pre-sensor-filter/PRE-NMS metrics. NOT official Top-K or AP."""
    q = x['count'] / 6; nr = native_rows(x)
    chosen = np.array([ix[np.argmax(scores[ix])] for f in np.unique(x['family'])
                       for ix in [np.flatnonzero(x['family'] == f)]])
    result = dict(mse=float(np.mean((scores - q) ** 2)), selected_q=float(q[chosen].mean()),
                  oracle_q=float(np.mean([q[x['family'] == f].max() for f in np.unique(x['family'])])),
                  mean_q=float(q.mean()))
    for name, rows in (('fixed_depth', nr), ('selected_depth', chosen)):
        order = rows[np.argsort(-scores[rows], kind='stable')]
        for k in (10, 50):
            top = order[:k]
            result[f'{name}_top{k}_zero'] = float((q[top] == 0).mean())
            result[f'{name}_top{k}_q'] = float(q[top].mean())
    e = x.get('edges', np.empty((0, 2), np.int64))
    e = e[x['count'][e[:, 0]] != x['count'][e[:, 1]]]
    ds = scores[e[:, 0]] - scores[e[:, 1]]
    correct = np.where(np.abs(ds) < 1e-9, .5, (ds * (q[e[:, 0]] - q[e[:, 1]]) > 0).astype(float))
    result['conflict_pair_agreement'] = float(correct.mean()) if len(e) else None
    result['comparable_conflict_edges'] = int(len(e))
    return result


def decode(bank, scores, policy):
    if bank.shape != (*scores.shape, 17) or scores.shape[1] != 4:
        raise ValueError('Invalid full candidate bank')
    if not np.isfinite(bank).all() or not np.isfinite(scores).all(): raise ValueError('Nonfinite inference')
    base = bank[..., 0]; old = base.argmax(1); new = scores.argmax(1)
    if policy not in ('native', 'rank_executed', 'rank_max', 'both', 'depth_only'): raise ValueError(policy)
    d = new if policy in ('both', 'depth_only') else old
    result = bank[np.arange(len(bank)), d].copy()
    result[:, 0] = (scores[np.arange(len(bank)), old] if policy == 'rank_executed' else
                    scores.max(1) if policy in ('both', 'rank_max') else base.max(1))
    return result
