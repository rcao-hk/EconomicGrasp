"""Shared accounting for attribution and full-path tests. No CAD imports."""
from __future__ import annotations

import csv
import gc
import io
import json
import ctypes
from pathlib import Path

import numpy as np
import torch
from rep_a_common import (aggregate, atomic_file, cdf_targets, digest, file_sha,
                          load_torch, metrics, save_json, tensors)


def source_digest(names):
    root = Path(__file__).resolve().parent
    return digest({name: file_sha(root/name) for name in names})


def checked_manifest(folder, protocol, name='protocol.json'):
    path = Path(folder)/name
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError(f'Protocol changed at {path}; use a new output directory')
    save_json(path, protocol)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=keys, restval='')
    writer.writeheader(); writer.writerows(rows)
    with atomic_file(path) as f:
        f.write(buf.getvalue().encode())


def release_memory():
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


def memory_status(min_free_gib=0):
    try:
        fields = {s.split(':')[0]: int(s.split()[1])
                  for s in Path('/proc/meminfo').read_text().splitlines()}
        available = fields['MemAvailable']/1024**2
        rss = next(int(s.split()[1])/1024**2 for s in Path('/proc/self/status').read_text().splitlines()
                   if s.startswith('VmRSS:'))
        if available < min_free_gib:
            raise MemoryError(f'MemAvailable={available:.2f} GiB below {min_free_gib}; resume later')
        return dict(rss_gib=rss, available_gib=available)
    except (FileNotFoundError, KeyError):
        return {}


def model_inputs(d, device):
    # Also supports newly generated candidates; never calls fixed-cache label checks.
    out = tensors(d, device)
    out['valid'] = torch.as_tensor(np.array(d['valid'], dtype=bool, copy=True), device=device)
    out['offsets_mm'] = torch.as_tensor(np.array(d['offsets_mm'], dtype=np.float32, copy=True), device=device)
    out['zero_index'] = int(d['zero_index'])
    return out


def load_scorer(path, cache_root, device):
    """Load old Rep-A/B and new retrained controls without changing their weights."""
    ck = load_torch(path)
    init = load_torch(Path(cache_root)/'reader_init.pt')
    if ck['contract'] != init['contract']:
        raise RuntimeError(f'Stage-1/cache mismatch: {path}')
    if ck.get('experiment') == 'Rep-B0-controls':
        from rep_b0_controls import RepB0Control
        model = RepB0Control(**ck['model_spec'])
    elif ck.get('experiment') == 'Rep-B':
        from rep_b_model import RepBModel
        model = RepBModel(**ck['model_spec'])
    else:
        from rep_a_model import RepAReaderScorer
        if ck.get('variant') not in ('A0','A1','A2','A3'):
            raise ValueError(f'Unknown checkpoint family: {path}')
        if ck['model_config'] != init['model_config']:
            raise RuntimeError('Rep-A reader configuration changed')
        model = RepAReaderScorer(init, ck['variant'])
    model.load_state_dict(ck['model'], strict=True)
    meta = dict(margin=float(ck['margin']), epoch=int(ck['epoch']),
                contract=ck['contract'], checkpoint_sha=file_sha(path),
                variant=ck.get('variant', ck.get('control')))
    del ck, init
    release_memory()
    return model.to(device).eval(), meta


def decision_metrics(prob, d, margin):
    """Tie-aware, selection-relevant metrics in addition to original all-pair accuracy."""
    result, selected = metrics(prob, d, margin)
    valid = np.asarray(d['valid'], bool)
    u = np.asarray(d['utility'], np.float64)
    score = np.asarray(prob).mean(-1)
    z = int(d['zero_index']); q = np.arange(u.shape[1])
    masked_score = np.where(valid, score, -np.inf)
    pred = masked_score.argmax(0)
    # Native wins score ties, matching the deployed strict-margin policy.
    pred = np.where(score[z] >= masked_score.max(0), z, pred)
    oracle_u = np.where(valid, u, -np.inf).max(0)
    alt_valid = valid.copy(); alt_valid[z] = False
    has_alt = alt_valid.any(0)
    best_alt_u = np.where(alt_valid, u, -np.inf).max(0)
    gap = best_alt_u-u[z]
    informative = has_alt & (np.abs(gap) > 1e-7)
    opportunity = has_alt & (gap > 1e-7)
    # For ties among exact best alternatives, use the model's highest-scored
    # tied candidate. Credit any utility-optimal action, not one arbitrary index.
    tied = alt_valid & np.isclose(u, best_alt_u[None], atol=1e-7, rtol=0)
    best_alt_score = np.where(tied, score, -np.inf).max(0)
    nv = best_alt_score-score[z]
    correct = np.where(np.abs(nv) <= 1e-12, .5, (nv*gap > 0).astype(float))
    result.update(
        top1_optimal_count=int(np.isclose(u[pred,q], oracle_u, atol=1e-7, rtol=0).sum()),
        forced_top1_regret_sum=float((oracle_u-u[pred,q]).sum()),
        selected_regret_sum=float((oracle_u-u[selected,q]).sum()),
        native_best_correct_sum=float(correct[informative].sum()),
        native_best_count=int(informative.sum()),
        opportunity_count=int(opportunity.sum()),
        opportunity_rescued_count=int((opportunity & (u[selected,q] > u[z]+1e-7)).sum()))
    # Near-native pairs are a harder metric than far-apart candidates.
    offsets = np.asarray(d.get('offsets_mm', np.zeros(len(u))))
    near = valid & (np.abs(offsets[:,None]) <= 20+1e-6)
    near[z] = False
    du, ds = u-u[z], score-score[z]
    pairs = near & (np.abs(du)>1e-7)
    near_correct = np.where(np.abs(ds)<=1e-12, .5, (du*ds>0).astype(float))
    result['near_native_correct_sum'] = float(near_correct[pairs].sum())
    result['near_native_pair_count'] = int(pairs.sum())
    return result, selected


def aggregate_decisions(rows):
    out = aggregate(rows)
    nq = out['num_queries']
    for key, num, den in (
        ('top1_optimal_recall','top1_optimal_count',None),
        ('forced_top1_regret','forced_top1_regret_sum',None),
        ('selected_regret','selected_regret_sum',None),
        ('native_vs_best_alt_accuracy','native_best_correct_sum','native_best_count'),
        ('opportunity_rescue_recall','opportunity_rescued_count','opportunity_count'),
        ('near_native_pair_accuracy','near_native_correct_sum','near_native_pair_count')):
        count = nq if den is None else sum(r[den] for r in rows)
        out[key] = sum(r[num] for r in rows)/count if count else None
    return out
