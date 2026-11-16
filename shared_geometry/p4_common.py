"""P4 shared contracts. Physical banks and the P2 reader are never trained here."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from .data import sha256, digest, signed, load_signed, atomic_json, atomic_npz, frame_key, make_plan
from .p0_relative_decision import scene_balanced, scene_bootstrap, metrics_for_family
from .p1_clean_visible import read_cache_index, load_cached, collate_cache
from .p2_geometry_diagnostics import load_shared
from .p3_action_geometry_diagnostics import pair_geometry_evidence, _sanity_check_input

SCHEMA = 'shared-visible-geometry-p4-v1'


def select_entries(entries, limit=0):
    """Smoke-only round-robin across RGB domains; all formal rows keep order."""
    if not limit: return entries
    if limit < 0: raise ValueError('Negative frame limit')
    domains=list(dict.fromkeys(e['row']['domain'] for e in entries))
    queues={d:[e for e in entries if e['row']['domain']==d] for d in domains}
    selected=[];position=0
    while len(selected)<min(limit,len(entries)):
        for d in domains:
            if position<len(queues[d]): selected.append(queues[d][position])
            if len(selected)==min(limit,len(entries)): break
        position+=1
    return selected


def source_hashes():
    root = Path(__file__).parent
    names = ('p4_common.py', 'p4_score_diagnostics.py', 'p4_depth_baseline.py', 'p4_depth_data.py',
             'geometry.py', 'model.py', 'p1_clean_visible.py', 'p2_geometry_diagnostics.py',
             'p3_action_geometry_diagnostics.py', 'runtime.py')
    return {name: sha256(root/name) for name in names if (root/name).is_file()}


def formal_rows(index, partition, smoke=False):
    """Validate actual scene/frame sets, not just the number of entries."""
    if partition not in ('train', 'validation') or index['partition'] != partition:
        raise ValueError('P4 uses training-domain train/validation only')
    rows = [e['row'] for e in index['entries']]
    keys = [frame_key(r) for r in rows]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError('Empty/duplicated P4 frames')
    if any(r['partition'] != partition or r['scene'] >= 100 or r['scene'] < 0 or
           r['domain'] not in ('original', 'gntrans') for r in rows):
        raise ValueError('Wrong frame domain/partition or official test access')
    if not smoke and set(keys) != {frame_key(r) for r in make_plan()[partition]}:
        raise ValueError('Formal P4 needs the original 2560/160-frame protocol')


def freeze_reader(weight, p1_index, device='cpu', smoke=False):
    model, protocol = load_shared(weight, device, smoke)
    if (protocol['val_fingerprint'] != p1_index['fingerprint'] or
        protocol['teacher_config'] != p1_index['contract']['target_config'] or
        protocol['source'] != p1_index['contract']['source']):
        raise ValueError('P2 reader and P1 validation provenance differ')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, protocol


def score_physics(model, physical, local, dimensions):
    """Differentiable readout of [B,F,7,75,6]; labels cannot enter this API."""
    if (physical.ndim != 5 or physical.shape[-2:] != (75, 6) or
        local.shape != (*physical.shape[:-1], 3) or
        dimensions.shape != (*physical.shape[:3], 3)):
        raise ValueError('Misaligned physical/local/dimension tensors')
    token = model.point_reader(torch.cat((physical, local/.05), -1))
    desc = torch.cat((token.mean(-2), token.amax(-2), dimensions/.1), -1)
    return model.quality(desc)[..., 0]


def tensor_inputs(batch, device):
    return tuple(torch.as_tensor(batch[k], dtype=torch.float32, device=device)
                 for k in ('local_xyz', 'bank_dims'))


def family_rows(logits, frame, row, offsets, state_source=None):
    result = []
    for f, z in enumerate(np.asarray(logits)):
        state = None if state_source is None else state_source[f, ..., 2:].argmax(-1)
        r = metrics_for_family(z, frame['quality_count'][f], offsets, state=state)
        r.update(scene=int(row['scene']), domain=row['domain'], anno=int(row['anno']),
                 family=int(frame['family_id'][f]),
                 native_count=int(frame['quality_count'][f, 0]),
                 physical_ids=frame['physical_id'][f].astype(str).tolist())
        r['no_rescue'] = bool(r['opportunity'] and not r['rescue'])
        result.append(r)
    return result


def paired_summary(records, baseline=None, n_boot=5000, seed=0):
    keys = ('selected_q', 'native_q', 'oracle_q', 'gain', 'move', 'rescue', 'harm',
            'equal', 'nativepair_correct', 'allpair_correct', 'native_signed_bias',
            'alt_signed_bias', 'native_squared_error', 'alt_squared_error')
    result = {k: scene_balanced(records, k) for k in keys}
    result['gain_vs_native_ci'] = scene_bootstrap(records, metric='gain', n_boot=n_boot, seed=seed)
    if baseline is not None:
        identities = lambda xs: [(r['scene'], r['domain'], r['anno'], r['family']) for r in xs]
        if identities(records) != identities(baseline):
            raise ValueError('Unpaired family identities')
        result['gain_vs_baseline_ci'] = scene_bootstrap(records, baseline, metric='gain', n_boot=n_boot, seed=seed)
    return result


def verify_p2(expected_path, index, weight, records, partial=False):
    if not expected_path:
        return False
    expected = load_signed(expected_path)
    if (expected['validation_fingerprint'] != index['fingerprint'] or
        expected['checkpoint_sha256'] != sha256(weight)):
        raise ValueError('Expected P2 result has different weights/cache')
    if not partial:
        for arm in ('predicted', 'clean_visible'):
            if abs(scene_balanced(records[arm], 'gain') - expected['summaries'][arm]['gain']) > 2e-6:
                raise RuntimeError('P2 source gain parity failed: '+arm)
    return not partial


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Use a new output directory: '+str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_records(path, records):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    return dict(path=Path(path).name, sha256=sha256(path), rows=len(records))


def finite_correlation(x, y):
    """Descriptive association only; no IID action-level significance test."""
    from scipy.stats import spearmanr
    a, b = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(n=len(a), pearson=None, spearman=None)
    return dict(n=len(a), pearson=float(np.corrcoef(a, b)[0, 1]),
                spearman=float(spearmanr(a, b).statistic))


def atomic_checkpoint(path, payload):
    path = Path(path); tmp = path.with_name(path.name+'.tmp')
    torch.save(payload, tmp); tmp.replace(path)
