"""Strict old-cache import, shared-pool preparation, and immutable factorial plan.

This round reuses ONLY the paid-for P1 Uniform ∪ Decision labels. It does not
claim to discover hard negatives outside that reservoir. A native relation
trace is required for training; a subset-only NMS rerun is not a substitute.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import BASE_SHA, SCHEMA, digest, sha, validate, stratify, write_json

FIELDS = ('features', 'base_logit', 'grasps', 'family', 'depth', 'count', 'bits', 'physical_id')


def signed(path, signature):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    value = obj.pop(signature, None)
    if value != digest(obj): raise ValueError(f'Invalid {signature}: {path}')
    obj[signature] = value
    return obj


def source_index(path, formal=True):
    obj = signed(path, 'index_sha256')
    if obj.get('schema') != 'quality-decision-p0p2-v1' or obj.get('kind') != 'decision-cache-index':
        raise ValueError('Expected audited previous P1 index; adapt schema explicitly, never guess')
    if obj.get('base_checkpoint_sha256') != BASE_SHA or not obj.get('config_digest'):
        raise ValueError('Wrong frozen GVAR identity')
    frames = {}
    for entry in obj['entries']:
        r = entry['row']; arm = entry['strategy']
        if (r['domain'] not in ('original', 'gntrans') or r['split'] != 'train' or
            not 0 <= int(r['scene']) < 100 or not 0 <= int(r['anno']) < 256 or
            arm not in ('uniform', 'decision')):
            raise ValueError('Unsupported frame/domain/strategy; test scenes forbidden')
        key = (r['domain'], int(r['scene']), int(r['anno']))
        if arm in frames.setdefault(key, {}): raise ValueError('Duplicate source frame/arm')
        frames[key][arm] = entry
    if any(set(v) != {'uniform', 'decision'} for v in frames.values()):
        raise ValueError('Missing paired Uniform/Decision frame')
    if formal:
        if len(frames) != 800: raise ValueError('Formal round requires all 800 paired frames')
        for s in range(100):
            for domain in ('original', 'gntrans'):
                if sum(k[:2] == (domain, s) for k in frames) != 4:
                    raise ValueError(f'Missing four accepted views: {domain}/{s}')
            a = {k[2] for k in frames if k[:2] == ('original', s)}
            b = {k[2] for k in frames if k[:2] == ('gntrans', s)}
            if a != b: raise ValueError('Original/GN-Trans view mismatch')
    return obj, sorted(frames.items())


def read_cache(entry):
    path = Path(entry['file'])
    if not path.is_file() or sha(path) != entry['sha256']:
        raise ValueError(f'Missing/changed source cache: {path}')
    with np.load(path, allow_pickle=False) as z:
        x = {k: z[k] for k in FIELDS}
    validate(x)
    if len(np.unique(x['family'])) != 64: raise ValueError('Expected 64 source families')
    return x


def union_frame(entries):
    u, d = read_cache(entries['uniform']), read_cache(entries['decision'])
    combined = {}
    for x in (u, d):
        for f in np.unique(x['family']):
            ix = np.flatnonzero(x['family'] == f)
            values = {k: x[k][ix] for k in FIELDS}
            if int(f) in combined:
                old = combined[int(f)]
                # All shared labels AND source features must agree. No relabel,
                # relocalization or tolerance broadening to merge two runtimes.
                for k in FIELDS:
                    if not np.array_equal(old[k], values[k]):
                        raise ValueError(f'Inconsistent shared family {f}/{k}')
            else:
                combined[int(f)] = values
    keys = sorted(combined)
    out = {k: np.concatenate([combined[f][k] for f in keys]) for k in FIELDS}
    out['uniform_families'] = np.unique(u['family']).astype(np.int64)
    out['decision_families'] = np.unique(d['family']).astype(np.int64)
    out['edges'] = np.empty((0, 2), np.int64)
    if len(np.intersect1d(out['uniform_families'], out['decision_families'])) < 32:
        raise ValueError('Missing 32 shared anchors')
    validate(out)
    return out


def frame_name(row):
    return f"{row['domain']}_scene{int(row['scene']):04d}_frame{int(row['anno']):04d}"


def read_pack(entry):
    if sha(entry['file']) != entry['sha256']: raise ValueError('Changed traced frame')
    with np.load(entry['file'], allow_pickle=False) as z:
        x = {k: z[k] for k in z.files}
    validate(x)
    return x


def prepare(source, traces, output, acceptance, formal=True):
    """Assemble traces and freeze A/B/C/D data assignments BEFORE fitting."""
    src, frames = source_index(source, formal=formal)
    if not Path(acceptance).is_file(): raise FileNotFoundError('Previous accepted label run required')
    acceptance_sha = sha(acceptance)
    entries = {}; trace_contract = None
    for path in traces:
        t = signed(path, 'manifest_sha256')
        c = t['contract']
        if c['source_index_sha256'] != src['index_sha256'] or c['acceptance_sha256'] != acceptance_sha:
            raise ValueError('Trace source labels/acceptance provenance differs')
        if trace_contract is None: trace_contract = c
        if trace_contract != c: raise ValueError('Mixed tracing runtime/seed/contracts')
        for e in t['entries']:
            name = frame_name(e['row'])
            if name in entries: raise ValueError('Duplicate trace shard frame')
            entries[name] = e
    wanted = {frame_name(es['uniform']['row']) for _, es in frames}
    if set(entries) != wanted: raise ValueError('Trace coverage not equal to source 800 frames')
    planned = []
    for _, es in frames:
        r = es['uniform']['row']; e = entries[frame_name(r)]
        # Revalidate the immutable source hashes against trace metadata.
        if e['source_sha256'] != {k: es[k]['sha256'] for k in ('uniform', 'decision')}:
            raise ValueError('Frame source changed since trace')
        if e['row'] != r or e['contract_sha256'] != digest(trace_contract):
            raise ValueError('Trace row/contract does not match planned source')
        x = read_pack(e)
        item = dict(e)
        if r['scene'] < 80:
            # Quality-stratified mining is restricted to PROBE TRAIN scenes.
            chosen, audit = stratify(x, 20261114 + r['scene'] * 1009 + r['anno'])
            item.update(stratified_families=chosen.tolist(), sampling_audit=audit)
        else:
            item.update(stratified_families=None, sampling_audit=None)
        planned.append(item)
    obj = dict(schema=SCHEMA, kind='factorial-plan', formal=bool(formal),
               base_checkpoint_sha256=BASE_SHA, base_config_digest=src['config_digest'],
               source_index_sha256=src['index_sha256'], source_index_path=str(Path(source).resolve()),
               accepted_label_report_sha256=acceptance_sha, trace_contract=trace_contract,
               trace_shards={str(p): sha(p) for p in traces}, entries=planned,
               primary_epochs=10, primary_policy='rank_executed',
               seeds=[20261113, 20261114, 20261115],
               train_scenes=[0, 80], validation_scenes=[80, 90], audit_scenes=[90, 100],
               audit_scenes_previously_inspected=True,
               budgets={'families': 64, 'actions': 256, 'random_anchors': 32,
                        'pair_cap': 2048, 'target_cap': 256, 'target_loss_mass': .5},
               no_new_labels=True, label_pool_scope='previous Uniform union Decision ONLY')
    obj['plan_sha256'] = digest(obj)
    if Path(output).exists(): raise FileExistsError('Never overwrite an experiment plan')
    write_json(output, obj)
    return obj


def load_plan(path, require_formal=True):
    p = signed(path, 'plan_sha256')
    if p.get('schema') != SCHEMA or p.get('kind') != 'factorial-plan': raise ValueError('Wrong plan')
    if require_formal and not p['formal']: raise ValueError('Smoke plan cannot train/evaluate formally')
    return p


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source-index', required=True)
    a.add_argument('--trace-manifest', action='append', required=True)
    a.add_argument('--accepted-label-report', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--smoke', action='store_true')
    x = a.parse_args()
    p = prepare(x.source_index, x.trace_manifest, x.output, x.accepted_label_report, not x.smoke)
    print(json.dumps({'plan_sha256': p['plan_sha256'], 'frames': len(p['entries']), 'formal': p['formal']}))

if __name__ == '__main__': main()
