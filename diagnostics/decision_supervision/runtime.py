"""Native relation replay and fixed-geometry inference; optional 117 runtime adapter.

Use the accepted historical runtime via --factory module:callable. No relaxed
load, no relabeling, no copied features across checkpoints. Compiled NMS is the
source of truth, including ties. This module does not reimplement its sorting.
"""
from __future__ import annotations
import argparse
import importlib
import inspect
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .core import (BASE_SHA, SCHEMA, digest, sha, write_json, write_npz,
                   native_rows, physical_ids, decode, predict)
from .data import source_index, union_frame, frame_name


def make_runtime(args):
    module, name = args.factory.split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    rt = factory(args.config, args.dataset_root, args.gntrans_root, args.device)
    if rt.config.get('checkpoint_sha256') != BASE_SHA or rt.config.get('backbone') != 'point_gvar':
        raise ValueError('Wrong frozen MIX50 GVAR identity')
    if float(rt.config.get('mix_training_fraction', .1)) != .5:
        raise ValueError('Not MIX50')
    src = inspect.getsourcefile(factory)
    return rt, {'factory': args.factory, 'factory_source_sha256': sha(src),
                'bound_config_sha256': sha(args.config)}


def filtered_indices(full, filtered):
    """Occurrence-aware mapping; duplicate geometries do not lose source IDs."""
    loc = defaultdict(deque)
    for i, row in enumerate(full):
        loc[row[:16].tobytes()].append(i)
    indices = []
    for row in filtered:
        key = row[:16].tobytes()
        if not loc[key]: raise ValueError('Sensor filter changed/reordered physical row values')
        indices.append(loc[key].popleft())
    return np.asarray(indices, np.int64)


def compiled_relations(full, allowed_queries, run_nms):
    """Top-50 native NMS survivors and their geometric conflict neighbors.

    allowed_queries are full-frame source query IDs that survived sensor filter.
    Every returned edge is verified by a TWO-ROW call to the installed compiled
    NMS. Edge direction is not a historical suppression assignment. Full native
    top-50 is determined before restricting to cached queries, never on a subset.
    """
    g, qids = full
    tagged = np.asarray(g, dtype=np.float64).copy(); tagged[:, 16] = np.arange(len(g))
    survivors = run_nms(tagged.copy())
    indices = np.asarray(survivors[:, 16], dtype=np.int64)
    if len(indices) != len(np.unique(indices)) or not np.isin(indices, np.arange(len(g))).all():
        raise RuntimeError('Compiled NMS did not preserve unique row identity')
    if not np.array_equal(survivors[:, :16], tagged[indices, :16]):
        raise RuntimeError('Compiled NMS changed physical actions/scores')
    if len(survivors) > 1 and np.any(np.diff(survivors[:, 0]) > 0):
        raise RuntimeError('Expected compiled native descending score order')
    allowed = set(map(int, allowed_queries))
    relevant = [i for i in range(len(g)) if int(qids[i]) in allowed]
    leaders = [int(i) for i in indices[:50] if int(qids[i]) in allowed]
    edges = set(); visited = set(); calls = 0
    for i in leaders:
        for j in relevant:
            if i == j: continue
            a, b = sorted((int(qids[i]), int(qids[j])))
            # A repeated pair needs no repeated compiler call.
            if (a, b) in visited: continue
            visited.add((a,b))
            pair = tagged[[i, j]].copy(); kept = run_nms(pair); calls += 1
            if len(kept) == 1: edges.add((a, b))
            elif len(kept) != 2: raise RuntimeError('Invalid pairwise compiled NMS result')
    return sorted(edges), dict(full_sensor_candidates=len(g), full_nms_survivors=len(indices),
                               cached_top50_survivors=len(leaders), pairwise_nms_calls=calls,
                               relation='native_top50_survivor_conflict_neighbor_NOT_directed_suppression')


def trace(args):
    source, frames = source_index(args.source_index, formal=not args.smoke)
    rt, runtime_identity = make_runtime(args)
    if rt.config_digest != source['config_digest']: raise ValueError('Source runtime config mismatch')
    if not 0 <= args.shard_index < args.shards: raise ValueError('Invalid shard')
    from graspnetAPI import GraspGroup
    import grasp_nms
    nms_file = inspect.getfile(grasp_nms)
    def nms(g): return GraspGroup(g).nms(.03, float(np.pi / 6)).grasp_group_array
    source_seed = int(source['contract']['seed'])
    contract = dict(schema=SCHEMA, source_index_sha256=source['index_sha256'],
                    acceptance_sha256=sha(args.accepted_label_report),
                    base_checkpoint_sha256=BASE_SHA, base_config_digest=rt.config_digest,
                    source_snapshot_seed=source_seed, runtime=runtime_identity,
                    compiled_nms_sha256=sha(nms_file), script_sha256=sha(__file__),
                    core_sha256=sha(Path(__file__).with_name('core.py')),
                    data_sha256=sha(Path(__file__).with_name('data.py')),
                    edge_definition='native-top50-survivor-conflict-neighbor',
                    no_cad_relabel=True, smoke=bool(args.smoke))
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    entries = []
    for num, (_, es) in enumerate(frames):
        if num % args.shards != args.shard_index: continue
        row = es['uniform']['row']; name = frame_name(row)
        path = root / 'frames' / (name + '.npz'); sidecar = path.with_suffix('.json')
        source_shas = {k: es[k]['sha256'] for k in ('uniform', 'decision')}
        if path.exists() or sidecar.exists():
            if not args.resume or not (path.exists() and sidecar.exists()): raise FileExistsError(path)
            old = json.loads(sidecar.read_text())
            if old['contract_sha256'] != digest(contract) or old['source_sha256'] != source_shas:
                raise ValueError('Trace resume contract mismatch')
            if sha(path) != old['sha256']: raise ValueError('Corrupt trace')
            entries.append(old); continue
        x = union_frame(es); snap = rt.snapshot(row, source_seed)
        bank = np.asarray(snap['grasps'])
        if bank.shape != (1024, 4, 17): raise ValueError('Require full 1024-query native forward')
        fam = x['family']; dep = x['depth']; actual = bank[fam, dep]
        if not np.array_equal(physical_ids(actual), x['physical_id'].astype(str)):
            raise ValueError('Native physical actions no longer match accepted labels')
        for field, current, cached in (
            ('geometry', actual[:, 1:16], x['grasps'][:, 1:16]),
            ('score', actual[:, 0], x['grasps'][:, 0]),
            ('feature', np.asarray(snap['features'])[fam, dep], x['features']),
            ('logit', np.asarray(snap['base_logit'])[fam, dep], x['base_logit'])):
            err = float(np.max(np.abs(current - cached)))
            if err > 1e-6: raise RuntimeError(f'Native cache parity failed: {field} {err}; use accepted runtime')
        winners = bank[np.arange(1024), bank[..., 0].argmax(1)].copy()
        filtered = rt.sensor_collision_filter(row, winners)
        ids = filtered_indices(winners, filtered)
        families = np.unique(fam)
        pair_queries, audit = compiled_relations((filtered, ids), families, nms)
        nr = native_rows(x); q_to_row = {int(fam[i]): int(i) for i in nr}
        edges = [sorted([q_to_row[a], q_to_row[b]]) for a, b in pair_queries]
        x['edges'] = np.asarray(sorted(set(map(tuple, edges))), np.int64).reshape(-1, 2)
        x['native_sensor_valid'] = np.isin(families, ids)
        write_npz(path, x)
        item = dict(row=row, file=str(path.resolve()), sha256=sha(path), source_sha256=source_shas,
                    contract_sha256=digest(contract), trace_audit=audit)
        write_json(sidecar, item); entries.append(item)
        print(f'[trace] {name} families={len(families)} edges={len(edges)}', flush=True)
    manifest = dict(schema=SCHEMA, contract=contract, entries=entries,
                    shard_index=args.shard_index, shards=args.shards)
    manifest['manifest_sha256'] = digest(manifest)
    out = root / f'trace_{args.shard_index:02d}_of_{args.shards:02d}.json'
    if out.exists() and not args.resume: raise FileExistsError(out)
    write_json(out, manifest)


def infer(args):
    from .train import load_head
    rt, runtime_identity = make_runtime(args)
    spec = {'file': args.weights, 'sha256': args.weights_sha256} if args.weights else None
    model, info = load_head(spec, args.device) if spec else (None, None)
    if model is not None and (info['config']['base_checkpoint_sha256'] != BASE_SHA or
                             info['config']['base_config_digest'] != rt.config_digest):
        raise ValueError('Probe/base mismatch')
    if info and (info['config'].get('smoke') or info.get('epoch') != 10):
        raise ValueError('Formal inference requires fixed epoch10, not smoke or alternate checkpoint selection')
    policies = args.policies.split(',')
    if len(policies) != len(set(policies)) or any(p not in ('native','rank_executed','rank_max','both','depth_only') for p in policies):
        raise ValueError('Invalid/duplicate policy')
    if args.limit < 0: raise ValueError('limit must be nonnegative')
    if model is None and policies != ['native']: raise ValueError('Native has no learned policies')
    root = Path(args.output)
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    scene_start = {'test_seen': 100, 'test_similar': 130, 'test_novel': 160}[args.split]
    frames = [(s, a) for s in range(scene_start, scene_start + 30) for a in range(0, 256, args.interval)]
    if args.limit: frames = frames[:args.limit]
    records = []
    for frame_number, (s, a) in enumerate(frames, 1):
        row = dict(domain='original', scene=s, anno=a, split=args.split, partition='test')
        snap = rt.snapshot(row, args.snapshot_seed); bank = np.asarray(snap['grasps'])
        x = dict(features=np.asarray(snap['features']).reshape(-1, 64),
                 base_logit=np.asarray(snap['base_logit']).reshape(-1),
                 grasps=bank.reshape(-1, 17), depth=np.tile(np.arange(4), 1024))
        with torch.inference_mode():
            scores = predict(model, x, args.device).sigmoid().cpu().numpy().reshape(1024, 4) if model else bank[..., 0]
        for policy in policies:
            g = decode(bank, scores, policy)
            filtered = rt.sensor_collision_filter(row, g)
            path = root / policy / f'scene_{s:04d}' / 'realsense' / f'{a:04d}.npy'
            path.parent.mkdir(parents=True, exist_ok=True); np.save(path, filtered)
            records.append(dict(scene=s, anno=a, policy=policy, file=str(path.relative_to(root)),
                                sha256=sha(path), count=len(filtered),
                                frozen_bank_sha256=hashlib.sha256(np.ascontiguousarray(bank[:,:,1:16]).tobytes()).hexdigest()))
        if frame_number % 26 == 0:
            print(f'[infer] {frame_number}/{len(frames)} scene={s} anno={a}', flush=True)
    m = dict(schema=SCHEMA, kind='inference', complete=not bool(args.limit),
             split=args.split, interval=args.interval, snapshot_seed=args.snapshot_seed,
             base_checkpoint_sha256=BASE_SHA, base_config_digest=rt.config_digest,
             runtime=runtime_identity, implementation_sha256={'runtime':sha(__file__), 'core':sha(Path(__file__).with_name('core.py'))}, probe_sha256=sha(args.weights) if args.weights else None,
             probe_arm=info['config'].get('arm') if info else 'native',
             probe_seed=info['config'].get('seed') if info else None,
             plan_sha256=info['config'].get('plan_sha256') if info else None,
             policies=policies, frames=len(frames), records=records,
             collision_protocol='original_sensor_voxel0.01_threshold0.01_approach0.05',
             labels_used=False)
    write_json(root / 'inference.json', m)


def evaluate(args):
    root = Path(args.inference); m = json.loads((root / 'inference.json').read_text())
    if m['schema'] != SCHEMA or not m['complete']: raise ValueError('Incomplete/smoke run')
    expected = 30 * len(range(0, 256, m['interval']))
    for p in m['policies']:
        rows = [r for r in m['records'] if r['policy'] == p]
        if len(rows) != expected or len({(r['scene'], r['anno']) for r in rows}) != expected:
            raise ValueError('Missing/duplicate formal inference frames')
    for r in m['records']:
        path = root / r['file']
        if sha(path) != r['sha256']: raise ValueError('Inference dump changed')
        g = np.load(path, allow_pickle=False)
        if g.ndim != 2 or g.shape[1] != 17 or not np.isfinite(g).all(): raise ValueError('Malformed grasp array')
    result = {}
    for policy in m['policies']:
        subprocess.run([sys.executable, 'eval.py', '--dataset_root', args.dataset_root,
                        '--dump_dir', str(root / policy), '--camera', 'realsense',
                        '--split', m['split'], '--sample_interval', str(m['interval']),
                        '--num_workers', str(args.workers)], check=True)
        path = root / policy / f"ap_{m['split']}_realsense.npy"
        ap = np.load(path, allow_pickle=False)
        if ap.shape != (30, expected // 30, 50, 6) or not np.isfinite(ap).all() or ap.min() < 0 or ap.max() > 1:
            raise ValueError('Invalid official AP')
        result[policy] = dict(sha256=sha(path), file=str(path.resolve()), ap_percent=float(100 * ap.mean()),
                             prefix={str(k): {'mean_quality':float(ap[:,:,k-1,:].mean()),
                                             'success_rate_mu1_2':float(ap[:,:,k-1,5].mean()),
                                             'padded_non_success_rate':float(1-ap[:,:,k-1,5].mean())}
                                     for k in (1,10,50)},
                             prefix_scope='Official budgets; missing candidate slots count as non-success, not observed false-positive actions')
    write_json(root / 'evaluation.json', dict(schema=SCHEMA, inference_sha256=sha(root / 'inference.json'), results=result))


def main():
    p = argparse.ArgumentParser(description=__doc__); sp = p.add_subparsers(dest='cmd', required=True)
    for name in ('trace', 'infer'):
        a = sp.add_parser(name)
        a.add_argument('--config', required=True); a.add_argument('--dataset-root', required=True)
        a.add_argument('--gntrans-root', default=''); a.add_argument('--device', default='cuda:0')
        a.add_argument('--factory', default='rare_grasp.runtime:Runtime')
        a.add_argument('--output', required=True)
        if name == 'trace':
            a.add_argument('--source-index', required=True); a.add_argument('--accepted-label-report', required=True)
            a.add_argument('--shards', type=int, default=1); a.add_argument('--shard-index', type=int, default=0)
            a.add_argument('--resume', action='store_true'); a.add_argument('--smoke', action='store_true')
        else:
            a.add_argument('--weights'); a.add_argument('--weights-sha256')
            a.add_argument('--split', choices=('test_seen', 'test_similar', 'test_novel'), required=True)
            a.add_argument('--interval', type=int, choices=(1, 10), default=10)
            a.add_argument('--policies', default='rank_executed,both')
            a.add_argument('--snapshot-seed', type=int, default=20261113); a.add_argument('--limit', type=int, default=0)
    a = sp.add_parser('eval'); a.add_argument('--inference', required=True)
    a.add_argument('--dataset-root', required=True); a.add_argument('--workers', type=int, default=16)
    x = p.parse_args(); {'trace': trace, 'infer': infer, 'eval': evaluate}[x.cmd](x)

if __name__ == '__main__': main()
