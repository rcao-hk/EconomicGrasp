"""Signed cache contracts; exact labels are keyed to complete physical actions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from interactive_grasp.contracts import (sha256, digest, signed, load_signed, atomic_json,
    atomic_npz, frame_key, frame_seed, physical_ids, make_plan, validate_plan)
from .geometry import action_bank, support_points, VisibleSurfaceTeacher, DEFAULT_OFFSETS_M
from .runtime import create_runtime, TargetLoader

SCHEMA = 'shared-visible-geometry-v1'


def code_hash():
    roots = [Path(__file__).parent, Path(__file__).parent.parent/'interactive_grasp']
    return digest({str(p.relative_to(r.parent)): sha256(p) for r in roots for p in sorted(r.glob('*.py'))})


def tensor_scene(arrays, device):
    # Explicit whitelist: labels, rendered depth, sensor mask and CAD never enter model.
    return {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device)
            for k in ('feature_map', 'pred_depth', 'K')}


# Deliberately limited to model inputs, never CAD/teacher/sensor fields.
MODEL_INPUT_KEYS = ('feature_map', 'pred_depth', 'K', 'bank', 'local_features')
TRAIN_TARGET_KEYS = ('quality_count', 'target_xyz', 'target_udf',
                     'target_projective', 'target_state', 'target_udf_mask',
                     'target_projective_mask')


def collate_inputs(frames, with_targets=False):
    """Stack independent scenes into an actual B dimension.

    Fail closed on ragged families, evidence grids or target point budgets;
    zero padding could introduce fake physical actions or mismatched geometry.
    Values stay in numpy until a single batch transfer in predict/train.
    """
    if not frames:
        raise ValueError('Empty multi-image batch')
    keys = MODEL_INPUT_KEYS + (TRAIN_TARGET_KEYS if with_targets else ())
    result = {}
    for key in keys:
        values = []
        expected = None
        for i, frame in enumerate(frames):
            if key not in frame:
                raise ValueError(f'Missing {key} in image {i}')
            v = np.asarray(frame[key])
            if expected is None:
                expected = v.shape
            elif v.shape != expected:
                raise ValueError(f'Ragged {key}: image 0 has {expected}, image {i} has {v.shape}; '
                                 'use shape buckets, never pad grasp actions')
            values.append(v)
        result[key] = np.stack(values, axis=0)
    # Crucial: the input-only mode never carries quality labels or rendered geometry.
    if result['bank'].ndim != 4 or result['bank'].shape[-1] != 17:
        raise ValueError('Batch grasp bank must be [B,F,K,17]')
    if result['feature_map'].ndim != 4 or result['pred_depth'].ndim != 3 or result['K'].shape[-2:] != (3,3):
        raise ValueError('Batch image geometry shapes invalid')
    return result


def collate_frames(frames):
    """Aligned training/validation batch including exact labels (never model input)."""
    return collate_inputs(frames, with_targets=True)


def family_indices(scores, n, seed):
    if not 1 <= n <= len(scores):
        raise ValueError('Invalid family budget')
    rng = np.random.default_rng(seed)
    # Keep uniform coverage, plus decision-relevant labels; same bank for all variants.
    high = np.argsort(-scores, kind='stable')[:min(128, len(scores))]
    chosen = rng.choice(high, size=min(n//2, len(high)), replace=False)
    rest = np.setdiff1d(np.arange(len(scores)), chosen)
    chosen = np.r_[chosen, rng.choice(rest, size=n-len(chosen), replace=False)]
    return np.sort(chosen).astype(np.int64)


def validate_cache(a):
    required = ('feature_map', 'pred_depth', 'K', 'bank', 'physical_id', 'quality_count',
                'family_id', 'local_features', 'target_xyz', 'target_udf', 'target_projective',
                'target_state', 'target_udf_mask', 'target_projective_mask')
    if any(k not in a for k in required):
        raise ValueError('Missing cache field')
    g = a['bank']
    if g.ndim != 3 or g.shape[-1] != 17 or g.shape[1] != 7:
        raise ValueError('Expected F x 7 x 17 bank')
    if not np.array_equal(physical_ids(g.reshape(-1, 17)).reshape(g.shape[:2]), a['physical_id']):
        raise ValueError('Physical action identity mismatch')
    if not np.array_equal(action_bank(g[:, 0]), g):
        raise ValueError('Bank does not follow fixed physical perturbation contract')
    rot = g[...,4:13].reshape(-1,3,3)
    if not np.allclose(rot.transpose(0,2,1)@rot, np.eye(3), atol=1e-3) or np.any(np.linalg.det(rot)<.999):
        raise ValueError('Invalid SO(3) grasp geometry')
    if np.any(g[...,1]<0) or np.any(g[...,1]>.100001) or np.any(g[...,15]<=0):
        raise ValueError('Invalid width/camera-plane geometry')
    q = a['quality_count']
    if q.shape != g.shape[:2] or q.dtype.kind not in 'iu' or ((q < 0) | (q > 6)).any():
        raise ValueError('Missing or invalid exact-action labels')
    if len(np.unique(a['family_id'])) != len(g):
        raise ValueError('Duplicate family IDs')
    if a['local_features'].ndim != 2 or len(a['local_features']) != len(g):
        raise ValueError('Local feature shape mismatch')
    n = len(a['target_xyz'])
    if a['target_xyz'].shape != (n, 3) or a['feature_map'].ndim != 3 or a['pred_depth'].ndim != 2:
        raise ValueError('Geometry shape mismatch')
    for name in ('udf', 'projective', 'state', 'udf_mask', 'projective_mask'):
        if a['target_'+name].shape != (n,):
            raise ValueError('Target vector mismatch')
    for k in ('target_udf_mask', 'target_projective_mask'):
        if a[k].dtype != np.bool_: raise ValueError('Geometry masks must be bool')
    if a['target_state'].dtype.kind not in 'iu' or not np.isin(a['target_state'], range(4)).all():
        raise ValueError('Invalid observation state')
    for k in required:
        if k != 'physical_id' and not np.isfinite(a[k]).all():
            raise ValueError('Nonfinite array: '+k)


def export(args):
    plan = load_signed(args.plan); validate_plan(plan)
    if args.field_points<32 or not 1<=args.families<=1024 or args.limit<0:
        raise ValueError('Invalid export budget')
    if args.partition not in ('train', 'validation'):
        raise ValueError('Geometry export only uses train/validation; infer never opens targets')
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError('Bad shard')
    rows = [r for r in plan[args.partition] if (args.domain == 'both' or r['domain'] == args.domain) and r['scene'] % args.shards == args.shard]
    if args.limit: rows = rows[:args.limit]
    runtime = create_runtime(args); teacher_loader = TargetLoader(args.targets)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=SCHEMA, plan=plan['fingerprint'], partition=args.partition,
                    source=runtime.provenance, code_sha256=code_hash(), target_config=teacher_loader.config_sha256,
                    seed=args.seed, offsets_m=list(DEFAULT_OFFSETS_M), families=args.families,
                    field_points=args.field_points, tau=args.tau, band=args.band,
                    sampling='half uniform, half uniformly from source top128; no GT-based selection')
    ch = digest(contract)
    for i, row in enumerate(rows):
        dest = root/(frame_key(row)+'.npz'); meta = dest.with_suffix('.json')
        if dest.exists() or meta.exists():
            if not args.resume or not (dest.is_file() and meta.is_file()):
                raise FileExistsError(dest)
            old = load_signed(meta)
            if old['contract_hash'] != ch or old['sha256'] != sha256(dest):
                raise ValueError('Cache resume contract/hash differs')
            for name in ('depth', 'intrinsics'):
                if sha256(old['target'][name+'_path']) != old['target'][name+'_sha256']:
                    raise ValueError('Geometry teacher source changed')
            continue
        a, parity = runtime.snapshot(row, args.seed)
        ids = family_indices(a['grasps'][:, 0], args.families, frame_seed(row, args.seed))
        bank = action_bank(a['grasps'][ids])
        count = np.asarray(runtime.label(row, bank.reshape(-1, 17)), dtype=np.int64).reshape(bank.shape[:2])
        depth, target_provenance = teacher_loader.load(row, a['K'], a['pred_depth'].shape)
        teacher = VisibleSurfaceTeacher(depth, a['K'], args.tau, args.band)
        support, _ = support_points(torch.from_numpy(bank))
        xyz, labels = teacher.sample(args.field_points, frame_seed(row, args.seed+71), support.numpy())
        arrays = {k: a[k] for k in ('feature_map', 'pred_depth', 'K')}
        arrays.update(bank=bank, physical_id=physical_ids(bank.reshape(-1, 17)).reshape(bank.shape[:2]),
                      quality_count=count, family_id=ids, local_features=a['features'][ids], target_xyz=xyz,
                      **{'target_'+k: v for k, v in labels.items()})
        validate_cache(arrays)
        atomic_npz(dest, arrays)
        atomic_json(meta, signed(dict(row=row, contract=contract, contract_hash=ch, target=target_provenance,
                                     parity=parity, sha256=sha256(dest))))
        print(f'[export] {i+1}/{len(rows)} {frame_key(row)} actions={count.size}', flush=True)


def index(args):
    plan = load_signed(args.plan); validate_plan(plan)
    root = Path(args.cache); entries = []; missing = []; contract = None
    for row in plan[args.partition]:
        f = root/(frame_key(row)+'.npz'); meta = f.with_suffix('.json')
        if not f.exists() or not meta.exists():
            missing.append(frame_key(row)); continue
        m = load_signed(meta)
        if m['row'] != row or sha256(f) != m['sha256'] or digest(m['contract']) != m['contract_hash']:
            raise ValueError('Frame/contract/hash mismatch')
        if m['contract']['plan'] != plan['fingerprint'] or m['contract']['partition'] != args.partition:
            raise ValueError('Manifest partition mismatch')
        if contract is None: contract = m['contract']
        if m['contract'] != contract: raise ValueError('Mixed cache contracts')
        with np.load(f, allow_pickle=False) as z: validate_cache(dict(z))
        entries.append(dict(row=row, path=str(f.resolve()), sha256=m['sha256']))
    if not entries or (missing and not args.smoke):
        raise ValueError(f'Incomplete cache: {len(missing)} frames missing')
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    atomic_json(out, signed(dict(schema=SCHEMA, partition=args.partition, complete=not missing,
                                contract=contract, entries=entries, missing=missing)))


def read_index(path, partition=None, smoke=False):
    idx = load_signed(path)
    if idx['schema'] != SCHEMA or (partition and idx['partition'] != partition):
        raise ValueError('Wrong index')
    if not idx['complete'] and not smoke: raise ValueError('Partial cache is smoke only')
    if not idx['entries']: raise ValueError('Empty index')
    seen = set()
    for e in idx['entries']:
        row = e['row']; key = frame_key(row)
        if row['scene'] >= 100 or row['partition'] != idx['partition'] or key in seen:
            raise ValueError('Split leakage or duplicate frame')
        seen.add(key)
        if sha256(e['path']) != e['sha256']: raise ValueError('Frame hash mismatch')
    return idx


def read_frame(e):
    with np.load(e['path'], allow_pickle=False) as z: a = dict(z)
    validate_cache(a)
    return a