"""Contracts/utilities for E1/E2; no legacy argv parser or heavy imports here."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

VERSION = 'center_cva_e1e2_v1'
MAIN_SHA = '52d09f925059bec3643610ecf1f1722894627ee5'
SPLITS = {'train': range(0, 100), 'test_seen': range(100, 130),
          'test_similar': range(130, 160), 'test_novel': range(160, 190)}
THRESHOLDS = (.2, .4, .6, .8, 1., 1.2)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(*arrays):
    h = hashlib.sha256()
    for x in arrays:
        a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def frame_stride(sample_interval=.1):
    """A fraction, NOT an integer stride. .1 -> frames 0,10,...,250."""
    f = float(sample_interval)
    if not 0 < f <= 1:
        raise ValueError('sample_interval must be a fraction in (0,1]')
    stride = int(round(1/f))
    if not math.isclose(1/stride, f, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError('Use a reciprocal integer fraction, such as 0.1')
    return stride


def schedule(split, sample_interval=.1):
    if split not in SPLITS:
        raise ValueError(split)
    return [(sid, aid) for sid in SPLITS[split]
            for aid in range(0, 256, frame_stride(sample_interval))]


def shard_frames(frames, shard_id=0, num_shards=1, max_frames=0):
    if not 0 <= shard_id < num_shards:
        raise ValueError('Invalid shard id/count')
    # Limit BEFORE sharding: changing worker count does not change smoke scope.
    frames = list(frames[:max_frames] if max_frames else frames)
    scenes = sorted({s for s, _ in frames})
    owned = set(scenes[shard_id::num_shards])
    return [(s, a) for s, a in frames if s in owned]


def parse_offsets(text):
    x = np.asarray([float(s) for s in text.split(',')], np.float32)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError('Non-finite/empty offsets')
    if len(np.unique(x)) != len(x) or np.sum(x == 0) != 1:
        raise ValueError('Offsets must be unique and contain exactly one zero')
    return x


def case_key(case):
    return case.replace(':', '_').replace('-', 'm').replace('+', 'p').replace('.', 'p')


def seed_for(seed, *parts):
    return int(hashlib.sha256(repr((seed, parts)).encode()).hexdigest()[:8], 16)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_case(seed):
    rng = np.random.default_rng(seed)
    family = int(rng.integers(3))
    if family == 0:
        return f'bias:{rng.uniform(-20, 20):.6f}'
    if family == 1:
        return f'scale:{rng.uniform(-.03, .03):.8f}'
    return f'smooth:{rng.uniform(1, 10):.6f}'


def perturb_depth(depth, case, seed=0):
    """Does not mutate input. Positive bias moves camera-z away from the camera."""
    if case == 'nominal':
        return depth
    kind, v = case.split(':'); v = float(v)
    if not math.isfinite(v):
        raise ValueError(case)
    if kind == 'bias':
        err = torch.full_like(depth, v/1000.)
    elif kind == 'scale':
        err = depth*v
    elif kind == 'smooth':
        if v < 0:
            raise ValueError(case)
        noise = np.random.default_rng(seed).normal(size=(1, 1, 8, 8)).astype(np.float32)
        field = torch.from_numpy(noise).to(depth)
        field = torch.nn.functional.interpolate(field, size=depth.shape[-2:], mode='bilinear', align_corners=False)
        field = field - field.mean()
        err = field / field.square().mean().sqrt().clamp_min(1e-8) * (v/1000.)
    else:
        raise ValueError(f'Unsupported error case {case}')
    return depth + err


def expand_centers(native, offsets, min_depth=.2, max_depth=1.):
    """[Q,17] -> [C,Q,17]. Camera-z offset, NOT insertion depth or ray arclength."""
    if native.ndim != 2 or native.shape[1] != 17:
        raise ValueError('native must be [Q,17]')
    if not torch.isfinite(native).all() or not bool((native[:, 15] > 0).all()):
        raise ValueError('Nonfinite/nonpositive native center')
    off = torch.as_tensor(offsets, device=native.device, dtype=native.dtype)
    z = native[:, 15]
    out = native.unsqueeze(0).repeat(len(off), 1, 1)
    ray = native[:, 13:16] / z[:, None]
    out[:, :, 13:16] = (z[None] + off[:, None]/1000.)[:, :, None] * ray[None]
    valid = (out[:, :, 15] > min_depth) & (out[:, :, 15] < max_depth)
    return out, valid


def cdf_targets(friction):
    f = np.asarray(friction, np.float32)
    t = np.asarray(THRESHOLDS, np.float32)
    return ((f[..., None] > 0) & (f[..., None] <= t + 1e-6)).astype(np.float32)


def select_centers(utility, valid, zero):
    """Zero-threshold utility decision; exact numerical ties stay native."""
    if utility.shape != valid.shape or utility.ndim != 2:
        raise ValueError('Expected matching [C,Q] utility/valid')
    if not bool(valid[zero].all()):
        raise ValueError('Native must be valid for every query')
    u = utility.masked_fill(~valid, -torch.inf)
    best = u.argmax(0)
    q = torch.arange(u.shape[1], device=u.device)
    return torch.where(u[best, q] > u[zero] + 1e-7, best, torch.full_like(best, zero))


@contextlib.contextmanager
def atomic_file(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            yield f
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path, obj):
    with atomic_file(path) as f:
        f.write(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode())


def save_npz(path, **obj):
    with atomic_file(path) as f:
        np.savez_compressed(f, **obj)


def save_torch(path, obj):
    with atomic_file(path) as f:
        torch.save(obj, f)


def load_torch(path):
    # Repository checkpoints are trusted local experiment artifacts.
    return torch.load(path, map_location='cpu', weights_only=False)


@contextlib.contextmanager
def lock(path, wait=False):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def manifest(root, obj):
    path = Path(root)/'protocol.json'
    with lock(Path(root)/'.protocol.lock', wait=True):
        if path.exists() and json.loads(path.read_text()) != obj:
            raise RuntimeError(f'Protocol mismatch at {path}; use a new output root')
        if not path.exists():
            save_json(path, obj)


def require_ram(gib):
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, value = line.split(':', 1); info[k] = int(value.strip().split()[0])
    available = info['MemAvailable'] / 2**20
    if available < gib:
        raise MemoryError(f'Available host RAM {available:.1f} GiB < required {gib:.1f} GiB')
    return available


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_dataset(root, split, camera='realsense'):
    from dataset.graspnet_dataset import GraspNetMultiDataset
    ds = GraspNetMultiDataset(root, split=split, camera=camera, num_points=20000,
                             remove_outlier=True, augment=False, load_label=False,
                             use_gt_depth=False, use_fuse_depth=False,
                             min_depth=.2, max_depth=1., bin_num=256)
    lookup = {(int(str(s).split('_')[-1]), i % 256): i
              for i, s in enumerate(ds.scene_list())}
    return ds, lookup


def get_batch(ds, lookup, sid, aid, device):
    from dataset.graspnet_dataset import collate_fn
    item = ds[lookup[(sid, aid)]]
    # Benchmark preprocessing is inherited; no depth/GT is passed to the model.
    keys = ('img', 'K', 'camera_pose_vec', 'camera_gravity_vec', 'scene_idx',
            'anno_idx', 'token_valid_mask')
    return move_batch(collate_fn([{k: item[k] for k in keys if k in item}]), device)


def batch_fingerprint(batch):
    return array_sha(*(batch[k] for k in ('img', 'K', 'camera_pose_vec', 'camera_gravity_vec', 'token_valid_mask') if k in batch))


def relative_cdf_loss(logits, target, valid, zero, variant='E1', relative_weight=1., beta=.1):
    """E1 = CDF BCE. E2 = same BCE + native-relative utility Huber; no new head."""
    if variant not in ('E1', 'E2') or beta <= 0 or relative_weight < 0:
        raise ValueError('Invalid loss configuration')
    if logits.shape != target.shape or logits.shape[:-1] != valid.shape or logits.shape[-1] != 6:
        raise ValueError('Expected [C,Q,6] logits/targets and [C,Q] validity')
    if not bool(valid[zero].all()) or not bool(valid.any()):
        raise ValueError('Missing native/valid labels')
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], target[valid])
    pred_u, true_u = logits.sigmoid().mean(-1), target.mean(-1)
    pair_valid = valid & valid[zero:zero+1]
    pair_valid = pair_valid.clone(); pair_valid[zero] = False
    if bool(pair_valid.any()):
        relative = torch.nn.functional.smooth_l1_loss(
            (pred_u-pred_u[zero])[pair_valid], (true_u-true_u[zero])[pair_valid], beta=beta)
    else:
        relative = logits.sum()*0.
    loss = bce + (relative_weight*relative if variant == 'E2' else 0.)
    return loss, {'cdf_bce': float(bce.detach()), 'relative_huber': float(relative.detach())}
