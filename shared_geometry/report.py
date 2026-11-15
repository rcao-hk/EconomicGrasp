"""Scene-paired inference summary; AP claims are separate from geometry metrics."""
from pathlib import Path
import numpy as np
from .data import signed, atomic_json, load_signed, sha256


def summarize(a):
    if a.bootstrap < 1: raise ValueError('Positive bootstrap count required')
    roots = {}
    for item in a.split_dir:
        split, path = item.split('=', 1)
        if split in roots: raise ValueError('Duplicate split')
        roots[split] = Path(path)
    required = ('test_seen', 'test_similar', 'test_novel')
    if set(roots) != set(required): raise ValueError('Three full official splits required')
    arrays = {}; identity = None; sources = {}
    for split in required:
        root = roots[split]; manifest = load_signed(root/'inference_manifest.json')
        done = load_signed(root/'evaluation_complete.json')
        if done['inference_sha256'] != manifest['fingerprint']: raise ValueError('Stale eval record')
        current = (manifest['source'], manifest['weights']['sha256'], manifest['protocol'])
        if identity is None: identity = current
        if current != identity: raise ValueError('Checkpoint/source changed between splits')
        arrays[split] = {}
        for mode in manifest['variants']:
            p = root/mode/f'ap_{split}_realsense.npy'
            if sha256(p) != done['methods'][mode]['sha256']: raise ValueError('AP hash mismatch')
            ap = np.load(p, allow_pickle=False)
            if ap.shape != (30, 256, 50, 6) or not np.isfinite(ap).all(): raise ValueError('Wrong AP array')
            arrays[split][mode] = ap.mean((1, 2, 3))*100
        sources[split] = done['fingerprint']
    rng = np.random.default_rng(a.seed); results = {}
    for mode in arrays[required[0]]:
        delta = np.stack([arrays[s][mode]-arrays[s]['native'] for s in required])
        boot = np.zeros(a.bootstrap)
        for d in delta:
            boot += d[rng.integers(0, 30, size=(a.bootstrap, 30))].mean(1)/3
        results[mode] = dict(split_ap={s: float(arrays[s][mode].mean()) for s in required},
                             mean_ap=float(np.mean([arrays[s][mode].mean() for s in required])),
                             delta_pp=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist())
    atomic_json(a.output, signed(dict(results=results, evaluation_sources=sources,
        note='Scene-paired, split-stratified bootstrap conditional on fixed weights; no training-seed uncertainty')))
