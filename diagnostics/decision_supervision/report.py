"""Paired official AP, per-seed factorial effects, and transparent seed variability."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .core import SCHEMA, ARMS, sha, write_json

SPLITS = ('test_seen', 'test_similar', 'test_novel')


def read_result(root, policy):
    root = Path(root)
    m = json.loads((root / 'inference.json').read_text())
    e = json.loads((root / 'evaluation.json').read_text())
    if m['schema'] != SCHEMA or not m['complete'] or e['inference_sha256'] != sha(root / 'inference.json'):
        raise ValueError('Incomplete/unpaired official evaluation')
    r = e['results'][policy]; path = Path(r['file'])
    if sha(path) != r['sha256']: raise ValueError('AP array hash mismatch')
    ap = np.load(path, allow_pickle=False)
    expected = (30, len(range(0, 256, m['interval'])), 50, 6)
    if ap.shape != expected or not np.isfinite(ap).all(): raise ValueError('Wrong AP array')
    return m, ap.mean(axis=(1, 2, 3)) * 100


def effects(a):
    # A/C share unstratified data, B/D stratified data; only C/D targeted loss.
    return {'B-A_sampling_random': a['B'] - a['A'],
            'C-A_targeting_raw': a['C'] - a['A'],
            'D-B_targeting_stratified': a['D'] - a['B'],
            'D-C_sampling_targeted': a['D'] - a['C'],
            'D-A_combined': a['D'] - a['A'],
            'interaction_D-C-B+A': a['D'] - a['C'] - a['B'] + a['A']}


def paired_ci(values, seed, draws):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    # Fixed number of scenes per split; never mix the three domains when drawing.
    if values.shape != (3, 30): raise ValueError('Require 3x30 scene effects')
    samples = np.stack([v[rng.integers(0, 30, (draws, 30))].mean(1) for v in values]).mean(0)
    return dict(mean_pp=float(values.mean()), ci95_pp=np.quantile(samples, [.025, .975]).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True, help='JSON: seed -> A/B/C/D -> split -> inference root')
    p.add_argument('--policy', default='rank_executed')
    p.add_argument('--output', required=True); p.add_argument('--bootstrap', type=int, default=5000)
    a = p.parse_args()
    if a.bootstrap < 1: raise ValueError('bootstrap must be positive')
    runs = json.loads(Path(a.runs).read_text()); output = Path(a.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}; rows = []; all_effects = {}; identity = None; physical_banks = {}
    for seed, arms in sorted(runs.items()):
        if set(arms) != set(ARMS): raise ValueError('All four arms required for factorial analysis')
        arrays = {}
        for arm in ARMS:
            if set(arms[arm]) != set(SPLITS): raise ValueError('All three splits required')
            values = []
            for split in SPLITS:
                m, scores = read_result(arms[arm][split], a.policy)
                if m['probe_arm'] != arm or str(m['probe_seed']) != str(seed) or m['split'] != split:
                    raise ValueError('Run mapping does not match saved arm/seed/split')
                current = {k: m[k] for k in ('base_checkpoint_sha256', 'base_config_digest',
                           'interval', 'snapshot_seed', 'runtime', 'implementation_sha256', 'collision_protocol', 'plan_sha256')}
                if identity is None: identity = current
                if identity != current: raise ValueError('Different evaluation protocols or training plans')
                current_banks={(r['scene'],r['anno']):r['frozen_bank_sha256'] for r in m['records'] if r['policy']==a.policy}
                if split in physical_banks and physical_banks[split] != current_banks:
                    raise ValueError('Frozen base physical banks differ across heads/seeds')
                physical_banks[split]=current_banks
                values.append(scores)
                scene_start = {'test_seen': 100, 'test_similar': 130, 'test_novel': 160}[split]
                rows.extend(dict(seed=seed, arm=arm, split=split, scene=scene_start+i,
                                 ap_percent=float(v)) for i, v in enumerate(scores))
            arrays[arm] = np.stack(values)
        eff = effects(arrays); all_effects[seed] = eff
        summaries[seed] = dict(ap={arm: dict(zip(SPLITS, v.mean(1).tolist())) for arm, v in arrays.items()},
                              effects={name: paired_ci(v, 20261114, a.bootstrap) for name, v in eff.items()},
                              split_effects={name: dict(zip(SPLITS, v.mean(1).tolist())) for name, v in eff.items()})
    means = {name: [e[name].mean() for e in all_effects.values()] for name in next(iter(all_effects.values()))}
    # Scene CI conditional on THESE heads; seed dispersion separately, not pseudo-IID 90*seeds.
    report = dict(schema=SCHEMA, policy=a.policy, per_seed=summaries,
                  fixed_heads_scene_ci={name: paired_ci(np.mean([e[name] for e in all_effects.values()], axis=0),
                                                       20261114, a.bootstrap) for name in means},
                  seed_dispersion={name: dict(values_pp=list(map(float, v)), mean_pp=float(np.mean(v)),
                                   std_pp=float(np.std(v, ddof=1)) if len(v)>1 else None) for name, v in means.items()},
                  inference_contract=identity, source_mapping_sha256=sha(a.runs),
                  limitation='Scene bootstrap is conditional on trained heads; seed spread reported separately. Reused test set is exploratory.')
    write_json(output / 'summary.json', report)
    with (output / 'per_scene.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['# 四臂配对官方AP', '', f'策略：{a.policy}。同split配对，scene是抽样单位；seed差异单列。', '',
             '| Contrast | Mean pp | Conditional scene CI | Seed std pp |', '|---|---:|---|---:|']
    for name, value in report['fixed_heads_scene_ci'].items():
        lines.append(f"| {name} | {value['mean_pp']:.6f} | {value['ci95_pp']} | {report['seed_dispersion'][name]['std_pp']} |")
    (output / 'RESULTS_CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

if __name__ == '__main__': main()
