#!/usr/bin/env python3
"""Paired scene-level A0-A3 factorial effects; no held-out model selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rep_a_common import save_json, seed_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-root', required=True)
    p.add_argument('--splits', default='test_similar,test_novel')
    p.add_argument('--bootstrap', type=int, default=10000)
    args = p.parse_args()
    root = Path(args.test_root)
    table, effects = [], []
    for split in args.splits.split(','):
        data, summaries = {}, {}
        for variant in ('A0', 'A1', 'A2', 'A3'):
            folder = root/variant/split
            if not (folder/'summary.json').exists():
                raise FileNotFoundError(f'Missing {folder}; all four cells required for factorial comparison')
            summaries[variant] = json.loads((folder/'summary.json').read_text())
            with open(folder/'per_frame.csv') as f:
                data[variant] = {(int(r['scene_id']),int(r['anno_id']),r['case']): r for r in csv.DictReader(f)}
            for case, values in summaries[variant]['cases'].items():
                table.append({'variant': variant, 'split': split, 'case': case, **values})
        base = data['A0']
        for v in ('A1','A2','A3'):
            if base.keys() != data[v].keys():
                raise ValueError(f'Unpaired frames/cases in {split}/{v}')
            for key, r in base.items():
                s = data[v][key]
                if (r['action_sha'], r['num_queries']) != (s['action_sha'], s['num_queries']):
                    raise ValueError(f'Unpaired actions at {key}')
            if summaries[v]['seed'] != summaries['A0']['seed'] or summaries[v]['cache_contract'] != summaries['A0']['cache_contract']:
                raise ValueError('Different perturbations or evidence contracts')
        for case in summaries['A0']['cases']:
            keys = [k for k in base if k[2] == case]
            for metric in ('selected_utility', 'success08', 'harm08', 'probability_drift'):
                vals = {v: np.array([float(data[v][k][metric]) for k in keys]) for v in data}
                contrast = {'structure_no_aug': vals['A2']-vals['A0'],
                            'augmentation_old_reader': vals['A1']-vals['A0'],
                            'structure_with_aug': vals['A3']-vals['A1'],
                            'augmentation_new_reader': vals['A3']-vals['A2'],
                            'interaction': (vals['A3']-vals['A2'])-(vals['A1']-vals['A0'])}
                sid = np.array([k[0] for k in keys])
                for name, diff in contrast.items():
                    scene = np.array([diff[sid==s].mean() for s in np.unique(sid)])
                    rng = np.random.default_rng(seed_for(split, case, metric, name))
                    means = scene[rng.integers(len(scene), size=(args.bootstrap, len(scene)))].mean(1)
                    lo, hi = np.quantile(means,[.025,.975])
                    effects.append({'split':split,'case':case,'metric':metric,'contrast':name,
                        'effect':float(scene.mean()),'ci_low':float(lo),'ci_high':float(hi),
                        'num_scenes':len(scene),'note':'scene percentile CI; single training seed; no multiplicity correction'})
    for name, rows in (('comparison',table),('factorial_effects',effects)):
        save_json(root/(name+'.json'), rows)
        with open(root/(name+'.csv'),'w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'Written {root}/comparison.csv and factorial_effects.csv')


if __name__ == '__main__':
    main()
