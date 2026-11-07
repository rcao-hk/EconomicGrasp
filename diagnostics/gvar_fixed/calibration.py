"""Independent-scene four-offset baseline, not a new probability/reader model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .analysis import load_rows, scene_ci
from .contracts import write_json, file_sha256


def fit_offsets(rows, ridge=.1):
    if ridge < 0: raise ValueError('Ridge must be nonnegative')
    groups = defaultdict(list)
    for r in rows:
        if r['condition'] == 'native': groups[(r['reference'], r['model'])].append(r)
    if not groups: raise ValueError('No native calibration scores')
    params = {}
    for (ref, model), group in groups.items():
        checks = {r['checkpoint_sha256'] for r in group}
        sources = {r['source_checkpoint_sha256'] for r in group}
        if len(checks) != 1 or len(sources) != 1: raise ValueError('Mixed calibration identities')
        offsets = []
        for d in range(4):
            records = [dict(scene=r['scene_id'], split=r['split'], sample=r['sample_id'],
                            residual=r['quality'] - r['score']) for r in group if r['depth_id'] == d]
            # Analytic minimizer of equal-cluster weighted (s+b-Q)^2 + ridge*b^2.
            value = scene_ci(records, 'residual', n_boot=0)['mean']
            if value is None: raise ValueError('Calibration lacks a depth bin')
            offsets.append(value / (1 + ridge))
        params['|'.join((ref, model))] = dict(offsets=offsets, checkpoint_sha256=next(iter(checks)),
                                            source_checkpoint_sha256=next(iter(sources)))
    return dict(kind='depth_additive_mse_offsets_v1', ridge=ridge,
                scene_ids=sorted({int(r['scene_id']) for r in rows}), parameters=params,
                semantics='Additive ranking offsets; adjusted score is NOT a calibrated probability; no clipping.',
                fitting='Validation-only equal split/scene/frame weighting; no offset search on evaluated scenes.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-csv', action='append', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ridge', type=float, default=.1)
    args = p.parse_args()
    if Path(args.output).exists(): raise FileExistsError('Do not overwrite a frozen calibration artifact')
    result = fit_offsets(load_rows(args.validation_csv), args.ridge)
    result['source_sha256'] = {f: file_sha256(f) for f in args.validation_csv}
    write_json(args.output, result)
    print(f"Saved offsets from {len(result['scene_ids'])} scenes to {args.output}")


if __name__ == '__main__': main()
