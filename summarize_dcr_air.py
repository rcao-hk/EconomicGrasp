#!/usr/bin/env python3
"""Summarize completed DCR-AIR official cells without filling missing cells."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from e1e2_common import atomic_file, save_json


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, restval='')
    writer.writeheader()
    writer.writerows(rows)
    with atomic_file(path) as f:
        f.write(buf.getvalue().encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root', required=True)
    args = p.parse_args()
    root = Path(args.work_root)
    rows = []
    for path in sorted((root / 'test' / 'official').glob('*/*/*/summary.json')):
        d = json.loads(path.read_text())
        rows.append({k: d[k] for k in (
            'variant', 'method', 'case', 'split', 'mean_accuracy',
            'sample_interval', 'seen_role')})
    if not rows:
        raise RuntimeError('No completed AIR official results')

    write_csv(root / 'comparison.csv', rows)
    save_json(root / 'comparison.json', rows)

    effects = []
    for split, case in sorted({(r['split'], r['case']) for r in rows}):
        cell = {
            r['method']: r['mean_accuracy']
            for r in rows
            if r['split'] == split and r['case'] == case
        }
        row = {'split': split, 'case': case, **cell}
        if 'air_stage1' in cell and 'dcr_stage1' in cell:
            row['air_over_dcr'] = cell['air_stage1'] - cell['dcr_stage1']
        if 'dcr_stage1' in cell and 'native' in cell:
            row['dcr_over_native'] = cell['dcr_stage1'] - cell['native']
        if 'air_stage1' in cell and 'native' in cell:
            row['air_over_native'] = cell['air_stage1'] - cell['native']
        effects.append(row)
    write_csv(root / 'air_effects.csv', effects)
    save_json(root / 'air_effects.json', effects)
    print(f'[AIR] {len(rows)} completed cells; {root / "air_effects.csv"}')


if __name__ == '__main__':
    main()
