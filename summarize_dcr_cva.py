#!/usr/bin/env python3
"""Summarize completed DCR official cells; never turn missing cells into zero."""
import argparse
import csv
import io
import json
from pathlib import Path
from e1e2_common import atomic_file, save_json


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=fields,restval='')
    w.writeheader(); w.writerows(rows)
    with atomic_file(path) as f: f.write(buf.getvalue().encode())


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--work-root',required=True)
    args=p.parse_args(); root=Path(args.work_root); rows=[]
    for path in sorted((root/'test').glob('*/official/*/*/*/summary.json')):
        d=json.loads(path.read_text())
        rows.append({k:d[k] for k in ('variant','method','case','split','mean_accuracy','sample_interval','seen_role')})
    if not rows: raise RuntimeError('No completed official results')
    write_csv(root/'comparison.csv',rows); save_json(root/'comparison.json',rows)
    effects=[]
    for key in sorted({(r['variant'],r['split'],r['case']) for r in rows}):
        cell={r['method']:r['mean_accuracy'] for r in rows if (r['variant'],r['split'],r['case'])==key}
        r=dict(zip(('variant','split','case'),key)); r.update(cell)
        for left,right,name in (('anchored','stage1','residual_over_fixed_rank'),
                                 ('stage1','local','fixed_over_local_score'),
                                 ('stage1','native','action_only_over_native')):
            if left in cell and right in cell: r[name]=cell[left]-cell[right]
        effects.append(r)
    write_csv(root/'ranking_effects.csv',effects)
    print(f'[DCR] {len(rows)} completed cells; {root/"ranking_effects.csv"}')


if __name__=='__main__':
    main()
