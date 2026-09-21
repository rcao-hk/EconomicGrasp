#!/usr/bin/env python3
"""Summaries/paired contrasts for B0 attribution and full-path evaluations."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rep_a_common import save_json,seed_for
from rep_followup_common import aggregate_decisions,write_csv


def weighted(rows,key,weight='num_queries'):
    use=[r for r in rows if key in r and r[key] not in ('',None)]
    return sum(float(r[key])*int(r[weight]) for r in use)/sum(int(r[weight]) for r in use) if use else None


def bootstrap_scene(values,seed,n=10000):
    v=np.asarray(values,float); rng=np.random.default_rng(seed)
    x=v[rng.integers(len(v),size=(n,len(v)))].mean(1)
    return dict(effect=float(v.mean()),ci_low=float(np.quantile(x,.025)),ci_high=float(np.quantile(x,.975)),num_scenes=len(v))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('attribution','fullpath'),required=True)
    p.add_argument('--root',required=True); p.add_argument('--splits',default='test_seen,test_similar,test_novel')
    args=p.parse_args(); root=Path(args.root); rows=[]; table=[]
    if args.kind=='attribution':
        for path in sorted(root.glob('*/*/per_frame.csv')):
            if path.parent.name not in args.splits.split(','): continue
            with path.open() as f: rows+=list(csv.DictReader(f))
        if not rows: raise FileNotFoundError('No attribution per_frame.csv')
        groups=sorted({(r['name'],r['split'],r['policy']) for r in rows})
        count_keys=('num_queries','num_actions','pair_count','native_best_count','opportunity_count','near_native_pair_count')
        str_keys=('name','split','policy','action_sha')
        for r in rows:
            for k,v in list(r.items()):
                if k not in str_keys: r[k]=int(v) if k in count_keys else float(v)
        for name,split,policy in groups:
            rr=[r for r in rows if (r['name'],r['split'],r['policy'])==(name,split,policy)]
            table.append(dict(name=name,split=split,policy=policy,**aggregate_decisions(rr)))
        effects=[]
        names=sorted({r['name'] for r in rows})
        for split in args.splits.split(','):
            for policy in ('fixed_0','val_selected'):
                lookup={name:{(int(r['scene_id']),int(r['anno_id'])):r for r in rows
                              if (r['name'],r['split'],r['policy'])==(name,split,policy)} for name in names}
                if 'full' not in lookup or not lookup['full']: continue
                base=lookup['full']
                for name,other in lookup.items():
                    if name=='full': continue
                    if base.keys()!=other.keys(): raise ValueError(f'Unpaired attribution frames: {name}/{split}')
                    for key in base:
                        if base[key]['action_sha']!=other[key]['action_sha']: raise ValueError('Action mismatch')
                    for metric in ('selected_utility','success08','harm08'):
                        scenes=sorted({k[0] for k in base})
                        means=[np.mean([other[k][metric]-base[k][metric] for k in base if k[0]==sid]) for sid in scenes]
                        effects.append(dict(split=split,policy=policy,contrast=f'{name}-full',metric=metric,
                            **bootstrap_scene(means,seed_for(name,split,policy,metric)),note='Scene CI, one training seed, uncorrected'))
        if effects: write_csv(root/'paired_effects.csv',effects); save_json(root/'paired_effects.json',effects)
    else:
        expected=sorted((root/'inference').glob('test_*/scene_*/ann_*.npz'))
        expected=[x for x in expected if x.parts[-3] in args.splits.split(',')]
        if not expected: raise FileNotFoundError('No fullpath inference outputs')
        for path in expected:
            target=root/'evaluation'/path.relative_to(root/'inference')
            if not target.exists(): raise FileNotFoundError(f'Exact evaluation incomplete: {target}')
            with np.load(target,allow_pickle=False) as d: rows+=json.loads(str(d['rows_json']))
        keys=('method','policy','split','mode','case')
        groups=sorted({tuple(r[k] for k in keys) for r in rows})
        columns=('selected_utility','native_utility','oracle_utility','utility_gain','success08','native_success08',
                 'success08_gain','rescue08','harm08','move_rate','pure_collision','empty',
                 'raw_top1_success08','raw_top10_success08','raw_top50_success08','depth_rms_mm')
        for group in groups:
            rr=[r for r in rows if tuple(r[k] for k in keys)==group]
            item={**dict(zip(keys,group)),**{c:weighted(rr,c) for c in columns},
                  'num_frames':len(rr),'num_queries':sum(r['num_queries'] for r in rr)}
            if 'native_best_count' in rr[0]: item.update(aggregate_decisions(rr))
            table.append(item)
        for item in table:
            base=next(r for r in table if (r['method'],r['policy'],r['split'],r['case'])==
                      (item['method'],item['policy'],item['split'],'nominal'))
            item['utility_drop_from_nominal']=base['selected_utility']-item['selected_utility']
            item['success08_drop_from_nominal']=base['success08']-item['success08']
        write_csv(root/'per_frame.csv',rows)
    write_csv(root/'comparison.csv',table); save_json(root/'comparison.json',table)
    print(f'Wrote {root}/comparison.csv (raw exact-action metrics, NOT official AP)')


if __name__=='__main__': main()
