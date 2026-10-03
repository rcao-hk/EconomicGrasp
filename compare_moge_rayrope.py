#!/usr/bin/env python3
"""Verified experiment table, NPY-derived AP and paired scene differences."""
from __future__ import annotations
import argparse,json,csv
from pathlib import Path
import numpy as np
from moge_rayrope.runtime import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--variants',default='baseline,moge,rayrope,moge_rayrope')
    p.add_argument('--splits',default='test_seen,test_similar,test_novel')
    p.add_argument('--collision',choices=['on','off','both'],default='both')
    p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args(); root=Path(a.root)
    vs=a.variants.split(','); splits=a.splits.split(','); modes=['on','off'] if a.collision=='both' else [a.collision]
    if a.bootstrap<0: raise ValueError('bootstrap must be nonnegative')
    rows=[]; scenes={}; train_protocols={}; ref=None
    budget_keys=('main_commit','dav2_sha256','train_fraction','eval_fraction','train_frames','val_frames',
                 'sampling_sha256','epochs','seed','effective_batch','lr','weight_decay','grad_clip','use_fuse_depth')
    for v in vs:
        tp=json.loads((root/v/'train'/'protocol.json').read_text()); train_protocols[v]=tp
        if tp['partial_run']: raise RuntimeError(f'{v} is a smoke run')
        budget={k:tp[k] for k in budget_keys}
        if ref is not None and budget!=ref: raise RuntimeError(f'{v}: training budget/source mismatch')
        ref=budget
        for mode in modes:
            row={'variant':v,'collision':mode}; scene=[]
            for split in splits:
                r=root/v/'test_native'/f'test_collision_{mode}'
                proto=json.loads((r/split/'protocol.json').read_text())
                if proto['depth_bias_mm']!=0: raise RuntimeError('Do not mix perturbed/native AP')
                d=r/'official'/split; summary=json.loads((d/'summary.json').read_text())
                arr=np.load(d/'accuracy.npy',allow_pickle=False)
                expected=(30,len(range(0,256,round(1/tp['eval_fraction']))),50,6)
                if int(summary['checkpoint_epoch'])+1!=int(tp['epochs']): raise RuntimeError('Incomplete training budget in AP result')
                if arr.shape!=expected or not np.isfinite(arr).all(): raise RuntimeError(f'Bad accuracy {d}')
                value=float(arr.mean())
                if abs(summary['mean_accuracy']-value)>1e-8: raise RuntimeError(f'Summary mismatch {d}')
                row[split]=100*value; scenes[(v,mode,split)]=arr.mean((1,2,3))
            row['Mean']=float(np.mean([row[s] for s in splits])); rows.append(row)
    contrasts={}
    if 'baseline' in vs:
        rng=np.random.default_rng(0)
        for v in vs:
            if v=='baseline': continue
            for mode in modes:
                for split in splits:
                    delta=scenes[(v,mode,split)]-scenes[('baseline',mode,split)]
                    ci=None
                    if a.bootstrap:
                        boot=delta[rng.integers(0,30,(a.bootstrap,30))].mean(-1)
                        ci=(100*np.quantile(boot,[.025,.975])).tolist()
                    contrasts[f'{v}/{mode}/{split}']={'delta_AP_pp':100*float(delta.mean()),
                        'scene_positive':int((delta>0).sum()),'scene_bootstrap95_pp':ci}
    out=root/'comparison'; out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'comparison.json',{'primary_collision':'on','rows':rows,'paired_contrasts':contrasts,
                                     'protocols':train_protocols,'warning':'Single-run paired scene CI is not across-training-seed uncertainty'})
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['variant','collision']+splits+['Mean']); w.writeheader(); w.writerows(rows)
    lines=['# MoGe + RayRoPE — matched 20% GraspNet','', '**Primary: collision-on.** Values recomputed from accuracy.npy.','',
           '| Variant | Collision | '+' | '.join(splits)+' | Mean |','|---|---|'+'---:|'*(len(splits)+1)]
    for row in rows: lines.append('| '+row['variant']+' | '+row['collision']+' | '+' | '.join(f'{row[k]:.3f}' for k in splits+['Mean'])+' |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
