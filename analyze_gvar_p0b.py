#!/usr/bin/env python3
"""P0-B: complete RGB-support × geometry-support 2×2 with Inverse-Hybrid.

Fixed/fixed: volume_fixed_rel; fixed/dynamic: volume_hybrid;
dynamic/fixed: volume_inverse_hybrid; dynamic/dynamic: volume_rel.
All independent e19 checkpoint APs with the same Original/10% mixed protocol.
Missing inverse-hybrid is PENDING, never filled with fake AP.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from analyze_gvar_scene_paired import (SPLITS, METRIC_NAMES, canonical_scene_ids,
                                        scene_metric, boot_indices, paired_metric,
                                        load_all, AuditError, sha256_file)
from analyze_gvar_p0_decision import audit_training_log

CELLS=('volume_fixed_rel','volume_hybrid','volume_inverse_hybrid','volume_rel')
NEW='volume_inverse_hybrid'
EXPECTED={'volume_fixed_rel':44.1730,'volume_hybrid':44.1167,'volume_rel':45.5176}
CONTRASTS=('rgb_effect_fixed_geom','rgb_effect_dynamic_geom',
           'geom_effect_fixed_rgb','geom_effect_dynamic_rgb',
           'rgb_mean_effect','geom_mean_effect','interaction')


def require(b,msg):
    if not b: raise AuditError(msg)


def mandatory(root,variant):
    paths=[root/'train'/variant/n for n in ('gvar_protocol.json','gntrans_mix_protocol.json',
                                           'gvar_epochs.jsonl','checkpoint_epoch_019.tar','checkpoint_epoch_020.tar')]
    for split in SPLITS:
        base=root/'eval'/f'{variant}_e19'/split
        paths += [base/n for n in ('gvar_inference_protocol.json','gvar_inference_summary.json',
                                   f'ap_{split}_realsense.npy')]
    return [str(p) for p in paths if not p.is_file()]


def _expr(name, f, fd, df, d):
    return {
        'rgb_effect_fixed_geom':df-f,
        'rgb_effect_dynamic_geom':d-fd,
        'geom_effect_fixed_rgb':fd-f,
        'geom_effect_dynamic_rgb':d-df,
        'rgb_mean_effect':((df-f)+(d-fd))/2,
        'geom_mean_effect':((fd-f)+(d-df))/2,
        'interaction':(d-df)-(fd-f),
    }[name]


def analyze(root:Path,out:Path,bootstrap=50000,seed=20261009,ci=.95,require_ready=False):
    require(root.is_dir(),f'Root missing: {root}')
    require(bootstrap>=200 and 0<ci<1,'Bad CI or bootstrap settings')
    require(out.resolve()!=root.resolve() and not out.resolve().is_relative_to((root/'train').resolve())
            and not out.resolve().is_relative_to((root/'eval').resolve()),
            'Output is not allowed under experiment train/eval directories')
    require(not out.exists() or not any(out.iterdir()),f'Nonempty output: {out}')
    missing={v:mandatory(root,v) for v in CELLS}
    require(not {v:paths for v,paths in missing.items() if v!=NEW and paths},
            f'Old controls incomplete: {missing}')
    status={'version':1,'new_model':NEW,'cells':CELLS,'epoch':19,'status':'pending' if missing[NEW] else 'ready',
            'missing_files':{v:p for v,p in missing.items() if p},'bootstrap':bootstrap,'seed':seed,'ci':ci}
    if missing[NEW]:
        out.mkdir(parents=True,exist_ok=True)
        (out/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        (out/'REPORT.md').write_text('# P0-B pending: missing Inverse-Hybrid official e19 AP\n\n'
                                     +'\n'.join('- '+p for p in missing[NEW])+'\n')
        if require_ready: raise AuditError('P0-B pending; status saved. Do not substitute smoke or e20 outputs.')
        return status
    records,audit=load_all(root,list(CELLS),epoch=19)
    for v in CELLS:
        audit_training_log(root,v)
        cp=root/'train'/v/'checkpoint_epoch_019.tar'
        require(sha256_file(cp)==records[v]['ckpt_sha256'],f'Checkpoint identity mismatch for {v}')
        if v in EXPECTED:
            ap=float(np.mean([records[v]['data'][s].tensor.mean(dtype='float64') for s in SPLITS]))*100
            require(abs(ap-EXPECTED[v])<=.02,f'Historic {v} AP changed, {ap} vs {EXPECTED[v]}')
    idx=boot_indices(bootstrap,seed)
    rows=[]; scenes=[]
    for metric in METRIC_NAMES:
        data={v:{s:scene_metric(records[v]['data'][s].tensor,metric) for s in SPLITS} for v in CELLS}
        for contrast in CONTRASTS:
            values={s:_expr(contrast, *(data[v][s] for v in CELLS)) for s in SPLITS}
            for split in (*SPLITS,'Mean'):
                rows.append({'metric':metric,'contrast':contrast,'split':split,
                             **paired_metric(values,idx,ci,split)})
            if metric=='AP':
                for split in SPLITS:
                    for scene,val in zip(canonical_scene_ids(split),values[split]):
                        scenes.append({'contrast':contrast,'split':split,'scene_id':scene,'delta_AP_pp':float(val*100)})
    out.mkdir(parents=True,exist_ok=True)
    def table(path,values):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    table(out/'factorial_effects.csv',rows);table(out/'factorial_scenes.csv',scenes)
    def fmt(c,split,metric='AP'):
        r=next(x for x in rows if x['contrast']==c and x['split']==split and x['metric']==metric)
        return f"{r['delta_pp']:+.3f} [{r['ci_low_pp']:+.3f},{r['ci_high_pp']:+.3f}]"
    report=['# GVAR P0-B: full RGB × geometry support 2×2 factorial', '',
            'Independent single-seed e19 models, same Original GraspNet evaluation protocol.',
            'Each model uses equal-capacity reader. Changing sampling location also changes associated metadata and validity masks.',
            f'{bootstrap:,} scene-cluster bootstrap draws. This is not a training-seed CI.', '',
            '| Model | RGB support | Geometry support | Mean AP (%) |', '|---|---|---|---:|']
    for v,rgb,geo in ((CELLS[0],'fixed','fixed'),(CELLS[1],'fixed','dynamic'),
                      (CELLS[2],'dynamic','fixed'),(CELLS[3],'dynamic','dynamic')):
        ap=float(np.mean([records[v]['data'][sp].tensor.mean(dtype='float64') for sp in SPLITS]))*100
        report.append(f'| {v} | {rgb} | {geo} | {ap:.3f} |')
    report+=['','## Conditional effects and interaction (AP pp [CI])','',
             '| Contrast | Seen | Similar | Novel | Mean |', '|---|---|---|---|---|']
    for c in CONTRASTS:
        report.append('| '+ ' | '.join((c,*(fmt(c,s) for s in (*SPLITS,'Mean'))))+' |')
    report+=['','## Novel μ=0.4/0.8 interaction','','| μ | Interaction [CI] |', '|---|---|']
    for mu in (.4,.8): report.append(f'| {mu} | {fmt("interaction","test_novel",f"AP_mu{mu:.1f}")} |')
    report+=['','The interaction estimates conditional support-strategy dependence, NOT a universal co-registration law.',
             'Each model generates potentially different physical grasps, so gains are not isolated candidate reranking.',
             'Scene resampling does not include training-seed variation or multiplicity adjustments.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    status.update({'status':'complete', 'checkpoint_sha256':{v:records[v]['ckpt_sha256'] for v in CELLS},
                   'source_files':audit, 'output':str(out)})
    (out/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=50000);p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--ci',type=float,default=.95);p.add_argument('--require-ready',action='store_true')
    a=p.parse_args();s=analyze(a.root,a.output_dir,a.bootstrap,a.seed,a.ci,a.require_ready)
    print(json.dumps({k:s[k] for k in ('status','cells','missing_files')},indent=2))

if __name__=='__main__':main()
