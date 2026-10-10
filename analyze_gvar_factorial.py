#!/usr/bin/env python3
"""Complete GVAR 2x2 factorial contrasts, with scene-cluster paired intervals.

Cells: fixed/no-rel, dynamic/no-rel, fixed/relative, dynamic/relative.
Interaction = (dynamic_rel - dynamic) - (fixed_rel - fixed).
Uses the same original GraspNet e19 AP arrays; no training/inference.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_gvar_scene_paired import (load_all, boot_indices, paired_metric, scene_metric,
                                        METRIC_NAMES, SPLITS, canonical_scene_ids, AuditError)

CELLS=('volume_fixed','volume','volume_fixed_rel','volume_rel')

def run(root:Path,out:Path,epoch:int=19,bootstrap:int=50000,seed:int=20261009,ci:float=.95):
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Choose empty analysis output: {out}')
    records,audit=load_all(root,list(CELLS),epoch)
    idx=boot_indices(bootstrap,seed)
    table=[];scene_rows=[]
    for metric in METRIC_NAMES:
        for split in (*SPLITS,'Mean'):
            def cell(c):
                return ({s:scene_metric(records[c]['data'][s].tensor,metric) for s in SPLITS}
                        if split=='Mean' else {split:scene_metric(records[c]['data'][split].tensor,metric)})
            fixed, dynamic, fixed_rel, dynamic_rel=(cell(c) for c in CELLS)
            for term in ('fixed_rel_effect','dynamic_rel_effect','dynamic_effect_no_rel',
                         'dynamic_effect_with_rel','geometry_mean_effect','dynamic_mean_effect','interaction'):
                def expr(s):
                    f,d,fr,dr=(x[s] for x in (fixed,dynamic,fixed_rel,dynamic_rel))
                    return {
                        'fixed_rel_effect':fr-f,
                        'dynamic_rel_effect':dr-d,
                        'dynamic_effect_no_rel':d-f,
                        'dynamic_effect_with_rel':dr-fr,
                        'geometry_mean_effect':((fr-f)+(dr-d))/2,
                        'dynamic_mean_effect':((d-f)+(dr-fr))/2,
                        'interaction':(dr-d)-(fr-f),
                    }[term]
                vectors={s:expr(s) for s in (SPLITS if split=='Mean' else (split,))}
                result=paired_metric(vectors,idx,ci,split)
                table.append({'metric':metric,'split':split,'contrast':term,**result})
                if metric=='AP' and split!='Mean':
                    scene_rows.extend({'contrast':term,'split':split,'scene_id':sid,'delta_pp':float(x*100)}
                                      for sid,x in zip(canonical_scene_ids(split),vectors[split]))
    out.mkdir(parents=True)
    def save(name,rows):
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('factorial_effects.csv',table);save('factorial_scene_effects.csv',scene_rows)
    overview=[x for x in table if x['metric']=='AP']
    report=['# GVAR 2x2 factorial action-support analysis','',
            'Fixed/no-rel = volume_fixed; dynamic/no-rel = volume; fixed/rel = volume_fixed_rel; dynamic/rel = volume_rel.',
            'Effects in AP percentage points (pp). Scene-paired bootstrap; all models independently generated candidate actions.',
            f'Bootstrap={bootstrap}, seed={seed}, CI={ci:.0%}. This is NOT across-training-seed uncertainty.',
            '', '| Contrast | Seen Δ [CI] | Similar Δ [CI] | Novel Δ [CI] | Mean Δ [CI] |',
            '|---|---|---|---|---|']
    for term in ('fixed_rel_effect','dynamic_rel_effect','dynamic_effect_no_rel',
                 'dynamic_effect_with_rel','geometry_mean_effect','dynamic_mean_effect','interaction'):
        def format(s):
            v=next(row for row in overview if row['contrast']==term and row['split']==s)
            return f"{v['delta_pp']:+.3f} [{v['ci_low_pp']:+.3f}, {v['ci_high_pp']:+.3f}]"
        report.append('| '+ ' | '.join([term,*(format(s) for s in (*SPLITS,'Mean'))])+' |')
    report+=['','**Interpretation:** The interaction isolates whether geometry benefit changes when probe sampling is dynamic rather than fixed.',
             'An interaction CI containing zero does not support a geometry×sampling synergy. Check Novel μ=0.4/0.8 separately.',
             'All CIs resample scenes only; one seed per cell and multiple tested contrasts prevent population-level or training-variance claims.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'factorial_manifest.json').write_text(json.dumps({'cells':CELLS,'epoch':epoch,'bootstrap':bootstrap,
      'seed':seed,'ci':ci,'input_files':audit,'root':str(root.resolve())},indent=2)+'\n')
    return {'rows':len(table),'scene_rows':len(scene_rows),'output':str(out)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epoch',type=int,default=19);p.add_argument('--bootstrap',type=int,default=50000)
    p.add_argument('--seed',type=int,default=20261009);p.add_argument('--ci',type=float,default=.95)
    a=p.parse_args();print(json.dumps(run(a.root,a.output_dir,a.epoch,a.bootstrap,a.seed,a.ci),indent=2))

if __name__=='__main__':main()
