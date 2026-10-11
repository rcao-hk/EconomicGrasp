#!/usr/bin/env python3
"""Compare historical outcome-selected P1 vs pre-registered Novel probe cohort.

Descriptive comparisons only: cohorts may overlap, are not independent, and
include strongly correlated gripper probes. This program does NOT compute
significance, training effects, or contact/collision labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def required(x, msg):
    if not x:
        raise ValueError(msg)


def load(path: Path):
    required(path.is_file(),f'Missing file {path}')
    if path.suffix=='.json':return json.loads(path.read_text())
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def scalar(x):
    return float(x) if x not in ('',None,'None') else None


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def compare(selected_dir: Path, uniform_dir: Path, out: Path, variant='volume_rel'):
    required(not out.exists() or not any(out.iterdir()),f'Nonempty result output: {out}')
    files={name:{tag:folder/name for tag,folder in [('selected',selected_dir),('uniform',uniform_dir)]}
           for name in ('P1_AUDIT.json','per_role.csv','ap_associations.csv')}
    old=load(files['P1_AUDIT.json']['selected'])
    new=load(files['P1_AUDIT.json']['uniform'])
    required(variant in old['variants'] and variant in new['variants'],f'Variant {variant} missing')
    sel_ids={(int(x['scene_id']),int(x['frame_id'])) for x in old['selected_frame_ids']}
    uni_ids={(int(x['scene_id']),int(x['frame_id'])) for x in new['selected_frame_ids']}
    required(len(sel_ids)==len(old['selected_frame_ids']) and len(uni_ids)==len(new['selected_frame_ids']),
             'Repeated frame identity in manifest')
    role_rows={tag:{x['role']:x for x in load(files['per_role.csv'][tag]) if x['variant']==variant}
               for tag in ('selected','uniform')}
    required(set(role_rows['selected'])==set(role_rows['uniform']), 'Different available gripper roles')
    metrics=('rendered_coverage','both_reference_coverage','paired_rendered_mae_mm',
             'paired_sensor_mae_mm','std_spearman_vs_error','scene_mean_within_frame_std_rho')
    rows=[]
    for role in sorted(role_rows['selected']):
        a=role_rows['selected'][role];b=role_rows['uniform'][role]
        for metric in metrics:
            x,y=scalar(a.get(metric)),scalar(b.get(metric))
            rows.append({'variant':variant,'role':role,'metric':metric,
                         'outcome_selected':x,'uniform':y,
                         'uniform_minus_selected':None if x is None or y is None else y-x,
                         'interpretation':'descriptive_only_not_independent'})
    links={tag:{(r['metric'],r['AP_outcome']):r for r in load(files['ap_associations.csv'][tag]) if r['variant']==variant}
           for tag in ('selected','uniform')}
    required(set(links['selected'])==set(links['uniform']), 'Different available AP association features')
    assoc=[]
    for key in sorted(links['selected']):
        a,b=links['selected'][key],links['uniform'][key]
        assoc.append({'variant':variant,'feature':key[0],'AP_outcome':key[1],
                      'selected_within_scene_pearson':scalar(a['within_scene_centered_pearson']),
                      'uniform_within_scene_pearson':scalar(b['within_scene_centered_pearson']),
                      'selected_pooled_spearman':scalar(a['pooled_spearman']),
                      'uniform_pooled_spearman':scalar(b['pooled_spearman']),
                      'interpretation':'descriptive_only_outcome_selection_bias'})
    out.mkdir(parents=True,exist_ok=True)
    for name,values in (('cohort_role_comparison.csv',rows),('cohort_AP_association.csv',assoc)):
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    manifest={'variant':variant,'selected_frames':len(sel_ids),'uniform_frames':len(uni_ids),
              'overlap_frames':len(sel_ids&uni_ids),'frame_overlap':sorted([list(x) for x in sel_ids&uni_ids]),
              'source_hashes':{name:{tag:sha256(path) for tag,path in refs.items()} for name,refs in files.items()},
              'limitations':['systematically sampled uniform frames not random IID',
                 'outcome-selected cohort biased by AP',
                 'cohorts may overlap; no significance test',
                 'within-scene frame/AP correlation observational, not causal']}
    (out/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=['# GVAR P1: outcome-selected vs AP-blind Novel diagnostic cohort','',
            f"Variant: {variant}. Old frames {len(sel_ids)}, new fixed-ID frames {len(uni_ids)}, overlap {len(sel_ids&uni_ids)}.",
            '', '| Feature | Outcome-selected frame-centered Pearson | AP-blind frame-centered Pearson |',
            '|---|---:|---:|']
    for row in assoc:
        if row['feature'] not in ('rendered_mae_mm','rendered_coverage','closing_rendered_coverage',
                                  'contact_rendered_coverage','mean_proxy_std_mm') or row['AP_outcome'] not in ('delta_AP_pp','delta_mu04_pp','delta_mu08_pp'):
            continue
        fmt=lambda x:'NA' if x is None else f'{x:+.3f}'
        report.append(f"| {row['feature']} vs {row['AP_outcome']} | {fmt(row['selected_within_scene_pearson'])} | {fmt(row['uniform_within_scene_pearson'])} |")
    report += ['', '**Interpretation limit:** this is a descriptive sensitivity check. Neither cohort supplies fixed-action exact grasp utility or a causal uncertainty effect.',
               'STD/gradient are predicted-depth proxies, not calibrated uncertainty. Probe validity is not confirmed free-space/contact visibility.', '']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected-dir',type=Path,required=True)
    p.add_argument('--uniform-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variant',default='volume_rel')
    args=p.parse_args(argv)
    try: out=compare(args.selected_dir,args.uniform_dir,args.output_dir,args.variant)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.exit(2,f'GVAR P1 cohort comparison FAILED: {exc}\n')
    print(json.dumps({k:out[k] for k in ('selected_frames','uniform_frames','overlap_frames')},indent=2))

if __name__=='__main__':main()
