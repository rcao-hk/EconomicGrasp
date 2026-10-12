#!/usr/bin/env python3
"""P0-C matched Original-versus-GN-Trans RGB performance and material domain gap.

All evaluations MUST use original sensor crop, camera metadata, sampled image
indices and collision cloud. Only RGB image content is switched to GN-Trans.
Every checkpoint is the same between its Original and GN-Trans pair. This is
NOT legacy GN-Trans virtual-depth preprocessing and not a raw-RGB-only system.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from analyze_gvar_scene_paired import SPLITS, METRIC_NAMES, scene_metric, paired_metric, boot_indices, canonical_scene_ids
from utils.gvar_p0_analysis import (load,paired_protocol,same_pair_except_checkpoint,
                                    fresh_output,require,write_csv)

VARIANTS=('baseline','volume_fixed','volume_rel')
DOMAINS=('original','gntrans')


def analyze(root:Path,out:Path,bootstrap=50000,seed=20261009,ci=.95):
    require(bootstrap>=200 and 0<ci<1,'Invalid bootstrap arguments')
    fresh_output(out)
    data={(v,d,s):load(root,v,d,'r0',s) for v in VARIANTS for d in DOMAINS for s in SPLITS}
    for variant in VARIANTS:
        first=data[variant,'original','test_seen']
        for split in SPLITS:
            a=data[variant,'original',split];b=data[variant,'gntrans',split]
            paired_protocol(a,b,compare_domain=True)
            require(a['protocol']['checkpoint_sha256']==first['protocol']['checkpoint_sha256'],
                    f'Different checkpoint across {variant} splits')
            require(b['protocol']['gvar_config']==a['protocol']['gvar_config'],
                    f'Architecture mismatch Original vs GN-Trans {variant}')
            require(b['protocol']['gntrans_rgb_root'] is not None and
                    b['protocol']['image_source']=='gntrans_rendered_rgb' and
                    a['protocol']['image_source']=='original_rgb',
                    f'Input domains incorrectly configured {variant}/{split}')
    for domain in DOMAINS:
        for split in SPLITS:
            for variant in VARIANTS[1:]:
                same_pair_except_checkpoint(data['baseline',domain,split],data[variant,domain,split])

    index=boot_indices(bootstrap,seed)
    results=[];scenes=[]
    for variant in VARIANTS:
        for metric in METRIC_NAMES:
            values={domain:{s:scene_metric(data[variant,domain,s]['ap'],metric) for s in SPLITS}
                    for domain in DOMAINS}
            baseline={domain:{s:scene_metric(data['baseline',domain,s]['ap'],metric) for s in SPLITS}
                      for domain in DOMAINS}
            exprs={
                'material_shift_gntrans_minus_original':{
                    s:values['gntrans'][s]-values['original'][s] for s in SPLITS},
                'improvement_vs_baseline_original':{
                    s:values['original'][s]-baseline['original'][s] for s in SPLITS},
                'improvement_vs_baseline_gntrans':{
                    s:values['gntrans'][s]-baseline['gntrans'][s] for s in SPLITS},
                # Positive DoD means the relative GN-Trans penalty became less
                # negative, though the *absolute* GN-Trans gap can still grow.
                'material_gap_change_vs_baseline':{
                    s:(values['gntrans'][s]-values['original'][s])-
                      (baseline['gntrans'][s]-baseline['original'][s]) for s in SPLITS},
            }
            for contrast,diffs in exprs.items():
                for split in (*SPLITS,'Mean'):
                    results.append({'variant':variant,'split':split,'metric':metric,'contrast':contrast,
                                    **paired_metric(diffs,index,ci,split)})
        for s in SPLITS:
            a=scene_metric(data[variant,'original',s]['ap'],'AP')
            b=scene_metric(data[variant,'gntrans',s]['ap'],'AP')
            ao=scene_metric(data['baseline','original',s]['ap'],'AP')
            bo=scene_metric(data['baseline','gntrans',s]['ap'],'AP')
            for scene,orig,trans,orig_b,trans_b in zip(canonical_scene_ids(s),a,b,ao,bo):
                scenes.append({'variant':variant,'split':s,'scene_id':scene,
                               'original_AP':float(orig*100),'gntrans_AP':float(trans*100),
                               'material_gap_pp':float((trans-orig)*100),
                               'original_adv_vs_baseline_pp':float((orig-orig_b)*100),
                               'gntrans_adv_vs_baseline_pp':float((trans-trans_b)*100),
                               'gap_change_vs_baseline_pp':float(((trans-orig)-(trans_b-orig_b))*100)})
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'domain_contrasts.csv',results)
    write_csv(out/'domain_scenes.csv',scenes)
    report=['# GVAR P0-C paired material-domain evaluation','',
            'One checkpoint per variant; Original and GN-Trans share **Original sensor depth crop, K, pixel sample, sensor collision**.',
            'GN-Trans replaces RGB ONLY. Its virtual depth is never consumed for cropping, network geometry or collision.',
            'This does not reproduce the legacy GN-Trans virtual-crop protocol; it isolates the input RGB shift.',
            f'{bootstrap:,} scene-stratified bootstrap draws, seed {seed}, CI={ci:.0%}.', '',
            '## Full Original and GN-Trans AP (%)', '',
            '| Variant | Original Seen/Similar/Novel/Mean | GN-Trans Seen/Similar/Novel/Mean | Material gap Mean [CI] (pp) |',
            '|---|---|---|---|']
    def lookup(v,s,c,m='AP'):
        matches=[r for r in results if r['variant']==v and r['split']==s and r['contrast']==c and r['metric']==m]
        assert len(matches)==1
        return matches[0]
    for v in VARIANTS:
        aps={domain:[100*float(scene_metric(data[v,domain,s]['ap'],'AP').mean()) for s in SPLITS]
             for domain in DOMAINS}
        for domain in DOMAINS:
            aps[domain].append(float(np.mean(aps[domain])))
        g=lookup(v,'Mean','material_shift_gntrans_minus_original')
        report.append(f'| {v} | '+ ' / '.join(f'{x:.3f}' for x in aps['original'])+' | '
                     + ' / '.join(f'{x:.3f}' for x in aps['gntrans'])+
                     f" | {g['delta_pp']:+.3f} [{g['ci_low_pp']:+.3f}, {g['ci_high_pp']:+.3f}] |")
    report+=['', '## GVAR advantage over baseline, in-domain and under appearance shift', '',
              '| Variant | Split | Original Δ [CI] | GN-Trans Δ [CI] | Material-gap change [CI] |',
              '|---|---|---|---|---|']
    for v in VARIANTS[1:]:
        for s in (*SPLITS,'Mean'):
            def fmt(c):
                r=lookup(v,s,c)
                return f"{r['delta_pp']:+.3f} [{r['ci_low_pp']:+.3f},{r['ci_high_pp']:+.3f}]"
            report.append('| '+ ' | '.join((v,s,fmt('improvement_vs_baseline_original'),
                          fmt('improvement_vs_baseline_gntrans'),fmt('material_gap_change_vs_baseline')))+' |')
    report+=['','## Novel friction thresholds', '', '| Variant | μ | Material gap [CI] | Gap-change vs baseline [CI] |',
              '|---|---|---|---|']
    for v in VARIANTS:
        for mu in (.4,.8):
            def fmt(c):
                r=lookup(v,'test_novel',c,f'AP_mu{mu:.1f}')
                return f"{r['delta_pp']:+.3f} [{r['ci_low_pp']:+.3f},{r['ci_high_pp']:+.3f}]"
            report.append(f"| {v} | {mu} | {fmt('material_shift_gntrans_minus_original')} | "
                          f"{fmt('material_gap_change_vs_baseline')} |")
    report+=['','**Interpretation:** A positive gap-change is the difference-in-differences under a matched-image intervention.',
             'It does NOT show an automatic increase in absolute GN-Trans AP, robust grasp quality at μ=0.4, or multi-seed stability.',
             'Every independently trained model produces a different physical grasp set; rank/AP differences are not pure reranking.',
             'The whole pipeline remains depth-assisted by original sensor preprocessing and collision checking.',
             'No multiple-comparison adjustment or training-seed uncertainty is included.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    audit={'variants':VARIANTS,'domains':DOMAINS,'splits':SPLITS,'bootstrap':bootstrap,
           'seed':seed,'ci':ci,'crop_and_collision':'original_sensor',
           'paired_rgb_only_change':True,
           'per_variant_checkpoint_sha256':{v:data[v,'original','test_seen']['protocol']['checkpoint_sha256'] for v in VARIANTS},
           'source_files':[p for rec in data.values() for p in rec['sources']]}
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    return {'variants':len(VARIANTS),'input_ap_arrays':len(data),'rows':len(results),'output':str(out)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=50000)
    p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--ci',type=float,default=.95)
    a=p.parse_args();print(json.dumps(analyze(a.root,a.output_dir,a.bootstrap,a.seed,a.ci),indent=2))

if __name__=='__main__':main()
