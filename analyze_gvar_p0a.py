#!/usr/bin/env python3
"""P0-A: paired reader counterfactual AP, same frozen volume_rel checkpoint.

Test-time stress probes: R0 default, R1 fixed RGB content, R2 fixed
position metadata, R3 fixed validity mask, R4 all three fixed. Predicted
relative geometry remains dynamic for ALL modes. No network is re-trained.
These interventions can create out-of-distribution token combinations and
are NOT causal estimates of re-trained architecture improvements.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from analyze_gvar_scene_paired import SPLITS, METRIC_NAMES, scene_metric, paired_metric, boot_indices, canonical_scene_ids
from utils.gvar_p0_analysis import load, paired_protocol, fresh_output, require, write_csv

MODES=('r0','r1','r2','r3','r4')
# NumPy-only analysis contract, independent of the torch/reader import.
# Fail if an output claims a different mode from the switches actually run.
EXPECTED_SOURCES={
    'r0':['dynamic','dynamic','dynamic'],
    'r1':['fixed','dynamic','dynamic'],
    'r2':['dynamic','fixed','dynamic'],
    'r3':['dynamic','dynamic','fixed'],
    'r4':['fixed','fixed','fixed'],
}



def analyze(root:Path,out:Path,bootstrap=50000,seed=20261009,ci=.95):
    require(bootstrap>=200 and 0<ci<1,'Invalid bootstrap interval/draws')
    fresh_output(out)
    all_data={(m,s):load(root,'volume_rel','original',m,s) for m in MODES for s in SPLITS}
    canonical=all_data['r0','test_seen']
    for mode in MODES:
        for split in SPLITS:
            d=all_data[mode,split]
            require(d['protocol'].get('reader_cf_sources')==EXPECTED_SOURCES[mode],
                    f'Incorrect reader_cf_sources for {mode}/{split}')
            paired_protocol(canonical if split=='test_seen' else all_data['r0',split],d)
            # All modes must use one and only one frozen trained checkpoint.
            require(d['protocol']['gvar_config']==canonical['protocol']['gvar_config'],
                    f'Architecture config differs across counterfactual modes {mode}')
    idx=boot_indices(bootstrap,seed)
    rows=[]; scenes=[]
    for mode in MODES:
        for metric in METRIC_NAMES:
            deltas={s:scene_metric(all_data[mode,s]['ap'],metric)-scene_metric(all_data['r0',s]['ap'],metric)
                    for s in SPLITS}
            for split in (*SPLITS,'Mean'):
                if split=='Mean':
                    current=sum(float(scene_metric(all_data[mode,s]['ap'],metric).mean()) for s in SPLITS)/3*100
                    ref=sum(float(scene_metric(all_data['r0',s]['ap'],metric).mean()) for s in SPLITS)/3*100
                else:
                    current=float(scene_metric(all_data[mode,split]['ap'],metric).mean())*100
                    ref=float(scene_metric(all_data['r0',split]['ap'],metric).mean())*100
                rows.append({'counterfactual':mode,'reference':'r0','split':split,'metric':metric,
                             'reference_pct':ref,'counterfactual_pct':current,
                             **paired_metric(deltas,idx,ci,split)})
        for s in SPLITS:
            vec=scene_metric(all_data[mode,s]['ap'],'AP')-scene_metric(all_data['r0',s]['ap'],'AP')
            for scene,delta in zip(canonical_scene_ids(s),vec):
                scenes.append({'mode':mode,'split':s,'scene_id':scene,'delta_AP_pp':float(delta*100)})
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'reader_cf_effects.csv',rows)
    write_csv(out/'reader_cf_scenes.csv',scenes)
    report=['# GVAR P0-A frozen-reader counterfactual AP','',
            'One **volume_rel e19 checkpoint**; identical Original GraspNet RGB, sensor crop and sensor-collision protocol.',
            'R1 fixed feature / dynamic metadata & mask; R2 dynamic feature / fixed metadata / dynamic mask;',
            'R3 dynamic feature & metadata / fixed mask; R4 fixed feature, metadata & mask.',
            'Predicted geometry stays dynamic. This is test-time OOD perturbation, NOT re-trained factorial.',
            f'{bootstrap:,} scene-stratified paired bootstrap draws, seed={seed}, CI={ci:.0%}.', '',
            '| Mode | Seen Δ [CI] | Similar Δ [CI] | Novel Δ [CI] | Mean Δ [CI] |',
            '|---|---|---|---|---|']
    for mode in MODES:
        vals=[]
        for s in (*SPLITS,'Mean'):
            r=next(x for x in rows if x['counterfactual']==mode and x['split']==s and x['metric']=='AP')
            vals.append(f"{r['delta_pp']:+.3f} [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}]")
        report.append('| '+ ' | '.join((mode,*vals)) + ' |')
    report+=['','## Novel friction-specific ΔAP against R0','','| Mode | μ=0.4 [CI] | μ=0.8 [CI] |',
             '|---|---|---|']
    for mode in MODES:
        outvals=[]
        for mu in (.4,.8):
            r=next(x for x in rows if x['counterfactual']==mode and x['split']=='test_novel' and x['metric']==f'AP_mu{mu:.1f}')
            outvals.append(f"{r['delta_pp']:+.3f} [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}]")
        report.append('| '+ ' | '.join((mode,*outvals))+' |')
    report+=['','**Caveats:** different inference-time interventions change candidate sets, score distribution and possibly collision survival.',
             'An R1–R4 degradation shows sensitivity of the frozen model and is not proof of a causal training-time component.',
             'The scene CI does not cover training seeds or multiplicity. The complete counterfactual modes are not independent experiments.',
             'See raw effects and audit for prefix@1/5/10/20/50 and all μ.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    audit={'modes':MODES,'split_names':SPLITS,'bootstrap':bootstrap,'seed':seed,'ci':ci,
           'checkpoint_sha256':canonical['protocol']['checkpoint_sha256'],
           'sampling_protocol':canonical['protocol']['sample_policy'],
           'source_files':[p for entry in all_data.values() for p in entry['sources']]}
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    return {'modes':len(MODES),'rows':len(rows),'scenes':len(scenes),'output':str(out)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=50000)
    p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--ci',type=float,default=.95)
    a=p.parse_args();print(json.dumps(analyze(a.root,a.output_dir,a.bootstrap,a.seed,a.ci),indent=2))

if __name__=='__main__':main()
