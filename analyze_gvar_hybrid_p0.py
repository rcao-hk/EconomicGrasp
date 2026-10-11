#!/usr/bin/env python3
"""P0 hybrid evidence-site audit: fixed RGB/dynamic geometry intermediate.

Uses three independently trained checkpoints (fixed/fixed, fixedRGB/dynGeom,
dynRGB/dynGeom). Reuses canonical GVAR e19 scene-paired protocol and bootstrap.
It does NOT evaluate fixed physical candidate grasps or claim a full factorial.
It never trains, infers or runs the GraspNet official evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from analyze_gvar_p0_decision import audit_training_log
from analyze_gvar_scene_paired import (
    AuditError, SPLITS, boot_indices, load_all, paired_metric, run as paired_run,
    scene_metric, METRIC_NAMES, sha256_file,
)

REFERENCE = 'volume_fixed_rel'
HYBRID = 'volume_hybrid'
DYNAMIC = 'volume_rel'
MODELS = (REFERENCE, HYBRID, DYNAMIC)


def require(x: bool, msg: str):
    if not x:
        raise AuditError(msg)


def pending_files(root: Path, variant: str) -> list[str]:
    """New experiment: avoid accidentally selecting partial smoke/e20 outputs."""
    training = root / 'train' / variant
    files = [training / 'gvar_protocol.json', training / 'gntrans_mix_protocol.json',
             training / 'gvar_epochs.jsonl', training / 'checkpoint_epoch_019.tar',
             training / 'checkpoint_epoch_020.tar']
    for split in SPLITS:
        where = root / 'eval' / f'{variant}_e19' / split
        files.extend((where / 'gvar_inference_protocol.json',
                      where / 'gvar_inference_summary.json',
                      where / f'ap_{split}_realsense.npy'))
    return [str(p) for p in files if not p.is_file()]


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def evaluate(root: Path, output: Path, bootstrap=50000, seed=20261009,
             ci=.95, require_ready=False) -> dict:
    require(root.is_dir(), f'Missing GVAR results root: {root}')
    require(bootstrap >= 200 and 0 < ci < 1, 'Invalid bootstrap settings')
    require(output.resolve() != root.resolve(), 'Output must not equal experiment root')
    require(not any(output.resolve().is_relative_to((root/x).resolve()) for x in ('train','eval')),
            'Output must not be under train/ or eval/')
    require(not output.exists() or not any(output.iterdir()),
            f'Refusing to overwrite nonempty output {output}')
    missing = {v: pending_files(root,v) for v in MODELS}
    invalid_controls = {v: x for v,x in missing.items() if x and v != HYBRID}
    require(not invalid_controls,
            f'Missing historical fixed_rel/volume_rel control evidence: {invalid_controls}')
    state = {'version': 1, 'models': MODELS, 'root': str(root.resolve()),
             'bootstrap': bootstrap, 'seed': seed, 'ci': ci,
             'status': 'pending' if missing[HYBRID] else 'ready',
             'missing_files': {v:x for v,x in missing.items() if x},
             'note': 'Three-point sequential path, not a fully crossed 2x2 factorial.'}
    if missing[HYBRID]:
        output.mkdir(parents=True, exist_ok=True)
        (output/'STATUS.json').write_text(json.dumps(state, indent=2)+'\n')
        (output/'REPORT.md').write_text('# P0 Hybrid pending\n\nHybrid e19 AP not available.\n\n'
                                         + '\n'.join('- '+s for s in missing[HYBRID])+'\n')
        if require_ready:
            raise AuditError(f'Hybrid incomplete. Status saved under {output}')
        return state

    records, audit = load_all(root, list(MODELS), epoch=19)
    training = [audit_training_log(root,v) for v in MODELS]  # all e0..e20, finite
    # Source identity must match the *actual* archived e19 checkpoint, not
    # merely three mutually consistent inference JSON files.
    for v in MODELS:
        ckpt = root / 'train' / v / 'checkpoint_epoch_019.tar'
        digest = sha256_file(ckpt)
        require(digest == records[v]['ckpt_sha256'],
                f'Official inference checkpoint SHA mismatch for {v}: {ckpt}')
    for v, expected in {'volume_fixed_rel':44.1730,'volume_rel':45.5176}.items():
        actual = sum(records[v]['data'][sp].tensor.mean(dtype='float64') for sp in SPLITS) / 3 * 100
        require(abs(actual - expected) <= .02,
                f'Historic {v} AP drift ({actual:.4f} vs expected {expected:.4f}): check protocol')
    output.mkdir(parents=True, exist_ok=True)
    paired_run(root, output/'paired', list(MODELS), REFERENCE, epoch=19,
               bootstrap=bootstrap,seed=seed,ci=ci,
               requested_contrasts=[f'{HYBRID}:{DYNAMIC}'])
    index = boot_indices(bootstrap,seed)
    table=[]
    for name in METRIC_NAMES:
        values = {v:{s:scene_metric(records[v]['data'][s].tensor,name) for s in SPLITS}
                  for v in MODELS}
        for split in (*SPLITS,'Mean'):
            def delta(v,w):
                return {s:values[v][s]-values[w][s]
                        for s in (SPLITS if split=='Mean' else (split,))}
            # Path effects only; not a claim of independent main effects.
            for label, v,w in (
                ('dynamic_geometry_given_fixed_rgb',HYBRID,REFERENCE),
                ('dynamic_rgb_given_dynamic_geometry',DYNAMIC,HYBRID),
                ('whole_path_dynamic_vs_fixed',DYNAMIC,REFERENCE),
            ):
                stat=paired_metric(delta(v,w),index,ci,split)
                table.append({'split':split,'metric':name,'contrast':label,'variant':v,'reference':w,**stat})
    p=output/'hybrid_path_effects.csv'
    with p.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    def look(s,c,m='AP'):
        x=[r for r in table if r['split']==s and r['contrast']==c and r['metric']==m]
        require(len(x)==1, f'Missing row {s}/{c}/{m}')
        return x[0]
    report=['# GVAR P0 Hybrid: split visual and geometric support','',
        'Physical support semantics: fixed_rgb/fixed_geometry → fixed_rgb/dynamic_geometry → dynamic_rgb/dynamic_geometry.',
        'Compare independently trained variants; scene-paired AP does not isolate candidate quality vs ranking.',
        f'Bootstrap={bootstrap} scene draws, CI={ci:.0%}; training seed coverage=1.',
        '', '| Contrast (pp [CI]) | Seen | Similar | Novel | Mean |',
        '|---|---|---|---|---|']
    for label in ('dynamic_geometry_given_fixed_rgb','dynamic_rgb_given_dynamic_geometry','whole_path_dynamic_vs_fixed'):
        cols=[]
        for s in (*SPLITS,'Mean'):
            r=look(s,label)
            cols.append(f"{r['delta_pp']:+.3f} [{r['ci_low_pp']:+.3f},{r['ci_high_pp']:+.3f}]")
        report.append('| '+ ' | '.join((label,*cols)) + ' |')
    report+=['','## Novel stricter/looser friction', '', '| Contrast | μ=0.4 | μ=0.8 |', '|---|---|---|']
    for label in ('dynamic_geometry_given_fixed_rgb','dynamic_rgb_given_dynamic_geometry'):
        cols=[]
        for mu in (0.4,0.8):
            r=look('test_novel',label,f'AP_mu{mu:.1f}')
            cols.append(f"{r['delta_pp']:+.3f} [{r['ci_low_pp']:+.3f},{r['ci_high_pp']:+.3f}]")
        report.append('| '+ ' | '.join((label,*cols))+' |')
    report+=['', '## Limits',
            '- Evidence assignment differs in support coordinates and their associated spatial metadata.',
            '- Three sequential configurations are not an independently crossed RGB×geometry 2×2 factorial.',
            '- A CI overlapping zero is not equivalence; CIs do not reflect training seeds or multiplicity.',
            '- A nominal geometry correspondence gain does not imply Novel μ=0.4 contact-quality gain.',
            '- Compare reader FLOPs/latency and ensure checkpoint, source and sensor-collision parity.',
            '']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    state.update(status='complete',train_audit=training,
                 checkpoint_sha256={v:records[v]['ckpt_sha256'] for v in MODELS},
                 input_audit=audit, output=str(output.resolve()))
    (output/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    return state


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=50000)
    p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--ci',type=float,default=.95)
    p.add_argument('--require-ready',action='store_true')
    args=p.parse_args(argv)
    try:
        state=evaluate(args.root,args.output_dir,args.bootstrap,args.seed,args.ci,args.require_ready)
    except (AuditError,KeyError,ValueError,OSError,TypeError) as exc:
        p.exit(2,f'GVAR hybrid P0 FAILED: {exc}\n')
    print(json.dumps({k:state[k] for k in ('status','models','missing_files')},indent=2))

if __name__=='__main__':main()
