#!/usr/bin/env python3
"""Nominal fixed-action test of a RETRAINED B0 control or original B0 checkpoint.

Seen evaluation is included and labelled validation_seen. Reports original
validation margin AND margin=0; never retunes on evaluation frames.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from rep_a_common import (SPLITS, array_sha, check_runtime_sources, digest, exclusive_run, file_sha,
                         list_frames, read_frame, save_json)
from rep_followup_common import (aggregate_decisions, checked_manifest, decision_metrics,
                                load_scorer, model_inputs, write_csv, source_digest)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--name',required=True)
    p.add_argument('--split',choices=('test_seen','test_similar','test_novel'),required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); sys.argv=[sys.argv[0]]
    check_runtime_sources(args.cache_root)
    paths=list_frames(args.cache_root,args.split,args.max_frames)
    protocol=dict(name=args.name,split=args.split,checkpoint_sha=file_sha(args.checkpoint),
        frame_contract=digest([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]),
        code=source_digest(('rep_b0_controls.py','rep_followup_common.py','test_rep_b0_controls.py')),
        seen_role='validation_seen' if args.split=='test_seen' else 'held_out',policies=['fixed_0','val_selected'])
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with exclusive_run(out/'.test.lock'):
        checked_manifest(out,protocol)
        if args.resume and (out/'summary.json').exists() and (out/'per_frame.csv').exists():
            print(f'[B0 ATTR] already complete: {out}'); return
        model,info=load_scorer(args.checkpoint,args.cache_root,torch.device(args.device))
        if info['variant'] not in ('B0','full','no_image','independent_rgb'):
            raise ValueError('Not a B0 attribution checkpoint')
        rows=[]
        for i,path in enumerate(paths):
            d=read_frame(path,info['contract']); t=model_inputs(d,args.device)
            prob=model(t).sigmoid().cpu().numpy()
            for policy,margin in (('fixed_0',0.),('val_selected',info['margin'])):
                met,_=decision_metrics(prob,d,margin)
                rows.append(dict(name=args.name,split=args.split,policy=policy,margin=margin,
                    scene_id=int(d['scene_id']),anno_id=int(d['anno_id']),
                    action_sha=array_sha(d['actions'],d['valid'],d['friction'],d['query_ids']),**met))
            if (i+1)%100==0: print(f'[B0 ATTR {args.name}] {i+1}/{len(paths)}',flush=True)
        write_csv(out/'per_frame.csv',rows)
        save_json(out/'summary.json',dict(protocol=protocol,checkpoint=info,
            policies={s:aggregate_decisions([r for r in rows if r['policy']==s])
                      for s in ('fixed_0','val_selected')}))


if __name__=='__main__': main()
