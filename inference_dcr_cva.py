#!/usr/bin/env python3
"""One forward pass -> local-score / Stage-1-score / anchored-score controls.

Same center decisions and physical actions for all three methods. No exact
labels/cache/CAD at inference. Old E1/E2 checkpoints are accepted as zero-
residual baselines, so no new training is needed to test output equivalence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from e1e2_common import (VERSION, atomic_file, case_key, digest, file_sha, get_batch,
    load_torch, lock, make_dataset, manifest, save_json, save_npz,
    schedule, seed_all, seed_for, shard_frames)
from dcr_cva_common import DCR_VERSION, make_outputs


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','stage1-checkpoint','checkpoint','output-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=('test_seen','test_similar','test_novel'),required=True)
    p.add_argument('--cases',default='nominal,bias:-20,bias:20')
    p.add_argument('--sample-interval',type=float,default=.1)
    p.add_argument('--query-limit',type=int,default=0)
    p.add_argument('--rank-strength',type=float,default=1.)
    p.add_argument('--shard-id',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    return p


def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    if not 0<=args.rank_strength<=1 or args.query_limit<0 or args.max_frames<0:
        raise ValueError('Invalid rank strength/query/frame count')
    ck=load_torch(args.checkpoint)
    if ck['version'] not in (VERSION,DCR_VERSION): raise RuntimeError('Unsupported checkpoint version')
    p=ck['protocol']; cfg=p['config']
    if p['reference_sha256']!=file_sha(args.stage1_checkpoint): raise RuntimeError('Wrong Stage-1 checkpoint')
    if args.sample_interval!=p['sample_interval']: raise RuntimeError('Training/inference schedule mismatch')
    cases=list(dict.fromkeys(['nominal']+[s.strip() for s in args.cases.split(',') if s.strip()]))
    device=torch.device(args.device); seed_all(cfg['seed'])
    from models.economicgrasp_cva_centers import load_reference, extract_depth_features
    from models.economicgrasp_cva_dcr import DecoupledCenterRankingCVA
    model=DecoupledCenterRankingCVA(load_reference(args.stage1_checkpoint,device),p['offsets_mm'],
        cfg['group_chunk'],cfg.get('rank_hidden',128),cfg.get('rank_bound',.5),cfg['seed']).to(device)
    if ck['version']==DCR_VERSION:
        model.load_learned_state(ck['model']); variant=p['variant']
    else:
        model.warm_start_corrector(ck['model']); variant=cfg['variant']+'_DCR_zero'
    model.eval()
    run={'version':DCR_VERSION,'variant':variant,'camera':p['camera'],
         'checkpoint_sha256':file_sha(args.checkpoint),'reference_sha256':p['reference_sha256'],
         'source_checkpoint_version':ck['version'],'sample_interval':args.sample_interval,
         'cases':cases,'offsets_mm':p['offsets_mm'],'query_limit':args.query_limit,
         'max_frames_per_split':args.max_frames,'query_chunk':cfg['query_chunk'],
         'rank_strength':args.rank_strength,'rank_bound':cfg.get('rank_bound',.5),
         'methods':['native','local','stage1','anchored'],
         'code_sha256':digest({n:file_sha(Path(__file__).resolve().parent/n) for n in
             ('inference_dcr_cva.py','dcr_cva_common.py','models/economicgrasp_cva_dcr.py','models/economicgrasp_cva_centers.py')})}
    del ck
    root=Path(args.output_root); manifest(root,run); signature=digest(run)
    frames=shard_frames(schedule(args.split,args.sample_interval),args.shard_id,args.num_shards,args.max_frames)
    if not frames:
        print('[DCR INFER] no frames assigned',flush=True); return
    ds,lookup=make_dataset(args.dataset_root,args.split,p['camera'])
    completed=skipped=0
    for sid,aid in frames:
        marker=root/'completed'/args.split/f'scene_{sid:04d}'/f'{aid:04d}.json'
        with lock(marker.with_suffix('.lock')):
            if args.resume and marker.exists():
                old=json.loads(marker.read_text())
                if old['signature']!=signature: raise RuntimeError('Changed inference contract')
                if all((root/n).is_file() for n in old['outputs']):
                    skipped+=1; continue
            elif marker.exists():
                raise FileExistsError(f'{marker}; use --resume')
            batch=get_batch(ds,lookup,sid,aid,device); paths=[]
            with torch.no_grad():
                pack=extract_depth_features(model.reference,batch)
                for case in cases:
                    logits,residual,bundle,_=model(batch,case=case,case_seed=seed_for(2030,sid,aid,case),
                        query_limit=args.query_limit,query_chunk=cfg['query_chunk'],depth_pack=pack)
                    outputs,sel=make_outputs(logits,residual,bundle,model.zero,args.rank_strength)
                    for mode,array in outputs.items():
                        rel=Path('dump')/mode/case_key(case)/f'scene_{sid:04d}'/p['camera']/f'{aid:04d}.npy'
                        with atomic_file(root/rel) as f: np.save(f,array.cpu().numpy(),allow_pickle=False)
                        paths.append(str(rel))
                    trace=Path('traces')/args.split/f'scene_{sid:04d}'/f'ann_{aid:04d}_{case_key(case)}.npz'
                    q=torch.arange(len(sel),device=device)
                    # Small trace supports score-only replay/diagnostics without
                    # another Stage-1/CVA forward pass or model retraining.
                    save_npz(root/trace, signature=np.array(signature),scene_id=np.array(sid),anno_id=np.array(aid),
                        case=np.array(case),selected=sel.cpu().numpy(),query_ids=bundle['query_ids'].cpu().numpy(),
                        offsets_mm=model.corrector.offsets_mm.cpu().numpy(),
                        local_utility=logits.sigmoid().mean(-1).cpu().numpy(),
                        selected_rank_residual=residual[sel,q].cpu().numpy(),
                        stage1_score=outputs['stage1'][:,0].cpu().numpy(),
                        anchored_score=outputs['anchored'][:,0].cpu().numpy())
                    paths.append(str(trace))
            save_json(marker,{'signature':signature,'scene_id':sid,'anno_id':aid,'outputs':paths})
            completed+=1
            del batch,pack,logits,residual,bundle,outputs
            print(f'[DCR INFER] {args.split} {sid}/{aid} new={completed} skip={skipped}',flush=True)
    save_json(root/f'infer_{args.split}_shard{args.shard_id}.json',
              {'signature':signature,'new':completed,'skipped':skipped,'scheduled':len(frames)})


if __name__=='__main__':
    main()
