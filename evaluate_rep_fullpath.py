#!/usr/bin/env python3
"""Fresh exact CAD/DexNet labels for full-path outputs, including Seen.

No labels from Rep-P0/Rep-A are accepted. Reuses only identical physical actions
within the SAME scene/frame. Evaluator retains at most one scene in memory.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from rep_a_common import (SPLITS, array_sha, cdf_targets, digest, exclusive_run, file_sha, save_npz)
from rep_followup_common import decision_metrics, memory_status, release_memory, source_digest


def physical_key(g):
    # Score and object_id are outputs/metadata, not physical action coordinates.
    return np.asarray(g[1:16],dtype=np.float32).tobytes()


def label_actions(evaluator, sid, aid, actions, valid, memo, chunk=128, min_free=0):
    missing={}
    for g in actions[valid]:
        key=physical_key(g)
        if key not in memo: missing.setdefault(key,g)
    keys=list(missing)
    for start in range(0,len(keys),chunk):
        memory_status(min_free)
        kk=keys[start:start+chunk]
        result=evaluator.evaluate(sid,aid,np.stack([missing[k] for k in kk]))
        for i,key in enumerate(kk):
            f=float(result.friction[i])
            if not np.isfinite(f): raise FloatingPointError('Evaluator returned an unlabelled valid action')
            memo[key]=(f,bool(result.collision_or_empty[i]),bool(result.pure_collision[i]),bool(result.empty[i]))
    friction=np.full(valid.shape,-1.,np.float32)
    collision=np.zeros(valid.shape,bool); pure=collision.copy(); empty=collision.copy()
    for k,q in zip(*np.nonzero(valid)):
        friction[k,q],collision[k,q],pure[k,q],empty[k,q]=memo[physical_key(actions[k,q])]
    return friction,collision,pure,empty,len(keys)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True); p.add_argument('--work-root',required=True)
    p.add_argument('--split',choices=tuple(SPLITS)[1:],required=True)
    p.add_argument('--shard-id',type=int,default=0); p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--eval-chunk',type=int,default=128); p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--fc-mode',choices=('official','reuse_contacts'),default='reuse_contacts')
    p.add_argument('--label-scope',choices=('selected','all'),default='selected',help='selected evaluates the union of executed actions plus native; all additionally computes oracle/ranking diagnostics')
    p.add_argument('--verify-n',type=int,default=0); p.add_argument('--min-host-free-gib',type=float,default=4.)
    p.add_argument('--resume',action='store_true'); p.add_argument('--repair-corrupt',action='store_true')
    args=p.parse_args(); sys.argv=[sys.argv[0]]
    if not 0<=args.shard_id<args.num_shards or args.eval_chunk<1: raise ValueError('Invalid worker/chunk')
    root=Path(args.work_root); protocol=json.loads((root/'protocol.json').read_text())
    files=sorted((root/'inference'/args.split).glob('scene_*/ann_*.npz'))
    scenes=sorted({int(p.parent.name.split('_')[-1]) for p in files})
    if not scenes: raise FileNotFoundError('No inference outputs')
    lo,hi=SPLITS[args.split]
    if any(not lo<=s<hi for s in scenes): raise RuntimeError('Split leakage in fullpath outputs')
    owned=set(scenes[args.shard_id::args.num_shards])
    frames={}
    for path in files:
        if int(path.parent.name.split('_')[-1]) in owned:
            frame=(path.parent.name,path.name[:8]); frames.setdefault(frame,[]).append(path)
    items=list(frames.items())
    if args.max_frames>0: items=items[:args.max_frames]
    # Lazy heavy import: NO CAD objects or models in the inference phase.
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    evaluator=ExactGraspNetActionEvaluator(args.dataset_root,protocol['camera'],split=args.split,
        collision_chunk=args.eval_chunk,fc_mode=args.fc_mode,verify_n=args.verify_n,strict=True)
    code=source_digest(('evaluate_rep_fullpath.py','exact_action_graspnet_evaluator.py'))
    current=None
    try:
        for (scene,frame),paths in items:
            sid=int(scene.split('_')[-1]); aid=int(frame.split('_')[-1])
            if sid!=current:
                evaluator.scene_cache.clear(); release_memory(); current=sid
            memo={}
            for path in paths:
                out=root/'evaluation'/args.split/scene/path.name
                out.parent.mkdir(parents=True,exist_ok=True)
                signature=digest(dict(inference_sha=file_sha(path),code=code,fc_mode=args.fc_mode,verify_n=args.verify_n,label_scope=args.label_scope))
                with exclusive_run(out.with_suffix('.lock')):
                    if out.exists() and args.resume:
                        try:
                            with np.load(out,allow_pickle=False) as z:
                                saved=str(z['signature']); _=json.loads(str(z['rows_json']))
                        except Exception:
                            if not args.repair_corrupt: raise
                            saved=None
                        if saved is not None:
                            if saved!=signature: raise RuntimeError(f'Changed exact-evaluation contract: {out}')
                            continue
                    elif out.exists(): raise FileExistsError(out)
                    with np.load(path,allow_pickle=False) as f: d={k:f[k] for k in f.files}
                    if str(d['signature'])!=digest(protocol): raise RuntimeError('Mixed inference protocols')
                    a=d['actions']; v=d['valid'].astype(bool)
                    if not np.isfinite(a[v]).all(): raise ValueError('Non-finite valid action')
                    label_mask=v.copy()
                    if args.label_scope=='selected':
                        label_mask[:]=False
                        label_mask[int(d['zero_index'])]=True
                        for picked in d['selected']: label_mask[picked,np.arange(a.shape[1])]=True
                        label_mask &= v
                    fr,col,pure,empty,n_new=label_actions(evaluator,sid,aid,a,label_mask,memo,args.eval_chunk,args.min_host_free_gib)
                    fr[~label_mask]=np.nan
                    y=cdf_targets(fr); utility=y.mean(-1)
                    labels=dict(valid=v,friction=fr,utility=utility,zero_index=d['zero_index'],offsets_mm=d['offsets_mm'])
                    zero=int(d['zero_index']); qq=np.arange(a.shape[1]); rows=[]
                    for i,(method,policy) in enumerate(zip(d['output_methods'],d['output_policies'])):
                        method,policy=str(method),str(policy); sel=d['selected'][i]
                        if not v[sel,qq].all(): raise RuntimeError('Inference selected invalid hypotheses')
                        us=utility[sel,qq]; ss=y[sel,qq,3]; ns=y[zero,:,3]
                        row=dict(method=method,policy=policy,split=args.split,scene_id=sid,anno_id=aid,
                            mode=str(d['mode']),case=str(d['case']),num_queries=len(qq),num_actions=int(v.sum()),
                            selected_utility=float(us.mean()),native_utility=float(utility[zero].mean()),
                            oracle_utility=float(np.where(v,utility,-np.inf).max(0).mean()) if args.label_scope=='all' else None,
                            label_scope=args.label_scope,
                            utility_gain=float((us-utility[zero]).mean()),success08=float(ss.mean()),
                            native_success08=float(ns.mean()),success08_gain=float((ss-ns).mean()),
                            rescue08=float(((ss==1)&(ns==0)).mean()),harm08=float(((ss==0)&(ns==1)).mean()),
                            move_rate=float((sel!=zero).mean()),pure_collision=float(pure[sel,qq].mean()),
                            empty=float(empty[sel,qq].mean()),margin=float(d['output_margins'][i]),
                            action_sha=array_sha(a,v),depth_rms_mm=float(d['depth_rms_mm']),
                            anchor_depth_rms_mm=float(d['anchor_depth_rms_mm']),
                            reader_depth_rms_mm=float(d['reader_depth_rms_mm']),
                            seen_role='validation_seen' if args.split=='test_seen' else 'held_out')
                        order=np.argsort(-d['rank_scores'][i],kind='stable')
                        for top in (1,10,50):
                            ids=order[:top]; row[f'raw_top{top}_success08']=float(ss[ids].mean())
                            row[f'raw_top{top}_count']=len(ids)
                        if method!='stage1_native' and args.label_scope=='all':
                            j=list(d['scorer_names']).index(method)
                            extra,check=decision_metrics(d['probabilities'][j],labels,row['margin'])
                            if not np.array_equal(check,sel): raise RuntimeError('Selection replay mismatch')
                            row.update(extra)
                        rows.append(row)
                    save_npz(out,dict(signature=np.array(signature),inference_sha=np.array(file_sha(path)),
                        rows_json=np.array(json.dumps(rows,allow_nan=False)),friction=fr,valid=v,evaluated_mask=label_mask,
                        collision=col,pure_collision=pure,empty=empty,newly_evaluated=np.array(n_new),
                        label_source=np.array('fresh exact CAD evaluation, NOT reused Rep-P0 labels')))
                    print(f'[FULLPATH EVAL] {args.split} {sid}/{aid} {d["mode"]}/{d["case"]} new={n_new} cache_scenes={len(evaluator.scene_cache)} {memory_status()}',flush=True)
            del memo; release_memory()
    finally:
        evaluator.scene_cache.clear(); release_memory()


if __name__=='__main__': main()
