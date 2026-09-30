#!/usr/bin/env python3
"""P1-2: online audit of transferred canonical labels versus exact actions.

The script never writes a reusable label/action cache. It samples live predicted
candidates, evaluates those exact [17D] grasps with the existing CAD/DexNet
evaluator, and writes only diagnostic result JSON/CSV.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

from mgf_p0_core import cdf_targets, exact_targets


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint',required=True)
    p.add_argument('--dataset-root',default='/data/robotarm/dataset/graspnet')
    p.add_argument('--output-root',required=True)
    p.add_argument('--split',choices=['test_seen','test_similar','test_novel'],required=True)
    p.add_argument('--frames-per-scene',type=int,default=1)
    p.add_argument('--queries',type=int,default=8)
    p.add_argument('--top-candidates',type=int,default=4)
    p.add_argument('--random-candidates',type=int,default=4)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--fc-mode',choices=['official','reuse_contacts'],default='official')
    p.add_argument('--verify-n',type=int,default=8)
    p.add_argument('--shard-id',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--merge',action='store_true')
    return p


def _bin(value, edges, labels):
    if value is None or not np.isfinite(value): return 'invalid'
    i=int(np.searchsorted(np.asarray(edges,float),float(value),side='right'))
    return labels[i]


def _aggregate(rows):
    groups={}
    def add(name,row):
        g=groups.setdefault(name,dict(n=0,threshold_agreement=0.,utility_abs_error=0.,
                                     any_success_correct=0,transfer_positive=0,exact_positive=0,
                                     false_positive=0,false_negative=0))
        g['n']+=1
        g['threshold_agreement']+=float(row['threshold_agreement'])
        g['utility_abs_error']+=abs(float(row['transfer_utility'])-float(row['exact_utility']))
        tp=bool(row['transfer_any_success']); ep=bool(row['exact_any_success'])
        g['any_success_correct']+=int(tp==ep); g['transfer_positive']+=int(tp); g['exact_positive']+=int(ep)
        g['false_positive']+=int(tp and not ep); g['false_negative']+=int((not tp) and ep)
    for row in rows:
        add('overall',row)
        add('center/'+row['center_bin'],row)
        add('width/'+row['width_bin'],row)
        add('collision/'+row['collision_group'],row)
    out={}
    for name,g in groups.items():
        n=g.pop('n')
        out[name]=dict(n=n,
            threshold_agreement=g['threshold_agreement']/n,
            utility_mae=g['utility_abs_error']/n,
            any_success_accuracy=g['any_success_correct']/n,
            transfer_positive_fraction=g['transfer_positive']/n,
            exact_positive_fraction=g['exact_positive']/n,
            false_positive_fraction=g['false_positive']/n,
            false_negative_fraction=g['false_negative']/n)
    return out


def merge(root,split):
    from metric_field_runtime import atomic_json,digest
    root=Path(root)/split
    protocol=json.loads((root/'protocol.json').read_text()); sig=digest(protocol)
    rows=[]
    for sid,ann in protocol['schedule']:
        path=root/'frames'/f'{sid:04d}_{ann:04d}.json'
        data=json.loads(path.read_text())
        if data['signature']!=sig: raise RuntimeError(f'Stale P1-2 result {path}')
        rows.extend(data['rows'])
    summary=_aggregate(rows)
    atomic_json(root/'summary.json',dict(protocol=protocol,frames=len(protocol['schedule']),
                candidates=len(rows),metrics=summary,
                caveat='Diagnostic transferred-vs-exact labels; not official GraspNet AP and never a training cache'))
    if rows:
        with (root/'per_action.csv').open('w',newline='') as f:
            keys=sorted({k for r in rows for k in r if k not in ('transfer_target','exact_target')})
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
            for r in rows: w.writerow({k:r.get(k) for k in keys})
    print(json.dumps(summary,indent=2))


@torch.no_grad()
def main():
    a=parser().parse_args()
    if a.merge: return merge(a.output_root,a.split)
    if min(a.frames_per_scene,a.queries,a.top_candidates+a.random_candidates,a.num_shards)<1:
        raise ValueError('Invalid audit counts')
    if not 0<=a.shard_id<a.num_shards: raise ValueError('Invalid shard')
    if not torch.cuda.is_available(): raise RuntimeError('P1-2 needs CUDA/GraspNet environment')

    from mgf_p0_online import FrozenSource,safe_batch,code_digest
    from metric_field_runtime import (make_dataset,dataset_schedule,atomic_json,
                                      ensure_manifest,digest,seed_all)
    from dataset.graspnet_dataset import collate_fn
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator

    seed_all(a.seed); source=FrozenSource(a.source_checkpoint,'cuda:0')
    full,_,indices=make_dataset(a.dataset_root,a.split,.1,labels=True,
                                use_fuse_depth=source.protocol.get('use_fuse_depth',False))
    scene_rows={}
    for idx,(sid,ann) in zip(indices,dataset_schedule(full,indices)):
        scene_rows.setdefault(sid,[]).append(idx)
    if a.frames_per_scene>26: raise ValueError('At most 26 stride-10 frames per scene')
    indices=[v[int(j*len(v)/a.frames_per_scene)] for v in scene_rows.values() for j in range(a.frames_per_scene)]
    schedule=dataset_schedule(full,indices)
    protocol=dict(version='mgf_online_p1_2_v1',source_sha256=source.checkpoint_sha,
                  code_sha256=code_digest(),split=a.split,schedule=schedule,
                  frames_per_scene=a.frames_per_scene,queries=a.queries,
                  top_candidates=a.top_candidates,random_candidates=a.random_candidates,
                  seed=a.seed,fc_mode=a.fc_mode,verify_n=a.verify_n,num_shards=a.num_shards,
                  dataset_root=str(Path(a.dataset_root).resolve()),
                  transferred_labels='existing <=5mm nearest canonical CDF/width labels from online matcher',
                  exact_labels='live actual predicted actions: CAD/table collision plus force closure',
                  center_strata_mm=[0,1,2,3,4,5],
                  width_strata_mm=[0,2,5,10,20],
                  collision_strata='clear / pure_collision / empty; stock evaluator exposes no continuous collision margin',
                  output_contract='JSON/CSV diagnostic results only; never consumed by training')
    root=Path(a.output_root)/a.split; ensure_manifest(root/'protocol.json',protocol); signature=digest(protocol)
    import fcntl
    lock=open(root/f'.shard_{a.shard_id}.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    evaluator=ExactGraspNetActionEvaluator(a.dataset_root,'realsense',split=a.split,
                                          fc_mode=a.fc_mode,verify_n=a.verify_n,strict=True)
    current_scene=None

    for pos,(idx,(sid,ann)) in enumerate(zip(indices,schedule)):
        if pos%a.num_shards!=a.shard_id: continue
        path=root/'frames'/f'{sid:04d}_{ann:04d}.json'
        if path.exists():
            if not a.resume: raise FileExistsError(path)
            if json.loads(path.read_text())['signature']!=signature: raise RuntimeError('Stale frame result')
            continue
        if current_scene!=sid:
            evaluator.scene_cache.clear(); current_scene=sid
        rng=np.random.default_rng(a.seed+sid*1000+ann)
        ctx=source(safe_batch(collate_fn([full[idx]]),'cuda:0',True))
        ep=ctx['ep']; b,q,A,D=ctx['shape']
        if b!=1: raise RuntimeError('P1-2 audits one image at a time')
        bins=ep['batch_grasp_cdf_bins_angle_depth'][0].long()
        valid=ep['batch_grasp_cdf_valid_mask'][0].bool()
        width_label=ep['batch_grasp_width_angle_depth'][0].float()
        width_valid=ep['batch_grasp_width_valid_mask_angle_depth'][0].bool()
        base_u=ep['grasp_cdf_pred_angle_depth'].sigmoid().mean(1)[0]
        action=ctx['actions'].reshape(q,A,D,17)
        predicted_width=action[...,1]
        point=ep['batch_grasp_point'][0]
        center=ep['xyz_graspable'][0]
        center_mm=(center-point).norm(dim=-1)*1000.

        q_valid=valid.flatten(1).any(1)
        qscore=base_u.flatten(1).masked_fill(~valid.flatten(1),-1).max(1).values
        possible=torch.where(q_valid)[0].cpu().numpy()
        if len(possible)==0: raise RuntimeError('No canonically valid query in audit frame')
        nq=min(a.queries,len(possible)); half=(nq+1)//2
        order=torch.argsort(qscore,descending=True,stable=True).cpu().numpy()
        top=np.asarray([x for x in order if x in set(possible)],dtype=int)[:half]
        rest=np.setdiff1d(possible,top)
        rand=rng.choice(rest,nq-half,replace=False) if nq>half else np.zeros(0,dtype=int)
        qids=np.concatenate([top,rand])

        selected=[]
        for qi in qids:
            vm=valid[qi].flatten().cpu().numpy()
            ids=np.flatnonzero(vm)
            scores=base_u[qi].flatten().cpu().numpy()
            order=ids[np.argsort(-scores[ids],kind='stable')]
            nt=min(a.top_candidates,len(order)); chosen=list(order[:nt])
            remain=np.setdiff1d(ids,np.asarray(chosen,dtype=int))
            nr=min(a.random_candidates,len(remain))
            if nr: chosen.extend(rng.choice(remain,nr,replace=False).tolist())
            for ci in chosen: selected.append((int(qi),int(ci)))
        if not selected: raise RuntimeError('No valid candidate selected')

        grasps=[]
        transfer_bins=[]; transfer_width=[]; transfer_width_valid=[]; cmm=[]; pwidth=[]; aid=[]; did=[]
        for qi,ci in selected:
            aa,dd=divmod(ci,D)
            grasps.append(action[qi,aa,dd].cpu().numpy())
            transfer_bins.append(int(bins[qi,aa,dd]))
            transfer_width.append(float(width_label[qi,aa,dd]))
            transfer_width_valid.append(bool(width_valid[qi,aa,dd]))
            cmm.append(float(center_mm[qi])); pwidth.append(float(predicted_width[qi,aa,dd]))
            aid.append(aa); did.append(dd)
        grasps=np.asarray(grasps,np.float32)
        exact=evaluator.evaluate(sid,ann,grasps)
        exact_t=exact_targets(exact,len(grasps))
        transfer_t=cdf_targets(torch.tensor(transfer_bins,dtype=torch.long)).cpu().numpy()
        transfer_u=transfer_t.mean(-1); exact_u=exact_t.mean(-1)
        rows=[]
        for i,(qi,ci) in enumerate(selected):
            wm=(abs(pwidth[i]-transfer_width[i])*1000. if transfer_width_valid[i] else None)
            collision_group=('empty' if bool(exact.empty[i]) else
                             'pure_collision' if bool(exact.pure_collision[i]) else 'clear')
            row=dict(scene=int(sid),ann=int(ann),query=qi,candidate=ci,angle=aid[i],depth=did[i],
                     center_match_mm=cmm[i],
                     center_bin=_bin(cmm[i],[1,2,3,4],['0-1','1-2','2-3','3-4','4-5']),
                     predicted_width_mm=1000*pwidth[i],
                     canonical_width_mm=(1000*transfer_width[i] if transfer_width_valid[i] else None),
                     width_mismatch_mm=wm,width_label_valid=transfer_width_valid[i],
                     width_bin=_bin(wm,[2,5,10,20],['0-2','2-5','5-10','10-20','20+']),
                     transfer_bin=transfer_bins[i],transfer_utility=float(transfer_u[i]),
                     exact_friction=float(exact.friction[i]),exact_utility=float(exact_u[i]),
                     transfer_any_success=bool(transfer_t[i].any()),exact_any_success=bool(exact_t[i].any()),
                     threshold_agreement=float((transfer_t[i]==exact_t[i]).mean()),
                     collision_or_empty=bool(exact.collision_or_empty[i]),
                     pure_collision=bool(exact.pure_collision[i]),empty=bool(exact.empty[i]),
                     collision_group=collision_group,
                     transfer_target=transfer_t[i].astype(int).tolist(),
                     exact_target=exact_t[i].astype(int).tolist())
            rows.append(row)
        atomic_json(path,dict(signature=signature,scene=sid,ann=ann,rows=rows,
                    metrics=_aggregate(rows),exact_evaluator_stats=exact.stats,
                    note='Result-only exact-action audit; no cache generated or consumed by training'))
        print(f'[P1-2] {a.split} scene={sid} ann={ann} queries={len(qids)} actions={len(rows)}',flush=True)
        del ctx,ep

if __name__=='__main__': main()
