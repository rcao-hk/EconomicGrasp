#!/usr/bin/env python3
"""P0-2 online fixed-action observation/action counterfactuals with exact labels.

No label-mining stage. Existing CAD/DexNet evaluator runs on actual actions in
this process. JSON/CSV files are audit RESULTS, never inputs to P0-1 training.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from mgf_p0_core import VERSION, THRESHOLDS, perturb_actions, exact_targets, comparison_metrics


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint',default='')
    p.add_argument('--controls-root',default='')
    p.add_argument('--variants',default='base,feature_only,full,cva')
    p.add_argument('--dataset-root',default='/data/robotarm/dataset/graspnet')
    p.add_argument('--gntrans-rgb-root',default='')
    p.add_argument('--output-root',required=True)
    p.add_argument('--split',choices=['test_seen','test_similar','test_novel'],required=True)
    p.add_argument('--mode',choices=['observation','action','both'],default='both')
    p.add_argument('--frames-per-scene',type=int,default=1)
    p.add_argument('--queries',type=int,default=8)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--ray-offsets',default='-0.02,-0.01,0.01,0.02',help='Camera Z offset in metres, along original image ray')
    p.add_argument('--roll-degrees',default='-15,15',help='Local approach-axis roll; must be multiples of the native angle step')
    p.add_argument('--width-offsets',default='-0.01,0.01',help='Metres; invalid widths are excluded, never clipped')
    p.add_argument('--fc-mode',choices=['official','reuse_contacts'],default='official')
    p.add_argument('--verify-n',type=int,default=8)
    p.add_argument('--shard-id',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--merge',action='store_true',help='Only verify coverage and aggregate completed report files')
    return p


def merge(root, split):
    from metric_field_runtime import atomic_json,digest
    root=Path(root)/split
    protocol=json.loads((root/'protocol.json').read_text()); signature=digest(protocol)
    aggregates={}; rows=[]
    for sid,ann in protocol['schedule']:
        path=root/'frames'/f'{sid:04d}_{ann:04d}.json'
        data=json.loads(path.read_text())
        if data['signature']!=signature: raise RuntimeError(f'Stale audit result {path}')
        for name,stats in data['metrics'].items():
            entry=aggregates.setdefault(name,{})
            for key,value in stats.items():
                if not isinstance(value,(float,int)) or value is None: continue
                weight=(stats.get('valid_delta_pairs',0) if key=='delta_sign_accuracy'
                        else stats.get('candidates',0) if key in ('utility_mae','utility_shift')
                        else stats.get('queries',1))
                if key in ('queries','candidates','valid_delta_pairs','target_tie_pairs'):
                    entry[key]=entry.get(key,0)+value
                elif weight:
                    acc=entry.setdefault(key,[0.,0.]); acc[0]+=value*weight; acc[1]+=weight
        rows.extend(data['action_results'])
    result={n:{k:(v[0]/v[1] if isinstance(v,list) else v) for k,v in s.items()} for n,s in aggregates.items()}
    atomic_json(root/'summary.json',dict(protocol=protocol,metrics=result,frames=len(protocol['schedule']),
                       caveat='Exact fixed-action diagnostics, NOT official GraspNet AP; no sensor-cloud post-filter'))
    import csv
    with (root/'per_action.csv').open('w',newline='') as f:
        keys=sorted({k for row in rows for k in row})
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result,indent=2))


@torch.no_grad()
def main():
    a=parser().parse_args()
    if a.merge: return merge(a.output_root,a.split)
    if min(a.frames_per_scene,a.queries,a.num_shards)<1 or not 0<=a.shard_id<a.num_shards:
        raise ValueError('Invalid audit counts/sharding')
    if not a.source_checkpoint: raise ValueError('--source-checkpoint required')
    if a.mode in ('observation','both') and not Path(a.gntrans_rgb_root).is_dir():
        raise ValueError('Observation audit requires existing --gntrans-rgb-root')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA/GraspNet environment required')
    from mgf_p0_online import (FrozenSource,make_control,load_control,safe_batch,paired_dataset,
                               verify_pair,fixed_queries,code_digest)
    from metric_field_runtime import make_dataset,dataset_schedule,atomic_json,ensure_manifest,digest,sha256_file,seed_all
    seed_all(a.seed)
    source=FrozenSource(a.source_checkpoint,'cuda:0')
    from dataset.graspnet_dataset import collate_fn
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    controls={}; control_hashes={}
    names=a.variants.split(',')
    if len(set(names))!=len(names) or 'base' not in names: raise ValueError('Unique variants including base required')
    for name in names:
        if name=='base': controls[name]=make_control(source,'base').eval(); continue
        path=Path(a.controls_root)/name/'train'/'checkpoint_latest.pt'
        ctrl,ck=load_control(source,path)
        if ctrl.variant!=name or ck['protocol']['partial_run']:
            raise ValueError(f'Incomplete/wrong control {path}')
        controls[name]=ctrl; control_hashes[name]=sha256_file(path)
    full,_,indices=make_dataset(a.dataset_root,a.split,.1,labels=False)
    material=paired_dataset(full,a.gntrans_rgb_root) if a.mode!='action' else None
    scene_rows={}
    for i,(sid,ann) in zip(indices,dataset_schedule(full,indices)): scene_rows.setdefault(sid,[]).append(i)
    if a.frames_per_scene>26: raise ValueError('At most 26 audit frames per scene at this schedule')
    indices=[v[int(j*len(v)/a.frames_per_scene)] for v in scene_rows.values() for j in range(a.frames_per_scene)]
    schedule=dataset_schedule(full,indices)
    def amounts(s):
        values=[float(x) for x in s.split(',') if x]
        if not all(math.isfinite(x) and x!=0 for x in values) or len(set(values))!=len(values):
            raise ValueError('Offsets must be unique finite nonzero values')
        return values
    offsets={'ray_z':amounts(a.ray_offsets),'roll': [math.radians(x) for x in amounts(a.roll_degrees)],
             'width':amounts(a.width_offsets)}
    protocol=dict(version=VERSION,source_sha256=source.checkpoint_sha,control_sha256=control_hashes,
                  variants=names,code_sha256=code_digest(),split=a.split,schedule=schedule,mode=a.mode,
                  queries=a.queries,seed=a.seed,offsets=offsets,fc_mode=a.fc_mode,verify_n=a.verify_n,
                  num_shards=a.num_shards,dataset_root=str(Path(a.dataset_root).resolve()),
                  gntrans_rgb_root=str(Path(a.gntrans_rgb_root).resolve()) if material else None,
                  labels='live CAD/table collision + exact force closure for each physical action; thresholds unchanged across visual materials',
                  feature_bundle='proposal/relative/metric latents from current RGB',
                  geometry_bundle='predicted numeric depth + first-surface profile; no observed/GT depth',
                  action_bank='original frozen-Base top-half and random-half queries, all AxD widths held fixed',
                  rotation='native in-plane roll grid; NOT arbitrary tilt',
                  width_baseline='CVA has no explicit width input; its width-perturbation score is invariant by construction',
                  preprocessing='Only colorpath replaced; original sensor-space crop and camera metadata preserved')
    root=Path(a.output_root)/a.split
    ensure_manifest(root/'protocol.json',protocol); signature=digest(protocol)
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
            if json.loads(path.read_text())['signature']!=signature: raise RuntimeError('Stale audit frame')
            continue
        if sid!=current_scene:
            evaluator.scene_cache.clear()  # bounded read-only CAD assets in RAM, no disk cache
            current_scene=sid
        rng=np.random.default_rng(a.seed+sid*1000+ann)
        raw=full[idx]; batch=safe_batch(collate_fn([raw]),'cuda:0',False)
        native=source(batch)
        utility=native['ep']['grasp_cdf_pred_angle_depth'].sigmoid().mean(1)[0].flatten(1).max(1).values
        qcount=min(a.queries,len(utility)); half=(qcount+1)//2
        top=torch.argsort(utility,descending=True,stable=True)[:half].cpu().numpy()
        other=np.setdiff1d(np.arange(len(utility)),top)
        query_ids=np.concatenate([top,rng.choice(other,qcount-half,replace=False)])
        bank=fixed_queries(native,torch.tensor(query_ids,device='cuda:0'))
        del native
        original=source(batch,fixed=bank)
        b,q,nangle,ndepth=original['shape']; nc=nangle*ndepth
        if b!=1: raise RuntimeError('Audit requires one image per forward')
        for value in offsets['roll']:
            if not math.isclose(value/(math.pi/nangle),round(value/(math.pi/nangle)),abs_tol=1e-5):
                raise ValueError('Roll offsets must match native angle bins')
        actions=original['actions'].clone()
        valid=((actions[...,1]>0)&(actions[...,1]<=source.model.max_width)&(actions[...,15]>0))[0].cpu().numpy()
        metrics={}; action_results=[]; eval_stats=[]

        def evaluate(act,mask):
            array=act[0].float().cpu().numpy(); flat=np.asarray(mask,bool).reshape(-1)
            target=np.zeros((len(array),6),np.float32)
            if flat.any():
                result=evaluator.evaluate(sid,ann,array[flat])
                target[flat]=exact_targets(result,int(flat.sum())); eval_stats.append(result.stats)
            return target.mean(-1)

        def score(ctx,act,roll_steps=0):
            result={}
            baseline=ctx['ep']['grasp_cdf_pred_angle_depth']
            base=torch.roll(baseline,shifts=-roll_steps,dims=3) if roll_steps else baseline
            for name,ctrl in controls.items():
                if name=='base': logits=base
                elif name=='cva':
                    logits=ctrl(ctx)
                    if roll_steps: logits=torch.roll(logits,shifts=-roll_steps,dims=3)
                else: logits=ctrl(ctx,actions=act,base_logits=base)
                result[name]=logits.sigmoid().mean(1)[0].flatten(1).cpu().numpy()
            return result

        original_scores=score(original,actions)
        truth=None
        if material is not None:
            # Same original depth/segmentation crop, ONLY paired RGB is replaced.
            raw_m=material[idx]; verify_pair(full,material,idx,raw,raw_m)
            mat_batch=safe_batch(collate_fn([raw_m]),'cuda:0',False)
            changed=source(mat_batch,fixed=bank)
            truth=evaluate(actions,valid).reshape(q,nc)
            contexts=[('original',original),('both',changed)]
            # Process hybrid contexts sequentially, not four simultaneous source models.
            for condition in ('original','both','visual_only','geometry_only'):
                if condition=='original': ctx=original
                elif condition=='both': ctx=changed
                elif condition=='visual_only': ctx=source(mat_batch,fixed=bank,geometry_from=original)
                else: ctx=source(batch,fixed=bank,geometry_from=changed)
                scores=score(ctx,actions)
                for name,s in scores.items():
                    metrics[f'observation/{condition}/{name}']=comparison_metrics(s,truth,valid.reshape(q,nc),original_scores[name])
                for qi in range(q):
                    for ci in range(nc):
                        row=dict(scene=sid,ann=ann,experiment='observation',condition=condition,query=int(query_ids[qi]),
                                 candidate=ci,valid=bool(valid[qi*nc+ci]),
                                 exact_utility=float(truth[qi,ci]) if valid[qi*nc+ci] else None)
                        row.update({f'utility_{name}':float(s[qi,ci]) for name,s in scores.items()})
                        action_results.append(row)
                if condition in ('visual_only','geometry_only'): del ctx
            del contexts,changed,mat_batch,raw_m
        if a.mode in ('action','both'):
            # Two fixed anchors/query: source-Base winner + random nonwinner.
            best=original_scores['base'].argmax(1)
            random=np.array([rng.choice(np.delete(np.arange(nc),j)) for j in best])
            anchor=np.stack([best,random],1); qi=np.arange(q)[:,None]
            flat_anchor=(qi*nc+anchor).reshape(-1)
            index=torch.tensor(flat_anchor,device=actions.device)
            va=valid[flat_anchor].reshape(q,2)
            y0=(truth[qi,anchor] if truth is not None else evaluate(actions[:,index],va).reshape(q,2))
            s0={k:v[qi,anchor] for k,v in original_scores.items()}
            all_y=[y0]; all_valid=[va]; all_scores={k:[v] for k,v in s0.items()}
            for condition,kind,amount in [('native','',0.)]+[(f'{k}:{v:g}',k,v) for k,vs in offsets.items() for v in vs]:
                if condition=='native':
                    act=actions; vm=valid; scores=original_scores; target=y0
                else:
                    act,vm_tensor=perturb_actions(actions,kind,amount,source.model.max_width)
                    vm=vm_tensor[0].cpu().numpy()
                    # Fixed centre intervention: re-run Base/CVA at the new
                    # physical centre, never reuse native centre labels/scores.
                    if kind=='ray_z':
                        shifted=copy.deepcopy(bank)
                        shifted['centres']=act.reshape(b,q,nangle,ndepth,17)[:,:,0,0,13:16]
                        ctx=source(batch,fixed=shifted)
                    else: ctx=original
                    step=int(round(amount/(math.pi/nangle))) if kind=='roll' else 0
                    scores=score(ctx,act,step)
                    target=evaluate(act[:,index],vm[flat_anchor]).reshape(q,2)
                    all_y.append(target); all_valid.append(vm[flat_anchor].reshape(q,2))
                    for name,s in scores.items(): all_scores[name].append(s[qi,anchor])
                    if kind=='ray_z': del ctx
                mask=vm[flat_anchor].reshape(q,2)
                for name,s in scores.items():
                    chosen=s[qi,anchor]
                    usable=mask&va; dy=target-y0; ds=chosen-s0[name]
                    informative=usable&(np.abs(dy)>1e-6)
                    sign_score=(np.sign(ds)==np.sign(dy)).astype(float)
                    sign_score[np.abs(ds)<1e-8]=.5
                    metrics[f'action/{condition}/{name}']={
                        'queries':int(usable.sum()),'candidates':int(mask.sum()),
                        'utility_mae':float(np.abs(chosen-target)[mask].mean()) if mask.any() else None,
                        'valid_delta_pairs':int(informative.sum()),'target_tie_pairs':int((usable&~informative).sum()),
                        'delta_sign_accuracy':float(sign_score[informative].mean()) if informative.any() else None}
                for ii in range(q):
                    for jj in range(2):
                        row=dict(scene=sid,ann=ann,experiment='action',condition=condition,query=int(query_ids[ii]),
                                 candidate=int(anchor[ii,jj]),anchor=jj,valid=bool(mask[ii,jj]),
                                 exact_utility=float(target[ii,jj]) if mask[ii,jj] else None)
                        row.update({f'utility_{name}':float(s[ii,anchor[ii,jj]]) for name,s in scores.items()})
                        action_results.append(row)
            ys=np.stack(all_y,-1).reshape(q*2,-1); ms=np.stack(all_valid,-1).reshape(q*2,-1)
            for name,vals in all_scores.items():
                metrics[f'action/family_selection/{name}']=comparison_metrics(np.stack(vals,-1).reshape(q*2,-1),ys,ms)
        atomic_json(path,dict(signature=signature,scene=sid,ann=ann,metrics=metrics,action_results=action_results,
                             exact_evaluator_stats=eval_stats,action_sha256=digest(actions.cpu().tolist()),
                             note='Results only; these exact labels are never consumed by P0-1 training'))
        print(f'[P0-2] scene={sid} ann={ann} queries={q} completed',flush=True)
        del original,batch,actions


if __name__=='__main__': main()
