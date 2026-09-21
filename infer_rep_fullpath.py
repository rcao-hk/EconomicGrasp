#!/usr/bin/env python3
"""Regenerate Stage-1 grasps under reader/anchor/joint depth errors, then rescore.

Includes Seen. This is NOT the old fixed-action stress test. R,w,d and seed
membership are decoded again. New actions are saved for fresh CAD evaluation.
Inference never imports the exact evaluator or consumes cached grasp labels.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from rep_a_common import (DEFAULT_CASES, SPLITS, atomic_file, check_runtime_sources, choose, digest,
    exclusive_run, file_sha, frame_identity, list_frames, parse_case, perturb_depth, save_json, save_npz, seed_for)
from rep_followup_common import checked_manifest, load_scorer, memory_status, release_memory, source_digest
from rep_fullpath_runtime import MODES, case_key, expand_actions, load_stage1, parse_offsets, routed_depths


def query_indices(native, limit):
    n=len(native)
    if limit<=0 or limit>=n: return np.arange(n,dtype=np.int64)
    # Same topk_uniform frame subset used by the fixed-action experiments.
    rank=np.argsort(-native[:,0],kind='stable'); nt=limit//2
    tail=rank[nt:]; pos=np.rint(np.linspace(0,len(tail)-1,limit-nt)).astype(int)
    return np.concatenate((rank[:nt],tail[pos]))


@torch.no_grad()
def score_queries(model, data, chunk):
    k,q=data['actions'].shape[:2]; result=[]
    for start in range(0,q,chunk):
        sub=dict(data)
        sub['actions']=data['actions'][:,start:start+chunk]
        sub['valid']=data['valid'][:,start:start+chunk]
        sub['token_ids']=data['token_ids'][start:start+chunk]
        # A0/A1 did not implement invalid-action sanitization; never feed them
        # an invalid off-support location. Its output remains excluded by valid.
        native=sub['actions'][int(data['zero_index']):int(data['zero_index'])+1].expand_as(sub['actions'])
        sub['actions']=torch.where(sub['valid'][...,None],sub['actions'],native)
        result.append(model(sub).sigmoid().cpu().numpy())
    return np.concatenate(result,axis=1)


def dump_outputs(root, payload, mode, case, split, camera, sid, aid):
    actions=payload['actions']; q=np.arange(actions.shape[1])
    for i,(method,policy) in enumerate(zip(payload['output_methods'],payload['output_policies'])):
        out=actions[payload['selected'][i],q].copy(); out[:,0]=payload['rank_scores'][i]
        path=Path(root)/'dump'/str(method)/str(policy)/mode/case_key(case)/split/f'scene_{sid:04d}'/camera/f'{aid:04d}.npy'
        with atomic_file(path) as f: np.save(f,out,allow_pickle=False)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True); p.add_argument('--stage1-checkpoint',required=True)
    p.add_argument('--cache-root',required=True,help='Rep-A cache only supplies frame schedule/calibration contract; NO action labels read')
    p.add_argument('--output-root',required=True); p.add_argument('--split',choices=tuple(SPLITS)[1:],required=True)
    p.add_argument('--scorer',action='append',default=[],help='NAME=checkpoint.pt; A0/A1/B0/control models supported')
    p.add_argument('--cases',default=DEFAULT_CASES); p.add_argument('--modes',default=','.join(MODES))
    p.add_argument('--offsets-mm',default='-40,-20,-10,0,10,20,40')
    p.add_argument('--pose-depth-mode',default='global_film'); p.add_argument('--camera',default='realsense')
    p.add_argument('--query-limit',type=int,default=64,help='0 means all Stage-1 queries; required for formal AP')
    p.add_argument('--score-query-chunk',type=int,default=64)
    p.add_argument('--shard-id',type=int,default=0); p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--max-frames',type=int,default=0); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--min-host-free-gib',type=float,default=4.)
    p.add_argument('--replay-atol',type=float,default=5e-5)
    p.add_argument('--resume',action='store_true'); p.add_argument('--repair-corrupt',action='store_true')
    args=p.parse_args(); sys.argv=[sys.argv[0]]
    if not 0<=args.shard_id<args.num_shards or args.score_query_chunk<1: raise ValueError('Invalid sharding/chunk')
    cases=list(dict.fromkeys(['nominal']+[s.strip() for s in args.cases.split(',') if s.strip()]))
    for case in cases: parse_case(case)
    modes=list(dict.fromkeys(args.modes.split(',')))
    if not modes or any(m not in MODES for m in modes): raise ValueError('Unknown path mode')
    manifest=check_runtime_sources(args.cache_root)
    if file_sha(args.stage1_checkpoint)!=manifest['stage1_sha256']:
        raise RuntimeError('Use the SAME Stage-1 checkpoint as Rep-A for the first controlled experiment')
    if manifest['camera']!=args.camera or manifest['pose_depth_mode']!=args.pose_depth_mode:
        raise ValueError('Camera/pose-depth-mode differs from cache contract')
    specs={}
    for item in args.scorer:
        name,path=item.split('=',1)
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name=='stage1_native' or name in specs:
            raise ValueError(f'Bad/duplicate scorer name: {name}')
        specs[name]=path
    offsets=parse_offsets(args.offsets_mm)
    paths=list_frames(args.cache_root,args.split)
    scenes=sorted({frame_identity(x)[0] for x in paths})
    owned=set(scenes[args.shard_id::args.num_shards])
    paths=[x for x in paths if frame_identity(x)[0] in owned]
    if args.max_frames>0: paths=paths[:args.max_frames]
    schedule=[('joint','nominal')]+[(mode,c) for c in cases if c!='nominal' for mode in modes]
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,manifest=manifest,stage1_sha=manifest['stage1_sha256'],camera=args.camera,
        scorers={n:file_sha(v) for n,v in specs.items()},cases=cases,modes=modes,offsets_mm=offsets.tolist(),
        query_limit=args.query_limit,seed=args.seed,feature_precision=manifest['feature_dtype'],
        code=source_digest(('rep_fullpath_runtime.py','infer_rep_fullpath.py','rep_followup_common.py',
                            'rep_b0_controls.py','rep_a_model.py','rep_b_model.py')),
        scope='Full Stage-1 depth->enhancement/anchor->ViewNet/CDF/width regeneration, then K-translation rescore',
        caveat='A1 augments the reader/scorer only, NOT a separately augmentation-trained full Stage-1',
        candidate_policy='Reselect image-FPS naturally; regenerate R,w,d; one output per selected query',
        workspace='Original dataset workspace crop; GT-free deployment crop is not tested')
    signature=digest(protocol)
    with exclusive_run(root/'.protocol.lock',wait=True): checked_manifest(root,protocol)
    memory_status(args.min_host_free_gib)
    # Late heavy imports: argparse must not leak into utils.arguments.
    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    device=torch.device(args.device)
    stage,replay,use_fuse=load_stage1(args.stage1_checkpoint,device,args.pose_depth_mode)
    scorers={name:load_scorer(path,args.cache_root,device) for name,path in specs.items()}
    dataset=GraspNetMultiDataset(args.dataset_root,split=args.split,camera=args.camera,num_points=20000,
        remove_outlier=True,augment=False,load_label=False,use_gt_depth=False,use_fuse_depth=use_fuse,
        min_depth=.2,max_depth=1.,bin_num=256)
    index={(int(str(s).split('_')[-1]),i%256):i for i,s in enumerate(dataset.scene_list())}
    completed=0
    for src in paths:
        sid,aid=frame_identity(src); folder=root/'inference'/args.split/f'scene_{sid:04d}'
        folder.mkdir(parents=True,exist_ok=True)
        with exclusive_run(folder/f'ann_{aid:04d}.lock'):
            pending=[]
            for mode,case in schedule:
                path=folder/f'ann_{aid:04d}_{mode}_{case_key(case)}.npz'
                if path.exists() and args.resume:
                    try:
                        with np.load(path,allow_pickle=False) as f: cached={k:f[k] for k in f.files}
                    except Exception:
                        if not args.repair_corrupt: raise
                        pending.append((mode,case,path)); continue
                    if str(cached['signature'])!=signature: raise RuntimeError(f'Changed inference contract: {path}')
                    # Recreate a missing/interrupted .npy dump without inference.
                    dump_outputs(root,cached,mode,case,args.split,args.camera,sid,aid)
                    del cached; continue
                if path.exists() and not args.resume: raise FileExistsError(path)
                pending.append((mode,case,path))
            if not pending: continue
            memory_status(args.min_host_free_gib)
            item=dataset[index[(sid,aid)]]
            allowed=('img','K','camera_pose_vec','camera_gravity_vec','scene_idx','anno_idx','token_valid_mask')
            batch=collate_fn([{k:item[k] for k in allowed if k in item}]); del item
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            batch.update(cva_export_angle_feature=False,cva_compute_diagnostics=False,geometry_compute_diagnostics=False)
            ep0=replay.capture(batch)
            native0=pred_decode_center_view_angle(ep0,use_cdf=True)[0].cpu().numpy()
            token0=ep0['kview_base_token_sel_idx'][0].cpu().numpy()
            depth0=replay.nominal_depth
            check=replay.run(batch,depth0,depth0)
            rebuilt=pred_decode_center_view_angle(check,use_cdf=True)[0].cpu().numpy()
            noop=float(np.max(np.abs(rebuilt-native0)))
            if noop>args.replay_atol: raise RuntimeError(f'Nominal full-path replay failed {sid}/{aid}: {noop}')
            if not np.array_equal(check['kview_base_token_sel_idx'][0].cpu().numpy(),token0):
                raise RuntimeError('Nominal image-FPS replay changed')
            del check,rebuilt
            for mode,case,path in pending:
                seed=seed_for(args.seed,args.split,sid,aid,parse_case(case)[0])
                pert,err=perturb_depth(depth0[0],case,seed)
                anchor,reader=routed_depths(depth0,pert[None],mode)
                ep=replay.run(batch,anchor,reader)
                native_all=pred_decode_center_view_angle(ep,use_cdf=True)[0].cpu().numpy()
                ids=query_indices(native_all,args.query_limit); native=native_all[ids]
                action,valid,zero=expand_actions(native,offsets)
                h,w=reader.shape[-2:]
                raw=replay.image_feature
                # Same quantization as the fixed-action training cache.
                raw=raw.to(torch.float16 if manifest['feature_dtype']=='float16' else torch.float32).float()
                tokens=ep['kview_base_token_sel_idx'][0][torch.as_tensor(ids,device=device)]
                data=dict(image_feature=raw,depth=reader[0],K=batch['K'][0],
                    objectness=ep['objectness_score'][0].reshape(2,h,w),
                    graspness=ep['graspness_score'][0].reshape(1,h,w),
                    actions=torch.from_numpy(action).to(device),valid=torch.from_numpy(valid).to(device),
                    offsets_mm=torch.from_numpy(offsets).to(device),zero_index=zero,token_ids=tokens)
                methods=['stage1_native']; policies=['native']; margins=[0.]
                chosen=[np.full(len(ids),zero,np.int64)]; scores=[native[:,0]]; probs=[]
                for name,(model,meta) in scorers.items():
                    prob=score_queries(model,data,args.score_query_chunk)
                    if not np.isfinite(prob).all(): raise FloatingPointError(f'{name}: non-finite scores')
                    probs.append(prob)
                    for policy,margin in (('fixed_0',0.),('val_selected',meta['margin'])):
                        sel=choose(prob.mean(-1),valid,zero,margin)
                        methods.append(name); policies.append(policy); margins.append(margin)
                        chosen.append(sel); scores.append(prob.mean(-1)[sel,np.arange(len(ids))])
                payload=dict(signature=np.array(signature),scene_id=np.array(sid),anno_id=np.array(aid),
                    split=np.array(args.split),mode=np.array(mode),case=np.array(case),actions=action,valid=valid,
                    offsets_mm=offsets,zero_index=np.array(zero),query_ids=ids,token_ids=tokens.cpu().numpy(),
                    original_native_score=native[:,0],output_methods=np.asarray(methods),output_policies=np.asarray(policies),
                    output_margins=np.asarray(margins),selected=np.stack(chosen),rank_scores=np.stack(scores),
                    scorer_names=np.asarray(list(scorers)),probabilities=np.stack(probs) if probs else np.empty((0,*valid.shape,6),np.float32),
                    replay_max_abs=np.array(noop),num_stage1_queries=np.array(len(native_all)),
                    depth_rms_mm=np.array(err['depth_rms_mm']),depth_bias_mm=np.array(err['depth_bias_mm']),
                    anchor_depth_rms_mm=np.array(float((anchor-depth0).square().mean().sqrt())*1000),
                    reader_depth_rms_mm=np.array(float((reader-depth0).square().mean().sqrt())*1000),
                    anchor_valid_fraction=np.array(ep['fullpath_anchor_valid_fraction']),
                    seed_set_jaccard=np.array(len(set(token0)&set(ep['kview_base_token_sel_idx'][0].cpu().numpy()))/max(1,len(set(token0)|set(ep['kview_base_token_sel_idx'][0].cpu().numpy())))))
                save_npz(path,payload); dump_outputs(root,payload,mode,case,args.split,args.camera,sid,aid)
                del ep,native_all,native,action,valid,data,raw,tokens,payload,pert,anchor,reader
            del ep0,batch,native0,token0,depth0
            replay.clear(); release_memory(); completed+=1
            print(f'[FULLPATH INFER] {args.split} scene={sid} ann={aid} completed={completed} {memory_status()}',flush=True)
    save_json(root/f'infer_{args.split}_shard{args.shard_id}.json',dict(signature=signature,completed_new_frames=completed,
        selected_frames=len(paths),shard=args.shard_id,num_shards=args.num_shards,seen_role='validation_seen' if args.split=='test_seen' else 'held_out'))


if __name__=='__main__': main()
