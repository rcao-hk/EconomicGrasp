#!/usr/bin/env python3
"""Frozen-source inference for trained P1-1/P1-3 controls."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint',required=True)
    p.add_argument('--control-checkpoint',required=True)
    p.add_argument('--dataset-root',default='/data/robotarm/dataset/graspnet')
    p.add_argument('--output-root',required=True)
    p.add_argument('--split',choices=['test_seen','test_similar','test_novel'],required=True)
    p.add_argument('--eval-fraction',type=float,default=.1)
    p.add_argument('--shard-id',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--collision',choices=['off','on','both'],default='off')
    p.add_argument('--collision-thresh',type=float,default=.01)
    p.add_argument('--voxel-size',type=float,default=.01)
    p.add_argument('--approach-dist',type=float,default=.05)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    return p


@torch.no_grad()
def main():
    a=parser().parse_args()
    if a.num_shards<1 or not 0<=a.shard_id<a.num_shards or a.batch_size<1 or a.workers<0 or a.max_frames<0:
        raise ValueError('Invalid inference counts')
    if not all(np.isfinite(x) and x>0 for x in (a.collision_thresh,a.voxel_size,a.approach_dist)):
        raise ValueError('Collision parameters must be finite positive')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA inference environment required')

    from mgf_p0_online import FrozenSource,safe_batch
    from mgf_p1_online import load_control,prepare_context,code_digest,VERSION as P1_VERSION
    from metric_field_runtime import (VERSION,make_dataset,dataset_schedule,ensure_manifest,
                                      digest,sha256_file,atomic_json,seed_all,worker_init)
    from inference_metric_grasp_field import atomic_npy
    seed_all(0)
    source=FrozenSource(a.source_checkpoint,'cuda:0')
    control,ck=load_control(source,a.control_checkpoint)
    protocol_train=ck['protocol']; epoch=ck['epoch']
    if protocol_train.get('partial_run',True):
        raise ValueError('Formal inference rejects partial P1 training')
    if abs(float(a.eval_fraction)-float(protocol_train['eval_fraction']))>1e-12:
        raise ValueError('Evaluation fraction differs from P1 training protocol')
    family,variant=protocol_train['family'],protocol_train['variant']
    control.eval()

    from dataset.graspnet_dataset import collate_fn
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    full,_,indices=make_dataset(a.dataset_root,a.split,a.eval_fraction,labels=False,max_frames=a.max_frames)
    schedule=dataset_schedule(full,indices)
    modes=['off','on'] if a.collision=='both' else [a.collision]
    roots={m:Path(a.output_root)/f'test_collision_{m}' for m in modes}
    manifests={}; locks=[]
    import fcntl
    for mode,root in roots.items():
        training=copy.deepcopy(source.protocol)
        training['eval_fraction']=a.eval_fraction
        training['partial_run']=bool(a.max_frames or protocol_train.get('partial_run',False))
        training['p1_finetuning']=protocol_train
        collision='none' if mode=='off' else dict(
            type='model_free_original_sensor',threshold=a.collision_thresh,
            voxel_size_m=a.voxel_size,approach_dist_m=a.approach_dist,network_input=False)
        manifest=dict(
            version=VERSION,experiment_version=P1_VERSION,split=a.split,schedule=schedule,
            code_sha256=code_digest(),checkpoint_sha256=source.checkpoint_sha,
            checkpoint_epoch=epoch,source_checkpoint_epoch=source.epoch,
            control_sha256=sha256_file(a.control_checkpoint),training_protocol=training,
            score_source=f'p1_{family}_{variant}',collision_filter=collision,
            max_frames=a.max_frames,evaluation_fraction=a.eval_fraction,batch_size=a.batch_size,
            num_shards=a.num_shards,seed=0,
            preprocessing='Original GraspNet crop/workspace; sensor depth never enters network')
        ensure_manifest(root/a.split/'protocol.json',manifest)
        lock=open(root/a.split/f'.shard_{a.shard_id}.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(lock)
        manifests[mode]=(manifest,digest(manifest))

    def paths(root,sid,ann):
        return (root/'dump'/f'scene_{sid:04d}'/'realsense'/f'{ann:04d}.npy',
                root/a.split/'completed'/f'{sid:04d}_{ann:04d}.json')

    pending=[]
    for pos,(idx,(sid,ann)) in enumerate(zip(indices,schedule)):
        if pos%a.num_shards!=a.shard_id: continue
        complete=True
        for mode,root in roots.items():
            path,mark=paths(root,sid,ann)
            if not a.resume and (path.exists() or mark.exists()): raise FileExistsError(path)
            ok=False
            if mark.is_file():
                row=json.loads(mark.read_text())
                if row['signature']!=manifests[mode][1]: raise RuntimeError(f'Stale marker {mark}')
                ok=path.is_file() and sha256_file(path)==row['output_sha256']
            complete &= ok
        if not complete: pending.append(idx)

    loader=DataLoader(Subset(full,pending),batch_size=a.batch_size,num_workers=a.workers,
                      collate_fn=collate_fn,shuffle=False,worker_init_fn=worker_init)
    if 'on' in modes:
        from graspnetAPI import GraspGroup
        from utils.collision_detector import ModelFreeCollisionDetectorTorch
    cursor=0
    for raw in loader:
        ctx=source(safe_batch(raw,'cuda:0',False))
        ctx=prepare_context(source,ctx,family,variant)
        ep=dict(ctx['ep']); ep['grasp_cdf_pred_angle_depth']=control(ctx)
        preds=pred_decode_center_view_angle(ep,use_cdf=True)
        for pred in preds:
            idx=pending[cursor]; sid,ann=dataset_schedule(full,[idx])[0]; cursor+=1
            original=pred.float().cpu().numpy()
            if original.ndim!=2 or original.shape[1]!=17 or not np.isfinite(original).all():
                raise RuntimeError('Invalid decoded grasps')
            arrays={'off':original}
            if 'on' in modes:
                cloud,_=full.get_data(idx,return_raw_cloud=True)
                detector=ModelFreeCollisionDetectorTorch(np.asarray(cloud,np.float32).reshape(-1,3),voxel_size=a.voxel_size)
                gg=GraspGroup(original)
                collision=detector.detect(gg,approach_dist=a.approach_dist,collision_thresh=a.collision_thresh)
                arrays['on']=gg[~collision.cpu().numpy()].grasp_group_array.astype(np.float32,copy=False)
            for mode,root in roots.items():
                path,mark=paths(root,sid,ann); arr=arrays[mode]
                if a.resume and mark.is_file() and path.is_file():
                    saved=json.loads(mark.read_text())
                    if saved['signature']==manifests[mode][1] and sha256_file(path)==saved['output_sha256']:
                        prior=np.load(path,allow_pickle=False)
                        if prior.shape!=arr.shape or not np.allclose(prior,arr,atol=1e-6,rtol=1e-6):
                            raise RuntimeError('Resume re-forward changed dump')
                        continue
                atomic_npy(path,arr)
                atomic_json(mark,dict(signature=manifests[mode][1],output_sha256=sha256_file(path),
                                      grasps=len(arr),grasps_before_collision=len(original),
                                      grasps_after_collision=len(arr)))
        del ctx,ep

    for mode,root in roots.items():
        atomic_json(root/a.split/f'shard_{a.shard_id}.json',
                    dict(signature=manifests[mode][1],written=cursor))


if __name__=='__main__': main()
