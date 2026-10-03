#!/usr/bin/env python3
"""Native CVA-CDF decoding for online MoGe/RayRoPE models; no feature cache."""
from __future__ import annotations
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from moge_rayrope.config import VERSION


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset-root',default='/data/robotarm/dataset/graspnet')
    p.add_argument('--output-root',required=True)
    p.add_argument('--split',choices=['test_seen','test_similar','test_novel'],required=True)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--shard-id',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--collision',choices=['on','off','both'],default='both')
    p.add_argument('--collision-thresh',type=float,default=.01)
    p.add_argument('--voxel-size',type=float,default=.01)
    p.add_argument('--approach-dist',type=float,default=.05)
    p.add_argument('--depth-bias-mm',type=float,default=0.,help='Diagnostic only: bias predicted numeric geometry before seed/view/group; no sensor input')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    return p


class SeededIndices(Dataset):
    """Frame-indexed CPU sampling, independent of worker/shard/resume schedule."""
    def __init__(self,base,indices): self.base=base; self.indices=indices
    def __len__(self): return len(self.indices)
    def __getitem__(self,pos):
        import random
        idx=self.indices[pos]; pr=random.getstate(); nr=np.random.get_state(); tr=torch.get_rng_state()
        try:
            seed=34153+idx; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            return self.base[idx]
        finally: random.setstate(pr); np.random.set_state(nr); torch.set_rng_state(tr)


@torch.no_grad()
def main():
    a=parser().parse_args(); sys.argv=[sys.argv[0]]
    if not torch.cuda.is_available(): raise RuntimeError('CUDA/GraspNet required')
    if min(a.batch_size,a.num_shards)<1 or min(a.workers,a.max_frames)<0 or not 0<=a.shard_id<a.num_shards:
        raise ValueError('Invalid inference counts')
    if not np.isfinite(a.depth_bias_mm) or abs(a.depth_bias_mm)>100:
        raise ValueError('Depth diagnostic bias must be finite and within +/-100 mm')
    if any(not np.isfinite(x) or x<=0 for x in (a.collision_thresh,a.voxel_size,a.approach_dist)):
        raise ValueError('Invalid collision configuration')
    from moge_rayrope.runtime import (load_model,make_dataset,move_batch,sha256_file,code_fingerprint,
        digest,ensure_manifest,atomic_json,atomic_npy,seed_all,worker_init)
    seed_all(0)
    model,ck=load_model(a.checkpoint,'cuda:0')
    p=ck['protocol']
    if p['partial_run']: raise ValueError('Use a complete trained checkpoint, not a smoke checkpoint')
    if int(ck['epoch'])+1!=int(p['epochs']): raise ValueError('Formal inference requires the completed fixed training budget')
    if code_fingerprint()!=p['code_sha256']: raise RuntimeError('Model code changed since training; audit before using this checkpoint')
    model.depth_bias_m=a.depth_bias_mm*.001
    base,_,indices,schedule=make_dataset(a.dataset_root,a.split,p['eval_fraction'],False,model.cfg,
        p['label_folder'],False,a.max_frames)
    from dataset.graspnet_dataset import collate_fn
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    modes=['on','off'] if a.collision=='both' else [a.collision]
    roots={mode:Path(a.output_root)/f'test_collision_{mode}' for mode in modes}
    manifests={}; signatures={}; locks=[]
    import fcntl
    checkpoint_sha=sha256_file(a.checkpoint)
    for mode,root in roots.items():
        collision='none' if mode=='off' else dict(type='model_free_original_sensor',threshold=a.collision_thresh,
            voxel_size_m=a.voxel_size,approach_dist_m=a.approach_dist,network_input=False)
        m=dict(version=VERSION,checkpoint_sha256=checkpoint_sha,checkpoint_epoch=ck['epoch'],
            code_sha256=code_fingerprint(),training_protocol=p,split=a.split,schedule=schedule,
            evaluation_fraction=p['eval_fraction'],dataset_root=str(Path(a.dataset_root).resolve()),
            batch_size=a.batch_size,num_shards=a.num_shards,seed=0,max_frames=a.max_frames,
            collision_filter=collision,primary_collision='on',depth_bias_mm=a.depth_bias_mm,
            score_source='native_CVA_CDF',top_view_count=1,
            preprocessing='Original GraspNet crop/workspace; annotation-independent cropping is NOT claimed',
            diagnostic='native' if a.depth_bias_mm==0 else 'predicted_numeric_depth_bias_before_seed_view_group')
        ensure_manifest(root/a.split/'protocol.json',m)
        f=open(root/a.split/f'.shard_{a.shard_id}.lock','a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(f)
        manifests[mode]=m; signatures[mode]=digest(m)
    assigned=[pos for pos in range(len(indices)) if pos%a.num_shards==a.shard_id]
    shard_indices=[indices[pos] for pos in assigned]
    def paths(mode,sid,aid):
        r=roots[mode]
        return r/'dump'/f'scene_{sid:04d}'/'realsense'/f'{aid:04d}.npy',r/a.split/'completed'/f'{sid:04d}_{aid:04d}.json'
    def complete(mode,sid,aid):
        path,mark=paths(mode,sid,aid)
        if mark.is_file():
            row=json.loads(mark.read_text())
            if row['signature']!=signatures[mode]: raise RuntimeError(f'Stale output marker {mark}')
            return path.is_file() and sha256_file(path)==row['output_sha256']
        return False
    for pos in assigned:
        sid,aid=schedule[pos]
        for mode in modes:
            path,mark=paths(mode,sid,aid)
            if not a.resume and (path.exists() or mark.exists()): raise FileExistsError(path)
    loader=DataLoader(SeededIndices(base,shard_indices),batch_size=a.batch_size,shuffle=False,
                      num_workers=a.workers,collate_fn=collate_fn,worker_init_fn=worker_init)
    if 'on' in modes:
        from graspnetAPI import GraspGroup
        from utils.collision_detector import ModelFreeCollisionDetectorTorch
    cursor=0; written=0; batches=0; timed_frames=0; forward_seconds=0.; collision_seconds=0.
    torch.cuda.reset_peak_memory_stats()
    wall_start=time.monotonic()
    for raw in loader:
        n=raw['img'].shape[0]; batch_positions=assigned[cursor:cursor+n]; cursor+=n
        # Keep original batch boundaries when resuming, including partial batches.
        if a.resume and all(complete(mode,*schedule[pos]) for pos in batch_positions for mode in modes): continue
        seed_all(73331+indices[batch_positions[0]])
        inputs=move_batch(raw,'cuda:0',False)
        torch.cuda.synchronize(); forward_start=time.monotonic()
        ep=model(inputs)
        preds=pred_decode_center_view_angle(ep,use_cdf=True)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-forward_start
        if batches>0: forward_seconds+=elapsed; timed_frames+=n
        batches+=1
        for j,pred in enumerate(preds):
            pos=batch_positions[j]; sid,aid=schedule[pos]; idx=indices[pos]
            arr=pred.detach().float().cpu().numpy()
            if arr.ndim!=2 or arr.shape[1]!=17 or not np.isfinite(arr).all(): raise RuntimeError('Malformed grasp dump')
            arrays={'off':arr}
            if 'on' in modes:
                collision_start=time.monotonic()
                cloud,_=base.get_data(idx,return_raw_cloud=True)
                detector=ModelFreeCollisionDetectorTorch(np.asarray(cloud,np.float32).reshape(-1,3),voxel_size=a.voxel_size)
                gg=GraspGroup(arr)
                coll=detector.detect(gg,approach_dist=a.approach_dist,collision_thresh=a.collision_thresh).cpu().numpy()
                arrays['on']=arr[~coll]
                if batches>1: collision_seconds+=time.monotonic()-collision_start
            for mode in modes:
                out=arrays[mode]; path,mark=paths(mode,sid,aid)
                if a.resume and complete(mode,sid,aid):
                    old=np.load(path,allow_pickle=False)
                    if old.shape!=out.shape or not np.allclose(old,out,rtol=1e-6,atol=1e-6):
                        raise RuntimeError('Resumed paired forward differs: use fresh output root')
                    continue
                atomic_npy(path,out)
                atomic_json(mark,dict(signature=signatures[mode],output_sha256=sha256_file(path),
                    grasps_before_collision=len(arr),grasps_after_collision=len(out)))
            written+=1
        if written%20==0: print(f'[MR INFER] {a.split} shard={a.shard_id} processed={cursor}/{len(assigned)}',flush=True)
        del ep,preds
    for mode,root in roots.items():
        atomic_json(root/a.split/f'shard_{a.shard_id}.json',dict(signature=signatures[mode],assigned=len(assigned),written=written,
            timing=dict(batch_size=a.batch_size,warmup_batches=1,timed_frames=timed_frames,
                forward_decode_seconds=forward_seconds,collision_raw_cloud_seconds=collision_seconds,
                forward_decode_ms_per_frame=1000*forward_seconds/timed_frames if timed_frames else None,
                wall_seconds_including_io=time.monotonic()-wall_start),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))

if __name__=='__main__': main()
