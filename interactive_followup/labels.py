"""Official CAD/Dex-Net label extension for *held-out Stage-1 validation scenes only*.

No test labels, no network updates. Full labels enable an independent
NMS/top-k validation *proxy*, which is not the official test AP.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from interactive_grasp.contracts import (load_signed,signed,atomic_json,atomic_npz,sha256,
                                         physical_ids,frame_key)
from interactive_grasp.data import load_index,load_frame


def validate_index(idx):
    if idx['partition']!='validation' or any(e['row']['scene']>=100 or e['row']['split']!='train' for e in idx['entries']):
        raise ValueError('Full labels only allowed on training-domain held-out validation scenes')
    if not idx['complete']:raise ValueError('Formal full-label validation requires complete index')


def label_one(e,index,out,evaluator,source_hash,resume=False):
    row=e['row'];key=frame_key(row)
    path=out/(key+'.npz');meta=path.with_suffix('.json')
    contract=dict(frame=row,index_sha256=index['fingerprint'],source=index['contract']['source'],
                  source_frame_sha256=e['sha256'],evaluator_source_sha256=source_hash,
                  evaluator_mode='official',purpose='validation_quality_proxy_only')
    if path.exists() or meta.exists():
        if not resume or not(path.exists() and meta.exists()):raise FileExistsError(f'Incomplete/existing labels: {path}')
        saved=load_signed(meta)
        if saved['contract']!=contract or sha256(path)!=saved['npz_sha256']:raise ValueError('Full-label cache changed since creation')
        return saved
    frame=load_frame(e,index['contract']['expected_queries'])
    g=frame['grasps'];ids=physical_ids(g)
    ev=evaluator.evaluate(row['scene'],row['anno'],g)
    friction=np.round(np.asarray(ev.friction,dtype=np.float64),2)
    bad=np.asarray(ev.collision_or_empty,dtype=bool)
    valid=(friction[:,None]>0)&(friction[:,None]<=np.asarray([.2,.4,.6,.8,1.,1.2])[None])&(~bad[:,None])
    count=valid.sum(-1).astype(np.int64)
    mask=frame['label_mask'];old=frame['quality_count']
    if not np.array_equal(count[mask],old[mask]):
        raise RuntimeError(f'Existing sparse exact labels differ from full re-evaluation at {key}')
    arrays=dict(physical_id=ids,quality_count=count,friction=friction.astype(np.float32),
                collision_or_empty=bad,assigned_obj=np.asarray(ev.assigned_obj,dtype=np.int64))
    atomic_npz(path,arrays)
    receipt=signed(dict(contract=contract,npz_sha256=sha256(path),frame_key=key,
                        num_grasps=len(g),num_sparse_verified=int(mask.sum()),
                        quality_mean=float(count.mean()/6),stats=ev.stats))
    atomic_json(meta,receipt)
    return receipt


def label(a):
    index=load_index(a.index,'validation',False);validate_index(index)
    if a.shards<1 or not 0<=a.shard<a.shards:raise ValueError('Bad shard parameters')
    entries=[e for i,e in enumerate(index['entries']) if i%a.shards==a.shard]
    if a.limit:entries=entries[:a.limit]
    if not entries:raise ValueError('No validation frames in selected shard')
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    import exact_action_graspnet_evaluator
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    src_hash=sha256(exact_action_graspnet_evaluator.__file__)
    evaluator=ExactGraspNetActionEvaluator(a.dataset_root,'realsense',split='train',fc_mode='official',strict=True)
    last_scene=None
    for i,e in enumerate(entries,1):
        scene=e['row']['scene']
        if scene!=last_scene:evaluator.scene_cache.clear();last_scene=scene
        rec=label_one(e,index,root,evaluator,src_hash,a.resume)
        print(f'[full-val-label] {i}/{len(entries)} {rec["frame_key"]} Q={rec["quality_mean"]:.4f}',flush=True)
    atomic_json(root/f'shard_{a.shard:03d}.json',signed(dict(index_sha256=index['fingerprint'],
               shard=a.shard,shards=a.shards,frames=len(entries),smoke_only=bool(a.limit),evaluator_sha256=src_hash)))


def read_full_labels(entry,index,root):
    row=entry['row'];key=frame_key(row)
    path=Path(root)/(key+'.npz');meta=path.with_suffix('.json')
    if not path.is_file() or not meta.is_file():raise FileNotFoundError(f'Full validation exact labels missing: {path}')
    rec=load_signed(meta)
    if (rec['contract']['frame']!=row or rec['contract']['index_sha256']!=index['fingerprint']
        or rec['contract']['source']!=index['contract']['source']
        or rec['contract']['source_frame_sha256']!=entry['sha256']
        or sha256(path)!=rec['npz_sha256']):raise ValueError(f'Full quality label provenance mismatch: {path}')
    with np.load(path,allow_pickle=False) as f:
        dat={k:f[k] for k in f.files}
    if dat['quality_count'].shape!=(index['contract']['expected_queries'],) or not np.array_equal(dat['physical_id'],physical_ids(load_frame(entry,index['contract']['expected_queries'])['grasps'])):
        raise ValueError(f'Full exact labels not aligned with executed actions: {path}')
    return dat


def verify(a):
    index=load_index(a.index,'validation',False);validate_index(index)
    n=0;sources=set()
    for e in index['entries']:
        data=read_full_labels(e,index,a.output)
        rec=load_signed(Path(a.output)/(frame_key(e['row'])+'.json'))
        sources.add(rec['contract']['evaluator_source_sha256']);n+=len(data['quality_count'])
    if len(sources)!=1:raise ValueError('Multiple evaluator versions in full validation labels')
    manifest=signed(dict(status='complete',partition='validation',index_sha256=index['fingerprint'],
        source=index['contract']['source'],n_frames=len(index['entries']),n_actions=n,
        evaluator_sha256=next(iter(sources))))
    p=Path(a.output)/'full_label_manifest.json'
    if p.exists():
        if load_signed(p)!=manifest:raise ValueError('Existing full-label manifest differs')
    else:atomic_json(p,manifest)
    print(json.dumps(manifest,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    label_p=sub.add_parser('label');label_p.add_argument('--index',required=True)
    label_p.add_argument('--dataset-root',required=True);label_p.add_argument('--output',required=True)
    label_p.add_argument('--shard',type=int,default=0);label_p.add_argument('--shards',type=int,default=1)
    label_p.add_argument('--limit',type=int,default=0);label_p.add_argument('--resume',action='store_true')
    verify_p=sub.add_parser('verify');verify_p.add_argument('--index',required=True)
    verify_p.add_argument('--output',required=True)
    a=p.parse_args();label(a) if a.stage=='label' else verify(a)

if __name__=='__main__':main()