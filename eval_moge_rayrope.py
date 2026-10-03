#!/usr/bin/env python3
"""Official GraspNet AP with exact dump/protocol checks; smoke results rejected."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from moge_rayrope.runtime import VERSION,digest,sha256_file,atomic_json,atomic_npy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',default='/data/robotarm/dataset/graspnet')
    p.add_argument('--inference-root',required=True)
    p.add_argument('--split',choices=['test_seen','test_similar','test_novel'],required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); sys.argv=[sys.argv[0]]
    if a.workers<1: raise ValueError('workers must be positive')
    root=Path(a.inference_root); protocol=json.loads((root/a.split/'protocol.json').read_text())
    if protocol['version']!=VERSION or protocol['split']!=a.split: raise ValueError('Wrong inference protocol')
    if protocol['max_frames'] or protocol['training_protocol']['partial_run']:
        raise ValueError('Formal evaluator refuses smoke/partial data')
    hashes=[]
    for sid,aid in protocol['schedule']:
        path=root/'dump'/f'scene_{sid:04d}'/'realsense'/f'{aid:04d}.npy'
        marker=root/a.split/'completed'/f'{sid:04d}_{aid:04d}.json'
        row=json.loads(marker.read_text()); sha=sha256_file(path)
        if row['signature']!=digest(protocol) or row['output_sha256']!=sha: raise RuntimeError(f'Stale dump {path}')
        hashes.append(sha)
    sig=digest({'protocol':protocol,'dumps':hashes}); out=root/'official'/a.split; out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with open(out/'.eval.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').exists():
            old=json.loads((out/'summary.json').read_text())
            if not a.resume: raise FileExistsError('Existing result: use --resume')
            if old['signature']!=sig: raise RuntimeError('Evaluation signature changed')
            return
        from graspnetAPI import GraspNetEval
        evaluator=GraspNetEval(a.dataset_root,camera='realsense',split=a.split)
        fn=getattr(evaluator,{'test_seen':'eval_seen','test_similar':'eval_similar','test_novel':'eval_novel'}[a.split])
        fraction=protocol['evaluation_fraction']
        result,reported=fn(str(root/'dump'),anno_sample_ratio=fraction,proc=a.workers)
        accuracy=np.asarray(result,dtype=np.float64)
        expected=(30,len(range(0,256,round(1/fraction))),50,6)
        if accuracy.shape!=expected or not np.isfinite(accuracy).all(): raise RuntimeError(f'Unexpected accuracy shape {accuracy.shape}')
        atomic_npy(out/'accuracy.npy',accuracy)
        atomic_json(out/'summary.json',dict(signature=sig,split=a.split,shape=list(accuracy.shape),
            mean_accuracy=float(accuracy.mean()),reported_ap=np.asarray(reported).tolist(),
            checkpoint_epoch=protocol['checkpoint_epoch'],collision_filter=protocol['collision_filter'],
            depth_bias_mm=protocol['depth_bias_mm'],model=protocol['training_protocol']['model']))
        print(f'[MR AP] {a.split}: mean_accuracy={float(accuracy.mean()):.8f}; reported={reported}',flush=True)

if __name__=='__main__': main()
