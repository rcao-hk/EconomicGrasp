#!/usr/bin/env python3
"""Run the repository's sampled GraspNet API protocol on full-query dumps.

Rejects pilot/subsampled-query runs and missing frames. Raw exact-action Top-K
metrics are deliberately NOT called AP. Requires the API's anno_sample_ratio
extension already used by this repository's eval.py.
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from rep_a_common import SPLITS,atomic_file,save_json,exclusive_run,digest,file_sha
from rep_fullpath_runtime import case_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True); p.add_argument('--work-root',required=True)
    p.add_argument('--split',choices=tuple(SPLITS)[1:],required=True)
    p.add_argument('--frame-stride',type=int,default=10)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    root=Path(args.work_root); protocol=json.loads((root/'protocol.json').read_text())
    if protocol['query_limit']!=0:
        raise RuntimeError('Official full-query comparison requires QUERY_LIMIT=0; use a NEW WORK_ROOT and rerun infer. Pilot metrics remain useful, but are not formal AP.')
    if args.frame_stride<1 or args.workers<1: raise ValueError('Invalid stride/workers')
    from graspnetAPI import GraspNetEval
    ge=GraspNetEval(root=args.dataset_root,camera=protocol['camera'],split=args.split)
    evaluate=getattr(ge,{'test_seen':'eval_seen','test_similar':'eval_similar','test_novel':'eval_novel'}[args.split])
    if 'anno_sample_ratio' not in inspect.signature(evaluate).parameters:
        raise RuntimeError('Installed graspnetAPI lacks anno_sample_ratio; use the same API as repo eval.py. Do not silently evaluate absent frames.')
    folders=sorted((root/'dump').glob(f'*/*/*/*/{args.split}'))
    if not folders: raise FileNotFoundError('No inference dumps')
    for folder in folders:
        method,policy,mode,case=folder.parts[-5:-1]
        expected=[folder/f'scene_{sid:04d}'/protocol['camera']/f'{aid:04d}.npy'
                  for sid in range(*SPLITS[args.split]) for aid in range(0,256,args.frame_stride)]
        missing=[str(x) for x in expected if not x.is_file()]
        if missing: raise FileNotFoundError(f'Incomplete formal dump: {missing[:3]} ({len(missing)} missing)')
        sig=digest(dict(protocol=protocol,split=args.split,stride=args.frame_stride,
                        dumps=[(str(x.relative_to(root)),file_sha(x)) for x in expected]))
        dest=root/'official'/method/policy/mode/case/args.split; dest.mkdir(parents=True,exist_ok=True)
        with exclusive_run(dest/'.lock'):
            if args.resume and (dest/'summary.json').exists():
                old=json.loads((dest/'summary.json').read_text())
                if old['signature']!=sig: raise RuntimeError('Official AP dump/protocol changed')
                continue
            res,ap=evaluate(str(folder),anno_sample_ratio=1./args.frame_stride,proc=args.workers)
            arr=np.asarray(res)
            with atomic_file(dest/'accuracy.npy') as f: np.save(f,arr,allow_pickle=False)
            save_json(dest/'summary.json',dict(signature=sig,method=method,policy=policy,mode=mode,case=case,
                split=args.split,seen_role='validation_seen' if args.split=='test_seen' else 'held_out',
                frame_stride=args.frame_stride,num_frames=len(expected),reported_ap=np.asarray(ap).tolist(),
                accuracy_shape=list(arr.shape),mean_accuracy=float(arr.mean()),
                protocol='Unfiltered inference dumps -> existing GraspNetEval API (its standard NMS/collision/force-closure)'))
            print(f'[OFFICIAL AP] {method}/{policy}/{mode}/{case}/{args.split}: {ap}',flush=True)


if __name__=='__main__': main()
