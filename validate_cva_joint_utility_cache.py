#!/usr/bin/env python3
"""Validate exact-evaluator CVA joint-utility cache files before training."""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

REQ=(
 'angle_feature','score_logits','depth_logits','width_raw','angle_id','depth_id',
 'view_rank','view_score','center_xyz','view_xyz','friction','pure_collision',
 'empty','center_id','assigned_obj','legacy_score','legacy_quality','scene_id','anno_id'
)
ALLOWED=np.asarray([-1.0,0.2,0.4,0.6,0.8,1.0,1.2],np.float32)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cache_dir',required=True); ap.add_argument('--max_files',type=int,default=0); ap.add_argument('--strict_friction_grid',type=int,default=1); args=ap.parse_args()
 files=sorted(glob.glob(os.path.join(args.cache_dir,'**','*.npz'),recursive=True))
 if args.max_files>0: files=files[:args.max_files]
 if not files: raise FileNotFoundError(f'No npz cache under {args.cache_dir}')
 bad=0; total=valid=safe08=collision=empty=0; dims=set(); scenes=set()
 for path in files:
  try:
   with np.load(path,allow_pickle=False) as d:
    missing=[k for k in REQ if k not in d]
    if missing: raise KeyError(f'missing {missing}')
    n=len(d['friction'])
    for k in REQ:
     if k in {'scene_id','anno_id'}: continue
     if len(d[k])!=n: raise ValueError(f'{k} length={len(d[k])}, expected {n}')
    if d['angle_feature'].ndim!=2: raise ValueError('angle_feature must be [N,C]')
    if d['score_logits'].shape!=(n,6): raise ValueError(f'score_logits={d["score_logits"].shape}')
    if d['depth_logits'].ndim!=2: raise ValueError('depth_logits must be [N,D+1]')
    for k in ('angle_feature','score_logits','depth_logits','width_raw','center_xyz','view_xyz','legacy_score'):
     if not np.isfinite(d[k]).all(): raise ValueError(f'{k} has NaN/Inf')
    f=np.asarray(d['friction'],np.float32)
    if int(args.strict_friction_grid):
     ok=np.min(np.abs(f[:,None]-ALLOWED[None]),axis=1)<5e-3
     if not ok.all(): raise ValueError(f'unexpected friction {np.unique(f[~ok])[:10]}')
    if np.any(d['pure_collision'].astype(bool)&d['empty'].astype(bool)):
     raise ValueError('pure_collision and empty overlap')
    total+=n; valid+=int((f>0).sum()); safe08+=int(((f>0)&(f<=.8)).sum()); collision+=int(d['pure_collision'].sum()); empty+=int(d['empty'].sum())
    dims.add((d['angle_feature'].shape[-1],d['depth_logits'].shape[-1])); scenes.add(int(np.asarray(d['scene_id']).reshape(-1)[0]))
   print(f'[OK] {os.path.relpath(path,args.cache_dir)} N={n}')
  except Exception as exc:
   bad+=1; print(f'[BAD] {path}: {exc}',file=sys.stderr)
 summary={'files':len(files),'bad':bad,'scenes':len(scenes),'candidate_dims':sorted(dims),'candidates':total,'valid_ratio':valid/max(total,1),'safe08_ratio':safe08/max(total,1),'pure_collision_ratio':collision/max(total,1),'empty_ratio':empty/max(total,1)}
 print(json.dumps(summary,indent=2))
 raise SystemExit(1 if bad else 0)
if __name__=='__main__': main()
