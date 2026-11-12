"""Streaming full-query inference, frozen-geometry dumps and official AP wrapper."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .contracts import (SCHEMA,load_signed,validate_plan,frame_key,signed,atomic_json,
                        atomic_npz,physical_ids,sha256,digest,parameter_source_hash)
from .export import get_runtime
from .data import tensor_frame
from .train import load_refiner


def load_models(items,device,source,allow_smoke=False):
    models={};protocols={};paths={};common=None
    for item in items:
        name,path=item.split('=',1)
        if name not in ('A','B','C','D') or name in models:raise ValueError('Unique A/B/C/D=checkpoint required')
        model,p=load_refiner(path,device,allow_smoke)
        if p['model']['variant']!=name or p['source']!=source:raise ValueError('Frozen source/model identity mismatch')
        normalized=dict(p);normalized.pop('model_description')
        normalized['model']=dict(p['model']);normalized['model'].pop('variant')
        if common is None:common=normalized
        if normalized!=common:raise ValueError('A/B/C/D data, seed, training budget or common dimensions differ')
        models[name]=model;protocols[name]=p;paths[name]=dict(path=str(Path(path).resolve()),sha256=sha256(path))
    if not models:raise ValueError('No models supplied')
    return models,protocols,paths


@torch.inference_mode()
def score_frame(models,protocols,arrays,device):
    inputs,_,_=tensor_frame(arrays,device)
    outputs={'native':arrays['grasps'][:,0].copy()}
    for name,model in models.items():
        dtype=protocols[name]['memory_dtype']
        # Match any declared cache quantization exactly; labels and physical
        # grasps remain FP32 and are never rounded for matching.
        inputs['memory']=torch.from_numpy(arrays['memory'].astype(dtype).astype(np.float32)).unsqueeze(0).to(device)
        score=model(**inputs).sigmoid()[0].cpu().numpy()
        if not np.isfinite(score).all():raise FloatingPointError('Nonfinite score')
        outputs[name]=score
    return outputs


def emit_frame(root,row,arrays,scores):
    records=[];keep=arrays['sensor_keep'];original=arrays['grasps'];ids=physical_ids(original)
    for name,score in scores.items():
        g=original.copy();g[:,0]=score
        if not np.array_equal(physical_ids(g),ids):raise RuntimeError('A score update changed executed geometry')
        g=g[keep]
        out=root/name/f"scene_{row['scene']:04d}"/'realsense'/f"{row['anno']:04d}.npy"
        out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():raise FileExistsError(out)
        np.save(out,g)
        records.append(dict(row=row,variant=name,path=str(out.relative_to(root)),sha256=sha256(out),
                            queries=len(ids),survivors=int(keep.sum())))
    atomic_npz(root/'traces'/(frame_key(row)+'.npz'),dict(
        grasps=original,physical_id=ids,sensor_keep=keep,
        **{f'score_{name}':v for name,v in scores.items()}))
    return records


def infer(a):
    plan=load_signed(a.plan);validate_plan(plan)
    rows=[r for r in plan['test'] if r['domain']==a.domain and r['split']==a.split]
    if a.limit:rows=rows[:a.limit]
    if not rows:raise ValueError('No planned test frames for domain/split')
    if not a.limit and not a.reference_root:
        raise ValueError('Formal run requires --reference-root to verify the existing frozen-MSE dumps')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    runtime=get_runtime(a)
    models,protocols,paths=load_models(a.weight,a.device,runtime.provenance,bool(a.limit))
    records=[];max_reference_error=0.
    for step,row in enumerate(rows):
        arrays,parity=runtime.snapshot(row,a.seed)
        for field in ('native_decode_max_abs','native_reader_replay_max_abs'):
            if field not in parity or not np.isfinite(parity[field]) or parity[field]>1e-6:
                raise RuntimeError('Frozen provider native replay failed')
        n=len(arrays['grasps'])
        if n!=1024:raise ValueError('Formal/full snapshot must contain 1024 queries')
        arrays['sensor_keep']=runtime.sensor_keep(row,arrays['grasps']).astype(bool)
        arrays['label_mask']=np.zeros(n,dtype=bool);arrays['quality_count']=np.full(n,-1,dtype=np.int64)
        if a.reference_root:
            path=Path(a.reference_root)/f"scene_{row['scene']:04d}"/'realsense'/f"{row['anno']:04d}.npy"
            old=np.load(path,allow_pickle=False);now=arrays['grasps'][arrays['sensor_keep']]
            if old.shape!=now.shape:raise RuntimeError(f'Source MSE dump count mismatch: {path}')
            err=float(np.max(np.abs(old-now))) if len(now) else 0.
            if not np.isfinite(err) or err>1e-6:raise RuntimeError(f'Frozen MSE dump parity failed: {path}: {err}')
            max_reference_error=max(max_reference_error,err)
        scores=score_frame(models,protocols,arrays,a.device)
        records.extend(emit_frame(root,row,arrays,scores))
        print(f'[infer] {step+1}/{len(rows)} {frame_key(row)} fixed={n}',flush=True)
    atomic_json(root/'inference_manifest.json',signed(dict(schema=SCHEMA,status='complete',
        plan_sha256=plan['fingerprint'],source=runtime.provenance,source_code_sha256=parameter_source_hash(),
        split=a.split,domain=a.domain,seed=a.seed,frames=len(rows),partial_smoke=bool(a.limit),
        source_reference_root=str(a.reference_root),source_reference_max_abs=max_reference_error,
        variants=['native',*models],weights=paths,protocols=protocols,records=records)))


def evaluate(a):
    root=Path(a.inference);m=load_signed(root/'inference_manifest.json')
    if m['partial_smoke'] or m['frames']!=7680:raise ValueError('Official eval requires 30 scenes x256 frames')
    starts={'test_seen':100,'test_similar':130,'test_novel':160};start=starts[m['split']]
    expected={(s,i) for s in range(start,start+30) for i in range(256)}
    for name in m['variants']:
        rows=[r for r in m['records'] if r['variant']==name]
        if len(rows)!=7680 or {(r['row']['scene'],r['row']['anno']) for r in rows}!=expected:
            raise ValueError('Incomplete/duplicate test scene/frame grid')
        for r in rows:
            if sha256(root/r['path'])!=r['sha256']:raise ValueError('Dump hash mismatch')
        subprocess.run([sys.executable,'eval.py','--dataset_root',a.dataset_root,'--dump_dir',str(root/name),
                        '--camera','realsense','--split',m['split'],'--sample_interval','1',
                        '--num_workers',str(a.workers)],check=True)
        path=root/name/f"ap_{m['split']}_realsense.npy"
        arr=np.load(path,allow_pickle=False)
        if arr.shape!=(30,256,50,6) or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:
            raise RuntimeError('Official AP contract failed')
    atomic_json(root/'evaluation_complete.json',signed(dict(schema=SCHEMA,status='complete',
        inference_sha256=m['fingerprint'],split=m['split'],domain=m['domain'],
        methods={name:dict(sha256=sha256(root/name/f"ap_{m['split']}_realsense.npy"),
                           mean_ap_percent=float(np.load(root/name/f"ap_{m['split']}_realsense.npy").mean()*100))
                 for name in m['variants']})))
