"""Optional test inference with alphas locked from held-out validation only.

No test labels are read or allowed; frozen actions and collision mask preserved.
Uses unchanged production GraspNet eval from interactive_grasp.cli eval.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from interactive_grasp.contracts import load_signed,validate_plan,sha256,physical_ids,signed,atomic_json,parameter_source_hash
from interactive_grasp.export import get_runtime
from interactive_grasp.infer import load_models,emit_frame
from interactive_grasp.data import tensor_frame
from .core import mix_logits


def locked_choices(path,source,weight_paths):
    sweep=load_signed(path)
    if sweep.get('partition')!='validation' or sweep.get('status')!='complete' or not sweep.get('full_exact_labels'):
        raise ValueError('Only complete all-action held-out validation scan can lock test alphas')
    if sweep.get('source')!=source:raise ValueError('Locked alphas belong to different frozen base')
    for name in 'ABCD':
        entry=sweep.get('weight_files',{}).get(name)
        if not entry or sha256(weight_paths[name])!=entry['sha256']:
            raise ValueError(f'{name}: trained scorer changed since validation alpha lock')
    if set(sweep.get('choices',{}))!=set(('A','B','C','D','D_pre_read')):
        raise ValueError('Incomplete locked alpha decisions')
    alphas={v:float(sweep['choices'][v]['alpha']) for v in 'ABCD'}
    for v in alphas:
        if sweep['choices'][v]['source']!='validation_only':raise ValueError('Test-derived alpha is forbidden')
    return sweep,alphas


@torch.inference_mode()
def infer(a):
    plan=load_signed(a.plan);validate_plan(plan)
    rows=[r for r in plan['test'] if r['domain']=='original' and r['split']==a.split]
    if a.limit:rows=rows[:a.limit]
    if not rows:raise ValueError('No test frames')
    if not a.limit and not a.reference_root:raise ValueError('Formal test requires native-MSE reference dump parity')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    rt=get_runtime(a)
    models,protocols,paths=load_models(a.weight,a.device,rt.provenance,bool(a.limit))
    if set(models)!=set('ABCD'):raise ValueError('Need all four trained A/B/C/D')
    file_by_variant={n:info['path'] for n,info in paths.items()}
    sweep,alphas=locked_choices(a.locked,rt.provenance,file_by_variant)
    current=parameter_source_hash()
    if sweep['current_interactive_code_sha256']!=current and not a.allow_code_drift:
        raise ValueError('Frozen Stage1 source changed since alpha selection; inspect changes before test')
    if bool(sweep.get('explicit_code_drift_override')) and not a.allow_code_drift:
        raise ValueError('Validation used an explicit source-code mismatch; test must explicitly acknowledge it')
    records=[]; max_err=0.
    for pos,row in enumerate(rows,1):
        snap,parity=rt.snapshot(row,a.seed)
        if len(snap['grasps'])!=1024 or any(parity.get(k,1)>1e-6 for k in ('native_decode_max_abs','native_reader_replay_max_abs')):
            raise RuntimeError('Frozen native forward mismatch')
        snap['sensor_keep']=rt.sensor_keep(row,snap['grasps']).astype(bool)
        ids=physical_ids(snap['grasps']);base_g=snap['grasps']
        if a.reference_root:
            path=Path(a.reference_root)/f"scene_{row['scene']:04d}"/'realsense'/f"{row['anno']:04d}.npy"
            old=np.load(path,allow_pickle=False);now=base_g[snap['sensor_keep']]
            if old.shape!=now.shape:raise RuntimeError(f'Native candidate count mismatch: {path}')
            err=float(np.max(np.abs(old-now))) if len(now) else 0.
            if err>1e-6:raise RuntimeError(f'Native candidate mismatch {err}: {path}')
            max_err=max(max_err,err)
        snap['label_mask']=np.zeros(1024,bool);snap['quality_count']=np.full(1024,-1,np.int64)
        inputs,_,_=tensor_frame(snap,a.device)
        scores={'native':base_g[:,0].copy()}
        for variant,model in models.items():
            inputs['memory']=torch.from_numpy(snap['memory'].astype(protocols[variant]['memory_dtype']).astype(np.float32)).unsqueeze(0).to(a.device)
            z=model(**inputs)[0].cpu().numpy().astype(float)
            scores[variant]=mix_logits(snap['base_logit'],z,alphas[variant],base_g[:,0]).astype(np.float32)
        for arr in scores.values():
            g=base_g.copy();g[:,0]=arr
            if not np.array_equal(physical_ids(g),ids):raise RuntimeError('Rescoring changed grasp geometry')
        records.extend(emit_frame(root,row,snap,scores))
        print(f'[confirm] {pos}/{len(rows)} scene={row["scene"]} anno={row["anno"]}',flush=True)
    atomic_json(root/'inference_manifest.json',signed(dict(status='complete',source=rt.provenance,
        source_code_sha256=current,split=a.split,domain='original',seed=a.seed,frames=len(rows),
        partial_smoke=bool(a.limit),source_reference_root=str(a.reference_root),
        source_reference_max_abs=max_err,variants=['native',*models],weights=paths,protocols=protocols,
        locked_validation_sha256=sha256(a.locked),locked_validation_fingerprint=sweep['fingerprint'],
        locked_alphas=alphas,action_geometry_frozen=True,test_labels_used=False,
        records=records)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','config','dataset-root','locked','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--gntrans-root',default='');p.add_argument('--weight',action='append',required=True)
    p.add_argument('--split',choices=('test_seen','test_similar','test_novel'),required=True)
    p.add_argument('--reference-root',default='')
    p.add_argument('--provider',default='interactive_grasp.native:NativeRuntime')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--allow-code-drift',action='store_true')
    a=p.parse_args();infer(a)

if __name__=='__main__':main()