"""Evidence mismatch, density, permutation and paired scene-level diagnostics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .core import *
from .model import intervene
from .pipeline import load_models,metrics,scene_mean


def donor_entry(entries,row):
    pool=[e for e in entries if e['row']['domain']==row['domain'] and e['row']['scene']!=row['scene']]
    if not pool:raise ValueError('Wrong-scene control requires another validation scene in same RGB domain')
    # Deterministic and label/score independent; same frame number preferred.
    return min(pool,key=lambda e:(e['row']['anno']!=row['anno'],e['row']['scene'],e['row']['anno']))


def scene_delta(rows,left,right,metric,bootstrap=5000,seed=0):
    """Row identity is (domain, scene, anno); RGB domains remain paired per scene."""
    maps=[]
    for variant,mode in (left,right):
        selected=[r for r in rows if r['variant']==variant and r['condition']==mode]
        d={(r['domain'],r['scene'],r['anno']):r[metric] for r in selected}
        if len(d)!=len(selected):raise ValueError('Duplicate paired diagnostic frame')
        maps.append(d)
    if maps[0].keys()!=maps[1].keys():raise ValueError('Unmatched diagnostic frames')
    scenes={}
    for key in maps[0]:
        x,y=maps[0][key],maps[1][key]
        if (x is None)!=(y is None):raise ValueError('Asymmetric missing labels')
        if x is not None:scenes.setdefault(key[1],{}).setdefault(key[0],[]).append(x-y)
    values=np.array([np.mean([np.mean(x) for x in d.values()]) for d in scenes.values()])
    if not len(values):return {'mean':None,'ci95':None,'scenes':0}
    rng=np.random.default_rng(seed);boot=values[rng.integers(0,len(values),size=(bootstrap,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'scenes':len(values),'unit':'scene cluster; Original/GNTrans paired inside scene; fixed weights'}


@torch.no_grad()
def diagnose(a):
    if a.limit<0 or a.bootstrap<1:raise ValueError('Invalid diagnostic bounds')
    idx=read_index(a.index,'validation',a.smoke)
    models,ps,weights=load_models(a.weight,idx['contract']['source'],a.device,a.smoke)
    if Path(a.output).exists():raise FileExistsError(a.output)
    rows=[];checks=[]
    entries=idx['entries'][:a.limit or None]
    for e in entries:
        arr=read_frame(e,idx['contract']);x,_,_=tensors(arr,a.device);n=len(arr['grasps']);donor=None
        for name,model in models.items():
            z=model(**x);base_score=z.sigmoid()[0].cpu().numpy()
            rows.append({**e['row'],'variant':name,'condition':'aligned',**metrics(arr,base_score)})
            gen=np.random.default_rng(frame_seed(e['row'],a.seed));order=gen.permutation(n);inv=np.argsort(order)
            shuffled=dict(x)
            for k in ('features','grasps','base_logit','action_valid'):shuffled[k]=x[k][:,order]
            zperm=model(**shuffled)[:,inv];p_error=float((z-zperm).abs().max())
            if p_error>a.permutation_tol:raise RuntimeError(f'Candidate permutation parity {p_error}')
            dup=model(**intervene(x,'duplicate_candidates'))[:,:n]
            check={**e['row'],'variant':name,'permutation_max_logit_error':p_error,
                   'duplicate_mean_logit_change':float((z-dup).abs().mean()),
                   'duplicate_max_logit_change':float((z-dup).abs().max())}
            rows.append({**e['row'],'variant':name,'condition':'duplicate_candidates',**metrics(arr,dup.sigmoid()[0].cpu().numpy())})
            if name=='scene':
                donor=donor_entry(idx['entries'],e['row']);darr=read_frame(donor,idx['contract']);dx,_,_=tensors(darr,a.device)
                check['donor_frame']=frame_key(donor['row'])
                for mode in ('scene_joint_permute','scene_appearance_shuffle','scene_swap','scene_mean'):
                    changed=intervene(x,mode,frame_seed(e['row'],a.seed),dx)
                    zz=model(**changed);err=float((z-zz).abs().max())
                    if mode=='scene_joint_permute' and err>a.permutation_tol:raise RuntimeError(f'Joint memory permutation failed {err}')
                    rows.append({**e['row'],'variant':name,'condition':mode,**metrics(arr,zz.sigmoid()[0].cpu().numpy())})
                    check[mode+'_max_logit_change']=err
            checks.append(check)
    contrasts={}
    compare=[(('scene','aligned'),('candidate','aligned')),(('scene','aligned'),('local','aligned')),
             (('candidate','aligned'),('local','aligned'))]
    if 'scene' in models:
        compare += [(('scene','aligned'),('scene',m)) for m in ('scene_appearance_shuffle','scene_swap','scene_mean')]
    for left,right in compare:
        if left[0] not in models or right[0] not in models:continue
        contrasts[f'{left}-{right}']={k:scene_delta(rows,left,right,k,a.bootstrap,a.seed)
            for k in ('mse','accuracy','labelled_top1_q','pre_nms_top1_q','pre_nms_top5_q','pre_nms_top50_q')}
    atomic_json(a.output,signed({'schema':SCHEMA,'index':idx['fingerprint'],'weights':weights,
        'records':rows,'checks':checks,'contrasts':contrasts,'limited_frames':bool(a.limit),
        'notes':['Validation only; test is not used to choose settings.',
                 'Joint permutation tests invariance; appearance shuffle breaks feature-position alignment.',
                 'Mismatch controls are inference interventions, not trained ablations.',
                 'Pre-NMS quality is not official AP; report coverage and empty surviving pools.']}))


def benchmark(a):
    import time
    if a.warmup<0 or a.repeats<1:raise ValueError('Invalid timing bounds')
    idx=read_index(a.index,'validation',a.smoke)
    models,ps,weights=load_models(a.weight,idx['contract']['source'],a.device,a.smoke)
    arr=read_frame(idx['entries'][0],idx['contract']);x,_,_=tensors(arr,a.device)
    def sync():
        if str(a.device).startswith('cuda'):torch.cuda.synchronize(a.device)
    rows={}
    with torch.inference_mode():
        for name,model in models.items():
            for _ in range(a.warmup):model(**x)
            sync()
            if str(a.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
            times=[]
            for _ in range(a.repeats):
                sync();start=time.perf_counter();model(**x);sync();times.append((time.perf_counter()-start)*1000)
            rows[name]={'extra_head_mean_ms':float(np.mean(times)),'extra_head_p50_ms':float(np.median(times)),
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'cuda_peak_bytes_all_loaded_heads_and_inputs':torch.cuda.max_memory_allocated(a.device) if str(a.device).startswith('cuda') else None}
    if Path(a.output).exists():raise FileExistsError(a.output)
    atomic_json(a.output,signed({'schema':SCHEMA,'weights':weights,'device':a.device,'torch':torch.__version__,
        'rows':rows,'scope':'Added scorer forward only; excludes frozen GVAR extraction, transfers, CAD and official evaluation.'}))
