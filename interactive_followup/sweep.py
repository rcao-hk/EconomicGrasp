"""Validation-only residual-strength sweep; optional exact-action AP-like replay.

No training, no test scenes, no graph/learned suppression, no test-tuned alpha.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from interactive_grasp.contracts import (load_signed,signed,atomic_json,sha256,
                                         parameter_source_hash)
from interactive_grasp.data import load_index,load_frame,tensor_frame
from interactive_grasp.infer import load_models
from .core import parse_alphas,mix_logits,quality_pairs,selection_stats,weighted_summary,choose_alpha,paired_scene_bootstrap
from .labels import read_full_labels,validate_index


def val_ap_proxy(grasps,prob,keep,quality):
    """Replay official NMS, per-object top-10, global top-50 on validation.

    This is NOT official test AP. Quality is exact CAD-DexNet per action.
    """
    from graspnetAPI import GraspGroup
    from interactive_grasp.contracts import physical_ids
    g=np.asarray(grasps,dtype=np.float32);prob=np.asarray(prob,dtype=np.float32)
    sensor=np.asarray(keep,dtype=bool)
    if len(g)!=len(prob) or sensor.shape!=(len(g),) or not np.isfinite(prob).all():raise ValueError('Replay input mismatch')
    source_ids=physical_ids(g);mapping={}
    for i,pid in enumerate(source_ids):
        if pid in mapping:
            j=mapping[pid]
            for key in ('assigned_obj','friction','collision_or_empty'):
                if quality[key][i]!=quality[key][j]:raise ValueError('Identical executed geometry has conflicting exact quality')
        else:mapping[pid]=i
    changed=g[sensor].copy();changed[:,0]=prob[sensor]
    remain=GraspGroup(changed).nms(0.03,np.deg2rad(30)).grasp_group_array
    if len(remain)==0:return 0.
    idx=np.asarray([mapping[i] for i in physical_ids(remain)],dtype=np.int64)
    rank=remain[:,0].astype(float)
    obj=np.asarray(quality['assigned_obj'])[idx]
    chosen=[]
    for oid in np.unique(obj):
        rows=np.flatnonzero(obj==oid)
        chosen.extend(rows[np.argsort(rank[rows])[::-1]][:10])
    chosen=np.asarray(chosen,dtype=np.int64)
    min_score=np.sort(rank[chosen])[::-1][min(49,len(chosen)-1)]
    chosen=chosen[rank[chosen]>=min_score]
    chosen=chosen[np.argsort(-rank[chosen])]
    physical=idx[chosen]
    bad=np.asarray(quality['collision_or_empty'],dtype=bool)[physical]
    friction=np.asarray(quality['friction'],dtype=float)[physical]
    m=np.asarray([.2,.4,.6,.8,1.,1.2])[None,:]
    ok=((friction[:,None]>0)&(friction[:,None]<=m+1e-6)&(~bad[:,None])).astype(float)
    cum=np.cumsum(ok,axis=0)
    k=np.arange(1,51,dtype=float)
    numer=np.zeros((50,6),dtype=float)
    numer[:min(len(cum),50)]=cum[:50]
    if len(cum) and len(cum)<50:numer[len(cum):]=cum[-1]
    return float((numer/k[:,None]).mean()*100)


def load_quality(e,index,full_root):
    if full_root is None:return None
    return read_full_labels(e,index,full_root)


def add_frame_rows(entry,frame,zmodels,alphas,full_quality=None):
    """Explicit separation of sparse-label and full-pool selection statistics."""
    base=np.asarray(frame['base_logit'],dtype=float)
    exact=np.asarray(frame['grasps'][:,0],dtype=float)
    lab=frame['label_mask'].astype(bool)
    sparse=frame['quality_count'].astype(np.int64)
    if not np.any(lab):raise ValueError('Missing validation labels')
    count=full_quality['quality_count'] if full_quality is not None else sparse
    mask=np.ones(len(count),dtype=bool) if full_quality is not None else lab
    rows=[]
    for name,z in zmodels.items():
        for alpha in alphas:
            score=mix_logits(base,z,float(alpha),exact)
            ys=count[mask]/6
            mse=float(np.mean((score[mask]-ys)**2))
            old=exact[mask]
            pair=quality_pairs(exact,score,count,mask)
            select=selection_stats(exact,score,count,mask,frame['sensor_keep'])
            result={**entry['row'],'variant':name,'alpha':float(alpha),'mse':mse,
                    'labelled_fraction':float(np.mean(mask)),
                    'labelled_best_q':float(np.max(ys)),
                    'score_residual_abs_mean':float(np.mean(np.abs(score-exact))),
                    'logit_delta_abs_mean':float(np.mean(np.abs(z-base))),
                    **{f'pair_{k}':v for k,v in pair.items()},**select}
            if full_quality is not None:
                result['proxy_ap']=val_ap_proxy(frame['grasps'],score,frame['sensor_keep'],full_quality)
            rows.append(result)
    return rows


def analyse(a):
    index=load_index(a.index,'validation',a.allow_partial)
    validate_index(index) if not a.allow_partial else None
    if any(e['row']['scene']>=100 or e['row']['partition']!='validation' for e in index['entries']):
        raise ValueError('No test/novel/similar/seen frame may tune alpha')
    alphas=parse_alphas(a.alphas)
    source=index['contract']['source'];models,protocols,paths=load_models(a.weight,a.device,source,a.allow_partial)
    if set(models)!=set('ABCD'):raise ValueError('Require all four A/B/C/D for controlled sweep')
    code=parameter_source_hash()
    changed={name:p.get('source_code_sha256') for name,p in protocols.items() if p.get('source_code_sha256')!=code}
    if changed and not a.allow_code_drift:
        raise ValueError('Stage1 training source-code hash differs from checkout; audit actual server changes, then pass --allow-code-drift explicitly')
    full_root=Path(a.full_labels_root) if a.full_labels_root else None
    if full_root is not None:
        manifest=load_signed(full_root/'full_label_manifest.json')
        if manifest.get('status')!='complete' or manifest['index_sha256']!=index['fingerprint'] or manifest['source']!=source:
            raise ValueError('Full held-out validation action labels are incomplete or from a different frozen base')
    if a.select_alpha and full_root is None:raise ValueError('Selecting alpha requires all 1024 frozen actions exact-labelled on validation')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.cpu_threads)
    rows=[]
    with torch.inference_mode():
        for i,e in enumerate(index['entries'][:a.limit or None],1):
            frame=load_frame(e,index['contract']['expected_queries'])
            inputs,_,_=tensor_frame(frame,a.device)
            base=torch.as_tensor(frame['base_logit'],device=a.device,dtype=torch.float32)
            zs={}
            for name,model in models.items():
                inputs['memory']=torch.from_numpy(frame['memory'].astype(protocols[name]['memory_dtype']).astype(np.float32)).unsqueeze(0).to(a.device)
                zs[name]=model(**inputs)[0].detach().cpu().numpy().astype(float)
                if name=='D':
                    zs['D_pre_read']=model(**inputs,bypass_interaction_query=True)[0].detach().cpu().numpy().astype(float)
            full=load_quality(e,index,full_root)
            if full is not None and not np.array_equal(frame['physical_id'],full['physical_id']):raise ValueError('Physical action changed since full validation labeling')
            rs=add_frame_rows(e,frame,zs,alphas,full)
            for row in rs:row['partition']='validation'
            rows.extend(rs)
            print(f'[validate-alpha] {i}/{len(index["entries"])} {e["row"]["domain"]} scene{e["row"]["scene"]:04d} frame{e["row"]["anno"]:04d}',flush=True)
    if not rows:raise RuntimeError('No validation results')
    keys=('mse','pair_new_agreement','pair_repair','pair_damage','top1_changed','top5_overlap','top50_overlap','top1_new_labelled','score_residual_abs_mean','logit_delta_abs_mean')
    if full_root is not None:keys+=('proxy_ap',)
    aggregates={}
    all_names=['A','B','C','D','D_pre_read']
    for name in all_names:
        aggregates[name]={}
        for alpha in alphas:
            subset=[r for r in rows if r['variant']==name and r['alpha']==float(alpha)]
            aggregates[name][str(float(alpha))]={k:weighted_summary(subset,k) for k in keys}
    choices={}
    if a.select_alpha:
        if a.limit or not index['complete']:raise ValueError('Cannot select alpha from partial/smoke validation')
        for name in ('A','B','C','D'):
            choices[name]=choose_alpha([r for r in rows if r['variant']==name],'proxy_ap',alphas)
        choices['D_pre_read']=choose_alpha([r for r in rows if r['variant']=='D_pre_read'],'proxy_ap',alphas)
    deltas={}
    if not a.limit and index['complete'] and full_root is not None:
        for a1,b1 in (('D','A'),('D','B'),('D','C'),('D','D_pre_read'),('C','A'),('B','A')):
            deltas[f'{a1}-{b1}']={str(float(alpha)):paired_scene_bootstrap(
                rows,a1,b1,float(alpha),'proxy_ap',n_boot=a.bootstrap,seed=a.seed)
                for alpha in alphas}
    summary=signed(dict(status='complete' if not a.limit else 'partial_smoke',index_sha256=index['fingerprint'],
        partition='validation',source=source,weight_files=paths,training_source_code_sha256=changed,
        current_interactive_code_sha256=code,explicit_code_drift_override=bool(a.allow_code_drift),
        full_exact_labels=full_root is not None,full_label_manifest_sha256=sha256(full_root/'full_label_manifest.json') if full_root else None,
        alpha_values=alphas.tolist(),num_frames=len(index['entries'][:a.limit or None]),
        metrics_scope='validation exact-action full-pool AP-like proxy' if full_root else 'sparse labelled subset + full-pool identity changes; NOT official AP',
        aggregate=aggregates,choices=choices,paired_scene_proxy_ap=deltas))
    atomic_json(output/'summary.json',summary)
    with (output/'per_frame.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Interactive grasp follow-up: held-out validation residual sweep','',
           'No official test scenes used. Alpha chosen only with full validation CAD labels.','',
           '| Variant | Alpha | Labelled Q-MSE | Pairs agreement | Top1 switched | Proxy AP (%) |',
           '|---|---:|---:|---:|---:|---:|']
    for name in all_names:
        for alpha in alphas:
            r=aggregates[name][str(float(alpha))]
            lines.append(f'|{name}|{alpha:.2f}|{r["mse"]:.6f}|{r["pair_new_agreement"]}|{r["top1_changed"]:.4f}|{r.get("proxy_ap")} |')
    lines.extend(['','## Locked validation choices','',json.dumps(choices,indent=2),
                  '', 'Warning: proxy AP is offline exact-quality validation ranking; it is NOT official GraspNet test AP.'])
    (output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'choices':choices,'output':str(output)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True);p.add_argument('--weight',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--full-labels-root')
    p.add_argument('--alphas',default='0,0.05,0.1,0.2,0.4,0.7,1')
    p.add_argument('--select-alpha',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--allow-code-drift',action='store_true');p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();analyse(a)

if __name__=='__main__':main()