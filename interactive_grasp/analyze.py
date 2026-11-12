"""Mechanism checks and paired split-stratified scene bootstrap for A/B/C/D."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .contracts import load_signed,signed,atomic_json,sha256,SCHEMA
from .data import load_index,load_frame,tensor_frame
from .train import load_refiner
from .objectives import pair_changes


def diagnose(a):
    index=load_index(a.index,a.partition,a.allow_partial)
    model,p=load_refiner(a.weight,a.device,a.allow_partial)
    if p['source']!=index['contract']['source']:raise ValueError('Weights/cache frozen source mismatch')
    rows=[]
    with torch.no_grad():
        for e in index['entries'][:a.limit or None]:
            frame=load_frame(e,index['contract']['expected_queries']);inputs,y,mask=tensor_frame(frame,a.device)
            z=model(**inputs);score=z.sigmoid()[0].cpu().numpy()
            # Equivariance is required; density invariance is measured, not assumed.
            rng=np.random.default_rng(a.seed)
            order=rng.permutation(z.shape[1]);inv=np.argsort(order)
            shuffled={k:v[:,order] for k,v in inputs.items()}
            shuffled_z=model(**shuffled)[:,inv]
            perm_error=float((z-shuffled_z).abs().max())
            if perm_error>1e-5:raise RuntimeError(f'Permutation equivariance failed: {perm_error}')
            duplicate=torch.arange(min(a.duplicates,z.shape[1]),device=z.device)
            duplicated={k:torch.cat((v,v[:,duplicate]),dim=1) for k,v in inputs.items()}
            new_z=model(**duplicated)[:,:z.shape[1]]
            changes=pair_changes(frame['grasps'][:,0],score,frame['quality_count'],frame['label_mask'])
            row={**e['row'],**changes,'permutation_max_logit_error':perm_error,
                 'duplicates':len(duplicate),'duplicate_context_mean_logit_change':float((z-new_z).abs().mean())}
            if model.cfg.variant=='D':
                control=model(**inputs,bypass_interaction_query=True).sigmoid()[0].cpu().numpy()
                row['pre_interaction_read_query_mean_score_change']=float(np.mean(np.abs(score-control)))
                row['pre_interaction_read_query_pairs']=pair_changes(frame['grasps'][:,0],control,
                                               frame['quality_count'],frame['label_mask'])
            rows.append(row)
    atomic_json(a.output,signed(dict(schema=SCHEMA,index_sha256=index['fingerprint'],
        weight_sha256=sha256(a.weight),variant=model.cfg.variant,records=rows,
        warning='Labelled-subset pairs; inference interventions are not independently trained ablations.')))


def paired_bootstrap(values, n=5000, seed=0):
    """values: list of split-specific per-scene deltas, percentage points."""
    if n<1 or any(len(x)==0 for x in values):raise ValueError('Invalid bootstrap inputs')
    rng=np.random.default_rng(seed)
    draws=np.mean([np.asarray(v)[rng.integers(0,len(v),size=(n,len(v)))].mean(1) for v in values],axis=0)
    return dict(mean_pp=float(np.mean([np.mean(v) for v in values])),
                ci95=np.quantile(draws,[.025,.975]).tolist(),
                unit='paired scene, stratified by test split; not training-seed uncertainty')


def summarize(a):
    splits=('test_seen','test_similar','test_novel');sources={}
    for item in a.split_dir:
        k,v=item.split('=',1)
        if k in sources or k not in splits:raise ValueError('Invalid/duplicate split input')
        sources[k]=Path(v)
    if set(sources)!=set(splits):raise ValueError('All three splits required')
    arrays={};identity=None;methods=None
    for split in splits:
        root=sources[split];m=load_signed(root/'inference_manifest.json');ev=load_signed(root/'evaluation_complete.json')
        if ev['inference_sha256']!=m['fingerprint'] or m['partial_smoke'] or m['split']!=split:
            raise ValueError('Incomplete/mismatched official evaluation')
        current=(m['source'],m['domain'],m['seed'],m['weights'])
        if identity is None:identity=current
        if current!=identity:raise ValueError('Source/heads/data protocol changed between splits')
        if methods is None:methods=set(m['variants'])
        if set(m['variants'])!=methods or methods!=set(('native','A','B','C','D')):
            raise ValueError('Require all A/B/C/D plus native under the same protocol')
        arrays[split]={}
        for name in sorted(methods):
            path=root/name/f'ap_{split}_realsense.npy'
            if sha256(path)!=ev['methods'][name]['sha256']:raise ValueError('AP hash mismatch')
            x=np.load(path)
            if x.shape!=(30,256,50,6) or not np.isfinite(x).all():raise ValueError('Invalid official AP')
            arrays[split][name]=x
    summary={split:{name:float(x.mean()*100) for name,x in arrays[split].items()} for split in splits}
    contrasts={}
    for left,right in (('A','native'),('B','A'),('C','A'),('D','B'),('D','C'),('D','A')):
        values=[(arrays[s][left]-arrays[s][right]).mean(axis=(1,2,3))*100 for s in splits]
        contrasts[f'{left}-{right}']=paired_bootstrap(values,a.bootstrap,a.seed)
    interaction=[(arrays[s]['D']-arrays[s]['C']-arrays[s]['B']+arrays[s]['A']).mean(axis=(1,2,3))*100 for s in splits]
    contrasts['D-C-B+A']=paired_bootstrap(interaction,a.bootstrap,a.seed)
    result=dict(schema=SCHEMA,per_split_ap_percent=summary,paired_contrasts=contrasts,
        mean_ap_percent={n:float(np.mean([summary[s][n] for s in splits])) for n in sorted(methods)},
        source_manifests={s:sha256(sources[s]/'inference_manifest.json') for s in splits},
        interpretation='2x2 AP interaction is an experimental contrast, not a physical causal contribution.')
    result['per_split_k_mu_percent']={
        split:{name:{
            'mean_prefix_ap_by_mu':(x.mean(axis=(0,1,2))*100).tolist(),
            'precision_at_k_by_mu':{
                str(k):(x[:,:,k-1,:].mean(axis=(0,1))*100).tolist() for k in (1,5,50)}
        } for name,x in arrays[split].items()} for split in splits}
    atomic_json(a.output,signed(result))
    if a.markdown:
        lines=['# Interactive Grasp Stage 1 — A/B/C/D','',
               '|Variant|Seen|Similar|Novel|Mean|','|---|---:|---:|---:|---:|']
        for n in ('native','A','B','C','D'):
            lines.append('|'+n+'|'+'|'.join(f'{summary[s][n]:.4f}' for s in splits)+f'|{result["mean_ap_percent"][n]:.4f}|')
        lines+=['','## Paired scene contrasts','']
        for k,v in contrasts.items():lines.append(f'- {k}: {v["mean_pp"]:+.4f} pp; CI {v["ci95"]}')
        lines+=['','Physical actions are fixed. Native official NMS/filters are unchanged.',
                'Confidence intervals condition on fixed trained weights; repeat training seeds separately.']
        Path(a.markdown).write_text('\n'.join(lines)+'\n',encoding='utf-8')
