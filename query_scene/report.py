"""Three-way official AP comparison; common scales/data and scene uncertainty."""
from pathlib import Path
import numpy as np
from .core import *
from interactive_grasp.analyze import paired_bootstrap


def summarize(a):
    paths={}
    splits=('test_seen','test_similar','test_novel')
    for item in a.split_dir:
        k,p=item.split('=',1)
        if k in paths or k not in splits:raise ValueError('Bad split mapping')
        paths[k]=Path(p)
    if set(paths)!=set(splits):raise ValueError('Need all three splits')
    arrays={};identity=None;required={'native',*VARIANTS}
    for split in splits:
        root=paths[split];m=load_signed(root/'inference_manifest.json');ev=load_signed(root/'evaluation_complete.json')
        if m['schema']!=SCHEMA or m['partial_smoke'] or m['frames']!=7680 or m['split']!=split or ev['inference_sha256']!=m['fingerprint']:
            raise ValueError('Incomplete or inconsistent evaluation')
        current=(m['source'],m['domain'],m['seed'],m['weights'],m['protocols'])
        if identity is None:identity=current
        if identity!=current or set(m['variants'])!=required:raise ValueError('Across-split model/protocol mismatch')
        arrays[split]={}
        for name in required:
            p=root/name/f'ap_{split}_realsense.npy'
            if sha256(p)!=ev['methods'][name]['sha256']:raise ValueError('AP hash mismatch')
            x=np.load(p,allow_pickle=False)
            if x.shape!=(30,256,50,6) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('AP shape/range')
            arrays[split][name]=x
    table={s:{n:float(x.mean()*100) for n,x in arrays[s].items()} for s in splits}
    contrasts={}
    for left,right in (('scene','candidate'),('scene','local'),('candidate','local'),('local','native'),('candidate','native'),('scene','native')):
        values=[(arrays[s][left]-arrays[s][right]).mean(axis=(1,2,3))*100 for s in splits]
        contrasts[f'{left}-{right}']=paired_bootstrap(values,a.bootstrap,a.seed)
    result={'schema':SCHEMA,'per_split_ap_percent':table,'paired_contrasts':contrasts,
            'mean_ap_percent':{n:float(np.mean([table[s][n] for s in splits])) for n in required},
            'source_manifests':{s:sha256(paths[s]/'inference_manifest.json') for s in splits},
            'by_mu_and_budget':{s:{n:{'ap_by_mu':(x.mean(axis=(0,1,2))*100).tolist(),
                'precision_at_k':{str(k):(x[:,:,k-1,:].mean(axis=(0,1))*100).tolist() for k in (1,5,10,50)}}
                for n,x in arrays[s].items()} for s in splits}}
    if Path(a.output).exists():raise FileExistsError(a.output)
    atomic_json(a.output,signed(result))
    if a.markdown:
        lines=['# Query–Scene Evidence Access','', '|Variant|Seen|Similar|Novel|Mean|','|---|---:|---:|---:|---:|']
        for name in ('native',*VARIANTS):lines.append('|'+name+'|'+'|'.join(f'{table[s][name]:.4f}' for s in splits)+f'|{result["mean_ap_percent"][name]:.4f}|')
        lines+=['','## Paired scene contrasts']+[f'- {k}: {v["mean_pp"]:+.4f} pp; CI {v["ci95"]}' for k,v in contrasts.items()]
        lines+=['','Actions, sensor filter and official NMS are fixed. Scene means the current model input crop.',
                'A gain needs matched Local/Candidate controls and evidence-correspondence checks.',
                'Intervals condition on fixed weights; not training-seed uncertainty or multiple-comparison adjusted.']
        Path(a.markdown).write_text('\n'.join(lines)+'\n',encoding='utf-8')
