"""Same physical action / alternate optical appearance diagnostic (not a loss)."""
from pathlib import Path
import numpy as np
import torch
from .data import read_index, read_frame, load_signed, atomic_json, signed, sha256
from .experiment import load_model, predict, family_metrics


@torch.no_grad()
def diagnose_paired(a):
    idx=read_index(a.index,'validation',a.smoke)
    model,p=load_model(a.weight,a.device,a.smoke)
    if p['mode']=='local': raise ValueError('Local frozen feature was action-specific; do not swap it across domains')
    if p['val_fingerprint']!=idx['fingerprint']: raise ValueError('Wrong validation identity')
    lookup={(e['row']['domain'],e['row']['scene'],e['row']['anno']):e for e in idx['entries']}
    records=[]
    for (domain,scene,anno),entry in lookup.items():
        if domain!='original': continue
        peer=lookup.get(('gntrans',scene,anno))
        if peer is None: raise ValueError('Missing paired appearance frame')
        metas=[load_signed(Path(e['path']).with_suffix('.json')) for e in (entry,peer)]
        ids=[m['target'].get('raw_geometry_identity') for m in metas]
        if not ids[0] or ids[0]!=ids[1]:
            raise ValueError('Paired diagnostic requires verified identical raw rendered geometry and camera')
        x,y=read_frame(entry),read_frame(peer)
        # Deliberately keep ORIGINAL exact physical actions and their exact labels.
        swapped=dict(x,**{k:y[k] for k in ('feature_map','pred_depth','K')})
        za,_=predict(model,p['mode'],x,a.device);zb,_=predict(model,p['mode'],swapped,a.device)
        za,zb=za.cpu().numpy(),zb.cpu().numpy()
        q=x['quality_count'];rows=np.arange(len(q))
        ca,cb=za.argmax(1),zb.argmax(1)
        records.append(dict(scene=scene,anno=anno,geometry_identity=ids[0],
            original=family_metrics(za,q),altered=family_metrics(zb,q),
            changed_choice=float(np.mean(ca!=cb)),
            equal_quality_change=float(np.mean(q[rows,ca]==q[rows,cb])),
            mean_score_difference=float(np.mean(np.abs(1/(1+np.exp(-np.clip(za,-60,60)))-1/(1+np.exp(-np.clip(zb,-60,60))))))))
    if not records:raise ValueError('No paired diagnostic records')
    if Path(a.output).exists():raise FileExistsError(a.output)
    atomic_json(a.output,signed(dict(records=records,weight_sha256=sha256(a.weight),
        note='One camera-frame action pool shared across optical domains; physical quality assumed unchanged only for optical/material-rendering changes at fixed friction. Not inference from ground-truth geometry.')))
