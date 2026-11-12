"""Train A/B/C/D on identical full-context, sparsely labelled frozen frames."""
from __future__ import annotations
from dataclasses import asdict
import random
import os
from pathlib import Path
import numpy as np
import torch
from .contracts import SCHEMA,atomic_json,signed,frame_seed,parameter_source_hash,sha256
from .data import load_index,load_frame,tensor_frame
from .model import ModelConfig,InteractiveGraspRefiner
from .objectives import sparse_quality_loss,pair_changes


def evaluate(model,index,device,pair_seed=0):
    model.eval();rows=[]
    with torch.no_grad():
        for e in index['entries']:
            a=load_frame(e,index['contract']['expected_queries'])
            inputs,count,mask=tensor_frame(a,device)
            z=model(**inputs)
            _,loss=sparse_quality_loss(z,count,mask,inputs['action_valid'],pair_seed=pair_seed)
            score=z.sigmoid()[0].cpu().numpy();base=a['grasps'][:,0];m=a['label_mask'];y=a['quality_count']
            ids=np.flatnonzero(m)
            row={**e['row'],**loss,**pair_changes(base,score,y,m),
                'labelled_subset_top1_q':float(y[ids[np.argmax(score[ids])]]/6),
                'native_labelled_subset_top1_q':float(y[ids[np.argmax(base[ids])]]/6),
                'labelled_fraction':float(m.mean())}
            rows.append(row)
    # Average views within scene, then scenes within domain, then equal domains.
    keys=('mse','pair','labelled_subset_top1_q','native_labelled_subset_top1_q')
    summary={}
    for key in keys:
        domains=[]
        for d in sorted({r['domain'] for r in rows}):
            scenes=[]
            for s in sorted({r['scene'] for r in rows if r['domain']==d}):
                scenes.append(np.mean([r[key] for r in rows if r['domain']==d and r['scene']==s]))
            domains.append(np.mean(scenes))
        summary[key]=float(np.mean(domains))
    return dict(summary=summary,per_frame=rows,
                scope='Sparse labelled subset, NOT full-pool Top1, insertion regret or official AP')


def atomic_torch(path,obj):
    path=Path(path);tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    torch.save(obj,tmp);os.replace(tmp,path)


def fit(a):
    train=load_index(a.train_index,'train',a.smoke);val=load_index(a.val_index,'validation',a.smoke)
    for key in ('source','provider','seed','memory_dtype','expected_queries','code_sha256','plan_sha256'):
        if train['contract'][key]!=val['contract'][key]:raise ValueError(f'Train/val source contract mismatch: {key}')
    if {r['row']['scene'] for r in train['entries']}&{r['row']['scene'] for r in val['entries']}:
        raise ValueError('Stage1 train/validation scene overlap, including across RGB domains')
    if any(e['row']['scene']>=100 for e in train['entries']+val['entries']):
        raise ValueError('Official test scene leaked into training/validation')
    if min(a.epochs,a.accumulate,a.max_pairs)<1 or a.lr<=0:raise ValueError('Invalid optimizer budget')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(a.cpu_threads)
    cfg=ModelConfig(feature_dim=train['feature_dim'],memory_dim=train['memory_dim'],variant=a.variant,
                    hidden_dim=a.hidden,heads=a.heads,layers=a.layers,seed=a.seed,read_chunk=a.read_chunk)
    model=InteractiveGraspRefiner(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    hyper={k:getattr(a,k) for k in ('epochs','accumulate','lr','weight_decay','max_pairs','seed','smoke')}
    protocol=dict(schema=SCHEMA,model=asdict(cfg),hyper=hyper,train_index=train['fingerprint'],
        val_index=val['fingerprint'],source=train['contract']['source'],
        memory_dtype=train['contract']['memory_dtype'],expected_queries=train['contract']['expected_queries'],
        source_code_sha256=parameter_source_hash(),loss=dict(mse=10.,pair=.5,ties='excluded'),
        physical_actions_frozen=True,primary_selection='final_epoch',model_description=model.describe())
    start=0;updates=0;history=[]
    if a.resume:
        ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if ck['protocol']!=protocol:raise ValueError('Resume protocol/code/data differs')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        start=ck['epoch'];updates=ck['updates'];history=ck['history']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if start>a.epochs:raise ValueError('Resume past requested final epoch')
    else:
        atomic_json(root/'protocol.json',signed(protocol))
        history.append(dict(epoch=0,updates=0,validation=evaluate(model,val,device)['summary']))
    for epoch in range(start+1,a.epochs+1):
        model.train();order=np.random.default_rng(a.seed+epoch*100003).permutation(len(train['entries']))
        stat=[]
        for step in range(0,len(order),a.accumulate):
            group=order[step:step+a.accumulate];optimizer.zero_grad(set_to_none=True)
            for pos in group:
                e=train['entries'][int(pos)];frame=load_frame(e,train['contract']['expected_queries'])
                inputs,y,label_mask=tensor_frame(frame,device)
                z=model(**inputs)
                loss,info=sparse_quality_loss(z,y,label_mask,inputs['action_valid'],
                    pair_seed=frame_seed(e['row'],a.seed+epoch*100003),max_pairs=a.max_pairs)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                (loss/len(group)).backward();stat.append(info)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step();updates+=1
        validation=evaluate(model,val,device)
        history.append(dict(epoch=epoch,updates=updates,
            train_mse=float(np.mean([x['mse'] for x in stat])),
            train_pair=float(np.mean([x['pair'] for x in stat])),
            used_pairs=int(sum(x['used_pairs'] for x in stat)),validation=validation['summary']))
        ck=dict(schema=SCHEMA,protocol=protocol,model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch,updates=updates,history=history,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        atomic_torch(root/'last.pt',ck)
        atomic_json(root/'history.json',history)
        atomic_json(root/'validation_last.json',validation)
        print(f'[train {a.variant}] epoch {epoch}/{a.epochs} updates={updates} val={validation["summary"]}',flush=True)
    atomic_json(root/'complete.json',signed(dict(status='complete',epochs=a.epochs,updates=updates,
        checkpoint_sha256=sha256(root/'last.pt'),smoke_only=a.smoke)))


def load_refiner(path,device,allow_smoke=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=SCHEMA or ck['epoch']!=ck['protocol']['hyper']['epochs']:
        raise ValueError('Only preregistered completed final-epoch model is accepted')
    if ck['protocol']['hyper']['smoke'] and not allow_smoke:raise ValueError('Smoke checkpoint forbidden in formal evaluation')
    model=InteractiveGraspRefiner(ModelConfig(**ck['protocol']['model'])).to(device)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    return model,ck['protocol']
