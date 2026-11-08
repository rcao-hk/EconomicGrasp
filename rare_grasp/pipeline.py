"""Manifest, exact-action cache, RARE training, full-query inference and AP.

Run `python -m rare_grasp.pipeline --help`. All long-running operations are
explicit; no default command starts training, mining, or official evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import torch
from .core import (SCHEMA, make_protocol, verify_protocol, frame_key, frame_seed,
                   sha256, digest, write_json, physical_ids, validate_cache,
                   ResidualRanker, ranking_loss, select_grasps, candidate_metrics)


def protocol_rows(path, partition, domain=None, split=None):
    p=json.loads(Path(path).read_text()); verify_protocol(p)
    rows=[r for r in p[partition] if (domain is None or r['domain']==domain)
          and (split is None or r['split']==split)]
    if not rows: raise ValueError('No frames match requested partition/domain/split')
    return p, rows


def ensure_new_dir(path):
    p=Path(path)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f'Output must be empty; no silent overwrite: {p}')
    p.mkdir(parents=True,exist_ok=True)
    return p


def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'wb') as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def mine(a):
    from .runtime import Runtime
    p,rows=protocol_rows(a.protocol,a.partition,a.domain,a.split)
    if not 0<=a.shard_index<a.num_shards or a.families<1 or a.families>1024:
        raise ValueError('Invalid shards/family count')
    rows=[r for i,r in enumerate(rows) if i%a.num_shards==a.shard_index]
    if a.limit: rows=rows[:a.limit]
    runtime=Runtime(a.config,a.dataset_root,a.gntrans_root,a.device)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    contract=dict(schema=SCHEMA,protocol_sha256=p['protocol_sha256'],partition=a.partition,
                  config_digest=runtime.config_digest,config=runtime.config,
                  families=a.families,seed=a.seed,angle_policy='frozen_native_winner',
                  sampling='uniform_queries_without_replacement',fc_mode='official',
                  labels='full_exact_executed_actions_no_legacy_mask',
                  collision_prefilter=False, feature_dtype='float32',
                  evaluator_source_sha256=sha256('exact_action_graspnet_evaluator.py'),
                  runtime_source_sha256=sha256(Path(__file__).with_name('runtime.py')),
                  dataset_root=str(Path(a.dataset_root).resolve()),
                  gntrans_root=str(Path(a.gntrans_root).resolve()))
    contract_hash=digest(contract)
    for n,row in enumerate(rows):
        key=frame_key(row);npz=root/(key+'.npz');meta=root/(key+'.json')
        if meta.exists():
            saved=json.loads(meta.read_text())
            if (saved['contract_hash']!=contract_hash or saved['row']!=row or
                    not npz.exists() or sha256(npz)!=saved['npz_sha256']):
                raise RuntimeError(f'Existing cache does not match contract: {key}')
            continue
        if npz.exists(): raise RuntimeError(f'Orphan cache {npz}; inspect rather than overwrite')
        snap=runtime.snapshot(row,a.seed)
        rng=np.random.default_rng(frame_seed(row,a.seed))
        ids=np.sort(rng.choice(1024,a.families,replace=False))
        grasps=snap['grasps'][ids].reshape(-1,17).copy()
        labels=runtime.label(row,grasps)
        arrays=dict(features=snap['features'][ids].reshape(-1,64).astype(np.float32),
                    base_logit=snap['base_logit'][ids].reshape(-1).astype(np.float32),
                    family=np.repeat(ids,4).astype(np.int64),depth=np.tile(np.arange(4),len(ids)),
                    grasps=grasps,physical_id=physical_ids(grasps),
                    native_angle=np.repeat(snap['angles'][ids],4),**labels)
        validate_cache(arrays)
        atomic_npz(npz,**arrays)
        write_json(meta,dict(row=row,contract=contract,contract_hash=contract_hash,
                            npz_sha256=sha256(npz),native_parity_max_abs=snap['native_parity_max_abs']))
        print(f'[mine {a.partition}] {n+1}/{len(rows)} {key}: {len(grasps)} exact actions',flush=True)


def index_cache(a):
    p,rows=protocol_rows(a.protocol,a.partition)
    root=Path(a.cache); entries=[];contract=None;missing=[]
    for row in rows:
        key=frame_key(row);f=root/(key+'.npz');meta=root/(key+'.json')
        if not meta.exists(): missing.append(key);continue
        rec=json.loads(meta.read_text())
        if rec['row']!=row or rec['contract']['partition']!=a.partition:
            raise ValueError('Cache split identity mismatch')
        if rec['contract']['protocol_sha256']!=p['protocol_sha256']:
            raise ValueError('Frame protocol fingerprint mismatch')
        if digest(rec['contract'])!=rec['contract_hash']:
            raise ValueError('Cache contract hash mismatch')
        if contract is None: contract=rec['contract']
        if rec['contract']!=contract: raise ValueError('Mixed model/label/sampling contracts')
        if not f.exists() or sha256(f)!=rec['npz_sha256']: raise ValueError(f'Corrupt cache {f}')
        with np.load(f,allow_pickle=False) as loaded: validate_cache(dict(loaded))
        entries.append(dict(row=row,file=str(f.resolve()),sha256=rec['npz_sha256']))
    if missing and not a.allow_partial:
        raise RuntimeError(f'{len(missing)} frames missing; e.g. {missing[:3]}')
    if not entries: raise RuntimeError('Empty cache')
    obj=dict(schema=SCHEMA,contract=contract,entries=entries,complete=not missing,
             expected_frames=len(rows),actual_frames=len(entries),missing=missing)
    obj['index_sha256']=digest(obj)
    write_json(a.output,obj)
    print(json.dumps({k:obj[k] for k in ('complete','actual_frames','expected_frames')}))


def load_index(path, partition, allow_partial=False):
    index=json.loads(Path(path).read_text())
    actual=index.pop('index_sha256')
    if digest(index)!=actual: raise ValueError('Index fingerprint mismatch')
    index['index_sha256']=actual
    if index['schema']!=SCHEMA or index['contract']['partition']!=partition:
        raise ValueError('Wrong cache schema/partition (test cannot train)')
    if not index['complete'] and not allow_partial: raise ValueError('Partial cache cannot be formal training')
    for e in index['entries']:
        if e['row']['partition']!=partition or sha256(e['file'])!=e['sha256']:
            raise ValueError('Corrupt frame or partition mismatch')
    return index


def arrays(entry):
    with np.load(entry['file'],allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def forward_arrays(model, item, device):
    f=torch.as_tensor(item['features'],dtype=torch.float32,device=device)
    b=torch.as_tensor(item['base_logit'],dtype=torch.float32,device=device)
    d=torch.as_tensor(item['depth'],dtype=torch.long,device=device)
    return model(f,b,d)


def validate_ranker(model,index,device):
    rows=[];model.eval()
    with torch.no_grad():
        for e in index['entries']:
            x=arrays(e);z=forward_arrays(model,x,device);s=z.sigmoid().cpu().numpy()
            m=candidate_metrics(s.reshape(-1,4),x['count'].reshape(-1,4))
            m.update(domain=e['row']['domain'],scene=e['row']['scene'],
                     mse=float(np.mean((s-x['count']/6)**2)))
            rows.append(m)
    # Explicit domain/scene balance (same-view frames are not IID scenes).
    keys=('selected_q','oracle_q','regret','oracle_at2','mse')
    domains=sorted({r['domain'] for r in rows})
    result={}
    for key in keys:
        result[key]=float(np.mean([np.mean([np.mean([r[key] for r in rows if
                       r['domain']==d and r['scene']==s]) for s in sorted({r['scene'] for r in rows if r['domain']==d})]) for d in domains]))
    return result


def fit(a):
    train=load_index(a.train_index,'train',a.smoke)
    val=load_index(a.val_index,'validation',a.smoke)
    for k in ('config_digest','protocol_sha256','families','seed'):
        if train['contract'][k]!=val['contract'][k]: raise ValueError(f'Train/val {k} differs')
    tr={(e['row']['domain'],e['row']['scene'],e['row']['anno']) for e in train['entries']}
    va={(e['row']['domain'],e['row']['scene'],e['row']['anno']) for e in val['entries']}
    if tr&va: raise ValueError('Same image in rank training and validation')
    if any(s>=100 for _,s,_ in tr|va): raise ValueError('Official test scenes forbidden for fitting/selection')
    if not a.smoke and (len(train['entries'])!=5200 or len(val['entries'])!=40):
        raise ValueError('Formal experiment requires 5200 train +40 held-out-view validation images')
    if a.epochs<1 or a.batch_frames<1 or a.lr<=0 or min(a.point_weight,a.pair_weight)<0:
        raise ValueError('Invalid training settings')
    root=ensure_new_dir(a.output);device=torch.device(a.device)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    model=ResidualRanker(hidden=a.hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    order_rng=np.random.default_rng(a.seed)
    history=[];updates=0
    settings=vars(a).copy();settings.pop('func',None)
    protocol=dict(schema=SCHEMA,training=settings,base_config=train['contract']['config'],
                  config_digest=train['contract']['config_digest'],
                  train_index_sha256=train['index_sha256'],val_index_sha256=val['index_sha256'],
                  source_contract=train['contract'],primary='fixed last epoch',
                  weights=dict(point=a.point_weight,pair=a.pair_weight if a.objective=='rare' else 0),
                  test_images_used=False,trainable_parameters=sum(p.numel() for p in model.parameters()))
    write_json(root/'training_protocol.json',protocol)
    def save(epoch):
        payload=dict(schema=SCHEMA,epoch=epoch,model=model.state_dict(),hidden=a.hidden,
                     protocol=protocol,optimizer=opt.state_dict(),updates=updates,
                     numpy_rng=order_rng.bit_generator.state,torch_rng=torch.get_rng_state())
        tmp=root/'last.pt.tmp';torch.save(payload,tmp);os.replace(tmp,root/'last.pt')
    history.append(dict(epoch=0,updates=0,validation=validate_ranker(model,val,device)))
    save(0)
    for epoch in range(1,a.epochs+1):
        model.train();order=order_rng.permutation(len(train['entries']));stats=[]
        for start in range(0,len(order),a.batch_frames):
            opt.zero_grad(set_to_none=True);losses=[]
            for i in order[start:start+a.batch_frames]:
                entry=train['entries'][int(i)];x=arrays(entry)
                z=forward_arrays(model,x,device)
                point,pair,report=ranking_loss(z,torch.as_tensor(x['count'],device=device),
                    torch.as_tensor(x['family'],device=device),scope=a.pair_scope,
                    max_pairs=a.max_pairs,seed=frame_seed(entry['row'],a.seed+epoch*100003))
                loss=a.point_weight*point+(a.pair_weight*pair if a.objective=='rare' else 0)
                losses.append(loss);stats.append([point.item(),pair.item(),report['used_pairs']])
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite RARE loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step();updates+=1
        validation=validate_ranker(model,val,device)
        history.append(dict(epoch=epoch,updates=updates,mean_point=float(np.mean(stats,axis=0)[0]),
                            mean_pair=float(np.mean(stats,axis=0)[1]),mean_pairs=float(np.mean(stats,axis=0)[2]),
                            validation=validation))
        save(epoch);write_json(root/'history.json',history)
        print(f'[fit {a.objective}] epoch {epoch}/{a.epochs} val={validation}',flush=True)
    write_json(root/'complete.json',dict(status='complete',epochs=a.epochs,updates=updates,
               checkpoint_sha256=sha256(root/'last.pt'),training_protocol_sha256=digest(protocol),
               smoke_only=bool(a.smoke)))


def infer(a):
    from .runtime import Runtime
    p,rows=protocol_rows(a.protocol,'test',a.domain,a.split)
    if a.limit: rows=rows[:a.limit]
    root=ensure_new_dir(a.output)
    rt=Runtime(a.config,a.dataset_root,a.gntrans_root,a.device)
    ranker=None;rk=None
    if a.ranker:
        rk=torch.load(a.ranker,map_location='cpu',weights_only=False)
        if rk['schema']!=SCHEMA or rk['protocol']['config_digest']!=rt.config_digest:
            raise ValueError('Ranker belongs to a different frozen base')
        if rk['protocol']['training'].get('smoke') and not a.limit:
            raise ValueError('Smoke-trained head forbidden for formal inference')
        ranker=ResidualRanker(hidden=rk['hidden']).to(rt.device)
        ranker.load_state_dict(rk['model'],strict=True);ranker.eval()
        if rk['epoch']!=rk['protocol']['training']['epochs']:
            raise ValueError('Primary inference uses preregistered final epoch only')
    policies=a.policies.split(',')
    if ranker is None and policies!=['native']:
        raise ValueError('No ranker: use --policies native')
    records=[]
    for n,row in enumerate(rows):
        snap=rt.snapshot(row,a.seed);bank=snap['grasps']
        if ranker is None: scores=bank[...,0]
        else:
            with torch.inference_mode():
                features=torch.from_numpy(snap['features']).to(rt.device)
                logit=torch.from_numpy(snap['base_logit']).to(rt.device)
                depths=torch.arange(4,device=rt.device)[None].expand(1024,4)
                scores=ranker(features,logit,depths).sigmoid().cpu().numpy()
        for policy in policies:
            result=select_grasps(bank,scores,policy)
            before=len(result)
            if not a.no_collision: result=rt.sensor_collision_filter(row,result)
            dest=root/policy/f"scene_{row['scene']:04d}"/'realsense'/f"{row['anno']:04d}.npy"
            dest.parent.mkdir(parents=True,exist_ok=True);np.save(dest,result)
            records.append(dict(row=row,policy=policy,path=str(dest.relative_to(root)),
                                prefilter_count=before,postfilter_count=len(result),sha256=sha256(dest)))
        # Exact frozen decision traces: these do not claim quality labels/AP.
        atomic_npz(root/'traces'/(frame_key(row)+'.npz'),native_scores=bank[...,0],
                   new_scores=scores,source_angles=snap['angles'],
                   widths=bank[...,1],centers=bank[:,0,13:16])
        print(f'[infer] {n+1}/{len(rows)} {frame_key(row)}',flush=True)
    write_json(root/'inference_manifest.json',dict(status='complete',schema=SCHEMA,
       config_digest=rt.config_digest,base_sha256=rt.config['checkpoint_sha256'],
       ranker_sha256=sha256(a.ranker) if a.ranker else None,domain=a.domain,split=a.split,
       seed=a.seed,policies=policies,frames=len(rows),smoke_only=bool(a.limit),
       collision='none' if a.no_collision else 'original_sensor:.01voxel,.01threshold,.05approach',
       angle='frozen_native_winner',width='native_per_depth_frozen',records=records))


def evaluate(a):
    root=Path(a.inference);manifest=json.loads((root/'inference_manifest.json').read_text())
    if manifest['smoke_only'] or manifest['frames']!=780 or manifest['status']!='complete':
        raise ValueError('Official 10% eval requires complete 30 scenes x26 frames (not smoke)')
    for item in manifest['records']:
        f=root/item['path']
        if not f.exists() or sha256(f)!=item['sha256']: raise ValueError('Dump corruption')
        x=np.load(f)
        if x.ndim!=2 or x.shape[1]!=17 or not np.isfinite(x).all(): raise ValueError('Bad dump')
    for policy in manifest['policies']:
        subprocess.run([sys.executable,'eval.py','--dataset_root',a.dataset_root,
                        '--dump_dir',str(root/policy),'--camera','realsense',
                        '--split',manifest['split'],'--num_workers',str(a.workers),
                        '--sample_interval','10'],check=True)
        out=root/policy/f"ap_{manifest['split']}_realsense.npy"
        ap=np.load(out)
        if ap.shape!=(30,26,50,6) or not np.isfinite(ap).all() or ap.min()<0 or ap.max()>1:
            raise RuntimeError(f'Official AP shape/range validation failed: {ap.shape}')
    write_json(root/'evaluation_complete.json',dict(status='complete',official=True,
        split=manifest['split'],domain=manifest['domain'],frame_count=780,
        policies={v:dict(ap_percent=float(np.load(root/v/f"ap_{manifest['split']}_realsense.npy").mean()*100),
                        sha256=sha256(root/v/f"ap_{manifest['split']}_realsense.npy")) for v in manifest['policies']}))


def compare(a):
    """Paired scene bootstrap with all three splits; supplied files must match."""
    if len(a.left)!=3 or len(a.right)!=3: raise ValueError('Provide Seen/Similar/Novel arrays in that order')
    deltas=[];means=[]
    for expected_split,l,r in zip(('test_seen','test_similar','test_novel'),a.left,a.right):
        metas=[]
        for path in (l,r):
            mf=Path(path).parent.parent/'inference_manifest.json'
            if not mf.exists(): raise ValueError('AP comparison requires inference provenance manifests')
            metas.append(json.loads(mf.read_text()))
        for key in ('config_digest','base_sha256','domain','split','frames','seed','collision','angle','width'):
            if metas[0][key]!=metas[1][key]: raise ValueError(f'AP comparison protocol mismatch: {key}')
        if metas[0]['split']!=expected_split: raise ValueError('AP arrays must be ordered Seen/Similar/Novel')
        x,y=np.load(l),np.load(r)
        if x.shape!=y.shape or x.shape!=(30,26,50,6): raise ValueError('AP protocol/shape mismatch')
        if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite AP')
        d=(y-x).mean(axis=(1,2,3))*100;deltas.append(d)
        means.append(dict(left=100*float(x.mean()),right=100*float(y.mean()),delta_pp=float(d.mean())))
    rng=np.random.default_rng(a.seed)
    boot=np.stack([d[rng.integers(0,30,size=(a.bootstrap,30))].mean(1) for d in deltas])
    out=dict(splits=dict(zip(('seen','similar','novel'),means)),
             mean_delta_pp=float(np.mean(deltas)),ci95=np.quantile(boot.mean(0),[.025,.975]).tolist(),
             scope='scene-cluster paired; not training-seed uncertainty',
             source_sha256={p:sha256(p) for p in a.left+a.right})
    write_json(a.output,out)


def diagnose_cache(a):
    index=load_index(a.index,a.partition,a.allow_partial)
    ck=torch.load(a.ranker,map_location='cpu',weights_only=False)
    if ck['protocol']['config_digest']!=index['contract']['config_digest']:
        raise ValueError('Cache and ranker frozen base identity mismatch')
    model=ResidualRanker(hidden=ck['hidden']).to(a.device)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    output=[]
    with torch.no_grad():
        for entry in index['entries']:
            x=arrays(entry);s=forward_arrays(model,x,a.device).sigmoid().cpu().numpy().reshape(-1,4)
            base=x['grasps'][:,0].reshape(-1,4);quality=x['count'].reshape(-1,4)
            b=candidate_metrics(base,quality);n=candidate_metrics(s,quality)
            if abs(b['oracle_q']-n['oracle_q'])>1e-12: raise RuntimeError('Oracle changed under scoring-only update')
            if abs((n['selected_q']-b['selected_q'])-(b['regret']-n['regret']))>1e-8:
                raise RuntimeError('Selected-gain / regret-reduction identity failed')
            output.append(dict(row=entry['row'],native=b,ranker=n))
    write_json(a.output,dict(schema=SCHEMA,partition=a.partition,records=output,
        source_index_sha256=index['index_sha256'],ranker_sha256=sha256(a.ranker),
        warning='Fixed sampled families; not AP. Bootstrap scenes, not individual actions.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='cmd',required=True)
    m=sp.add_parser('manifest');m.add_argument('--output',required=True)
    m.set_defaults(func=lambda a:write_json(a.output,make_protocol()))
    b=sp.add_parser('bind');b.add_argument('--template',required=True);b.add_argument('--checkpoint',required=True);b.add_argument('--output',required=True)
    def bind(a):
        from .runtime import bind_config
        bind_config(a.template,a.checkpoint,a.output)
    b.set_defaults(func=bind)
    m=sp.add_parser('mine');m.set_defaults(func=mine)
    for k in ('config','protocol','dataset-root','output'):m.add_argument('--'+k,required=True)
    m.add_argument('--gntrans-root',default='');m.add_argument('--partition',choices=('train','validation','test'),required=True)
    m.add_argument('--domain',choices=('original','gntrans'));m.add_argument('--split')
    m.add_argument('--families',type=int,default=32);m.add_argument('--seed',type=int,default=0)
    m.add_argument('--shard-index',type=int,default=0);m.add_argument('--num-shards',type=int,default=1)
    m.add_argument('--limit',type=int,default=0);m.add_argument('--device',default='cuda:0')
    i=sp.add_parser('index');i.set_defaults(func=index_cache)
    for k in ('protocol','cache','output'):i.add_argument('--'+k,required=True)
    i.add_argument('--partition',required=True,choices=('train','validation','test'));i.add_argument('--allow-partial',action='store_true')
    f=sp.add_parser('fit');f.set_defaults(func=fit)
    for k in ('train-index','val-index','output'):f.add_argument('--'+k,required=True)
    f.add_argument('--objective',choices=('point','rare'),required=True)
    f.add_argument('--pair-scope',choices=('image','family'),default='image');f.add_argument('--max-pairs',type=int,default=2048)
    f.add_argument('--point-weight',type=float,default=10);f.add_argument('--pair-weight',type=float,default=.5)
    f.add_argument('--epochs',type=int,default=20);f.add_argument('--batch-frames',type=int,default=4)
    f.add_argument('--hidden',type=int,default=64);f.add_argument('--lr',type=float,default=1e-4);f.add_argument('--weight-decay',type=float,default=1e-4)
    f.add_argument('--seed',type=int,default=0);f.add_argument('--device',default='cuda:0');f.add_argument('--smoke',action='store_true')
    inf=sp.add_parser('infer');inf.set_defaults(func=infer)
    for k in ('config','protocol','dataset-root','output'):inf.add_argument('--'+k,required=True)
    inf.add_argument('--gntrans-root',default='');inf.add_argument('--ranker');inf.add_argument('--domain',required=True,choices=('original','gntrans'))
    inf.add_argument('--split',required=True,choices=('test_seen','test_similar','test_novel'))
    inf.add_argument('--policies',default='both,depth_only,rank_only');inf.add_argument('--seed',type=int,default=0)
    inf.add_argument('--device',default='cuda:0');inf.add_argument('--limit',type=int,default=0);inf.add_argument('--no-collision',action='store_true')
    ev=sp.add_parser('eval');ev.set_defaults(func=evaluate)
    ev.add_argument('--dataset-root',required=True);ev.add_argument('--inference',required=True);ev.add_argument('--workers',type=int,default=16)
    c=sp.add_parser('compare');c.set_defaults(func=compare)
    c.add_argument('--left',action='append',required=True);c.add_argument('--right',action='append',required=True)
    c.add_argument('--output',required=True);c.add_argument('--seed',type=int,default=0);c.add_argument('--bootstrap',type=int,default=5000)
    dg=sp.add_parser('diagnose-cache');dg.set_defaults(func=diagnose_cache)
    for k in ('index','ranker','output'):dg.add_argument('--'+k,required=True)
    dg.add_argument('--partition',choices=('validation','test'),default='test')
    dg.add_argument('--allow-partial',action='store_true');dg.add_argument('--device',default='cpu')
    a=p.parse_args();a.func(a)


if __name__=='__main__':main()
