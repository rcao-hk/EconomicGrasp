"""Cache, train, stream inference, and official evaluation for evidence access."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .core import *
from .model import Config,ContextScorer,intervene
from .runtime import create
from interactive_grasp.objectives import sparse_quality_loss,pair_changes
from interactive_grasp.train import atomic_torch


class Reuse:
    """Read immutable Stage1 or new experiment labels, never join by query ID."""
    def __init__(self,paths,source):
        self.rows={};self.identities={}
        for path in paths:
            idx=load_signed(path);c=idx['contract']
            if idx.get('schema') not in (SCHEMA,'interactive-frozen-grasp-stage1-v1'):raise ValueError('Unsupported reuse index schema')
            for k in ('base_sha256','selector_sha256'):
                if c['source'][k]!=source[k]:raise ValueError('Reuse source checkpoint mismatch')
            self.identities[str(Path(path).resolve())]=sha256(path)
            for e in idx['entries']:
                key=frame_key(e['row'])
                if key in self.rows:raise ValueError('Duplicate frame in reuse indices')
                self.rows[key]=e
    def lookup(self,row,ids):
        y=np.full(len(ids),-1,dtype=np.int64);e=self.rows.get(frame_key(row))
        if e is None:return y
        if sha256(e['path'])!=e['sha256']:raise ValueError('Reuse file SHA mismatch')
        with np.load(e['path'],allow_pickle=False) as z:
            if not np.array_equal(physical_ids(z['grasps']),z['physical_id']):raise ValueError('Reuse physical mismatch')
            table={}
            for pid,q,known in zip(z['physical_id'],z['quality_count'],z['label_mask']):
                if not known:continue
                if not 0<=q<=6:raise ValueError('Bad old quality')
                if str(pid) in table and table[str(pid)]!=int(q):raise ValueError('Conflicting same-action labels')
                table[str(pid)]=int(q)
        return np.array([table.get(str(pid),-1) for pid in ids],dtype=np.int64)


def export(a):
    plan=load_signed(a.plan);validate_plan(plan)
    if not 0<=a.shard<a.shards or not 1<=a.labels<=1024 or a.partition not in ('train','validation'):raise ValueError('Export bounds')
    rows=[r for r in plan[a.partition] if not a.domain or r['domain']==a.domain]
    rows=[r for i,r in enumerate(rows) if i%a.shards==a.shard][:a.limit or None]
    if not rows:raise ValueError('Empty export')
    rt=create(a);reuse=Reuse(a.reuse_index,rt.provenance)
    contract={'schema':SCHEMA,'source':rt.provenance,'provider':a.provider,'code':code_hash(),
              'plan':plan['fingerprint'],'partition':a.partition,'grid_size':a.grid_size,'queries':1024,
              'seed':a.seed,'labels':a.labels,'dtype':'float32','reuse':reuse.identities,'label_evaluator':'official_exact_actions_no_nearest_anchor_mask'}
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        path=root/(frame_key(row)+'.npz');meta=path.with_suffix('.json')
        if path.exists() or meta.exists():
            if not a.resume or not(path.exists() and meta.exists()):raise FileExistsError(path)
            r=load_signed(meta)
            if r['contract']!=contract or r['row']!=row or sha256(path)!=r['sha256']:raise ValueError('Resume cache changed')
            continue
        arr,parity=rt.snapshot(row,a.seed);n=len(arr['grasps'])
        arr['physical_id']=physical_ids(arr['grasps']);arr['sensor_keep']=rt.sensor_keep(row,arr['grasps']).astype(bool)
        selected=np.sort(np.random.default_rng(frame_seed(row,a.seed)).choice(n,a.labels,replace=False))
        y=np.full(n,-1,dtype=np.int64);mask=np.zeros(n,dtype=bool);mask[selected]=True
        y[selected]=reuse.lookup(row,arr['physical_id'][selected]);missing=selected[y[selected]<0]
        if len(missing):y[missing]=rt.label(row,arr['grasps'][missing])
        arr['quality_count']=y;arr['label_mask']=mask
        validate(arr,1024,a.grid_size)
        if shutil.disk_usage(root).free < sum(v.nbytes for v in arr.values())+a.reserve_gib*(1<<30):raise OSError('Insufficient disk reserve')
        atomic_npz(path,arr);atomic_json(meta,signed({'row':row,'contract':contract,'sha256':sha256(path),
            'parity':parity,'labels_reused':int(len(selected)-len(missing)),'labels_new':int(len(missing))}))
        print(f'[export] {i+1}/{len(rows)} {frame_key(row)} labelled={mask.sum()} new={len(missing)}',flush=True)


def index(a):
    plan=load_signed(a.plan);validate_plan(plan);root=Path(a.cache);entries=[];c=None;dims=None;missing=[]
    for row in plan[a.partition]:
        p=root/(frame_key(row)+'.npz');m=p.with_suffix('.json')
        if not m.exists():missing.append(frame_key(row));continue
        r=load_signed(m)
        if r['row']!=row or r['contract']['plan']!=plan['fingerprint'] or r['contract']['partition']!=a.partition:raise ValueError('Frame/plan mismatch')
        if c is None:c=r['contract']
        if c!=r['contract'] or sha256(p)!=r['sha256']:raise ValueError('Mixed/corrupt cache')
        e={'row':row,'path':str(p.resolve()),'sha256':r['sha256']};arr=read_frame(e,c)
        d=(arr['features'].shape[1],arr['scene_features'].shape[1])
        if dims is None:dims=d
        if dims!=d:raise ValueError('Feature dimensions vary')
        entries.append(e)
    if not entries or (missing and not a.smoke):raise ValueError(f'Missing {len(missing)} frames')
    if Path(a.output).exists():raise FileExistsError(a.output)
    atomic_json(a.output,signed({'schema':SCHEMA,'partition':a.partition,'complete':not missing,
        'missing':missing,'contract':c,'entries':entries,'feature_dim':dims[0],'scene_dim':dims[1]}))


def scene_mean(rows,key):
    ds=[]
    for domain in sorted({r['domain'] for r in rows}):
        ss=[]
        for scene in sorted({r['scene'] for r in rows if r['domain']==domain}):
            v=[r[key] for r in rows if r['domain']==domain and r['scene']==scene and r[key] is not None]
            if v:ss.append(float(np.mean(v)))
        if ss:ds.append(float(np.mean(ss)))
    return float(np.mean(ds)) if ds else None


def metrics(a,score):
    mask=a['label_mask'];y=a['quality_count'];base=a['grasps'][:,0]
    out=pair_changes(base,score,y,mask)
    out.update(mse=float(np.mean((score[mask]-y[mask]/6)**2)),native_mse=float(np.mean((base[mask]-y[mask]/6)**2)),
               labels=int(mask.sum()),residual_score_abs=float(np.mean(np.abs(score-base))))
    ids=np.flatnonzero(mask);old=ids[np.argsort(-base[ids],kind='stable')];new=ids[np.argsort(-score[ids],kind='stable')]
    out['labelled_top1_q']=float(y[new[0]]/6);out['native_labelled_top1_q']=float(y[old[0]]/6)
    # A fully labelled post-sensor pool can support pre-NMS top-K quality.
    survivors=np.flatnonzero(a['sensor_keep']);full=mask[survivors].all()
    out['sensor_candidates']=len(survivors);out['survivor_labels_complete']=bool(full)
    for k in (1,5,10,50):
        ns=survivors[np.argsort(-score[survivors],kind='stable')[:k]]
        bs=survivors[np.argsort(-base[survivors],kind='stable')[:k]]
        out[f'pre_nms_top{k}_q']=float(y[ns].mean()/6) if full and len(ns) else None
        out[f'native_pre_nms_top{k}_q']=float(y[bs].mean()/6) if full and len(bs) else None
        out[f'top{k}_label_coverage']=float(mask[ns].mean()) if len(ns) else None
    return out


@torch.no_grad()
def evaluate_model(model,idx,device):
    model.eval();rows=[]
    for e in idx['entries']:
        arr=read_frame(e,idx['contract']);x,y,mask=tensors(arr,device)
        z=model(**x);rows.append({**e['row'],**metrics(arr,z.sigmoid()[0].cpu().numpy())})
    keys=('mse','native_mse','accuracy','base_accuracy','labelled_top1_q','native_labelled_top1_q',
          'pre_nms_top1_q','pre_nms_top5_q','pre_nms_top50_q','residual_score_abs')
    return {'summary':{k:scene_mean(rows,k) for k in keys},'records':rows,
            'scope':'exact labelled actions; pre-NMS metrics are NOT official AP'}


def fit(a):
    tr=read_index(a.train_index,'train',a.smoke);va=read_index(a.val_index,'validation',a.smoke)
    for k in ('source','provider','code','plan','grid_size','queries','seed','dtype'):
        if tr['contract'][k]!=va['contract'][k]:raise ValueError(f'Train/val differ: {k}')
    if {e['row']['scene'] for e in tr['entries']}&{e['row']['scene'] for e in va['entries']}:raise ValueError('Scene split leakage')
    if (tr['feature_dim'],tr['scene_dim'])!=(va['feature_dim'],va['scene_dim']):raise ValueError('Dimension mismatch')
    if tr['contract']['code']!=code_hash():raise ValueError('Cache/current experiment code identity differs')
    if min(a.epochs,a.accumulate,a.max_pairs)<1 or a.lr<=0:raise ValueError('Bad optimizer settings')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(a.cpu_threads)
    cfg=Config(tr['feature_dim'],tr['scene_dim'],a.variant,a.hidden,a.heads,a.layers,a.seed,a.residual_scale)
    model=ContextScorer(cfg).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    hyper={k:getattr(a,k) for k in ('epochs','accumulate','lr','weight_decay','max_pairs','seed','smoke','train_scene_shuffle')}
    if a.train_scene_shuffle and a.variant!='scene':raise ValueError('Training spatial-shuffle control is Scene-only')
    protocol={'schema':SCHEMA,'model':asdict(cfg),'description':model.describe(),'hyper':hyper,
              'source':tr['contract']['source'],'plan':tr['contract']['plan'], 'code':code_hash(),
              'train_index':tr['fingerprint'],'validation_index':va['fingerprint'],
              'grid_size':tr['contract']['grid_size'],'loss':{'mse':10.,'pair':.5,'ties':'excluded'},
              'selection':'fixed_final_epoch','expected_queries':tr['contract']['queries']}
    start=updates=0;history=[]
    if a.resume:
        ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if ck['protocol']!=protocol:raise ValueError('Resume config/code/data mismatch')
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch'];updates=ck['updates'];history=ck['history']
        if start>a.epochs:raise ValueError('Resume past planned epoch')
    else:
        atomic_json(root/'protocol.json',signed(protocol))
        check=read_frame(va['entries'][0],va['contract']);xx,_,_=tensors(check,a.device)
        with torch.no_grad():
            parity=float((model(**xx)-xx['base_logit']).abs().max())
        if parity>1e-6:raise RuntimeError('Epoch-zero baseline parity failed')
        history=[{'epoch':0,'native_logit_max_abs':parity,'validation':evaluate_model(model,va,a.device)['summary']}]
        # Epoch zero must be native; retain a snapshot for audit, never test-select it.
        atomic_torch(root/'epoch0.pt',{'schema':SCHEMA,'protocol':protocol,'model':model.state_dict(),'epoch':0})
    for epoch in range(start+1,a.epochs+1):
        order=np.random.default_rng(a.seed+100003*epoch).permutation(len(tr['entries']));train_stats=[];model.train()
        for step in range(0,len(order),a.accumulate):
            group=order[step:step+a.accumulate];opt.zero_grad(set_to_none=True)
            for pos in group:
                e=tr['entries'][int(pos)];arr=read_frame(e,tr['contract']);x,y,mask=tensors(arr,a.device)
                rng=frame_seed(e['row'],a.seed+100003*epoch)
                if a.train_scene_shuffle:x=intervene(x,'scene_appearance_shuffle',rng)
                z=model(**x);loss,info=sparse_quality_loss(z,y,mask,x['action_valid'],pair_seed=rng,max_pairs=a.max_pairs)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss/len(group)).backward();train_stats.append(info)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1
        val=evaluate_model(model,va,a.device)
        history.append({'epoch':epoch,'updates':updates,'train_mse':float(np.mean([s['mse'] for s in train_stats])),
            'train_pair':float(np.mean([s['pair'] for s in train_stats])),'eligible_pairs':int(sum(s['eligible_pairs'] for s in train_stats)),
            'used_pairs':int(sum(s['used_pairs'] for s in train_stats)),'validation':val['summary']})
        atomic_torch(root/'last.pt',{'schema':SCHEMA,'protocol':protocol,'model':model.state_dict(),
                                   'optimizer':opt.state_dict(),'epoch':epoch,'updates':updates,'history':history})
        atomic_json(root/'history.json',history);atomic_json(root/'validation_last.json',val)
        print(f'[fit {a.variant}] epoch={epoch} updates={updates} {val["summary"]}',flush=True)
    atomic_json(root/'complete.json',signed({'schema':SCHEMA,'status':'complete','checkpoint':sha256(root/'last.pt'),
                                            'epochs':a.epochs,'updates':updates,'smoke':a.smoke}))


def load_model(path,device,smoke=False):
    ck=torch.load(path,map_location='cpu',weights_only=False);p=ck['protocol']
    if ck.get('schema')!=SCHEMA or ck['epoch']!=p['hyper']['epochs']:raise ValueError('Need final completed model')
    if p['code']!=code_hash():raise ValueError('Model implementation/source SHA differs from training')
    if p['hyper']['smoke'] and not smoke:raise ValueError('Smoke model forbidden')
    model=ContextScorer(Config(**p['model'])).to(device);model.load_state_dict(ck['model'],strict=True);model.eval()
    return model,p


def load_models(items,source,device,smoke=False):
    models={};protocols={};identities={};common=None
    for item in items:
        name,path=item.split('=',1)
        if name not in VARIANTS or name in models:raise ValueError('Unique local/candidate/scene=checkpoint required')
        model,p=load_model(path,device,smoke)
        if p['source']!=source or p['model']['variant']!=name:raise ValueError('Source/head identity mismatch')
        comparison={k:v for k,v in p.items() if k!='description'};comparison['model']=dict(p['model']);comparison['model'].pop('variant')
        if common is None:common=comparison
        if comparison!=common:raise ValueError('Training budget/data/shared settings differ')
        models[name]=model;protocols[name]=p;identities[name]={'path':str(Path(path).resolve()),'sha256':sha256(path)}
    if not models:raise ValueError('No models')
    return models,protocols,identities


@torch.no_grad()
def infer(a):
    from interactive_grasp.infer import emit_frame
    plan=load_signed(a.plan);validate_plan(plan)
    rows=[r for r in plan['test'] if r['domain']==a.domain and r['split']==a.split][:a.limit or None]
    if not rows:raise ValueError('No test frames')
    if not a.limit and not a.reference_root:raise ValueError('Formal run needs frozen MSE pre-NMS reference dumps')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True);rt=create(a)
    models,ps,weights=load_models(a.weight,rt.provenance,a.device,bool(a.limit));records=[];max_error=0.
    for p in ps.values():
        if p['plan']!=plan['fingerprint'] or p['grid_size']!=a.grid_size:raise ValueError('Plan/grid mismatch')
    for i,row in enumerate(rows):
        arr,parity=rt.snapshot(row,a.seed);n=len(arr['grasps']);arr['physical_id']=physical_ids(arr['grasps'])
        arr['sensor_keep']=rt.sensor_keep(row,arr['grasps']).astype(bool);arr['label_mask']=np.zeros(n,dtype=bool)
        arr['quality_count']=np.full(n,-1,dtype=np.int64);validate(arr,1024,a.grid_size)
        if a.reference_root:
            path=Path(a.reference_root)/f'scene_{row["scene"]:04d}'/'realsense'/f'{row["anno"]:04d}.npy'
            old=np.load(path,allow_pickle=False);now=arr['grasps'][arr['sensor_keep']]
            if old.shape!=now.shape:raise ValueError('Source reference shape mismatch')
            error=float(np.max(np.abs(old-now))) if len(now) else 0.
            if error>1e-6 or not np.isfinite(error):raise RuntimeError(f'Native reference parity failed {error}')
            max_error=max(max_error,error)
        x,_,_=tensors(arr,a.device);scores={'native':arr['grasps'][:,0]}
        for name,model in models.items():
            scores[name]=model(**x).sigmoid()[0].cpu().numpy()
            if not np.isfinite(scores[name]).all():raise FloatingPointError('Nonfinite scores')
        records.extend(emit_frame(root,row,arr,scores))
        print(f'[infer] {i+1}/{len(rows)} {frame_key(row)}',flush=True)
    atomic_json(root/'inference_manifest.json',signed({'schema':SCHEMA,'status':'complete','source':rt.provenance,
        'source_code_sha256':code_hash(),'split':a.split,'domain':a.domain,'seed':a.seed,'frames':len(rows),
        'plan_sha256':plan['fingerprint'],'partial_smoke':bool(a.limit),'source_reference_max_abs':max_error,
        'source_reference_root':a.reference_root,'variants':['native',*models],'weights':weights,'protocols':ps,'records':records}))


def official_eval(a):
    from interactive_grasp.infer import evaluate
    evaluate(a) # unmodified production eval.py; complete 30x256 frames required
