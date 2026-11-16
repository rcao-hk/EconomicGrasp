"""P1: matched Pred-Analytic vs privileged Clean-Visible quality-reader study.

The *only* new input in the clean arm is the rendered-visible depth available
in offline train/validation. No teacher information enters test inference.
The two arms share physically identical banks, exact labels, readout module,
optimizer, seeds, batches, fixed 20-epoch budget and validation selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .data import (SCHEMA,read_index,read_frame,sha256,digest,signed,load_signed,
                   atomic_json,atomic_npz,frame_key)
from .geometry import support_points, VisibleSurfaceTeacher
from .runtime import TargetLoader
from .model import ModelConfig, PredictedGeometryControl, quality_loss
from .experiment import family_metrics, balanced
from .p0_relative_decision import scene_bootstrap

P1_SCHEMA='shared-visible-geometry-p1-analytic-v1'
ARMS=('predicted','clean_visible')


def physical_tokens(teacher,xyz):
    labels=teacher.label(np.asarray(xyz,dtype=np.float32).reshape(-1,3))
    return np.concatenate((labels['udf'][:,None],labels['projective'][:,None],
                 np.eye(4,dtype=np.float32)[labels['state']]),axis=-1).astype(np.float32)


def validated_target(index,entry,frame,loader):
    """Check actual rendered teacher against the original frame's signed sidecar."""
    if index['partition'] not in ('train','validation') or entry['row']['partition'] not in ('train','validation'):
        raise ValueError('No test or unknown partition may access rendered geometry')
    if index['contract']['target_config']!=loader.config_sha256:
        raise ValueError('Rendered teacher config differs from original export')
    sidecar=Path(entry['path']).with_suffix('.json')
    if not sidecar.is_file():raise FileNotFoundError('Original signed target provenance required')
    old=load_signed(sidecar)
    if old['row']!=entry['row'] or old['sha256']!=entry['sha256'] or old['contract']['target_config']!=loader.config_sha256:
        raise ValueError('Original frame/teacher signature mismatch')
    depth,provenance=loader.load(entry['row'],frame['K'],frame['pred_depth'].shape)
    for k in ('config_sha256','raw_geometry_identity','depth_sha256','intrinsics_sha256'):
        if provenance[k]!=old['target'][k]:raise ValueError(f'Clean visible teacher changed: {k}')
    return depth,provenance


def make_cache_frame(frame,clean_depth,tau,band):
    g=torch.as_tensor(frame['bank'],dtype=torch.float32)
    world,local=support_points(g)
    xyz=world.reshape(-1,3).detach().cpu().numpy()
    sources={'predicted':frame['pred_depth'],'clean_visible':clean_depth}
    arrays={k:frame[k].copy() for k in ('quality_count','physical_id','family_id')}
    arrays.update(local_xyz=local.cpu().numpy().astype(np.float32),
                  bank_dims=frame['bank'][...,1:4].astype(np.float32))
    for name,depth in sources.items():
        teacher=VisibleSurfaceTeacher(depth,frame['K'],tau,band)
        tokens=physical_tokens(teacher,xyz).reshape(*g.shape[:2],75,6)
        arrays[name+'_physics']=tokens
    validate_cache_frame(arrays)
    return arrays


def validate_cache_frame(arr):
    q=np.asarray(arr['quality_count']);f,k=q.shape
    if k!=7 or f<1 or q.dtype.kind not in 'iu' or np.any((q<0)|(q>6)):
        raise ValueError('Expected exact seven-action quality labels')
    if arr['local_xyz'].shape!=(f,k,75,3) or arr['bank_dims'].shape!=(f,k,3):
        raise ValueError('Geometry support mismatch')
    if arr['physical_id'].shape!=(f,k) or len(arr['family_id'])!=f:
        raise ValueError('Physical action ID/size mismatch')
    for name in ARMS:
        t=arr[name+'_physics']
        if t.shape!=(f,k,75,6) or not np.isfinite(t).all():
            raise ValueError('Analytic physical evidence is malformed')
        if not ((t[...,0]>=0)&(t[...,0]<=1)).all() or not ((t[...,1]>=-1)&(t[...,1]<=1)).all():
            raise ValueError('Invalid distance range')
        if not np.allclose(t[...,2:].sum(-1),1.,atol=1e-6):
            raise ValueError('Observation state is not one-hot')
    if not np.isfinite(arr['local_xyz']).all() or not np.isfinite(arr['bank_dims']).all():
        raise ValueError('Nonfinite support positions')


def build(a):
    idx=read_index(a.index,None,a.smoke)
    if idx['partition'] not in ('train','validation'):
        raise ValueError('Clean-visible build is train/validation only')
    if a.limit<0 or (a.limit and not a.smoke):raise ValueError('Partial build must be marked --smoke')
    loader=TargetLoader(a.targets)
    if idx['contract']['target_config']!=loader.config_sha256:
        raise ValueError('Teacher config identity mismatch')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    entries=[];source=idx['entries'][:a.limit] if a.limit else idx['entries']
    # Both arms are materialized within one atomic frame to guarantee pairing.
    contract=dict(schema=P1_SCHEMA,partition=idx['partition'],original_index_sha256=sha256(a.index),
                  original_index_fingerprint=idx['fingerprint'],source=idx['contract']['source'],
                  plan=idx['contract']['plan'],target_config=loader.config_sha256,
                  tau=idx['contract']['tau'],band=idx['contract']['band'],
                  offsets_m=idx['contract']['offsets_m'],code_sha256=sha256(__file__),
                  input_modes=list(ARMS),full=not bool(a.limit))
    ch=digest(contract)
    for i,e in enumerate(source):
        dest=root/(frame_key(e['row'])+'.npz');meta=dest.with_suffix('.json')
        if dest.exists() or meta.exists():
            if not a.resume or not (dest.is_file() and meta.is_file()):raise FileExistsError(dest)
            old=load_signed(meta)
            if old['contract_hash']!=ch or old['source_frame_sha256']!=e['sha256'] or old['sha256']!=sha256(dest):
                raise ValueError('P1 cache resume contract mismatch')
            # Source rendered geometry must still exist and match prior identity.
            for key in ('depth','intrinsics'):
                if sha256(old['teacher'][key+'_path'])!=old['teacher'][key+'_sha256']:
                    raise ValueError('Rendered source changed during resume')
        else:
            frame=read_frame(e)
            clean,provenance=validated_target(idx,e,frame,loader)
            vals=make_cache_frame(frame,clean,idx['contract']['tau'],idx['contract']['band'])
            atomic_npz(dest,vals)
            atomic_json(meta,signed(dict(row=e['row'],contract_hash=ch,source_frame_sha256=e['sha256'],
                                         teacher=provenance,sha256=sha256(dest))))
        entries.append(dict(row=e['row'],path=str(dest.resolve()),sha256=sha256(dest),
                            original_frame_sha256=e['sha256']))
        if (i+1)%100==0 or i+1==len(source):
            print(f'[p1 build {idx["partition"]}] {i+1}/{len(source)}',flush=True)
    manifest=root/'index.json'
    if manifest.exists():
        old=load_signed(manifest)
        if old['contract']!=contract or old['entries']!=entries:raise FileExistsError('Existing P1 index disagrees')
    else:
        atomic_json(manifest,signed(dict(schema=P1_SCHEMA,partition=idx['partition'],
                          complete=bool(idx['complete'] and not a.limit),contract=contract,entries=entries)))


def read_cache_index(path,partition=None,smoke=False):
    idx=load_signed(path)
    if idx['schema']!=P1_SCHEMA or (partition and idx['partition']!=partition):raise ValueError('Wrong P1 index')
    if not idx['complete'] and not smoke:raise ValueError('P1 smoke cache forbidden in formal training')
    if not idx['entries']:raise ValueError('Empty P1 index')
    seen=set()
    for e in idx['entries']:
        key=(e['row']['domain'],e['row']['scene'],e['row']['anno'])
        if e['row']['partition']!=idx['partition'] or e['row']['scene']>=100 or key in seen:
            raise ValueError('Mixed partition, test leakage or duplicate P1 row')
        seen.add(key)
        if sha256(e['path'])!=e['sha256']:raise ValueError('P1 cache corruption')
    return idx


def load_cached(e):
    with np.load(e['path'],allow_pickle=False) as z:a=dict(z)
    validate_cache_frame(a)
    return a


def score_cached(model,batch,arm,device):
    if arm not in ARMS:raise ValueError('Invalid P1 geometry source')
    phys=torch.as_tensor(batch[arm+'_physics'],device=device,dtype=torch.float32)
    local=torch.as_tensor(batch['local_xyz'],device=device,dtype=torch.float32)
    dims=torch.as_tensor(batch['bank_dims'],device=device,dtype=torch.float32)
    if phys.ndim!=5 or phys.shape[-2:]!=(75,6) or local.shape!=(*phys.shape[:-1],3) or dims.shape!=(*phys.shape[:3],3):
        raise ValueError('P1 aligned batched [B,F,K,75,6] evidence required')
    # EXACT same 9D point reader + mean/max pooling + 3D dimensions head used
    # by the existing PredictedGeometryControl; no hidden source tag is passed.
    token=model.point_reader(torch.cat((phys,local/.05),-1))
    descriptor=torch.cat((token.mean(-2),token.amax(-2),dims/.1),-1)
    return model.quality(descriptor)[...,0]


def collate_cache(frames):
    if not frames:raise ValueError('Empty cache batch')
    keys=('predicted_physics','clean_visible_physics','local_xyz','bank_dims','quality_count')
    for k in keys:
        shapes={f[k].shape for f in frames}
        if len(shapes)!=1:raise ValueError('Ragged evidence or action families: '+k)
    return {k:np.stack([f[k] for f in frames]) for k in keys}


def balanced_gain(rows):
    return balanced(rows,'gain')


@torch.no_grad()
def evaluate(model,arm,index,device,batch_size):
    model.eval();rows=[]
    entries=index['entries']
    for start in range(0,len(entries),batch_size):
        chunk=entries[start:start+batch_size]
        a=collate_cache([load_cached(e) for e in chunk])
        z=score_cached(model,a,arm,device).cpu().numpy()
        for i,e in enumerate(chunk):
            m=family_metrics(z[i],a['quality_count'][i])
            rows.append(dict(row=e['row'],metrics=m))
    return dict(summary={k:balanced(rows,k) for k in ('native_q','selected_q','oracle_q','gain','regret',
                    'rescue','harm','move','pair_accuracy','quality_mse')},rows=rows)


def fit(a):
    if a.arm not in ARMS or min(a.batch_size,a.accumulate,a.val_batch_size,a.epochs,a.cpu_threads)<1:
        raise ValueError('Invalid P1 train settings')
    if a.lr<=0 or a.weight_decay<0:raise ValueError('Invalid learning rate or weight decay')
    tr=read_cache_index(a.train_index,'train',a.smoke)
    va=read_cache_index(a.val_index,'validation',a.smoke)
    for field in ('source','plan','target_config','tau','band','offsets_m'):
        if tr['contract'][field]!=va['contract'][field]:raise ValueError('P1 train/val source mismatch: '+field)
    if {x['row']['scene'] for x in tr['entries']} & {x['row']['scene'] for x in va['entries']}:
        raise ValueError('P1 scene holdout violated')
    if not a.smoke and (len(tr['entries']),len(va['entries']))!=(2560,160):
        raise ValueError('Expected complete formal P1 train/validation manifests')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.cpu_threads);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=ModelConfig(feature_channels=1,hidden=a.hidden,truncation_m=tr['contract']['tau'])
    # Same class, same state-dict keys and initialization for both arms.
    model=PredictedGeometryControl(cfg).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    effective=a.batch_size*a.accumulate
    settings=dict(schema=P1_SCHEMA,arm=a.arm,train_index=tr['fingerprint'],val_index=va['fingerprint'],
             train_source=tr['contract']['source'],target_config=tr['contract']['target_config'],
             code_sha256=sha256(__file__),architecture='PredictedGeometryControl.point_reader+quality',
             quality_loss='10*MSE(sigmoid(z),count/6)+0.5*within-family pair',
             seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,accumulate=a.accumulate,
             val_batch_size=a.val_batch_size,effective_batch=effective,lr=a.lr,weight_decay=a.weight_decay,
             cfg=asdict(cfg),trainable_parameters=sum(p.numel() for p in model.parameters()),smoke=bool(a.smoke),
             selection='max scene/domain balanced selected Q gain, ties earliest, admit only if >0')
    rng=np.random.default_rng(a.seed);best=-float('inf');start=0;updates=0;history=[]
    if a.resume:
        ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if ck['protocol']!=settings:raise ValueError('P1 resume contract differs')
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        best,start,updates,history=ck['best'],ck['epoch'],ck['updates'],ck['history']
    atomic_json(root/'protocol.json',signed(settings))
    for epoch in range(start+1,a.epochs+1):
        model.train();order=rng.permutation(len(tr['entries']));losses=[]
        opt.zero_grad(set_to_none=True)
        for k in range(0,len(order),effective):
            group=order[k:k+effective]
            for b in range(0,len(group),a.batch_size):
                idxs=group[b:b+a.batch_size]
                batch=collate_cache([load_cached(tr['entries'][int(j)]) for j in idxs])
                z=score_cached(model,batch,a.arm,a.device)
                count=torch.as_tensor(batch['quality_count'],device=a.device,dtype=torch.long)
                loss,_=quality_loss(z,count)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite P1 loss')
                (loss*(len(idxs)/len(group))).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True);updates+=1
        val=evaluate(model,a.arm,va,a.device,a.val_batch_size)
        gain=val['summary']['gain'];improved=gain>best+1e-12
        if improved:best=gain
        history.append(dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),validation=val['summary']))
        payload=dict(schema=P1_SCHEMA,protocol=settings,model=model.state_dict(),optimizer=opt.state_dict(),
                     epoch=epoch,updates=updates,best=best,history=history,rng=rng.bit_generator.state,
                     torch_rng=torch.get_rng_state())
        tmp=root/'last.pt.tmp';torch.save(payload,tmp);tmp.replace(root/'last.pt')
        if improved:
            torch.save(payload,root/'best.pt');atomic_json(root/'selected_validation.json',signed(val))
        atomic_json(root/'history.json',signed(dict(history=history)))
        print(f'[P1 {a.arm}] epoch {epoch}/{a.epochs}, update={updates}, gain={gain:.6f}',flush=True)
    selected=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    atomic_json(root/'selection.json',signed(dict(admitted=best>0,selected_gain=best,
                best_epoch=selected['epoch'],best_weight_sha256=sha256(root/'best.pt'),
                validation_index=va['fingerprint'],note='Privilege is OFFLINE ONLY; no official test inference.')))


def load_model(path,device='cpu',allow_smoke=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['schema']!=P1_SCHEMA:raise ValueError('Not a matched P1 analytic checkpoint')
    p=ck['protocol']
    if p['smoke'] and not allow_smoke:raise ValueError('Smoke model forbidden for formal comparison')
    model=PredictedGeometryControl(ModelConfig(**p['cfg']))
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),p


def compare(a):
    idx=read_cache_index(a.index,'validation',a.smoke)
    models={};summary={};results={}
    for arm,path in [('predicted',a.predicted_weight),('clean_visible',a.clean_weight)]:
        model,protocol=load_model(path,a.device,a.smoke)
        if protocol['val_index']!=idx['fingerprint'] or protocol['arm']!=arm:
            raise ValueError('P1 validation/model arm mismatch')
        if arm=='clean_visible' and protocol['target_config']!=idx['contract']['target_config']:
            raise ValueError('Clean teacher config mismatch')
        models[arm]=dict(path=str(Path(path).resolve()),sha256=sha256(path),protocol=protocol)
        result=evaluate(model,arm,idx,a.device,a.batch_size)
        summary[arm]=result['summary'];results[arm]=result['rows']
    # Existing scene-bootstrap helper expects flat family records; expand by
    # frame and use each frame's selected quality difference as the sample unit.
    def flat(rows):
        return [dict(scene=r['row']['scene'],domain=r['row']['domain'],gain=r['metrics']['gain']) for r in rows]
    contrast=scene_bootstrap(flat(results['clean_visible']),flat(results['predicted']),
                             metric='gain',n_boot=a.bootstrap,seed=a.seed)
    extra={}
    if getattr(a,'h2_weight',None):
        if not getattr(a,'original_val_index',None):
            raise ValueError('Existing H2 comparator requires --original-val-index')
        if sha256(a.original_val_index)!=idx['contract']['original_index_sha256']:
            raise ValueError('H2 original validation index differs from P1 input source')
        from .data import read_index
        from .experiment import load_model as load_original_model, evaluate_index
        original=read_index(a.original_val_index,'validation',a.smoke)
        if original['fingerprint']!=idx['contract']['original_index_fingerprint']:
            raise ValueError('P1 original validation source fingerprint mismatch')
        if [x['row'] for x in original['entries']]!=[x['row'] for x in idx['entries']]:
            raise ValueError('H2 comparison frame identities not aligned')
        h2,p=load_original_model(a.h2_weight,a.device,a.smoke)
        if p['mode']!='h2' or p['val_fingerprint']!=original['fingerprint']:
            raise ValueError('Wrong H2 checkpoint/validation source')
        h2_results=evaluate_index(h2,'h2',original,a.device,batch_size=a.batch_size)
        summary['learned_h2']=h2_results['summary'];results['learned_h2']=h2_results['rows']
        models['learned_h2']=dict(path=str(Path(a.h2_weight).resolve()),sha256=sha256(a.h2_weight),
                                   protocol=p,comparison_role='descriptive: different geometry supervision and train objective')
        for arm in ARMS:
            extra[f'{arm}_minus_learned_h2']=scene_bootstrap(flat(results[arm]),flat(results['learned_h2']),
                         metric='gain',n_boot=a.bootstrap,seed=a.seed)
    if Path(a.output).exists():raise FileExistsError(a.output)
    atomic_json(a.output,signed(dict(schema=P1_SCHEMA,partition='validation',
          index_sha256=sha256(a.index),index_fingerprint=idx['fingerprint'],
          source_models=models,summary=summary,paired_clean_minus_predicted=contrast,
          descriptive_learned_h2_contrasts=extra,per_frame=results,
          privileged_clean_visible=True,official_ap_not_run=True,
          note='Clean-rendered geometry is privileged train/validation evidence only. H2 is a descriptive non-matched-training comparison, not part of the two-arm controlled contrast.')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build');b.add_argument('--index',required=True);b.add_argument('--targets',required=True)
    b.add_argument('--output',required=True);b.add_argument('--resume',action='store_true')
    b.add_argument('--smoke',action='store_true');b.add_argument('--limit',type=int,default=0)
    b.set_defaults(func=build)
    t=sub.add_parser('train');t.add_argument('--train-index',required=True);t.add_argument('--val-index',required=True)
    t.add_argument('--arm',choices=ARMS,required=True);t.add_argument('--output',required=True)
    t.add_argument('--batch-size',type=int,default=4);t.add_argument('--accumulate',type=int,default=1)
    t.add_argument('--val-batch-size',type=int,default=4);t.add_argument('--epochs',type=int,default=20)
    t.add_argument('--hidden',type=int,default=64);t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--weight-decay',type=float,default=1e-4);t.add_argument('--seed',type=int,default=0)
    t.add_argument('--cpu-threads',type=int,default=2);t.add_argument('--device',default='cuda:0')
    t.add_argument('--resume',action='store_true');t.add_argument('--smoke',action='store_true')
    t.set_defaults(func=fit)
    c=sub.add_parser('compare');c.add_argument('--index',required=True)
    c.add_argument('--predicted-weight',required=True);c.add_argument('--clean-weight',required=True)
    c.add_argument('--batch-size',type=int,default=4);c.add_argument('--device',default='cuda:0')
    c.add_argument('--seed',type=int,default=0);c.add_argument('--bootstrap',type=int,default=2000)
    c.add_argument('--output',required=True);c.add_argument('--smoke',action='store_true')
    c.add_argument('--h2-weight',default='',help='Optional existing learned H2 checkpoint for descriptive comparison')
    c.add_argument('--original-val-index',default='',help='Exact original signed validation index for H2 comparison')
    c.set_defaults(func=compare)
    a=parser.parse_args();a.func(a)

if __name__=='__main__':main()