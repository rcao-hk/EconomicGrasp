"""P5-1: separate geometric-source quality from quality-reader adaptation.

Offline matched original/GN-Trans train/validation only. Reuses the depth-only
selected P4 corrector, the immutable P1 physics/physical IDs/quality counts,
then trains two *independent, matched* 13k-parameter readouts (Pred/Corrected).
Clean-Visible is a privileged diagnostic reference, never an inference input.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch

from .data import (sha256, digest, signed, load_signed, atomic_json, atomic_npz,
                   frame_key)
from .geometry import support_points, VisibleSurfaceTeacher
from .model import ModelConfig, PredictedGeometryControl, quality_loss
from .p1_clean_visible import (read_cache_index, load_cached, physical_tokens, score_cached)
from .p4_depth_data import read_depth_index, read_features
from .p4_depth_baseline import load_corrector
from .p4_common import (formal_rows, select_entries, freeze_reader, score_physics,
                        family_rows, paired_summary, scene_balanced, scene_bootstrap,
                        atomic_checkpoint, new_directory, write_records, verify_p2)

SCHEMA = 'shared-visible-geometry-p5-reader-v1'
ARMS = ('predicted', 'corrected')


def frozen_corrector(path, depth_index, device, smoke):
    model, protocol, epoch = load_corrector(path, device, smoke)
    if protocol['val_fingerprint'] != depth_index['fingerprint'] and depth_index['partition']=='validation':
        raise ValueError('Depth corrector validation fingerprint mismatch')
    if protocol['teacher_config'] != depth_index['contract']['target_config']:
        raise ValueError('Corrector/target configuration mismatch')
    if protocol['train_source'] != depth_index['contract']['source']:
        raise ValueError('Corrector/source identity mismatch')
    if not smoke:
        selection=load_signed(Path(path).parent/'selection.json')
        if (not selection['complete'] or selection['selected_epoch'] != epoch or
            selection['best_sha256'] != sha256(path) or not selection['no_action_quality_used_for_selection']):
            raise ValueError('Expected selected, completed, depth-only P4 corrector')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model, protocol, epoch


def aligned_indices(depth_index, p1_index, partition, smoke):
    di=read_depth_index(depth_index,partition,smoke)
    pi=read_cache_index(p1_index,partition,smoke)
    formal_rows(di,partition,smoke);formal_rows(pi,partition,smoke)
    if (di['contract']['original_index_fingerprint'] != pi['contract']['original_index_fingerprint'] or
        di['contract']['target_config'] != pi['contract']['target_config'] or
        di['contract']['source'] != pi['contract']['source']):
        raise ValueError('P4/P1 source, teacher or original-bank identity mismatch')
    dp={frame_key(e['row']):e for e in di['entries']}
    pp={frame_key(e['row']):e for e in pi['entries']}
    if set(dp)!=set(pp):raise ValueError('P1/P4 frame sets differ')
    return di,pi,[(dp[k],pp[k]) for k in [frame_key(e['row']) for e in di['entries']]]


def make_corrected_physics(model,source_path,p1_array,tau,band,device):
    """Model sees only RGB features + predicted depth, not GT, Q or action.

    Physical actions are supplied only to the *post-hoc* analytic geometry read.
    """
    with np.load(source_path, allow_pickle=False) as z:
        input_t={k:np.asarray(z[k],np.float32) for k in ('feature_map','pred_depth','K')}
        bank=np.asarray(z['bank'],np.float32)
        pid=np.asarray(z['physical_id'])
        count=np.asarray(z['quality_count'])
        family_id=np.asarray(z['family_id'])
    if not np.array_equal(pid,p1_array['physical_id']) or not np.array_equal(count,p1_array['quality_count']) or not np.array_equal(family_id,p1_array['family_id']):
        raise ValueError('P1 corrected-cache physical IDs/quality labels changed')
    with torch.no_grad():
        depth=model({k:torch.as_tensor(input_t[k],device=device)[None] for k in ('feature_map','pred_depth')})[0].cpu().numpy()
    points,local=support_points(torch.from_numpy(bank))
    if not np.allclose(local.numpy(),p1_array['local_xyz'],rtol=0,atol=1e-6):
        raise ValueError('P1 corrected-cache support point indices changed')
    xyz=points.reshape(-1,3).numpy()
    result=physical_tokens(VisibleSurfaceTeacher(depth,input_t['K'],tau,band),xyz)
    result=result.reshape(*bank.shape[:2],75,6)
    if not np.isfinite(result).all():raise ValueError('Nonfinite corrected geometry')
    return result


def build(a):
    if a.limit<0 or (a.limit and not a.smoke):raise ValueError('Partial P5 cache build requires --smoke')
    di,pi,pairs=aligned_indices(a.depth_index,a.p1_index,a.partition,a.smoke)
    corrector,prot,epoch=frozen_corrector(a.depth_weight,di,a.device,a.smoke)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    code_sha=sha256(__file__)
    contract=dict(schema=SCHEMA,kind='cache',partition=a.partition,
        depth_index_fingerprint=di['fingerprint'],p1_index_fingerprint=pi['fingerprint'],
        original_index_fingerprint=di['contract']['original_index_fingerprint'],
        teacher_config=pi['contract']['target_config'],source=pi['contract']['source'],
        plan=pi['contract']['plan'],tau=pi['contract']['tau'],band=pi['contract']['band'],
        offsets_m=pi['contract']['offsets_m'],depth_weight_sha256=sha256(a.depth_weight),
        depth_weight_epoch=epoch,code_sha256=code_sha,partial=bool(a.limit))
    ch=digest(contract);entries=[]
    subset=select_entries([{'row':x['row'],'paired':(x,y)} for x,y in pairs],a.limit)
    for n,item in enumerate(subset,1):
        de,pe=item['paired'];row=de['row'];key=frame_key(row)
        dest=root/(key+'_corrected.npz');meta=dest.with_suffix('.json')
        identity=dict(row=row,contract_hash=ch,source_sha256=de['source_sha256'],
                      p1_sha256=pe['sha256'],depth_weight_sha256=sha256(a.depth_weight))
        if dest.exists() or meta.exists():
            if not a.resume or not dest.is_file() or not meta.is_file():raise FileExistsError(dest)
            m=load_signed(meta)
            if any(m.get(k)!=v for k,v in identity.items()) or sha256(dest)!=m['sha256']:
                raise ValueError('P5 cache resume identity changed')
        else:
            old=load_cached(pe)
            corr=make_corrected_physics(corrector,de['source_path'],old,pi['contract']['tau'],pi['contract']['band'],a.device)
            atomic_npz(dest,dict(corrected_physics=corr,physical_id=old['physical_id'],
                family_id=old['family_id'],quality_count=old['quality_count']))
            atomic_json(meta,signed(dict(**identity,sha256=sha256(dest))))
        entries.append(dict(row=row,source_path=de['source_path'],source_sha256=de['source_sha256'],
                            p1_path=pe['path'],p1_sha256=pe['sha256'],
                            corrected_path=str(dest.resolve()),corrected_sha256=sha256(dest)))
        if n%100==0 or n==len(subset):print(f'[P5 corrected cache/{a.partition}] {n}/{len(subset)}',flush=True)
    manifest=root/'index.json'
    result=signed(dict(schema=SCHEMA,kind='cache',partition=a.partition,
        complete=bool(di['complete'] and pi['complete'] and not a.limit),contract=contract,entries=entries))
    if manifest.exists():
        if not a.resume or load_signed(manifest)!=result:
            raise FileExistsError('P5 cache manifest already exists with different identity')
    else: atomic_json(manifest,result)
    return result


def read_index(path,partition=None,smoke=False):
    idx=load_signed(path)
    if idx.get('schema')!=SCHEMA or idx.get('kind')!='cache' or (partition and idx['partition']!=partition):
        raise ValueError('Wrong P5 reader cache index')
    if not idx['complete'] and not smoke:raise ValueError('Incomplete P5 cache forbidden for formal training')
    if idx['contract']['code_sha256']!=sha256(__file__):
        raise ValueError('P5 cache construction code changed; do not silently reuse cache')
    rows=[e['row'] for e in idx['entries']]
    formal_rows(idx,idx['partition'],smoke)
    if len(set(frame_key(r) for r in rows))!=len(rows):raise ValueError('Repeated P5 rows')
    for e in idx['entries']:
        if (sha256(e['source_path'])!=e['source_sha256'] or sha256(e['p1_path'])!=e['p1_sha256'] or
            sha256(e['corrected_path'])!=e['corrected_sha256']):
            raise ValueError('P5 source/geometry cache changed')
        side=load_signed(Path(e['corrected_path']).with_suffix('.json'))
        if (side['contract_hash']!=digest(idx['contract']) or side['sha256']!=e['corrected_sha256'] or
            side['source_sha256']!=e['source_sha256'] or side['p1_sha256']!=e['p1_sha256']):
            raise ValueError('P5 cache sidecar mismatch')
    return idx


def read_sample(e):
    with np.load(e['corrected_path'],allow_pickle=False) as z: c={k:z[k] for k in z.files}
    with np.load(e['p1_path'],allow_pickle=False) as z: p={k:z[k] for k in z.files}
    if any(not np.array_equal(c[k],p[k]) for k in ('physical_id','family_id','quality_count')):
        raise ValueError('P5 cache lost exact physical-action alignment')
    physical=np.asarray(c['corrected_physics'],np.float32)
    if physical.shape!=p['predicted_physics'].shape or physical.shape[-2:]!=(75,6):
        raise ValueError('P5 corrected geometry shape mismatch')
    if not np.isfinite(physical).all() or not np.allclose(physical[...,2:].sum(-1),1,atol=1e-6):
        raise ValueError('P5 corrected physical evidence malformed')
    return dict(**p,corrected_physics=physical)


def collate(samples):
    if not samples:raise ValueError('Empty P5 batch')
    keys=('predicted_physics','corrected_physics','clean_visible_physics','local_xyz',
          'bank_dims','quality_count','physical_id','family_id')
    out={}
    for name in keys:
        items=[s[name] for s in samples]
        if len({x.shape for x in items})!=1:raise ValueError('P5 ragged action bank: '+name)
        out[name]=np.stack(items)
    return out


def forward(model,batch,arm,device):
    if arm not in ('predicted','corrected','clean_visible'):
        raise ValueError('Unexpected P5 geometry source')
    physical=torch.as_tensor(batch[arm+'_physics'],device=device,dtype=torch.float32)
    loc=torch.as_tensor(batch['local_xyz'],device=device,dtype=torch.float32)
    dim=torch.as_tensor(batch['bank_dims'],device=device,dtype=torch.float32)
    return score_physics(model,physical,loc,dim)


@torch.no_grad()
def evaluate(model,index,arm,device,batch_size=4):
    model.eval();rows=[]
    for start in range(0,len(index['entries']),batch_size):
        block=index['entries'][start:start+batch_size]
        batch=collate([read_sample(e) for e in block]);logits=forward(model,batch,arm,device).cpu().numpy()
        for n,e in enumerate(block):
            rows.extend(family_rows(logits[n],{k:batch[k][n] for k in ('quality_count','physical_id','family_id')},
                 e['row'],index['contract']['offsets_m'],batch[arm+'_physics'][n]))
    return rows


def fit(a):
    if a.arm not in ARMS or min(a.batch_size,a.accumulate,a.val_batch_size,a.epochs,a.cpu_threads)<1:
        raise ValueError('Invalid training settings')
    if a.lr<=0 or a.weight_decay<0 or getattr(a,'stop_after_epoch',0)<0:raise ValueError('Invalid optimizer/stop setting')
    tr=read_index(a.train_index,'train',a.smoke);va=read_index(a.val_index,'validation',a.smoke)
    for k in ('teacher_config','source','plan','tau','band','offsets_m','depth_weight_sha256'):
        if tr['contract'][k]!=va['contract'][k]:raise ValueError('P5 train/val mismatch: '+k)
    if {x['row']['scene'] for x in tr['entries']} & {x['row']['scene'] for x in va['entries']}:
        raise ValueError('Train/validation scene overlap')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.cpu_threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=ModelConfig(feature_channels=1,hidden=a.hidden,truncation_m=tr['contract']['tau'])
    model=PredictedGeometryControl(cfg).to(a.device)
    optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    settings=dict(schema=SCHEMA,kind='reader',arm=a.arm,
        train_fingerprint=tr['fingerprint'],val_fingerprint=va['fingerprint'],
        depth_weight_sha256=tr['contract']['depth_weight_sha256'],
        p1_fingerprint=va['contract']['p1_index_fingerprint'],
        teacher_config=tr['contract']['teacher_config'],source=tr['contract']['source'],
        seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,accumulate=a.accumulate,
        val_batch_size=a.val_batch_size,lr=a.lr,weight_decay=a.weight_decay,
        parameters=sum(p.numel() for p in model.parameters()),cfg=asdict(cfg),
        selection='scene/domain balanced selected-Q gain, ties earliest',
        objective='10*MSE(sigmoid(z),count/6)+0.5*within-family pair',smoke=bool(a.smoke),
        code_sha256=sha256(__file__))
    rng=np.random.default_rng(a.seed);start=0;updates=0;best=-float('inf');history=[]
    if a.resume:
        ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if ck['schema']!=SCHEMA or ck['protocol']!=settings:raise ValueError('P5 resume protocol mismatch')
        model.load_state_dict(ck['model'],strict=True);optim.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'].cpu())
        start,best,updates,history=ck['epoch'],ck['best'],ck['updates'],ck['history']
    atomic_json(root/'protocol.json',signed(settings))
    effective=a.batch_size*a.accumulate
    limit=getattr(a,'stop_after_epoch',0)
    end=min(limit,a.epochs) if limit else a.epochs
    if end<start:raise ValueError('Requested stop precedes resumed epoch')
    for epoch in range(start+1,end+1):
        model.train();order=rng.permutation(len(tr['entries']));opt_losses=[]
        optim.zero_grad(set_to_none=True)
        for s in range(0,len(order),effective):
            group=order[s:s+effective]
            for u in range(0,len(group),a.batch_size):
                ids=group[u:u+a.batch_size]
                b=collate([read_sample(tr['entries'][int(i)]) for i in ids])
                logits=forward(model,b,a.arm,a.device)
                q=torch.as_tensor(b['quality_count'],device=a.device,dtype=torch.long)
                loss,_=quality_loss(logits,q)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite P5 quality loss')
                (loss*len(ids)/len(group)).backward();opt_losses.append((float(loss.detach()),len(ids)))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optim.step();optim.zero_grad(set_to_none=True);updates+=1
        with torch.no_grad():rows=evaluate(model,va,a.arm,a.device,a.val_batch_size)
        gain=scene_balanced(rows,'gain');is_best=gain>best+1e-12
        if is_best:best=gain
        history.append(dict(epoch=epoch,updates=updates,
            train_loss=sum(x*n for x,n in opt_losses)/sum(n for x,n in opt_losses),
            validation_gain=gain,validation_selected_q=scene_balanced(rows,'selected_q')))
        ck=dict(schema=SCHEMA,protocol=settings,model=model.state_dict(),optimizer=optim.state_dict(),
            epoch=epoch,updates=updates,best=best,history=history,rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state())
        atomic_checkpoint(root/'last.pt',ck)
        if is_best:
            atomic_checkpoint(root/'best.pt',ck)
            atomic_json(root/'selected_validation.json',signed(dict(epoch=epoch,gain=gain,
                selected_q=scene_balanced(rows,'selected_q'))))
        atomic_json(root/'history.json',signed(dict(history=history)))
        print(f'[P5 reader/{a.arm}] epoch={epoch}, updates={updates}, gain={gain:+.6f}',flush=True)
    chosen=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    atomic_json(root/'selection.json',signed(dict(best_epoch=chosen['epoch'],
        selected_gain=best,admitted=best>0,complete=(end==a.epochs),weight_sha256=sha256(root/'best.pt'),
        validation_fingerprint=va['fingerprint'])))


def load_reader(path,index,device='cpu',smoke=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=SCHEMA or ck.get('protocol',{}).get('kind')!='reader':
        raise ValueError('Not a P5 reader checkpoint')
    p=ck['protocol']
    if not smoke and p['smoke']:raise ValueError('Smoke reader in formal comparison')
    if (p['val_fingerprint']!=index['fingerprint'] or
        p['depth_weight_sha256']!=index['contract']['depth_weight_sha256'] or
        p['teacher_config']!=index['contract']['teacher_config']):
        raise ValueError('Reader training/cache identity mismatch')
    if not smoke:
        selection=load_signed(Path(path).parent/'selection.json')
        if not selection['complete'] or selection['weight_sha256']!=sha256(path):
            raise ValueError('Reader is incomplete or weight changed')
    model=PredictedGeometryControl(ModelConfig(**p['cfg'])).to(device)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    return model,p


def compare(a):
    if a.batch_size<1 or a.bootstrap<1:raise ValueError('Invalid comparison budget')
    idx=read_index(a.index,'validation',a.smoke)
    models={}
    for arm,weight in (('predicted',a.predicted_weight),('corrected',a.corrected_weight)):
        m,p=load_reader(weight,idx,a.device,a.smoke)
        if p['arm']!=arm:raise ValueError('P5 reader arm/checkpoint mismatch')
        models[arm]=m
    pi=read_cache_index(a.p1_index,'validation',a.smoke)
    if idx['contract']['p1_index_fingerprint']!=pi['fingerprint']:
        raise ValueError('P5 matched P1 validation mismatch')
    p2,p2p=freeze_reader(a.shared_weight,pi,a.device,a.smoke)
    refs={}
    modes={
        'predicted_p2_fixed':(p2,'predicted'),
        'corrected_p2_fixed':(p2,'corrected'),
        'clean_p2_fixed':(p2,'clean_visible'),
        'predicted_retrained':(models['predicted'],'predicted'),
        'corrected_adapted':(models['corrected'],'corrected'),
        'pred_reader_on_corrected':(models['predicted'],'corrected'),
        'corrected_reader_on_predicted':(models['corrected'],'predicted'),
    }
    recs={k:[] for k in modes}
    for start in range(0,len(idx['entries']),a.batch_size):
        block=idx['entries'][start:start+a.batch_size];b=collate([read_sample(e) for e in block])
        for name,(m,source) in modes.items():
            with torch.no_grad():z=forward(m,b,source,a.device).cpu().numpy()
            for j,e in enumerate(block):
                recs[name].extend(family_rows(z[j],{k:b[k][j] for k in ('quality_count','physical_id','family_id')},
                    e['row'],idx['contract']['offsets_m'],b[source+'_physics'][j]))
    scores={name:paired_summary(rows,recs['predicted_p2_fixed'] if name!='predicted_p2_fixed' else None,
                            a.bootstrap,a.seed) for name,rows in recs.items()}
    if not a.smoke:
        expected=load_signed(a.expected_p4)
        if (expected.get('study')!='P4-1 ordinary-depth, frozen P2 readout' or
            expected['sources']['corrector_sha256']!=idx['contract']['depth_weight_sha256'] or
            expected['sources']['reader_sha256']!=sha256(a.shared_weight)):
            raise ValueError('P4 comparison source mismatch')
        for k,src in (('predicted','predicted_p2_fixed'),('corrected','corrected_p2_fixed'),('clean_visible','clean_p2_fixed')):
            if abs(scores[src]['gain']-expected['scores'][k]['gain'])>2e-6:
                raise RuntimeError('P5 geometry cache does not reproduce P4 frozen readout: '+k)
    dest=new_directory(a.output)
    contrasts={
        'corrected_adapted_minus_corrected_fixed':('corrected_adapted','corrected_p2_fixed'),
        'corrected_adapted_minus_pred_retrained':('corrected_adapted','predicted_retrained'),
        'pred_retrained_minus_pred_fixed':('predicted_retrained','predicted_p2_fixed'),
        'corrected_fixed_minus_pred_fixed':('corrected_p2_fixed','predicted_p2_fixed'),
        'clean_fixed_minus_corrected_fixed':('clean_p2_fixed','corrected_p2_fixed')}
    paired={k:scene_bootstrap(recs[u],recs[v],metric='gain',n_boot=a.bootstrap,seed=a.seed)
            for k,(u,v) in contrasts.items()}
    # Difference-in-differences controls for refitting either geometry reader.
    # Uses exactly matched physical action families, not independent families.
    diff_rows=[]
    keys=('scene','domain','anno','family','native_count')
    for fixedp,fixedc,retrain,adapt in zip(recs['predicted_p2_fixed'],recs['corrected_p2_fixed'],
                                           recs['predicted_retrained'],recs['corrected_adapted']):
        if not all(tuple(item[k] for k in keys)==tuple(fixedp[k] for k in keys)
                   for item in (fixedc,retrain,adapt)):
            raise ValueError('P5 DiD physical action identity mismatch')
        did=((adapt['gain']-retrain['gain'])-(fixedc['gain']-fixedp['gain']))
        diff_rows.append(dict(scene=fixedp['scene'],domain=fixedp['domain'],gain=float(did)))
    paired['difference_in_differences_geometry_adaptation']=scene_bootstrap(
        diff_rows,metric='gain',n_boot=a.bootstrap,seed=a.seed)
    subgroup={}
    for name,rr in recs.items():
        subgroup[name]={}
        for tag,pred in (('native_full_count6',lambda r:r['native_count']==6),
                         ('native_not_full',lambda r:r['native_count']<6),
                         ('has_improvement_opportunity',lambda r:r['opportunity']),
                         ('no_improvement_opportunity',lambda r:not r['opportunity'])):
            sub=[x for x in rr if pred(x)]
            subgroup[name][tag]=dict(n=len(sub),gain=scene_balanced(sub,'gain'),
                rescue=scene_balanced(sub,'rescue'),harm=scene_balanced(sub,'harm'))
    report=dict(schema=SCHEMA,kind='matched-reader-comparison',partition='validation',
        validation_fingerprint=idx['fingerprint'],
        sources=dict(p1_index_fingerprint=pi['fingerprint'],depth_weight_sha256=idx['contract']['depth_weight_sha256'],
            p2_reader_sha256=sha256(a.shared_weight),p5_pred_reader_sha256=sha256(a.predicted_weight),
            p5_corrected_reader_sha256=sha256(a.corrected_weight)),
        scores=scores,paired=paired,subgroups=subgroup,weights_matched_budget=True,
        training_quality_labels_only=True,depth_model_frozen=True,
        cross_source_scores_are_OOD=True,official_ap_not_run=True,
        note='Reader adaptation is compared to a separately retrained Pred reader; all use the same fixed physical bank and labels. Validation-selected weights and single seed; intervals descriptive.')
    report['artifacts']={k:write_records(dest/(k+'_families.jsonl.gz'),v) for k,v in recs.items()}
    atomic_json(dest/'summary.json',signed(report))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('build');b.add_argument('--partition',choices=('train','validation'),required=True)
    b.add_argument('--depth-index',required=True);b.add_argument('--p1-index',required=True)
    b.add_argument('--depth-weight',required=True);b.add_argument('--output',required=True)
    b.add_argument('--resume',action='store_true');b.add_argument('--limit',type=int,default=0)
    t=s.add_parser('train');t.add_argument('--arm',choices=ARMS,required=True)
    t.add_argument('--train-index',required=True);t.add_argument('--val-index',required=True)
    t.add_argument('--output',required=True);t.add_argument('--epochs',type=int,default=20)
    t.add_argument('--batch-size',type=int,default=4);t.add_argument('--accumulate',type=int,default=1)
    t.add_argument('--val-batch-size',type=int,default=4);t.add_argument('--hidden',type=int,default=64)
    t.add_argument('--lr',type=float,default=1e-4);t.add_argument('--weight-decay',type=float,default=1e-4)
    t.add_argument('--cpu-threads',type=int,default=2);t.add_argument('--seed',type=int,default=0)
    t.add_argument('--resume',action='store_true')
    t.add_argument('--stop-after-epoch',type=int,default=0)
    c=s.add_parser('compare');c.add_argument('--index',required=True);c.add_argument('--p1-index',required=True)
    c.add_argument('--predicted-weight',required=True);c.add_argument('--corrected-weight',required=True)
    c.add_argument('--shared-weight',required=True);c.add_argument('--expected-p4',default='')
    c.add_argument('--output',required=True);c.add_argument('--batch-size',type=int,default=4)
    c.add_argument('--bootstrap',type=int,default=5000);c.add_argument('--seed',type=int,default=0)
    for sub in (b,t,c):sub.add_argument('--device',default='cuda:0');sub.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.func(args)

if __name__=='__main__':main()
