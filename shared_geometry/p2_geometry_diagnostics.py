"""P2: native-relative diagnostics for matched P1 readers, and a shared-weight
Predicted/Clean-Visible reader control. Offline train/validation only.

Never uses rendered depth at deployment or touches original caches/checkpoints.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch

from .data import atomic_json, signed, sha256, digest
from .model import ModelConfig, PredictedGeometryControl, quality_loss
from .p1_clean_visible import (ARMS, P1_SCHEMA, read_cache_index, load_cached,
    collate_cache, score_cached, load_model as load_p1_model)
from .p0_relative_decision import metrics_for_family, summarize, scene_bootstrap, scene_balanced

SCHEMA = 'shared-visible-geometry-p2-source-control-v1'


def _family_records(z, batch, rows, arm, offsets):
    """Return exact same P0 per-family definitions on physically aligned P1 banks."""
    records = []
    scores = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    q = np.asarray(batch['quality_count'])
    states = np.argmax(np.asarray(batch[arm+'_physics'])[..., 2:], axis=-1)
    if scores.shape != q.shape or scores.ndim != 3 or scores.shape[0] != len(rows):
        raise ValueError('Predictions/labels do not align by image/family/action')
    for b, row in enumerate(rows):
        for f in range(scores.shape[1]):
            rec = metrics_for_family(scores[b,f], q[b,f], offsets, state=states[b,f])
            rec.update(scene=int(row['scene']), domain=row['domain'], anno=int(row['anno']),
                       family=int(batch['family_id'][b,f]), native_count=int(q[b,f,0]),
                       no_rescue=bool(rec['opportunity'] and not rec['rescue']))
            for a in rec['alternatives']:
                a['true_benefit'] = bool(a['delta_q'] > 0)
            records.append(rec)
    return records


def _pair_stats(records):
    """Breakdown cannot be replaced by global all-pair accuracy."""
    has_opportunity = lambda r: r['opportunity']
    none = lambda r: not r['opportunity']
    full = lambda r: r['native_count'] == 6
    low = lambda r: r['native_count'] < 6
    bins = {
        'native_full_count6': full,
        'native_below_full': low,
        'opportunity': has_opportunity,
        'no_opportunity': none,
    }
    out = {}
    for name, pred in bins.items():
        items = [r for r in records if pred(r)]
        out[name] = dict(count=len(items), gain=scene_balanced(items,'gain'),
            move=scene_balanced(items,'move'), rescue=scene_balanced(items,'rescue'),
            harm=scene_balanced(items,'harm'),
            native_signed_bias=scene_balanced(items,'native_signed_bias'),
            alt_signed_bias=scene_balanced(items,'alt_signed_bias'),
            native_pair_accuracy=scene_balanced(items,'nativepair_correct'))
    return out


def _source_contrast(pred, clean, n_boot, seed):
    if len(pred) != len(clean):
        raise ValueError('Paired reader predictions must cover identical families')
    for a,b in zip(pred,clean):
        if (a['scene'],a['domain'],a['anno'],a['family'],a['native_count']) != (b['scene'],b['domain'],b['anno'],b['family'],b['native_count']):
            raise ValueError('Source comparison mixed physical actions')
    result = {}
    for metric in ('gain','rescue','harm','move','native_signed_bias',
                   'alt_signed_bias','nativepair_correct','allpair_correct'):
        # Missing pair accuracy may differ by scene when all labels tie.
        # For these metrics, report descriptive aggregates; paired CI is valid
        # only if the same scene/domain cells have finite observations.
        try:
            result[metric] = scene_bootstrap(clean,pred,metric=metric,n_boot=n_boot,seed=seed)
        except ValueError:
            result[metric] = dict(clean=scene_balanced(clean,metric),
                                  predicted=scene_balanced(pred,metric),
                                  paired_ci=None)
    return result


def _evaluate_models(models, index, device, batch_size, *, limit=0):
    if batch_size < 1 or limit < 0:
        raise ValueError('Invalid batch size or limit')
    records = {arm:[] for arm in ARMS}
    entries = index['entries'][:limit] if limit else index['entries']
    offsets = index['contract']['offsets_m']
    for start in range(0,len(entries),batch_size):
        sub = entries[start:start+batch_size]
        frames = [load_cached(e) for e in sub]
        batch = collate_cache(frames)
        batch['family_id'] = np.stack([f['family_id'] for f in frames])
        rows = [e['row'] for e in sub]
        for arm in ARMS:
            with torch.no_grad():
                logits = score_cached(models[arm],batch,arm,device)
            records[arm].extend(_family_records(logits,batch,rows,arm,offsets))
    return records


def _describe(records, bootstrap, seed):
    return dict(summaries={arm:summarize(v) for arm,v in records.items()},
                native_relative_subgroups={arm:_pair_stats(v) for arm,v in records.items()},
                clean_minus_predicted=_source_contrast(records['predicted'],records['clean_visible'],bootstrap,seed))


def diagnostic(a):
    idx = read_cache_index(a.index,'validation',a.smoke)
    if a.limit and not a.smoke:
        raise ValueError('Partial diagnostic requires --smoke')
    if a.bootstrap < 1 or a.batch_size < 1:
        raise ValueError('Invalid diagnostic budget')
    models = {}; sources = {}
    for arm,path in zip(ARMS,(a.predicted_weight,a.clean_weight)):
        model,p = load_p1_model(path,a.device,a.smoke)
        if p['arm']!=arm or p['val_index']!=idx['fingerprint'] or p['target_config']!=idx['contract']['target_config']:
            raise ValueError('P1 checkpoint/cache provenance mismatch')
        models[arm] = model
        sources[arm] = dict(checkpoint=sha256(path),selected_epoch=torch.load(path,map_location='cpu',weights_only=False)['epoch'])
    records = _evaluate_models(models,idx,a.device,a.batch_size,limit=a.limit)
    report = _describe(records,a.bootstrap,a.seed)
    report.update(schema=SCHEMA,study='matched_p1_native_relative',partition='validation',
                  validation_fingerprint=idx['fingerprint'],cache_index_sha256=sha256(a.index),
                  models=sources,smoke=bool(a.smoke),per_family=records,
                  note='Only scene-balanced offline exact-action Q, not AP. Selected P1 checkpoints used the same validation set; CIs are descriptive.')
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    atomic_json(dest,signed(report))
    return report


def _check_indices(train_idx,val_idx,smoke):
    train=read_cache_index(train_idx,'train',smoke)
    val=read_cache_index(val_idx,'validation',smoke)
    for key in ('source','plan','target_config','tau','band','offsets_m','original_index_fingerprint'):
        # Partition-specific original index fingerprints are EXPECTED to differ.
        if key=='original_index_fingerprint':continue
        if train['contract'][key]!=val['contract'][key]:
            raise ValueError('Shared reader source/teacher mismatch: '+key)
    train_scenes={e['row']['scene'] for e in train['entries']}
    val_scenes={e['row']['scene'] for e in val['entries']}
    if train_scenes & val_scenes:
        raise ValueError('Train/validation scene leakage')
    if not smoke and (len(train['entries']),len(val['entries']))!=(2560,160):
        raise ValueError('Formal P2 training requires full P1 train/validation caches')
    return train,val


def _shared_eval(model,idx,device,batch_size):
    records=_evaluate_models(dict.fromkeys(ARMS,model),idx,device,batch_size)
    # Checkpoint selection must not optimize only the privileged clean arm.
    gains={arm:scene_balanced(rows,'gain') for arm,rows in records.items()}
    mean_gain=float(np.mean(list(gains.values())))
    return dict(records=records,gains=gains,mean_gain=mean_gain,
                selected_q={arm:scene_balanced(rows,'selected_q') for arm,rows in records.items()})


def fit_shared(a):
    if min(a.batch_size,a.accumulate,a.val_batch_size,a.epochs,a.cpu_threads)<1 or a.lr<=0 or a.weight_decay<0:
        raise ValueError('Invalid P2 training settings')
    train,val=_check_indices(a.train_index,a.val_index,a.smoke)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.cpu_threads)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=ModelConfig(feature_channels=1,hidden=a.hidden,truncation_m=train['contract']['tau'])
    model=PredictedGeometryControl(cfg).to(a.device)
    optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    # The same quality-reader parameters receive both geometry sources; both
    # losses count once per underlying image, not twice the number of images.
    protocol=dict(schema=SCHEMA,study='shared_reader',train_fingerprint=train['fingerprint'],
        val_fingerprint=val['fingerprint'],teacher_config=train['contract']['target_config'],
        source=train['contract']['source'],code_sha256=sha256(__file__),
        architecture='PredictedGeometryControl.point_reader+quality, one shared copy',
        objective='0.5*(10MSE+0.5pair) per source, same exact-action bank',
        checkpoint_selection='mean of predicted+clean selected-Q gains on validation; earliest tie',
        seed=a.seed,hidden=a.hidden,epochs=a.epochs,batch_size=a.batch_size,
        val_batch_size=a.val_batch_size,accumulate=a.accumulate,lr=a.lr,
        weight_decay=a.weight_decay,smoke=bool(a.smoke),
        parameters=sum(p.numel() for p in model.parameters()),cfg=asdict(cfg))
    rng=np.random.default_rng(a.seed);best=-float('inf');start=0;updates=0;history=[]
    if a.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if ck['schema']!=SCHEMA or ck['protocol']!=protocol:
            raise ValueError('P2 shared-reader resume protocol mismatch')
        model.load_state_dict(ck['model'],strict=True)
        optim.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'])
        start,updates,best,history=ck['epoch'],ck['updates'],ck['best'],ck['history']
    atomic_json(out/'protocol.json',signed(protocol))
    effective=a.batch_size*a.accumulate
    for epoch in range(start+1,a.epochs+1):
        model.train();order=rng.permutation(len(train['entries']));losses=[]
        optim.zero_grad(set_to_none=True)
        for step in range(0,len(order),effective):
            indices=order[step:step+effective]
            for j in range(0,len(indices),a.batch_size):
                sample=indices[j:j+a.batch_size]
                batch=collate_cache([load_cached(train['entries'][int(i)]) for i in sample])
                counts=torch.as_tensor(batch['quality_count'],device=a.device,dtype=torch.long)
                zpred=score_cached(model,batch,'predicted',a.device)
                zclean=score_cached(model,batch,'clean_visible',a.device)
                lp,_=quality_loss(zpred,counts)
                lc,_=quality_loss(zclean,counts)
                loss=(lp+lc)*0.5
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite shared-reader loss')
                (loss*(len(sample)/len(indices))).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optim.step();optim.zero_grad(set_to_none=True);updates+=1
        model.eval()
        with torch.no_grad():valid=_shared_eval(model,val,a.device,a.val_batch_size)
        selected=float(valid['mean_gain']);improved=selected>best+1e-12
        if improved:best=selected
        history.append(dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),
                            validation=dict(gains=valid['gains'],mean_gain=selected,
                                            selected_q=valid['selected_q'])))
        payload=dict(schema=SCHEMA,protocol=protocol,model=model.state_dict(),
                     optimizer=optim.state_dict(),epoch=epoch,updates=updates,best=best,
                     history=history,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        temp=out/'last.pt.tmp';torch.save(payload,temp);temp.replace(out/'last.pt')
        if improved:
            torch.save(payload,out/'best.pt')
            atomic_json(out/'selected_validation.json',signed(dict(
                epoch=epoch,gains=valid['gains'],mean_gain=selected,
                selected_q=valid['selected_q'])))
        atomic_json(out/'history.json',signed(dict(history=history)))
        print(f'[P2 shared] epoch={epoch} updates={updates} mean-gain={selected:.6f}',flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    atomic_json(out/'selection.json',signed(dict(best_epoch=ck['epoch'],
           selected_mean_gain=best,admitted=best>0,weight_sha256=sha256(out/'best.pt'),
           note='Privileged clean only offline; this is not a deployment checkpoint.')))


def load_shared(path,device='cpu',smoke=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=SCHEMA or ck.get('protocol',{}).get('study')!='shared_reader':
        raise ValueError('Not a P2 shared reader checkpoint')
    protocol=ck['protocol']
    if protocol['smoke'] and not smoke:raise ValueError('Smoke checkpoint forbidden in formal diagnosis')
    model=PredictedGeometryControl(ModelConfig(**protocol['cfg']))
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),protocol


def compare_shared(a):
    idx=read_cache_index(a.index,'validation',a.smoke)
    model,p=load_shared(a.weight,a.device,a.smoke)
    if p['val_fingerprint']!=idx['fingerprint'] or p['teacher_config']!=idx['contract']['target_config']:
        raise ValueError('P2 shared model/validation/teacher mismatch')
    records=_evaluate_models(dict.fromkeys(ARMS,model),idx,a.device,a.batch_size)
    report=_describe(records,a.bootstrap,a.seed)
    report.update(schema=SCHEMA,study='shared_reader_fixed_weights',partition='validation',
                  checkpoint_sha256=sha256(a.weight),cache_index_sha256=sha256(a.index),
                  validation_fingerprint=idx['fingerprint'],per_family=records,
                  paired_physical_ids=True,privileged_clean_visible=True,official_ap_not_run=True,
                  note='One frozen reader under predicted vs clean geometry, trained on BOTH sources. Validation used for checkpoint selection; CI is descriptive, not confirmatory.')
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    atomic_json(dest,signed(report))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='cmd',required=True)
    d=s.add_parser('diagnose-p1',help='P0-style native-relative diagnostics of BOTH matched P1 readers')
    d.add_argument('--index',required=True)
    d.add_argument('--predicted-weight',required=True)
    d.add_argument('--clean-weight',required=True)
    d.add_argument('--output',required=True)
    t=s.add_parser('shared-train',help='One quality reader trained on both geometry sources, same exact labels')
    t.add_argument('--train-index',required=True)
    t.add_argument('--val-index',required=True)
    t.add_argument('--output',required=True)
    t.add_argument('--epochs',type=int,default=20)
    t.add_argument('--hidden',type=int,default=64)
    t.add_argument('--accumulate',type=int,default=1)
    t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--weight-decay',type=float,default=1e-4)
    t.add_argument('--cpu-threads',type=int,default=2)
    t.add_argument('--resume',action='store_true')
    c=s.add_parser('shared-compare',help='Compare frozen shared reader on two geometry sources')
    c.add_argument('--index',required=True)
    c.add_argument('--weight',required=True)
    c.add_argument('--output',required=True)
    for q in (d,t,c):
        q.add_argument('--device',default='cuda:0')
        q.add_argument('--batch-size',type=int,default=4)
        q.add_argument('--val-batch-size',type=int,default=4) if q is t else None
        q.add_argument('--bootstrap',type=int,default=5000) if q is not t else None
        q.add_argument('--seed',type=int,default=0)
        q.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.cmd=='diagnose-p1':diagnostic(args)
    elif args.cmd=='shared-train':fit_shared(args)
    else:compare_shared(args)


if __name__=='__main__':main()