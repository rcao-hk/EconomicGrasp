"""P4-1: ordinary 2D metric-depth correction, followed by analytical geometry.

The corrector sees frozen RGB features and predicted depth ONLY. No grasp/Q,
region weighting, implicit field or reader gradient is used during training.
Primary checkpoint selection uses depth MAE, never action Q or official AP.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .p4_common import (SCHEMA, source_hashes, formal_rows, freeze_reader, score_physics,
    tensor_inputs, family_rows, paired_summary, verify_p2, new_directory, write_records,
    atomic_checkpoint, sha256, signed, atomic_json, atomic_npz, load_signed, frame_key,
    read_cache_index, load_cached, collate_cache, scene_balanced)
from .p4_depth_data import build, read_depth_index, DepthDataset, collate_depth, read_features
from .data import read_index, read_frame
from .geometry import VisibleSurfaceTeacher, support_points
from .p1_clean_visible import physical_tokens
from .experiment import balanced

MODEL_SCHEMA='shared-visible-geometry-p4-plain-depth-v1'


@dataclass
class DepthConfig:
    feature_channels: int = 128
    hidden: int = 32
    max_correction_m: float = .05
    depth_scale_m: float = 2.


class PlainDepthCorrector(nn.Module):
    """Shared full-image residual, independent of any candidate action set.

    Bounded *output* correction (unlike merely multiplying an unbounded head).
    Invalid original pixels stay invalid; this baseline does not invent support
    in missing rays. Zero initialization exactly recovers the original depth.
    """
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        if cfg.hidden<4 or cfg.feature_channels<1 or cfg.max_correction_m<=0 or cfg.depth_scale_m<=0:
            raise ValueError('Invalid depth architecture configuration')
        groups=4 if cfg.hidden%4==0 else 1
        self.project=nn.Sequential(nn.Conv2d(cfg.feature_channels,cfg.hidden,1),nn.GELU(),
                                  nn.Conv2d(cfg.hidden,cfg.hidden,3,padding=1),nn.GELU())
        self.refine=nn.Sequential(nn.Conv2d(cfg.hidden+2,cfg.hidden,3,padding=1),
            nn.GroupNorm(groups,cfg.hidden),nn.GELU(),
            nn.Conv2d(cfg.hidden,cfg.hidden,3,padding=1),nn.GroupNorm(groups,cfg.hidden),nn.GELU(),
            nn.Conv2d(cfg.hidden,1,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)

    def forward(self,inputs):
        # No labels, camera poses, actions, score priors or reader features added.
        feature=inputs['feature_map'];depth=inputs['pred_depth']
        if feature.ndim!=4 or depth.ndim!=3 or feature.shape[0]!=depth.shape[0]:
            raise ValueError('Expected actual BxCxHfxWf features and BxHxW depth')
        valid=torch.isfinite(depth)&(depth>1e-6)
        if not torch.isfinite(feature).all() or not torch.isfinite(depth).all() or (depth<0).any():
            raise ValueError('Nonfinite RGB features or invalid metric depth')
        feature=F.interpolate(self.project(feature),size=depth.shape[-2:],mode='bilinear',align_corners=False)
        raw=self.refine(torch.cat((feature,depth[:,None]/self.cfg.depth_scale_m,valid[:,None].to(depth)),1))[:,0]
        delta=self.cfg.max_correction_m*raw.tanh()
        return torch.where(valid,(depth+delta).clamp_min(1e-6),torch.zeros_like(depth))


def depth_loss(pred,target,original,gradient_weight=.5,normalizer_m=.01):
    """Ordinary pixel L1 + 2D first-difference loss; image-balanced means."""
    if pred.shape!=target.shape or original.shape!=target.shape or pred.ndim!=3:
        raise ValueError('Depth loss needs aligned BxHxW')
    if gradient_weight<0 or normalizer_m<=0:raise ValueError('Invalid depth loss scale')
    losses=[];metrics=[]
    for p,t,o in zip(pred,target,original):
        valid=torch.isfinite(t)&(t>1e-6)&torch.isfinite(o)&(o>1e-6)
        if not valid.any():raise ValueError('Frame has no common valid depth supervision')
        error=p-t; l1=error[valid].abs().mean(); grads=[]
        for axis in (0,1):
            if axis==0:
                mask=valid[1:]&valid[:-1];diff=error[1:]-error[:-1]
            else:
                mask=valid[:,1:]&valid[:,:-1];diff=error[:,1:]-error[:,:-1]
            if mask.any():grads.append(diff[mask].abs().mean())
        gradient=torch.stack(grads).mean() if grads else p.sum()*0
        losses.append((l1+gradient_weight*gradient)/normalizer_m)
        metrics.append(dict(mae_mm=float(l1.detach()*1000),
            rmse_mm=float(error[valid].square().mean().sqrt().detach()*1000),
            gradient_mae_mm=float(gradient.detach()*1000),
            valid_fraction=float(valid.float().mean()),
            teacher_valid_fraction=float((t>1e-6).float().mean()),
            correction_abs_mm=float((p-o)[valid].abs().mean().detach()*1000)))
    return torch.stack(losses).mean(),metrics


def transfer(batch,device):
    # K is returned by the data loader for provenance, not used by this network.
    return {k:batch['inputs'][k].to(device) for k in ('feature_map','pred_depth')},batch['target'].to(device)


def make_loader(dataset,batch_size,workers,order=None,seed=0):
    ds=dataset if order is None else Subset(dataset,[int(x) for x in order])
    gen=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,
                      collate_fn=collate_depth,generator=gen,pin_memory=False)


@torch.no_grad()
def evaluate_depth(model,index,device,batch_size=4,workers=0):
    model.eval();rows=[]
    for batch in make_loader(DepthDataset(index),batch_size,workers):
        inputs,target=transfer(batch,device);out=model(inputs)
        _,after=depth_loss(out,target,inputs['pred_depth'])
        _,before=depth_loss(inputs['pred_depth'],target,inputs['pred_depth'])
        for j,(row,a,b) in enumerate(zip(batch['rows'],after,before)):
            valid=(target[j]>1e-6)&(inputs['pred_depth'][j]>1e-6)
            needed=(target[j]-inputs['pred_depth'][j]).abs()
            delta=(out[j]-inputs['pred_depth'][j]).abs()
            a['teacher_correction_exceeds_bound_fraction']=float((needed[valid]>model.cfg.max_correction_m).float().mean())
            a['correction_near_bound_fraction']=float((delta[valid]>=.99*model.cfg.max_correction_m).float().mean())
            rows.append(dict(row=row,metrics=dict(a,baseline_mae_mm=b['mae_mm'],
                                                 baseline_gradient_mae_mm=b['gradient_mae_mm'])))
    return dict(summary={k:balanced(rows,k) for k in rows[0]['metrics']},rows=rows)


def fit(a):
    if min(a.batch_size,a.accumulate,a.val_batch_size,a.epochs,a.cpu_threads)<1 or a.workers<0:
        raise ValueError('Invalid P4-1 batch/training budget')
    if a.lr<=0 or a.weight_decay<0 or a.gradient_weight<0 or a.stop_after_epoch<0:
        raise ValueError('Invalid optimizer/loss/stop setting')
    tr=read_depth_index(a.train_index,'train',a.smoke);va=read_depth_index(a.val_index,'validation',a.smoke)
    for k in ('source','plan','target_config'):
        if tr['contract'][k]!=va['contract'][k]:raise ValueError('Depth train/validation provenance differs')
    if {e['row']['scene'] for e in tr['entries']} & {e['row']['scene'] for e in va['entries']}:
        raise ValueError('Depth train/validation scene overlap')
    if tr['entries'][0]['feature_shape'][0]!=va['entries'][0]['feature_shape'][0]:raise ValueError('Channel mismatch')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.cpu_threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=DepthConfig(tr['entries'][0]['feature_shape'][0],a.hidden,a.max_correction_m)
    model=PlainDepthCorrector(cfg).to(a.device)
    optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    protocol=dict(schema=MODEL_SCHEMA,architecture=asdict(cfg),source_hashes=source_hashes(),
        train_fingerprint=tr['fingerprint'],val_fingerprint=va['fingerprint'],
        train_source=tr['contract']['source'],teacher_config=tr['contract']['target_config'],
        seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,accumulate=a.accumulate,
        val_batch_size=a.val_batch_size,lr=a.lr,weight_decay=a.weight_decay,
        gradient_weight=a.gradient_weight,normalizer_m=.01,smoke=bool(a.smoke),
        parameters=sum(p.numel() for p in model.parameters()),
        objective='whole-image valid pixel L1 + gradient_weight * horizontal/vertical first-difference L1',
        selection='minimum scene/domain-balanced validation depth MAE, earliest tie, epoch0 eligible',
        no_quality_supervision=True,no_reader_updates=True,no_action_conditioning=True)
    rng=np.random.default_rng(a.seed);updates=0;start=0;history=[]
    if a.resume:
        old=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if old['protocol']!=protocol:raise ValueError('Changed P4-1 resume contract')
        model.load_state_dict(old['model'],strict=True);optim.load_state_dict(old['optimizer'])
        rng.bit_generator.state=old['numpy_rng'];torch.set_rng_state(old['torch_rng'].cpu())
        start,updates,best,history=old['epoch'],old['updates'],old['best'],old['history']
    else:
        val=evaluate_depth(model,va,a.device,a.val_batch_size,a.workers)
        if abs(val['summary']['mae_mm']-val['summary']['baseline_mae_mm'])>1e-5:
            raise RuntimeError('Zero initialized depth does not reproduce baseline')
        best=val['summary']['mae_mm'];history=[dict(epoch=0,updates=0,validation=val['summary'])]
    atomic_json(root/'protocol.json',signed(protocol))
    def payload(epoch):
        return dict(schema=MODEL_SCHEMA,protocol=protocol,epoch=epoch,updates=updates,best=best,
            model=model.state_dict(),optimizer=optim.state_dict(),history=history,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
    if not a.resume:
        atomic_checkpoint(root/'best.pt',payload(0));atomic_checkpoint(root/'last.pt',payload(0))
        atomic_json(root/'selected_validation.json',signed(val))
    end=min(a.epochs,a.stop_after_epoch) if a.stop_after_epoch else a.epochs
    if end<start:raise ValueError('Requested stop precedes resumed epoch')
    ds=DepthDataset(tr);effective=a.batch_size*a.accumulate
    for epoch in range(start+1,end+1):
        model.train();order=rng.permutation(len(ds));optim.zero_grad(set_to_none=True);losses=[]
        loader=make_loader(ds,a.batch_size,a.workers,order,seed=a.seed+epoch)
        processed=0
        for i,batch in enumerate(loader):
            inputs,target=transfer(batch,a.device);prediction=model(inputs)
            loss,_=depth_loss(prediction,target,inputs['pred_depth'],a.gradient_weight)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite depth loss')
            micro=len(batch['rows']); group_start=(processed//effective)*effective
            group_size=min(effective,len(ds)-group_start)
            (loss*micro/group_size).backward();losses.append((float(loss.detach()),micro));processed+=micro
            if processed%effective==0 or processed==len(ds):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optim.step();optim.zero_grad(set_to_none=True);updates+=1
        val=evaluate_depth(model,va,a.device,a.val_batch_size,a.workers)
        value=val['summary']['mae_mm'];improved=value<best-1e-12
        if improved:best=value
        history.append(dict(epoch=epoch,updates=updates,
            train_loss=sum(v*n for v,n in losses)/sum(n for v,n in losses),validation=val['summary']))
        atomic_checkpoint(root/'last.pt',payload(epoch))
        if improved:
            atomic_checkpoint(root/'best.pt',payload(epoch));atomic_json(root/'selected_validation.json',signed(val))
        atomic_json(root/'history.json',signed(dict(history=history)))
        print(f'[P4 depth] epoch={epoch} updates={updates} val-MAE={value:.6f}mm',flush=True)
    chosen=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    atomic_json(root/'selection.json',signed(dict(selected_epoch=chosen['epoch'],
        selected_depth_mae_mm=best,baseline_depth_mae_mm=history[0]['validation']['mae_mm'],
        depth_admitted=bool(chosen['epoch']>0 and best<history[0]['validation']['mae_mm']),
        action_admitted_not_evaluated=True,completed_epochs=end,complete=end==a.epochs,
        best_sha256=sha256(root/'best.pt'),no_action_quality_used_for_selection=True)))
    return model


def load_corrector(path,device='cpu',smoke=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=MODEL_SCHEMA:raise ValueError('Not a P4 ordinary-depth checkpoint')
    p=ck['protocol']
    if p['smoke'] and not smoke:raise ValueError('Smoke corrector cannot enter formal evaluation')
    model=PlainDepthCorrector(DepthConfig(**p['architecture'])).to(device)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    return model,p,ck['epoch']


def support_errors(pred,corr,clean,tau):
    """Same support sites/masks across all three inputs; masked absence is null."""
    known=(pred[...,2:].argmax(-1)!=3)&(corr[...,2:].argmax(-1)!=3)&(clean[...,2:].argmax(-1)!=3)
    rel_mask=known[:,1:]&known[:,:1]
    result={}
    for arm,phys in (('predicted',pred),('corrected',corr)):
        for c,name in ((0,'udf'),(1,'projective')):
            absolute=np.abs(phys[...,c]-clean[...,c])*tau*1000
            relative=np.abs((phys[:,1:,:,c]-phys[:,:1,:,c])-(clean[:,1:,:,c]-clean[:,:1,:,c]))*tau*1000
            result[arm+'_'+name+'_mae_mm']=float(absolute[known].mean()) if known.any() else None
            result[arm+'_'+name+'_delta_mae_mm']=float(relative[rel_mask].mean()) if rel_mask.any() else None
    result['common_observed_fraction']=float(known.mean())
    result['relative_common_observed_fraction']=float(rel_mask.mean())
    return result


def evaluate_actions(a):
    if a.batch_size<1 or a.bootstrap<1 or a.limit<0 or (a.limit and not a.smoke):raise ValueError('Invalid evaluation budget')
    if not a.smoke and not a.expected_p2:
        raise ValueError('Formal depth/action evaluation requires the signed P2 reference')
    di=read_depth_index(a.depth_index,'validation',a.smoke)
    pi=read_cache_index(a.p1_index,'validation',a.smoke);formal_rows(pi,'validation',a.smoke)
    oi=read_index(a.original_index,'validation',a.smoke)
    if (di['contract']['original_index_fingerprint']!=oi['fingerprint'] or
        pi['contract']['original_index_fingerprint']!=oi['fingerprint'] or
        pi['contract']['original_index_sha256']!=sha256(a.original_index)):
        raise ValueError('P1, original and depth validation provenance differs')
    original_by_key={frame_key(e['row']):e for e in oi['entries']}
    p1_by_key={frame_key(e['row']):e for e in pi['entries']}
    depth_keys=[frame_key(e['row']) for e in di['entries']]
    if (not set(depth_keys)<=set(original_by_key) or not set(depth_keys)<=set(p1_by_key) or
        (not a.smoke and (set(depth_keys)!=set(original_by_key) or set(depth_keys)!=set(p1_by_key)))):
        raise ValueError('P1, original and depth validation frame identities differ')
    originals=[original_by_key[k] for k in depth_keys]
    paired=[p1_by_key[k] for k in depth_keys]
    if any(d['source_sha256']!=o['sha256'] or d['row']!=o['row'] or d['row']!=p['row']
           for d,o,p in zip(di['entries'],originals,paired)):
        raise ValueError('Matched frame provenance changed')
    corrector,protocol,epoch=load_corrector(a.weight,a.device,a.smoke)
    if not a.smoke:
        selection=load_signed(Path(a.weight).parent/'selection.json')
        if (not selection['complete'] or selection['best_sha256']!=sha256(a.weight) or
            selection['selected_epoch']!=epoch):
            raise ValueError('Formal evaluation requires completed depth-only selection, not a hand-picked epoch')
    if protocol['val_fingerprint']!=di['fingerprint']:raise ValueError('Depth checkpoint/index mismatch')
    reader,reader_protocol=freeze_reader(a.shared_weight,pi,a.device,a.smoke)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=new_directory(a.output);(out/'frames').mkdir()
    records={arm:[] for arm in ('predicted','corrected','clean_visible')};depth_rows=[];geometry_rows=[];output_frames=[]
    n=min(a.limit,len(di['entries'])) if a.limit else len(di['entries'])
    ds=DepthDataset(di);tau=pi['contract']['tau'];band=pi['contract']['band']
    pred_parity=0.
    for start in range(0,n,a.batch_size):
        end=min(start+a.batch_size,n);items=[ds[i] for i in range(start,end)]
        db=collate_depth(items);inputs,target=transfer(db,a.device)
        with torch.inference_mode():corrected=corrector(inputs)
        _,dm=depth_loss(corrected,target,inputs['pred_depth'])
        _,bm=depth_loss(inputs['pred_depth'],target,inputs['pred_depth'])
        frames=[load_cached(paired[i]) for i in range(start,end)]
        batch=collate_cache(frames);phys=[]
        for b,i in enumerate(range(start,end)):
            old=read_frame(originals[i]);cached=frames[b]
            if (not np.array_equal(old['physical_id'],cached['physical_id']) or
                not np.array_equal(old['quality_count'],cached['quality_count']) or
                not np.array_equal(old['family_id'],cached['family_id']) or
                originals[i]['sha256']!=paired[i]['original_frame_sha256']):
                raise ValueError('Same frame does not contain the same physical action bank')
            points,local=support_points(torch.from_numpy(old['bank']))
            if not np.allclose(local.numpy(),cached['local_xyz'],atol=1e-6):raise ValueError('Support coordinates changed')
            xyz=points.reshape(-1,3).numpy()
            baseline=physical_tokens(VisibleSurfaceTeacher(old['pred_depth'],old['K'],tau,band),xyz).reshape(cached['predicted_physics'].shape)
            pred_parity=max(pred_parity,float(np.abs(baseline-cached['predicted_physics']).max()))
            if pred_parity>1e-6:raise RuntimeError('Analytical Predicted/P1 cache parity failed')
            depth=corrected[b].cpu().numpy()
            physics=physical_tokens(VisibleSurfaceTeacher(depth,old['K'],tau,band),xyz).reshape(baseline.shape)
            phys.append(physics)
            row=di['entries'][i]['row']
            depth_rows.append(dict(row=row,metrics=dict(dm[b],baseline_mae_mm=bm[b]['mae_mm'])))
            geometry_rows.append(dict(row=row,metrics=support_errors(baseline,physics,cached['clean_visible_physics'],tau)))
            path=out/'frames'/(frame_key(row)+'.npz')
            atomic_npz(path,dict(corrected_depth=depth,corrected_physics=physics,
                      physical_id=cached['physical_id'],family_id=cached['family_id']))
            output_frames.append(dict(row=row,path=str(path.resolve()),sha256=sha256(path),
                original_sha256=originals[i]['sha256'],p1_frame_sha256=paired[i]['sha256']))
        batch['corrected_physics']=np.stack(phys);local,dims=tensor_inputs(batch,a.device)
        for arm in records:
            with torch.inference_mode():
                z=score_physics(reader,torch.as_tensor(batch[arm+'_physics'],device=a.device),local,dims).cpu().numpy()
            for b,i in enumerate(range(start,end)):
                records[arm].extend(family_rows(z[b],frames[b],di['entries'][i]['row'],
                                   pi['contract']['offsets_m'],batch[arm+'_physics'][b]))
        print(f'[P4 depth/actions] {end}/{n} frames',flush=True)
    checked=verify_p2(a.expected_p2,pi,a.shared_weight,records,bool(a.limit) or n!=len(pi['entries']))
    summaries={arm:paired_summary(rr,None if arm=='predicted' else records['predicted'],a.bootstrap,a.seed) for arm,rr in records.items()}
    report=dict(schema=SCHEMA,study='P4-1 ordinary-depth, frozen P2 readout',smoke=bool(a.smoke),
        n_frames=n,n_families=len(records['predicted']),selected_epoch=epoch,
        sources=dict(corrector_sha256=sha256(a.weight),reader_sha256=sha256(a.shared_weight),
            corrector_protocol=protocol,reader_protocol=reader_protocol,p1_index=pi['fingerprint'],
            depth_index=di['fingerprint'],source_hashes=source_hashes()),
        scores=summaries,depth_metrics={k:balanced(depth_rows,k) for k in depth_rows[0]['metrics']},
        support_geometry={k:balanced(geometry_rows,k) for k in geometry_rows[0]['metrics']},
        corrected_action_admitted=summaries['corrected']['gain']>0,
        analytical_predicted_parity_max_abs=pred_parity,published_p2_gain_parity_checked=checked,
        action_bank_changed=False,quality_labels_regenerated=False,reader_updated=False,
        official_ap_not_run=True,privileged_clean_visible_reference=True,
        warning='Corrector selected only by depth MAE; Q cannot reselect epochs. Clean is offline privilege. '
                'Same fixed action bank; no end-to-end proposal update or official AP. Negative depth-baseline '
                'results do not establish the necessity of an implicit field.')
    report['artifacts']=dict(frames=output_frames,
        families={arm:write_records(out/(arm+'_families.jsonl.gz'),rr) for arm,rr in records.items()},
        depth=write_records(out/'depth_metrics.jsonl.gz',depth_rows),
        geometry=write_records(out/'support_geometry.jsonl.gz',geometry_rows))
    atomic_json(out/'summary.json',signed(report))
    lines=['# P4-1：普通2D深度修正验证','',
           '固定动作bank和P2 reader；深度模型仅以像素/梯度监督训练。Q不是AP。','',
           '|来源|Selected Q×100|相对native Q pp|','|---|---:|---:|']
    for arm,s in summaries.items():lines.append(f"|{arm}|{s['selected_q']*100:.5f}|{s['gain']*100:+.5f}|")
    lines+=['',f"深度选定epoch={epoch}；没有按动作Q重选。",'完整来源身份与场景配对区间见 summary.json。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def predict_depth(a):
    model,protocol,epoch=load_corrector(a.weight,a.device,a.smoke)
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    with np.load(a.inputs,allow_pickle=False) as z:
        x={k:torch.from_numpy(np.asarray(z[k],np.float32)).to(a.device) for k in ('feature_map','pred_depth')}
    single=x['feature_map'].ndim==3
    if single:x={k:v.unsqueeze(0) for k,v in x.items()}
    with torch.inference_mode():depth=model(x).cpu().numpy()
    atomic_npz(path,dict(corrected_depth=depth[0] if single else depth))
    atomic_json(path.with_suffix('.json'),signed(dict(schema=MODEL_SCHEMA,epoch=epoch,
        input_sha256=sha256(a.inputs),weight_sha256=sha256(a.weight),output_sha256=sha256(path),
        no_teacher_access=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    b=s.add_parser('build');b.add_argument('--index',required=True);b.add_argument('--targets',required=True)
    b.add_argument('--output',required=True);b.add_argument('--resume',action='store_true');b.add_argument('--limit',type=int,default=0);b.set_defaults(func=build)
    t=s.add_parser('train');t.add_argument('--train-index',required=True);t.add_argument('--val-index',required=True)
    t.add_argument('--output',required=True);t.add_argument('--epochs',type=int,default=20);t.add_argument('--batch-size',type=int,default=4)
    t.add_argument('--accumulate',type=int,default=1);t.add_argument('--val-batch-size',type=int,default=4)
    t.add_argument('--hidden',type=int,default=32);t.add_argument('--max-correction-m',type=float,default=.05)
    t.add_argument('--gradient-weight',type=float,default=.5);t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--weight-decay',type=float,default=1e-4);t.add_argument('--workers',type=int,default=2)
    t.add_argument('--cpu-threads',type=int,default=2);t.add_argument('--seed',type=int,default=0)
    t.add_argument('--resume',action='store_true');t.add_argument('--stop-after-epoch',type=int,default=0);t.set_defaults(func=fit)
    e=s.add_parser('evaluate');e.add_argument('--depth-index',required=True);e.add_argument('--p1-index',required=True)
    e.add_argument('--original-index',required=True);e.add_argument('--weight',required=True);e.add_argument('--shared-weight',required=True)
    e.add_argument('--expected-p2',default='');e.add_argument('--output',required=True);e.add_argument('--batch-size',type=int,default=4)
    e.add_argument('--bootstrap',type=int,default=5000);e.add_argument('--seed',type=int,default=0);e.add_argument('--limit',type=int,default=0);e.set_defaults(func=evaluate_actions)
    i=s.add_parser('predict');i.add_argument('--weight',required=True);i.add_argument('--inputs',required=True);i.add_argument('--output',required=True);i.set_defaults(func=predict_depth)
    for sub in (b,t,e,i):sub.add_argument('--smoke',action='store_true')
    for sub in (t,e,i):sub.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.func(args)


if __name__=='__main__':main()
