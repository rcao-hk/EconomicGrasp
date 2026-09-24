#!/usr/bin/env python3
"""Jointly train correction representation and an isolated ranking residual.

Reuses E1/E2 exact-action cache; never evaluates CAD/DexNet during training.
Selection checkpoints use the SAME Seen selected-utility criterion as E1.
Ranking metrics are reported separately, and no held-out split is tuned.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from e1e2_common import (VERSION, digest, file_sha, get_batch, load_torch, lock,
    make_dataset, manifest, relative_cdf_loss, save_json, save_torch, seed_all,
    seed_for, select_centers)
from train_e1e2_cva import cache_paths, cached_bundle, check_input, read_cache
from dcr_cva_common import (DCR_VERSION, code_fingerprint, make_outputs,
                           rank_metrics, selected_rank_loss)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','stage1-checkpoint','cache-root','output-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--init-checkpoint',default='',help='E1/E2 warm start; empty starts from Stage-1')
    p.add_argument('--selection-loss',choices=('cdf','cdf_relative'),default='cdf')
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--rank-lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--grad-accum',type=int,default=4)
    p.add_argument('--relative-weight',type=float,default=1.)
    p.add_argument('--relative-beta',type=float,default=.1)
    p.add_argument('--rank-anchor-weight',type=float,default=.1)
    p.add_argument('--rank-hidden',type=int,default=128)
    p.add_argument('--rank-bound',type=float,default=.5,help='Maximum absolute log-odds change')
    p.add_argument('--query-chunk',type=int,default=64)
    p.add_argument('--group-chunk',type=int,default=512)
    p.add_argument('--seed',type=int,default=2032)
    p.add_argument('--val-every',type=int,default=1)
    p.add_argument('--no-error-training',action='store_true')
    p.add_argument('--max-train-frames',type=int,default=0)
    p.add_argument('--max-val-frames',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    return p


def grad_norm(module):
    return sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)


@torch.no_grad()
def validate(model, paths, ds, lookup, signature, device, chunk):
    from models.economicgrasp_cva_centers import extract_depth_features
    model.eval(); rows=[]
    for path in paths:
        fr=read_cache(path,signature)
        sid,aid=int(fr['scene_id']),int(fr['anno_id'])
        batch=get_batch(ds,lookup,sid,aid,device)
        pack=extract_depth_features(model.reference,batch)
        check_input(fr,batch,pack[0])
        for ci,case in enumerate(fr['cases']):
            bundle=cached_bundle(fr,ci,device)
            logits,residual,_,_=model(batch,bundle,case=str(case),case_seed=int(fr['case_seeds'][ci]),
                                     query_chunk=chunk,depth_pack=pack)
            target=torch.as_tensor(fr['target'][ci],device=device).float()
            outputs,sel=make_outputs(logits,residual,bundle,model.zero)
            q=torch.arange(len(sel),device=device); y=target.mean(-1)
            chosen=y[sel,q]; native=y[model.zero]
            oracle=y.masked_fill(~bundle['valid'],-torch.inf).max(0).values
            row={'case':str(case),'queries':len(q),
                 'selected_utility':float(chosen.mean()),'utility_gain':float((chosen-native).mean()),
                 'native_utility':float(native.mean()),'oracle_regret':float((oracle-chosen).mean()),
                 'move_rate':float((sel!=model.zero).float().mean())}
            for mode in ('local','stage1','anchored'):
                for k,v in rank_metrics(outputs[mode][:,0],chosen).items():
                    row[f'{mode}_{k}']=v
            rows.append(row)
        del batch,pack,fr
    by_case={}
    for case in sorted({r['case'] for r in rows}):
        rr=[r for r in rows if r['case']==case]
        n=sum(r['queries'] for r in rr)
        stats={'queries':n,'frames':len(rr)}
        for k in rr[0]:
            if k in ('case','queries'): continue
            use=[r for r in rr if r[k] is not None]
            # Pair concordance is weighted by informative pair count.
            weight=k.replace('pair_concordance','rank_pairs') if k.endswith('pair_concordance') else 'queries'
            den=sum(r[weight] for r in use)
            stats[k]=sum(r[k]*r[weight] for r in use)/den if den else None
        by_case[case]=stats
    metric=float(np.mean([r['selected_utility'] for r in by_case.values()]))
    return metric,by_case


def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    if min(args.epochs,args.grad_accum,args.val_every,args.query_chunk,args.group_chunk)<1:
        raise ValueError('Epoch/chunk/accumulation counts must be positive')
    if min(args.lr,args.rank_lr)<=0 or min(args.weight_decay,args.relative_weight,args.rank_anchor_weight)<0:
        raise ValueError('Invalid optimizer/loss configuration')
    seed_all(args.seed)
    cp=json.loads((Path(args.cache_root)/'protocol.json').read_text())
    ref_sha=file_sha(args.stage1_checkpoint)
    if cp['version']!=VERSION or cp['reference_sha256']!=ref_sha:
        raise RuntimeError('E1 cache/Stage-1/version mismatch; do not relabel by query index')
    signature_cache=digest(cp)
    paths=cache_paths(args.cache_root,cp,'train',args.max_train_frames)
    vp=cache_paths(args.cache_root,cp,'test_seen',args.max_val_frames)
    if not paths or not vp: raise RuntimeError('Empty train/validation schedule')
    device=torch.device(args.device)
    from models.economicgrasp_cva_centers import load_reference
    from models.economicgrasp_cva_dcr import DecoupledCenterRankingCVA
    model=DecoupledCenterRankingCVA(load_reference(args.stage1_checkpoint,device),cp['offsets_mm'],
        args.group_chunk,args.rank_hidden,args.rank_bound,args.seed).to(device)
    init_sha=None
    if args.init_checkpoint:
        ck=load_torch(args.init_checkpoint); init_sha=file_sha(args.init_checkpoint)
        if ck['version']!=VERSION or ck['protocol']['reference_sha256']!=ref_sha:
            raise RuntimeError('Warm start must be an E1/E2 checkpoint using this Stage-1')
        if ck['protocol']['cache_signature']!=signature_cache:
            raise RuntimeError('Warm start/action cache mismatch')
        model.warm_start_corrector(ck['model']); del ck
    correction_params=[p for p in model.corrector.parameters() if p.requires_grad]
    ranking_params=list(model.ranker.parameters())
    # Independent clipping and optimizer states: rank gradients cannot change
    # the correction update through a shared global gradient norm either.
    opt=torch.optim.AdamW(correction_params,lr=args.lr,weight_decay=args.weight_decay)
    ropt=torch.optim.AdamW(ranking_params,lr=args.rank_lr,weight_decay=args.weight_decay)
    ds,lookup=make_dataset(args.dataset_root,'train',cp['camera'])
    vds,vlookup=make_dataset(args.dataset_root,'test_seen',cp['camera'])
    config={k:v for k,v in vars(args).items() if k not in ('dataset_root','stage1_checkpoint',
        'cache_root','output_root','init_checkpoint','device','resume','epochs')}
    run={'version':DCR_VERSION,'code_sha256':code_fingerprint(),'cache_signature':signature_cache,
         'reference_sha256':ref_sha,'init_sha256':init_sha,'config':config,
         'variant':'DCR_'+args.selection_loss,'camera':cp['camera'],'offsets_mm':cp['offsets_mm'],
         'sample_interval':cp['sample_interval'],'train_frames':len(paths),'val_frames':len(vp),
         'checkpoint_selection':'Seen macro selected exact utility, same as E1; ranking proxy is diagnostic only',
         'rank_gradient':'detached CVA latents/CDF; separate optimizer and gradient clipping',
         'frozen':'Stage-1 depth/backbone/proposal/R/w/insertion depth',
         'trainable':'online E1 corrector plus bounded zero-init cross-query ranking head',
         'ranking_target':'exact utility of the actually selected cached physical actions'}
    out=Path(args.output_root); manifest(out,run); signature=digest(run)
    history=[]; start=0; best=-float('inf'); latest=out/'checkpoint_latest.pt'
    with lock(out/'.train.lock'):
        if latest.exists():
            if not args.resume: raise FileExistsError(f'{latest}; use --resume')
            ck=load_torch(latest)
            if ck['signature']!=signature: raise RuntimeError('DCR resume contract changed')
            model.load_learned_state(ck['model']); opt.load_state_dict(ck['optimizer'])
            ropt.load_state_dict(ck['rank_optimizer'])
            start=ck['epoch']+1; best=ck['best_metric']; history=ck['history']
            random.setstate(ck['rng_python']); np.random.set_state(ck['rng_numpy'])
            torch.set_rng_state(ck['rng_torch'])
            if torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck['rng_cuda'])
            del ck
        else:
            # Keep an explicit warm-start/zero-residual baseline. Do not force
            # a worse fine-tuned selector to replace an already-good E1 model.
            metric,by_case=validate(model,vp,vds,vlookup,signature_cache,device,args.query_chunk)
            best=metric
            row={'epoch':-1,'val_macro_utility':metric,'val_cases':by_case,
                 'note':'initial selector with zero ranking residual'}
            initial={'version':DCR_VERSION,'signature':signature,'protocol':run,'epoch':-1,
                     'model':model.learned_state(),'best_metric':best}
            save_torch(out/'checkpoint_initial.pt',initial)
            save_torch(out/'checkpoint_best.pt',initial)
            save_json(out/'initial.json',row); save_json(out/'best.json',row)
            print('[DCR INITIAL] '+json.dumps(row),flush=True)
        for epoch in range(start,args.epochs):
            model.train(); opt.zero_grad(set_to_none=True); ropt.zero_grad(set_to_none=True)
            order=list(paths); random.Random(seed_for(args.seed,'order',epoch)).shuffle(order)
            totals={}
            for i,path in enumerate(order):
                fr=read_cache(path,signature_cache); sid,aid=int(fr['scene_id']),int(fr['anno_id'])
                ci=0 if args.no_error_training else seed_for(args.seed,epoch,sid,aid)%len(fr['cases'])
                batch=get_batch(ds,lookup,sid,aid,device); bundle=cached_bundle(fr,ci,device)
                logits,residual,_,depth=model(batch,bundle,case=str(fr['cases'][ci]),
                    case_seed=int(fr['case_seeds'][ci]),query_chunk=args.query_chunk)
                check_input(fr,batch,depth); target=torch.as_tensor(fr['target'][ci],device=device)
                loss,parts=relative_cdf_loss(logits,target,bundle['valid'],model.zero,
                    'E1' if args.selection_loss=='cdf' else 'E2',args.relative_weight,args.relative_beta)
                sel=select_centers(logits.detach().sigmoid().mean(-1),bundle['valid'],model.zero)
                rloss,rparts=selected_rank_loss(residual,bundle['actions'][model.zero,:,0],sel,target,args.rank_anchor_weight)
                if not bool(torch.isfinite(loss+rloss)): raise FloatingPointError('Nonfinite DCR loss')
                gsize=min(args.grad_accum,len(order)-(i//args.grad_accum)*args.grad_accum)
                # Audit BEFORE correction backward: ranking alone must have zero
                # gradient in every corrector/reference parameter.
                (rloss/gsize).backward()
                if i==0 and epoch==start:
                    if any(p.grad is not None for p in model.corrector.parameters()):
                        raise RuntimeError('Ranking gradient leaked into the correction representation')
                (loss/gsize).backward()
                if i==0 and epoch==start:
                    norms={n:grad_norm(m) for n,m in model.corrector.named_children()}
                    if any(p.grad is not None for p in model.reference.parameters()):
                        raise RuntimeError('Gradient leaked into immutable Stage-1')
                    for name in ('image_adapter','enhancer','group','decoder'):
                        if norms[name]<=0: raise RuntimeError(f'No correction gradient in {name}')
                    save_json(out/'gradient_check.json',{'correction':norms,'ranker':grad_norm(model.ranker),
                        'ranking_to_corrector':0,'reference':0,'rank_pairs_in_audit_frame':rparts['rank_pairs']})
                if (i+1)%args.grad_accum==0 or i+1==len(order):
                    torch.nn.utils.clip_grad_norm_(correction_params,5.,error_if_nonfinite=True)
                    torch.nn.utils.clip_grad_norm_(ranking_params,5.,error_if_nonfinite=True)
                    opt.step(); ropt.step(); opt.zero_grad(set_to_none=True); ropt.zero_grad(set_to_none=True)
                for k,v in {'selection_loss':float(loss.detach()),'rank_loss':float(rloss.detach()),**parts,**rparts}.items():
                    totals[k]=totals.get(k,0.)+v
                if (i+1)%50==0:
                    print(f'[DCR] epoch={epoch} frame={i+1}/{len(order)} selection={totals["selection_loss"]/(i+1):.5f} rank={totals["rank_loss"]/(i+1):.5f}',flush=True)
                del fr,batch,bundle,logits,residual,loss,rloss,depth
            row={'epoch':epoch,'train':{k:v/len(order) for k,v in totals.items()}}
            improved=False
            if (epoch+1)%args.val_every==0 or epoch+1==args.epochs:
                metric,by_case=validate(model,vp,vds,vlookup,signature_cache,device,args.query_chunk)
                row.update(val_macro_utility=metric,val_cases=by_case)
                improved=metric>best; best=max(best,metric)
            history.append(row)
            ck={'version':DCR_VERSION,'signature':signature,'protocol':run,'epoch':epoch,
                'model':model.learned_state(),'optimizer':opt.state_dict(),'rank_optimizer':ropt.state_dict(),
                'history':history,'best_metric':best,'rng_python':random.getstate(),
                'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
                'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            save_torch(latest,ck)
            if improved:
                keep={k:v for k,v in ck.items() if k not in ('optimizer','rank_optimizer','history') and not k.startswith('rng_')}
                save_torch(out/'checkpoint_best.pt',keep); save_json(out/'best.json',row)
            save_json(out/'metrics.json',history); print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
