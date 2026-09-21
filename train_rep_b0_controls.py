#!/usr/bin/env python3
"""Retrain B0 visual-content and no-cross-candidate controls with the original BCE."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rep_a_common import (check_runtime_sources, cdf_targets, digest, exclusive_run, list_frames,
    load_torch, read_frame, save_json, save_torch, seed_for, tune_margin)
from rep_b0_controls import CONTROLS, RepB0Control
from rep_followup_common import model_inputs, source_digest, release_memory


@torch.no_grad()
def validate(model, paths, contract, device):
    model.eval()
    predictions, labels = [], []
    for path in paths:
        d = read_frame(path, contract)
        predictions.append(model(model_inputs(d, device)).sigmoid().cpu().numpy())
        labels.append({k:d[k] for k in ('valid','utility','friction','zero_index')})
    return tune_margin(predictions, labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--control', choices=CONTROLS, required=True)
    p.add_argument('--reference-checkpoint', help='Inherit original B0 architecture, NOT its trained weights')
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-accum-steps', type=int, default=1); p.add_argument('--grad-clip', type=float, default=5.)
    p.add_argument('--dim', type=int, default=0); p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=2); p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--resume', action='store_true')
    p.add_argument('--max-train-frames', type=int, default=0); p.add_argument('--max-val-frames', type=int, default=0)
    p.add_argument('--progress-every', type=int, default=100)
    args = p.parse_args(); sys.argv = [sys.argv[0]]
    if args.epochs < 1 or args.grad_accum_steps < 1 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError('Invalid training configuration')
    manifest = check_runtime_sources(args.cache_root)
    init = load_torch(Path(args.cache_root)/'reader_init.pt')
    spec = dict(control=args.control, channels=int(init['model_config']['channels']),
        dim=int(args.dim or init['model_config']['out_dim']), heads=args.heads,
        layers=args.layers, dropout=args.dropout, prior_sigma_mm=30., variant='B0')
    contract = init['contract']; del init; release_memory()
    if args.reference_checkpoint:
        ref = load_torch(args.reference_checkpoint)
        if ref.get('experiment') != 'Rep-B' or ref.get('variant') != 'B0' or ref['contract'] != contract:
            raise ValueError('Reference must be the B0 checkpoint trained on this cache contract')
        if int(ref['model_spec']['channels']) != spec['channels']:
            raise ValueError('Reference B0 image-channel count differs')
        spec = dict(ref['model_spec'], control=args.control)
        del ref; release_memory()
    train = list_frames(args.cache_root,'train',args.max_train_frames)
    val = list_frames(args.cache_root,'test_seen',args.max_val_frames)
    config = {k:v for k,v in vars(args).items() if k not in
        ('output_dir','cache_root','resume','epochs','device','progress_every')}
    code = source_digest(('rep_b0_controls.py','rep_b_model.py','rep_a_model.py','train_rep_b0_controls.py'))
    signature = digest(dict(config=config, spec=spec, contract=contract, code=code, manifest=manifest,
        frames=[(str(x.relative_to(args.cache_root)),x.stat().st_size,x.stat().st_mtime_ns) for x in train+val]))
    device = torch.device(args.device); out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with exclusive_run(out/'.train.lock'):
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        model = RepB0Control(**spec).to(device)
        optim = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        start, history, best_key = 0, [], (-float('inf'),)*3
        latest = out/'checkpoint_latest.pt'
        if latest.exists():
            if not args.resume: raise FileExistsError(f'{latest}: use --resume or a fresh output-dir')
            ck = load_torch(latest)
            if ck['signature'] != signature: raise RuntimeError('Resume training/data/code contract changed')
            model.load_state_dict(ck['model'],strict=True); optim.load_state_dict(ck['optimizer'])
            start, history, best_key = ck['epoch']+1, ck['history'], tuple(ck['best_key'])
            torch.set_rng_state(ck['torch_rng'])
            if torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck['cuda_rng'])
            del ck
        save_json(out/'protocol.json',dict(signature=signature, model_spec=spec, config=config,
            train_frames=len(train), val_frames=len(val), parameter_count=sum(x.numel() for x in model.parameters()),
            objective='Same six-threshold BCE as B0; no ranking or new supervision',
            validation='Nominal test_seen, 100-129; also reported as validation_seen, not independent held-out',
            sample_interval='Inherited unchanged from Rep-A cache'))
        for epoch in range(start,args.epochs):
            model.train(); order=list(train)
            # Match the original B0 frame-order policy.
            random.Random(seed_for(args.seed,'rep_b_order',epoch)).shuffle(order)
            total, seen, updates = 0., 0, 0
            for begin in range(0,len(order),args.grad_accum_steps):
                chunk=order[begin:begin+args.grad_accum_steps]; optim.zero_grad(set_to_none=True)
                for path in chunk:
                    d=read_frame(path,contract); t=model_inputs(d,device)
                    logits=model(t); y=torch.from_numpy(cdf_targets(d['friction'])).to(device)
                    loss=F.binary_cross_entropy_with_logits(logits[t['valid']],y[t['valid']])
                    if not torch.isfinite(loss): raise FloatingPointError(str(path))
                    (loss/len(chunk)).backward(); total+=float(loss.detach()); seen+=1
                    del d,t,logits,y,loss
                if epoch==start and updates==0:
                    grad={n:sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
                          for n,m in model.named_children()}
                    if any(not np.isfinite(v) or v<=0 for v in grad.values()):
                        raise RuntimeError(f'Component gradient check failed: {grad}')
                    save_json(out/'gradient_check.json',grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
                optim.step(); updates+=1
                if seen%max(1,args.progress_every)<len(chunk):
                    print(f'[B0 {args.control}] epoch={epoch} {seen}/{len(order)} loss={total/seen:.5f}',flush=True)
            best,sweep=validate(model,val,contract,device)
            key=(best['selected_utility'],-best['harm08'],-best['move_rate']); improved=key>best_key
            if improved: best_key=key
            row=dict(epoch=epoch,train_loss=total/seen,updates=updates,val=best); history.append(row)
            ck=dict(experiment='Rep-B0-controls',control=args.control,variant=args.control,
                model_spec=spec,model=model.state_dict(),optimizer=optim.state_dict(),contract=contract,
                signature=signature,epoch=epoch,history=history,best_key=best_key,margin=float(best['margin']),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            if improved:
                save_torch(out/'checkpoint_best.pt',ck); save_json(out/'best.json',row)
                save_json(out/'margin_sweep_best.json',sweep)
            save_torch(latest,ck); save_json(out/'metrics.json',history)
            print(json.dumps(row),flush=True)


if __name__=='__main__': main()
