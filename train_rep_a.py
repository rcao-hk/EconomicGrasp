#!/usr/bin/env python3
"""Train Rep-A A0/A1/A2/A3 on fixed actions, with full reader/scorer gradients."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from rep_a_common import (check_runtime_sources, VARIANTS, aggregate, cdf_targets, digest, exclusive_run,
    list_frames, load_torch, metrics, perturb_depth, read_frame, save_json,
    save_torch, seed_for, tensors, training_case, tune_margin)
from rep_a_model import RepAReaderScorer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--grad-clip', type=float, default=5.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-train-frames', type=int, default=0)
    p.add_argument('--max-val-frames', type=int, default=0)
    p.add_argument('--max-bias-mm', type=float, default=20.)
    p.add_argument('--max-scale', type=float, default=.03)
    p.add_argument('--smooth-mm', type=float, default=10.)
    p.add_argument('--nominal-prob', type=float, default=.25)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--progress-every', type=int, default=100)
    return p


@torch.no_grad()
def validation(model, paths, contract, device):
    model.eval()
    predictions, frames = [], []
    for path in paths:
        d = read_frame(path, contract)
        prob = model(tensors(d, device)).sigmoid().cpu().numpy()
        predictions.append(prob)
        # Do not retain dense image maps across validation frames.
        frames.append({k: d[k] for k in ('valid','utility','friction','zero_index')})
    return tune_margin(predictions, frames)


def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    if args.epochs < 1 or args.grad_accum_steps < 1 or not 0 <= args.nominal_prob <= 1:
        raise ValueError('Invalid epochs/accumulation/probability')
    if min(args.max_bias_mm, args.max_scale, args.smooth_mm) < 0:
        raise ValueError('Negative augmentation amplitude')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly use --device cpu for a CPU smoke test')
    check_runtime_sources(args.cache_root)
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    init = load_torch(Path(args.cache_root)/'reader_init.pt')
    contract = init['contract']
    train_paths = list_frames(args.cache_root, 'train', args.max_train_frames)
    val_paths = list_frames(args.cache_root, 'test_seen', args.max_val_frames)
    config = vars(args).copy()
    for name in ('epochs','resume','device','progress_every','output_dir','cache_root'):
        config.pop(name)
    signature = digest({'config': config, 'contract': contract,
        'files': [(str(p.relative_to(args.cache_root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in train_paths+val_paths]})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out/'.train.lock'):
        model = RepAReaderScorer(init, args.variant).to(device)
        del init
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        latest = out/'checkpoint_latest.pt'
        start, history, best_key = 0, [], (-float('inf'), -float('inf'), -float('inf'))
        if latest.exists():
            if not args.resume:
                raise FileExistsError(f'{latest}; use --resume or a new output-dir')
            ck = load_torch(latest)
            if ck['signature'] != signature:
                raise RuntimeError('Training/data contract changed; cannot resume into this output-dir')
            model.load_state_dict(ck['model'], strict=True)
            optimizer.load_state_dict(ck['optimizer'])
            start, history, best_key = ck['epoch']+1, ck['history'], tuple(ck['best_key'])
            torch.set_rng_state(ck['torch_rng'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ck['cuda_rng'])
            del ck
        save_json(out/'protocol.json', {'variant': args.variant, 'signature': signature,
            'cache_contract': contract, 'config': config, 'epochs_requested': args.epochs,
            'train_frames': len(train_paths), 'val_frames': len(val_paths),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen': 'DINO, proposal DPT, metric depth, candidates; cached pre-enhancer features',
            'trainable': 'spatial enhancer, CVA grouping, action scorer, optional independent image reader',
            'validation': 'unperturbed PREDICTED depth only; margin selected here, never on held-out',
            'not_official_AP': True})
        for epoch in range(start, args.epochs):
            model.train()
            order = list(train_paths)
            random.Random(seed_for(args.seed, 'order', epoch)).shuffle(order)
            sums, cases, seen, updates = 0., {}, 0, 0
            for offset in range(0, len(order), args.grad_accum_steps):
                chunk = order[offset:offset+args.grad_accum_steps]
                optimizer.zero_grad(set_to_none=True)
                for path in chunk:
                    d = read_frame(path, contract)
                    t = tensors(d, device)
                    # Separate deterministic RNG, shared A1/A3 regardless of architecture/dropout.
                    s = seed_for(args.seed, 'train', epoch, int(d['scene_id']), int(d['anno_id']))
                    case = training_case(s, args.max_bias_mm, args.max_scale, args.smooth_mm,
                                         args.nominal_prob) if VARIANTS[args.variant][1] else 'nominal'
                    dep, _ = perturb_depth(t['depth'], case, s)
                    logits = model(t, dep)
                    mask = torch.from_numpy(d['valid'].astype(bool)).to(device)
                    y = torch.from_numpy(cdf_targets(d['friction'])).to(device)
                    loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss: {path} / {case}')
                    (loss/len(chunk)).backward()
                    sums += float(loss.detach()); seen += 1
                    kind = case.split(':')[0]; cases[kind] = cases.get(kind, 0)+1
                    del logits, loss, d, t, dep, mask, y
                if epoch == start and updates == 0:
                    norms = {name: sum(float(p.grad.detach().norm()) for p in part.parameters()
                                      if p.grad is not None) for name, part in model.named_children()}
                    if any(v <= 0 or not np.isfinite(v) for v in norms.values()):
                        raise RuntimeError(f'Missing/non-finite component gradients: {norms}')
                    save_json(out/'gradient_check.json', norms)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step(); updates += 1
                if seen % max(args.progress_every, 1) < len(chunk):
                    print(f'[REP-A {args.variant}] epoch={epoch} frames={seen}/{len(order)} loss={sums/seen:.5f}', flush=True)
            best, sweep = validation(model, val_paths, contract, device)
            key = (best['selected_utility'], -best['harm08'], -best['move_rate'])
            improved = key > best_key
            if improved:
                best_key = key
            row = {'epoch': epoch, 'train_loss': sums/seen, 'updates': updates,
                   'augmentation_counts': cases, 'val': best}
            history.append(row)
            payload = {'variant': args.variant, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'signature': signature,
                'contract': contract, 'model_config': model.config, 'margin': best['margin'],
                'history': history, 'best_key': best_key, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            if improved:
                save_torch(out/'checkpoint_best.pt', payload)
                save_json(out/'best.json', row)
                save_json(out/'margin_sweep_best.json', sweep)
            save_torch(latest, payload)
            save_json(out/'metrics.json', history)
            print(json.dumps(row, sort_keys=True), flush=True)
        print(f'[REP-A {args.variant}] finished; best={out / "checkpoint_best.pt"}', flush=True)


if __name__ == '__main__':
    main()
