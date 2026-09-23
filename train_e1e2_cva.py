#!/usr/bin/env python3
"""Train the online-image CVA representation, without online CAD/DexNet calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from e1e2_common import (VERSION, array_sha, batch_fingerprint, digest, file_sha, get_batch,
    load_torch, lock, make_dataset, manifest, relative_cdf_loss, save_json, save_torch,
    schedule, seed_all, seed_for, select_centers)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--stage1-checkpoint', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--variant', choices=('E1', 'E2'), required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--relative-weight', type=float, default=1.)
    p.add_argument('--relative-beta', type=float, default=.1)
    p.add_argument('--query-chunk', type=int, default=64)
    p.add_argument('--group-chunk', type=int, default=512)
    p.add_argument('--seed', type=int, default=2031)
    p.add_argument('--val-every', type=int, default=1)
    p.add_argument('--no-error-training', action='store_true')
    p.add_argument('--max-train-frames', type=int, default=0)
    p.add_argument('--max-val-frames', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    return p


def cache_paths(root, protocol, split, limit=0):
    expected = schedule(split, protocol['sample_interval'])
    if protocol['max_frames_per_split']:
        expected = expected[:protocol['max_frames_per_split']]
    if limit:
        expected = expected[:limit]
    files = [Path(root)/split/f'scene_{s:04d}'/f'ann_{a:04d}.npz' for s, a in expected]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Incomplete {split} cache ({len(missing)} missing); first: {missing[0]}')
    return files


def read_cache(path, signature):
    with np.load(path, allow_pickle=False) as z:
        fr = {k: z[k] for k in z.files}
    if str(fr['signature']) != signature:
        raise RuntimeError(f'Cache signature mismatch at {path}')
    if str(fr['action_sha']) != array_sha(fr['actions'][..., 1:16]):
        raise RuntimeError(f'Exact-action identity mismatch at {path}')
    return fr


def cached_bundle(fr, ci, device):
    return {k: torch.as_tensor(fr[k][ci].copy(), device=device)
            for k in ('actions', 'valid', 'token_ids', 'view_xyz', 'angle_ids', 'depth_ids')}


def check_input(fr, batch, actual_depth):
    if str(fr['input_sha']) != batch_fingerprint(batch):
        raise RuntimeError('Cached RGB/crop/K/pose differs from online input; labels cannot be reused')
    expected = torch.as_tensor(fr['nominal_depth'], device=actual_depth.device)
    error = float((actual_depth[0]-expected).abs().max())
    if error > 1e-5:
        raise RuntimeError(f'Frozen depth changed (max_abs={error}); do not reuse cached physical actions')


@torch.no_grad()
def validate(model, paths, dataset, lookup, signature, device, query_chunk):
    model.eval(); rows = []
    for path in paths:
        fr = read_cache(path, signature)
        sid, aid = int(fr['scene_id']), int(fr['anno_id'])
        batch = get_batch(dataset, lookup, sid, aid, device)
        for ci, case in enumerate(fr['cases']):
            bundle = cached_bundle(fr, ci, device)
            logits, _, nominal_depth = model(batch, bundle, str(case), int(fr['case_seeds'][ci]), query_chunk=query_chunk)
            check_input(fr, batch, nominal_depth)
            target = torch.as_tensor(fr['target'][ci], device=device)
            u, y = logits.sigmoid().mean(-1), target.mean(-1)
            sel = select_centers(u, bundle['valid'], model.zero)
            q = torch.arange(y.shape[1], device=device)
            chosen = y[sel, q]; native = y[model.zero]
            oracle = y.masked_fill(~bundle['valid'], -torch.inf).max(0).values
            rows.append({'scene_id': sid, 'anno_id': aid, 'case': str(case), 'queries': len(q),
                         'selected_utility': float(chosen.mean()), 'native_utility': float(native.mean()),
                         'utility_gain': float((chosen-native).mean()), 'oracle_regret': float((oracle-chosen).mean()),
                         'move_rate': float((sel != model.zero).float().mean())})
    by_case = {}
    for case in sorted({r['case'] for r in rows}):
        rr = [r for r in rows if r['case'] == case]
        n = sum(r['queries'] for r in rr)
        by_case[case] = {k: sum(r[k]*r['queries'] for r in rr)/n
                         for k in ('selected_utility', 'native_utility', 'utility_gain', 'oracle_regret', 'move_rate')}
        by_case[case]['queries'] = n
    metric = float(np.mean([v['selected_utility'] for v in by_case.values()]))
    return metric, by_case


def main():
    args = parser().parse_args(); sys.argv = [sys.argv[0]]
    if min(args.epochs, args.grad_accum, args.val_every, args.query_chunk, args.group_chunk) < 1:
        raise ValueError('Epoch/accumulation/chunk counts must be positive')
    seed_all(args.seed)
    protocol = json.loads((Path(args.cache_root)/'protocol.json').read_text())
    ref_sha = file_sha(args.stage1_checkpoint)
    if protocol['reference_sha256'] != ref_sha or protocol['version'] != VERSION:
        raise RuntimeError('Cache/reference/version mismatch')
    cache_signature = digest(protocol)
    paths = cache_paths(args.cache_root, protocol, 'train', args.max_train_frames)
    val_paths = cache_paths(args.cache_root, protocol, 'test_seen', args.max_val_frames)
    from models.economicgrasp_cva_centers import CenterHypothesisCVA, load_reference
    device = torch.device(args.device)
    reference = load_reference(args.stage1_checkpoint, device)
    model = CenterHypothesisCVA(reference, protocol['offsets_mm'], args.group_chunk).to(device)
    # No depth/backbone/seed/view/native-pose parameters can enter the optimizer.
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    train_ds, train_lookup = make_dataset(args.dataset_root, 'train', protocol['camera'])
    val_ds, val_lookup = make_dataset(args.dataset_root, 'test_seen', protocol['camera'])
    config = {k: v for k, v in vars(args).items() if k not in
              ('dataset_root', 'stage1_checkpoint', 'cache_root', 'output_root', 'device', 'resume', 'epochs')}
    run = {'version': VERSION, 'cache_signature': cache_signature, 'reference_sha256': ref_sha,
           'config': config, 'camera': protocol['camera'], 'sample_interval': protocol['sample_interval'],
           'offsets_mm': protocol['offsets_mm'], 'train_frames': len(paths), 'val_frames': len(val_paths),
           'trainable': 'online DPT image adapter, enhancer, CVA grouping, CDF and zero-init action adapter',
           'frozen': 'foundation backbone, metric depth incl. global_film, native proposal/view/R/w/insertion depth',
           'checkpoint_selection': 'Seen macro exact selected utility; held-out data never select checkpoints'}
    out = Path(args.output_root); manifest(out, run); signature = digest(run)
    start, history, best = 0, [], -float('inf')
    latest = out/'checkpoint_latest.pt'
    with lock(out/'.train.lock'):
        if latest.exists():
            if not args.resume:
                raise FileExistsError(f'{latest}; use --resume')
            ck = load_torch(latest)
            if ck['signature'] != signature:
                raise RuntimeError('Training resume contract mismatch')
            model.load_learned_state(ck['model']); optimizer.load_state_dict(ck['optimizer'])
            start, history, best = ck['epoch']+1, ck['history'], ck['best_metric']
            random.setstate(ck['rng_python']); np.random.set_state(ck['rng_numpy'])
            torch.set_rng_state(ck['rng_torch'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ck['rng_cuda'])
            del ck
        for epoch in range(start, args.epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            order = list(paths); random.Random(seed_for(args.seed, 'order', epoch)).shuffle(order)
            totals = {'loss': 0., 'cdf_bce': 0., 'relative_huber': 0.}
            for i, path in enumerate(order):
                fr = read_cache(path, cache_signature)
                sid, aid = int(fr['scene_id']), int(fr['anno_id'])
                ci = 0 if args.no_error_training else seed_for(args.seed, epoch, sid, aid) % len(fr['cases'])
                batch = get_batch(train_ds, train_lookup, sid, aid, device)
                bundle = cached_bundle(fr, ci, device)
                logits, _, nominal_depth = model(batch, bundle, str(fr['cases'][ci]),
                                                int(fr['case_seeds'][ci]), query_chunk=args.query_chunk)
                check_input(fr, batch, nominal_depth)
                target = torch.as_tensor(fr['target'][ci], device=device)
                loss, parts = relative_cdf_loss(logits, target, bundle['valid'], model.zero,
                                                args.variant, args.relative_weight, args.relative_beta)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                group_start = (i//args.grad_accum)*args.grad_accum
                group_size = min(args.grad_accum, len(order)-group_start)
                (loss/group_size).backward()
                if i == 0 and epoch == start:
                    norms = {name: sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
                             for name, module in model.named_children()}
                    if any(p.grad is not None for p in model.reference.parameters()):
                        raise RuntimeError('Gradient leaked into immutable proposal generator')
                    for name in ('image_adapter', 'enhancer', 'group', 'decoder'):
                        if norms[name] <= 0:
                            raise RuntimeError(f'No end-to-end gradient in {name}')
                    save_json(out/'gradient_check.json', norms)
                if (i+1) % args.grad_accum == 0 or i+1 == len(order):
                    torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                totals['loss'] += float(loss.detach())
                for k in parts:
                    totals[k] += parts[k]
                if (i+1) % 50 == 0:
                    print(f'[{args.variant}] epoch={epoch} frame={i+1}/{len(order)} loss={totals["loss"]/(i+1):.5f}', flush=True)
                del fr, batch, bundle, logits, nominal_depth, loss
            row = {'epoch': epoch, 'train': {k: v/len(order) for k, v in totals.items()}}
            improved = False
            if (epoch+1) % args.val_every == 0 or epoch+1 == args.epochs:
                metric, by_case = validate(model, val_paths, val_ds, val_lookup, cache_signature, device, args.query_chunk)
                row.update(val_macro_utility=metric, val_cases=by_case)
                improved = metric > best
                best = max(best, metric)
            history.append(row)
            ck = {'version': VERSION, 'signature': signature, 'protocol': run, 'epoch': epoch,
                  'model': model.learned_state(), 'optimizer': optimizer.state_dict(),
                  'history': history, 'best_metric': best, 'rng_python': random.getstate(),
                  'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                  'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            save_torch(latest, ck)
            if improved:
                best_ck = {k: v for k, v in ck.items() if k not in ('optimizer', 'history') and not k.startswith('rng_')}
                save_torch(out/'checkpoint_best.pt', best_ck); save_json(out/'best.json', row)
            save_json(out/'metrics.json', history)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
