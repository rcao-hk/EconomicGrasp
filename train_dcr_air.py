#!/usr/bin/env python3
"""Train DCR-AIR with a frozen DCR corrector and independent image evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from dcr_air_common import AIR_VERSION, code_fingerprint, evidence_cdf_loss
from e1e2_common import (VERSION, digest, file_sha, get_batch, load_torch, lock,
                         make_dataset, manifest, save_json, save_torch, seed_all,
                         seed_for, select_centers)
from train_e1e2_cva import cache_paths, cached_bundle, check_input, read_cache


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'stage1-checkpoint', 'dcr-checkpoint',
                 'cache-root', 'output-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--anchor-weight', type=float, default=.01)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--residual-bound', type=float, default=1.0)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--query-chunk', type=int, default=64)
    p.add_argument('--seed', type=int, default=2041)
    p.add_argument('--val-every', type=int, default=1)
    p.add_argument('--no-error-training', action='store_true')
    p.add_argument('--max-train-frames', type=int, default=0)
    p.add_argument('--max-val-frames', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    return p


def grad_norm(module):
    return sum(float(x.grad.norm()) for x in module.parameters()
               if x.grad is not None)


@torch.no_grad()
def validate(model, paths, ds, lookup, cache_signature, device, query_chunk):
    from models.economicgrasp_cva_centers import extract_depth_features
    model.eval()
    rows = []
    for path in paths:
        fr = read_cache(path, cache_signature)
        sid, aid = int(fr['scene_id']), int(fr['anno_id'])
        batch = get_batch(ds, lookup, sid, aid, device)
        pack = extract_depth_features(model.reference, batch)
        check_input(fr, batch, pack[0])
        for ci, case in enumerate(fr['cases']):
            bundle = cached_bundle(fr, ci, device)
            fused, base, residual, _, _, diag = model(
                batch, bundle, case=str(case),
                case_seed=int(fr['case_seeds'][ci]),
                query_chunk=query_chunk, depth_pack=pack)
            target = torch.as_tensor(fr['target'][ci], device=device).float()
            y = target.mean(-1)
            base_u = base.sigmoid().mean(-1)
            air_u = fused.sigmoid().mean(-1)
            base_sel = select_centers(base_u, bundle['valid'], model.zero)
            air_sel = select_centers(air_u, bundle['valid'], model.zero)
            q = torch.arange(y.shape[1], device=device)
            native = y[model.zero]
            dcr_exact = y[base_sel, q]
            air_exact = y[air_sel, q]
            oracle = y.masked_fill(~bundle['valid'], -torch.inf).max(0).values
            rows.append({
                'case': str(case),
                'queries': len(q),
                'air_selected_utility': float(air_exact.mean()),
                'dcr_selected_utility': float(dcr_exact.mean()),
                'air_over_dcr_exact': float((air_exact - dcr_exact).mean()),
                'air_utility_gain': float((air_exact - native).mean()),
                'dcr_utility_gain': float((dcr_exact - native).mean()),
                'air_oracle_regret': float((oracle - air_exact).mean()),
                'dcr_oracle_regret': float((oracle - dcr_exact).mean()),
                'air_move_rate': float((air_sel != model.zero).float().mean()),
                'dcr_move_rate': float((base_sel != model.zero).float().mean()),
                'selection_change_rate': float((air_sel != base_sel).float().mean()),
                'air_abs_residual': float(residual.abs().mean()),
                'visible_ratio': float(diag['visible_ratio'].mean()),
            })
        del batch, pack, fr

    by_case = {}
    fields = [k for k in rows[0] if k not in ('case', 'queries')]
    for case in sorted({r['case'] for r in rows}):
        rr = [r for r in rows if r['case'] == case]
        n = sum(r['queries'] for r in rr)
        stats = {'queries': n, 'frames': len(rr)}
        for key in fields:
            stats[key] = sum(r[key] * r['queries'] for r in rr) / n
        by_case[case] = stats
    metric = float(np.mean([x['air_selected_utility'] for x in by_case.values()]))
    return metric, by_case


def _checkpoint(model, optimizer, run, signature, epoch, best, history):
    return {
        'version': AIR_VERSION,
        'signature': signature,
        'protocol': run,
        'epoch': epoch,
        'model': model.learned_state(),
        'optimizer': optimizer.state_dict(),
        'best_metric': best,
        'history': history,
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    if min(args.epochs, args.grad_accum, args.val_every,
           args.query_chunk, args.hidden) < 1:
        raise ValueError('Epoch/chunk/hidden counts must be positive')
    if args.lr <= 0 or min(args.weight_decay, args.anchor_weight) < 0:
        raise ValueError('Invalid optimizer/loss configuration')
    if not 0 < args.residual_bound <= 4:
        raise ValueError('Invalid residual bound')
    seed_all(args.seed)

    cache_protocol = json.loads((Path(args.cache_root) / 'protocol.json').read_text())
    ref_sha = file_sha(args.stage1_checkpoint)
    if cache_protocol['version'] != VERSION or cache_protocol['reference_sha256'] != ref_sha:
        raise RuntimeError('E1 cache/Stage-1 mismatch')
    cache_signature = digest(cache_protocol)
    train_paths = cache_paths(args.cache_root, cache_protocol, 'train',
                              args.max_train_frames)
    val_paths = cache_paths(args.cache_root, cache_protocol, 'test_seen',
                            args.max_val_frames)
    if not train_paths or not val_paths:
        raise RuntimeError('Empty train/validation schedule')

    dcr_sha = file_sha(args.dcr_checkpoint)
    dcr_ck = load_torch(args.dcr_checkpoint)
    if dcr_ck.get('protocol', {}).get('cache_signature') != cache_signature:
        raise RuntimeError('DCR checkpoint/action cache mismatch')
    if dcr_ck['protocol']['reference_sha256'] != ref_sha:
        raise RuntimeError('DCR checkpoint/Stage-1 mismatch')
    del dcr_ck

    device = torch.device(args.device)
    from models.economicgrasp_cva_air import ActionImageDCR, load_frozen_dcr
    frozen_dcr, dcr_protocol = load_frozen_dcr(
        args.stage1_checkpoint, args.dcr_checkpoint, device)
    if list(map(float, dcr_protocol['offsets_mm'])) != list(map(float, cache_protocol['offsets_mm'])):
        raise RuntimeError('DCR/cache center grid mismatch')
    model = ActionImageDCR(
        frozen_dcr, args.hidden, args.residual_bound, args.seed).to(device)

    params = list(model.reader.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay)

    train_ds, train_lookup = make_dataset(
        args.dataset_root, 'train', cache_protocol['camera'])
    val_ds, val_lookup = make_dataset(
        args.dataset_root, 'test_seen', cache_protocol['camera'])

    config = {k: v for k, v in vars(args).items() if k not in (
        'dataset_root', 'stage1_checkpoint', 'dcr_checkpoint',
        'cache_root', 'output_root', 'device', 'resume', 'epochs')}
    run = {
        'version': AIR_VERSION,
        'code_sha256': code_fingerprint(),
        'cache_signature': cache_signature,
        'reference_sha256': ref_sha,
        'base_dcr_sha256': dcr_sha,
        'base_dcr_variant': dcr_protocol['variant'],
        'config': config,
        'camera': cache_protocol['camera'],
        'offsets_mm': cache_protocol['offsets_mm'],
        'sample_interval': cache_protocol['sample_interval'],
        'train_frames': len(train_paths),
        'val_frames': len(val_paths),
        'checkpoint_selection': 'Seen macro exact utility after AIR center selection',
        'frozen': 'complete DCR corrector/ranker + Stage-1 model',
        'trainable': 'pre-enhancer action-aligned image reader + bounded scalar CDF logit residual',
        'evidence_contract':
            'same physical action projected to image; no active depth/case/CAD input to AIR',
        'score_contract':
            'AIR changes center choice only; official ranking remains frozen Stage-1 score',
    }
    out = Path(args.output_root)
    manifest(out, run)
    signature = digest(run)

    history = []
    start = 0
    best = -float('inf')
    latest = out / 'checkpoint_latest.pt'
    with lock(out / '.train.lock'):
        if latest.exists():
            if not args.resume:
                raise FileExistsError(f'{latest}; use --resume')
            ck = load_torch(latest)
            if ck['signature'] != signature:
                raise RuntimeError('AIR resume contract changed')
            model.load_learned_state(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            start = ck['epoch'] + 1
            best = ck['best_metric']
            history = ck['history']
            random.setstate(ck['rng_python'])
            np.random.set_state(ck['rng_numpy'])
            torch.set_rng_state(ck['rng_torch'])
            if torch.cuda.is_available() and ck['rng_cuda']:
                torch.cuda.set_rng_state_all(ck['rng_cuda'])
            del ck
        else:
            metric, by_case = validate(
                model, val_paths, val_ds, val_lookup,
                cache_signature, device, args.query_chunk)
            # The structural experiment must start as an exact no-op over DCR.
            # Fail before training if the real checkpoint/data path violates that
            # contract; a synthetic unit test alone is not sufficient.
            if any(abs(v['selection_change_rate']) > 0 or
                   abs(v['air_over_dcr_exact']) > 1e-12
                   for v in by_case.values()):
                raise RuntimeError(
                    'Zero-init AIR failed exact DCR selection equivalence')
            best = metric
            initial_row = {
                'epoch': -1,
                'val_macro_utility': metric,
                'val_cases': by_case,
                'note': 'zero-init AIR exactly reproduces frozen DCR center selection',
            }
            initial = _checkpoint(
                model, optimizer, run, signature, -1, best, history)
            save_torch(out / 'checkpoint_initial.pt', initial)
            save_torch(out / 'checkpoint_best.pt', initial)
            save_json(out / 'initial.json', initial_row)
            save_json(out / 'best.json', initial_row)
            print('[AIR INITIAL] ' + json.dumps(initial_row), flush=True)

        for epoch in range(start, args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            order = list(train_paths)
            random.Random(seed_for(args.seed, 'order', epoch)).shuffle(order)
            totals = {'loss': 0., 'cdf_bce': 0., 'air_anchor': 0.,
                      'air_abs_residual': 0., 'frames': 0}
            grad_audit = None

            for i, path in enumerate(order):
                fr = read_cache(path, cache_signature)
                sid, aid = int(fr['scene_id']), int(fr['anno_id'])
                ci = (0 if args.no_error_training else
                      seed_for(args.seed, epoch, sid, aid) % len(fr['cases']))
                batch = get_batch(train_ds, train_lookup, sid, aid, device)
                bundle = cached_bundle(fr, ci, device)
                fused, _, residual, _, depth, _ = model(
                    batch, bundle, case=str(fr['cases'][ci]),
                    case_seed=int(fr['case_seeds'][ci]),
                    query_chunk=args.query_chunk)
                check_input(fr, batch, depth)
                target = torch.as_tensor(
                    fr['target'][ci], device=device).float()
                loss, parts = evidence_cdf_loss(
                    fused, target, bundle['valid'], residual,
                    args.anchor_weight)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite AIR loss')

                group_start = (i // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(order) - group_start)
                (loss / group_size).backward()

                if grad_audit is None:
                    grad_audit = {
                        'reader': grad_norm(model.reader),
                        'frozen_dcr': grad_norm(model.base),
                    }
                    if grad_audit['reader'] <= 0:
                        raise RuntimeError('AIR reader received zero gradient')
                    if grad_audit['frozen_dcr'] != 0:
                        raise RuntimeError('Gradient leaked into frozen DCR')
                    save_json(out / 'gradient_check.json', grad_audit)

                totals['loss'] += float(loss.detach())
                for key in ('cdf_bce', 'air_anchor', 'air_abs_residual'):
                    totals[key] += parts[key]
                totals['frames'] += 1

                step = ((i + 1) % args.grad_accum == 0 or i + 1 == len(order))
                if step:
                    torch.nn.utils.clip_grad_norm_(params, 5.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                del fr, batch, bundle, fused, residual, depth, target, loss

            row = {
                'epoch': epoch,
                **{k: (v / totals['frames'] if k != 'frames' else v)
                   for k, v in totals.items()},
            }
            if (epoch + 1) % args.val_every == 0:
                metric, by_case = validate(
                    model, val_paths, val_ds, val_lookup,
                    cache_signature, device, args.query_chunk)
                row['val_macro_utility'] = metric
                row['val_cases'] = by_case
                if metric > best + 1e-12:
                    best = metric
                    save_torch(
                        out / 'checkpoint_best.pt',
                        _checkpoint(model, optimizer, run, signature,
                                    epoch, best, history + [row]))
                    save_json(out / 'best.json', row)
            history.append(row)
            save_json(out / 'metrics.json', history)
            save_torch(
                latest, _checkpoint(
                    model, optimizer, run, signature, epoch, best, history))
            print('[AIR TRAIN] ' + json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
