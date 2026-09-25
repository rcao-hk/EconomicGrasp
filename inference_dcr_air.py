#!/usr/bin/env python3
"""Infer DCR-AIR and dump native/DCR/AIR actions under frozen Stage-1 ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from dcr_air_common import AIR_VERSION, make_air_outputs
from e1e2_common import (atomic_file, case_key, digest, file_sha, get_batch,
                         load_torch, lock, make_dataset, manifest, save_json,
                         save_npz, schedule, seed_all, seed_for, shard_frames)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'stage1-checkpoint', 'dcr-checkpoint',
                 'checkpoint', 'output-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--split',
                   choices=('test_seen', 'test_similar', 'test_novel'),
                   required=True)
    p.add_argument('--cases',
                   default='nominal,bias:-15,bias:15,bias:-25,bias:25,'
                           'scale:-0.03,scale:0.03,smooth:5,smooth:10')
    p.add_argument('--sample-interval', type=float, default=.1)
    p.add_argument('--query-limit', type=int, default=0)
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    return p


def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    if args.query_limit < 0 or args.max_frames < 0:
        raise ValueError('Invalid query/frame count')

    ck = load_torch(args.checkpoint)
    if ck.get('version') != AIR_VERSION:
        raise RuntimeError('Unsupported AIR checkpoint')
    protocol = ck['protocol']
    cfg = protocol['config']
    if protocol['reference_sha256'] != file_sha(args.stage1_checkpoint):
        raise RuntimeError('Wrong Stage-1 checkpoint')
    if protocol['base_dcr_sha256'] != file_sha(args.dcr_checkpoint):
        raise RuntimeError('Wrong base DCR checkpoint')
    if args.sample_interval != protocol['sample_interval']:
        raise RuntimeError('Training/inference frame schedule mismatch')

    cases = list(dict.fromkeys(
        ['nominal'] + [x.strip() for x in args.cases.split(',') if x.strip()]))
    device = torch.device(args.device)
    seed_all(cfg['seed'])

    from models.economicgrasp_cva_air import ActionImageDCR, load_frozen_dcr
    base, dcr_protocol = load_frozen_dcr(
        args.stage1_checkpoint, args.dcr_checkpoint, device)
    if list(map(float, dcr_protocol['offsets_mm'])) != list(map(float, protocol['offsets_mm'])):
        raise RuntimeError('AIR/base DCR offset-grid mismatch')
    model = ActionImageDCR(
        base, cfg['hidden'], cfg['residual_bound'], cfg['seed']).to(device)
    model.load_learned_state(ck['model'])
    model.eval()

    run = {
        'version': AIR_VERSION,
        'variant': 'DCR_AIR',
        'camera': protocol['camera'],
        'checkpoint_sha256': file_sha(args.checkpoint),
        'base_dcr_sha256': protocol['base_dcr_sha256'],
        'reference_sha256': protocol['reference_sha256'],
        'sample_interval': args.sample_interval,
        'cases': cases,
        'offsets_mm': protocol['offsets_mm'],
        'query_limit': args.query_limit,
        'max_frames_per_split': args.max_frames,
        'query_chunk': cfg['query_chunk'],
        'methods': ['native', 'dcr_stage1', 'air_stage1'],
        'score_contract': 'all methods use frozen Stage-1 query score',
        'code_sha256': protocol['code_sha256'],
    }
    del ck

    root = Path(args.output_root)
    manifest(root, run)
    signature = digest(run)
    frames = shard_frames(
        schedule(args.split, args.sample_interval),
        args.shard_id, args.num_shards, args.max_frames)
    if not frames:
        print('[AIR INFER] no frames assigned', flush=True)
        return

    ds, lookup = make_dataset(
        args.dataset_root, args.split, protocol['camera'])
    from models.economicgrasp_cva_centers import extract_depth_features

    completed = skipped = 0
    for sid, aid in frames:
        marker = (root / 'completed' / args.split /
                  f'scene_{sid:04d}' / f'{aid:04d}.json')
        with lock(marker.with_suffix('.lock')):
            if args.resume and marker.exists():
                old = json.loads(marker.read_text())
                if old['signature'] != signature:
                    raise RuntimeError('Changed AIR inference contract')
                if all((root / p).is_file() for p in old['outputs']):
                    skipped += 1
                    continue
            elif marker.exists():
                raise FileExistsError(f'{marker}; use --resume')

            batch = get_batch(ds, lookup, sid, aid, device)
            paths = []
            with torch.no_grad():
                pack = extract_depth_features(model.reference, batch)
                for case in cases:
                    fused, base_logits, residual, bundle, _, diag = model(
                        batch, case=case,
                        case_seed=seed_for(2030, sid, aid, case),
                        query_limit=args.query_limit,
                        query_chunk=cfg['query_chunk'],
                        depth_pack=pack)
                    outputs, base_sel, air_sel = make_air_outputs(
                        base_logits, fused, bundle, model.zero)
                    for mode, array in outputs.items():
                        rel = (Path('dump') / mode / case_key(case) /
                               f'scene_{sid:04d}' / protocol['camera'] /
                               f'{aid:04d}.npy')
                        with atomic_file(root / rel) as f:
                            np.save(f, array.cpu().numpy(), allow_pickle=False)
                        paths.append(str(rel))

                    trace = (Path('traces') / args.split /
                             f'scene_{sid:04d}' /
                             f'ann_{aid:04d}_{case_key(case)}.npz')
                    save_npz(
                        root / trace,
                        signature=np.array(signature),
                        scene_id=np.array(sid),
                        anno_id=np.array(aid),
                        case=np.array(case),
                        query_ids=bundle['query_ids'].cpu().numpy(),
                        offsets_mm=model.base.corrector.offsets_mm.cpu().numpy(),
                        base_selected=base_sel.cpu().numpy(),
                        air_selected=air_sel.cpu().numpy(),
                        base_local_utility=base_logits.sigmoid().mean(-1).cpu().numpy(),
                        air_local_utility=fused.sigmoid().mean(-1).cpu().numpy(),
                        air_logit_residual=residual.cpu().numpy(),
                        visible_ratio=diag['visible_ratio'].cpu().numpy(),
                        attention_max=diag['attention_max'].cpu().numpy(),
                        stage1_score=outputs['native'][:, 0].cpu().numpy(),
                    )
                    paths.append(str(trace))
            save_json(marker, {
                'signature': signature,
                'scene_id': sid,
                'anno_id': aid,
                'outputs': paths,
            })
            completed += 1
            del batch, pack, fused, base_logits, residual, bundle, outputs
            print(
                f'[AIR INFER] {args.split} {sid}/{aid} '
                f'new={completed} skip={skipped}', flush=True)

    save_json(root / f'infer_{args.split}_shard{args.shard_id}.json', {
        'signature': signature,
        'new': completed,
        'skipped': skipped,
        'scheduled': len(frames),
    })


if __name__ == '__main__':
    main()
