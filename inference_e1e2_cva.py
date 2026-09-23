#!/usr/bin/env python3
"""Online RGB inference for E1/E2; no training cache, labels or evaluator input."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from e1e2_common import (VERSION, atomic_file, digest, file_sha, get_batch, load_torch, lock,
    make_dataset, manifest, save_json, schedule, seed_all, seed_for, select_centers, shard_frames, case_key)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--stage1-checkpoint', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--split', choices=('test_seen', 'test_similar', 'test_novel'), required=True)
    p.add_argument('--cases', default='nominal,bias:-20,bias:20')
    p.add_argument('--sample-interval', type=float, default=.1)
    p.add_argument('--query-limit', type=int, default=0, help='0=all queries; unrelated to 10 percent frame schedule')
    p.add_argument('--score-source', choices=('model', 'stage1'), default='model')
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    return p


def main():
    args = parser().parse_args(); sys.argv = [sys.argv[0]]
    ck = load_torch(args.checkpoint)
    if ck['version'] != VERSION:
        raise RuntimeError('Checkpoint version mismatch')
    protocol = ck['protocol']
    if protocol['reference_sha256'] != file_sha(args.stage1_checkpoint):
        raise RuntimeError('Wrong immutable Stage-1 checkpoint')
    if args.sample_interval != protocol['sample_interval']:
        raise ValueError('Training/inference frame schedules must match for this experiment')
    cases = list(dict.fromkeys(['nominal'] + args.cases.split(',')))
    config = protocol['config']; seed_all(config['seed'])
    from models.economicgrasp_cva_centers import CenterHypothesisCVA, load_reference, extract_depth_features
    device = torch.device(args.device)
    model = CenterHypothesisCVA(load_reference(args.stage1_checkpoint, device),
                               protocol['offsets_mm'], config['group_chunk']).to(device)
    model.load_learned_state(ck['model']); model.eval(); del ck
    run = {'version': VERSION, 'checkpoint_sha256': file_sha(args.checkpoint),
           'reference_sha256': protocol['reference_sha256'], 'variant': config['variant'],
           'camera': protocol['camera'], 'cases': cases, 'sample_interval': args.sample_interval,
           'query_limit': args.query_limit, 'score_source': args.score_source,
           'max_frames_per_split': args.max_frames, 'query_chunk': config['query_chunk'],
           'offsets_mm': protocol['offsets_mm'], 'seen_role': 'validation_seen'}
    root = Path(args.output_root); manifest(root, run); signature = digest(run)
    frames = shard_frames(schedule(args.split, args.sample_interval), args.shard_id,
                          args.num_shards, args.max_frames)
    ds, lookup = make_dataset(args.dataset_root, args.split, protocol['camera'])
    completed = skipped = 0
    for sid, aid in frames:
        marker = root/'completed'/args.split/f'scene_{sid:04d}'/f'{aid:04d}.json'
        with lock(marker.with_suffix('.lock')):
            if args.resume and marker.exists():
                m = json.loads(marker.read_text())
                if m['signature'] != signature:
                    raise RuntimeError('Inference resume signature mismatch')
                if all((root/p).is_file() for p in m['outputs']):
                    skipped += 1; continue
            batch = get_batch(ds, lookup, sid, aid, device)
            outputs = []
            with torch.no_grad():
                pack = extract_depth_features(model.reference, batch)
                for case in cases:
                    logits, bundle, _ = model(batch, case=case,
                        case_seed=seed_for(2030, sid, aid, case), query_limit=args.query_limit,
                        query_chunk=config['query_chunk'], depth_pack=pack)
                    utility = logits.sigmoid().mean(-1)
                    selected = select_centers(utility, bundle['valid'], model.zero)
                    q = torch.arange(utility.shape[1], device=device)
                    native = bundle['native'].clone()
                    chosen = bundle['actions'][selected, q].clone()
                    chosen[:, 0] = (utility[selected, q] if args.score_source == 'model' else native[:, 0])
                    for method, value in [('model', chosen), ('native', native)]:
                        path = Path('dump')/method/case_key(case)/f'scene_{sid:04d}'/protocol['camera']/f'{aid:04d}.npy'
                        with atomic_file(root/path) as f:
                            np.save(f, value.cpu().numpy(), allow_pickle=False)
                        outputs.append(str(path))
            save_json(marker, {'signature': signature, 'outputs': outputs, 'scene_id': sid, 'anno_id': aid})
            del pack, batch, logits, bundle
            completed += 1
            print(f'[{config["variant"]} INFER] {args.split} {sid}/{aid} new={completed} skip={skipped}', flush=True)
    save_json(root/f'infer_{args.split}_shard{args.shard_id}.json',
              {'signature': signature, 'new': completed, 'skipped': skipped, 'scheduled': len(frames)})


if __name__ == '__main__':
    main()
