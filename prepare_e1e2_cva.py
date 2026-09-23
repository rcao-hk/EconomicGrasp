#!/usr/bin/env python3
"""Offline exact labels for E1/E2. Never cache trainable image representations."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch
from e1e2_common import (VERSION, MAIN_SHA, array_sha, batch_fingerprint, cdf_targets,
    digest, file_sha, get_batch, lock, make_dataset, manifest, parse_offsets,
    random_case, require_ram, save_json, save_npz, schedule, seed_all, seed_for, shard_frames)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--stage1-checkpoint', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--split', choices=('train', 'test_seen'), required=True)
    p.add_argument('--camera', default='realsense')
    p.add_argument('--sample-interval', type=float, default=.1)
    p.add_argument('--query-limit', type=int, default=64)
    p.add_argument('--offsets-mm', default='-40,-20,-10,0,10,20,40')
    p.add_argument('--val-cases', default='nominal,bias:-20,bias:20')
    p.add_argument('--seed', type=int, default=2030)
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--voxel-size', type=float, default=.005)
    p.add_argument('--eval-chunk', type=int, default=128)
    p.add_argument('--fc-mode', choices=('reuse_contacts', 'official'), default='reuse_contacts')
    p.add_argument('--verify-n', type=int, default=8)
    p.add_argument('--min-host-free-gib', type=float, default=6.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--repair-corrupt', action='store_true')
    return p


def exact_evaluator(args):
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    from graspnetAPI.utils.eval_utils import voxel_sample_points, create_table_points
    class E1E2Evaluator(ExactGraspNetActionEvaluator):
        def _scene_models(self, scene_id):
            if scene_id not in self.scene_cache:
                models, dex, _ = self.eval.get_scene_models(int(scene_id), ann_id=0)
                self.scene_cache[scene_id] = ([voxel_sample_points(m, args.voxel_size) for m in models], dex)
            return self.scene_cache[scene_id]
    result = E1E2Evaluator(args.dataset_root, args.camera, split=args.split,
                          collision_chunk=args.eval_chunk, fc_mode=args.fc_mode,
                          verify_n=args.verify_n, strict=True)
    result.table = create_table_points(1., 1., .05, dx=-.5, dy=-.5, dz=-.05,
                                      grid_size=args.voxel_size)
    return result


def main():
    args = parser().parse_args(); sys.argv = [sys.argv[0]]
    if args.query_limit < 1 or args.eval_chunk < 1 or args.voxel_size <= 0:
        raise ValueError('Invalid query/geometry/chunk configuration')
    seed_all(args.seed)
    offsets = parse_offsets(args.offsets_mm)
    val_cases = list(dict.fromkeys(['nominal'] + args.val_cases.split(',')))
    protocol = {
        'version': VERSION, 'main_base': MAIN_SHA, 'reference_sha256': file_sha(args.stage1_checkpoint),
        'camera': args.camera, 'sample_interval': args.sample_interval,
        'query_limit': args.query_limit, 'query_sampling': 'uniform_decode_order',
        'offsets_mm': offsets.tolist(), 'val_cases': val_cases, 'seed': args.seed,
        'max_frames_per_split': args.max_frames, 'voxel_size': args.voxel_size,
        'fc_mode': args.fc_mode, 'verify_n': args.verify_n,
        'training_cases': 'nominal plus one deterministic bias/scale/smooth case per frame',
        'action_contract': 'fixed reference; same R/w/insertion depth, camera-z center offsets only',
        'image_features_cached': False, 'pose_depth_mode': 'global_film',
    }
    out = Path(args.output_root); manifest(out, protocol); signature = digest(protocol)
    frames = shard_frames(schedule(args.split, args.sample_interval), args.shard_id,
                          args.num_shards, args.max_frames)
    if not frames:
        print('[E1E2 CACHE] no frames assigned to this shard', flush=True)
        return
    from models.economicgrasp_cva_centers import load_reference, extract_depth_features, reference_candidates
    device = torch.device(args.device)
    reference = load_reference(args.stage1_checkpoint, device)
    ds, lookup = make_dataset(args.dataset_root, args.split, args.camera)
    evaluator = exact_evaluator(args)
    current_scene = None; done = skipped = 0
    for sid, aid in frames:
        path = out/args.split/f'scene_{sid:04d}'/f'ann_{aid:04d}.npz'
        with lock(path.with_suffix('.lock')):
            if path.exists():
                if not args.resume:
                    raise FileExistsError(f'{path}; use --resume')
                try:
                    with np.load(path, allow_pickle=False) as z:
                        if str(z['signature']) != signature:
                            raise RuntimeError(f'Cache protocol mismatch: {path}')
                        if str(z['action_sha']) != array_sha(z['actions'][..., 1:16]):
                            raise ValueError(f'Corrupt action array: {path}')
                        _ = z['target'].shape
                    skipped += 1; continue
                except (OSError, ValueError, EOFError):
                    if not args.repair_corrupt:
                        raise
            if current_scene != sid:
                evaluator.scene_cache.clear(); gc.collect(); current_scene = sid
            require_ram(args.min_host_free_gib)
            batch = get_batch(ds, lookup, sid, aid, device)
            pack = extract_depth_features(reference, batch)
            cs_seed = seed_for(args.seed, sid, aid)
            cases = ['nominal', random_case(cs_seed)] if args.split == 'train' else val_cases
            seeds = [seed_for(args.seed, sid, aid, c) for c in cases]
            saved = {k: [] for k in ('actions', 'valid', 'target', 'friction', 'token_ids',
                                     'view_xyz', 'angle_ids', 'depth_ids', 'query_ids')}
            for case, cseed in zip(cases, seeds):
                bundle, active = reference_candidates(reference, batch, pack, case, cseed, offsets, args.query_limit)
                if done == 0 and case == 'nominal':
                    from models.economicgrasp_bip3d import pred_decode_center_view_angle
                    with torch.no_grad():
                        direct = pred_decode_center_view_angle(reference(dict(batch)), use_cdf=True)[0]
                    keep = (torch.arange(len(direct), device=device) if args.query_limit >= len(direct)
                            else torch.linspace(0, len(direct)-1, args.query_limit, device=device).round().long())
                    if not torch.allclose(direct[keep], bundle['native'], atol=1e-5, rtol=0):
                        raise RuntimeError('Nominal cached-depth replay does not reproduce main Stage-1')
                    del direct
                a = bundle['actions'].cpu().numpy(); valid = bundle['valid'].cpu().numpy()
                ids = np.flatnonzero(valid.reshape(-1))
                result = evaluator.evaluate(sid, aid, a.reshape(-1, 17)[ids])
                fr = np.full(valid.size, -1., np.float32); fr[ids] = result.friction; fr = fr.reshape(valid.shape)
                saved['target'].append(cdf_targets(fr)); saved['friction'].append(fr)
                for k in ('actions', 'valid', 'token_ids', 'view_xyz', 'angle_ids', 'depth_ids', 'query_ids'):
                    saved[k].append(bundle[k].detach().cpu().numpy())
            arrays = {k: np.stack(v) for k, v in saved.items()}
            save_npz(path, signature=np.array(signature), scene_id=np.array(sid), anno_id=np.array(aid),
                     cases=np.asarray(cases), case_seeds=np.asarray(seeds, np.int64),
                     nominal_depth=pack[0][0].cpu().numpy(), K=batch['K'][0].cpu().numpy(),
                     input_sha=np.array(batch_fingerprint(batch)),
                     action_sha=np.array(array_sha(arrays['actions'][..., 1:16])), **arrays)
            done += 1
            print(f'[E1E2 CACHE] {args.split} {sid}/{aid} cases={len(cases)} new={done} skip={skipped}', flush=True)
            del batch, pack, bundle, active, arrays, saved, result
    evaluator.scene_cache.clear()
    save_json(out/f'prepare_{args.split}_shard{args.shard_id}.json',
              {'signature': signature, 'new_frames': done, 'skipped_frames': skipped,
               'scheduled': len(frames), 'shard_id': args.shard_id, 'num_shards': args.num_shards})


if __name__ == '__main__':
    main()
