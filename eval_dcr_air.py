#!/usr/bin/env python3
"""GraspNet API evaluation for DCR-AIR on the exact sampled frame schedule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from dcr_air_common import AIR_METHODS, AIR_VERSION
from e1e2_common import (atomic_file, case_key, digest, file_sha, lock,
                         save_json, schedule)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--inference-root', required=True)
    p.add_argument('--split',
                   choices=('test_seen', 'test_similar', 'test_novel'),
                   required=True)
    p.add_argument('--methods', default='native,dcr_stage1,air_stage1')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    sys.argv = [sys.argv[0]]

    root = Path(args.inference_root)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['version'] != AIR_VERSION:
        raise RuntimeError('Not a DCR-AIR inference protocol')
    if protocol['query_limit'] != 0 or protocol['max_frames_per_split'] != 0:
        raise RuntimeError(
            'Official comparison requires full-query inference and complete schedule')
    if args.workers < 1:
        raise ValueError('workers must be positive')

    from graspnetAPI import GraspNetEval
    ge = GraspNetEval(
        args.dataset_root, protocol['camera'], split=args.split)
    evaluate = getattr(ge, {
        'test_seen': 'eval_seen',
        'test_similar': 'eval_similar',
        'test_novel': 'eval_novel',
    }[args.split])
    frames = schedule(args.split, protocol['sample_interval'])

    for method in [x.strip() for x in args.methods.split(',') if x.strip()]:
        if method not in AIR_METHODS:
            raise ValueError(method)
        for case in protocol['cases']:
            dump = root / 'dump' / method / case_key(case)
            paths = [
                dump / f'scene_{sid:04d}' / protocol['camera'] /
                f'{aid:04d}.npy'
                for sid, aid in frames
            ]
            if any(not path.is_file() for path in paths):
                raise FileNotFoundError(
                    f'Incomplete {args.split}/{method}/{case}; finish inference first')
            sig = digest({
                'protocol': protocol,
                'split': args.split,
                'method': method,
                'case': case,
                'dump_sha256': [file_sha(path) for path in paths],
            })
            out = root / 'official' / method / case_key(case) / args.split
            with lock(out / '.run.lock'):
                summary = out / 'summary.json'
                if args.resume and summary.is_file():
                    if json.loads(summary.read_text())['signature'] != sig:
                        raise RuntimeError('Official result signature mismatch')
                    continue
                result, ap = evaluate(
                    str(dump),
                    anno_sample_ratio=protocol['sample_interval'],
                    proc=args.workers)
                accuracy = np.asarray(result)
                expected_frames = len(
                    range(0, 256, round(1 / protocol['sample_interval'])))
                if accuracy.shape[:2] != (30, expected_frames):
                    raise RuntimeError(
                        f'Unexpected GraspNet API schedule: {accuracy.shape}')
                with atomic_file(out / 'accuracy.npy') as f:
                    np.save(f, accuracy, allow_pickle=False)
                save_json(summary, {
                    'signature': sig,
                    'case': case,
                    'split': args.split,
                    'method': method,
                    'variant': protocol['variant'],
                    'reported_ap': np.asarray(ap).tolist(),
                    'mean_accuracy': float(accuracy.mean()),
                    'sample_interval': protocol['sample_interval'],
                    'shape': list(accuracy.shape),
                    'seen_role':
                        'validation_seen' if args.split == 'test_seen'
                        else 'held_out',
                })
                print(
                    f'[AIR AP] {method} {case} {args.split}: {ap}',
                    flush=True)


if __name__ == '__main__':
    main()
