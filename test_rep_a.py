#!/usr/bin/env python3
"""Fixed-action depth stress test. No labels, candidates or margins are retuned.

Reports nominal/corrupted performance AND probability/representation drift and
within-ray discrimination. This is NOT official GraspNet AP or end-to-end pose
recovery: physical actions remain the exact same Rep-P0 candidates in every case.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from rep_a_common import (check_runtime_sources, DEFAULT_CASES, SPLITS, aggregate, array_sha, digest,
    exclusive_run, file_sha, list_frames, load_torch, metrics, parse_case,
    perturb_depth, read_frame, save_json, seed_for, tensors, atomic_file)
from rep_a_model import RepAReaderScorer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--split', choices=SPLITS, default='test_similar')
    p.add_argument('--cases', default=DEFAULT_CASES)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--save-per-query', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--progress-every', type=int, default=50)
    return p


@torch.no_grad()
def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    cases = list(dict.fromkeys(['nominal']+[x.strip() for x in args.cases.split(',') if x.strip()]))
    for c in cases:
        parse_case(c)
    check_runtime_sources(args.cache_root)
    device = torch.device(args.device)
    paths = list_frames(args.cache_root, args.split, args.max_frames)
    signature = digest({'checkpoint': file_sha(args.checkpoint), 'split': args.split,
        'cases': cases, 'seed': args.seed, 'save_per_query': args.save_per_query,
        'frames': [(p.name, p.parent.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out/'.test.lock'):
        if args.resume and (out/'summary.json').exists():
            saved = json.loads((out/'summary.json').read_text())
            if saved['signature'] != signature:
                raise RuntimeError('Test signature changed; use another output-dir')
            print(f'[REP-A TEST] already complete: {out}', flush=True)
            return
        ck = load_torch(args.checkpoint)
        init = load_torch(Path(args.cache_root)/'reader_init.pt')
        if ck['contract'] != init['contract']:
            raise RuntimeError('Checkpoint and cache come from different Stage-1/export contracts')
        if ck['model_config'] != init['model_config']:
            raise RuntimeError('Model configuration mismatch')
        model = RepAReaderScorer(init, ck['variant']).to(device)
        model.load_state_dict(ck['model'], strict=True); model.eval()
        margin, variant, epoch, contract = ck['margin'], ck['variant'], ck['epoch'], ck['contract']
        del ck, init
        rows = []
        qfile = gzip.open(out/'per_query.csv.gz.tmp', 'wt', newline='') if args.save_per_query else None
        qw = None
        try:
            for i, path in enumerate(paths):
                d = read_frame(path, contract)
                t = tensors(d, device)
                before_sha = array_sha(d['actions'], d['valid'], d['friction'], d['query_ids'])
                logits0, rep0 = model(t, return_repr=True)
                p0 = logits0.sigmoid().cpu().numpy()
                _, k0 = metrics(p0, d, margin)
                for case in cases:
                    seed = seed_for(args.seed, args.split, int(d['scene_id']), int(d['anno_id']), parse_case(case)[0])
                    depth, err = perturb_depth(t['depth'], case, seed)
                    logits, rep = (logits0, rep0) if case == 'nominal' else model(t, depth, return_repr=True)
                    prob = logits.sigmoid().cpu().numpy()
                    met, selected = metrics(prob, d, margin)
                    valid = d['valid'].astype(bool)
                    cosine = (1-F.cosine_similarity(rep.float(), rep0.float(), dim=-1)).reshape(valid.shape).cpu().numpy()
                    relative = ((rep-rep0).norm(dim=-1)/rep0.norm(dim=-1).clamp_min(1e-6)).reshape(valid.shape).cpu().numpy()
                    # Separate high-score population without assuming saved query order.
                    ids = np.argsort(-d['native_score'], kind='stable')[:max(1, len(d['native_score'])//2)]
                    small = {key: d[key][:, ids] for key in ('valid', 'friction', 'utility')}
                    small['zero_index'] = d['zero_index']
                    top, _ = metrics(prob[:, ids], small, margin)
                    row = {'variant': variant, 'split': args.split, 'scene_id': int(d['scene_id']),
                        'anno_id': int(d['anno_id']), 'action_sha': before_sha, 'case': case,
                        'margin': margin, **met, **err,
                        'probability_drift': float(np.abs(prob-p0)[valid].mean()),
                        'representation_cosine_distance': float(cosine[valid].mean()),
                        'representation_relative_l2': float(relative[valid].mean()),
                        'selection_turnover': float((selected != k0).mean()),
                        'top_half_utility_gain': top['utility_gain'],
                        'top_half_success08_gain': top['success08_gain']}
                    rows.append(row)
                    if qfile:
                        for q in range(len(selected)):
                            k = int(selected[q]); z = int(d['zero_index'])
                            rec = {'variant': variant, 'split': args.split, 'scene_id': int(d['scene_id']),
                                'anno_id': int(d['anno_id']), 'query_id': int(d['query_ids'][q]),
                                'case': case, 'selected_k': k, 'selected_offset_mm': float(d['offsets_mm'][k]),
                                'native_score': float(d['native_score'][q]),
                                'native_utility': float(d['utility'][z,q]),
                                'selected_utility': float(d['utility'][k,q]),
                                'oracle_utility': float(np.where(valid[:,q],d['utility'][:,q],-np.inf).max())}
                            if qw is None:
                                qw = csv.DictWriter(qfile, fieldnames=list(rec)); qw.writeheader()
                            qw.writerow(rec)
                if before_sha != array_sha(d['actions'], d['valid'], d['friction'], d['query_ids']):
                    raise RuntimeError('A reader or perturbation mutated actions/labels')
                del d, t, logits0, rep0, logits, rep, prob, depth
                if (i+1) % max(1,args.progress_every) == 0:
                    print(f'[REP-A TEST {variant}] {i+1}/{len(paths)} frames x {len(cases)} cases', flush=True)
        finally:
            if qfile:
                qfile.close()
        if qfile:
            (out/'per_query.csv.gz.tmp').replace(out/'per_query.csv.gz')
        with atomic_file(out/'per_frame.csv') as f:
            import io
            text = io.StringIO(); w = csv.DictWriter(text, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows); f.write(text.getvalue().encode())
        case_results = {}
        extra = ('probability_drift', 'representation_cosine_distance', 'representation_relative_l2',
                 'selection_turnover', 'depth_rms_mm', 'depth_bias_mm', 'clamped_fraction',
                 'top_half_utility_gain', 'top_half_success08_gain')
        for case in cases:
            r = [x for x in rows if x['case'] == case]
            case_results[case] = aggregate(r)
            for key in extra:
                case_results[case][key] = sum(x[key]*x['num_queries'] for x in r)/sum(x['num_queries'] for x in r)
        clean = case_results['nominal']
        for r in case_results.values():
            r['utility_drop_from_nominal'] = clean['selected_utility']-r['selected_utility']
            r['success08_drop_from_nominal'] = clean['success08']-r['success08']
        result = {'signature': signature, 'variant': variant, 'split': args.split,
            'checkpoint': str(Path(args.checkpoint).resolve()), 'epoch': epoch,
            'validation_selected_margin': margin, 'cache_contract': contract, 'seed': args.seed,
            'fixed_actions_unchanged': True, 'not_official_AP': True, 'cases': case_results}
        save_json(out/'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
