#!/usr/bin/env python3
"""Audit DCR-vs-E1 center-selection behavior from saved DCR inference traces.

This script is label-free: it reports *what center each corrector chose*, not
whether a changed action is an exact-action rescue/harm. It expects traces from
DCR-E1-4 (or any two inference_dcr_cva.py runs with the same Stage-1 protocol).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from e1e2_common import atomic_file, save_json


def _csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, restval='')
    writer.writeheader()
    writer.writerows(rows)
    with atomic_file(path) as f:
        f.write(buf.getvalue().encode())


def _load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _scalar_str(x):
    a = np.asarray(x)
    return str(a.item() if a.ndim == 0 else a.reshape(-1)[0])


def _selected_offsets(trace):
    selected = np.asarray(trace['selected'], dtype=np.int64).reshape(-1)
    offsets = np.asarray(trace['offsets_mm'], dtype=np.float64).reshape(-1)
    if selected.size == 0 or selected.min() < 0 or selected.max() >= len(offsets):
        raise RuntimeError('Invalid selected-index trace')
    return offsets[selected], offsets


def _top_indices(score, k):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if not np.isfinite(score).all():
        raise RuntimeError('Non-finite Stage-1 score in trace')
    return np.argsort(-score, kind='stable')[:min(int(k), len(score))]


def _append_group(store, key, e_off, d_off, e_margin, d_margin):
    bucket = store[key]
    bucket['e_off'].append(np.asarray(e_off, np.float64))
    bucket['d_off'].append(np.asarray(d_off, np.float64))
    bucket['e_margin'].append(np.asarray(e_margin, np.float64))
    bucket['d_margin'].append(np.asarray(d_margin, np.float64))


def _cat(parts):
    return np.concatenate(parts) if parts else np.empty((0,), np.float64)


def _offset_stats(x):
    x = np.asarray(x, np.float64)
    if not len(x):
        return {}
    stay = np.isclose(x, 0.)
    return dict(
        stay_rate=float(stay.mean()),
        move_rate=float((~stay).mean()),
        negative_rate=float((x < 0).mean()),
        positive_rate=float((x > 0).mean()),
        mean_offset_mm=float(x.mean()),
        median_offset_mm=float(np.median(x)),
        mean_abs_offset_mm=float(np.abs(x).mean()),
    )


def _transition_stats(e, d):
    e = np.asarray(e, np.float64)
    d = np.asarray(d, np.float64)
    if e.shape != d.shape or not len(e):
        return {}
    es, ds = np.isclose(e, 0.), np.isclose(d, 0.)
    both_move = (~es) & (~ds)
    opposite = both_move & (np.sign(e) != np.sign(d))
    return dict(
        same_selection_rate=float(np.isclose(e, d).mean()),
        e1_stay_to_dcr_move_rate=float((es & ~ds).mean()),
        e1_move_to_dcr_stay_rate=float((~es & ds).mean()),
        both_move_rate=float(both_move.mean()),
        opposite_direction_rate_all=float(opposite.mean()),
        opposite_direction_rate_given_both_move=float(opposite.sum() / max(1, both_move.sum())),
        dcr_minus_e1_offset_mean_mm=float((d - e).mean()),
        dcr_minus_e1_abs_offset_mean_mm=float((np.abs(d) - np.abs(e)).mean()),
    )


def _margin_from_trace(trace):
    u = np.asarray(trace['local_utility'], dtype=np.float64)
    selected = np.asarray(trace['selected'], dtype=np.int64).reshape(-1)
    offsets = np.asarray(trace['offsets_mm'], dtype=np.float64).reshape(-1)
    zero = np.flatnonzero(np.isclose(offsets, 0.))
    if u.ndim != 2 or len(zero) != 1 or u.shape[1] != len(selected):
        raise RuntimeError('Malformed local-utility trace')
    q = np.arange(len(selected))
    return u[selected, q] - u[int(zero[0])]


def _validate_protocol(dcr_root: Path, e1_root: Path):
    dp = json.loads((dcr_root / 'protocol.json').read_text())
    ep = json.loads((e1_root / 'protocol.json').read_text())
    keys = ('reference_sha256', 'sample_interval', 'offsets_mm',
            'query_limit', 'max_frames_per_split', 'cases')
    mismatch = {k: (dp.get(k), ep.get(k)) for k in keys if dp.get(k) != ep.get(k)}
    if mismatch:
        raise RuntimeError(f'DCR/E1 inference protocols are not action-aligned: {mismatch}')
    return dp, ep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dcr-root', required=True,
                   help='DCR inference root, e.g. WORK_ROOT/test/dcr')
    p.add_argument('--e1-root', required=True,
                   help='E1 zero-residual inference root, e.g. WORK_ROOT/test/e1_ref')
    p.add_argument('--output-root', required=True)
    p.add_argument('--splits', default='test_seen,test_similar,test_novel')
    p.add_argument('--cases', default='',
                   help='Optional comma-separated filter; empty uses all traced cases')
    p.add_argument('--topks', default='10,50',
                   help='Per-frame Stage-1 score prefixes in addition to all queries')
    args = p.parse_args()

    dcr_root, e1_root, out = Path(args.dcr_root), Path(args.e1_root), Path(args.output_root)
    dp, _ = _validate_protocol(dcr_root, e1_root)
    splits = {x.strip() for x in args.splits.split(',') if x.strip()}
    cases = {x.strip() for x in args.cases.split(',') if x.strip()}
    topks = sorted({int(x) for x in args.topks.split(',') if x.strip()})
    if any(k < 1 for k in topks):
        raise ValueError('topks must be positive')

    dcr_paths = sorted((dcr_root / 'traces').glob('*/scene_*/ann_*.npz'))
    if not dcr_paths:
        raise FileNotFoundError(f'No traces under {dcr_root / "traces"}')

    groups = defaultdict(lambda: defaultdict(list))
    transition_counts = defaultdict(lambda: defaultdict(int))
    offset_counts = defaultdict(lambda: defaultdict(int))
    frames = defaultdict(int)
    used_files = 0

    for dpath in dcr_paths:
        rel = dpath.relative_to(dcr_root / 'traces')
        split = rel.parts[0]
        if split not in splits:
            continue
        epath = e1_root / 'traces' / rel
        if not epath.is_file():
            raise FileNotFoundError(f'Missing aligned E1 trace: {epath}')
        d, e = _load(dpath), _load(epath)
        case = _scalar_str(d['case'])
        if cases and case not in cases:
            continue
        if _scalar_str(e['case']) != case:
            raise RuntimeError(f'Case mismatch for {rel}')
        for key in ('scene_id', 'anno_id', 'query_ids', 'offsets_mm', 'stage1_score'):
            if not np.array_equal(np.asarray(d[key]), np.asarray(e[key])):
                raise RuntimeError(f'Alignment mismatch for {key}: {rel}')

        d_off, offsets = _selected_offsets(d)
        e_off, e_offsets = _selected_offsets(e)
        if not np.array_equal(offsets, e_offsets):
            raise RuntimeError(f'Offset-grid mismatch: {rel}')
        d_margin, e_margin = _margin_from_trace(d), _margin_from_trace(e)
        score = np.asarray(d['stage1_score'], np.float64).reshape(-1)
        if len(score) != len(d_off):
            raise RuntimeError(f'Score/query mismatch: {rel}')

        subsets = {'all': np.arange(len(score))}
        subsets.update({f'top{k}': _top_indices(score, k) for k in topks})
        for subset, idx in subsets.items():
            key = (split, case, subset)
            _append_group(groups, key, e_off[idx], d_off[idx], e_margin[idx], d_margin[idx])
            frames[key] += 1
            for method, arr in (('e1', e_off[idx]), ('dcr', d_off[idx])):
                for off in offsets:
                    offset_counts[(split, case, subset, method)][float(off)] += int(np.isclose(arr, off).sum())
            for eo, do in zip(e_off[idx], d_off[idx]):
                transition_counts[(split, case, subset)][(float(eo), float(do))] += 1
        used_files += 1

    if not used_files:
        raise RuntimeError('Filters removed every trace')

    summary_rows, hist_rows, trans_rows = [], [], []
    for (split, case, subset), bucket in sorted(groups.items()):
        e_off, d_off = _cat(bucket['e_off']), _cat(bucket['d_off'])
        e_margin, d_margin = _cat(bucket['e_margin']), _cat(bucket['d_margin'])
        row = {
            'split': split, 'case': case, 'subset': subset,
            'frames': frames[(split, case, subset)], 'queries': len(e_off),
        }
        row.update({f'e1_{k}': v for k, v in _offset_stats(e_off).items()})
        row.update({f'dcr_{k}': v for k, v in _offset_stats(d_off).items()})
        row.update(_transition_stats(e_off, d_off))
        row['e1_selected_local_margin_mean'] = float(e_margin.mean())
        row['dcr_selected_local_margin_mean'] = float(d_margin.mean())
        summary_rows.append(row)

        for method in ('e1', 'dcr'):
            counts = offset_counts[(split, case, subset, method)]
            den = sum(counts.values())
            for off in sorted(counts):
                hist_rows.append({
                    'split': split, 'case': case, 'subset': subset, 'method': method,
                    'offset_mm': off, 'count': counts[off], 'rate': counts[off] / max(1, den),
                })
        counts = transition_counts[(split, case, subset)]
        den = sum(counts.values())
        for (eo, do), count in sorted(counts.items()):
            trans_rows.append({
                'split': split, 'case': case, 'subset': subset,
                'e1_offset_mm': eo, 'dcr_offset_mm': do,
                'count': count, 'rate': count / max(1, den),
            })

    out.mkdir(parents=True, exist_ok=True)
    _csv(out / 'selection_audit.csv', summary_rows)
    _csv(out / 'offset_histogram.csv', hist_rows)
    _csv(out / 'transition_matrix.csv', trans_rows)
    save_json(out / 'audit_meta.json', {
        'dcr_root': str(dcr_root),
        'e1_root': str(e1_root),
        'trace_files': used_files,
        'splits': sorted(splits),
        'cases_filter': sorted(cases),
        'topks': topks,
        'protocol': {k: dp.get(k) for k in (
            'reference_sha256', 'sample_interval', 'offsets_mm',
            'query_limit', 'max_frames_per_split', 'cases')},
        'interpretation_boundary':
            'Label-free selection behavior only; no exact-action rescue/harm claims.',
        'outputs': ['selection_audit.csv', 'offset_histogram.csv', 'transition_matrix.csv'],
    })
    print(f'[DCR AUDIT] {used_files} aligned traces -> {out}', flush=True)


if __name__ == '__main__':
    main()
