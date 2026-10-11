#!/usr/bin/env python3
"""P0: verify GVAR fixed_rel readiness, then complete paired 2x2 factorial.

This script does not train, infer, or evaluate grasps. It waits *by inspection*
for the previously requested e19 official AP and e20 training completion of
volume_fixed_rel. A pending run is reported as pending, never filled/imputed.

When ready, it reuses analyze_gvar_scene_paired and analyze_gvar_factorial,
including all source manifests, canonical frame fingerprints and scene-cluster
bootstrap. Percentile intervals quantify scene sampling, NOT training seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

from analyze_gvar_factorial import run as factorial_run
from analyze_gvar_scene_paired import SPLITS, AuditError, load_all, run as paired_run

VERSION = 1
CONTROLS = ('baseline', 'slot', 'volume_fixed', 'volume', 'volume_rel')
NEW = 'volume_fixed_rel'
ALL = (*CONTROLS, NEW)
EXPECTED_AP = {
    'baseline': 42.923436,
    'slot': 42.796338,
    'volume_fixed': 44.821,
    'volume': 44.598192,
    'volume_rel': 45.517636,
}
REQUIRED_EPOCHS = 21


def require(flag: bool, msg: str) -> None:
    if not flag:
        raise AuditError(msg)


def expected_paths(root: Path, variant: str, epoch: int) -> list[Path]:
    train = root / 'train' / variant
    paths = [train / 'gvar_protocol.json', train / 'gntrans_mix_protocol.json',
             train / 'gvar_epochs.jsonl', train / f'checkpoint_epoch_{epoch:03d}.tar',
             train / f'checkpoint_epoch_{REQUIRED_EPOCHS-1:03d}.tar']
    for split in SPLITS:
        parent = root / 'eval' / f'{variant}_e{epoch}' / split
        paths += [parent / 'gvar_inference_protocol.json',
                  parent / 'gvar_inference_summary.json',
                  parent / f'ap_{split}_realsense.npy']
    return list(dict.fromkeys(paths))


def audit_training_log(root: Path, variant: str) -> dict[str, Any]:
    path = root / 'train' / variant / 'gvar_epochs.jsonl'
    rows = []
    with path.open('r', encoding='utf-8') as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                x = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise AuditError(f'{path}:{lineno}: invalid JSON: {exc}') from exc
            require(isinstance(x, dict) and 'epoch' in x,
                    f'{path}:{lineno}: missing epoch field')
            rows.append(x)
    ids = [int(x['epoch']) for x in rows]
    require(ids == list(range(REQUIRED_EPOCHS)),
            f'{variant}: expected completed e0..e{REQUIRED_EPOCHS-1}, found {ids}')
    for row in rows:
        for field in ('train_loss', 'val_loss'):
            require(field in row and math.isfinite(float(row[field])),
                    f'{variant}: invalid {field} at epoch {row["epoch"]}')
    return {'variant': variant, 'epoch_count': len(rows),
            'last_epoch': ids[-1], 'train_loss_e20': float(rows[-1]['train_loss']),
            'val_loss_e20': float(rows[-1]['val_loss'])}


def rows_from_csv(path: Path) -> list[dict]:
    require(path.is_file(), f'Missing expected output CSV: {path}')
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _metric(table: list[dict], split: str, name: str, metric='AP') -> dict:
    candidates = [r for r in table if r['split'] == split and r['contrast'] == name
                  and r['metric'] == metric]
    require(len(candidates) == 1,
            f'Expected one factorial row for {split}/{name}/{metric}, found {len(candidates)}')
    return candidates[0]


def _fmt(r: dict) -> str:
    return (f"{float(r['delta_pp']):+.3f} "
            f"[{float(r['ci_low_pp']):+.3f}, {float(r['ci_high_pp']):+.3f}]")


def interpret_factorial(rows: list[dict]) -> dict:
    interaction_mean = _metric(rows, 'Mean', 'interaction')
    interaction_novel = _metric(rows, 'test_novel', 'interaction')
    novel_04 = _metric(rows, 'test_novel', 'interaction', 'AP_mu0.4')
    novel_08 = _metric(rows, 'test_novel', 'interaction', 'AP_mu0.8')
    def tag(r):
        lo, hi = float(r['ci_low_pp']), float(r['ci_high_pp'])
        epsilon = 1e-4  # AP pp: ignore floating-point-only zero-width fluctuations
        return ('positive_scene_interval' if lo > epsilon else
                'negative_scene_interval' if hi < -epsilon else 'inconclusive_scene_interval')
    return {'mean_interaction': interaction_mean, 'novel_interaction': interaction_novel,
            'novel_mu04_interaction': novel_04, 'novel_mu08_interaction': novel_08,
            'mean_interval_direction': tag(interaction_mean),
            'novel_interval_direction': tag(interaction_novel),
            'not_a_multi_seed_inference': True}


def _check_output_target(root: Path, out: Path) -> None:
    root, out = root.resolve(), out.resolve()
    require(out != root, 'Cannot write into GVAR experiment root')
    for directory in ('train', 'eval'):
        require(not out.is_relative_to(root / directory),
                f'Analysis output may not be under {directory}/')
    require(not (out.exists() and any(out.iterdir())),
            f'Analysis output must be absent or empty, refusing overwrite: {out}')


def run(root: Path, output_dir: Path, epoch=19, bootstrap=50000, seed=20261009,
        ci=.95, ap_tolerance=.02, require_ready=False) -> dict:
    require(epoch == 19, 'This P0 decision reproduces the fixed e19 protocol only')
    require(0 < ci < 1 and bootstrap >= 200, 'Invalid CI or bootstrap draws')
    require(root.is_dir(), f'Missing experiment results root: {root}')
    _check_output_target(root, output_dir)
    missing = {v: [str(p) for p in expected_paths(root, v, epoch) if not p.is_file()]
               for v in ALL}
    missing = {v: paths for v, paths in missing.items() if paths}
    if any(v in missing for v in CONTROLS):
        # Old controls must exist: otherwise we cannot distinguish incomplete P0
        # from a wrong root, deleted evidence, or mismatched older experiments.
        raise AuditError(f'Missing historical control evidence: '
                         f'{json.dumps({v: p for v,p in missing.items() if v in CONTROLS}, indent=2)}')
    pending = NEW in missing
    state = {'version': VERSION, 'root': str(root.resolve()), 'epoch': epoch,
             'required_variants': list(ALL), 'new_variant': NEW,
             'missing_files': missing, 'status': 'pending' if pending else 'ready',
             'bootstrap': bootstrap, 'bootstrap_seed': seed, 'scene_ci': ci,
             'warning': 'Scene CIs do not cover training-seed variation or multiple comparisons.'}
    if pending:
        state['next_action'] = ('Wait for fixed_rel e20 training and all three e19 '
                                'Original GraspNet official AP arrays; do NOT retrain automatically.')
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'P0_STATUS.json').write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n')
        (output_dir / 'P0_DECISION.md').write_text(
            '# P0 pending: fixed_rel factorial not available\n\n'
            + 'Missing:\n' + '\n'.join(f'- `{p}`' for p in missing[NEW])
            + '\n\nNo interaction, AP or confidence interval has been invented.\n')
        if require_ready:
            raise AuditError(f'P0 still pending, status saved to {output_dir / "P0_STATUS.json"}')
        return state

    # Validate ALL six variants and their complete per-split AP/provenance before
    # emitting any "success" claim or substituting a missing result.
    records, _ = load_all(root, list(ALL), epoch)
    training = [audit_training_log(root, v) for v in ALL]
    observed = {v: float(sum(records[v]['data'][sp].tensor.mean(dtype='float64')
                             for sp in SPLITS) / len(SPLITS) * 100) for v in ALL}
    for v, expected in EXPECTED_AP.items():
        require(abs(observed[v] - expected) <= ap_tolerance,
                f'Historical {v} Mean AP changed: {observed[v]:.6f} vs {expected:.6f}; '
                'audit checkpoint, decoder and AP protocol before deciding.')

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_out = output_dir / 'paired_six'
    factorial_out = output_dir / 'factorial'
    paired_run(root, paired_out, list(ALL), 'baseline', epoch, bootstrap, seed, ci,
               requested_contrasts=['volume_fixed:volume_rel',
                                    'volume_fixed:volume_fixed_rel',
                                    'volume_fixed_rel:volume_rel',
                                    'volume:volume_rel'])
    factorial_run(root, factorial_out, epoch=epoch, bootstrap=bootstrap, seed=seed, ci=ci)
    factorial_rows = rows_from_csv(factorial_out / 'factorial_effects.csv')
    interpretation = interpret_factorial(factorial_rows)
    table = rows_from_csv(paired_out / 'paired_summary.csv')
    decision = [
        '# GVAR P0 — completed fixed/dynamic × relative-geometry factorial', '',
        f'Source: `{root}`. Single training seed; e{epoch}; 10% Original + 10% GN-Trans.',
        f'Scene-paired bootstrap {bootstrap:,} draws, seed {seed}, {ci:.0%} percentile CIs.', '',
        '## Six models — official original GraspNet AP (%)', '',
        '| Variant | Seen | Similar | Novel | Mean |', '|---|---:|---:|---:|---:|'
    ]
    for v in ALL:
        ap = {r['split']: float(r['variant_pct']) for r in table if r['variant'] == v}
        decision.append(f"| {v} | {ap['test_seen']:.3f} | {ap['test_similar']:.3f} | "
                        f"{ap['test_novel']:.3f} | {ap['Mean']:.3f} |")
    decision += ['', '## Factorial effects: delta AP pp [scene-bootstrap CI]', '',
                 '| Contrast | Seen | Similar | Novel | Mean |', '|---|---|---|---|---|']
    terms = ('fixed_rel_effect', 'dynamic_rel_effect', 'dynamic_effect_no_rel',
             'dynamic_effect_with_rel', 'geometry_mean_effect',
             'dynamic_mean_effect', 'interaction')
    for t in terms:
        decision.append('| ' + ' | '.join((t, *(_fmt(_metric(factorial_rows, s, t))
                          for s in (*SPLITS, 'Mean')))) + ' |')
    decision += ['', '## Interaction under strict and lenient friction', '',
                 f"Novel μ=0.4: {_fmt(_metric(factorial_rows, 'test_novel', 'interaction', 'AP_mu0.4'))} pp.",
                 f"Novel μ=0.8: {_fmt(_metric(factorial_rows, 'test_novel', 'interaction', 'AP_mu0.8'))} pp.", '',
                 f"Mean interaction interval: **{interpretation['mean_interval_direction']}**.",
                 f"Novel interaction interval: **{interpretation['novel_interval_direction']}**.", '',
                 'No zero-crossing is descriptive scene-level evidence, not a multi-seed or '
                 'multiple-testing-adjusted result; a CI overlapping zero is not equivalence.',
                 'Different independently trained models can generate different physical grasps; '
                 'prefix changes cannot be attributed exclusively to ranking.',
                 'Dynamic support should not be declared essential without a supported '
                 'interaction and a practically meaningful Novel/μ=0.4 improvement.', '',
                 'Detailed evidence: `factorial/factorial_effects.csv`, '
                 '`paired_six/paired_summary.csv`, per-scene deltas and input SHA256 audit.']
    (output_dir / 'P0_DECISION.md').write_text('\n'.join(decision) + '\n')
    state.update(status='complete', observed_mean_ap=observed, training_logs=training,
                 interpretation=interpretation, source='existing official e19 AP NPY')
    (output_dir / 'P0_STATUS.json').write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n')
    return state


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=19)
    p.add_argument('--bootstrap', type=int, default=50000)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--ci', type=float, default=.95)
    p.add_argument('--ap-tolerance', type=float, default=.02,
                   help='Maximum allowed AP-point drift of five historical controls')
    p.add_argument('--require-ready', action='store_true',
                   help='Exit nonzero if fixed_rel is not complete (still leave status artifact)')
    a=p.parse_args(argv)
    try:
        result=run(a.root,a.output_dir,a.epoch,a.bootstrap,a.seed,a.ci,
                   a.ap_tolerance,a.require_ready)
    except (AuditError, FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f'GVAR P0 audit FAILED: {exc}\n')
    print(json.dumps({'status':result['status'],'out':str(a.output_dir),
                      'mean_ap':result.get('observed_mean_ap')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
