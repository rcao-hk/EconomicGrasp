"""Selection utility on identical action banks; paired scene-cluster statistics."""
from __future__ import annotations

import csv
from collections import defaultdict, Counter
import json
from pathlib import Path

import numpy as np

from .contracts import MUS, VERSION, physical_id, file_sha256, write_json

PHYSICAL = ['width_m', 'height_m', 'insertion_depth_m'] + [f'r{i}{j}' for i in range(3) for j in range(3)] + ['center_x', 'center_y', 'center_z']
CONTEXT = ['bank_id', 'geometry_sha256', 'source_checkpoint_sha256']
FAILURES = ('empty_grasp', 'collision', 'force_closure_failure', 'success')


def load_rows(paths):
    rows, seen, physical, runtime_ids = [], set(), {}, {}
    for path in paths:
        with open(path, newline='', encoding='utf-8-sig') as f:
            rd = csv.DictReader(f)
            required = set(PHYSICAL + CONTEXT + ['version', 'sample_id', 'scene_id', 'anno_id', 'split',
                'reference', 'model', 'condition', 'checkpoint_sha256', 'family_id', 'depth_id',
                'candidate_id', 'quality', 'score', 'failure', 'friction'] + [f'success_mu_{m:.1f}' for m in MUS])
            if not required <= set(rd.fieldnames or []):
                raise ValueError(f'{path}: missing columns {sorted(required - set(rd.fieldnames or []))}')
            for raw in rd:
                if raw['version'] != VERSION:
                    raise ValueError('Unrecognized candidate table contract')
                r = dict(raw)
                for k in PHYSICAL + ['quality', 'score', 'friction']:
                    r[k] = float(r[k])
                    if not np.isfinite(r[k]): raise ValueError(f'Nonfinite {k}')
                for k in ('scene_id', 'anno_id', 'depth_id'):
                    r[k] = int(r[k])
                expected_sample = f'scene{r["scene_id"]:04d}_frame{r["anno_id"]:04d}'
                inferred_split = ('seen' if 100 <= r['scene_id'] < 130 else 'similar'
                                  if 130 <= r['scene_id'] < 160 else 'novel' if 160 <= r['scene_id'] < 190 else None)
                if r['sample_id'] != expected_sample or r['split'] != inferred_split or not 0 <= r['anno_id'] < 256:
                    raise ValueError('Scene/frame/split identity mismatch')
                if not 0 <= r['score'] <= 1:
                    raise ValueError('Raw score must be a six-threshold sigmoid mean in [0,1]')
                r['bits'] = np.array([int(r[f'success_mu_{m:.1f}']) for m in MUS])
                expected_bits = (r['friction'] > 0) & (r['friction'] <= MUS + 1e-7)
                if not np.array_equal(r['bits'], expected_bits):
                    raise ValueError('Friction and explicit six-mu labels disagree')
                if not np.isin(r['bits'], [0, 1]).all() or np.any(np.diff(r['bits']) < 0):
                    raise ValueError('Invalid six-threshold success vector')
                if not np.isclose(r['bits'].mean(), r['quality'], atol=1e-8, rtol=0):
                    raise ValueError('Quality must equal exact six-mu success fraction')
                if r['failure'] not in FAILURES or (r['failure'] == 'success') != (r['quality'] > 0):
                    raise ValueError('Contradictory exact-action failure labels')
                d = r['depth_id']
                if d not in range(4) or not np.isclose(r['insertion_depth_m'], .01 * (d + 1), atol=1e-7):
                    raise ValueError('Depth ID differs from physical insertion depth')
                grasp = np.array([r['score'], *[r[k] for k in PHYSICAL], -1], dtype=np.float32)
                if physical_id(grasp) != r['candidate_id']:
                    raise ValueError('Candidate ID does not identify the saved physical action')
                key = (r['reference'], r['sample_id'], r['family_id'], d)
                uid = (*key, r['model'], r['condition'])
                if uid in seen: raise ValueError(f'Duplicate score row across inputs: {uid}')
                seen.add(uid)
                signature = tuple(r[k] for k in CONTEXT + PHYSICAL) + (r['candidate_id'], r['quality'], r['failure'], *r['bits'])
                if key in physical and physical[key] != signature:
                    raise ValueError(f'Physical actions/labels/geometry changed between treatments: {key}')
                physical[key] = signature
                model = r['model']
                if model in runtime_ids and runtime_ids[model] != r['checkpoint_sha256']:
                    raise ValueError('A scorer model cannot silently mix checkpoint identities')
                runtime_ids[model] = r['checkpoint_sha256']
                rows.append(r)
    if not rows: raise ValueError('Empty candidate input')
    # Validate four-bin complete families and invariant center/rotation/height.
    for key, family in group_rows(rows).items():
        if sorted(family) != list(range(4)):
            raise ValueError(f'Incomplete four-depth family: {key}')
        for d in range(1, 4):
            for k in PHYSICAL[1:2] + PHYSICAL[3:]:
                if family[d][k] != family[0][k]:
                    raise ValueError(f'Insertion family changed {k}: {key}')
    return rows


def group_rows(rows):
    groups = defaultdict(dict)
    for r in rows:
        key = (r['reference'], r['model'], r['condition'], r['sample_id'], r['family_id'])
        if r['depth_id'] in groups[key]: raise ValueError('Repeated depth in one treatment')
        groups[key][r['depth_id']] = r
    return groups


def pairwise_agreement(score, quality):
    correct = comparable = 0
    for i in range(len(score)):
        for j in range(i + 1, len(score)):
            dq, ds = quality[i] - quality[j], score[i] - score[j]
            if abs(dq) <= 1e-8: continue
            comparable += 1
            correct += .5 if abs(ds) <= 1e-9 else float(dq * ds > 0)
    return correct / comparable if comparable else None, comparable


def family_statistics(family):
    r = family[0]
    score = np.array([family[d]['score'] for d in range(4)])
    quality = np.array([family[d]['quality'] for d in range(4)])
    order = np.argsort(-score, kind='stable')  # score tie -> smallest physical depth ID
    chosen, second = int(order[0]), int(order[1])
    agreement, comparable = pairwise_agreement(score, quality)
    result = dict(reference=r['reference'], model=r['model'], condition=r['condition'],
                  sample=r['sample_id'], scene=r['scene_id'], split=r['split'], family=r['family_id'],
                  selected_depth=chosen, selected_id=family[chosen]['candidate_id'],
                  selected_quality=quality[chosen], oracle_quality=quality.max(),
                  regret=quality.max() - quality[chosen], pairwise_agreement=agreement,
                  comparable_pairs=comparable, oracle_at2=quality[order[:2]].max(),
                  hard_negative_gain=quality[second] - quality[chosen],
                  hard_negative_better=float(quality[second] > quality[chosen] + 1e-8),
                  selected_failure=family[chosen]['failure'], selected_score=score[chosen])
    for t, mu in enumerate(MUS):
        result[f'selected_mu_{mu:.1f}'] = family[chosen]['bits'][t]
    return result


def scene_ci(records, metric, n_boot=5000, seed=20261107):
    """Equal families within frame, frames within scene, scenes within split.

    Primary contrasts are paired per-family *before* resampling; references are
    never pooled as independent scenes. Unrankable all-tie families are excluded
    only from ordering metrics, not from utility/regret.
    """
    if n_boot < 0: raise ValueError('n_boot must be nonnegative')
    nested = defaultdict(lambda: defaultdict(list))
    n_valid = 0
    for r in records:
        v = r.get(metric)
        if v is None or not np.isfinite(v): continue
        nested[(r['split'], r['scene'])][r['sample']].append(float(v)); n_valid += 1
    split_values = defaultdict(list)
    for (split, scene), frames in sorted(nested.items()):
        split_values[split].append(np.mean([np.mean(v) for v in frames.values()]))
    if not split_values:
        return dict(mean=None, ci95=None, scenes=0, frames=0, records=0)
    mean = float(np.mean([np.mean(v) for v in split_values.values()]))
    ci = None
    if n_boot and all(len(v) >= 2 for v in split_values.values()):
        rng = np.random.default_rng(seed)
        draws = np.zeros(n_boot)
        for values in split_values.values():
            a = np.array(values)
            draws += a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1) / len(split_values)
        ci = np.quantile(draws, [.025, .975]).tolist()
    return dict(mean=mean, ci95=ci, scenes=len(nested),
                frames=sum(len(v) for v in nested.values()), records=n_valid,
                weighting='equal_split_scene_frame_family',
                warning='CI unavailable: fewer than two scenes in a represented split' if ci is None and n_boot else None)


def comparison(left, right, n_boot, seed):
    """Positive gain means LEFT selects better than RIGHT at exactly same oracle."""
    index = lambda rows: {(r['sample'], r['family']): r for r in rows}
    a, b = index(left), index(right)
    if a.keys() != b.keys():
        raise ValueError('Matched comparison cannot discard missing families or frames')
    paired = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x['reference'] != y['reference'] or not np.isclose(x['oracle_quality'], y['oracle_quality'], atol=1e-12):
            raise ValueError('Different references/oracles; not a selection-only comparison')
        gain = x['selected_quality'] - y['selected_quality']
        reduction = y['regret'] - x['regret']
        if not np.isclose(gain, reduction, atol=1e-12):
            raise RuntimeError('Selection gain must equal regret reduction on a frozen bank')
        pair = dict(sample=x['sample'], scene=x['scene'], split=x['split'], family=x['family'],
                    selected_quality_gain=gain, regret_reduction=reduction,
                    selected_action_changed=float(x['selected_id'] != y['selected_id']),
                    repair=float(y['selected_quality'] == 0 and x['selected_quality'] > 0),
                    damage=float(y['selected_quality'] > 0 and x['selected_quality'] == 0),
                    right_failure=y['selected_failure'], left_failure=x['selected_failure'],
                    right_selected_depth=y['selected_depth'], left_selected_depth=x['selected_depth'],
                    pairwise_gain=(x['pairwise_agreement'] - y['pairwise_agreement']
                                   if x['pairwise_agreement'] is not None and y['pairwise_agreement'] is not None else None))
        for mu in MUS:
            k = f'selected_mu_{mu:.1f}'; pair[k + '_gain'] = x[k] - y[k]
        paired.append(pair)
    metrics = ('selected_quality_gain', 'regret_reduction', 'selected_action_changed', 'repair',
               'damage', 'pairwise_gain') + tuple(f'selected_mu_{m:.1f}_gain' for m in MUS)
    return dict(metrics={k: scene_ci(paired, k, n_boot, seed) for k in metrics},
                repair_by_right_failure={k: sum(r['repair'] for r in paired if r['right_failure'] == k) for k in FAILURES},
                damage_by_left_failure={k: sum(r['damage'] for r in paired if r['left_failure'] == k) for k in FAILURES},
                per_family=paired)


def apply_calibration(rows, artifact):
    """Only apply validation-fitted offsets; forbid ANY repeated scene ID."""
    scene_ids = {int(r['scene_id']) for r in rows}
    if scene_ids & set(artifact['scene_ids']):
        raise ValueError('Calibration/evaluation scenes overlap; frames or references cannot hide leakage')
    if artifact['kind'] != 'depth_additive_mse_offsets_v1':
        raise ValueError('Unknown calibration contract')
    out = list(rows)
    for r in rows:
        if r['condition'] != 'native': continue
        key = '|'.join((r['reference'], r['model']))
        params = artifact['parameters'].get(key)
        if params is None or params['checkpoint_sha256'] != r['checkpoint_sha256'] or params['source_checkpoint_sha256'] != r['source_checkpoint_sha256']:
            raise ValueError('Calibration reference/scorer checkpoint or protocol missing/mismatched')
        item = dict(r); item['score'] += params['offsets'][r['depth_id']]
        item['condition'] = 'native_calibrated'
        out.append(item)
    return out


def analyze(rows, n_boot=5000, seed=20261107):
    stats = [family_statistics(v) for v in group_rows(rows).values()]
    treatments = defaultdict(list)
    for r in stats:
        treatments[(r['reference'], r['model'], r['condition'])].append(r)
    summaries = {}
    summary_metrics = ('selected_quality', 'oracle_quality', 'regret', 'pairwise_agreement',
                       'oracle_at2', 'hard_negative_gain', 'hard_negative_better')
    for key, group in sorted(treatments.items()):
        summaries['/'.join(key)] = dict(
            metrics={m: scene_ci(group, m, n_boot, seed) for m in summary_metrics},
            families=len(group), rankable_families=sum(r['comparable_pairs'] > 0 for r in group),
            selected_failures=dict(Counter(r['selected_failure'] for r in group)),
            by_split={s: {m: scene_ci([r for r in group if r['split'] == s], m, n_boot, seed)
                          for m in summary_metrics} for s in sorted({r['split'] for r in group})})
    contrasts = {}
    for reference in sorted({r['reference'] for r in stats}):
        g = (reference, 'gvar', 'native')
        if g not in treatments: raise ValueError('Every reference requires native GVAR scores')
        for key in sorted(treatments):
            if key[0] != reference or key == g: continue
            if key[1] == 'point' and key[2] != 'native': continue
            if key[2] == 'native_calibrated': continue
            name = reference + '/gvar_native-minus-' + key[1] + '_' + key[2]
            contrasts[name] = comparison(treatments[g], treatments[key], n_boot, seed)
        pcal, gcal = (reference, 'point', 'native_calibrated'), (reference, 'gvar', 'native_calibrated')
        if pcal in treatments and gcal in treatments:
            contrasts[reference + '/calibrated_gvar-minus-calibrated_point'] = comparison(treatments[gcal], treatments[pcal], n_boot, seed)
        for model in ('point', 'gvar'):
            cal, native = (reference, model, 'native_calibrated'), (reference, model, 'native')
            if cal in treatments and native in treatments:
                contrasts[reference + '/' + model + '_calibrated-minus-native'] = comparison(treatments[cal], treatments[native], n_boot, seed)
    # Across-query ranking is secondary; each scorer chooses one depth per family.
    frame_groups = defaultdict(list)
    for r in stats:
        frame_groups[(r['reference'], r['model'], r['condition'], r['sample'])].append(r)
    ranking = defaultdict(list)
    for (ref, model, cond, sample), group in frame_groups.items():
        ordered = sorted(group, key=lambda r: (-r['selected_score'], r['family']))
        if len(ordered) < 2: continue
        a, b = ordered[:2]
        ranking['/'.join((ref, model, cond))].append(dict(sample=sample, scene=a['scene'], split=a['split'],
            top1_quality=a['selected_quality'], rank2_minus_rank1=b['selected_quality'] - a['selected_quality']))
    depth_profiles = {}
    by_depth = defaultdict(list)
    for r in rows:
        by_depth[(r['reference'], r['model'], r['condition'], r['depth_id'])].append(r)
    for key, values in sorted(by_depth.items()):
        records = [dict(sample=v['sample_id'], scene=v['scene_id'], split=v['split'],
                        score=v['score'], quality=v['quality'],
                        **{f'mu_{mu:.1f}': v['bits'][t] for t, mu in enumerate(MUS)},
                        **{reason: float(v['failure'] == reason) for reason in FAILURES}) for v in values]
        metrics = ('score', 'quality') + FAILURES + tuple(f'mu_{m:.1f}' for m in MUS)
        depth_profiles['/'.join(map(str, key))] = {
            'actions': len(values), 'depth_cm': key[-1] + 1,
            'metrics': {k: scene_ci(records, k, n_boot, seed) for k in metrics}}
    return dict(treatments=summaries, contrasts=contrasts, depth_profiles=depth_profiles,
                across_query={k: {m: scene_ci(v, m, n_boot, seed) for m in ('top1_quality', 'rank2_minus_rank1')}
                              for k, v in ranking.items()},
                per_family=stats, units='six-mu utility, NOT AP',
                limitations=[
                    'Reference banks are reported separately; they are not independent extra scenes.',
                    'Same physical bank and same oracle are verified before measuring selection gain.',
                    'Frozen-reader interventions are test-time sensitivity, not retrained architectural effects.',
                    'Cross-model common-geometry replay preserves each model\'s own features, but does not erase training-history confounds.',
                    'Scores are not conditioned on external executed width; width is frozen from the reference model.',
                    'Scene bootstrap does not cover training-seed variance or post-hoc hypothesis selection.',
                    'Geometry-bias conditions are GVAR-stream-only stress tests, not deployment calibration.'
                ])


def render_report(result):
    lines = ['# GVAR 固定物理动作诊断', '', '**这些数值是六摩擦阈值平均质量，不是官方 AP。**', '',
             '## G1/G2：共用候选及 oracle 的选择对比', '',
             '| 比较（左减右） | Selected Q gain | 95% scene CI | Regret reduction | Repair | Damage |',
             '|---|---:|---|---:|---:|---:|']
    for name, c in result['contrasts'].items():
        m = c['metrics']; fmt = lambda k: f"{m[k]['mean']:.6f}" if m[k]['mean'] is not None else '—'
        lines.append(f"| {name} | {fmt('selected_quality_gain')} | {m['selected_quality_gain']['ci95']} | {fmt('regret_reduction')} | {fmt('repair')} | {fmt('damage')} |")
    lines += ['', '## 每个参考候选库的独立结果', '', '| Reference / model / condition | Q selected | Oracle | Regret | Pairwise | Oracle@2 |', '|---|---:|---:|---:|---:|---:|']
    for name, value in result['treatments'].items():
        m = value['metrics']
        fmt = lambda k: f"{m[k]['mean']:.6f}" if m[k]['mean'] is not None else '—'
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(k) for k in ('selected_quality', 'oracle_quality', 'regret', 'pairwise_agreement', 'oracle_at2')) + ' |')
    lines += ['', '## 解释边界', ''] + ['- ' + v for v in result['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', action='append', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--markdown', default=None)
    p.add_argument('--calibration', default=None)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--expected-frames', type=int, default=0)
    p.add_argument('--expected-scenes', type=int, default=0)
    p.add_argument('--expected-families-per-reference', type=int, default=0)
    p.add_argument('--require-cross-model', action='store_true')
    p.add_argument('--require-controls', action='store_true')
    args = p.parse_args()
    rows = load_rows(args.csv)
    for label, expected, actual in (
        ('frames', args.expected_frames, len({r['sample_id'] for r in rows})),
        ('scenes', args.expected_scenes, len({r['scene_id'] for r in rows})),
    ):
        if expected and expected != actual:
            raise ValueError(f'Expected {expected} {label}, found {actual}')
    for reference in {r['reference'] for r in rows}:
        subset = [r for r in rows if r['reference'] == reference]
        available = {(r['model'], r['condition']) for r in subset}
        required = {('gvar', 'native')}
        if args.require_cross_model: required.add(('point', 'native'))
        if args.require_controls: required |= {('gvar', 'rgb_fixed'), ('gvar', 'perm_cycle')}
        if not required <= available:
            raise ValueError(f'{reference}: missing expected treatments {required - available}')
        families = len({(r['sample_id'], r['family_id']) for r in subset})
        if args.expected_families_per_reference and families != args.expected_families_per_reference:
            raise ValueError(f'{reference}: expected {args.expected_families_per_reference} families, found {families}')
    if args.calibration:
        rows = apply_calibration(rows, json.loads(Path(args.calibration).read_text()))
    result = analyze(rows, args.bootstrap, args.seed)
    result['source_sha256'] = {f: file_sha256(f) for f in args.csv}
    if args.calibration: result['calibration_sha256'] = file_sha256(args.calibration)
    write_json(args.output, result)
    if args.markdown:
        path = Path(args.markdown); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_report(result), encoding='utf-8')
    print(json.dumps({'output': args.output, 'treatments': len(result['treatments']), 'contrasts': len(result['contrasts'])}))


if __name__ == '__main__': main()
