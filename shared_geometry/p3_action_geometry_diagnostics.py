"""P3-0: paired action-decision flips and action-relative visible-geometry evidence.

This is an OFFLINE diagnostic on the existing signed P1 validation cache and a
frozen P2 shared reader. Clean rendered-visible geometry is privileged and is
never exposed to an RGB-only deployment model. No labels or source IDs enter
``score_cached``. There is no training, CAD relabeling, or official-test access.

Important geometry contract: the 75 points are a *gripper-relative stencil*.
Index j across two actions identifies the same location in the gripper stencil,
not the same 3D camera/world coordinate. Difference-of-differences therefore
measures the action-induced change in geometry **along corresponding gripper
stencil sites**, not a surface distance between colocated camera points.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import atomic_json, load_signed, sha256, signed
from .p0_relative_decision import metrics_for_family, scene_balanced, scene_bootstrap
from .p1_clean_visible import collate_cache, load_cached, read_cache_index, score_cached
from .p2_geometry_diagnostics import load_shared

SCHEMA = 'shared-visible-geometry-p3-action-delta-v1'
ARMS = ('predicted', 'clean_visible')


def stencil_regions():
    """Predeclared disjoint, coarse GRIPPER-STENCIL regions (not contact labels)."""
    x, y, _ = np.indices((5, 5, 3))
    x, y = x.reshape(-1), y.reshape(-1)
    regions = {
        'palm_proxy': x == 0,
        'lateral_finger_proxy': (x > 0) & ((y == 0) | (y == 4)),
        'opening_core_proxy': (x > 0) & (y >= 1) & (y <= 3),
    }
    if sum(int(v.sum()) for v in regions.values()) != 75 or not np.all(sum(regions.values()) == 1):
        raise AssertionError('Stencil region masks must partition all 75 support sites')
    return regions


REGIONS = stencil_regions()


def _mean_or_none(values, mask):
    arr = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if arr.shape != m.shape:
        raise ValueError('Evidence metric / mask shape mismatch')
    return float(arr[m].mean()) if np.any(m) else None


def pair_geometry_evidence(pred, clean, k, tau_m, *, regions=REGIONS):
    """Compare action-induced geometry deltas at matching stencil indices.

    Input is 7 x 75 x 6: truncated visible UDF, projective z residual and
    four observation state probabilities (one-hot for analytic P1 cache).
    Never relabel unobserved points as free/occupied. A continuous delta metric
    is defined only where BOTH sources observe BOTH physical support sites.
    """
    pred = np.asarray(pred, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if pred.shape != clean.shape or pred.ndim != 3 or pred.shape[1:] != (75, 6):
        raise ValueError('Aligned [K,75,6] analytic geometry required')
    if not 1 <= k < len(pred) or not 0 < tau_m <= .2:
        raise ValueError('Invalid alternative action index or geometry truncation')
    if not np.isfinite(pred).all() or not np.isfinite(clean).all():
        raise ValueError('Nonfinite geometry')
    p_state = pred[..., 2:].argmax(-1)
    c_state = clean[..., 2:].argmax(-1)
    # State 3 means unobserved. For the UDF it is not necessarily invalid in
    # the original KDTree computation; here we deliberately use a conservative
    # mask to avoid interpreting a missing ray as geometrically known.
    joint_known = ((p_state[0] != 3) & (p_state[k] != 3)
                   & (c_state[0] != 3) & (c_state[k] != 3))
    clean_known = (c_state[0] != 3) & (c_state[k] != 3)
    clean_near = (np.minimum(clean[0, :, 0], clean[k, :, 0]) <= .25)
    p_delta = pred[k, :, :2] - pred[0, :, :2]
    c_delta = clean[k, :, :2] - clean[0, :, :2]
    errors = np.abs(p_delta - c_delta) * (tau_m * 1000.)
    new_udf_error = np.abs(pred[k, :, 0] - clean[k, :, 0]) * tau_m * 1000.
    native_udf_error = np.abs(pred[0, :, 0] - clean[0, :, 0]) * tau_m * 1000.
    # This is a *categorical transition* diagnostic, not a 3D occupancy label.
    correct_transition = (p_state[0] == c_state[0]) & (p_state[k] == c_state[k])
    p_changed = p_state[k] != p_state[0]
    c_changed = c_state[k] != c_state[0]

    def group(mask):
        used = np.asarray(mask, dtype=bool)
        valid = used & joint_known
        near = valid & clean_near
        clean_observed = used & clean_known
        return {
            'n_stencil': int(used.sum()),
            'n_clean_observed': int(clean_observed.sum()),
            'n_both_observed': int(valid.sum()),
            'n_near_surface_both_observed': int(near.sum()),
            'udf_delta_mae_mm': _mean_or_none(errors[:, 0], valid),
            'projective_delta_mae_mm': _mean_or_none(errors[:, 1], valid),
            'near_udf_delta_mae_mm': _mean_or_none(errors[:, 0], near),
            'near_projective_delta_mae_mm': _mean_or_none(errors[:, 1], near),
            'native_udf_mae_mm': _mean_or_none(native_udf_error, valid),
            'alternative_udf_mae_mm': _mean_or_none(new_udf_error, valid),
            'state_pair_match_rate': _mean_or_none(correct_transition.astype(float), used),
            'state_change_match_rate': _mean_or_none((p_changed == c_changed).astype(float), used),
            'clean_near_fraction': _mean_or_none(clean_near.astype(float), used),
            'both_observed_fraction': _mean_or_none(joint_known.astype(float), used),
            'clean_visible_unobserved_fraction': _mean_or_none(((c_state[0] == 3) | (c_state[k] == 3)).astype(float), used),
        }

    out = {'all': group(np.ones(75, dtype=bool))}
    for name, mask in regions.items():
        if mask.shape != (75,):
            raise ValueError('Stencil region has invalid shape: ' + name)
        out[name] = group(mask)
    return out


def _decision_label(k, q):
    if k == 0:
        return 'keep'
    if q[k] > q[0]:
        return 'rescue'
    if q[k] < q[0]:
        return 'harm'
    return 'equal_quality_move'


def _flip_kind(kp, kc, q):
    if kp == kc:
        return 'same_action'
    pred = _decision_label(kp, q)
    clean = _decision_label(kc, q)
    if pred == 'harm' and clean == 'keep':
        return 'harm_to_keep'
    if pred == 'keep' and clean == 'rescue':
        return 'keep_to_rescue'
    if pred == 'harm' and clean == 'rescue':
        return 'harm_to_rescue'
    if pred == 'harm' and clean == 'harm' and q[kc] > q[kp]:
        return 'harm_to_less_harmful_move'
    if pred == 'rescue' and clean == 'harm':
        return 'rescue_to_harm'
    if pred == 'keep' and clean == 'harm':
        return 'keep_to_harm'
    if pred == 'rescue' and clean == 'rescue' and q[kc] > q[kp]:
        return 'rescue_to_better_rescue'
    return 'other_changed_action'


def decision_flips(pred_logits, clean_logits, counts, offsets_m):
    """Decisions on exactly the same physical action family; no GT in logits."""
    zp = np.asarray(pred_logits, dtype=np.float64)
    zc = np.asarray(clean_logits, dtype=np.float64)
    q = np.asarray(counts, dtype=np.int64) / 6.
    if zp.shape != zc.shape or zp.shape != q.shape or zp.ndim != 1 or len(zp) != len(offsets_m):
        raise ValueError('Inconsistent action/logit/label family')
    kp, kc = int(np.argmax(zp)), int(np.argmax(zc))
    catp, catc = _decision_label(kp, q), _decision_label(kc, q)
    return {
        'pred_choice': kp, 'clean_choice': kc,
        'pred_offset_mm': round(float(offsets_m[kp]) * 1000, 5),
        'clean_offset_mm': round(float(offsets_m[kc]) * 1000, 5),
        'pred_decision': catp, 'clean_decision': catc,
        'pred_selected_q': float(q[kp]), 'clean_selected_q': float(q[kc]),
        'native_q': float(q[0]), 'oracle_q': float(q.max()),
        'pred_gain': float(q[kp] - q[0]), 'clean_gain': float(q[kc] - q[0]),
        'source_gain_diff': float(q[kc] - q[kp]),
        'flipped': bool(kp != kc),
        'clean_better': bool(q[kc] > q[kp]),
        'clean_worse': bool(q[kc] < q[kp]),
        'same_q_different_action': bool(kp != kc and q[kp] == q[kc]),
        'flip_type': _flip_kind(kp, kc, q),
        'has_opportunity': bool(q.max() > q[0]),
        'native_is_full': bool(counts[0] == 6),
        'pred_top_logit_margin': float(zp[kp] - max(np.delete(zp, kp))),
        'clean_top_logit_margin': float(zc[kc] - max(np.delete(zc, kc))),
    }


def _grouped_by_scene(records, metric, selector=None):
    selected = [r for r in records if selector is None or selector(r)]
    return scene_balanced(selected, metric)


def _mean_or_none_vector(values):
    valid = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(valid)) if valid else None


def _region_summaries(records):
    out = {}
    for key in ('all',) + tuple(REGIONS):
        out[key] = {}
        for name in ('udf_delta_mae_mm', 'projective_delta_mae_mm',
                     'near_udf_delta_mae_mm', 'near_projective_delta_mae_mm',
                     'both_observed_fraction', 'state_change_match_rate'):
            # Values are averaged per family (six perturbations) before the
            # scene/domain reduction. Do not treat 75*6 queries as IID scenes.
            col = f'geometry_{key}_{name}'
            out[key][name] = scene_balanced(records, col)
    return out


def aggregate(records, bootstrap=5000, seed=0):
    if not records:
        raise ValueError('No P3 family records')
    keys = ('pred_gain', 'clean_gain', 'source_gain_diff', 'flipped', 'clean_better',
            'clean_worse', 'same_q_different_action', 'has_opportunity')
    summary = {key: scene_balanced(records, key) for key in keys}
    summary['n_families'] = len(records)
    summary['n_scenes'] = len({r['scene'] for r in records})
    summary['n_domain_scene_cells'] = len({(r['scene'], r['domain']) for r in records})
    summary['flip_type_counts'] = dict(Counter(r['flip_type'] for r in records))
    summary['decision_transition_counts'] = dict(Counter(f"{r['pred_decision']} -> {r['clean_decision']}" for r in records))
    summary['flip_types'] = {}
    for ty, n in sorted(summary['flip_type_counts'].items()):
        def subset(r, ty=ty): return r['flip_type'] == ty
        summary['flip_types'][ty] = {
            'n': n,
            'source_gain_diff': _grouped_by_scene(records, 'source_gain_diff', subset),
            'pred_gain': _grouped_by_scene(records, 'pred_gain', subset),
            'clean_gain': _grouped_by_scene(records, 'clean_gain', subset),
            'near_udf_delta_mae_mm': _grouped_by_scene(records, 'geometry_all_near_udf_delta_mae_mm', subset),
        }
    summary['native_full_vs_other'] = {
        name: {'n': sum(sel(r) for r in records),
               'gain_diff': _grouped_by_scene(records, 'source_gain_diff', sel),
               'pred_gain': _grouped_by_scene(records, 'pred_gain', sel),
               'clean_gain': _grouped_by_scene(records, 'clean_gain', sel),
               'flip_fraction': _grouped_by_scene(records, 'flipped', sel)}
        for name, sel in (('native_full', lambda r: r['native_is_full']),
                          ('native_below_full', lambda r: not r['native_is_full']),
                          ('has_opportunity', lambda r: r['has_opportunity']),
                          ('no_opportunity', lambda r: not r['has_opportunity']))
    }
    summary['regions'] = _region_summaries(records)
    # Six fixed ray offsets, every non-tied native-relative pair counted once.
    # Useful for separating source-induced pair repairs from source-induced
    # failures, rather than correlating only the final selected action.
    pairs = [dict(scene=r['scene'], domain=r['domain'], **a)
             for r in records for a in r['alternatives']]
    summary['by_offset_mm'] = {}
    for off in sorted({a['offset_mm'] for a in pairs}):
        row = [a for a in pairs if a['offset_mm'] == off]
        comparable = [a for a in row if a['pred_pair_correct'] is not None]
        summary['by_offset_mm'][str(off)] = {
            'pairs': len(row), 'non_tied_pairs': len(comparable),
            'pred_pair_accuracy': scene_balanced(comparable, 'pred_pair_correct'),
            'clean_pair_accuracy': scene_balanced(comparable, 'clean_pair_correct'),
            'source_pair_repair': scene_balanced(comparable, 'source_pair_repair'),
            'source_pair_damage': scene_balanced(comparable, 'source_pair_damage'),
            'mean_true_delta_q': scene_balanced(row, 'delta_q'),
            'mean_predicted_delta_probability': scene_balanced(row, 'predicted_delta_probability'),
            'mean_clean_delta_probability': scene_balanced(row, 'clean_delta_probability'),
            'udf_delta_mae_mm': scene_balanced(row, 'relative_udf_delta_error_mm'),
            'projective_delta_mae_mm': scene_balanced(row, 'relative_projective_delta_error_mm'),
        }
    def pair_status(a):
        if a['pred_pair_correct'] is None: return 'exact_quality_tie'
        if a['source_pair_repair']: return 'pred_wrong_clean_correct'
        if a['source_pair_damage']: return 'pred_correct_clean_wrong'
        if a['pred_pair_correct']: return 'both_correct'
        return 'both_incorrect'
    summary['pair_repair_error_association'] = {}
    for kind in ('pred_wrong_clean_correct','pred_correct_clean_wrong',
                 'both_correct','both_incorrect','exact_quality_tie'):
        subset = [a for a in pairs if pair_status(a) == kind]
        summary['pair_repair_error_association'][kind] = {
            'n': len(subset),
            'udf_delta_mae_mm': scene_balanced(subset, 'relative_udf_delta_error_mm'),
            'projective_delta_mae_mm': scene_balanced(subset, 'relative_projective_delta_error_mm'),
            'near_udf_delta_mae_mm': scene_balanced(subset, 'relative_near_udf_delta_error_mm'),
            'both_observed_fraction': scene_balanced(subset, 'relative_both_observed_fraction'),
        }
    summary['geometry_error_bands'] = {}
    bands = (('0_to_2mm', 0., 2.), ('2_to_5mm', 2., 5.),
             ('5_to_10mm', 5., 10.), ('at_least_10mm', 10., float('inf')))
    for key in ('geometry_all_udf_delta_mae_mm', 'geometry_all_projective_delta_mae_mm'):
        entry = {}
        for tag, lo, hi in bands:
            def cond(r, lo=lo, hi=hi, key=key):
                v = r.get(key)
                return v is not None and lo <= v < hi
            entry[tag] = {'n': sum(cond(r) for r in records),
                          'clean_better_fraction': _grouped_by_scene(records, 'clean_better', cond),
                          'gain_diff': _grouped_by_scene(records, 'source_gain_diff', cond)}
        entry['no_joint_observation'] = sum(r.get(key) is None for r in records)
        summary['geometry_error_bands'][key] = entry
    # Correct scene-level paired comparison: mean within domain/scene, then
    # equal domains per scene and bootstrap over *scenes*, not 5k action families.
    summary['paired_clean_minus_predicted'] = scene_bootstrap(
        [dict(scene=r['scene'], domain=r['domain'], gain=r['source_gain_diff']) for r in records],
        metric='gain', n_boot=bootstrap, seed=seed)
    return summary


def _family_record(row, fam_id, physical_ids, zp, zc, q, offsets, pred, clean, tau):
    flip = decision_flips(zp, zc, q, offsets)
    full = metrics_for_family(zp, q, offsets)
    clean_full = metrics_for_family(zc, q, offsets)
    alternatives = []
    geom = []
    for k in range(1, len(q)):
        g = pair_geometry_evidence(pred, clean, k, tau)
        geom.append(g)
        truth = int(np.sign(int(q[k]) - int(q[0])))
        pred_cmp = int(np.sign(zp[k] - zp[0]))
        clean_cmp = int(np.sign(zc[k] - zc[0]))
        alternatives.append({
            'index': k, 'offset_mm': round(float(offsets[k])*1000, 5),
            'delta_q': float((int(q[k])-int(q[0]))/6.),
            'predicted_delta_probability': float(full['alternatives'][k-1]['pred_delta']),
            'clean_delta_probability': float(clean_full['alternatives'][k-1]['pred_delta']),
            'pred_pair_correct': None if truth == 0 else bool(pred_cmp == truth),
            'clean_pair_correct': None if truth == 0 else bool(clean_cmp == truth),
            'source_pair_repair': bool(truth != 0 and pred_cmp != truth and clean_cmp == truth),
            'source_pair_damage': bool(truth != 0 and pred_cmp == truth and clean_cmp != truth),
            'relative_udf_delta_error_mm': g['all']['udf_delta_mae_mm'],
            'relative_projective_delta_error_mm': g['all']['projective_delta_mae_mm'],
            'relative_near_udf_delta_error_mm': g['all']['near_udf_delta_mae_mm'],
            'relative_both_observed_fraction': g['all']['both_observed_fraction'],
            'geometry': g,
        })
    record = {**flip, 'scene': int(row['scene']), 'domain': row['domain'],
              'anno': int(row['anno']), 'family': int(fam_id),
              'physical_native': str(physical_ids[0]),
              'physical_pred_selected': str(physical_ids[flip['pred_choice']]),
              'physical_clean_selected': str(physical_ids[flip['clean_choice']]),
              'native_count': int(q[0]), 'alternatives': alternatives}
    for key in ('all',) + tuple(REGIONS):
        for name in ('udf_delta_mae_mm', 'projective_delta_mae_mm',
                     'near_udf_delta_mae_mm', 'near_projective_delta_mae_mm',
                     'both_observed_fraction', 'state_change_match_rate'):
            record[f'geometry_{key}_{name}'] = _mean_or_none_vector(g[key][name] for g in geom)
    # Fixed choice indices from the two sources, no GT selection of regions.
    for source, k in (('pred', flip['pred_choice']), ('clean', flip['clean_choice'])):
        if k:
            record[f'{source}_selected_geometry'] = geom[k-1]
        else:
            record[f'{source}_selected_geometry'] = None
    return record


def _sanity_check_input(frame, row, index):
    if row['partition'] != 'validation' or row['scene'] >= 100:
        raise ValueError('P3 requires validation-only scene/partition')
    counts = frame['quality_count']
    if counts.ndim != 2 or counts.shape[1] != 7:
        raise ValueError('Expected seven exact-action labels per family')
    if len(frame['family_id']) != counts.shape[0] or frame['physical_id'].shape != counts.shape:
        raise ValueError('Physical ID / family size mismatch')
    if not np.isfinite(frame['local_xyz']).all():
        raise ValueError('Nonfinite gripper support coordinates')
    # P1 signed cache already binds predicted and clean tensors to a single
    # physical_id array. Stencil positions must be the same relative to each
    # perturbed action (translation-only bank), up to FP32 construction error.
    if not np.allclose(frame['local_xyz'], frame['local_xyz'][:, :1], atol=1e-6, rtol=0):
        raise ValueError('Action-relative stencil index alignment failed')
    if tuple(index['contract']['offsets_m'])[0] != 0:
        raise ValueError('Native action is not index zero')


def _score_hybrid(model, framebatch, data, device):
    fake = dict(framebatch)
    fake['predicted_physics'] = data
    with torch.inference_mode():
        return score_cached(model, fake, 'predicted', device).detach().cpu().numpy()


def _make_hybrid(pred, clean, mask, *, forward=True):
    if mask.shape != (75,):
        raise ValueError('Invalid intervention stencil mask')
    # Replace complete 6D physical tokens, not just one component. Status
    # one-hot remains valid and geometry is internally consistent per token.
    base, replacement = (pred, clean) if forward else (clean, pred)
    result = base.copy()
    result[..., mask, :] = replacement[..., mask, :]
    return result


def diagnose(args):
    if args.batch_size < 1 or args.bootstrap < 1 or args.limit < 0:
        raise ValueError('Invalid diagnostic budget')
    if args.limit and not args.smoke:
        raise ValueError('Partial validation diagnostics require --smoke')
    idx = read_cache_index(args.index, 'validation', args.smoke)
    if not args.smoke and len(idx['entries']) != 160:
        raise ValueError('Formal P3 requires full 160-frame validation')
    model, protocol = load_shared(args.shared_weight, args.device, args.smoke)
    if protocol['val_fingerprint'] != idx['fingerprint'] or protocol['teacher_config'] != idx['contract']['target_config']:
        raise ValueError('Frozen P2 reader does not match the signed P1 cache')
    weight_sha = sha256(args.shared_weight)
    if getattr(args, 'expected_p2', ''):
        check = load_signed(args.expected_p2)
        if check['validation_fingerprint'] != idx['fingerprint'] or check['checkpoint_sha256'] != weight_sha:
            raise ValueError('Expected P2 comparison does not match this validation/weight')
    model.eval()
    inputs = idx['entries'][:args.limit] if args.limit else idx['entries']
    tau = idx['contract']['tau']
    offsets = idx['contract']['offsets_m']
    records = []
    intervention_records = {name: [] for name in REGIONS}
    inverse_records = {name: [] for name in REGIONS}
    max_all_swap_error = 0.
    for start in range(0, len(inputs), args.batch_size):
        entries = inputs[start:start+args.batch_size]
        frames = [load_cached(e) for e in entries]
        for fr, en in zip(frames, entries):
            _sanity_check_input(fr, en['row'], idx)
        batch = collate_cache(frames)
        pred = batch['predicted_physics']
        clean = batch['clean_visible_physics']
        with torch.inference_mode():
            z_pred = score_cached(model, batch, 'predicted', args.device).detach().cpu().numpy()
            z_clean = score_cached(model, batch, 'clean_visible', args.device).detach().cpu().numpy()
        if not getattr(args, 'skip_interventions', False):
            # Full-source replacement must exactly recover the frozen reader
            # output; it is only a numerical provenance/slicing sanity check.
            all_mask = np.ones(75, dtype=bool)
            z_all = _score_hybrid(model, batch, _make_hybrid(pred, clean, all_mask), args.device)
            max_all_swap_error = max(max_all_swap_error, float(np.max(np.abs(z_all - z_clean))))
            if max_all_swap_error > 1e-5:
                raise RuntimeError('Hybrid full replacement fails fixed-reader parity')
            hybrid_pred, hybrid_clean = {}, {}
            for region, mask in REGIONS.items():
                hybrid_pred[region] = _score_hybrid(model, batch, _make_hybrid(pred, clean, mask), args.device)
                hybrid_clean[region] = _score_hybrid(model, batch, _make_hybrid(pred, clean, mask, forward=False), args.device)
        for b, en in enumerate(entries):
            for f in range(len(frames[b]['family_id'])):
                rec = _family_record(en['row'], frames[b]['family_id'][f], frames[b]['physical_id'][f],
                                     z_pred[b, f], z_clean[b, f], frames[b]['quality_count'][f],
                                     offsets, pred[b, f], clean[b, f], tau)
                records.append(rec)
                if not getattr(args, 'skip_interventions', False):
                    q = frames[b]['quality_count'][f].astype(float)/6.
                    for region in REGIONS:
                        k_in = int(np.argmax(hybrid_pred[region][b, f]))
                        k_out = int(np.argmax(hybrid_clean[region][b, f]))
                        intervention_records[region].append(dict(
                            scene=rec['scene'],domain=rec['domain'],gain=float(q[k_in]-q[0]),
                            comparison_gain=float(q[k_in]-q[int(rec['pred_choice'])]),
                            move=bool(k_in!=0),selected_index=k_in,
                            improved_vs_pred=bool(q[k_in]>q[int(rec['pred_choice'])]),
                            harmed_vs_pred=bool(q[k_in]<q[int(rec['pred_choice'])])))
                        inverse_records[region].append(dict(
                            scene=rec['scene'],domain=rec['domain'],gain=float(q[k_out]-q[0]),
                            comparison_gain=float(q[k_out]-q[int(rec['clean_choice'])]),
                            move=bool(k_out!=0),selected_index=k_out))
    report = dict(schema=SCHEMA, study='P3-0 fixed-reader action-relative geometry and choice flips',
                  partition='validation', privileged_clean_visible=True, official_ap_not_run=True,
                  trained_new_weights=False, source=dict(
                      p1_validation_index_sha256=sha256(args.index),
                      p1_validation_fingerprint=idx['fingerprint'],
                      frozen_p2_shared_weight_sha256=weight_sha,
                      frozen_p2_protocol=protocol,
                      expected_p2_comparison_sha256=sha256(args.expected_p2) if getattr(args,'expected_p2','') else None),
                  n_frames=len(inputs),n_action_families=len(records),
                  geometry_definitions=dict(truncation_m=tau, near_visible_udf_threshold_normalized=.25,
                      regions={name:int(mask.sum()) for name,mask in REGIONS.items()},
                      note='All region names denote STENCIL proxies; not established contact physics. Delta compares matching gripper-relative stencil indices, not colocated camera points.'),
                  summary=aggregate(records, bootstrap=args.bootstrap, seed=args.seed),
                  per_family=records,
                  warning='All contrasts are exploratory on validation-selected single-seed P2 weights; clean is privileged. Region replacements are distribution-shift interventions and not a causal contribution decomposition.')
    if not getattr(args, 'skip_interventions', False):
        report['region_interventions'] = dict(full_swap_logit_parity_max_error=max_all_swap_error,
            predicted_to_clean={name:dict(gain=scene_balanced(rr,'gain'),
                move=scene_balanced(rr,'move'),
                improvement_vs_pred=scene_balanced(rr,'comparison_gain'),
                improved_fraction=scene_balanced(rr,'improved_vs_pred'),
                harmed_fraction=scene_balanced(rr,'harmed_vs_pred'),
                paired_gain_vs_predicted=scene_bootstrap(rr, metric='comparison_gain', n_boot=args.bootstrap, seed=args.seed))
                for name, rr in intervention_records.items()},
            clean_to_predicted={name:dict(gain=scene_balanced(rr,'gain'),
                degradation_vs_clean=scene_balanced(rr,'comparison_gain'),
                paired_gain_vs_clean=scene_bootstrap(rr, metric='comparison_gain', n_boot=args.bootstrap, seed=args.seed))
                for name, rr in inverse_records.items()})
    if getattr(args, 'expected_p2', '') and not args.limit:
        check = load_signed(args.expected_p2)
        for name, metric in [('predicted', 'pred_gain'), ('clean_visible','clean_gain')]:
            got = scene_balanced(records, metric)
            original = float(check['summaries'][name]['gain'])
            if abs(got - original)>2e-6:
                raise RuntimeError(f'P3 source-parity failed for {name}: {got} vs P2 {original}')
        report['source']['published_p2_gain_parity_checked'] = True
    elif getattr(args,'expected_p2',''):
        report['source']['published_p2_gain_parity_checked'] = False  # partial smoke
    path = Path(args.output)
    if path.exists():
        raise FileExistsError('Cannot overwrite existing P3 diagnostics')
    atomic_json(path, signed(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True, help='Signed P1 validation cache index')
    p.add_argument('--shared-weight', required=True, help='Frozen P2 shared-reader best.pt')
    p.add_argument('--expected-p2', default='', help='Optional signed P2 comparison for exact gain parity')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-interventions', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    report = diagnose(args)
    print(f"[P3] validated {report['n_frames']} frames, {report['n_action_families']} families", flush=True)
    print(f"[P3] clean-pred gain = {report['summary']['paired_clean_minus_predicted']['mean']*100:+.4f} Q-pp", flush=True)


if __name__ == '__main__':
    main()