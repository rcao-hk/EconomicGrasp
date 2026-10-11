#!/usr/bin/env python3
"""P1: conditional gripper-probe reliability, matched reference errors and AP links.

Consumes *previously exported* AP-selected Novel frames; it does not run a
model, render grasp outcomes, or use any depth reference for inference.
- Scene x role: validity coverage, matched predicted-vs-rendered/sensor MAE,
  and conditional (within-frame) error-risk Spearman diagnostics.
- Same physical probe: compare rendered/sensor absolute depth errors ONLY if
  both reference validity masks are satisfied. No extrapolated reference gaps.
- Frame-to-AP: within-scene-centered descriptive associations, not causation.
  Frames were selected *using AP outcome*; 5 scenes and 4 frames per scene are
  not a population sample, and hundreds of probes are not independent trials.

Predicted-depth local STD and gradient are proxies, NOT calibrated uncertainty.
Do not infer gripper contact, free space, or physical grasp success from these
projected visual-depth probes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable

import numpy as np

VERSION = 1
ROLES = ('left_contact', 'right_contact', 'closing', 'finger_body', 'palm', 'approach')
PROXIES = ('proxy_depth_std_m', 'proxy_depth_gradient_mpx', 'uncertainty_sigma_m')
TARGETS = ('delta_AP_pp', 'delta_mu04_pp', 'delta_mu08_pp')
METRICS = ('rendered_coverage', 'sensor_coverage', 'both_reference_coverage',
           'foreground_coverage', 'rendered_mae_mm', 'paired_rendered_mae_mm',
           'paired_sensor_mae_mm', 'paired_error_difference_mm',
           'mean_proxy_std_mm', 'closing_rendered_coverage',
           'contact_rendered_coverage', 'approach_rendered_coverage')


class DataError(ValueError):
    pass


def check(flag: bool, message: str) -> None:
    if not flag:
        raise DataError(message)


def read_json(path: Path) -> dict:
    check(path.is_file(), f'Missing JSON source: {path}')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise DataError(f'Cannot parse JSON {path}: {exc}') from exc
    check(isinstance(data, dict), f'Expected JSON object: {path}')
    return data


def source_hash(path: Path) -> str:
    check(path.is_file(), f'Missing source: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    check(path.is_file(), f'Missing CSV source: {path}')
    with path.open('r', newline='', encoding='utf-8') as stream:
        rd = csv.DictReader(stream)
        check(rd.fieldnames is not None and set(required).issubset(rd.fieldnames),
              f'CSV missing fields {set(required) - set(rd.fieldnames or [])}: {path}')
        rows = list(rd)
    check(bool(rows), f'Empty CSV source: {path}')
    return rows


def num(x: str | None, *, where: str, optional=False) -> float | None:
    if x is None or x == '' or x == 'None':
        if optional:
            return None
        raise DataError(f'Missing numeric value: {where}')
    try:
        v = float(x)
    except ValueError as exc:
        raise DataError(f'Invalid number {where}={x!r}') from exc
    check(math.isfinite(v), f'Nonfinite number {where}={v}')
    return v


def boolean(x: str | bool | None, *, where: str) -> bool:
    if x in (True, 'True', 'true', '1', 'TRUE'):
        return True
    if x in (False, 'False', 'false', '0', 'FALSE'):
        return False
    raise DataError(f'Invalid boolean {where}={x!r}')


def average(values: Iterable[float | None]) -> float | None:
    arr = [v for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(arr, dtype=np.float64)) if arr else None


def tie_rank(data: np.ndarray) -> np.ndarray:
    order = np.argsort(data, kind='stable')
    ranks = np.empty(data.shape[0], dtype=np.float64)
    sorted_values = data[order]
    unique, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    del unique
    for start, count in zip(starts, counts):
        ranks[order[start:start+count]] = start + (count - 1) / 2.0
    return ranks


def spearman(x: Iterable[float], y: Iterable[float], *, min_n: int = 10) -> float | None:
    x, y = np.asarray(list(x), dtype=np.float64), np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_n:
        return None
    a, b = tie_rank(x[mask]), tie_rank(y[mask])
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def pearson_within_scene(frames: list[dict], metric: str, target: str) -> tuple[float | None, int, int]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in frames:
        if r.get(metric) is not None and r.get(target) is not None:
            grouped[r['scene_id']].append(r)
    xx, yy, used = [], [], 0
    for _, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        x = np.asarray([r[metric] for r in rows], dtype=np.float64)
        y = np.asarray([r[target] for r in rows], dtype=np.float64)
        xx.extend((x-x.mean()).tolist()); yy.extend((y-y.mean()).tolist())
        used += 1
    if len(xx) < 3 or np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return None, used, len(xx)
    return float(np.corrcoef(xx, yy)[0, 1]), used, len(xx)


def group_stats(items: list[dict], min_probes=15) -> dict[str, Any]:
    n = len(items)
    check(n > 0, 'Empty probe group')
    projection = [r for r in items if r['projection_valid']]
    fg = [r for r in items if r['foreground_valid']]
    rendered = [r for r in items if r['rendered_valid']]
    sensor = [r for r in items if r['sensor_valid']]
    both = [r for r in items if r['rendered_valid'] and r['sensor_valid']]
    rend_err = average(r['rendered_error_m'] for r in rendered)
    rend_both = average(r['rendered_error_m'] for r in both)
    sensor_both = average(r['sensor_error_m'] for r in both)
    stats: dict[str, Any] = {
        'probes': n, 'projection_valid': len(projection),
        'foreground_valid': len(fg), 'rendered_valid': len(rendered),
        'sensor_valid': len(sensor), 'both_reference_valid': len(both),
        'projection_coverage': len(projection)/n,
        'foreground_coverage': len(fg)/n,
        'rendered_coverage': len(rendered)/n,
        'sensor_coverage': len(sensor)/n,
        'both_reference_coverage': len(both)/n,
        'rendered_mae_mm': 1000*rend_err if rend_err is not None else None,
        'sensor_mae_mm_unmatched': (1000*average(r['sensor_error_m'] for r in sensor)
                                    if sensor else None),
        'paired_rendered_mae_mm': 1000*rend_both if rend_both is not None else None,
        'paired_sensor_mae_mm': 1000*sensor_both if sensor_both is not None else None,
        'paired_error_difference_mm': (1000*(rend_both-sensor_both)
                                       if rend_both is not None and sensor_both is not None else None),
        # Paired means, not unsigned ground-truth-vs-sensor surface distance.
        'scene_CI': 'not estimated: probe samples are clustered within grasp/frame/scene'
    }
    for col, short in (('proxy_depth_std_m', 'std'),
                       ('proxy_depth_gradient_mpx', 'gradient'),
                       ('uncertainty_sigma_m', 'external_sigma')):
        selected = [r for r in rendered if r[col] is not None]
        stats[f'{short}_count'] = len(selected)
        stats[f'{short}_spearman_vs_error'] = spearman(
            (r[col] for r in selected), (r['rendered_error_m'] for r in selected),
            min_n=min_probes)
    return stats


def _row_num(row: dict, key: str, where: str, optional=False) -> float | None:
    return num(row.get(key), where=f'{where}.{key}', optional=optional)


def check_inputs(probe_dir: Path, cases_dir: Path, variants: list[str]) -> tuple[list[dict], list[dict], dict, list[dict], dict]:
    selection_file = cases_dir / 'selected_frames.json'
    manifest_file = probe_dir / 'manifest.json'
    selection = read_json(selection_file)
    manifest = read_json(manifest_file)
    check(selection.get('split') == 'test_novel' and selection.get('epoch') == 19,
          'Expected e19 test_novel AP-selected cases')
    check(manifest.get('source_selection_sha256') == source_hash(selection_file),
          'Probe-export selection hash does not match selected_frames.json')
    check(all(v in manifest.get('variants', []) for v in variants),
          f'Variants missing in probe manifest: {variants} vs {manifest.get("variants")}')
    check(manifest.get('uncertainty_provenance') is not None,
          'Missing uncertainty provenance in probe manifest')
    check(manifest.get('rank_topk', 0) > 0, 'Missing probe top-K contract')

    frame_rows = read_csv(cases_dir/'novel_frames.csv',
                          ('scene_id','frame_id','delta_AP_pp','delta_mu04_pp','delta_mu08_pp','selection'))
    frame_lookup = {}
    for raw in frame_rows:
        key = (int(raw['scene_id']), int(raw['frame_id']))
        check(key not in frame_lookup, f'Duplicate AP frame: {key}')
        frame_lookup[key] = raw
    requested = {}
    for x in selection.get('selected_frames', []):
        k = (int(x['scene_id']), int(x['frame_id']))
        check(160 <= k[0] <= 189 and k[1] in range(0,256,10),f'Noncanonical selected frame {k}')
        check(k not in requested, f'Duplicate selected frame {k}')
        check(k in frame_lookup, f'Selected frame absent in novel_frames.csv: {k}')
        raw = frame_lookup[k]
        check(x['selection'] == raw['selection'], f'Frame AP selection mismatch {k}')
        require_delta = num(x['delta_AP_pp'], where=f'selected {k}.delta')
        check(abs(require_delta - num(raw['delta_AP_pp'], where=f'AP frame {k}.delta')) <= 1e-5,
              f'AP delta mismatch for selected frame {k}')
        requested[k] = raw
    check(bool(requested), 'No selected frames available')
    raw_probes = read_csv(probe_dir/'probe_records.csv',
                          ('variant','scene_id','frame_id','role','grasp_rank','probe_idx',
                           'projection_valid','foreground_valid','rendered_valid','sensor_valid',
                           'rendered_error_m','sensor_error_m',*PROXIES,'source_frame_delta_AP_pp'))
    raw_frame_summary = read_csv(probe_dir/'frame_probe_summary.csv',
                                 ('variant','scene_id','frame_id','status','num_grasps','total_probes',
                                  'valid_rendered_object_probes','source_frame_delta_AP_pp'))
    probe_records = []
    observed_keys = set()
    for i, r in enumerate(raw_probes):
        if r['variant'] not in variants:
            continue
        key=(int(r['scene_id']),int(r['frame_id']))
        check(key in requested, f'Probe references nonselected frame {key}')
        role=r['role']
        check(role in ROLES, f'Unknown probe role {role}')
        rank, probeidx=int(r['grasp_rank']), int(r['probe_idx'])
        check(1 <= rank <= int(manifest['rank_topk']) and 0 <= probeidx < 36,
              f'Probe rank/index out of range at row {i}')
        probe_key=(r['variant'],*key,rank,probeidx)
        check(probe_key not in observed_keys, f'Duplicate physical probe identity {probe_key}')
        observed_keys.add(probe_key)
        check(abs(num(r['source_frame_delta_AP_pp'],where=f'probe row {i}.AP')
                  -num(requested[key]['delta_AP_pp'],where=f'AP selection {key}'))<1e-5,
              f'Probe/AP mismatch at {probe_key}')
        flags={k:boolean(r[k],where=f'{probe_key}.{k}') for k in
               ('projection_valid','foreground_valid','rendered_valid','sensor_valid')}
        check((not flags['foreground_valid'] or flags['projection_valid']) and
              (not flags['rendered_valid'] or flags['foreground_valid']) and
              (not flags['sensor_valid'] or flags['foreground_valid']),
              f'Illogical validity masks at {probe_key}')
        rend=_row_num(r,'rendered_error_m',str(probe_key),optional=True)
        sensor=_row_num(r,'sensor_error_m',str(probe_key),optional=True)
        check((rend is not None)==flags['rendered_valid'],
              f'Rendered validity/error inconsistency at {probe_key}')
        check((sensor is not None)==flags['sensor_valid'],
              f'Sensor validity/error inconsistency at {probe_key}')
        check((rend is None or rend>=0) and (sensor is None or sensor>=0),
              f'Negative depth error at {probe_key}')
        probe_records.append({'variant':r['variant'],'scene_id':key[0], 'frame_id':key[1],
                              'role':role,'grasp_rank':rank,'probe_idx':probeidx,
                              **flags,'rendered_error_m':rend,'sensor_error_m':sensor,
                              **{k:_row_num(r,k,str(probe_key),optional=True) for k in PROXIES},
                              'selection':requested[key]['selection']})
    check(probe_records, 'No matching probe records')
    frame_summary = {}
    for r in raw_frame_summary:
        if r['variant'] not in variants:
            continue
        k=(r['variant'],int(r['scene_id']),int(r['frame_id']))
        check(k not in frame_summary,f'Duplicate frame summary {k}')
        check(k[1:] in requested,f'Frame summary not in AP selection {k}')
        check(abs(num(r['source_frame_delta_AP_pp'],where=str(k))
                  -num(requested[k[1:]]['delta_AP_pp'],where=str(k)))<1e-5,
              f'Frame summary/AP mismatch at {k}')
        frame_summary[k]=r
    group_by_frame=defaultdict(list)
    for r in probe_records:
        group_by_frame[(r['variant'],r['scene_id'],r['frame_id'])].append(r)
    for v in variants:
        for scene, frame in requested:
            k=(v,scene,frame)
            check(k in frame_summary, f'Missing frame probe summary {k}')
            summary=frame_summary[k]
            total=int(float(summary['total_probes']))
            grasp_count=int(float(summary['num_grasps']))
            count=len(group_by_frame[k])
            check(total == grasp_count*36 == count,
                  f'Inconsistent 36-probe/grasp count for {k}: {total}, {grasp_count}, {count}')
            check(int(float(summary['valid_rendered_object_probes'])) ==
                  sum(r['rendered_valid'] for r in group_by_frame[k]),
                  f'Frame rendered valid-count mismatch: {k}')
    check(int(manifest.get('num_probe_records',0)) >= len(probe_records),
          'Probe file longer than provenance-record count')
    return probe_records, list(requested.keys()), manifest, frame_lookup, requested


def _write_csv(path: Path, rows: list[dict]) -> None:
    check(bool(rows),f'Refusing empty CSV: {path}')
    fields = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=fields)
        w.writeheader();w.writerows(rows)


def _weighted_frame_corr(rows: list[dict], col: str, min_probes: int) -> tuple[float | None,int]:
    by_frame = defaultdict(list)
    for r in rows:
        if r['rendered_valid'] and r[col] is not None:
            by_frame[r['frame_id']].append(r)
    per_frame_rho = [spearman((r[col] for r in items),
                               (r['rendered_error_m'] for r in items),min_n=min_probes)
                     for items in by_frame.values()]
    valid = [r for r in per_frame_rho if r is not None]
    return average(valid), len(valid)


def summarise(probes: list[dict], selected: list[tuple[int,int]], frame_ap: dict,
              variants: list[str], min_probes: int, min_frames: int):
    by_scene_role,by_role,by_frame_role,by_frame,by_scene = [defaultdict(list) for _ in range(5)]
    for r in probes:
        v,s,f,role=r['variant'],r['scene_id'],r['frame_id'],r['role']
        by_scene_role[(v,s,role)].append(r)
        by_role[(v,role)].append(r)
        by_frame_role[(v,s,f,role)].append(r)
        by_frame[(v,s,f)].append(r)
        by_scene[(v,s)].append(r)

    per_scene_role=[]
    for (v,s,role),rows in sorted(by_scene_role.items()):
        stats=group_stats(rows,min_probes)
        entry={'variant':v,'scene_id':s,'role':role,'n_frames':len({r['frame_id'] for r in rows}),**stats}
        for proxy,short in ((PROXIES[0],'std'),(PROXIES[1],'gradient'),(PROXIES[2],'external_sigma')):
            corr,count=_weighted_frame_corr(rows,proxy,min_probes)
            entry[f'{short}_within_frame_spearman_mean']=corr
            entry[f'{short}_within_frame_count']=count
        per_scene_role.append(entry)
    per_scene = [{'variant':v,'scene_id':s,'n_frames':len({r['frame_id'] for r in rows}),
                  **group_stats(rows,min_probes)} for (v,s),rows in sorted(by_scene.items())]
    per_frame_role = [{'variant':v,'scene_id':s,'frame_id':f,'role':role,
                       'selection':frame_ap[(s,f)]['selection'],
                       **{t:float(num(frame_ap[(s,f)][t],where=f'AP.{s}.{f}.{t}')) for t in TARGETS},
                       **group_stats(rows,min_probes)}
                      for (v,s,f,role),rows in sorted(by_frame_role.items())]
    per_frame = []
    for (v,s,f),rows in sorted(by_frame.items()):
        stats=group_stats(rows,min_probes)
        role_cov={role: group_stats(by_frame_role[(v,s,f,role)],min_probes)['rendered_coverage']
                  for role in ROLES if (v,s,f,role) in by_frame_role}
        check(set(role_cov)==set(ROLES),f'Missing role in {(v,s,f)}')
        contact = (role_cov['left_contact']+role_cov['right_contact'])/2
        ap=frame_ap[(s,f)]
        per_frame.append({'variant':v,'scene_id':s,'frame_id':f,
                          'selection':ap['selection'],
                          **{t:float(num(ap[t],where=f'AP.{s}.{f}.{t}')) for t in TARGETS},
                          **stats,
                          'closing_rendered_coverage':role_cov['closing'],
                          'contact_rendered_coverage':contact,
                          'approach_rendered_coverage':role_cov['approach'],
                          'mean_proxy_std_mm':1000*average(r['proxy_depth_std_m'] for r in rows if r['rendered_valid'])
                              if average(r['proxy_depth_std_m'] for r in rows if r['rendered_valid']) is not None else None})
    per_role=[]
    for (v,role),rows in sorted(by_role.items()):
        stats=group_stats(rows,min_probes)
        matching=[r for r in per_scene_role if r['variant']==v and r['role']==role]
        per_role.append({'variant':v,'role':role,'scene_count':len(matching),**stats,
                         'scene_mean_rendered_coverage':average(r['rendered_coverage'] for r in matching),
                         'scene_mean_rendered_mae_mm':average(r['rendered_mae_mm'] for r in matching),
                         'scene_mean_within_frame_std_rho':average(r['std_within_frame_spearman_mean'] for r in matching)})

    links=[]
    for v in variants:
        rows=[r for r in per_frame if r['variant']==v]
        check(len(rows)==len(selected),f'Expected exactly one probe-summary row for each selected frame: {v}')
        for metric in METRICS:
            for target in TARGETS:
                valid=[r for r in rows if r.get(metric) is not None and r.get(target) is not None]
                rho=spearman((r[metric] for r in valid),(r[target] for r in valid),min_n=max(min_frames,3))
                scene_rhos=[]
                for s in sorted({r['scene_id'] for r in valid}):
                    x=[r for r in valid if r['scene_id']==s]
                    sr=spearman((r[metric] for r in x),(r[target] for r in x),min_n=min_frames)
                    if sr is not None:
                        scene_rhos.append(sr)
                corr,n_scenes,n_frames=pearson_within_scene(rows,metric,target)
                links.append({'variant':v,'metric':metric,'AP_outcome':target,
                              'n_frame_pairs':len(valid),
                              'pooled_spearman':rho,
                              'mean_within_scene_spearman':average(scene_rhos),
                              'scenes_with_spearman':len(scene_rhos),
                              'within_scene_centered_pearson':corr,
                              'centered_scene_count':n_scenes,
                              'centered_frame_count':n_frames,
                              'validity':'descriptive_only; outcome_selected; not causal or independent-frame inference'})
    return {'scene_role':per_scene_role,'scene':per_scene,'role':per_role,
            'frame_role':per_frame_role,'frame':per_frame,'ap_association':links}


def run(probe_dir: Path, cases_dir: Path, output_dir: Path, variants: list[str],
        min_probes: int = 15, min_frames: int = 3) -> dict:
    check(variants and len(variants)==len(set(variants)),f'Duplicate/empty variants: {variants}')
    check(min_probes>=3 and min_frames>=3,'Minimum sample thresholds must be >=3')
    check(not(output_dir.exists() and any(output_dir.iterdir())),
          f'Choose a fresh output directory; refusing overwrite: {output_dir}')
    probes,selected,manifest,frame_lookup,requested=check_inputs(probe_dir,cases_dir,variants)
    tables=summarise(probes,selected,frame_lookup,variants,min_probes,min_frames)
    # No outputs are created until all input/annotation provenance is verified.
    output_dir.mkdir(parents=True,exist_ok=True)
    names={'scene_role':'per_scene_role.csv','scene':'per_scene.csv',
           'role':'per_role.csv','frame_role':'per_frame_role.csv',
           'frame':'per_frame.csv','ap_association':'ap_associations.csv'}
    for k,path in names.items():
        _write_csv(output_dir/path,tables[k])
    provenance_files=[probe_dir/'manifest.json',probe_dir/'probe_records.csv',
                      probe_dir/'frame_probe_summary.csv',cases_dir/'selected_frames.json',
                      cases_dir/'novel_frames.csv']
    audit={'version':VERSION,'source_files':[{'file':str(p),'sha256':source_hash(p)} for p in provenance_files],
           'variants':variants,'selected_scenes':sorted({s for s,f in selected}),
           'selected_frame_ids':[{'scene_id':s,'frame_id':f} for s,f in selected],
           'probe_count':len(probes),'min_probes':min_probes,'min_frames':min_frames,
           'export_topk':int(manifest['rank_topk']),
           'limitations':['AP-selected 5 scenes/20 frames, not an unbiased Novel distribution',
                          'All probes per grasp/scene correlated; no independent-probe confidence intervals',
                          'Matched reference means require identical per-probe validity masks',
                          'Difference of two absolute errors is NOT GT-vs-sensor depth disagreement',
                          'STD/gradient are heuristic predicted-depth proxies, not calibrated sigma',
                          'Different model variants evaluate different physical grasp candidates']}
    (output_dir/'P1_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    lines=['# GVAR P1: conditional geometry-evidence audit','',
           f"Variants: {', '.join(variants)}. AP-selected scenes: {audit['selected_scenes']}; "
           f"unique frames: {len(selected)}; probe records: {len(probes)}.",
           '', '## Per-scene reference-valid probe coverage and depth error', '',
           '| Variant | Scene | Rendered valid | Both refs valid | Rendered MAE (mm) | Matched rendered MAE (mm) | Matched sensor MAE (mm) |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in tables['scene']:
        def f(key,pct=False):
            x=r.get(key)
            return 'NA' if x is None else f'{x*100:.1f}%' if pct else f'{x:.2f}'
        lines.append(f"| {r['variant']} | {r['scene_id']:04d} | "
                     f"{f('rendered_coverage',True)} | {f('both_reference_coverage',True)} | "
                     f"{f('rendered_mae_mm')} | {f('paired_rendered_mae_mm')} | {f('paired_sensor_mae_mm')} |")
    lines += ['', '## Role-conditioned reliability', '',
              '| Variant | Role | Rendered valid | Matched valid | Matched rendered MAE (mm) | Matched sensor MAE (mm) | Mean scene-wise within-frame STD rho |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in tables['role']:
        def f(key,pct=False):
            x=r.get(key)
            return 'NA' if x is None else f'{x*100:.1f}%' if pct else f'{x:.3f}'
        lines.append(f"| {r['variant']} | {r['role']} | {f('rendered_coverage',True)} | "
                     f"{f('both_reference_coverage',True)} | {f('paired_rendered_mae_mm')} | "
                     f"{f('paired_sensor_mae_mm')} | {f('scene_mean_within_frame_std_rho')} |")
    lines += ['', '## Frame-level AP associations (exploratory only)', '',
              '| Variant | Feature | AP endpoint | Pooled Spearman | Within-scene centered Pearson | Scenes |',
              '|---|---|---|---:|---:|---:|']
    for r in tables['ap_association']:
        if r['AP_outcome'] not in ('delta_AP_pp','delta_mu04_pp'):
            continue
        if r['metric'] not in ('rendered_coverage','both_reference_coverage','rendered_mae_mm',
                              'contact_rendered_coverage','closing_rendered_coverage',
                              'approach_rendered_coverage'):
            continue
        fmt=lambda x: 'NA' if x is None else f'{x:+.3f}'
        lines.append(f"| {r['variant']} | {r['metric']} | {r['AP_outcome']} | "
                     f"{fmt(r['pooled_spearman'])} | {fmt(r['within_scene_centered_pearson'])} | "
                     f"{r['centered_scene_count']} |")
    lines += ['', '## Scientific interpretation limits', '',
              '- A rendered reference and a sensor reference may have DIFFERENT valid masks. '
              'Only columns starting `paired_` compare them on identical probes; '
              'unmatched MAEs must not be subtracted.',
              '- Local depth STD and gradient are *predicted-depth-derived proxies*, '
              'not learned calibrated uncertainty. If external sigma is absent, no calibration is claimed.',
              '- AP-selected 3 worst + 1 best frames in 5 scenes create outcome-selection bias. '
              'Within-scene correlations are descriptive, not causal or population estimates.',
              '- Same physical probe identity is preserved within a variant, but independent '
              'GVAR models output different physical candidates. This analysis cannot isolate reranking.',
              '- `rendered_valid` means compatible projected foreground/pred/reference masks; '
              'it is NOT ground-truth 3D visibility, contact success or occupancy.',
              '', 'Artifacts: `per_scene_role.csv`, `per_scene.csv`, `per_role.csv`, '
              '`per_frame_role.csv`, `per_frame.csv`, `ap_associations.csv`, '
              '`P1_AUDIT.json`.']
    (output_dir/'P1_REPORT.md').write_text('\n'.join(lines)+'\n')
    return {'status':'complete','variants':variants,'scenes':len(audit['selected_scenes']),
            'frames':len(selected),'probes':len(probes), 'out':str(output_dir)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-dir',type=Path,required=True)
    p.add_argument('--cases-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['volume_rel'])
    p.add_argument('--min-probes',type=int,default=15)
    p.add_argument('--min-frames',type=int,default=3)
    a=p.parse_args(argv)
    try:
        state=run(a.probe_dir,a.cases_dir,a.output_dir,a.variants,
                  a.min_probes,a.min_frames)
    except (DataError,FileNotFoundError,KeyError,TypeError,OSError,ValueError) as exc:
        p.exit(2,f'GVAR P1 conditional analysis FAILED: {exc}\n')
    print(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
