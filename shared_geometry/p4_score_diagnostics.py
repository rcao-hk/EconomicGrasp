"""P4-0: frozen-reader margin/geometry coupling and equal-count region controls.

No training, score thresholds, new region weights or official test-set access.
Path attributions and token swaps describe this reader, not physical causality.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .p4_common import (SCHEMA, source_hashes, formal_rows, freeze_reader, select_entries,
    score_physics, tensor_inputs, family_rows, paired_summary, verify_p2,
    new_directory, write_records, finite_correlation, sha256, signed, atomic_json,
    read_cache_index, load_cached, collate_cache, scene_balanced, scene_bootstrap)
from .p3_action_geometry_diagnostics import REGIONS, pair_geometry_evidence, _sanity_check_input


def matched_masks(n_points=15, repeats=16, seed=0):
    """Fixed before observing data/Q. Equal point count AND equal height counts.

    Additional controls preserve each region subset's exact (approach,height)
    histogram, drawing opening coordinates at random. For the complete palm
    layer that control is necessarily identical, and is explicitly marked so.
    """
    if n_points < 3 or n_points > 15 or n_points % 3 or repeats < 1:
        raise ValueError('Common count must be a multiple of 3 in [3,15]')
    x, y, z = np.indices((5, 5, 3)); x, z = x.ravel(), z.ravel()
    rng = np.random.default_rng(seed)
    pools = dict(REGIONS, random_all=np.ones(75, bool))
    result = {}
    for label, pool in pools.items():
        samples = []
        for _ in range(repeats):
            selected = np.concatenate([rng.choice(np.flatnonzero(pool & (z == h)),
                                         n_points//3, replace=False) for h in range(3)])
            mask = np.zeros(75, bool); mask[selected] = True; samples.append(mask)
        result[label] = np.stack(samples)
    for label in REGIONS:
        matched = []
        for source in result[label]:
            mask = np.zeros(75, bool)
            for a in range(5):
                for h in range(3):
                    cell = (x == a) & (z == h)
                    count = int((source & cell).sum())
                    if count:
                        mask[rng.choice(np.flatnonzero(cell), count, replace=False)] = True
            matched.append(mask)
        result[label+'_xz_matched'] = np.stack(matched)
    for masks in result.values():
        if not np.all(masks.sum(1) == n_points):
            raise AssertionError('Unequal replacement cardinality')
    return result


def margin_attributions(model, predicted, clean, local, dims, steps=8):
    """Gradient*evidence-difference for each action, then subtract native.

    All actions are independently scored by the fixed point reader, so gradient
    of sum(logits) is the collection of per-action input gradients. IG uses a
    straight path in normalized physical-token space, NOT realizable 3D scenes.
    Every parameter is frozen; only input tensors receive autograd gradients.
    """
    if steps < 0 or predicted.shape != clean.shape:
        raise ValueError('Bad attribution path')
    difference = clean-predicted

    def gradient_at(t):
        x = (predicted+t*difference).detach().requires_grad_(True)
        z = score_physics(model, x, local, dims)
        gradient = torch.autograd.grad(z.sum(), x, create_graph=False)[0]
        return z.detach(), gradient.detach()

    with torch.enable_grad():
        zp, gp = gradient_at(0.)
        zc, gc = gradient_at(1.)
        avg = None
        if steps:
            avg = torch.zeros_like(difference)
            for t in (np.arange(steps)+.5)/steps:
                _, grad = gradient_at(float(t)); avg += grad/steps
    def relative(v):
        return v[:, :, 1:]-v[:, :, :1]
    endpoint = relative(gp*difference)
    endpoint_clean = relative(gc*difference)
    actual = relative(zc)-relative(zp)
    out = dict(predicted_logits=zp, clean_logits=zc, actual_margin_change=actual,
               pred_endpoint=endpoint, clean_endpoint=endpoint_clean)
    if avg is not None:
        integrated = relative(avg*difference)
        out.update(integrated=integrated,
                   completeness_residual=integrated.sum((-1,-2))-actual)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def describe_attribution(a, region):
    # Signed sum estimates margin change; absolute sum measures sensitivity.
    selected = a[region]
    return dict(signed=float(selected.sum()), abs_sum=float(np.abs(selected).sum()),
                distance_signed=float(selected[:, :2].sum()),
                state_signed=float(selected[:, 2:].sum()))


def error_sign(value, dq):
    if dq == 0: return None
    if value == 0: return .5
    return float(np.sign(value) == np.sign(dq))


def coupling_summary(records):
    fields = ('abs_margin_change', 'udf_delta_mae_mm', 'projective_delta_mae_mm',
              'pred_correct', 'clean_correct', 'true_delta_q', 'taylor_abs_error',
              'integrated_abs_error')
    def group(rows):
        metrics = {k: scene_balanced(rows, k) for k in fields}
        cells = defaultdict(list)
        for r in rows: cells[(r['scene'], r['domain'])].append(r)
        correlations = []
        for (s, d), cell in sorted(cells.items()):
            c = {}
            for field in ('udf_delta_mae_mm','projective_delta_mae_mm','taylor_abs_magnitude'):
                c[field] = finite_correlation(
                    [np.nan if r[field] is None else r[field] for r in cell],
                    [r['abs_margin_change'] for r in cell])
            correlations.append(dict(scene=s, domain=d, associations=c))
        return dict(n_pairs=len(rows), means=metrics, by_scene_domain_correlations=correlations)
    return dict(all_pairs=group(records),
        by_true_gap_count={str(i):group([r for r in records if r['abs_delta_count']==i]) for i in range(7)},
        by_pair_status={status:group([r for r in records if r['pair_status']==status])
                        for status in sorted({r['pair_status'] for r in records})},
        by_offset_mm={str(o):group([r for r in records if r['offset_mm']==o])
                      for o in sorted({r['offset_mm'] for r in records})})


def run(a):
    if min(a.batch_size, a.bootstrap, a.repeats) < 1 or a.ig_steps < 0 or a.limit < 0:
        raise ValueError('Invalid P4-0 budget')
    if a.limit and not a.smoke: raise ValueError('Partial runs require --smoke')
    if not a.smoke and not a.expected_p2:
        raise ValueError('Formal P4-0 requires the signed P2 parity reference')
    index = read_cache_index(a.index, 'validation', a.smoke)
    formal_rows(index, 'validation', a.smoke)
    model, protocol = freeze_reader(a.weight, index, a.device, a.smoke)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = new_directory(a.output)
    masks = matched_masks(a.points, a.repeats, a.seed)
    entries = select_entries(index['entries'],a.limit)
    records = dict(predicted=[], clean_visible=[]); pairs = []
    swap_records = {label:{'forward':[], 'reverse':[]} for label in masks}
    parity_error = 0.
    for start in range(0, len(entries), a.batch_size):
        sub = entries[start:start+a.batch_size]; frames = [load_cached(e) for e in sub]
        for f, e in zip(frames, sub): _sanity_check_input(f, e['row'], index)
        batch = collate_cache(frames)
        local, dims = tensor_inputs(batch, a.device)
        pred = torch.as_tensor(batch['predicted_physics'], device=a.device, dtype=torch.float32)
        clean = torch.as_tensor(batch['clean_visible_physics'], device=a.device, dtype=torch.float32)
        attrs = margin_attributions(model, pred, clean, local, dims, a.ig_steps)
        zp, zc = attrs['predicted_logits'], attrs['clean_logits']
        with torch.no_grad():
            full_swap = torch.where(torch.ones((75,1),device=a.device,dtype=torch.bool), clean, pred)
            zfull = score_physics(model, full_swap, local, dims).cpu().numpy()
        parity_error = max(parity_error, float(np.abs(zfull-zc).max()))
        if parity_error > 1e-5: raise RuntimeError('Full source swap failed logit parity')
        for b, (frame, entry) in enumerate(zip(frames, sub)):
            row = entry['row']
            for arm, z in (('predicted', zp), ('clean_visible', zc)):
                records[arm].extend(family_rows(z[b], frame, row, index['contract']['offsets_m'], frame[arm+'_physics']))
            for f in range(len(frame['family_id'])):
                for k in range(1, 7):
                    dq = int(frame['quality_count'][f,k])-int(frame['quality_count'][f,0])
                    mp, mc = float(zp[b,f,k]-zp[b,f,0]), float(zc[b,f,k]-zc[b,f,0])
                    geo = pair_geometry_evidence(frame['predicted_physics'][f], frame['clean_visible_physics'][f], k, index['contract']['tau'])
                    attribution = {}
                    for name, values in attrs.items():
                        if name not in ('pred_endpoint', 'clean_endpoint', 'integrated'): continue
                        attribution[name] = {region:describe_attribution(values[b,f,k-1], mask)
                                             for region, mask in dict(all=np.ones(75,bool), **REGIONS).items()}
                    estimate = attribution['pred_endpoint']['all']['signed']
                    r = dict(scene=int(row['scene']), domain=row['domain'], anno=int(row['anno']),
                        family=int(frame['family_id'][f]), action=k,
                        native_id=str(frame['physical_id'][f,0]), alternative_id=str(frame['physical_id'][f,k]),
                        offset_mm=float(index['contract']['offsets_m'][k]*1000),
                        true_delta_q=dq/6., abs_delta_count=abs(dq), pred_margin=mp, clean_margin=mc,
                        abs_margin_change=abs(mc-mp), sign_flip=bool(np.sign(mp)!=np.sign(mc)),
                        pred_correct=error_sign(mp,dq), clean_correct=error_sign(mc,dq),
                        udf_delta_mae_mm=geo['all']['udf_delta_mae_mm'],
                        projective_delta_mae_mm=geo['all']['projective_delta_mae_mm'],
                        taylor_abs_error=abs(estimate-(mc-mp)), taylor_abs_magnitude=abs(estimate),
                        pair_status=('quality_tie' if dq==0 else
                            ('correct' if error_sign(mp,dq)==1 else 'tie' if mp==0 else 'wrong')+'->'+
                            ('correct' if error_sign(mc,dq)==1 else 'tie' if mc==0 else 'wrong')),
                        integrated_abs_error=None,
                        geometry_by_region=geo, attribution=attribution)
                    if a.ig_steps:
                        r['integrated_abs_error'] = float(abs(attrs['completeness_residual'][b,f,k-1]))
                    pairs.append(r)
        # Average decisions over masks; never average scores then run argmax.
        for label, draws in masks.items():
            for direction in ('forward', 'reverse'):
                baseline = zp if direction=='forward' else zc
                primary, replacement = (pred, clean) if direction=='forward' else (clean, pred)
                gains, moves, switches = [], [], []
                q = batch['quality_count']/6.
                base_choice = baseline.argmax(-1)
                base_q = np.take_along_axis(q, base_choice[...,None], -1)[...,0]
                for mask in draws:
                    m = torch.as_tensor(mask, device=a.device).reshape(1,1,1,75,1)
                    with torch.no_grad():
                        z = score_physics(model, torch.where(m,replacement,primary), local,dims).cpu().numpy()
                    choice = z.argmax(-1)
                    value = np.take_along_axis(q, choice[...,None], -1)[...,0]
                    gains.append(value-base_q); moves.append(choice!=0); switches.append(choice!=base_choice)
                gm, mm, sm = np.mean(gains,0), np.mean(moves,0), np.mean(switches,0)
                for b, (fr,e) in enumerate(zip(frames,sub)):
                    for f, family in enumerate(fr['family_id']):
                        swap_records[label][direction].append(dict(
                            scene=int(e['row']['scene']),domain=e['row']['domain'],anno=int(e['row']['anno']),
                            family=int(family), gain=float(gm[b,f]), move=float(mm[b,f]),
                            switch=float(sm[b,f])))
        print(f'[P4-0] {min(start+a.batch_size,len(entries))}/{len(entries)} frames', flush=True)
    checked = verify_p2(a.expected_p2, index, a.weight, records, bool(a.limit))
    region_results = {}
    for label, directions in swap_records.items():
        region_results[label] = {}
        for direction, rr in directions.items():
            result = dict(mean=scene_balanced(rr,'gain'),
                          paired_ci=scene_bootstrap(rr,metric='gain',n_boot=a.bootstrap,seed=a.seed),
                          move=scene_balanced(rr,'move'), switch=scene_balanced(rr,'switch'))
            if label in REGIONS:
                for control in ('random_all', label+'_xz_matched'):
                    result['vs_'+control] = scene_bootstrap(rr,swap_records[control][direction],
                                                   metric='gain',n_boot=a.bootstrap,seed=a.seed)
            region_results[label][direction] = result
    report = dict(schema=SCHEMA,study='P4-0',smoke=bool(a.smoke),n_frames=len(entries),
        n_families=len(records['predicted']),n_pairs=len(pairs),
        inputs=dict(index_sha256=sha256(a.index),index_fingerprint=index['fingerprint'],
                    weight_sha256=sha256(a.weight),reader_protocol=protocol),
        source_hashes=source_hashes(),official_ap_not_run=True,trained_new_weights=False,
        full_swap_logit_error=parity_error,published_p2_gain_parity_checked=checked,
        scores={arm:paired_summary(v,None if arm=='predicted' else records['predicted'],a.bootstrap,a.seed)
                for arm,v in records.items()},
        coupling=coupling_summary(pairs), equal_count_regions=region_results,
        mask_design=dict(points=a.points,repeats=a.repeats,seed=a.seed,
             indices={k:[np.flatnonzero(m).tolist() for m in v] for k,v in masks.items()},
             palm_matched_is_self=bool(np.array_equal(masks['palm_proxy'],masks['palm_proxy_xz_matched'])),
             aggregation='argmax each draw, then per-family mean, then paired-scene bootstrap'),
        ig_steps=a.ig_steps,
        warning='Normalized-token gradients/path integrals and hybrid swaps are model diagnostics, not physical causes. '
                'Soft state interpolation is off-manifold. CIs condition on fixed weights and fixed masks; '
                'mask repetitions and families are not independent scenes; no multiple-comparison correction.')
    report['artifacts'] = dict(pairs=write_records(output/'pairs.jsonl.gz',pairs),
        families={arm:write_records(output/(arm+'_families.jsonl.gz'),rr) for arm,rr in records.items()},
        swaps={label:{d:write_records(output/(label+'_'+d+'.jsonl.gz'),rr) for d,rr in dd.items()}
               for label,dd in swap_records.items()})
    atomic_json(output/'summary.json',signed(report))
    text = ['# P4-0：评分差与几何差、等点数替换', '',
            '冻结P2权重；指标为离线Q，不是AP。区域名称是模板代理。', '',
            '|区域（同为%d点）|Pred→Clean ΔQ pp|Clean→Pred ΔQ pp|' % a.points,
            '|---|---:|---:|']
    for name in masks:
        text.append(f"|{name}|{region_results[name]['forward']['mean']*100:.5f}|{region_results[name]['reverse']['mean']*100:.5f}|")
    text += ['', '完整区间、分真实质量差/位移的关联、积分残差和输入身份见 summary.json。',
             '随机对照保持高度分层；xz匹配对照进一步保持approach/height直方图。',
             '掌部15点的xz对照必然相同，不能作为独立空间反证。']
    (output/'RESULTS_CN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True);p.add_argument('--weight',required=True)
    p.add_argument('--expected-p2',default='');p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--device',default='cuda:0')
    p.add_argument('--points',type=int,default=15);p.add_argument('--repeats',type=int,default=16)
    p.add_argument('--ig-steps',type=int,default=8);p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--smoke',action='store_true');run(p.parse_args())


if __name__=='__main__':main()
