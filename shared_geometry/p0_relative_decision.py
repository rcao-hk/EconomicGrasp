"""P0: explain why good all-pair ranking fails native-relative action choice.

Read-only validation diagnostic. No checkpoint fitting, GT in model inputs, test
access, post-hoc threshold selection, or assumptions about physical occupancy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .data import read_index, read_frame, collate_inputs, sha256, signed, atomic_json
from .experiment import load_model, predict
from .geometry import support_points, VisibleSurfaceTeacher


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60., 60.)))


def metrics_for_family(z, counts, offsets, native_logit=None, state=None):
    """All scalar qualities use the existing six-level exact-action target."""
    z = np.asarray(z, dtype=np.float64); counts = np.asarray(counts, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=float)
    if z.ndim != 1 or len(z) != len(offsets) or z.shape != counts.shape or not np.isfinite(z).all():
        raise ValueError('Invalid family logits/labels/offset contract')
    if not (offsets[0] == 0 and (counts >= 0).all() and (counts <= 6).all()):
        raise ValueError('Native index zero and valid 0..6 quality are required')
    q = counts.astype(float) / 6
    p = sigmoid(z)
    k = int(np.argmax(z))  # Native wins exact ties.
    true_delta = q[1:] - q[0]
    estimated_delta = p[1:] - p[0]
    best_other = int(np.argmax(z[1:]) + 1)
    _, j = np.triu_indices(len(z), 1)
    # For the all-pair diagnostic count only ground-truth non-ties.
    ii, jj = np.triu_indices(len(z), 1)
    not_tie = q[ii] != q[jj]
    correct = np.sign(z[ii] - z[jj]) * np.sign(q[ii] - q[jj])
    rel_mask = true_delta != 0
    rel_correct = np.sign(estimated_delta[rel_mask])*np.sign(true_delta[rel_mask])
    alt = []
    for u in range(1, len(z)):
        truth = 'benefit' if true_delta[u-1] > 0 else 'harm' if true_delta[u-1] < 0 else 'tie'
        alt.append(dict(offset_mm=round(float(offsets[u]*1000), 6), truth=truth,
                        native_q=float(q[0]), q=float(q[u]), delta_q=float(true_delta[u-1]),
                        pred_delta=float(estimated_delta[u-1]), pred_above_native=bool(z[u] > z[0]),
                        wrong_improvement=bool(z[u] > z[0] and q[u] < q[0]),
                        missed_improvement=bool(z[u] <= z[0] and q[u] > q[0])))
    result = dict(native_q=float(q[0]), selected_q=float(q[k]), oracle_q=float(q.max()),
                  gain=float(q[k]-q[0]), oracle_gain=float(q.max()-q[0]),
                  move=bool(k!=0), choice=k, choice_offset_mm=round(float(offsets[k]*1000), 6),
                  rescue=bool(q[k] > q[0]), harm=bool(q[k] < q[0]), equal=bool(q[k] == q[0]),
                  opportunity=bool(q.max() > q[0]),
                  top_margin=float(z[k] - max(np.delete(z, k))) if len(z)>1 else None,
                  max_alt_gap=float(z[best_other]-z[0]),
                  native_probability=float(p[0]), selected_probability=float(p[k]),
                  native_abs_error=float(abs(p[0]-q[0])),
                  selected_abs_error=float(abs(p[k]-q[k])),
                  native_squared_error=float((p[0]-q[0])**2),
                  alt_squared_error=float(np.mean((p[1:]-q[1:])**2)),
                  native_signed_bias=float(p[0]-q[0]),
                  alt_signed_bias=float(np.mean(p[1:]-q[1:])),
                  allpair_correct=float(np.mean(np.where(correct[not_tie]==0,.5,correct[not_tie]>0))) if not_tie.any() else None,
                  nativepair_correct=float(np.mean(np.where(rel_correct==0,.5,rel_correct>0))) if rel_mask.any() else None,
                  nativepair_n=int(rel_mask.sum()),
                  alternatives=alt)
    if state is not None:
        state = np.asarray(state)
        if state.shape != (len(z), 75):
            raise ValueError('Expected a predicted-geometry state for 75 probes per action')
        result.update(native_unknown_fraction=float((state[0]==3).mean()),
                      selected_unknown_fraction=float((state[k]==3).mean()),
                      selected_behind_fraction=float((state[k]==2).mean()),
                      selected_surface_fraction=float((state[k]==1).mean()))
    if native_logit is not None:
        prior = float(sigmoid(native_logit))
        result.update(native_prior_probability=prior,
                      native_prior_abs_error=float(abs(prior-q[0])),
                      native_prior_squared_error=float((prior-q[0])**2),
                      native_prior_signed_bias=float(prior-q[0]))
    return result


def scene_balanced(records, metric, *, predicate=None):
    """Equal RGB-domain weight, then equal scene, then family; do not IID bootstrap families."""
    grouped=defaultdict(list)
    for r in records:
        if predicate is not None and not predicate(r):continue
        v = r.get(metric)
        if v is None:continue
        grouped[(r['domain'], int(r['scene']))].append(float(v))
    domains=sorted({k[0] for k in grouped})
    means=[]
    for domain in domains:
        values=[np.mean(v) for (d,_),v in sorted(grouped.items()) if d==domain]
        if values:means.append(float(np.mean(values)))
    return float(np.mean(means)) if means else None


def scene_bootstrap(records_a, records_b=None, metric='gain', n_boot=2000, seed=0):
    """Paired scene bootstrap, requiring every scene/domain to exist in both modes."""
    def per_scene(recs):
        grouped=defaultdict(list)
        for r in recs:
            if r.get(metric) is not None:grouped[(int(r['scene']),r['domain'])].append(float(r[metric]))
        return {k:float(np.mean(v)) for k,v in grouped.items()}
    a=per_scene(records_a);b=per_scene(records_b) if records_b is not None else None
    if b is not None and set(a)!=set(b):raise ValueError('Unpaired validation scene/domain coverage')
    sceneids=sorted({k[0] for k in a})
    if not sceneids or n_boot<1: raise ValueError('Empty scenes or invalid bootstrap count')
    values=[]
    for sid in sceneids:
        domains=sorted(k[1] for k in a if k[0]==sid)
        x=[a[(sid,d)]-(b[(sid,d)] if b is not None else 0.) for d in domains]
        if not x:raise ValueError('Empty scene')
        values.append(float(np.mean(x)))
    gen=np.random.default_rng(seed);v=np.asarray(values)
    draws=v[gen.integers(0,len(v),size=(n_boot,len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                n_scenes=len(v),unit='paired scenes, within-scene RGB domains equally weighted; fixed weights')


def summarize(records):
    keys=('gain','move','rescue','harm','equal','opportunity','oracle_gain','top_margin',
          'native_abs_error','selected_abs_error','native_squared_error','alt_squared_error',
          'native_signed_bias','alt_signed_bias','native_prior_squared_error',
          'native_prior_abs_error','nativepair_correct','allpair_correct',
          'selected_unknown_fraction','selected_behind_fraction')
    summary={k:scene_balanced(records,k) for k in keys}
    summary['headroom_recovery']=((summary['gain']/summary['oracle_gain'])
                                  if summary['oracle_gain'] is not None and summary['oracle_gain']>0 else None)
    summary['false_improvement_among_moves']=scene_balanced(records,'harm',predicate=lambda r:r['move'])
    summary['missed_opportunity_given_opportunity']=scene_balanced(records,'no_rescue',predicate=lambda r:r['opportunity'])
    offsets=sorted({a['offset_mm'] for r in records for a in r['alternatives']})
    by_offset={}
    for offset in offsets:
        sub=[dict(r,**a) for r in records for a in r['alternatives'] if a['offset_mm']==offset]
        by_offset[str(offset)]=dict(pair_n=len(sub),true_benefit=scene_balanced(sub,'true_benefit'),
            wrong_improvement=scene_balanced(sub,'wrong_improvement'),
            missed_improvement=scene_balanced(sub,'missed_improvement'),
            predicted_above_native=scene_balanced(sub,'pred_above_native'),
            delta_q=scene_balanced(sub,'delta_q'),pred_delta=scene_balanced(sub,'pred_delta'))
    summary['by_offset_mm']=by_offset
    buckets={}
    for q0 in range(7):
        sub=[r for r in records if r['native_count']==q0]
        buckets[str(q0)]=dict(n=len(sub),selected_gain=scene_balanced(sub,'gain'),
                             move=scene_balanced(sub,'move'),harm=scene_balanced(sub,'harm'),
                             rescue=scene_balanced(sub,'rescue'),opportunity=scene_balanced(sub,'opportunity'))
    summary['by_native_quality_count']=buckets
    margin_bins=[('non_move_preferred',lambda r:r['max_alt_gap']<=0),
                 ('low_positive_0_to_0.1',lambda r:0<r['max_alt_gap']<=.1),
                 ('medium_positive_0.1_to_0.5',lambda r:.1<r['max_alt_gap']<=.5),
                 ('high_positive_above_0.5',lambda r:r['max_alt_gap']>.5)]
    summary['by_native_vs_best_alternative_margin']={
        tag:dict(n=sum(pred(r) for r in records),
                 gain=scene_balanced(records,'gain',predicate=pred),
                 harm=scene_balanced(records,'harm',predicate=pred),
                 rescue=scene_balanced(records,'rescue',predicate=pred))
        for tag,pred in margin_bins}
    positions=sorted({r['choice_offset_mm'] for r in records})
    summary['by_selected_offset_mm']={str(offset):
        dict(n=sum(r['choice_offset_mm']==offset for r in records),
             gain=scene_balanced(records,'gain',predicate=lambda r:r['choice_offset_mm']==offset),
             harm=scene_balanced(records,'harm',predicate=lambda r:r['choice_offset_mm']==offset),
             rescue=scene_balanced(records,'rescue',predicate=lambda r:r['choice_offset_mm']==offset))
        for offset in positions}
    if any(r.get('selected_unknown_fraction') is not None for r in records):
        intervals=[('known_le_0.05',lambda r:r.get('selected_unknown_fraction',1)<=.05),
                   ('mixed_0.05_to_0.25',lambda r:.05<r.get('selected_unknown_fraction',-1)<.25),
                   ('mostly_unknown_ge_0.25',lambda r:r.get('selected_unknown_fraction',-1)>=.25)]
        summary['by_selected_unobserved_ratio']={
            tag:dict(n=sum(pred(r) for r in records),
                     gain=scene_balanced(records,'gain',predicate=pred),
                     harm=scene_balanced(records,'harm',predicate=pred),
                     rescue=scene_balanced(records,'rescue',predicate=pred))
            for tag,pred in intervals}
    scene_rows=defaultdict(list)
    for r in records:scene_rows[(r['domain'],r['scene'])].append(r)
    summary['by_scene_domain']={f'{d}:scene_{int(s):04d}':
        dict(n=len(rr),gain=float(np.mean([r['gain'] for r in rr])),
             harm=float(np.mean([r['harm'] for r in rr])),
             rescue=float(np.mean([r['rescue'] for r in rr])))
        for (d,s),rr in sorted(scene_rows.items())}
    return summary


def run(a):
    idx=read_index(a.index,'validation',a.smoke)
    weight_specs={}
    for item in a.weight:
        if '=' not in item:raise ValueError('Use --weight name=/absolute/best.pt')
        key,path=item.split('=',1)
        if key in weight_specs or not key: raise ValueError('Duplicate/empty model name')
        weight_specs[key]=path
    if not weight_specs:raise ValueError('At least one checkpoint required')
    if a.batch_size<1 or a.bootstrap<1 or a.limit<0:raise ValueError('Invalid diagnostic budget')
    if a.limit and not a.smoke:raise ValueError('Partial validation P0 requires --smoke')
    modes={};model_ids={}
    for label,path in weight_specs.items():
        model,prot=load_model(path,a.device,a.smoke)
        for k in ('source','plan','offsets_m'):
            if prot[k]!=idx['contract'][k]:raise ValueError(f'Checkpoint {label} does not match cache {k}')
        if prot['val_fingerprint']!=idx['fingerprint']:raise ValueError('Validation index identity mismatch')
        modes[label]=(model,prot['mode']);model_ids[label]=dict(path=str(Path(path).resolve()),sha256=sha256(path),mode=prot['mode'])
    records={label:[] for label in modes}
    entries=idx['entries'][:a.limit] if a.limit else idx['entries']
    offsets=idx['contract']['offsets_m']
    for start in range(0,len(entries),a.batch_size):
        block=entries[start:start+a.batch_size]
        frames=[read_frame(e) for e in block]
        batch=frames[0] if len(frames)==1 else collate_inputs(frames)
        for label,(model,mode) in modes.items():
            with torch.inference_mode():
                z,_=predict(model,mode,batch,a.device)
            all_z=z.detach().cpu().numpy()
            if len(frames)==1:all_z=all_z[None]
            for j,(e,f) in enumerate(zip(block,frames)):
                state=None
                if a.visibility:
                    # This is a diagnostic *predicted* visible-state readout, not GT.
                    bank=torch.as_tensor(f['bank'])
                    sites,_=support_points(bank)
                    # Only 7 physical hypotheses per family; KDTree built once.
                    teacher=VisibleSurfaceTeacher(f['pred_depth'],f['K'],idx['contract']['tau'])
                    state=teacher.label(sites.reshape(-1,3).numpy())['state'].reshape(*bank.shape[:2],75)
                for fam in range(len(f['bank'])):
                    base_p = float(np.clip(f['bank'][fam,0,0], 1e-7, 1-1e-7))
                    base_logit = float(np.log(base_p/(1-base_p)))
                    s=metrics_for_family(all_z[j,fam],f['quality_count'][fam],offsets,
                                         native_logit=base_logit,
                                         state=None if state is None else state[fam])
                    s.update(scene=int(e['row']['scene']),domain=e['row']['domain'],anno=int(e['row']['anno']),
                             family=int(f['family_id'][fam]),native_count=int(f['quality_count'][fam,0]),
                             no_rescue=bool(s['opportunity'] and not s['rescue']))
                    for alt in s['alternatives']:
                        alt['true_benefit']=bool(alt['delta_q']>0)
                    records[label].append(s)
    summaries={label:summarize(rr) for label,rr in records.items()}
    bootstrap={label:scene_bootstrap(rr,metric='gain',n_boot=a.bootstrap,seed=a.seed)
               for label,rr in records.items()}
    comparisons={}
    names=sorted(records)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            comparisons[f'{left}-{right}']=scene_bootstrap(records[left],records[right],metric='gain',n_boot=a.bootstrap,seed=a.seed)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    atomic_json(out,signed(dict(schema='shared-visible-geometry-p0-v1',partition='validation',
             validation_index_sha256=sha256(a.index),validation_fingerprint=idx['fingerprint'],
             truncated_smoke=bool(a.limit),checkpoint_sources=model_ids,
             summaries=summaries,bootstrap=bootstrap,comparisons=comparisons,
             per_family=records,metrics_not_official_ap=True,
             note='P0: all-pair quality vs native-relative decision and max-selection; ties are not errors; labels used only offline.')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True);p.add_argument('--weight',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--visibility',action='store_true')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()