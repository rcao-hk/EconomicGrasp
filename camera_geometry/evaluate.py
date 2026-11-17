"""Same-ray geometry metrics and verified same-surface distance response; never AP."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from .data import inputs_for_views, common_queries, stack_inputs, load_reference, digest
from .geometry import transform_uv, sample, inside, rays, height_from_z

VIEWS = ('full','roi','workspace')


def grouped_mean_ci(rows, key, bootstrap=1000, seed=0):
    grouped = defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row.get(key) is not None: grouped[row['group']][row['domain']].append(row[key])
    values = [np.mean([np.mean(v) for v in ds.values()]) for ds in grouped.values()]
    if not values: return {'mean':None,'ci95':None,'n_groups':0}
    a = np.asarray(values); rng = np.random.default_rng(seed)
    means = np.mean(a[rng.integers(len(a),size=(bootstrap,len(a)))],axis=1)
    return {'mean':float(a.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'n_groups':len(a),'weak_group_count':len(a)<10}


def summarize_rows(rows, bootstrap=1000):
    out = {}
    for crop in VIEWS+('cross_crop',):
        part = [r for r in rows if r['crop']==crop]
        keys = ('mae_mm','bias_mm','xyz_mae_mm','invalid_fraction','eligible_fraction',
                'gt_out_of_range_fraction','height_mae_mm','gt_height_out_of_range_fraction') if crop!='cross_crop' else ('range_mm','p95_range_mm')
        out[crop] = {k:grouped_mean_ci(part,k,bootstrap) for k in keys}
    return out


def distance_response(tracks, min_span_m=.02):
    pool = defaultdict(list)
    truth = {}
    for t in tracks:
        physical = (t['group'],t['track'])
        if physical in truth and not np.allclose(truth[physical],t['xyz_table'],atol=1e-6):
            raise ValueError('Same track id describes different physical points')
        truth[physical] = t['xyz_table']
        pool[(t['group'],t['domain'],t['condition'],t['crop'],t['track'])].append(t)
    aggregates = defaultdict(lambda:[0.,0.,0])
    for key, ts in pool.items():
        x,y = np.asarray([[t['target_z_m'],t['pred_z_m']] for t in ts]).T
        if len(x)<2 or np.ptp(x)<min_span_m: continue
        dx,dy = x-x.mean(),y-y.mean()
        a = aggregates[key[:4]]; a[0] += float(dx@dy); a[1] += float(dx@dx); a[2] += 1
    rows = [dict(group=k[0],domain=k[1],condition=k[2],crop=k[3],beta=v[0]/v[1],n_tracks=v[2])
            for k,v in aggregates.items() if v[1]>0]
    return {'status':'AVAILABLE' if rows else 'NOT_AVAILABLE_no_verified_tracks_with_sufficient_span',
        'definition':'within-track centered covariance/variance, then equal group/domain aggregation',
        'min_true_span_m':min_span_m,'rows':rows,
        'by_crop':{c:grouped_mean_ci([r for r in rows if r['crop']==c],'beta') for c in VIEWS}}


@torch.no_grad()
def evaluate_model(model, entries, cfg, device):
    model.eval(); rows=[]; tracks=[]; track_rejected=0
    hw = tuple(cfg['image_hw']); n = cfg['eval_rays']
    for e in entries:
        views = inputs_for_views(e,VIEWS,hw)
        seed = int(digest([cfg['seed'],e['id'],'evaluation'])[:16],16)
        uv,target,coverage = common_queries(e,views,hw,n,np.random.default_rng(seed))
        inp = stack_inputs(views,device)
        out = model(**inp)
        crop_uv = torch.stack([transform_uv(uv,v['A']) for v in views]).to(device)
        zs = sample(out['z'][:,None],crop_uv)[...,0].cpu()
        eligibility = sample(out['eligible'][:,None].float(),crop_uv,mode='nearest')[...,0].cpu()>.5
        common = eligibility.all(0)
        if not common.any(): raise ValueError('No jointly eligible eval rays: '+e['id'])
        if not torch.isfinite(zs).all(): raise FloatingPointError('Nonfinite eval prediction')
        rnorm = rays(uv,torch.tensor(e['K'],dtype=torch.float32)).norm(dim=-1)
        height_target = None
        if 'plane' in views[0]:
            height_target = height_from_z(target[None],uv[None],torch.tensor(e['K'],dtype=torch.float32)[None],
                                         views[0]['plane'][None])[0]
        for i,c in enumerate(VIEWS):
            height_error = None; height_outside = None
            if height_target is not None:
                hp = height_from_z(zs[i:i+1],uv[None],torch.tensor(e['K'],dtype=torch.float32)[None],views[0]['plane'][None])[0]
                height_error = float((hp[common]-height_target[common]).abs().mean()*1000)
                height_outside = float(((height_target[common]<cfg['height_min_m'])|
                    (height_target[common]>cfg['height_max_m'])).float().mean())
            err = zs[i,common]-target[common]
            rows.append(dict(id=e['id'],group=e['group'],domain=e['domain'],crop=c,
                condition=e.get('condition','unspecified'),distance_m=e.get('distance_m'),
                mae_mm=float(err.abs().mean()*1000),bias_mm=float(err.mean()*1000),
                p95_error_mm=float(torch.quantile(err.abs(),.95)*1000),
                xyz_mae_mm=float((err.abs()*rnorm[common]).mean()*1000),
                invalid_fraction=float((zs[i,common]<=0).float().mean()),
                eligible_fraction=float(common.float().mean()),common_foreground_fraction=coverage,
                gt_out_of_range_fraction=float((target[common]>=cfg['max_depth_m']).float().mean()),
                n_rays=int(common.sum()),height_mae_mm=height_error,gt_height_out_of_range_fraction=height_outside))
        spread = zs[:,common].amax(0)-zs[:,common].amin(0)
        rows.append(dict(id=e['id'],group=e['group'],domain=e['domain'],crop='cross_crop',
            condition=e.get('condition','unspecified'),distance_m=e.get('distance_m'),
            range_mm=float(spread.mean()*1000),p95_range_mm=float(torch.quantile(spread,.95)*1000)))
        if e.get('tracks'):
            gt,fg = load_reference(e)
            T = np.asarray(e['T_table_from_cam']); K = np.asarray(e['K'])
            for t in e['tracks']:
                x = (np.asarray(t['xyz_table'])-T[:3,3]) @ T[:3,:3]
                if x[2]<=0: track_rejected+=1; continue
                projected = K @ x; p = projected[:2]/projected[2]
                q = np.asarray(t['uv']); u,v = np.rint(q).astype(int)
                valid = (np.linalg.norm(p-q)<.05 and 0<=u<gt.shape[1] and 0<=v<gt.shape[0])
                if valid: valid = bool(fg[v,u] and abs(float(gt[v,u])-x[2])<=cfg.get('track_tolerance_m',.005))
                loc = torch.stack([transform_uv(torch.tensor(q,dtype=torch.float32)[None],vw['A']) for vw in views])
                if not valid or not inside(loc,hw,margin=1.).all(): track_rejected+=1; continue
                loc = loc.to(device)
                ok = sample(out['eligible'][:,None].float(),loc,mode='nearest')[...,0]>.5
                if not ok.all(): track_rejected+=1; continue
                pred = sample(out['z'][:,None],loc)[:,0,0].cpu().numpy()
                for c,z in zip(VIEWS,pred):
                    tracks.append(dict(group=e['group'],domain=e['domain'],condition=e.get('condition','unspecified'),
                        crop=c,track=t['id'],xyz_table=t['xyz_table'],target_z_m=float(x[2]),pred_z_m=float(z),id=e['id']))
    cells = {}
    for condition,distance in sorted({(r['condition'],str(r['distance_m'])) for r in rows}):
        subset = [r for r in rows if r['condition']==condition and str(r['distance_m'])==distance]
        cells[f'{condition}/distance={distance}'] = summarize_rows(subset,cfg['bootstrap'])
    return {'metrics':summarize_rows(rows,cfg['bootstrap']), 'condition_distance_cells':cells,
        'rows':rows,'distance_response':distance_response(tracks,cfg.get('min_track_span_m',.02)),
        'track_rows':tracks,'rejected_track_observations':track_rejected,
        'official_ap_not_run':True,'metric_scope':'sampled common foreground rays; group-cluster exploratory intervals'}
