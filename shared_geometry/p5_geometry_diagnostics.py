"""P5-0: why improved 2D depth does not improve action-query geometry.

Read-only analysis of P4 frozen-reader evaluation: whole-frame vs projected
support/edges, 75-site UDF/projective/state, action-relative deltas, nearest
visible-surface correspondence and matched physical action decisions.
Clean depths and exact Q are used only for offline diagnostics.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from .data import (sha256,signed,load_signed,atomic_json,frame_key,read_index)
from .p1_clean_visible import read_cache_index,load_cached,collate_cache
from .p4_depth_data import read_depth_index
from .p4_common import (formal_rows,freeze_reader,score_physics,paired_summary,
                        scene_balanced,scene_bootstrap,new_directory,write_records,
                        finite_correlation)
from .p3_action_geometry_diagnostics import REGIONS
from .geometry import support_points,pixels_and_valid,VisibleSurfaceTeacher
from .p4_common import select_entries
from .p0_relative_decision import metrics_for_family

SCHEMA = 'shared-visible-geometry-p5-geometry-v1'


def depth_edges(clean, threshold_m=.02):
    """1-pixel visible surface edge: missing/valid or >threshold adjacent depth."""
    if clean.ndim!=2 or threshold_m<=0:raise ValueError('Finite HxW depth and positive threshold required')
    valid=clean>1e-6;edges=np.zeros_like(valid)
    for axis in (0,1):
        if axis==0:
            a,b=clean[:-1],clean[1:];av,bv=valid[:-1],valid[1:]
            diff=(av!=bv) | (av&bv&(np.abs(a-b)>threshold_m))
            edges[:-1] |= diff;edges[1:] |= diff
        else:
            a,b=clean[:,:-1],clean[:,1:];av,bv=valid[:,:-1],valid[:,1:]
            diff=(av!=bv) | (av&bv&(np.abs(a-b)>threshold_m))
            edges[:,:-1] |= diff;edges[:,1:] |= diff
    return edges


def project_support(xyz,K,shape,radius=2):
    uv,valid=pixels_and_valid(xyz,K,shape)
    out=np.zeros(shape,dtype=bool)
    ij=np.rint(uv[valid]).astype(np.int64)
    if len(ij):out[ij[:,1],ij[:,0]]=True
    return binary_dilation(out,iterations=radius) if radius else out


def pixel_split(pred,corr,clean,support,K,edge_threshold_m=.02,radius=2):
    """Four exclusive support/edge bins on the *same* triple-valid pixels."""
    for z in (pred,corr,clean):
        if z.ndim!=2 or z.shape!=pred.shape or not np.isfinite(z).all() or (z<0).any():
            raise ValueError('Finite, nonnegative and aligned depth maps required')
    valid=(pred>1e-6)&(corr>1e-6)&(clean>1e-6)
    edge=depth_edges(clean,edge_threshold_m)
    local=project_support(support,K,clean.shape,radius)
    ae=np.abs(pred-clean)*1000;be=np.abs(corr-clean)*1000;improvement=ae-be
    regions={'full':np.ones_like(valid),
             'projected_support':local,'away_from_support':~local,
             'support_and_edge':local&edge,'support_and_smooth':local&~edge,
             'away_and_edge':~local&edge,'away_and_smooth':~local&~edge,
             'depth_edge':edge,'depth_smooth':~edge}
    out={}
    for name,mask in regions.items():
        chosen=valid&mask
        out[name]=dict(n=int(chosen.sum()),image_fraction=float(chosen.mean()),
            pred_mae_mm=float(ae[chosen].mean()) if chosen.any() else None,
            corrected_mae_mm=float(be[chosen].mean()) if chosen.any() else None,
            improvement_mm=float(improvement[chosen].mean()) if chosen.any() else None,
            corrected_wins_fraction=float((improvement[chosen]>0).mean()) if chosen.any() else None,
            correction_abs_mm=float(np.abs(corr[chosen]-pred[chosen]).mean()*1000) if chosen.any() else None)
    out['metadata']=dict(valid_fraction=float(valid.mean()),support_pixel_fraction=float(local.mean()),
        edge_pixel_fraction=float(edge.mean()),edge_threshold_m=edge_threshold_m,
        dilation_radius_px=radius)
    return out


def nearest_pixel_drift(pred,corr,xyz,K):
    """Find whether the *closest 3D visible surface sample* changes image pixel.

    This index drift is not necessarily a model error, only a KDTree diagnostic.
    """
    def nn(depth):
        v,u=np.where(depth>1e-6)
        if not len(u):raise ValueError('No visible surface')
        Kinv=np.linalg.inv(K)
        pix=np.stack((u,v,np.ones_like(u)),1)
        point=(pix@Kinv.T)*depth[v,u,None]
        inds=cKDTree(point).query(xyz,k=1,workers=1)[1]
        return np.stack((u[inds],v[inds]),1)
    u=nn(pred);v=nn(corr)
    distance=np.linalg.norm(u-v,axis=1)
    return dict(n_points=len(distance),changed_pixel_fraction=float((distance>0).mean()),
                mean_nearest_pixel_shift=float(distance.mean()),p95_nearest_pixel_shift=float(np.quantile(distance,.95)))


def geometry_regional(pred,corr,clean,tau):
    """Errors on fixed common masks, separately absolute and action-relative."""
    if pred.shape!=corr.shape or pred.shape!=clean.shape or pred.ndim!=4 or pred.shape[-2:]!=(75,6):
        raise ValueError('Expected aligned [F,K,75,6] physical geometry')
    state=[x[...,2:].argmax(-1) for x in (pred,corr,clean)]
    known=np.logical_and.reduce([s!=3 for s in state]);relation=known[:,1:]&known[:,:1]
    near=(np.minimum(clean[:,1:,:,0],clean[:,:1,:,0])<=.25)&relation
    result={}
    mm=tau*1000
    for arm,a in (('predicted',pred),('corrected',corr)):
        out={}
        for region,sites in dict(all=np.ones(75,bool),**REGIONS).items():
            abs_mask=known&sites[None,None,:]
            rel_mask=relation&sites[None,None,:]
            near_mask=near&sites[None,None,:]
            for channel,name in ((0,'udf'),(1,'projective')):
                abse=np.abs(a[...,channel]-clean[...,channel])*mm
                rele=np.abs((a[:,1:,:,channel]-a[:,:1,:,channel])-
                            (clean[:,1:,:,channel]-clean[:,:1,:,channel]))*mm
                out.setdefault(region,{})[name+'_mae_mm']=float(abse[abs_mask].mean()) if abs_mask.any() else None
                out[region][name+'_relative_mae_mm']=float(rele[rel_mask].mean()) if rel_mask.any() else None
                out[region][name+'_near_relative_mae_mm']=float(rele[near_mask].mean()) if near_mask.any() else None
            out[region]['state_agreement']=float((state[0][...,sites]==state[2][...,sites]).mean()) if arm=='predicted' else float((state[1][...,sites]==state[2][...,sites]).mean())
            out[region]['common_fraction']=float(abs_mask[...,sites].mean())
        result[arm]=out
    return result


def family_geometry_features(pred,corr,clean,tau):
    """Per-family signed improvement; keeps null for empty valid masks."""
    known=np.logical_and.reduce([a[...,2:].argmax(-1)!=3 for a in (pred,corr,clean)])
    rel=known[:,1:] & known[:,:1]
    out=[]
    for i in range(len(pred)):
        row={}
        for channel,name in ((0,'udf'),(1,'projective')):
            p=np.abs((pred[i,1:,:,channel]-pred[i,:1,:,channel])-
                     (clean[i,1:,:,channel]-clean[i,:1,:,channel]))*tau*1000
            c=np.abs((corr[i,1:,:,channel]-corr[i,:1,:,channel])-
                     (clean[i,1:,:,channel]-clean[i,:1,:,channel]))*tau*1000
            m=rel[i]
            row['relative_'+name+'_error_pred_mm']=float(p[m].mean()) if m.any() else None
            row['relative_'+name+'_error_corrected_mm']=float(c[m].mean()) if m.any() else None
            row['relative_'+name+'_error_improvement_mm']=float((p[m]-c[m]).mean()) if m.any() else None
        for region,sites in REGIONS.items():
            m=rel[i]&sites[None,:]
            p=np.abs((pred[i,1:,:,0]-pred[i,:1,:,0])-
                     (clean[i,1:,:,0]-clean[i,:1,:,0]))*tau*1000
            c=np.abs((corr[i,1:,:,0]-corr[i,:1,:,0])-
                     (clean[i,1:,:,0]-clean[i,:1,:,0]))*tau*1000
            row[region+'_relative_udf_improvement_mm']=float((p[m]-c[m]).mean()) if m.any() else None
        out.append(row)
    return out


def validate_inputs(args):
    di=read_depth_index(args.depth_index,'validation',args.smoke)
    pi=read_cache_index(args.p1_index,'validation',args.smoke)
    oi=read_index(args.original_index,'validation',args.smoke)
    formal_rows(di,'validation',args.smoke);formal_rows(pi,'validation',args.smoke)
    expected=load_signed(args.p4_summary)
    if (expected.get('study')!='P4-1 ordinary-depth, frozen P2 readout' or
        (expected.get('smoke',False) and not args.smoke)):
        raise ValueError('P4 summary has mismatched experiment/smoke status')
    if (expected['sources']['depth_index']!=di['fingerprint'] or
        expected['sources']['p1_index']!=pi['fingerprint'] or
        expected['sources']['reader_sha256']!=sha256(args.shared_weight)):
        raise ValueError('P4 source provenance mismatch')
    if (di['contract']['original_index_fingerprint']!=oi['fingerprint'] or
        pi['contract']['original_index_fingerprint']!=oi['fingerprint']):
        raise ValueError('P4/P1 original frame identity mismatch')
    P={frame_key(e['row']):e for e in pi['entries']}
    O={frame_key(e['row']):e for e in oi['entries']}
    D={frame_key(e['row']):e for e in di['entries']}
    A={frame_key(e['row']):e for e in expected['artifacts']['frames']}
    if not(set(A)==set(D) and (args.smoke or set(D)==set(P)==set(O))):
        raise ValueError('P4 corrected depth artifacts do not cover intended frames')
    return di,pi,expected,[(D[k],P[k],O[k],A[k]) for k in [frame_key(e['row']) for e in di['entries']]]


def run(a):
    if (a.bootstrap<1 or a.limit<0 or (a.limit and not a.smoke) or
        a.edge_threshold_m<=0 or a.support_radius_px<0):
        raise ValueError('Invalid diagnostic configuration')
    di,pi,expected,paired=validate_inputs(a)
    reader,_=freeze_reader(a.shared_weight,pi,a.device,a.smoke)
    root=new_directory(a.output)
    offsets=pi['contract']['offsets_m'];tau=pi['contract']['tau']
    all_records={'predicted':[],'corrected':[],'clean_visible':[]}
    frame_rows=[];family_rows_local=[];regional=[];nn_metrics=[]
    selected=select_entries([dict(row=x[0]['row'],payload=x) for x in paired],a.limit)
    for index,choice in enumerate(selected,1):
        de,pe,oe,ae=choice['payload']
        if (ae['sha256']!=sha256(ae['path']) or pe['sha256']!=ae['p1_frame_sha256'] or
            oe['sha256']!=ae['original_sha256']):
            raise ValueError('P4 corrected artifact or physical bank changed')
        with np.load(de['source_path'],allow_pickle=False) as z:
            bank=np.asarray(z['bank'],np.float32);K=np.asarray(z['K'],np.float32)
            pred_depth=np.asarray(z['pred_depth'],np.float32)
            pid=np.asarray(z['physical_id']);q=np.asarray(z['quality_count']);fam=np.asarray(z['family_id'])
        with np.load(de['target_path'],allow_pickle=False) as z:clean_depth=np.asarray(z['visible_depth'],np.float32)
        with np.load(ae['path'],allow_pickle=False) as z:
            corr_depth=np.asarray(z['corrected_depth'],np.float32)
            cp=np.asarray(z['corrected_physics'],np.float32)
            if not np.array_equal(z['physical_id'],pid):raise ValueError('P4 cached physical action identity mismatch')
        old=load_cached(pe)
        if not np.array_equal(q,old['quality_count']) or not np.array_equal(pid,old['physical_id']) or not np.array_equal(fam,old['family_id']):
            raise ValueError('P1/P4 original quality or action identity mismatch')
        if cp.shape!=old['predicted_physics'].shape:raise ValueError('Corrected physics mismatch')
        points,local=support_points(torch.from_numpy(bank))
        if not np.allclose(local.numpy(),old['local_xyz'],atol=1e-6,rtol=0):
            raise ValueError('Physical support definition drift')
        xyz=points.reshape(-1,3).numpy()
        pixel=pixel_split(pred_depth,corr_depth,clean_depth,xyz,K,
                          a.edge_threshold_m,a.support_radius_px)
        regional.append(dict(row=de['row'],metrics=geometry_regional(
            old['predicted_physics'],cp,old['clean_visible_physics'],tau)))
        nn=nearest_pixel_drift(pred_depth,corr_depth,xyz,K)
        nn_metrics.append(dict(row=de['row'],metrics=nn))
        frame_rows.append(dict(row=de['row'],pixels=pixel,nearest_surface=nn))
        batch=dict(predicted_physics=old['predicted_physics'][None],corrected_physics=cp[None],
                   clean_visible_physics=old['clean_visible_physics'][None],
                   local_xyz=old['local_xyz'][None],bank_dims=old['bank_dims'][None])
        loc=torch.from_numpy(batch['local_xyz']).to(a.device);dims=torch.from_numpy(batch['bank_dims']).to(a.device)
        estimates={}
        for arm in all_records:
            with torch.no_grad():
                z=score_physics(reader,torch.as_tensor(batch[arm+'_physics'],device=a.device),loc,dims)[0].cpu().numpy()
            estimates[arm]=z
        features=family_geometry_features(old['predicted_physics'],cp,old['clean_visible_physics'],tau)
        for f in range(len(q)):
            family_id=int(fam[f]);joint=dict(row=de['row'],family=family_id)
            r=dict(scene=int(de['row']['scene']),domain=de['row']['domain'],anno=int(de['row']['anno']),
                   family=family_id,native_count=int(q[f,0]),**features[f])
            for arm in all_records:
                m=metrics_for_family(estimates[arm][f],q[f],offsets)
                m.update(scene=r['scene'],domain=r['domain'],anno=r['anno'],family=family_id,
                    native_count=r['native_count'])
                all_records[arm].append(m)
                r[arm+'_chosen_index']=int(m['choice'])
                r[arm+'_gain']=float(m['gain'])
            r['corrected_minus_predicted_gain']=r['corrected_gain']-r['predicted_gain']
            family_rows_local.append(r)
        if index%20==0 or index==len(selected):print(f'[P5 geometry] {index}/{len(selected)} frames',flush=True)
    if not a.limit:
        # Reconstruct the P4 whole-frame metric under precisely the same mask.
        for arm in all_records:
            if abs(scene_balanced(all_records[arm],'gain')-expected['scores'][arm]['gain'])>2e-6:
                raise RuntimeError('P5 frozen P4 action gain parity failed: '+arm)
    score_summary={k:paired_summary(v,all_records['predicted'] if k!='predicted' else None,
                     a.bootstrap,a.seed) for k,v in all_records.items()}
    pixel_summary={}
    for key in frame_rows[0]['pixels']:
        if key=='metadata':continue
        pixel_summary[key]={}
        for metric in ('n','image_fraction','pred_mae_mm','corrected_mae_mm','improvement_mm',
                       'corrected_wins_fraction','correction_abs_mm'):
            rows=[dict(row=rr['row'],metrics={metric:rr['pixels'][key][metric]}) for rr in frame_rows]
            from .experiment import balanced
            pixel_summary[key][metric]=balanced(rows,metric)
    geometry_summary={}
    for region in regional[0]['metrics']['predicted']:
        geometry_summary[region]={}
        for metric in regional[0]['metrics']['predicted'][region]:
            from .experiment import balanced
            geometry_summary[region][metric]={source:balanced([
                dict(row=r['row'],metrics={metric:r['metrics'][source][region][metric]}) for r in regional],metric)
                for source in ('predicted','corrected')}
    from .experiment import balanced
    if not a.limit:
        for before,observed in (('baseline_mae_mm',pixel_summary['full']['pred_mae_mm']),
                                ('mae_mm',pixel_summary['full']['corrected_mae_mm'])):
            if abs(expected['depth_metrics'][before]-observed)>1e-3:
                raise RuntimeError('P5 pixel/depth evaluation parity failed: '+before)
    nearest_summary={k:balanced([dict(row=r['row'],metrics={k:r['metrics'][k]}) for r in nn_metrics],k)
                     for k in nn_metrics[0]['metrics']}
    association={k:finite_correlation([r[k] if r[k] is not None else np.nan for r in family_rows_local],
                                    [r['corrected_minus_predicted_gain'] for r in family_rows_local])
                 for k in ('relative_udf_error_improvement_mm','relative_projective_error_improvement_mm',
                            'palm_proxy_relative_udf_improvement_mm',
                            'lateral_finger_proxy_relative_udf_improvement_mm',
                            'opening_core_proxy_relative_udf_improvement_mm')}
    ci=scene_bootstrap(all_records['corrected'],all_records['predicted'],metric='gain',
                       n_boot=a.bootstrap,seed=a.seed)
    report=dict(schema=SCHEMA,partition='validation',partial_smoke=bool(a.limit),
        n_frames=len(frame_rows),n_families=len(family_rows_local),
        sources=dict(p4_summary_sha256=sha256(a.p4_summary),p2_reader_sha256=sha256(a.shared_weight),
                     p1_fingerprint=pi['fingerprint'],depth_fingerprint=di['fingerprint']),
        depth_patches=dict(edge_threshold_m=a.edge_threshold_m,support_radius_px=a.support_radius_px,
                           edge_includes_missing_valid_boundary=True),
        pixel_regions=pixel_summary,geometry_regions=geometry_summary,nearest_pixel_drift=nearest_summary,
        scores=score_summary,corrected_minus_predicted_gain_ci=ci,
        geometry_improvement_vs_action_gain_descriptive=association,
        claims=dict(reader_updated=False,depth_model_retrained=False,
            official_ap_not_run=True,privileged_clean_visible_offline_only=True),
        note='Region masks are projected gripper-stencil proxies, not contact GT. Association is descriptive; no IID p-values. Nearest pixel drift is not automatically geometric improvement. Q is exact-action, not AP.')
    report['artifacts']=dict(per_frame=write_records(root/'pixel_geometry_frames.jsonl.gz',frame_rows),
        per_family=write_records(root/'action_geometry_families.jsonl.gz',family_rows_local),
        per_region=write_records(root/'region_geometry_frames.jsonl.gz',regional))
    atomic_json(root/'summary.json',signed(report))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--depth-index',required=True);p.add_argument('--p1-index',required=True)
    p.add_argument('--original-index',required=True);p.add_argument('--p4-summary',required=True)
    p.add_argument('--shared-weight',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--edge-threshold-m',type=float,default=.02)
    p.add_argument('--support-radius-px',type=int,default=2)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();run(a)

if __name__=='__main__':main()
