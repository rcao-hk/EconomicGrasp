#!/usr/bin/env python3
"""CPU-only gripper-probe geometry-error / uncertainty calibration analysis.

Inputs: exported aligned RGB/predicted/rendered/sensor-depth sidecars, EXISTING
GraspGroup dumps and an AP-only selected-frame manifest. All sampled probe
locations are determined by each archived physical grasp and the matching
GVAR reader variant. Unknown/background samples are explicitly excluded.

Local predicted-depth STD/gradient are *heuristic proxies*, not learned or
calibrated uncertainty. External sigma, when provided, must be provenance-tagged
by the exporter. No probe-level grasp-outcome labels or causal claims are made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F

from models.gripper_volume_reader import GVARConfig, ROLE_NAMES, canonical_probes, project_probes, sample_map

VARIANTS=("volume_fixed","volume","volume_rel","volume_fixed_rel")


def check(condition, msg):
    if not condition: raise ValueError(msg)


def sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def append_csv(path, rows):
    check(rows,f'Empty CSV {path}')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def predicted_proxies(pred):
    """Depth-only local/std gradient proxies; no GT used in their calculation."""
    x=torch.as_tensor(pred,dtype=torch.float32)[None,None]
    valid=torch.isfinite(x)&(x>0)
    cleaned=torch.where(valid,x,0.)
    weight=F.avg_pool2d(valid.float(),3,1,1)
    avg=F.avg_pool2d(cleaned,3,1,1)/weight.clamp_min(1e-6)
    var=(F.avg_pool2d(cleaned.square(),3,1,1)/weight.clamp_min(1e-6)-avg.square()).clamp_min(0)
    std=torch.where(weight>0,torch.sqrt(var),torch.nan)
    gx=F.pad((cleaned[...,1:]-cleaned[...,:-1]).abs(),(0,1,0,0))
    gy=F.pad((cleaned[...,1:,:]-cleaned[...,:-1,:]).abs(),(0,0,0,1))
    edge=torch.sqrt(gx.square()+gy.square())
    return std,edge


def _sample(v: np.ndarray, grid: torch.Tensor):
    x=torch.as_tensor(np.asarray(v),dtype=torch.float32)[None,None]
    return sample_map(x,grid)[...,0].numpy()


def _ranks(a):
    """Tie-aware average ordinal ranks for Spearman correlation."""
    _,inverse,count=np.unique(a,return_inverse=True,return_counts=True)
    first=np.cumsum(count)-count
    values=first+(count-1)/2
    return values[inverse].astype(float)


def spearman(a,b):
    a=np.asarray(a);b=np.asarray(b); ok=np.isfinite(a)&np.isfinite(b)
    if np.count_nonzero(ok)<10:return None
    aa,bb=_ranks(a[ok]),_ranks(b[ok])
    if np.std(aa)<1e-12 or np.std(bb)<1e-12:return None
    return float(np.corrcoef(aa,bb)[0,1])


def curves(rows:list[dict],bins:int):
    """Summarize risk/calibration by observed probe. NOT an independent sample CI."""
    data=[]; risk=[]; summary=[]
    for variant in sorted(set(r['variant'] for r in rows)):
        for score_key,score_type in (("proxy_depth_std_m","proxy_std"),
                                     ("proxy_depth_gradient_mpx","proxy_gradient"),
                                     ("uncertainty_sigma_m","external_sigma")):
            items=[r for r in rows if r['variant']==variant and r['rendered_error_m'] is not None
                   and r[score_key] is not None and np.isfinite(r[score_key])]
            if not items:continue
            score=np.asarray([float(x[score_key]) for x in items]);err=np.asarray([float(x['rendered_error_m']) for x in items])
            order=np.argsort(score,kind='stable')
            ranks=np.array_split(order,bins)
            for k,sel in enumerate(ranks):
                if len(sel):data.append({"variant":variant,"score_type":score_type,"bin":k,
                   "count":len(sel),"mean_uncertainty_or_proxy_m":float(np.mean(score[sel])),
                   "mean_abs_depth_error_m":float(np.mean(err[sel])),"median_abs_error_m":float(np.median(err[sel]))})
            for coverage in (1.,.9,.75,.5,.25):
                n=max(1,int(math.ceil(len(order)*coverage)))
                ids=order[:n]
                risk.append({"variant":variant,"score_type":score_type,"coverage":coverage,"n_probes":n,
                             "mean_abs_error_m":float(np.mean(err[ids]))})
            corr=spearman(score,err)
            ans={"variant":variant,"score_type":score_type,"n_valid_probes":len(err),
                 "mean_abs_error_m":float(err.mean()),"spearman_proxy_vs_abs_error":corr,
                 "error_gt_10mm_rate":float(np.mean(err>.01)),"score_mean_m":float(score.mean()),
                 "scene_cluster_CI":"not estimated; probes and selected frames not independent"}
            if score_type=='external_sigma':
                # Empirical intervals are descriptive; Gaussian nominal coverage
                # only if this map is an estimated standard deviation in metres.
                for z,pct in ((1.,'68'),(1.645,'90'),(1.96,'95')):
                    ans[f'coverage_le_{z:g}sigma_vs_{pct}pct']=float(np.mean(err<=z*np.maximum(score,1e-6)))
            summary.append(ans)
    return data,risk,summary


def records_for_frame(sidecar:Path,grasp:Path,variant:str,scene:int,frame:int,rank_limit:int,selection:dict,
                      cfg:GVARConfig):
    check(grasp.is_file(),f'Missing archived grasp dump {grasp}')
    with np.load(sidecar,allow_pickle=False) as d:
        required={'pred_depth_m','gt_depth_m','sensor_depth_m','foreground_mask','K','crop_rgb'}
        check(required<=set(d.files),f'Missing sidecar fields {required-set(d.files)}: {sidecar}')
        maps={key:np.array(d[key]) for key in d.files}
    pred,gt,sensor,fg=(maps[x] for x in ('pred_depth_m','gt_depth_m','sensor_depth_m','foreground_mask'))
    H,W=pred.shape
    check(gt.shape==sensor.shape==fg.shape==(H,W),f'Nonaligned depth references {sidecar}')
    K=np.array(maps['K'],dtype=np.float32)
    check(K.shape==(3,3) and np.isfinite(K).all(),f'Invalid crop intrinsics {sidecar}')
    grasps=np.load(grasp,allow_pickle=False)
    check(grasps.ndim==2 and grasps.shape[1]==17 and np.isfinite(grasps).all(),f'Bad grasp dump {grasp}')
    order=np.argsort(-grasps[:,0],kind='stable')[:rank_limit]
    raw=grasps[order]
    if not len(raw):return [], {"scene_id":scene,"frame_id":frame,"variant":variant,"valid_probe_fraction":0.,"status":"zero_grasps"}
    centers=torch.from_numpy(raw[:,13:16].astype(np.float32))
    rotations=torch.from_numpy(raw[:,4:13].reshape(-1,3,3).astype(np.float32))
    insertion=torch.from_numpy(raw[:,3].astype(np.float32))
    check(bool(torch.isfinite(insertion).all() and (insertion>0).all() and (insertion<.2).all()),f'Implausible insertion depths in {grasp}')
    d_used=(torch.full_like(insertion,cfg.fixed_insertion_m) if variant in ('volume_fixed','volume_fixed_rel') else insertion)
    probes,roles=canonical_probes(d_used,cfg)
    xyz,uv,grid,valid=project_probes(centers,rotations,probes,torch.from_numpy(K),(H,W))
    # Bilinear interpolation across invalid/GT holes is forbidden.
    depth_pred_ok=np.isfinite(pred)&(pred>0)
    gt_ok=np.isfinite(gt)&(gt>0)
    sensor_ok=np.isfinite(sensor)&(sensor>0)
    foreground=(fg>0).astype(np.float32)
    valid_render=(_sample((depth_pred_ok & gt_ok).astype('float32'),grid)>.999)
    valid_sensor=(_sample((depth_pred_ok & sensor_ok).astype('float32'),grid)>.999)
    valid_fg=(_sample(foreground,grid)>.999)
    zpred=_sample(np.where(depth_pred_ok,pred,0),grid)
    zgt=_sample(np.where(gt_ok,gt,0),grid)
    zsensor=_sample(np.where(sensor_ok,sensor,0),grid)
    std,grad=predicted_proxies(pred)
    pxstd=sample_map(std,grid)[...,0].numpy()
    pxgrad=sample_map(grad,grid)[...,0].numpy()
    sigma=None
    if 'uncertainty_sigma_m' in maps:
        s=maps['uncertainty_sigma_m']
        check(s.shape==(H,W) and np.isfinite(s).all() and (s>=0).all(),f'Invalid uncertainty sigma {sidecar}')
        sigma=_sample(s,grid)
    report=[]; valid_count=0
    proj=valid.numpy(); uvn=uv.numpy()
    for n in range(len(raw)):
        for k in range(36):
            fg_pass=bool(proj[n,k] and valid_fg[n,k]); gt_pass=bool(fg_pass and valid_render[n,k]); sensor_pass=bool(fg_pass and valid_sensor[n,k]);
            if gt_pass:valid_count+=1
            report.append({"variant":variant,"scene_id":scene,"frame_id":frame,"selection":selection['selection'],
                "grasp_rank":n+1,"grasp_score":float(raw[n,0]),"grasp_width_m":float(raw[n,1]),
                "insertion_depth_m":float(raw[n,3]),"support_depth_m":float(d_used[n]),
                "role":ROLE_NAMES[int(roles[k])],"probe_idx":k,"u":float(uvn[n,k,0]),"v":float(uvn[n,k,1]),
                "projection_valid":bool(proj[n,k]),"foreground_valid":fg_pass,
                "rendered_valid":gt_pass,"sensor_valid":sensor_pass,
                "rendered_error_m":float(abs(zpred[n,k]-zgt[n,k])) if gt_pass else None,
                "sensor_error_m":float(abs(zpred[n,k]-zsensor[n,k])) if sensor_pass else None,
                "proxy_depth_std_m":float(pxstd[n,k]) if gt_pass and np.isfinite(pxstd[n,k]) else None,
                "proxy_depth_gradient_mpx":float(pxgrad[n,k]) if gt_pass and np.isfinite(pxgrad[n,k]) else None,
                "uncertainty_sigma_m":float(sigma[n,k]) if sigma is not None and gt_pass else None,
                "source_frame_delta_AP_pp":float(selection['delta_AP_pp'])})
    frame_stats={"scene_id":scene,"frame_id":frame,"variant":variant,"status":"ok","num_grasps":len(raw),
                 "total_probes":len(raw)*36,"valid_rendered_object_probes":valid_count,
                 "valid_probe_fraction":valid_count/(len(raw)*36),"rendered_MAE_m":np.mean([r['rendered_error_m'] for r in report if r['rendered_error_m'] is not None]) if valid_count else None,
                 "source_frame_delta_AP_pp":float(selection['delta_AP_pp'])}
    return report,frame_stats


def render_frame_figure(sidecar:Path, rows:list[dict],output:Path, title:str):
    """Diagnostic overlay of archived top-1 projected GVAR probes (not GT contacts)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(sidecar,allow_pickle=False) as d:
        rgb=np.asarray(d['crop_rgb']);pred=np.asarray(d['pred_depth_m'])
        gt=np.asarray(d['gt_depth_m']);fg=np.asarray(d['foreground_mask'])>0
    delta=np.where(fg & np.isfinite(gt)&(gt>0),np.abs(pred-gt)*1000.,np.nan)
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    axes[0,0].imshow(rgb);axes[0,0].set_title('RGB / top-1 projected probes')
    pal=plt.get_cmap('tab10')
    for j,role in enumerate(ROLE_NAMES):
        xy=[(r['u'],r['v']) for r in rows if r['role']==role and r['grasp_rank']==1 and r['projection_valid']]
        if xy:
            axes[0,0].scatter([x for x,y in xy],[y for x,y in xy],s=15,label=role,color=pal(j))
    axes[0,0].legend(fontsize=7,loc='upper right',ncol=2)
    axes[0,1].imshow(pred,cmap='magma',vmin=.3,vmax=1.0);axes[0,1].set_title('Predicted depth (m)')
    axes[1,0].imshow(gt,cmap='magma',vmin=.3,vmax=1.0);axes[1,0].set_title('Rendered/fused reference (m)')
    h=axes[1,1].imshow(delta,cmap='viridis',vmin=0.,vmax=60.)
    axes[1,1].set_title('|pred-reference| (mm), visible object mask')
    fig.colorbar(h,ax=axes[1,1],shrink=.75)
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(title)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=140);plt.close(fig)


def run(root:Path,sidecars:Path,selection_path:Path,out:Path,variants:list[str],topk:int,bins:int, render_max:int=0):
    check(variants and len(variants)==len(set(variants)) and all(v in VARIANTS for v in variants),'Variants must be GVAR volume reader variants')
    check(topk>0 and bins>=2 and render_max>=0,'Bad topk/bins/render_max')
    check(not(out.exists() and any(out.iterdir())),f'Nonempty output {out}; choose fresh directory')
    chosen=json.loads(selection_path.read_text())
    check(chosen.get('split')=='test_novel' and chosen.get('epoch')==19,'Selection must be e19 Novel')
    selection_sha=sha(selection_path)
    rows,frames,manifests=[],[],{}
    to_render=[]
    for v in variants:
        manifest_path=sidecars/v/'export_manifest.json';m=json.loads(manifest_path.read_text())
        check(m['selection_sha256']==selection_sha and m['variant']==v,'Mismatched sidecar selection/variant')
        proto=json.loads((root/'eval'/f'{v}_e19'/'test_novel'/'gvar_inference_protocol.json').read_text())
        check(proto['checkpoint_sha256']==m['checkpoint_sha256'],'Sidecar checkpoint did not generate official AP grasps')
        manifests[v]=m
        cfg=GVARConfig(**proto['gvar_config'])
        for record in chosen['selected_frames']:
            s,f=int(record['scene_id']),int(record['frame_id'])
            sidecar=sidecars/v/f'scene_{s:04d}'/f'{f:04d}.npz'
            check(sidecar.is_file(),f'Missing depth sidecar {sidecar}')
            grasp=root/'eval'/f'{v}_e19'/'test_novel'/f'scene_{s:04d}'/'realsense'/f'{f:04d}.npy'
            r,stat=records_for_frame(sidecar,grasp,v,s,f,topk,record,cfg)
            rows.extend(r);frames.append(stat)
            if render_max and len(to_render)<render_max:
                to_render.append((sidecar,r,f'{v} scene_{s:04d} / frame {f:04d}',v,s,f))
    check(rows,'No exported probes to analyze')
    per_scene=[]
    for v in variants:
        for sid in sorted(set(x['scene_id'] for x in frames)):
            items=[x for x in rows if x['variant']==v and x['scene_id']==sid and x['rendered_error_m'] is not None]
            per_scene.append({"variant":v,"scene_id":sid,"valid_object_probes":len(items),
                              "MAE_m":float(np.mean([x['rendered_error_m'] for x in items])) if items else None,
                              "mean_proxy_std_m":float(np.mean([x['proxy_depth_std_m'] for x in items if x['proxy_depth_std_m'] is not None])) if items else None})
    binrows,riskrows,cal=curves(rows,bins)
    out.mkdir(parents=True,exist_ok=True)
    for sidecar,rr,title,v,s,f in to_render:
        render_frame_figure(sidecar,rr,out/'figures'/v/f'scene_{s:04d}_frame_{f:04d}.png',title)
    for name,data in [('probe_records.csv',rows),('frame_probe_summary.csv',frames),('scene_probe_summary.csv',per_scene),
                      ('calibration_bins.csv',binrows),('risk_coverage.csv',riskrows),('calibration_summary.csv',cal)]:
        if data:append_csv(out/name,data)
    source={v:{"uncertainty_source":m['uncertainty_source'],"checkpoint_sha256":m['checkpoint_sha256']} for v,m in manifests.items()}
    meta={"version":1,"source_selection_sha256":selection_sha,"source_selection":str(selection_path),
          "variants":variants,"archived_eval_root":str(root),"sidecar_root":str(sidecars),
          "uncertainty_provenance":source,"num_probe_records":len(rows),"rank_topk":topk,"num_bins":bins,
          "object_region_only":True,"geometry_reference":"rendered_gt vs predicted for visible object pixels; sensor additionally reported",
          "uncertainty_proxy_caveat":"local depth std/gradient NOT learned uncertainty; selected frames NOT representative sample",
          "physical_probe_caveat":"projected probes sample visible ray depth; they do NOT directly evaluate gripper contact or exact DexNet utility"}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    lines=['# GVAR gripper-probe geometry reliability diagnostics','',
           'This is an **outcome-selected** frame subset (not full Novel evaluation). Probe pixels are correlated within frames/scenes.',
           'The measured target is absolute **predicted vs rendered visible object-depth** error, not physical contact success.',
           'Local predicted-depth std/gradient are **heuristic proxies**, not learned calibrated uncertainty. External sigma provenance is recorded.',
           '', '| Variant | Score | Valid probes | MAE (mm) | Spearman(score, abs error) |', '|---|---|---:|---:|---:|']
    for a in cal:
        lines.append(f"| {a['variant']} | {a['score_type']} | {a['n_valid_probes']} | {a['mean_abs_error_m']*1000:.2f} | {a['spearman_proxy_vs_abs_error'] if a['spearman_proxy_vs_abs_error'] is not None else 'NA'} |")
    lines+=['','**Do not use this diagnostic to infer a causal effect of uncertainty-aware grasping.**',
            'Next: if a genuinely RGB-only external sigma map predicts probe errors on separate (non-selected) Novel scenes, test its addition inside action evidence aggregation.',
            'A privileged depth-assisted sigma is allowed as a diagnostic only, not as RGB-only inference input.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selection-json',type=Path,required=True)
    p.add_argument('--sidecar-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['volume_rel'])
    p.add_argument('--topk',type=int,default=20)
    p.add_argument('--bins',type=int,default=10)
    p.add_argument('--render-max',type=int,default=0,help='Render first N selected frame/variant diagnostic panels')
    args=p.parse_args(argv)
    meta=run(args.root,args.sidecar_root,args.selection_json,args.output_dir,args.variants,args.topk,args.bins,args.render_max)
    print(json.dumps({"variants":meta['variants'],"probe_records":meta['num_probe_records'],"output":str(args.output_dir)},indent=2))

if __name__=='__main__':main()
