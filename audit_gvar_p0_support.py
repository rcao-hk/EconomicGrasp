#!/usr/bin/env python3
"""P0-A nontraining audit of dynamic-vs-fixed configured-probe support positions.

Uses the frozen AP-blind Novel subset, saved RGB checkpoint depth sidecars,
archived *post-collision* physical grasps and crop-aware K. No learned model
is run; no reference depth/GT is loaded for the geometric projection audit.
Unknown space is never labeled free or occupied. The measured mask-conflict
fraction quantifies an intervention-specific attention-masking mechanism, not
true camera visibility or grasp contact quality.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from models.gripper_volume_reader import GVARConfig, canonical_probes, project_probes, sample_map
from analyze_gvar_scene_paired import sha256_file

ROLES=('left_contact','right_contact','closing','finger_body','palm','approach')


def require(v,msg):
    if not v:raise ValueError(msg)


def digest(p:Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(records):
    grouped=defaultdict(list)
    for r in records:
        grouped[(r['role'],r['depth_bin'])].append(r)
    rows=[]
    for (role,depth_bin),values in sorted(grouped.items()):
        vec=lambda key:np.asarray([x[key] for x in values])
        valid=vec('both_projected_valid').astype(bool)
        distances=np.asarray([x['pixel_shift_px'] for x in values if x['both_projected_valid']], dtype=float)
        rows.append({'role':role,'depth_bin':depth_bin,'count':len(values),
                     'rgb_fixed_projected_ratio':float(vec('fixed_projected_valid').mean()),
                     'geom_dynamic_projected_ratio':float(vec('dynamic_projected_valid').mean()),
                     'geom_dynamic_depth_known_ratio':float(vec('geom_depth_known').mean()),
                     'geometry_known_but_fixed_rgb_masked_ratio':float(vec('geom_known_rgb_mask_conflict').mean()),
                     'mean_pixel_shift_px':float(distances.mean()) if distances.size else None,
                     'p95_pixel_shift_px':float(np.quantile(distances,.95)) if distances.size else None,
                     'paired_projection_count':int(valid.sum())})
    return rows


def analyze(root,selection_json,sidecars,out,topk=20):
    require(topk>0,'topk must be positive')
    require(not out.exists() or not any(out.iterdir()),f'Refuse existing output {out}')
    selector=json.loads(selection_json.read_text())
    require(selector.get('split')=='test_novel' and selector.get('epoch')==19,'Requires AP-blind e19 Novel selection')
    require(selector.get('selection_policy')=='pre_registered_fixed_frame_ids' and selector.get('AP_blind_selection') is True,
            'Selection must be AP-independent, not worst/best outcome selection')
    keys=[(int(r['scene_id']),int(r['frame_id'])) for r in selector['selected_frames']]
    require(len(keys)==len(set(keys)) and len(keys)==120 and
            {s for s,f in keys}==set(range(160,190)) and
            all(f in (0,80,160,240) for s,f in keys), 'Expected 120 fixed-ID Novel frames')
    path=sidecars/'volume_rel'/'export_manifest.json'
    saved=json.loads(path.read_text())
    require(saved['variant']=='volume_rel' and saved['selection_sha256']==digest(selection_json),
            'Depth sidecars do not share frozen AP-blind frame selection')
    original=root/'eval/volume_rel_e19/test_novel/gvar_inference_protocol.json'
    proto=json.loads(original.read_text())
    require(proto['checkpoint_sha256']==saved['checkpoint_sha256'],
            'Depth model checkpoint differs from officially evaluated volume_rel model')
    cfg=GVARConfig(**proto['gvar_config'])
    records=[]
    for scene,frame in keys:
        npz=sidecars/'volume_rel'/f'scene_{scene:04d}'/f'{frame:04d}.npz'
        grasp=root/'eval/volume_rel_e19/test_novel'/f'scene_{scene:04d}'/'realsense'/f'{frame:04d}.npy'
        require(npz.is_file() and grasp.is_file(),f'Missing saved depth or grasp {scene}/{frame}')
        with np.load(npz,allow_pickle=False) as d:
            require({'K','pred_depth_m'}<=set(d.files),f'Depth/K sidecar invalid {npz}')
            K=np.asarray(d['K'],dtype=np.float32);depth=np.asarray(d['pred_depth_m'],dtype=np.float32)
        require(K.shape==(3,3) and depth.ndim==2,'Invalid depth coordinates')
        allgrasps=np.load(grasp,allow_pickle=False)
        require(allgrasps.ndim==2 and allgrasps.shape[1]==17 and np.isfinite(allgrasps).all(),
                f'Invalid physical grasps {grasp}')
        order=np.argsort(-allgrasps[:,0],kind='stable')[:topk]
        g=allgrasps[order]
        if not len(g):continue
        centers=torch.as_tensor(g[:,13:16].copy(),dtype=torch.float32)
        rotations=torch.as_tensor(g[:,4:13].copy().reshape(-1,3,3),dtype=torch.float32)
        depths=torch.as_tensor(g[:,3].copy(),dtype=torch.float32)
        require(bool(torch.isfinite(depths).all() and (depths>0).all() and (depths<.2).all()),
                f'Invalid grasp insertion depth {grasp}')
        fixed,roles=canonical_probes(torch.full_like(depths,cfg.fixed_insertion_m),cfg)
        dynamic,roles2=canonical_probes(depths,cfg)
        require(torch.equal(roles,roles2),'Role ordering changed')
        H,W=depth.shape
        _,uv_fixed,grid_fixed,valid_fixed=project_probes(centers,rotations,fixed,torch.from_numpy(K),(H,W))
        _,uv_dynamic,grid_dynamic,valid_dynamic=project_probes(centers,rotations,dynamic,torch.from_numpy(K),(H,W))
        depth_valid=np.isfinite(depth)&(depth>0)
        known=sample_map(torch.from_numpy(depth_valid.astype(np.float32))[None,None],grid_dynamic)[...,0]>.999
        known=known & valid_dynamic
        distance=torch.norm(uv_dynamic-uv_fixed,dim=-1).numpy()
        for k in range(len(g)):
            depth_bin=int(round(float(g[k,3])/.01))
            for j in range(len(roles)):
                vf=bool(valid_fixed[k,j]);vd=bool(valid_dynamic[k,j]);kd=bool(known[k,j])
                records.append({'scene_id':scene,'frame_id':frame,'rank':k+1,
                                'depth_bin':depth_bin,'role':ROLES[int(roles[j])],
                                'probe_idx':j,'fixed_projected_valid':int(vf),
                                'dynamic_projected_valid':int(vd),
                                'geom_depth_known':int(kd),
                                'geom_known_rgb_mask_conflict':int(kd and not vf),
                                'both_projected_valid':int(vf and vd),
                                'pixel_shift_px':float(distance[k,j]) if vf and vd else None})
    require(records,'No physical grasps to audit')
    summary=summarize(records)
    out.mkdir(parents=True,exist_ok=True)
    for p,rows in ((out/'probe_shift.csv',records),(out/'role_depth_summary.csv',summary)):
        with p.open('w',newline='',encoding='utf-8') as handle:
            w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest={'selection_sha256':digest(selection_json),'source_checkpoint_sha256':saved['checkpoint_sha256'],
              'source_files':[{'path':str(p),'sha256':sha256_file(p)} for p in (path,original,selection_json)],
              'topk':topk,'probe_records':len(records),'sample_count':len(keys),
              'warning':'Known depth along ray is NOT true 3D visibility, free space or contact success.'}
    (out/'AUDIT.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=['# P0-A projected support-position audit', '',
            f'AP-blind Novel frames: {len(keys)}, archived physical top-{topk} grasps, {len(records)} probes.',
            'Pixel displacement is defined between *both valid* fixed and dynamic projections.',
            'Geometry-known-but-fixed-RGB-masked includes only valid predicted ray-depth samples, not ground-truth visibility.',
            '', '| Role | Insertion bin | Mean shift px | Geometry-known but fixed RGB masked |',
            '|---|---:|---:|---:|']
    for r in summary:
        report.append(f"| {r['role']} | {r['depth_bin']} | "
                      f"{r['mean_pixel_shift_px'] if r['mean_pixel_shift_px'] is not None else 'NA'} | "
                      f"{r['geometry_known_but_fixed_rgb_masked_ratio']:.3f} |")
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selection-json',type=Path,required=True)
    p.add_argument('--sidecar-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--topk',type=int,default=20)
    a=p.parse_args();v=analyze(a.root,a.selection_json,a.sidecar_root,a.output_dir,a.topk)
    print(json.dumps({'frames':v['sample_count'],'probes':v['probe_records']},indent=2))

if __name__=='__main__':main()
