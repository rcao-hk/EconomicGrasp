"""GPU-free regression suite for GVAR hybrid and AP-blind Novel cohort.

All AP/sidecar inputs in this file are synthetic and must not be interpreted as
real GVAR validation. Actual CUDA/model/official AP require server smoke tests.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from models.gripper_volume_reader import (
    GVARConfig, GripperVolumeReader, VARIANTS,
    canonical_probes, support_probe_depths, project_probes,
)
from test_gvar import inputs
from test_gvar_scene_paired import synthetic_gvar, SPLITS
from test_gvar_p0_p1_followup import p1_demo, put_json, write_csv
from select_gvar_uniform_novel import selected_ids, run as select_run, DEFAULT_FRAMES, DEFAULT_SCENES
from analyze_gvar_hybrid_p0 import evaluate, AuditError
from analyze_gvar_p1_conditional import run as conditional_run
from compare_gvar_p1_cohorts import compare


def csv_data(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def _gvar_controls(root, ready=False):
    synthetic_gvar(root, ('volume_fixed','volume_rel'))
    for variant in ('volume_fixed_rel', *(('volume_hybrid',) if ready else ())):
        shutil.copytree(root/'train/volume_fixed',root/'train'/variant)
        p=root/'train'/variant/'gvar_protocol.json'
        data=json.loads(p.read_text());data['gvar_config']['variant']=variant;put_json(p,data)
        for split in SPLITS:
            original=root/'eval/volume_fixed_e19'/split
            target=root/'eval'/f'{variant}_e19'/split
            shutil.copytree(original,target)
            j=target/'gvar_inference_protocol.json'
            d=json.loads(j.read_text());d['gvar_config']['variant']=variant
            d['checkpoint_sha256']=hashlib.sha256(variant.encode()).hexdigest()
            put_json(j,d)
    for variant in ('volume_fixed_rel','volume_rel',*(('volume_hybrid',) if ready else ())):
        log=root/'train'/variant/'gvar_epochs.jsonl'
        with log.open('w') as stream:
            for i in range(21):stream.write(json.dumps({'epoch':i,'train_loss':.8,'val_loss':1.0})+'\n')
        for i in (19,20):
            (root/'train'/variant/f'checkpoint_epoch_{i:03d}.tar').write_bytes(variant.encode())
    for variant,percent in {'volume_rel':45.5176,'volume_fixed_rel':44.1730,
                            **({'volume_hybrid':44.8} if ready else {})}.items():
        for split in SPLITS:
            path=root/'eval'/f'{variant}_e19'/split/f'ap_{split}_realsense.npy'
            array=np.load(path)
            array[:]=np.float32(percent/100.)
            np.save(path,array)


def test_hybrid_probe_stream_coordinates_are_distinct():
    assert 'volume_hybrid' in VARIANTS
    depths=torch.tensor([.01,.04])
    config=GVARConfig(variant='volume_hybrid')
    rgb=support_probe_depths(depths,config,'rgb')
    geometry=support_probe_depths(depths,config,'geometry')
    assert rgb.tolist()==pytest.approx([.025,.025])
    torch.testing.assert_close(geometry,depths)
    xr,_=canonical_probes(rgb,config)
    xg,_=canonical_probes(geometry,config)
    torch.testing.assert_close(xr[0],xr[1])
    torch.testing.assert_close(xg[1]-xg[0],torch.tensor([.03,0.,0.]).expand(36,3))
    # Fixed RGB and dynamic geometry actually project to different pixels.
    K=torch.tensor([[140.,0.,31.5],[0.,140.,31.5],[0.,0.,1.]])
    R=torch.eye(3).repeat(2,1,1)
    xyz=torch.tensor([[0.,0.,.6],[0.,0.,.6]])
    _,uvr,_,_=project_probes(xyz,R,xr,K,(64,64))
    _,uvg,_,_=project_probes(xyz,R,xg,K,(64,64))
    torch.testing.assert_close(uvr[0],uvr[1])
    assert not torch.allclose(uvg[0],uvg[1])


def test_hybrid_has_exact_same_parameters_and_detached_depth():
    base,_,_=inputs('volume_rel')
    hybrid,q,c=inputs('volume_hybrid')
    assert list(hybrid.state_dict())==list(base.state_dict())
    assert sum(x.numel() for x in hybrid.parameters())==sum(x.numel() for x in base.parameters())
    hybrid.load_state_dict(base.state_dict())
    y=hybrid(q,c)
    assert y.shape==q.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert q.grad is not None and c['pregeom'].grad is not None
    assert c['depth'].grad is None and c['centers'].grad is None
    assert all(x.grad is not None for x in hybrid.parameters())


def test_hybrid_rgb_and_geometry_are_not_implicitly_colocated(monkeypatch):
    import models.gripper_volume_reader as mod
    reader,q,context=inputs('volume_hybrid',Q=1,A=1)
    grids=[]
    original=mod.action_relative_geometry
    def capture(depth,grid,uv,xyz,centers,rotations,K,valid,scale):
        grids.append(grid.clone())
        return original(depth,grid,uv,xyz,centers,rotations,K,valid,scale)
    monkeypatch.setattr(mod,'action_relative_geometry',capture)
    reader(q,context)
    # At least D=4 independent geometry grids; test first and last insertion.
    assert len(grids)>=1 and torch.isfinite(torch.cat(grids)).all()
    valid_sizes=[g.shape[0] for g in grids]
    assert sum(valid_sizes)==q.shape[3]
    g=torch.cat(grids,dim=0)
    assert not torch.allclose(g[0],g[-1])


def test_hybrid_cuda_not_claimed_by_cpu_tests():
    # Smoke test is deliberately CPU-only; full forward uses server CUDA extensions.
    assert torch.tensor(1.).device.type=='cpu'


def test_default_uniform_frame_ids_and_scene_count():
    frozen=selected_ids(DEFAULT_SCENES,DEFAULT_FRAMES)
    assert len(frozen)==120
    assert frozen[:4]==[(160,0),(160,80),(160,160),(160,240)]
    assert frozen[-4:]==[(189,0),(189,80),(189,160),(189,240)]
    with pytest.raises(AuditError,match='Frame IDs'):
        selected_ids(DEFAULT_SCENES,(0,77,160,240))
    with pytest.raises(AuditError,match='unique'):
        selected_ids((160,160),(0,80))


def test_uniform_selection_is_independent_of_ap_order(tmp_path):
    root=tmp_path/'gvar';synthetic_gvar(root,('baseline','volume_rel'))
    out=tmp_path/'uniform';stats=select_run(root,out)
    assert stats['frames']==120 and stats['scenes']==30
    doc=json.loads((out/'selected_frames.json').read_text())
    assert doc['AP_blind_selection'] is True
    assert doc['selection_policy']=='pre_registered_fixed_frame_ids'
    assert {x['selection'] for x in doc['selected_frames']}=={'uniform'}
    assert {x['frame_id'] for x in doc['selected_frames']}==set(DEFAULT_FRAMES)
    assert len(csv_data(out/'novel_frames.csv'))==120
    # Perturb the AP tensor arbitrarily (while keeping probabilities valid).
    p=root/'eval/volume_rel_e19/test_novel/ap_test_novel_realsense.npy'
    arr=np.load(p);arr[:,0]*=.01
    np.save(p,arr)
    out2=tmp_path/'uniform2';stats2=select_run(root,out2)
    assert stats2['selection_sha256']==stats['selection_sha256']
    q=json.loads((out2/'selected_frames.json').read_text())
    assert [(x['scene_id'],x['frame_id']) for x in q['selected_frames']]==[ (x['scene_id'],x['frame_id']) for x in doc['selected_frames'] ]
    assert [x['delta_AP_pp'] for x in q['selected_frames']] != [x['delta_AP_pp'] for x in doc['selected_frames']]


def test_uniform_selection_rejects_nonempty_out_and_bad_epoch(tmp_path):
    root=tmp_path/'gvar';synthetic_gvar(root,('baseline','volume_rel'))
    out=tmp_path/'keep';out.mkdir();(out/'other.txt').write_text('keep')
    with pytest.raises(AuditError,match='overwrite'):
        select_run(root,out)
    assert (out/'other.txt').read_text()=='keep'
    with pytest.raises(AuditError,match='e19'):
        select_run(root,tmp_path/'no',epoch=20)


def test_hybrid_status_pending_without_fake_ap(tmp_path):
    root=tmp_path/'gvar';_gvar_controls(root,ready=False)
    status=evaluate(root,tmp_path/'out',bootstrap=200)
    assert status['status']=='pending'
    assert not (tmp_path/'out/paired').exists()
    assert 'Hybrid e19 AP not available' in (tmp_path/'out/REPORT.md').read_text()


def test_hybrid_completed_produces_direct_path_effects(tmp_path):
    root=tmp_path/'gvar';_gvar_controls(root,ready=True)
    stat=evaluate(root,tmp_path/'out',bootstrap=250,seed=33)
    assert stat['status']=='complete'
    rows=csv_data(tmp_path/'out/hybrid_path_effects.csv')
    assert len(rows)==4*12*3
    def get(label):
        return next(float(x['delta_pp']) for x in rows if x['split']=='Mean' and x['metric']=='AP' and x['contrast']==label)
    assert get('dynamic_geometry_given_fixed_rgb')==pytest.approx(44.8-44.173,abs=.001)
    assert get('dynamic_rgb_given_dynamic_geometry')==pytest.approx(45.5176-44.8,abs=.001)
    assert get('whole_path_dynamic_vs_fixed')==pytest.approx(get('dynamic_geometry_given_fixed_rgb')+get('dynamic_rgb_given_dynamic_geometry'),abs=1e-8)
    assert 'Novel stricter/looser friction' in (tmp_path/'out/REPORT.md').read_text()


def test_hybrid_fails_if_checkpoint_hash_does_not_match(tmp_path):
    root=tmp_path/'gvar';_gvar_controls(root,ready=True)
    (root/'train/volume_hybrid/checkpoint_epoch_019.tar').write_bytes(b'wrong')
    with pytest.raises(AuditError,match='SHA mismatch'):
        evaluate(root,tmp_path/'out',bootstrap=200)


def test_uniform_cohort_runs_existing_probe_conditional_and_compares(tmp_path):
    (tmp_path/'old').mkdir()
    old_cases,old_probes=p1_demo(tmp_path/'old')
    old_out=tmp_path/'old_stats'
    conditional_run(old_probes,old_cases,old_out,['volume_rel'],min_probes=3,min_frames=3)
    root=tmp_path/'gvar';synthetic_gvar(root,('baseline','volume_rel'))
    cases=tmp_path/'uniform_cases';select_run(root,cases)
    selection=json.loads((cases/'selected_frames.json').read_text())
    probes=tmp_path/'uniform_probes';probes.mkdir()
    rows=[];frame_stats=[]
    for selected in selection['selected_frames']:
        s=selected['scene_id'];f=selected['frame_id'];delta=selected['delta_AP_pp']
        for role_idx,role in enumerate(('left_contact','right_contact','closing','finger_body','palm','approach')):
            for j in range(6):
                idx=role_idx*6+j
                valid=(j>=1)
                rows.append({'variant':'volume_rel','scene_id':s,'frame_id':f,'role':role,'grasp_rank':1,'probe_idx':idx,
                    'projection_valid':True,'foreground_valid':True,'rendered_valid':True,
                    'sensor_valid':valid,'rendered_error_m':.01+.002*(f//80)+.0001*idx,
                    'sensor_error_m':.012+.0002*idx if valid else None,
                    'proxy_depth_std_m':.001+.00005*idx,
                    'proxy_depth_gradient_mpx':.0002+.00001*idx,
                    'uncertainty_sigma_m':None,'source_frame_delta_AP_pp':delta})
        frame_stats.append({'variant':'volume_rel','scene_id':s,'frame_id':f,'status':'ok',
            'num_grasps':1,'total_probes':36,'valid_rendered_object_probes':36,
            'source_frame_delta_AP_pp':delta})
    probes.joinpath('probe_records.csv');write_csv(probes/'probe_records.csv',rows)
    write_csv(probes/'frame_probe_summary.csv',frame_stats)
    selection_hash=hashlib.sha256((cases/'selected_frames.json').read_bytes()).hexdigest()
    put_json(probes/'manifest.json',{'source_selection_sha256':selection_hash,'variants':['volume_rel'],
            'uncertainty_provenance':{'volume_rel':{'uncertainty_source':'none'}},
            'rank_topk':1,'num_probe_records':len(rows)})
    uniform_out=tmp_path/'uniform_stats'
    result=conditional_run(probes,cases,uniform_out,['volume_rel'],min_probes=3,min_frames=3)
    assert result['scenes']==30 and result['frames']==120 and result['probes']==4320
    assert len(csv_data(uniform_out/'per_scene_role.csv'))==180
    out=tmp_path/'compared';report=compare(old_out,uniform_out,out)
    assert report['uniform_frames']==120 and report['selected_frames']==8
    assert report['overlap_frames']==2
    assert len(csv_data(out/'cohort_AP_association.csv'))==36
    assert 'descriptive' in (out/'REPORT.md').read_text()
