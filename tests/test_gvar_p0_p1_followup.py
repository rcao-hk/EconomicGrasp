"""P0 readiness/factorial and P1 conditional probe analysis synthetic regressions."""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'tests'))
from test_gvar_scene_paired import synthetic_gvar
from analyze_gvar_p0_decision import run as p0_run, AuditError
from analyze_gvar_p1_conditional import run as p1_run, DataError, spearman, pearson_within_scene


def put_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def get_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def p0_demo(root,ready=False):
    controls=('baseline','slot','volume_fixed','volume','volume_rel')
    synthetic_gvar(root,controls)
    if ready:
        shutil.copytree(root/'train/volume_fixed',root/'train/volume_fixed_rel')
        p=root/'train/volume_fixed_rel/gvar_protocol.json'
        data=json.loads(p.read_text());data['gvar_config']['variant']='volume_fixed_rel';put_json(p,data)
        for split in ('test_seen','test_similar','test_novel'):
            dst=root/'eval/volume_fixed_rel_e19'/split
            shutil.copytree(root/'eval/volume_fixed_e19'/split,dst)
            meta=dst/'gvar_inference_protocol.json'
            j=json.loads(meta.read_text());j['gvar_config']['variant']='volume_fixed_rel'
            j['checkpoint_sha256']=hashlib.sha256(b'volume_fixed_rel').hexdigest()
            put_json(meta,j)
            ap=dst/f'ap_{split}_realsense.npy'
            np.save(ap,np.load(ap)+np.float32(.01))
    for v in (*controls,*(['volume_fixed_rel'] if ready else [])):
        folder=root/'train'/v
        with (folder/'gvar_epochs.jsonl').open('w') as f:
            for i in range(21):f.write(json.dumps({'epoch':i,'train_loss':.8+i*.001,'val_loss':1.0+i*.001})+'\n')
        for e in (19,20): (folder/f'checkpoint_epoch_{e:03d}.tar').write_bytes(b'mock checkpoint')


def p1_demo(root):
    cases=root/'cases';probes=root/'probes';cases.mkdir();probes.mkdir()
    ap_rows=[]; chosen=[]; probe_rows=[]; summary=[]
    for scene in (165,167):
        for i,frame in enumerate((0,10,20,30)):
            delta=[-3.,-2.,-1.,1.][i]+(scene-165)*.05
            cls='best' if i==3 else 'worst'
            selected={'scene_id':scene,'frame_id':frame,'selection':cls,
                      'delta_AP_pp':delta,'delta_mu04_pp':delta*.5,'delta_mu08_pp':delta*1.2}
            chosen.append(selected)
            ap_rows.append({'scene_id':scene,'frame_id':frame,'delta_AP_pp':delta,
                            'delta_mu04_pp':delta*.5,'delta_mu08_pp':delta*1.2,
                            'selection':cls})
            valid_count=0
            for k in range(36):
                role=('left_contact','right_contact','closing','finger_body','palm','approach')[k//6]
                render=(k%6 < 4 + (i%2))  # valid role/frames vary
                sensor=(k%6 >= 1)  # not the same validity mask
                valid_count+=int(render)
                render_err=.015+.002*i+.001*(k%6)
                sensor_err=.005+.001*(k%6)
                probe_rows.append({
                    'variant':'volume_rel','scene_id':scene,'frame_id':frame,'role':role,
                    'grasp_rank':1,'probe_idx':k,'projection_valid':True,
                    'foreground_valid':True,'rendered_valid':render,'sensor_valid':sensor,
                    'rendered_error_m':render_err if render else None,
                    'sensor_error_m':sensor_err if sensor else None,
                    'proxy_depth_std_m':.001+.0001*(k%6) if render else None,
                    'proxy_depth_gradient_mpx':.003+.00015*(k%6) if render else None,
                    'uncertainty_sigma_m':None,'source_frame_delta_AP_pp':delta})
            summary.append({'variant':'volume_rel','scene_id':scene,'frame_id':frame,
                            'status':'ok','num_grasps':1,'total_probes':36,
                            'valid_rendered_object_probes':valid_count,
                            'valid_probe_fraction':valid_count/36,
                            'source_frame_delta_AP_pp':delta})
    selected={'split':'test_novel','epoch':19,'reference':'baseline','comparison':'volume_rel',
              'selected_frames':chosen}
    put_json(cases/'selected_frames.json',selected)
    write_csv(cases/'novel_frames.csv',ap_rows)
    write_csv(probes/'probe_records.csv',probe_rows)
    write_csv(probes/'frame_probe_summary.csv',summary)
    identity=hashlib.sha256((cases/'selected_frames.json').read_bytes()).hexdigest()
    put_json(probes/'manifest.json',{'variants':['volume_rel'],'rank_topk':1,
             'source_selection_sha256':identity,'num_probe_records':len(probe_rows),
             'uncertainty_provenance':{'volume_rel':{'uncertainty_source':'none'}}})
    return cases,probes


def test_p0_pending_never_invents_factorial(tmp_path):
    root=tmp_path/'gvar';p0_demo(root)
    status=p0_run(root,tmp_path/'analysis',bootstrap=200,ap_tolerance=100)
    assert status['status']=='pending'
    assert 'volume_fixed_rel' in status['missing_files']
    assert not (tmp_path/'analysis/factorial').exists()
    assert 'No interaction' in (tmp_path/'analysis/P0_DECISION.md').read_text()


def test_p0_pending_can_fail_explicitly(tmp_path):
    root=tmp_path/'gvar';p0_demo(root)
    with pytest.raises(AuditError,match='pending'):
        p0_run(root,tmp_path/'analysis',bootstrap=200,ap_tolerance=100,require_ready=True)
    assert (tmp_path/'analysis/P0_STATUS.json').exists()


def test_p0_complete_factorial_and_paired_ci(tmp_path):
    root=tmp_path/'gvar';p0_demo(root,ready=True)
    out=tmp_path/'analysis'
    status=p0_run(root,out,bootstrap=250,seed=7,ap_tolerance=100)
    assert status['status']=='complete'
    assert (out/'factorial/factorial_effects.csv').exists()
    assert (out/'paired_six/paired_summary.csv').exists()
    assert len(get_csv(out/'factorial/factorial_effects.csv'))==7*12*4
    assert len(get_csv(out/'paired_six/paired_summary.csv'))==6*4
    r=next(x for x in get_csv(out/'factorial/factorial_effects.csv')
           if x['metric']=='AP' and x['split']=='Mean' and x['contrast']=='interaction')
    assert float(r['delta_pp'])==pytest.approx(0,abs=1e-5)
    assert status['interpretation']['mean_interval_direction']=='inconclusive_scene_interval'
    assert 'Novel μ=0.4' in (out/'P0_DECISION.md').read_text()


def test_p0_controls_ap_drift_fail_fast(tmp_path):
    root=tmp_path/'gvar';p0_demo(root,ready=True)
    with pytest.raises(AuditError,match='Historical baseline Mean AP changed'):
        p0_run(root,tmp_path/'analysis',bootstrap=200,ap_tolerance=.02)
    assert not (tmp_path/'analysis').exists()


def test_p0_refuses_training_directory_outputs(tmp_path):
    root=tmp_path/'gvar';p0_demo(root)
    with pytest.raises(AuditError,match='under train/'):
        p0_run(root,root/'train/new_analysis',bootstrap=200,ap_tolerance=100)


def test_p0_truncated_e20_training_log_fails(tmp_path):
    root=tmp_path/'gvar';p0_demo(root,ready=True)
    path=root/'train/volume_fixed_rel/gvar_epochs.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(AuditError,match='expected completed'):
        p0_run(root,tmp_path/'analysis',bootstrap=200,ap_tolerance=100)


def test_p1_reports_matched_reference_errors_by_role_and_scene(tmp_path):
    cases,probes=p1_demo(tmp_path)
    out=tmp_path/'result'
    stat=p1_run(probes,cases,out,['volume_rel'],min_probes=3,min_frames=3)
    assert stat['status']=='complete'
    assert stat['frames']==8 and stat['probes']==8*36
    scene_role=get_csv(out/'per_scene_role.csv')
    assert len(scene_role)==2*6
    closing=next(x for x in scene_role if x['scene_id']=='165' and x['role']=='closing')
    assert float(closing['rendered_coverage'])==pytest.approx(4.5/6,abs=1e-6)
    assert int(closing['both_reference_valid'])>0
    # Only paired set may be subtracted; base rendered MAE is on a different mask.
    assert float(closing['paired_error_difference_mm'])>0
    assert float(closing['paired_rendered_mae_mm'])>float(closing['paired_sensor_mae_mm'])
    links=get_csv(out/'ap_associations.csv')
    assert len(links)==12*3
    assert all(row['validity'].startswith('descriptive_only') for row in links)
    assert 'not an unbiased Novel distribution' in (out/'P1_AUDIT.json').read_text()
    assert 'independent' in (out/'P1_REPORT.md').read_text()


def test_p1_within_scene_association_avoids_scene_offset(tmp_path):
    rows=[]
    for s in (160,161):
        for i in range(4):
            rows.append({'scene_id':s, 'x': i+100*(s-160), 'y':i-100*(s-160)})
    within,ns,n=pearson_within_scene(rows,'x','y')
    assert within==pytest.approx(1.) and ns==2 and n==8
    assert spearman([1]*10,range(10)) is None


def test_p1_bad_selection_hash_fails_without_artifacts(tmp_path):
    cases,probes=p1_demo(tmp_path)
    dat=json.loads((cases/'selected_frames.json').read_text());dat['selected_frames'][0]['selection']='best'
    put_json(cases/'selected_frames.json',dat)
    with pytest.raises(DataError,match='selection hash'):
        p1_run(probes,cases,tmp_path/'out',['volume_rel'])
    assert not (tmp_path/'out').exists()


def test_p1_duplicate_probe_fails_without_artifacts(tmp_path):
    cases,probes=p1_demo(tmp_path)
    rows=get_csv(probes/'probe_records.csv');rows.append(dict(rows[0]))
    write_csv(probes/'probe_records.csv',rows)
    with pytest.raises(DataError,match='Duplicate physical probe'):
        p1_run(probes,cases,tmp_path/'out',['volume_rel'])
    assert not (tmp_path/'out').exists()


def test_p1_wrong_frame_ap_fails_without_artifacts(tmp_path):
    cases,probes=p1_demo(tmp_path)
    rows=get_csv(probes/'probe_records.csv');rows[0]['source_frame_delta_AP_pp']='999'
    write_csv(probes/'probe_records.csv',rows)
    with pytest.raises(DataError,match='Probe/AP mismatch'):
        p1_run(probes,cases,tmp_path/'out',['volume_rel'])


def test_p1_rejects_invalid_reference_mask(tmp_path):
    cases,probes=p1_demo(tmp_path)
    rows=get_csv(probes/'probe_records.csv')
    a=next(x for x in rows if x['rendered_valid']=='True')
    a['foreground_valid']='False'
    write_csv(probes/'probe_records.csv',rows)
    with pytest.raises(DataError,match='Illogical validity'):
        p1_run(probes,cases,tmp_path/'out',['volume_rel'])


def test_p1_rejects_nonempty_output(tmp_path):
    cases,probes=p1_demo(tmp_path)
    out=tmp_path/'out';out.mkdir();(out/'preserve.txt').write_text('not yours')
    with pytest.raises(DataError,match='refusing overwrite'):
        p1_run(probes,cases,out,['volume_rel'])
    assert (out/'preserve.txt').read_text()=='not yours'
