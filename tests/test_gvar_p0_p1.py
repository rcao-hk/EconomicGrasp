"""P0/P1 CPU regressions: action support factorial, scene/frame alignment, uncertainty."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_gvar_scene_paired import synthetic_gvar
from models.gripper_volume_reader import GVARConfig, GripperVolumeReader, VARIANTS, canonical_probes
from diagnose_gvar_novel_failures import analyze, frame_ap, grasp_stats
from export_gvar_probe_sidecars import selected_frames
from analyze_gvar_probe_uncertainty import predicted_proxies, records_for_frame, curves, spearman


def create_grasps(root, variants, scenes, frames, epoch=19):
    for v in variants:
        for scene in scenes:
            for f in frames:
                p=root/'eval'/f'{v}_e{epoch}'/'test_novel'/f'scene_{scene:04d}'/'realsense'/f'{f:04d}.npy'
                p.parent.mkdir(parents=True,exist_ok=True)
                arr=np.zeros((4,17),dtype=np.float32)
                arr[:,0]=[0.8,0.6,0.4,0.2]
                arr[:,1]=.06;arr[:,2]=.02;arr[:,3]=[.01,.02,.03,.04]
                arr[:,4:13]=np.eye(3).reshape(-1);arr[:,13:16]=[0,0,.6]
                np.save(p,arr)


def test_fixed_rel_config_capacity_matches():
    assert 'volume_fixed_rel' in VARIANTS
    models=[]
    for v in ('volume_fixed','volume','volume_rel','volume_fixed_rel'):
        cfg=GVARConfig(variant=v,reader_dim=16,reader_heads=4,reader_dropout=0.,action_chunk=4,activation_checkpoint=False)
        models.append(GripperVolumeReader(8,16,cfg))
    assert len(set(sum(p.numel() for p in m.parameters()) for m in models))==1
    assert all(list(models[0].state_dict())==list(m.state_dict()) for m in models)


def test_fixed_rel_probes_constant_but_reader_geometry_trainable():
    from test_gvar import inputs
    r,q,c=inputs('volume_rel')
    r.cfg=GVARConfig(variant='volume_fixed_rel',reader_dim=16,reader_heads=4,reader_dropout=0.,action_chunk=7,activation_checkpoint=False)
    out=r(q,c);assert out.shape==q.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert c['depth'].grad is None and c['centers'].grad is None
    assert c['pregeom'].grad is not None
    assert all(p.grad is not None for p in r.parameters())
    d=torch.tensor([.01,.04]);x,_=canonical_probes(torch.full_like(d,r.cfg.fixed_insertion_m),r.cfg)
    torch.testing.assert_close(x[0],x[1])


def test_fixed_rel_geometry_branch_is_active():
    from test_gvar import inputs
    baseline,q,c=inputs('volume_fixed')
    rel,_,_=inputs('volume_rel')
    rel.cfg=GVARConfig(variant='volume_fixed_rel',reader_dim=16,reader_heads=4,reader_dropout=0.,action_chunk=7,activation_checkpoint=False)
    baseline.load_state_dict(rel.state_dict())
    assert torch.max((baseline(q,c)-rel(q,c)).abs())>1e-7


def test_selected_frames_validation(tmp_path):
    p=tmp_path/'sel.json'
    p.write_text(json.dumps({'split':'test_novel','selected_frames':[{'scene_id':167,'frame_id':0},{'scene_id':167,'frame_id':10},{'scene_id':167,'frame_id':0}]}))
    assert selected_frames(p)==[(167,0),(167,10)]
    p.write_text(json.dumps({'split':'test_novel','selected_frames':[{'scene_id':167,'frame_id':3}]}))
    with pytest.raises(ValueError,match='Non-canonical'):selected_frames(p)


def test_novel_case_selection(tmp_path):
    root=tmp_path/'gvar';synthetic_gvar(root)
    create_grasps(root,['baseline','volume_rel'],[165,167],[0,10,20,30])
    out=tmp_path/'cases'
    meta=analyze(root,out,[165,167],['baseline','volume_rel'],'baseline','volume_rel',worst_frames=3,best_frames=1,topk=2)
    assert len(meta['selected_frames'])==8
    assert len({(r['scene_id'],r['frame_id']) for r in meta['selected_frames']})==8
    assert (out/'novel_frames.csv').exists() and (out/'selected_frames.json').exists()
    a=np.load(root/'eval/volume_rel_e19/test_novel/ap_test_novel_realsense.npy')
    assert all(x.shape==(26,) for x in frame_ap(a,167))


def test_invalid_grasp_dump_fails(tmp_path):
    p=tmp_path/'bad.npy';np.save(p,np.ones((2,16)))
    with pytest.raises(ValueError,match='Malformed'):grasp_stats(p,topk=5,require_grasps=True)
    with pytest.raises(FileNotFoundError):grasp_stats(tmp_path/'missing.npy',5,True)


def test_uncertainty_proxy_uses_only_predicted_depth():
    z=np.ones((64,64),dtype='float32')*.6;z[:,32:]=.7
    s,g=predicted_proxies(z)
    assert s.shape==g.shape==(1,1,64,64)
    assert s[0,0,30,31]>0 and g[0,0,30,31]>0


def test_probe_error_and_sigma_sampling(tmp_path):
    side=tmp_path/'frame.npz'
    K=np.array([[120.,0.,31.5],[0.,120.,31.5],[0.,0.,1.]],dtype=np.float32)
    np.savez_compressed(side,pred_depth_m=np.full((64,64),.6,dtype='float32'),
        gt_depth_m=np.full((64,64),.62,dtype='float32'),
        sensor_depth_m=np.full((64,64),.58,dtype='float32'),
        foreground_mask=np.ones((64,64),dtype='uint8'),
        crop_rgb=np.ones((64,64,3),dtype='uint8'),K=K,
        uncertainty_sigma_m=np.full((64,64),.02,dtype='float32'))
    create_grasps(tmp_path,['volume_rel'],[167],[0])
    grasp=tmp_path/'eval/volume_rel_e19/test_novel/scene_0167/realsense/0000.npy'
    records,stats=records_for_frame(side,grasp,'volume_rel',167,0,2,
       {'selection':'worst','delta_AP_pp':-1.},GVARConfig(variant='volume_rel'))
    good=[r for r in records if r['rendered_valid']]
    assert good and stats['valid_rendered_object_probes']>0
    assert all(r['rendered_error_m']==pytest.approx(.02,abs=1e-6) for r in good)
    assert all(r['uncertainty_sigma_m']==pytest.approx(.02,abs=1e-6) for r in good)


def test_risk_coverage_does_not_create_spurious_external_sigma():
    rows=[{'variant':'volume_rel','rendered_error_m':(i+1)*.001,
           'proxy_depth_std_m':(i+1)*.0003,'proxy_depth_gradient_mpx':(i+1)*.0002,
           'uncertainty_sigma_m':None} for i in range(100)]
    bins,risk,summary=curves(rows,10)
    assert len(summary)==2 and len(risk)==10 and len(bins)==20
    assert spearman([1]*20,list(range(20))) is None
    rows=[dict(x,uncertainty_sigma_m=(i+1)*.0013) for i,x in enumerate(rows)]
    _,_,summary=curves(rows,10)
    ext=next(s for s in summary if s['score_type']=='external_sigma')
    assert ext['spearman_proxy_vs_abs_error']==pytest.approx(1.)
    assert ext['coverage_le_1sigma_vs_68pct']==pytest.approx(1.)


def test_factorial_additive_example(tmp_path):
    from analyze_gvar_factorial import run as factorial_run
    import shutil,csv
    root=tmp_path/'gvar';synthetic_gvar(root,('volume_fixed','volume','volume_rel'))
    shutil.copytree(root/'train/volume_fixed',root/'train/volume_fixed_rel')
    p=root/'train/volume_fixed_rel/gvar_protocol.json';j=json.loads(p.read_text());j['gvar_config']['variant']='volume_fixed_rel';p.write_text(json.dumps(j))
    for split in ('test_seen','test_similar','test_novel'):
        src=root/'eval/volume_fixed_e19'/split;dst=root/'eval/volume_fixed_rel_e19'/split;shutil.copytree(src,dst)
        p=dst/'gvar_inference_protocol.json';j=json.loads(p.read_text());j['gvar_config']['variant']='volume_fixed_rel'
        j['checkpoint_sha256']=hashlib.sha256(b'volume_fixed_rel').hexdigest();p.write_text(json.dumps(j))
        f=dst/f'ap_{split}_realsense.npy';np.save(f,np.load(f)+np.float32(.01))
    result=factorial_run(root,tmp_path/'factorial',bootstrap=1000,seed=7)
    assert result['rows']==7*12*4
    with (tmp_path/'factorial/factorial_effects.csv').open() as stream:table=list(csv.DictReader(stream))
    interaction=next(x for x in table if x['metric']=='AP' and x['split']=='Mean' and x['contrast']=='interaction')
    assert float(interaction['delta_pp'])==pytest.approx(0.,abs=1e-5)


def test_empty_grasp_records_explicit(tmp_path):
    side=tmp_path/'x.npz';grasp=tmp_path/'empty.npy'
    d=np.ones((64,64),dtype='float32')
    np.savez_compressed(side,pred_depth_m=d,gt_depth_m=d,sensor_depth_m=d,
                        foreground_mask=d.astype('uint8'),crop_rgb=np.ones((64,64,3),dtype='uint8'),K=np.eye(3))
    np.save(grasp,np.empty((0,17),dtype='float32'))
    rows,stats=records_for_frame(side,grasp,'volume_rel',167,0,5,{'selection':'worst','delta_AP_pp':0.},GVARConfig(variant='volume_rel'))
    assert rows==[] and stats['status']=='zero_grasps'
