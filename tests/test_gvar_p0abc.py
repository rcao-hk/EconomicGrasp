"""Synthetic-contract and CPU tests for P0-A/B/C; NO server datasets required."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from models.gripper_volume_reader import (GVARConfig, GripperVolumeReader, COUNTERFACTUAL_SOURCES,
                                          VARIANTS, support_probe_depths)
from utils.gvar_p0_sampling import per_frame_sampling_seed, gntrans_rgb_path
from analyze_gvar_scene_paired import SPLITS, SHAPE, _fingerprint
from analyze_gvar_p0a import analyze as analyze_cf
from analyze_gvar_p0c import analyze as analyze_domain
from analyze_gvar_p0b import analyze as analyze_inverse
from utils.gvar_p0_analysis import ProtocolError
from test_gvar import inputs


def _json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True)+'\n')


def _synthetic_p0(root,kind):
    modes=('r0','r1','r2','r3','r4') if kind=='p0a' else ('r0',)
    variants=('volume_rel',) if kind=='p0a' else ('baseline','volume_fixed','volume_rel')
    domains=('original',) if kind=='p0a' else ('original','gntrans')
    for v in variants:
        ck=hashlib.sha256(v.encode()).hexdigest()
        for mode in modes:
            for domain in domains:
                for split in SPLITS:
                    folder=root/v/domain/mode/split
                    proto={'p0_contract_version':1,'source_domain':domain,
                        'reader_cf_mode':mode,'reader_cf_sources':list(COUNTERFACTUAL_SOURCES[mode]),
                        'gvar_config':{'variant':v},
                        'completed_epoch':19,'camera':'realsense','checkpoint_sha256':ck,
                        'split':split,'dataset_root':'/dataset/graspnet','gntrans_rgb_root':
                          '/dataset/GN-Trans' if domain=='gntrans' else None,
                        'selected_samples':780,'frame_stride':10,'frame_fingerprint':_fingerprint(split),
                        'preprocess_cloud_source':'original_sensor',
                        'preprocess_protocol':'paired_rgb_original_sensor_crop_v1',
                        'collision_source':'original_sensor','collision_thresh':.01,
                        'collision_voxel_size':.01,'gt_depth_network_input':False,
                        'sample_policy':'per_frame_seed_v1','network_geometry':'predicted metric depth',
                        'sample_seed':20261012,'batch_size':3,'num_workers':4,'seed':0,'max_batches':0,
                        'image_source':'original_rgb' if domain=='original' else 'gntrans_rendered_rgb'}
                    _json(folder/'gvar_p0_protocol.json',proto)
                    _json(folder/'gvar_p0_summary.json',{'complete':True,'valid_dump_count':780,'expected_count':780})
                    offsets={v:.00 for v in variants}
                    if kind=='p0a':
                        offsets={'r0':0.,'r1':-.02,'r2':-.01,'r3':-.005,'r4':-.03}
                        val=.4+offsets[mode]
                    else:
                        val=.4 + {'baseline':0.,'volume_fixed':.02,'volume_rel':.04}[v]
                        if domain=='gntrans':val-= {'baseline':.1,'volume_fixed':.09,'volume_rel':.075}[v]
                    # smooth scene-level heterogeneity for nondegenerate CI
                    a=np.full(SHAPE,val,dtype='float32')
                    a+=np.arange(30,dtype='float32')[:,None,None,None]*.0001
                    np.save(folder/f'ap_{split}_realsense.npy',a)


def test_sampling_same_per_frame_across_domain_and_workers():
    idxs=(0,10,25,256*30,256*30+250)
    assert [per_frame_sampling_seed(i,20261012) for i in idxs] == [per_frame_sampling_seed(i,20261012) for i in reversed(idxs)][::-1]
    assert len({per_frame_sampling_seed(i,20261012) for i in idxs})==len(idxs)
    with pytest.raises(ValueError):per_frame_sampling_seed(-1,0)
    with pytest.raises(ValueError):per_frame_sampling_seed(0,2**32)
    assert gntrans_rgb_path(Path('/data/trans'),'scene_0165',80)==Path('/data/trans/scenes/00165/0080_color.png')
    with pytest.raises(ValueError):gntrans_rgb_path(Path('/data/trans'),'scene_165',80)


def test_counterfactual_modes_and_training_safety():
    assert list(COUNTERFACTUAL_SOURCES)==['r0','r1','r2','r3','r4']
    reader,queries,ctx=inputs('volume_rel')
    with pytest.raises(RuntimeError):reader.set_counterfactual('r2')
    reader.eval()
    with torch.no_grad():
        base=reader(queries,ctx)
        reader.set_counterfactual('r0')
        torch.testing.assert_close(base,reader(queries,ctx),atol=0,rtol=0)
        for mode in ('r1','r2','r3','r4'):
            reader.set_counterfactual(mode)
            output=reader(queries,ctx)
            assert output.shape==base.shape and torch.isfinite(output).all()
            assert (base-output).abs().max()>1e-7
    with pytest.raises(ValueError):reader.set_counterfactual('bad')
    fixed,_,_=inputs('volume_fixed_rel');fixed.eval()
    with pytest.raises(ValueError):fixed.set_counterfactual('r1')


def test_r4_matches_hybrid_when_transplanting_reader_weights():
    old,q,c=inputs('volume_rel')
    hybrid,_,_=inputs('volume_hybrid')
    hybrid.load_state_dict(old.state_dict(),strict=True)
    old.eval();hybrid.eval();old.set_counterfactual('r4')
    with torch.no_grad():
        torch.testing.assert_close(old(q,c),hybrid(q,c),atol=0,rtol=0)


def test_inverse_hybrid_capacity_and_geometry_support():
    assert 'volume_inverse_hybrid' in VARIANTS
    cfg=GVARConfig(variant='volume_inverse_hybrid',reader_dim=16,reader_heads=4)
    depths=torch.tensor([.01,.04])
    assert torch.equal(support_probe_depths(depths,cfg,'rgb'),depths)
    assert torch.allclose(support_probe_depths(depths,cfg,'geometry'),torch.tensor([.025,.025]))
    a,q,c=inputs('volume_inverse_hybrid')
    b,_,_=inputs('volume_rel')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    result=a(q,c)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert c['depth'].grad is None and c['centers'].grad is None
    assert c['pregeom'].grad is not None
    assert all(p.grad is not None for p in a.parameters())


def test_p0a_pairs_and_cis(tmp_path):
    root=tmp_path/'cf';_synthetic_p0(root,'p0a')
    info=analyze_cf(root,tmp_path/'report',bootstrap=250,seed=7)
    assert info['rows']==5*12*4
    import csv
    with (tmp_path/'report/reader_cf_effects.csv').open() as f:rows=list(csv.DictReader(f))
    r1=next(x for x in rows if x['counterfactual']=='r1' and x['metric']=='AP' and x['split']=='Mean')
    assert float(r1['delta_pp'])==pytest.approx(-2.,abs=.0001)
    assert float(r1['ci_high_pp'])<0
    baseline=next(x for x in rows if x['counterfactual']=='r0' and x['metric']=='AP' and x['split']=='Mean')
    assert float(baseline['delta_pp'])==pytest.approx(0.)


def test_p0a_rejects_mismatched_protocol(tmp_path):
    root=tmp_path/'cf';_synthetic_p0(root,'p0a')
    p=root/'volume_rel/original/r2/test_novel/gvar_p0_protocol.json'
    d=json.loads(p.read_text());d['sample_seed']=123;_json(p,d)
    with pytest.raises(ProtocolError,match='sample_seed'):
        analyze_cf(root,tmp_path/'out',bootstrap=200)
    assert not (tmp_path/'out').exists()


def test_p0a_rejects_mislabeled_sources(tmp_path):
    root=tmp_path/'cf';_synthetic_p0(root,'p0a')
    p=root/'volume_rel/original/r2/test_novel/gvar_p0_protocol.json'
    d=json.loads(p.read_text())
    d['reader_cf_sources']=['dynamic','dynamic','dynamic']
    _json(p,d)
    with pytest.raises(ProtocolError,match='reader_cf_sources'):
        analyze_cf(root,tmp_path/'report',bootstrap=200)
    assert not (tmp_path/'report').exists()


def test_p0a_rejects_missing_ap(tmp_path):
    root=tmp_path/'cf';_synthetic_p0(root,'p0a')
    (root/'volume_rel/original/r2/test_seen/ap_test_seen_realsense.npy').unlink()
    with pytest.raises(ProtocolError,match='Missing official AP'):
        analyze_cf(root,tmp_path/'out',bootstrap=200)


def test_p0c_measures_domain_gap_and_difference_of_differences(tmp_path):
    root=tmp_path/'domain';_synthetic_p0(root,'p0c')
    stat=analyze_domain(root,tmp_path/'out',bootstrap=500,seed=3)
    assert stat['input_ap_arrays']==3*2*3
    import csv
    with (tmp_path/'out/domain_contrasts.csv').open() as f:rows=list(csv.DictReader(f))
    gap=next(x for x in rows if x['variant']=='volume_rel' and x['split']=='Mean' and x['metric']=='AP' and x['contrast']=='material_shift_gntrans_minus_original')
    dod=next(x for x in rows if x['variant']=='volume_rel' and x['split']=='Mean' and x['metric']=='AP' and x['contrast']=='material_gap_change_vs_baseline')
    assert float(gap['delta_pp'])==pytest.approx(-7.5,abs=1e-3)
    assert float(dod['delta_pp'])==pytest.approx(2.5,abs=1e-3)


def test_p0c_rejects_checkpoint_mismatch_between_domains(tmp_path):
    root=tmp_path/'domain';_synthetic_p0(root,'p0c')
    p=root/'volume_rel/gntrans/r0/test_novel/gvar_p0_protocol.json'
    d=json.loads(p.read_text());d['checkpoint_sha256']='0'*64;_json(p,d)
    with pytest.raises(ProtocolError,match='checkpoint_sha256'):
        analyze_domain(root,tmp_path/'out',bootstrap=200)


def test_p0c_rejects_different_sampling_or_collision(tmp_path):
    root=tmp_path/'domain';_synthetic_p0(root,'p0c')
    p=root/'volume_fixed/gntrans/r0/test_novel/gvar_p0_protocol.json'
    d=json.loads(p.read_text());d['collision_source']='virtual_gt';_json(p,d)
    with pytest.raises(ProtocolError,match='collision_source'):
        analyze_domain(root,tmp_path/'out',bootstrap=200)


def test_own_cli_help_does_not_import_graspnet_cuda():
    result=subprocess.run([sys.executable,str(ROOT/'inference_gvar_p0.py'),'--help'],
                          text=True,capture_output=True,timeout=10)
    assert result.returncode==0 and 'p0_cf_mode' in result.stdout


def test_scripts_bash_syntax():
    for fn in ('run_gvar_p0_eval.sh','run_gvar_inverse_hybrid_train.sh'):
        x=subprocess.run(['bash','-n',str(ROOT/'scripts'/fn)],capture_output=True,text=True)
        assert x.returncode==0,x.stderr


def test_registration_audit_renders_shared_mask_outline(tmp_path):
    from PIL import Image
    from check_gvar_p0_rgb_registration import inspect, boundary
    root=tmp_path/'original';trans=tmp_path/'gntrans'
    a=root/'scenes/scene_0165/realsense/rgb/0080.png'
    b=trans/'scenes/00165/0080_color.png'
    label=root/'scenes/scene_0165/realsense/label/0080.png'
    for p in (a,b,label):p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.full((80,100,3),120,dtype='uint8'),'RGB').save(a)
    Image.fromarray(np.full((80,100,3),220,dtype='uint8'),'RGB').save(b)
    seg=np.zeros((80,100),dtype='uint8');seg[25:50,40:70]=3
    Image.fromarray(seg).save(label)
    audit=inspect(root,trans,tmp_path/'registration',(165,),(80,),max_width=128)
    assert len(audit)==1 and audit[0]['boundary_pixels']>0
    assert (tmp_path/'registration/scene_0165_frame_0080.png').exists()
    assert 'not proof' in (tmp_path/'registration/REGISTRATION_AUDIT.json').read_text().lower()
    with pytest.raises(ValueError):inspect(root,trans,tmp_path/'different',(165,),(81,))
    assert boundary(seg>0).shape==seg.shape

def test_p0a_reader_checkpoint_preflight(tmp_path):
    """Saved reader parameters must strict-load and survive R0--R4 unchanged."""
    cfg=GVARConfig(variant='volume_rel',reader_dim=16,reader_heads=4,action_chunk=16)
    reader=GripperVolumeReader(feat_dim=8,hidden_dim=32,cfg=cfg)
    ck={'gvar_contract_version':1,'gvar_config':cfg.to_dict(),
        'detach_policy':{'E':True,'Q':True,'C':True},'completed_epoch':19,
        'model_state_dict':{'kview_grasp_module.decoder.reader.'+k:v
                            for k,v in reader.state_dict().items()}}
    checkpoint=tmp_path/'e19.tar'
    torch.save(ck,checkpoint)
    from verify_gvar_p0a_checkpoint import verify
    result=verify(checkpoint)
    assert result['status']=='reader_checkpoint_preflight_passed'
    assert result['all_reader_state_unchanged']
    assert result['r0_replay_max_abs']==0.0 and result['r4_hybrid_same_weights_max_abs']==0.0
    assert set(result['counterfactual_max_abs_delta_from_r0'])=={'r1','r2','r3','r4'}
    ck['completed_epoch']=20
    torch.save(ck,checkpoint)
    with pytest.raises(ValueError,match='completed_epoch=19'):
        verify(checkpoint)


@pytest.mark.parametrize("count",(18,36))
def test_p0_counterfactual_modes_support_both_probe_counts(count):
    from dataclasses import replace
    reader,queries,ctx=inputs("volume_rel")
    reader.cfg=replace(reader.cfg,num_probes=count)
    hybrid,_,_=inputs("volume_hybrid")
    hybrid.cfg=replace(hybrid.cfg,num_probes=count)
    hybrid.load_state_dict(reader.state_dict(),strict=True)
    reader.eval();hybrid.eval()
    with torch.no_grad():
        r0=reader(queries,ctx)
        reader.set_counterfactual("r0")
        torch.testing.assert_close(r0,reader(queries,ctx),rtol=0,atol=0)
        for mode in ("r1","r2","r3","r4"):
            reader.set_counterfactual(mode)
            assert torch.isfinite(reader(queries,ctx)).all()
        torch.testing.assert_close(reader(queries,ctx),hybrid(queries,ctx),rtol=0,atol=0)

