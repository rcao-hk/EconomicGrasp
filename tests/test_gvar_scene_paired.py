"""End-to-end synthetic regression tests for the strict GVAR scene-paired audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_gvar_scene_paired import (AuditError, SHAPE, SPLITS, FRAMES, _fingerprint,
                                        boot_indices, paired_metric, scene_metric, run, main)

VARIANTS = ("baseline", "slot", "volume", "volume_rel")


def _save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def synthetic_gvar(root: Path, variants=VARIANTS):
    for variant in variants:
        folder = root / "train" / variant
        train = {"gvar_config": {"variant": variant}, "base_main_sha": "52d09f9",
                 "train_original": 2600, "train_gntrans": 2600, "validation_original_seen": 780,
                 "validation_gntrans": 0, "frame_stride": 10, "smoke_max_batches": 0,
                 "per_rank_batch": 3, "world_size": 3, "global_batch": 9,
                 "detach_policy": {"E": True, "Q": True, "C": True},
                 "network_geometry": "predicted metric depth", "train_steps_per_epoch": 578,
                 "seed": 0, "architecture_config": {"pose_depth_mode": "global_film", "kview_mode": "A1", "use_cdf": True}}
        mix = {"train_fraction_each_domain": 0.1, "eval_fraction_each_domain": 0.1,
               "original_train_count": 2600, "gntrans_train_count": 2600,
               "mixed_train_count": 5200, "paired_scene_frame_sampling": True,
               "gntrans_fused_background": True, "gntrans_observed_depth_network_input": False,
               "gntrans_object_depth_supervision": "rendered_gt",
               "original_train_index_sha256": "a"*64, "gntrans_train_index_sha256": "b"*64,
               "original_seen_index_sha256": "c"*64, "gntrans_seen_index_sha256": "d"*64}
        _save_json(folder / "gvar_protocol.json", train)
        _save_json(folder / "gntrans_mix_protocol.json", mix)
        for split in SPLITS:
            folder = root / "eval" / f"{variant}_e19" / split
            ck = hashlib.sha256(variant.encode()).hexdigest()
            protocol = {"gvar_config": {"variant": variant}, "split": split,
                        "camera": "realsense", "completed_epoch": 19, "selected_samples": 780,
                        "frame_stride": 10, "batch_size": 3, "gvar_contract_version": 1, "collision_source": "original_sensor",
                        "network_geometry": "predicted metric depth", "depth_assisted_dataset_preprocessing": True,
                        "collision_thresh": 0.01, "collision_voxel_size": 0.01,
                        "frame_fingerprint": _fingerprint(split), "checkpoint_sha256": ck,
                        "seed": 0, "dataset_root": "/dataset/graspnet"}
            _save_json(folder / "gvar_inference_protocol.json", protocol)
            _save_json(folder / "gvar_inference_summary.json", {"complete":True, "valid_dump_count":780,"expected_count":780})
            ids = np.arange(30,dtype=np.float32).reshape(30,1,1,1)
            frames = np.arange(26,dtype=np.float32).reshape(1,26,1,1)
            ranks = np.arange(50,dtype=np.float32).reshape(1,1,50,1)
            mus = np.arange(6,dtype=np.float32).reshape(1,1,1,6)
            baseline = .18 + ids*.001 + frames*.0002 + ranks*.0001 + mus*.0004
            off = {"baseline":0., "slot": -.002, "volume": .02, "volume_rel":.03,"volume_fixed": .01}[variant]
            shift = (ids - 14.5) * .0003
            data = baseline + (off + shift if variant != "baseline" else 0.)
            np.save(folder / f"ap_{split}_realsense.npy", np.broadcast_to(data,SHAPE).astype("float32"))


@pytest.fixture
def demo(tmp_path):
    root=tmp_path/"gvar"; synthetic_gvar(root); return root


def _load_csv(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))


def test_report_has_all_rows_paired_confidence_and_provenance(demo, tmp_path):
    out=tmp_path/"out"
    r=run(demo,out,list(VARIANTS),"baseline",19,1000,123,0.95)
    assert r["rows"]==16 and r["scene_rows"]==360 and r["frame_rows"]==9360
    paired=_load_csv(out/"paired_summary.csv")
    assert {x['split'] for x in paired}==set(SPLITS)|{'Mean'}
    v=next(x for x in paired if x['variant']=='volume_rel' and x['split']=='Mean')
    assert float(v['delta_pp'])==pytest.approx(3.,abs=.002)
    assert float(v['ci_low_pp'])>0
    assert float(v['ci_high_pp'])>float(v['ci_low_pp'])
    assert v['scene_wins']=='90'
    assert float(next(x for x in paired if x['variant']=='slot' and x['split']=='Mean')['delta_pp'])<0
    md=(out/'REPORT.md').read_text()
    assert "NOT candidate recall" in md
    assert "cannot be attributed solely to reranking" in md
    cross=_load_csv(out/'cross_variant_contrasts.csv')
    direct=next(x for x in cross if x['reference']=='volume' and x['variant']=='volume_rel' and x['split']=='Mean' and x['metric']=='AP')
    assert float(direct['delta_pp']) == pytest.approx(1.,abs=.002)
    assert float(direct['ci_low_pp']) > 0
    assert (out/'cross_variant_scenes.csv').exists()
    audit=json.loads((out/'gvar_scene_paired_audit.json').read_text())
    assert audit['bootstrap_unit'].startswith('scene')
    assert audit['input_count']==4*(2+3*3)
    scenes=_load_csv(out/'scene_level.csv')
    frames=_load_csv(out/'frame_level.csv')
    assert len(scenes)==4*3*30 and len(frames)==4*3*30*26
    assert frames[0]['frame_id']=='0' and frames[25]['frame_id']=='250'
    # AP array was averaged over K and mu, preserving percentage units.
    assert float(frames[0]['baseline_AP'])==pytest.approx(float(frames[0]['variant_AP']),abs=1e-6)


def test_scene_bootstrap_is_stratified_and_deterministic():
    x=np.linspace(-.05,.05,30)
    vectors={s:x.copy() for s in SPLITS}
    idx=boot_indices(800,77)
    a=paired_metric(vectors,idx,.95,'Mean')
    b=paired_metric(vectors,boot_indices(800,77),.95,'Mean')
    assert a==b
    assert a['scene_wins']==45 and a['scene_losses']==45 and a['scene_count']==90
    assert a['ci_low_pp']<0<a['ci_high_pp']
    assert a['ci_high_pp']-a['ci_low_pp']>0.5  # scene clusters, not 23400 i.i.d. elements


def test_scene_metric_exact_dimension_semantics():
    a=np.zeros(SHAPE,dtype=np.float32)
    a[:,:,:,1]=.2
    a[:,:,:,3]=.6
    a[:,:,0,:]+=0.1
    assert scene_metric(a,'AP_mu0.4').shape==(30,)
    assert scene_metric(a,'AP_mu0.4')[0]==pytest.approx(.202)
    assert scene_metric(a,'prefix_precision@1')[0]==pytest.approx(.1+(.2+.6)/6)


def test_bad_frame_fingerprint_rejected_before_any_output(demo,tmp_path):
    p=demo/'eval/volume_e19/test_novel/gvar_inference_protocol.json'
    data=json.loads(p.read_text());data['frame_fingerprint']='0'*64;_save_json(p,data)
    with pytest.raises(AuditError,match='fingerprint'):
        run(demo,tmp_path/'not_created',list(VARIANTS),'baseline',19,200,2,.95)
    assert not (tmp_path/'not_created').exists()


def test_different_dataset_root_rejected(demo,tmp_path):
    p=demo/'eval/volume_rel_e19/test_similar/gvar_inference_protocol.json'
    data=json.loads(p.read_text());data['dataset_root']='/another';_save_json(p,data)
    with pytest.raises(AuditError,match='dataset_root'):
        run(demo,tmp_path/'out',list(VARIANTS),'baseline',19,200,2,.95)


def test_training_split_fingerprint_mismatch_rejected(demo,tmp_path):
    p=demo/'train/slot/gntrans_mix_protocol.json'
    data=json.loads(p.read_text()); data['original_train_index_sha256']='d'*64;_save_json(p,data)
    with pytest.raises(AuditError,match='training index'):
        run(demo,tmp_path/'out',list(VARIANTS),'baseline',19,200,2,.95)


def test_epoch_checkpoint_inconsistency_rejected(demo,tmp_path):
    p=demo/'eval/slot_e19/test_seen/gvar_inference_protocol.json'
    data=json.loads(p.read_text());data['checkpoint_sha256']='1'*64;_save_json(p,data)
    with pytest.raises(AuditError,match='DIFFERENT checkpoints'):
        run(demo,tmp_path/'out',list(VARIANTS),'baseline',19,200,2,.95)


def test_inference_summary_incomplete_rejected(demo,tmp_path):
    p=demo/'eval/volume_e19/test_seen/gvar_inference_summary.json'
    _save_json(p,{"complete":False,"valid_dump_count":779,"expected_count":780})
    with pytest.raises(AuditError,match='Incomplete'):
        run(demo,tmp_path/'out',list(VARIANTS),'baseline',19,200,2,.95)


@pytest.mark.parametrize('kind', ['nan','wrong_shape','too_big'])
def test_corrupt_ap_rejected(demo,tmp_path,kind):
    p=demo/'eval/baseline_e19/test_novel/ap_test_novel_realsense.npy'
    a=np.load(p)
    if kind=='nan': a[0,0,0,0]=np.nan
    if kind=='wrong_shape': a=a[:-1]
    if kind=='too_big': a[0,0,0,0]=2
    np.save(p,a)
    with pytest.raises(AuditError):
        run(demo,tmp_path/'out',list(VARIANTS),'baseline',19,200,2,.95)


def test_volume_fixed_missing_fails_explicitly(demo,tmp_path):
    with pytest.raises(AuditError,match='Missing required JSON'):
        run(demo,tmp_path/'out',list(VARIANTS)+['volume_fixed'],'baseline',19,200,2,.95)


def test_optional_volume_fixed_when_complete(tmp_path):
    root=tmp_path/'gvar';synthetic_gvar(root,('baseline','volume_fixed'))
    r=run(root,tmp_path/'out',['baseline','volume_fixed'],'baseline',19,200,22,.95)
    assert r['rows']==8 and r['mean_ap']['volume_fixed']>r['mean_ap']['baseline']


def test_refuse_overwrite_of_unrelated_results(demo,tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'someone_elses_experiment.txt').write_text('keep')
    with pytest.raises(AuditError,match='Refusing to overwrite'):
        run(demo,out,list(VARIANTS),'baseline',19,200,2,.95,overwrite=True)
    assert (out/'someone_elses_experiment.txt').read_text()=='keep'


def test_rerun_requires_explicit_overwrite(demo,tmp_path):
    out=tmp_path/'out'
    run(demo,out,['baseline','slot'],'baseline',19,200,2,.95)
    with pytest.raises(AuditError,match='Refusing to overwrite'):
        run(demo,out,['baseline','slot'],'baseline',19,200,2,.95)
    r=run(demo,out,['baseline','slot'],'baseline',19,200,2,.95,overwrite=True)
    assert r['rows']==8


def test_script_entrypoint_error_code(demo,tmp_path,capsys):
    code=main(['--root',str(demo),'--output-dir',str(tmp_path/'out'),
               '--variants','baseline','volume_fixed','--bootstrap','200'])
    assert code==2
    assert 'FAILED:' in capsys.readouterr().err


def test_training_global_batch_mismatch_rejected(demo,tmp_path):
    p=demo/'train/slot/gvar_protocol.json'
    data=json.loads(p.read_text());data['global_batch']=3;_save_json(p,data)
    with pytest.raises(AuditError,match='global_batch'):
        run(demo,tmp_path/'out',list(VARIANTS),'baseline',19,200,2,.95)


def test_explicit_contrast_validation(demo,tmp_path):
    with pytest.raises(AuditError,match='both variants'):
        run(demo,tmp_path/'bad',list(VARIANTS),'baseline',19,200,2,.95,
            requested_contrasts=['volume_fixed:volume'])
    out=tmp_path/'good'
    run(demo,out,list(VARIANTS),'baseline',19,200,2,.95,
        requested_contrasts=['baseline:volume'])
    rows=_load_csv(out/'cross_variant_contrasts.csv')
    assert any(x['reference']=='baseline' and x['variant']=='volume' for x in rows)


def test_prevent_stale_cross_variant_contrasts_on_overwrite(demo,tmp_path):
    output=tmp_path/'out'
    run(demo,output,list(VARIANTS),'baseline',19,200,2,.95)
    with pytest.raises(AuditError,match='overwrite existing analysis variants'):
        run(demo,output,['baseline','slot'],'baseline',19,200,2,.95,overwrite=True)
    assert (output/'cross_variant_contrasts.csv').exists()
