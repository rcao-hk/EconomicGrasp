"""P0-B readiness and support-site diagnostics with synthetic canonical GVAR data."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from test_gvar_scene_paired import synthetic_gvar
import analyze_gvar_p0b as p0b
from audit_gvar_p0_support import summarize


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj)+'\n')


def make_data(root,include_new=False):
    synthetic_gvar(root,('volume','volume_rel'))
    offset={'volume_fixed_rel':0.,'volume_hybrid':.01,'volume_inverse_hybrid':.02,'volume_rel':.04}
    # Existing old 'volume_rel' is used but AP values overwritten; this is a
    # synthetic regression fixture, not a claim about the real experiment.
    for v in ('volume_fixed_rel','volume_hybrid', 'volume_inverse_hybrid'):
        if v=='volume_inverse_hybrid' and not include_new:continue
        shutil.copytree(root/'train/volume',root/'train'/v)
        pp=root/'train'/v/'gvar_protocol.json'
        train=json.loads(pp.read_text());train['gvar_config']['variant']=v;put(pp,train)
        for split in ('test_seen','test_similar','test_novel'):
            orig=root/'eval/volume_e19'/split
            dest=root/'eval'/f'{v}_e19'/split
            shutil.copytree(orig,dest)
            p=dest/'gvar_inference_protocol.json'
            d=json.loads(p.read_text());d['gvar_config']['variant']=v
            ck=root/'train'/v/'checkpoint_epoch_019.tar'; ck.write_bytes(v.encode())
            d['checkpoint_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest()
            put(p,d)
    for v in ('volume_fixed_rel','volume_hybrid','volume_rel','volume_inverse_hybrid'):
        if v=='volume_inverse_hybrid' and not include_new:continue
        ck=root/'train'/v/'checkpoint_epoch_019.tar'
        if not ck.exists():ck.write_bytes(v.encode())
        (root/'train'/v/'checkpoint_epoch_020.tar').write_bytes(b'e20'+v.encode())
        epoch=root/'train'/v/'gvar_epochs.jsonl'
        epoch.write_text(''.join(json.dumps({'epoch':i,'train_loss':.8,'val_loss':1.0})+'\n' for i in range(21)))
        for split in ('test_seen','test_similar','test_novel'):
            p=root/'eval'/f'{v}_e19'/split/'gvar_inference_protocol.json'
            d=json.loads(p.read_text());d['checkpoint_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest()
            put(p,d)
            arr=np.full((30,26,50,6),.25+offset[v],dtype=np.float32)
            np.save(p.with_name(f'ap_{split}_realsense.npy'),arr)


def test_p0b_pending_is_explicit(tmp_path):
    root=tmp_path/'gvar';make_data(root)
    result=p0b.analyze(root,tmp_path/'out',bootstrap=200)
    assert result['status']=='pending'
    assert 'volume_inverse_hybrid' in result['missing_files']
    assert (tmp_path/'out/STATUS.json').exists()
    assert not (tmp_path/'out/factorial_effects.csv').exists()


def test_p0b_full_2x2_and_interaction_ci(tmp_path,monkeypatch):
    root=tmp_path/'gvar';make_data(root,include_new=True)
    monkeypatch.setattr(p0b,'EXPECTED',{})
    result=p0b.analyze(root,tmp_path/'out',bootstrap=400,seed=3)
    assert result['status']=='complete'
    import csv
    with (tmp_path/'out/factorial_effects.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==7*12*4
    effect=next(x for x in rows if x['metric']=='AP' and x['split']=='Mean' and x['contrast']=='interaction')
    assert float(effect['delta_pp'])==pytest.approx(1.0,abs=1e-4)  # (.04-.02)-(.01-.00)=+.01
    assert float(effect['ci_low_pp'])>0


def test_p0b_checkpoint_identity_rejected(tmp_path,monkeypatch):
    root=tmp_path/'gvar';make_data(root,include_new=True)
    monkeypatch.setattr(p0b,'EXPECTED',{})
    ck=root/'train/volume_inverse_hybrid/checkpoint_epoch_019.tar'
    ck.write_bytes(b'wrong checkpoint contents')
    with pytest.raises(p0b.AuditError,match='Checkpoint identity mismatch'):
        p0b.analyze(root,tmp_path/'out',bootstrap=200)
    assert not (tmp_path/'out').exists()


def test_projection_summary_fractions():
    data=[]
    for d in (1,4):
        for role in ('palm','closing'):
            for i in range(10):
                fixed=i<5; geom=i<8;known=i<7
                data.append({'depth_bin':d,'role':role,'pixel_shift_px':float(i) if fixed else None,
                             'fixed_projected_valid':int(fixed), 'dynamic_projected_valid':int(geom),
                             'geom_depth_known':int(known),
                             'geom_known_rgb_mask_conflict':int(known and not fixed),
                             'both_projected_valid':int(fixed and geom)})
    r=summarize(data)
    assert len(r)==4
    assert all(x['geometry_known_but_fixed_rgb_masked_ratio']==pytest.approx(.2) for x in r)
    assert all(x['mean_pixel_shift_px']==pytest.approx(2.) for x in r)
    assert all(x['paired_projection_count']==5 for x in r)
