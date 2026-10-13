"""Synthetic tests for P0-A historical e19 identity gate."""
import hashlib
import json
from pathlib import Path
import pytest
from verify_gvar_p0a_checkpoint import historical_identity

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj),encoding='utf-8')

def make_root(tmp_path):
    root=tmp_path/'gvar'
    train=root/'train/volume_rel'
    cfg={'variant':'volume_rel','reader_dim':64}
    ckhash=hashlib.sha256(b'historical-e19-model').hexdigest()
    write(train/'gvar_protocol.json',{'gvar_config':cfg,
        'detach_policy':{'E':True,'Q':True,'C':True},
        'train_original':2600,'train_gntrans':2600,
        'validation_original_seen':780,'validation_gntrans':0,
        'world_size':3,'per_rank_batch':3,'global_batch':9,
        'seed':0,'frame_stride':10})
    write(train/'gntrans_mix_protocol.json',{
        'original_train_count':2600,'gntrans_train_count':2600,
        'mixed_train_count':5200,'paired_scene_frame_sampling':True})
    for split in ('test_seen','test_similar','test_novel'):
        folder=root/'eval/volume_rel_e19'/split
        write(folder/'gvar_inference_protocol.json',{
            'checkpoint_sha256':ckhash,'completed_epoch':19,
            'gvar_config':cfg,'split':split,'camera':'realsense',
            'batch_size':3,'selected_samples':780,'frame_stride':10,
            'collision_source':'original_sensor',
            'collision_thresh':.01,'collision_voxel_size':.01,
            'network_geometry':'predicted metric depth',
            'frame_fingerprint':'a'*64})
        write(folder/'gvar_inference_summary.json',{
            'complete':True,'valid_dump_count':780,'expected_count':780})
        (folder/f'ap_{split}_realsense.npy').write_bytes(b'synthetic-not-real-AP')
    return root,ckhash

def test_historical_checksum_matches_all_three_splits(tmp_path):
    root,digest=make_root(tmp_path)
    status=historical_identity(root,digest)
    assert status['status']=='matches_all_historical_volume_rel_e19_evaluations'
    assert len(status['historical_manifest_sources'])==3

def test_historical_identity_rejects_changed_checkpoint(tmp_path):
    root,digest=make_root(tmp_path)
    with pytest.raises(ValueError,match='checkpoint_sha256'):
        historical_identity(root,'0'*64)

def test_historical_identity_rejects_incomplete_split(tmp_path):
    root,digest=make_root(tmp_path)
    path=root/'eval/volume_rel_e19/test_novel/gvar_inference_summary.json'
    obj=json.loads(path.read_text());obj['complete']=False;write(path,obj)
    with pytest.raises(ValueError,match='inference is incomplete'):
        historical_identity(root,digest)
