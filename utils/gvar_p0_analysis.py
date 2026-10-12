"""Shared strict read-only input audit for GVAR P0-A/P0-C AP tensors."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_gvar_scene_paired import SPLITS, SHAPE, _fingerprint


class ProtocolError(ValueError):
    pass


def require(cond,msg):
    if not bool(cond): raise ProtocolError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    require(path.is_file(),f'Missing manifest: {path}')
    try: d=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as e: raise ProtocolError(f'Invalid JSON {path}: {e}') from e
    require(isinstance(d,dict),f'Manifest must be dict {path}')
    return d


def output_dir(root,variant,domain,mode,split):
    return root/variant/domain/mode/split


def load(root,variant,domain,mode,split):
    require(split in SPLITS,f'Unsupported split {split}')
    folder=output_dir(root,variant,domain,mode,split)
    mp=folder/'gvar_p0_protocol.json'
    sp=folder/'gvar_p0_summary.json'
    ap_path=folder/f'ap_{split}_realsense.npy'
    proto,summary=read_json(mp),read_json(sp)
    for key,want in {'p0_contract_version':1,'source_domain':domain,'reader_cf_mode':mode,
                     'split':split,'completed_epoch':19,'camera':'realsense',
                     'selected_samples':780,'frame_stride':10,'preprocess_cloud_source':'original_sensor',
                     'preprocess_protocol':'paired_rgb_original_sensor_crop_v1',
                     'collision_source':'original_sensor','collision_thresh':.01,'collision_voxel_size':.01,
                     'gt_depth_network_input':False,'sample_policy':'per_frame_seed_v1',
                     'network_geometry':'predicted metric depth','max_batches':0,
                     'frame_fingerprint':_fingerprint(split)}.items():
        require(proto.get(key)==want,f'{mp} mismatch {key}: {proto.get(key)!r} != {want!r}')
    require(proto.get('gvar_config',{}).get('variant')==variant,f'{mp}: variant mismatch')
    require(summary.get('complete') is True and summary.get('valid_dump_count')==780
            and summary.get('expected_count')==780, f'{sp}: incomplete inference')
    ck=proto.get('checkpoint_sha256')
    require(isinstance(ck,str) and len(ck)==64 and all(c in '0123456789abcdef' for c in ck),
            f'{mp}: invalid checkpoint SHA256')
    require(ap_path.is_file(),f'Missing official AP tensor: {ap_path}')
    try: ap=np.load(ap_path,allow_pickle=False)
    except (OSError,ValueError) as e: raise ProtocolError(f'Invalid AP tensor {ap_path}: {e}') from e
    require(ap.shape==SHAPE and ap.dtype.kind=='f' and np.isfinite(ap).all(),
            f'Official AP must be finite float {SHAPE}, found {ap.shape}: {ap_path}')
    require(float(ap.min())>=-1e-6 and float(ap.max())<=1+1e-6,
            f'Official AP outside [0,1]: {ap_path}')
    files=[{'path':str(p),'sha256':sha(p)} for p in (mp,sp,ap_path)]
    return {'variant':variant,'domain':domain,'mode':mode,'split':split,'protocol':proto,
            'summary':summary,'ap':ap,'sources':files}


def paired_protocol(a,b,compare_domain=False):
    x,y=a['protocol'],b['protocol']
    for key in ('checkpoint_sha256','completed_epoch','camera','dataset_root','frame_fingerprint',
                'selected_samples','frame_stride','preprocess_cloud_source','preprocess_protocol',
                'collision_source','collision_thresh','collision_voxel_size','sample_policy',
                'sample_seed','batch_size','num_workers','seed','network_geometry','gt_depth_network_input'):
        require(x.get(key)==y.get(key),f'Paired protocol mismatch {key}: {a["variant"]}/{a["domain"]}/{a["mode"]} '
                f'vs {b["variant"]}/{b["domain"]}/{b["mode"]}')
    if not compare_domain:
        require(x.get('source_domain')==y.get('source_domain'),f'Domain mismatch in mode comparison')
    else:
        require(a['variant']==b['variant'] and a['mode']==b['mode'],
                'Material shift comparison must hold model and counterfactual mode fixed')
    return True


def same_pair_except_checkpoint(a,b):
    """Different models under EXACT same sampling and collision convention."""
    x,y=a['protocol'],b['protocol']
    for key in ('source_domain','reader_cf_mode','completed_epoch','camera','dataset_root',
                'frame_fingerprint','selected_samples','frame_stride',
                'preprocess_cloud_source','preprocess_protocol','collision_source',
                'collision_thresh','collision_voxel_size','sample_policy','sample_seed',
                'batch_size','num_workers','seed','network_geometry','gt_depth_network_input'):
        require(x.get(key)==y.get(key),f'Different model comparison has protocol mismatch in {key}')


def write_csv(path,rows):
    require(bool(rows),f'No rows for {path}')
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fresh_output(path):
    path=Path(path)
    require(not path.exists() or not any(path.iterdir()),
            f'Output nonempty; choose new directory: {path}')
    return path
