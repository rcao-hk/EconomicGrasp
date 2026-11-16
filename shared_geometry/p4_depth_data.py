"""P4-1 target index: reuse immutable RGB features, store only aligned depth GT.

Feature maps are NOT copied a second time. Targets are available only in the
training-domain train/validation loader; the prediction API takes inputs only.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import read_index, sha256, digest, signed, load_signed, atomic_json, atomic_npz, frame_key
from .runtime import TargetLoader
from .p1_clean_visible import validated_target
from .p4_common import formal_rows, select_entries

DEPTH_SCHEMA = 'shared-visible-geometry-p4-depth-index-v1'


def read_features(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: np.asarray(z[k], dtype=np.float32) for k in ('feature_map','pred_depth','K')}
    if (a['feature_map'].ndim!=3 or a['pred_depth'].ndim!=2 or a['K'].shape!=(3,3) or
        not all(np.isfinite(v).all() for v in a.values()) or
        (a['pred_depth']<0).any() or not (a['pred_depth']>1e-6).any()):
        raise ValueError('Invalid original RGB feature/depth/camera inputs')
    return a


def build(a):
    idx=read_index(a.index,None,a.smoke);formal_rows(idx,idx['partition'],a.smoke)
    if a.limit<0 or (a.limit and not a.smoke):raise ValueError('Partial export is smoke only')
    loader=TargetLoader(a.targets)
    if loader.config_sha256!=idx['contract']['target_config']:
        raise ValueError('Use the exact original rendered-visible target config')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    contract=dict(original_index_sha256=sha256(a.index),original_index_fingerprint=idx['fingerprint'],
        original_contract=idx['contract'],target_config=loader.config_sha256,
        source=idx['contract']['source'],plan=idx['contract']['plan'],
        data_code_sha256=sha256(__file__),partition=idx['partition'])
    ch=digest(contract);entries=[]
    subset=select_entries(idx['entries'],a.limit)
    for i,e in enumerate(subset):
        f=read_features(e['path'])
        clean,provenance=validated_target(idx,e,f,loader)
        if clean.shape!=f['pred_depth'].shape or not (clean>1e-6).any():
            raise ValueError('No aligned visible teacher or invalid shape')
        dest=out/(frame_key(e['row'])+'_target.npz');meta=dest.with_suffix('.json')
        identity=dict(row=e['row'],source_frame_sha256=e['sha256'],contract_hash=ch,
                      teacher=provenance)
        if dest.exists() or meta.exists():
            if not a.resume or not(dest.is_file() and meta.is_file()):raise FileExistsError(dest)
            old=load_signed(meta)
            if any(old[k]!=v for k,v in identity.items()) or sha256(dest)!=old['sha256']:
                raise ValueError('P4 depth target resume provenance mismatch')
        else:
            atomic_npz(dest,dict(visible_depth=clean.astype(np.float32)))
            atomic_json(meta,signed(dict(**identity,sha256=sha256(dest))))
        entries.append(dict(row=e['row'],source_path=e['path'],source_sha256=e['sha256'],
            target_path=str(dest.resolve()),target_sha256=sha256(dest),
            feature_shape=list(f['feature_map'].shape),depth_shape=list(clean.shape)))
        if (i+1)%128==0 or i+1==len(subset):print(f'[P4 depth targets] {i+1}/{len(subset)}',flush=True)
    index=signed(dict(schema=DEPTH_SCHEMA,partition=idx['partition'],
        complete=bool(idx['complete'] and not a.limit),contract=contract,entries=entries))
    path=out/'index.json'
    if path.exists():
        if not a.resume or load_signed(path)!=index:raise FileExistsError('Different depth target index')
    else:atomic_json(path,index)
    return index


def read_depth_index(path,partition=None,smoke=False,verify_files=True):
    idx=load_signed(path)
    if idx['schema']!=DEPTH_SCHEMA or (partition and idx['partition']!=partition):
        raise ValueError('Wrong P4 depth index')
    if not idx['complete'] and not smoke:raise ValueError('Partial depth cache forbidden')
    formal_rows(idx,idx['partition'],smoke)
    if len({tuple(e['feature_shape']) for e in idx['entries']})!=1 or len({tuple(e['depth_shape']) for e in idx['entries']})!=1:
        raise ValueError('Use explicit shape buckets; do not pad physical image geometry')
    if verify_files:
        for e in idx['entries']:
            if sha256(e['source_path'])!=e['source_sha256'] or sha256(e['target_path'])!=e['target_sha256']:
                raise ValueError('Changed original features or depth targets')
            side=load_signed(Path(e['target_path']).with_suffix('.json'))
            if (side['row']!=e['row'] or side['sha256']!=e['target_sha256'] or
                side['source_frame_sha256']!=e['source_sha256'] or
                side['contract_hash']!=digest(idx['contract'])):
                raise ValueError('Depth target sidecar/index mismatch')
    return idx


class DepthDataset(Dataset):
    def __init__(self,index):self.index=index
    def __len__(self):return len(self.index['entries'])
    def __getitem__(self,i):
        e=self.index['entries'][i];inputs=read_features(e['source_path'])
        with np.load(e['target_path'],allow_pickle=False) as z:
            target=np.asarray(z['visible_depth'],np.float32)
        if target.shape!=inputs['pred_depth'].shape or not np.isfinite(target).all() or (target<0).any():
            raise ValueError('Malformed depth target')
        return dict(inputs=inputs,target=target,row=e['row'])


def collate_depth(items):
    if not items:raise ValueError('Empty image batch')
    def stack(xs):
        if len({x.shape for x in xs})!=1:raise ValueError('Ragged depth/image batch')
        return torch.from_numpy(np.stack(xs)).float()
    # Explicitly separate inputs and labels; quality_count/GT depth never enter forward.
    return dict(inputs={k:stack([x['inputs'][k] for x in items]) for k in ('feature_map','pred_depth','K')},
                target=stack([x['target'] for x in items]),rows=[x['row'] for x in items])
