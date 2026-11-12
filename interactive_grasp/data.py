"""Indexed full-context frames with sparse exact-action supervision."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .contracts import load_signed, sha256, validate_frame, SCHEMA


def load_index(path, partition=None, allow_smoke=False):
    index=load_signed(path)
    if index.get('schema')!=SCHEMA or (partition and index['partition']!=partition):
        raise ValueError('Cache schema/partition mismatch')
    if not index['complete'] and not allow_smoke: raise ValueError('Incomplete cache forbidden for formal run')
    if not index['entries']: raise ValueError('Empty index')
    seen=set()
    for e in index['entries']:
        key=(e['row']['domain'],e['row']['scene'],e['row']['anno'])
        if key in seen or e['row']['partition']!=index['partition']: raise ValueError('Duplicate/wrong frame partition')
        seen.add(key)
        if sha256(e['path'])!=e['sha256']: raise ValueError(f'Corrupt cache {e["path"]}')
    return index


def load_frame(entry, expected_queries):
    with np.load(entry['path'],allow_pickle=False) as f:
        a={k:f[k] for k in f.files}
    validate_frame(a,expected_queries)
    return a


def tensor_frame(a, device):
    # Same full context for every variant; labels remain separate from inputs.
    inputs={k:torch.from_numpy(np.asarray(a[k])).unsqueeze(0).to(device=device,dtype=torch.float32)
            for k in ('features','grasps','base_logit','memory')}
    inputs['memory_valid']=torch.from_numpy(a['memory_valid']).unsqueeze(0).to(device)
    inputs['action_valid']=torch.ones_like(inputs['base_logit'],dtype=torch.bool)
    label_mask=torch.from_numpy(a['label_mask']).unsqueeze(0).to(device)
    count=torch.from_numpy(a['quality_count']).unsqueeze(0).to(device=device,dtype=torch.long)
    return inputs,count,label_mask


class ExactLabelReuse:
    """Adapter for legacy RARE per-frame NPZ labels, joined by physical action ID.

    Never joins on query ID alone. Must be same base SHA, frame/domain, official
    evaluator mode. Nonmatches remain unlabeled; caller explicitly re-evaluates.
    """
    def __init__(self, index_paths, base_sha):
        import json
        from .contracts import digest
        self.entries={};self.sources={}
        for path in index_paths:
            obj=json.loads(Path(path).read_text())
            fingerprint=obj.pop('index_sha256',None)
            if not fingerprint or digest(obj)!=fingerprint: raise ValueError('Legacy index hash mismatch')
            contract=obj['contract'];config=contract['config']
            if config['checkpoint_sha256']!=base_sha or contract.get('fc_mode')!='official':
                raise ValueError('Reuse labels require exact base and official evaluator provenance')
            self.sources[str(Path(path).resolve())]=sha256(path)
            for e in obj['entries']:
                row=e['row'];key=(row['domain'],row['scene'],row['anno'])
                if key in self.entries: raise ValueError('Duplicate frame across reuse indices')
                self.entries[key]=e

    def lookup(self,row,ids):
        from .contracts import physical_ids
        result=np.full(len(ids),-1,dtype=np.int64)
        key=(row['domain'],row['scene'],row['anno'])
        e=self.entries.get(key)
        if e is None:return result
        if sha256(e['file'])!=e['sha256']:raise ValueError('Reuse file hash mismatch')
        with np.load(e['file'],allow_pickle=False) as f:
            old_ids=f['physical_id'].astype(str);bits=f['bits'];cnt=f['count']
            if not np.array_equal(physical_ids(f['grasps']),old_ids):raise ValueError('Reuse physical IDs invalid')
            if bits.shape!=(len(cnt),6) or not np.isin(bits,[0,1]).all() or not np.array_equal(bits.sum(-1),cnt):
                raise ValueError('Reuse exact bits/count inconsistent')
            if np.any(np.diff(bits.astype(int),axis=-1)<0):
                raise ValueError('Reuse friction-success bits must be monotone')
            lookup={}
            for pid,y in zip(old_ids,cnt):
                if pid in lookup and lookup[pid]!=int(y):raise ValueError('Conflicting quality for identical action')
                lookup[pid]=int(y)
        for i,pid in enumerate(ids):result[i]=lookup.get(str(pid),-1)
        return result
