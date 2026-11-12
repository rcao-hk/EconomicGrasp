"""Full-context evidence export and immutable sparse-label indexing."""
from __future__ import annotations
import importlib
import json
import shutil
from pathlib import Path
import numpy as np
from .contracts import (SCHEMA,load_signed,validate_plan,frame_key,frame_seed,physical_ids,
                        validate_frame,signed,atomic_json,atomic_npz,sha256,digest,parameter_source_hash)
from .data import ExactLabelReuse


def get_runtime(a):
    module,name=a.provider.split(':',1)
    loaded=importlib.import_module(module)
    cls=getattr(loaded,name)
    runtime=cls(a.config,a.dataset_root,a.gntrans_root,a.device)
    if module!='interactive_grasp.native':
        if not getattr(loaded,'__file__',None):
            raise ValueError('Custom provider must have an auditable source file')
        runtime.provenance['adapter_source_sha256']=sha256(loaded.__file__)
    for method in ('snapshot','sensor_keep','label'):
        if not callable(getattr(runtime,method,None)):raise TypeError(f'Provider lacks {method}')
    for field in ('base_sha256','selector_sha256','config_sha256'):
        if not runtime.provenance.get(field):raise ValueError('Provider missing frozen source identity')
    return runtime


def export(a):
    plan=load_signed(a.plan);validate_plan(plan)
    rows=plan[a.partition]
    if a.domain: rows=[r for r in rows if r['domain']==a.domain]
    if not 0<=a.shard<a.shards:raise ValueError('Invalid export shard')
    rows=[r for i,r in enumerate(rows) if i%a.shards==a.shard]
    if a.limit:rows=rows[:a.limit]
    if not rows or a.labels<0 or a.labels>1024:raise ValueError('Invalid export selection')
    if a.partition!='test' and a.labels==0:raise ValueError('Train/validation require labels')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    runtime=get_runtime(a)
    reuse=ExactLabelReuse(a.reuse_index,runtime.provenance['base_sha256'])
    contract=dict(schema=SCHEMA,plan_sha256=plan['fingerprint'],partition=a.partition,
                  source=runtime.provenance,provider=a.provider,code_sha256=parameter_source_hash(),
                  seed=a.seed,label_queries=a.labels,memory_dtype=a.memory_dtype,
                  expected_queries=1024,reuse_indices=reuse.sources,
                  input_stage='full_pre_sensor_filter',labels='exact_executed_actions_sparse')
    csha=digest(contract)
    total=0
    for step,row in enumerate(rows):
        key=frame_key(row);path=root/(key+'.npz');meta=root/(key+'.json')
        if path.exists() or meta.exists():
            if not a.resume or not(path.exists() and meta.exists()):raise FileExistsError(path)
            old=load_signed(meta)
            if old['contract_sha256']!=csha or old['row']!=row or sha256(path)!=old['npz_sha256']:
                raise ValueError('Existing cache contract/hash mismatch')
            total+=path.stat().st_size
            continue
        arrays,parity=runtime.snapshot(row,a.seed)
        n=len(arrays['grasps'])
        if n!=1024:raise RuntimeError('Export must retain all 1024 queries')
        # Gate missing/failed native checks also for explicitly supplied adapters.
        for field in ('native_decode_max_abs','native_reader_replay_max_abs'):
            if field not in parity or not np.isfinite(parity[field]) or parity[field]>1e-6:
                raise RuntimeError(f'Provider did not pass {field}')
        ids=physical_ids(arrays['grasps']);arrays['physical_id']=ids
        label_mask=np.zeros(n,dtype=bool)
        selected=np.sort(np.random.default_rng(frame_seed(row,a.seed)).choice(n,a.labels,replace=False))
        label_mask[selected]=True
        counts=np.full(n,-1,dtype=np.int64)
        matched=reuse.lookup(row,ids[selected]);counts[selected]=matched
        missing=selected[matched<0]
        if len(missing):counts[missing]=runtime.label(row,arrays['grasps'][missing])
        arrays['label_mask']=label_mask;arrays['quality_count']=counts
        arrays['sensor_keep']=runtime.sensor_keep(row,arrays['grasps']).astype(bool)
        mem=arrays['memory'];stored=mem.astype(a.memory_dtype)
        quant_error=float(np.max(np.abs(mem-stored.astype(mem.dtype))))
        arrays['memory']=stored
        validate_frame(arrays,1024)
        required_bytes=sum(x.nbytes for x in arrays.values())
        free=shutil.disk_usage(root).free
        if free < required_bytes + a.reserve_gib*(1<<30):
            raise OSError('Not enough disk reserve; do not shrink context/top-k silently')
        atomic_npz(path,arrays)
        metadata=signed(dict(schema=SCHEMA,row=row,contract=contract,contract_sha256=csha,
                npz_sha256=sha256(path),native_parity=parity,
                memory_quantization_max_abs=quant_error,
                exact_labels_reused=int(np.sum(matched>=0)),new_labels=int(len(missing))))
        atomic_json(meta,metadata);total+=path.stat().st_size
        print(f'[export] {step+1}/{len(rows)} {key} full={n} labelled={a.labels} '
              f'reused={np.sum(matched>=0)} stored={path.stat().st_size/2**20:.1f}MiB',flush=True)
        if step==0:
            estimated=path.stat().st_size*len(plan[a.partition])
            print(f'[storage] extrapolated compressed partition bytes={estimated}; '
                  'measure representative frames before full export',flush=True)
    atomic_json(root/f'export_receipt_shard{a.shard:03d}.json',signed(dict(
        contract=contract,contract_sha256=csha,shard=a.shard,shards=a.shards,
        frames=len(rows),limit=a.limit,bytes=total)))


def index(a):
    plan=load_signed(a.plan);validate_plan(plan)
    rows=plan[a.partition];root=Path(a.cache);entries=[];contract=None;dims=None;missing=[]
    for row in rows:
        path=root/(frame_key(row)+'.npz');meta=path.with_suffix('.json')
        if not meta.exists():missing.append(frame_key(row));continue
        rec=load_signed(meta)
        if rec['row']!=row or rec['contract']['plan_sha256']!=plan['fingerprint'] or rec['contract']['partition']!=a.partition:
            raise ValueError('Frame plan/partition mismatch')
        if digest(rec['contract'])!=rec['contract_sha256']:raise ValueError('Bad frame contract')
        if contract is None:contract=rec['contract']
        if contract!=rec['contract']:raise ValueError('Mixed cache source contracts')
        if sha256(path)!=rec['npz_sha256']:raise ValueError('Corrupt frame data')
        with np.load(path,allow_pickle=False) as f:
            data={k:f[k] for k in f.files}
        validate_frame(data,contract['expected_queries'])
        current=(data['features'].shape[-1],data['memory'].shape[-1])
        if dims is None:dims=current
        if current!=dims:raise ValueError('Mixed feature/memory dimensions')
        entries.append(dict(row=row,path=str(path.resolve()),sha256=rec['npz_sha256']))
    if not entries or (missing and not a.allow_partial):raise ValueError(f'Missing {len(missing)} planned frames')
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    atomic_json(dest,signed(dict(schema=SCHEMA,partition=a.partition,complete=not missing,
        expected_frames=len(rows),missing=missing,contract=contract,entries=entries,
        feature_dim=dims[0],memory_dim=dims[1])))
    print(f'[index] {len(entries)}/{len(rows)} frames; complete={not missing}')
