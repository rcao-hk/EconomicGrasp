"""Immutable frame/action contracts. All writable files are new experiment assets."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

SCHEMA = 'interactive-frozen-grasp-stage1-v1'
BASE_MAIN = '7a956d37aaa427a32e598f21fdc6b910ed3c2704'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def atomic_npz(path, arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with open(tmp,'wb') as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def signed(obj):
    if 'fingerprint' in obj: raise ValueError('Already signed')
    return {**obj,'fingerprint':digest(obj)}


def load_signed(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    expected=obj.pop('fingerprint',None)
    if expected is None or digest(obj)!=expected: raise ValueError(f'Fingerprint mismatch: {path}')
    obj['fingerprint']=expected
    return obj


def frame_key(row):
    return f"{row['domain']}_scene{int(row['scene']):04d}_frame{int(row['anno']):04d}"


def frame_seed(row, seed):
    return int(digest([frame_key(row),int(seed)])[:8],16)


def physical_ids(g):
    g=np.asarray(g,dtype='<f4')
    if g.ndim!=2 or g.shape[1]!=17 or not np.isfinite(g).all():
        raise ValueError('Expected finite [N,17] grasps')
    return np.asarray([hashlib.sha256(x[1:16].tobytes()).hexdigest() for x in g],dtype='<U64')


def validate_frame(a, expected_queries=1024):
    required=('grasps','physical_id','features','base_logit','memory','memory_valid',
              'query_id','angle_id','depth_id','label_mask','quality_count','sensor_keep')
    if any(k not in a for k in required): raise ValueError('Missing frame arrays')
    g=a['grasps'];n=len(g)
    if n!=expected_queries or n<1 or g.shape!=(n,17): raise ValueError('Full candidate context required')
    if not np.array_equal(a['query_id'],np.arange(n)):
        raise ValueError('One original-order action per query required; no pre-top-k')
    if not np.array_equal(physical_ids(g),a['physical_id']): raise ValueError('Physical action hash mismatch')
    if a['features'].ndim!=2 or len(a['features'])!=n: raise ValueError('Bad action features')
    if a['memory'].ndim!=3 or a['memory'].shape[:2]!=(n,36): raise ValueError('Expected 36 GVAR evidence tokens/action')
    if a['memory_valid'].shape!=(n,36) or a['memory_valid'].dtype!=np.bool_:
        raise ValueError('True evidence-valid mask required')
    for name in ('base_logit','angle_id','depth_id','label_mask','quality_count','sensor_keep'):
        if a[name].shape!=(n,): raise ValueError(f'Bad vector {name}')
    if a['label_mask'].dtype!=np.bool_ or a['sensor_keep'].dtype!=np.bool_:
        raise ValueError('Masks must be bool')
    if a['quality_count'].dtype.kind not in 'iu': raise ValueError('Quality counts must be integers')
    lab=a['label_mask'];y=a['quality_count']
    if np.any((y[lab]<0)|(y[lab]>6)) or np.any(y[~lab]!=-1):
        raise ValueError('Missing labels must be -1, never implicit negatives')
    for name in ('grasps','features','base_logit','memory'):
        if not np.isfinite(a[name]).all(): raise ValueError(f'Nonfinite {name}')
    if np.any(a['memory'][~a['memory_valid']]!=0): raise ValueError('Invalid memory must be zero')
    if not np.allclose(1/(1+np.exp(-a['base_logit'].astype(float))),g[:,0],rtol=0,atol=1e-6):
        raise ValueError('Frozen source score/logit mismatch')
    if np.any((g[:,1]<0)|(g[:,1]>.100001)) or np.any((a['angle_id']<0)|(a['angle_id']>=12)):
        raise ValueError('Width or angle outside primary contract')
    if np.any((a['depth_id']<0)|(a['depth_id']>=4)) or not np.allclose(g[:,3],(a['depth_id']+1)*.01,atol=1e-7):
        raise ValueError('Actual insertion does not match depth id')
    rot=g[:,4:13].reshape(-1,3,3)
    if not np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-3) or np.any(np.linalg.det(rot)<.999):
        raise ValueError('Invalid grasp SO(3) rotation')


def make_plan():
    """Pilot across 80 train scenes, 20 new-stage validation scenes.

    Historical frozen generator/MSE may have seen all train scenes: this is a
    stage-1 scene holdout, NOT whole-system unseen-object evaluation.
    All pilot training/validation frames lie in the 50% even-frame pool.
    """
    def row(d,s,a,part):
        split='train' if s<100 else ('test_seen' if s<130 else 'test_similar' if s<160 else 'test_novel')
        return dict(domain=d,scene=s,anno=a,split=split,partition=part)
    train=[row(d,s,a,'train') for d in ('original','gntrans') for s in range(80) for a in range(0,256,16)]
    val=[row(d,s,a,'validation') for d in ('original','gntrans') for s in range(80,100) for a in (8,72,136,200)]
    test=[row('original',s,a,'test') for s in range(100,190) for a in range(256)]
    return signed(dict(schema=SCHEMA,base_main=BASE_MAIN,train=train,validation=val,test=test,
                       note='Frozen base/MSE may have seen validation scenes; only added-stage holdout.'))


def validate_plan(plan):
    if plan.get('schema')!=SCHEMA: raise ValueError('Wrong plan schema')
    sets={}
    for part in ('train','validation','test'):
        rows=plan[part]; keys=[frame_key(r) for r in rows]
        if not rows or len(set(keys))!=len(keys): raise ValueError('Empty/duplicate frames')
        for r in rows:
            if r['partition']!=part or r['domain'] not in ('original','gntrans') or not 0<=r['anno']<256:
                raise ValueError('Invalid frame identity')
            s=r['scene']; split='train' if s<100 else 'test_seen' if s<130 else 'test_similar' if s<160 else 'test_novel'
            if not 0<=s<190 or r['split']!=split or ((s<100)!=(part!='test')):
                raise ValueError('Official test scenes cannot train/validate')
        sets[part]={r['scene'] for r in rows}
    if sets['train']&sets['validation']: raise ValueError('Train/validation scenes overlap across RGB domains')


def parameter_source_hash():
    root=Path(__file__).parent
    return digest({p.name:sha256(p) for p in sorted(root.glob('*.py'))})
