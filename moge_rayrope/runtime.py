"""Training/inference I/O, provenance and baseline configuration utilities."""
from __future__ import annotations
import hashlib,json,os,random,sys,tempfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from .config import VERSION,MAIN_COMMIT

MAIN_FILES={
 'models/economicgrasp_bip3d.py':'dfc498775f4b4e93d35352f9e424958519b13b7f',
 'models/economicgrasp_depth.py':'f507c8d009329d9d036b3dbe6d1c9b4ee3d993ad',
 'models/dinov2_dpt.py':'cad4e7d8d17231d3c00a37bb50410bcb838a2f1e',
 'models/kview_query_transformer.py':'85d073af1d0fb88f8dcd7f1c8c435879ca758e08',
 'models/loss_economicgrasp_depth_kview_transformer.py':'72b4b33e64491c6a1f753e51fab834332af2a013',
 'utils/arguments.py':'0d4eea41952048f1f59b53b10258d7005abe952b',
 'dataset/graspnet_dataset.py':'a7b6551df08301cc86c37f51e8b12179e9f1db6f',
 'dataset/cdf_label_adapter.py':'5aaf0417f574522059f2b74db6109508765eb6dd',
}
ROOT=Path(__file__).resolve().parents[1]


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_main_contract():
    result={}
    for name,expected in MAIN_FILES.items():
        data=(ROOT/name).read_bytes()
        actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if actual!=expected:
            raise RuntimeError(f'{name} differs from pinned main {MAIN_COMMIT}: {actual} != {expected}. '
                               'Do not silently combine another experimental branch; review/rebase the adapter first.')
        result[name]=actual
    return result


def code_fingerprint():
    h=hashlib.sha256()
    files=list((ROOT/'moge_rayrope').glob('*.py'))
    for directory in ('models','utils','dataset'):
        files.extend((ROOT/directory).rglob('*.py'))
    files += [ROOT/n for n in ('train_moge_rayrope.py','inference_moge_rayrope.py','eval_moge_rayrope.py')]
    for p in sorted(files): h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent); os.close(fd)
    try: torch.save(value,temp); os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def atomic_npy(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: np.save(f,value,allow_pickle=False)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def ensure_manifest(path,value):
    """Serialize creation by lock; independent inference shards share a manifest."""
    import fcntl
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(str(path)+'.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if path.exists():
            if digest(json.loads(path.read_text()))!=digest(value): raise RuntimeError(f'Stale/incompatible manifest: {path}')
        else: atomic_json(path,value)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s=torch.initial_seed()%(2**32); random.seed(s); np.random.seed(s)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])


def configure_main(cfg,root,label_folder,use_fuse_depth=True):
    # utils.arguments parses argv at import. Our own parser has already run.
    old=sys.argv; sys.argv=[old[0]]
    try:
        from utils.arguments import cfgs
    finally: sys.argv=old
    options=dict(dataset_root=str(root),camera='realsense',num_point=20000,m_point=cfg.seeds,
        num_view=300,num_angle=12,num_depth=4,grasp_max_width=.1,graspness_threshold=.1,
        objectness_loss_weight=1.,graspness_loss_weight=10.,view_loss_weight=100.,
        score_loss_weight=1.,width_loss_weight=10.,depth_prob_loss_weight=10.,
        voxel_size=.005,min_depth=cfg.min_depth,max_depth=cfg.max_depth,bin_num=256,
        multi_modal=True,extend_angle=True,use_cdf=True,use_gt_depth=False,
        use_obs_depth=False,use_depth_comp=False,use_fuse_depth=bool(use_fuse_depth),
        graspness_mode='scene',pose_depth_mode=cfg.pose_mode,kview_mode='A1',kview_k=1,
        use_top4_view_infer=False,kview_use_collision=False,vis_dir=None,
        kview_group_chunk=cfg.group_chunk,cva_label_folder=label_folder,cdf_label_folder=label_folder)
    for k,v in options.items(): setattr(cfgs,k,v)
    return options


def make_dataset(root,split,fraction,labels,cfg,label_folder,use_fuse_depth=True,max_frames=0):
    from dataset.graspnet_dataset import GraspNetMultiDataset
    from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
    if not np.isfinite(fraction) or not 0<fraction<=1: raise ValueError('Invalid frame fraction')
    step=round(1/fraction)
    if abs(fraction*step-1)>1e-7: raise ValueError('Fraction must be an inverse integer (stride sampling)')
    base=GraspNetMultiDataset(root,camera='realsense',split=split,voxel_size=.005,
        num_points=20000,remove_outlier=True,augment=False,load_label=labels,
        use_gt_depth=False,use_fuse_depth=bool(use_fuse_depth),graspness_mode='scene',
        min_depth=cfg.min_depth,max_depth=cfg.max_depth,bin_num=256,depth_strides=1,
        extend_angle=True,load_grasp_payload=False)
    dataset=base
    if labels:
        dataset=CVAExtendedLabelAdapter(base,dataset_root=root,use_cdf=True,
                        label_folder=label_folder,num_angle=12,num_depth=4)
    schedule=[]; indices=[]
    for i,(scene,frame) in enumerate(zip(base.scenename,base.frameid)):
        sid=int(str(scene).split('_')[-1]); aid=int(frame)
        if aid%step==0: indices.append(i); schedule.append([sid,aid])
    if max_frames: indices=indices[:max_frames]; schedule=schedule[:max_frames]
    expected=5200 if split=='train' and fraction==.2 else 780 if split.startswith('test_') and fraction==.1 else None
    if expected is not None and not max_frames and len(indices)!=expected:
        raise RuntimeError(f'Unexpected {split} frame count {len(indices)} != {expected}')
    return base,Subset(dataset,indices),indices,schedule


def move_batch(raw,device,labels=True):
    from .model import CPU_LABELS
    def move(x):
        if torch.is_tensor(x): return x.to(device)
        if isinstance(x,dict): return {k:move(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return type(x)(move(y) for y in x)
        return x
    out={}
    # These are not used by the regression/CDF model. Never transfer large
    # unused depth-bin targets or observed numeric depth into the network.
    drop={'depth','sensor_depth_m','obs_depth_m','input_depth','depth_prob_gt','depth_prob_mask'}
    for key,val in raw.items():
        if key in drop: continue
        if not labels and (key in CPU_LABELS or key.startswith('gt_') or 'label' in key or key=='cdf_thresholds'):
            continue
        out[key]=val if key in CPU_LABELS else move(val)
    return out


def load_model(checkpoint,device='cuda',verify=True):
    from .config import ModelConfig
    from .model import EconomicGraspMoGeRayRoPE
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('version')!=VERSION: raise ValueError('Not a MoGe/RayRoPE checkpoint')
    p=ck['protocol']; cfg=ModelConfig(**p['model'])
    if verify: verify_main_contract()
    configure_main(cfg,p['dataset_root'],p['label_folder'],p['use_fuse_depth'])
    dav=ROOT/'checkpoints'/f'depth_anything_v2_{cfg.encoder}.pth'
    if sha256_file(dav)!=p['dav2_sha256']: raise RuntimeError('DAV2 weights changed')
    model=EconomicGraspMoGeRayRoPE(cfg).to(device); model.load_state_dict(ck['model'],strict=True); model.eval()
    return model,ck
