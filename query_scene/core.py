"""Shared-scene sampling and immutable full-candidate evidence contracts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from interactive_grasp.contracts import (sha256, digest, signed, load_signed, atomic_json,
    atomic_npz, physical_ids, frame_key, frame_seed, make_plan, validate_plan)

SCHEMA = 'query-scene-evidence-v1'
VARIANTS = ('local', 'candidate', 'scene')


def code_hash():
    roots = (Path(__file__).parent, Path(__file__).parent.parent / 'interactive_grasp')
    return digest({f'{r.name}/{p.name}': sha256(p) for r in roots for p in sorted(r.glob('*.py'))})


@torch.no_grad()
def sample_scene(pregeom, depth, K, image_hw, grid_size=32):
    """Uniform image-grid memory, independent of proposals, scores and GT masks.

    Tokens use the frozen reader's *unprojected pregeom map*, not its 36 probes.
    Pixel centers and camera K refer to the model input crop. Invalid predicted
    depth removes only metric geometry, never an otherwise valid RGB token.
    """
    if pregeom.ndim != 4 or pregeom.shape[0] != 1 or depth.ndim != 4 or depth.shape[:2] != (1,1):
        raise ValueError('Expected single-image feature [1,C,H,W] and depth [1,1,H,W]')
    h, w = map(int, image_hw)
    if depth.shape[-2:] != (h,w) or K.shape != (1,3,3) or grid_size < 2:
        raise ValueError('Camera/depth/grid shape mismatch')
    if not torch.isfinite(pregeom).all() or not torch.isfinite(K).all() or min(h,w)<2:
        raise ValueError('Nonfinite feature/camera or invalid image size')
    if K[0,0,0] <= 0 or K[0,1,1] <= 0 or not torch.allclose(K[0,2], K.new_tensor([0,0,1])):
        raise ValueError('Invalid pinhole intrinsics')
    # Align-corners=False cell centers, shared by image and feature sampling.
    x = (torch.arange(grid_size, device=pregeom.device, dtype=torch.float32)+.5)/grid_size*2-1
    yy, xx = torch.meshgrid(x,x,indexing='ij'); grid=torch.stack((xx,yy),-1)[None]
    rgb=F.grid_sample(pregeom.float(),grid,align_corners=False,mode='bilinear',padding_mode='border')[0]
    rgb=rgb.permute(1,2,0).reshape(-1,pregeom.shape[1])
    valid_d=torch.isfinite(depth)&(depth>0)
    clean=torch.where(valid_d,depth,0).float()
    z=F.grid_sample(clean,grid,align_corners=False,mode='bilinear',padding_mode='border').flatten()
    weight=F.grid_sample(valid_d.float(),grid,align_corners=False,mode='bilinear',padding_mode='border').flatten()
    known=weight>.999999; z=torch.where(known,z,0)
    u=(xx.flatten()+1)*w/2-.5; v=(yy.flatten()+1)*h/2-.5
    ray=torch.stack((u,v,torch.ones_like(u)),-1) @ torch.linalg.inv(K[0].float()).T
    ray=ray/ray[:,2:3]
    xyz=ray*z[:,None]
    geo=torch.cat((xx.flatten()[:,None],yy.flatten()[:,None],ray[:,:2],xyz,known.float()[:,None]),-1)
    return {'scene_features':rgb.cpu().numpy().copy(), 'scene_geometry':geo.cpu().numpy().copy(),
            'scene_valid':np.ones(len(rgb),dtype=bool)}


def validate(a, queries=1024, grid_size=32):
    n=queries; m=grid_size**2
    for key,shape in {'grasps':(n,17),'base_logit':(n,), 'query_id':(n,),
                      'sensor_keep':(n,), 'label_mask':(n,), 'quality_count':(n,),
                      'scene_geometry':(m,8),'scene_valid':(m,)}.items():
        if key not in a or a[key].shape!=shape: raise ValueError(f'Bad/missing {key} shape')
    if a['features'].ndim!=2 or a['features'].shape[0]!=n or a['scene_features'].ndim!=2 or a['scene_features'].shape[0]!=m:
        raise ValueError('Feature/context dimensions')
    if not np.array_equal(a['query_id'],np.arange(n)) or not np.array_equal(physical_ids(a['grasps']),a['physical_id']):
        raise ValueError('Full native query order / executed action identity mismatch')
    for key in ('sensor_keep','label_mask','scene_valid'):
        if a[key].dtype!=np.bool_: raise ValueError('Masks must be bool')
    for key in ('grasps','base_logit','features','scene_features','scene_geometry'):
        if not np.isfinite(a[key]).all():raise ValueError(f'Nonfinite {key}')
    lab=a['label_mask'];y=a['quality_count']
    if y.dtype.kind not in 'iu' or np.any((y[lab]<0)|(y[lab]>6)) or np.any(y[~lab]!=-1):
        raise ValueError('Unknown labels must be -1, not negatives')
    if not np.all((a['grasps'][:,0]>0)&(a['grasps'][:,0]<1)):
        raise ValueError('Frozen MSE probabilities must be in (0,1)')
    expected=np.log(a['grasps'][:,0].astype(float)/(1-a['grasps'][:,0].astype(float)))
    if not np.allclose(expected,a['base_logit'],atol=2e-5,rtol=1e-5):raise ValueError('Frozen score/logit mismatch')
    rot=a['grasps'][:,4:13].reshape(-1,3,3)
    if not np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-3) or np.any(np.linalg.det(rot)<.999):
        raise ValueError('Invalid rotations')
    if np.any((a['grasps'][:,1]<0)|(a['grasps'][:,1]>.100001)):raise ValueError('Executed width outside source contract')
    if not np.isin(a['scene_geometry'][:,-1],[0,1]).all():raise ValueError('Bad predicted-depth validity flag')
    if np.any(a['scene_geometry'][a['scene_geometry'][:,-1]==0,4:7]!=0):raise ValueError('Unknown depth leaked into metric coordinates')


def tensors(a,device):
    x={k:torch.from_numpy(a[k]).unsqueeze(0).to(device=device,dtype=torch.float32)
       for k in ('features','grasps','base_logit','scene_features','scene_geometry')}
    x['scene_valid']=torch.from_numpy(a['scene_valid']).unsqueeze(0).to(device)
    x['action_valid']=torch.ones_like(x['base_logit'],dtype=torch.bool)
    return x,torch.from_numpy(a['quality_count']).unsqueeze(0).to(device),torch.from_numpy(a['label_mask']).unsqueeze(0).to(device)


def read_frame(e,c):
    with np.load(e['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
    validate(a,c['queries'],c['grid_size']);return a


def read_index(path,part=None,smoke=False):
    idx=load_signed(path)
    if idx['schema']!=SCHEMA or (part and idx['partition']!=part):raise ValueError('Wrong index schema/partition')
    if not idx['complete'] and not smoke:raise ValueError('Partial index forbidden for formal run')
    seen=set()
    for e in idx['entries']:
        row=e['row'];key=frame_key(row)
        if key in seen or row['partition']!=idx['partition']:raise ValueError('Duplicate/partition mismatch')
        if row['scene']>=100 and idx['partition']!='test':raise ValueError('Test scenes in fitting data')
        seen.add(key)
        if sha256(e['path'])!=e['sha256']:raise ValueError('Cache SHA mismatch')
    if not seen:raise ValueError('Empty index')
    return idx
