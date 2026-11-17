"""Explicit raw-image manifests; references/segmentation never enter model.forward."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .geometry import check_K, crop_rgb, transform_uv, inside, plane_from_table, pixel_grid

SCHEMA = 'camera-crop-plane-v1'
PARTITIONS = ('train', 'dev', 'holdout', 'exploratory')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def array(path):
    return np.load(path, allow_pickle=False) if Path(path).suffix == '.npy' else np.asarray(Image.open(path))


def load_input(e):
    rgb = np.asarray(Image.open(e['rgb_path']).convert('RGB'), dtype=np.float32)/255.
    K = torch.tensor(e['K'], dtype=torch.float32); check_K(K)
    if list(rgb.shape[:2]) != e['raw_hw']: raise ValueError('RGB shape changed: '+e['id'])
    return torch.from_numpy(rgb.copy()).permute(2,0,1), K


def load_reference(e):
    z = np.asarray(array(e['depth_path']), dtype=np.float32)*float(e['depth_scale_m'])
    if list(z.shape) != e['raw_hw'] or not np.isfinite(z).all() or (z < 0).any():
        raise ValueError('Expected finite nonnegative raw-camera z reference: '+e['id'])
    mask = z > 0
    if e.get('mask_path'):
        fg = array(e['mask_path'])
        if fg.shape != z.shape: raise ValueError('Mask must be raw HxW, not RGB')
        mask &= fg > 0
    if mask.sum() == 0: raise ValueError('No supervised foreground rays: '+e['id'])
    return torch.from_numpy(z.copy()), torch.from_numpy(mask.copy())


def validate_entries(entries):
    ids, groups, objects = set(), {}, {}
    for e in entries:
        if e['id'] in ids: raise ValueError('Duplicate frame id: '+e['id'])
        ids.add(e['id']); p = e['partition']
        if p not in PARTITIONS: raise ValueError('Unknown partition')
        if e.get('official_test', False) and p in ('train','dev'):
            raise ValueError('Official test cannot train or select models')
        if p == 'holdout' and e.get('previously_used', True):
            raise ValueError('Previously observed data must be exploratory, not independent holdout')
        groups.setdefault(e['group'], set()).add(p)
        for obj in e.get('object_ids', []): objects.setdefault(obj, set()).add(p)
        if not e.get('reference_source') or not e.get('crop_source'):
            raise ValueError('Explicit reference/crop provenance required')
        if not 0 < float(e['depth_scale_m']) <= 1: raise ValueError('Invalid metre conversion')
        if set(e['crop_boxes']) != {'full','roi','workspace'}:
            raise ValueError('Provide full, roi, workspace boxes explicitly')
        h, w = e['raw_hw']
        if not np.allclose(e['crop_boxes']['full'], [0,0,w,h]):
            raise ValueError('full must retain original field of view')
        for box in e['crop_boxes'].values():
            if len(box) != 4 or not np.isfinite(box).all() or not box[2] > box[0] or not box[3] > box[1]:
                raise ValueError('Invalid crop box')
        check_K(torch.tensor(e['K'], dtype=torch.float64))
        for key in ('pose_vec','gravity_vec'):
            v = np.asarray(e[key], dtype=float)
            if v.shape != (3,) or not np.isfinite(v).all() or abs(np.linalg.norm(v)-1) > 1e-3:
                raise ValueError('Unit orientation metadata required: '+key)
        if 'T_table_from_cam' in e:
            T = torch.tensor(e['T_table_from_cam'], dtype=torch.float64)
            pl = plane_from_table(T)
            if e.get('plane_source') not in ('calibrated','renderer') or not e.get('plane_provenance'):
                raise ValueError('Plane must come from declared calibration or renderer, not target depth fitting')
            if not np.allclose(e['gravity_vec'], pl[:3].numpy(), atol=1e-3):
                raise ValueError('Gravity/world-up and table normal disagree')
            if not np.allclose(e['pose_vec'], -T[:3,2].numpy(), atol=1e-3):
                raise ValueError('Pose convention: -optical axis in table frame')
        tracks = e.get('tracks', [])
        if len({t['id'] for t in tracks}) != len(tracks): raise ValueError('Duplicate track ID in frame')
        if tracks and 'T_table_from_cam' not in e: raise ValueError('Metric track audit requires table/world frame')
    if any(len(ps) > 1 for ps in groups.values()):
        raise ValueError('Same physical group crosses partitions (including paired appearance domains)')
    if any('holdout' in ps and len(ps) > 1 for ps in objects.values()):
        raise ValueError('Declared holdout objects overlap development/training/exploratory objects')


def prepare(spec_path, output):
    if Path(output).exists(): raise FileExistsError(output)
    spec = read_json(spec_path); base = Path(spec_path).resolve().parent
    entries = spec['entries']
    if not entries: raise ValueError('Empty data specification')
    for e in entries:
        for key in ('rgb_path','depth_path','mask_path'):
            if e.get(key):
                p = Path(e[key]); e[key] = str((p if p.is_absolute() else base/p).resolve())
        rgb = np.asarray(Image.open(e['rgb_path']))
        e['raw_hw'] = list(rgb.shape[:2])
        e.setdefault('crop_boxes', {})
        e['crop_boxes'].setdefault('full', [0,0,rgb.shape[1],rgb.shape[0]])
        if 'T_table_from_cam' in e:
            T = np.asarray(e['T_table_from_cam'], dtype=float)
            e.setdefault('pose_vec', (-T[:3,2]).tolist())
            e.setdefault('gravity_vec', T[2,:3].tolist())
        e['source_sha256'] = {k:sha256(e[k]) for k in ('rgb_path','depth_path','mask_path') if e.get(k)}
        load_input(e); z, valid = load_reference(e)
        e['reference_audit'] = {'valid_pixels': int(valid.sum()),
            'median_z_m': float(z[valid].median()), 'max_z_m': float(z[valid].max())}
    validate_entries(entries)
    result = {'schema':SCHEMA, 'entries':entries, 'provenance':spec.get('provenance',''),
              'split_claim':'declared groups/objects checked; dataset history needs human audit'}
    result['fingerprint'] = digest(result)
    atomic_json(output, result)
    return result


def load_manifest(path, verify=True):
    m = read_json(path); fp = m.pop('fingerprint')
    if m['schema'] != SCHEMA or digest(m) != fp: raise ValueError('Manifest signature mismatch')
    m['fingerprint'] = fp; validate_entries(m['entries'])
    if verify:
        for e in m['entries']:
            for k, h in e['source_sha256'].items():
                if sha256(e[k]) != h: raise ValueError('Source file changed: '+e[k])
    return m


def inputs_for_views(e, names, hw):
    rgb, K = load_input(e)
    out = []
    for name in names:
        image, kc, A, valid = crop_rgb(rgb, K, e['crop_boxes'][name], hw)
        row = dict(rgb=image, K=kc, A=A, pixel_valid=valid,
                   pose=torch.tensor(e['pose_vec'], dtype=torch.float32),
                   gravity=torch.tensor(e['gravity_vec'], dtype=torch.float32))
        if 'T_table_from_cam' in e:
            row['plane'] = plane_from_table(torch.tensor(e['T_table_from_cam'], dtype=torch.float32))
        out.append(row)
    return out


def common_queries(e, views, hw, n, rng):
    z, fg = load_reference(e)
    uv = pixel_grid(z.shape).reshape(-1,2)
    valid = fg.reshape(-1).clone()
    for v in views:
        valid &= inside(transform_uv(uv, v['A']), hw, margin=1.)
    candidates = torch.where(valid)[0].numpy()
    if len(candidates) < n:
        raise ValueError(f"{e['id']}: only {len(candidates)} common foreground rays, need {n}; no silent padding/drop")
    idx = rng.choice(candidates, n, replace=False)
    return uv[idx], z.reshape(-1)[idx], float(len(candidates)/int(fg.sum()))


def stack_inputs(rows, device):
    # Deliberate whitelist: reference depth, mask, distance tags and IDs cannot leak.
    keys = ['rgb','K','pose','gravity']
    if all('plane' in r for r in rows): keys += ['plane']
    return {k:torch.stack([r[k] for r in rows]).to(device) for k in keys}


def import_gc6d(spec_path, output):
    """Import real GC6D BOP paths. Never infer that its JSON 'w2c' is actually w2c.

    Scene selection/partitions and transform semantics are explicit. A scene's
    world frame is NOT assumed to be a calibrated table frame.
    """
    cfg = read_json(spec_path); root = Path(cfg['dataset_root']).resolve()
    if not cfg.get('scenes'): raise ValueError('Register explicit scene partitions before importing')
    if 'REPLACE' in cfg.get('orientation_provenance','REPLACE'):
        raise ValueError('Verify orientation provenance before importing')
    dest = Path(output).resolve()
    if dest.exists(): raise FileExistsError(dest)
    offsets = {'realsense-d415':1,'realsense-d435':2,'azure-kinect':3,'zivid':4}
    camera = cfg['camera']; offset = offsets[camera]
    semantics = cfg['camera_transform_semantics']
    if semantics not in ('c2w','w2c'): raise ValueError('Explicit c2w/w2c interpretation required')
    official = set(map(int,read_json(root/'split_info/grasp_test_scene_ids.json')))
    records=[]; metadata_hashes={}
    for scene in cfg['scenes']:
        sid=int(scene['scene']); folder=root/'scenes'/f'{sid:06d}'
        cams=read_json(folder/'scene_camera.json'); gts=read_json(folder/'scene_gt.json')
        metadata_hashes[str(folder/'scene_camera.json')]=sha256(folder/'scene_camera.json')
        metadata_hashes[str(folder/'scene_gt.json')]=sha256(folder/'scene_gt.json')
        keys=sorted(int(k) for k in cams if (int(k)-offset)%4==0 and 0<=(int(k)-offset)//4<13)
        if 'image_ids' in scene:
            wanted=set(map(int,scene['image_ids']))
            if not wanted<=set(keys): raise ValueError('Requested view is absent or belongs to another camera')
            keys=[k for k in keys if k in wanted]
        if not keys: raise ValueError('No selected camera frames')
        for image in keys:
            info=cams[str(image)]; stem=f'{image:06d}'
            rgb=folder/'rgb'/f'{stem}.png'; label=folder/'label'/f'{stem}.png'
            im=np.asarray(Image.open(rgb)); h,w=im.shape[:2]
            fg=np.asarray(Image.open(label))>0
            if fg.ndim!=2 or fg.shape!=(h,w) or not fg.any(): raise ValueError('Invalid GC6D foreground label')
            y,x=np.where(fg); margin=int(cfg.get('workspace_margin_px',24))
            adaptive=[max(0,int(x.min())-margin),max(0,int(y.min())-margin),
                      min(w,int(x.max())+margin+1),min(h,int(y.max())+margin+1)]
            boxes=scene.get('crop_boxes_by_image',{}).get(str(image))
            if boxes is None:
                boxes={'full':[0,0,w,h], 'roi':(np.asarray(cfg['fixed_roi_fraction'])*[w,h,w,h]).tolist(),
                       'workspace':adaptive}
                crop_source='fixed full/ROI; workspace from supplied segmentation bbox (privileged preprocessing)'
            else: crop_source='explicit exported per-view crop boxes; '+cfg['crop_provenance']
            T=np.eye(4); T[:3,:3]=np.asarray(info['cam_R_w2c']).reshape(3,3)
            T[:3,3]=np.asarray(info['cam_t_w2c']).reshape(3)/1000.
            if semantics=='w2c': T=np.linalg.inv(T)
            depth_source=cfg['reference_source']
            if depth_source=='sensor':
                depth=folder/'depth'/f'{stem}.png'
                scale=float(info['depth_scale'])/1000.  # explicit BOP scale; missing metadata fails
            elif depth_source=='rendered':
                depth=Path(cfg['rendered_depth_template'].format(root=str(root),scene=sid,image=image,camera=camera))
                scale=float(cfg['rendered_depth_scale_m'])  # never silently borrow sensor units
            else: raise ValueError('reference_source must be sensor or rendered')
            up=np.asarray(cfg['world_up'],dtype=float); up=up/np.linalg.norm(up)
            e=dict(id=f'gc6d_{sid:06d}_{stem}_{camera}',group=f'gc6d_scene_{sid:06d}',
                object_ids=sorted({f'gc6d_obj_{int(o["obj_id"])}' for o in gts[str(image)]}),
                partition=scene['partition'],previously_used=scene['previously_used'],official_test=sid in official,
                domain='gc6d',rgb_path=str(rgb),depth_path=str(depth),mask_path=str(label),depth_scale_m=scale,
                K=np.asarray(info['cam_K']).reshape(3,3).tolist(),pose_vec=(-T[:3,2]).tolist(),
                gravity_vec=(T[:3,:3].T@up).tolist(),reference_source=depth_source+'; '+cfg['reference_provenance'],
                crop_source=crop_source,crop_boxes=boxes,condition='real_gc6d',
                orientation_provenance=cfg['orientation_provenance'])
            calibration=scene.get('T_table_from_world',cfg.get('T_table_from_world'))
            if calibration is not None:
                table=np.asarray(calibration,dtype=float)@T
                e.update(T_table_from_cam=table.tolist(),plane_source='calibrated',
                    plane_provenance=scene.get('plane_provenance',cfg.get('plane_provenance','')),
                    pose_vec=(-table[:3,2]).tolist(),gravity_vec=table[2,:3].tolist())
            records.append(e)
    dest.mkdir(parents=True)
    atomic_json(dest/'records.json',{'entries':records,'provenance':{
        'config_sha256':sha256(spec_path),'metadata_sha256':metadata_hashes,'transform_semantics':semantics,
        'same_surface_tracks':'not synthesized from frame medians; supply audited tracks separately'}})
    return prepare(dest/'records.json',dest/'manifest.json')
