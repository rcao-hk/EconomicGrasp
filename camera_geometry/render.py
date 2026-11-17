"""Deterministic CPU CAD probe renderer; diagnostic RGB, NOT photorealistic GC6D data.

Real camera translations change perspective and z-buffer. Crops are subsequently
formed from each raw render. No digital zoom is labelled as a physical movement.
"""
from __future__ import annotations
from pathlib import Path
import math
import re
import numpy as np
from PIL import Image
from .data import read_json, atomic_json, sha256, prepare


def camera_pose(distance, tilt_deg):
    a = math.radians(tilt_deg)
    c = np.array([distance*math.sin(a),0.,distance*math.cos(a)])
    z = -c/np.linalg.norm(c)
    x = np.cross(z,[0.,0.,1.])
    x = np.array([1.,0.,0.]) if np.linalg.norm(x)<1e-8 else x/np.linalg.norm(x)
    y = np.cross(z,x)
    T = np.eye(4); T[:3,:3] = np.stack((x,y,z),1); T[:3,3] = c
    return T


def rasterize(vertices, faces, K, T, hw, color=(.65,.35,.18)):
    """Pinhole z-buffer with perspective-correct z; no face decimation."""
    H,W = hw
    v,u = np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
    rays = np.stack((u,v,np.ones_like(u)),-1) @ np.linalg.inv(K).T
    n,b = T[2,:3],T[2,3]
    dot = rays @ n
    good = dot < -1e-6
    depth = np.full((H,W),np.inf)
    depth[good] = -b/dot[good]
    rgb = np.full((H,W,3),.55,dtype=np.float64)
    mask = np.zeros((H,W),np.uint8)
    camera = (vertices-T[:3,3]) @ T[:3,:3]
    if (camera[:,2] <= .01).any(): raise ValueError('Mesh crosses camera near plane; change physical camera pose')
    pix = camera @ K.T; pix = pix[:,:2]/pix[:,2:3]
    light = np.array([.3,-.4,1.]); light /= np.linalg.norm(light)
    for face in faces:
        p = pix[face]; z = camera[face,2]
        lo = np.maximum(np.ceil(p.min(0)).astype(int),[0,0])
        hi = np.minimum(np.floor(p.max(0)).astype(int),[W-1,H-1])
        if (hi<lo).any(): continue
        yy,xx = np.meshgrid(np.arange(lo[1],hi[1]+1),np.arange(lo[0],hi[0]+1),indexing='ij')
        den = (p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den)<1e-10: continue
        w0 = ((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/den
        w1 = ((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/den
        w2 = 1-w0-w1
        invz = w0/z[0]+w1/z[1]+w2/z[2]
        zz = 1/np.maximum(invz,1e-12)
        region = np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        use = (w0>=-1e-8)&(w1>=-1e-8)&(w2>=-1e-8)&(zz<depth[region])
        norm = np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]])
        norm /= max(np.linalg.norm(norm),1e-12)
        shade = .35+.65*abs(norm@light)
        depth[region][use] = zz[use]; rgb[region][use] = np.asarray(color)*shade
        mask[region][use] = 1
    depth[~np.isfinite(depth)] = 0
    return np.clip(rgb*255,0,255).astype(np.uint8),depth.astype(np.float32),mask


def load_mesh(path, scale):
    if Path(path).suffix == '.npz':
        with np.load(path,allow_pickle=False) as x:
            v,f = x['vertices'].copy(),x['faces'].copy()
    else:
        import trimesh  # optional, only needed for PLY/OBJ/STL input
        mesh = trimesh.load(path,force='mesh',process=False)
        v,f = np.asarray(mesh.vertices),np.asarray(mesh.faces)
    v = np.asarray(v,dtype=np.float64)*float(scale); f = np.asarray(f,dtype=np.int64)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(v).all():
        raise ValueError('Finite triangle mesh required')
    if not len(f) or f.min()<0 or f.max()>=len(v): raise ValueError('Invalid triangle indices')
    # Rigid translation ONLY. Never normalize size: object dimensions retain metres.
    shift = np.array([*(.5*(v[:,:2].min(0)+v[:,:2].max(0))),v[:,2].min()])
    return v-shift,f,shift


def render(spec_path, output):
    cfg = read_json(spec_path); root = Path(output).resolve()
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    H,W = cfg.get('raw_hw',[480,640]); f0 = float(cfg.get('focal_px',600))
    entries = []; provenance = []; mesh_partitions = {}
    for obj in cfg['objects']:
        ident = obj['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',ident): raise ValueError('Safe unique object id required')
        path = Path(obj['mesh']); path = path if path.is_absolute() else Path(spec_path).parent/path
        if float(obj['scale_to_m'])<=0: raise ValueError('Explicit positive CAD-to-metre scale required')
        mesh_hash = sha256(path)
        if mesh_hash in mesh_partitions and mesh_partitions[mesh_hash] != obj['partition']:
            raise ValueError('Same CAD file crosses partitions under different object names')
        mesh_partitions[mesh_hash] = obj['partition']
        vertices,faces,shift = load_mesh(path,obj['scale_to_m'])
        if len(faces)>cfg.get('max_faces',200000): raise ValueError('Mesh exceeds declared CPU budget; no silent simplification')
        distances = cfg['train_distances_m'] if obj['partition']=='train' else cfg['eval_distances_m']
        face_ids = np.unique(np.linspace(0,len(faces)-1,min(len(faces),cfg.get('track_count',256))).astype(int))
        points = vertices[faces[face_ids]].mean(1)
        for tilt in cfg.get('tilts_deg',[0,35]):
            for mode in cfg.get('focal_modes',['fixed','matched_table_projection']):
                if mode not in ('fixed','matched_table_projection'): raise ValueError('Unknown focal intervention')
                for dist in distances:
                    if dist<=0 or not 0<=tilt<80: raise ValueError('Invalid physical camera pose')
                    f = f0 if mode=='fixed' else f0*dist/cfg.get('reference_distance_m',.45)
                    K = np.array([[f,0,(W-1)/2],[0,f,(H-1)/2],[0,0,1.]])
                    T = camera_pose(dist,tilt)
                    rgb,z,mask = rasterize(vertices,faces,K,T,(H,W),obj.get('color',[.65,.35,.18]))
                    ys,xs = np.where(mask>0)
                    if len(xs)<cfg.get('min_foreground_pixels',512): raise ValueError('Insufficient foreground in '+ident)
                    key = f'{ident}_t{tilt}_d{dist}_{mode}'; dest = root/key; dest.mkdir()
                    Image.fromarray(rgb).save(dest/'rgb.png'); Image.fromarray(mask).save(dest/'mask.png')
                    np.save(dest/'depth.npy',z)
                    m = cfg.get('workspace_margin_px',24)
                    work = [max(0,int(xs.min())-m),max(0,int(ys.min())-m),
                            min(W,int(xs.max())+m+1),min(H,int(ys.max())+m+1)]
                    roi_frac = np.asarray(cfg.get('fixed_roi_fraction',[.1,.1,.9,.9]))
                    roi = (roi_frac*np.array([W,H,W,H])).tolist()
                    cam = (points-T[:3,3]) @ T[:3,:3]
                    uv = cam @ K.T; uv = uv[:,:2]/uv[:,2:3]
                    tracks = []
                    for j,p in enumerate(uv):
                        u,v = np.rint(p).astype(int)
                        if 1<=u<W-1 and 1<=v<H-1 and mask[v,u] and abs(z[v,u]-cam[j,2])<.002:
                            tracks.append({'id':f'face{face_ids[j]}','uv':p.tolist(),'xyz_table':points[j].tolist()})
                    entries.append(dict(id=key,group=ident,object_ids=[ident],partition=obj['partition'],
                        domain='controlled_cad',previously_used=obj['previously_used'],official_test=False,
                        rgb_path=str(dest/'rgb.png'),depth_path=str(dest/'depth.npy'),mask_path=str(dest/'mask.png'),
                        depth_scale_m=1.,K=K.tolist(),T_table_from_cam=T.tolist(),
                        plane_source='renderer',plane_provenance='Known table z=0; camera look-at transform in metres',
                        reference_source='CPU triangle z-buffer optical z; table plane; no hidden-surface depth',
                        crop_source='fixed full/ROI; workspace uses rendered foreground (privileged diagnostic crop)',
                        crop_boxes={'full':[0,0,W,H],'roi':roi,'workspace':work},
                        distance_m=float(dist),condition=f'{mode}_tilt{tilt}',tracks=tracks))
        provenance.append({'object':ident,'mesh':str(path.resolve()),'sha256':sha256(path),
                           'scale_to_m':obj['scale_to_m'],'translation_removed_m':shift.tolist()})
    raw = root/'records.json'
    atomic_json(raw,{'entries':entries,'provenance':{'renderer':'flat CAD color + deterministic light; no textures/shadows',
        'meshes':provenance,'config_sha256':sha256(spec_path),
        'matched_projection':'f/d fixed for table plane; 3D object parallax is NOT removed'}})
    return prepare(raw,root/'manifest.json')
