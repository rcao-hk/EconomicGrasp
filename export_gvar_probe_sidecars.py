#!/usr/bin/env python3
"""Export *diagnostic-only* aligned depth sidecars for selected GVAR frames.

Loads a FULL e19 GVAR checkpoint, uses the original RGB inference graph and
GraspNet's existing crop-aware GT/sensor-depth alignment, but passes ONLY RGB,
K and pose/image indices to the model. Reference depths are never model inputs.
No grasp re-evaluation, no gradient update and no learned-uncertainty fabrication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(4<<20),b''):d.update(c)
    return d.hexdigest()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='GVAR deployment results root')
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--selection-json',type=Path,required=True)
    p.add_argument('--variant',choices=('baseline','slot','volume_fixed','volume','volume_rel','volume_fixed_rel'),default='volume_rel')
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epoch',type=int,default=19)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-frames',type=int,default=0,help='Only for smoke; 0=all')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--uncertainty-map-root',type=Path,default=None,
                   help='Optional *external* per-frame estimated sigma depth map; MUST be same cropped resolution')
    p.add_argument('--uncertainty-units',choices=('m','mm'),default='m')
    p.add_argument('--uncertainty-source',choices=('rgb_only','depth_assisted_privileged'),default=None)
    return p


def require(cond,msg):
    if not cond: raise ValueError(msg)


def selected_frames(p:Path):
    dat=json.loads(p.read_text());require(dat.get('split')=='test_novel','Only Novel selection accepted')
    rows=dat['selected_frames']
    require(len(rows)>0,'Empty selection')
    keys=[]
    for row in rows:
        s,f=int(row['scene_id']),int(row['frame_id'])
        require(160<=s<190 and f in range(0,256,10),f'Non-canonical scene/frame {s}/{f}')
        keys.append((s,f))
    return sorted(set(keys))


def run(args):
    require(args.max_frames>=0,'max_frames must be nonnegative')
    require(args.uncertainty_map_root is None or args.uncertainty_source is not None,
            '--uncertainty-source required with external uncertainty maps')
    require(args.output_dir.resolve()!=args.root.resolve(),'Do not overwrite experimental root')
    import_path=args.root/'eval'/f'{args.variant}_e{args.epoch}'/'test_novel'/'gvar_inference_protocol.json'
    base_proto=json.loads(import_path.read_text())
    require(base_proto['gvar_config']['variant']==args.variant,'Variant mismatch with evaluated result')
    require(base_proto['completed_epoch']==args.epoch,'Wrong checkpoint epoch')
    ck=args.checkpoint or args.root/'train'/args.variant/f'checkpoint_epoch_{args.epoch:03d}.tar'
    require(ck.is_file(),f'Missing checkpoint {ck}')
    ck_sha=sha(ck)
    require(ck_sha==base_proto['checkpoint_sha256'],'Checkpoint is not the one used for archived AP inference')
    require(str(args.dataset_root.resolve())==str(Path(base_proto['dataset_root']).resolve()),'Dataset root mismatch with AP inference protocol')
    frames=selected_frames(args.selection_json)
    selection_sha=sha(args.selection_json)
    if args.max_frames:frames=frames[:args.max_frames]
    # Legacy utils.arguments parses sys.argv at import time. Our own args are
    # consumed before any model/dataset imports; no global parser edits.
    sys.argv=[sys.argv[0]]
    from inference_gvar import InferenceDataset, FIXED_KEYS
    from utils.arguments import cfgs
    from models.economicgrasp_gvar import EconomicGraspGVAR
    from utils.gvar_runtime import validate_checkpoint
    from dataset.graspnet_dataset import GraspNetMultiDataset
    from PIL import Image
    import cv2

    checkpoint=torch.load(ck,map_location='cpu',weights_only=False)
    config=validate_checkpoint(checkpoint)
    require(config.variant==args.variant,'Checkpoint model variant mismatch')
    for k,v in checkpoint['architecture_config'].items():setattr(cfgs,k,v)
    cfgs.multi_modal=True
    cfgs.use_cdf=True
    cfgs.camera='realsense'
    cfgs.use_obs_depth=False
    cfgs.use_gt_depth=False
    cfgs.use_top4_view_infer=False
    cfgs.kview_mode='A1'
    cfgs.pose_depth_mode='global_film'
    torch.manual_seed(int(checkpoint.get('gvar_protocol',{}).get('seed',0)))
    np.random.seed(int(checkpoint.get('gvar_protocol',{}).get('seed',0)))

    class DiagnosticDataset(InferenceDataset):
        def _crop_box_from_mask(self,mask):
            box=super()._crop_box_from_mask(mask)
            self._last_crop=box
            return box
        def get_data(self,index,return_raw_cloud=False):
            # Bypass filtered InferenceDataset.get_data; take GraspNet's exact
            # cropped-and-resized reference maps in the same K/crop coordinates.
            sample=GraspNetMultiDataset.get_data(self,index,return_raw_cloud=return_raw_cloud)
            if return_raw_cloud:return sample
            required={'img','img_idxs','K','gt_depth_m','sensor_depth_m'}
            require(required.issubset(sample),f'Missing reference/input keys {required-set(sample)}')
            x0,y0,x1,y1=self._last_crop
            seg=np.asarray(Image.open(self.labelpath[index]))
            fg=cv2.resize((seg[y0:y1,x0:x1]>0).astype('uint8'),
                          (sample['gt_depth_m'].shape[1],sample['gt_depth_m'].shape[0]),
                          interpolation=cv2.INTER_NEAREST)
            sample['foreground_mask']=fg
            return {k:sample[k] for k in (required|{'camera_pose_vec','camera_gravity_vec','scene_idx','anno_idx','dataset_idx','foreground_mask'}) if k in sample}

    dataset=DiagnosticDataset(str(args.dataset_root),camera='realsense',split='test_novel',
                              num_points=cfgs.num_point,remove_outlier=True,augment=False,
                              load_label=False,use_gt_depth=False,use_fuse_depth=False)
    model=EconomicGraspGVAR(min_depth=cfgs.min_depth,max_depth=cfgs.max_depth,bin_num=cfgs.bin_num,
                              is_training=False,use_obs_depth=False,use_cdf=True,use_depth_comp=False,
                              pose_depth_mode='global_film',vis_dir=None,gvar_config=config)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    device=torch.device(args.device)
    model=model.to(device).eval()
    dest=args.output_dir/args.variant
    manifest={"version":1,"variant":args.variant,"checkpoint_sha256":ck_sha,
              "selection_sha256":selection_sha,"epoch":args.epoch,"dataset_root":str(args.dataset_root.resolve()),
              "source_inference_protocol_sha256":sha(import_path),"uncertainty_source":args.uncertainty_source or 'none',
              "uncertainty_units":args.uncertainty_units if args.uncertainty_map_root else None,
              "max_frames":args.max_frames,"frames":[{'scene_id':s,'frame_id':f} for s,f in frames]}
    mp=dest/'export_manifest.json'
    if dest.exists() and any(dest.iterdir()):
        require(args.resume and mp.is_file() and json.loads(mp.read_text())==manifest,
                f'Nonempty/mismatched sidecar destination {dest}; use new dir or --resume with exact manifest')
    dest.mkdir(parents=True,exist_ok=True)
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    n=0
    for s,f in frames:
        path=dest/f'scene_{s:04d}'/f'{f:04d}.npz'
        if args.resume and path.is_file():
            with np.load(path,allow_pickle=False) as saved:
                require({'pred_depth_m','gt_depth_m','sensor_depth_m','K','foreground_mask','crop_rgb'}<=set(saved.files),f'Incomplete sidecar {path}')
            continue
        index=(s-160)*256+f
        require(int(dataset.frameid[index])==f and dataset.scenename[index]==f'scene_{s:04d}',f'Dataset row mismatch {s}/{f}')
        sample=dataset[index]
        gt=np.asarray(sample.pop('gt_depth_m'),dtype=np.float32)
        sensor=np.asarray(sample.pop('sensor_depth_m'),dtype=np.float32)
        fg=np.asarray(sample.pop('foreground_mask'),dtype=np.uint8)
        img=sample['img']
        rgb=img.detach().float().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
        rgb=np.clip(rgb.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
        crop_rgb=(rgb*255).astype(np.uint8)
        features={k: (torch.as_tensor(v).to(device).unsqueeze(0) if not torch.is_tensor(v) else v.to(device).unsqueeze(0)) for k,v in sample.items() if k in FIXED_KEYS}
        features['cva_export_angle_feature']=False
        features['cva_compute_diagnostics']=False
        with torch.inference_mode():
            ep=model(features)
            pred=ep['depth_net_pred']
            used=ep['depth_map_used_for_geometry']
            require(pred.shape==used.shape and torch.allclose(pred,used,rtol=0,atol=1e-6),'GT/observed depth leaked into model geometry')
            pred=pred.detach().cpu().numpy()[0,0].astype(np.float32)
        require(pred.shape==gt.shape==sensor.shape==fg.shape, f'Crop misalignment at {s}/{f}')
        values={"pred_depth_m":pred,"gt_depth_m":gt,"sensor_depth_m":sensor,
                "foreground_mask":fg,"crop_rgb":crop_rgb,"K":np.asarray(sample['K'],dtype=np.float32)}
        if args.uncertainty_map_root:
            source=args.uncertainty_map_root/f'scene_{s:04d}'/f'{f:04d}.npy'
            require(source.is_file(),f'Missing external uncertainty map {source}')
            sigma=np.load(source,allow_pickle=False).astype(np.float32)
            require(sigma.shape==pred.shape and np.isfinite(sigma).all() and (sigma>=0).all(),f'Invalid sigma map {source}')
            if args.uncertainty_units=='mm':sigma/=1000
            values['uncertainty_sigma_m']=sigma
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as stream: np.savez_compressed(stream,**values)
        tmp.replace(path)
        n+=1
        print(f'[EXPORT] {args.variant} scene_{s:04d}/{f:04d}, pred/ref/fg={pred.shape}',flush=True)
    print(f'Completed {n}/{len(frames)} new sidecars into {dest}',flush=True)


if __name__=='__main__':
    p=parser();a=p.parse_args();run(a)
