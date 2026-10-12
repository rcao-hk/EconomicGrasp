#!/usr/bin/env python3
"""P0-A/C controlled inference: frozen reader counterfactual or matched RGB shift.

Original and GN-Trans share original sensor-derived crop/workspace/K/image-FPS
pixel sampling and the same original sensor-cloud collision filter. GN-Trans
replaces *only* the image bytes. This is intentionally distinct from the
legacy GN-Trans virtual-depth crop/evaluation path. No sensor/GT depth is
passed as model input; depth-assisted preprocessing/filtering is disclosed.

Every physical grasp dump is indexed by canonical scene/frame identity. Each
frame uses a domain-invariant, stateless pixel-sampling seed. All metadata and
inputs required for a paired comparison are audited in output manifests.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time


def _own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--p0_domain', choices=('original','gntrans'), required=True)
    p.add_argument('--p0_cf_mode', choices=('r0','r1','r2','r3','r4'), default='r0')
    p.add_argument('--p0_gntrans_rgb_root', type=Path, default=None)
    p.add_argument('--p0_sampling_seed', type=int, default=20261012)
    p.add_argument('--p0_max_batches', type=int, default=0)
    p.add_argument('--p0_resume', action='store_true')
    if '--help' in sys.argv or '-h' in sys.argv:
        p.print_help()
        print('Common GVAR inference flags: --dataset_root, --checkpoint_path, '
              '--save_dir, --test_mode, --sample_interval 0.1, --batch_size 3, '
              '--num_workers 4, --multi_modal, --use_cdf. '
              'See scripts/run_gvar_p0_eval.sh.')
        raise SystemExit(0)
    args, rest = p.parse_known_args()
    if args.p0_max_batches < 0:
        p.error('p0_max_batches must be nonnegative')
    if args.p0_domain == 'gntrans' and args.p0_gntrans_rgb_root is None:
        p.error('--p0_gntrans_rgb_root is required for GN-Trans')
    if args.p0_domain == 'original' and args.p0_gntrans_rgb_root is not None:
        p.error('--p0_gntrans_rgb_root must not be set for Original')
    if args.p0_domain == 'gntrans' and args.p0_cf_mode != 'r0':
        p.error('GN-Trans P0-C tests the original unperturbed reader (r0) only')
    sys.argv = [sys.argv[0], *rest]
    return args


P0 = _own_args()
# Imported AFTER removal of P0 flags. The legacy GVAR inference module then
# removes its own flags BEFORE importing utils.arguments, preserving main parser.
import inference_gvar as original
import numpy as np
import torch
from PIL import Image
from graspnetAPI import GraspGroup
from torch.utils.data import DataLoader, Subset
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.gvar_runtime import (atomic_json, frame_fingerprint,
                                selected_frame_indices, validate_checkpoint)
from dataset.graspnet_dataset import collate_fn
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_gvar import EconomicGraspGVAR
from models.gripper_volume_reader import COUNTERFACTUAL_SOURCES

P0_CONTRACT = 1
from utils.gvar_p0_sampling import per_frame_sampling_seed, gntrans_rgb_path


class PairedRGBDataset(original.InferenceDataset):
    """Sensor-matched, domain-invariant gripper preprocessing.

    No GT/observed depth tensors reach the model: returns FIXED_KEYS only.
    Rendered/virtual GN-Trans depth is deliberately NOT loaded at inference.
    """
    def __init__(self, root, split, domain, gntrans_rgb_root, sample_seed, **kwargs):
        super().__init__(root, split=split, **kwargs)
        self.domain = domain
        self.sample_seed = int(sample_seed)
        self.original_colorpath = tuple(self.colorpath)
        if domain == 'gntrans':
            assert gntrans_rgb_root is not None
            self.rgb_root = Path(gntrans_rgb_root).resolve()
            self.colorpath = [str(gntrans_rgb_path(self.rgb_root, scene, int(frame)))
                              for scene,frame in zip(self.scenename,self.frameid)]
        else:
            self.rgb_root = None
        if len(self.colorpath) != len(self.original_colorpath):
            raise RuntimeError('RGB source indexing mismatch')

    def validate_rgb_size(self, indices):
        # Fail before inference: a GN-Trans image with a different image plane
        # cannot be interpreted as a material-only change under the same K.
        for i in indices:
            path = Path(self.colorpath[i])
            original_path = Path(self.original_colorpath[i])
            if not path.is_file() or not original_path.is_file():
                raise FileNotFoundError(f'RGB domain pair absent: {path} or {original_path}')
            with Image.open(path) as img, Image.open(original_path) as org:
                if img.size != org.size or img.mode != 'RGB':
                    raise ValueError(f'GN-Trans RGB plane/channel mismatch: {path}: {img.size}/{img.mode} '
                                     f'versus original {original_path}: {org.size}/{org.mode}')

    def get_data(self,index,return_raw_cloud=False):
        if return_raw_cloud:
            return super().get_data(index,return_raw_cloud=True)
        # Parent's masked-image seed pixel sampling uses np.random.choice.
        # Restore worker RNG after this single call; each frame is invariant to
        # domains, num_workers, batch size and the run's ordering.
        old = np.random.get_state()
        np.random.seed(per_frame_sampling_seed(int(index),self.sample_seed))
        try:
            return super().get_data(index,return_raw_cloud=False)
        finally:
            np.random.set_state(old)


def _hash_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _output_path(dataset,idx,out):
    return out/dataset.scenename[idx]/cfgs.camera/f'{int(dataset.frameid[idx]):04d}.npy'


def main():
    if (cfgs.test_mode not in ('test_seen','test_similar','test_novel') or
        cfgs.camera != 'realsense' or not cfgs.multi_modal or not cfgs.use_cdf or
        cfgs.use_obs_depth or cfgs.use_gt_depth):
        raise ValueError('Requires Original GraspNet RealSense, predicted-depth RGB network, CDF, 10% test split')
    if cfgs.collision_thresh != .01 or cfgs.collision_voxel_size != .01:
        raise ValueError('P0 requires matched original-sensor collision threshold and voxel=0.01m')
    if not cfgs.checkpoint_path or not Path(cfgs.checkpoint_path).is_file():
        raise FileNotFoundError('Full e19 GVAR checkpoint required')
    ckpath=Path(cfgs.checkpoint_path)
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    conf=validate_checkpoint(ck)
    if cfgs.test_mode not in ('test_seen','test_similar','test_novel'):
        raise ValueError('unsupported split')
    if int(ck['completed_epoch'])!=19:
        raise ValueError('Frozen P0 controls must use completed e19 checkpoint')
    if P0.p0_cf_mode!='r0' and conf.variant!='volume_rel':
        raise ValueError('Only volume_rel checkpoint supports counterfactual reader interventions')
    if bool(getattr(cfgs,'use_top4_view_infer',False)):
        raise ValueError('P0 protocol is A1 Top-1, not Top-4')
    # Preserve architecture from checkpoint, including volume variant/pose flags.
    for k,v in ck['architecture_config'].items():
        setattr(cfgs,k,v)
    if (cfgs.kview_mode != 'A1' or cfgs.pose_depth_mode != 'global_film'
        or not cfgs.use_cdf or bool(ck['architecture_config'].get('use_top4_view_infer', False))):
        raise ValueError('P0 requires checkpoint with A1/global_film/CDF')
    cfgs.use_top4_view_infer=False
    torch.manual_seed(int(cfgs.seed));np.random.seed(int(cfgs.seed))
    dataset=PairedRGBDataset(cfgs.dataset_root, cfgs.test_mode,
        P0.p0_domain, P0.p0_gntrans_rgb_root, P0.p0_sampling_seed,
        camera='realsense',num_points=cfgs.num_point,remove_outlier=True,
        augment=False,load_label=False,use_gt_depth=False)
    indices=selected_frame_indices(dataset,cfgs.sample_interval)
    dataset.validate_rgb_size(indices)
    # Verify that changing RGB does NOT change preprocessed metric action inputs.
    # Each pixel-index sample is seeded by its frame ID independently of domain.
    parity = {'frame_pairs_checked': 0, 'fixed_fields_checked': [], 'image_mean_abs_difference': []}
    if P0.p0_domain == 'gntrans':
        control = PairedRGBDataset(cfgs.dataset_root, cfgs.test_mode, 'original', None,
            P0.p0_sampling_seed, camera='realsense', num_points=cfgs.num_point,
            remove_outlier=True, augment=False, load_label=False, use_gt_depth=False)
        fixed_keys = sorted(original.FIXED_KEYS - {'img'})
        for j in (indices[0], indices[len(indices)//2], indices[-1]):
            source = control.get_data(j)
            shifted = dataset.get_data(j)
            for key in fixed_keys:
                a,b=source[key],shifted[key]
                a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                b = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
                if not np.array_equal(a,b, equal_nan=True):
                    raise RuntimeError(f'Matched RGB preprocessing changed {key} at frame index {j}')
            parity['image_mean_abs_difference'].append(float((source['img']-shifted['img']).abs().mean()))
            parity['frame_pairs_checked']+=1
        parity['fixed_fields_checked'] = fixed_keys
    digest=_hash_file(ckpath)
    out=Path(cfgs.save_dir)
    protocol={
        'p0_contract_version':P0_CONTRACT, 'source_domain':P0.p0_domain,
        'reader_cf_mode':P0.p0_cf_mode, 'reader_cf_sources':list(COUNTERFACTUAL_SOURCES[P0.p0_cf_mode]),
        'gvar_config':conf.to_dict(),'completed_epoch':19, 'camera':'realsense',
        'checkpoint_sha256':digest,'checkpoint':str(ckpath.resolve()),
        'split':cfgs.test_mode,'dataset_root':str(Path(cfgs.dataset_root).resolve()),
        'gntrans_rgb_root':str(P0.p0_gntrans_rgb_root.resolve()) if P0.p0_gntrans_rgb_root else None,
        'selected_samples':len(indices),'frame_stride':10,
        'frame_fingerprint':frame_fingerprint(dataset,indices),
        'image_source':'gntrans_rendered_rgb' if P0.p0_domain=='gntrans' else 'original_rgb',
        'preprocess_cloud_source':'original_sensor', 'preprocess_protocol':'paired_rgb_original_sensor_crop_v1',
        'collision_source':'original_sensor','collision_thresh':float(cfgs.collision_thresh),
        'collision_voxel_size':float(cfgs.collision_voxel_size),
        'network_geometry':'predicted metric depth', 'gt_depth_network_input':False,
        'sample_policy':'per_frame_seed_v1','sample_seed':int(P0.p0_sampling_seed),
        'batch_size':int(cfgs.batch_size),'num_workers':int(cfgs.num_workers),'seed':int(cfgs.seed),
        'max_batches':int(P0.p0_max_batches),
        'paired_input_parity':parity,
    }
    mp=out/'gvar_p0_protocol.json'
    if mp.exists():
        if not P0.p0_resume or json.loads(mp.read_text())!=protocol:
            raise ValueError('Output already has a DIFFERENT protocol or --p0_resume not enabled')
    elif out.exists() and list(out.glob('scene_*/*/*.npy')):
        raise ValueError('Unmanifested existing grasp dumps; choose new output')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(mp,protocol)
    todo=[idx for idx in indices if not (P0.p0_resume and original.valid_dump(_output_path(dataset,idx,out)))]
    loader=DataLoader(Subset(dataset,todo),batch_size=cfgs.batch_size,shuffle=False,
        num_workers=cfgs.num_workers,collate_fn=collate_fn,pin_memory=False,
        worker_init_fn=original.worker_init,
        generator=torch.Generator().manual_seed(int(cfgs.seed)),
        persistent_workers=cfgs.num_workers>0)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=EconomicGraspGVAR(min_depth=cfgs.min_depth,max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,is_training=False,use_obs_depth=False,
        use_depth_comp=False,use_cdf=True,pose_depth_mode=cfgs.pose_depth_mode,
        vis_dir=None,gvar_config=conf).to(device)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.eval()
    if P0.p0_cf_mode!='r0':
        model.kview_grasp_module.decoder.reader.set_counterfactual(P0.p0_cf_mode)
    processed=0;started=time.perf_counter()
    for step,batch in enumerate(loader):
        if P0.p0_max_batches and step>=P0.p0_max_batches:break
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        batch['cva_export_angle_feature']=False
        batch['cva_compute_diagnostics']=False
        with torch.inference_mode():
            ep=model(batch)
            pred,used=ep['depth_net_pred'],ep['depth_map_used_for_geometry']
            if not torch.allclose(pred,used,rtol=0,atol=1e-6):
                raise RuntimeError('P0 network geometry is not exclusively predicted depth')
            grasps=pred_decode_center_view_angle(ep,use_cdf=True)
        for j,prediction in enumerate(grasps):
            idx=todo[step*cfgs.batch_size+j]
            gg=GraspGroup(prediction.detach().cpu().numpy())
            cloud,_=dataset.get_data(idx,return_raw_cloud=True)
            detector=ModelFreeCollisionDetectorTorch(np.asarray(cloud).reshape(-1,3),
                                                     voxel_size=cfgs.collision_voxel_size)
            collision=detector.detect(gg,approach_dist=.05,collision_thresh=cfgs.collision_thresh)
            gg=gg[~collision.detach().cpu().numpy()]
            path=_output_path(dataset,idx,out)
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.npy.tmp')
            with tmp.open('wb') as f:
                np.save(f,gg.grasp_group_array,allow_pickle=False)
            tmp.replace(path)
            processed+=1
        if step%25==0:
            print(f'[P0] {P0.p0_domain} {conf.variant} {P0.p0_cf_mode} '
                  f'{processed}/{len(todo)} sec/frame={(time.perf_counter()-started)/max(processed,1):.3f}',flush=True)
    completed=sum(original.valid_dump(_output_path(dataset,i,out)) for i in indices)
    complete=completed==len(indices) and not P0.p0_max_batches
    atomic_json(out/'gvar_p0_summary.json',{'complete':bool(complete),
        'valid_dump_count':completed,'expected_count':len(indices),'processed_this_run':processed,
        'elapsed_seconds':time.perf_counter()-started})
    if not complete and not P0.p0_max_batches:
        raise RuntimeError(f'Incomplete P0 inference: {completed}/{len(indices)}')
    print(f'[P0] {P0.p0_domain} {conf.variant}/{P0.p0_cf_mode} complete={complete}, '
          f'{completed}/{len(indices)}',flush=True)


if __name__=='__main__':
    main()
