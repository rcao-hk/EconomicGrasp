"""Fixed original-GraspNet geometry shared by RS and paired GN-Trans RGB.

Reproduces main GraspNetMultiDataset's workspace/crop/K path but avoids
unneeded 20k random point sampling, grasp labels and dense depth-bin targets.
Uses sensor depth + segmentation for workspace/crop (privileged preprocessing),
rendered depth solely as evaluation target, and supplied table calibration.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import scipy.io as scio

from .metrics import SPLITS, deterministic_frames, scene_ids


class FixedGraspNetFrames(Dataset):
    def __init__(self, dataset_root, *, gntrans_rgb_root=None, camera="realsense",
                 splits=SPLITS, frames_per_scene=8, max_scenes=None,
                 require_paired_rgb=False):
        from dataset.graspnet_dataset import GraspNetMultiDataset
        if camera != "realsense":
            raise ValueError("Registered MIX50 protocol uses GraspNet RealSense only.")
        self.root=Path(dataset_root).resolve()
        self.gntrans_root=Path(gntrans_rgb_root).resolve() if gntrans_rgb_root else None
        self.paired=bool(require_paired_rgb)
        if self.paired and self.gntrans_root is None:
            raise ValueError("--appearance requires GN-Trans RGB root.")
        self.datasets={}
        self.tasks=[]
        for split in splits:
            if split not in SPLITS: raise ValueError("Unknown split: "+split)
            underlying="train" if split.startswith("train_") else split
            if underlying not in self.datasets:
                self.datasets[underlying]=GraspNetMultiDataset(
                    str(self.root),camera=camera,split=underlying,
                    num_points=1,remove_outlier=True,augment=False,
                    load_label=False,use_gt_depth=False,use_fuse_depth=False,
                    depth_strides=448,
                )
            scenes=scene_ids(split)
            if max_scenes is not None:
                if not 0 < max_scenes <= len(scenes):
                    raise ValueError(f"max_scenes={max_scenes} outside selected split")
                scenes=scenes[:max_scenes]
            for scene in scenes:
                for frame in deterministic_frames(split,frames_per_scene):
                    self.tasks.append((split,underlying,int(scene),int(frame)))
        if len(self.tasks)!=len(set(self.tasks)):
            raise ValueError("Duplicate split-scene-frame tasks.")

    def __len__(self): return len(self.tasks)

    def _load_task(self, idx, include_rgb=True):
        split,underlying,scene_id,frame=self.tasks[idx]
        ds=self.datasets[underlying]
        offset=scene_id-(0 if underlying=="train" else {
            "test_seen":100,"test_similar":130,"test_novel":160}[underlying])
        ds_idx=offset*256+frame
        scene=f"scene_{scene_id:04d}"
        if ds.scenename[ds_idx]!=scene or int(ds.frameid[ds_idx])!=frame:
            raise ValueError(f"Scene/frame index mismatch: {split} {scene} {frame}.")
        raw_sensor=np.asarray(Image.open(ds.depthpath[ds_idx]))
        raw_seg=np.asarray(Image.open(ds.labelpath[ds_idx]))
        raw_gt=np.asarray(Image.open(ds.gtdepthpath[ds_idx]))
        if raw_sensor.ndim!=2 or raw_seg.shape!=raw_sensor.shape or raw_gt.shape!=raw_seg.shape:
            raise ValueError("Raw sensor/seg/rendered GT dimensions differ.")
        meta=scio.loadmat(ds.metapath[ds_idx])
        K_orig=np.asarray(meta["intrinsic_matrix"],dtype=np.float32)
        if K_orig.shape!=(3,3):
            raise ValueError("Expected 3x3 original intrinsics.")
        factor=float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
        if not np.isfinite(factor) or factor<=0: raise ValueError("Invalid factor_depth.")
        from utils.data_utils import (
            CameraInfo,create_point_cloud_from_depth_image,get_workspace_mask
        )
        H,W=raw_seg.shape
        info=CameraInfo(float(W),float(H),float(K_orig[0,0]),float(K_orig[1,1]),
                        float(K_orig[0,2]),float(K_orig[1,2]),factor)
        cloud=create_point_cloud_from_depth_image(raw_sensor,info,organized=True)
        poses,align=ds._load_camera_geometry(scene)
        T=align @ poses[frame]
        workspace=get_workspace_mask(cloud,raw_seg,trans=T,organized=True,outlier=.02)
        mask=(raw_sensor>0)&workspace
        box=ds._crop_box_from_mask(mask)
        x0,y0,x1,y1=box
        if not (0<=x0<x1<=W and 0<=y0<y1<=H):
            raise ValueError("Invalid production crop.")
        resized=(448,448)
        K=ds.resize_intrinsics_with_crop(K_orig,box,out_hw=resized)
        # Training target: same GraspNet rendered GT in object foreground.
        # Original TSDF background supervision is irrelevant to this fixed
        # foreground-only diagnostic and MUST NOT silently enter the mask.
        gt_m=ds.build_fused_gt_depth_m(raw_gt,raw_seg,ds_idx,factor)
        gt=cv2.resize(gt_m[y0:y1,x0:x1],resized,interpolation=cv2.INTER_NEAREST)
        seg=cv2.resize(raw_seg[y0:y1,x0:x1],resized,interpolation=cv2.INTER_NEAREST)
        plane=ds.build_camera_plane_metadata(scene,frame)
        pose,gravity,tilt=ds.build_camera_orientation_metadata(scene,frame)
        sample={
            "split":split,"scene":scene,"frame":frame,
            "gt":torch.from_numpy(gt.astype(np.float32,copy=True)),
            "seg":torch.from_numpy(seg.astype(np.int32,copy=True)),
            "K":torch.from_numpy(np.asarray(K,dtype=np.float32).copy()),
            "plane":torch.from_numpy(np.asarray(plane,dtype=np.float32).copy()),
            "pose":torch.from_numpy(np.asarray(pose,dtype=np.float32).copy()),
            "camera_tilt_deg":float(tilt),
            "crop_box":torch.tensor(box,dtype=torch.long),
            "raw_h":H,"raw_w":W,
        }
        if include_rgb:
            real=np.asarray(Image.open(ds.colorpath[ds_idx]),dtype=np.float32)/255.
            if real.shape!=(H,W,3): raise ValueError("Original raw RGB alignment mismatch.")
            sample["rgb"]=ds.img_transforms(real[y0:y1,x0:x1].copy())
            if self.paired:
                if self.gntrans_root is None:raise ValueError("GN-Trans root missing.")
                path=self.gntrans_root/"scenes"/f"{scene_id:05d}"/f"{frame:04d}_color.png"
                if not path.is_file(): raise FileNotFoundError(str(path))
                trans=np.asarray(Image.open(path),dtype=np.float32)/255.
                if trans.shape!=real.shape:
                    raise ValueError(f"GN-Trans and RS must share raw pixel geometry: {path}")
                sample["rgb_trans"]=ds.img_transforms(trans[y0:y1,x0:x1].copy())
                sample["trans_rgb_path"]=str(path)
        return sample


    def verify_native_preprocessing(self, idx, tolerance=1e-5):
        """Check exact crop/K/RGB/rendered target parity with main's get_data.

        Important: main's get_data samples a random sparse point, but it does
        not use that random sample to determine workspace or crop geometry.
        """
        sample=self._load_task(idx,include_rgb=True)
        split,underlying,scene_id,frame=self.tasks[idx]
        ds=self.datasets[underlying]
        start=0 if underlying=="train" else {
            "test_seen":100,"test_similar":130,"test_novel":160}[underlying]
        native=ds.get_data((scene_id-start)*256+frame)
        err={}
        for key,a,b in (
            ("rgb",sample["rgb"].numpy(),native["img"].numpy()),
            ("K",sample["K"].numpy(),np.asarray(native["K"])),
            ("rendered_gt_m",sample["gt"].numpy(),np.asarray(native["gt_depth_m"])),
            ("camera_plane",sample["plane"].numpy(),np.asarray(native["camera_plane"])),
            ("camera_pose_vec",sample["pose"].numpy(),np.asarray(native["camera_pose_vec"])),
        ):
            if a.shape!=b.shape:
                raise ValueError(f"Main loader {key} shape differs: {a.shape} vs {b.shape}")
            diff=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            err[key+"_max_diff"]=diff
            if diff>tolerance:
                raise ValueError(
                    f"{split}/{scene_id}/{frame} mismatch in main {key}, max={diff:.6g}."
                )
        # Production main get_data does not expose raw crop_box; K/rgb/GT
        # parity above jointly checks the derived crop without inventing a key.
        return dict(split=split,scene=sample["scene"],frame=frame,
                    absolute_tolerance=tolerance,**err)

    def __getitem__(self, idx):
        return self._load_task(idx,include_rgb=True)

    def for_prior(self, idx):
        return self._load_task(idx,include_rgb=False)


def build_registry(dataset_root,gntrans_root,splits,frames_per_scene,max_scenes=None,
                   require_paired_rgb=False):
    return FixedGraspNetFrames(
        dataset_root,gntrans_rgb_root=gntrans_root,
        splits=splits,frames_per_scene=frames_per_scene,
        max_scenes=max_scenes,require_paired_rgb=require_paired_rgb
    )
