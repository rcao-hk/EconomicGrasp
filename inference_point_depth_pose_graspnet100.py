#!/usr/bin/env python3
"""Original GraspNet inference for both point heads, Original-only or mixed e19.

Training may use full Original or paired Original/GN-Trans; test is Original
RealSense at the requested evaluation fraction.

RGB, calibration/pose and image-space sample indices reach the network; all
metric XYZ sent to the grasp backbone come from the detached predicted DPT
depth. The historical GraspNet workspace/collision preprocessing still uses
RealSense depth and must be disclosed as such.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import argparse
import sys

_p=argparse.ArgumentParser(add_help=False)
_p.add_argument("--point_head",choices=("legacy","cdf"),required=True)
_p.add_argument("--expected_train_domain",choices=("auto","original","mixed"),default="auto")
_p.add_argument("--point_rgb_fusion",choices=("none","dino_multiscale"),default="none")
HEAD_ARGS,_remaining=_p.parse_known_args()
sys.argv=[sys.argv[0],*_remaining]

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from graspnetAPI import GraspGroup

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_depth import economicgrasp_depth_baseline, pred_decode
from models.point_grasp_cdf_head import decode_point_cdf
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.point_depth_training_protocol import checkpoint_training_protocol
from utils.point_depth_mixed_contract import (
    network_rgb_pose_inputs,
    scene_frame_signature,
    scene_stride_indices,
)


class DeterministicDataset(Dataset):
    """Frame-local numpy sampling seed, independent of DataLoader workers."""
    def __init__(self, base, seed):
        self.base = base
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        state = np.random.get_state()
        h = hashlib.sha256(f"pointdepth:{self.seed}:{int(idx)}".encode()).digest()
        try:
            np.random.seed(int.from_bytes(h[:4], "little"))
            return self.base[idx]
        finally:
            np.random.set_state(state)


def _load_checkpoint_strict(model, path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    domain,protocol,manifest_key=checkpoint_training_protocol(
        ckpt,expected_head=HEAD_ARGS.point_head,
    )
    if HEAD_ARGS.expected_train_domain!="auto" and HEAD_ARGS.expected_train_domain!=domain:
        raise ValueError(
            f"Expected training domain {HEAD_ARGS.expected_train_domain}, "
            f"but checkpoint declares {domain}"
        )
    saved_rgb=ckpt.get("point_rgb_fusion",protocol.get("point_rgb_fusion","none"))
    if saved_rgb!=HEAD_ARGS.point_rgb_fusion or protocol.get("point_rgb_fusion","none")!=saved_rgb:
        raise ValueError(
            f"Point RGB fusion checkpoint/manifest/config mismatch: "
            f"checkpoint={saved_rgb!r}, request={HEAD_ARGS.point_rgb_fusion!r}"
        )
    if saved_rgb!="none" and not protocol.get("shared_depth_checkpoint_sha256"):
        raise ValueError("A2 must come from a verified shared-frozen depth checkpoint")
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    return domain,protocol,manifest_key


def _file_sha256(path):
    dig = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def inference():
    if not cfgs.checkpoint_path or not cfgs.save_dir or not cfgs.test_mode:
        raise ValueError("--checkpoint_path, --save_dir, --test_mode are required")
    if cfgs.test_mode not in ("test_seen","test_similar","test_novel"):
        raise ValueError("Only official Original test_seen/test_similar/test_novel supported")
    if cfgs.camera != "realsense":
        raise ValueError("Controlled comparison expects RealSense")
    if str(cfgs.pose_depth_mode) != "global_film":
        raise ValueError("Expected --pose_depth_mode global_film")
    if cfgs.use_gt_depth or cfgs.use_obs_depth or cfgs.use_fuse_depth:
        raise ValueError("No GT/observed/fused depth may be used as network geometry")
    if cfgs.min_depth != 0.2 or cfgs.max_depth != 1.0:
        raise ValueError("Depth range mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("Original point-centric MinkowskiEngine requires CUDA")
    out = Path(cfgs.save_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty inference result directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    full = GraspNetMultiDataset(
        cfgs.dataset_root,
        camera=cfgs.camera,
        split=cfgs.test_mode,
        num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=False,
    )
    indices=scene_stride_indices(full,float(cfgs.sample_interval))
    if float(cfgs.sample_interval)==0.1 and len(indices)!=780:
        raise RuntimeError(f"Expected 780 frames, got {len(indices)}")
    selected=Subset(DeterministicDataset(full,cfgs.seed),indices)
    loader=DataLoader(
        selected,batch_size=cfgs.batch_size,shuffle=False,num_workers=cfgs.num_workers,
        collate_fn=collate_fn,pin_memory=False,
        persistent_workers=cfgs.num_workers>0,
    )
    model=economicgrasp_depth_baseline(
        num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode="global_film",
        point_use_cdf=HEAD_ARGS.point_head=="cdf",
        point_rgb_fusion=HEAD_ARGS.point_rgb_fusion,
        vis_dir=None,
    ).cuda()
    train_domain,contract,manifest_key=_load_checkpoint_strict(model,cfgs.checkpoint_path)
    model.eval()
    max_batches=int(os.environ.get("MAX_BATCHES","0"))
    if max_batches<0:
        raise ValueError("MAX_BATCHES must be >=0")
    (out/"inference_protocol.json").write_text(json.dumps({
        "architecture":"economicgrasp_depth_baseline",
        "completed_epoch":19,
        "checkpoint_sha256":_file_sha256(cfgs.checkpoint_path),
        "checkpoint_manifest_key":manifest_key,
        "checkpoint_training_protocol":contract,
        "point_head":HEAD_ARGS.point_head,
        "point_rgb_fusion":HEAD_ARGS.point_rgb_fusion,
        "shared_depth_checkpoint_sha256":contract.get("shared_depth_checkpoint_sha256"),
        "train_domain":train_domain,
        "train_fraction":(
            contract.get("mix_train_fraction",contract.get("train_fraction",1.0))
            if train_domain=="mixed" else 1.0
        ),
        "training_frame_count":contract.get("train_frame_count",contract.get("train_count")),
        "trained_on":(
            "paired_Original_GraspNet_plus_GNTrans" if train_domain=="mixed"
            else "100percent_Original_GraspNet_RealSense"
        ),
        "test_mode":cfgs.test_mode,
        "sample_interval_fraction":float(cfgs.sample_interval),
        "dataset_count":len(selected),
        "scene_frame_sha256":scene_frame_signature(full,indices),
        "inference_sampling":"stateless_numpy_seed_from_dataset_idx",
        "sampling_seed":int(cfgs.seed),
        "max_batches":max_batches,
        "status":"smoke_only" if max_batches else "formal_inference",
        "geometry_input":"predicted_DPT_depth_only",
        "preprocessing":"original_RealSense_sensor_depth_crop_and_point_sampling",
        "collision_cloud":"original_RealSense_sensor",
        "collision_thresh":float(cfgs.collision_thresh),
        "collision_voxel_size":float(cfgs.collision_voxel_size),
        "batch_size":int(cfgs.batch_size),
    },indent=2,sort_keys=True)+"\n")
    processed=0
    start=time.monotonic()
    for batch_index,batch in enumerate(loader):
        if max_batches and batch_index>=max_batches:
            break
        net_input=network_rgb_pose_inputs(batch,pose_depth_mode="global_film")
        for key,value in net_input.items():
            if torch.is_tensor(value):
                net_input[key]=value.cuda(non_blocking=False)
        with torch.no_grad():
            output=model(net_input)
            if output.get("depth_map_used_for_geometry")!="depth_net_pred":
                raise RuntimeError("External depth entered the point-cloud grasp pipeline")
            grasps=(decode_point_cdf(
                output,num_angle=cfgs.num_angle,num_depth=cfgs.num_depth,
                max_width=cfgs.grasp_max_width,
            ) if HEAD_ARGS.point_head=="cdf" else pred_decode(output))
        for j,pred in enumerate(grasps):
            idx=indices[batch_index*cfgs.batch_size+j]
            frame=int(full.frameid[idx])
            scene=str(full.scenename[idx])
            values=pred.detach().float().cpu().numpy()
            if values.ndim!=2 or values.shape[1]!=17 or not np.isfinite(values).all():
                raise FloatingPointError(f"Invalid predicted grasp array {scene}/{frame}")
            group=GraspGroup(values)
            if cfgs.collision_thresh>0:
                sensor_cloud,_=full.get_data(idx,return_raw_cloud=True)
                detector=ModelFreeCollisionDetectorTorch(
                    sensor_cloud.reshape(-1,3),voxel_size=cfgs.collision_voxel_size
                )
                collision=detector.detect(
                    group,approach_dist=0.05,collision_thresh=cfgs.collision_thresh
                )
                group=group[~collision.detach().cpu().numpy()]
            dst=out/scene/cfgs.camera/f"{frame:04d}.npy"
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite: {dst}")
            group.save_npy(str(dst))
            processed+=1
        if batch_index%20==0:
            print(f"[POINT-INFER] split={cfgs.test_mode} batches={batch_index+1}/{len(loader)} "
                  f"frames={processed}/{len(selected)} elapsed={time.monotonic()-start:.1f}s",
                  flush=True)
    if not max_batches and processed!=len(selected):
        raise RuntimeError(f"Missing formal frames: processed={processed} expected={len(selected)}")
    (out/"inference_summary.json").write_text(json.dumps({
        "processed_frames":processed,
        "expected_frames":len(selected),
        "complete":processed==len(selected),
        "smoke":bool(max_batches),
        "wall_s":time.monotonic()-start,
    },indent=2)+"\n")
    print(f"[POINT-INFER] completed {processed} frames, smoke={bool(max_batches)}")


if __name__=="__main__":
    inference()
