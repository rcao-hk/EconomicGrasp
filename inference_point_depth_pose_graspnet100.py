#!/usr/bin/env python3
"""Original GraspNet inference for legacy/CDF point baseline (full-data-trained e19).

Training uses ALL 100 Original GraspNet scenes x 256 frames; evaluation 10% or 100%.

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
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "economicgrasp_depth_baseline":
        raise ValueError("Not a point-centric depth-baseline checkpoint")
    if checkpoint.get("pose_depth_mode") != cfgs.pose_depth_mode:
        raise ValueError("Checkpoint pose mode differs from model mode")
    if checkpoint.get("completed_epoch") != 19:
        raise ValueError("Official full-data comparison must use fixed completed_epoch=19")
    if checkpoint.get("point_head") != HEAD_ARGS.point_head:
        raise ValueError("Wrong point head selected for checkpoint")
    contract = checkpoint.get("full100_protocol")
    if not isinstance(contract,dict) or contract.get("train_frame_count")!=25600:
        raise ValueError("Checkpoint is missing the complete GraspNet training contract")
    if contract.get("head")!=HEAD_ARGS.point_head:
        raise ValueError("Mismatching head in embedded full100 protocol")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return contract


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
        vis_dir=None,
    ).cuda()
    contract=_load_checkpoint_strict(model,cfgs.checkpoint_path)
    model.eval()
    max_batches=int(os.environ.get("MAX_BATCHES","0"))
    if max_batches<0:
        raise ValueError("MAX_BATCHES must be >=0")
    (out/"inference_protocol.json").write_text(json.dumps({
        "architecture":"economicgrasp_depth_baseline",
        "completed_epoch":19,
        "checkpoint_sha256":_file_sha256(cfgs.checkpoint_path),
        "checkpoint_full100_protocol":contract,
        "point_head":HEAD_ARGS.point_head,
        "trained_on":"100percent_Original_GraspNet_RealSense",
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
