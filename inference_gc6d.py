#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC6D inference + evaluation script for EconomicGrasp / ecograsp_dpt checkpoints.

This script is adapted from the existing EconomicGrasp inference script and adds:
  - GraspClutter6D dataset loading with GraspClutter6DMultiDataset
  - GC6D dump layout: <save_dir>/<scene_id:06d>/<camera>/<img_id:06d>.npy
  - GC6D official evaluation through GraspClutter6DEval.eval_all()
  - optional model-free collision filtering before saving
  - optional GraspNet-1B fallback mode for sanity/comparison

Required project files:
  - dataset/gc6d_ecograsp_dpt_dataloader.py
  - graspclutter6dAPI installed/importable
  - graspnetAPI installed/importable

Typical GC6D inference + eval:

CUDA_VISIBLE_DEVICES=0 python test_gc6d_economicgrasp.py \
  --dataset gc6d \
  --dataset_root /data2/robotarm/dataset/GraspClutter6D \
  --camera realsense-d435 \
  --test_mode test \
  --checkpoint_path /path/to/checkpoint.tar \
  --model_variant depth_baseline \
  --save_dir /data2/robotarm/result/grasp/gc6d_dump/ecograsp_gc6d \
  --infer \
  --eval \
  --batch_size 4 \
  --num_workers 4 \
  --collision_thresh 0.01

If neither --infer nor --eval is given, both are run by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# graspnetAPI may still use deprecated np aliases.
np.float = np.float64  # type: ignore[attr-defined]

import torch
from torch.utils.data import DataLoader, Subset

from graspnetAPI import GraspGroup, GraspNetEval
from graspclutter6dAPI import GraspClutter6D, GraspClutter6DEval

from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn
from dataset.gc6d_dataset import GraspClutter6DMultiDataset


GC6D_CAMERAS = ("realsense-d415", "realsense-d435", "azure-kinect", "zivid")
GC6D_CAMERA_ALIASES = {"realsense": "realsense-d435", "kinect": "azure-kinect"}
G1B_CAMERAS = ("realsense", "kinect")


def parse_args():
    parser = argparse.ArgumentParser("EconomicGrasp inference/eval on GraspClutter6D.")
    parser.add_argument("--dataset", default="gc6d", choices=["gc6d", "graspnet"],
                        help="Dataset to run inference/evaluation on.")
    parser.add_argument("--dataset_root", required=True,
                        help="GC6D root if --dataset gc6d; GraspNet-1B root if --dataset graspnet.")
    parser.add_argument("--camera", default="realsense-d435",
                        help="GC6D: realsense-d415/realsense-d435/azure-kinect/zivid. "
                             "GraspNet: realsense/kinect.")
    parser.add_argument("--test_mode", default="test",
                        help="GC6D: normally 'test'. GraspNet: test_seen/test_similar/test_novel/test.")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--infer", action="store_true", help="Run inference and save grasp npy files.")
    parser.add_argument("--eval", action="store_true", help="Run official evaluation on --save_dir.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="During inference, skip samples whose output npy already exists.")
    parser.add_argument("--save_nocollision", action="store_true",
                        help="Also save predictions before model-free collision filtering under <save_dir>_nocollision.")

    # Dataset / dataloader
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    # GC6D dataloader settings. These should match your generation settings.
    parser.add_argument("--graspness_mode", default="instance", choices=["scene", "instance"])
    parser.add_argument("--mask_mode", default="workspace_depth",
                        choices=["depth", "label_depth", "bbox_depth", "workspace_depth"])
    parser.add_argument("--workspace_outlier", type=float, default=0.02)
    parser.add_argument("--workspace_pose_mode", default="json", choices=["json", "inverse", "none"])
    parser.add_argument("--workspace_depth_trunc", type=float, default=0.0)
    parser.add_argument("--factor_depth_mode", default="bop", choices=["bop", "camera", "fixed"])
    parser.add_argument("--fixed_factor_depth", type=float, default=None)
    parser.add_argument("--extend_angle", type=int, default=1)

    # Depth branch settings
    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=1.0)
    parser.add_argument("--bin_num", type=int, default=256)
    parser.add_argument("--use_gt_depth", action="store_true",
                        help="Use virtual depth as network depth input. Usually false for inference.")
    parser.add_argument("--use_obs_depth", action="store_true")
    parser.add_argument("--use_fuse_depth", action="store_true",
                        help="For GC6D this should normally be false because virtual_scenes is already fused.")

    # Model variant. Keep this small and explicit so the script fails early if the checkpoint/model mismatch.
    parser.add_argument("--model_variant", default="depth_baseline",
                        choices=["depth_baseline", "dpt_rotnet", "economicgrasp"],
                        help="Which EconomicGrasp model/pred_decode pair to instantiate.")
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--vis_every", type=int, default=1000)

    # Post-processing / evaluation
    parser.add_argument("--collision_thresh", type=float, default=0.0,
                        help="Model-free collision threshold. <=0 disables filtering.")
    parser.add_argument("--collision_voxel_size", type=float, default=0.01)
    parser.add_argument("--eval_workers", type=int, default=None)
    parser.add_argument("--sample_interval", type=float, default=1.0,
                        help="Subsample frames for inference. 1.0 = all; 0.1 = every 10th frame. "
                             "Use full 1.0 for final official evaluation.")
    parser.add_argument("--max_width", type=float, default=None,
                        help="Clamp grasp width before save. Default: GC6D 0.14, GraspNet 0.10.")
    parser.add_argument("--device", default="cuda:0")

    return parser.parse_args()


def canonical_camera(dataset: str, camera: str) -> str:
    if dataset == "gc6d":
        camera = GC6D_CAMERA_ALIASES.get(camera, camera)
        if camera not in GC6D_CAMERAS:
            raise ValueError(f"Invalid GC6D camera={camera}. Expected one of {GC6D_CAMERAS}")
        return camera
    if dataset == "graspnet":
        if camera == "realsense-d435":
            camera = "realsense"
        elif camera == "azure-kinect":
            camera = "kinect"
        if camera not in G1B_CAMERAS:
            raise ValueError(f"Invalid GraspNet camera={camera}. Expected one of {G1B_CAMERAS}")
        return camera
    raise ValueError(dataset)


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_model_and_decoder(args):
    if args.model_variant == "depth_baseline":
        from models.economicgrasp_depth import economicgrasp_depth_baseline, pred_decode
        net = economicgrasp_depth_baseline(
            seed_feat_dim=512,
            depth_stride=1,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            is_training=False,
            use_obs_depth=args.use_obs_depth,
            vis_dir=args.vis_dir,
            vis_every=args.vis_every,
        )
        return net, pred_decode

    if args.model_variant == "dpt_rotnet":
        from models.economicgrasp_bip3d import economicgrasp_dpt_rotnet as economicgrasp_dpt
        from models.economicgrasp_bip3d import pred_decode_rotnet_cva as pred_decode
        net = economicgrasp_dpt(
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            bin_num=args.bin_num,
            is_training=False,
            use_obs_depth=args.use_obs_depth,
            vis_dir=args.vis_dir,
            vis_every=args.vis_every,
        )
        return net, pred_decode

    if args.model_variant == "economicgrasp":
        from models.economicgrasp import economicgrasp, pred_decode
        net = economicgrasp(seed_feat_dim=512, is_training=False)
        return net, pred_decode

    raise ValueError(args.model_variant)


def load_checkpoint(net, checkpoint_path: str, device: torch.device):
    if checkpoint_path is None:
        raise ValueError("--checkpoint_path is required for inference.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    try:
        net.load_state_dict(state)
    except RuntimeError:
        # Common fallback for DataParallel prefixes.
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        net.load_state_dict(new_state)

    print(f"-> loaded checkpoint {checkpoint_path}")


def build_dataset(args):
    load_label = False

    if args.dataset == "gc6d":
        return GraspClutter6DMultiDataset(
            root=args.dataset_root,
            camera=args.camera,
            split=args.test_mode,
            num_points=args.num_point,
            voxel_size=args.voxel_size,
            remove_outlier=None,
            augment=False,
            load_label=load_label,
            use_gt_depth=args.use_gt_depth,
            use_fuse_depth=False,  # GC6D virtual_scenes is already fused.
            graspness_mode=args.graspness_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            bin_num=args.bin_num,
            depth_strides=1,
            extend_angle=bool(args.extend_angle),
            mask_mode=args.mask_mode,
            workspace_outlier=args.workspace_outlier,
            workspace_pose_mode=args.workspace_pose_mode,
            workspace_depth_trunc=args.workspace_depth_trunc,
            factor_depth_mode=args.factor_depth_mode,
            fixed_factor_depth=args.fixed_factor_depth,
        )

    if args.dataset == "graspnet":
        if args.model_variant == "economicgrasp":
            return GraspNetDataset(
                args.dataset_root,
                split=args.test_mode,
                camera=args.camera,
                num_points=args.num_point,
                remove_outlier=True,
                augment=False,
                load_label=load_label,
            )
        return GraspNetMultiDataset(
            args.dataset_root,
            split=args.test_mode,
            camera=args.camera,
            num_points=args.num_point,
            remove_outlier=True,
            augment=False,
            load_label=load_label,
            use_gt_depth=args.use_gt_depth,
            use_fuse_depth=args.use_fuse_depth,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            bin_num=args.bin_num,
            depth_strides=1,
            extend_angle=bool(args.extend_angle),
            graspness_mode=args.graspness_mode,
        )

    raise ValueError(args.dataset)


def build_eval_subset(dataset, sample_interval: float, annos_per_scene: int):
    total = len(dataset)
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")

    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices = []
    num_scenes = (total + annos_per_scene - 1) // annos_per_scene

    for scene_i in range(num_scenes):
        start = scene_i * annos_per_scene
        end = min((scene_i + 1) * annos_per_scene, total)
        scene_len = end - start
        local = list(range(0, scene_len, stride))
        indices.extend([start + i for i in local])

    return Subset(dataset, indices), indices


def build_dataloader(args):
    full_dataset = build_dataset(args)
    annos_per_scene = 13 if args.dataset == "gc6d" else 256
    eval_dataset, sampled_indices = build_eval_subset(full_dataset, args.sample_interval, annos_per_scene)

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
    )

    print(f"Dataset:          {args.dataset}")
    print(f"Camera:           {args.camera}")
    print(f"Split/test_mode:   {args.test_mode}")
    print(f"Total samples:     {len(full_dataset)}")
    print(f"Evaluated samples: {len(eval_dataset)}")
    print(f"sample_interval:   {args.sample_interval}")
    return full_dataset, eval_dataset, loader, sampled_indices


def _to_device_batch(batch_data, device, pin_memory=False):
    for key in batch_data:
        if "list" in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device, non_blocking=pin_memory)
        elif "graph" in key:
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].to(device, non_blocking=pin_memory)
        else:
            if torch.is_tensor(batch_data[key]):
                batch_data[key] = batch_data[key].to(device, non_blocking=pin_memory)
    return batch_data


def _scalar_int(x):
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item())
    return int(np.asarray(x).reshape(-1)[0])


def get_sample_identity(args, full_dataset, data_idx: int):
    """Return scene_dir_name, frame_file_stem, ann_id for saving."""
    if args.dataset == "gc6d":
        scene_id, img_id, ann_id = full_dataset.samples[int(data_idx)]
        return f"{int(scene_id):06d}", f"{int(img_id):06d}", int(ann_id)

    # GraspNet path convention.
    scene_name = full_dataset.scene_list()[int(data_idx)]
    ann_id = int(data_idx) % 256
    return scene_name, f"{ann_id:04d}", ann_id


def maybe_clamp_grasp_width(gg: GraspGroup, max_width: float):
    if max_width is None or len(gg) == 0:
        return gg
    arr = gg.grasp_group_array
    arr[:, 1] = np.minimum(arr[:, 1], float(max_width))
    gg.grasp_group_array = arr
    return gg


def run_inference(args, net, pred_decode, full_dataset, loader, sampled_indices, device):
    batch_interval = 20
    net.eval()
    tic = time.time()

    default_max_width = 0.14 if args.dataset == "gc6d" else 0.10
    max_width = default_max_width if args.max_width is None else args.max_width

    for batch_idx, batch_data in enumerate(loader):
        batch_data = _to_device_batch(batch_data, device, pin_memory=args.pin_memory)

        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        cur_bs = len(grasp_preds)
        for i in range(cur_bs):
            subset_data_idx = batch_idx * args.batch_size + i
            if subset_data_idx >= len(sampled_indices):
                continue
            data_idx = sampled_indices[subset_data_idx]
            scene_name, frame_name, _ = get_sample_identity(args, full_dataset, data_idx)

            save_dir = os.path.join(args.save_dir, scene_name, args.camera)
            save_path = os.path.join(save_dir, frame_name + ".npy")

            if args.skip_existing and os.path.exists(save_path):
                continue

            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)
            gg = maybe_clamp_grasp_width(gg, max_width=max_width)

            if args.save_nocollision:
                no_collision_dir = os.path.join(args.save_dir + "_nocollision", scene_name, args.camera)
                os.makedirs(no_collision_dir, exist_ok=True)
                gg.save_npy(os.path.join(no_collision_dir, frame_name + ".npy"))

            if args.collision_thresh > 0:
                cloud, _ = full_dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=args.collision_voxel_size,
                )
                collision_mask = mfcdetector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=args.collision_thresh,
                )
                collision_mask = collision_mask.detach().cpu().numpy()
                gg = gg[~collision_mask]

            os.makedirs(save_dir, exist_ok=True)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            denom = max(batch_interval, 1)
            print(f"Eval batch: {batch_idx}, avg interval time: {(toc - tic) / denom:.4f}s")
            tic = time.time()


def run_gc6d_eval(args):
    if args.test_mode != "test":
        raise ValueError("GC6D official evaluation should use --test_mode test.")

    ge = GraspClutter6DEval(root=args.dataset_root, camera=args.camera, split=args.test_mode)
    proc = args.eval_workers if args.eval_workers is not None else args.num_workers
    res, ap = ge.eval_all(args.save_dir, proc=proc)

    # Official GC6D helper returns ap as [AP, AP0.4, AP0.8] in the reference script.
    out = {
        "dataset": args.dataset,
        "split": args.test_mode,
        "camera": args.camera,
        "dump_dir": args.save_dir,
        "ap": float(ap[0]),
        "ap_0.4": float(ap[1]),
        "ap_0.8": float(ap[2]),
    }

    out_path = os.path.join(args.save_dir, f"eval_results_{args.test_mode}_{args.camera}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved evaluation results to: {out_path}")
    return res, ap


def run_graspnet_eval(args):
    ge = GraspNetEval(root=args.dataset_root, camera=args.camera, split=args.test_mode)
    proc = args.eval_workers if args.eval_workers is not None else args.num_workers

    if args.test_mode == "test_seen":
        res, ap = ge.eval_seen(args.save_dir, proc=proc)
    elif args.test_mode == "test_similar":
        res, ap = ge.eval_similar(args.save_dir, proc=proc)
    elif args.test_mode == "test_novel":
        res, ap = ge.eval_novel(args.save_dir, proc=proc)
    elif args.test_mode == "test":
        res, ap = ge.eval_all(args.save_dir, proc=proc)
    else:
        raise ValueError(f"Unknown GraspNet test_mode={args.test_mode}")

    res_arr = np.array(res).reshape(-1, 6)
    res_mean = np.mean(res_arr, axis=0)
    out = {
        "dataset": args.dataset,
        "split": args.test_mode,
        "camera": args.camera,
        "dump_dir": args.save_dir,
        "ap": float(np.mean(res_mean)),
        "ap_0.4": float(res_mean[1]),
        "ap_0.8": float(res_mean[3]),
        "res_mean": res_mean.tolist(),
    }

    out_path = os.path.join(args.save_dir, f"eval_results_{args.test_mode}_{args.camera}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved evaluation results to: {out_path}")
    return res, ap


def run_eval(args):
    if args.dataset == "gc6d":
        return run_gc6d_eval(args)
    return run_graspnet_eval(args)


def main():
    args = parse_args()
    args.camera = canonical_camera(args.dataset, args.camera)
    args.dataset_root = str(Path(args.dataset_root).expanduser())
    args.save_dir = str(Path(args.save_dir).expanduser())

    run_infer = args.infer or ((not args.infer) and (not args.eval))
    run_eval_flag = args.eval or ((not args.infer) and (not args.eval))

    if args.dataset == "gc6d" and args.test_mode != "test":
        print("[warn] GC6D official eval is normally split=test. "
              "Inference can run on other splits, but --eval will require test.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if run_infer:
        net, pred_decode = build_model_and_decoder(args)
        net.to(device)
        load_checkpoint(net, args.checkpoint_path, device)
        full_dataset, eval_dataset, loader, sampled_indices = build_dataloader(args)
        run_inference(args, net, pred_decode, full_dataset, loader, sampled_indices, device)

    if run_eval_flag:
        run_eval(args)


if __name__ == "__main__":
    main()
