#!/usr/bin/env python3
"""GVAR inference on original GraspNet 0,10,...,250 frames in each test scene."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

from utils.gvar_runtime import (parse_gvar_args, validate_checkpoint, selected_frame_indices,
    frame_fingerprint, atomic_json, CONTRACT_VERSION)
EXPLICIT_ARGV = list(sys.argv)
G = parse_gvar_args(inference=True)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_gvar import EconomicGraspGVAR
from models.economicgrasp_bip3d import pred_decode_center_view_angle

FIXED_KEYS = {"img", "img_idxs", "K", "camera_pose_vec", "camera_gravity_vec", "scene_idx", "anno_idx", "dataset_idx"}


class InferenceDataset(GraspNetMultiDataset):
    # Keep main's depth-assisted crop/workspace/image-index preprocessing EXACTLY.
    # Only remove unused training payload; no giant 448*448*256 depth target.
    def build_depth_prob_gt(self, depth_m_resized):
        return np.empty((1, 0, 0), np.float32), np.empty((1, 0), np.float32)

    def get_data(self, index, return_raw_cloud=False):
        output = super().get_data(index, return_raw_cloud=return_raw_cloud)
        if return_raw_cloud:
            return output
        return {key: value for key, value in output.items() if key in FIXED_KEYS}


def worker_init(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))


def output_path(dataset, index, out):
    return out / dataset.scenename[index] / cfgs.camera / f"{int(dataset.frameid[index]):04d}.npy"


def valid_dump(path):
    try:
        arr = np.load(path, allow_pickle=False)
        return arr.ndim == 2 and arr.shape[1] == 17 and bool(np.isfinite(arr).all())
    except (OSError, ValueError, EOFError):
        return False


def main():
    if cfgs.test_mode not in ("test_seen", "test_similar", "test_novel"):
        raise ValueError("Select one original GraspNet test split")
    if cfgs.camera != "realsense" or not cfgs.multi_modal:
        raise ValueError("This controlled experiment requires RealSense and --multi_modal")
    if cfgs.use_obs_depth or cfgs.use_gt_depth:
        raise ValueError("Observed/GT geometry network input is forbidden")
    if not cfgs.checkpoint_path or not Path(cfgs.checkpoint_path).is_file():
        raise FileNotFoundError("--checkpoint_path is missing or not a file")
    ck = torch.load(cfgs.checkpoint_path, map_location="cpu", weights_only=False)
    config = validate_checkpoint(ck)
    if "--gvar_action_chunk" in EXPLICIT_ARGV:
        config = replace(config, action_chunk=G.gvar_action_chunk)
    if G.gvar_variant != "auto" and G.gvar_variant != config.variant:
        raise ValueError(f"Requested {G.gvar_variant}, checkpoint is {config.variant}")
    # Architecture comes from the checkpoint, never silently from today's defaults.
    for key, value in ck["architecture_config"].items():
        if "--" + key in EXPLICIT_ARGV and getattr(cfgs, key, None) != value:
            raise ValueError(f"Explicit CLI {key} conflicts with checkpoint")
        setattr(cfgs, key, value)
    if not cfgs.use_cdf or cfgs.kview_mode != "A1" or cfgs.use_top4_view_infer:
        raise ValueError("Only CDF A1/Top-1 checkpoints belong to this experiment")
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required")
    out = Path(cfgs.save_dir)
    torch.manual_seed(cfgs.seed)
    np.random.seed(cfgs.seed)
    dataset = InferenceDataset(cfgs.dataset_root, split=cfgs.test_mode, camera=cfgs.camera,
        num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False, use_gt_depth=False)
    indices = selected_frame_indices(dataset, cfgs.sample_interval)
    # Hash state identity cheaply via checkpoint bytes once, not each frame.
    digest = hashlib.sha256()
    with open(cfgs.checkpoint_path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    protocol = {
        "gvar_contract_version": CONTRACT_VERSION, "gvar_config": config.to_dict(),
        "checkpoint": str(Path(cfgs.checkpoint_path).resolve()), "checkpoint_sha256": digest.hexdigest(),
        "completed_epoch": ck["completed_epoch"], "split": cfgs.test_mode, "camera": cfgs.camera,
        "dataset_root": str(Path(cfgs.dataset_root).resolve()), "selected_samples": len(indices),
        "frame_stride": 10, "frame_fingerprint": frame_fingerprint(dataset, indices),
        "collision_thresh": cfgs.collision_thresh, "collision_voxel_size": cfgs.collision_voxel_size,
        "collision_source": "original_sensor" if cfgs.collision_thresh > 0 else "none",
        "network_geometry": "predicted metric depth", "depth_assisted_dataset_preprocessing": True,
        "batch_size": cfgs.batch_size, "seed": cfgs.seed, "smoke_max_batches": G.gvar_max_batches,
    }
    manifest = out / "gvar_inference_protocol.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if not G.gvar_resume_inference or prior != protocol:
            raise ValueError("Existing inference output/config; choose a new output path or resume the IDENTICAL run")
    elif out.exists() and any(out.glob("scene_*/*/*.npy")):
        raise ValueError("Unmanifested grasp dumps in output directory; refusing to mix runs")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, protocol)
    todo = [i for i in indices if not (G.gvar_resume_inference and valid_dump(output_path(dataset, i, out)))]
    loader = DataLoader(Subset(dataset, todo), batch_size=cfgs.batch_size, shuffle=False,
        num_workers=cfgs.num_workers, worker_init_fn=worker_init, collate_fn=collate_fn, pin_memory=False,
        generator=torch.Generator().manual_seed(cfgs.seed), persistent_workers=cfgs.num_workers > 0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EconomicGraspGVAR(min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, bin_num=cfgs.bin_num,
        is_training=False, use_obs_depth=False, use_depth_comp=False, use_cdf=True,
        pose_depth_mode=cfgs.pose_depth_mode, vis_dir=None, gvar_config=config).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    started, processed = time.perf_counter(), 0
    for step, batch in enumerate(loader):
        if G.gvar_max_batches and step >= G.gvar_max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["cva_export_angle_feature"] = False
        batch["cva_compute_diagnostics"] = False
        with torch.inference_mode():
            ep = model(batch)
            if not torch.allclose(ep["depth_net_pred"], ep["depth_map_used_for_geometry"], atol=1e-6, rtol=0):
                raise RuntimeError("Predicted-geometry contract violated")
            grasps = pred_decode_center_view_angle(ep, use_cdf=True)
        for j, prediction in enumerate(grasps):
            idx = todo[step * cfgs.batch_size + j]
            gg = GraspGroup(prediction.detach().cpu().numpy())
            if cfgs.collision_thresh > 0:
                cloud, _ = dataset.get_data(idx, return_raw_cloud=True)
                detector = ModelFreeCollisionDetectorTorch(np.asarray(cloud).reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
                mask = detector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~mask.detach().cpu().numpy()]
            path = output_path(dataset, idx, out)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic numeric dump; equivalent to GraspGroup.save_npy, no pickle.
            tmp = path.with_suffix(".npy.tmp")
            with open(tmp, "wb") as f:
                np.save(f, gg.grasp_group_array, allow_pickle=False)
            tmp.replace(path)
            processed += 1
        if step % 20 == 0:
            print(f"[GVAR-INFER] {config.variant}/{cfgs.test_mode} {processed}/{len(todo)} "
                  f"sec/frame={(time.perf_counter()-started)/max(processed,1):.3f}", flush=True)
    count = sum(valid_dump(output_path(dataset, i, out)) for i in indices)
    complete = count == len(indices) and not G.gvar_max_batches
    atomic_json(out / "gvar_inference_summary.json", {"complete": complete, "valid_dump_count": count,
        "expected_count": len(indices), "processed_this_run": processed, "elapsed_seconds": time.perf_counter()-started})
    if not complete and not G.gvar_max_batches:
        raise RuntimeError(f"Incomplete inference: {count}/{len(indices)}")
    print(f"[GVAR-INFER] complete={complete}, {count}/{len(indices)} frames", flush=True)


if __name__ == "__main__":
    main()
