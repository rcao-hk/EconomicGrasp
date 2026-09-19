#!/usr/bin/env python3
"""Mine Rep-P0 fixed-action geometry-source cache.

For each RGB-only Stage-1 native query:
  1) decode one complete grasp action;
  2) create same-ray translation hypotheses while preserving R/width/depth;
  3) label every valid physical action with the exact CAD/DexNet evaluator;
  4) extract the same gripper-centric descriptor from pred/sensor/rendered/CAD
     geometry.

The geometry source never changes the action being evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--split", default="train", choices=("train","test_seen","test_similar","test_novel"))
    p.add_argument("--camera", default="realsense")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--query_eval_num", type=int, default=64)
    p.add_argument("--query_eval_mode", default="topk_uniform", choices=("all","topk","uniform","topk_uniform"))
    p.add_argument("--offsets_mm", default="-40,-20,-10,0,10,20,40")
    p.add_argument("--sources", default="pred,sensor,rendered,cad_full")
    p.add_argument("--voxel_size", type=float, default=0.008)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fc_mode", default="reuse_contacts", choices=("reuse_contacts","official"))
    p.add_argument("--verify_n", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=20)
    return p.parse_args()


ARGS = parse_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from rep_p0_geometry_common import (
    DescriptorConfig,
    GEOMETRY_SOURCES,
    backproject_depth_map,
    build_translation_ray_actions,
    describe_actions,
    friction_utility,
    load_full_cad_scene_cloud,
    parse_offsets_mm,
    select_query_indices,
    zero_offset_index,
)
from utils.arguments import cfgs


def subset_indices(total: int, interval: float):
    if interval <= 0:
        raise ValueError("sample_interval must be positive.")
    stride = max(1, int(round(1.0 / float(interval))))
    out = []
    for start in range(0, total, 256):
        out.extend(range(start, min(start + 256, total), stride))
    return out


def move_batch(batch, device):
    for key in ("point_clouds","cloud_colors","coordinates_for_voxel"):
        batch.pop(key, None)
    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(f"Unexpected list-valued key {key!r}; use load_label=False.")
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def load_checkpoint_model(path: str, device):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError("Rep-P0 requires a full EconomicGrasp checkpoint with model_state_dict.")
    source = str(ckpt.get("geometry_depth_source", "pred"))
    if source not in ("", "pred"):
        raise RuntimeError(
            f"Rep-P0 action generator must be RGB-predicted geometry; checkpoint source={source!r}."
        )
    pose_mode = str(ckpt.get("pose_depth_mode", "global_film"))
    use_fuse_depth = bool(ckpt.get("use_fuse_depth", False))

    cfgs.use_top4_view_infer = False
    cfgs.kview_mode = "A1"
    cfgs.kview_k = 1
    cfgs.use_cdf = True
    cfgs.use_obs_depth = False
    cfgs.pose_depth_mode = pose_mode

    model = economicgrasp_dpt_student(
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=pose_mode,
        camera_pose_key=str(ckpt.get("camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(ckpt.get("camera_gravity_key", "camera_gravity_vec")),
        pose_hidden_dim=int(ckpt.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(ckpt.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(ckpt.get("ray_gravity_mid_dim", 32)),
        use_cdf=True,
        vis_dir=None,
    ).to(device)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optional = ("rgb_geometry_diagnostics.",)
    missing = [k for k in result.missing_keys if not k.startswith(optional)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(optional)]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return ckpt, model, pose_mode, use_fuse_depth


def evaluate_actions(evaluator, scene_id, anno_id, actions, valid):
    K, Q, _ = actions.shape
    friction = np.full((K, Q), np.nan, dtype=np.float32)
    assigned = np.full((K, Q), -1, dtype=np.int64)
    collision = np.full((K, Q), -1, dtype=np.int8)
    pure = np.full((K, Q), -1, dtype=np.int8)
    empty = np.full((K, Q), -1, dtype=np.int8)
    flat_valid = valid.reshape(-1)
    ids = np.flatnonzero(flat_valid)
    if len(ids):
        res = evaluator.evaluate(scene_id, anno_id, actions.reshape(-1, 17)[ids])
        friction.reshape(-1)[ids] = res.friction
        assigned.reshape(-1)[ids] = res.assigned_obj
        collision.reshape(-1)[ids] = res.collision_or_empty.astype(np.int8)
        pure.reshape(-1)[ids] = res.pure_collision.astype(np.int8)
        empty.reshape(-1)[ids] = res.empty.astype(np.int8)
        return friction, assigned, collision, pure, empty, res.stats
    return friction, assigned, collision, pure, empty, {}


def main():
    sources = tuple(x.strip() for x in ARGS.sources.split(",") if x.strip())
    unknown = sorted(set(sources) - set(GEOMETRY_SOURCES))
    if unknown:
        raise ValueError(f"Unknown geometry source(s): {unknown}; supported={GEOMETRY_SOURCES}")
    if not sources:
        raise ValueError("At least one source is required.")

    offsets = parse_offsets_mm(ARGS.offsets_mm)
    zidx = zero_offset_index(offsets)
    if ARGS.num_shards < 1 or not (0 <= ARGS.shard_id < ARGS.num_shards):
        raise ValueError("Invalid shard_id/num_shards.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint, model, pose_mode, use_fuse_depth = load_checkpoint_model(
        ARGS.checkpoint_path, device
    )

    dataset = GraspNetMultiDataset(
        ARGS.dataset_root,
        split=ARGS.split,
        camera=ARGS.camera,
        num_points=20000,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=use_fuse_depth,
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
    )
    indices = subset_indices(len(dataset), ARGS.sample_interval)
    indices = [idx for pos, idx in enumerate(indices) if pos % ARGS.num_shards == ARGS.shard_id]
    if ARGS.max_samples > 0:
        indices = indices[:ARGS.max_samples]

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=ARGS.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(ARGS.num_workers > 0),
    )
    evaluator = ExactGraspNetActionEvaluator(
        ARGS.dataset_root,
        ARGS.camera,
        split=ARGS.split,
        fc_mode=ARGS.fc_mode,
        verify_n=ARGS.verify_n,
        strict=True,
    )
    desc_cfg = DescriptorConfig(voxel_size=ARGS.voxel_size)

    out_root = Path(ARGS.output_root) / ARGS.split
    out_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "Rep-P0 fixed-action geometry-source diagnosis",
        "split": ARGS.split,
        "camera": ARGS.camera,
        "checkpoint": os.path.abspath(ARGS.checkpoint_path),
        "sample_interval": ARGS.sample_interval,
        "query_eval_num": ARGS.query_eval_num,
        "query_eval_mode": ARGS.query_eval_mode,
        "offsets_mm": offsets.tolist(),
        "zero_index": zidx,
        "action_contract": "native R/width/height/insertion-depth fixed; translation shifts on same camera ray",
        "sources": list(sources),
        "voxel_size": ARGS.voxel_size,
        "descriptor_feature_dim": desc_cfg.feature_dim,
        "descriptor_hist_bins": list(desc_cfg.hist_bins),
        "pose_depth_mode": pose_mode,
        "use_fuse_depth_for_rendered_target": use_fuse_depth,
        "shard_id": ARGS.shard_id,
        "num_shards": ARGS.num_shards,
    }
    with (out_root / f"protocol_shard_{ARGS.shard_id:02d}.json").open("w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)

    processed = 0
    total_actions = 0
    total_eval_sec = 0.0
    start_all = time.perf_counter()

    for local_i, batch in enumerate(loader):
        batch = move_batch(batch, device)
        scene_id = int(batch["scene_idx"].reshape(-1)[0].item())
        anno_id = int(batch["anno_idx"].reshape(-1)[0].item())
        out_dir = out_root / f"scene_{scene_id:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ann_{anno_id:04d}.npz"
        if out_path.exists() and not ARGS.overwrite:
            continue

        batch["cva_export_angle_feature"] = False
        batch["cva_compute_diagnostics"] = False
        batch["geometry_compute_diagnostics"] = False
        with torch.inference_mode():
            ep = model(batch)
            native_all = pred_decode_center_view_angle(ep, use_cdf=True)[0]
        qidx = select_query_indices(native_all, ARGS.query_eval_num, ARGS.query_eval_mode)
        native = native_all.index_select(0, qidx).detach().cpu().numpy().astype(np.float32)
        actions, valid = build_translation_ray_actions(
            native,
            offsets,
            min_depth=ARGS.min_depth,
            max_depth=ARGS.max_depth,
        )
        if not np.all(valid[zidx]):
            raise RuntimeError("Native candidate became invalid in Rep-P0 action generation.")

        t_eval = time.perf_counter()
        friction, assigned, collision, pure, empty, eval_stats = evaluate_actions(
            evaluator, scene_id, anno_id, actions, valid
        )
        total_eval_sec += time.perf_counter() - t_eval
        utility = friction_utility(friction)

        Kcam = batch["K"][0].detach().cpu().numpy().astype(np.float32)
        source_points = {}
        if "pred" in sources:
            pred_depth = ep["depth_map_used_for_geometry"][0]
            if pred_depth.dim() == 3:
                pred_depth = pred_depth[0]
            source_points["pred"] = backproject_depth_map(
                pred_depth.detach().cpu().numpy(),
                Kcam,
                voxel_size=ARGS.voxel_size,
                min_depth=ARGS.min_depth,
                max_depth=ARGS.max_depth,
            )
        if "sensor" in sources:
            source_points["sensor"] = backproject_depth_map(
                batch["sensor_depth_m"][0].detach().cpu().numpy(),
                Kcam,
                voxel_size=ARGS.voxel_size,
                min_depth=0.05,
                max_depth=2.0,
            )
        if "rendered" in sources:
            source_points["rendered"] = backproject_depth_map(
                batch["gt_depth_m"][0].detach().cpu().numpy(),
                Kcam,
                voxel_size=ARGS.voxel_size,
                min_depth=0.05,
                max_depth=2.0,
            )
        if "cad_full" in sources:
            source_points["cad_full"] = load_full_cad_scene_cloud(
                evaluator, scene_id, anno_id, ARGS.voxel_size
            )

        payload = {
            "actions": actions.astype(np.float32),
            "valid": valid.astype(np.uint8),
            "friction": friction.astype(np.float32),
            "utility": utility.astype(np.float32),
            "assigned_obj": assigned.astype(np.int16),
            "collision_or_empty": collision.astype(np.int8),
            "pure_collision": pure.astype(np.int8),
            "empty": empty.astype(np.int8),
            "offsets_mm": offsets.astype(np.float32),
            "zero_index": np.asarray(zidx, dtype=np.int16),
            "scene_id": np.asarray(scene_id, dtype=np.int16),
            "anno_id": np.asarray(anno_id, dtype=np.int16),
            "query_ids": qidx.detach().cpu().numpy().astype(np.int16),
            "native_score": native[:, 0].astype(np.float32),
        }
        for source in sources:
            feat = describe_actions(source_points[source], actions, valid, desc_cfg)
            payload[f"feat_{source}"] = feat.astype(np.float16)
            payload[f"points_{source}"] = np.asarray(len(source_points[source]), dtype=np.int32)

        np.savez_compressed(out_path, **payload)
        processed += 1
        total_actions += int(valid.sum())

        if ARGS.progress_every > 0 and processed % ARGS.progress_every == 0:
            elapsed = time.perf_counter() - start_all
            print(
                f"[REP-P0-MINE] split={ARGS.split} shard={ARGS.shard_id}/{ARGS.num_shards} "
                f"frames={processed} scene={scene_id:04d} ann={anno_id:04d} "
                f"valid_actions={total_actions} elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = {
        **protocol,
        "processed_frames": processed,
        "valid_actions": total_actions,
        "exact_eval_sec": total_eval_sec,
        "wall_sec": time.perf_counter() - start_all,
    }
    with (out_root / f"summary_shard_{ARGS.shard_id:02d}.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
