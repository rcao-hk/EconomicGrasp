#!/usr/bin/env python3
"""Predict online metric-grasp-field grasps with optional sensor-cloud collision filtering."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from metric_field_runtime import (
    VERSION, atomic_json, code_fingerprint, construct_model, dataset_schedule,
    digest, ensure_manifest, make_dataset, move_batch, seed_all, sha256_file, worker_init,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="/data/robotarm/dataset/graspnet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--split", choices=("test_seen", "test_similar", "test_novel"), required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--top4", action="store_true")
    p.add_argument(
        "--score-source",
        choices=("field", "base", "blend"),
        default="field",
        help=(
            "CDF scorer used for final angle-depth decoding. 'field' uses the "
            "Metric Grasp Field output; 'base' restores Base CVA-CDF logits; "
            "'blend' interpolates Base/Field logits."
        ),
    )
    p.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help=(
            "For --score-source blend: final=(1-alpha)*base + alpha*field. "
            "Use alpha in [0,1]."
        ),
    )
    p.add_argument(
        "--collision-thresh",
        type=float,
        default=0.0,
        help=(
            "Model-free collision IoU threshold. <=0 disables collision "
            "filtering. Positive values use the original GraspNet sensor cloud "
            "as a post-hoc system-level filter."
        ),
    )
    p.add_argument(
        "--collision-voxel-size",
        type=float,
        default=0.01,
        help="Voxel size (m) for the model-free sensor-cloud collision detector.",
    )
    p.add_argument(
        "--collision-approach-dist",
        type=float,
        default=0.05,
        help="Collision-free approach distance (m) before the grasp pose.",
    )
    p.add_argument("--resume", action="store_true")
    return p


def atomic_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


@torch.no_grad()
def main():
    args = parser().parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.max_frames < 0:
        raise ValueError("Invalid batch/worker/frame setting")
    if args.collision_thresh < 0:
        raise ValueError("--collision-thresh must be >=0")
    if not np.isfinite(args.blend_alpha) or not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError("--blend-alpha must lie in [0,1]")
    if args.collision_voxel_size <= 0 or args.collision_approach_dist <= 0:
        raise ValueError(
            "--collision-voxel-size and --collision-approach-dist must be >0"
        )
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard setting")
    if not torch.cuda.is_available():
        raise RuntimeError("Online inference needs the main CUDA/GraspNet environment")
    device = torch.device("cuda:0")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("version") != VERSION:
        raise ValueError("Not a DAV2 metric-grasp-field checkpoint")
    protocol, state = ck["protocol"], ck["model"]
    epoch = int(ck["epoch"])
    del ck
    official = Path("checkpoints") / f"depth_anything_v2_{protocol['encoder']}.pth"
    if sha256_file(official) != protocol["dav2_sha256"]:
        raise RuntimeError("Official DAV2 checkpoint differs from training")
    seed_all(protocol["seed"])
    model = construct_model(protocol, device=device, top4=args.top4)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    from dataset.graspnet_dataset import collate_fn
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    use_collision_filter = args.collision_thresh > 0
    if use_collision_filter:
        from graspnetAPI import GraspGroup
        from utils.collision_detector import ModelFreeCollisionDetectorTorch
    eval_fraction = float(
        protocol.get("eval_fraction", protocol["sample_fraction"])
    )
    full, _, indices = make_dataset(
        args.dataset_root,
        args.split,
        eval_fraction,
        labels=False,
        max_frames=args.max_frames,
    )
    schedule = dataset_schedule(full, indices)
    root = Path(args.output_root)
    collision_filter = (
        {
            "type": "model_free_original_sensor",
            "threshold": float(args.collision_thresh),
            "voxel_size_m": float(args.collision_voxel_size),
            "approach_dist_m": float(args.collision_approach_dist),
            "network_input": False,
        }
        if use_collision_filter
        else "none"
    )
    run = {"version": VERSION, "checkpoint_sha256": sha256_file(args.checkpoint),
           "checkpoint_epoch": epoch, "training_protocol": protocol,
           "code_sha256": code_fingerprint(), "split": args.split,
           "schedule": schedule, "top4": args.top4,
           "evaluation_fraction": eval_fraction,
           "score_source": args.score_source,
           "blend_alpha": (
               float(args.blend_alpha)
               if args.score_source == "blend"
               else None
           ),
           "max_frames": args.max_frames, "collision_filter": collision_filter,
           "prediction_modalities": (
               "RGB + camera metadata; depth internally predicted; "
               + ("original sensor depth used only for post-hoc collision filtering"
                  if use_collision_filter else "no sensor-depth post-filter")
           ),
           "preprocessing": "unchanged main GraspNetMultiDataset crop/workspace protocol"}
    sig = digest(run)
    manifest = root/args.split/"protocol.json"
    ensure_manifest(manifest, run)
    owned = [(idx, pair) for pos, (idx, pair) in enumerate(zip(indices, schedule))
             if pos % args.num_shards == args.shard_id]
    # A shard lock prevents concurrent launchers from processing the same work.
    import fcntl
    lock = open(root/args.split/f".shard_{args.shard_id}.lock", "a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, skipped = [], 0
    for idx, (sid, aid) in owned:
        path = root/"dump"/f"scene_{sid:04d}"/"realsense"/f"{aid:04d}.npy"
        marker = root/args.split/"completed"/f"{sid:04d}_{aid:04d}.json"
        if not args.resume and (path.exists() or marker.exists()):
            raise FileExistsError(f"Existing output {path}; use --resume")
        if args.resume and marker.is_file():
            m = json.loads(marker.read_text())
            if m["signature"] != sig:
                raise RuntimeError(f"Stale marker {marker}")
            if path.is_file() and sha256_file(path) == m["output_sha256"]:
                skipped += 1
                continue
        pending.append(idx)
    loader = DataLoader(Subset(full, pending), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn,
                        worker_init_fn=worker_init, pin_memory=False)
    names = full.scene_list()
    done = 0
    grasps_before_filter = 0
    grasps_after_filter = 0
    for raw in loader:
        batch = move_batch(raw, device, inference=True)
        ep = model(batch)
        if args.score_source in ("base", "blend"):
            base_logits = ep.get("mgf_base_cdf_logits")
            field_logits = ep.get("grasp_cdf_pred_angle_depth")
            if base_logits is None or field_logits is None:
                raise RuntimeError(
                    "Base/blend decode requested but field/base CDF logits are missing"
                )
            if tuple(base_logits.shape) != tuple(field_logits.shape):
                raise RuntimeError(
                    "Base/field CDF shape mismatch: "
                    f"base={tuple(base_logits.shape)} "
                    f"field={tuple(field_logits.shape)}"
                )
            # Diagnostic intervention only: preserve the SAME centers, selected
            # views, widths, metric depth, and checkpoint; change only the
            # final CDF tensor consumed by pred_decode_center_view_angle().
            if args.score_source == "base":
                ep["grasp_cdf_pred_angle_depth"] = base_logits
            else:
                alpha = float(args.blend_alpha)
                ep["grasp_cdf_pred_angle_depth"] = (
                    (1.0 - alpha) * base_logits + alpha * field_logits
                )
        predictions = pred_decode_center_view_angle(ep, use_cdf=True)
        for pred in predictions:
            data_idx = pending[done]
            sid, aid = int(names[data_idx].split("_")[-1]), data_idx % 256
            array = pred.detach().float().cpu().numpy()
            if array.ndim != 2 or array.shape[-1] != 17 or not np.isfinite(array).all():
                raise RuntimeError(f"Invalid decoded grasps for {sid}/{aid}")

            before = int(len(array))
            if use_collision_filter and before > 0:
                # Match the historical EconomicGrasp CVA evaluation protocol:
                # the RGB network remains unchanged, while a model-free
                # detector uses the paired original GraspNet sensor point cloud
                # only as a post-hoc system-level filter.
                cloud, _ = full.get_data(data_idx, return_raw_cloud=True)
                gg = GraspGroup(array)
                detector = ModelFreeCollisionDetectorTorch(
                    np.asarray(cloud, dtype=np.float32).reshape(-1, 3),
                    voxel_size=args.collision_voxel_size,
                )
                collision = detector.detect(
                    gg,
                    approach_dist=args.collision_approach_dist,
                    collision_thresh=args.collision_thresh,
                )
                keep = ~collision.detach().cpu().numpy()
                array = gg[keep].grasp_group_array.astype(np.float32, copy=False)

            after = int(len(array))
            grasps_before_filter += before
            grasps_after_filter += after

            path = root/"dump"/f"scene_{sid:04d}"/"realsense"/f"{aid:04d}.npy"
            atomic_npy(path, array)
            atomic_json(
                root/args.split/"completed"/f"{sid:04d}_{aid:04d}.json",
                {
                    "signature": sig,
                    "output_sha256": sha256_file(path),
                    "grasps": after,
                    "grasps_before_collision": before,
                    "grasps_after_collision": after,
                    "collision_rejected": before - after,
                },
            )
            done += 1
        if done % 20 < args.batch_size:
            print(f"[MGF INFER] {args.split} shard={args.shard_id} done={done}/{len(pending)} skipped={skipped}", flush=True)
    retention = (
        float(grasps_after_filter) / float(grasps_before_filter)
        if grasps_before_filter > 0 else 1.0
    )
    atomic_json(
        root/args.split/f"shard_{args.shard_id}.json",
        {
            "signature": sig,
            "written": done,
            "skipped": skipped,
            "assigned": len(owned),
            "grasps_before_collision": grasps_before_filter,
            "grasps_after_collision": grasps_after_filter,
            "collision_retention_ratio": retention,
        },
    )
    lock.close()


if __name__ == "__main__":
    main()
