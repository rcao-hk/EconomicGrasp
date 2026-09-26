#!/usr/bin/env python3
"""Predict online metric-grasp-field grasps; no P0/P1 cache or sensor filtering."""
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
    full, _, indices = make_dataset(args.dataset_root, args.split, protocol["sample_fraction"],
                                     labels=False, max_frames=args.max_frames)
    schedule = dataset_schedule(full, indices)
    root = Path(args.output_root)
    run = {"version": VERSION, "checkpoint_sha256": sha256_file(args.checkpoint),
           "checkpoint_epoch": epoch, "training_protocol": protocol,
           "code_sha256": code_fingerprint(), "split": args.split,
           "schedule": schedule, "top4": args.top4,
           "max_frames": args.max_frames, "collision_filter": "none",
           "prediction_modalities": "RGB + camera metadata; depth internally predicted",
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
    for raw in loader:
        batch = move_batch(raw, device, inference=True)
        ep = model(batch)
        predictions = pred_decode_center_view_angle(ep, use_cdf=True)
        for pred in predictions:
            data_idx = pending[done]
            sid, aid = int(names[data_idx].split("_")[-1]), data_idx % 256
            array = pred.detach().float().cpu().numpy()
            if array.ndim != 2 or array.shape[-1] != 17 or not np.isfinite(array).all():
                raise RuntimeError(f"Invalid decoded grasps for {sid}/{aid}")
            path = root/"dump"/f"scene_{sid:04d}"/"realsense"/f"{aid:04d}.npy"
            atomic_npy(path, array)
            atomic_json(root/args.split/"completed"/f"{sid:04d}_{aid:04d}.json",
                        {"signature": sig, "output_sha256": sha256_file(path), "grasps": len(array)})
            done += 1
        if done % 20 < args.batch_size:
            print(f"[MGF INFER] {args.split} shard={args.shard_id} done={done}/{len(pending)} skipped={skipped}", flush=True)
    atomic_json(root/args.split/f"shard_{args.shard_id}.json",
                {"signature": sig, "written": done, "skipped": skipped, "assigned": len(owned)})
    lock.close()


if __name__ == "__main__":
    main()
