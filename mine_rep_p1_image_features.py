#!/usr/bin/env python3
"""Cache frozen pre-enhancer image feature maps for Rep-P1.

The exact K=7 actions and CAD/DexNet labels are never regenerated here.  This
script reads the formal Rep-P0 cache, extracts one frozen image feature map per
RGB frame from the same Stage-1 checkpoint, and binds the feature cache to the
Rep-P0 physical-action tensor with a SHA256 digest.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from rep_p1_common import (
    IMAGE_FEATURE_SOURCE,
    REP_P1_VERSION,
    action_digest,
    cache_paths,
    extract_pre_enhancer_feature,
    load_p0_frame,
    load_stage1_reference,
    sha256_file,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--stage1_checkpoint", required=True)
    p.add_argument("--p0_cache_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument(
        "--split",
        required=True,
        choices=("train", "test_seen", "test_similar", "test_novel"),
    )
    p.add_argument("--camera", default="realsense")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--repair_invalid_cache", action="store_true")
    p.add_argument("--progress_every", type=int, default=50)
    return p.parse_args()


def atomic_save_npz(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def move_batch(batch, device):
    for key in ("point_clouds", "cloud_colors", "coordinates_for_voxel"):
        batch.pop(key, None)
    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Unexpected list-valued dataset field {key!r}; use load_label=False"
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def validate_cache(path: Path, p0, stage1_sha: str):
    try:
        with np.load(path, allow_pickle=False) as d:
            required = {
                "pre_feature", "K", "image_hw", "scene_id", "anno_id",
                "action_digest", "feature_source", "stage1_sha256",
                "version",
            }
            missing = sorted(required - set(d.files))
            if missing:
                return False, f"missing {missing}"
            if str(np.asarray(d["version"]).reshape(-1)[0]) != REP_P1_VERSION:
                return False, "version mismatch"
            if str(np.asarray(d["feature_source"]).reshape(-1)[0]) != IMAGE_FEATURE_SOURCE:
                return False, "feature source mismatch"
            if str(np.asarray(d["stage1_sha256"]).reshape(-1)[0]) != stage1_sha:
                return False, "Stage-1 checkpoint mismatch"
            scene = int(np.asarray(d["scene_id"]).reshape(-1)[0])
            anno = int(np.asarray(d["anno_id"]).reshape(-1)[0])
            if (scene, anno) != (p0["scene_id"], p0["anno_id"]):
                return False, "scene/anno mismatch"
            expected = action_digest(p0["actions"], p0["valid"], p0["offsets"])
            if str(np.asarray(d["action_digest"]).reshape(-1)[0]) != expected:
                return False, "physical-action digest mismatch"
            feat = np.asarray(d["pre_feature"])
            K = np.asarray(d["K"])
            hw = np.asarray(d["image_hw"])
            if feat.ndim != 3 or K.shape != (3, 3) or hw.shape != (2,):
                return False, "malformed feature/K/image_hw"
            if not np.isfinite(feat).all() or not np.isfinite(K).all():
                return False, "non-finite cache"
        return True, "validated"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.repair_invalid_cache and not args.resume:
        raise ValueError("--repair_invalid_cache requires --resume")
    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise ValueError("Invalid shard configuration")

    p0_split = Path(args.p0_cache_root) / args.split
    paths = cache_paths(p0_split)
    if not paths:
        raise RuntimeError(f"No Rep-P0 cache files under {p0_split}")

    # Scene-level sharding preserves locality and avoids assigning one scene to
    # multiple feature miners.
    scene_ids = sorted({
        int(p.parent.name.split("_")[-1])
        for p in paths
    })
    owned = {
        scene for pos, scene in enumerate(scene_ids)
        if pos % args.num_shards == args.shard_id
    }
    paths = [
        p for p in paths
        if int(p.parent.name.split("_")[-1]) in owned
    ]
    if args.max_samples > 0:
        paths = paths[: args.max_samples]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stage1_sha = sha256_file(args.stage1_checkpoint)
    reference, meta = load_stage1_reference(
        args.stage1_checkpoint,
        device,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        bin_num=args.bin_num,
    )

    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn

    dataset = GraspNetMultiDataset(
        args.dataset_root,
        split=args.split,
        camera=args.camera,
        num_points=1,
        remove_outlier=False,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=meta["use_fuse_depth"],
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        bin_num=args.bin_num,
    )
    scene_pos = {
        int(name.split("_")[-1]): i
        for i, name in enumerate(dataset.sceneIds)
    }

    out_split = Path(args.output_root) / args.split
    out_split.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": REP_P1_VERSION,
        "experiment": "Rep-P1 frozen pre-enhancer image feature cache",
        "split": args.split,
        "camera": args.camera,
        "feature_source": IMAGE_FEATURE_SOURCE,
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
        "stage1_sha256": stage1_sha,
        "p0_cache_root": str(p0_split.resolve()),
        "physical_action_contract":
            "exact Rep-P0 K=7 action tensor; image cache bound by action_digest",
        "sharding": "scene_level_modulo",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "owned_scenes": sorted(owned),
    }
    (out_split / f"protocol_shard_{args.shard_id:02d}.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True)
    )

    done = resumed = repaired = 0
    for i, p0_path in enumerate(paths):
        p0 = load_p0_frame(p0_path, need_geo=False)
        out = out_split / p0_path.relative_to(p0_split)
        if out.exists() and not args.overwrite:
            if args.resume:
                ok, reason = validate_cache(out, p0, stage1_sha)
                if ok:
                    resumed += 1
                    continue
                if not args.repair_invalid_cache:
                    raise RuntimeError(f"Invalid existing {out}: {reason}")
                repaired += 1
            else:
                raise FileExistsError(f"{out}; use --resume or --overwrite")

        if p0["scene_id"] not in scene_pos:
            raise RuntimeError(
                f"Scene {p0['scene_id']} absent from dataset split {args.split}"
            )
        dataset_idx = scene_pos[p0["scene_id"]] * 256 + p0["anno_id"]
        batch = collate_fn([dataset[dataset_idx]])
        batch = move_batch(batch, device)
        scene = int(batch["scene_idx"].reshape(-1)[0].item())
        anno = int(batch["anno_idx"].reshape(-1)[0].item())
        if (scene, anno) != (p0["scene_id"], p0["anno_id"]):
            raise RuntimeError(
                f"Dataset/P0 mismatch {scene}/{anno} vs "
                f"{p0['scene_id']}/{p0['anno_id']}"
            )

        feature = extract_pre_enhancer_feature(reference, batch)
        h, w = map(int, batch["img"].shape[-2:])
        payload = {
            "version": np.asarray(REP_P1_VERSION),
            "feature_source": np.asarray(IMAGE_FEATURE_SOURCE),
            "stage1_sha256": np.asarray(stage1_sha),
            "scene_id": np.asarray(scene, dtype=np.int16),
            "anno_id": np.asarray(anno, dtype=np.int16),
            "action_digest": np.asarray(
                action_digest(p0["actions"], p0["valid"], p0["offsets"])
            ),
            "pre_feature": feature[0].half().cpu().numpy(),
            "K": batch["K"][0].float().cpu().numpy(),
            "image_hw": np.asarray([h, w], dtype=np.int16),
        }
        atomic_save_npz(out, payload)
        done += 1
        del batch, feature, payload

        if args.progress_every > 0 and (done + resumed) % args.progress_every == 0:
            print(
                f"[REP-P1-CACHE] split={args.split} "
                f"shard={args.shard_id}/{args.num_shards} "
                f"done={done} resumed={resumed} repaired={repaired} "
                f"total={len(paths)}",
                flush=True,
            )

    summary = {
        **protocol,
        "selected_frames": len(paths),
        "written": done,
        "resumed": resumed,
        "repaired": repaired,
    }
    (out_split / f"summary_shard_{args.shard_id:02d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
