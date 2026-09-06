#!/usr/bin/env python3
"""Run the existing CVA-CDF RGB student inference for matched depth controls.

Each variant/split has its own dump and manifest. Repeating a command skips
complete matching dumps; an interrupted split is recomputed in full. Different
checkpoints/protocols must use a different prediction_root.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from cva_depth_evaluation import (CONTRACT_VERSION, add_selection_arguments, annotation_ids,
                                  check_dumps, read_json, sha256_file, write_json)


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--base_checkpoint", help="The original Stage-1/2 checkpoint used to initialize all arms.")
    parser.add_argument("--controls_dir", help="Training group directory containing none/, foreground/, anchor/.")
    parser.add_argument("--checkpoint_name", default="checkpoint.tar", help="Same filename in every trained arm.")
    parser.add_argument("--camera", choices=("realsense", "kinect"), default="realsense")
    parser.add_argument("--frame_stride", type=int, default=1, help="Annotation IDs 0,K,2K,...; 1 = full benchmark.")
    parser.add_argument("--topk_views", type=int, choices=(1, 4), default=1)
    parser.add_argument("--collision_thresh", type=float, default=0.0,
                        help="0 disables captured-cloud collision postprocessing; preserve your comparison protocol.")
    parser.add_argument("--collision_voxel_size", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m_point", type=int, default=1024)
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--graspness_threshold", type=float, default=0.1)
    parser.add_argument("--graspness_mode", default="scene")
    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=1.0)
    parser.add_argument("--bin_num", type=int, default=256)
    parser.add_argument("--dry_run", action="store_true", help="Validate checkpoints and print child commands without inference.")
    args = parser.parse_args(argv)
    if "base" in args.variants and not args.base_checkpoint:
        parser.error("--base_checkpoint is required when variants includes base.")
    if any(v != "base" for v in args.variants) and not args.controls_dir:
        parser.error("--controls_dir is required for trained variants.")
    if Path(args.checkpoint_name).name != args.checkpoint_name or args.checkpoint_name in (".", ".."):
        parser.error("--checkpoint_name must be a filename inside each variant directory.")
    if (not 1 <= args.frame_stride <= 256 or min(args.batch_size, args.m_point, args.num_point, args.bin_num) < 1
            or args.num_workers < 0 or not 0 <= args.seed < 2**32):
        parser.error("Invalid frame_stride, batch/point/bin count, num_workers or seed.")
    floats = (args.min_depth, args.max_depth, args.collision_thresh, args.collision_voxel_size, args.graspness_threshold)
    if (not all(math.isfinite(v) for v in floats) or not 0 < args.min_depth < args.max_depth
            or args.collision_thresh < 0 or args.collision_voxel_size <= 0 or args.graspness_threshold < 0):
        parser.error("Invalid depth bounds, collision parameters or graspness_threshold.")
    return args


def checkpoint_record(path, variant, args):
    import torch

    path = Path(path).resolve()
    before = path.stat()
    # Checkpoints are the user's full training snapshots, including metadata.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"distill_contract_version": 2, "geometry_depth_source": "pred", "seed_selection_mode": "image_fps",
                "depth_head_executed": True, "legacy_dataset_use_gt_depth": False}
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Expected a full RGB student checkpoint: {path}")
    if checkpoint.get("distill_stage") not in (1, 2) or any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Checkpoint is not a contract-v2 RGB/pred-depth/image-FPS student: {path}")
    if checkpoint["distill_stage"] == 2 and checkpoint.get("teacher_geometry_depth_source") != "gt":
        raise ValueError(f"Stage-2 checkpoint does not record a clean-depth teacher: {path}")
    required = ("pose_depth_mode", "use_fuse_depth")
    if any(k not in checkpoint for k in required):
        raise ValueError(f"Missing pose/fuse-depth metadata: {path}")
    trained = checkpoint.get("depth_geometry_args", {})
    if variant != "base" and (checkpoint.get("depth_geometry_contract_version") != 1 or trained.get("variant") != variant):
        raise ValueError(f"Checkpoint does not belong to the {variant} depth control: {path}")
    for key in ("camera", "min_depth", "max_depth", "bin_num", "m_point", "graspness_mode"):
        if key in trained and trained[key] != getattr(args, key):
            raise ValueError(f"{path}: trained {key}={trained[key]!r} differs from inference {getattr(args, key)!r}.")
    for key in ("min_depth", "max_depth"):
        saved = checkpoint.get("distillation_config", {}).get(key)
        if saved is not None and saved != getattr(args, key):
            raise ValueError(f"{path}: saved {key}={saved} differs from inference {getattr(args, key)}.")
    contract = {k: checkpoint[k] for k in required}
    for key, default in (("pose_hidden_dim", 64), ("ray_gravity_hidden_dim", 64), ("ray_gravity_mid_dim", 32),
                         ("camera_pose_key", "camera_pose_vec"), ("camera_gravity_key", "camera_gravity_vec")):
        contract[key] = checkpoint.get(key, default)
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Checkpoint changed during preflight: {path}. Wait for training or use a saved epoch file.")
    return {"path": str(path), "sha256": digest, "input_contract": contract,
            "distill_stage": checkpoint["distill_stage"], "epoch": checkpoint.get("epoch"),
            "training_args": trained, "training_source_revision": checkpoint.get("source_revision")}


def prediction_protocol(args, input_contract):
    keys = ("camera", "frame_stride", "topk_views", "collision_thresh", "collision_voxel_size", "batch_size",
            "num_workers", "seed", "m_point", "num_point", "graspness_threshold", "graspness_mode",
            "min_depth", "max_depth", "bin_num")
    return {**{key: getattr(args, key) for key in keys}, "dataset_root": str(Path(args.dataset_root).resolve()),
            "input_contract": input_contract, "use_cdf": True, "extend_angle": True, "kview_mode": "A1",
            "geometry_depth_source": "pred", "seed_selection_mode": "image_fps", "grasp_max_width": 0.1,
            "num_view": 300, "num_angle": 12, "num_depth": 4, "pre_eval_nms": False}


def inference_command(args, record, split, dump_dir):
    command = [sys.executable, "-u", str(ROOT / "inference_cva_distill.py"),
               "--checkpoint_path", record["path"], "--save_dir", str(dump_dir), "--test_mode", split,
               "--distill_stage", str(record["distill_stage"]), "--multi_modal", "--inference", "--use_cdf",
               "--extend_angle", "--kview_mode", "A1", "--sample_interval", repr(1.0 / args.frame_stride),
               "--grasp_max_width", "0.1", "--num_view", "300", "--num_angle", "12", "--num_depth", "4"]
    for key in ("dataset_root", "camera", "collision_thresh", "collision_voxel_size", "batch_size", "num_workers",
                "seed", "m_point", "num_point", "graspness_threshold", "graspness_mode", "min_depth", "max_depth", "bin_num"):
        command.extend((f"--{key}", str(getattr(args, key))))
    if record["input_contract"]["use_fuse_depth"]:
        command.append("--use_fuse_depth")
    if args.topk_views == 4:
        command.append("--use_top4_view_infer")
    return command


def run_logged(command, path):
    with Path(path).open("a", encoding="utf-8") as log:
        log.write(f"\n[COMMAND] {shlex.join(command)}\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def assert_checkpoint_unchanged(record):
    if sha256_file(record["path"]) != record["sha256"]:
        raise ValueError(f"Checkpoint changed during inference: {record['path']}. "
                         "Wait for training or select an immutable epoch file and a new prediction_root.")


def main(argv=None):
    args = parse_args(argv)
    args.dataset_root = str(Path(args.dataset_root).resolve())
    records = {}
    for variant in args.variants:
        path = args.base_checkpoint if variant == "base" else Path(args.controls_dir) / variant / args.checkpoint_name
        records[variant] = checkpoint_record(path, variant, args)
    reference = records[args.variants[0]]["input_contract"]
    if any(record["input_contract"] != reference for record in records.values()):
        raise ValueError("Compared checkpoints have different pose/fuse-depth input contracts.")
    trained = [record for name, record in records.items() if name != "base"]
    if len({record["training_args"].get("train_scope") for record in trained}) > 1:
        raise ValueError("Compared controls use different train_scope values.")
    if len({record["epoch"] for record in trained}) > 1:
        print("[NOTE] Controls are from different epochs (e.g. best_geometry.tar); do not call this equal-budget comparison.", flush=True)
    protocol = prediction_protocol(args, reference)
    code_hashes = {name: sha256_file(ROOT / name) for name in (
        "inference_cva_depth_controls.py", "inference_cva_distill.py", "cva_depth_evaluation.py",
        "models/economicgrasp_dpt_distill.py", "models/economicgrasp_bip3d.py", "models/kview_query_transformer.py",
        "dataset/graspnet_dataset.py", "utils/arguments.py", "utils/collision_detector.py")}
    jobs = []
    # Preflight every destination before the first GPU job.
    for variant, record in records.items():
        for split in args.splits:
            dump_dir = Path(args.prediction_root).resolve() / variant / split
            manifest_path = dump_dir / "inference_manifest.json"
            identity = {"contract_version": CONTRACT_VERSION, "checkpoint_sha256": record["sha256"],
                        "protocol": protocol, "split": split, "inference_code_sha256": code_hashes}
            previous = read_json(manifest_path) if manifest_path.is_file() else None
            if previous and (previous.get("identity") != identity or previous.get("variant") != variant):
                raise ValueError(f"Different checkpoint/protocol/code already occupies {dump_dir}. Use a new prediction_root.")
            if not previous and dump_dir.exists() and any(dump_dir.iterdir()):
                raise ValueError(f"Nonempty dump directory has no manifest: {dump_dir}. Use a new prediction_root.")
            jobs.append((variant, split, record, dump_dir, identity, previous))
    for variant, split, record, dump_dir, identity, previous in jobs:
        command = inference_command(args, record, split, dump_dir)
        print(f"[DEPTH INFER] {variant}/{split}: {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        assert_checkpoint_unchanged(record)
        if previous and previous.get("status") == "complete":
            coverage = check_dumps(dump_dir, split, args.camera, args.frame_stride)
            if coverage != previous.get("coverage"):
                raise ValueError(f"Completed prediction files changed: {dump_dir}. Use a new prediction_root.")
            print(f"[SKIP] Complete matching inference: {dump_dir}", flush=True)
            continue
        manifest = {"variant": variant, "identity": identity, "checkpoint": record, "status": "running",
                    "command": command, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "annotation_ids": annotation_ids(args.frame_stride)}
        write_json(dump_dir / "inference_manifest.json", manifest)
        start = time.perf_counter()
        run_logged(command, dump_dir / "inference.log")
        assert_checkpoint_unchanged(record)
        manifest.update(status="complete", elapsed_seconds=time.perf_counter() - start,
                        coverage=check_dumps(dump_dir, split, args.camera, args.frame_stride))
        write_json(dump_dir / "inference_manifest.json", manifest)


if __name__ == "__main__":
    main()
