#!/usr/bin/env python3
"""Run the existing CVA-CDF RGB student inference for matched depth controls.

Each variant/split has its own dump and manifest. Repeating a command skips
complete matching dumps; an interrupted split is recomputed in full. Different
checkpoints/protocols must use a different prediction_root.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from cva_depth_evaluation import (CONTRACT_VERSION, add_selection_arguments, annotation_ids,
                                  check_dumps, read_json, sha256_file, write_json)


ROOT = Path(__file__).resolve().parent


def parse_gpu_ids(value):
    identifiers = tuple(part.strip() for part in value.split(","))
    if any(not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)", item) for item in identifiers):
        raise argparse.ArgumentTypeError("gpu_ids must contain CUDA device indices or GPU/MIG UUIDs, e.g. 1,2.")
    identifiers = tuple(str(int(item)) if item.isdigit() else item for item in identifiers)
    if len(set(identifiers)) != len(identifiers):
        raise argparse.ArgumentTypeError("gpu_ids must not contain duplicate devices.")
    return identifiers


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
    parser.add_argument("--gpu_ids", type=parse_gpu_ids,
                        help="One concurrent variant/split job per listed GPU, e.g. 1,2. Omit for legacy serial inference.")
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
    if args.gpu_ids is not None:
        command.append("--require_cuda")
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


def build_jobs(args):
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
            jobs.append({"variant": variant, "split": split, "checkpoint": record, "dump_dir": dump_dir,
                         "identity": identity, "previous": previous,
                         "command": inference_command(args, record, split, dump_dir)})
    return jobs


def prepare_job(args, job, gpu_id=None):
    record, dump_dir, previous = job["checkpoint"], job["dump_dir"], job["previous"]
    assert_checkpoint_unchanged(record)
    if previous and previous.get("status") == "complete":
        coverage = check_dumps(dump_dir, job["split"], args.camera, args.frame_stride)
        if coverage != previous.get("coverage"):
            raise ValueError(f"Completed prediction files changed: {dump_dir}. Use a new prediction_root.")
        print(f"[SKIP] Complete matching inference: {dump_dir}", flush=True)
        return None
    manifest = {"variant": job["variant"], "identity": job["identity"], "checkpoint": record, "status": "running",
                "command": job["command"], "cuda_visible_devices": gpu_id if gpu_id is not None else os.environ.get("CUDA_VISIBLE_DEVICES"),
                "annotation_ids": annotation_ids(args.frame_stride)}
    write_json(dump_dir / "inference_manifest.json", manifest)
    return {**job, "manifest": manifest, "started": time.perf_counter(), "log_path": dump_dir / "inference.log"}


def complete_job(args, state):
    assert_checkpoint_unchanged(state["checkpoint"])
    state["manifest"].update(status="complete", elapsed_seconds=time.perf_counter() - state["started"],
                             coverage=check_dumps(state["dump_dir"], state["split"], args.camera, args.frame_stride))
    write_json(state["dump_dir"] / "inference_manifest.json", state["manifest"])


def signal_job(process, force=False):
    try:
        if os.name == "posix":
            # Include DataLoader workers, even if the model process already exited.
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif process.poll() is None:
            process.kill() if force else process.terminate()
    except ProcessLookupError:
        pass


def stop_jobs(active):
    states = list(active.values())
    for state in states:
        signal_job(state["process"])
    deadline = time.monotonic() + 5
    for state in states:
        process = state["process"]
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signal_job(process, force=True)
            process.wait()
        finally:
            # Terminate any surviving workers in a failed POSIX process group.
            signal_job(process, force=True)
            state["log"].close()


def run_gpu_jobs(args, jobs):
    """Use independent processes, one per device; never split/reorder a frame set."""
    pending, active = deque(jobs), {}
    print(f"[SCHEDULE] GPUs={','.join(args.gpu_ids)} jobs={len(jobs)} batch_per_GPU={args.batch_size}", flush=True)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("GPU inference interrupted; stopping child processes.")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or active:
            # Check failures before assigning more work to any device.
            for gpu_id, state in list(active.items()):
                code = state["process"].poll()
                if code is None:
                    continue
                state["log"].close()
                if code:
                    with state["log_path"].open(encoding="utf-8", errors="replace") as stream:
                        tail = "".join(deque(stream, maxlen=20))
                    raise RuntimeError(f"GPU {gpu_id}: {state['variant']}/{state['split']} exited with {code}. "
                                       f"Log: {state['log_path']}\n{tail}")
                complete_job(args, state)
                print(f"[DONE] GPU={gpu_id} {state['variant']}/{state['split']} "
                      f"seconds={time.perf_counter() - state['started']:.1f}", flush=True)
                del active[gpu_id]
            for gpu_id in args.gpu_ids:
                while gpu_id not in active and pending:
                    state = prepare_job(args, pending.popleft(), gpu_id)
                    if state is None:
                        continue
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id)
                    log = state["log_path"].open("a", encoding="utf-8")
                    log.write(f"\n[GPU {gpu_id}] {shlex.join(state['command'])}\n")
                    log.flush()
                    try:
                        process = subprocess.Popen(state["command"], cwd=ROOT, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
                    except BaseException:
                        log.close()
                        raise
                    active[gpu_id] = {**state, "process": process, "log": log}
                    print(f"[RUN] GPU={gpu_id} {state['variant']}/{state['split']} log={state['log_path']}", flush=True)
            if active:
                time.sleep(0.2)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop_jobs(active)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)


def main(argv=None):
    args = parse_args(argv)
    jobs = build_jobs(args)
    if args.dry_run:
        if args.gpu_ids is not None:
            print(f"[PLAN] GPUs={','.join(args.gpu_ids)}: one variant/split job per GPU, next job goes to a free GPU.")
        for job in jobs:
            print(f"[DEPTH INFER] {job['variant']}/{job['split']}: {shlex.join(job['command'])}", flush=True)
    elif args.gpu_ids is not None:
        run_gpu_jobs(args, jobs)
    else:
        for job in jobs:
            print(f"[DEPTH INFER] {job['variant']}/{job['split']}: {shlex.join(job['command'])}", flush=True)
            state = prepare_job(args, job)
            if state is not None:
                run_logged(state["command"], state["log_path"])
                complete_job(args, state)


if __name__ == "__main__":
    main()
