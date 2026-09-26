#!/usr/bin/env python3
"""Official GraspNet AP for complete MGF dumps (no sensor collision filter)."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from metric_field_runtime import VERSION, atomic_json, digest, sha256_file
from inference_metric_grasp_field import atomic_npy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="/data/robotarm/dataset/graspnet")
    p.add_argument("--inference-root", required=True)
    p.add_argument("--split", choices=("test_seen", "test_similar", "test_novel"), required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    sys.argv = [sys.argv[0]]
    root = Path(args.inference_root)
    protocol = json.loads((root/args.split/"protocol.json").read_text())
    if protocol["version"] != VERSION or protocol["split"] != args.split:
        raise ValueError("Wrong inference protocol")
    if protocol["max_frames"] or protocol["training_protocol"]["partial_run"]:
        raise ValueError("Refusing formal AP on a smoke/partial run")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    hashes = []
    for sid, aid in protocol["schedule"]:
        path = root/"dump"/f"scene_{sid:04d}"/"realsense"/f"{aid:04d}.npy"
        marker = root/args.split/"completed"/f"{sid:04d}_{aid:04d}.json"
        if not path.is_file() or not marker.is_file():
            raise FileNotFoundError(f"Incomplete dump: {path}")
        sha = sha256_file(path)
        m = json.loads(marker.read_text())
        if m["signature"] != digest(protocol) or m["output_sha256"] != sha:
            raise ValueError(f"Dump/marker mismatch: {path}")
        hashes.append(sha)
    sig = digest({"protocol": protocol, "dump_sha256": hashes})
    out = root/"official"/args.split
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with open(out/".eval.lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/"summary.json").is_file():
            if not args.resume:
                raise FileExistsError("Existing AP result; use --resume")
            old = json.loads((out/"summary.json").read_text())
            if old["signature"] != sig:
                raise RuntimeError("Existing evaluation does not match the dumps")
            return
        from graspnetAPI import GraspNetEval
        evaluator = GraspNetEval(args.dataset_root, camera="realsense", split=args.split)
        evaluate = getattr(evaluator, {"test_seen": "eval_seen", "test_similar": "eval_similar",
                                       "test_novel": "eval_novel"}[args.split])
        fraction = protocol["training_protocol"]["sample_fraction"]
        result, ap = evaluate(str(root/"dump"), anno_sample_ratio=fraction, proc=args.workers)
        accuracy = np.asarray(result)
        frame_count = len(range(0, 256, round(1/fraction)))
        if accuracy.shape[:2] != (30, frame_count) or not np.isfinite(accuracy).all():
            raise RuntimeError(f"Unexpected GraspNet result shape/values: {accuracy.shape}")
        atomic_npy(out/"accuracy.npy", accuracy)
        atomic_json(out/"summary.json", {"signature": sig, "split": args.split,
                    "checkpoint_epoch": protocol["checkpoint_epoch"], "reported_ap": np.asarray(ap).tolist(),
                    "mean_accuracy": float(accuracy.mean()), "shape": list(accuracy.shape),
                    "collision_filter": "none", "seen_role": "validation" if args.split == "test_seen" else "held_out"})
        print(f"[MGF AP] {args.split}: {ap}", flush=True)


if __name__ == "__main__":
    main()
