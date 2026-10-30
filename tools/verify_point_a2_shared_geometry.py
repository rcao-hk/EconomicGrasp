#!/usr/bin/env python3
"""Verify A1/A2 checkpoints use *exactly identical frozen depth weights*.

Does not require MinkowskiEngine, GraspNetAPI or scene data.
Optional --a3 compares a full image-centric checkpoint's depth weights too.
This verifies model-state equality, NOT that A3 was trained with fixed depth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def read_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model_state_dict"), dict):
        raise ValueError(f"{path} is missing a full model_state_dict checkpoint")
    return ckpt


def depth_state(ckpt):
    model = ckpt["model_state_dict"]
    prefix = "depth_net."
    state = {k[len(prefix):]: v for k, v in model.items() if k.startswith(prefix)}
    if not state:
        raise ValueError("Checkpoint contains no depth_net.* weights")
    if not all(torch.is_tensor(v) for v in state.values()):
        raise TypeError("Non-tensor value in depth_net checkpoint state")
    return state


def compare_depth_states(ref, other):
    if ref.keys() != other.keys():
        miss = sorted(ref.keys() - other.keys())
        extra = sorted(other.keys() - ref.keys())
        raise ValueError(f"Depth parameter keys differ: missing={miss[:8]}, extra={extra[:8]}")
    bad = []
    for name in sorted(ref):
        a, b = ref[name], other[name]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            bad.append(name)
    if bad:
        raise AssertionError(f"Shared depth weights differ at {len(bad)} keys: {bad[:12]}")
    return len(ref)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_point_model(path, expected_fusion, expected_domain):
    ckpt = read_checkpoint(path)
    protocol = ckpt.get("full100_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("A1/A2 checkpoint missing point experiment protocol")
    if ckpt.get("completed_epoch") != 19:
        raise ValueError(f"{path} is not the preregistered completed epoch19")
    if ckpt.get("point_head") != "cdf" or protocol.get("head") != "cdf":
        raise ValueError("A1/A2 must both use identical CDF head labels")
    if ckpt.get("point_rgb_fusion", "none") != expected_fusion:
        raise ValueError(f"{path}: expected RGB fusion {expected_fusion}")
    if protocol.get("point_rgb_fusion", "none") != expected_fusion:
        raise ValueError("Checkpoint RGB fuse flag/manifest disagree")
    if protocol.get("train_domain", "original") != expected_domain:
        raise ValueError("A1/A2 training domains differ")
    if protocol.get("depth_training_mode") not in (None, "shared_frozen_depth"):
        raise ValueError("Unexpected depth training mode")
    source = protocol.get("shared_depth_checkpoint_sha256")
    if not source or len(source) != 64:
        raise ValueError("Missing frozen shared-depth checkpoint sha256")
    return ckpt, protocol, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--a3", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-domain", choices=("mixed", "original"), default="mixed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    a1, p1, source1 = check_point_model(args.a1, "none", args.expected_domain)
    a2, p2, source2 = check_point_model(args.a2, "dino_multiscale", args.expected_domain)
    if source1 != source2:
        raise AssertionError("A1/A2 use different shared frozen DPT checkpoint sources")
    for key in ("training_scene_frame_sha256", "validation_scene_frame_sha256",
                "gntrans_train_count", "original_train_count", "train_frame_count",
                "seed", "epochs", "batch_size_per_rank", "global_batch_size",
                "gntrans_depth_target", "original_depth_target"):
        if p1.get(key) != p2.get(key):
            raise AssertionError(f"A1/A2 protocol mismatch: {key}")
    state1 = depth_state(a1)
    depth_count = compare_depth_states(state1, depth_state(a2))
    summary = {
        "result": "PASS: A1/A2 exact frozen DPT state equality",
        "point_head": "cdf",
        "source_sha256": source1,
        "depth_state_tensors": depth_count,
        "train_domain": args.expected_domain,
        "a1_checkpoint_sha256": file_sha256(args.a1),
        "a2_checkpoint_sha256": file_sha256(args.a2),
        "a1_point_rgb_fusion": "none",
        "a2_point_rgb_fusion": "dino_multiscale",
        "limitations": "Weights identical; compare per-frame image/K/pose/pixel indices and A3 frozen training separately.",
    }
    if args.source:
        if file_sha256(args.source) != source1:
            raise AssertionError("External source checkpoint SHA differs from A1/A2 manifest")
        summary["source_depth_tensors"] = compare_depth_states(
            state1, depth_state(read_checkpoint(args.source))
        )
    if args.a3:
        summary["a3_depth_tensors"] = compare_depth_states(
            state1, depth_state(read_checkpoint(args.a3))
        )
        summary["a3_depth_weights_equal"] = True
    report = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refuse to overwrite geometry report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
