"""Validate point-depth baseline checkpoint training domains and sampling.

Pure Python: importable by unit tests and both inference entrypoints without
loading torch, GraspNetAPI, or MinkowskiEngine. A matching *nominal* frame
count alone does not establish a compatible source/domain contract: 100%
Original and 50% Original + 50% GN-Trans both contain 25,600 frames.
"""
from __future__ import annotations

import math


def expected_frames_per_domain(fraction: float, *, scenes: int = 100) -> int:
    """Same per-scene stride rule as scene_stride_indices (256 images/scene)."""
    fraction = float(fraction)
    if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError(f"Invalid training fraction: {fraction}")
    stride = 1 if fraction >= 1.0 else max(1, round(1.0 / fraction))
    return scenes * len(range(0, 256, stride))


def validate_training_protocol(protocol, *, head: str, source: str):
    """Return (domain, validated_protocol), fail closed on incompatible runs.

    source='full100_protocol' covers both original-only and mixed domains in the
    newer dual-head trainer, retaining the original checkpoint key for
    backwards compatibility. source='mixed_protocol' covers the historical
    Legacy-only mixed trainer; it has never trained Point-CDF.
    """
    if head not in ("legacy", "cdf"):
        raise ValueError(f"Unrecognized point head {head}")
    if not isinstance(protocol, dict):
        raise ValueError(f"Missing {source} checkpoint manifest")
    if source == "full100_protocol":
        if protocol.get("head") != head:
            raise ValueError("Checkpoint head / training manifest mismatch")
        if protocol.get("pose_depth_mode") != "global_film":
            raise ValueError("Expected global_film metric depth")
        if int(protocol.get("max_train_batches", 0)) != 0 or int(protocol.get("max_val_batches", 0)) != 0:
            raise ValueError("Partial-training/smoke checkpoint cannot be evaluated")
        domain = protocol.get("train_domain", "original")
        if domain == "original":
            if int(protocol.get("train_frame_count", -1)) != 25600:
                raise ValueError("Original-only training must use all 25,600 frames")
            if protocol.get("gntrans_train_count", 0) != 0:
                raise ValueError("Original-only training contains GN-Trans frames")
        elif domain == "mixed":
            fraction = float(protocol.get("mix_train_fraction", -1))
            target = expected_frames_per_domain(fraction)
            original = int(protocol.get("original_train_count", -1))
            trans = int(protocol.get("gntrans_train_count", -1))
            if original != target or trans != target:
                raise ValueError(
                    f"Incorrect paired-domain counts for fraction={fraction}: "
                    f"Original={original}, GN-Trans={trans}, expected={target}"
                )
            if int(protocol.get("train_frame_count", -1)) != 2 * target:
                raise ValueError("Mixed training frame total disagrees with per-domain counts")
            if protocol.get("paired_scene_frame_sampling") is not True:
                raise ValueError("Mixed training lacks paired frame sampling guarantee")
            sig1 = protocol.get("original_train_scene_frame_sha256")
            sig2 = protocol.get("gntrans_train_scene_frame_sha256")
            if not isinstance(sig1, str) or len(sig1) != 64 or sig1 != sig2:
                raise ValueError("Original/GN-Trans frame signatures do not match")
            if not protocol.get("gntrans_rgb_root"):
                raise ValueError("Mixed checkpoint is missing GN-Trans RGB origin")
        else:
            raise ValueError(f"Unknown training domain: {domain}")
        return domain, protocol
    if source == "mixed_protocol":
        if head != "legacy":
            raise ValueError("Historical mixed checkpoint supports Legacy head only")
        if protocol.get("pose_depth_mode", "global_film") != "global_film":
            raise ValueError("Historical mixed checkpoint has wrong pose depth mode")
        fraction = float(protocol.get("train_fraction", -1))
        target = expected_frames_per_domain(fraction)
        if (int(protocol.get("original_train_count", -1)) != target
                or int(protocol.get("gntrans_train_count", -1)) != target
                or int(protocol.get("train_count", -1)) != 2 * target):
            raise ValueError("Legacy mixed checkpoint has inconsistent training counts")
        if protocol.get("paired_scene_frame_sampling") is not True:
            raise ValueError("Historical mixed training is missing pairing assertion")
        sig1 = protocol.get("original_train_sha256")
        sig2 = protocol.get("gntrans_train_sha256")
        if not isinstance(sig1, str) or len(sig1) != 64 or sig1 != sig2:
            raise ValueError("Legacy mixed checkpoint has mismatched frame hashes")
        return "mixed", protocol
    raise ValueError(f"Unknown checkpoint manifest source: {source}")


def checkpoint_training_protocol(checkpoint, *, expected_head: str):
    """Check full checkpoint source and return (domain, protocol, source key)."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Point checkpoint must be a dictionary")
    if checkpoint.get("architecture") != "economicgrasp_depth_baseline":
        raise ValueError("Incorrect point-depth baseline architecture")
    if checkpoint.get("pose_depth_mode") != "global_film":
        raise ValueError("Point checkpoint must use global_film pose-aware depth")
    saved_head = checkpoint.get("point_head")
    if saved_head is None and "mixed_protocol" in checkpoint:
        saved_head = "legacy"  # historical pre-CDF checkpoint contract
    if saved_head != expected_head:
        raise ValueError(f"Wrong checkpoint head: expected {expected_head}, got {saved_head}")
    if checkpoint.get("completed_epoch") != 19:
        raise ValueError("Official point comparison requires completed_epoch=19")

    key = ("full100_protocol" if "full100_protocol" in checkpoint
           else "mixed_protocol" if "mixed_protocol" in checkpoint
           else None)
    if key is None:
        raise ValueError("Checkpoint contains no recognized training manifest")
    domain, validated = validate_training_protocol(
        checkpoint[key], head=expected_head, source=key)
    return domain, validated, key
