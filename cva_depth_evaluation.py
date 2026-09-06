"""Small, model-independent helpers for matched depth-control inference/AP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


VARIANTS = ("base", "none", "foreground", "anchor")
SPLITS = ("test_seen", "test_similar", "test_novel")
CONTRACT_VERSION = 1
FRICTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def csv_choices(value, choices):
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items or len(set(items)) != len(items) or set(items) - set(choices):
        raise argparse.ArgumentTypeError(f"Use unique comma-separated values from {choices}.")
    return items


def add_selection_arguments(parser):
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--variants", type=lambda value: csv_choices(value, VARIANTS), default=VARIANTS)
    parser.add_argument("--splits", type=lambda value: csv_choices(value, SPLITS), default=SPLITS)


def scene_ids(split):
    first = {"test_seen": 100, "test_similar": 130, "test_novel": 160}[split]
    return list(range(first, first + 30))


def annotation_ids(frame_stride):
    if not 1 <= frame_stride <= 256:
        raise ValueError("frame_stride must be an integer in [1, 256].")
    return list(range(0, 256, frame_stride))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dumps(dump_dir, split, camera, frame_stride):
    """Require exact frame coverage and finite GraspGroup arrays; empty is valid.

    The stat fingerprint detects ordinary edits/replacements for AP caching.
    It is not a cryptographic content fingerprint of the prediction arrays.
    """
    import numpy as np

    dump_dir = Path(dump_dir)
    expected = {
        Path(f"scene_{scene:04d}") / camera / f"{anno:04d}.npy"
        for scene in scene_ids(split) for anno in annotation_ids(frame_stride)
    }
    actual = {path.relative_to(dump_dir) for path in dump_dir.glob(f"scene_*/{camera}/*.npy")}
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(
            f"Incomplete/mixed dumps in {dump_dir}: expected={len(expected)}, actual={len(actual)}, "
            f"missing={len(missing)}, extra={len(extra)}; "
            f"first missing={sorted(map(str, missing))[:3]}, first extra={sorted(map(str, extra))[:3]}"
        )
    fingerprint = hashlib.sha256()
    empty = 0
    for relative in sorted(expected):
        path = dump_dir / relative
        array = np.load(path, allow_pickle=False)
        if (array.ndim != 2 or array.shape[1] != 17 or array.dtype.kind not in "fiu"
                or not np.isfinite(array).all()):
            raise ValueError(f"Invalid GraspGroup dump (expected finite N x 17 numeric array): {path}")
        empty += int(len(array) == 0)
        stat = path.stat()
        fingerprint.update(f"{relative.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return {"frames": len(expected), "empty_grasp_frames": empty,
            "file_stat_fingerprint": fingerprint.hexdigest()}


def read_completed_manifest(dump_dir, variant, split):
    path = Path(dump_dir) / "inference_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No inference manifest: {path}. Use inference_cva_depth_controls.py first.")
    record = read_json(path)
    if (record.get("status") != "complete" or record.get("variant") != variant
            or record.get("identity", {}).get("split") != split
            or record.get("identity", {}).get("contract_version") != CONTRACT_VERSION):
        raise ValueError(f"Inference is incomplete or its manifest does not match {variant}/{split}: {path}")
    return record


def summarize_accuracy(result, split, frame_stride, returned_ap):
    """GraspNet AP: mean precision over frames, ranks 1..50 and six frictions."""
    import numpy as np

    result = np.asarray(result, dtype=np.float64)
    expected = (len(scene_ids(split)), len(annotation_ids(frame_stride)), 50, len(FRICTIONS))
    if result.shape != expected:
        raise ValueError(f"GraspNet accuracy shape {result.shape} != {expected}; check frame sampling/API version.")
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("GraspNet returned non-finite or out-of-range precision values.")
    ap = float(result.mean())
    returned = float(np.asarray(returned_ap, dtype=np.float64).mean())
    if not np.isfinite(returned) or abs(returned - ap) > 1e-6:
        raise ValueError(f"Evaluator AP={returned} differs from accuracy.mean()={ap}.")
    return {"result_shape": list(result.shape), "ap": ap, "ap_04": float(result[..., 1].mean()),
            "ap_08": float(result[..., 3].mean()), "ap_percent": 100 * ap,
            "ap_04_percent": 100 * float(result[..., 1].mean()),
            "ap_08_percent": 100 * float(result[..., 3].mean()),
            "evaluator_returned_ap": returned,
            "scene_ap_percent": dict(zip(map(str, scene_ids(split)), (100 * result.mean(axis=(1, 2, 3))).tolist()))}
