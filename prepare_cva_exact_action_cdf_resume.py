#!/usr/bin/env python3
"""Scene-granular resume preparation for current CVA-CDF exact-action mining.

A scene is considered complete only when it contains exactly the annotation
files implied by the miner's ``sample_interval`` and every expected NPZ passes
a lightweight current-cache contract check.  If a scene is incomplete, the
entire ``scene_xxxx`` directory is removed so the next mining run regenerates
that scene from scratch.

This helper intentionally supports only the current CVA-CDF cache schema.  It
has no compatibility path for legacy joint-utility caches.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from exact_action_cdf_common import (
    CACHE_SCHEMA_VERSION,
    atomic_save_json,
    sha256_file,
)
from exact_action_cdf_cache import REQUIRED_ARRAYS, validate_cache_arrays


ANN_RE = re.compile(r"^ann_(\d{4})\.npz$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--scene_start", type=int, default=0)
    p.add_argument("--scene_end", type=int, default=99)
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--pending_file", required=True)
    p.add_argument("--output_json", default="")
    p.add_argument("--clean_incomplete", type=int, choices=(0, 1), default=1)
    p.add_argument("--fail_if_pending", type=int, choices=(0, 1), default=0)
    p.add_argument("--verbose", type=int, choices=(0, 1), default=1)
    return p.parse_args()


def expected_annos(sample_interval: float) -> Tuple[int, ...]:
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    stride = 1 if sample_interval >= 1.0 else max(1, int(round(1.0 / sample_interval)))
    return tuple(range(0, 256, stride))


def _scalar_string(array: np.ndarray) -> str:
    value = np.asarray(array).reshape(-1)
    if value.size != 1:
        raise ValueError(f"expected scalar string, got {np.asarray(array).shape}")
    return str(value[0])


def _scalar_int(array: np.ndarray) -> int:
    value = np.asarray(array).reshape(-1)
    if value.size != 1:
        raise ValueError(f"expected scalar integer, got {np.asarray(array).shape}")
    return int(value[0])


def quick_validate_file(
    path: Path,
    *,
    scene_id: int,
    anno_id: int,
    checkpoint_sha256: str,
) -> None:
    """Check that an existing frame is safe to count as complete.

    Mining publishes final NPZ files atomically.  Resume still reads every NPZ
    member so ZIP/CRC corruption is detected, then runs the current-cache
    structural/provenance validator.  Expensive value-distribution checks are
    left to the normal full-cache validator at the end of mining.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise RuntimeError("zero-length cache file")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    meta = validate_cache_arrays(
        arrays,
        path=str(path),
        expected_checkpoint_sha256=checkpoint_sha256,
        check_values=False,
    )
    if meta.scene_id != scene_id or meta.anno_id != anno_id:
        raise RuntimeError(
            f"metadata scene/anno={meta.scene_id}/{meta.anno_id}, "
            f"expected {scene_id}/{anno_id}"
        )


@dataclass
class SceneStatus:
    scene_id: int
    complete: bool
    expected_frames: int
    existing_npz: int
    missing_annos: List[int]
    unexpected_annos: List[int]
    invalid_files: Dict[str, str]
    cleaned: bool = False


def inspect_scene(
    cache_dir: Path,
    scene_id: int,
    annos: Sequence[int],
    checkpoint_sha256: str,
) -> SceneStatus:
    scene_dir = cache_dir / f"scene_{scene_id:04d}"
    expected_set = set(int(x) for x in annos)
    existing: Dict[int, Path] = {}
    invalid: Dict[str, str] = {}

    if scene_dir.is_dir():
        # Temp files are safe to remove: final NPZ publication is atomic.
        for tmp in scene_dir.glob("*.tmp.*"):
            try:
                tmp.unlink()
            except OSError:
                pass
        for path in sorted(scene_dir.glob("ann_*.npz")):
            match = ANN_RE.match(path.name)
            if match is None:
                continue
            existing[int(match.group(1))] = path

    existing_set = set(existing)
    missing = sorted(expected_set - existing_set)
    unexpected = sorted(existing_set - expected_set)

    # Only validate expected files. Unexpected frame files make the scene
    # incomplete anyway and the whole directory will be regenerated.
    if not missing and not unexpected:
        for anno_id in annos:
            path = existing[int(anno_id)]
            try:
                quick_validate_file(
                    path,
                    scene_id=scene_id,
                    anno_id=int(anno_id),
                    checkpoint_sha256=checkpoint_sha256,
                )
            except Exception as exc:  # report exact file, reset whole scene later
                invalid[path.name] = repr(exc)

    complete = not missing and not unexpected and not invalid
    return SceneStatus(
        scene_id=scene_id,
        complete=complete,
        expected_frames=len(annos),
        existing_npz=len(existing),
        missing_annos=missing,
        unexpected_annos=unexpected,
        invalid_files=invalid,
    )


def main() -> None:
    args = parse_args()
    if args.scene_start < 0 or args.scene_end > 189 or args.scene_start > args.scene_end:
        raise ValueError("expected 0 <= scene_start <= scene_end <= 189")
    if int(args.max_samples) > 0:
        raise RuntimeError(
            "Scene-granular resume requires --max_samples=-1 because a global "
            "sample cap does not define a complete per-scene file set."
        )

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    checkpoint_sha = sha256_file(str(checkpoint))
    annos = expected_annos(float(args.sample_interval))

    statuses: List[SceneStatus] = []
    pending: List[int] = []
    complete: List[int] = []

    for scene_id in range(int(args.scene_start), int(args.scene_end) + 1):
        status = inspect_scene(cache_dir, scene_id, annos, checkpoint_sha)
        if status.complete:
            complete.append(scene_id)
            if args.verbose:
                print(
                    f"[RESUME][KEEP] scene_{scene_id:04d}: "
                    f"{status.expected_frames}/{status.expected_frames} frames complete",
                    flush=True,
                )
        else:
            pending.append(scene_id)
            scene_dir = cache_dir / f"scene_{scene_id:04d}"
            reasons: List[str] = []
            if status.missing_annos:
                preview = ",".join(f"{x:04d}" for x in status.missing_annos[:8])
                suffix = "..." if len(status.missing_annos) > 8 else ""
                reasons.append(f"missing={len(status.missing_annos)}[{preview}{suffix}]")
            if status.unexpected_annos:
                preview = ",".join(f"{x:04d}" for x in status.unexpected_annos[:8])
                suffix = "..." if len(status.unexpected_annos) > 8 else ""
                reasons.append(
                    f"unexpected={len(status.unexpected_annos)}[{preview}{suffix}]"
                )
            if status.invalid_files:
                preview = ",".join(list(status.invalid_files)[:4])
                reasons.append(f"invalid={len(status.invalid_files)}[{preview}]")
            if not scene_dir.exists():
                reasons.append("scene_dir_absent")
            if bool(args.clean_incomplete) and scene_dir.exists():
                shutil.rmtree(scene_dir)
                status.cleaned = True
            if args.verbose:
                action = "RESET" if status.cleaned else "PENDING"
                print(
                    f"[RESUME][{action}] scene_{scene_id:04d}: "
                    f"existing={status.existing_npz}/{status.expected_frames}; "
                    + "; ".join(reasons),
                    flush=True,
                )
        statuses.append(status)

    pending_path = Path(args.pending_file).expanduser().resolve()
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pending_path.with_name(pending_path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        for scene_id in pending:
            handle.write(f"{scene_id}\n")
    os.replace(tmp, pending_path)

    payload = {
        "cache_dir": str(cache_dir),
        "schema_version": CACHE_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "scene_start": int(args.scene_start),
        "scene_end": int(args.scene_end),
        "sample_interval": float(args.sample_interval),
        "expected_annos": list(annos),
        "expected_frames_per_scene": len(annos),
        "num_complete_scenes": len(complete),
        "num_pending_scenes": len(pending),
        "complete_scenes": complete,
        "pending_scenes": pending,
        "clean_incomplete": bool(args.clean_incomplete),
        "statuses": [asdict(status) for status in statuses],
    }
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else cache_dir / "_resume_scene_status.json"
    )
    atomic_save_json(payload, str(output_json))
    print(
        f"[RESUME] complete={len(complete)} pending={len(pending)} "
        f"expected_frames_per_scene={len(annos)} pending_file={pending_path}",
        flush=True,
    )
    print(f"[RESUME] status_json={output_json}", flush=True)

    if bool(args.fail_if_pending) and pending:
        raise SystemExit(
            "Scene-level completeness check failed after mining; pending scenes: "
            + ",".join(str(x) for x in pending)
        )


if __name__ == "__main__":
    main()
