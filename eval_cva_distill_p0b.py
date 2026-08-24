#!/usr/bin/env python3
"""Evaluate P0-B variant dumps with the official GraspNet evaluator."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from graspnetAPI import GraspNetEval


SUPPORTED_VARIANTS = (
    "student",
    "teacher_full",
    "teacher_common",
    "oracle_hybrid",
)


def _parse_variants(text: str) -> Tuple[str, ...]:
    variants = tuple(x.strip() for x in text.split(",") if x.strip())
    if not variants:
        raise ValueError("--variants cannot be empty.")
    unknown = sorted(set(variants) - set(SUPPORTED_VARIANTS))
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}; supported={list(SUPPORTED_VARIANTS)}."
        )
    return variants


def _scene_ids(split: str) -> range:
    if split == "test_seen":
        return range(100, 130)
    if split == "test_similar":
        return range(130, 160)
    if split == "test_novel":
        return range(160, 190)
    raise ValueError(f"Unsupported split: {split!r}.")


def _expected_files(
    dump_dir: Path,
    *,
    split: str,
    camera: str,
    sample_interval: int,
) -> Iterable[Path]:
    for scene_id in _scene_ids(split):
        for anno_id in range(0, 256, sample_interval):
            yield (
                dump_dir
                / f"scene_{scene_id:04d}"
                / camera
                / f"{anno_id:04d}.npy"
            )


def _check_complete(
    dump_dir: Path,
    *,
    split: str,
    camera: str,
    sample_interval: int,
) -> Dict[str, object]:
    missing: List[str] = []
    expected = 0
    for path in _expected_files(
        dump_dir,
        split=split,
        camera=camera,
        sample_interval=sample_interval,
    ):
        expected += 1
        if not path.is_file() and len(missing) < 20:
            missing.append(str(path))
    actual = sum(
        1
        for scene_id in _scene_ids(split)
        for _ in (dump_dir / f"scene_{scene_id:04d}" / camera).glob("*.npy")
    )
    if missing:
        raise FileNotFoundError(
            f"P0-B dump is incomplete: dump={dump_dir}, expected={expected}, "
            f"actual={actual}. First missing files: {missing}"
        )
    return {
        "expected_files": expected,
        "actual_files_in_split_scene_dirs": actual,
    }


def _call_evaluator(
    evaluator: GraspNetEval,
    *,
    split: str,
    dump_dir: Path,
    num_workers: int,
    sample_interval: int,
):
    if split == "test_seen":
        method = evaluator.eval_seen
    elif split == "test_similar":
        method = evaluator.eval_similar
    elif split == "test_novel":
        method = evaluator.eval_novel
    else:
        raise ValueError(split)

    # The user's current GraspNetAPI fork accepts anno_sample_ratio.  The
    # upstream API does not.  Full evaluation works with either API; sampled
    # evaluation requires the fork used by the existing repository eval.py.
    if sample_interval == 1:
        try:
            return method(
                str(dump_dir),
                anno_sample_ratio=1.0,
                proc=num_workers,
            )
        except TypeError as exc:
            if "anno_sample_ratio" not in str(exc):
                raise
            return method(str(dump_dir), proc=num_workers)

    try:
        return method(
            str(dump_dir),
            anno_sample_ratio=1.0 / float(sample_interval),
            proc=num_workers,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Sampled P0-B evaluation requires the repository's GraspNetAPI "
            "extension with anno_sample_ratio. Use --sample_interval 1 for "
            "the official full evaluation, or install the same API fork used "
            "by eval.py."
        ) from exc


def _scalar(value) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Evaluator returned an empty AP value.")
    return float(array.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument(
        "--split",
        required=True,
        choices=("test_seen", "test_similar", "test_novel"),
    )
    parser.add_argument(
        "--variants",
        default=",".join(SUPPORTED_VARIANTS),
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Evaluate annotation IDs 0,K,2K,...; official full evaluation uses K=1.",
    )
    parser.add_argument("--skip_complete_check", action="store_true")
    args = parser.parse_args()

    if args.sample_interval <= 0:
        raise ValueError("--sample_interval must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive.")

    prediction_root = Path(args.prediction_root).resolve()
    meta_dir = prediction_root / "_p0b_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    variants = _parse_variants(args.variants)

    all_results: Dict[str, Dict[str, object]] = {}
    for variant in variants:
        dump_dir = prediction_root / variant
        if not dump_dir.is_dir():
            raise FileNotFoundError(f"P0-B variant dump directory not found: {dump_dir}")
        completeness = {}
        if not args.skip_complete_check:
            completeness = _check_complete(
                dump_dir,
                split=args.split,
                camera=args.camera,
                sample_interval=args.sample_interval,
            )

        print(
            f"[P0-B][EVAL] variant={variant} split={args.split} "
            f"camera={args.camera} dump={dump_dir}",
            flush=True,
        )
        evaluator = GraspNetEval(
            root=args.dataset_root,
            camera=args.camera,
            split=args.split,
        )
        result, ap_returned = _call_evaluator(
            evaluator,
            split=args.split,
            dump_dir=dump_dir,
            num_workers=args.num_workers,
            sample_interval=args.sample_interval,
        )
        result = np.asarray(result, dtype=np.float64)
        if result.ndim < 1 or result.shape[-1] < 4:
            raise RuntimeError(
                f"Unexpected GraspNet result shape for {variant}: {result.shape}."
            )

        ap = float(result.mean())
        ap04 = float(result[..., 1].mean())
        ap08 = float(result[..., 3].mean())
        returned = _scalar(ap_returned)
        if abs(ap - returned) > 1.0e-8:
            print(
                f"[P0-B][WARN] result.mean={ap:.9f} differs from evaluator "
                f"AP={returned:.9f}; recording both.",
                flush=True,
            )

        npy_path = dump_dir / f"ap_{args.split}_{args.camera}.npy"
        np.save(npy_path, result)
        record: Dict[str, object] = {
            "protocol": "P0-B-official-AP-oracle-hybrid-v1",
            "variant": variant,
            "split": args.split,
            "camera": args.camera,
            "dump_dir": str(dump_dir),
            "sample_interval": int(args.sample_interval),
            "result_shape": list(result.shape),
            "ap": ap,
            "ap0.4": ap04,
            "ap0.8": ap08,
            "evaluator_returned_ap": returned,
            "ap_percent": 100.0 * ap,
            "ap0.4_percent": 100.0 * ap04,
            "ap0.8_percent": 100.0 * ap08,
            "result_npy": str(npy_path),
            **completeness,
        }
        json_path = meta_dir / f"eval_{variant}_{args.split}_{args.camera}.json"
        json_path.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        all_results[variant] = record
        print(
            f"[P0-B][EVAL] {variant}: AP={100.0 * ap:.4f}, "
            f"AP0.8={100.0 * ap08:.4f}, AP0.4={100.0 * ap04:.4f}",
            flush=True,
        )

    combined_path = meta_dir / f"eval_all_{args.split}_{args.camera}.json"
    combined_path.write_text(
        json.dumps(all_results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[P0-B][EVAL] saved {combined_path}")


if __name__ == "__main__":
    main()
