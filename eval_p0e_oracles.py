#!/usr/bin/env python3
"""Evaluate P0-E oracle dumps with the official GraspNet evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from graspnetAPI import GraspNetEval

from p0e_oracle_common import DEFAULT_VARIANTS, PROTOCOL_VERSION, scene_ids_for_split


def _parse_variants(text: str) -> Tuple[str, ...]:
    variants = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not variants:
        raise ValueError("--variants cannot be empty.")
    if len(variants) != len(set(variants)):
        raise ValueError(f"Duplicate variants are unsupported: {variants}.")
    for variant in variants:
        if any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in variant
        ):
            raise ValueError(f"Unsafe variant name {variant!r}.")
    return variants


def _expected_files(
    dump_dir: Path,
    *,
    split: str,
    camera: str,
    sample_interval: int,
) -> Iterable[Path]:
    for scene_id in scene_ids_for_split(split):
        for anno_id in range(0, 256, int(sample_interval)):
            yield dump_dir / f"scene_{scene_id:04d}" / camera / f"{anno_id:04d}.npy"


def _check_complete(
    dump_dir: Path,
    *,
    split: str,
    camera: str,
    sample_interval: int,
) -> Dict[str, object]:
    expected = list(
        _expected_files(
            dump_dir,
            split=split,
            camera=camera,
            sample_interval=sample_interval,
        )
    )
    missing = [str(path) for path in expected if not path.is_file()]
    actual = sum(
        1
        for scene_id in scene_ids_for_split(split)
        for _path in (dump_dir / f"scene_{scene_id:04d}" / camera).glob("*.npy")
    )
    if missing:
        raise FileNotFoundError(
            f"P0-E dump is incomplete: dump={dump_dir}, expected={len(expected)}, "
            f"actual={actual}. First missing files: {missing[:20]}"
        )
    return {
        "expected_files": len(expected),
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

    if int(sample_interval) == 1:
        try:
            return method(str(dump_dir), anno_sample_ratio=1.0, proc=int(num_workers))
        except TypeError as error:
            if "anno_sample_ratio" not in str(error):
                raise
            return method(str(dump_dir), proc=int(num_workers))

    try:
        return method(
            str(dump_dir),
            anno_sample_ratio=1.0 / float(sample_interval),
            proc=int(num_workers),
        )
    except TypeError as error:
        raise RuntimeError(
            "Sampled P0-E evaluation requires the repository GraspNetAPI fork "
            "with anno_sample_ratio. Use --sample_interval 1 for full official "
            "evaluation or install the same fork used by eval.py."
        ) from error


def _scalar(value) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Evaluator returned an empty AP value.")
    return float(array.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument(
        "--split",
        required=True,
        choices=("test_seen", "test_similar", "test_novel"),
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--skip_complete_check", action="store_true")
    args = parser.parse_args()

    if int(args.num_workers) <= 0:
        raise ValueError("--num_workers must be positive.")
    if int(args.sample_interval) <= 0:
        raise ValueError("--sample_interval must be positive.")

    prediction_root = Path(args.prediction_root).resolve()
    meta_root = prediction_root / "_p0e_meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    variants = _parse_variants(args.variants)

    all_results: Dict[str, Dict[str, object]] = {}
    for variant in variants:
        dump_dir = prediction_root / variant
        if not dump_dir.is_dir():
            raise FileNotFoundError(f"P0-E variant directory not found: {dump_dir}")
        completeness: Dict[str, object] = {}
        if not bool(args.skip_complete_check):
            completeness = _check_complete(
                dump_dir,
                split=args.split,
                camera=args.camera,
                sample_interval=int(args.sample_interval),
            )

        print(
            f"[P0-E][EVAL] variant={variant} split={args.split} dump={dump_dir}",
            flush=True,
        )
        evaluator = GraspNetEval(
            root=args.dataset_root,
            camera=args.camera,
            split=args.split,
        )
        result, returned_ap = _call_evaluator(
            evaluator,
            split=args.split,
            dump_dir=dump_dir,
            num_workers=int(args.num_workers),
            sample_interval=int(args.sample_interval),
        )
        result = np.asarray(result, dtype=np.float64)
        if result.ndim < 1 or result.shape[-1] < 4:
            raise RuntimeError(
                f"Unexpected GraspNet result shape for {variant}: {result.shape}."
            )
        ap = float(result.mean())
        ap04 = float(result[..., 1].mean())
        ap08 = float(result[..., 3].mean())
        returned = _scalar(returned_ap)
        if abs(ap - returned) > 1.0e-8:
            print(
                f"[P0-E][WARN] result.mean={ap:.9f} differs from evaluator "
                f"AP={returned:.9f}; recording both.",
                flush=True,
            )

        result_path = dump_dir / f"ap_{args.split}_{args.camera}.npy"
        np.save(result_path, result)
        record: Dict[str, object] = {
            "protocol": PROTOCOL_VERSION,
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
            "result_npy": str(result_path),
            **completeness,
        }
        json_path = meta_root / f"eval_{variant}_{args.split}_{args.camera}.json"
        json_path.write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        all_results[variant] = record
        print(
            f"[P0-E][EVAL] {variant}: AP={100.0 * ap:.4f}, "
            f"AP0.8={100.0 * ap08:.4f}, AP0.4={100.0 * ap04:.4f}",
            flush=True,
        )

    combined_path = meta_root / f"eval_all_{args.split}_{args.camera}.json"
    combined_path.write_text(
        json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[P0-E][EVAL] saved {combined_path}", flush=True)


if __name__ == "__main__":
    main()
