#!/usr/bin/env python3
"""Evaluate matched depth-control dumps with GraspNetAPI and export AP tables.

Camera, frame sampling, decoder and collision settings come from inference
manifests. No model checkpoint or GPU is required by this launcher.
"""

from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
import time

from cva_depth_evaluation import (FRICTIONS, add_selection_arguments, check_dumps, read_completed_manifest,
                                  read_json, sha256_file, summarize_accuracy, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--num_workers", type=int, default=4, help="CPU evaluation worker processes.")
    parser.add_argument("--check_only", action="store_true", help="Validate all dumps without loading GraspNetAPI.")
    parser.add_argument("--force", action="store_true", help="Recompute even if matching AP results are cached.")
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num_workers must be positive.")
    return args


def evaluator_method(evaluator, split, frame_stride):
    method = getattr(evaluator, {"test_seen": "eval_seen", "test_similar": "eval_similar", "test_novel": "eval_novel"}[split])
    parameters = inspect.signature(method).parameters
    supports_sampling = "anno_sample_ratio" in parameters
    if frame_stride != 1 and not supports_sampling:
        raise RuntimeError(
            "FRAME_STRIDE > 1 requires the GraspNetAPI fork with anno_sample_ratio, as used by eval.py. "
            "The installed API cannot evaluate sampled frames. Generate a new FRAME_STRIDE=1 dump for "
            "official full evaluation, or use your existing API fork. No frames will be silently omitted."
        )
    return method, ({"anno_sample_ratio": 1.0 / frame_stride} if supports_sampling else {})


def call_evaluator(evaluator, split, dump_dir, num_workers, frame_stride):
    method, kwargs = evaluator_method(evaluator, split, frame_stride)
    return method(str(dump_dir), proc=num_workers, **kwargs)


def evaluator_fingerprint(evaluator_class):
    source = Path(inspect.getfile(evaluator_class)).resolve()
    package = source.parent
    # Include the geometry/NMS implementation as well as the public AP wrapper.
    paths = [source, package / "utils" / "eval_utils.py", package / "utils" / "config.py", package / "grasp.py"]
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def prepare_jobs(args):
    jobs, common_protocol, common_code = [], None, None
    for variant in args.variants:
        for split in args.splits:
            dump_dir = Path(args.prediction_root).resolve() / variant / split
            manifest = read_completed_manifest(dump_dir, variant, split)
            protocol = manifest["identity"]["protocol"]
            code = manifest["identity"]["inference_code_sha256"]
            if common_protocol is None:
                common_protocol, common_code = protocol, code
            if protocol != common_protocol or code != common_code:
                raise ValueError(f"Inference protocol/code differs across compared dumps: {dump_dir}")
            coverage = check_dumps(dump_dir, split, protocol["camera"], protocol["frame_stride"])
            if coverage != manifest.get("coverage"):
                raise ValueError(f"Prediction files changed since inference completed: {dump_dir}")
            print(f"[CHECK] {variant}/{split}: {coverage['frames']} frames; empty={coverage['empty_grasp_frames']}", flush=True)
            jobs.append((variant, split, dump_dir, manifest, coverage))
    return jobs


def comparison_rows(records):
    lookup = {(record["variant"], record["split"]): record for record in records}
    rows = []
    for split in dict.fromkeys(record["split"] for record in records):
        for treatment, control in (("none", "base"), ("foreground", "none"), ("anchor", "foreground"), ("anchor", "none")):
            if (treatment, split) not in lookup or (control, split) not in lookup:
                continue
            a, b = lookup[treatment, split], lookup[control, split]
            rows.append({"split": split, "comparison": f"{treatment} - {control}",
                         **{key + "_pp": a[key + "_percent"] - b[key + "_percent"] for key in ("ap", "ap_04", "ap_08")}})
    return rows


def write_tables(root, records):
    root = Path(root)
    columns = ("variant", "split", "camera", "frame_stride", "topk_views", "collision_thresh", "checkpoint_epoch",
               "ap_percent", "ap_04_percent", "ap_08_percent", "frames", "empty_grasp_frames", "checkpoint_path")
    rows = []
    for record in records:
        protocol = record["evaluation_identity"]["inference_identity"]["protocol"]
        rows.append({**record, **protocol, **record["coverage"]})
    with (root / "ap_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    deltas = comparison_rows(records)
    with (root / "ap_deltas.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "comparison", "ap_pp", "ap_04_pp", "ap_08_pp"))
        writer.writeheader()
        writer.writerows(deltas)
    write_json(root / "ap_summary.json", {"metric": "GraspNet precision averaged over ranks 1..50 and six frictions",
                                        "frictions": FRICTIONS, "results": records, "deltas_percentage_points": deltas})


def main(argv=None):
    args = parse_args(argv)
    jobs = prepare_jobs(args)
    if args.check_only:
        return
    import numpy as np
    from graspnetAPI import GraspNetEval

    fingerprint = evaluator_fingerprint(GraspNetEval)
    evaluators = {}
    # Check sampling API support for every split before expensive evaluation.
    for _, split, _, manifest, _ in jobs:
        if split not in evaluators:
            protocol = manifest["identity"]["protocol"]
            evaluator = GraspNetEval(root=args.dataset_root, camera=protocol["camera"], split=split)
            evaluator_method(evaluator, split, protocol["frame_stride"])
            evaluators[split] = evaluator
    records = []
    for variant, split, dump_dir, manifest, coverage in jobs:
        protocol = manifest["identity"]["protocol"]
        npy_path = dump_dir / f"ap_{split}_{protocol['camera']}.npy"
        json_path = npy_path.with_suffix(".json")
        identity = {"inference_identity": manifest["identity"], "coverage": coverage, "evaluator_source_sha256": fingerprint,
                    "dataset_root": str(Path(args.dataset_root).resolve()), "top_k": 50, "frictions": list(FRICTIONS),
                    "summary_code_sha256": sha256_file(Path(__file__).with_name("cva_depth_evaluation.py")),
                    "evaluation_code_sha256": sha256_file(__file__)}
        cached = read_json(json_path) if json_path.is_file() else None
        if (not args.force and cached and cached.get("evaluation_identity") == identity and npy_path.is_file()
                and cached.get("result_sha256") == sha256_file(npy_path)):
            record = cached
            print(f"[SKIP] Matching AP cached: {variant}/{split}", flush=True)
        else:
            print(f"[DEPTH AP] {variant}/{split} camera={protocol['camera']} stride={protocol['frame_stride']}", flush=True)
            start = time.perf_counter()
            result, returned_ap = call_evaluator(evaluators[split], split, dump_dir, args.num_workers, protocol["frame_stride"])
            metrics = summarize_accuracy(result, split, protocol["frame_stride"], returned_ap)
            np.save(npy_path, np.asarray(result, dtype=np.float64), allow_pickle=False)
            record = {"variant": variant, "split": split, "evaluation_identity": identity, "coverage": coverage,
                      "checkpoint_path": manifest["checkpoint"]["path"], "checkpoint_epoch": manifest["checkpoint"].get("epoch"),
                      "checkpoint_sha256": manifest["checkpoint"]["sha256"], "dump_dir": str(dump_dir),
                      "result_npy": str(npy_path), "result_sha256": sha256_file(npy_path), "num_workers": args.num_workers,
                      "elapsed_seconds": time.perf_counter() - start, **metrics}
            write_json(json_path, record)
        records.append(record)
        # Preserve completed comparisons even if a later job is interrupted.
        write_tables(args.prediction_root, records)
        print(f"[AP] {variant}/{split}: AP={record['ap_percent']:.3f}, "
              f"AP0.4={record['ap_04_percent']:.3f}, AP0.8={record['ap_08_percent']:.3f}", flush=True)
    print(f"[SAVE] {Path(args.prediction_root).resolve() / 'ap_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
