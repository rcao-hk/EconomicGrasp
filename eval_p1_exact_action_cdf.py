#!/usr/bin/env python3
"""Strict Base-vs-Exact official evaluation for the P1 CDF-only probe."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

MODES: Tuple[str, ...] = ("base", "exact")
SPLITS: Tuple[str, ...] = ("test_seen", "test_similar", "test_novel")


@dataclass(frozen=True)
class ApRecord:
    mode: str
    split: str
    ap: float
    ap04: float
    ap08: float
    evaluator_returned_ap: float
    result_shape: Tuple[int, ...]
    dump_dir: str
    result_npy: str

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["result_shape"] = list(self.result_shape)
        row["ap0.4"] = self.ap04
        row["ap0.8"] = self.ap08
        row["ap_percent"] = 100.0 * self.ap
        row["ap0.4_percent"] = 100.0 * self.ap04
        row["ap0.8_percent"] = 100.0 * self.ap08
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--expected_collision_thresh", type=float, default=0.01)
    parser.add_argument("--expected_collision_voxel_size", type=float, default=0.01)
    parser.add_argument("--skip_complete_check", action="store_true")
    parser.add_argument("--skip_manifest_check", action="store_true")
    parser.add_argument("--overwrite_ap_arrays", action="store_true")
    return parser.parse_args()


def _json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _scene_ids(split: str) -> range:
    bounds = {
        "test_seen": (100, 130),
        "test_similar": (130, 160),
        "test_novel": (160, 190),
    }
    if split not in bounds:
        raise ValueError(f"Unsupported split {split!r}")
    return range(*bounds[split])


def _annos(interval: int) -> range:
    if interval <= 0:
        raise ValueError("sample_interval must be positive")
    return range(0, 256, interval)


def _prediction_paths(
    root: Path, split: str, camera: str, interval: int
) -> Iterable[Path]:
    for scene_id in _scene_ids(split):
        for anno_id in _annos(interval):
            yield root / f"scene_{scene_id:04d}" / camera / f"{anno_id:04d}.npy"


def _check_complete(
    root: Path, split: str, camera: str, interval: int
) -> Dict[str, int]:
    expected_paths = list(_prediction_paths(root, split, camera, interval))
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete P1 dump {root}: missing {len(missing)}/{len(expected_paths)}; "
            f"examples={missing[:20]}"
        )
    actual = sum(
        1
        for scene_id in _scene_ids(split)
        for path in (root / f"scene_{scene_id:04d}" / camera).glob("[0-9][0-9][0-9][0-9].npy")
        if path.is_file()
    )
    return {"expected_files": len(expected_paths), "actual_prediction_files": actual}


def _close(name: str, value: Any, expected: float, atol: float = 1e-9) -> None:
    actual = float(value)
    if not math.isfinite(actual) or abs(actual - expected) > atol:
        raise RuntimeError(f"{name}={actual!r}; expected={expected!r}")


def _validate_inference_manifest(
    dump_dir: Path,
    *,
    mode: str,
    split: str,
    camera: str,
    sample_interval: int,
    expected_collision_thresh: float,
    expected_collision_voxel_size: float,
) -> Dict[str, Any]:
    path = dump_dir / "_inference_complete.json"
    manifest = _json(path)
    expected_text = {
        "status": "complete",
        "mode": mode,
        "test_mode": split,
        "camera": camera,
        "geometry_depth_source": "pred",
    }
    for key, expected in expected_text.items():
        if str(manifest.get(key)) != expected:
            raise RuntimeError(
                f"{path}: {key}={manifest.get(key)!r}; expected={expected!r}"
            )
    if int(manifest.get("top_views", -1)) != 1:
        raise RuntimeError(f"{path}: P1 must use Top-1")
    fraction = 1.0 if sample_interval == 1 else 1.0 / sample_interval
    _close(f"{path}: sample_interval", manifest.get("sample_interval"), fraction, 1e-8)
    _close(
        f"{path}: collision_thresh",
        manifest.get("collision_thresh"),
        expected_collision_thresh,
        1e-10,
    )
    _close(
        f"{path}: collision_voxel_size",
        manifest.get("collision_voxel_size"),
        expected_collision_voxel_size,
        1e-10,
    )
    expected_samples = len(_scene_ids(split)) * len(_annos(sample_interval))
    if int(manifest.get("processed_samples", -1)) != expected_samples:
        raise RuntimeError(
            f"{path}: processed_samples={manifest.get('processed_samples')}; "
            f"expected={expected_samples}"
        )

    probe = manifest.get("exact_action_metadata", {})
    if mode == "base":
        if probe not in ({}, None):
            raise RuntimeError(f"{path}: Base contains exact-action metadata")
    else:
        if not isinstance(probe, Mapping) or not bool(probe.get("head_only_update")):
            raise RuntimeError(f"{path}: Exact is not a declared head-only update")
        updated = list(probe.get("updated_state_keys", []))
        if len(updated) != 2 or not all(
            "decoder.cdf_head." in str(key) for key in updated
        ):
            raise RuntimeError(f"{path}: invalid updated_state_keys={updated}")
    return manifest


def summarize_result_array(result: np.ndarray) -> Dict[str, float]:
    result = np.asarray(result, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] < 4:
        raise ValueError(f"Unexpected GraspNet result shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("GraspNet result contains NaN/Inf")
    return {
        "ap": float(result.mean()),
        "ap0.4": float(result[..., 1].mean()),
        "ap0.8": float(result[..., 3].mean()),
    }


def _official_eval(
    dataset_root: str,
    camera: str,
    split: str,
    dump_dir: Path,
    num_workers: int,
    interval: int,
) -> Tuple[np.ndarray, float]:
    try:
        from graspnetAPI import GraspNetEval
    except ImportError as exc:
        raise RuntimeError("Activate the environment containing graspnetAPI") from exc

    evaluator = GraspNetEval(root=dataset_root, camera=camera, split=split)
    method = {
        "test_seen": evaluator.eval_seen,
        "test_similar": evaluator.eval_similar,
        "test_novel": evaluator.eval_novel,
    }[split]
    if interval == 1:
        try:
            result, returned = method(str(dump_dir), anno_sample_ratio=1.0, proc=num_workers)
        except TypeError as exc:
            if "anno_sample_ratio" not in str(exc):
                raise
            result, returned = method(str(dump_dir), proc=num_workers)
    else:
        try:
            result, returned = method(
                str(dump_dir), anno_sample_ratio=1.0 / interval, proc=num_workers
            )
        except TypeError as exc:
            raise RuntimeError(
                "Sampled evaluation requires the repository GraspNetAPI fork with "
                "anno_sample_ratio; otherwise use --sample_interval 1."
            ) from exc
    returned_array = np.asarray(returned, dtype=np.float64)
    return np.asarray(result, dtype=np.float64), float(returned_array.mean())


def _training_diagnostics(train_dir: Path) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (train_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or int(rows[0].get("epoch", -1)) != 0:
        raise RuntimeError("metrics.jsonl does not start with the epoch-0 Base row")
    base_val = rows[0].get("val")
    best = _json(train_dir / "best.json")
    best_val = best.get("val_metrics")
    contract = _json(train_dir / "probe_contract.json")
    if not isinstance(base_val, Mapping) or not isinstance(best_val, Mapping):
        raise RuntimeError("Invalid Base/Best validation metrics")
    if contract.get("objective") != "ordinary_unbalanced_bce_on_exact_evaluator_cdf":
        raise RuntimeError("Training objective is not the required CDF-only BCE")
    trainable = list(contract.get("trainable_parameters", []))
    if trainable != ["cdf_head.weight", "cdf_head.bias"]:
        raise RuntimeError(f"Unexpected trainable parameters {trainable}")

    def finite(mapping: Mapping[str, Any], key: str) -> float | None:
        value = mapping.get(key)
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    base_loss, best_loss = finite(base_val, "loss"), finite(best_val, "loss")
    if base_loss is None or best_loss is None:
        raise RuntimeError("Missing finite validation CDF BCE")
    output: Dict[str, Any] = {
        "best_epoch": int(best.get("best_epoch", -1)),
        "base_val": dict(base_val),
        "best_val": dict(best_val),
        "val_loss_gain": base_loss - best_loss,
        "objective": contract["objective"],
        "trainable_parameters": trainable,
        "probe_contract": contract,
    }
    for key in ("utility_mae", "selected_regret", "selected_invalid"):
        base_value, best_value = finite(base_val, key), finite(best_val, key)
        if base_value is not None and best_value is not None:
            output[f"{key}_gain"] = base_value - best_value
    return output


def determine_learnability_status(
    *, val_loss_gain: float, official_mean_ap_delta: float, tolerance: float = 1e-10
) -> str:
    if val_loss_gain > tolerance and official_mean_ap_delta > tolerance:
        return "learnable_and_transfers_to_official_ap"
    if val_loss_gain > tolerance:
        return "locally_learnable_without_positive_official_ap_transfer"
    return "learnability_not_demonstrated"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def build_markdown_report(summary: Mapping[str, Any]) -> str:
    training = summary["training_diagnostics"]
    status = summary["decision"]["status"]
    lines = [
        "# P1 Exact-Action CDF Utility Learnability Report",
        "",
        "## Protocol",
        "",
        "Base and Exact execute the same Stage-1 RGB-only graph. Only the existing "
        "monotonic CDF head is updated with ordinary threshold-wise CDF BCE. No "
        "ranking, hard-negative, auxiliary, collision, width, utility-regression, "
        "or KD loss is used.",
        "",
        f"- Test sample interval: `{summary['protocol']['sample_interval']}`",
        f"- Collision threshold/voxel: `{summary['protocol']['collision_thresh']}` / "
        f"`{summary['protocol']['collision_voxel_size']}`",
        "- Grasp-view policy: `Top-1`",
        "",
        "## Cached validation learnability",
        "",
        f"- Best epoch: **{training['best_epoch']}**",
        f"- Base / best CDF BCE: **{training['base_val']['loss']:.8f}** / "
        f"**{training['best_val']['loss']:.8f}**",
        f"- Validation CDF BCE gain: **{training['val_loss_gain']:+.8f}**",
    ]
    for key, label in (
        ("utility_mae_gain", "Utility-MAE gain"),
        ("selected_regret_gain", "Center-wise selection-regret gain"),
        ("selected_invalid_gain", "Selected-invalid-rate gain"),
    ):
        if key in training:
            lines.append(f"- {label}: **{training[key]:+.8f}**")
    lines += [
        "",
        "## Official GraspNet AP",
        "",
        "| Split | Base AP | Exact-CDF AP | ΔAP | Base AP@0.4 | Exact AP@0.4 | "
        "ΔAP@0.4 | Base AP@0.8 | Exact AP@0.8 | ΔAP@0.8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["delta_rows"]:
        lines.append(
            "| {split} | {base_ap:.4f} | {exact_ap:.4f} | {delta_ap:+.4f} | "
            "{base_ap04:.4f} | {exact_ap04:.4f} | {delta_ap04:+.4f} | "
            "{base_ap08:.4f} | {exact_ap08:.4f} | {delta_ap08:+.4f} |".format(**row)
        )
    interpretations = {
        "learnable_and_transfers_to_official_ap":
            "The frozen Stage-1 feature contains linearly decodable evaluator-aligned "
            "action utility, and the learned CDF improves paired official AP.",
        "locally_learnable_without_positive_official_ap_transfer":
            "The exact-action CDF is decodable on the fixed cache, but the gain does "
            "not transfer to official AP. Diagnose cache/admission mismatch before "
            "adding ranking losses.",
        "learnability_not_demonstrated":
            "The head-only probe does not demonstrate learnability. Test a richer "
            "gripper-conditioned representation before adding ranking or hard-negative loss.",
    }
    lines += [
        "",
        "## Decision",
        "",
        f"**{status}**",
        "",
        interpretations[status],
        "",
        "## Interpretation boundary",
        "",
        "P1 measures linear decodability and official-AP transfer for fixed Top-1-view "
        "Student actions. It does not test Top-4 hypotheses, proposal transfer, local "
        "pose refinement, or a learned gripper-conditioned field.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.sample_interval <= 0 or args.num_workers <= 0:
        raise ValueError("sample_interval and num_workers must be positive")
    prediction_root = Path(args.prediction_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ap_dir = output_dir / "ap_arrays"
    ap_dir.mkdir(exist_ok=True)

    records: List[ApRecord] = []
    manifests: Dict[str, Dict[str, Any]] = {}
    reference_sha: str | None = None
    for mode in MODES:
        for split in SPLITS:
            dump = prediction_root / mode / split
            if not dump.is_dir():
                raise FileNotFoundError(dump)
            completeness = (
                {} if args.skip_complete_check else
                _check_complete(dump, split, args.camera, args.sample_interval)
            )
            if not args.skip_manifest_check:
                manifest = _validate_inference_manifest(
                    dump,
                    mode=mode,
                    split=split,
                    camera=args.camera,
                    sample_interval=args.sample_interval,
                    expected_collision_thresh=args.expected_collision_thresh,
                    expected_collision_voxel_size=args.expected_collision_voxel_size,
                )
                current_sha = str(manifest.get("reference_base_checkpoint_sha256", ""))
                if not current_sha:
                    raise RuntimeError(f"{dump}: missing reference Base SHA256")
                if reference_sha is None:
                    reference_sha = current_sha
                elif current_sha != reference_sha:
                    raise RuntimeError("Paired manifests use different Base checkpoints")
                manifests[f"{mode}/{split}"] = {**manifest, **completeness}

            print(f"[P1][OFFICIAL] {mode}/{split}: {dump}", flush=True)
            result, returned = _official_eval(
                args.dataset_root,
                args.camera,
                split,
                dump,
                args.num_workers,
                args.sample_interval,
            )
            metric = summarize_result_array(result)
            result_path = ap_dir / f"{mode}_{split}_{args.camera}.npy"
            if result_path.exists() and not args.overwrite_ap_arrays:
                old = np.load(result_path)
                if old.shape != result.shape or not np.array_equal(old, result):
                    raise FileExistsError(
                        f"Different AP array exists at {result_path}; pass "
                        "--overwrite_ap_arrays to replace it."
                    )
            else:
                np.save(result_path, result)
            records.append(
                ApRecord(
                    mode, split, metric["ap"], metric["ap0.4"], metric["ap0.8"],
                    returned, tuple(result.shape), str(dump), str(result_path)
                )
            )

    lookup = {(row.mode, row.split): row for row in records}
    deltas: List[Dict[str, Any]] = []
    for split in SPLITS:
        base, exact = lookup[("base", split)], lookup[("exact", split)]
        deltas.append({
            "split": split,
            "base_ap": base.ap, "exact_ap": exact.ap, "delta_ap": exact.ap-base.ap,
            "base_ap04": base.ap04, "exact_ap04": exact.ap04,
            "delta_ap04": exact.ap04-base.ap04,
            "base_ap08": base.ap08, "exact_ap08": exact.ap08,
            "delta_ap08": exact.ap08-base.ap08,
        })
    mean = {"split": "mean"}
    for key in (
        "base_ap", "exact_ap", "delta_ap", "base_ap04", "exact_ap04",
        "delta_ap04", "base_ap08", "exact_ap08", "delta_ap08"
    ):
        mean[key] = float(np.mean([row[key] for row in deltas]))
    deltas.append(mean)

    training = _training_diagnostics(Path(args.train_dir).resolve())
    status = determine_learnability_status(
        val_loss_gain=float(training["val_loss_gain"]),
        official_mean_ap_delta=float(mean["delta_ap"]),
    )
    summary: Dict[str, Any] = {
        "protocol": {
            "name": "P1-exact-action-CDF-head-only-v1",
            "camera": args.camera,
            "sample_interval": args.sample_interval,
            "collision_thresh": args.expected_collision_thresh,
            "collision_voxel_size": args.expected_collision_voxel_size,
            "top_views": 1,
            "loss": "ordinary_unbalanced_thresholdwise_CDF_BCE_only",
        },
        "training_diagnostics": training,
        "official_records": [row.to_dict() for row in records],
        "delta_rows": deltas,
        "decision": {
            "status": status,
            "val_loss_gain": training["val_loss_gain"],
            "official_mean_ap_delta": mean["delta_ap"],
        },
        "inference_manifests": manifests,
    }

    long_rows = [row.to_dict() for row in records]
    _write_csv(
        output_dir / "p1_official_ap_long.csv",
        long_rows,
        ("mode", "split", "ap", "ap0.4", "ap0.8", "ap_percent",
         "ap0.4_percent", "ap0.8_percent", "evaluator_returned_ap",
         "result_shape", "dump_dir", "result_npy"),
    )
    _write_csv(
        output_dir / "p1_delta_summary.csv",
        deltas,
        ("split", "base_ap", "exact_ap", "delta_ap", "base_ap04",
         "exact_ap04", "delta_ap04", "base_ap08", "exact_ap08", "delta_ap08"),
    )
    (output_dir / "p1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = build_markdown_report(summary)
    (output_dir / "P1_REPORT.md").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"[P1][SAVE] {output_dir / 'p1_summary.json'}", flush=True)
    print(f"[P1][SAVE] {output_dir / 'P1_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
