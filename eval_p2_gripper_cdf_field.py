#!/usr/bin/env python3
"""Strict official GraspNet evaluation for Base and four scratch P2 variants."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Mapping, Sequence, Tuple


def _consume_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--train_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    args = parser.parse_args()
    # Consume evaluator-specific CLI before importing the models package;
    # models/__init__.py imports utils.arguments and its global parser.
    sys.argv[:] = [sys.argv[0]]
    return args


ARGS = _consume_args()

import numpy as np
from graspnetAPI import GraspNetEval

from exact_action_cdf_common import atomic_save_json
from models.p2_gripper_cdf_field import P2_VARIANTS


MODES = ("base", *P2_VARIANTS)
SPLITS = ("test_seen", "test_similar", "test_novel")
FRICTION = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
INCREMENTAL_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("p2_0", "base", "P2-0 minus Base"),
    ("p2_a", "p2_0", "P2-A minus P2-0"),
    ("p2_b", "p2_a", "P2-B minus P2-A"),
    ("p2_c", "p2_b", "P2-C minus P2-B"),
)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_manifests(
    prediction_root: str,
    modes: Sequence[str],
    camera: str,
    sample_interval: int,
):
    expected_fraction = 1.0 if sample_interval == 1 else 1.0 / float(sample_interval)
    lineage = None
    manifests = {}
    for mode in modes:
        for split in SPLITS:
            path = os.path.join(
                prediction_root, mode, split, "_inference_complete.json"
            )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing inference manifest: {path}")
            payload = read_json(path)
            if str(payload.get("status")) != "complete":
                raise RuntimeError(f"Incomplete inference: {path}")
            for key, expected in {
                "mode": mode,
                "test_mode": split,
                "camera": camera,
            }.items():
                if str(payload.get(key)) != str(expected):
                    raise RuntimeError(
                        f"{path}: {key}={payload.get(key)!r}, expected={expected!r}"
                    )
            for key, expected in {
                "sample_interval": expected_fraction,
                "collision_thresh": 0.01,
                "collision_voxel_size": 0.01,
                "top_views": 1.0,
            }.items():
                actual = float(payload.get(key, float("nan")))
                if not np.isfinite(actual) or abs(actual - expected) > 1e-9:
                    raise RuntimeError(f"{path}: {key}={actual}, expected={expected}")
            current_lineage = str(
                payload.get("reference_base_checkpoint_sha256", "")
            )
            if not current_lineage:
                raise RuntimeError(f"{path}: missing Base checkpoint lineage")
            if lineage is None:
                lineage = current_lineage
            elif current_lineage != lineage:
                raise RuntimeError("P2 inference manifests mix Base checkpoint lineages")
            if mode == "base":
                if bool(payload.get("scratch_three_layer_mlp", False)):
                    raise RuntimeError(f"{path}: Base unexpectedly declares P2 MLP")
            else:
                if not bool(payload.get("scratch_three_layer_mlp", False)):
                    raise RuntimeError(f"{path}: P2 mode is not a scratch MLP")
                if bool(payload.get("uses_p1_checkpoint", True)):
                    raise RuntimeError(f"{path}: P2 mode consumes a P1 checkpoint")
                if bool(payload.get("uses_stage1_or_p1_residual", True)):
                    raise RuntimeError(f"{path}: residual-on-Stage1/P1 is forbidden")
            manifests[f"{mode}/{split}"] = payload
    return manifests, lineage


def evaluate_one(dataset_root, dump_dir, camera, split, interval, workers):
    evaluator = GraspNetEval(root=dataset_root, camera=camera, split=split)
    ratio = 1.0 / float(interval)
    if split == "test_seen":
        result, _ = evaluator.eval_seen(
            dump_dir, anno_sample_ratio=ratio, proc=workers
        )
    elif split == "test_similar":
        result, _ = evaluator.eval_similar(
            dump_dir, anno_sample_ratio=ratio, proc=workers
        )
    else:
        result, _ = evaluator.eval_novel(
            dump_dir, anno_sample_ratio=ratio, proc=workers
        )
    array = np.asarray(result, dtype=np.float64)
    if array.ndim != 4 or array.shape[-1] != len(FRICTION):
        raise RuntimeError(f"Unexpected official AP array shape {array.shape}")
    metrics = {
        "AP": float(array.mean()),
        "AP0.4": float(array[..., 1].mean()),
        "AP0.8": float(array[..., 3].mean()),
    }
    for index, friction in enumerate(FRICTION):
        metrics[f"AP@{friction:.1f}"] = float(array[..., index].mean())
    return array, metrics


def load_training_summary(train_root: str, variant: str):
    best_path = os.path.join(train_root, variant, "best.json")
    contract_path = os.path.join(train_root, variant, "probe_contract.json")
    if not os.path.isfile(best_path) or not os.path.isfile(contract_path):
        raise FileNotFoundError(f"Missing training summary for {variant}")
    best = read_json(best_path)
    contract = read_json(contract_path)
    if str(contract.get("variant")) != variant:
        raise RuntimeError(f"Training contract variant mismatch for {variant}")
    if bool(contract.get("uses_p1_checkpoint", True)):
        raise RuntimeError(f"{variant}: training contract consumed a P1 checkpoint")
    if bool(contract.get("uses_stage1_or_p1_residual", True)):
        raise RuntimeError(f"{variant}: training contract is residual-based")
    predictor = contract.get("predictor_contract", {})
    if int(predictor.get("num_linear_layers", -1)) != 3:
        raise RuntimeError(f"{variant}: predictor is not a three-layer MLP")
    return {"best": best, "contract": contract}


def write_csv(path: str, rows: List[Mapping[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_scene_statistics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, float]:
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError(
            f"Paired arrays must share [scene,anno,rank,friction], got "
            f"{reference.shape}/{candidate.shape}"
        )
    reference_scene = reference.mean(axis=(1, 2, 3))
    candidate_scene = candidate.mean(axis=(1, 2, 3))
    delta = candidate_scene - reference_scene
    result = {
        "num_scenes": int(delta.size),
        "mean_delta_ap": float(delta.mean()),
        "median_delta_ap": float(np.median(delta)),
        "improved_scenes": int((delta > 0.0).sum()),
        "tied_scenes": int(np.isclose(delta, 0.0, atol=1e-12).sum()),
        "degraded_scenes": int((delta < 0.0).sum()),
    }
    samples = max(0, int(bootstrap_samples))
    if samples > 0 and delta.size > 0:
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, delta.size, size=(samples, delta.size))
        means = delta[indices].mean(axis=1)
        result["bootstrap_ci95_low"] = float(np.quantile(means, 0.025))
        result["bootstrap_ci95_high"] = float(np.quantile(means, 0.975))
    else:
        result["bootstrap_ci95_low"] = float("nan")
        result["bootstrap_ci95_high"] = float("nan")
    return result


def main():
    args = ARGS
    if args.sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    arrays_dir = os.path.join(output_dir, "ap_arrays")
    os.makedirs(arrays_dir, exist_ok=True)

    manifests, lineage = validate_manifests(
        args.prediction_root,
        MODES,
        args.camera,
        int(args.sample_interval),
    )
    training = {
        variant: load_training_summary(args.train_root, variant)
        for variant in P2_VARIANTS
    }
    parameter_counts = {
        int(training[variant]["contract"]["trainable_parameter_count"])
        for variant in P2_VARIANTS
    }
    if len(parameter_counts) != 1:
        raise RuntimeError(
            f"P2 variants are not capacity matched: parameter_counts={parameter_counts}"
        )
    common_parameter_count = next(iter(parameter_counts))

    metrics_by_mode: Dict[str, Dict[str, Dict[str, float]]] = {}
    arrays_by_mode: Dict[str, Dict[str, np.ndarray]] = {}
    long_rows = []
    for mode in MODES:
        metrics_by_mode[mode] = {}
        arrays_by_mode[mode] = {}
        for split in SPLITS:
            dump = os.path.join(args.prediction_root, mode, split)
            array, metrics = evaluate_one(
                args.dataset_root,
                dump,
                args.camera,
                split,
                int(args.sample_interval),
                int(args.num_workers),
            )
            np.save(
                os.path.join(arrays_dir, f"{mode}_{split}_{args.camera}.npy"),
                array,
            )
            metrics_by_mode[mode][split] = metrics
            arrays_by_mode[mode][split] = array
            for metric, value in metrics.items():
                long_rows.append(
                    {
                        "mode": mode,
                        "split": split,
                        "metric": metric,
                        "value": value,
                    }
                )
            print(f"[P2-EVAL] {mode}/{split}: {metrics}", flush=True)

    for mode in MODES:
        mean_metrics = {}
        for metric in metrics_by_mode[mode][SPLITS[0]]:
            mean_metrics[metric] = float(
                np.mean(
                    [metrics_by_mode[mode][split][metric] for split in SPLITS]
                )
            )
            long_rows.append(
                {
                    "mode": mode,
                    "split": "mean",
                    "metric": metric,
                    "value": mean_metrics[metric],
                }
            )
        metrics_by_mode[mode]["mean"] = mean_metrics

    delta_rows = []
    for candidate, reference, label in INCREMENTAL_PAIRS:
        for split in (*SPLITS, "mean"):
            row = {
                "comparison": label,
                "candidate": candidate,
                "reference": reference,
                "split": split,
            }
            for metric in ("AP", "AP0.4", "AP0.8"):
                row[f"{metric}_reference"] = metrics_by_mode[reference][split][metric]
                row[f"{metric}_candidate"] = metrics_by_mode[candidate][split][metric]
                row[f"delta_{metric}"] = (
                    metrics_by_mode[candidate][split][metric]
                    - metrics_by_mode[reference][split][metric]
                )
            delta_rows.append(row)

    vs_base_rows = []
    for mode in P2_VARIANTS:
        for split in (*SPLITS, "mean"):
            vs_base_rows.append(
                {
                    "mode": mode,
                    "split": split,
                    "base_AP": metrics_by_mode["base"][split]["AP"],
                    "mode_AP": metrics_by_mode[mode][split]["AP"],
                    "delta_AP": metrics_by_mode[mode][split]["AP"]
                    - metrics_by_mode["base"][split]["AP"],
                }
            )

    scene_rows = []
    for comparison_index, (candidate, reference, label) in enumerate(
        INCREMENTAL_PAIRS
    ):
        for split_index, split in enumerate(SPLITS):
            stats = paired_scene_statistics(
                arrays_by_mode[reference][split],
                arrays_by_mode[candidate][split],
                bootstrap_samples=int(args.bootstrap_samples),
                seed=(
                    int(args.bootstrap_seed)
                    + comparison_index * 1009
                    + split_index * 9176
                ),
            )
            scene_rows.append(
                {
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "split": split,
                    **stats,
                }
            )

    write_csv(os.path.join(output_dir, "p2_official_ap_long.csv"), long_rows)
    write_csv(os.path.join(output_dir, "p2_incremental_summary.csv"), delta_rows)
    write_csv(os.path.join(output_dir, "p2_delta_vs_base.csv"), vs_base_rows)
    write_csv(
        os.path.join(output_dir, "p2_paired_scene_summary.csv"), scene_rows
    )

    base_mean = metrics_by_mode["base"]["mean"]["AP"]
    mean_deltas = {
        mode: metrics_by_mode[mode]["mean"]["AP"] - base_mean
        for mode in P2_VARIANTS
    }
    best_mode = max(mean_deltas, key=mean_deltas.get)
    if mean_deltas[best_mode] > 0:
        decision = "scratch_cdf_mlp_improves_over_base"
    else:
        decision = "no_scratch_p2_variant_improves_over_base"

    summary = {
        "protocol": "P2-capacity-matched-three-layer-scratch-CDF-v2",
        "reference_base_checkpoint_sha256": lineage,
        "sample_interval": int(args.sample_interval),
        "camera": args.camera,
        "collision_thresh": 0.01,
        "modes": list(MODES),
        "metrics": metrics_by_mode,
        "deltas_vs_base_mean_ap": mean_deltas,
        "best_p2_variant": best_mode,
        "decision": decision,
        "common_trainable_parameter_count": common_parameter_count,
        "training": training,
        "incremental_pairs": [list(item) for item in INCREMENTAL_PAIRS],
        "paired_scene_statistics": scene_rows,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "manifests": manifests,
    }
    atomic_save_json(summary, os.path.join(output_dir, "p2_summary.json"))

    lines = [
        "# P2 Capacity-Matched Scratch CDF Report",
        "",
        f"- Decision: `{decision}`",
        f"- Best P2 variant: `{best_mode}`",
        f"- Common trainable parameter count: `{common_parameter_count}`",
        f"- Sample interval: `{args.sample_interval}`",
        "- Test-time collision threshold: `0.01`",
        "- All variants: same three-Linear-layer MLP, random scratch initialization",
        "- Only loss: ordinary unbalanced exact-action CDF BCE",
        "- No P1 checkpoint and no residual-on-Stage1/P1 prediction",
        "",
        "## Official AP",
        "",
        "| Mode | Seen | Similar | Novel | Mean | Δ Mean vs Base |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        values = [100.0 * metrics_by_mode[mode][split]["AP"] for split in SPLITS]
        mean = 100.0 * metrics_by_mode[mode]["mean"]["AP"]
        delta = 100.0 * (metrics_by_mode[mode]["mean"]["AP"] - base_mean)
        lines.append(
            f"| {mode} | {values[0]:.3f} | {values[1]:.3f} | "
            f"{values[2]:.3f} | {mean:.3f} | {delta:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Incremental representation gains",
            "",
            "| Comparison | Seen ΔAP | Similar ΔAP | Novel ΔAP | Mean ΔAP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate, reference, label in INCREMENTAL_PAIRS:
        deltas = [
            100.0
            * (
                metrics_by_mode[candidate][split]["AP"]
                - metrics_by_mode[reference][split]["AP"]
            )
            for split in (*SPLITS, "mean")
        ]
        lines.append(
            f"| {label} | {deltas[0]:+.3f} | {deltas[1]:+.3f} | "
            f"{deltas[2]:+.3f} | {deltas[3]:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Validation-cache fit",
            "",
            "| Variant | Best epoch | Base BCE | P2 BCE | Δ BCE | Base utility MAE | P2 utility MAE | Base regret | P2 regret | Parameters |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in P2_VARIANTS:
        best = training[variant]["best"]
        val = best.get("val_metrics", {})
        contract = training[variant]["contract"]
        base_loss = float(val.get("base_loss", float("nan")))
        p2_loss = float(val.get("loss", float("nan")))
        lines.append(
            f"| {variant} | {int(best.get('best_epoch', -1))} | "
            f"{base_loss:.6f} | {p2_loss:.6f} | {p2_loss-base_loss:+.6f} | "
            f"{float(val.get('base_utility_mae', float('nan'))):.6f} | "
            f"{float(val.get('utility_mae', float('nan'))):.6f} | "
            f"{float(val.get('base_selected_regret', float('nan'))):.6f} | "
            f"{float(val.get('selected_regret', float('nan'))):.6f} | "
            f"{int(contract.get('trainable_parameter_count', -1))} |"
        )

    lines.extend(
        [
            "",
            "## Paired scene-level stability",
            "",
            "| Comparison | Split | Mean ΔAP | 95% bootstrap CI | Improved | Degraded |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in scene_rows:
        lines.append(
            f"| {row['comparison']} | {row['split']} | "
            f"{100.0*float(row['mean_delta_ap']):+.3f} | "
            f"[{100.0*float(row['bootstrap_ci95_low']):+.3f}, "
            f"{100.0*float(row['bootstrap_ci95_high']):+.3f}] | "
            f"{int(row['improved_scenes'])} | {int(row['degraded_scenes'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `P2-0 - Base` isolates the effect of replacing the original linear CDF head with the common nonlinear three-layer MLP trained on exact-action labels.",
            "- `P2-A - P2-0` adds explicit metric center/view/angle/depth/width action encoding.",
            "- `P2-B - P2-A` adds projected DPT evidence from left finger, right finger, closing, palm, and approach regions.",
            "- `P2-C - P2-B` adds signed predicted-depth residual statistics at the same gripper support points.",
            "",
            "All four P2 variants instantiate the same fixed-width input layout and identical three-layer MLP parameter count; inactive evidence blocks are hard-masked to zero. The study therefore controls head architecture and nominal capacity more tightly than the earlier residual-adapter design.",
            "",
            "P2 changes only action scoring. It does not alter center proposals, view proposals, width generation, Top-1 policy, or the proposal-recall ceiling identified by P0-E.",
        ]
    )
    report_path = os.path.join(output_dir, "P2_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(f"[SAVE] {report_path}", flush=True)


if __name__ == "__main__":
    main()
