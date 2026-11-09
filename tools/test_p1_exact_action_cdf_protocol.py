#!/usr/bin/env python3
"""Synthetic tests for the P1 report/protocol helpers."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eval_p1_exact_action_cdf.py"
spec = importlib.util.spec_from_file_location("eval_p1", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_result_summary() -> None:
    result = np.zeros((2, 3, 4, 6), dtype=np.float64)
    result[..., 1] = 0.4
    result[..., 3] = 0.8
    metrics = module.summarize_result_array(result)
    assert abs(metrics["ap0.4"] - 0.4) < 1e-12
    assert abs(metrics["ap0.8"] - 0.8) < 1e-12
    expected_mean = (0.4 + 0.8) / 6.0
    assert abs(metrics["ap"] - expected_mean) < 1e-12
    record = module.ApRecord(
        mode="base",
        split="test_seen",
        ap=metrics["ap"],
        ap04=metrics["ap0.4"],
        ap08=metrics["ap0.8"],
        evaluator_returned_ap=metrics["ap"],
        result_shape=tuple(result.shape),
        dump_dir="/tmp/base",
        result_npy="/tmp/base.npy",
    ).to_dict()
    assert record["ap0.4"] == metrics["ap0.4"]
    assert record["ap0.8"] == metrics["ap0.8"]


def test_status() -> None:
    assert module.determine_learnability_status(
        val_loss_gain=0.01, official_mean_ap_delta=0.001
    ) == "learnable_and_transfers_to_official_ap"
    assert module.determine_learnability_status(
        val_loss_gain=0.01, official_mean_ap_delta=-0.001
    ) == "locally_learnable_without_positive_official_ap_transfer"
    assert module.determine_learnability_status(
        val_loss_gain=0.0, official_mean_ap_delta=0.01
    ) == "learnability_not_demonstrated"


def test_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = {
            "status": "complete",
            "mode": "exact",
            "test_mode": "test_seen",
            "camera": "realsense",
            "geometry_depth_source": "pred",
            "top_views": 1,
            "sample_interval": 0.1,
            "collision_thresh": 0.01,
            "collision_voxel_size": 0.01,
            "processed_samples": 30 * 26,
            "reference_base_checkpoint_sha256": "abc",
            "exact_action_metadata": {
                "head_only_update": True,
                "updated_state_keys": [
                    "x.decoder.cdf_head.weight",
                    "x.decoder.cdf_head.bias",
                ],
            },
        }
        (root / "_inference_complete.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        loaded = module._validate_inference_manifest(
            root,
            mode="exact",
            split="test_seen",
            camera="realsense",
            sample_interval=10,
            expected_collision_thresh=0.01,
            expected_collision_voxel_size=0.01,
        )
        assert loaded["mode"] == "exact"


def test_report() -> None:
    delta = {
        "split": "mean",
        "base_ap": 0.45,
        "exact_ap": 0.47,
        "delta_ap": 0.02,
        "base_ap04": 0.35,
        "exact_ap04": 0.37,
        "delta_ap04": 0.02,
        "base_ap08": 0.55,
        "exact_ap08": 0.57,
        "delta_ap08": 0.02,
    }
    summary = {
        "protocol": {
            "sample_interval": 10,
            "collision_thresh": 0.01,
            "collision_voxel_size": 0.01,
        },
        "training_diagnostics": {
            "best_epoch": 3,
            "base_val": {"loss": 0.2},
            "best_val": {"loss": 0.1},
            "val_loss_gain": 0.1,
        },
        "delta_rows": [delta],
        "decision": {"status": "learnable_and_transfers_to_official_ap"},
    }
    report = module.build_markdown_report(summary)
    assert "Exact-CDF AP" in report
    assert "learnable_and_transfers_to_official_ap" in report


def main() -> None:
    test_result_summary()
    test_status()
    test_manifest()
    test_report()
    print("P1 protocol tests passed (4/4).")


if __name__ == "__main__":
    main()
