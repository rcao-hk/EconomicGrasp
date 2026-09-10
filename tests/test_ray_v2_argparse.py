"""Regression test: --v2_* and --ray_* flags must precede legacy parse_args()."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_v2_training_args_are_consumed_before_legacy_parser():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import sys
sys.argv = [
    "train_ray_grasp_v2.py",
    "--dataset_root", "/tmp/graspnet",
    "--batch_size", "1",
    "--v2_train_sample_interval", "0.1",
    "--v2_eval_sample_interval", "0.1",
    "--v2_target_temperature", "0.1",
    "--v2_listwise_weight", "1.0",
    "--v2_calibration_weight", "0.5",
    "--v2_max_batches", "0",
]
from utils.ray_grasp_v2_runtime import parse_v2_cli
v2, ray, cfg = parse_v2_cli(training=True)
assert v2.v2_train_sample_interval == 0.1
assert v2.v2_eval_sample_interval == 0.1
assert v2.v2_target_temperature == 0.1
assert v2.v2_listwise_weight == 1.0
assert v2.v2_calibration_weight == 0.5
assert v2.v2_max_batches == 0
assert cfg.dataset_root == "/tmp/graspnet"
assert cfg.batch_size == 1
assert all(not token.startswith("--v2_") for token in sys.argv)
assert all(not token.startswith("--ray_") for token in sys.argv)
print("P2_V2_ARGPARSE_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "P2_V2_ARGPARSE_OK" in result.stdout
