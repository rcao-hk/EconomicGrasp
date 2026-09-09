"""Regression test for the P2/legacy argparse import-order bug."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_training_ray_args_are_consumed_before_legacy_parser():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import sys
sys.argv = [
    "train_ray_grasp.py",
    "--dataset_root", "/tmp/graspnet",
    "--batch_size", "1",
    "--ray_train_sample_interval", "0.1",
    "--ray_eval_sample_interval", "0.1",
    "--ray_max_batches", "0",
    "--ray_support_weight", "1.0",
]
from utils.ray_grasp_runtime import parse_cli
args, cfg = parse_cli(training=True)
assert args.ray_train_sample_interval == 0.1
assert args.ray_eval_sample_interval == 0.1
assert args.ray_max_batches == 0
assert args.ray_support_weight == 1.0
assert cfg.dataset_root == "/tmp/graspnet"
assert cfg.batch_size == 1
assert all(not token.startswith("--ray_") for token in sys.argv)
print("P2_ARGPARSE_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "P2_ARGPARSE_OK" in result.stdout
