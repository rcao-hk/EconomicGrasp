from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_p5_args_are_consumed_before_legacy_parser():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import sys
sys.argv = [
    "train_p5.py",
    "--dataset_root", "/tmp/graspnet",
    "--batch_size", "1",
    "--p5_train_sample_interval", "0.1",
    "--p5_eval_sample_interval", "0.1",
    "--p5_neighbors", "8",
    "--p5_corrupt_prob", "0.8",
    "--p5_max_batches", "0",
]
from utils.p5_runtime import parse_p5_cli
args, cfg = parse_p5_cli(training=True)
assert args.p5_train_sample_interval == 0.1
assert args.p5_eval_sample_interval == 0.1
assert args.p5_neighbors == 8
assert args.p5_corrupt_prob == 0.8
assert cfg.dataset_root == "/tmp/graspnet"
assert cfg.batch_size == 1
assert all(not token.startswith("--p5_") for token in sys.argv)
print("P5_ARGPARSE_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "P5_ARGPARSE_OK" in result.stdout
