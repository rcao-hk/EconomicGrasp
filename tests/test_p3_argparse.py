from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_p3_args_are_consumed_before_legacy_global_parser():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import sys
sys.argv = [
    "train_p3_ray.py",
    "--dataset_root", "/tmp/graspnet",
    "--batch_size", "1",
    "--p3_train_sample_interval", "0.1",
    "--p3_eval_sample_interval", "0.1",
    "--p3_max_batches", "0",
    "--p3_joint_weight", "1.0",
    "--p3_offsets_mm=-40,-20,-10,0,10,20,40",
]
from utils.p3_ray_runtime import parse_p3_cli
args, cfg = parse_p3_cli(training=True)
assert args.p3_train_sample_interval == 0.1
assert args.p3_eval_sample_interval == 0.1
assert args.p3_max_batches == 0
assert args.p3_joint_weight == 1.0
assert args.p3_offsets_mm.startswith("-40")
assert cfg.dataset_root == "/tmp/graspnet"
assert cfg.batch_size == 1
assert all(not token.startswith("--p3_") for token in sys.argv)
print("P3_ARGPARSE_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "P3_ARGPARSE_OK" in result.stdout
