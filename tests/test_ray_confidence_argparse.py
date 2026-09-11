from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_rc_args_are_consumed_before_legacy_parser():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import sys
sys.argv = [
    "train_ray_confidence.py",
    "--dataset_root", "/tmp/graspnet",
    "--batch_size", "1",
    "--rc_train_sample_interval", "0.1",
    "--rc_eval_sample_interval", "0.1",
    "--rc_max_batches", "0",
    "--rc_ranking_weight", "1.0",
    "--rc_offsets_mm=-40,-20,-10,0,10,20,40",
]
from utils.ray_confidence_runtime import parse_rc_cli
args, cfg = parse_rc_cli(training=True)
assert args.rc_train_sample_interval == 0.1
assert args.rc_eval_sample_interval == 0.1
assert args.rc_max_batches == 0
assert args.rc_ranking_weight == 1.0
assert cfg.dataset_root == "/tmp/graspnet"
assert cfg.batch_size == 1
assert all(not token.startswith("--rc_") for token in sys.argv)
print("RC_ARGPARSE_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "RC_ARGPARSE_OK" in result.stdout
