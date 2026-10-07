"""Small experiment runtime; parse GVAR flags BEFORE utils.arguments import."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from models.gripper_volume_reader import GVARConfig, VARIANTS

BASE_MAIN_SHA = "52d09f925059bec3643610ecf1f1722894627ee5"
CONTRACT_VERSION = 1
BRANCH = "exp/gripper-volume-action-reader"


def parse_gvar_args(inference=False):
    parser = argparse.ArgumentParser(add_help=False, description="GVAR experiment-specific flags. Common CVA arguments are forwarded to utils.arguments.")
    parser.add_argument("--gvar_variant", choices=(("auto",) + VARIANTS if inference else VARIANTS), default="auto" if inference else "volume")
    parser.add_argument("--gvar_reader_dim", type=int, default=64)
    parser.add_argument("--gvar_reader_heads", type=int, default=4)
    parser.add_argument("--gvar_action_chunk", type=int, default=512)
    parser.add_argument("--gvar_activation_checkpoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gvar_max_batches", type=int, default=0, help="Smoke limit; 0=all. Partial dumps must NOT be evaluated.")
    parser.add_argument("--gvar_resume_inference", action="store_true")
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        print("\nCommon flags: --dataset_root --gntrans_rgb_root (train) --log_dir (train)\n"
              "--checkpoint_path (eval/resume) --save_dir --test_mode --sample_interval\n"
              "--batch_size --num_workers --use_cdf --multi_modal --pose_depth_mode\n"
              "See scripts/run_gvar_train.sh and scripts/run_gvar_eval.sh for complete commands.")
        raise SystemExit(0)
    args, rest = parser.parse_known_args()
    if args.gvar_max_batches < 0:
        parser.error("gvar_max_batches must be nonnegative")
    sys.argv = [sys.argv[0], *rest]
    return args


def config_from_args(args):
    return GVARConfig(variant=args.gvar_variant, reader_dim=args.gvar_reader_dim,
                      reader_heads=args.gvar_reader_heads, action_chunk=args.gvar_action_chunk,
                      activation_checkpoint=bool(args.gvar_activation_checkpoint))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def finite_metrics(values):
    return {k: float(v) for k, v in values.items() if math.isfinite(float(v))}


def selected_frame_indices(dataset, fraction=0.1):
    if abs(float(fraction) - 0.1) > 1e-10:
        raise ValueError("This GVAR experiment fixes evaluation to 0.1 (frame IDs 0,10,...,250)")
    rows = []
    for i, (scene, fid) in enumerate(zip(dataset.scenename, dataset.frameid)):
        if int(fid) % 10 == 0:
            rows.append(i)
    if len(rows) != 780:
        raise ValueError(f"Expected 780 frames in one test split, got {len(rows)}")
    return rows


def frame_fingerprint(dataset, indices):
    text = "\n".join(f"{dataset.scenename[i]}/{int(dataset.frameid[i]):04d}" for i in indices)
    return hashlib.sha256(text.encode()).hexdigest()


def architecture_config(cfgs):
    keys = {"min_depth", "max_depth", "bin_num", "num_angle", "num_depth", "num_view", "m_point",
            "graspness_threshold", "pose_depth_mode", "use_cdf", "use_top4_view_infer"}
    return {k: v for k, v in vars(cfgs).items() if k in keys or k.startswith("kview_")}


def validate_checkpoint(checkpoint, config=None):
    if not isinstance(checkpoint, dict) or checkpoint.get("gvar_contract_version") != CONTRACT_VERSION:
        raise ValueError("Expected a GVAR full checkpoint, not an old Stage-1/P5/raw-state checkpoint")
    loaded = GVARConfig(**checkpoint["gvar_config"])
    if config is not None and loaded.to_dict() != config.to_dict():
        raise ValueError(f"GVAR configuration mismatch: checkpoint={loaded.to_dict()}, requested={config.to_dict()}")
    if checkpoint.get("detach_policy") != {"E": True, "Q": True, "C": True}:
        raise ValueError("Checkpoint does not document full E/Q/C detach")
    return loaded


class LimitedLoader:
    """Bound a real loader without altering the scene/frame sampling contract."""
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, int(limit)
    def __len__(self):
        return min(len(self.loader), self.limit) if self.limit else len(self.loader)
    def __iter__(self):
        from itertools import islice
        return islice(iter(self.loader), len(self))
