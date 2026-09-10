"""Argparse-safe runtime for P2-v2.

Do not import ``models`` at module import time.  The repository-wide parser in
utils.arguments executes parse_args() during import, so all --v2_* and --ray_*
flags must be consumed first.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch


def parse_v2_cli(training: bool):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v2_hidden", type=int, default=128)
    parser.add_argument("--v2_layers", type=int, default=2)
    parser.add_argument("--v2_heads", type=int, default=4)
    parser.add_argument("--v2_dropout", type=float, default=0.10)
    parser.add_argument("--v2_zero_bias_init", type=float, default=0.5)
    parser.add_argument("--v2_max_batches", type=int, default=0)
    if training:
        parser.add_argument("--v2_train_sample_interval", type=float, default=0.1)
        parser.add_argument("--v2_eval_sample_interval", type=float, default=0.1)
        parser.add_argument("--v2_target_temperature", type=float, default=0.1)
        parser.add_argument("--v2_listwise_weight", type=float, default=1.0)
        parser.add_argument("--v2_calibration_weight", type=float, default=0.5)
        parser.add_argument("--v2_log_every", type=int, default=20)
    else:
        parser.add_argument(
            "--v2_selection",
            choices=("relational", "zero", "raw", "supported"),
            default="relational",
        )
        parser.add_argument(
            "--v2_final_score",
            choices=("raw", "contextual", "product"),
            default="raw",
        )
        parser.add_argument("--v2_run_eval", action="store_true")
        parser.add_argument("--v2_eval_only", action="store_true")
        parser.add_argument("--v2_eval_workers", type=int, default=10)

    original = list(sys.argv[1:])
    args, remaining = parser.parse_known_args()
    args.explicit_v2 = sorted({
        token.split("=", 1)[0]
        for token in original
        if token.startswith("--v2_")
    })
    sys.argv = [sys.argv[0], *remaining]

    # The v1 parser consumes --ray_* first, then imports the shared cfg parser.
    # utils.ray_grasp_runtime is intentionally model-import-free.
    from utils.ray_grasp_runtime import parse_cli
    ray_args, cfg = parse_cli(training=training)

    if args.v2_hidden <= 0 or args.v2_layers <= 0 or args.v2_heads <= 0:
        raise ValueError("P2-v2 hidden/layers/heads must be positive.")
    if args.v2_hidden % args.v2_heads != 0:
        raise ValueError("--v2_hidden must be divisible by --v2_heads.")
    if not (0.0 <= args.v2_dropout < 1.0):
        raise ValueError("--v2_dropout must lie in [0,1).")
    if not math.isfinite(args.v2_zero_bias_init):
        raise ValueError("--v2_zero_bias_init must be finite.")
    if args.v2_max_batches < 0:
        raise ValueError("--v2_max_batches must be >= 0.")
    if training:
        for name in ("v2_train_sample_interval", "v2_eval_sample_interval"):
            value = float(getattr(args, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"--{name} must lie in (0,1].")
        if args.v2_target_temperature <= 0 or not math.isfinite(args.v2_target_temperature):
            raise ValueError("--v2_target_temperature must be finite and positive.")
        for name in ("v2_listwise_weight", "v2_calibration_weight"):
            value = float(getattr(args, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"--{name} must be finite and nonnegative.")
        if args.v2_listwise_weight <= 0:
            raise ValueError("P2-v2 requires a positive listwise loss weight.")
        if args.v2_log_every <= 0:
            raise ValueError("--v2_log_every must be positive.")
    else:
        if args.v2_eval_workers <= 0:
            raise ValueError("--v2_eval_workers must be positive.")
    return args, ray_args, cfg


def _validate_base_checkpoint(ck, cfg, ray_args):
    if ck.get("distill_stage") != 1 or ck.get("distill_contract_version") != 2:
        raise ValueError("P2-v2 requires a controlled Stage-1-derived P2 checkpoint.")
    if ck.get("geometry_depth_source") != "pred" or ck.get("seed_selection_mode") != "image_fps":
        raise ValueError("P2-v2 requires predicted geometry and image-FPS query ownership.")
    if ck.get("legacy_dataset_use_gt_depth", True) or not ck.get("depth_head_executed", False):
        raise ValueError("P2-v2 checkpoint violates the RGB predicted-depth contract.")
    if ck.get("p1_center_recenter", False):
        raise ValueError("P2-v2 must not initialize from a P1 recenter checkpoint.")
    if ck.get("ray_contract_version") != 1:
        raise ValueError("P2-v2 requires the trained P2-v1 ray-field contract as its base.")
    if int(ck.get("ray_smoke_max_batches", 0)) != 0:
        raise ValueError("Do not initialize P2-v2 from a P2-v1 smoke checkpoint.")
    if "ray_offsets_mm" not in ck or "ray_hidden" not in ck or "ray_loss_weights" not in ck:
        raise ValueError("P2-v1 checkpoint is missing ray-grid/training metadata.")
    if int(ck.get("m_point", cfg.m_point)) != int(cfg.m_point):
        raise ValueError("P2-v2 m_point must equal the P2-v1 query budget.")
    explicit = set(ray_args.explicit_shared)
    for key in ("use_fuse_depth", "pose_depth_mode"):
        if key not in ck:
            raise ValueError(f"P2-v1 checkpoint is missing {key} metadata.")
        if f"--{key}" in explicit and getattr(cfg, key) != ck[key]:
            raise ValueError(f"--{key} disagrees with the P2-v1 checkpoint.")
        setattr(cfg, key, ck[key])
    for key in ("min_depth", "max_depth", "bin_num", "num_view", "num_angle", "num_depth"):
        if key in ck and float(ck[key]) != float(getattr(cfg, key)):
            raise ValueError(f"Checkpoint {key}={ck[key]} differs from CLI={getattr(cfg, key)}.")


def load_v2_model(args, ray_args, cfg, device, *, require_v2: bool = False):
    """Load P2-v1 for fresh v2 training, or a strict P2-v2 checkpoint."""
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        raise ValueError("P2-v2 requires a full checkpoint with model_state_dict and metadata.")
    _validate_base_checkpoint(ck, cfg, ray_args)
    is_v2 = "ray_v2_contract_version" in ck
    if require_v2 and not is_v2:
        raise ValueError("P2-v2 inference requires a trained P2-v2 checkpoint.")

    from models.economicgrasp_ray_v2 import RayConditionedGraspV2, RAY_V2_CONTRACT_VERSION
    if is_v2 and int(ck["ray_v2_contract_version"]) != RAY_V2_CONTRACT_VERSION:
        raise ValueError("P2-v2 checkpoint contract version mismatch.")

    names = ("v2_hidden", "v2_layers", "v2_heads", "v2_dropout")
    if is_v2:
        for name in names:
            if name not in ck:
                raise ValueError(f"P2-v2 checkpoint is missing {name}.")
            if f"--{name}" in set(args.explicit_v2):
                requested = getattr(args, name)
                saved = ck[name]
                if float(requested) != float(saved):
                    raise ValueError(f"Explicit --{name}={requested} disagrees with checkpoint={saved}.")
            setattr(args, name, ck[name])
        offsets = tuple(float(x) for x in ck["ray_offsets_mm"])
        zero_bias = float(ck.get("v2_zero_bias_init", args.v2_zero_bias_init))
    else:
        offsets = tuple(float(x) for x in ck["ray_offsets_mm"])
        zero_bias = float(args.v2_zero_bias_init)

    model = RayConditionedGraspV2(
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        bin_num=cfg.bin_num,
        pose_depth_mode=cfg.pose_depth_mode,
        camera_pose_key=ck.get("camera_pose_key", "camera_pose_vec"),
        camera_gravity_key=ck.get("camera_gravity_key", "camera_gravity_vec"),
        pose_hidden_dim=int(ck.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(ck.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(ck.get("ray_gravity_mid_dim", 32)),
        ray_offsets_mm=offsets,
        ray_hidden=int(ck["ray_hidden"]),
        ray_checkpoint_decoder=False,
        v2_hidden=int(args.v2_hidden),
        v2_layers=int(args.v2_layers),
        v2_heads=int(args.v2_heads),
        v2_dropout=float(args.v2_dropout),
        v2_zero_bias=zero_bias,
    ).to(device)
    if is_v2:
        model.load_state_dict({k.removeprefix("module."): v for k, v in ck["model_state_dict"].items()}, strict=True)
    else:
        model.load_v1(ck["model_state_dict"])
    return model, ck, is_v2, offsets
