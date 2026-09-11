"""Argparse-safe runtime for the ray-conditioned confidence experiment.

Do not import ``models`` at module import time. ``utils.arguments`` parses the
process argv during import, therefore all --rc_* flags must be consumed first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

DEFAULT_OFFSETS_MM = (-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0)


def _parse_offsets(value):
    values = tuple(float(x) for x in (value.split(",") if isinstance(value, str) else value))
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Ray-confidence offsets must be finite and non-empty.")
    if len(set(values)) != len(values) or values.count(0.0) != 1:
        raise ValueError("Ray-confidence offsets must be unique and contain exactly one zero.")
    if max(abs(x) for x in values) > 250.0:
        raise ValueError("Ray-confidence offsets exceed 250 mm; check units.")
    return values


def parse_rc_cli(training: bool):
    parser = argparse.ArgumentParser(add_help=False, description="Ray-conditioned confidence flags")
    parser.add_argument("--rc_offsets_mm", default=None,
                        help="Comma-separated camera-z offsets in mm; pass negatives as --rc_offsets_mm=-40,...")
    parser.add_argument("--rc_hidden", type=int, default=128)
    parser.add_argument("--rc_layers", type=int, default=2)
    parser.add_argument("--rc_heads", type=int, default=4)
    parser.add_argument("--rc_dropout", type=float, default=0.10)
    parser.add_argument("--rc_init_bias", type=float, default=4.0)
    parser.add_argument("--rc_max_batches", type=int, default=0)
    if training:
        parser.add_argument("--rc_train_sample_interval", type=float, default=0.1)
        parser.add_argument("--rc_eval_sample_interval", type=float, default=0.1)
        parser.add_argument("--rc_calibration_weight", type=float, default=1.0)
        parser.add_argument("--rc_ranking_weight", type=float, default=1.0)
        parser.add_argument("--rc_rank_temperature", type=float, default=0.10)
        parser.add_argument("--rc_log_every", type=int, default=20)
    else:
        parser.add_argument("--rc_score_mode", choices=("raw", "calibrated"), default="calibrated")
        parser.add_argument("--rc_run_eval", action="store_true")
        parser.add_argument("--rc_eval_only", action="store_true")
        parser.add_argument("--rc_eval_workers", type=int, default=10)

    original = list(sys.argv[1:])
    args, remaining = parser.parse_known_args()
    args.explicit_rc = sorted({token.split("=", 1)[0] for token in original if token.startswith("--rc_")})
    explicit_shared = sorted({token.split("=", 1)[0] for token in remaining if token.startswith("--")})
    sys.argv = [sys.argv[0], *remaining]
    from utils.arguments import cfgs

    cfgs.multi_modal = True
    cfgs.use_cdf = True
    cfgs.extend_angle = True
    cfgs.graspness_mode = cfgs.graspness_mode or "scene"
    for key in ("use_obs_depth", "use_gt_depth", "use_depth_comp", "kview_use_collision", "use_top4_view_infer", "pin_memory"):
        if bool(getattr(cfgs, key, False)):
            raise ValueError(f"Ray-confidence experiment does not allow --{key}.")
    if cfgs.kview_mode != "A1" or cfgs.oracle_mode:
        raise ValueError("Ray-confidence experiment requires native Top-1 A1 and no oracle mode.")
    if cfgs.graspness_mode not in ("scene", "instance"):
        raise ValueError("graspness_mode must be scene or instance.")
    if min(cfgs.batch_size, cfgs.m_point) <= 0 or cfgs.num_workers < 0 or cfgs.eval_num_workers < 0:
        raise ValueError("Invalid batch/query/worker configuration.")
    if args.rc_hidden <= 0 or args.rc_layers <= 0 or args.rc_heads <= 0:
        raise ValueError("rc_hidden/layers/heads must be positive.")
    if args.rc_hidden % args.rc_heads != 0:
        raise ValueError("--rc_hidden must be divisible by --rc_heads.")
    if not 0.0 <= args.rc_dropout < 1.0 or args.rc_max_batches < 0:
        raise ValueError("Invalid rc_dropout or rc_max_batches.")
    if not math.isfinite(args.rc_init_bias):
        raise ValueError("rc_init_bias must be finite.")

    if training:
        for name in ("rc_train_sample_interval", "rc_eval_sample_interval"):
            x = float(getattr(args, name))
            if not math.isfinite(x) or not 0.0 < x <= 1.0:
                raise ValueError(f"--{name} must lie in (0,1].")
        for name in ("rc_calibration_weight", "rc_ranking_weight"):
            x = float(getattr(args, name))
            if not math.isfinite(x) or x < 0.0:
                raise ValueError(f"--{name} must be finite and non-negative.")
        if args.rc_calibration_weight + args.rc_ranking_weight <= 0:
            raise ValueError("At least one ray-confidence loss weight must be positive.")
        if args.rc_rank_temperature <= 0 or not math.isfinite(args.rc_rank_temperature):
            raise ValueError("rc_rank_temperature must be finite and positive.")
        if args.rc_log_every <= 0:
            raise ValueError("rc_log_every must be positive.")
        if "--learning_rate" not in explicit_shared:
            cfgs.learning_rate = 1.0e-4
    else:
        if args.rc_eval_workers <= 0:
            raise ValueError("rc_eval_workers must be positive.")
        if "--sample_interval" not in explicit_shared:
            cfgs.sample_interval = 0.1
    args.explicit_shared = explicit_shared
    return args, cfgs


def init_distributed(seed: int):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Full ray-confidence EconomicGrasp execution requires CUDA.")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=int(os.environ.get("DDP_TIMEOUT_SEC", "3600"))),
        )
    random.seed(int(seed) + rank)
    np.random.seed(int(seed) + rank)
    torch.manual_seed(int(seed) + rank)
    torch.cuda.manual_seed_all(int(seed) + rank)
    return rank, world, torch.device("cuda", local_rank)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def stride_from_fraction(value: float) -> int:
    x = float(value)
    if not math.isfinite(x) or not 0.0 < x <= 1.0:
        raise ValueError("Sampling fraction must lie in (0,1].")
    return max(1, int(round(1.0 / x)))


def scene_stratified_indices(dataset, fraction: float):
    if not hasattr(dataset, "scenename") or len(dataset.scenename) != len(dataset):
        raise RuntimeError("GraspNet dataset must expose one scenename entry per frame.")
    stride = stride_from_fraction(fraction)
    groups = {}
    for idx, scene in enumerate(dataset.scenename):
        groups.setdefault(str(scene), []).append(idx)
    return [idx for group in groups.values() for idx in group[::stride]]


def _worker_init(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataset(cfg, split: str, fraction: float, labels: bool):
    from dataset.graspnet_dataset import GraspNetMultiDataset
    kwargs = dict(
        camera=cfg.camera,
        split=split,
        num_points=cfg.num_point,
        remove_outlier=True,
        augment=False,
        use_gt_depth=False,
        use_fuse_depth=cfg.use_fuse_depth,
        graspness_mode=cfg.graspness_mode,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        bin_num=cfg.bin_num,
        load_label=labels,
    )
    if labels:
        kwargs.update(voxel_size=cfg.voxel_size, depth_strides=1, extend_angle=True, load_grasp_payload=False)
    base = GraspNetMultiDataset(cfg.dataset_root, **kwargs)
    indices = scene_stratified_indices(base, fraction)
    dataset = base
    if labels:
        from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
        dataset = CVAExtendedLabelAdapter(
            base,
            dataset_root=cfg.dataset_root,
            use_cdf=True,
            label_folder=cfg.cdf_label_folder,
            num_angle=cfg.num_angle,
            num_depth=cfg.num_depth,
        )
    return base, dataset, indices


def make_loader(dataset, indices, cfg, rank: int, world: int, training: bool):
    from dataset.graspnet_dataset import collate_fn
    sampler = None
    if training:
        selected = Subset(dataset, indices)
        sampler = DistributedSampler(
            selected, num_replicas=world, rank=rank, shuffle=True,
            seed=cfg.seed, drop_last=False,
        )
    else:
        selected = Subset(dataset, indices[rank::world])
    generator = torch.Generator().manual_seed(int(cfg.seed) + rank)
    workers = cfg.num_workers if training else cfg.eval_num_workers
    loader = DataLoader(
        selected,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init,
        generator=generator,
        persistent_workers=False,
    )
    return loader, sampler


def move_batch(batch: dict, device: torch.device):
    drop = {"point_clouds", "cloud_colors", "coordinates_for_voxel", "sensor_depth_m", "obs_depth_m"}
    return {
        key: (value.to(device, non_blocking=False) if torch.is_tensor(value) else value)
        for key, value in batch.items()
        if key not in drop
    }


def _validate_stage1(checkpoint, cfg, args):
    if checkpoint.get("distill_stage") != 1 or checkpoint.get("distill_contract_version") != 2:
        raise ValueError("Fresh ray-confidence training requires controlled Stage-1 (stage=1, contract=2).")
    if checkpoint.get("geometry_depth_source") != "pred" or checkpoint.get("seed_selection_mode") != "image_fps":
        raise ValueError("Stage-1 checkpoint must use predicted geometry and image-FPS queries.")
    if not bool(checkpoint.get("depth_head_executed", False)) or bool(checkpoint.get("legacy_dataset_use_gt_depth", True)):
        raise ValueError("Stage-1 checkpoint violates the RGB predicted-depth contract.")
    if checkpoint.get("p1_center_recenter", False) or "ray_contract_version" in checkpoint or "p3_contract_version" in checkpoint:
        raise ValueError("Fresh ray-confidence training must start from original Stage-1, not P1/P2/P3.")
    if "use_fuse_depth" not in checkpoint or "pose_depth_mode" not in checkpoint:
        raise ValueError("Stage-1 checkpoint is missing geometry metadata.")
    explicit = set(args.explicit_shared)
    for key in ("use_fuse_depth", "pose_depth_mode"):
        if f"--{key}" in explicit and getattr(cfg, key) != checkpoint[key]:
            raise ValueError(f"--{key} disagrees with checkpoint metadata.")
        setattr(cfg, key, checkpoint[key])


def load_rc_model(args, cfg, device, *, require_rc: bool = False):
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        raise FileNotFoundError(f"Ray-confidence checkpoint not found: {cfg.checkpoint_path}")
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Ray-confidence experiment requires a full checkpoint with metadata.")
    is_rc = "rc_contract_version" in checkpoint
    if require_rc and not is_rc:
        raise ValueError("Ray-confidence inference requires a trained ray-confidence checkpoint.")

    if is_rc:
        if int(checkpoint["rc_contract_version"]) != 1:
            raise ValueError("Unsupported ray-confidence checkpoint contract.")
        if checkpoint.get("rc_base_main_sha") != "3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2":
            raise ValueError("Ray-confidence checkpoint has the wrong main ancestry.")
        if int(checkpoint.get("rc_smoke_max_batches", 0)) != 0 and require_rc:
            raise ValueError("Official inference rejects smoke ray-confidence checkpoints.")
        if checkpoint.get("distill_stage") != 1 or checkpoint.get("geometry_depth_source") != "pred":
            raise ValueError("Ray-confidence checkpoint lost Stage-1 ancestry metadata.")
        for key in ("use_fuse_depth", "pose_depth_mode"):
            setattr(cfg, key, checkpoint[key])
        offsets = tuple(float(x) for x in checkpoint["rc_offsets_mm"])
        hidden = int(checkpoint["rc_hidden"])
        layers = int(checkpoint["rc_layers"])
        heads = int(checkpoint["rc_heads"])
        dropout = float(checkpoint["rc_dropout"])
        init_bias = float(checkpoint["rc_init_bias"])
        for name, saved in (
            ("rc_hidden", hidden), ("rc_layers", layers), ("rc_heads", heads),
            ("rc_dropout", dropout), ("rc_init_bias", init_bias),
        ):
            if f"--{name}" in set(args.explicit_rc) and float(getattr(args, name)) != float(saved):
                raise ValueError(f"Explicit --{name} disagrees with checkpoint={saved}.")
        if args.rc_offsets_mm is not None and _parse_offsets(args.rc_offsets_mm) != offsets:
            raise ValueError("Do not change the trained ray-confidence grid at inference/resume.")
    else:
        _validate_stage1(checkpoint, cfg, args)
        offsets = _parse_offsets(args.rc_offsets_mm or DEFAULT_OFFSETS_MM)
        hidden, layers, heads = args.rc_hidden, args.rc_layers, args.rc_heads
        dropout, init_bias = args.rc_dropout, args.rc_init_bias

    from models.economicgrasp_ray_confidence import economicgrasp_dpt_ray_confidence
    model = economicgrasp_dpt_ray_confidence(
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        bin_num=cfg.bin_num,
        pose_depth_mode=cfg.pose_depth_mode,
        camera_pose_key=checkpoint.get("camera_pose_key", "camera_pose_vec"),
        camera_gravity_key=checkpoint.get("camera_gravity_key", "camera_gravity_vec"),
        pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(checkpoint.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(checkpoint.get("ray_gravity_mid_dim", 32)),
        rc_offsets_mm=offsets,
        rc_hidden=hidden,
        rc_layers=layers,
        rc_heads=heads,
        rc_dropout=dropout,
        rc_init_bias=init_bias,
    ).to(device)
    if is_rc:
        state = {key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
    else:
        model.load_stage1(checkpoint["model_state_dict"])
    return model, checkpoint, is_rc, offsets


def reduce_statistics(stats, device, world):
    names = sorted(stats)
    packed = torch.tensor([stats[name] for name in names], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(packed)
    values = packed.cpu().tolist()
    return {
        name: {"sum": total, "count": count, "mean": total / count if count else None}
        for name, (total, count) in zip(names, values)
    }


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
