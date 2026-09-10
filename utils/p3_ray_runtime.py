"""Argparse-safe runtime for P3 ray-evidence aggregation.

Do not import anything under ``models`` at module import time.  This repository's
``utils.arguments`` executes parse_args() during import, so P3-specific flags
must be consumed first.
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
        raise ValueError("P3 offsets must be finite and non-empty.")
    if len(set(values)) != len(values) or values.count(0.0) != 1:
        raise ValueError("P3 offsets must be unique and contain exactly one zero.")
    if max(abs(x) for x in values) > 250.0:
        raise ValueError("P3 offsets exceed 250 mm; check units.")
    return values


def parse_p3_cli(training: bool):
    parser = argparse.ArgumentParser(add_help=False, description="P3 ray-evidence specific flags")
    parser.add_argument("--p3_offsets_mm", default=None,
                        help="Comma-separated camera-z offsets in mm; for negatives use --p3_offsets_mm=-40,...")
    parser.add_argument("--p3_hidden", type=int, default=128)
    parser.add_argument("--p3_layers", type=int, default=2)
    parser.add_argument("--p3_heads", type=int, default=4)
    parser.add_argument("--p3_dropout", type=float, default=0.10)
    parser.add_argument("--p3_max_batches", type=int, default=0,
                        help="Smoke cap only; never use for a reported result.")
    if training:
        parser.add_argument("--p3_train_sample_interval", type=float, default=0.1)
        parser.add_argument("--p3_eval_sample_interval", type=float, default=0.1)
        parser.add_argument("--p3_cdf_weight", type=float, default=1.0)
        parser.add_argument("--p3_width_weight", type=float, default=10.0)
        parser.add_argument("--p3_viability_weight", type=float, default=1.0)
        parser.add_argument("--p3_joint_weight", type=float, default=1.0)
        parser.add_argument("--p3_log_every", type=int, default=20)
    else:
        parser.add_argument("--p3_selection_score", choices=("joint", "raw"), default="joint")
        parser.add_argument("--p3_final_score", choices=("same", "joint", "raw"), default="same")
        parser.add_argument("--p3_force_zero", action="store_true")
        parser.add_argument("--p3_run_eval", action="store_true")
        parser.add_argument("--p3_eval_only", action="store_true")
        parser.add_argument("--p3_eval_workers", type=int, default=10)

    original = list(sys.argv[1:])
    args, remaining = parser.parse_known_args()
    args.explicit_p3 = sorted({token.split("=", 1)[0] for token in original if token.startswith("--p3_")})
    explicit_shared = sorted({token.split("=", 1)[0] for token in remaining if token.startswith("--")})
    sys.argv = [sys.argv[0], *remaining]
    from utils.arguments import cfgs

    # P3 is intentionally a strict RGB-only, image-FPS, CDF, Top-1 experiment.
    cfgs.multi_modal = True
    cfgs.use_cdf = True
    cfgs.extend_angle = True
    cfgs.graspness_mode = cfgs.graspness_mode or "scene"
    for key in ("use_obs_depth", "use_gt_depth", "use_depth_comp", "kview_use_collision", "use_top4_view_infer", "pin_memory"):
        if bool(getattr(cfgs, key, False)):
            raise ValueError(f"P3 does not allow --{key}.")
    if cfgs.kview_mode != "A1":
        raise ValueError("P3-v1 requires native Top-1 KView mode A1.")
    if cfgs.oracle_mode:
        raise ValueError("P3 does not allow oracle_mode.")
    if cfgs.graspness_mode not in ("scene", "instance"):
        raise ValueError("graspness_mode must be scene or instance.")
    if min(cfgs.batch_size, cfgs.m_point) <= 0 or cfgs.num_workers < 0 or cfgs.eval_num_workers < 0:
        raise ValueError("Invalid P3 batch/query/worker configuration.")
    if args.p3_hidden <= 0 or args.p3_layers <= 0 or args.p3_heads <= 0:
        raise ValueError("P3 hidden/layers/heads must be positive.")
    if args.p3_hidden % args.p3_heads != 0:
        raise ValueError("--p3_hidden must be divisible by --p3_heads.")
    if not 0.0 <= args.p3_dropout < 1.0 or args.p3_max_batches < 0:
        raise ValueError("Invalid P3 dropout or smoke batch cap.")
    if training:
        for name in ("p3_train_sample_interval", "p3_eval_sample_interval"):
            x = float(getattr(args, name))
            if not math.isfinite(x) or not 0.0 < x <= 1.0:
                raise ValueError(f"--{name} must lie in (0,1].")
        for name in ("p3_cdf_weight", "p3_width_weight", "p3_viability_weight", "p3_joint_weight"):
            x = float(getattr(args, name))
            if not math.isfinite(x) or x < 0.0:
                raise ValueError(f"--{name} must be finite and non-negative.")
        if args.p3_cdf_weight <= 0 or args.p3_joint_weight <= 0 or args.p3_log_every <= 0:
            raise ValueError("P3 requires positive CDF/joint weights and log interval.")
        if "--learning_rate" not in explicit_shared:
            cfgs.learning_rate = 1.0e-4
    else:
        if args.p3_eval_workers <= 0:
            raise ValueError("--p3_eval_workers must be positive.")
        if "--sample_interval" not in explicit_shared:
            cfgs.sample_interval = 0.1
    args.explicit_shared = explicit_shared
    return args, cfgs


def init_distributed(seed: int):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Full P3 EconomicGrasp execution requires CUDA; pure tensor tests are CPU-safe.")
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
        dist.destroy_process_group()  # no final barrier: preserve original exception on rank failure


def stride_from_fraction(value: float) -> int:
    x = float(value)
    if not math.isfinite(x) or not 0.0 < x <= 1.0:
        raise ValueError("Sampling fraction must be in (0,1].")
    return max(1, int(round(1.0 / x)))


def scene_stratified_indices(dataset, fraction: float):
    if not hasattr(dataset, "scenename") or len(dataset.scenename) != len(dataset):
        raise RuntimeError("GraspNet dataset must expose one scenename entry per frame.")
    stride = stride_from_fraction(fraction)
    groups = {}
    for idx, scene in enumerate(dataset.scenename):
        groups.setdefault(str(scene), []).append(idx)
    return [idx for scene_indices in groups.values() for idx in scene_indices[::stride]]


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
        kwargs.update(
            voxel_size=cfg.voxel_size,
            depth_strides=1,
            extend_angle=True,
            load_grasp_payload=False,
        )
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
            selected,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=False,
        )
    else:
        # Unique, non-padded validation/inference shards.
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
    result = {}
    for key, value in batch.items():
        if key in drop:
            continue
        # Variable-size cached object labels intentionally remain CPU lists.
        result[key] = value.to(device, non_blocking=False) if torch.is_tensor(value) else value
    return result


def _validate_stage1_contract(checkpoint, cfg, args):
    if checkpoint.get("distill_stage") != 1 or checkpoint.get("distill_contract_version") != 2:
        raise ValueError("P3 requires the controlled Stage-1 RGB checkpoint (distill stage 1, contract 2).")
    if checkpoint.get("geometry_depth_source") != "pred" or checkpoint.get("seed_selection_mode") != "image_fps":
        raise ValueError("P3 requires predicted geometry and image-FPS seed ownership.")
    if not bool(checkpoint.get("depth_head_executed", False)) or bool(checkpoint.get("legacy_dataset_use_gt_depth", True)):
        raise ValueError("Checkpoint violates the P3 RGB predicted-depth contract.")
    if checkpoint.get("p1_center_recenter", False) or "ray_contract_version" in checkpoint:
        raise ValueError("Fresh P3 must initialize from original Stage-1, not P1/P2.")
    if "use_fuse_depth" not in checkpoint or "pose_depth_mode" not in checkpoint:
        raise ValueError("Stage-1 checkpoint is missing use_fuse_depth/pose_depth_mode metadata.")
    explicit = set(args.explicit_shared)
    for key in ("use_fuse_depth", "pose_depth_mode"):
        if f"--{key}" in explicit and getattr(cfg, key) != checkpoint[key]:
            raise ValueError(f"--{key} disagrees with Stage-1 checkpoint metadata.")
        setattr(cfg, key, checkpoint[key])


def load_p3_model(args, cfg, device, *, require_p3: bool = False):
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        raise FileNotFoundError(f"P3 checkpoint not found: {cfg.checkpoint_path}")
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("P3 requires a full Stage-1/P3 checkpoint with metadata.")
    is_p3 = "p3_contract_version" in checkpoint
    if require_p3 and not is_p3:
        raise ValueError("P3 inference requires a trained P3 checkpoint.")

    if is_p3:
        if int(checkpoint["p3_contract_version"]) != 1:
            raise ValueError("Unsupported P3 checkpoint contract version.")
        if checkpoint.get("p3_base_main_sha") != "3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2":
            raise ValueError("P3 checkpoint was not built from the controlled main base.")
        # P3 checkpoints preserve the original Stage-1 geometry contract.
        if checkpoint.get("distill_stage") != 1 or checkpoint.get("geometry_depth_source") != "pred":
            raise ValueError("P3 checkpoint has invalid Stage-1 ancestry metadata.")
        if int(checkpoint.get("p3_smoke_max_batches", 0)) != 0 and require_p3:
            raise ValueError("Official P3 inference rejects smoke checkpoints.")
        for key in ("use_fuse_depth", "pose_depth_mode"):
            setattr(cfg, key, checkpoint[key])
        offsets = tuple(float(x) for x in checkpoint["p3_offsets_mm"])
        hidden = int(checkpoint["p3_hidden"])
        layers = int(checkpoint["p3_layers"])
        heads = int(checkpoint["p3_heads"])
        dropout = float(checkpoint["p3_dropout"])
        for name, saved in (("p3_hidden", hidden), ("p3_layers", layers), ("p3_heads", heads), ("p3_dropout", dropout)):
            if f"--{name}" in set(args.explicit_p3) and float(getattr(args, name)) != float(saved):
                raise ValueError(f"Explicit --{name} disagrees with P3 checkpoint ({saved}).")
        if args.p3_offsets_mm is not None and _parse_offsets(args.p3_offsets_mm) != offsets:
            raise ValueError("Do not change the trained P3 ray grid at inference/resume.")
    else:
        _validate_stage1_contract(checkpoint, cfg, args)
        offsets = _parse_offsets(args.p3_offsets_mm or DEFAULT_OFFSETS_MM)
        hidden, layers, heads, dropout = args.p3_hidden, args.p3_layers, args.p3_heads, args.p3_dropout

    from models.economicgrasp_ray_p3 import economicgrasp_dpt_p3_ray
    model = economicgrasp_dpt_p3_ray(
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        bin_num=cfg.bin_num,
        pose_depth_mode=cfg.pose_depth_mode,
        camera_pose_key=checkpoint.get("camera_pose_key", "camera_pose_vec"),
        camera_gravity_key=checkpoint.get("camera_gravity_key", "camera_gravity_vec"),
        pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(checkpoint.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(checkpoint.get("ray_gravity_mid_dim", 32)),
        p3_offsets_mm=offsets,
        p3_hidden=hidden,
        p3_layers=layers,
        p3_heads=heads,
        p3_dropout=dropout,
    ).to(device)
    if is_p3:
        state = {key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
    else:
        model.load_stage1(checkpoint["model_state_dict"])
    return model, checkpoint, is_p3, offsets


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def reduce_statistics(stats, device, world):
    names = sorted(stats)
    packed = torch.tensor([stats[name] for name in names], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(packed)
    values = packed.cpu().tolist()
    return {
        name: {"sum": total, "count": count, "mean": total / count if count > 0 else None}
        for name, (total, count) in zip(names, values)
    }
