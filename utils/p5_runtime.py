"""Argparse-safe runtime for P5 geometry-aware grasp repair.

Do not import anything under ``models`` at module import time.  The repository's
legacy global parser runs while importing utils.arguments, so every --p5_* flag
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


def parse_p5_cli(training: bool):
    p = argparse.ArgumentParser(add_help=False, description="P5 grasp-repair flags")
    p.add_argument("--p5_hidden", type=int, default=128)
    p.add_argument("--p5_layers", type=int, default=2)
    p.add_argument("--p5_heads", type=int, default=4)
    p.add_argument("--p5_neighbors", type=int, default=8)
    p.add_argument("--p5_dropout", type=float, default=0.10)
    p.add_argument("--p5_max_delta_m", type=float, default=0.06)
    p.add_argument("--p5_target_radius_m", type=float, default=0.06)
    p.add_argument("--p5_corrupt_prob", type=float, default=0.8)
    p.add_argument("--p5_scene_bias_sigma_m", type=float, default=0.012)
    p.add_argument("--p5_scale_sigma", type=float, default=0.025)
    p.add_argument("--p5_region_sigma_m", type=float, default=0.015)
    p.add_argument("--p5_region_grid", type=int, default=7)
    p.add_argument("--p5_max_batches", type=int, default=0,
                   help="Smoke cap only; never use for a reported AP result.")
    if training:
        p.add_argument("--p5_train_sample_interval", type=float, default=0.1)
        p.add_argument("--p5_eval_sample_interval", type=float, default=0.1)
        p.add_argument("--p5_repair_weight", type=float, default=1.0)
        p.add_argument("--p5_unknown_identity_weight", type=float, default=0.02)
        p.add_argument("--p5_smooth_l1_beta_m", type=float, default=0.005)
        p.add_argument("--p5_log_every", type=int, default=20)
    else:
        p.add_argument("--p5_mode", choices=("repair", "native"), default="repair")
        p.add_argument("--p5_run_eval", action="store_true")
        p.add_argument("--p5_eval_only", action="store_true")
        p.add_argument("--p5_eval_workers", type=int, default=10)

    original = list(sys.argv[1:])
    args, remaining = p.parse_known_args()
    args.explicit_p5 = sorted({x.split("=", 1)[0] for x in original if x.startswith("--p5_")})
    explicit_shared = sorted({x.split("=", 1)[0] for x in remaining if x.startswith("--")})
    sys.argv = [sys.argv[0], *remaining]
    from utils.arguments import cfgs

    cfgs.multi_modal = True
    cfgs.use_cdf = True
    cfgs.extend_angle = True
    cfgs.graspness_mode = cfgs.graspness_mode or "scene"
    for key in ("use_obs_depth", "use_gt_depth", "use_depth_comp", "kview_use_collision",
                "use_top4_view_infer", "pin_memory"):
        if bool(getattr(cfgs, key, False)):
            raise ValueError(f"P5 does not allow --{key}.")
    if cfgs.kview_mode != "A1" or cfgs.oracle_mode:
        raise ValueError("P5 requires native Top-1 A1 and no oracle mode.")
    if cfgs.graspness_mode not in ("scene", "instance"):
        raise ValueError("P5 graspness_mode must be scene or instance.")
    if min(cfgs.batch_size, cfgs.m_point) <= 0 or cfgs.num_workers < 0 or cfgs.eval_num_workers < 0:
        raise ValueError("Invalid P5 batch/query/worker configuration.")
    if args.p5_hidden <= 0 or args.p5_layers <= 0 or args.p5_heads <= 0 or args.p5_neighbors <= 0:
        raise ValueError("P5 hidden/layers/heads/neighbors must be positive.")
    if args.p5_hidden % args.p5_heads != 0:
        raise ValueError("--p5_hidden must be divisible by --p5_heads.")
    if not 0.0 <= args.p5_dropout < 1.0 or args.p5_max_batches < 0:
        raise ValueError("Invalid P5 dropout/max_batches.")
    if args.p5_max_delta_m <= 0 or args.p5_target_radius_m <= 0:
        raise ValueError("P5 max delta/target radius must be positive.")
    if not 0.0 <= args.p5_corrupt_prob <= 1.0:
        raise ValueError("P5 corruption probability must lie in [0,1].")
    if min(args.p5_scene_bias_sigma_m, args.p5_scale_sigma, args.p5_region_sigma_m) < 0:
        raise ValueError("P5 corruption sigmas must be non-negative.")
    if args.p5_region_grid < 2:
        raise ValueError("P5 region grid must be >=2.")

    if training:
        for name in ("p5_train_sample_interval", "p5_eval_sample_interval"):
            x = float(getattr(args, name))
            if not math.isfinite(x) or not 0.0 < x <= 1.0:
                raise ValueError(f"--{name} must lie in (0,1].")
        if args.p5_repair_weight <= 0 or args.p5_unknown_identity_weight < 0:
            raise ValueError("P5 repair weight must be positive and identity weight non-negative.")
        if args.p5_smooth_l1_beta_m <= 0 or args.p5_log_every <= 0:
            raise ValueError("P5 beta/log interval must be positive.")
        if "--learning_rate" not in explicit_shared:
            cfgs.learning_rate = 1.0e-4
    else:
        if args.p5_eval_workers <= 0:
            raise ValueError("P5 eval workers must be positive.")
        if "--sample_interval" not in explicit_shared:
            cfgs.sample_interval = 0.1
    args.explicit_shared = explicit_shared
    return args, cfgs


def init_distributed(seed: int):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Full P5 EconomicGrasp execution requires CUDA.")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", init_method="env://",
                                timeout=timedelta(seconds=int(os.environ.get("DDP_TIMEOUT_SEC", "3600"))))
    random.seed(int(seed) + rank)
    np.random.seed(int(seed) + rank)
    torch.manual_seed(int(seed) + rank)
    torch.cuda.manual_seed_all(int(seed) + rank)
    return rank, world, torch.device("cuda", local)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def stride_from_fraction(value: float) -> int:
    x = float(value)
    if not math.isfinite(x) or not 0.0 < x <= 1.0:
        raise ValueError("P5 sampling fraction must lie in (0,1].")
    return max(1, int(round(1.0 / x)))


def scene_stratified_indices(dataset, fraction: float):
    if not hasattr(dataset, "scenename") or len(dataset.scenename) != len(dataset):
        raise RuntimeError("P5 requires one scenename entry per frame.")
    stride = stride_from_fraction(fraction)
    groups = {}
    for i, scene in enumerate(dataset.scenename):
        groups.setdefault(str(scene), []).append(i)
    return [i for group in groups.values() for i in group[::stride]]


def _worker_init(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataset(cfg, split: str, fraction: float, labels: bool):
    from dataset.graspnet_dataset import GraspNetMultiDataset
    kw = dict(
        camera=cfg.camera, split=split, num_points=cfg.num_point,
        remove_outlier=True, augment=False, use_gt_depth=False,
        use_fuse_depth=cfg.use_fuse_depth, graspness_mode=cfg.graspness_mode,
        min_depth=cfg.min_depth, max_depth=cfg.max_depth, bin_num=cfg.bin_num,
        load_label=labels,
    )
    if labels:
        kw.update(voxel_size=cfg.voxel_size, depth_strides=1, extend_angle=True,
                  load_grasp_payload=False)
    base = GraspNetMultiDataset(cfg.dataset_root, **kw)
    indices = scene_stratified_indices(base, fraction)
    dataset = base
    if labels:
        from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
        dataset = CVAExtendedLabelAdapter(
            base, dataset_root=cfg.dataset_root, use_cdf=True,
            label_folder=cfg.cdf_label_folder, num_angle=cfg.num_angle,
            num_depth=cfg.num_depth,
        )
    return base, dataset, indices


def make_loader(dataset, indices, cfg, rank: int, world: int, training: bool):
    from dataset.graspnet_dataset import collate_fn
    sampler = None
    if training:
        selected = Subset(dataset, indices)
        sampler = DistributedSampler(selected, num_replicas=world, rank=rank,
                                     shuffle=True, seed=cfg.seed, drop_last=False)
    else:
        selected = Subset(dataset, indices[rank::world])
    gen = torch.Generator().manual_seed(int(cfg.seed) + rank)
    workers = cfg.num_workers if training else cfg.eval_num_workers
    loader = DataLoader(
        selected, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
        num_workers=workers, pin_memory=False, collate_fn=collate_fn,
        worker_init_fn=_worker_init, generator=gen, persistent_workers=False,
    )
    return loader, sampler


def move_batch(batch: dict, device: torch.device):
    drop = {"point_clouds", "cloud_colors", "coordinates_for_voxel", "sensor_depth_m",
            "obs_depth_m", "gt_depth_m"}
    return {
        k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
        for k, v in batch.items() if k not in drop
    }


def _validate_stage1(ck, cfg, args):
    if ck.get("distill_stage") != 1 or ck.get("distill_contract_version") != 2:
        raise ValueError("Fresh P5 requires controlled Stage-1 (stage=1, contract=2).")
    if ck.get("geometry_depth_source") != "pred" or ck.get("seed_selection_mode") != "image_fps":
        raise ValueError("P5 Stage-1 must use predicted geometry and image-FPS.")
    if not bool(ck.get("depth_head_executed", False)) or bool(ck.get("legacy_dataset_use_gt_depth", True)):
        raise ValueError("P5 Stage-1 violates the RGB predicted-depth contract.")
    forbidden = ("p1_center_recenter", "ray_contract_version", "p3_contract_version",
                 "rc_contract_version", "p5_contract_version")
    if bool(ck.get("p1_center_recenter", False)) or any(k in ck for k in forbidden[1:]):
        raise ValueError("Fresh P5 must initialize from original Stage-1, not P1/P2/P3/P4/P5.")
    if "use_fuse_depth" not in ck or "pose_depth_mode" not in ck:
        raise ValueError("P5 Stage-1 checkpoint is missing geometry metadata.")
    explicit = set(args.explicit_shared)
    for key in ("use_fuse_depth", "pose_depth_mode"):
        if f"--{key}" in explicit and getattr(cfg, key) != ck[key]:
            raise ValueError(f"--{key} disagrees with P5 initialization checkpoint.")
        setattr(cfg, key, ck[key])


def load_p5_model(args, cfg, device, *, require_p5: bool = False):
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        raise FileNotFoundError(f"P5 checkpoint not found: {cfg.checkpoint_path}")
    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        raise ValueError("P5 requires a full Stage-1/P5 checkpoint with metadata.")
    is_p5 = "p5_contract_version" in ck
    if require_p5 and not is_p5:
        raise ValueError("P5 inference requires a trained P5 checkpoint.")

    names = (
        "p5_hidden", "p5_layers", "p5_heads", "p5_neighbors", "p5_dropout",
        "p5_max_delta_m", "p5_target_radius_m", "p5_corrupt_prob",
        "p5_scene_bias_sigma_m", "p5_scale_sigma", "p5_region_sigma_m", "p5_region_grid",
    )
    if is_p5:
        if int(ck["p5_contract_version"]) != 1:
            raise ValueError("Unsupported P5 checkpoint contract.")
        if int(ck.get("p5_smoke_max_batches", 0)) != 0 and require_p5:
            raise ValueError("Official P5 inference rejects smoke checkpoints.")
        if ck.get("distill_stage") != 1 or ck.get("geometry_depth_source") != "pred":
            raise ValueError("P5 checkpoint lost Stage-1 ancestry metadata.")
        for key in ("use_fuse_depth", "pose_depth_mode"):
            setattr(cfg, key, ck[key])
        for name in names:
            if name not in ck:
                raise ValueError(f"P5 checkpoint is missing {name}.")
            if f"--{name}" in set(args.explicit_p5) and float(getattr(args, name)) != float(ck[name]):
                raise ValueError(f"Explicit --{name} disagrees with checkpoint={ck[name]}.")
            setattr(args, name, ck[name])
    else:
        _validate_stage1(ck, cfg, args)

    from models.economicgrasp_p5 import economicgrasp_dpt_p5
    model = economicgrasp_dpt_p5(
        min_depth=cfg.min_depth, max_depth=cfg.max_depth, bin_num=cfg.bin_num,
        pose_depth_mode=cfg.pose_depth_mode,
        camera_pose_key=ck.get("camera_pose_key", "camera_pose_vec"),
        camera_gravity_key=ck.get("camera_gravity_key", "camera_gravity_vec"),
        pose_hidden_dim=int(ck.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(ck.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(ck.get("ray_gravity_mid_dim", 32)),
        p5_hidden=int(args.p5_hidden), p5_layers=int(args.p5_layers),
        p5_heads=int(args.p5_heads), p5_neighbors=int(args.p5_neighbors),
        p5_dropout=float(args.p5_dropout), p5_max_delta_m=float(args.p5_max_delta_m),
        p5_target_radius_m=float(args.p5_target_radius_m),
        p5_corrupt_prob=float(args.p5_corrupt_prob),
        p5_scene_bias_sigma_m=float(args.p5_scene_bias_sigma_m),
        p5_scale_sigma=float(args.p5_scale_sigma),
        p5_region_sigma_m=float(args.p5_region_sigma_m),
        p5_region_grid=int(args.p5_region_grid),
    ).to(device)
    if is_p5:
        state = {k.removeprefix("module."): v for k, v in ck["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
    else:
        model.load_stage1(ck["model_state_dict"])
    return model, ck, is_p5


def reduce_statistics(stats, device, world):
    names = sorted(stats)
    packed = torch.tensor([stats[k] for k in names], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(packed)
    vals = packed.cpu().tolist()
    return {k: {"sum": s, "count": n, "mean": s / n if n else None}
            for k, (s, n) in zip(names, vals)}


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
