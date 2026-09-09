"""Runtime helpers for the isolated P2 experiment; import-safe until parse_cli."""
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
from torch.utils.data import DataLoader, Subset, DistributedSampler

from models.ray_grasp_ops import DEFAULT_OFFSETS_MM, parse_offsets


def parse_cli(training: bool):
    p = argparse.ArgumentParser(add_help=False, description="P2 ray-specific arguments (shared flags below)")
    p.add_argument("--ray_offsets_mm", default=None,
                   help="Comma-separated mm; pass negative values as --ray_offsets_mm=-40,-20,-10,0,10,20,40")
    p.add_argument("--ray_hidden", type=int, default=128)
    p.add_argument("--ray_no_checkpoint_decoder", action="store_true")
    p.add_argument("--ray_max_batches", type=int, default=0, help="Smoke cap only; never a full result.")
    if training:
        p.add_argument("--ray_train_sample_interval", type=float, default=.1)
        p.add_argument("--ray_eval_sample_interval", type=float, default=.1)
        p.add_argument("--ray_support_weight", type=float, default=1.)
        p.add_argument("--ray_log_every", type=int, default=20)
    else:
        p.add_argument("--ray_score_mode", choices=("auto", "raw", "supported"), default="auto")
        p.add_argument("--ray_selection", choices=("best", "zero"), default="best")
        p.add_argument("--ray_run_eval", action="store_true")
        p.add_argument("--ray_eval_only", action="store_true")
        p.add_argument("--ray_eval_workers", type=int, default=10)
    if "--help" in sys.argv or "-h" in sys.argv:
        p.print_help()
    args, remaining = p.parse_known_args()
    explicit = {x.split("=")[0] for x in remaining if x.startswith("--")}
    sys.argv = [sys.argv[0], *remaining]
    from utils.arguments import cfgs
    cfgs.multi_modal, cfgs.use_cdf, cfgs.extend_angle = True, True, True
    cfgs.graspness_mode = cfgs.graspness_mode or "scene"
    if cfgs.graspness_mode not in ("scene", "instance"):
        raise ValueError("graspness_mode must be scene or instance.")
    for key in ("use_obs_depth", "use_gt_depth", "use_depth_comp", "kview_use_collision", "use_top4_view_infer", "pin_memory"):
        if bool(getattr(cfgs, key, False)):
            raise ValueError(f"P2 v1 does not allow --{key}.")
    if cfgs.kview_mode != "A1" or cfgs.oracle_mode:
        raise ValueError("P2 v1 requires native Top-1 A1 and no oracle mode.")
    if args.ray_max_batches < 0 or args.ray_hidden <= 0:
        raise ValueError("Invalid ray batch cap/hidden size.")
    if min(cfgs.batch_size, cfgs.m_point) <= 0 or cfgs.num_workers < 0:
        raise ValueError("Invalid batch size, query count or worker count.")
    if not training and "--sample_interval" not in explicit:
        cfgs.sample_interval = .1
    if training and "--learning_rate" not in explicit:
        cfgs.learning_rate = 1e-4
    args.explicit_shared = sorted(explicit)
    return args, cfgs


def init_distributed(seed=0):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("Full EconomicGrasp requires CUDA extensions. CPU is supported only by the tensor tests.")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(seconds=int(os.environ.get("DDP_TIMEOUT_SEC", 3600))))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    return rank, world, torch.device("cuda", local)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()  # no final barrier on error


def stride_from_fraction(fraction):
    x = float(fraction)
    if not math.isfinite(x) or not 0 < x <= 1:
        raise ValueError("Sampling fraction must be finite and in (0,1].")
    return max(1, int(round(1. / x)))


def scene_indices(dataset, fraction):
    names = dataset.scenename
    if len(names) != len(dataset):
        raise RuntimeError("scenename must be a per-frame array.")
    groups = {}
    for i, scene in enumerate(names):
        groups.setdefault(scene, []).append(i)
    stride = stride_from_fraction(fraction)
    indices = [i for group in groups.values() for i in group[::stride]]
    return indices


def worker_init(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataset(cfg, split, fraction, labels):
    from dataset.graspnet_dataset import GraspNetMultiDataset
    kw = dict(camera=cfg.camera, split=split, num_points=cfg.num_point,
              remove_outlier=True, augment=False, use_gt_depth=False,
              use_fuse_depth=cfg.use_fuse_depth, graspness_mode=cfg.graspness_mode,
              min_depth=cfg.min_depth, max_depth=cfg.max_depth, bin_num=cfg.bin_num,
              load_label=labels)
    if labels:
        kw.update(voxel_size=cfg.voxel_size, depth_strides=1, extend_angle=True, load_grasp_payload=False)
    base = GraspNetMultiDataset(cfg.dataset_root, **kw)
    indices = scene_indices(base, fraction)
    dataset = base
    if labels:
        from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
        dataset = CVAExtendedLabelAdapter(base, dataset_root=cfg.dataset_root, use_cdf=True,
                                         label_folder=cfg.cdf_label_folder,
                                         num_angle=cfg.num_angle, num_depth=cfg.num_depth)
    return base, dataset, indices


def make_loader(dataset, indices, cfg, rank, world, training):
    from dataset.graspnet_dataset import collate_fn
    sampler = None
    if training:
        selected = Subset(dataset, indices)
        sampler = DistributedSampler(selected, num_replicas=world, rank=rank, shuffle=True,
                                     seed=cfg.seed, drop_last=False)
    else:
        selected = Subset(dataset, indices[rank::world])  # no duplicated validation frames
    generator = torch.Generator().manual_seed(cfg.seed + rank)
    workers = cfg.num_workers if training else cfg.eval_num_workers
    loader = DataLoader(selected, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
                        num_workers=workers, pin_memory=False, collate_fn=collate_fn,
                        worker_init_fn=worker_init, generator=generator, persistent_workers=False)
    return loader, sampler


def move_batch(batch, device):
    drop = {"point_clouds", "cloud_colors", "coordinates_for_voxel", "sensor_depth_m", "obs_depth_m"}
    # In particular, variable-size CDF cache lists must remain pageable CPU tensors.
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k not in drop}


def load_model(args, cfg, device):
    from models.economicgrasp_ray import RayConditionedGrasp, RAY_CONTRACT_VERSION
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        raise ValueError("A full Stage-1 or P2 checkpoint with metadata is required.")
    is_ray = "ray_contract_version" in ck
    if is_ray and ck["ray_contract_version"] != RAY_CONTRACT_VERSION:
        raise ValueError("P2 checkpoint contract version mismatch.")
    if ck.get("distill_stage") != 1 or ck.get("distill_contract_version") != 2:
        raise ValueError("Initialization requires the controlled Stage-1 RGB checkpoint.")
    if ck.get("geometry_depth_source") != "pred" or ck.get("seed_selection_mode") != "image_fps":
        raise ValueError("Checkpoint must use predicted geometry and image-FPS.")
    if ck.get("legacy_dataset_use_gt_depth", True) or not ck.get("depth_head_executed", False):
        raise ValueError("Invalid legacy depth/geometry checkpoint protocol.")
    if ck.get("p1_center_recenter", False):
        raise ValueError("Use the original Stage-1 checkpoint, not a P1 recenter model.")
    if "use_fuse_depth" not in ck or "pose_depth_mode" not in ck:
        raise ValueError("Missing depth construction metadata.")
    explicit = set(args.explicit_shared)
    for key in ("use_fuse_depth", "pose_depth_mode"):
        if f"--{key}" in explicit and getattr(cfg, key) != ck[key]:
            raise ValueError(f"--{key} disagrees with checkpoint metadata.")
        setattr(cfg, key, ck[key])
    for key in ("min_depth", "max_depth", "bin_num", "num_view", "num_angle", "num_depth"):
        if key in ck and float(ck[key]) != float(getattr(cfg, key)):
            raise ValueError(f"Checkpoint {key}={ck[key]} differs from CLI={getattr(cfg, key)}.")
    offsets = parse_offsets(args.ray_offsets_mm or (ck.get("ray_offsets_mm") if is_ray else DEFAULT_OFFSETS_MM))
    if is_ray and offsets != tuple(ck["ray_offsets_mm"]):
        raise ValueError("Do not change the trained ray grid at inference/resume; use --ray_selection zero for a slice.")
    if is_ray and int(ck.get("m_point", cfg.m_point)) != cfg.m_point:
        raise ValueError("Keep m_point equal to the trained ray checkpoint to preserve the query budget.")
    hidden = int(ck["ray_hidden"]) if is_ray else args.ray_hidden
    model = RayConditionedGrasp(min_depth=cfg.min_depth, max_depth=cfg.max_depth, bin_num=cfg.bin_num,
        pose_depth_mode=cfg.pose_depth_mode, camera_pose_key=ck.get("camera_pose_key", "camera_pose_vec"),
        camera_gravity_key=ck.get("camera_gravity_key", "camera_gravity_vec"),
        pose_hidden_dim=int(ck.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(ck.get("ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(ck.get("ray_gravity_mid_dim", 32)),
        ray_offsets_mm=offsets, ray_hidden=hidden,
        ray_checkpoint_decoder=not args.ray_no_checkpoint_decoder).to(device)
    if is_ray:
        model.load_state_dict(ck["model_state_dict"], strict=True)
    else:
        model.load_stage1(ck["model_state_dict"])
    return model, ck, is_ray, offsets


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def reduce_statistics(stats, device, world):
    # Each rank must register the same keys, including ranks with zero eval frames.
    names = sorted(stats)
    packed = torch.tensor([stats[k] for k in names], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(packed)
    values = packed.cpu().tolist()
    return {k: {"sum": s, "count": n, "mean": s / n if n else None}
            for k, (s, n) in zip(names, values)}
