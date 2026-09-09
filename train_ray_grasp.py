#!/usr/bin/env python3
"""Train the shared ray-conditioned grasp tail; never call an analytic evaluator.

Start from a controlled Stage-1 checkpoint, or resume a full P2 checkpoint.
Frozen modules stay in eval mode. Validation uses unique, non-padded DDP shards.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# IMPORTANT: do not import anything from ``models`` before parse_cli() has
# consumed all --ray_* arguments. models/__init__.py imports legacy modules,
# which import utils.arguments and execute its global parse_args().
from utils.ray_grasp_runtime import (
    parse_cli, init_distributed, cleanup, load_model, build_dataset, make_loader,
    move_batch, json_write, reduce_statistics,
)


def sampling_record(indices):
    values = np.asarray(indices, dtype="<i8")
    return {"count": len(indices), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def new_stats(k):
    names = ("cdf", "width", "support", "base_label_point_support", "any_label_point_support",
             "selected_label_point_support", "selected_nonzero", "any_cdf_label_support",
             "selected_cdf_label_support", *[f"selected_k{i}" for i in range(k)])
    return {name: [0., 0.] for name in names}


def add_stats(stats, values):
    for name, (total, count) in values.items():
        stats[name][0] += float(total.detach().double().item())
        stats[name][1] += float(count.detach().double().item())


def objective(sums, weights, world, synchronize):
    """Correct DDP weighting even when ranks have different valid-label counts."""
    names = ("cdf", "width", "support")
    counts = torch.stack([sums[n][1].detach().double() for n in names])
    if synchronize and world > 1:
        dist.all_reduce(counts)
    factor = world if synchronize else 1
    return sum(weights[n] * sums[n][0] * factor / counts[i].clamp_min(1).to(sums[n][0])
               for i, n in enumerate(names))


def epoch_loop(network, raw, loader, cfg, args, device, world, weights, epoch, optimizer=None):
    # Safe here: main() has already called parse_cli(), so the repository-global
    # argparse instance has seen only shared arguments.
    from models.ray_grasp_ops import loss_sums, metric_sums

    training = optimizer is not None
    raw.train(training)
    stats = new_stats(len(raw.kview_grasp_module.offsets_m))
    start = time.monotonic()
    for step, batch in enumerate(loader):
        if args.ray_max_batches and step >= args.ray_max_batches:
            break
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            out = network(batch, with_labels=True)
            sums = loss_sums(out["ray_hypotheses"])
            loss = objective(sums, weights, world, synchronize=training)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, step={step}.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Only trainable parameters are clipped/updated; frozen BN buffers
                # and the dense depth estimator are untouched.
                norm = torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad],
                                                     max_norm=10., error_if_nonfinite=True)
                optimizer.step()
        add_stats(stats, sums)
        add_stats(stats, metric_sums(out["ray_hypotheses"], out["ray_offsets_m"]))
        if training and step % args.ray_log_every == 0 and int(os.environ.get("RANK", 0)) == 0:
            print(f"[RAY][TRAIN] epoch={epoch} step={step}/{len(loader)} "
                  f"local_scaled_loss={loss.item():.6f} grad_norm={float(norm):.5f} "
                  f"elapsed={time.monotonic()-start:.1f}s", flush=True)
        del out, sums, batch, loss
    return reduce_statistics(stats, device, world)


def main():
    # This must be the first operation that can reach utils.arguments.
    args, cfg = parse_cli(training=True)
    if not math.isfinite(args.ray_support_weight) or args.ray_support_weight < 0 or args.ray_log_every <= 0:
        raise ValueError("Support weight must be nonnegative and log interval positive.")
    if cfg.max_epoch <= 0 or cfg.learning_rate <= 0 or cfg.ckpt_save_interval <= 0:
        raise ValueError("Epoch count, learning rate and checkpoint interval must be positive.")
    if cfg.eval_num_workers < 0:
        raise ValueError("eval_num_workers must be nonnegative.")
    rank, world, device = init_distributed(cfg.seed)
    try:
        raw, source, is_ray, offsets = load_model(args, cfg, device)
        from models.economicgrasp_ray import RAY_CONTRACT_VERSION, MAIN_BASE_SHA
        if bool(cfg.resume) != bool(is_ray):
            raise ValueError("Use Stage-1 without --resume; use a P2 checkpoint with --resume.")
        train_base, train_data, train_idx = build_dataset(cfg, "train", args.ray_train_sample_interval, True)
        val_base, val_data, val_idx = build_dataset(cfg, "test_seen", args.ray_eval_sample_interval, True)
        sampling = {"protocol": "per-scene-stride-v1", "train": sampling_record(train_idx),
                    "validation": sampling_record(val_idx),
                    "train_fraction": args.ray_train_sample_interval,
                    "validation_fraction": args.ray_eval_sample_interval}
        weights = {"cdf": float(cfg.score_loss_weight), "width": float(cfg.width_loss_weight),
                   "support": args.ray_support_weight}
        if any(not math.isfinite(x) or x < 0 for x in weights.values()) or weights["cdf"] <= 0:
            raise ValueError("CDF loss weight must be positive; other loss weights must be nonnegative.")
        if is_ray:
            for key, value in (("ray_sampling", sampling), ("ray_loss_weights", weights),
                               ("ray_max_epoch", cfg.max_epoch), ("m_point", cfg.m_point),
                               ("ray_smoke_max_batches", args.ray_max_batches)):
                if source.get(key) != value:
                    raise ValueError(f"Resume protocol differs for {key}: {source.get(key)!r} vs {value!r}")
        train_loader, sampler = make_loader(train_data, train_idx, cfg, rank, world, True)
        val_loader, _ = make_loader(val_data, val_idx, cfg, rank, world, False)
        params = [p for p in raw.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch)
        start_epoch, best = 0, float("inf")
        if is_ray:
            optimizer.load_state_dict(source["optimizer_state_dict"])
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_epoch, best = int(source["next_epoch"]), float(source["best_validation_objective"])
        if start_epoch >= cfg.max_epoch:
            raise ValueError("Resume checkpoint has already reached max_epoch.")
        # Explicit input movement is essential: DDP device_ids=[local] would
        # recursively transfer the enormous variable-size CPU label caches.
        network = DDP(raw, device_ids=None, broadcast_buffers=False,
                      find_unused_parameters=False) if world > 1 else raw
        outdir = Path(cfg.log_dir)
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            if not cfg.resume and (outdir / "checkpoint_latest.tar").exists():
                raise FileExistsError("Choose a new log_dir or explicitly --resume the existing P2 run.")
            json_write(outdir / "ray_training_protocol.json", {
                "base_main_sha": MAIN_BASE_SHA, "init_checkpoint": str(cfg.checkpoint_path),
                "ray_offsets_mm": offsets, "sampling": sampling, "loss_weights": weights,
                "trainable_parameters": sum(p.numel() for p in params), "world_size": world,
                "base_frozen": True, "online_evaluator": False, "cfg": vars(cfg),
                "ray_args": vars(args), "smoke_only": bool(args.ray_max_batches),
                "train_scene_count": len(set(train_base.scenename)),
                "validation_scene_count": len(set(val_base.scenename)),
            })
            print(f"[RAY] train={len(train_idx)}/{len(train_base)} val={len(val_idx)}/{len(val_base)} "
                  f"rays={cfg.m_point} K={len(offsets)} trainable={sum(p.numel() for p in params)} "
                  f"steps_per_rank={len(train_loader)} world={world}; decoder-only; evaluator=OFF", flush=True)
        for epoch in range(start_epoch, cfg.max_epoch):
            # Explicit per-epoch RNG schedule makes resumption independent of the
            # number of validation batches on each unique rank shard.
            epoch_seed = cfg.seed + 1009 * epoch + rank
            random.seed(epoch_seed); np.random.seed(epoch_seed); torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            val_loader.generator.manual_seed(cfg.seed + rank)
            sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]
            train_stats = epoch_loop(network, raw, train_loader, cfg, args, device, world, weights, epoch, optimizer)
            # Bypass DDP for unequal validation shard sizes; synchronize only the
            # final sufficient statistics, never one collective per val batch.
            val_stats = epoch_loop(raw, raw, val_loader, cfg, args, device, world, weights, epoch)
            if val_stats["cdf"]["count"] == 0:
                raise RuntimeError("Validation has no valid CDF targets; inspect cache/grid coverage.")
            val_objective = sum(weights[n] * (val_stats[n]["mean"] or 0.) for n in weights)
            improved = val_objective < best
            best = min(best, val_objective)
            scheduler.step()
            if rank == 0:
                record = {"epoch": epoch, "learning_rate": lr, "train": train_stats,
                          "validation": val_stats, "validation_objective": val_objective,
                          "best_validation_objective": best, "smoke_only": bool(args.ray_max_batches)}
                with (outdir / "ray_epochs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                print(f"[RAY][VAL] epoch={epoch} objective={val_objective:.6f} "
                      f"base_support={val_stats['base_label_point_support']['mean']:.4f} "
                      f"ray_coverage={val_stats['any_label_point_support']['mean']:.4f} "
                      f"selected_support={val_stats['selected_label_point_support']['mean']:.4f} "
                      f"best={best:.6f}", flush=True)
                # Retain original geometry/camera metadata, not original optimizer
                # state. Only the P2 tail optimizer/scheduler is serialized below.
                state = {key: source[key] for key in (
                    "distill_stage", "distill_contract_version", "geometry_depth_source",
                    "seed_selection_mode", "depth_head_executed", "legacy_dataset_use_gt_depth",
                    "camera_pose_key", "camera_gravity_key", "pose_hidden_dim",
                    "ray_gravity_hidden_dim", "ray_gravity_mid_dim") if key in source}
                state.update(model_state_dict=raw.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), epoch=epoch, next_epoch=epoch + 1,
                    best_validation_objective=best, ray_contract_version=RAY_CONTRACT_VERSION,
                    ray_offsets_mm=list(offsets), ray_hidden=raw.ray_hidden, ray_sampling=sampling,
                    ray_loss_weights=weights, ray_max_epoch=cfg.max_epoch, ray_base_main_sha=MAIN_BASE_SHA,
                    ray_base_checkpoint=source.get("ray_base_checkpoint", str(cfg.checkpoint_path)),
                    ray_smoke_max_batches=args.ray_max_batches, ray_base_frozen=True,
                    ray_label_protocol="main-cdf-nearest-point-5mm;separate-geometric-label-support",
                    ray_online_evaluator=False, pose_depth_mode=cfg.pose_depth_mode,
                    use_fuse_depth=cfg.use_fuse_depth, min_depth=cfg.min_depth, max_depth=cfg.max_depth,
                    bin_num=cfg.bin_num, m_point=cfg.m_point, num_view=cfg.num_view,
                    num_angle=cfg.num_angle, num_depth=cfg.num_depth)
                paths = [outdir / "checkpoint_latest.tar"]
                if improved:
                    paths.append(outdir / "checkpoint_best.tar")
                if (epoch + 1) % cfg.ckpt_save_interval == 0 or epoch + 1 == cfg.max_epoch:
                    paths.append(outdir / f"checkpoint_epoch_{epoch:03d}.tar")
                for path in paths:
                    temp = str(path) + ".tmp"
                    torch.save(state, temp)
                    os.replace(temp, path)
        if world > 1:
            dist.barrier()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
