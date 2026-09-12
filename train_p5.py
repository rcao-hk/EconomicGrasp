#!/usr/bin/env python3
"""Train P5 geometry-error-aware, gripper-centric grasp repair."""
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

from utils.p5_runtime import parse_p5_cli, load_p5_model

LOSS_NAMES = ("repair", "unknown_identity")
METRIC_NAMES = (
    "p5_target_known_ratio", "p5_native_target_dist_m", "p5_repaired_target_dist_m",
    "p5_repair_improvement_m", "p5_native_within5mm", "p5_repaired_within5mm",
    "p5_pred_delta_abs_m", "p5_target_delta_abs_m",
    "p5_query_corruption_abs_m", "p5_keypoint_visible_ratio",
)


def _sampling_record(indices):
    arr = np.asarray(indices, dtype="<i8")
    return {"count": len(indices), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def _new_stats():
    return {name: [0.0, 0.0] for name in (*LOSS_NAMES, *METRIC_NAMES)}


def _add(stats, values):
    for name, (total, count) in values.items():
        if name not in stats:
            raise KeyError(f"Unexpected P5 statistic {name!r}.")
        stats[name][0] += float(total.detach().double().item())
        stats[name][1] += float(count.detach().double().item())


def _objective(sums, args, device, world, training):
    counts = torch.stack([sums[n][1].detach().double() for n in LOSS_NAMES]).to(device)
    if training and world > 1:
        dist.all_reduce(counts)
    def mean(name, idx):
        if float(counts[idx].item()) <= 0:
            return sums[name][0] * 0.0
        factor = float(world) if training else 1.0
        return sums[name][0] * factor / counts[idx].to(sums[name][0])
    repair = mean("repair", 0)
    identity = mean("unknown_identity", 1)
    total = float(args.p5_repair_weight) * repair + identity
    return total, repair.detach(), identity.detach()


def _loop(network, raw, loader, args, device, world, epoch, optimizer=None):
    from utils.p5_ops import repair_loss_sums, repair_metric_sums
    from utils.p5_runtime import move_batch, reduce_statistics

    training = optimizer is not None
    raw.train(training)
    stats = _new_stats()
    started = time.monotonic()
    for step, batch in enumerate(loader):
        if args.p5_max_batches and step >= args.p5_max_batches:
            break
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            out = network(batch, with_labels=True)
            targets = {
                "target_delta_local": out["p5_target_delta_local"],
                "target_known": out["p5_target_known"],
                "target_distance_m": out["p5_target_distance_m"],
                "target_utility": out["p5_target_utility"],
            }
            sums = repair_loss_sums(
                out["p5_delta_local"], targets,
                beta_m=float(args.p5_smooth_l1_beta_m),
                unknown_identity_weight=float(args.p5_unknown_identity_weight),
            )
            loss, repair_mean, identity_mean = _objective(
                sums, args, device, world, training
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite P5 loss at epoch={epoch}, step={step}.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                trainable = [p for p in raw.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0, error_if_nonfinite=True)
                optimizer.step()
        _add(stats, sums)
        metrics = repair_metric_sums(
            out["p5_delta_local"].detach(), out["p5_rotation"].detach(),
            out["p5_proposal_center"].detach(), targets,
        )
        _add(stats, metrics)
        one = out["p5_delta_local"].new_tensor(1.0)
        _add(stats, {
            "p5_query_corruption_abs_m": (out["P5: query corruption abs m"].detach(), one),
            "p5_keypoint_visible_ratio": (out["p5_keypoint_visible_ratio"].detach(), one),
        })
        if training and step % int(args.p5_log_every) == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[P5][TRAIN] epoch={epoch} step={step}/{len(loader)} loss={loss.item():.6f} "
                f"repair={repair_mean.item():.6f} identity={identity_mean.item():.6f} "
                f"known={float(out['p5_target_known'].float().mean()):.3f} "
                f"corrupt={float(out['P5: query corruption abs m']):.4f}m "
                f"grad={float(grad_norm):.4f} elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
        del out, targets, sums, batch, loss
    return reduce_statistics(stats, device, world)


def _append(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
        f.flush()


def main():
    args, cfg = parse_p5_cli(training=True)
    from utils.p5_runtime import init_distributed, cleanup_distributed, build_dataset, make_loader, json_write
    if cfg.max_epoch <= 0 or cfg.learning_rate <= 0 or cfg.ckpt_save_interval <= 0:
        raise ValueError("P5 max_epoch, learning_rate and checkpoint interval must be positive.")
    rank, world, device = init_distributed(cfg.seed)
    try:
        raw, source, is_p5 = load_p5_model(args, cfg, device)
        from models.economicgrasp_p5 import P5_CONTRACT_VERSION, P5_CONTROLLED_MAIN_SHA
        if bool(cfg.resume) != bool(is_p5):
            raise ValueError(
                "Fresh P5: pass the controlled Stage-1 checkpoint without --resume. "
                "Resume P5: pass a P5 checkpoint with --resume."
            )
        train_base, train_data, train_idx = build_dataset(
            cfg, "train", args.p5_train_sample_interval, labels=True
        )
        val_base, val_data, val_idx = build_dataset(
            cfg, "test_seen", args.p5_eval_sample_interval, labels=True
        )
        sampling = {
            "protocol": "per-scene-stride-v1",
            "train": _sampling_record(train_idx),
            "validation": _sampling_record(val_idx),
            "train_fraction": float(args.p5_train_sample_interval),
            "validation_fraction": float(args.p5_eval_sample_interval),
        }
        loss_cfg = {
            "repair_weight": float(args.p5_repair_weight),
            "unknown_identity_weight": float(args.p5_unknown_identity_weight),
            "smooth_l1_beta_m": float(args.p5_smooth_l1_beta_m),
            "unknown_is_negative": False,
        }
        corruption_cfg = {
            "train_only": True,
            "probability": float(args.p5_corrupt_prob),
            "scene_bias_sigma_m": float(args.p5_scene_bias_sigma_m),
            "scale_sigma": float(args.p5_scale_sigma),
            "region_sigma_m": float(args.p5_region_sigma_m),
            "region_grid": int(args.p5_region_grid),
        }
        if is_p5:
            for key, expected in (("p5_sampling", sampling), ("p5_loss", loss_cfg),
                                  ("p5_corruption", corruption_cfg),
                                  ("p5_max_epoch", int(cfg.max_epoch)),
                                  ("p5_smoke_max_batches", int(args.p5_max_batches))):
                if source.get(key) != expected:
                    raise ValueError(f"P5 resume protocol differs for {key}: {source.get(key)!r} vs {expected!r}")

        train_loader, sampler = make_loader(train_data, train_idx, cfg, rank, world, True)
        val_loader, _ = make_loader(val_data, val_idx, cfg, rank, world, False)
        trainable = [p for p in raw.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("P5 has no trainable repair parameters.")
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch)
        start_epoch, best_dist, best_within = 0, float("inf"), -float("inf")
        if is_p5:
            optimizer.load_state_dict(source["optimizer_state_dict"])
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_epoch = int(source["next_epoch"])
            best_dist = float(source["best_validation_repaired_target_dist_m"])
            best_within = float(source["best_validation_repaired_within5mm"])
        if start_epoch >= cfg.max_epoch:
            raise ValueError("P5 checkpoint already reached max_epoch.")

        network = DDP(raw, device_ids=None, broadcast_buffers=False, find_unused_parameters=False) if world > 1 else raw
        outdir = Path(cfg.log_dir)
        log_path = outdir / "log_train_p5.txt"
        jsonl_path = outdir / "p5_epochs.jsonl"
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            if not cfg.resume and (outdir / "checkpoint_latest.tar").exists():
                raise FileExistsError("Choose a new P5 log_dir or explicitly resume it.")
            json_write(outdir / "p5_training_protocol.json", {
                "experiment": "p5-geometry-error-aware-gripper-repair",
                "controlled_main_sha": P5_CONTROLLED_MAIN_SHA,
                "init_checkpoint": str(Path(cfg.checkpoint_path).resolve()),
                "stage1_frozen": True,
                "repair_translation_only": True,
                "same_view_angle_depth_supervision": True,
                "missing_annotation_is_negative": False,
                "gripper_keypoints": 11,
                "cross_ray_neighbors": int(args.p5_neighbors),
                "target_radius_m": float(args.p5_target_radius_m),
                "max_delta_m": float(args.p5_max_delta_m),
                "architecture": {
                    "hidden": int(args.p5_hidden), "layers": int(args.p5_layers),
                    "heads": int(args.p5_heads), "dropout": float(args.p5_dropout),
                },
                "sampling": sampling, "loss": loss_cfg, "corruption": corruption_cfg,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "checkpoint_selection": "minimum validation repaired-target distance; within-5mm also saved",
                "online_analytic_evaluator": False,
                "smoke_only": bool(args.p5_max_batches), "world_size": world,
                "train_scene_count": len(set(train_base.scenename)),
                "validation_scene_count": len(set(val_base.scenename)),
            })
            _append(log_path,
                    f"P5 init={cfg.checkpoint_path} train={len(train_idx)}/{len(train_base)} "
                    f"val={len(val_idx)}/{len(val_base)} trainable={sum(p.numel() for p in trainable)} "
                    f"world={world} lr={cfg.learning_rate} evaluator=OFF")

        for epoch in range(start_epoch, cfg.max_epoch):
            epoch_seed = int(cfg.seed) + 5003 * epoch + rank
            random.seed(epoch_seed); np.random.seed(epoch_seed); torch.manual_seed(epoch_seed); torch.cuda.manual_seed_all(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            val_loader.generator.manual_seed(int(cfg.seed) + rank)
            sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]
            train_stats = _loop(network, raw, train_loader, args, device, world, epoch, optimizer)
            val_stats = _loop(raw, raw, val_loader, args, device, world, epoch)
            dist_m = val_stats["p5_repaired_target_dist_m"]["mean"]
            within = val_stats["p5_repaired_within5mm"]["mean"]
            if dist_m is None or within is None or not math.isfinite(dist_m) or not math.isfinite(within):
                raise RuntimeError("P5 validation produced no finite repair metrics.")
            improved_dist = dist_m < best_dist
            improved_within = within > best_within
            best_dist, best_within = min(best_dist, dist_m), max(best_within, within)
            scheduler.step()

            if rank == 0:
                record = {
                    "epoch": epoch, "learning_rate": lr, "train": train_stats,
                    "validation": val_stats,
                    "validation_repaired_target_dist_m": dist_m,
                    "validation_repaired_within5mm": within,
                    "best_validation_repaired_target_dist_m": best_dist,
                    "best_validation_repaired_within5mm": best_within,
                    "smoke_only": bool(args.p5_max_batches),
                }
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                line = (
                    f"epoch={epoch:02d} lr={lr:.8g} known={val_stats['p5_target_known_ratio']['mean']:.4f} "
                    f"nativeDist={val_stats['p5_native_target_dist_m']['mean']:.5f} "
                    f"repairDist={dist_m:.5f} improve={val_stats['p5_repair_improvement_m']['mean']:.5f} "
                    f"native5={val_stats['p5_native_within5mm']['mean']:.4f} repaired5={within:.4f} "
                    f"predDelta={val_stats['p5_pred_delta_abs_m']['mean']:.5f} "
                    f"trainCorrupt={train_stats['p5_query_corruption_abs_m']['mean']:.5f} "
                    f"bestDist={best_dist:.5f} best5={best_within:.4f}"
                )
                _append(log_path, line); print("[P5][VAL] " + line, flush=True)

                state = {k: source[k] for k in (
                    "distill_stage", "distill_contract_version", "seed_selection_mode",
                    "geometry_depth_source", "depth_head_executed", "pose_depth_mode",
                    "camera_pose_key", "camera_gravity_key", "pose_hidden_dim",
                    "ray_gravity_hidden_dim", "ray_gravity_mid_dim", "use_fuse_depth",
                    "legacy_dataset_use_gt_depth",
                ) if k in source}
                state.update(
                    model_state_dict=raw.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), epoch=epoch, next_epoch=epoch + 1,
                    p5_contract_version=P5_CONTRACT_VERSION,
                    p5_controlled_main_sha=P5_CONTROLLED_MAIN_SHA,
                    p5_base_checkpoint=source.get("p5_base_checkpoint", str(Path(cfg.checkpoint_path).resolve())),
                    p5_hidden=int(args.p5_hidden), p5_layers=int(args.p5_layers),
                    p5_heads=int(args.p5_heads), p5_neighbors=int(args.p5_neighbors),
                    p5_dropout=float(args.p5_dropout), p5_max_delta_m=float(args.p5_max_delta_m),
                    p5_target_radius_m=float(args.p5_target_radius_m),
                    p5_corrupt_prob=float(args.p5_corrupt_prob),
                    p5_scene_bias_sigma_m=float(args.p5_scene_bias_sigma_m),
                    p5_scale_sigma=float(args.p5_scale_sigma),
                    p5_region_sigma_m=float(args.p5_region_sigma_m),
                    p5_region_grid=int(args.p5_region_grid),
                    p5_sampling=sampling, p5_loss=loss_cfg, p5_corruption=corruption_cfg,
                    p5_max_epoch=int(cfg.max_epoch), p5_smoke_max_batches=int(args.p5_max_batches),
                    best_validation_repaired_target_dist_m=best_dist,
                    best_validation_repaired_within5mm=best_within,
                )
                torch.save(state, outdir / "checkpoint_latest.tar")
                if (epoch + 1) % int(cfg.ckpt_save_interval) == 0:
                    torch.save(state, outdir / f"checkpoint_epoch_{epoch:03d}.tar")
                if improved_dist:
                    torch.save(state, outdir / "checkpoint_best_distance.tar")
                if improved_within:
                    torch.save(state, outdir / "checkpoint_best_5mm.tar")
            if world > 1:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
