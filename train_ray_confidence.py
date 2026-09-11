#!/usr/bin/env python3
"""Train only the ray-conditioned confidence/ranking head.

The complete controlled Stage-1 RGB grasp model is frozen and kept in eval mode.
Multi-depth evidence cannot change the predicted center, view, angle, insertion
depth or width; it only learns a multiplicative score gate. No analytic
GraspNet/Dex-Net evaluator is called during training.
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

from utils.ray_confidence_runtime import parse_rc_cli, load_rc_model

LOSS_NAMES = ("calib_pos", "calib_neg", "rank")
METRIC_NAMES = (
    "rc_known_ratio", "rc_positive_ratio", "rc_gate_mean", "rc_gate_positive",
    "rc_gate_negative", "rc_raw_bce", "rc_calibrated_bce", "rc_score_suppression",
    "rc_raw_top1_target", "rc_raw_top10_target", "rc_raw_top50_target",
    "rc_cal_top1_target", "rc_cal_top10_target", "rc_cal_top50_target",
    "rc_zero_point_support",
)


def _sampling_record(indices):
    values = np.asarray(indices, dtype="<i8")
    return {"count": len(indices), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def _new_stats():
    return {name: [0.0, 0.0] for name in (*LOSS_NAMES, *METRIC_NAMES)}


def _add(stats, values):
    for name, (total, count) in values.items():
        if name not in stats:
            raise KeyError(f"Unexpected ray-confidence statistic {name!r}.")
        stats[name][0] += float(total.detach().double().item())
        stats[name][1] += float(count.detach().double().item())


def _objective(sums, args, device, world, training):
    counts = torch.stack([sums[name][1].detach().double() for name in LOSS_NAMES]).to(device)
    if training and world > 1:
        dist.all_reduce(counts)

    def global_mean(name, index):
        if float(counts[index].item()) <= 0:
            return sums[name][0] * 0.0
        factor = float(world) if training else 1.0
        return sums[name][0] * factor / counts[index].to(sums[name][0])

    pos = global_mean("calib_pos", 0)
    neg = global_mean("calib_neg", 1)
    present = []
    if float(counts[0].item()) > 0:
        present.append(pos)
    if float(counts[1].item()) > 0:
        present.append(neg)
    calibration = torch.stack(present).mean() if present else pos * 0.0
    ranking = global_mean("rank", 2)
    total = args.rc_calibration_weight * calibration + args.rc_ranking_weight * ranking
    return total, calibration.detach(), ranking.detach()


def _loop(network, raw, loader, args, device, world, epoch, optimizer=None):
    from utils.ray_confidence_ops import confidence_loss_sums, confidence_metric_sums
    from utils.ray_confidence_runtime import move_batch, reduce_statistics

    training = optimizer is not None
    raw.train(training)
    stats = _new_stats()
    started = time.monotonic()
    for step, batch in enumerate(loader):
        if args.rc_max_batches and step >= args.rc_max_batches:
            break
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            out = network(batch, with_labels=True)
            targets = {
                "target_score": out["rc_target_score"],
                "target_known": out["rc_target_known"],
                "target_positive": out["rc_target_positive"],
                "ideal_gate": out["rc_ideal_gate"],
            }
            sums = confidence_loss_sums(
                out["rc_raw_score"],
                out["rc_confidence_logit"],
                targets,
                rank_temperature=args.rc_rank_temperature,
            )
            loss, calibration_mean, ranking_mean = _objective(
                sums, args, device, world, training
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite ray-confidence loss epoch={epoch} step={step}.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                trainable = [p for p in raw.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0, error_if_nonfinite=True)
                optimizer.step()

        _add(stats, sums)
        metrics = confidence_metric_sums(
            out["rc_raw_score"].detach(),
            out["rc_confidence_logit"].detach(),
            targets,
        )
        known = out["rc_zero_point_known"].bool()
        support = out["rc_zero_point_support"].bool()
        metrics["rc_zero_point_support"] = (
            support.masked_select(known).float().sum(),
            known.sum().to(out["rc_raw_score"].dtype),
        )
        _add(stats, metrics)

        if training and step % args.rc_log_every == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[RC][TRAIN] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.6f} calib={calibration_mean.item():.6f} "
                f"rank={ranking_mean.item():.6f} grad={float(grad_norm):.5f} "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
        del out, targets, sums, batch, loss
    return reduce_statistics(stats, device, world)


def _append(path: Path, text: str):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()


def main():
    args, cfg = parse_rc_cli(training=True)
    from utils.ray_confidence_runtime import (
        init_distributed, cleanup_distributed, build_dataset, make_loader, json_write
    )

    if cfg.max_epoch <= 0 or cfg.learning_rate <= 0 or cfg.ckpt_save_interval <= 0:
        raise ValueError("max_epoch, learning_rate and ckpt_save_interval must be positive.")
    rank, world, device = init_distributed(cfg.seed)
    try:
        raw, source, is_rc, offsets = load_rc_model(args, cfg, device)
        from models.economicgrasp_ray_confidence import RC_CONTRACT_VERSION, RC_BASE_MAIN_SHA
        if bool(cfg.resume) != bool(is_rc):
            raise ValueError(
                "Fresh RC training: pass original Stage-1 without --resume. "
                "Resume RC: pass an RC checkpoint with --resume."
            )

        train_base, train_data, train_idx = build_dataset(
            cfg, "train", args.rc_train_sample_interval, labels=True
        )
        val_base, val_data, val_idx = build_dataset(
            cfg, "test_seen", args.rc_eval_sample_interval, labels=True
        )
        sampling = {
            "protocol": "per-scene-stride-v1",
            "train": _sampling_record(train_idx),
            "validation": _sampling_record(val_idx),
            "train_fraction": float(args.rc_train_sample_interval),
            "validation_fraction": float(args.rc_eval_sample_interval),
        }
        loss_cfg = {
            "calibration_weight": float(args.rc_calibration_weight),
            "ranking_weight": float(args.rc_ranking_weight),
            "rank_temperature": float(args.rc_rank_temperature),
            "calibration_balance": "equal-positive-zero",
            "final_score_formula": "stage1_raw_score * sigmoid(confidence_logit)",
        }
        if is_rc:
            for key, expected in (
                ("rc_sampling", sampling),
                ("rc_loss", loss_cfg),
                ("rc_max_epoch", int(cfg.max_epoch)),
                ("rc_smoke_max_batches", int(args.rc_max_batches)),
            ):
                if source.get(key) != expected:
                    raise ValueError(f"Resume protocol differs for {key}: {source.get(key)!r} vs {expected!r}")

        train_loader, sampler = make_loader(train_data, train_idx, cfg, rank, world, True)
        val_loader, _ = make_loader(val_data, val_idx, cfg, rank, world, False)
        trainable = [p for p in raw.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("Ray-confidence model has no trainable parameters.")
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch)

        start_epoch = 0
        best_rank = float("inf")
        best_top10 = -float("inf")
        if is_rc:
            optimizer.load_state_dict(source["optimizer_state_dict"])
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_epoch = int(source["next_epoch"])
            best_rank = float(source["best_validation_rank_loss"])
            best_top10 = float(source["best_validation_cal_top10_target"])
        if start_epoch >= cfg.max_epoch:
            raise ValueError("RC checkpoint already reached max_epoch.")

        network = DDP(raw, device_ids=None, broadcast_buffers=False, find_unused_parameters=False) if world > 1 else raw
        outdir = Path(cfg.log_dir)
        log_path = outdir / "log_train_rc.txt"
        jsonl_path = outdir / "rc_epochs.jsonl"
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            if not cfg.resume and (outdir / "checkpoint_latest.tar").exists():
                raise FileExistsError("Choose a new RC log_dir or explicitly resume it.")
            json_write(outdir / "rc_training_protocol.json", {
                "experiment": "ray-conditioned-grasp-confidence-calibration",
                "base_main_sha": RC_BASE_MAIN_SHA,
                "init_checkpoint": str(Path(cfg.checkpoint_path).resolve()),
                "ray_offsets_mm": list(offsets),
                "rc_hidden": int(args.rc_hidden),
                "rc_layers": int(args.rc_layers),
                "rc_heads": int(args.rc_heads),
                "rc_dropout": float(args.rc_dropout),
                "rc_init_bias": float(args.rc_init_bias),
                "sampling": sampling,
                "loss": loss_cfg,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "stage1_frozen": True,
                "pose_intervention": "none",
                "center_intervention": "none",
                "cdf_representation_intervention": "none",
                "only_deployable_change": "final grasp score/ranking",
                "online_analytic_evaluator": False,
                "smoke_only": bool(args.rc_max_batches),
                "world_size": world,
                "train_scene_count": len(set(train_base.scenename)),
                "validation_scene_count": len(set(val_base.scenename)),
            })
            _append(
                log_path,
                f"RC init={cfg.checkpoint_path} train={len(train_idx)}/{len(train_base)} "
                f"val={len(val_idx)}/{len(val_base)} K={len(offsets)} "
                f"trainable={sum(p.numel() for p in trainable)} world={world} "
                f"lr={cfg.learning_rate} stage1_frozen=1 score_only=1 evaluator=OFF",
            )

        for epoch in range(start_epoch, cfg.max_epoch):
            epoch_seed = int(cfg.seed) + 4013 * epoch + rank
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            val_loader.generator.manual_seed(int(cfg.seed) + rank)
            sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]

            train_stats = _loop(network, raw, train_loader, args, device, world, epoch, optimizer)
            val_stats = _loop(raw, raw, val_loader, args, device, world, epoch)
            rank_loss = val_stats["rank"]["mean"]
            top10 = val_stats["rc_cal_top10_target"]["mean"]
            if rank_loss is None or top10 is None or not math.isfinite(rank_loss) or not math.isfinite(top10):
                raise RuntimeError("RC validation produced no finite ranking metrics.")
            improved_rank = rank_loss < best_rank
            improved_top10 = top10 > best_top10
            best_rank = min(best_rank, rank_loss)
            best_top10 = max(best_top10, top10)
            scheduler.step()

            if rank == 0:
                record = {
                    "epoch": epoch,
                    "learning_rate": lr,
                    "train": train_stats,
                    "validation": val_stats,
                    "validation_rank_loss": rank_loss,
                    "validation_cal_top10_target": top10,
                    "best_validation_rank_loss": best_rank,
                    "best_validation_cal_top10_target": best_top10,
                    "smoke_only": bool(args.rc_max_batches),
                }
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                line = (
                    f"epoch={epoch:02d} lr={lr:.8g} rank={rank_loss:.6f} "
                    f"rawTop10={val_stats['rc_raw_top10_target']['mean']:.6f} "
                    f"calTop10={top10:.6f} rawTop1={val_stats['rc_raw_top1_target']['mean']:.6f} "
                    f"calTop1={val_stats['rc_cal_top1_target']['mean']:.6f} "
                    f"gate={val_stats['rc_gate_mean']['mean']:.4f} "
                    f"gatePos={val_stats['rc_gate_positive']['mean']:.4f} "
                    f"gateNeg={val_stats['rc_gate_negative']['mean']:.4f} "
                    f"rawBCE={val_stats['rc_raw_bce']['mean']:.5f} "
                    f"calBCE={val_stats['rc_calibrated_bce']['mean']:.5f} "
                    f"support={val_stats['rc_zero_point_support']['mean']:.4f} "
                    f"bestRank={best_rank:.6f} bestTop10={best_top10:.6f}"
                )
                _append(log_path, line)
                print("[RC][VAL] " + line, flush=True)

                state = {
                    key: source[key]
                    for key in (
                        "distill_stage", "distill_contract_version", "seed_selection_mode",
                        "geometry_depth_source", "depth_head_executed", "pose_depth_mode",
                        "camera_pose_key", "camera_gravity_key", "pose_hidden_dim",
                        "ray_gravity_hidden_dim", "ray_gravity_mid_dim", "use_fuse_depth",
                        "legacy_dataset_use_gt_depth",
                    )
                    if key in source
                }
                state.update(
                    model_state_dict=raw.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch,
                    next_epoch=epoch + 1,
                    rc_contract_version=RC_CONTRACT_VERSION,
                    rc_base_main_sha=RC_BASE_MAIN_SHA,
                    rc_base_checkpoint=source.get("rc_base_checkpoint", str(Path(cfg.checkpoint_path).resolve())),
                    rc_offsets_mm=list(offsets),
                    rc_hidden=int(args.rc_hidden),
                    rc_layers=int(args.rc_layers),
                    rc_heads=int(args.rc_heads),
                    rc_dropout=float(args.rc_dropout),
                    rc_init_bias=float(args.rc_init_bias),
                    rc_sampling=sampling,
                    rc_loss=loss_cfg,
                    rc_max_epoch=int(cfg.max_epoch),
                    rc_smoke_max_batches=int(args.rc_max_batches),
                    best_validation_rank_loss=float(best_rank),
                    best_validation_cal_top10_target=float(best_top10),
                    score_only_intervention=True,
                )
                torch.save(state, outdir / "checkpoint_latest.tar")
                if (epoch + 1) % int(cfg.ckpt_save_interval) == 0 or epoch + 1 == cfg.max_epoch:
                    torch.save(state, outdir / f"checkpoint_epoch_{epoch:03d}.tar")
                if improved_rank:
                    torch.save(state, outdir / "checkpoint_best_rank.tar")
                if improved_top10:
                    torch.save(state, outdir / "checkpoint_best_top10.tar")

        if world > 1:
            dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
