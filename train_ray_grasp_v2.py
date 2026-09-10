#!/usr/bin/env python3
"""Train only the P2-v2 ray-wise cross-depth Transformer selector.

Fresh v2 training requires a completed P2-v1 checkpoint (use e19/latest from the
first experiment).  The complete v1 network stays frozen/eval.  No GraspNet or
Dex-Net analytic evaluator is invoked during training.
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

# Argparse safety: this module imports no models before parse_v2_cli().
from utils.ray_grasp_v2_runtime import parse_v2_cli, load_v2_model


def _sampling_record(indices):
    values = np.asarray(indices, dtype="<i8")
    return {"count": len(indices), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def _stat_names(k):
    return (
        "listwise", "calib_pos", "calib_neg",
        "v2_positive_ray_ratio", "v2_selected_target_utility", "v2_oracle_target_utility",
        "v2_selection_regret", "v2_oracle_hit", "v2_selected_label_point_support",
        "v2_selected_cdf_label_support", "v2_selected_nonzero", "v2_selected_offset_abs_m",
        "v2_selected_offset_signed_m", "v2_selector_entropy", "v2_zero_probability",
        *[f"v2_selected_k{i}" for i in range(k)],
    )


def _new_stats(k):
    return {name: [0.0, 0.0] for name in _stat_names(k)}


def _add_stats(stats, values):
    for name, (total, count) in values.items():
        if name not in stats:
            raise KeyError(f"Unexpected P2-v2 statistic {name!r}.")
        stats[name][0] += float(total.detach().double().item())
        stats[name][1] += float(count.detach().double().item())


def _training_objective(sums, args, device, world):
    """DDP-correct global means with balanced positive/zero calibration."""
    names = ("listwise", "calib_pos", "calib_neg")
    counts = torch.stack([sums[name][1].detach().double() for name in names]).to(device)
    if world > 1:
        dist.all_reduce(counts)

    def global_mean(name, idx):
        if float(counts[idx].item()) <= 0:
            return sums[name][0] * 0.0
        # DDP averages gradients across ranks, hence the world multiplier.
        return sums[name][0] * float(world) / counts[idx].to(sums[name][0])

    listwise = global_mean("listwise", 0)
    calibration_terms = []
    if float(counts[1].item()) > 0:
        calibration_terms.append(global_mean("calib_pos", 1))
    if float(counts[2].item()) > 0:
        calibration_terms.append(global_mean("calib_neg", 2))
    calibration = (
        torch.stack(calibration_terms).mean()
        if calibration_terms
        else listwise * 0.0
    )
    total = args.v2_listwise_weight * listwise + args.v2_calibration_weight * calibration
    return total, listwise.detach(), calibration.detach()


def _loop(network, raw, loader, args, device, world, epoch, optimizer=None):
    from models.ray_grasp_v2_ops import (
        selector_loss_sums,
        selector_metric_sums,
    )

    training = optimizer is not None
    raw.train(training)
    stats = _new_stats(len(raw.kview_grasp_module.offsets_m))
    started = time.monotonic()
    for step, batch in enumerate(loader):
        if args.v2_max_batches and step >= args.v2_max_batches:
            break
        from utils.ray_grasp_runtime import move_batch
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            out = network(batch, with_labels=True)
            targets = {
                "target_utility": out["ray_v2_target_utility"],
                "known": out["ray_v2_target_known"],
                "support": out["ray_v2_target_support"],
                "cdf_support": out["ray_v2_target_cdf_support"],
            }
            losses = selector_loss_sums(
                out["ray_v2_selector_logits"],
                targets,
                target_temperature=args.v2_target_temperature,
            )
            loss, listwise_mean, calibration_mean = _training_objective(
                losses, args, device, world if training else 1
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite P2-v2 loss at epoch={epoch}, step={step}.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                trainable = [p for p in raw.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, 10.0, error_if_nonfinite=True
                )
                optimizer.step()
        _add_stats(stats, losses)
        _add_stats(stats, selector_metric_sums(
            out["ray_v2_selector_logits"].detach(),
            targets,
            out["ray_offsets_m"],
        ))
        if training and step % args.v2_log_every == 0 and int(os.environ.get("RANK", 0)) == 0:
            print(
                f"[P2-V2][TRAIN] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.6f} listwise={listwise_mean.item():.6f} "
                f"calib={calibration_mean.item():.6f} grad={float(grad_norm):.5f} "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
        del out, targets, losses, batch, loss

    from utils.ray_grasp_runtime import reduce_statistics
    return reduce_statistics(stats, device, world)


def _append_log(path: Path, text: str):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()


def main():
    args, ray_args, cfg = parse_v2_cli(training=True)
    from utils.ray_grasp_runtime import (
        init_distributed, cleanup, build_dataset, make_loader, json_write
    )

    if cfg.max_epoch <= 0 or cfg.learning_rate <= 0 or cfg.ckpt_save_interval <= 0:
        raise ValueError("max_epoch, learning_rate and ckpt_save_interval must be positive.")
    rank, world, device = init_distributed(cfg.seed)
    try:
        raw, source, is_v2, offsets = load_v2_model(args, ray_args, cfg, device)
        from models.economicgrasp_ray_v2 import RAY_V2_CONTRACT_VERSION

        if bool(cfg.resume) != bool(is_v2):
            raise ValueError(
                "Fresh P2-v2: pass a P2-v1 checkpoint without --resume. "
                "Resume: pass a P2-v2 checkpoint with --resume."
            )
        train_base, train_data, train_idx = build_dataset(
            cfg, "train", args.v2_train_sample_interval, labels=True
        )
        val_base, val_data, val_idx = build_dataset(
            cfg, "test_seen", args.v2_eval_sample_interval, labels=True
        )
        sampling = {
            "protocol": "per-scene-stride-v1",
            "train": _sampling_record(train_idx),
            "validation": _sampling_record(val_idx),
            "train_fraction": args.v2_train_sample_interval,
            "validation_fraction": args.v2_eval_sample_interval,
        }
        loss_cfg = {
            "listwise_weight": float(args.v2_listwise_weight),
            "calibration_weight": float(args.v2_calibration_weight),
            "target_temperature": float(args.v2_target_temperature),
            "calibration_balance": "equal-positive-zero-when-both-present",
        }
        if is_v2:
            for key, expected in (("ray_v2_sampling", sampling), ("ray_v2_loss", loss_cfg),
                                  ("ray_v2_max_epoch", cfg.max_epoch),
                                  ("ray_v2_smoke_max_batches", args.v2_max_batches)):
                if source.get(key) != expected:
                    raise ValueError(
                        f"Resume protocol differs for {key}: {source.get(key)!r} vs {expected!r}"
                    )

        train_loader, sampler = make_loader(train_data, train_idx, cfg, rank, world, True)
        val_loader, _ = make_loader(val_data, val_idx, cfg, rank, world, False)
        trainable = [p for p in raw.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("P2-v2 has no trainable selector parameters.")
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch
        )
        start_epoch, best_regret = 0, float("inf")
        if is_v2:
            optimizer.load_state_dict(source["optimizer_state_dict"])
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_epoch = int(source["next_epoch"])
            best_regret = float(source["best_validation_selection_regret"])
        if start_epoch >= cfg.max_epoch:
            raise ValueError("P2-v2 checkpoint has already reached max_epoch.")

        network = DDP(
            raw, device_ids=None, broadcast_buffers=False, find_unused_parameters=False
        ) if world > 1 else raw
        outdir = Path(cfg.log_dir)
        protocol_path = outdir / "ray_v2_training_protocol.json"
        human_log = outdir / "log_train_v2.txt"
        jsonl = outdir / "ray_v2_epochs.jsonl"
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            if not cfg.resume and (outdir / "checkpoint_latest.tar").exists():
                raise FileExistsError("Choose a new v2 log_dir or explicitly --resume it.")
            json_write(protocol_path, {
                "experiment": "p2-v2-ray-wise-cross-depth-transformer",
                "init_checkpoint": str(Path(cfg.checkpoint_path).resolve()),
                "init_is_v2": is_v2,
                "ray_offsets_mm": list(offsets),
                "ray_v1_hidden": int(source["ray_hidden"]),
                "v2_hidden": int(args.v2_hidden),
                "v2_layers": int(args.v2_layers),
                "v2_heads": int(args.v2_heads),
                "v2_dropout": float(args.v2_dropout),
                "v2_zero_bias_init": float(args.v2_zero_bias_init),
                "sampling": sampling,
                "loss": loss_cfg,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "base_p2_v1_frozen": True,
                "checkpoint_selection": "minimum validation cross-depth selection regret",
                "online_analytic_evaluator": False,
                "smoke_only": bool(args.v2_max_batches),
                "world_size": world,
                "train_scene_count": len(set(train_base.scenename)),
                "validation_scene_count": len(set(val_base.scenename)),
            })
            _append_log(
                human_log,
                f"P2-v2 init={cfg.checkpoint_path} train={len(train_idx)}/{len(train_base)} "
                f"val={len(val_idx)}/{len(val_base)} K={len(offsets)} "
                f"trainable={sum(p.numel() for p in trainable)} world={world} "
                f"lr={cfg.learning_rate} selector_only=1 evaluator=OFF",
            )
            print(
                f"[P2-V2] train={len(train_idx)}/{len(train_base)} val={len(val_idx)}/{len(val_base)} "
                f"K={len(offsets)} trainable={sum(p.numel() for p in trainable)} world={world}",
                flush=True,
            )

        for epoch in range(start_epoch, cfg.max_epoch):
            epoch_seed = cfg.seed + 2027 * epoch + rank
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            val_loader.generator.manual_seed(cfg.seed + rank)
            sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]

            train_stats = _loop(
                network, raw, train_loader, args, device, world, epoch, optimizer
            )
            # Unique validation shards: bypass DDP forward; synchronize only final stats.
            val_stats = _loop(raw, raw, val_loader, args, device, world, epoch)
            regret = val_stats["v2_selection_regret"]["mean"]
            if regret is None or not math.isfinite(regret):
                raise RuntimeError("Validation produced no finite positive-ray selection regret.")
            improved = regret < best_regret
            best_regret = min(best_regret, regret)
            scheduler.step()

            if rank == 0:
                record = {
                    "epoch": epoch,
                    "learning_rate": lr,
                    "train": train_stats,
                    "validation": val_stats,
                    "validation_selection_regret": regret,
                    "best_validation_selection_regret": best_regret,
                    "smoke_only": bool(args.v2_max_batches),
                }
                with jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                line = (
                    f"epoch={epoch:02d} lr={lr:.8g} regret={regret:.6f} "
                    f"selectedU={val_stats['v2_selected_target_utility']['mean']:.6f} "
                    f"oracleU={val_stats['v2_oracle_target_utility']['mean']:.6f} "
                    f"hit={val_stats['v2_oracle_hit']['mean']:.4f} "
                    f"support={val_stats['v2_selected_label_point_support']['mean']:.4f} "
                    f"cdf_support={val_stats['v2_selected_cdf_label_support']['mean']:.4f} "
                    f"nonzero={val_stats['v2_selected_nonzero']['mean']:.4f} "
                    f"zeroP={val_stats['v2_zero_probability']['mean']:.4f} "
                    f"best_regret={best_regret:.6f}"
                )
                _append_log(human_log, line)
                print("[P2-V2][VAL] " + line, flush=True)

                # Preserve the complete v1 checkpoint contract required by the loader.
                state = {
                    key: source[key]
                    for key in (
                        "distill_stage", "distill_contract_version", "geometry_depth_source",
                        "seed_selection_mode", "depth_head_executed", "legacy_dataset_use_gt_depth",
                        "camera_pose_key", "camera_gravity_key", "pose_hidden_dim",
                        "ray_gravity_hidden_dim", "ray_gravity_mid_dim", "pose_depth_mode",
                        "use_fuse_depth", "min_depth", "max_depth", "bin_num", "m_point",
                        "num_view", "num_angle", "num_depth", "ray_contract_version",
                        "ray_offsets_mm", "ray_hidden", "ray_loss_weights", "ray_label_protocol",
                        "ray_base_main_sha", "ray_base_checkpoint", "ray_base_frozen",
                    )
                    if key in source
                }
                # A v2 checkpoint remains a completed non-smoke P2-v1 base contract.
                state["ray_smoke_max_batches"] = 0
                state.update(
                    model_state_dict=raw.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch,
                    next_epoch=epoch + 1,
                    ray_v2_contract_version=RAY_V2_CONTRACT_VERSION,
                    ray_v2_sampling=sampling,
                    ray_v2_loss=loss_cfg,
                    ray_v2_max_epoch=cfg.max_epoch,
                    ray_v2_smoke_max_batches=args.v2_max_batches,
                    ray_v2_base_checkpoint=source.get(
                        "ray_v2_base_checkpoint", str(Path(cfg.checkpoint_path).resolve())
                    ),
                    base_p2_v1_frozen=True,
                    v2_hidden=int(args.v2_hidden),
                    v2_layers=int(args.v2_layers),
                    v2_heads=int(args.v2_heads),
                    v2_dropout=float(args.v2_dropout),
                    v2_zero_bias_init=float(args.v2_zero_bias_init),
                    best_validation_selection_regret=best_regret,
                    checkpoint_selection="minimum_validation_selection_regret",
                )
                paths = [outdir / "checkpoint_latest.tar"]
                if improved:
                    paths.extend([
                        outdir / "checkpoint_best.tar",
                        outdir / "checkpoint_best_selector.tar",
                    ])
                if (epoch + 1) % cfg.ckpt_save_interval == 0 or epoch + 1 == cfg.max_epoch:
                    paths.append(outdir / f"checkpoint_epoch_{epoch:03d}.tar")
                for path in paths:
                    tmp = str(path) + ".tmp"
                    torch.save(state, tmp)
                    os.replace(tmp, path)

        if world > 1:
            dist.barrier()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
