#!/usr/bin/env python3
"""Train P3 selection-free cross-depth evidence aggregation.

Fresh training starts from the controlled Stage-1 e15-style RGB checkpoint.
The complete Stage-1 model remains frozen/eval; only the residual cross-depth
aggregator and contextual viability head are optimized.  No analytic GraspNet
or Dex-Net evaluator is called during training.
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

# Import-order contract: this runtime consumes all --p3_* flags before any
# models module can trigger utils.arguments.parse_args().
from utils.p3_ray_runtime import parse_p3_cli, load_p3_model


LOSS_NAMES = ("cdf", "width", "viability_pos", "viability_neg", "joint_pos", "joint_neg")


def _sampling_record(indices):
    x = np.asarray(indices, dtype="<i8")
    return {"count": len(indices), "sha256": hashlib.sha256(x.tobytes()).hexdigest()}


def _metric_names(k):
    names = [
        "p3_base_point_support", "p3_any_point_support", "p3_any_cdf_support",
    ]
    for mode in ("joint", "raw"):
        names.extend([
            f"p3_{mode}_selected_point_support",
            f"p3_{mode}_selected_cdf_support",
            f"p3_{mode}_selected_nonzero",
            f"p3_{mode}_offset_abs_m",
            f"p3_{mode}_offset_signed_m",
            f"p3_{mode}_selected_target_utility",
            f"p3_{mode}_oracle_target_utility",
            f"p3_{mode}_selection_regret",
        ])
        names.extend(f"p3_{mode}_selected_k{i}" for i in range(k))
    return tuple(names)


def _new_stats(k):
    return {name: [0.0, 0.0] for name in (*LOSS_NAMES, *_metric_names(k), "aggregation_delta_norm")}


def _add(stats, values):
    for name, pair in values.items():
        if name not in stats:
            raise KeyError(f"Unexpected P3 statistic {name!r}.")
        total, count = pair
        stats[name][0] += float(total.detach().double().item())
        stats[name][1] += float(count.detach().double().item())


def _global_component_means(sums, device, world, training):
    counts = torch.stack([sums[name][1].detach().double() for name in LOSS_NAMES]).to(device)
    if training and world > 1:
        dist.all_reduce(counts)

    def mean(name, index):
        if float(counts[index].item()) <= 0:
            return sums[name][0] * 0.0
        # DDP averages parameter gradients across ranks. Multiply local sums by
        # world so the resulting averaged gradient equals the global mean.
        factor = float(world) if training else 1.0
        return sums[name][0] * factor / counts[index].to(sums[name][0])

    return {name: mean(name, i) for i, name in enumerate(LOSS_NAMES)}


def _objective(sums, args, device, world, training):
    parts = _global_component_means(sums, device, world, training)
    viability_terms = [parts[name] for name in ("viability_pos", "viability_neg") if sums[name][1].item() > 0]
    joint_terms = [parts[name] for name in ("joint_pos", "joint_neg") if sums[name][1].item() > 0]
    viability = torch.stack(viability_terms).mean() if viability_terms else parts["cdf"] * 0.0
    joint = torch.stack(joint_terms).mean() if joint_terms else parts["cdf"] * 0.0
    total = (
        args.p3_cdf_weight * parts["cdf"]
        + args.p3_width_weight * parts["width"]
        + args.p3_viability_weight * viability
        + args.p3_joint_weight * joint
    )
    detached = {
        "cdf": parts["cdf"].detach(), "width": parts["width"].detach(),
        "viability": viability.detach(), "joint": joint.detach(),
    }
    return total, detached


def _loop(network, raw, loader, args, device, world, epoch, optimizer=None):
    from utils.p3_ray_ops import loss_sums, metric_sums
    from utils.p3_ray_runtime import move_batch, reduce_statistics

    training = optimizer is not None
    raw.train(training)
    stats = _new_stats(int(raw.kview_grasp_module.offsets_m.numel()))
    started = time.monotonic()
    for step, batch in enumerate(loader):
        if args.p3_max_batches and step >= args.p3_max_batches:
            break
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            out = network(batch, with_labels=True)
            sums = loss_sums(out)
            loss, parts = _objective(sums, args, device, world, training)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite P3 loss at epoch={epoch}, step={step}.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                trainable = [p for p in raw.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0, error_if_nonfinite=True)
                optimizer.step()
        _add(stats, sums)
        _add(stats, metric_sums(out))
        delta = out["D: P3 aggregation delta norm"].detach().double()
        _add(stats, {"aggregation_delta_norm": (delta, delta.new_tensor(1.0))})
        if training and step % args.p3_log_every == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[P3][TRAIN] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.6f} cdf={parts['cdf'].item():.5f} "
                f"width={parts['width'].item():.5f} viability={parts['viability'].item():.5f} "
                f"joint={parts['joint'].item():.5f} grad={float(grad_norm):.5f} "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
        del out, sums, batch, loss
    return reduce_statistics(stats, device, world)


def _append(path: Path, text: str):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()


def main():
    args, cfg = parse_p3_cli(training=True)
    from utils.p3_ray_runtime import (
        init_distributed, cleanup_distributed, build_dataset, make_loader, json_write
    )
    if cfg.max_epoch <= 0 or cfg.learning_rate <= 0 or cfg.ckpt_save_interval <= 0:
        raise ValueError("P3 max_epoch, learning_rate and checkpoint interval must be positive.")

    rank, world, device = init_distributed(cfg.seed)
    try:
        raw, source, is_p3, offsets = load_p3_model(args, cfg, device)
        from models.economicgrasp_ray_p3 import P3_CONTRACT_VERSION, P3_BASE_MAIN_SHA
        if bool(cfg.resume) != bool(is_p3):
            raise ValueError(
                "Fresh P3: use the controlled Stage-1 checkpoint without --resume. "
                "Resume P3: pass a P3 checkpoint with --resume."
            )

        train_base, train_data, train_idx = build_dataset(
            cfg, "train", args.p3_train_sample_interval, labels=True
        )
        val_base, val_data, val_idx = build_dataset(
            cfg, "test_seen", args.p3_eval_sample_interval, labels=True
        )
        sampling = {
            "protocol": "per-scene-stride-v1",
            "train": _sampling_record(train_idx),
            "validation": _sampling_record(val_idx),
            "train_fraction": float(args.p3_train_sample_interval),
            "validation_fraction": float(args.p3_eval_sample_interval),
        }
        loss_cfg = {
            "cdf_weight": float(args.p3_cdf_weight),
            "width_weight": float(args.p3_width_weight),
            "viability_weight": float(args.p3_viability_weight),
            "joint_weight": float(args.p3_joint_weight),
            "viability_balance": "equal-positive-negative",
            "joint_balance": "equal-positive-zero",
        }
        if is_p3:
            for key, expected in (
                ("p3_sampling", sampling),
                ("p3_loss", loss_cfg),
                ("p3_max_epoch", int(cfg.max_epoch)),
                ("p3_smoke_max_batches", int(args.p3_max_batches)),
            ):
                if source.get(key) != expected:
                    raise ValueError(f"P3 resume protocol differs for {key}: {source.get(key)!r} vs {expected!r}")

        train_loader, sampler = make_loader(train_data, train_idx, cfg, rank, world, True)
        val_loader, _ = make_loader(val_data, val_idx, cfg, rank, world, False)
        trainable = [p for p in raw.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("P3 has no trainable evidence-aggregation parameters.")
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch)
        start_epoch = 0
        best_utility = -float("inf")
        best_regret = float("inf")
        if is_p3:
            optimizer.load_state_dict(source["optimizer_state_dict"])
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_epoch = int(source["next_epoch"])
            best_utility = float(source["best_validation_selected_target_utility"])
            best_regret = float(source["best_validation_selection_regret"])
        if start_epoch >= cfg.max_epoch:
            raise ValueError("P3 checkpoint already reached max_epoch.")

        # Inputs are moved explicitly. device_ids=None prevents DDP from trying
        # to scatter the large variable-size CPU label lists.
        network = DDP(raw, device_ids=None, broadcast_buffers=False, find_unused_parameters=False) if world > 1 else raw
        outdir = Path(cfg.log_dir)
        log_path = outdir / "log_train_p3.txt"
        jsonl_path = outdir / "p3_epochs.jsonl"
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            if not cfg.resume and (outdir / "checkpoint_latest.tar").exists():
                raise FileExistsError("Choose a new P3 log_dir or resume the existing run explicitly.")
            json_write(outdir / "p3_training_protocol.json", {
                "experiment": "p3-selection-free-cross-depth-evidence-aggregation",
                "base_main_sha": P3_BASE_MAIN_SHA,
                "init_checkpoint": str(Path(cfg.checkpoint_path).resolve()),
                "ray_offsets_mm": list(offsets),
                "p3_hidden": args.p3_hidden,
                "p3_layers": args.p3_layers,
                "p3_heads": args.p3_heads,
                "p3_dropout": args.p3_dropout,
                "sampling": sampling,
                "loss": loss_cfg,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "stage1_frozen": True,
                "hard_depth_selection_before_grasp_head": False,
                "final_candidate_axis": "ray-depth x in-plane-angle x insertion-depth",
                "checkpoint_selection": "max validation selected target utility; regret also tracked",
                "online_analytic_evaluator": False,
                "smoke_only": bool(args.p3_max_batches),
                "world_size": world,
                "train_scene_count": len(set(train_base.scenename)),
                "validation_scene_count": len(set(val_base.scenename)),
            })
            _append(
                log_path,
                f"P3 init={cfg.checkpoint_path} train={len(train_idx)}/{len(train_base)} "
                f"val={len(val_idx)}/{len(val_base)} K={len(offsets)} "
                f"trainable={sum(p.numel() for p in trainable)} world={world} "
                f"lr={cfg.learning_rate} stage1_frozen=1 evaluator=OFF",
            )

        for epoch in range(start_epoch, cfg.max_epoch):
            epoch_seed = int(cfg.seed) + 3011 * epoch + rank
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            val_loader.generator.manual_seed(int(cfg.seed) + rank)
            sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]

            train_stats = _loop(network, raw, train_loader, args, device, world, epoch, optimizer)
            # Validation uses unique rank shards, therefore bypass DDP forward and
            # reduce only sufficient statistics after all local batches finish.
            val_stats = _loop(raw, raw, val_loader, args, device, world, epoch)
            selected_utility = val_stats["p3_joint_selected_target_utility"]["mean"]
            regret = val_stats["p3_joint_selection_regret"]["mean"]
            if selected_utility is None or regret is None or not math.isfinite(selected_utility) or not math.isfinite(regret):
                raise RuntimeError("P3 validation produced no finite joint selection metrics.")
            improved_utility = selected_utility > best_utility
            improved_regret = regret < best_regret
            best_utility = max(best_utility, selected_utility)
            best_regret = min(best_regret, regret)
            scheduler.step()

            if rank == 0:
                record = {
                    "epoch": epoch,
                    "learning_rate": lr,
                    "train": train_stats,
                    "validation": val_stats,
                    "validation_selected_target_utility": selected_utility,
                    "validation_selection_regret": regret,
                    "best_validation_selected_target_utility": best_utility,
                    "best_validation_selection_regret": best_regret,
                    "smoke_only": bool(args.p3_max_batches),
                }
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                line = (
                    f"epoch={epoch:02d} lr={lr:.8g} selectedU={selected_utility:.6f} "
                    f"regret={regret:.6f} pointSupport={val_stats['p3_joint_selected_point_support']['mean']:.4f} "
                    f"anySupport={val_stats['p3_any_point_support']['mean']:.4f} "
                    f"nonzero={val_stats['p3_joint_selected_nonzero']['mean']:.4f} "
                    f"rawU={val_stats['p3_raw_selected_target_utility']['mean']:.6f} "
                    f"delta={val_stats['aggregation_delta_norm']['mean']:.5f} "
                    f"bestU={best_utility:.6f} bestRegret={best_regret:.6f}"
                )
                _append(log_path, line)
                print("[P3][VAL] " + line, flush=True)

                # Preserve only the Stage-1 ancestry metadata required to
                # reconstruct the frozen base plus the complete P3 contract.
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
                    p3_contract_version=P3_CONTRACT_VERSION,
                    p3_base_main_sha=P3_BASE_MAIN_SHA,
                    p3_base_checkpoint=source.get("p3_base_checkpoint", str(Path(cfg.checkpoint_path).resolve())),
                    p3_offsets_mm=list(offsets),
                    p3_hidden=raw.p3_hidden,
                    p3_layers=raw.p3_layers,
                    p3_heads=raw.p3_heads,
                    p3_dropout=raw.p3_dropout,
                    p3_sampling=sampling,
                    p3_loss=loss_cfg,
                    p3_max_epoch=int(cfg.max_epoch),
                    p3_smoke_max_batches=int(args.p3_max_batches),
                    p3_stage1_frozen=True,
                    p3_hard_depth_selection_before_grasp_head=False,
                    best_validation_selected_target_utility=best_utility,
                    best_validation_selection_regret=best_regret,
                    min_depth=cfg.min_depth,
                    max_depth=cfg.max_depth,
                    bin_num=cfg.bin_num,
                    m_point=cfg.m_point,
                    num_view=cfg.num_view,
                    num_angle=cfg.num_angle,
                    num_depth=cfg.num_depth,
                )
                paths = [outdir / "checkpoint_latest.tar"]
                if improved_utility:
                    paths.append(outdir / "checkpoint_best_utility.tar")
                if improved_regret:
                    paths.append(outdir / "checkpoint_best_regret.tar")
                if (epoch + 1) % cfg.ckpt_save_interval == 0 or epoch + 1 == cfg.max_epoch:
                    paths.append(outdir / f"checkpoint_epoch_{epoch:03d}.tar")
                for path in paths:
                    temp = str(path) + ".tmp"
                    torch.save(state, temp)
                    os.replace(temp, path)

        if world > 1:
            dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
