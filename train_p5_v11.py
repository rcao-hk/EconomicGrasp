#!/usr/bin/env python3
"""Train P5-v1.1: repair-to-valid-set with balanced geometry errors.

Differences from P5-v1:
  1. proposals already within the 5-mm valid set receive an identity target;
  2. training alternates native and corrupted batches (50/50 by construction);
  3. every epoch evaluates the same Seen subset twice: native geometry and a
     deterministic corrupted geometry condition;
  4. a "best native" checkpoint is written only when repair beats the native
     baseline in both set violation and within-safe ratio.

No online GraspNet/Dex-Net evaluator is called during training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _parse_v11_flags():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--p5v11_safe_radius_m", type=float, default=0.005)
    p.add_argument("--p5v11_safe_identity_weight", type=float, default=1.0)
    p.add_argument("--p5v11_val_corrupt_seed", type=int, default=55117)
    args, remaining = p.parse_known_args()
    if args.p5v11_safe_radius_m < 0:
        raise ValueError("--p5v11_safe_radius_m must be non-negative.")
    if args.p5v11_safe_identity_weight < 0:
        raise ValueError("--p5v11_safe_identity_weight must be non-negative.")
    sys.argv = [sys.argv[0], *remaining]
    return args


V11 = _parse_v11_flags()
from utils.p5_runtime import parse_p5_cli, load_p5_model

LOSS_NAMES = ("repair", "safe_identity", "unknown_identity")
METRIC_NAMES = (
    "p5v11_target_known_ratio", "p5v11_safe_ratio", "p5v11_repairable_ratio",
    "p5v11_native_label_dist_m", "p5v11_repaired_label_dist_m",
    "p5v11_native_set_violation_m", "p5v11_repaired_set_violation_m",
    "p5v11_set_improvement_m", "p5v11_native_within_safe",
    "p5v11_repaired_within_safe", "p5v11_safe_noharm_ratio",
    "p5v11_repairable_capture_ratio", "p5v11_pred_delta_safe_m",
    "p5v11_pred_delta_repairable_m", "p5v11_pred_delta_unknown_m",
    "p5v11_pred_delta_all_m", "p5v11_query_corruption_abs_m",
    "p5v11_keypoint_visible_ratio", "p5v11_corrupt_branch_ratio",
)


def _sampling_record(indices):
    arr = np.asarray(indices, dtype="<i8")
    return {"count": len(indices), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def _new_stats():
    return {name: [0.0, 0.0] for name in (*LOSS_NAMES, *METRIC_NAMES)}


def _add(stats, values):
    for name, (total, count) in values.items():
        if name not in stats:
            raise KeyError(f"Unexpected P5-v1.1 statistic {name!r}.")
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
    safe_identity = mean("safe_identity", 1)
    unknown_identity = mean("unknown_identity", 2)
    total = (
        float(args.p5_repair_weight) * repair
        + float(V11.p5v11_safe_identity_weight) * safe_identity
        + float(args.p5_unknown_identity_weight) * unknown_identity
    )
    return total, repair.detach(), safe_identity.detach(), unknown_identity.detach()


def _configure_geometry(raw, *, training: bool, condition: str, step: int):
    """Configure P5-v1 without changing its serialized architecture.

    During evaluation, child modules remain in eval mode.  For the corrupted
    validation pass we set only the root ``training`` flag so the existing P5
    forward executes the corruption branch without enabling dropout.
    """
    if training:
        raw.train(True)
        if condition != "balanced":
            raise ValueError("Training condition must be 'balanced'.")
        corrupt = bool(step % 2 == 1)
        raw.p5_corrupt_prob = 1.0 if corrupt else 0.0
        return corrupt

    raw.train(False)
    if condition == "native":
        raw.p5_corrupt_prob = 0.0
        return False
    if condition == "corrupt":
        raw.p5_corrupt_prob = 1.0
        # P5-v1 checks self.training to activate corruption. Children stay eval.
        raw.training = True
        return True
    raise ValueError(f"Unknown P5-v1.1 condition={condition!r}.")


def _loop(network, raw, loader, args, device, world, epoch, *, condition, optimizer=None):
    from utils.p5_v11_ops import (
        prepare_repair_to_set_targets,
        repair_to_set_loss_sums,
        repair_to_set_metric_sums,
    )
    from utils.p5_runtime import move_batch, reduce_statistics

    training = optimizer is not None
    original_prob = float(raw.p5_corrupt_prob)
    stats = _new_stats()
    started = time.monotonic()
    try:
        if not training:
            _configure_geometry(raw, training=False, condition=condition, step=0)
        for step, batch in enumerate(loader):
            if args.p5_max_batches and step >= args.p5_max_batches:
                break
            corrupt_branch = _configure_geometry(
                raw, training=training, condition=condition, step=step
            ) if training else (condition == "corrupt")
            batch = move_batch(batch, device)
            with torch.set_grad_enabled(training):
                out = network(batch, with_labels=True)
                base_targets = {
                    "target_delta_local": out["p5_target_delta_local"],
                    "target_known": out["p5_target_known"],
                    "target_distance_m": out["p5_target_distance_m"],
                    "target_utility": out["p5_target_utility"],
                }
                targets = prepare_repair_to_set_targets(
                    base_targets, safe_radius_m=float(V11.p5v11_safe_radius_m)
                )
                sums = repair_to_set_loss_sums(
                    out["p5_delta_local"], targets,
                    beta_m=float(args.p5_smooth_l1_beta_m),
                )
                loss, repair_mean, safe_mean, unknown_mean = _objective(
                    sums, args, device, world, training
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite P5-v1.1 loss epoch={epoch} step={step}."
                    )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    trainable = [p for p in raw.parameters() if p.requires_grad]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, 10.0, error_if_nonfinite=True
                    )
                    optimizer.step()

            _add(stats, sums)
            _add(stats, repair_to_set_metric_sums(
                out["p5_delta_local"].detach(), out["p5_rotation"].detach(),
                out["p5_proposal_center"].detach(), targets,
            ))
            one = out["p5_delta_local"].new_tensor(1.0)
            _add(stats, {
                "p5v11_query_corruption_abs_m": (
                    out["P5: query corruption abs m"].detach(), one
                ),
                "p5v11_keypoint_visible_ratio": (
                    out["p5_keypoint_visible_ratio"].detach(), one
                ),
                "p5v11_corrupt_branch_ratio": (
                    one * float(corrupt_branch), one
                ),
            })
            if training and step % int(args.p5_log_every) == 0 and int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[P5-v1.1][TRAIN] epoch={epoch} step={step}/{len(loader)} "
                    f"branch={'corrupt' if corrupt_branch else 'native'} "
                    f"loss={loss.item():.6f} repair={repair_mean.item():.6f} "
                    f"safe={safe_mean.item():.6f} unknown={unknown_mean.item():.6f} "
                    f"known={float(targets['target_known'].float().mean()):.3f} "
                    f"corrupt={float(out['P5: query corruption abs m']):.4f}m "
                    f"grad={float(grad_norm):.4f} elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
            del out, base_targets, targets, sums, batch, loss
    finally:
        raw.p5_corrupt_prob = original_prob
        if not training:
            raw.training = False
    return reduce_statistics(stats, device, world)


def _append(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
        f.flush()


def _fmt(x):
    return "NA" if x is None else f"{x:.6f}"


def main():
    args, cfg = parse_p5_cli(training=True)
    from utils.p5_runtime import (
        init_distributed, cleanup_distributed, build_dataset, make_loader, json_write
    )
    if float(V11.p5v11_safe_radius_m) >= float(args.p5_target_radius_m):
        raise ValueError("P5-v1.1 safe radius must be smaller than target radius.")
    if cfg.max_epoch <= 0 or cfg.learning_rate <= 0 or cfg.ckpt_save_interval <= 0:
        raise ValueError("P5-v1.1 max_epoch/lr/checkpoint interval must be positive.")

    rank, world, device = init_distributed(cfg.seed)
    try:
        raw, source, is_p5 = load_p5_model(args, cfg, device)
        from models.economicgrasp_p5 import P5_CONTRACT_VERSION, P5_CONTROLLED_MAIN_SHA
        if bool(cfg.resume) != bool(is_p5):
            raise ValueError(
                "Fresh P5-v1.1: use Stage-1 without --resume; resume only a P5-v1.1 checkpoint."
            )
        if is_p5 and int(source.get("p5_protocol_version", -1)) != 2:
            raise ValueError("P5-v1.1 resume requires p5_protocol_version=2.")

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
            "safe_identity_weight": float(V11.p5v11_safe_identity_weight),
            "unknown_identity_weight": float(args.p5_unknown_identity_weight),
            "smooth_l1_beta_m": float(args.p5_smooth_l1_beta_m),
            "safe_radius_m": float(V11.p5v11_safe_radius_m),
            "unknown_is_negative": False,
        }
        corruption_cfg = {
            "training_balance": "alternate-native-corrupt-batches-50-50",
            "corrupt_branch_probability": 1.0,
            "native_branch_probability": 0.0,
            "scene_bias_sigma_m": float(args.p5_scene_bias_sigma_m),
            "scale_sigma": float(args.p5_scale_sigma),
            "region_sigma_m": float(args.p5_region_sigma_m),
            "region_grid": int(args.p5_region_grid),
            "validation_corrupt_seed": int(V11.p5v11_val_corrupt_seed),
        }
        if is_p5:
            for key, expected in (
                ("p5_sampling", sampling), ("p5_v11_loss", loss_cfg),
                ("p5_v11_corruption", corruption_cfg),
                ("p5_max_epoch", int(cfg.max_epoch)),
                ("p5_smoke_max_batches", int(args.p5_max_batches)),
            ):
                if source.get(key) != expected:
                    raise ValueError(
                        f"P5-v1.1 resume protocol differs for {key}: "
                        f"{source.get(key)!r} vs {expected!r}"
                    )

        train_loader, sampler = make_loader(train_data, train_idx, cfg, rank, world, True)
        val_loader, _ = make_loader(val_data, val_idx, cfg, rank, world, False)
        trainable = [p for p in raw.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("P5-v1.1 has no trainable repair parameters.")
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch
        )
        start_epoch = 0
        best_native = None
        best_corrupt = None
        if is_p5:
            optimizer.load_state_dict(source["optimizer_state_dict"])
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_epoch = int(source["next_epoch"])
            best_native = source.get("best_native_set_violation_m")
            best_corrupt = source.get("best_corrupt_set_violation_m")
        if start_epoch >= cfg.max_epoch:
            raise ValueError("P5-v1.1 checkpoint already reached max_epoch.")

        network = DDP(
            raw, device_ids=None, broadcast_buffers=False, find_unused_parameters=False
        ) if world > 1 else raw
        outdir = Path(cfg.log_dir)
        log_path = outdir / "log_train_p5_v11.txt"
        jsonl_path = outdir / "p5_v11_epochs.jsonl"
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            if not cfg.resume and (outdir / "checkpoint_latest.tar").exists():
                raise FileExistsError("Choose a new P5-v1.1 log_dir or resume explicitly.")
            json_write(outdir / "p5_v11_training_protocol.json", {
                "experiment": "p5-v1.1-repair-to-valid-set",
                "p5_protocol_version": 2,
                "controlled_main_sha": P5_CONTROLLED_MAIN_SHA,
                "init_checkpoint": str(Path(cfg.checkpoint_path).resolve()),
                "stage1_frozen": True,
                "repair_translation_only": True,
                "missing_annotation_is_negative": False,
                "safe_region_target": "identity",
                "balanced_native_corrupt_training": True,
                "dual_validation": ["native", "deterministic_corrupt"],
                "sampling": sampling, "loss": loss_cfg,
                "corruption": corruption_cfg,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "checkpoint_selection": (
                    "write best-native only if repaired set violation < native baseline "
                    "and repaired within-safe ratio >= native within-safe ratio"
                ),
                "online_analytic_evaluator": False,
                "smoke_only": bool(args.p5_max_batches), "world_size": world,
                "train_scene_count": len(set(train_base.scenename)),
                "validation_scene_count": len(set(val_base.scenename)),
            })
            _append(
                log_path,
                f"P5-v1.1 init={cfg.checkpoint_path} train={len(train_idx)}/{len(train_base)} "
                f"val={len(val_idx)}/{len(val_base)} safe={V11.p5v11_safe_radius_m} "
                f"trainable={sum(p.numel() for p in trainable)} world={world} "
                f"lr={cfg.learning_rate} evaluator=OFF",
            )

        for epoch in range(start_epoch, cfg.max_epoch):
            epoch_seed = int(cfg.seed) + 5003 * epoch + rank
            random.seed(epoch_seed); np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed); torch.cuda.manual_seed_all(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            val_loader.generator.manual_seed(int(cfg.seed) + rank)
            sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]

            train_stats = _loop(
                network, raw, train_loader, args, device, world, epoch,
                condition="balanced", optimizer=optimizer,
            )
            # Native validation: exact deployment geometry.
            val_native = _loop(
                raw, raw, val_loader, args, device, world, epoch,
                condition="native", optimizer=None,
            )
            # Corrupted validation: reset RNG every epoch so the geometry errors
            # are identical across checkpoints.
            corrupt_seed = int(V11.p5v11_val_corrupt_seed) + rank
            random.seed(corrupt_seed); np.random.seed(corrupt_seed)
            torch.manual_seed(corrupt_seed); torch.cuda.manual_seed_all(corrupt_seed)
            val_corrupt = _loop(
                raw, raw, val_loader, args, device, world, epoch,
                condition="corrupt", optimizer=None,
            )

            n_base = val_native["p5v11_native_set_violation_m"]["mean"]
            n_rep = val_native["p5v11_repaired_set_violation_m"]["mean"]
            n_base5 = val_native["p5v11_native_within_safe"]["mean"]
            n_rep5 = val_native["p5v11_repaired_within_safe"]["mean"]
            c_base = val_corrupt["p5v11_native_set_violation_m"]["mean"]
            c_rep = val_corrupt["p5v11_repaired_set_violation_m"]["mean"]
            c_base5 = val_corrupt["p5v11_native_within_safe"]["mean"]
            c_rep5 = val_corrupt["p5v11_repaired_within_safe"]["mean"]
            vals = (n_base, n_rep, n_base5, n_rep5, c_base, c_rep, c_base5, c_rep5)
            if any(x is None or not math.isfinite(x) for x in vals):
                raise RuntimeError("P5-v1.1 validation produced non-finite mechanism metrics.")

            native_positive = (n_rep < n_base - 1e-9) and (n_rep5 >= n_base5 - 1e-9)
            corrupt_positive = (c_rep < c_base - 1e-9) and (c_rep5 >= c_base5 - 1e-9)
            improved_native = native_positive and (best_native is None or n_rep < best_native)
            improved_corrupt = corrupt_positive and (best_corrupt is None or c_rep < best_corrupt)
            if improved_native:
                best_native = float(n_rep)
            if improved_corrupt:
                best_corrupt = float(c_rep)
            scheduler.step()

            if rank == 0:
                record = {
                    "epoch": epoch, "learning_rate": lr, "train": train_stats,
                    "validation_native": val_native,
                    "validation_corrupt": val_corrupt,
                    "native_mechanism_positive": bool(native_positive),
                    "corrupt_mechanism_positive": bool(corrupt_positive),
                    "best_native_set_violation_m": best_native,
                    "best_corrupt_set_violation_m": best_corrupt,
                    "smoke_only": bool(args.p5_max_batches),
                }
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                line = (
                    f"epoch={epoch:02d} lr={lr:.8g} "
                    f"N[{n_base:.5f}->{n_rep:.5f},5mm {n_base5:.4f}->{n_rep5:.4f},"
                    f"safeNoHarm={val_native['p5v11_safe_noharm_ratio']['mean']:.4f}] "
                    f"C[{c_base:.5f}->{c_rep:.5f},5mm {c_base5:.4f}->{c_rep5:.4f}] "
                    f"trainBranchCorrupt={train_stats['p5v11_corrupt_branch_ratio']['mean']:.3f} "
                    f"trainCorrupt={train_stats['p5v11_query_corruption_abs_m']['mean']:.5f} "
                    f"nativePositive={int(native_positive)} corruptPositive={int(corrupt_positive)} "
                    f"bestN={_fmt(best_native)} bestC={_fmt(best_corrupt)}"
                )
                _append(log_path, line)
                print("[P5-v1.1][VAL] " + line, flush=True)

                state = {k: source[k] for k in (
                    "distill_stage", "distill_contract_version", "seed_selection_mode",
                    "geometry_depth_source", "depth_head_executed", "pose_depth_mode",
                    "camera_pose_key", "camera_gravity_key", "pose_hidden_dim",
                    "ray_gravity_hidden_dim", "ray_gravity_mid_dim", "use_fuse_depth",
                    "legacy_dataset_use_gt_depth",
                ) if k in source}
                state.update(
                    model_state_dict=raw.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch, next_epoch=epoch + 1,
                    p5_contract_version=P5_CONTRACT_VERSION,
                    p5_protocol_version=2,
                    p5_controlled_main_sha=P5_CONTROLLED_MAIN_SHA,
                    p5_base_checkpoint=source.get(
                        "p5_base_checkpoint", str(Path(cfg.checkpoint_path).resolve())
                    ),
                    p5_hidden=int(args.p5_hidden), p5_layers=int(args.p5_layers),
                    p5_heads=int(args.p5_heads), p5_neighbors=int(args.p5_neighbors),
                    p5_dropout=float(args.p5_dropout),
                    p5_max_delta_m=float(args.p5_max_delta_m),
                    p5_target_radius_m=float(args.p5_target_radius_m),
                    # Serialized architecture keeps a neutral 0.5 corruption
                    # probability; v1.1 training overrides it per batch.
                    p5_corrupt_prob=0.5,
                    p5_scene_bias_sigma_m=float(args.p5_scene_bias_sigma_m),
                    p5_scale_sigma=float(args.p5_scale_sigma),
                    p5_region_sigma_m=float(args.p5_region_sigma_m),
                    p5_region_grid=int(args.p5_region_grid),
                    p5_sampling=sampling,
                    p5_loss={
                        "repair_weight": float(args.p5_repair_weight),
                        "unknown_identity_weight": float(args.p5_unknown_identity_weight),
                        "smooth_l1_beta_m": float(args.p5_smooth_l1_beta_m),
                        "unknown_is_negative": False,
                    },
                    p5_corruption={
                        "train_only": True, "probability": 0.5,
                        "scene_bias_sigma_m": float(args.p5_scene_bias_sigma_m),
                        "scale_sigma": float(args.p5_scale_sigma),
                        "region_sigma_m": float(args.p5_region_sigma_m),
                        "region_grid": int(args.p5_region_grid),
                    },
                    p5_v11_loss=loss_cfg, p5_v11_corruption=corruption_cfg,
                    p5_max_epoch=int(cfg.max_epoch),
                    p5_smoke_max_batches=int(args.p5_max_batches),
                    best_native_set_violation_m=best_native,
                    best_corrupt_set_violation_m=best_corrupt,
                )
                torch.save(state, outdir / "checkpoint_latest.tar")
                if (epoch + 1) % int(cfg.ckpt_save_interval) == 0:
                    torch.save(state, outdir / f"checkpoint_epoch_{epoch:03d}.tar")
                if improved_native:
                    torch.save(state, outdir / "checkpoint_best_native.tar")
                if improved_corrupt:
                    torch.save(state, outdir / "checkpoint_best_corrupt.tar")
            if world > 1:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
