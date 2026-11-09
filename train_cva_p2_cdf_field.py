#!/usr/bin/env python3
"""Train one capacity-matched P2 scratch CDF MLP with CDF BCE only."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, Mapping


def _consume_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=("p2_0", "p2_a", "p2_b", "p2_c"),
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--expected_pose_depth_mode", default="global_film")
    parser.add_argument("--expected_use_fuse_depth", type=int, choices=(0, 1), default=1)
    parser.add_argument("--val_scene_start", type=int, default=90)
    parser.add_argument("--require_all_scenes", type=int, choices=(0, 1), default=1)
    parser.add_argument("--min_frames_per_scene", type=int, default=26)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16, help="Frames per batch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    # Importing models.p2_gripper_cdf_field executes models/__init__.py first.
    # That package imports utils.arguments, whose global parser would otherwise
    # try to parse these trainer-specific flags and abort as "unrecognized".
    sys.argv[:] = [sys.argv[0]]
    return args


ARGS = _consume_args()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exact_action_cdf_common import atomic_save_json, friction_to_cdf_target
from models.p2_gripper_cdf_field import (
    P2_PROBE_VERSION,
    P2ScratchCdfMLP,
    P2FieldConfig,
    active_evidence_blocks,
    validate_variant,
)
from p2_gripper_field_cache import (
    P2GripperFieldCacheDataset,
    collate_p2_gripper_field,
    scan_p2_cache,
)
from p2_gripper_field_common import (
    save_p2_predictor_checkpoint,
    validate_base_checkpoint,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed((seed + worker_id) % (2**32 - 1))
    random.seed(seed + worker_id)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def predictor_kwargs(batch: Mapping[str, torch.Tensor], variant: str):
    active = set(active_evidence_blocks(variant))
    kwargs = {}
    if "pose" in active:
        kwargs["action_pose_feature"] = batch["action_pose_feature"]
    if "projected" in active:
        kwargs["projected_field_feature"] = batch["projected_field_feature"]
    if "ray_depth" in active:
        kwargs["ray_depth_feature"] = batch["ray_depth_feature"]
    return kwargs


def selection_metrics(logits, target, center_group):
    """Select over all angle rows and all depth anchors for each center."""
    utility = torch.sigmoid(logits.float()).mean(dim=-1)  # [R,D]
    true_utility = target.float().mean(dim=-1)             # [R,D]
    selected = []
    oracle = []
    regret = []
    invalid = []
    for group in torch.unique(center_group):
        mask = center_group == group
        pred = utility[mask].reshape(-1)
        true = true_utility[mask].reshape(-1)
        if pred.numel() == 0:
            continue
        index = pred.argmax()
        chosen = true[index]
        best = true.max()
        selected.append(chosen)
        oracle.append(best)
        regret.append((best - chosen).clamp_min(0.0))
        invalid.append((chosen <= 0.0).float())
    if not selected:
        return {
            "selected_target_utility": float("nan"),
            "oracle_target_utility": float("nan"),
            "selected_regret": float("nan"),
            "selected_invalid": float("nan"),
            "num_centers": 0.0,
        }
    return {
        "selected_target_utility": float(torch.stack(selected).mean().item()),
        "oracle_target_utility": float(torch.stack(oracle).mean().item()),
        "selected_regret": float(torch.stack(regret).mean().item()),
        "selected_invalid": float(torch.stack(invalid).mean().item()),
        "num_centers": float(len(selected)),
    }


def accumulate(sums, counts, metrics, weight):
    for key, value in metrics.items():
        if key == "num_centers" or not np.isfinite(value):
            continue
        sums[key] = sums.get(key, 0.0) + float(value) * float(weight)
        counts[key] = counts.get(key, 0.0) + float(weight)


def train_epoch(model, loader, optimizer, scaler, device, amp_enabled, variant):
    model.train()
    total_loss = 0.0
    total_actions = 0
    raw_abs_sum = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        target = friction_to_cdf_target(batch["friction"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, raw = model(
                batch["base_feature"],
                **predictor_kwargs(batch, variant),
            )
            # The only objective in P2.
            loss = F.binary_cross_entropy_with_logits(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        actions = int(target.shape[0] * target.shape[1])
        total_loss += float(loss.detach().item()) * actions
        raw_abs_sum += float(raw.detach().abs().mean().item()) * actions
        total_actions += actions
    return {
        "loss": total_loss / max(total_actions, 1),
        "raw_output_abs_mean": raw_abs_sum / max(total_actions, 1),
        "num_actions": float(total_actions),
    }


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, variant):
    model.eval()
    sums = {
        "loss": 0.0,
        "base_loss": 0.0,
        "utility_abs": 0.0,
        "base_utility_abs": 0.0,
        "raw_abs": 0.0,
    }
    total_actions = 0
    selection_sums: Dict[str, float] = {}
    selection_counts: Dict[str, float] = {}
    base_selection_sums: Dict[str, float] = {}
    base_selection_counts: Dict[str, float] = {}

    for batch in loader:
        batch = move_batch(batch, device)
        target = friction_to_cdf_target(batch["friction"])
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, raw = model(
                batch["base_feature"],
                **predictor_kwargs(batch, variant),
            )
            loss = F.binary_cross_entropy_with_logits(logits, target)
            base_loss = F.binary_cross_entropy_with_logits(
                batch["base_cdf_logits"], target
            )

        actions = int(target.shape[0] * target.shape[1])
        utility = torch.sigmoid(logits.float()).mean(dim=-1)
        base_utility = torch.sigmoid(batch["base_cdf_logits"].float()).mean(dim=-1)
        true_utility = target.float().mean(dim=-1)
        sums["loss"] += float(loss.item()) * actions
        sums["base_loss"] += float(base_loss.item()) * actions
        sums["utility_abs"] += float((utility - true_utility).abs().sum().item())
        sums["base_utility_abs"] += float(
            (base_utility - true_utility).abs().sum().item()
        )
        sums["raw_abs"] += float(raw.float().abs().mean().item()) * actions
        total_actions += actions

        current = selection_metrics(logits, target, batch["center_group"])
        current_base = selection_metrics(
            batch["base_cdf_logits"], target, batch["center_group"]
        )
        accumulate(
            selection_sums,
            selection_counts,
            current,
            current["num_centers"],
        )
        accumulate(
            base_selection_sums,
            base_selection_counts,
            current_base,
            current_base["num_centers"],
        )

    metrics = {
        "loss": sums["loss"] / max(total_actions, 1),
        "base_loss": sums["base_loss"] / max(total_actions, 1),
        "utility_mae": sums["utility_abs"] / max(total_actions, 1),
        "base_utility_mae": sums["base_utility_abs"] / max(total_actions, 1),
        "raw_output_abs_mean": sums["raw_abs"] / max(total_actions, 1),
        "num_actions": float(total_actions),
    }
    for key, value in selection_sums.items():
        metrics[key] = value / max(selection_counts.get(key, 0.0), 1.0)
    for key, value in base_selection_sums.items():
        metrics[f"base_{key}"] = value / max(
            base_selection_counts.get(key, 0.0), 1.0
        )
    return metrics


def main():
    args = ARGS
    variant = validate_variant(args.variant)
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch_size, and hidden_dim must be positive")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(int(args.seed))

    _, _, base_sha = validate_base_checkpoint(
        args.base_checkpoint,
        expected_pose_depth_mode=args.expected_pose_depth_mode,
        expected_use_fuse_depth=bool(args.expected_use_fuse_depth),
    )
    metadata, inventory, failures = scan_p2_cache(
        args.cache_dir,
        expected_source_base_checkpoint_sha256=base_sha,
        strict=True,
        check_values=True,
    )
    if failures:
        raise RuntimeError(f"P2 cache has failures: {failures[:3]}")
    counts = Counter(m.scene_id for m in metadata)
    if bool(args.require_all_scenes):
        missing = sorted(set(range(100)) - set(counts))
        insufficient = {
            scene: int(counts.get(scene, 0))
            for scene in range(100)
            if int(counts.get(scene, 0)) < int(args.min_frames_per_scene)
        }
        if missing or insufficient:
            raise RuntimeError(f"P2 formal cache incomplete: {missing}, {insufficient}")

    field_config = P2FieldConfig(**dict(inventory.field_config))
    dataset_common = dict(
        val_scene_start=int(args.val_scene_start),
        variant=variant,
        expected_source_base_checkpoint_sha256=base_sha,
        expected_field_config_sha256=field_config.sha256(),
        metadata_override=metadata,
        inventory_override=inventory,
    )
    train_dataset = P2GripperFieldCacheDataset(
        args.cache_dir, split="train", **dataset_common
    )
    val_dataset = P2GripperFieldCacheDataset(
        args.cache_dir, split="val", **dataset_common
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_p2_gripper_field,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_enabled = bool(args.amp) and device.type == "cuda"
    model = P2ScratchCdfMLP(
        variant=variant,
        base_feature_dim=inventory.feature_dim,
        image_feature_dim=inventory.image_feature_dim,
        num_depths=inventory.num_depths,
        num_thresholds=inventory.num_thresholds,
        hidden_dim=int(args.hidden_dim),
        increment_bias=float(inventory.cdf_increment_bias),
    ).to(device)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.min_learning_rate),
    )
    scaler = make_scaler(amp_enabled)

    contract = {
        "probe_version": P2_PROBE_VERSION,
        "variant": variant,
        "active_evidence_blocks": list(active_evidence_blocks(variant)),
        "source_base_checkpoint": os.path.abspath(args.base_checkpoint),
        "source_base_checkpoint_sha256": base_sha,
        "cache_inventory": inventory.to_dict(),
        "field_config": field_config.to_dict(),
        "field_config_sha256": field_config.sha256(),
        "predictor_contract": model.contract(),
        "architecture": "capacity_matched_three_linear_layer_scratch_mlp",
        "trainable_parameter_count": int(trainable_parameter_count),
        "training_initialization": "random_scratch_xavier_uniform",
        "uses_p1_checkpoint": False,
        "uses_stage1_or_p1_residual": False,
        "objective": "ordinary_unbalanced_exact_action_cdf_bce_only",
        "val_scene_start": int(args.val_scene_start),
        "train_frames": len(train_dataset),
        "val_frames": len(val_dataset),
        "epochs": int(args.epochs),
        "batch_size_frames": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "min_learning_rate": float(args.min_learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
    }
    atomic_save_json(contract, os.path.join(output_dir, "probe_contract.json"))

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    init_val = evaluate(model, val_loader, device, False, variant)
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(json.dumps({"epoch": 0, "phase": "scratch_init", "val": init_val}) + "\n")
    print(f"[SCRATCH-INIT] {json.dumps(init_val, sort_keys=True)}", flush=True)

    best_val_loss = float(init_val["loss"])
    best_epoch = 0
    best_path = os.path.join(output_dir, "checkpoint_best_p2_scratch.tar")
    save_p2_predictor_checkpoint(
        best_path,
        model,
        variant=variant,
        epoch=0,
        train_metrics={},
        val_metrics=init_val,
        source_base_checkpoint_sha256=base_sha,
        field_config=field_config,
        cache_contract=contract,
        optimizer_state_dict=optimizer.state_dict(),
    )
    atomic_save_json(
        {"best_epoch": 0, "best_val_loss": best_val_loss, "val_metrics": init_val},
        os.path.join(output_dir, "best.json"),
    )

    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            variant,
        )
        val_metrics = evaluate(model, val_loader, device, amp_enabled, variant)
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_sec": time.time() - epoch_start,
            "elapsed_sec": time.time() - started,
        }
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[EPOCH {epoch:03d}] {json.dumps(record, sort_keys=True)}", flush=True)

        save_p2_predictor_checkpoint(
            os.path.join(output_dir, "checkpoint_last_p2_scratch.tar"),
            model,
            variant=variant,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            source_base_checkpoint_sha256=base_sha,
            field_config=field_config,
            cache_contract=contract,
            optimizer_state_dict=optimizer.state_dict(),
        )
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            save_p2_predictor_checkpoint(
                best_path,
                model,
                variant=variant,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                source_base_checkpoint_sha256=base_sha,
                field_config=field_config,
                cache_contract=contract,
                optimizer_state_dict=optimizer.state_dict(),
            )
            atomic_save_json(
                {
                    "best_epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                os.path.join(output_dir, "best.json"),
            )
        if int(args.save_interval) > 0 and epoch % int(args.save_interval) == 0:
            save_p2_predictor_checkpoint(
                os.path.join(output_dir, f"checkpoint_epoch_{epoch:03d}_p2_scratch.tar"),
                model,
                variant=variant,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                source_base_checkpoint_sha256=base_sha,
                field_config=field_config,
                cache_contract=contract,
            )

    print(
        f"[DONE] variant={variant} best_epoch={best_epoch} "
        f"best_val_loss={best_val_loss:.6f} checkpoint={best_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
