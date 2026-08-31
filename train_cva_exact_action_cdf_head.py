#!/usr/bin/env python3
"""Fine-tune only the existing current CVA-CDF head on exact action labels.

The cached center, Top-1 view, in-plane angle, depth anchor, predicted width,
and CDF-head input all come from one frozen corrected Stage-1 checkpoint. Only
that checkpoint's existing ``decoder.cdf_head`` weights and bias are optimized.
The objective is ordinary unbalanced BCE over the six exact evaluator CDF
outcomes. No auxiliary head, scorer MLP, ranking loss, balancing, KD, or legacy
compatibility path is included.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exact_action_cdf_cache import (
    ExactActionCdfCacheDataset,
    collate_exact_action_cdf,
    scan_cache,
)
from exact_action_cdf_common import (
    CACHE_SCHEMA_VERSION,
    FRICTION_THRESHOLDS,
    PROBE_VERSION,
    CurrentCdfHeadOnly,
    atomic_save_json,
    friction_to_cdf_target,
    merge_cdf_head_into_full_checkpoint,
    validate_current_stage1_cdf_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed((seed + worker_id) % (2**32 - 1))
    random.seed(seed + worker_id)


def make_grad_scaler(enabled: bool):
    """Use the current AMP API with a compatibility fallback."""
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _selection_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    center_group: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate ranking within each cached fixed center over angle x depth."""
    probability = torch.sigmoid(logits)
    predicted_utility = probability.mean(dim=-1)  # [R,D]
    target_utility = target.mean(dim=-1)           # [R,D]

    selected_targets = []
    oracle_targets = []
    regrets = []
    selected_invalid = []
    for group_id in torch.unique(center_group):
        mask = center_group == group_id
        pred = predicted_utility[mask].reshape(-1)
        true = target_utility[mask].reshape(-1)
        if pred.numel() == 0:
            continue
        selected_index = torch.argmax(pred)
        selected = true[selected_index]
        oracle = true.max()
        selected_targets.append(selected)
        oracle_targets.append(oracle)
        regrets.append((oracle - selected).clamp_min(0.0))
        selected_invalid.append((selected <= 0.0).float())

    if not selected_targets:
        return {
            "selected_target_utility": float("nan"),
            "oracle_target_utility": float("nan"),
            "selected_regret": float("nan"),
            "selected_invalid": float("nan"),
            "num_centers": 0.0,
        }
    return {
        "selected_target_utility": float(torch.stack(selected_targets).mean().item()),
        "oracle_target_utility": float(torch.stack(oracle_targets).mean().item()),
        "selected_regret": float(torch.stack(regrets).mean().item()),
        "selected_invalid": float(torch.stack(selected_invalid).mean().item()),
        "num_centers": float(len(selected_targets)),
    }


def _accumulate_weighted(
    sums: Dict[str, float],
    counts: Dict[str, float],
    metrics: Mapping[str, float],
    weight: float,
) -> None:
    for key, value in metrics.items():
        if not np.isfinite(value):
            continue
        sums[key] = sums.get(key, 0.0) + float(value) * float(weight)
        counts[key] = counts.get(key, 0.0) + float(weight)


def train_epoch(
    model: CurrentCdfHeadOnly,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_actions = 0
    for batch in loader:
        batch = move_batch(batch, device)
        feature = batch["feature"]
        target = friction_to_cdf_target(batch["friction"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(feature)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        actions = int(target.shape[0] * target.shape[1])
        total_loss += float(loss.detach().item()) * actions
        total_actions += actions
    return {
        "loss": total_loss / max(total_actions, 1),
        "num_actions": float(total_actions),
    }


@torch.no_grad()
def evaluate(
    model: CurrentCdfHeadOnly,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    *,
    verify_base_reconstruction: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_base_loss = 0.0
    total_actions = 0
    mae_sum = 0.0
    base_mae_sum = 0.0
    candidate_count = 0
    selection_sums: Dict[str, float] = {}
    selection_counts: Dict[str, float] = {}
    base_selection_sums: Dict[str, float] = {}
    base_selection_counts: Dict[str, float] = {}
    reconstruction_max_abs = 0.0
    reconstruction_abs_sum = 0.0
    reconstruction_numel = 0

    for batch in loader:
        batch = move_batch(batch, device)
        target = friction_to_cdf_target(batch["friction"])
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(batch["feature"])
            loss = F.binary_cross_entropy_with_logits(logits, target)
            base_loss = F.binary_cross_entropy_with_logits(
                batch["base_cdf_logits"], target
            )

        if verify_base_reconstruction:
            difference = (
                logits.float() - batch["base_cdf_logits"].float()
            ).abs()
            current_error = float(difference.max().item())
            reconstruction_max_abs = max(reconstruction_max_abs, current_error)
            reconstruction_abs_sum += float(difference.sum().item())
            reconstruction_numel += int(difference.numel())
            # The cache itself has already passed a strict check from the actual
            # deployed Conv1d output to the final endpoint. Compact replay can
            # select a different TF32 convolution kernel and differ by ~1e-2.
            if current_error > 5e-2:
                raise RuntimeError(
                    "The compact current-head replay differs too much from the "
                    "cached deployed CDF output "
                    f"(max_abs_error={current_error:.3e}). Check the checkpoint "
                    "SHA and regenerate the cache."
                )

        actions = int(target.shape[0] * target.shape[1])
        total_loss += float(loss.item()) * actions
        total_base_loss += float(base_loss.item()) * actions
        total_actions += actions

        utility = torch.sigmoid(logits.float()).mean(dim=-1)
        base_utility = torch.sigmoid(batch["base_cdf_logits"].float()).mean(dim=-1)
        target_utility = target.float().mean(dim=-1)
        candidates = int(target_utility.numel())
        mae_sum += float((utility - target_utility).abs().sum().item())
        base_mae_sum += float((base_utility - target_utility).abs().sum().item())
        candidate_count += candidates

        current_selection = _selection_metrics(
            logits.float(), target.float(), batch["center_group"]
        )
        base_selection = _selection_metrics(
            batch["base_cdf_logits"].float(),
            target.float(),
            batch["center_group"],
        )
        center_weight = current_selection.pop("num_centers")
        base_center_weight = base_selection.pop("num_centers")
        _accumulate_weighted(
            selection_sums, selection_counts, current_selection, center_weight
        )
        _accumulate_weighted(
            base_selection_sums,
            base_selection_counts,
            base_selection,
            base_center_weight,
        )

    metrics: Dict[str, float] = {
        "loss": total_loss / max(total_actions, 1),
        "base_loss": total_base_loss / max(total_actions, 1),
        "utility_mae": mae_sum / max(candidate_count, 1),
        "base_utility_mae": base_mae_sum / max(candidate_count, 1),
        "num_actions": float(total_actions),
    }
    if verify_base_reconstruction:
        metrics["base_compact_replay_max_abs"] = float(reconstruction_max_abs)
        metrics["base_compact_replay_mean_abs"] = (
            reconstruction_abs_sum / max(reconstruction_numel, 1)
        )
    for key, value in selection_sums.items():
        metrics[key] = value / max(selection_counts.get(key, 0.0), 1.0)
    for key, value in base_selection_sums.items():
        metrics[f"base_{key}"] = value / max(
            base_selection_counts.get(key, 0.0), 1.0
        )
    return metrics


def _save_head_checkpoint(
    path: str,
    model: CurrentCdfHeadOnly,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    contract: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "probe_version": PROBE_VERSION,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": dict(train_metrics),
            "val_metrics": dict(val_metrics),
            "contract": dict(contract),
        },
        path,
    )


def _save_full_checkpoint(
    path: str,
    base_checkpoint: Mapping[str, Any],
    checkpoint_contract,
    model: CurrentCdfHeadOnly,
    probe_metadata: Mapping[str, Any],
) -> None:
    merged = merge_cdf_head_into_full_checkpoint(
        base_checkpoint,
        checkpoint_contract,
        model,
        probe_metadata=probe_metadata,
    )
    torch.save(merged, path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("Learning rates must be positive/non-negative")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(int(args.seed))

    base_checkpoint, checkpoint_contract = validate_current_stage1_cdf_checkpoint(
        args.base_checkpoint,
        expected_pose_depth_mode=args.expected_pose_depth_mode,
        expected_use_fuse_depth=bool(args.expected_use_fuse_depth),
    )
    metadata, inventory, failures = scan_cache(
        args.cache_dir,
        expected_checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
        strict=True,
        check_values=True,
    )
    if failures:
        raise RuntimeError(f"Strict cache scan unexpectedly retained failures: {failures[:3]}")
    if inventory.feature_dim != checkpoint_contract.feature_dim:
        raise RuntimeError("Cache/checkpoint CDF feature dimensions differ.")
    if inventory.num_angles != 12:
        raise RuntimeError(
            f"Current probe requires the 12-angle CVA lattice, got "
            f"A={inventory.num_angles}."
        )
    if inventory.num_depths != checkpoint_contract.num_depths:
        raise RuntimeError("Cache/checkpoint depth dimensions differ.")
    if inventory.num_thresholds != checkpoint_contract.num_thresholds:
        raise RuntimeError("Cache/checkpoint CDF threshold dimensions differ.")

    frame_counts = Counter(item.scene_id for item in metadata)
    if bool(args.require_all_scenes):
        missing_scenes = sorted(set(range(100)) - set(frame_counts))
        insufficient = {
            scene: count
            for scene, count in sorted(frame_counts.items())
            if count < int(args.min_frames_per_scene)
        }
        if missing_scenes or insufficient:
            raise RuntimeError(
                "Formal 10%-cache validation requires complete scene coverage. "
                f"missing_scenes={missing_scenes}, "
                f"insufficient_frames={insufficient}. Use --require_all_scenes 0 "
                "only for a smoke test."
            )

    train_dataset = ExactActionCdfCacheDataset(
        args.cache_dir,
        split="train",
        val_scene_start=int(args.val_scene_start),
        expected_checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
    )
    val_dataset = ExactActionCdfCacheDataset(
        args.cache_dir,
        split="val",
        val_scene_start=int(args.val_scene_start),
        expected_checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_exact_action_cdf,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_exact_action_cdf,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=worker_init_fn,
    )

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested_device}")
    device = torch.device(requested_device)
    amp_enabled = bool(args.amp) and device.type == "cuda"

    model = CurrentCdfHeadOnly(
        feature_dim=checkpoint_contract.feature_dim,
        num_depths=checkpoint_contract.num_depths,
        num_thresholds=checkpoint_contract.num_thresholds,
        increment_bias=inventory.cdf_increment_bias,
    )
    model.load_from_full_state(
        base_checkpoint["model_state_dict"], checkpoint_contract
    )
    model.to(device)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable_names != ["cdf_head.weight", "cdf_head.bias"]:
        raise RuntimeError(
            "Exact-action probe must optimize only the existing current CDF head; "
            f"trainable parameters are {trainable_names}."
        )
    scaler = make_grad_scaler(amp_enabled)
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

    probe_contract = {
        "probe_version": PROBE_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "base_checkpoint": checkpoint_contract.to_dict(),
        "cache_inventory": inventory.to_dict(),
        "training_scope": "existing_decoder_cdf_head_weight_and_bias_only",
        "trainable_parameters": trainable_names,
        "objective": "ordinary_unbalanced_bce_on_exact_evaluator_cdf",
        "friction_thresholds": list(FRICTION_THRESHOLDS),
        "val_scene_start": int(args.val_scene_start),
        "train_frames": len(train_dataset),
        "val_frames": len(val_dataset),
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "min_learning_rate": float(args.min_learning_rate),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "batch_size_frames": int(args.batch_size),
        "seed": int(args.seed),
    }
    atomic_save_json(probe_contract, os.path.join(output_dir, "probe_contract.json"))

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    start = time.time()

    # Epoch 0 is the untouched Stage-1 head on the exact validation cache.
    # Run it in float32 so the cache/head reconstruction check is not polluted
    # by inference autocast rounding.
    base_val = evaluate(
        model,
        val_loader,
        device,
        False,
        verify_base_reconstruction=True,
    )
    best_val_loss = float(base_val["loss"])
    best_epoch = 0
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": 0, "phase": "base", "val": base_val}) + "\n")
    print(f"[BASE] {json.dumps(base_val, sort_keys=True)}", flush=True)

    base_save_metadata = {
        **probe_contract,
        "selected_epoch": 0,
        "train_metrics": {},
        "val_metrics": dict(base_val),
    }
    _save_head_checkpoint(
        os.path.join(output_dir, "head_checkpoint_best.tar"),
        model,
        optimizer,
        0,
        {},
        base_val,
        probe_contract,
    )
    _save_full_checkpoint(
        os.path.join(output_dir, "checkpoint_best_exact_action.tar"),
        base_checkpoint,
        checkpoint_contract,
        model,
        base_save_metadata,
    )
    atomic_save_json(
        {
            "best_epoch": 0,
            "best_val_loss": best_val_loss,
            "val_metrics": base_val,
        },
        os.path.join(output_dir, "best.json"),
    )

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, amp_enabled, scaler
        )
        val_metrics = evaluate(model, val_loader, device, amp_enabled)
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_sec": time.time() - epoch_start,
            "elapsed_sec": time.time() - start,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[EPOCH {epoch:03d}] {json.dumps(record, sort_keys=True)}", flush=True)

        head_last = os.path.join(output_dir, "head_checkpoint_last.tar")
        full_last = os.path.join(output_dir, "checkpoint_last_exact_action.tar")
        metadata_for_save = {
            **probe_contract,
            "selected_epoch": epoch,
            "train_metrics": dict(train_metrics),
            "val_metrics": dict(val_metrics),
        }
        _save_head_checkpoint(
            head_last,
            model,
            optimizer,
            epoch,
            train_metrics,
            val_metrics,
            probe_contract,
        )
        _save_full_checkpoint(
            full_last,
            base_checkpoint,
            checkpoint_contract,
            model,
            metadata_for_save,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            _save_head_checkpoint(
                os.path.join(output_dir, "head_checkpoint_best.tar"),
                model,
                optimizer,
                epoch,
                train_metrics,
                val_metrics,
                probe_contract,
            )
            _save_full_checkpoint(
                os.path.join(output_dir, "checkpoint_best_exact_action.tar"),
                base_checkpoint,
                checkpoint_contract,
                model,
                metadata_for_save,
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
            _save_full_checkpoint(
                os.path.join(
                    output_dir, f"checkpoint_epoch_{epoch:03d}_exact_action.tar"
                ),
                base_checkpoint,
                checkpoint_contract,
                model,
                metadata_for_save,
            )

    print(
        f"[DONE] best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}; "
        f"full checkpoint={os.path.join(output_dir, 'checkpoint_best_exact_action.tar')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
