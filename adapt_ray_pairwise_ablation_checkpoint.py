#!/usr/bin/env python3
"""Expand a feature-ablation selector checkpoint to the full-input contract.

The held-out exact-action tester composes the historical full selector input

  [F0_sel, Fk_sel, Fk_sel-F0_sel, F0_mean, Fk_mean-F0_mean,
   score0, scorek, scorek-score0, normalized_offset].

Feature-ablation selectors were trained on subsets of those blocks.  This script
embeds the trained first-layer weights into the matching full-input columns and
sets all omitted columns to zero.  Feature normalization is expanded in the same
way.  Therefore the resulting checkpoint is functionally identical to the
original ablation selector while remaining directly consumable by
``test_ray_pairwise_selector.py``.

No network is retrained and no test information is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils.ray_pairwise_feature_ablation import FEATURE_MODES, feature_ablation_dim
from utils.ray_pairwise_selector import RayPairwiseSelector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_checkpoint", required=True)
    p.add_argument("--output_checkpoint", required=True)
    p.add_argument("--verify_atol", type=float, default=1e-6)
    p.add_argument("--verify_samples", type=int, default=64)
    return p.parse_args()


def feature_indices(group_dim: int, mode: str) -> torch.Tensor:
    c = int(group_dim)
    full_dim = 5 * c + 4
    if mode == "raw_offset":
        ids = list(range(5 * c, full_dim))
    elif mode == "selected_residual":
        ids = list(range(0, 3 * c))
    elif mode == "mean_residual":
        ids = list(range(3 * c, 5 * c))
    elif mode == "selected_mean":
        ids = list(range(0, 5 * c))
    elif mode == "full":
        ids = list(range(full_dim))
    else:
        raise ValueError(f"Unknown feature mode {mode!r}; choose from {FEATURE_MODES}")
    return torch.tensor(ids, dtype=torch.long)


def main():
    args = parse_args()
    src = Path(args.input_checkpoint)
    dst = Path(args.output_checkpoint)
    ckpt = torch.load(src, map_location="cpu")

    mode = str(ckpt.get("feature_mode", ""))
    if mode not in FEATURE_MODES:
        raise RuntimeError(
            f"Checkpoint {src} does not contain a valid feature_mode; got {mode!r}."
        )
    if "group_feature_dim" not in ckpt:
        raise KeyError("Ablation checkpoint missing group_feature_dim.")
    group_dim = int(ckpt["group_feature_dim"])
    subset_dim = int(ckpt["feature_dim"])
    expected_subset_dim = feature_ablation_dim(group_dim, mode)
    full_dim = 5 * group_dim + 4
    if subset_dim != expected_subset_dim:
        raise RuntimeError(
            f"Feature dimension mismatch for mode={mode}: checkpoint={subset_dim}, "
            f"expected={expected_subset_dim}."
        )

    indices = feature_indices(group_dim, mode)
    if int(indices.numel()) != subset_dim:
        raise RuntimeError("Feature-index mapping size mismatch.")

    state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v
             for k, v in ckpt["selector_state_dict"].items()}
    first_key = "net.0.weight"
    if first_key not in state:
        raise KeyError(f"Selector state is missing {first_key!r}.")
    old_weight = state[first_key]
    if old_weight.ndim != 2 or old_weight.shape[1] != subset_dim:
        raise RuntimeError(
            f"Unexpected first-layer shape {tuple(old_weight.shape)} for subset_dim={subset_dim}."
        )
    new_weight = old_weight.new_zeros((old_weight.shape[0], full_dim))
    new_weight[:, indices] = old_weight
    state[first_key] = new_weight

    old_mean = ckpt["feature_mean"].detach().cpu().float().reshape(-1)
    old_std = ckpt["feature_std"].detach().cpu().float().reshape(-1)
    if old_mean.numel() != subset_dim or old_std.numel() != subset_dim:
        raise RuntimeError("Feature normalization dimension does not match checkpoint feature_dim.")
    full_mean = torch.zeros(full_dim, dtype=old_mean.dtype)
    full_std = torch.ones(full_dim, dtype=old_std.dtype)
    full_mean[indices] = old_mean
    full_std[indices] = old_std

    # Fail-fast mathematical equivalence check before writing anything.
    hidden = int(ckpt["hidden_dim"])
    dropout = float(ckpt["dropout"])
    original = RayPairwiseSelector(subset_dim, hidden, dropout)
    original.load_state_dict(ckpt["selector_state_dict"])
    original.eval()
    expanded = RayPairwiseSelector(full_dim, hidden, dropout)
    expanded.load_state_dict(state)
    expanded.eval()

    gen = torch.Generator().manual_seed(20260917)
    x_full = torch.randn(max(1, int(args.verify_samples)), full_dim, generator=gen)
    x_subset = x_full.index_select(1, indices)
    with torch.no_grad():
        y_old = original((x_subset - old_mean) / old_std.clamp_min(1e-5))
        y_new = expanded((x_full - full_mean) / full_std.clamp_min(1e-5))
    max_abs = float((y_old - y_new).abs().max().item())
    if max_abs > float(args.verify_atol):
        raise RuntimeError(
            f"Expanded-checkpoint equivalence failed: max_abs={max_abs:.3e} "
            f"> atol={args.verify_atol:.3e}."
        )

    out = dict(ckpt)
    out["selector_state_dict"] = state
    out["feature_mean"] = full_mean
    out["feature_std"] = full_std
    out["feature_dim"] = full_dim
    out["heldout_source_feature_mode"] = mode
    out["heldout_source_feature_dim"] = subset_dim
    out["heldout_full_feature_dim"] = full_dim
    out["heldout_feature_indices"] = indices
    out["heldout_adapter_equivalence_max_abs"] = max_abs
    out["heldout_adapter_note"] = (
        "Exact zero-column embedding into full pairwise feature contract; no retraining."
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(
        f"[ABLATE-ADAPT] mode={mode} subset_dim={subset_dim} full_dim={full_dim} "
        f"max_abs={max_abs:.3e} -> {dst}",
        flush=True,
    )


if __name__ == "__main__":
    main()
