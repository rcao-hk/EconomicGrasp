"""Utilities for frozen multi-center exact-action ray diagnostics."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


def parse_offsets_mm(text):
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one ray offset is required.")
    if not any(abs(x) < 1e-9 for x in values):
        raise ValueError("Ray offsets must include 0 mm as native control.")
    if len({round(x, 9) for x in values}) != len(values):
        raise ValueError("Ray offsets contain duplicates.")
    return tuple(values)


def build_ray_center_hypotheses(native_xyz, token_idx, camera_K, image_hw, offsets_mm, min_depth, max_depth):
    """Construct same-pixel camera-z hypotheses around native centers."""
    if native_xyz.dim() != 3 or native_xyz.shape[-1] != 3:
        raise ValueError("native_xyz must be [B,M,3].")
    if token_idx.shape != native_xyz.shape[:2]:
        raise ValueError("token_idx does not match native center shape.")
    H, W = int(image_hw[0]), int(image_hw[1])
    if bool(((token_idx < 0) | (token_idx >= H * W)).any()):
        raise ValueError("token_idx contains out-of-range indices.")

    offsets = torch.as_tensor(list(offsets_mm), device=native_xyz.device, dtype=native_xyz.dtype).view(-1, 1, 1) / 1000.0
    z = native_xyz[..., 2].unsqueeze(0) + offsets
    valid = torch.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))

    idx = token_idx.long()
    u = (idx % W).to(native_xyz.dtype).unsqueeze(0)
    v = (idx // W).to(native_xyz.dtype).unsqueeze(0)
    fx = camera_K[:, 0, 0].to(native_xyz).view(1, -1, 1)
    fy = camera_K[:, 1, 1].to(native_xyz).view(1, -1, 1)
    cx = camera_K[:, 0, 2].to(native_xyz).view(1, -1, 1)
    cy = camera_K[:, 1, 2].to(native_xyz).view(1, -1, 1)
    x = (u - cx) / fx.clamp_min(1e-6) * z
    y = (v - cy) / fy.clamp_min(1e-6) * z
    centers = torch.stack((x, y, z), dim=-1)
    return centers.contiguous(), valid.contiguous()


def friction_utility(friction):
    f = np.asarray(friction, dtype=np.float32)
    mus = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
    return (((f[..., None] > 0.0) & (f[..., None] <= mus)).mean(axis=-1)).astype(np.float32)


def select_raw_score(raw_score, valid):
    s = np.asarray(raw_score, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if s.shape != v.shape or s.ndim != 2:
        raise ValueError("raw_score and valid must share [K,N].")
    if np.any(~v.any(axis=0)):
        raise ValueError("Every query needs at least one valid hypothesis.")
    return np.argmax(np.where(v, s, -np.inf), axis=0).astype(np.int64)


def select_exact_oracle(utility, raw_score, valid):
    u = np.asarray(utility, dtype=np.float64)
    s = np.asarray(raw_score, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if not (u.shape == s.shape == v.shape) or u.ndim != 2:
        raise ValueError("utility, raw_score and valid must share [K,N].")
    if np.any(~v.any(axis=0)):
        raise ValueError("Every query needs at least one valid hypothesis.")
    masked_u = np.where(v, u, -np.inf)
    best_u = masked_u.max(axis=0, keepdims=True)
    tie = v & np.isclose(masked_u, best_u, atol=1e-8, rtol=0.0)
    return np.argmax(np.where(tie, s, -np.inf), axis=0).astype(np.int64)


def gather_kn(matrix, k_index):
    arr = np.asarray(matrix)
    idx = np.asarray(k_index, dtype=np.int64)
    if arr.ndim < 2 or idx.shape != (arr.shape[1],):
        raise ValueError("Expected [K,N,...] and [N] indices.")
    return arr[idx, np.arange(arr.shape[1])]
