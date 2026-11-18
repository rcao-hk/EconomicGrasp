"""Calibrated support-plane geometry for the RGB height-only DPT path.

No Pose/Gravity/Plane FiLM; the known plane enters ONLY the analytic decoder.
Optical-axis depth z (metres), not Euclidean range. Image pixel centres are integers.
"""
from __future__ import annotations

import numpy as np
import torch


def plane_from_camera_to_table(T):
    """Return [n_x,n_y,n_z,b] from T_table_from_camera (metres).

    table z=0 is an externally calibrated support plane, NEVER inferred from depth.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("T_table_from_cam must be a finite 4x4 transform.")
    if not np.allclose(T[3], [0., 0., 0., 1.], atol=1e-4):
        raise ValueError("Invalid homogeneous last row.")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=5e-3) or not np.isclose(
        np.linalg.det(R), 1., atol=5e-3
    ):
        raise ValueError("Expected proper rigid camera-to-table rotation.")
    n, b = R[2], float(T[2, 3])
    if not 0.05 < b < 3.0:
        raise ValueError(
            f"Calibrated support-plane camera height b={b:.6f}m outside "
            "(0.05,3.0)m. Check table origin, extrinsics, and mm/m units."
        )
    return np.asarray([n[0], n[1], n[2], b], dtype=np.float32)


def height_from_depth(depth, K, plane):
    """Inverse relation for controlled unit tests; depth [B,1,H,W]."""
    ray_dot, _ = ray_plane_alignment(K, plane, depth.shape[-2:])
    return depth * ray_dot + plane[:, 3].view(-1, 1, 1, 1)


def ray_plane_alignment(K, plane, image_hw):
    """Return n dot r for r=K^-1[u,v,1], r_z=1, and valid downward-facing rays."""
    if K.ndim != 3 or K.shape[1:] != (3, 3):
        raise ValueError("Expected crop-adjusted K [B,3,3].")
    if plane.ndim != 2 or plane.shape != (K.shape[0], 4):
        raise ValueError("Expected calibrated camera-plane parameters [B,4].")
    if not torch.isfinite(K).all() or not torch.isfinite(plane).all():
        raise ValueError("Non-finite intrinsics or plane.")
    if (K[:, 0, 0] <= 0).any() or (K[:, 1, 1] <= 0).any():
        raise ValueError("Non-positive focal length.")
    if not torch.allclose(
        K[:, 2, :], K.new_tensor([0., 0., 1.]).expand(K.shape[0], 3), atol=1e-5
    ):
        raise ValueError("K must be pinhole with final row [0,0,1].")
    normal_norm = torch.linalg.vector_norm(plane[:, :3], dim=1)
    if not torch.allclose(normal_norm, torch.ones_like(normal_norm), atol=5e-3):
        raise ValueError("Plane normals must be unit length.")
    if (plane[:, 3] <= .05).any() or (plane[:, 3] >= 3.).any():
        raise ValueError("Invalid camera-to-plane metre offset.")

    height, width = map(int, image_hw)
    if height <= 1 or width <= 1:
        raise ValueError("Expected positive image resolution.")
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=K.dtype, device=K.device),
        torch.arange(width, dtype=K.dtype, device=K.device), indexing="ij"
    )
    uv1 = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(-1, 3)
    ray = torch.matmul(uv1.unsqueeze(0), torch.linalg.inv(K).transpose(1, 2))
    ray = ray / ray[..., 2:3]
    dot = (ray * plane[:, None, :3]).sum(-1).reshape(-1, 1, height, width)
    return dot, ray.reshape(-1, height, width, 3)


def height_to_depth(
    raw_height_logits, K, plane, *, height_min_m=-.05, height_max_m=.35,
    min_downward_dot=.1,
):
    """(z, predicted_height, ray_eligible), all [B,1,H,W].

    No learned geometric conditioning. Invalid near-parallel / upward rays
    produce z=0 and an explicit false mask, not an invented plausible depth.
    The returned valid mask must also gate pixel selection and depth supervision.
    Never clip a valid but inaccurate predicted z to the GT range: preserve its
    gradient so the supervised depth loss can correct it.
    """
    if raw_height_logits.ndim != 4 or raw_height_logits.shape[1] != 1:
        raise ValueError("raw_height_logits must be [B,1,H,W].")
    if not (-3.0 < height_min_m < height_max_m < 3.0):
        raise ValueError("Height interval must be increasing and in metres.")
    if not 0 < min_downward_dot < 1:
        raise ValueError("min_downward_dot must be in (0,1).")
    dot, _ = ray_plane_alignment(K, plane, raw_height_logits.shape[-2:])
    eligible = dot < -min_downward_dot
    h = height_min_m + (height_max_m-height_min_m) * raw_height_logits.sigmoid()
    denom = torch.where(eligible, dot, -torch.ones_like(dot))
    z = (h-plane[:, 3].view(-1, 1, 1, 1))/denom
    z = torch.where(eligible, z, torch.zeros_like(z))
    if not torch.isfinite(z).all():
        raise FloatingPointError("Analytic height-to-depth returned non-finite z.")
    return z, h, eligible
