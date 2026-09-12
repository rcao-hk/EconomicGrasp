"""Pure tensor operations for P5 geometry-error-aware grasp repair.

P5 separates annotation availability from grasp utility. Training may corrupt
the predicted dense geometry to expose deployment-like metric errors, but repair
supervision is used only when the exact predicted view/angle/depth operation is
known positive at a nearby annotated grasp center. Missing nearby labels are
never converted into negative grasp labels.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def structured_depth_corruption(
    depth: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
    probability: float = 0.8,
    scene_bias_sigma_m: float = 0.012,
    scale_sigma: float = 0.025,
    region_sigma_m: float = 0.015,
    region_grid: int = 7,
):
    """Apply spatially correlated metric-depth errors for training only."""
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError("P5 depth must be [B,1,H,W].")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("P5 corruption probability must lie in [0,1].")
    if min(scene_bias_sigma_m, scale_sigma, region_sigma_m) < 0:
        raise ValueError("P5 corruption sigmas must be non-negative.")
    if int(region_grid) < 2:
        raise ValueError("P5 region_grid must be >= 2.")
    B, _, H, W = depth.shape
    dtype, device = depth.dtype, depth.device
    active = (torch.rand(B, 1, 1, 1, device=device) < float(probability)).to(dtype)
    scene_bias = torch.randn(B, 1, 1, 1, device=device, dtype=dtype) * float(scene_bias_sigma_m)
    scale = torch.randn(B, 1, 1, 1, device=device, dtype=dtype) * float(scale_sigma)
    low = torch.randn(B, 1, int(region_grid), int(region_grid), device=device, dtype=dtype)
    low = low - low.mean(dim=(-2, -1), keepdim=True)
    low = low / low.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    region = F.interpolate(low, size=(H, W), mode="bicubic", align_corners=False)
    region = region * float(region_sigma_m)
    delta = active * (scene_bias + depth * scale + region)
    out = (depth + delta).clamp(float(min_depth), float(max_depth))
    actual = out - depth
    diag = {
        "active_ratio": active.mean(),
        "abs_mean_m": actual.abs().mean(),
        "signed_mean_m": actual.mean(),
        "abs_max_m": actual.abs().amax(),
    }
    return out.contiguous(), diag


def gather_depth(depth: torch.Tensor, token_idx: torch.Tensor) -> torch.Tensor:
    if depth.dim() != 4 or depth.shape[1] != 1 or token_idx.dim() != 2:
        raise ValueError("Expected depth [B,1,H,W] and token_idx [B,Q].")
    B, _, H, W = depth.shape
    if token_idx.shape[0] != B:
        raise ValueError("P5 depth/query batch mismatch.")
    return depth[:, 0].reshape(B, H * W).gather(1, token_idx.long())


def backproject_token_depth(token_idx: torch.Tensor, z: torch.Tensor,
                            camera_k: torch.Tensor, image_width: int) -> torch.Tensor:
    if token_idx.shape != z.shape or camera_k.shape != (z.shape[0], 3, 3):
        raise ValueError("P5 backprojection shape mismatch.")
    u = (token_idx % int(image_width)).to(z)
    v = (token_idx // int(image_width)).to(z)
    K = camera_k.to(z)
    fx, fy = K[:, 0, 0, None], K[:, 1, 1, None]
    cx, cy = K[:, 0, 2, None], K[:, 1, 2, None]
    if not bool(((fx > 0) & (fy > 0)).all()):
        raise ValueError("P5 camera focal lengths must be positive.")
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack((x, y, z), dim=-1)


def cdf_mean_utility(cdf_logits: torch.Tensor) -> torch.Tensor:
    if cdf_logits.dim() != 5:
        raise ValueError("P5 CDF logits must be [B,T,Q,A,D].")
    return torch.sigmoid(cdf_logits.float()).mean(dim=1)


def select_native_operation(cdf_logits: torch.Tensor, width_pred: torch.Tensor):
    utility = cdf_mean_utility(cdf_logits)
    B, Q, A, D = utility.shape
    flat = utility.reshape(B, Q, A * D)
    op = flat.argmax(dim=-1)
    score = flat.gather(-1, op[..., None]).squeeze(-1)
    angle = torch.div(op, D, rounding_mode="floor")
    depth = torch.remainder(op, D)
    if width_pred.shape != (B, D, Q, A):
        raise ValueError(f"P5 width_pred shape {tuple(width_pred.shape)} != {(B,D,Q,A)}")
    width_qad = width_pred.float().permute(0, 2, 3, 1).contiguous()
    b = torch.arange(B, device=op.device)[:, None].expand(B, Q)
    q = torch.arange(Q, device=op.device)[None, :].expand(B, Q)
    width_raw = width_qad[b, q, angle, depth]
    width_m = torch.clamp(1.2 * width_raw / 10.0, min=0.0, max=0.1)
    insertion_m = (depth.float() + 1.0) * 0.01
    return angle, depth, score, width_m, insertion_m, utility


def compact_cdf_utility(bins: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    if int(num_thresholds) <= 0:
        raise ValueError("P5 num_thresholds must be positive.")
    bins = bins.long()
    if bool(((bins < 0) | (bins > int(num_thresholds))).any()):
        raise ValueError("P5 compact CDF bins must lie in [0,T].")
    return torch.where(
        bins > 0,
        (float(num_thresholds) - bins.float() + 1.0) / float(num_thresholds),
        torch.zeros_like(bins, dtype=torch.float32),
    )


def build_repair_targets(
    *,
    proposal_center: torch.Tensor,
    rotation: torch.Tensor,
    nearest_label_center: torch.Tensor,
    cdf_bins: torch.Tensor,
    angle_idx: torch.Tensor,
    depth_idx: torch.Tensor,
    num_thresholds: int,
    target_radius_m: float,
):
    """Build coherent center-repair targets without treating unknown as failure."""
    if proposal_center.shape != nearest_label_center.shape or proposal_center.dim() != 3:
        raise ValueError("P5 repair centers must be [B,Q,3].")
    B, Q, _ = proposal_center.shape
    if rotation.shape != (B, Q, 3, 3) or cdf_bins.dim() != 4:
        raise ValueError("P5 repair rotation/CDF shape mismatch.")
    _, Q2, A, D = cdf_bins.shape
    if Q2 != Q or angle_idx.shape != (B, Q) or depth_idx.shape != (B, Q):
        raise ValueError("P5 repair operation shape mismatch.")
    b = torch.arange(B, device=proposal_center.device)[:, None].expand(B, Q)
    q = torch.arange(Q, device=proposal_center.device)[None, :].expand(B, Q)
    a = angle_idx.long().clamp(0, A - 1)
    d = depth_idx.long().clamp(0, D - 1)
    selected_bin = cdf_bins[b, q, a, d]
    selected_utility = compact_cdf_utility(selected_bin, int(num_thresholds)).to(proposal_center)
    delta_cam = nearest_label_center - proposal_center
    distance = delta_cam.norm(dim=-1)
    positive_same_action = selected_bin > 0
    known = torch.isfinite(distance) & (distance <= float(target_radius_m)) & positive_same_action
    delta_local = torch.matmul(rotation.transpose(-1, -2), delta_cam.unsqueeze(-1)).squeeze(-1)
    return {
        "target_delta_local": delta_local.detach(),
        "target_known": known.detach(),
        "target_distance_m": distance.detach(),
        "target_utility": selected_utility.detach(),
        "same_action_positive": positive_same_action.detach(),
    }


def build_gripper_keypoints_local(width_m: torch.Tensor, insertion_m: torch.Tensor) -> torch.Tensor:
    """Eleven sparse keypoints covering contacts, fingers, palm and approach space."""
    if width_m.shape != insertion_m.shape or width_m.dim() != 2:
        raise ValueError("P5 width/insertion must be [B,Q].")
    w = width_m.clamp(0.0, 0.1)
    d = insertion_m.clamp(0.005, 0.06)
    z0 = torch.zeros_like(w)
    half = 0.5 * w
    outer = (half + 0.01).clamp(max=0.06)
    back = torch.full_like(w, -0.03)
    forward = d + 0.02
    h = torch.full_like(w, 0.01)
    def p(x, y, z):
        return torch.stack((x, y, z), dim=-1)
    pts = [
        p(z0, z0, z0),
        p(d, half, z0), p(d, -half, z0),
        p(d, outer, z0), p(d, -outer, z0),
        p(z0, half, z0), p(z0, -half, z0),
        p(back, z0, z0), p(forward, z0, z0),
        p(d, z0, h), p(d, z0, -h),
    ]
    return torch.stack(pts, dim=2)


def transform_keypoints(center: torch.Tensor, rotation: torch.Tensor,
                        local_points: torch.Tensor) -> torch.Tensor:
    if center.dim() != 3 or rotation.shape != (*center.shape[:2], 3, 3):
        raise ValueError("P5 transform center/rotation shape mismatch.")
    if local_points.shape[:2] != center.shape[:2] or local_points.shape[-1] != 3:
        raise ValueError("P5 local keypoint shape mismatch.")
    world = torch.matmul(rotation.unsqueeze(2), local_points.unsqueeze(-1)).squeeze(-1)
    return world + center.unsqueeze(2)


def project_keypoints(points: torch.Tensor, camera_k: torch.Tensor, H: int, W: int):
    if points.dim() != 4 or points.shape[-1] != 3:
        raise ValueError("P5 points must be [B,Q,L,3].")
    B = points.shape[0]
    if camera_k.shape != (B, 3, 3):
        raise ValueError("P5 camera intrinsics shape mismatch.")
    K = camera_k.to(points)
    x, y, z = points.unbind(dim=-1)
    safe_z = z.clamp_min(1e-5)
    u = K[:, 0, 0, None, None] * x / safe_z + K[:, 0, 2, None, None]
    v = K[:, 1, 1, None, None] * y / safe_z + K[:, 1, 2, None, None]
    visible = (z > 1e-5) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    return torch.stack((u, v), dim=-1), visible


def sample_map_at_uv(feature_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    if feature_map.dim() != 4 or uv.dim() != 4 or uv.shape[-1] != 2:
        raise ValueError("P5 feature/uv shape mismatch.")
    B, _, H, W = feature_map.shape
    if uv.shape[0] != B:
        raise ValueError("P5 feature/uv batch mismatch.")
    gx = 2.0 * uv[..., 0] / max(W - 1, 1) - 1.0
    gy = 2.0 * uv[..., 1] / max(H - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.permute(0, 2, 3, 1).contiguous()


def repair_loss_sums(
    pred_delta_local: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    *,
    beta_m: float = 0.005,
    unknown_identity_weight: float = 0.02,
):
    if pred_delta_local.shape != targets["target_delta_local"].shape:
        raise ValueError("P5 predicted/target delta shape mismatch.")
    known = targets["target_known"].bool()
    target = targets["target_delta_local"].to(pred_delta_local)
    loss_map = F.smooth_l1_loss(pred_delta_local, target, beta=float(beta_m), reduction="none").sum(-1)
    repair_sum = loss_map.masked_select(known).sum()
    repair_count = known.sum().to(pred_delta_local.dtype)
    unknown = ~known
    identity_map = pred_delta_local.norm(dim=-1)
    identity_sum = identity_map.masked_select(unknown).sum() * float(unknown_identity_weight)
    identity_count = unknown.sum().to(pred_delta_local.dtype)
    return {
        "repair": (repair_sum, repair_count),
        "unknown_identity": (identity_sum, identity_count),
    }


@torch.no_grad()
def repair_metric_sums(pred_delta_local: torch.Tensor, rotation: torch.Tensor,
                       proposal_center: torch.Tensor, targets: Mapping[str, torch.Tensor]):
    known = targets["target_known"].bool()
    target_local = targets["target_delta_local"].to(pred_delta_local)
    target_center = proposal_center + torch.matmul(rotation, target_local.unsqueeze(-1)).squeeze(-1)
    repaired = proposal_center + torch.matmul(rotation, pred_delta_local.unsqueeze(-1)).squeeze(-1)
    native_dist = (proposal_center - target_center).norm(dim=-1)
    repaired_dist = (repaired - target_center).norm(dim=-1)
    improvement = native_dist - repaired_dist
    def pair(v, m):
        m = m.bool()
        v = v.to(pred_delta_local)
        return v.masked_select(m).sum(), m.sum().to(v.dtype)
    allmask = torch.ones_like(known)
    return {
        "p5_target_known_ratio": pair(known.float(), allmask),
        "p5_native_target_dist_m": pair(native_dist, known),
        "p5_repaired_target_dist_m": pair(repaired_dist, known),
        "p5_repair_improvement_m": pair(improvement, known),
        "p5_native_within5mm": pair((native_dist < 0.005).float(), known),
        "p5_repaired_within5mm": pair((repaired_dist < 0.005).float(), known),
        "p5_pred_delta_abs_m": pair(pred_delta_local.norm(dim=-1), allmask),
        "p5_target_delta_abs_m": pair(target_local.norm(dim=-1), known),
    }
