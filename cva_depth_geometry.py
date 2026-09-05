"""GT-only sampling and depth losses for the CVA depth-collapse experiments.

This module depends only on PyTorch. It never evaluates a grasp, reads a mesh,
or constructs supervision from a predicted depth, predicted score, or mask.
All distances and losses use metres. The functions are also used by the CPU
self-tests and the diagnostic script.
"""

from dataclasses import asdict, dataclass
import math
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairConfig:
    anchors_per_image: int = 128
    pairs_per_anchor: int = 8
    radius_min_m: float = 0.005
    radius_max_m: float = 0.03
    visibility_tolerance_m: float = 0.005
    control_depth_tolerance_m: float = 0.01
    min_depth: float = 0.2
    max_depth: float = 1.0
    huber_beta_m: float = 0.005

    def __post_init__(self):
        if self.anchors_per_image < 1 or self.pairs_per_anchor < 1:
            raise ValueError("Pair counts must be positive.")
        if not 0 < self.radius_min_m <= self.radius_max_m:
            raise ValueError("Require 0 < radius_min_m <= radius_max_m.")
        if not 0 < self.min_depth < self.max_depth:
            raise ValueError("Invalid depth interval.")
        if min(self.visibility_tolerance_m, self.control_depth_tolerance_m,
               self.huber_beta_m) <= 0:
            raise ValueError("Depth tolerances and Huber beta must be positive.")

    def to_dict(self):
        return asdict(self)


def depth_b1hw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3:
        depth = depth[:, None]
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth [B,1,H,W], got {tuple(depth.shape)}")
    return depth


def valid_depth(gt: torch.Tensor, cfg: PairConfig) -> torch.Tensor:
    gt = depth_b1hw(gt)
    return torch.isfinite(gt) & (gt >= cfg.min_depth) & (gt <= cfg.max_depth)


def metric_depth_loss(pred, gt, cfg: PairConfig, normalization="full_image"):
    """Match main's masked full-image L1 by default; mask comes only from GT."""
    pred, gt = depth_b1hw(pred), depth_b1hw(gt).to(pred)
    if pred.shape != gt.shape:
        raise ValueError("Predicted and GT depth must have exactly the same grid.")
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError("Non-finite predicted depth.")
    mask = valid_depth(gt, cfg)
    error = torch.where(mask, pred - torch.nan_to_num(gt), 0.0).abs()
    if normalization == "full_image":
        return error.mean()
    if normalization == "valid_pixels":
        return error.sum() / mask.sum().clamp_min(1)
    raise ValueError(f"Unknown depth normalization: {normalization}")


def contrast_depth(pred, beta: float, mean_mask=None):
    """Do not clamp the intervention: clipping would confound contrast and mean."""
    pred = depth_b1hw(pred)
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError("Contrast scan requires finite baseline depth.")
    if beta < 0 or not math.isfinite(beta):
        raise ValueError("Contrast beta must be finite and nonnegative.")
    mask = torch.ones_like(pred, dtype=torch.bool) if mean_mask is None else mean_mask.bool()
    if mask.shape != pred.shape or not bool(mask.flatten(1).any(1).all()):
        raise ValueError("Each image needs a nonempty fixed mean mask.")
    if beta == 1:
        # Preserve exact values at discrete query/matching boundaries.
        return pred
    mean = torch.where(mask, pred, 0.0).sum((1, 2, 3), keepdim=True)
    mean = mean / mask.sum((1, 2, 3), keepdim=True)
    return mean + float(beta) * (pred - mean)


def _foreground(batch, b: int, gt, cfg):
    mask = valid_depth(gt[b:b + 1], cfg)[0, 0]
    label = batch.get("objectness_label_tok")
    if label is None:
        raise KeyError("GT foreground sampling needs objectness_label_tok.")
    return mask & (label[b].to(gt.device).reshape(mask.shape) == 1)


@torch.no_grad()
def visible_anchor_pool(batch: Dict, b: int, cfg: PairConfig) -> Dict[str, torch.Tensor]:
    """Project cached object-frame anchors and reject occluded/off-image points.

    The returned XYZs are the actual cached anchors transformed into the camera
    frame, not backprojections of predicted depth. At most one point is kept per
    pixel, preferring the point closest to the rendered visible GT surface.
    """
    gt = depth_b1hw(batch["gt_depth_m"]).float()
    _, _, height, width = gt.shape
    device = gt.device
    k = batch["K"][b].to(device=device, dtype=torch.float32)
    foreground = _foreground(batch, b, gt, cfg).flatten()
    depths = gt[b, 0].flatten()
    point_lists = batch["grasp_points_list"][b]
    poses = batch["object_poses_list"][b]
    if len(point_lists) != len(poses):
        raise ValueError("Object poses and cached anchor lists are not aligned.")
    xyzs, pixels, owners, errors = [], [], [], []
    for object_i, (points, pose) in enumerate(zip(point_lists, poses)):
        points = torch.as_tensor(points, device=device, dtype=torch.float32)
        pose = torch.as_tensor(pose, device=device, dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] != 3 or pose.shape not in ((3, 4), (4, 4)):
            raise ValueError("Expected anchors [P,3] and object-to-camera pose [3,4]/[4,4].")
        xyz = points @ pose[:3, :3].T + pose[:3, 3]
        projected = xyz @ k.T
        uv = projected[:, :2] / projected[:, 2:3].clamp_min(1e-8)
        uv = torch.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1).round().long()
        valid = torch.isfinite(xyz).all(1) & (xyz[:, 2] >= cfg.min_depth) & (xyz[:, 2] <= cfg.max_depth)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        index = uv[:, 1].clamp(0, height - 1) * width + uv[:, 0].clamp(0, width - 1)
        distance = (depths[index] - xyz[:, 2]).abs()
        valid &= foreground[index] & (distance <= cfg.visibility_tolerance_m)
        xyzs.append(xyz[valid])
        pixels.append(index[valid])
        errors.append(distance[valid])
        owners.append(torch.full_like(index[valid], object_i))
    if not xyzs:
        return {"xyz": gt.new_empty((0, 3)), "pixel": torch.empty(0, device=device, dtype=torch.long),
                "object": torch.empty(0, device=device, dtype=torch.long)}
    xyz, pixel, owner, error = map(torch.cat, (xyzs, pixels, owners, errors))
    order = torch.argsort(error, stable=True)
    order = order[torch.argsort(pixel[order], stable=True)]
    keep = torch.ones(order.numel(), device=device, dtype=torch.bool)
    if order.numel() > 1:
        keep[1:] = pixel[order][1:] != pixel[order][:-1]
    take = order[keep]
    return {"xyz": xyz[take], "pixel": pixel[take], "object": owner[take]}


def _cpu_randint(n, shape, generator, device):
    return torch.randint(n, shape, generator=generator, device="cpu").to(device)


@torch.no_grad()
def matched_depth_pairs(batch: Dict, cfg: PairConfig, seed: int) -> Dict:
    """Return anchor and foreground controls with a common acceptance mask.

    Controls use identical integer image offsets and GT center depth within the
    configured tolerance. Thus pair counts and pixel-distance distributions are
    exactly matched; metric scale is approximately matched, and logged. Sampling
    depends only on GT and the seed. Object-balanced anchor draws avoid letting
    dense objects dominate. No accepted pairs means a connected zero loss.
    """
    gt = depth_b1hw(batch["gt_depth_m"]).float()
    batch_size, _, height, width = gt.shape
    device = gt.device
    n = cfg.anchors_per_image * cfg.pairs_per_anchor
    a_i = torch.zeros((batch_size, n), dtype=torch.long, device=device)
    a_j, c_i, c_j = a_i.clone(), a_i.clone(), a_i.clone()
    accepted = torch.zeros((batch_size, n), dtype=torch.bool, device=device)
    pool_counts = []
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for b in range(batch_size):
        pool = visible_anchor_pool(batch, b, cfg)
        pool_counts.append(int(pool["pixel"].numel()))
        if not pool["pixel"].numel():
            continue
        foreground = _foreground(batch, b, gt, cfg).flatten()
        foreground_ids = foreground.nonzero().flatten()
        z = gt[b, 0].flatten()
        objects = pool["object"].unique(sorted=True)
        picks = torch.empty(cfg.anchors_per_image, dtype=torch.long, device=device)
        object_draw = torch.randint(len(objects), (cfg.anchors_per_image,), generator=generator, device="cpu")
        # Group draws by object: avoid one GPU nonzero/synchronization per
        # sampled anchor while retaining uniform object and within-object draws.
        for object_index, owner in enumerate(objects):
            slots = (object_draw == object_index).nonzero().flatten()
            if not len(slots):
                continue
            rows = (pool["object"] == owner).nonzero().flatten()
            picks[slots.to(device)] = rows[_cpu_randint(len(rows), (len(slots),), generator, device)]
        centers = pool["pixel"][picks].repeat_interleave(cfg.pairs_per_anchor)
        rand = torch.rand((n, 2), generator=generator).to(device)
        radius = cfg.radius_min_m + rand[:, 0] * (cfg.radius_max_m - cfg.radius_min_m)
        angle = rand[:, 1] * (2 * torch.pi)
        k = batch["K"][b].to(device)
        du = (radius * angle.cos() * k[0, 0] / z[centers]).round().long()
        dv = (radius * angle.sin() * k[1, 1] / z[centers]).round().long()

        def endpoints(start):
            u, v = start % width + du, start // width + dv
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            end = v.clamp(0, height - 1) * width + u.clamp(0, width - 1)
            return end, inside & foreground[end] & ((du != 0) | (dv != 0))

        end, valid_anchor = endpoints(centers)
        controls = torch.zeros_like(centers)
        control_end = torch.zeros_like(centers)
        valid_control = torch.zeros_like(valid_anchor)
        # Vectorized, bounded rejection sampling; reject both experimental arms
        # when no depth-matched valid foreground control can be found.
        for _ in range(16):
            candidates = foreground_ids[_cpu_randint(len(foreground_ids), (n,), generator, device)]
            candidates_end, valid = endpoints(candidates)
            valid &= (z[candidates] - z[centers]).abs() <= cfg.control_depth_tolerance_m
            take = valid & ~valid_control
            controls[take], control_end[take] = candidates[take], candidates_end[take]
            valid_control |= valid
        a_i[b], a_j[b], c_i[b], c_j[b] = centers, end, controls, control_end
        accepted[b] = valid_anchor & valid_control
    return {"anchor_i": a_i, "anchor_j": a_j, "foreground_i": c_i,
            "foreground_j": c_j, "valid": accepted, "anchor_pool_counts": pool_counts}


def relative_depth_loss(pred, gt, pairs, variant, cfg: PairConfig):
    if variant not in ("anchor", "foreground"):
        raise ValueError("Relative-depth variant must be anchor or foreground.")
    pred = depth_b1hw(pred).flatten(1)
    gt = depth_b1hw(gt).to(pred).flatten(1)
    left, right = pairs[f"{variant}_i"], pairs[f"{variant}_j"]
    mask = pairs["valid"]
    delta_pred = pred.gather(1, left) - pred.gather(1, right)
    delta_gt = gt.gather(1, left) - gt.gather(1, right)
    safe_error = torch.where(mask, delta_pred - torch.nan_to_num(delta_gt), 0.0)
    loss_map = F.smooth_l1_loss(safe_error, torch.zeros_like(safe_error),
                              beta=cfg.huber_beta_m, reduction="none")
    # Equal weight per image. Images with no pairs contribute zero, not NaN.
    per_image = loss_map.sum(1) / mask.sum(1).clamp_min(1)
    return per_image.mean()


@torch.no_grad()
def depth_metrics(pred, gt, cfg: PairConfig, foreground=None) -> Dict[str, float]:
    pred, gt = depth_b1hw(pred), depth_b1hw(gt).to(pred)
    valid = valid_depth(gt, cfg)
    if foreground is not None:
        valid &= foreground.reshape_as(valid).to(valid.device) == 1
    count = int(valid.sum())
    result = {"valid_pixels": count, "valid_fraction": float(valid.float().mean()),
              "pred_nonfinite_fraction": float((~torch.isfinite(pred)).float().mean()),
              "pred_out_of_range_fraction": float(((pred < cfg.min_depth) | (pred > cfg.max_depth)).float().mean())}
    if not count:
        return {**result, "mae_m": None, "bias_m": None,
                "pred_std_m": None, "gt_std_m": None, "std_ratio": None}
    error = pred[valid] - gt[valid]
    # Image-wise spatial std: across-image mean differences must not hide collapse.
    pred_std, gt_std = [], []
    for p, g, mask in zip(pred, gt, valid):
        if bool(mask.any()):
            pred_std.append(p[mask].std(unbiased=False))
            gt_std.append(g[mask].std(unbiased=False))
    ps, gs = torch.stack(pred_std).mean(), torch.stack(gt_std).mean()
    return {**result, "mae_m": float(error.abs().mean()), "bias_m": float(error.mean()),
            "pred_std_m": float(ps), "gt_std_m": float(gs),
            "std_ratio": float(ps / gs.clamp_min(1e-8))}


@torch.no_grad()
def pair_metrics(pred, gt, pairs, cfg: PairConfig, camera_k=None):
    pred_map = depth_b1hw(pred)
    width = pred_map.shape[-1]
    pred, gt = pred_map.flatten(1), depth_b1hw(gt).to(pred).flatten(1)
    mask = pairs["valid"]
    result = {"pair_count": int(mask.sum()), "pair_capacity": mask.numel(),
              "pair_images": int(mask.any(1).sum()),
              "visible_anchor_count": sum(pairs["anchor_pool_counts"])}
    for variant in ("anchor", "foreground"):
        left, right = pairs[f"{variant}_i"], pairs[f"{variant}_j"]
        dp = pred.gather(1, left) - pred.gather(1, right)
        dg = gt.gather(1, left) - gt.gather(1, right)
        result[f"{variant}_relative_mae_m"] = float((dp - dg).abs()[mask].mean()) if bool(mask.any()) else None
        if camera_k is not None and bool(mask.any()):
            du = (left % width - right % width).float()
            dv = (left // width - right // width).float()
            k = camera_k.to(pred)
            nominal_radius = ((du / k[:, 0, 0, None]) ** 2 + (dv / k[:, 1, 1, None]) ** 2).sqrt() * gt.gather(1, left)
            result[f"{variant}_mean_lateral_radius_m"] = float(nominal_radius[mask].mean())
    return result
