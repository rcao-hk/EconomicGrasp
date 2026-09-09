"""Tensor-only ray grasp operations; no dataset, CUDA extension or CLI imports."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

DEFAULT_OFFSETS_MM = (-40., -20., -10., 0., 10., 20., 40.)


def parse_offsets(text: str | Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in (text.split(",") if isinstance(text, str) else text))
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Ray offsets must be a nonempty finite list in millimetres.")
    if len(set(values)) != len(values) or values.count(0.) != 1:
        raise ValueError("Ray offsets must be unique and include exactly one zero.")
    if max(map(abs, values)) > 250:
        raise ValueError("Ray offsets exceed 250 mm; check the units.")
    return values


def backproject(uv: torch.Tensor, z: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    """uv [B,M,2], z [B,M]; depth means camera z, not Euclidean ray length."""
    if uv.shape != (*z.shape, 2) or camera.shape != (z.shape[0], 3, 3):
        raise ValueError("Invalid uv/z/intrinsics shapes.")
    fx, fy = camera[:, 0, 0, None], camera[:, 1, 1, None]
    if not bool(((fx > 0) & (fy > 0)).all()):
        raise ValueError("Camera focal lengths must be positive.")
    return torch.stack(((uv[..., 0] - camera[:, 0, 2, None]) / fx * z,
                        (uv[..., 1] - camera[:, 1, 2, None]) / fy * z, z), -1)


def ray_hypotheses(base_xyz: torch.Tensor, pixel_idx: torch.Tensor,
                   camera: torch.Tensor, image_width: int,
                   offsets_m: torch.Tensor, min_depth: float, max_depth: float):
    """Return [B,M,K,3] centers, [B,M,K] validity, [B,M,K,5] descriptors.

    Out-of-range depths are made numerically safe but always masked from losses
    and decode. The zero-offset center is kept bitwise equal to the base center.
    """
    if base_xyz.shape != (*pixel_idx.shape, 3) or offsets_m.ndim != 1:
        raise ValueError("Expected base [B,M,3], pixels [B,M], offsets [K].")
    if not bool(torch.isfinite(base_xyz).all()):
        raise FloatingPointError("Nonfinite base query center.")
    z0 = base_xyz[..., 2].detach()
    uv = torch.stack((pixel_idx % image_width, pixel_idx // image_width), -1).to(z0)
    ray = backproject(uv, torch.ones_like(z0), camera.to(z0))
    depth = z0[..., None] + offsets_m.to(z0)
    valid = torch.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    safe_z = depth.clamp(min_depth, max_depth)
    xyz = ray[..., None, :] * safe_z[..., None]
    zero_idx = torch.where(offsets_m == 0)[0]
    if zero_idx.numel() != 1:
        raise ValueError("Exactly one zero-offset hypothesis is required.")
    xyz = xyz.clone()
    xyz[:, :, int(zero_idx.item()), :] = base_xyz.detach()
    scale = max(float(offsets_m.abs().max()), .01)
    zspan = max(max_depth - min_depth, 1e-6)
    rx = ray[..., 0, None].expand_as(depth)
    ry = ray[..., 1, None].expand_as(depth)
    desc = torch.stack((rx, ry, 2 * (z0[..., None].expand_as(depth) - min_depth) / zspan - 1,
                        2 * (safe_z - min_depth) / zspan - 1,
                        offsets_m.to(z0).view(1, 1, -1).expand_as(depth) / scale), -1)
    return xyz, valid, desc


def candidate_utilities(parts: Sequence[Mapping], supported: bool = True):
    """[B,M,K,A,D] joint utility; validity here uses prediction range only."""
    quality = torch.stack([p["grasp_cdf_pred_angle_depth"].float().sigmoid().mean(1)
                           for p in parts], dim=2)
    in_range = torch.stack([p["ray_in_range"] for p in parts], dim=2).bool()
    if supported:
        support = torch.stack([p["ray_support_logits"].float().sigmoid() for p in parts], 2)
        quality = quality * support[..., None, None]
    return quality.masked_fill(~in_range[..., None, None], -1.)


def select_hypotheses(parts: Sequence[Mapping], offsets_m: torch.Tensor,
                      supported: bool = True, selection: str = "best"):
    """Choose one k per image ray. Ties prefer smaller |offset|, then list order."""
    utility = candidate_utilities(parts, supported)
    best_op = utility.flatten(-2).max(-1).values  # B,M,K
    if best_op.shape[-1] != offsets_m.numel():
        raise ValueError("Hypothesis count/offset count mismatch.")
    if selection == "zero":
        k = torch.full_like(best_op[..., 0], int((offsets_m == 0).nonzero()[0]), dtype=torch.long)
    elif selection == "best":
        order = torch.argsort(offsets_m.abs(), stable=True).to(best_op.device)
        k = order[best_op.index_select(-1, order).argmax(-1)]
    else:
        raise ValueError("selection must be best or zero.")
    score = best_op.gather(-1, k[..., None]).squeeze(-1)
    return k, score, utility


def loss_sums(parts: Sequence[Mapping]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Global-normalizable CDF/width/support sufficient statistics.

    CDF and width retain main's masks and scales. Off-support hypotheses are NOT
    assigned zero force-closure labels. The separate support BCE learns the
    finite 5-mm label-domain membership, not physical grasp validity.
    """
    out = {}
    for p in parts:
        logits = p["grasp_cdf_pred_angle_depth"].float().permute(0, 2, 3, 4, 1)
        bins = p["batch_grasp_cdf_bins_angle_depth"].long()
        valid = p["batch_grasp_cdf_valid_mask"].bool() & p["ray_in_range"][..., None, None]
        if bins.shape != logits.shape[:-1] or valid.shape != bins.shape:
            raise ValueError("CDF labels must match [B,M,A,D].")
        T = logits.shape[-1]
        if p["batch_grasp_cdf_thresholds"].numel() != T:
            raise ValueError("CDF threshold count mismatch.")
        if bool(((bins < 0) | (bins > T)).any()):
            raise ValueError("Invalid compact CDF bins.")
        t = torch.arange(T, device=logits.device)
        target = ((bins[..., None] > 0) & (t >= bins[..., None] - 1)).to(logits)
        cdf = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        cdf_mask = valid[..., None].expand_as(cdf)
        wp = p["grasp_width_pred_angle_depth"].float().permute(0, 2, 3, 1)
        wt = p["batch_grasp_width_angle_depth"].float() * 10.
        wm = p["batch_grasp_width_valid_mask_angle_depth"].bool() & valid
        if wp.shape != wt.shape or wm.shape != wp.shape:
            raise ValueError("Width shape mismatch.")
        width = F.smooth_l1_loss(wp, wt, reduction="none")
        support = F.binary_cross_entropy_with_logits(p["ray_support_logits"].float(),
                                                      p["ray_support_target"].float(), reduction="none")
        sm = p["ray_support_known"].bool() & p["ray_in_range"]
        for name, value, mask in (("cdf", cdf, cdf_mask), ("width", width, wm),
                                  ("support", support, sm)):
            pair = (value.masked_select(mask).sum(), mask.sum().to(value.dtype))
            if name not in out:
                out[name] = pair
            else:
                out[name] = (out[name][0] + pair[0], out[name][1] + pair[1])
    if not out:
        raise ValueError("No ray hypotheses.")
    return out


@torch.no_grad()
def metric_sums(parts: Sequence[Mapping], offsets_m: torch.Tensor):
    """Small paired coverage/selection metrics; never an analytic AP oracle."""
    k, score, utility = select_hypotheses(parts, offsets_m, supported=True)
    sup = torch.stack([p["ray_support_target"] for p in parts], 2).bool()
    known = torch.stack([p["ray_support_known"] & p["ray_in_range"] for p in parts], 2)
    sup &= known
    zero = int((offsets_m == 0).nonzero()[0])
    selected_sup = sup.gather(-1, k[..., None]).squeeze(-1)
    any_known = known.any(-1)
    vals = {
        "base_label_point_support": (sup[..., zero].float(), known[..., zero]),
        "any_label_point_support": (sup.any(-1).float(), any_known),
        "selected_label_point_support": (selected_sup.float(), any_known),
        "selected_nonzero": ((k != zero).float(), score >= 0),
    }
    # Coverage of actual CDF labels includes missing-view masks, unlike support.
    cv = torch.stack([p["batch_grasp_cdf_valid_mask"].bool().any(-1).any(-1)
                      & p["ray_in_range"] for p in parts], 2)
    vals["any_cdf_label_support"] = (cv.any(-1).float(), any_known)
    vals["selected_cdf_label_support"] = (cv.gather(-1, k[..., None]).squeeze(-1).float(), any_known)
    for i in range(len(parts)):
        vals[f"selected_k{i}"] = ((k == i).float(), score >= 0)
    return {name: (v.masked_select(m).sum(), m.sum().to(v)) for name, (v, m) in vals.items()}
