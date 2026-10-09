"""CVA-CDF objective and controlled U0-U4 confidence/uncertainty losses.

U0 point; U1 expected/fixed; U2 learned interval score;
U3 learned Laplace NLL with detached depth; U4 the same Laplace
confidence-weighted depth regression *without* a depth stop-gradient.
The grasp->geometry gradient boundary remains fixed for every variant.
"""
from __future__ import annotations
import math
import torch
from .geometry import shape_losses, ray_consistency_loss, interval_score

LOG10 = math.log(10.0)


def laplace_depth_objective(mean, halfwidth, target, valid, *,
                            detach_mean=True, valid_normalize=True,
                            reference_halfwidth=.02):
    """Laplace NLL with h90 = log(10)*b; all optical-depth units are metres.

    A reference b0 multiplier makes depth gradients at initialization the
    same magnitude as metric L1; the log term prevents unlimited uncertainty.
    """
    mu = mean.detach() if detach_mean else mean
    if target.ndim == 3:
        target = target[:, None]
    if mu.shape != target.shape or halfwidth.shape != target.shape:
        raise ValueError("Laplace geometry shapes must agree")
    if valid.shape != target.shape:
        raise ValueError("Laplace valid mask shape mismatch")
    b0 = float(reference_halfwidth) / LOG10
    b = (halfwidth.float() / LOG10).clamp_min(1e-6)
    safe = torch.nan_to_num(target.float(), nan=0., posinf=0., neginf=0.)
    nll = b0 * ((mu.float() - safe).abs() / b + (b / b0).log())
    m = valid.to(nll.dtype)
    if valid_normalize:
        return (nll * m).sum() / m.sum().clamp_min(1.)
    # Legacy depth loss averages over all pixels, not only valid pixels.
    return (nll * m).mean()


def objective(ep, model_cfg, loss_cfg):
    from models.loss_economicgrasp_depth_kview_transformer import get_loss
    from utils.arguments import cfgs

    total, ep = get_loss(ep, use_cdf=True)
    old_depth = ep["A: DepthReg Loss"]
    zero = total * 0
    extras = {"global_shape": zero, "local_shape": zero,
              "reprojection": zero, "interval": zero}
    g = ep.get("mr_geometry")
    gt, valid = None, None

    if model_cfg.use_moge or (model_cfg.use_rayrope and model_cfg.uncertainty == "learned"):
        if "gt_depth_m" not in ep:
            raise KeyError("Geometry supervision requires gt_depth_m")
        gt = ep["gt_depth_m"].float()
        if gt.ndim == 3:
            gt = gt[:, None]
        if gt.ndim != 4 or gt.shape[1] != 1:
            raise ValueError("gt_depth_m must be [B,1,H,W]")
        valid = (torch.isfinite(gt) & (gt > model_cfg.min_depth)
                 & (gt < model_cfg.max_depth))
        safe = torch.where(valid, gt, torch.zeros_like(gt))
        if g is None:
            raise KeyError("Missing mr_geometry")
        if model_cfg.use_moge:
            extras.update(shape_losses(g["points"], safe, ep["K"],
                                       gt.shape[-2:], loss_cfg))
            extras["reprojection"] = ray_consistency_loss(g["canonical"], g["rays"])
        if model_cfg.use_rayrope and model_cfg.uncertainty == "learned":
            mode = model_cfg.uncertainty_loss
            if mode == "interval":
                extras["interval"] = interval_score(
                    g["depth"], g["sigma"], safe, valid, model_cfg.interval_coverage)
            elif mode == "laplace_decoupled":
                extras["interval"] = laplace_depth_objective(
                    g["depth"], g["sigma"], safe, valid,
                    detach_mean=True, valid_normalize=True,
                    reference_halfwidth=model_cfg.fixed_halfwidth)
            elif mode == "laplace_joint":
                joint = laplace_depth_objective(
                    g["depth"], g["sigma"], safe, valid,
                    detach_mean=False, valid_normalize=False,
                    reference_halfwidth=model_cfg.fixed_halfwidth)
                new_depth = float(cfgs.depth_prob_loss_weight) * joint
                total = total - old_depth + new_depth
                ep["B: DepthReg Loss"] = joint
                ep["A: DepthReg Loss"] = new_depth
            else:
                raise ValueError(f"Unexpected uncertainty_loss {mode}")

    task = total - ep["A: DepthReg Loss"]
    weighted = (loss_cfg.global_shape * extras["global_shape"]
                + loss_cfg.local_shape * extras["local_shape"]
                + loss_cfg.reprojection * extras["reprojection"]
                + loss_cfg.interval * extras["interval"])
    total = total + weighted

    stats = {k: v.detach() for k, v in extras.items()}
    stats.update(loss=total.detach(), task=task.detach(),
                 metric_depth=ep["A: DepthReg Loss"].detach())
    depth = ep.get("depth_map_pred")
    if depth is not None:
        dd = depth.detach().float()
        stats["depth_mean_m"] = dd.mean()
        stats["depth_spatial_std_m"] = dd.flatten(1).std(dim=1).mean()
        stats["depth_out_of_range_fraction"] = (
            (~torch.isfinite(dd)) | (dd <= model_cfg.min_depth)
            | (dd >= model_cfg.max_depth)).float().mean()
    if gt is not None and g is not None and "sigma" in g:
        residual = (g["depth"].detach().float() - torch.nan_to_num(gt)).abs()
        h = g["sigma"].detach().float()
        n = valid.sum().clamp_min(1)
        stats["interval_coverage90"] = ((residual <= h) & valid).sum().float() / n
        stats["interval_mean_halfwidth_m"] = (h * valid).sum() / n
        stats["interval_upper_saturation"] = (
            ((h >= model_cfg.sigma_max - 1e-4) & valid).sum().float() / n)
    width = ep.get("grasp_width_pred_angle_depth")
    if width is not None:
        w = width.detach()
        stats["decoded_width_mean_m"] = (1.2*w.float()/10).clamp(0,.1).mean()
        stats["decoded_width_zero_fraction"] = (w <= 0).float().mean()
    if model_cfg.use_moge:
        stats["metric_anchor_mean_m"] = g["anchor"].detach().mean()
        stats["shape_gauge_mean"] = g["gauge"].detach().mean()
    return total, task, stats
