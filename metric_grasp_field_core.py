"""DAV2 metric ray evidence and physical-action readout (pure PyTorch).

Depth CDF is a distribution over camera Z in metres, NOT the grasp CDF over
friction thresholds. All numeric geometry entering the grasp reader is detached.
The field models FIRST-SURFACE evidence; 'behind' never means occupied.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class MetricFieldConfig:
    min_depth: float = 0.2
    max_depth: float = 1.0
    bins: int = 160
    hidden: int = 64
    field_stride: int = 4
    surface_epsilon: float = 0.005
    prior_sigma: float = 0.03
    fixed_sigma: float = 0.02
    residual_scale: float = 0.04
    action_chunk: int = 256
    checkpoint_chunks: bool = True
    evidence_mode: str = "learned"

    def __post_init__(self):
        if not 0 < self.min_depth < self.max_depth:
            raise ValueError("Invalid camera-depth range")
        if min(self.bins, self.hidden, self.field_stride, self.action_chunk) < 1:
            raise ValueError("Counts must be positive")
        if self.bins < 2 or self.hidden % 8:
            raise ValueError("Need >=2 depth bins and hidden divisible by 8")
        if min(self.surface_epsilon, self.prior_sigma, self.fixed_sigma,
               self.residual_scale) <= 0:
            raise ValueError("Metric tolerances must be positive")
        if self.evidence_mode not in ("hard", "fixed", "learned"):
            raise ValueError("evidence_mode must be hard/fixed/learned")


def resize(x, hw):
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=True)


def sample_map(x, uv, image_hw):
    """[B,C,Hf,Wf], [B,...,2] image pixels -> [B,...,C]."""
    h, w = image_hw
    grid = torch.stack((2 * uv[..., 0] / (w - 1) - 1,
                        2 * uv[..., 1] / (h - 1) - 1), -1)
    b = x.shape[0]
    sampled = F.grid_sample(x, grid.reshape(b, -1, 1, 2).to(x.dtype),
                            align_corners=True, padding_mode="zeros")
    return sampled[:, :, :, 0].transpose(1, 2).reshape(*uv.shape[:-1], x.shape[1])


def project_points(points, K, image_hw):
    """Camera-Z convention; no Euclidean-range/optical-depth interchange."""
    h, w = image_hw
    shape = (K.shape[0],) + (1,) * (points.ndim - 2)
    z = points[..., 2]
    zz = z.clamp_min(1e-6)
    u = K[:, 0, 0].reshape(shape) * points[..., 0] / zz + K[:, 0, 2].reshape(shape)
    v = K[:, 1, 1].reshape(shape) * points[..., 1] / zz + K[:, 1, 2].reshape(shape)
    ok = (torch.isfinite(points).all(-1) & (z > 0) &
          (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1))
    uv = torch.nan_to_num(torch.stack((u, v), -1), nan=-1e6, posinf=1e6, neginf=-1e6)
    return uv, ok


def cdf_at(prob, z, zmin, zmax):
    """Continuous CDF: uniform density inside each fixed metric bin.

    prob [...,L], z [...]. Outside the supported depth range the CDF is 0/1.
    """
    l = prob.shape[-1]
    t = ((z - zmin) * l / (zmax - zmin)).clamp(0, l)
    k = t.floor().long().clamp(max=l - 1)
    frac = t - k
    prefix = F.pad(prob.cumsum(-1), (1, 0))
    return (prefix.gather(-1, k.unsqueeze(-1)).squeeze(-1) +
            frac * prob.gather(-1, k.unsqueeze(-1)).squeeze(-1))


def evidence_at(prob, z, cfg):
    """Same learned mean for all three ablations; no grasp gradient to prob."""
    prob = prob.detach().float()
    prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-8)
    centres = torch.linspace(cfg.min_depth, cfg.max_depth, cfg.bins + 1,
                             device=prob.device, dtype=prob.dtype)
    centres = (centres[:-1] + centres[1:]) * 0.5
    mean = (prob * centres).sum(-1)
    eps = cfg.surface_epsilon
    if cfg.evidence_mode == "learned":
        lo = cdf_at(prob, z - eps, cfg.min_depth, cfg.max_depth)
        hi = cdf_at(prob, z + eps, cfg.min_depth, cfg.max_depth)
    elif cfg.evidence_mode == "fixed":
        # Analytic Gaussian avoids changing the mean by truncating at bin edges.
        den = cfg.fixed_sigma * math.sqrt(2.0)
        lo = 0.5 * (1 + torch.erf((z - eps - mean) / den))
        hi = 0.5 * (1 + torch.erf((z + eps - mean) / den))
    else:
        lo = (mean <= z - eps).float()
        hi = (mean <= z + eps).float()
    return torch.stack((torch.tanh((z - mean) / cfg.residual_scale),
                        1 - hi, (hi - lo).clamp_min(0), lo), -1)


class MetricRayEvidenceHead(nn.Module):
    """Geometry-supervised profile; task losses must only use detached output."""
    def __init__(self, relative_dim, metric_dim, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.relative_proj = nn.Conv2d(relative_dim, h, 1)
        self.metric_proj = nn.Conv2d(metric_dim, h, 1)
        self.net = nn.Sequential(nn.Conv2d(2*h + 5, h, 3, padding=1),
                                 nn.GroupNorm(8, h), nn.GELU(),
                                 nn.Conv2d(h, h, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(h, cfg.bins, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        edges = torch.linspace(cfg.min_depth, cfg.max_depth, cfg.bins + 1)
        self.register_buffer("centres", (edges[:-1] + edges[1:]) / 2)

    def forward(self, relative_feat, relative_inverse_depth, metric_feat, depth, image_hw, camera_K):
        cfg = self.cfg
        hw = tuple(max(2, n // cfg.field_stride) for n in image_hw)
        r = resize(relative_inverse_depth.detach(), hw)
        # Relative decoder is affine-ambiguous. Standardize only its relative cue.
        r = (r - r.mean((-2, -1), keepdim=True)) / r.std((-2, -1), keepdim=True).clamp_min(1e-4)
        mu = resize(depth, hw)
        # Crop-aware camera rays; metric depth remains optical-axis Z, not range.
        if camera_K.shape != (depth.shape[0], 3, 3):
            raise ValueError("Expected camera intrinsics [B,3,3]")
        k = camera_K.detach().to(mu)
        yy, xx = torch.meshgrid(
            torch.linspace(0, image_hw[0]-1, hw[0], device=mu.device, dtype=mu.dtype),
            torch.linspace(0, image_hw[1]-1, hw[1], device=mu.device, dtype=mu.dtype), indexing="ij")
        rx = (xx[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None].clamp_min(1e-6)
        ry = (yy[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None].clamp_min(1e-6)
        rays = F.normalize(torch.stack((rx, ry, torch.ones_like(rx)), 1), dim=1)
        x = torch.cat((self.relative_proj(resize(relative_feat.detach(), hw)),
                       self.metric_proj(resize(metric_feat, hw)),
                       r.clamp(-5, 5), mu / cfg.max_depth, rays), 1)
        prior = -0.5 * ((self.centres[None, :, None, None] - mu) / cfg.prior_sigma).square()
        logits = self.net(x) + prior
        return logits, logits.float().softmax(1)


class TaskFeatureAdapter(nn.Module):
    """Task features have no trainable path into the metric decoder/profile."""
    def __init__(self, proposal_dim, relative_dim, metric_dim, hidden):
        super().__init__()
        self.proposal_proj = nn.Conv2d(proposal_dim, hidden, 1)
        self.relative_proj = nn.Conv2d(relative_dim, hidden, 1)
        self.metric_proj = nn.Conv2d(metric_dim, hidden, 1)
        self.fuse = nn.Sequential(nn.Conv2d(3*hidden, hidden, 1),
                                  nn.GroupNorm(8, hidden), nn.GELU())

    def forward(self, proposal, relative, metric, hw):
        return self.fuse(torch.cat((self.proposal_proj(resize(proposal, hw)),
                                   self.relative_proj(resize(relative.detach(), hw)),
                                   self.metric_proj(resize(metric.detach(), hw))), 1))


def gripper_support(actions):
    """27 physical support points, 5 roles; GraspNet local=(point-t)@R.

    Closing/contact bands:12; finger bodies:4+4; palm:3; approach:4.
    actions [B,N,17]. No external translation embedding is added to the scorer.
    """
    w, h, d = actions[..., 1], actions[..., 2], actions[..., 3]
    zero = torch.zeros_like(w)
    points, roles = [], []
    for xf in (0.15, 0.5, 0.85):
        for yf in (-0.45, 0.45):
            for zf in (-0.3, 0.3):
                points.append(torch.stack((d - .06 + .06*xf, w*yf, h*zf), -1)); roles.append(0)
    for role, sign in ((1, -1), (2, 1)):
        for xf in (0.25, 0.75):
            for zf in (-0.3, 0.3):
                points.append(torch.stack((d - .06 + .06*xf, sign*(w/2 + .005), h*zf), -1)); roles.append(role)
    for yf in (-0.4, 0., 0.4):
        points.append(torch.stack((d - .065, (w + .02)*yf, zero), -1)); roles.append(3)
    for back in (.08, .095):
        for yf in (-.4, .4):
            points.append(torch.stack((d - back, (w + .02)*yf, zero), -1)); roles.append(4)
    local = torch.stack(points, -2)
    rot = actions[..., 4:13].reshape(*actions.shape[:-1], 3, 3)
    world = local @ rot.transpose(-1, -2) + actions[..., 13:16].unsqueeze(-2)
    return world, local, torch.tensor(roles, device=actions.device)


def monotone_grasp_logits(raw):
    """Monotonic over friction thresholds only, NOT over translation/depth."""
    return torch.cat((raw[..., :1], raw[..., :1] + F.softplus(raw[..., 1:]).cumsum(-1)), -1)


class GraspFieldReadout(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.role_embed = nn.Embedding(5, h)
        self.point = nn.Sequential(nn.Linear(h + 8, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(5*h + 3), nn.Linear(5*h + 3, h),
                                  nn.GELU(), nn.Linear(h, 6))
        with torch.no_grad():
            self.head[-1].bias[0] = -2.
            self.head[-1].bias[1:] = -3.

    def _chunk(self, feature, prob, actions, K, image_hw):
        xyz, local, roles = gripper_support(actions.float())
        uv, valid = project_points(xyz, K.float(), image_hw)
        # Out-of-range Z is UNKNOWN, not empty/occupied. It must not gain certainty.
        valid = valid & (xyz[..., 2] >= self.cfg.min_depth) & (xyz[..., 2] <= self.cfg.max_depth)
        visual = sample_map(feature.float(), uv, image_hw)
        p = sample_map(prob.detach().float(), uv, image_hw)
        evidence = evidence_at(p, xyz[..., 2], self.cfg)
        evidence = torch.cat((evidence, valid[..., None].float()), -1)
        tokens = self.point(torch.cat((visual, evidence, local / .1), -1))
        tokens = tokens + self.role_embed(roles)[None, None]
        tokens = tokens * valid[..., None]
        pooled = []
        for role in range(5):
            m = (roles == role)[None, None] & valid
            pooled.append((tokens * m[..., None]).sum(-2) / m.sum(-1, keepdim=True).clamp_min(1))
        size = actions[..., 1:4] / actions.new_tensor([.1, .02, .04])
        return monotone_grasp_logits(self.head(torch.cat((*pooled, size), -1)))

    def forward(self, feature, prob, actions, K, image_hw):
        if actions.ndim != 3 or actions.shape[-1] != 17:
            raise ValueError("Expected [B,N,17] actions")
        # One compulsory boundary for *all* task-side numeric geometry.
        actions, prob, K = actions.detach(), prob.detach(), K.detach()
        outputs = []
        for chunk in actions.split(self.cfg.action_chunk, dim=1):
            if self.training and self.cfg.checkpoint_chunks and torch.is_grad_enabled():
                def run(f, p, a, k):
                    return self._chunk(f, p, a, k, image_hw)
                y = checkpoint(run, feature, prob, chunk, K, use_reentrant=False)
            else:
                y = self._chunk(feature, prob, chunk, K, image_hw)
            outputs.append(y)
        return torch.cat(outputs, 1)


def geometry_loss(logits, depth, gt, cfg):
    """Online depth supervision; mask from finite GT only, no prediction mask."""
    if gt.ndim == 3:
        gt = gt[:, None]
    gt = gt.float()
    target = F.interpolate(gt, size=logits.shape[-2:], mode="nearest")[:, 0]
    valid = torch.isfinite(target) & (target >= cfg.min_depth) & (target <= cfg.max_depth)
    clean = torch.where(valid, target, torch.full_like(target, cfg.min_depth))
    index = ((clean - cfg.min_depth) / (cfg.max_depth - cfg.min_depth) * cfg.bins).long().clamp(0, cfg.bins - 1)
    ce_map = F.cross_entropy(logits.float(), index, reduction="none")
    edges = torch.linspace(cfg.min_depth, cfg.max_depth, cfg.bins + 1, device=logits.device)
    mu = (logits.float().softmax(1) * ((edges[:-1]+edges[1:])/2)[None, :, None, None]).sum(1)
    ce = ce_map[valid].mean() if valid.any() else logits.sum()*0
    mean_l1 = (mu-clean).abs()[valid].mean() if valid.any() else logits.sum()*0
    gt_full = F.interpolate(gt, size=depth.shape[-2:], mode="nearest")
    mask = torch.isfinite(gt_full) & (gt_full >= cfg.min_depth) & (gt_full <= cfg.max_depth)
    # Keep main's full-image normalization for the existing metric-depth loss.
    # Remove nonfinite GT before arithmetic (NaN * 0 is still NaN).
    clean_gt = torch.where(mask, gt_full, depth.detach().float())
    depth_l1 = ((depth.float() - clean_gt).abs() * mask).mean()
    return {"profile_ce": ce, "profile_mean_l1": mean_l1, "depth_l1": depth_l1}
