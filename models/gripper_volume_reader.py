"""Pure-PyTorch, insertion-conditioned gripper-volume evidence reader.

No dataset, argparse, CUDA extension, evaluator, or teacher imports. Coordinates
follow GraspNet: local x=approach, y=opening, z=height. The palm lies behind
x=d-finger_length. Width is a FIXED support envelope, not a ground-truth width
or a claim that the quality target describes an exactly evaluated new action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

VARIANTS = ("baseline", "slot", "volume_fixed", "volume", "volume_rel", "volume_fixed_rel")
ROLE_NAMES = ("left_contact", "right_contact", "closing", "finger_body", "palm", "approach")


@dataclass(frozen=True)
class GVARConfig:
    variant: str = "volume"
    reader_dim: int = 64
    reader_heads: int = 4
    reader_dropout: float = 0.05
    action_chunk: int = 512
    activation_checkpoint: bool = True
    envelope_width_m: float = 0.06
    height_m: float = 0.02
    finger_length_m: float = 0.06
    finger_thickness_m: float = 0.01
    approach_m: float = 0.05
    fixed_insertion_m: float = 0.025
    geometry_scale_m: float = 0.06

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown GVAR variant {self.variant!r}; expected {VARIANTS}")
        if self.reader_dim < 8 or self.reader_heads < 1 or self.reader_dim % self.reader_heads:
            raise ValueError("reader_dim must be >=8 and divisible by reader_heads")
        if self.action_chunk < 1 or not 0 <= self.reader_dropout < 1:
            raise ValueError("Invalid reader chunk/dropout")
        for key in ("envelope_width_m", "height_m", "finger_length_m", "finger_thickness_m", "approach_m", "fixed_insertion_m", "geometry_scale_m"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_probes(depths: torch.Tensor, cfg: GVARConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """depths [N] -> probes [N,36,3], roles [36]; metres, not ray-z offsets."""
    if depths.ndim != 1 or not torch.is_floating_point(depths):
        raise ValueError("depths must be a floating [N] tensor")
    w, h, length, thick = cfg.envelope_width_m, cfg.height_m, cfg.finger_length_m, cfg.finger_thickness_m
    pts = []
    # Two contact-side strips: three positions along the finger, two heights.
    for side in (-1., 1.):
        pts.extend([[-length * f, side * w / 2, z * h / 4]
                    for f in (0.75, 0.40, 0.05) for z in (-1., 1.)])
    pts.extend([[-length * f, side * w / 4, side * h / 4]
                for f in (0.75, 0.40, 0.05) for side in (-1., 1.)])
    pts.extend([[-length * f, side * (w + thick) / 2, 0.]
                for f in (0.85, 0.45, 0.05) for side in (-1., 1.)])
    pts.extend([[-length - thick / 2, y, z * h / 4]
                for y in (-(w + thick) / 2, 0., (w + thick) / 2) for z in (-1., 1.)])
    pts.extend([[-length - thick - cfg.approach_m * f, side * (w + thick) / 2, 0.]
                for f in (0.15, 0.50, 0.85) for side in (-1., 1.)])
    template = depths.new_tensor(pts)
    offset = torch.stack((depths, torch.zeros_like(depths), torch.zeros_like(depths)), -1)
    probes = template.unsqueeze(0) + offset.unsqueeze(1)
    roles = torch.arange(6, device=depths.device).repeat_interleave(6)
    return probes, roles


def project_probes(centers, rotations, probes, K, image_hw):
    """All coordinates in camera metres. Invalid/behind-camera probes stay masked."""
    H, W = (int(v) for v in image_hw)
    if min(H, W) < 2:
        raise ValueError("image dimensions must be >=2")
    xyz = torch.einsum("nij,npj->npi", rotations, probes) + centers[:, None, :]
    pixels = torch.einsum("ij,npj->npi", K, xyz)
    uv = pixels[..., :2] / pixels[..., 2:3].clamp_min(1e-6)
    valid = (torch.isfinite(xyz).all(-1) & torch.isfinite(uv).all(-1) & (xyz[..., 2] > 1e-4)
             & (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1)
             & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1))
    norm = uv.new_tensor([W - 1, H - 1])
    grid = torch.nan_to_num(uv / norm * 2 - 1, nan=2., posinf=2., neginf=-2.).clamp(-2, 2)
    return xyz, uv, grid, valid


def sample_map(feature_map: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """[1,C,Hf,Wf], [N,P,2] (normalised original image coords) -> [N,P,C]."""
    if feature_map.ndim != 4 or feature_map.shape[0] != 1:
        raise ValueError("sample_map expects one image [1,C,H,W]")
    result = F.grid_sample(feature_map, grid[None].to(feature_map.dtype), mode="bilinear",
                           padding_mode="zeros", align_corners=True)
    return result[0].permute(1, 2, 0).contiguous()


def action_relative_geometry(depth_map, grid, uv, xyz, centers, rotations, K, valid, scale):
    """Return six SOFT evidence values: local surface xyz, ray dz, known, valid.

    Unknown or occluded space is NOT classified as free/occupied. Geometry is
    detached, but grid_sample is not globally detached (coordinate gradients to
    a separately supplied physical action can still exist).
    """
    depth = depth_map.detach()
    finite = torch.isfinite(depth) & (depth > 0)
    clean = torch.where(finite, depth, torch.zeros_like(depth))
    sampled = sample_map(clean, grid)[..., 0]
    coverage = sample_map(finite.to(depth.dtype), grid)[..., 0]
    known = valid & (coverage > 0.999) & (sampled > 0)
    hom = torch.cat((uv, torch.ones_like(uv[..., :1])), -1)
    rays = torch.einsum("ij,npj->npi", torch.linalg.inv(K.float()).to(hom.dtype), hom)
    rays = rays / rays[..., 2:3].clamp_min(1e-6)
    surface = rays * sampled[..., None]
    local = torch.einsum("nji,npj->npi", rotations, surface - centers[:, None, :])
    local = torch.nan_to_num(local / scale, nan=0., posinf=0., neginf=0.).clamp(-5, 5)
    dz = torch.nan_to_num((sampled - xyz[..., 2]) / scale).clamp(-5, 5)
    local = torch.where(known[..., None], local, torch.zeros_like(local))
    dz = torch.where(known, dz, torch.zeros_like(dz))
    return torch.cat((local, dz[..., None], known[..., None].to(local.dtype), valid[..., None].to(local.dtype)), -1)


class ActionDepthAdapter(nn.Module):
    """Same d embedding and residual MLP in all non-baseline variants."""
    def __init__(self, hidden_dim, num_depth, dropout):
        super().__init__()
        self.depth_embedding = nn.Embedding(num_depth, hidden_dim)
        nn.init.normal_(self.depth_embedding.weight, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))

    def make_queries(self, angle_features):
        return angle_features.unsqueeze(-2) + self.depth_embedding.weight

    def finish(self, queries):
        return queries + self.ffn(self.norm(queries))


class GripperVolumeReader(nn.Module):
    """Residual evidence update for [B,Q,A,D,H] action queries.

    volume_fixed/volume/volume_rel/volume_fixed_rel have IDENTICAL parameters. Geometry inputs in
    non-rel variants are zeros, retaining a capacity-matched comparison. The
    original metric grouping remains upstream: this is not a pure RGB replacement.
    """
    def __init__(self, feat_dim: int, hidden_dim: int, cfg: GVARConfig):
        super().__init__()
        if cfg.variant not in ("volume_fixed", "volume", "volume_rel", "volume_fixed_rel"):
            raise ValueError("Reader requires a volume variant")
        self.cfg = cfg
        C = cfg.reader_dim
        self.feature_projection = nn.Conv2d(feat_dim, C, 1)
        self.query_projection = nn.Linear(hidden_dim, C)
        self.role_embedding = nn.Embedding(6, C)
        self.metadata_projection = nn.Linear(5, C)  # local xyz, actual d, in-image
        self.geometry_projection = nn.Linear(6, C, bias=False)
        self.attn = nn.MultiheadAttention(C, cfg.reader_heads, dropout=cfg.reader_dropout, batch_first=True)
        self.norm = nn.LayerNorm(C)
        self.out_projection = nn.Linear(C, hidden_dim)
        self.last_debug = {}

    def _read_chunk(self, queries, fmap, depth, centers, rotations, K, depths, image_hw):
        cfg = self.cfg
        probe_depths = (torch.full_like(depths, cfg.fixed_insertion_m)
                        if cfg.variant in ("volume_fixed", "volume_fixed_rel") else depths)
        probes, roles = canonical_probes(probe_depths, cfg)
        xyz, uv, grid, valid = project_probes(centers, rotations, probes, K, image_hw)
        patch = sample_map(fmap, grid)
        meta = torch.cat((probes / cfg.geometry_scale_m,
                          depths[:, None, None].expand(-1, 36, 1) / cfg.geometry_scale_m,
                          valid[..., None].to(probes.dtype)), -1).to(patch.dtype)
        if cfg.variant in ("volume_rel", "volume_fixed_rel"):
            geom = action_relative_geometry(depth, grid, uv, xyz, centers, rotations, K, valid, cfg.geometry_scale_m)
        else:
            geom = patch.new_zeros((*patch.shape[:2], 6))
        tokens = patch + self.role_embedding(roles)[None] + self.metadata_projection(meta)
        tokens = tokens + self.geometry_projection(geom.to(tokens.dtype))
        # Zero invalid data; allow one zero key only in all-invalid rows to avoid NaNs.
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        padding = ~valid
        all_invalid = padding.all(-1)
        padding = padding.clone()
        padding[:, 0] &= ~all_invalid
        query = self.query_projection(queries)[:, None]
        attended, weights = self.attn(query, tokens, tokens, key_padding_mask=padding, need_weights=True)
        delta = self.out_projection(self.norm(attended[:, 0]))
        delta = torch.where(all_invalid[:, None], torch.zeros_like(delta), delta)
        role_mass = weights[:, 0].detach().reshape(-1, 6, 6).sum(-1)
        role_mass = torch.where(all_invalid[:, None], torch.zeros_like(role_mass), role_mass)
        stats = torch.cat((valid.float().mean(-1, keepdim=True), all_invalid.float()[:, None], role_mass), -1)
        return delta, stats.detach()

    def forward(self, queries: torch.Tensor, context: Dict[str, Any]):
        if queries.ndim != 5:
            raise ValueError("Action queries must be [B,Q,A,D,H]")
        B, Q, A, D, hidden = queries.shape
        pregeom = context["pregeom"]
        centers = context["centers"].detach().reshape(B, Q * A, 3)
        rotations = context["rotations"].detach().reshape(B, Q * A, 3, 3)
        K = context["K"].detach()
        depth = context["depth"].detach()
        image_hw = context["image_hw"]
        fmap = self.feature_projection(pregeom)  # project before upsampling (bounded channels)
        if tuple(fmap.shape[-2:]) != tuple(image_hw):
            fmap = F.interpolate(fmap, size=image_hw, mode="bilinear", align_corners=False)
        # Existing GraspNet/CDF decoder: bin d means (d+1)*0.01 metres.
        insertion = torch.arange(1, D + 1, device=queries.device, dtype=centers.dtype) * 0.01
        flat = queries.reshape(B, Q * A * D, hidden)
        outputs, totals = [], queries.new_zeros(8, dtype=torch.float32)
        nstats = 0
        for b in range(B):
            chunks = []
            for start in range(0, flat.shape[1], self.cfg.action_chunk):
                end = min(start + self.cfg.action_chunk, flat.shape[1])
                ids = torch.arange(start, end, device=queries.device)
                parent, depth_id = ids // D, ids % D
                inputs = (flat[b, start:end], fmap[b:b+1], depth[b:b+1],
                          centers[b, parent], rotations[b, parent], K[b], insertion[depth_id])
                def run(*args):
                    return self._read_chunk(*args, image_hw=image_hw)
                if self.training and self.cfg.activation_checkpoint and torch.is_grad_enabled():
                    delta, stat = checkpoint(run, *inputs, use_reentrant=False)
                else:
                    delta, stat = run(*inputs)
                chunks.append(delta)
                totals = totals + stat.sum(0)
                nstats += stat.shape[0]
            outputs.append(torch.cat(chunks, 0))
        self.last_debug = {
            "D: GVAR valid projection": totals[0] / max(nstats, 1),
            "D: GVAR all invalid": totals[1] / max(nstats, 1),
            "D: GVAR probes per action": queries.new_tensor(36.),
            "D: GVAR actions": queries.new_tensor(float(Q * A * D)),
        }
        for i, name in enumerate(ROLE_NAMES):
            self.last_debug[f"D: GVAR attention {name}"] = totals[i+2] / max(nstats, 1)
        return queries + torch.stack(outputs, 0).reshape_as(queries)


def apply_depthwise_heads(width_features, score_features, width_head, cdf_head, increment_bias):
    """Use the ORIGINAL per-depth output weights on depth-specific features.

    Features [B,Q,A,D,64]; returns width [B,D,Q,A], monotonic logits [B,T,Q,A,D].
    No candidate is translated, no new depth bin, label or quality loss is added.
    """
    B, Q, A, D, C = width_features.shape
    if score_features.shape != width_features.shape:
        raise ValueError("Width/score feature shapes differ")
    if width_head.weight.shape != (D, C, 1) or cdf_head.out_channels % D:
        raise ValueError("Output heads do not match D and channel count")
    T = cdf_head.out_channels // D
    width = torch.einsum("bqadc,dc->bdqa", width_features, width_head.weight[..., 0])
    width = width + width_head.bias[None, :, None, None]
    raw = torch.einsum("bqadc,dtc->bdtqa", score_features, cdf_head.weight[..., 0].reshape(D, T, C))
    raw = raw + cdf_head.bias.reshape(1, D, T, 1, 1)
    first = raw[:, :, :1]
    logits = torch.cat((first, first + torch.cumsum(F.softplus(raw[:, :, 1:] + increment_bias), dim=2)), dim=2)
    return width, logits.permute(0, 2, 3, 4, 1).contiguous()
