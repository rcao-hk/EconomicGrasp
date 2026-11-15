"""An action-independent image-conditioned field, then a strict geometry bottleneck.

No candidate count, proposal score, hidden RGB latent, or grasp identity is passed
from the scene field to the quality reader. H1/H2 differ only in geometry loss.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import support_points, relative_coordinates


@dataclass(frozen=True)
class ModelConfig:
    feature_channels: int
    hidden: int = 64
    truncation_m: float = .03
    point_chunk: int = 16384
    family_chunk: int = 32


class SharedField(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.adapter = nn.Sequential(nn.Conv2d(cfg.feature_channels, cfg.hidden, 1), nn.GELU(),
                                     nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1))
        self.net = nn.Sequential(nn.Linear(cfg.hidden*2+8, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 6))

    def encode(self, scene):
        return self.adapter(scene['feature_map'][None])

    def query(self, encoded, scene, xyz):
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError('Field accepts camera-frame [P,3], never a grasp token')
        out = []
        h, w = scene['pred_depth'].shape
        K = scene['K']
        for chunk in xyz.split(self.cfg.point_chunk):
            pixel = chunk @ K.T
            uv = pixel[:, :2]/chunk[:, 2:3].clamp_min(1e-6)
            grid = (uv+.5)/chunk.new_tensor([w, h])*2-1
            valid = (chunk[:, 2] > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
            image = F.grid_sample(encoded, grid[None, :, None], align_corners=False,
                                  padding_mode='zeros')[0, :, :, 0].T
            depth = F.grid_sample(scene['pred_depth'][None, None], grid[None, :, None],
                                  align_corners=False, mode='nearest')[0, 0, :, 0]
            residual = ((depth-chunk[:, 2])/self.cfg.truncation_m).clamp(-3, 3)
            geo = torch.cat((chunk, grid, depth[:, None], residual[:, None], valid[:, None].to(chunk)), -1)
            global_feature = encoded.mean((-2, -1))[0].expand(len(chunk), -1)
            raw = self.net(torch.cat((image, global_feature, geo), -1))
            out.append(raw)
        return torch.cat(out)

    @staticmethod
    def physical(raw):
        # Two verifiable distances + four observation-state probabilities.
        return torch.cat((raw[:, :1].sigmoid(), raw[:, 1:2].tanh(), raw[:, 2:].softmax(-1)), -1)


class FieldGraspModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.field = SharedField(cfg)
        self.point_reader = nn.Sequential(nn.Linear(9, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(2*cfg.hidden+3, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))

    def action_scores(self, encoded, scene, bank, access='fresh'):
        if bank.ndim != 3 or bank.shape[-1] != 17 or access not in ('fresh', 'stale'):
            raise ValueError('Expected family x alternatives x 17, fresh/stale access')
        scores = []
        for g in bank.split(self.cfg.family_chunk):
            if access == 'fresh':
                points, local = support_points(g)
            else:
                # Old physical support sites; express them in the NEW action frame.
                # New actions are known, but no new scene sites are queried.
                old, _ = support_points(g[:, :1])
                points = old.expand(-1, g.shape[1], -1, -1)
                local = relative_coordinates(points, g)
            raw = self.field.query(encoded, scene, points.reshape(-1, 3))
            physics = self.field.physical(raw).reshape(*points.shape[:-1], 6)
            token = self.point_reader(torch.cat((physics, local/.05), -1))
            descriptor = torch.cat((token.mean(-2), token.amax(-2), g[..., 1:4]/.1), -1)
            scores.append(self.quality(descriptor)[..., 0])
        return torch.cat(scores)

    def forward(self, scene, bank, access='fresh'):
        return self.action_scores(self.field.encode(scene), scene, bank, access)


def geometry_loss(raw, targets):
    def masked(v, m):
        return v[m].mean() if m.any() else raw.sum()*0
    udf = masked((raw[:, 0].sigmoid()-targets['udf']).abs(), targets['udf_mask'])
    proj = masked((raw[:, 1].tanh()-targets['projective']).abs(), targets['projective_mask'])
    state = F.cross_entropy(raw[:, 2:], targets['state'])
    return udf + proj + state, {'udf_l1': udf.detach(), 'projective_l1': proj.detach(), 'state_ce': state.detach()}


def quality_loss(logits, counts, max_pairs=512):
    if logits.shape != counts.shape or counts.dtype != torch.long or ((counts < 0) | (counts > 6)).any():
        raise ValueError('Aligned exact-action integer counts 0..6 required')
    mse = F.mse_loss(logits.sigmoid(), counts.to(logits)/6)
    pairs = []
    # Same-family comparisons: no preferred movement direction, no quality ties.
    for z, q in zip(logits, counts):
        i, j = torch.triu_indices(len(z), len(z), 1, device=z.device)
        good = q[i] != q[j]
        i, j = i[good], j[good]
        if len(i):
            pairs.append(F.softplus(-torch.sign((q[i]-q[j]).to(z))*(z[i]-z[j])).mean())
    pair = torch.stack(pairs).mean() if pairs else logits.sum()*0
    return 10*mse + .5*pair, {'mse': mse.detach(), 'pair': pair.detach()}

class LocalControl(nn.Module):
    """Bounded frozen-feature reference, not the main geometry representation.

    One parent feature plus explicit action change; cannot acquire new evidence.
    Only this reference reuses the source score. Field models cannot access it.
    """
    def __init__(self, feature_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim+18, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, bank):
        f = features[:, None].expand(-1, bank.shape[1], -1)
        delta = (bank[..., 13:16]-bank[:, :1, 13:16])/.02
        geometry = torch.cat((bank[..., 1:4]/.1, bank[..., 4:13], bank[..., 13:16], delta), -1)
        correction = .2*self.net(torch.cat((f, geometry), -1))[..., 0].tanh()
        prior = torch.logit(bank[:, :1, 0].clamp(1e-7, 1-1e-7))
        return prior + correction

class PredictedGeometryControl(FieldGraspModel):
    """Same quality readout on analytic geometry from RGB PREDICTED depth only."""
    def __init__(self, cfg):
        super().__init__(cfg)
        del self.field  # no unused learned field parameters in the control

    def forward(self, scene, bank):
        from .geometry import VisibleSurfaceTeacher
        import numpy as np
        teacher = VisibleSurfaceTeacher(scene['pred_depth'].detach().cpu().numpy(),
                                        scene['K'].detach().cpu().numpy(), self.cfg.truncation_m)
        scores = []
        for g in bank.split(self.cfg.family_chunk):
            points, local = support_points(g)
            labels = teacher.label(points.reshape(-1, 3).detach().cpu().numpy())
            physics = np.concatenate((labels['udf'][:, None], labels['projective'][:, None],
                                      np.eye(4, dtype=np.float32)[labels['state']]), -1)
            physics = torch.as_tensor(physics, device=g.device, dtype=g.dtype).reshape(*points.shape[:-1], 6)
            token = self.point_reader(torch.cat((physics, local/.05), -1))
            descriptor = torch.cat((token.mean(-2), token.amax(-2), g[..., 1:4]/.1), -1)
            scores.append(self.quality(descriptor)[..., 0])
        return torch.cat(scores)
