"""One image-conditioned field per scene; batched queries never mix scenes.

Legacy [C,H,W] / [F,7,17] inputs remain supported. True-batch inputs use
[B,C,H,W], [B,P,3], [B,F,7,17] and return [B,P,6]/[B,F,7].
There is no cross-image pooling or scene-state sharing.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import support_points, relative_coordinates


@dataclass(frozen=True)
class ModelConfig:
    feature_channels: int
    hidden: int = 64
    truncation_m: float = .03
    point_chunk: int = 16384  # query points per *scene*, independent of B
    family_chunk: int = 32    # families per *scene*, independent of B


def normalize_scene(scene):
    """Return a batched view of the original scene; no hidden labels are read."""
    fmap = scene['feature_map']
    if fmap.ndim not in (3, 4):
        raise ValueError('feature_map must be [C,H,W] or [B,C,H,W]')
    single = fmap.ndim == 3
    fmap = fmap.unsqueeze(0) if single else fmap
    depth = scene['pred_depth']
    K = scene['K']
    if single:
        if depth.ndim != 2 or K.shape != (3, 3):
            raise ValueError('Legacy scene depth/K must be [H,W]/[3,3]')
        depth = depth.unsqueeze(0)
        K = K.unsqueeze(0)
    if depth.ndim != 3 or K.ndim != 3 or depth.shape[0] != fmap.shape[0] or K.shape != (fmap.shape[0],3,3):
        raise ValueError('Batched scene must have depth [B,H,W] and K [B,3,3]')
    if fmap.shape[1] <= 0 or min(fmap.shape[-2:]) < 1 or min(depth.shape[-2:]) < 2:
        raise ValueError('Invalid scene spatial shape')
    return {'feature_map': fmap, 'pred_depth': depth, 'K': K}, single


def normalize_bank(bank, batch_size):
    if bank.ndim == 3:
        if batch_size != 1:
            raise ValueError('An unbatched action bank cannot represent multiple scenes')
        return bank.unsqueeze(0), True
    if bank.ndim != 4 or bank.shape[0] != batch_size or bank.shape[-1] != 17:
        raise ValueError('Action bank must be [F,K,17] or [B,F,K,17]')
    if bank.shape[1] < 1 or bank.shape[2] < 1:
        raise ValueError('Empty action family/alternative dimension')
    return bank, False


class SharedField(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.adapter = nn.Sequential(nn.Conv2d(cfg.feature_channels, cfg.hidden, 1), nn.GELU(),
                                     nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1))
        self.net = nn.Sequential(nn.Linear(cfg.hidden * 2 + 8, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 6))

    def encode(self, scene):
        batched, _ = normalize_scene(scene)
        return self.adapter(batched['feature_map'])

    def query(self, encoded, scene, xyz):
        """Probe B independent fields with B independent point sets.

        ``point_chunk`` bounds P for each scene; no B*P image expansion and no
        cross-scene averaging. Legacy query [P,3] returns [P,6].
        """
        if encoded.ndim != 4:
            raise ValueError('Encoded scene must be [B,C,Hf,Wf]')
        batch, _ = normalize_scene(scene)
        B = encoded.shape[0]
        if batch['pred_depth'].shape[0] != B or batch['feature_map'].shape[0] != B:
            raise ValueError('Encoded and geometry batch sizes disagree')
        legacy = xyz.ndim == 2
        if legacy:
            if B != 1 or xyz.shape[-1] != 3:
                raise ValueError('Unbatched [P,3] probes require one encoded scene')
            xyz = xyz.unsqueeze(0)
        elif xyz.ndim != 3 or xyz.shape[0] != B or xyz.shape[2] != 3:
            raise ValueError('Batched probes must be [B,P,3]')
        if xyz.shape[1] < 1:
            raise ValueError('Empty query point batch')
        if self.cfg.point_chunk < 1:
            raise ValueError('point_chunk must be positive')
        h, w = batch['pred_depth'].shape[-2:]
        K = batch['K'].to(device=xyz.device, dtype=xyz.dtype)
        global_feature = encoded.mean((-2, -1))  # [B,C]; never mean across B
        outputs = []
        for chunk in xyz.split(self.cfg.point_chunk, dim=1):
            pixels = torch.matmul(chunk, K.transpose(-1, -2))
            uv = pixels[..., :2] / chunk[..., 2:3].clamp_min(1e-6)
            grid = (uv + .5) / chunk.new_tensor([w, h]) * 2 - 1
            valid = ((chunk[..., 2] > 1e-6) & (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
                     & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1))
            # [B,C,P,1] -> [B,P,C]; old single-image interpolation contract unchanged.
            sample_grid = grid.unsqueeze(2)
            image = F.grid_sample(encoded, sample_grid, align_corners=False,
                                  padding_mode='zeros').squeeze(-1).transpose(1, 2)
            depth = F.grid_sample(batch['pred_depth'].unsqueeze(1), sample_grid,
                                  align_corners=False, mode='nearest').squeeze(1).squeeze(-1)
            residual = ((depth - chunk[..., 2]) / self.cfg.truncation_m).clamp(-3, 3)
            geo = torch.cat((chunk, grid, depth.unsqueeze(-1), residual.unsqueeze(-1),
                             valid.unsqueeze(-1).to(chunk)), -1)
            global_chunk = global_feature.unsqueeze(1).expand(-1, chunk.shape[1], -1)
            outputs.append(self.net(torch.cat((image, global_chunk, geo), -1)))
        out = torch.cat(outputs, dim=1)
        return out[0] if legacy else out

    @staticmethod
    def physical(raw):
        # Works for [P,6], [B,P,6], and any leading family/action dimensions.
        return torch.cat((raw[..., :1].sigmoid(), raw[..., 1:2].tanh(),
                          raw[..., 2:].softmax(-1)), -1)


class FieldGraspModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.field = SharedField(cfg)
        self.point_reader = nn.Sequential(nn.Linear(9, cfg.hidden), nn.GELU(),
                                          nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(2 * cfg.hidden + 3, cfg.hidden),
                                     nn.GELU(), nn.Linear(cfg.hidden, 1))

    def action_scores(self, encoded, scene, bank, access='fresh'):
        if access not in ('fresh', 'stale'):
            raise ValueError('Expected fresh or stale evidence')
        g, legacy = normalize_bank(bank, encoded.shape[0])
        if self.cfg.family_chunk < 1:
            raise ValueError('family_chunk must be positive')
        scores = []
        for piece in g.split(self.cfg.family_chunk, dim=1):
            if access == 'fresh':
                points, local = support_points(piece)
            else:
                old, _ = support_points(piece[:, :, :1])
                points = old.expand(-1, -1, piece.shape[2], -1, -1)
                local = relative_coordinates(points, piece)
            raw = self.field.query(encoded, scene, points.reshape(g.shape[0], -1, 3))
            physics = self.field.physical(raw).reshape(*points.shape[:-1], 6)
            token = self.point_reader(torch.cat((physics, local / .05), -1))
            descriptor = torch.cat((token.mean(-2), token.amax(-2), piece[..., 1:4] / .1), -1)
            scores.append(self.quality(descriptor)[..., 0])
        z = torch.cat(scores, dim=1)
        return z[0] if legacy else z

    def forward(self, scene, bank, access='fresh'):
        return self.action_scores(self.field.encode(scene), scene, bank, access)


def geometry_loss(raw, targets):
    """Per-image average, then batch average; no image dominates by valid count."""
    single = raw.ndim == 2
    if single:
        raw = raw.unsqueeze(0)
    if raw.ndim != 3 or raw.shape[-1] != 6:
        raise ValueError('Geometry predictions must be [P,6] or [B,P,6]')
    expected = raw.shape[:2]
    t = {}
    for name in ('udf', 'projective', 'state', 'udf_mask', 'projective_mask'):
        x = targets[name]
        if single:
            if x.ndim != 1: raise ValueError('Single-frame target must be [P]')
            x = x.unsqueeze(0)
        if x.shape != expected: raise ValueError('Geometry target/batch mismatch: '+name)
        t[name] = x
    loss_vec = []; udf_vec = []; proj_vec = []; state_vec = []
    for b in range(raw.shape[0]):
        r = raw[b]
        um, pm = t['udf_mask'][b], t['projective_mask'][b]
        udf = (r[um, 0].sigmoid() - t['udf'][b, um]).abs().mean() if um.any() else r.sum()*0
        proj = (r[pm, 1].tanh() - t['projective'][b, pm]).abs().mean() if pm.any() else r.sum()*0
        state = F.cross_entropy(r[:, 2:], t['state'][b].long())
        udf_vec.append(udf); proj_vec.append(proj); state_vec.append(state)
        loss_vec.append(udf + proj + state)
    return torch.stack(loss_vec).mean(), {'udf_l1': torch.stack(udf_vec).mean().detach(),
        'projective_l1': torch.stack(proj_vec).mean().detach(),
        'state_ce': torch.stack(state_vec).mean().detach()}


def quality_loss(logits, counts, max_pairs=512):
    """Per-image family quality, not cross-image pairwise ranking."""
    legacy = logits.ndim == 2
    if legacy:
        logits, counts = logits.unsqueeze(0), counts.unsqueeze(0)
    if logits.ndim != 3 or logits.shape != counts.shape or counts.dtype != torch.long or ((counts < 0) | (counts > 6)).any():
        raise ValueError('Aligned exact-action integer counts [B,F,K] 0..6 required')
    all_mse, all_pair = [], []
    for b in range(logits.shape[0]):
        z, q = logits[b], counts[b]
        all_mse.append(F.mse_loss(z.sigmoid(), q.to(z) / 6))
        pairs = []
        for zs, qs in zip(z, q):
            i, j = torch.triu_indices(len(zs), len(zs), 1, device=z.device)
            good = qs[i] != qs[j]
            i, j = i[good], j[good]
            if len(i):
                pairs.append(F.softplus(-torch.sign((qs[i] - qs[j]).to(z)) * (zs[i] - zs[j])).mean())
        all_pair.append(torch.stack(pairs).mean() if pairs else z.sum() * 0)
    mse = torch.stack(all_mse).mean()
    pair = torch.stack(all_pair).mean()
    return 10 * mse + .5 * pair, {'mse': mse.detach(), 'pair': pair.detach()}


class LocalControl(nn.Module):
    """Bounded frozen-feature reference; legacy and batched independent actions."""
    def __init__(self, feature_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim + 18, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, bank):
        single = bank.ndim == 3
        if single:
            if features.ndim != 2: raise ValueError('Single-image features must be [F,C]')
            bank, features = bank.unsqueeze(0), features.unsqueeze(0)
        if bank.ndim != 4 or features.ndim != 3 or features.shape[:2] != bank.shape[:2]:
            raise ValueError('Batched features/bank must be [B,F,C]/[B,F,K,17]')
        f = features.unsqueeze(2).expand(-1, -1, bank.shape[2], -1)
        delta = (bank[..., 13:16] - bank[:, :, :1, 13:16]) / .02
        geometry = torch.cat((bank[..., 1:4] / .1, bank[..., 4:13],
                              bank[..., 13:16], delta), -1)
        correction = .2 * self.net(torch.cat((f, geometry), -1))[..., 0].tanh()
        prior = torch.logit(bank[:, :, :1, 0].clamp(1e-7, 1 - 1e-7))
        result = prior + correction
        return result[0] if single else result


class PredictedGeometryControl(FieldGraspModel):
    """Same quality readout on analytical geometry from RGB predicted depth only.

    The geometry reference is CPU/KDTree based; one teacher is constructed for
    each independent scene in the batch before the shared Torch quality head.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        del self.field  # no unused learned parameters

    def forward(self, scene, bank):
        from .geometry import VisibleSurfaceTeacher
        batched, _ = normalize_scene(scene)
        g, legacy = normalize_bank(bank, batched['feature_map'].shape[0])
        teachers = [VisibleSurfaceTeacher(batched['pred_depth'][b].detach().cpu().numpy(),
                    batched['K'][b].detach().cpu().numpy(), self.cfg.truncation_m)
                    for b in range(g.shape[0])]
        result = []
        for piece in g.split(self.cfg.family_chunk, dim=1):
            points, local = support_points(piece)
            physical = []
            for b, teacher in enumerate(teachers):
                labels = teacher.label(points[b].reshape(-1, 3).detach().cpu().numpy())
                v = np.concatenate((labels['udf'][:, None], labels['projective'][:, None],
                                    np.eye(4, dtype=np.float32)[labels['state']]), -1)
                physical.append(torch.as_tensor(v, device=piece.device, dtype=piece.dtype).reshape(*points.shape[1:-1], 6))
            physics = torch.stack(physical)
            tokens = self.point_reader(torch.cat((physics, local / .05), -1))
            descriptor = torch.cat((tokens.mean(-2), tokens.amax(-2), piece[..., 1:4] / .1), -1)
            result.append(self.quality(descriptor)[..., 0])
        logits = torch.cat(result, dim=1)
        return logits[0] if legacy else logits