"""DCR-AIR: frozen DCR corrector plus independent action-aligned image evidence."""
from __future__ import annotations

import torch
from torch import nn

from dcr_air_common import action_keypoints_camera, project_keypoints, sample_image_features
from dcr_cva_common import DCR_VERSION
from e1e2_common import file_sha, load_torch
from models.economicgrasp_cva_centers import extract_depth_features, load_reference
from models.economicgrasp_cva_dcr import DecoupledCenterRankingCVA


def load_frozen_dcr(stage1_checkpoint, dcr_checkpoint, device):
    ck = load_torch(dcr_checkpoint)
    if ck.get('version') != DCR_VERSION:
        raise RuntimeError('AIR requires a trained DCR checkpoint')
    protocol = ck['protocol']
    if protocol['reference_sha256'] != file_sha(stage1_checkpoint):
        raise RuntimeError('DCR/Stage-1 checkpoint mismatch')
    cfg = protocol['config']
    base = DecoupledCenterRankingCVA(
        load_reference(stage1_checkpoint, device),
        protocol['offsets_mm'],
        cfg['group_chunk'],
        cfg.get('rank_hidden', 128),
        cfg.get('rank_bound', .5),
        cfg.get('seed', 2032),
    ).to(device)
    base.load_learned_state(ck['model'])
    base.eval().requires_grad_(False)
    del ck
    return base, protocol


class ActionImageEvidenceReader(nn.Module):
    """Read pre-enhancer image evidence at physical gripper-region projections.

    The reader never sees the active depth map, corruption case identity, CAD
    labels, or exact utility. Geometry enters only through the explicit action
    used to place the gripper key points in the image.
    """
    def __init__(self, feature_dim: int, hidden: int = 128, bound: float = 1.0):
        super().__init__()
        if min(feature_dim, hidden) < 1 or not 0 < bound <= 4:
            raise ValueError('Invalid AIR dimensions/bound')
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.bound = float(bound)
        self.num_keypoints = 13
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(15, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.type_embed = nn.Parameter(torch.empty(self.num_keypoints, hidden))
        nn.init.normal_(self.type_embed, std=.02)
        self.attn = nn.Linear(hidden, 1)
        self.residual = nn.Sequential(
            nn.LayerNorm(2 * hidden + 1),
            nn.Linear(2 * hidden + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer(
            'action_scale',
            torch.tensor([.10, .02, .04] + [1.] * 9 + [1., 1., 1.],
                         dtype=torch.float32),
        )

    def forward(self, pre_feature, K, image_hw, actions, valid):
        if actions.shape[:-1] != valid.shape:
            raise ValueError('Action/valid shape mismatch')
        points = action_keypoints_camera(actions)
        uv, visible = project_keypoints(points, K, image_hw)
        visible = visible & valid[..., None]
        sampled = sample_image_features(pre_feature, uv, visible, image_hw)
        if sampled.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f'Pre-enhancer channel mismatch: got {sampled.shape[-1]}, '
                f'expected {self.feature_dim}')

        action = actions[..., 1:16].float() / self.action_scale.to(actions)
        action_h = self.action_proj(action)
        point_h = self.feature_proj(sampled)
        point_h = point_h + self.type_embed.to(point_h)[None, None]
        point_h = point_h + action_h.unsqueeze(-2)

        score = self.attn(torch.tanh(point_h)).squeeze(-1)
        # Avoid NaNs for an entirely invisible candidate: mask, multiply, then
        # renormalize explicitly instead of softmax(-inf,...,-inf).
        score = score.masked_fill(~visible, -1e4)
        weight = torch.softmax(score, -1) * visible.to(score.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (weight.unsqueeze(-1) * point_h).sum(-2)
        vis_ratio = visible.float().mean(-1, keepdim=True)
        x = torch.cat((pooled, action_h, vis_ratio), -1)
        residual = self.bound * torch.tanh(self.residual(x).squeeze(-1))
        residual = residual * valid.to(residual.dtype)
        return residual, {
            'visible_ratio': vis_ratio.squeeze(-1),
            'attention_max': weight.max(-1).values,
        }


class ActionImageDCR(nn.Module):
    """A zero-init image residual over a frozen, already-trained DCR corrector."""
    def __init__(self, frozen_dcr, hidden=128, bound=1.0, seed=2041):
        super().__init__()
        self.base = frozen_dcr.eval().requires_grad_(False)
        feat_dim = int(getattr(self.base.reference, 'seed_feature_dim', 128))
        # Adding AIR must not perturb any RNG-dependent base behavior.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.reader = ActionImageEvidenceReader(feat_dim, hidden, bound)

    @property
    def zero(self):
        return self.base.zero

    @property
    def reference(self):
        return self.base.reference

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def learned_state(self):
        return {k: v.detach().cpu() for k, v in self.reader.state_dict().items()}

    def load_learned_state(self, state):
        self.reader.load_state_dict(state, strict=True)

    def forward(self, batch, bundle=None, case='nominal', case_seed=0,
                query_limit=0, query_chunk=64, depth_pack=None):
        if batch['img'].shape[0] != 1:
            raise ValueError('AIR follows the one-frame E1/DCR protocol')
        pack = (extract_depth_features(self.reference, batch)
                if depth_pack is None else depth_pack)
        with torch.no_grad():
            base_logits, bundle, depth = self.base.corrector(
                batch, bundle, case=case, case_seed=case_seed,
                query_limit=query_limit, query_chunk=query_chunk,
                depth_pack=pack)
            h, w = batch['img'].shape[-2:]
            pre_feature, _ = self.base.corrector.image_adapter(
                pack[4], h // 14, w // 14)
        residual, diagnostics = self.reader(
            pre_feature.detach(), batch['K'], (h, w),
            bundle['actions'], bundle['valid'])
        # One scalar logit shift preserves the base six-threshold CDF ordering.
        fused_logits = base_logits.detach() + residual[..., None]
        return (fused_logits, base_logits.detach(), residual,
                bundle, depth, diagnostics)
