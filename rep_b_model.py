"""Rep-B: hypothesis-conditioned image grasp representation.

The representation is formed from pre-enhancer image features and the physical
K-ray grasp hypotheses. Predicted depth, when enabled, enters only as a soft
candidate prior. It never gates or relocates RGB evidence.

Variants
--------
B0: hypothesis-conditioned RGB + cross-hypothesis relation; no depth prior.
B1: B0 + soft predicted-depth prior; nominal-depth training.
B2: B0 + soft predicted-depth prior; depth-error augmentation during training.
"""
from __future__ import annotations

import torch
from torch import nn

from rep_a_model import IndependentImageReader

REP_B_VARIANTS = {
    "B0": {"use_prior": False, "augment_depth": False},
    "B1": {"use_prior": True, "augment_depth": False},
    "B2": {"use_prior": True, "augment_depth": True},
}


class RepBModel(nn.Module):
    """Read image evidence for every physical hypothesis, then reason within ray.

    Input actions are [K,Q,17]. The finite-size gripper projection changes with
    the *physical* candidate translation, so RGB evidence is hypothesis
    conditioned even for candidates that share approximately the same center
    pixel. The relational encoder sees only the K candidates belonging to the
    same ray/query.

    Predicted depth is optional and deliberately late: it is converted to a
    smooth residual prior relative to each candidate's camera-z coordinate.
    It does not participate in image projection, masking, or feature sampling.
    """

    def __init__(
        self,
        channels: int,
        dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        prior_sigma_mm: float = 30.0,
        variant: str = "B0",
    ):
        super().__init__()
        if variant not in REP_B_VARIANTS:
            raise ValueError(f"Unknown Rep-B variant: {variant}")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if prior_sigma_mm <= 0:
            raise ValueError("prior_sigma_mm must be positive")

        self.variant = variant
        self.use_prior = bool(REP_B_VARIANTS[variant]["use_prior"])
        self.augment_depth = bool(REP_B_VARIANTS[variant]["augment_depth"])
        self.dim = int(dim)
        self.prior_sigma_m = float(prior_sigma_mm) / 1000.0

        # Same geometric image sampler used in Rep-A, but now its output is not
        # merely an additive residual: it is the primary per-hypothesis token.
        self.image_reader = IndependentImageReader(
            channels=channels, dim=dim, heads=heads, dropout=dropout
        )

        # Relative physical hypothesis identity. This is an action property,
        # not an observation-depth estimate.
        self.offset_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Absolute action geometry is retained so a robust representation does
        # not become invariant to the physical grasp itself.
        self.action_embed = nn.Sequential(
            nn.Linear(15, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Allocate common modules before the B1/B2-only prior MLP so B0/B1/B2
        # share identical initialization for common parameters under same seed.
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relational = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(dim),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 6),
        )

        self.prior_embed = (
            nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            if self.use_prior
            else None
        )

    @staticmethod
    def _safe_actions(actions: torch.Tensor, valid: torch.Tensor, zero_index: int):
        """Replace invalid actions by native action only for finite computation.

        Invalid candidates remain masked in attention/loss/selection, so this
        replacement cannot turn them into valid supervision.
        """
        if actions.ndim != 3 or actions.shape[-1] != 17:
            raise ValueError(f"Expected actions [K,Q,17], got {tuple(actions.shape)}")
        if valid.shape != actions.shape[:2]:
            raise ValueError("valid/action shape mismatch")
        native = actions[int(zero_index):int(zero_index)+1].expand_as(actions)
        safe = torch.where(valid[..., None], actions, native)
        if not torch.isfinite(safe).all():
            raise FloatingPointError("Rep-B actions remain non-finite after invalid masking")
        return safe

    def _offset_features(self, offsets_mm: torch.Tensor, q: int):
        # Normalize to a scale that keeps the formal +/-40 mm support O(1).
        off = offsets_mm.float() / 40.0
        feat = torch.stack((off, off.abs(), off.square()), -1)  # [K,3]
        return self.offset_embed(feat)[:, None, :].expand(-1, q, -1)

    def _soft_prior(
        self,
        depth: torch.Tensor,
        token_ids: torch.Tensor,
        actions: torch.Tensor,
    ):
        """Soft likelihood-like prior around observed depth at the seed pixel.

        depth: [1,H,W], token_ids: [Q], actions: [K,Q,17].
        The prior is smooth and finite; invalid observed depth is marked by a
        zero validity channel rather than hard-masking any hypothesis.
        """
        if not self.use_prior:
            return None
        if depth.ndim != 3 or depth.shape[0] != 1:
            raise ValueError(f"Expected depth [1,H,W], got {tuple(depth.shape)}")
        flat = depth[0].reshape(-1)
        ids = token_ids.long()
        if torch.any(ids < 0) or torch.any(ids >= flat.numel()):
            raise ValueError("token_ids outside depth map")
        obs = flat.index_select(0, ids)  # [Q]
        obs_valid = torch.isfinite(obs) & (obs > 0)
        obs_safe = torch.where(obs_valid, obs, torch.zeros_like(obs))

        candidate_z = actions[..., 15]  # [K,Q], camera z in metres
        delta = (candidate_z - obs_safe[None, :]) / self.prior_sigma_m
        delta = torch.nan_to_num(delta, nan=0.0, posinf=20.0, neginf=-20.0)
        delta = delta.clamp(-20.0, 20.0)
        gaussian = torch.exp(-0.5 * delta.square())
        valid_channel = obs_valid.float()[None, :].expand_as(delta)
        prior_feat = torch.stack((delta, gaussian, valid_channel), -1)
        # If observation is invalid, suppress all prior information.
        prior_feat = prior_feat * valid_channel[..., None]
        return self.prior_embed(prior_feat)

    def encode(self, data: dict, depth: torch.Tensor | None = None, return_components=False):
        required = ("image_feature", "K", "actions", "valid", "zero_index", "offsets_mm", "token_ids")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Rep-B missing inputs: {missing}")

        actions = data["actions"].float()
        valid = data["valid"].bool()
        zero_index = int(data["zero_index"])
        safe_actions = self._safe_actions(actions, valid, zero_index)
        k, q = safe_actions.shape[:2]

        feature = data["image_feature"].float()[None]
        K = data["K"].float()[None]
        # Image coordinate system matches the cached depth map/FOV. B0 does
        # not consume depth values, but image_hw is needed for projection.
        d_for_shape = data["depth"] if depth is None else depth
        h, w = d_for_shape.shape[-2:]

        image = self.image_reader(
            feature,
            safe_actions.reshape(-1, 17),
            K,
            (h, w),
        ).reshape(k, q, self.dim)

        action = self.action_embed(safe_actions[..., 1:16])
        offset = self._offset_features(data["offsets_mm"], q)
        prior = None
        if self.use_prior:
            prior = self._soft_prior(
                data["depth"] if depth is None else depth,
                data["token_ids"],
                safe_actions,
            )

        tokens = image + action + offset
        if prior is not None:
            tokens = tokens + prior

        # Transformer is batch=Q, sequence=K. Invalid hypotheses never provide
        # keys/values to another hypothesis.
        seq = tokens.permute(1, 0, 2)  # [Q,K,D]
        key_padding = (~valid).permute(1, 0)  # [Q,K]
        if torch.any(key_padding.all(-1)):
            raise ValueError("Every ray must contain at least one valid hypothesis")
        related = self.relational(seq, src_key_padding_mask=key_padding)
        rep = related.permute(1, 0, 2)  # [K,Q,D]
        rep = torch.where(valid[..., None], rep, torch.zeros_like(rep))

        if return_components:
            return rep, {
                "image": image,
                "action": action,
                "offset": offset,
                "prior": prior,
                "pre_relation": tokens,
            }
        return rep

    def forward(
        self,
        data: dict,
        depth: torch.Tensor | None = None,
        return_repr: bool = False,
        return_components: bool = False,
    ):
        encoded = self.encode(data, depth=depth, return_components=return_components)
        if return_components:
            rep, components = encoded
        else:
            rep, components = encoded, None
        logits = self.scorer(rep)
        if return_repr and return_components:
            return logits, rep, components
        if return_repr:
            return logits, rep
        if return_components:
            return logits, components
        return logits
