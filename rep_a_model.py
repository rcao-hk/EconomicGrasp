"""Experiment A: actual main-branch CVA reader, with an optional RGB bypass.

The baseline imports the original enhancer/grouping; it is not a new surrogate
point-cloud reader. All four cells train enhancer, grouping and action scorer.
A2/A3 additionally train an action-projected pre-enhancer image reader. DINO,
proposal DPT and metric-depth predictor are fixed in the export stage.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from rep_a_common import VARIANTS


class IndependentImageReader(nn.Module):
    """40 samples in 5 gripper regions; NO estimated depth input or masking."""
    def __init__(self, channels, dim=128, heads=4, dropout=.1):
        super().__init__()
        # Corners within closing region, left/right finger, palm, approach.
        corners = torch.cartesian_prod(torch.tensor([.15, .85]),
                                       torch.tensor([.15, .85]), torch.tensor([.15, .85]))
        self.register_buffer("corners", corners, persistent=False)
        self.role = nn.Embedding(5, dim)
        self.token = nn.Sequential(nn.Linear(channels+3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query = nn.Sequential(nn.Linear(15, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def sample_points(self, actions):
        a = actions.reshape(-1, 17)
        w, h, d = a[:, 1], a[:, 2], a[:, 3]
        lo_x, hi_x = d-.06, d
        # [N,5,3] lower/upper corners in the actual gripper frame.
        lo = torch.stack((
            torch.stack((lo_x, -w/2, -h/2), -1),
            torch.stack((lo_x, -w/2-.01, -h/2), -1),
            torch.stack((lo_x, w/2, -h/2), -1),
            torch.stack((lo_x-.01, -w/2-.01, -h/2), -1),
            torch.stack((lo_x-.04, -w/2-.01, -h/2), -1)), 1)
        hi = torch.stack((
            torch.stack((hi_x, w/2, h/2), -1),
            torch.stack((hi_x, -w/2, h/2), -1),
            torch.stack((hi_x, w/2+.01, h/2), -1),
            torch.stack((lo_x, w/2+.01, h/2), -1),
            torch.stack((lo_x-.01, w/2+.01, h/2), -1)), 1)
        local = lo[:, :, None] + (hi-lo)[:, :, None]*self.corners[None, None]
        local = local.reshape(-1, 40, 3)
        rotation = a[:, 4:13].reshape(-1, 3, 3)
        xyz = torch.einsum("nmc,njc->nmj", local, rotation) + a[:, None, 13:16]
        return local, xyz

    def forward(self, feature, actions, K, image_hw):
        """feature [1,C,Hf,Wf], actions [N,17], K [1,3,3]."""
        a = actions.reshape(-1, 17)
        local, xyz = self.sample_points(a)
        h, w = image_hw
        projected = xyz @ K[0].T
        uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
        valid = (xyz[..., 2] > .01) & torch.isfinite(uv).all(-1)
        valid &= (uv[..., 0] >= 0) & (uv[..., 0] <= w-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)
        # Matches the half-pixel convention of interpolate(...,align_corners=False).
        grid = 2*(uv+.5)/uv.new_tensor([w, h])-1
        grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
        feat = F.grid_sample(feature, grid[None], padding_mode="zeros", align_corners=False)
        feat = feat[0].permute(1, 2, 0)
        roles = self.role(torch.arange(5, device=a.device)).repeat_interleave(8, 0)
        kv = self.token(torch.cat((feat, local/.1), -1)) + roles[None]
        kv = kv * valid[..., None]
        mask = ~valid
        empty = mask.all(-1)
        # Safe finite fallback: zero-valued key, with output suppressed below.
        mask = mask.clone()
        mask[empty, 0] = False
        q = self.query(a[:, 1:16])[:, None]
        v, _ = self.attn(q, kv, kv, key_padding_mask=mask, need_weights=False)
        x = self.norm1(q+v)
        x = self.norm2(x+self.ffn(x))[:, 0]
        return x * (~empty)[:, None]


class RepAReaderScorer(nn.Module):
    def __init__(self, init, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        # Both source files are pure torch/numpy; no detector/evaluator is built.
        from models.grasp_spatial_enhancer import GraspSpatialEnhancer
        from models.kview_query_transformer import KViewQueryTransformerConfig, ViewConditionedAttentionGrouping
        self.variant = variant
        self.config = init["model_config"]
        self.enhancer = GraspSpatialEnhancer(**self.config["enhancer"])
        group_config = KViewQueryTransformerConfig(**self.config["group_config"])
        self.group = ViewConditionedAttentionGrouping(
            seed_feature_dim=self.config["channels"], feat_dim=self.config["channels"],
            out_dim=self.config["out_dim"], config=group_config)
        self.enhancer.load_state_dict(init["enhancer_state"], strict=True)
        self.group.load_state_dict(init["group_state"], strict=True)
        dim = self.config["out_dim"]
        self.action_embed = nn.Sequential(nn.Linear(15, dim), nn.GELU(), nn.Linear(dim, dim))
        # Identical scoring head/initialization in all cells. Width/R/depth stay fixed.
        self.scorer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                    nn.Dropout(.1), nn.Linear(dim, 6))
        # Allocate AFTER shared modules, so common initialization is paired.
        self.rgb_reader = IndependentImageReader(self.config["channels"], dim) if VARIANTS[variant][0] else None

    def encode_components(self, data, depth):
        """Return trained representation components before fusion.

        Depth affects only the original enhancer/grouping path. The RGB
        component reads pre-enhancer image features and never receives depth.
        This supports Rep-A-P1 checkpoint interventions without retraining.
        """
        raw = data["image_feature"][None]
        K = data["K"][None]
        depth = depth[None]  # [1,1,H,W]
        h, w = depth.shape[-2:]
        a = data["actions"].reshape(-1, 17)
        k, q = data["actions"].shape[:2]
        token_ids = data["token_ids"].reshape(1, q).expand(k, q).reshape(1, k*q)
        enhanced, _ = self.enhancer(raw, depth_prob=None, depth_map=depth, K=K,
                                    image_hw=(h, w), return_maps=False)
        feat_map = F.interpolate(enhanced, size=(h, w), mode="bilinear", align_corners=False)
        seed = feat_map.flatten(2).gather(2, token_ids[:, None].expand(-1, feat_map.shape[1], -1))
        depth_rep = self.group(seed_features=seed, token_sel_idx=token_ids,
                          seed_xyz=a[None, :, 13:16], top_view_rot=a[:, 4:13].reshape(1, -1, 3, 3),
                          feat_map=feat_map, depth_map=depth,
                          objectness_logits=data["objectness"][None],
                          graspness_map=data["graspness"][None], camera_K=K, end_points={})
        depth_rep = depth_rep[0].T
        action_rep = self.action_embed(a[:, 1:16])
        rgb_rep = None
        if self.rgb_reader is not None:
            # Preserve the exact fusion scale used during A2/A3 training.
            rgb_rep = .1*self.rgb_reader(raw, a, K, (h, w))
        return {"depth": depth_rep, "rgb": rgb_rep, "action": action_rep}

    def encode(self, data, depth, intervention="full", return_components=False):
        comps = self.encode_components(data, depth)
        if intervention == "full":
            rep = comps["depth"] + comps["action"]
            if comps["rgb"] is not None:
                rep = rep + comps["rgb"]
        elif intervention == "no_rgb":
            rep = comps["depth"] + comps["action"]
        elif intervention == "rgb_only":
            if comps["rgb"] is None:
                raise ValueError("rgb_only requires an A2/A3 checkpoint with the RGB reader.")
            # Do not renormalize: keep the learned 0.1 branch scale.
            rep = comps["rgb"] + comps["action"]
        else:
            raise ValueError(f"Unknown Rep-A intervention: {intervention!r}")
        return (rep, comps) if return_components else rep

    def forward(self, data, depth=None, return_repr=False, intervention="full",
                return_components=False):
        encoded = self.encode(
            data,
            data["depth"] if depth is None else depth,
            intervention=intervention,
            return_components=return_components,
        )
        if return_components:
            rep, comps = encoded
        else:
            rep, comps = encoded, None
        logits = self.scorer(rep).reshape(*data["actions"].shape[:2], 6)
        if return_repr and return_components:
            return logits, rep, comps
        if return_repr:
            return logits, rep
        if return_components:
            return logits, comps
        return logits
