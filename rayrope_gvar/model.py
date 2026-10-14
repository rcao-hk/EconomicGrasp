"""Optionally attach the original action-conditioned GVAR reader after Point RayRoPE.

This is an *additive* adapter over EconomicGraspMoGeRayRoPE: no modifications to
pinned main files, no depth/teacher input, no alternate grasp labels. When
GVAR is disabled the model's state_dict and execution are identical to U0
RayRoPE Point (under equal seeds/config and the same runtime).
"""
from __future__ import annotations
from moge_rayrope.model import EconomicGraspMoGeRayRoPE
from .decoder import GripperVolumeCDFDecoder
from .reader import GVARConfig


class EconomicGraspPointGVAR(EconomicGraspMoGeRayRoPE):
    def __init__(self, cfg, *, enable_gvar=False, gvar_config=None):
        if cfg.use_moge or not cfg.use_rayrope or cfg.ray_encoding != "point" or cfg.uncertainty != "fixed":
            raise ValueError("Joint study requires MoGe-off, RayRoPE Point, and no learned sigma")
        if cfg.pose_mode != "global_film" or cfg.use_shape_tokens or cfg.uncertainty_loss != "interval":
            raise ValueError("The feasibility study locks the original pose and loss conventions")
        super().__init__(cfg)
        self.gvar_enabled = bool(enable_gvar)
        self.gvar_config = gvar_config or GVARConfig(variant="volume_rel")
        self._gvar_pregeom = None
        if self.gvar_config.variant != "volume_rel":
            raise ValueError("Joint study supports GVAR volume_rel, not alternative reader modes")
        if self.gvar_enabled:
            base = self.base
            if base.use_obs_depth or base.geometry_depth_source != "pred" or base.seed_selection_mode != "image_fps":
                raise ValueError("GVAR requires predicted geometry, RGB-only input and image-FPS seeds")
            if not base.use_cdf or base.kview_config.mode != "A1":
                raise ValueError("GVAR requires CDF / A1 Top-1 decoder")
            decoder = base.kview_grasp_module.decoder
            base.kview_grasp_module.decoder = GripperVolumeCDFDecoder(
                decoder, base.seed_feature_dim, self.gvar_config)
            # Keep reader input on pre-geometry appearance features; only numeric
            # depth/poses are detached. Permanent hooks have no state_dict keys.
            base.proposal_head.register_forward_hook(self._capture_pregeom)
            base.kview_grasp_module.group.register_forward_pre_hook(
                self._supply_gvar_context, with_kwargs=True)

    def _capture_pregeom(self, module, args, output):
        if not isinstance(output, (tuple,list)) or len(output)!=2:
            raise RuntimeError("Expected proposal_head output (image_features, logits)")
        self._gvar_pregeom=output[0]

    def _supply_gvar_context(self, module, args, kwargs):
        if args or self._gvar_pregeom is None:
            raise RuntimeError("GVAR/RayRoPE context requires pre-geometry features and kwargs")
        ep=kwargs.get("end_points")
        if not isinstance(ep,dict) or "_gvar_context" in ep:
            raise RuntimeError("Invalid or stale GVAR context")
        ep["_gvar_context"]={
            "pregeom":self._gvar_pregeom,
            "centers":kwargs["seed_xyz"].detach(),
            "rotations":kwargs["top_view_rot"].detach(),
            "K":kwargs["camera_K"].detach(),
            "depth":kwargs["depth_map"].detach(),
            "image_hw":tuple(kwargs["depth_map"].shape[-2:]),
        }

    def forward(self,batch):
        self._gvar_pregeom=None
        try:
            ep=super().forward(batch)
            if self.gvar_enabled and "_gvar_context" in ep:
                raise RuntimeError("GVAR decoder failed to consume the action context")
            return ep
        finally:
            self._gvar_pregeom=None
            batch.pop("_gvar_context",None)
