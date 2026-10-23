"""Optional Point RayRoPE grouping and GVAR readout for the main CVA model.

Ported from exp/moge-rayrope-grasp at bb2cf57. Both default off. Numeric
geometry is detached; appearance features retain gradients in either branch.
"""
from dataclasses import asdict

import torch


def design_options(args):
    return dict(
        enable_point_rayrope=bool(getattr(args, "enable_point_rayrope", False)),
        enable_gvar=bool(getattr(args, "enable_gvar", False)),
        rayrope_group_chunk=int(getattr(args, "rayrope_group_chunk", 32)),
        gvar_action_chunk=int(getattr(args, "gvar_action_chunk", 256)),
    )


def supply_gvar_context(module, args, kwargs):
    if args:
        raise RuntimeError("GVAR grouping requires keyword inputs")
    ep = kwargs["end_points"]
    if "_gvar_context" in ep:
        raise RuntimeError("Stale GVAR context")
    ep["_gvar_context"] = dict(
        pregeom=ep.pop("_gvar_pregeom"),
        centers=kwargs["seed_xyz"].detach(),
        rotations=kwargs["top_view_rot"].detach(),
        K=kwargs["camera_K"].detach(),
        depth=kwargs["depth_map"].detach(),
        image_hw=tuple(kwargs["depth_map"].shape[-2:]),
    )


def install_grasp_designs(model, *, enable_point_rayrope=False, enable_gvar=False,
                          rayrope_group_chunk=32, gvar_action_chunk=256):
    """No module construction/RNG consumption when both designs are disabled."""
    model.grasp_design = dict(enable_point_rayrope=bool(enable_point_rayrope),
                             enable_gvar=bool(enable_gvar))
    if not (enable_point_rayrope or enable_gvar):
        return
    if model.use_obs_depth or model.use_gt_xyz_for_train or model.geometry_depth_source != "pred":
        raise ValueError("Point RayRoPE/GVAR require predicted RGB geometry")
    if not model.use_cdf or model.kview_config.mode != "A1":
        raise ValueError("Point RayRoPE/GVAR require --use_cdf --kview_mode A1")
    group_module = model.kview_grasp_module
    if enable_point_rayrope:
        from moge_rayrope.config import ModelConfig
        from moge_rayrope.attention import GraspRayRoPEGrouping

        class PointGrouping(GraspRayRoPEGrouping):
            def forward(self, *args, **kwargs):
                # Point RoPE uses zero interval width and no shape tokens.
                self.geometry_context = {"sigma": torch.zeros_like(kwargs["depth_map"])}
                try:
                    return super().forward(*args, **kwargs)
                finally:
                    self.geometry_context = None

        cfg = ModelConfig(use_rayrope=True, ray_encoding="point",
                          group_chunk=rayrope_group_chunk)
        group_module.group = PointGrouping(cfg, model.seed_feature_dim,
                                          group_module.decoder.input_proj.in_channels)
        model.grasp_design["rayrope"] = asdict(cfg)
    if enable_gvar:
        from rayrope_gvar.reader import GVARConfig
        from rayrope_gvar.decoder import GripperVolumeCDFDecoder
        cfg = GVARConfig(variant="volume_rel", action_chunk=gvar_action_chunk,
                         activation_checkpoint=True)
        group_module.decoder = GripperVolumeCDFDecoder(
            group_module.decoder, model.seed_feature_dim, cfg)
        group_module.group.register_forward_pre_hook(supply_gvar_context, with_kwargs=True)
        model.grasp_design["gvar"] = asdict(cfg)


def validate_checkpoint_design(model, checkpoint):
    """Reject silent changes to parameter-free settings (grid/chunk/flags)."""
    expected = checkpoint.get("grasp_design") if isinstance(checkpoint, dict) else None
    actual = model.grasp_design
    if expected is None:
        if actual["enable_point_rayrope"] or actual["enable_gvar"]:
            raise ValueError("Enabled designs require a checkpoint with grasp_design metadata")
    elif expected != actual:
        raise ValueError(f"Checkpoint grasp design mismatch: saved={expected}, requested={actual}")
