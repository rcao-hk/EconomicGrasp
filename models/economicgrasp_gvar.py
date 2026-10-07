"""GVAR integration into main@52d09f9. Existing main files remain unchanged."""
from __future__ import annotations

import torch
from torch import nn

from models.economicgrasp_bip3d import economicgrasp_dpt
from models.kview_query_transformer import CenterViewAngleCDFCandidateDecoder
from models.gripper_volume_reader import GVARConfig, ActionDepthAdapter, GripperVolumeReader, apply_depthwise_heads


class GripperVolumeCDFDecoder(CenterViewAngleCDFCandidateDecoder):
    def __init__(self, base, feat_dim, config):
        # Transfer the already initialized common stem/heads without consuming
        # another RNG stream or changing their state-dict names.
        nn.Module.__init__(self)
        for key in ("num_angle", "num_depth", "num_cdf_thresholds", "cdf_increment_bias", "hidden_dim", "branch_attn_chunk_size", "debug_prefix"):
            setattr(self, key, getattr(base, key))
        for key, module in base.named_children():
            self.add_module(key, module)
        self.gvar_config = config
        self.action_adapter = ActionDepthAdapter(self.hidden_dim, self.num_depth, self.score_dropout.p)
        self.reader = (None if config.variant == "slot"
                       else GripperVolumeReader(feat_dim, self.hidden_dim, config))

    def forward(self, group_features_angle, end_points):
        B, _, QA = group_features_angle.shape
        Q = int(end_points["kview_angle_query_base_q"])
        A, D = self.num_angle, self.num_depth
        if QA != Q * A or int(end_points["kview_angle_query_num_angle"]) != A:
            raise ValueError("GVAR received inconsistent Q/A shape")
        context = end_points.pop("_gvar_context", None)
        x = self.input_proj(group_features_angle).transpose(1, 2).reshape(B, Q, A, self.hidden_dim)
        x = x + self.angle_embed(torch.arange(A, device=x.device).view(1, 1, A))
        x = x.reshape(B * Q, A, self.hidden_dim)
        for layer in self.layers:
            x = layer(x)
        angle_feature = x.reshape(B, Q, A, self.hidden_dim)
        if bool(end_points.get("cva_export_angle_feature", False)):
            end_points["cva_angle_feature"] = angle_feature
        action = self.action_adapter.make_queries(angle_feature)
        if self.reader is not None:
            if context is None:
                raise RuntimeError("Missing gripper-reader context; instantiate EconomicGraspGVAR")
            action = self.reader(action, context)
            end_points.update(self.reader.last_debug)
        action = self.action_adapter.finish(action)
        flat = action.reshape(B, Q * A * D, self.hidden_dim).transpose(1, 2).contiguous()
        width = self.width_proj(flat).transpose(1, 2).reshape(-1, 1, 64)
        score = self.score_proj(flat).transpose(1, 2).reshape(-1, 1, 64)
        branches = self._run_branch_attention(torch.cat((width, score), 1))
        width = self.width_dropout(branches[:, 0]).reshape(B, Q, A, D, 64)
        score = self.score_dropout(branches[:, 1]).reshape(B, Q, A, D, 64)
        widths, logits = apply_depthwise_heads(width, score, self.width_head, self.cdf_head, self.cdf_increment_bias)
        if not torch.isfinite(widths).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite GVAR outputs")
        end_points["grasp_width_pred_angle_depth"] = widths
        end_points["grasp_cdf_pred_angle_depth"] = logits
        with torch.no_grad():
            end_points["D: GVAR insertion feature spread"] = action.detach().std(dim=3, unbiased=False).mean()
        return end_points


class EconomicGraspGVAR(economicgrasp_dpt):
    """Same input/label/decoder interface; all task->metric-depth paths detached.

    E: GraspSpatialEnhancer detaches depth-derived geometry.
    Q: seed selection/backprojection receives detached depth values.
    C: CVA support depth values and new-reader depth values are detached.
    Image features, attention, CDF and depth-supervision graphs remain trainable.
    """
    def __init__(self, *args, gvar_config=None, **kwargs):
        cfg = gvar_config or GVARConfig()
        super().__init__(*args, **kwargs)
        self.gvar_config = cfg
        if not self.use_cdf or self.use_obs_depth or self.geometry_depth_source != "pred":
            raise ValueError("GVAR requires RGB predicted geometry and CDF")
        if self.seed_selection_mode != "image_fps":
            raise ValueError("GVAR requires main's image-FPS proposal mode")
        self.spatial_enhancer.detach_depth_grad = True
        self.kview_config.detach_depth = True
        self.kview_config.detach_aux_maps = True
        self._gvar_pregeom = None
        if cfg.variant != "baseline":
            local = self.kview_grasp_module
            local.decoder = GripperVolumeCDFDecoder(local.decoder, self.seed_feature_dim, cfg)
            if cfg.variant != "slot":
                self.proposal_head.register_forward_hook(self._capture_pregeom)
                local.group.register_forward_pre_hook(self._attach_context, with_kwargs=True)

    def _select_graspable_seed_queries(self, *args, **kwargs):
        if args or "depth_map" not in kwargs:
            raise RuntimeError("Main seed-selection signature changed; audit Q detach before running")
        kwargs = dict(kwargs)
        kwargs["depth_map"] = kwargs["depth_map"].detach()
        return super()._select_graspable_seed_queries(**kwargs)

    def _capture_pregeom(self, module, args, output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("proposal_head must return (path1, proposal_logits)")
        self._gvar_pregeom = output[0]  # NOT detached: train visual representation

    def _attach_context(self, module, args, kwargs):
        if args or self._gvar_pregeom is None:
            raise RuntimeError("Missing pre-geometry map or changed grouping signature")
        ep = kwargs["end_points"]
        ep["_gvar_context"] = {
            "pregeom": self._gvar_pregeom,
            "centers": kwargs["seed_xyz"].detach(),
            "rotations": kwargs["top_view_rot"].detach(),
            "K": kwargs["camera_K"].detach(),
            "depth": kwargs["depth_map"].detach(),
            "image_hw": tuple(kwargs["depth_map"].shape[-2:]),
        }

    def forward(self, end_points):
        self._gvar_pregeom = None
        try:
            output = super().forward(end_points)
            output["D: GVAR E detach"] = output["depth_net_pred"].new_tensor(1.)
            output["D: GVAR Q detach"] = output["depth_net_pred"].new_tensor(1.)
            output["D: GVAR C detach"] = output["depth_net_pred"].new_tensor(1.)
            return output
        finally:
            self._gvar_pregeom = None
            end_points.pop("_gvar_context", None)
