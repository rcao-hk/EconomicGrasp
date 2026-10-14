"""Native GVAR CDF/width decoder transplanted under a Point RayRoPE group.

The class body is sourced from GVAR branch 37ebe98fe46d.
It preserves the existing CDF threshold and width-head weights/contract.
"""
from __future__ import annotations
import torch
from torch import nn
from models.kview_query_transformer import CenterViewAngleCDFCandidateDecoder
from .reader import GVARConfig, ActionDepthAdapter, GripperVolumeReader, apply_depthwise_heads


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
                raise RuntimeError("Missing gripper-reader context; instantiate EconomicGraspPointGVAR")
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
