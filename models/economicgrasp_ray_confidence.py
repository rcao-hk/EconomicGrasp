"""Ray-conditioned confidence calibration with an unchanged Stage-1 grasp pose.

The controlled Stage-1 RGB model predicts its native image-FPS query, metric
center, Top-1 approach view, in-plane angle, insertion depth and width exactly as
before. Additional centers along the same camera ray are used only as frozen
local evidence for a small confidence Transformer. The only deployable change is

    final_score = native_Stage1_score * confidence_gate.

No alternate center is ever decoded and no baseline CDF feature is modified.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .economicgrasp_dpt_distill import economicgrasp_dpt_student
from utils.ray_confidence_ops import (
    DEFAULT_RC_OFFSETS_MM,
    build_confidence_targets,
    build_ray_centers,
    parse_offsets,
    select_raw_operation,
)

RC_CONTRACT_VERSION = 1
RC_BASE_MAIN_SHA = "3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2"

RC_LABEL_KEYS = (
    "object_poses_list",
    "grasp_points_list",
    "view_graspness_list",
    "top_view_index_list",
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
    "cdf_thresholds",
)


class RayConfidenceHead(nn.Module):
    """Compare K local hypotheses and emit one multiplicative gate per base ray."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        init_bias: float = 4.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0 or int(num_layers) <= 0 or int(num_heads) <= 0:
            raise ValueError("Ray-confidence hidden/layers/heads must be positive.")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("Ray-confidence hidden_dim must be divisible by num_heads.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Ray-confidence dropout must lie in [0,1).")

        # Five ray descriptors + normalized selected angle/depth + raw Stage-1 score.
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim + 8),
            nn.Linear(self.feature_dim + 8, self.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.constant_(self.output[-1].bias, float(init_bias))

    def forward(
        self,
        selected_angle_features: torch.Tensor,  # B,Q,K,C
        ray_descriptor: torch.Tensor,           # B,Q,K,5
        in_range: torch.Tensor,                 # B,Q,K
        raw_score: torch.Tensor,                # B,Q
        angle_idx: torch.Tensor,                # B,Q
        depth_idx: torch.Tensor,                # B,Q
        num_angle: int,
        num_depth: int,
        zero_index: int,
    ):
        if selected_angle_features.dim() != 4:
            raise ValueError("selected_angle_features must be [B,Q,K,C].")
        B, Q, K, C = selected_angle_features.shape
        if C != self.feature_dim or ray_descriptor.shape != (B, Q, K, 5):
            raise ValueError("Ray-confidence feature/descriptor shape mismatch.")
        if in_range.shape != (B, Q, K) or raw_score.shape != (B, Q):
            raise ValueError("Ray-confidence mask/score shape mismatch.")
        if not (0 <= int(zero_index) < K):
            raise ValueError("Invalid zero_index.")
        if not bool(in_range.any(dim=-1).all()):
            raise RuntimeError("Every native ray must retain at least one valid evidence depth.")

        angle_norm = (angle_idx.float() / max(float(num_angle - 1), 1.0) * 2.0 - 1.0)
        depth_norm = (depth_idx.float() / max(float(num_depth - 1), 1.0) * 2.0 - 1.0)
        operation = torch.stack((angle_norm, depth_norm, raw_score.detach().float()), dim=-1)
        operation = operation.unsqueeze(2).expand(B, Q, K, 3)
        x = torch.cat(
            (selected_angle_features.detach().float(), ray_descriptor.detach().float(), operation),
            dim=-1,
        )
        x = self.input_proj(x).reshape(B * Q, K, self.hidden_dim)
        padding = (~in_range).reshape(B * Q, K)
        hidden = self.encoder(x, src_key_padding_mask=padding).reshape(B, Q, K, self.hidden_dim)

        zero_hidden = hidden[:, :, int(zero_index)]
        valid = in_range.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        confidence_logit = self.output(torch.cat((zero_hidden, pooled), dim=-1)).squeeze(-1)
        gate = torch.sigmoid(confidence_logit)
        calibrated_score = raw_score.detach().float() * gate
        return confidence_logit, gate, calibrated_score


class RayConfidenceEvidenceModule(nn.Module):
    """Frozen native CVA path plus auxiliary multi-depth evidence extraction."""

    def __init__(
        self,
        core: nn.Module,
        offsets_mm: Sequence[float],
        min_depth: float,
        max_depth: float,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        init_bias: float = 4.0,
    ) -> None:
        super().__init__()
        self.core = core
        self.core.requires_grad_(False)
        self.core.eval()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        offsets = torch.tensor(parse_offsets(offsets_mm), dtype=torch.float32) * 1.0e-3
        self.register_buffer("offsets_m", offsets)
        zero = torch.nonzero(offsets == 0, as_tuple=False).flatten()
        if zero.numel() != 1:
            raise ValueError("Ray-confidence grid needs one zero offset.")
        self.zero_index = int(zero.item())
        feature_dim = int(self.core.decoder.input_proj.in_channels)
        self.confidence = RayConfidenceHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            init_bias=init_bias,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.training = bool(mode)
        self.core.eval()
        self.confidence.train(mode)
        return self

    @staticmethod
    def _zero_labels(labels: Mapping[str, Any], center: torch.Tensor, view_inds: torch.Tensor,
                     in_range: torch.Tensor):
        from utils.label_generation import process_grasp_labels_cdf_width
        ep = dict(labels)
        ep["xyz_graspable"] = center
        ep["grasp_top_view_inds"] = view_inds
        with torch.no_grad():
            _, ep = process_grasp_labels_cdf_width(ep)
            nearest = ep["batch_grasp_point"]
            distance = (center - nearest).float().norm(dim=-1)
            point_known = torch.isfinite(distance) & in_range
            point_support = (distance < 0.005) & point_known
        return ep, point_support.detach(), point_known.detach()

    def forward(
        self,
        seed_features: torch.Tensor,
        seed_xyz: torch.Tensor,
        token_sel_idx: torch.Tensor,
        feat_map: torch.Tensor,
        depth_map: torch.Tensor,
        camera_K: torch.Tensor,
        end_points: dict,
        is_training: bool,
        process_grasp_labels_fn=None,
        process_grasp_labels_kwargs=None,
        topview_debug_fn=None,
        depth_prob=None,
        objectness_logits=None,
        graspness_map=None,
        img=None,
    ) -> dict:
        if process_grasp_labels_fn is not None:
            raise RuntimeError("Ray-confidence model owns post-prediction zero-center label matching.")
        labels = end_points.pop("_rc_labels", None)
        B, _C, _H, W = feat_map.shape

        # Reproduce Stage-1 Top-1 path exactly and freeze it completely.
        with torch.no_grad():
            work = dict(end_points)
            work, residual = self.core._call_view_net(
                seed_features=seed_features,
                token_sel_idx=token_sel_idx,
                camera_K=camera_K,
                depth_map=depth_map,
                depth_prob=depth_prob,
                end_points=work,
            )
            seed_features_base = seed_features + residual
            seed_q, xyz_q, token_q, _view_rot, work = self.core.selector(
                seed_features=seed_features_base,
                seed_xyz=seed_xyz,
                token_sel_idx=token_sel_idx,
                view_score=work["view_score"],
                end_points=work,
                is_training=False,
                forced_view_inds=None,
            )
            view_xyz = work["grasp_top_view_xyz"].detach()
            view_inds = work["grasp_top_view_inds"].detach().long()
            Q = int(xyz_q.shape[1])
            if Q != int(seed_xyz.shape[1]):
                raise RuntimeError("Ray-confidence experiment requires one native Top-1 view per image-FPS ray.")
            centers, in_range, descriptor = build_ray_centers(
                xyz_q.detach(), token_q.detach(), camera_K, W,
                self.offsets_m, self.min_depth, self.max_depth,
            )

            groups = []
            for ki in range(self.offsets_m.numel()):
                local_ep = {}
                seed_a, xyz_a, token_a, rot_a, local_ep = self.core._expand_angle_queries(
                    seed_q, centers[:, :, ki], token_q, view_xyz, local_ep
                )
                group = self.core.group(
                    seed_features=seed_a,
                    token_sel_idx=token_a,
                    seed_xyz=xyz_a,
                    top_view_rot=rot_a,
                    feat_map=feat_map,
                    depth_map=depth_map,
                    objectness_logits=objectness_logits,
                    graspness_map=graspness_map,
                    camera_K=camera_K,
                    end_points=local_ep,
                )
                A = int(self.core.num_angle)
                if group.shape[-1] != Q * A:
                    raise RuntimeError("Frozen grouping changed Q*A ordering.")
                groups.append(
                    group.view(B, group.shape[1], Q, A).permute(0, 2, 3, 1).contiguous()
                )

            # Only the exact zero-center representation goes through the native
            # frozen CDF/width decoder. These outputs define the unchanged pose.
            zero_group = groups[self.zero_index]
            Cg = int(zero_group.shape[-1])
            A = int(zero_group.shape[2])
            flat_zero = zero_group.permute(0, 3, 1, 2).reshape(B, Cg, Q * A).contiguous()
            decoder_ep = {"kview_angle_query_base_q": Q, "kview_angle_query_num_angle": A}
            zero_decoded = self.core.decoder(flat_zero, decoder_ep)
            cdf = zero_decoded["grasp_cdf_pred_angle_depth"].detach()
            width = zero_decoded["grasp_width_pred_angle_depth"].detach()
            angle_idx, depth_idx, raw_score, _ = select_raw_operation(cdf)

        # Gather the local feature of the *already selected native angle* from
        # every evidence depth. This cannot change the selected grasp operation.
        group_stack = torch.stack(groups, dim=2).detach()  # B,Q,K,A,C
        Kdepth = int(group_stack.shape[2])
        Cg = int(group_stack.shape[-1])
        gather_idx = angle_idx[:, :, None, None, None].expand(B, Q, Kdepth, 1, Cg)
        selected_angle_features = group_stack.gather(3, gather_idx).squeeze(3)
        confidence_logit, gate, calibrated = self.confidence(
            selected_angle_features,
            descriptor,
            in_range,
            raw_score,
            angle_idx,
            depth_idx,
            num_angle=A,
            num_depth=int(cdf.shape[-1]),
            zero_index=self.zero_index,
        )

        # Restore the ordinary zero-center endpoint contract. Main decoding of
        # pose and width therefore remains identical to Stage-1.
        end_points.update(work)
        end_points["xyz_graspable"] = xyz_q
        end_points["token_sel_xyz"] = xyz_q
        end_points["token_sel_idx"] = token_q
        end_points["grasp_top_view_xyz"] = view_xyz
        end_points["grasp_top_view_inds"] = view_inds
        end_points["grasp_cdf_pred_angle_depth"] = cdf
        end_points["grasp_width_pred_angle_depth"] = width
        end_points["rc_offsets_m"] = self.offsets_m
        end_points["rc_evidence_centers"] = centers
        end_points["rc_in_range"] = in_range
        end_points["rc_raw_score"] = raw_score
        end_points["rc_angle_idx"] = angle_idx
        end_points["rc_depth_idx"] = depth_idx
        end_points["rc_confidence_logit"] = confidence_logit
        end_points["rc_confidence_gate"] = gate
        end_points["rc_calibrated_score"] = calibrated

        if labels is not None:
            label_ep, point_support, point_known = self._zero_labels(
                labels, xyz_q, view_inds, in_range[:, :, self.zero_index]
            )
            targets = build_confidence_targets(
                label_ep["batch_grasp_cdf_bins_angle_depth"].detach(),
                label_ep["batch_grasp_cdf_valid_mask"].detach(),
                point_support,
                point_known,
                angle_idx,
                depth_idx,
                raw_score,
                int(label_ep["batch_grasp_cdf_thresholds"].numel()),
            )
            end_points["rc_target_score"] = targets["target_score"]
            end_points["rc_target_known"] = targets["target_known"]
            end_points["rc_target_positive"] = targets["target_positive"]
            end_points["rc_ideal_gate"] = targets["ideal_gate"]
            end_points["rc_zero_point_support"] = point_support
            end_points["rc_zero_point_known"] = point_known
        return end_points


class economicgrasp_dpt_ray_confidence(economicgrasp_dpt_student):
    """Frozen Stage-1 RGB grasp model plus trainable ray-confidence head."""

    def __init__(
        self,
        *args: Any,
        rc_offsets_mm=DEFAULT_RC_OFFSETS_MM,
        rc_hidden: int = 128,
        rc_layers: int = 2,
        rc_heads: int = 4,
        rc_dropout: float = 0.10,
        rc_init_bias: float = 4.0,
        **kwargs: Any,
    ) -> None:
        kwargs.update(is_training=False, use_cdf=True, use_obs_depth=False, vis_dir=None)
        super().__init__(*args, **kwargs)
        original = self.kview_grasp_module
        self.kview_grasp_module = RayConfidenceEvidenceModule(
            original,
            offsets_mm=rc_offsets_mm,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            hidden_dim=rc_hidden,
            num_layers=rc_layers,
            num_heads=rc_heads,
            dropout=rc_dropout,
            init_bias=rc_init_bias,
        )
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.kview_grasp_module.confidence.parameters():
            param.requires_grad_(True)
        self.rc_hidden = int(rc_hidden)
        self.rc_layers = int(rc_layers)
        self.rc_heads = int(rc_heads)
        self.rc_dropout = float(rc_dropout)
        self.rc_init_bias = float(rc_init_bias)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.training = bool(mode)
        self.kview_grasp_module.train(mode)
        return self

    def load_stage1(self, state: Mapping[str, torch.Tensor]) -> None:
        mapped = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            if key.startswith("kview_grasp_module."):
                key = key.replace("kview_grasp_module.", "kview_grasp_module.core.", 1)
            mapped[key] = value
        result = self.load_state_dict(mapped, strict=False)
        allowed_missing = (
            "kview_grasp_module.confidence.",
            "kview_grasp_module.offsets_m",
            "rgb_geometry_diagnostics.",
        )
        missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
        unexpected = [key for key in result.unexpected_keys if not key.startswith("rgb_geometry_diagnostics.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Ray-confidence Stage-1 loading mismatch: missing={missing}, unexpected={unexpected}"
            )

    def forward(self, batch: dict, with_labels: bool = False):
        labels = None
        if with_labels:
            missing = [key for key in RC_LABEL_KEYS if key not in batch]
            if missing:
                raise KeyError(f"Ray-confidence training is missing CDF-cache fields: {missing}")
            labels = {key: batch[key] for key in RC_LABEL_KEYS}
        data = {
            key: value
            for key, value in batch.items()
            if key not in RC_LABEL_KEYS
            and not key.endswith("_list")
            and key not in (
                "gt_depth_m", "point_clouds", "cloud_colors", "coordinates_for_voxel",
                "sensor_depth_m", "obs_depth_m",
            )
        }
        for key in ("image_fps_seed_idx_override", "oracle_view_inds_override", "oracle_mode"):
            if key in data:
                raise RuntimeError(f"Ray-confidence experiment forbids override {key!r}.")
        data.update(
            cva_force_process_grasp_labels=False,
            cva_compute_diagnostics=False,
            geometry_compute_diagnostics=False,
            cva_export_angle_feature=False,
            _rc_labels=labels,
        )
        out = super().forward(data)
        pred = out.get("depth_net_pred", None)
        used = out.get("depth_map_used_for_geometry", None)
        if pred is None or used is None or pred.shape != used.shape:
            raise RuntimeError("Ray-confidence RGB geometry contract is incomplete.")
        if float((pred - used).abs().max().item()) > 1e-6:
            raise RuntimeError("Ray-confidence model must use exactly Stage-1 predicted metric depth.")
        return out


@torch.no_grad()
def decode_ray_confidence_grasps(end_points: Mapping[str, torch.Tensor], score_mode: str = "calibrated"):
    """Decode the native Stage-1 pose and optionally change only its score."""
    from models.economicgrasp_bip3d import pred_decode_center_view_angle

    if score_mode not in ("raw", "calibrated"):
        raise ValueError("score_mode must be raw or calibrated.")
    decoded = pred_decode_center_view_angle(dict(end_points), use_cdf=True)
    gate = end_points["rc_confidence_gate"].float()
    outputs = []
    for bi, grasp in enumerate(decoded):
        pred = grasp.clone()
        if pred.shape[0] != gate.shape[1]:
            raise RuntimeError(
                f"Native decoder returned {pred.shape[0]} grasps for Q={gate.shape[1]}; "
                "cannot guarantee score-only intervention."
            )
        if score_mode == "calibrated":
            # Preserve the decoder's exact native score semantics and apply only
            # the learned multiplicative confidence gate.
            pred[:, 0] = pred[:, 0] * gate[bi].to(pred)
        outputs.append(pred)
    return outputs
