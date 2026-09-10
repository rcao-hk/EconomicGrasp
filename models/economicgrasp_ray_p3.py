"""P3: selection-free multi-depth evidence aggregation for RGB-only grasping.

The model starts from the controlled Stage-1 RGB checkpoint on main.  It keeps
K physical center hypotheses alive through local grasp representation formation,
aggregates their evidence with cross-depth self-attention, and predicts one
joint grasp field over K x angle x insertion-depth.  There is no learned or
hard depth selector before the grasp head; a metric center is chosen only at the
unavoidable final grasp decode.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .economicgrasp_dpt_distill import economicgrasp_dpt_student
from utils.p3_ray_ops import DEFAULT_P3_OFFSETS_MM, build_ray_centers, parse_offsets

P3_CONTRACT_VERSION = 1
P3_BASE_MAIN_SHA = "3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2"

P3_LABEL_KEYS = (
    "object_poses_list",
    "grasp_points_list",
    "view_graspness_list",
    "top_view_index_list",
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
    "cdf_thresholds",
)


class P3CrossDepthEvidenceAggregator(nn.Module):
    """Residual evidence exchange across K depths of the same ray and angle.

    Input local grasp features already depend on each physical center through
    the original metric grouping.  The Transformer sees only K hypotheses for
    one (image ray, in-plane angle), never different image rays.  The residual
    output is zero-initialized so a fresh P3 model starts from the independent
    Stage-1 representation rather than perturbing it randomly.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0 or int(num_layers) <= 0 or int(num_heads) <= 0:
            raise ValueError("P3 hidden/layers/heads must be positive.")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("P3 hidden_dim must be divisible by num_heads.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("P3 dropout must lie in [0,1).")

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim + 5),
            nn.Linear(self.feature_dim + 5, self.hidden_dim),
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
        self.delta_norm = nn.LayerNorm(self.hidden_dim)
        self.delta_head = nn.Linear(self.hidden_dim, self.feature_dim)
        self.viability_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.viability_head[-1].weight)
        nn.init.zeros_(self.viability_head[-1].bias)

    def forward(
        self,
        group_features: torch.Tensor,  # [B,Q,K,A,C]
        ray_descriptor: torch.Tensor,  # [B,Q,K,5]
        in_range: torch.Tensor,        # [B,Q,K]
    ):
        if group_features.dim() != 5:
            raise ValueError("P3 group_features must be [B,Q,K,A,C].")
        B, Q, K, A, C = group_features.shape
        if C != self.feature_dim:
            raise ValueError(f"P3 feature dim {C} != configured {self.feature_dim}.")
        if ray_descriptor.shape != (B, Q, K, 5) or in_range.shape != (B, Q, K):
            raise ValueError("P3 descriptor/range shape mismatch.")
        if not bool(in_range.any(dim=-1).all()):
            raise RuntimeError("Each P3 ray must retain at least one in-range center.")

        # One Transformer sequence per (batch, image ray, in-plane angle).
        local = group_features.permute(0, 1, 3, 2, 4).contiguous()  # B,Q,A,K,C
        desc = ray_descriptor.unsqueeze(2).expand(B, Q, A, K, 5)
        x = torch.cat((local, desc.to(local)), dim=-1)
        x = self.input_proj(x).reshape(B * Q * A, K, self.hidden_dim)
        padding = (~in_range).unsqueeze(2).expand(B, Q, A, K).reshape(B * Q * A, K)
        hidden = self.encoder(x, src_key_padding_mask=padding)
        hidden = hidden.reshape(B, Q, A, K, self.hidden_dim)

        delta = self.delta_head(self.delta_norm(hidden))
        contextual = local + delta
        contextual = contextual.permute(0, 1, 3, 2, 4).contiguous()  # B,Q,K,A,C

        # Viability is a contextual property of one physical center.  Average
        # the angle-specific evidence only after cross-depth interaction.
        viability_hidden = hidden.mean(dim=2)  # B,Q,K,H
        viability = self.viability_head(viability_hidden).squeeze(-1)
        viability = viability.masked_fill(~in_range, -20.0)

        with torch.no_grad():
            delta_norm = delta.float().norm(dim=-1).mean()
        return contextual, viability, delta_norm


class P3RayEvidenceModule(nn.Module):
    """Wrap the main CVA module without using its independent per-depth forward.

    View prediction is executed once at the native image-FPS query.  The same
    predicted approach view is then used for every physical center on that ray,
    isolating center-depth representation.  Local grouping is evaluated at all
    K centers under no-grad because the complete Stage-1 model is frozen.
    Cross-depth aggregation happens before the frozen CDF/width decoder.
    """

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
    ) -> None:
        super().__init__()
        self.core = core
        self.core.requires_grad_(False)
        self.core.eval()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        offsets = torch.tensor(parse_offsets(offsets_mm), dtype=torch.float32) * 1.0e-3
        self.register_buffer("offsets_m", offsets)
        feature_dim = int(self.core.config.head_model_dim)
        self.aggregator = P3CrossDepthEvidenceAggregator(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def train(self, mode: bool = True):
        # Never reactivate dropout/BN in the frozen Stage-1 representation.
        super().train(False)
        self.training = bool(mode)
        self.core.eval()
        self.aggregator.train(mode)
        return self

    @staticmethod
    def _label_match(labels: Mapping[str, Any], center: torch.Tensor, view_inds: torch.Tensor,
                     in_range: torch.Tensor):
        from utils.label_generation import process_grasp_labels_cdf_width
        ep = dict(labels)
        ep["xyz_graspable"] = center
        ep["grasp_top_view_inds"] = view_inds
        with torch.no_grad():
            _, ep = process_grasp_labels_cdf_width(ep)
            nearest = ep["batch_grasp_point"]
            distance = (center - nearest).float().norm(dim=-1)
            known = torch.isfinite(distance) & in_range
            support = (distance < 0.005) & known
        return {
            "cdf_bins": ep["batch_grasp_cdf_bins_angle_depth"].detach(),
            "cdf_valid": ep["batch_grasp_cdf_valid_mask"].detach() & in_range[..., None, None],
            "width_label": ep["batch_grasp_width_angle_depth"].detach(),
            "width_valid": ep["batch_grasp_width_valid_mask_angle_depth"].detach() & in_range[..., None, None],
            "point_support": support.detach(),
            "point_known": known.detach(),
        }

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
            raise RuntimeError("P3 owns its post-prediction label matching; outer label processing must be disabled.")
        labels = end_points.pop("_p3_labels", None)
        B, _C, H, W = feat_map.shape

        # Stage-1 view field + Top-1 selector are frozen and executed once.
        with torch.no_grad():
            work = dict(end_points)
            work, res_feat = self.core._call_view_net(
                seed_features=seed_features,
                token_sel_idx=token_sel_idx,
                camera_K=camera_K,
                depth_map=depth_map,
                depth_prob=depth_prob,
                end_points=work,
            )
            seed_features_base = seed_features + res_feat
            seed_features_q, seed_xyz_q, token_idx_q, _view_rot_q, work = self.core.selector(
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
            Q = int(seed_xyz_q.shape[1])
            if Q != int(seed_xyz.shape[1]):
                raise RuntimeError(
                    "P3-v1 fixes one native Top-1 view per image-FPS ray; "
                    f"selector changed Q from {seed_xyz.shape[1]} to {Q}."
                )
            centers, in_range, descriptor = build_ray_centers(
                seed_xyz_q.detach(), token_idx_q.detach(), camera_K,
                W, self.offsets_m, self.min_depth, self.max_depth,
            )

            groups = []
            label_parts = []
            for ki in range(self.offsets_m.numel()):
                local_ep = {}
                seed_a, xyz_a, token_a, rot_a, local_ep = self.core._expand_angle_queries(
                    seed_features_q, centers[:, :, ki], token_idx_q, view_xyz, local_ep
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
                    raise RuntimeError("P3 grouping changed Q*A ordering.")
                groups.append(
                    group.view(B, group.shape[1], Q, A).permute(0, 2, 3, 1).contiguous()
                )
                if labels is not None:
                    label_parts.append(self._label_match(
                        labels, centers[:, :, ki], view_inds, in_range[:, :, ki]
                    ))

        group_stack = torch.stack(groups, dim=2)  # B,Q,K,A,C
        contextual, viability, delta_norm = self.aggregator(
            group_stack.detach(), descriptor.detach(), in_range
        )
        Kdepth = int(self.offsets_m.numel())
        A = int(self.core.num_angle)
        Cg = int(contextual.shape[-1])
        flat_group = contextual.permute(0, 4, 1, 2, 3).reshape(B, Cg, Q * Kdepth * A).contiguous()
        decoder_ep = {
            "kview_angle_query_base_q": int(Q * Kdepth),
            "kview_angle_query_num_angle": int(A),
        }
        # Frozen decoder remains differentiable wrt the contextual representation.
        decoded = self.core.decoder(flat_group, decoder_ep)
        cdf_flat = decoded["grasp_cdf_pred_angle_depth"]      # B,T,QK,A,D
        width_flat = decoded["grasp_width_pred_angle_depth"] # B,D,QK,A
        T = int(cdf_flat.shape[1])
        D = int(cdf_flat.shape[-1])
        cdf = cdf_flat.reshape(B, T, Q, Kdepth, A, D).contiguous()
        width = width_flat.reshape(B, D, Q, Kdepth, A).contiguous()

        end_points.update(work)
        end_points["p3_centers"] = centers
        end_points["p3_in_range"] = in_range
        end_points["p3_ray_descriptor"] = descriptor
        end_points["p3_offsets_m"] = self.offsets_m
        end_points["p3_view_xyz"] = view_xyz
        end_points["p3_view_inds"] = view_inds
        end_points["p3_token_idx"] = token_idx_q
        end_points["p3_cdf_logits"] = cdf
        end_points["p3_width_pred"] = width
        end_points["p3_viability_logits"] = viability
        end_points["D: P3 aggregation delta norm"] = delta_norm.reshape(())
        end_points["D: P3 K"] = cdf.new_tensor(float(Kdepth)).reshape(())

        # Keep the main outer CDF contract structurally valid.  P3 inference
        # must use decode_p3_grasps(), because the true candidate axis is KxAxD.
        flat_centers = centers.reshape(B, Q * Kdepth, 3)
        flat_tokens = token_idx_q.unsqueeze(-1).expand(B, Q, Kdepth).reshape(B, Q * Kdepth)
        flat_views = view_xyz.unsqueeze(2).expand(B, Q, Kdepth, 3).reshape(B, Q * Kdepth, 3)
        end_points["xyz_graspable"] = flat_centers
        end_points["token_sel_xyz"] = flat_centers
        end_points["token_sel_idx"] = flat_tokens
        end_points["grasp_top_view_xyz"] = flat_views
        end_points["grasp_top_view_inds"] = view_inds.unsqueeze(-1).expand(B, Q, Kdepth).reshape(B, Q * Kdepth)
        end_points["grasp_cdf_pred_angle_depth"] = cdf_flat
        end_points["grasp_width_pred_angle_depth"] = width_flat

        if labels is not None:
            if len(label_parts) != Kdepth:
                raise RuntimeError("P3 label stack count mismatch.")
            end_points["p3_cdf_bins"] = torch.stack([x["cdf_bins"] for x in label_parts], dim=2)
            end_points["p3_cdf_valid"] = torch.stack([x["cdf_valid"] for x in label_parts], dim=2)
            end_points["p3_width_label"] = torch.stack([x["width_label"] for x in label_parts], dim=2)
            end_points["p3_width_valid"] = torch.stack([x["width_valid"] for x in label_parts], dim=2)
            end_points["p3_point_support"] = torch.stack([x["point_support"] for x in label_parts], dim=2)
            end_points["p3_point_known"] = torch.stack([x["point_known"] for x in label_parts], dim=2)
        return end_points


class economicgrasp_dpt_p3_ray(economicgrasp_dpt_student):
    """Controlled P3 model: frozen Stage-1 + trainable cross-depth aggregator."""

    def __init__(
        self,
        *args: Any,
        p3_offsets_mm=DEFAULT_P3_OFFSETS_MM,
        p3_hidden: int = 128,
        p3_layers: int = 2,
        p3_heads: int = 4,
        p3_dropout: float = 0.10,
        **kwargs: Any,
    ) -> None:
        kwargs.update(is_training=False, use_cdf=True, use_obs_depth=False, vis_dir=None)
        super().__init__(*args, **kwargs)
        original = self.kview_grasp_module
        self.kview_grasp_module = P3RayEvidenceModule(
            original,
            offsets_mm=p3_offsets_mm,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            hidden_dim=p3_hidden,
            num_layers=p3_layers,
            num_heads=p3_heads,
            dropout=p3_dropout,
        )
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.kview_grasp_module.aggregator.parameters():
            param.requires_grad_(True)
        self.p3_hidden = int(p3_hidden)
        self.p3_layers = int(p3_layers)
        self.p3_heads = int(p3_heads)
        self.p3_dropout = float(p3_dropout)
        self.train(False)

    def train(self, mode: bool = True):
        # Parent call in eval mode prevents frozen DPT/ViewNet/CVA dropout drift.
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
            "kview_grasp_module.aggregator.",
            "kview_grasp_module.offsets_m",
            "rgb_geometry_diagnostics.",
        )
        missing = [k for k in result.missing_keys if not k.startswith(allowed_missing)]
        unexpected = [k for k in result.unexpected_keys if not k.startswith("rgb_geometry_diagnostics.")]
        if missing or unexpected:
            raise RuntimeError(
                f"P3 Stage-1 loading mismatch: missing={missing}, unexpected={unexpected}"
            )

    def forward(self, batch: dict, with_labels: bool = False):
        labels = None
        if with_labels:
            missing = [key for key in P3_LABEL_KEYS if key not in batch]
            if missing:
                raise KeyError(f"P3 training is missing CDF-cache fields: {missing}")
            labels = {key: batch[key] for key in P3_LABEL_KEYS}

        # No GT geometry or captured point cloud enters the network path.
        data = {
            key: value
            for key, value in batch.items()
            if key not in P3_LABEL_KEYS
            and not key.endswith("_list")
            and key not in (
                "gt_depth_m", "point_clouds", "cloud_colors", "coordinates_for_voxel",
                "sensor_depth_m", "obs_depth_m",
            )
        }
        for key in ("image_fps_seed_idx_override", "oracle_view_inds_override", "oracle_mode"):
            if key in data:
                raise RuntimeError(f"P3 forbids query/oracle override {key!r}.")
        data.update(
            cva_force_process_grasp_labels=False,
            cva_compute_diagnostics=False,
            geometry_compute_diagnostics=False,
            cva_export_angle_feature=False,
            _p3_labels=labels,
        )
        out = super().forward(data)
        pred = out.get("depth_net_pred", None)
        used = out.get("depth_map_used_for_geometry", None)
        if pred is None or used is None or pred.shape != used.shape:
            raise RuntimeError("P3 RGB geometry contract is incomplete.")
        if float((pred - used).abs().max().item()) > 1e-6:
            raise RuntimeError("P3 geometry must be exactly the Stage-1 predicted metric depth.")
        return out


@torch.no_grad()
def decode_p3_grasps(
    end_points: Mapping[str, torch.Tensor],
    *,
    selection_score: str = "joint",
    final_score: str = "same",
    force_zero: bool = False,
):
    """Decode exactly one grasp per native image-FPS ray.

    Cross-depth evidence has already formed the K-conditioned grasp field.  This
    function performs only final task decode.  `selection_score` chooses the
    KxAxD operation by raw CDF or viability-gated joint utility; `final_score`
    optionally keeps raw CDF for cross-ray ranking as a diagnostic control.
    """
    from utils.label_generation import batch_viewpoint_params_to_matrix
    from utils.p3_ray_ops import final_indices, predicted_joint_utility, predicted_raw_utility

    k, angle_idx, depth_idx, selected_score, _ = final_indices(
        end_points, score_mode=selection_score, force_zero=force_zero
    )
    raw_utility = predicted_raw_utility(end_points["p3_cdf_logits"])
    joint_utility = predicted_joint_utility(
        end_points["p3_cdf_logits"], end_points["p3_viability_logits"]
    )
    B, Q = k.shape
    A = int(raw_utility.shape[-2])
    D = int(raw_utility.shape[-1])
    centers = end_points["p3_centers"].float()
    views = end_points["p3_view_xyz"].float()
    width_qkad = end_points["p3_width_pred"].float().permute(0, 2, 3, 4, 1).contiguous()
    outputs = []
    for bi in range(B):
        row = torch.arange(Q, device=k.device)
        kk = k[bi]
        aa = angle_idx[bi]
        dd = depth_idx[bi]
        center = centers[bi, row, kk]
        width = width_qkad[bi, row, kk, aa, dd].unsqueeze(-1)
        raw_score = raw_utility[bi, row, kk, aa, dd].unsqueeze(-1)
        joint_score = joint_utility[bi, row, kk, aa, dd].unsqueeze(-1)
        if final_score == "same":
            score = selected_score[bi].unsqueeze(-1)
        elif final_score == "raw":
            score = raw_score
        elif final_score == "joint":
            score = joint_score
        else:
            raise ValueError("P3 final_score must be same/raw/joint.")
        angle = aa.float() * (np.pi / float(A))
        insertion = (dd.float().unsqueeze(-1) + 1.0) * 0.01
        approach = -views[bi]
        rot = batch_viewpoint_params_to_matrix(approach, angle).reshape(Q, 9)
        width = torch.clamp(1.2 * width / 10.0, min=0.0, max=0.1)
        height = torch.full_like(score, 0.02)
        obj = torch.full_like(score, -1.0)
        pred = torch.cat((score, width, height, insertion, rot, center, obj), dim=-1)
        outputs.append(pred[score.squeeze(-1) >= 0])
    return outputs, k
