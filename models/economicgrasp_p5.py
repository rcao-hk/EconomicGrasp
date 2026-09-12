"""P5: geometry-error-aware grasp repair with gripper-centric evidence.

P5 starts from the controlled Stage-1 RGB model.  Stage-1 remains frozen and
produces its native image-FPS query, Top-1 view, angle/depth operation, width and
score.  During training only, a structured corruption is applied to the
*predicted* dense metric geometry and the proposal center is rebuilt from that
corrupted depth.  A trainable repair module observes no clean depth: it receives
raw/pre-geometry visual evidence plus a frozen geometry-conditioned map rebuilt
from the corrupted depth.

Route 1: supervision repairs the corrupted proposal toward a nearby annotated
grasp center only when the exact frozen (view, angle, insertion-depth) operation
is positive there.  Queries without such evidence are unknown, not negatives.

Route 2: evidence is sampled at sparse gripper keypoints and then exchanged
among nearby image-FPS rays before predicting a 3-D residual in the grasp frame.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .economicgrasp_dpt_distill import economicgrasp_dpt_student
from utils.p5_ops import (
    backproject_token_depth,
    build_gripper_keypoints_local,
    build_repair_targets,
    gather_depth,
    project_keypoints,
    repair_metric_sums,
    sample_map_at_uv,
    select_native_operation,
    structured_depth_corruption,
    transform_keypoints,
)

P5_CONTRACT_VERSION = 1
# Method was designed against this controlled main codebase.  P5 checkpoint
# metadata also records the concrete branch base used at training time.
P5_CONTROLLED_MAIN_SHA = "3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2"

P5_LABEL_KEYS = (
    "object_poses_list",
    "grasp_points_list",
    "view_graspness_list",
    "top_view_index_list",
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
    "cdf_thresholds",
)


class P5GripperEvidenceEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 128, layers: int = 2,
                 heads: int = 4, dropout: float = 0.10, num_keypoints: int = 11):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.num_keypoints = int(num_keypoints)
        if self.hidden <= 0 or layers <= 0 or heads <= 0 or self.hidden % heads != 0:
            raise ValueError("P5 keypoint encoder hidden/layers/heads are invalid.")
        # RGB(3) + pre-geometry feature(C) + corrupted-geometry feature(C)
        # + signed depth residual(1) + gripper-local xyz(3) + visibility(1).
        in_dim = 2 * self.feature_dim + 8
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, self.hidden), nn.GELU()
        )
        self.role = nn.Embedding(self.num_keypoints, self.hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=int(heads), dim_feedforward=4 * self.hidden,
            dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.output = nn.Sequential(
            nn.LayerNorm(2 * self.hidden), nn.Linear(2 * self.hidden, self.hidden), nn.GELU()
        )

    def forward(self, keypoint_features: torch.Tensor, visible: torch.Tensor):
        if keypoint_features.dim() != 4:
            raise ValueError("P5 keypoint_features must be [B,Q,L,C].")
        B, Q, L, _ = keypoint_features.shape
        if L != self.num_keypoints or visible.shape != (B, Q, L):
            raise ValueError("P5 keypoint/visibility shape mismatch.")
        x = self.input_proj(keypoint_features)
        roles = self.role(torch.arange(L, device=x.device)).view(1, 1, L, self.hidden)
        x = (x + roles).reshape(B * Q, L, self.hidden)
        padding = (~visible.bool()).reshape(B * Q, L)
        # Center keypoint (0) is always numerically defined; make it available if
        # projection edge cases would otherwise mask all tokens.
        all_masked = padding.all(dim=-1)
        if bool(all_masked.any()):
            padding = padding.clone()
            padding[all_masked, 0] = False
        h = self.encoder(x, src_key_padding_mask=padding).reshape(B, Q, L, self.hidden)
        valid = (~padding).reshape(B, Q, L).to(h.dtype).unsqueeze(-1)
        pooled = (h * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        center = h[:, :, 0]
        return self.output(torch.cat((center, pooled), dim=-1))


class P5CrossRayContext(nn.Module):
    """Local cross-ray consistency over nearby image-FPS queries."""
    def __init__(self, hidden: int = 128, heads: int = 4, neighbors: int = 8,
                 dropout: float = 0.05):
        super().__init__()
        self.hidden = int(hidden)
        self.neighbors = int(neighbors)
        if self.hidden % int(heads) != 0 or self.neighbors <= 0:
            raise ValueError("P5 cross-ray heads/neighbors are invalid.")
        self.rel = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.attn = nn.MultiheadAttention(hidden, int(heads), dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, query_feature: torch.Tensor, uv_norm: torch.Tensor, z: torch.Tensor):
        if query_feature.dim() != 3 or uv_norm.shape != (*query_feature.shape[:2], 2):
            raise ValueError("P5 cross-ray query/uv shape mismatch.")
        B, Q, H = query_feature.shape
        if z.shape != (B, Q):
            raise ValueError("P5 cross-ray z shape mismatch.")
        if Q <= 1:
            return query_feature
        k = min(self.neighbors, Q - 1)
        # Image-plane locality defines the neighborhood; feature attention is
        # free to reject neighbors across object/appearance boundaries.
        dist = torch.cdist(uv_norm.float(), uv_norm.float(), p=2)
        eye = torch.eye(Q, device=dist.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        idx = torch.topk(dist, k=k, dim=-1, largest=False).indices  # B,Q,k
        b = torch.arange(B, device=idx.device)[:, None, None].expand(B, Q, k)
        neigh_h = query_feature[b, idx]
        neigh_uv = uv_norm[b, idx]
        neigh_z = z[b, idx]
        rel = torch.cat((neigh_uv - uv_norm.unsqueeze(2),
                         (neigh_z - z.unsqueeze(2)).unsqueeze(-1)), dim=-1)
        kv = neigh_h + self.rel(rel.to(neigh_h))
        q = query_feature.reshape(B * Q, 1, H)
        kv = kv.reshape(B * Q, k, H)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm(query_feature + out.reshape(B, Q, H))


class P5RepairHead(nn.Module):
    def __init__(self, hidden: int = 128, max_delta_m: float = 0.06):
        super().__init__()
        self.max_delta_m = float(max_delta_m)
        if self.max_delta_m <= 0:
            raise ValueError("P5 max_delta_m must be positive.")
        self.net = nn.Sequential(
            nn.LayerNorm(2 * hidden + 6),
            nn.Linear(2 * hidden + 6, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, evidence: torch.Tensor, context: torch.Tensor, scalars: torch.Tensor):
        raw = torch.tanh(self.net(torch.cat((evidence, context, scalars), dim=-1)))
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        magnitude = norm.clamp(max=1.0) * self.max_delta_m
        return raw / norm * magnitude


class economicgrasp_dpt_p5(economicgrasp_dpt_student):
    """Frozen Stage-1 grasp model plus trainable P5 center-repair module."""
    def __init__(
        self,
        *args: Any,
        p5_hidden: int = 128,
        p5_layers: int = 2,
        p5_heads: int = 4,
        p5_neighbors: int = 8,
        p5_dropout: float = 0.10,
        p5_max_delta_m: float = 0.06,
        p5_target_radius_m: float = 0.06,
        p5_corrupt_prob: float = 0.8,
        p5_scene_bias_sigma_m: float = 0.012,
        p5_scale_sigma: float = 0.025,
        p5_region_sigma_m: float = 0.015,
        p5_region_grid: int = 7,
        **kwargs: Any,
    ) -> None:
        kwargs.update(is_training=False, use_cdf=True, use_obs_depth=False, vis_dir=None)
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad_(False)

        self.p5_evidence = P5GripperEvidenceEncoder(
            feature_dim=self.seed_feature_dim, hidden=p5_hidden, layers=p5_layers,
            heads=p5_heads, dropout=p5_dropout,
        )
        self.p5_context = P5CrossRayContext(
            hidden=p5_hidden, heads=p5_heads, neighbors=p5_neighbors, dropout=p5_dropout * 0.5
        )
        self.p5_repair = P5RepairHead(hidden=p5_hidden, max_delta_m=p5_max_delta_m)
        for module in (self.p5_evidence, self.p5_context, self.p5_repair):
            module.requires_grad_(True)

        self.p5_hidden = int(p5_hidden)
        self.p5_layers = int(p5_layers)
        self.p5_heads = int(p5_heads)
        self.p5_neighbors = int(p5_neighbors)
        self.p5_dropout = float(p5_dropout)
        self.p5_max_delta_m = float(p5_max_delta_m)
        self.p5_target_radius_m = float(p5_target_radius_m)
        self.p5_corrupt_prob = float(p5_corrupt_prob)
        self.p5_scene_bias_sigma_m = float(p5_scene_bias_sigma_m)
        self.p5_scale_sigma = float(p5_scale_sigma)
        self.p5_region_sigma_m = float(p5_region_sigma_m)
        self.p5_region_grid = int(p5_region_grid)

        self._p5_proposal_capture = None
        def _capture(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("P5 proposal_head hook expected (feature, logits).")
            self._p5_proposal_capture = (output[0].detach(), output[1].detach())
        self._p5_hook_handle = self.proposal_head.register_forward_hook(_capture)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.training = bool(mode)
        self.p5_evidence.train(mode)
        self.p5_context.train(mode)
        self.p5_repair.train(mode)
        return self

    def load_stage1(self, state: Mapping[str, torch.Tensor]) -> None:
        state = {k.removeprefix("module."): v for k, v in state.items()}
        result = self.load_state_dict(state, strict=False)
        allowed_missing = ("p5_evidence.", "p5_context.", "p5_repair.", "rgb_geometry_diagnostics.")
        missing = [k for k in result.missing_keys if not k.startswith(allowed_missing)]
        unexpected = [k for k in result.unexpected_keys if not k.startswith("rgb_geometry_diagnostics.")]
        if missing or unexpected:
            raise RuntimeError(f"P5 Stage-1 loading mismatch: missing={missing}, unexpected={unexpected}")

    @staticmethod
    def _normalized_uv(token_idx: torch.Tensor, H: int, W: int, dtype) -> torch.Tensor:
        u = (token_idx % W).to(dtype) / max(float(W - 1), 1.0) * 2.0 - 1.0
        v = (token_idx // W).to(dtype) / max(float(H - 1), 1.0) * 2.0 - 1.0
        return torch.stack((u, v), dim=-1)

    def _geometry_conditioned_map(self, pre_feature: torch.Tensor, depth_map: torch.Tensor,
                                  K: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        H, W = img.shape[-2:]
        with torch.no_grad():
            enhanced, _ = self.spatial_enhancer(
                feat_2d=pre_feature,
                depth_prob=None,
                depth_map=depth_map,
                K=K,
                image_hw=(H, W),
                return_maps=False,
                img=img,
            )
            return F.interpolate(enhanced, size=(H, W), mode="bilinear", align_corners=False).detach()

    def forward(self, batch: dict, with_labels: bool = False):
        labels = None
        if with_labels:
            missing = [k for k in P5_LABEL_KEYS if k not in batch]
            if missing:
                raise KeyError(f"P5 training is missing CDF-cache fields: {missing}")
            labels = {k: batch[k] for k in P5_LABEL_KEYS}

        data = {
            k: v for k, v in batch.items()
            if k not in P5_LABEL_KEYS and not k.endswith("_list")
            and k not in ("gt_depth_m", "point_clouds", "cloud_colors", "coordinates_for_voxel",
                          "sensor_depth_m", "obs_depth_m")
        }
        for key in ("image_fps_seed_idx_override", "oracle_view_inds_override", "oracle_mode"):
            if key in data:
                raise RuntimeError(f"P5 forbids query/oracle override {key!r}.")
        data.update(cva_force_process_grasp_labels=False, cva_compute_diagnostics=False,
                    geometry_compute_diagnostics=False, cva_export_angle_feature=False)

        self._p5_proposal_capture = None
        with torch.no_grad():
            out = super().forward(data)
        if self._p5_proposal_capture is None:
            raise RuntimeError("P5 failed to capture the pre-geometry DPT proposal feature.")
        pre_feature, _proposal_logits = self._p5_proposal_capture

        depth_native = out.get("depth_net_pred", None)
        depth_used = out.get("depth_map_used_for_geometry", None)
        if depth_native is None or depth_used is None or depth_native.shape != depth_used.shape:
            raise RuntimeError("P5 Stage-1 depth contract is incomplete.")
        if float((depth_native - depth_used).abs().max().item()) > 1e-6:
            raise RuntimeError("P5 must initialize from the native predicted-depth Stage-1 path.")
        B, _, H, W = depth_native.shape
        Kcam = data["K"]

        if self.training:
            depth_evidence, corruption_diag = structured_depth_corruption(
                depth_native.detach(), min_depth=self.min_depth, max_depth=self.max_depth,
                probability=self.p5_corrupt_prob,
                scene_bias_sigma_m=self.p5_scene_bias_sigma_m,
                scale_sigma=self.p5_scale_sigma,
                region_sigma_m=self.p5_region_sigma_m,
                region_grid=self.p5_region_grid,
            )
        else:
            depth_evidence = depth_native.detach()
            zero = depth_native.new_zeros(())
            corruption_diag = {"active_ratio": zero, "abs_mean_m": zero,
                               "signed_mean_m": zero, "abs_max_m": zero}

        token_idx = out.get("kview_base_token_sel_idx", out.get("token_sel_idx", None))
        view_xyz = out.get("grasp_top_view_xyz", None)
        view_inds = out.get("grasp_top_view_inds", None)
        cdf = out.get("grasp_cdf_pred_angle_depth", None)
        width_pred = out.get("grasp_width_pred_angle_depth", None)
        if any(x is None for x in (token_idx, view_xyz, view_inds, cdf, width_pred)):
            raise RuntimeError("P5 Stage-1 output is missing native CVA tensors.")
        if token_idx.dim() != 2 or view_xyz.shape != (*token_idx.shape, 3):
            raise RuntimeError("P5 requires one native Top-1 view per image-FPS ray.")
        Q = token_idx.shape[1]

        z_evidence = gather_depth(depth_evidence, token_idx)
        proposal_center = backproject_token_depth(token_idx, z_evidence, Kcam, W)
        angle_idx, depth_idx, raw_score, width_m, insertion_m, _ = select_native_operation(cdf, width_pred)

        from utils.label_generation import batch_viewpoint_params_to_matrix
        angle_rad = angle_idx.float() * (math.pi / float(self.num_angle))
        rotation = batch_viewpoint_params_to_matrix(
            -view_xyz.reshape(-1, 3), angle_rad.reshape(-1)
        ).reshape(B, Q, 3, 3)

        pre_448 = F.interpolate(pre_feature, size=(H, W), mode="bilinear", align_corners=False).detach()
        geom_evidence = self._geometry_conditioned_map(pre_feature, depth_evidence, Kcam, data["img"])
        local_points = build_gripper_keypoints_local(width_m, insertion_m)
        world_points = transform_keypoints(proposal_center, rotation, local_points)
        uv, visible = project_keypoints(world_points, Kcam, H, W)
        rgb_samples = sample_map_at_uv(data["img"].detach(), uv)
        pre_samples = sample_map_at_uv(pre_448, uv)
        geom_samples = sample_map_at_uv(geom_evidence, uv)
        depth_samples = sample_map_at_uv(depth_evidence, uv).squeeze(-1)
        signed_depth_residual = (depth_samples - world_points[..., 2]).unsqueeze(-1)
        keypoint_features = torch.cat((
            rgb_samples.float(), pre_samples.float(), geom_samples.float(),
            signed_depth_residual.float(), (local_points / 0.10).float(),
            visible.float().unsqueeze(-1),
        ), dim=-1)
        evidence = self.p5_evidence(keypoint_features, visible)
        uv_norm = self._normalized_uv(token_idx, H, W, evidence.dtype)
        context = self.p5_context(evidence, uv_norm, proposal_center[..., 2])
        z_norm = 2.0 * (proposal_center[..., 2] - self.min_depth) / max(self.max_depth - self.min_depth, 1e-6) - 1.0
        scalars = torch.stack((raw_score.to(evidence), width_m.to(evidence), insertion_m.to(evidence),
                               z_norm.to(evidence), uv_norm[..., 0], uv_norm[..., 1]), dim=-1)
        delta_local = self.p5_repair(evidence, context, scalars)
        delta_camera = torch.matmul(rotation, delta_local.unsqueeze(-1)).squeeze(-1)
        repaired_center = proposal_center + delta_camera
        valid_z = (repaired_center[..., 2] > self.min_depth) & (repaired_center[..., 2] < self.max_depth)
        repaired_center = torch.where(valid_z.unsqueeze(-1), repaired_center, proposal_center)

        out["p5_proposal_center"] = proposal_center
        out["p5_repaired_center"] = repaired_center
        out["p5_delta_local"] = delta_local
        out["p5_rotation"] = rotation
        out["p5_native_angle_idx"] = angle_idx
        out["p5_native_depth_idx"] = depth_idx
        out["p5_native_score"] = raw_score
        out["p5_native_width_m"] = width_m
        out["p5_depth_evidence"] = depth_evidence
        out["p5_keypoint_visible_ratio"] = visible.float().mean().reshape(())
        for name, value in corruption_diag.items():
            out[f"P5: corruption {name}"] = value.reshape(())
        z_native = gather_depth(depth_native.detach(), token_idx)
        out["P5: query corruption abs m"] = (z_evidence - z_native).abs().mean().reshape(())

        if labels is not None:
            from utils.label_generation import process_grasp_labels_cdf_width
            ep = dict(labels)
            ep["xyz_graspable"] = proposal_center.detach()
            ep["grasp_top_view_inds"] = view_inds.detach().long()
            with torch.no_grad():
                _, ep = process_grasp_labels_cdf_width(ep)
            targets = build_repair_targets(
                proposal_center=proposal_center.detach(),
                rotation=rotation.detach(),
                nearest_label_center=ep["batch_grasp_point"].detach(),
                cdf_bins=ep["batch_grasp_cdf_bins_angle_depth"].detach(),
                angle_idx=angle_idx.detach(), depth_idx=depth_idx.detach(),
                num_thresholds=int(ep["batch_grasp_cdf_thresholds"].numel()),
                target_radius_m=self.p5_target_radius_m,
            )
            out["p5_target_delta_local"] = targets["target_delta_local"]
            out["p5_target_known"] = targets["target_known"]
            out["p5_target_distance_m"] = targets["target_distance_m"]
            out["p5_target_utility"] = targets["target_utility"]
            out["p5_same_action_positive"] = targets["same_action_positive"]
            # Precompute metrics here for easy smoke inspection; trainer also
            # aggregates exact sufficient statistics through p5_ops.
            metrics = repair_metric_sums(delta_local.detach(), rotation.detach(),
                                          proposal_center.detach(), targets)
            for name, (total, count) in metrics.items():
                out[f"D: {name}"] = total / count.clamp_min(1.0)
        return out


@torch.no_grad()
def decode_p5_grasps(end_points: Mapping[str, torch.Tensor], use_repair: bool = True):
    """Keep Stage-1 score/view/angle/depth/width; replace translation only."""
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    decoded = pred_decode_center_view_angle(dict(end_points), use_cdf=True)
    repaired = end_points["p5_repaired_center"]
    proposal = end_points["p5_proposal_center"]
    centers = repaired if use_repair else proposal
    outputs = []
    for bi, grasp in enumerate(decoded):
        pred = grasp.clone()
        if pred.shape[0] != centers.shape[1]:
            raise RuntimeError(
                f"P5 decoder returned {pred.shape[0]} grasps for Q={centers.shape[1]}; "
                "translation-only intervention cannot be guaranteed."
            )
        # GraspNet representation: score,width,height,depth,R(9),translation(3),obj.
        pred[:, 13:16] = centers[bi].to(pred)
        outputs.append(pred)
    return outputs
