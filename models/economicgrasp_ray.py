"""Discrete ray-conditioned CVA-CDF field, based on main 3f3c08d.

Frozen RGB frontend; fixed metric z offsets; one shared, trainable CVA decoder.
K ray depths and D gripper insertion depths are different axes. No teacher or
online analytic evaluator is used. This first version does not train dense depth
or add cross-ray attention. See experiments/P2_RAY_GRASP.md for label semantics.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .economicgrasp_dpt_distill import economicgrasp_dpt_student
from .ray_grasp_ops import DEFAULT_OFFSETS_MM, parse_offsets, ray_hypotheses, select_hypotheses

RAY_CONTRACT_VERSION = 1
MAIN_BASE_SHA = "3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2"
LABEL_KEYS = (
    "object_poses_list", "grasp_points_list", "view_graspness_list", "top_view_index_list",
    "grasp_cdf_bins_list", "grasp_widths_depth_list", "grasp_width_valids_depth_list", "cdf_thresholds",
)
PART_KEYS = (
    "xyz_graspable", "token_sel_xyz", "token_sel_idx", "grasp_top_view_xyz", "grasp_top_view_inds",
    "kview_query_parent", "kview_query_view_rank", "kview_effective_k_int",
    "grasp_cdf_pred_angle_depth", "grasp_width_pred_angle_depth",
    "ray_support_logits", "ray_in_range", "ray_offset_m", "ray_descriptor", "ray_context_feature",
    "batch_grasp_point", "batch_grasp_cdf_bins_angle_depth", "batch_grasp_cdf_valid_mask",
    "batch_grasp_cdf_thresholds", "batch_grasp_width_angle_depth",
    "batch_grasp_width_valid_mask_angle_depth", "ray_support_target", "ray_support_known",
)


class RayDecoder(nn.Module):
    """Offset conditioning after frozen center-conditioned CVA grouping."""
    def __init__(self, base: nn.Module, hidden: int = 128, checkpoint_decoder: bool = True):
        super().__init__()
        self.base = base
        channels = int(base.input_proj.in_channels)
        self.ray_embed = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.ray_support = nn.Sequential(nn.LayerNorm(channels + 5), nn.Linear(channels + 5, hidden),
                                         nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.ray_embed[-1].weight)
        nn.init.zeros_(self.ray_embed[-1].bias)
        self.checkpoint_decoder = bool(checkpoint_decoder)

    def _predict(self, group: torch.Tensor, descriptor: torch.Tensor):
        B, C, QA = group.shape
        Q, A = descriptor.shape[1], int(self.base.num_angle)
        if descriptor.shape != (B, Q, 5) or QA != Q * A:
            raise RuntimeError("Ray decoder requires base-major angle order [B,C,M*A].")
        group = group.detach()  # first experiment deliberately freezes all grouping gradients
        embedding = self.ray_embed(descriptor).transpose(1, 2).unsqueeze(-1)
        conditioned = (group.reshape(B, C, Q, A) + embedding).reshape(B, C, QA)
        support_input = torch.cat((group.reshape(B, C, Q, A).mean(-1).transpose(1, 2), descriptor), -1)
        support = self.ray_support(support_input).squeeze(-1)
        out = self.base(conditioned, {"kview_angle_query_base_q": Q, "kview_angle_query_num_angle": A})
        return out["grasp_cdf_pred_angle_depth"], out["grasp_width_pred_angle_depth"], support

    def forward(self, group_features_angle, end_points):
        descriptor = end_points["ray_descriptor"].to(group_features_angle)
        B, C, QA = group_features_angle.shape
        Q, A = descriptor.shape[1], int(self.base.num_angle)
        if QA != Q * A:
            raise RuntimeError("Ray context extraction requires base-major [Q*A] ordering.")
        # Additive endpoint only: no v1 parameter or prediction is changed. P2-v2
        # consumes this frozen local feature after all K hypotheses are evaluated.
        end_points["ray_context_feature"] = (
            group_features_angle.detach().reshape(B, C, Q, A).mean(-1).transpose(1, 2).contiguous()
        )  # [B,Q,C]
        end_points["ray_descriptor"] = descriptor.detach()
        if self.training and self.checkpoint_decoder and torch.is_grad_enabled():
            cdf, width, support = checkpoint(self._predict, group_features_angle.detach(), descriptor,
                                              use_reentrant=False, preserve_rng_state=True)
        else:
            cdf, width, support = self._predict(group_features_angle, descriptor)
        end_points.update(grasp_cdf_pred_angle_depth=cdf, grasp_width_pred_angle_depth=width,
                          ray_support_logits=support)
        return end_points


class RaySweep(nn.Module):
    """Evaluate all fixed hypotheses; image backbone is outside this module."""
    def __init__(self, core, offsets_mm, min_depth, max_depth, hidden=128, checkpoint_decoder=True):
        super().__init__()
        self.core = core
        self.core.decoder = RayDecoder(core.decoder, hidden, checkpoint_decoder)
        self.register_buffer("offsets_m", torch.tensor(parse_offsets(offsets_mm), dtype=torch.float32) * .001)
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        self.core.config.vis_dir = None

    def train(self, mode=True):
        super().train(False)
        self.training = bool(mode)
        self.core.decoder.train(mode)
        return self

    def forward(self, *, seed_features, seed_xyz, token_sel_idx, feat_map, depth_map,
                camera_K, end_points, **kwargs):
        labels = end_points.pop("_ray_labels", None)
        centers, in_range, desc = ray_hypotheses(seed_xyz, token_sel_idx, camera_K,
                                                depth_map.shape[-1], self.offsets_m,
                                                self.min_depth, self.max_depth)
        native = dict(end_points)
        native.update(cva_compute_diagnostics=False, cva_export_angle_feature=False,
                      cva_force_process_grasp_labels=False)
        parts, reference_view = [], None
        for ki in range(self.offsets_m.numel()):
            # Fresh endpoint dictionary: never reuse stale zero-center labels.
            p = dict(native)
            p.update(xyz_graspable=centers[:, :, ki], token_sel_xyz=centers[:, :, ki],
                     token_sel_idx=token_sel_idx, ray_descriptor=desc[:, :, ki],
                     ray_in_range=in_range[:, :, ki], ray_offset_m=self.offsets_m[ki])
            p = self.core(seed_features=seed_features.detach(), seed_xyz=centers[:, :, ki],
                          token_sel_idx=token_sel_idx, feat_map=feat_map.detach(),
                          depth_map=depth_map.detach(), camera_K=camera_K, end_points=p,
                          is_training=False, process_grasp_labels_fn=None,
                          process_grasp_labels_kwargs=None, topview_debug_fn=None,
                          depth_prob=kwargs.get("depth_prob"),
                          objectness_logits=kwargs.get("objectness_logits"),
                          graspness_map=kwargs.get("graspness_map"), img=None)
            if not torch.equal(p["token_sel_idx"], token_sel_idx):
                raise RuntimeError("Ray sweep changed ordered image-FPS pixels.")
            if not torch.allclose(p["xyz_graspable"], centers[:, :, ki], atol=1e-6, rtol=0):
                raise RuntimeError("CVA did not consume the requested ray-depth center.")
            if int(p["kview_effective_k_int"]) != 1:
                raise RuntimeError("P2 v1 holds one native Top-1 view per ray; disable Top-4.")
            if reference_view is None:
                reference_view = p["grasp_top_view_inds"].detach().clone()
            elif not torch.equal(reference_view, p["grasp_top_view_inds"]):
                raise RuntimeError("Frozen view selection changed across ray depths.")
            if labels is not None:
                from utils.label_generation import process_grasp_labels_cdf_width
                p.update(labels)
                with torch.no_grad():
                    _, p = process_grasp_labels_cdf_width(p)
                    distance = (p["xyz_graspable"] - p["batch_grasp_point"]).norm(dim=-1)
                    p["ray_support_known"] = torch.isfinite(distance)
                    p["ray_support_target"] = distance < .005
            parts.append({key: p[key] for key in PART_KEYS if key in p})
        # Keep zero-ray conventional keys only for the inherited output contract.
        # The custom loss/decode always consumes ray_hypotheses, not these aliases.
        zero = int((self.offsets_m == 0).nonzero()[0])
        end_points.update(parts[zero])
        end_points["ray_hypotheses"] = parts
        end_points["ray_offsets_m"] = self.offsets_m
        end_points["ray_base_pixel_idx"] = token_sel_idx
        end_points["ray_base_xyz"] = seed_xyz
        return end_points


class RayConditionedGrasp(economicgrasp_dpt_student):
    def __init__(self, *args: Any, ray_offsets_mm=DEFAULT_OFFSETS_MM, ray_hidden=128,
                 ray_checkpoint_decoder=True, **kwargs: Any):
        kwargs.update(is_training=False, use_cdf=True, use_obs_depth=False,
                      vis_dir=None)
        super().__init__(*args, **kwargs)
        self.kview_grasp_module = RaySweep(self.kview_grasp_module, ray_offsets_mm,
                                           self.min_depth, self.max_depth,
                                           ray_hidden, ray_checkpoint_decoder)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.kview_grasp_module.core.decoder.parameters():
            p.requires_grad_(True)
        self.ray_hidden = int(ray_hidden)
        self.ray_checkpoint_decoder = bool(ray_checkpoint_decoder)
        self.train(False)

    def train(self, mode=True):
        # requires_grad=False alone does not freeze BN statistics or dropout.
        super().train(False)
        self.training = bool(mode)
        self.kview_grasp_module.train(mode)
        return self

    def forward(self, batch, with_labels=False):
        labels = None
        if with_labels:
            missing = [k for k in LABEL_KEYS if k not in batch]
            if missing:
                raise KeyError(f"Ray training is missing CDF cache fields: {missing}")
            labels = {k: batch[k] for k in LABEL_KEYS}
        # Privileged tensors cannot reach the RGB frontend/view/group/decoder.
        # Labels are inserted ONLY after each hypothesis's network predictions.
        data = {k: v for k, v in batch.items() if k not in LABEL_KEYS and not k.endswith("_list")
                and k not in ("gt_depth_m", "point_clouds", "cloud_colors", "coordinates_for_voxel",
                              "sensor_depth_m", "obs_depth_m")}
        for key in ("image_fps_seed_idx_override", "kview_forced_view_inds", "oracle_mode"):
            if key in data:
                raise RuntimeError(f"Unexpected oracle/query override: {key}")
        data.update(cva_force_process_grasp_labels=False, cva_compute_diagnostics=False,
                    geometry_compute_diagnostics=False, cva_export_angle_feature=False,
                    _ray_labels=labels)
        out = super().forward(data)
        predicted = out["depth_net_pred"]
        used = out["depth_map_used_for_geometry"]
        if predicted.shape != used.shape or not torch.allclose(predicted, used, atol=1e-6, rtol=0):
            raise RuntimeError("Ray frontend must retain the native RGB-predicted dense geometry.")
        return out

    def load_stage1(self, state):
        mapped = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            if key.startswith("kview_grasp_module.decoder."):
                key = key.replace("kview_grasp_module.decoder.", "kview_grasp_module.core.decoder.base.", 1)
            elif key.startswith("kview_grasp_module."):
                key = key.replace("kview_grasp_module.", "kview_grasp_module.core.", 1)
            mapped[key] = value
        result = self.load_state_dict(mapped, strict=False)
        allowed = ("kview_grasp_module.core.decoder.ray_", "kview_grasp_module.offsets_m",
                   "rgb_geometry_diagnostics.")
        missing = [k for k in result.missing_keys if not k.startswith(allowed)]
        unexpected = [k for k in result.unexpected_keys if not k.startswith("rgb_geometry_diagnostics.")]
        if missing or unexpected:
            raise RuntimeError(f"Stage-1 loading mismatch: missing={missing}, unexpected={unexpected}")


@torch.no_grad()
def decode_ray_grasps(end_points, supported=True, selection="best"):
    """One grasp per base ray: joint k/angle/insertion-depth selection.

    Delegates pose/width conventions (including main's 1.2 width multiplier)
    to main's decoder. Candidate budget stays M rather than inflating to M*K.
    """
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    parts = end_points["ray_hypotheses"]
    k, score, _ = select_hypotheses(parts, end_points["ray_offsets_m"], supported, selection)
    decoded = [pred_decode_center_view_angle(p, use_cdf=True) for p in parts]
    outputs = []
    for bi in range(k.shape[0]):
        all_grasps = torch.stack([g[bi] for g in decoded], dim=1)  # M,K,17
        if all_grasps.shape[:2] != (k.shape[1], len(parts)):
            raise RuntimeError("Main decoder changed per-ray candidate ordering/count.")
        row = torch.arange(k.shape[1], device=k.device)
        pred = all_grasps[row, k[bi]].clone()
        pred[:, 0] = score[bi]
        outputs.append(pred[score[bi] >= 0])
    return outputs, k
