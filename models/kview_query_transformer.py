"""
Complete K-view query transformer local grasp module for economicgrasp_dpt.

Purpose
-------
This module replaces the old single-view local analysis path:

    GeometryAwareDenseFieldViewNet
      -> MetricRegionCropGrouping / 2D ROI local analysis
      -> Grasp_Head_Local_Interaction_Dropout

with:

    GeometryAwareDenseFieldViewNet
      -> A0/A1/A2/A3 K-view query selection
      -> view-conditioned cross-attention grouping over local 2D/DPT features
      -> within-center K-view transformer grasp head with output-head dropout

A0-A3 modes
-----------
A0: train top1,        infer top1
A1: train multinomial, infer top1
A2: train multinomial, infer topK
A3: train multi-view,  infer topK

The final prediction tensors remain compatible with the existing EconomicGrasp
loss/decode interface:

    grasp_angle_pred: [B, num_angle + 1, Q]
    grasp_depth_pred: [B, num_depth + 1, Q]
    grasp_score_pred: [B, 6,             Q]
    grasp_width_pred: [B, 1,             Q]

where Q=M for top1/sampled-single-view modes and Q=M*K for topK/multi-view
modes.

Important
---------
This file deliberately has NO fallback implementation for
`batch_viewpoint_params_to_matrix`. Pass the repository function directly when
instantiating KViewQueryTransformerLocalGraspModule:

    kview_module = KViewQueryTransformerLocalGraspModule(
        ...,
        batch_viewpoint_params_to_matrix_fn=batch_viewpoint_params_to_matrix,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _rank0_only() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _entropy(prob: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return -(prob * torch.log(prob.clamp_min(eps))).sum(dim=dim)


def _safe_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(()) if x.numel() == 1 else x.mean().reshape(())


@torch.no_grad()
def _bincount_float(x: torch.Tensor, n: int) -> torch.Tensor:
    """Small helper that returns a float histogram on x.device."""
    if x is None or n <= 0:
        return torch.zeros((max(int(n), 1),), device="cpu", dtype=torch.float32)
    y = x.detach().reshape(-1).long()
    y = y[(y >= 0) & (y < int(n))]
    if y.numel() == 0:
        return torch.zeros((int(n),), device=x.device, dtype=torch.float32)
    return torch.bincount(y, minlength=int(n)).float()


@torch.no_grad()
def _hist_ratio(hist: torch.Tensor, idx: int) -> torch.Tensor:
    denom = hist.sum().clamp_min(1.0)
    if idx < 0 or idx >= hist.numel():
        return hist.new_tensor(0.0).reshape(())
    return (hist[int(idx)] / denom).reshape(())


@torch.no_grad()
def _safe_mean_masked(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        if not bool(mask.any()):
            return x.new_tensor(0.0).reshape(())
        x = x[mask]
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device if torch.is_tensor(x) else "cpu").reshape(())
    return x.float().mean().reshape(())


@torch.no_grad()
def _tensor_to_numpy(x: Any) -> Optional[np.ndarray]:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return None


@torch.no_grad()
def _add_hist_to_endpoints(
    end_points: Dict[str, Any],
    name: str,
    hist: torch.Tensor,
    scalar_prefix: Optional[str] = None,
    invalid_idx: Optional[int] = None,
) -> None:
    """Store histogram tensor and a few scalar ratios for training logs."""
    if not torch.is_tensor(hist):
        return
    hist = hist.detach().float()
    end_points[name] = hist.cpu()
    if scalar_prefix is None:
        return
    total = hist.sum().clamp_min(1.0)
    end_points[f"{scalar_prefix} hist total"] = total.reshape(())
    if hist.numel() > 0:
        end_points[f"{scalar_prefix} argmax"] = hist.argmax().float().reshape(())
        end_points[f"{scalar_prefix} bin0 ratio"] = (hist[0] / total).reshape(())
    if invalid_idx is not None and 0 <= int(invalid_idx) < hist.numel():
        end_points[f"{scalar_prefix} invalid ratio"] = (hist[int(invalid_idx)] / total).reshape(())


def _find_first_tensor(end_points: Dict[str, Any], names: Sequence[str]) -> Optional[torch.Tensor]:
    for k in names:
        v = end_points.get(k, None)
        if torch.is_tensor(v):
            return v
    return None


@torch.no_grad()
def _extract_query_valid_mask(end_points: Dict[str, Any], B: int, Q: int) -> Optional[torch.Tensor]:
    """Return query-level valid mask used by c2_2 losses, if available.

    In c2_2 loss, angle/depth/score/width losses are computed with:
        valid_mask = end_points['batch_valid_mask']  # [B,Q]

    Therefore label histograms should be reported both on all queries and on
    this valid subset. Otherwise invalid queries with default label 0 can make
    angle-bin-0 collapse look like a label distribution issue.
    """
    for key in [
        "batch_valid_mask", "grasp_valid_mask", "valid_mask",
        "batch_grasp_valid_mask", "kview_query_valid_mask",
    ]:
        m = end_points.get(key, None)
        if not torch.is_tensor(m):
            continue
        try:
            if m.shape == (B, Q):
                return m.detach().bool()
            if m.dim() == 3 and m.shape == (B, 1, Q):
                return m[:, 0, :].detach().bool()
            if m.dim() == 3 and m.shape == (B, Q, 1):
                return m[:, :, 0].detach().bool()
        except Exception:
            continue
    return None


@torch.no_grad()
def _derive_pose_labels_from_endpoints(
    end_points: Dict[str, Any],
    num_angle: int,
    num_depth: int,
    B: int,
    Q: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Best-effort extraction of query-level angle/depth/score labels.

    Direct label keys differ across EconomicGrasp variants. If direct labels are
    absent, derive angle/depth labels from batch_grasp_score [B,Q,A,D] by taking
    the best angle-depth pair. Invalid/no-positive queries are assigned to the
    final invalid bins A and D.
    """
    A, D = int(num_angle), int(num_depth)
    angle_label = _find_first_tensor(end_points, [
        # c2_2 loss uses this key directly:
        #   grasp_angle_label = end_points["batch_grasp_rotations"].long()
        "batch_grasp_rotations",
        "batch_grasp_angle", "batch_grasp_angle_label", "batch_grasp_angle_labels",
        "grasp_angle_label", "grasp_angle_labels", "angle_label", "angle_labels",
        "batch_grasp_angle_cls", "batch_grasp_angle_class",
    ])
    depth_label = _find_first_tensor(end_points, [
        "batch_grasp_depth", "batch_grasp_depth_label", "batch_grasp_depth_labels",
        "grasp_depth_label", "grasp_depth_labels", "depth_label", "depth_labels",
        "batch_grasp_depth_cls", "batch_grasp_depth_class",
    ])
    score_label = _find_first_tensor(end_points, [
        "batch_grasp_score_label", "batch_grasp_score_labels", "grasp_score_label",
        "score_label", "score_labels", "batch_score_label",
    ])

    def _shape_ok(x: torch.Tensor) -> bool:
        return torch.is_tensor(x) and x.shape[:2] == (B, Q)

    if angle_label is not None and _shape_ok(angle_label):
        angle_label = angle_label.detach().long().clamp(0, A)
    else:
        angle_label = None
    if depth_label is not None and _shape_ok(depth_label):
        depth_label = depth_label.detach().long().clamp(0, D)
    else:
        depth_label = None

    # Most EconomicGrasp code keeps batch_grasp_score as [B,Q,A,D] after label matching.
    score = end_points.get("batch_grasp_score", None)
    if torch.is_tensor(score):
        s = score.detach()
        # Try [B,Q,A,D].
        if s.dim() >= 4 and s.shape[0] == B and s.shape[1] == Q and s.shape[-2] == A and s.shape[-1] == D:
            sd = s.reshape(B, Q, A, D)
            flat = sd.reshape(B, Q, A * D)
            best, arg = flat.max(dim=-1)
            valid = torch.isfinite(best) & (best > 0)
            a = (arg // D).long().clamp(0, A - 1)
            d = (arg % D).long().clamp(0, D - 1)
            a = torch.where(valid, a, torch.full_like(a, A))
            d = torch.where(valid, d, torch.full_like(d, D))
            if angle_label is None:
                angle_label = a
            if depth_label is None:
                depth_label = d
            if score_label is None:
                # For histogram only. Assumes score is in [0,1] or at least monotonic.
                sb = torch.round(best.float().clamp(0.0, 1.0) * 5.0).long().clamp(0, 5)
                score_label = torch.where(valid, sb, torch.zeros_like(sb))
        # Fallback layout: [B,Q,D,A].
        elif s.dim() >= 4 and s.shape[0] == B and s.shape[1] == Q and s.shape[-2] == D and s.shape[-1] == A:
            sd = s.reshape(B, Q, D, A).transpose(2, 3).contiguous()
            flat = sd.reshape(B, Q, A * D)
            best, arg = flat.max(dim=-1)
            valid = torch.isfinite(best) & (best > 0)
            a = (arg // D).long().clamp(0, A - 1)
            d = (arg % D).long().clamp(0, D - 1)
            a = torch.where(valid, a, torch.full_like(a, A))
            d = torch.where(valid, d, torch.full_like(d, D))
            if angle_label is None:
                angle_label = a
            if depth_label is None:
                depth_label = d
            if score_label is None:
                sb = torch.round(best.float().clamp(0.0, 1.0) * 5.0).long().clamp(0, 5)
                score_label = torch.where(valid, sb, torch.zeros_like(sb))
        elif s.dim() >= 2 and s.shape[0] == B and s.shape[1] == Q and score_label is None:
            # Direct per-query score label fallback.
            sq = s.reshape(B, Q, -1).max(dim=-1).values
            score_label = torch.round(sq.float().clamp(0.0, 1.0) * 5.0).long().clamp(0, 5)

    if score_label is not None and torch.is_tensor(score_label):
        if score_label.shape == (B, Q):
            score_label = score_label.detach()
            if score_label.dtype.is_floating_point:
                score_label = torch.round(score_label.float().clamp(0.0, 1.0) * 5.0).long().clamp(0, 5)
            else:
                score_label = score_label.long().clamp(0, 5)
        else:
            score_label = None

    return angle_label, depth_label, score_label


def _hist_counts(values: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Return float histogram counts for integer values on the same device."""
    if values is None:
        return torch.zeros(num_bins)
    v = values.detach().reshape(-1).long()
    v = v[(v >= 0) & (v < int(num_bins))]
    if v.numel() == 0:
        return torch.zeros(num_bins, device=values.device, dtype=torch.float32)
    return torch.bincount(v, minlength=int(num_bins)).float()


def _add_hist_logs(
    end_points: Dict[str, Any],
    prefix: str,
    values: torch.Tensor,
    num_bins: int,
    max_bins_to_log: int = 32,
) -> None:
    """Add histogram count/fraction scalars to end_points for logger visibility."""
    if not torch.is_tensor(values):
        return
    counts = _hist_counts(values, num_bins=num_bins)
    total = counts.sum().clamp_min(1.0)
    frac = counts / total
    end_points[f"{prefix} hist entropy"] = _entropy(frac, dim=0).reshape(())
    end_points[f"{prefix} hist maxfrac"] = frac.max().reshape(())
    end_points[f"{prefix} hist argmax"] = torch.argmax(frac).float().reshape(())
    for i in range(min(int(num_bins), int(max_bins_to_log))):
        end_points[f"{prefix} hist {i:02d}"] = frac[i].reshape(())


def _extract_class_label(
    end_points: Dict[str, Any],
    candidate_keys: Sequence[str],
    B: int,
    Q: int,
    num_classes: int,
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    """Best-effort extraction of class labels shaped [B,Q].

    Supports integer labels [B,Q], [B,1,Q], [B,Q,1], or one-hot/logit-like
    tensors [B,C,Q] / [B,Q,C]. This is intentionally conservative: if no
    unambiguous key is found, it returns (None, None).
    """
    for key in candidate_keys:
        t = end_points.get(key, None)
        if not torch.is_tensor(t):
            continue
        x = t.detach()
        try:
            if x.shape == (B, Q):
                y = x
            elif x.dim() == 3 and x.shape == (B, 1, Q):
                y = x[:, 0, :]
            elif x.dim() == 3 and x.shape == (B, Q, 1):
                y = x[:, :, 0]
            elif x.dim() == 3 and x.shape == (B, num_classes, Q):
                y = torch.argmax(x.float(), dim=1)
            elif x.dim() == 3 and x.shape == (B, Q, num_classes):
                y = torch.argmax(x.float(), dim=2)
            else:
                continue
            if not torch.is_floating_point(y):
                yy = y.long()
            else:
                # Accept float labels only when they look class-like.
                yr = torch.round(y)
                diff = (y.float() - yr.float()).abs()
                finite = torch.isfinite(diff)
                if bool(finite.any()) and float(diff[finite].max().item()) > 1e-3:
                    continue
                yy = yr.long()
            if yy.shape == (B, Q):
                return key, yy.clamp(min=0, max=int(num_classes) - 1)
        except Exception:
            continue
    return None, None


def _angle_label_keys() -> Sequence[str]:
    return [
        # c2_2 loss uses this key directly. Keep it first to avoid accidentally
        # falling back to tensors derived from score labels.
        "batch_grasp_rotations",
        "batch_grasp_angle_label", "batch_grasp_angle", "grasp_angle_label",
        "angle_label", "batch_grasp_angle_cls", "grasp_angle_cls",
        "batch_angle_label", "angle_cls",
    ]


def _depth_label_keys() -> Sequence[str]:
    return [
        "batch_grasp_depth_label", "batch_grasp_depth", "grasp_depth_label",
        "depth_label", "batch_grasp_depth_cls", "grasp_depth_cls",
        "batch_depth_label", "depth_cls",
    ]


def _score_label_keys() -> Sequence[str]:
    # Do not include batch_grasp_score here: it is usually [B,Q,A,D] continuous
    # grasp-quality labels, not a per-query class label. It is handled separately
    # by _derive_pose_labels_from_endpoints().
    return [
        "batch_grasp_score_label", "grasp_score_label",
        "score_label", "batch_grasp_score_cls", "grasp_score_cls",
        "batch_score_label", "score_cls",
    ]


@dataclass
class KViewQueryTransformerConfig:
    # A0/A1/A2/A3 ablation interface.
    mode: str = "A1"
    num_query_views: int = 8
    sample_temperature: float = 1.0
    sample_from: str = "softmax"  # softmax | sigmoid_norm | relu_norm
    include_top1_in_multiview_train: bool = True
    mask_top1_when_sampling_extra: bool = True
    replacement: bool = False
    detach_sample_prob: bool = True

    # If True, reuse the sampled view index produced by GeometryAwareDenseFieldViewNet
    # during training for sampled-view modes. This keeps A1/A2 training exactly
    # aligned with the ViewNet multinomial branch instead of sampling a second time
    # inside KViewQuerySelector. For A3, the reused ViewNet sample is inserted as
    # the first view hypothesis and the remaining K-1 views are sampled by this
    # selector. A0 is still deterministic top-1 and ignores this option.
    reuse_viewnet_sample_in_single_view_train: bool = True
    reuse_viewnet_sample_in_multiview_train: bool = True

    # View-conditioned local attention grouping.
    patch_size: int = 8
    metric_radius: float = 0.08
    radius_px_min: float = 8.0
    radius_px_max: float = 64.0
    depth_norm_scale: float = 0.08
    grouping_model_dim: int = 256
    grouping_num_heads: int = 4
    grouping_dropout: float = 0.05
    grouping_ffn_ratio: float = 2.0
    grouping_max_queries_per_chunk: int = 2048
    detach_depth: bool = True
    detach_aux_maps: bool = True
    use_gripper_projected_axes: bool = True

    # K-view transformer grasp head.
    head_model_dim: int = 256
    head_hidden_dim: int = 64
    head_num_layers: int = 2
    head_num_heads: int = 4
    head_attn_dropout: float = 0.05
    head_dropout_p: float = 0.15
    use_view_embedding: bool = True
    use_xyz_embedding: bool = True
    predict_view_quality: bool = True

    # Debug / visualization.
    vis_dir: Optional[str] = None
    vis_every: int = 500
    vis_num_queries: int = 256
    vis_num_patch_queries: int = 16
    save_npz: bool = True
    debug_prefix: str = "D: KV"

    eps: float = 1e-6


# -----------------------------------------------------------------------------
# K-view query selection
# -----------------------------------------------------------------------------


class KViewQuerySelector(nn.Module):
    """Selects K view hypotheses and expands seed tensors to query tensors.

    This module has no fallback for batch_viewpoint_params_to_matrix. The repo
    function must be passed directly.
    """

    def __init__(
        self,
        num_view: int,
        view_dirs: torch.Tensor,
        batch_viewpoint_params_to_matrix_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: KViewQueryTransformerConfig,
    ):
        super().__init__()
        self.num_view = int(num_view)
        self.config = config
        if batch_viewpoint_params_to_matrix_fn is None:
            raise ValueError("Pass repo batch_viewpoint_params_to_matrix directly; no fallback is used.")
        self.batch_viewpoint_params_to_matrix_fn = batch_viewpoint_params_to_matrix_fn

        if view_dirs.shape != (self.num_view, 3):
            raise ValueError(f"view_dirs must be [{self.num_view}, 3], got {tuple(view_dirs.shape)}")
        self.register_buffer("view_dirs", F.normalize(view_dirs.float(), dim=-1), persistent=False)

    def _normalize_view_score_shape(self, view_score: torch.Tensor) -> torch.Tensor:
        if view_score.dim() != 3:
            raise ValueError(f"view_score must be [B,M,V] or [B,V,M], got {tuple(view_score.shape)}")
        if view_score.shape[-1] == self.num_view:
            return view_score.contiguous()
        if view_score.shape[1] == self.num_view:
            return view_score.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot infer view dimension from view_score={tuple(view_score.shape)}, num_view={self.num_view}")

    def _make_prob(self, view_score: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        tau = max(float(cfg.sample_temperature), cfg.eps)
        if cfg.sample_from == "softmax":
            prob = F.softmax(view_score / tau, dim=-1)
        elif cfg.sample_from == "sigmoid_norm":
            w = torch.sigmoid(view_score / tau).clamp_min(cfg.eps)
            prob = w / w.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
        elif cfg.sample_from == "relu_norm":
            w = F.relu(view_score).clamp_min(cfg.eps)
            prob = w / w.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
        elif cfg.sample_from == "minmax_norm":
            # Match GeometryAwareDenseFieldViewNet._select_top_view_inds() training behavior.
            w = view_score.detach().float() if cfg.detach_sample_prob else view_score.float()
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            wmin = w.amin(dim=-1, keepdim=True)
            wmax = w.amax(dim=-1, keepdim=True)
            w = (w - wmin) / (wmax - wmin).clamp_min(cfg.eps)
            row_sum = w.sum(dim=-1, keepdim=True)
            prob_mm = w / row_sum.clamp_min(cfg.eps)
            uniform = torch.full_like(prob_mm, 1.0 / float(self.num_view))
            prob = torch.where(row_sum > cfg.eps, prob_mm, uniform)
        else:
            raise ValueError(f"Unknown sample_from={cfg.sample_from}")
        prob = torch.nan_to_num(
            prob,
            nan=1.0 / self.num_view,
            posinf=1.0 / self.num_view,
            neginf=1.0 / self.num_view,
        )
        return prob / prob.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)

    def _strategy(self, is_training: bool) -> Tuple[str, int]:
        mode = str(self.config.mode).upper()
        K = max(int(self.config.num_query_views), 1)
        if mode == "A0":
            return "top1", 1
        if mode == "A1":
            return ("sample", 1) if is_training else ("top1", 1)
        if mode == "A2":
            return ("sample", 1) if is_training else ("topk", K)
        if mode == "A3":
            return ("samplek", K) if is_training else ("topk", K)
        raise ValueError(f"Unsupported k-view mode={self.config.mode}; expected A0/A1/A2/A3")

    def _select_view_indices(
        self,
        view_score: torch.Tensor,
        is_training: bool,
        forced_view_inds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        B, M, V = view_score.shape
        strategy, Keff = self._strategy(is_training)
        Keff = min(Keff, V)
        prob = self._make_prob(view_score)
        sample_prob = prob.detach() if self.config.detach_sample_prob else prob

        forced = None
        if forced_view_inds is not None:
            forced = forced_view_inds.detach().long().to(device=view_score.device)
            if forced.dim() == 2:
                forced = forced.unsqueeze(-1)
            if forced.dim() != 3 or forced.shape[:2] != (B, M):
                raise ValueError(
                    f"forced_view_inds must be [B,M] or [B,M,Kf], got {tuple(forced_view_inds.shape)}; "
                    f"expected B={B}, M={M}."
                )
            forced = forced.clamp(0, V - 1)

        if forced is not None and strategy == "sample":
            # Reuse the ViewNet sampled view exactly for A1/A2 training.
            view_inds = forced[..., :1].contiguous()
            view_rank = torch.full_like(view_inds, fill_value=-2)

        elif forced is not None and strategy == "samplek":
            # Reuse the ViewNet sampled view as the first hypothesis, then sample
            # remaining K-1 views from the selector distribution.
            n_forced = min(forced.shape[-1], Keff)
            first = forced[..., :n_forced].contiguous()
            n_extra = Keff - n_forced
            if n_extra > 0:
                prob_extra = sample_prob.clone()
                # Avoid duplicating forced views when possible.
                prob_extra.scatter_(dim=-1, index=first, value=0.0)
                denom = prob_extra.sum(dim=-1, keepdim=True)
                prob_extra = torch.where(denom > self.config.eps, prob_extra / denom.clamp_min(self.config.eps), sample_prob)
                sampled = torch.multinomial(
                    prob_extra.reshape(B * M, V),
                    num_samples=n_extra,
                    replacement=bool(self.config.replacement),
                ).view(B, M, n_extra)
                view_inds = torch.cat([first, sampled], dim=-1)
                view_rank = torch.cat([torch.full_like(first, fill_value=-2), torch.full_like(sampled, fill_value=-1)], dim=-1)
            else:
                view_inds = first
                view_rank = torch.full_like(view_inds, fill_value=-2)

        elif strategy == "top1":
            view_inds = torch.argmax(view_score, dim=-1, keepdim=True)
            view_rank = torch.zeros_like(view_inds)

        elif strategy == "topk":
            view_inds = torch.topk(view_score, k=Keff, dim=-1).indices
            view_rank = torch.arange(Keff, device=view_score.device).view(1, 1, Keff).expand(B, M, Keff)

        elif strategy == "sample":
            view_inds = torch.multinomial(sample_prob.reshape(B * M, V), num_samples=1, replacement=True).view(B, M, 1)
            view_rank = torch.full_like(view_inds, fill_value=-1)

        elif strategy == "samplek":
            if self.config.include_top1_in_multiview_train and Keff > 1:
                top1 = torch.argmax(view_score, dim=-1, keepdim=True)
                n_sample = Keff - 1
                prob_extra = sample_prob
                if self.config.mask_top1_when_sampling_extra:
                    prob_extra = prob_extra.clone()
                    prob_extra.scatter_(dim=-1, index=top1, value=0.0)
                    denom = prob_extra.sum(dim=-1, keepdim=True)
                    # If the distribution degenerates, use the original probability.
                    prob_extra = torch.where(denom > self.config.eps, prob_extra / denom.clamp_min(self.config.eps), sample_prob)
                sampled = torch.multinomial(
                    prob_extra.reshape(B * M, V),
                    num_samples=n_sample,
                    replacement=bool(self.config.replacement),
                ).view(B, M, n_sample)
                view_inds = torch.cat([top1, sampled], dim=-1)
                view_rank = torch.cat([torch.zeros_like(top1), torch.full_like(sampled, fill_value=-1)], dim=-1)
            else:
                view_inds = torch.multinomial(
                    sample_prob.reshape(B * M, V),
                    num_samples=Keff,
                    replacement=bool(self.config.replacement),
                ).view(B, M, Keff)
                view_rank = torch.full_like(view_inds, fill_value=-1)
        else:
            raise RuntimeError(f"Unknown strategy={strategy}")

        view_prob_sel = torch.gather(prob, dim=-1, index=view_inds.clamp(0, V - 1))
        return view_inds.long(), view_rank.long(), view_prob_sel, int(Keff), prob

    def _make_view_rot(self, view_inds_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Q = view_inds_q.shape
        dirs = self.view_dirs.to(device=view_inds_q.device, dtype=torch.float32)
        view_xyz = dirs.index_select(0, view_inds_q.reshape(-1)).view(B, Q, 3).contiguous()
        flat_view = view_xyz.reshape(-1, 3)
        flat_angle = torch.zeros(flat_view.shape[0], device=flat_view.device, dtype=flat_view.dtype)
        # Use repo convention directly. No fallback.
        flat_rot = self.batch_viewpoint_params_to_matrix_fn(-flat_view, flat_angle)
        view_rot = flat_rot.view(B, Q, 3, 3).contiguous()
        return view_xyz, view_rot

    @torch.no_grad()
    def _add_debug(
        self,
        end_points: Dict[str, Any],
        view_score: torch.Tensor,
        view_prob: torch.Tensor,
        view_inds: torch.Tensor,
        view_prob_sel: torch.Tensor,
        Keff: int,
    ) -> None:
        B, M, V = view_score.shape
        top2 = torch.topk(view_score, k=min(2, V), dim=-1).values
        margin = top2[..., 0] - top2[..., 1] if V >= 2 else torch.zeros_like(view_score[..., 0])
        ent = _entropy(view_prob, dim=-1, eps=self.config.eps)
        if Keff > 1:
            uniq = torch.ones((B, M, Keff), device=view_inds.device, dtype=torch.bool)
            for kk in range(1, Keff):
                uniq[..., kk] = (view_inds[..., kk:kk + 1] != view_inds[..., :kk]).all(dim=-1)
            unique_ratio = uniq.float().sum(dim=-1).mean() / float(Keff)
        else:
            unique_ratio = view_score.new_tensor(1.0)
        p = self.config.debug_prefix
        end_points[f"{p}Q K"] = view_score.new_tensor(float(Keff)).reshape(())
        end_points[f"{p}Q Q"] = view_score.new_tensor(float(M * Keff)).reshape(())
        end_points[f"{p}Q view entropy"] = ent.mean().reshape(())
        end_points[f"{p}Q view margin"] = margin.mean().reshape(())
        end_points[f"{p}Q selected prob"] = view_prob_sel.mean().reshape(())
        end_points[f"{p}Q unique ratio"] = unique_ratio.reshape(())

        top1_idx = torch.argmax(view_score, dim=-1)
        _add_hist_logs(end_points, f"{p}Q selected view", view_inds.reshape(-1), V, max_bins_to_log=0)
        _add_hist_logs(end_points, f"{p}Q top1 view", top1_idx.reshape(-1), V, max_bins_to_log=0)

        # Selector histograms / per-query values for visualization.
        selected_hist = _bincount_float(view_inds.reshape(-1), V)
        top1_idx = torch.argmax(view_score, dim=-1)
        top1_hist = _bincount_float(top1_idx.reshape(-1), V)
        _add_hist_to_endpoints(end_points, "kview_debug_selected_view_hist", selected_hist, f"{p}Q selected view")
        _add_hist_to_endpoints(end_points, "kview_debug_top1_view_hist", top1_hist, f"{p}Q top1 view")
        end_points["kview_debug_view_entropy0"] = ent[0].detach().cpu() if B > 0 else ent.detach().cpu()
        end_points["kview_debug_view_margin0"] = margin[0].detach().cpu() if B > 0 else margin.detach().cpu()
        end_points["kview_debug_top1_view_inds"] = top1_idx.detach()

    def forward(
        self,
        seed_features: torch.Tensor,
        seed_xyz: torch.Tensor,
        token_sel_idx: torch.Tensor,
        view_score: torch.Tensor,
        end_points: Dict[str, Any],
        is_training: bool,
        forced_view_inds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        view_score = self._normalize_view_score_shape(view_score)
        B, C, M = seed_features.shape
        if seed_xyz.shape[:2] != (B, M):
            raise ValueError(f"seed_xyz must be [B,M,3], got {tuple(seed_xyz.shape)}")
        if token_sel_idx.shape != (B, M):
            raise ValueError(f"token_sel_idx must be [B,M], got {tuple(token_sel_idx.shape)}")
        if view_score.shape[:2] != (B, M):
            raise ValueError(f"view_score {tuple(view_score.shape)} does not match seeds [B,M]=[{B},{M}]")

        view_inds, view_rank, view_prob_sel, Keff, view_prob = self._select_view_indices(
            view_score,
            is_training,
            forced_view_inds=forced_view_inds,
        )
        Q = M * Keff
        parent = torch.arange(M, device=seed_features.device, dtype=torch.long).view(1, M, 1).expand(B, M, Keff)

        parent_q = parent.reshape(B, Q).contiguous()
        view_inds_q = view_inds.reshape(B, Q).contiguous()
        view_rank_q = view_rank.reshape(B, Q).contiguous()
        view_prob_q = view_prob_sel.reshape(B, Q).contiguous()

        seed_features_q = seed_features.unsqueeze(-1).expand(B, C, M, Keff).reshape(B, C, Q).contiguous()
        seed_xyz_q = seed_xyz.unsqueeze(2).expand(B, M, Keff, 3).reshape(B, Q, 3).contiguous()
        token_sel_idx_q = token_sel_idx.unsqueeze(-1).expand(B, M, Keff).reshape(B, Q).contiguous()
        view_xyz_q, view_rot_q = self._make_view_rot(view_inds_q)

        # Base tensors for diagnostics / label restoration.
        end_points["kview_base_xyz_graspable"] = seed_xyz
        end_points["kview_base_token_sel_idx"] = token_sel_idx
        end_points["kview_base_seed_features"] = seed_features
        end_points["kview_base_M"] = int(M)
        end_points["kview_effective_k_int"] = int(Keff)
        end_points["kview_mode"] = str(self.config.mode).upper()

        # Query-level tensors used by label matching, grouping, and decode.
        end_points["xyz_graspable"] = seed_xyz_q
        end_points["token_sel_xyz"] = seed_xyz_q
        end_points["token_sel_idx"] = token_sel_idx_q
        end_points["grasp_top_view_inds"] = view_inds_q
        end_points["grasp_top_view_xyz"] = view_xyz_q
        end_points["grasp_top_view_rot"] = view_rot_q

        end_points["kview_query_parent"] = parent_q
        end_points["kview_query_view_inds"] = view_inds_q
        end_points["kview_query_view_rank"] = view_rank_q
        end_points["kview_query_view_prob"] = view_prob_q
        end_points["kview_view_prob"] = view_prob.detach()

        self._add_debug(end_points, view_score.detach(), view_prob.detach(), view_inds.detach(), view_prob_sel.detach(), Keff)

        # Debug: confirm whether the ViewNet sampled view has been reused.
        p = self.config.debug_prefix
        forced_used = forced_view_inds is not None
        end_points[f"{p}Q forced view used"] = view_score.new_tensor(float(forced_used)).reshape(())
        if forced_used:
            forced_dbg = forced_view_inds.detach().long().to(device=view_inds.device)
            if forced_dbg.dim() == 2:
                forced_dbg = forced_dbg.unsqueeze(-1)
            forced_dbg = forced_dbg.clamp(0, self.num_view - 1)
            n_cmp = min(forced_dbg.shape[-1], view_inds.shape[-1])
            diff = (view_inds[..., :n_cmp] != forced_dbg[..., :n_cmp]).float().mean()
            end_points[f"{p}Q forced view diff"] = diff.reshape(())
            end_points["kview_viewnet_forced_view_inds"] = forced_dbg[..., 0].detach()
        else:
            end_points[f"{p}Q forced view diff"] = view_score.new_tensor(0.0).reshape(())

        return seed_features_q, seed_xyz_q, token_sel_idx_q, view_rot_q, end_points


# -----------------------------------------------------------------------------
# View-conditioned attention grouping
# -----------------------------------------------------------------------------


class ViewConditionedAttentionGrouping(nn.Module):
    """View-conditioned local 2D/DPT attention grouping.

    For each center-view query, this module constructs a gripper-frame local patch
    in image coordinates. The patch axes are obtained by projecting the gripper
    rotation's local y/z axes into the image. A query token cross-attends to the
    sampled patch tokens, yielding one grouped feature per center-view query.
    """

    def __init__(
        self,
        seed_feature_dim: int,
        feat_dim: int,
        out_dim: int,
        config: KViewQueryTransformerConfig,
    ):
        super().__init__()
        self.seed_feature_dim = int(seed_feature_dim)
        self.feat_dim = int(feat_dim)
        self.out_dim = int(out_dim)
        self.config = config
        C = int(config.grouping_model_dim)

        self.seed_proj = nn.Sequential(
            nn.Conv1d(seed_feature_dim, C, 1),
            nn.GroupNorm(8 if C % 8 == 0 else 1, C),
            nn.ReLU(inplace=True),
            nn.Conv1d(C, C, 1),
        )
        self.xyz_embed = nn.Sequential(nn.Linear(3, C), nn.GELU(), nn.Linear(C, C))
        self.view_rot_embed = nn.Sequential(nn.Linear(9, C), nn.GELU(), nn.Linear(C, C))

        # sampled token = image feature + depth_delta + fg_prob + graspness + valid + local offsets(x,y)
        self.patch_proj = nn.Sequential(
            nn.Linear(feat_dim + 1 + 1 + 1 + 1 + 2, C),
            nn.GELU(),
            nn.Linear(C, C),
        )

        self.cross_attn = nn.MultiheadAttention(
            C,
            int(config.grouping_num_heads),
            dropout=float(config.grouping_dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(C)
        hidden = int(C * float(config.grouping_ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(C, hidden),
            nn.GELU(),
            nn.Dropout(float(config.grouping_dropout)),
            nn.Linear(hidden, C),
            nn.Dropout(float(config.grouping_dropout)),
        )
        self.norm2 = nn.LayerNorm(C)
        self.out_proj = nn.Conv1d(C, out_dim, 1)

        S = int(config.patch_size)
        if S < 2:
            raise ValueError("patch_size must be >= 2")
        lin = torch.linspace(-1.0, 1.0, S)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        offsets = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [P,2]
        self.register_buffer("unit_offsets", offsets.float(), persistent=False)

    @staticmethod
    def _project_xyz_to_uv(xyz: torch.Tensor, camera_K: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """xyz [B,Q,3], camera_K [B,3,3] -> uv [B,Q,2]."""
        z = xyz[..., 2].clamp_min(eps)
        fx = camera_K[:, 0, 0].view(-1, 1)
        fy = camera_K[:, 1, 1].view(-1, 1)
        cx = camera_K[:, 0, 2].view(-1, 1)
        cy = camera_K[:, 1, 2].view(-1, 1)
        u = xyz[..., 0] / z * fx + cx
        v = xyz[..., 1] / z * fy + cy
        return torch.stack([u, v], dim=-1)

    def _clamp_vec_radius(self, vec: torch.Tensor, fallback: torch.Tensor, radius_px: torch.Tensor) -> torch.Tensor:
        eps = self.config.eps
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        use_fallback = (~torch.isfinite(norm)) | (norm < eps)
        vec = torch.where(use_fallback, fallback, vec)
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)
        target = norm.clamp(float(self.config.radius_px_min), float(self.config.radius_px_max))
        # If the projection is nearly singular, use the depth-derived fallback radius.
        target = torch.where(use_fallback, radius_px.unsqueeze(-1), target)
        return vec / norm * target

    def _make_view_conditioned_grid(
        self,
        seed_xyz: torch.Tensor,
        token_sel_idx: torch.Tensor,
        top_view_rot: torch.Tensor,
        depth_map: torch.Tensor,
        camera_K: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds normalized grid [B,Q,P,2] and valid mask [B,Q,P].

        Also returns center/projection and projected local axes for debug.
        """
        B, Q, _ = seed_xyz.shape
        device = seed_xyz.device
        dtype = seed_xyz.dtype

        u0 = (token_sel_idx % W).to(dtype=dtype)
        v0 = (token_sel_idx // W).to(dtype=dtype)
        center_uv = torch.stack([u0, v0], dim=-1)  # [B,Q,2]
        center_proj_uv = self._project_xyz_to_uv(seed_xyz, camera_K)
        center_proj_err = torch.linalg.norm(center_proj_uv - center_uv, dim=-1)  # [B,Q]

        z = seed_xyz[..., 2].clamp_min(1e-6)
        fx = camera_K[:, 0, 0].view(B, 1).to(dtype=dtype)
        fy = camera_K[:, 1, 1].view(B, 1).to(dtype=dtype)
        fmean = 0.5 * (fx + fy)
        radius_px = (fmean * float(self.config.metric_radius) / z).clamp(
            float(self.config.radius_px_min),
            float(self.config.radius_px_max),
        )  # [B,Q]

        if self.config.use_gripper_projected_axes and top_view_rot is not None:
            # GraspNet rotation convention: columns are local axes. x is approach;
            # y/z span the gripper local plane after zero in-plane angle.
            axis_y = top_view_rot[..., :, 1].to(dtype=dtype)
            axis_z = top_view_rot[..., :, 2].to(dtype=dtype)
            uv_y = self._project_xyz_to_uv(seed_xyz + axis_y * float(self.config.metric_radius), camera_K)
            uv_z = self._project_xyz_to_uv(seed_xyz + axis_z * float(self.config.metric_radius), camera_K)
            vec_y = uv_y - center_uv
            vec_z = uv_z - center_uv
        else:
            vec_y = torch.zeros((B, Q, 2), device=device, dtype=dtype)
            vec_z = torch.zeros((B, Q, 2), device=device, dtype=dtype)
            vec_y[..., 0] = radius_px
            vec_z[..., 1] = radius_px

        fallback_y = torch.zeros_like(vec_y)
        fallback_z = torch.zeros_like(vec_z)
        fallback_y[..., 0] = radius_px
        fallback_z[..., 1] = radius_px
        vec_y = self._clamp_vec_radius(vec_y, fallback_y, radius_px)
        vec_z = self._clamp_vec_radius(vec_z, fallback_z, radius_px)

        offsets = self.unit_offsets.to(device=device, dtype=dtype)  # [P,2]
        # local x offset follows projected gripper-y; local y offset follows projected gripper-z.
        patch_uv = (
            center_uv.unsqueeze(2)
            + offsets[:, 0].view(1, 1, -1, 1) * vec_y.unsqueeze(2)
            + offsets[:, 1].view(1, 1, -1, 1) * vec_z.unsqueeze(2)
        )  # [B,Q,P,2]

        valid = (
            torch.isfinite(patch_uv).all(dim=-1)
            & (patch_uv[..., 0] >= 0.0)
            & (patch_uv[..., 0] <= float(W - 1))
            & (patch_uv[..., 1] >= 0.0)
            & (patch_uv[..., 1] <= float(H - 1))
        )

        x_norm = patch_uv[..., 0] / max(float(W - 1), 1.0) * 2.0 - 1.0
        y_norm = patch_uv[..., 1] / max(float(H - 1), 1.0) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).clamp(-2.0, 2.0).contiguous()
        radius_dbg = 0.5 * (torch.linalg.norm(vec_y, dim=-1) + torch.linalg.norm(vec_z, dim=-1))
        return grid, valid, radius_dbg, patch_uv, center_uv, center_proj_uv, center_proj_err, vec_y, vec_z

    def _sample_map(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """x [B,C,H,W], grid [B,Q,P,2] -> [B,Q,P,C]."""
        sampled = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # [B,C,Q,P]
        return sampled.permute(0, 2, 3, 1).contiguous()

    def _sample_aux_maps(
        self,
        grid: torch.Tensor,
        seed_xyz: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        objectness_logits: Optional[torch.Tensor],
        graspness_map: Optional[torch.Tensor],
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Q, P, _ = grid.shape
        device = grid.device
        dtype = seed_xyz.dtype

        if depth_map is not None:
            dmap = depth_map
            if dmap.dim() == 3:
                dmap = dmap.unsqueeze(1)
            dmap = dmap[:, :1]
            if self.config.detach_depth:
                dmap = dmap.detach()
            depth_patch = self._sample_map(dmap.to(dtype=dtype), grid).squeeze(-1)  # [B,Q,P]
            depth_delta = (depth_patch - seed_xyz[..., 2].unsqueeze(-1)) / max(float(self.config.depth_norm_scale), self.config.eps)
            depth_delta = torch.nan_to_num(depth_delta, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)
        else:
            depth_delta = torch.zeros((B, Q, P), device=device, dtype=dtype)

        if objectness_logits is not None:
            obj = objectness_logits
            if self.config.detach_aux_maps:
                obj = obj.detach()
            if obj.shape[1] >= 2:
                obj_fg = F.softmax(obj[:, :2], dim=1)[:, 1:2]
            else:
                obj_fg = torch.sigmoid(obj[:, :1])
            fg_patch = self._sample_map(obj_fg.to(dtype=dtype), grid).squeeze(-1)
        else:
            fg_patch = torch.zeros((B, Q, P), device=device, dtype=dtype)

        if graspness_map is not None:
            gmap = graspness_map
            if gmap.dim() == 3:
                gmap = gmap.unsqueeze(1)
            if self.config.detach_aux_maps:
                gmap = gmap.detach()
            grasp_patch = self._sample_map(gmap[:, :1].to(dtype=dtype), grid).squeeze(-1)
        else:
            grasp_patch = torch.zeros((B, Q, P), device=device, dtype=dtype)

        valid_f = valid.to(dtype=dtype)
        return depth_delta, fg_patch, grasp_patch, valid_f

    @torch.no_grad()
    def _add_debug(
        self,
        end_points: Dict[str, Any],
        valid_sum: torch.Tensor,
        valid_count: int,
        radius_sum: torch.Tensor,
        radius_count: int,
        ent_sum: torch.Tensor,
        maxp_sum: torch.Tensor,
        attn_count: int,
        first_patch_uv: Optional[torch.Tensor],
        *,
        center_proj_err_sum: torch.Tensor,
        center_proj_err_max: torch.Tensor,
        center_proj_err_count: int,
        patch_fg_sum: torch.Tensor,
        attn_fg_sum: torch.Tensor,
        patch_grasp_sum: torch.Tensor,
        attn_grasp_sum: torch.Tensor,
        depth_delta_abs_sum: torch.Tensor,
        attn_depth_delta_abs_sum: torch.Tensor,
        patch_stat_count: int,
        attn_stat_count: int,
        first_patch_attn: Optional[torch.Tensor],
        first_patch_fg: Optional[torch.Tensor],
        first_patch_grasp: Optional[torch.Tensor],
        first_patch_depth_delta: Optional[torch.Tensor],
        first_patch_valid: Optional[torch.Tensor],
        first_center_uv: Optional[torch.Tensor],
        first_center_proj_uv: Optional[torch.Tensor],
        first_center_proj_err: Optional[torch.Tensor],
        first_radius_dbg: Optional[torch.Tensor],
    ) -> None:
        p = self.config.debug_prefix
        end_points[f"{p}CA valid ratio"] = (valid_sum / max(float(valid_count), 1.0)).reshape(())
        end_points[f"{p}CA radius px"] = (radius_sum / max(float(radius_count), 1.0)).reshape(())
        end_points[f"{p}CA attn entropy"] = (ent_sum / max(float(attn_count), 1.0)).reshape(())
        end_points[f"{p}CA attn maxprob"] = (maxp_sum / max(float(attn_count), 1.0)).reshape(())

        # Coordinate-system sanity check. If this is large, token_sel_idx, seed_xyz,
        # and camera_K are not in the same resized-image coordinate system.
        end_points[f"{p}CA center proj err px"] = (center_proj_err_sum / max(float(center_proj_err_count), 1.0)).reshape(())
        end_points[f"{p}CA center proj err max px"] = center_proj_err_max.reshape(())

        # Attention foreground/object evidence. Patch means are valid-weighted;
        # attention means use the actual cross-attention weights.
        end_points[f"{p}CA patch fg"] = (patch_fg_sum / max(float(patch_stat_count), 1.0)).reshape(())
        end_points[f"{p}CA attn fg"] = (attn_fg_sum / max(float(attn_stat_count), 1.0)).reshape(())
        end_points[f"{p}CA patch graspness"] = (patch_grasp_sum / max(float(patch_stat_count), 1.0)).reshape(())
        end_points[f"{p}CA attn graspness"] = (attn_grasp_sum / max(float(attn_stat_count), 1.0)).reshape(())
        end_points[f"{p}CA depth delta abs"] = (depth_delta_abs_sum / max(float(patch_stat_count), 1.0)).reshape(())
        end_points[f"{p}CA attn depth delta abs"] = (attn_depth_delta_abs_sum / max(float(attn_stat_count), 1.0)).reshape(())

        if first_patch_uv is not None:
            # [num_vis_queries,P,2] for batch item 0 only.
            end_points["kview_debug_patch_uv0"] = first_patch_uv.detach().cpu()
        for key, val in {
            "kview_debug_patch_attn0": first_patch_attn,
            "kview_debug_patch_fg0": first_patch_fg,
            "kview_debug_patch_graspness0": first_patch_grasp,
            "kview_debug_patch_depth_delta0": first_patch_depth_delta,
            "kview_debug_patch_valid0": first_patch_valid,
            "kview_debug_center_uv0": first_center_uv,
            "kview_debug_center_proj_uv0": first_center_proj_uv,
            "kview_debug_center_proj_err0": first_center_proj_err,
            "kview_debug_radius_px0": first_radius_dbg,
        }.items():
            if val is not None:
                end_points[key] = val.detach().cpu()

    def forward(
        self,
        seed_features: torch.Tensor,
        token_sel_idx: torch.Tensor,
        seed_xyz: torch.Tensor,
        top_view_rot: torch.Tensor,
        feat_map: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        objectness_logits: Optional[torch.Tensor],
        graspness_map: Optional[torch.Tensor],
        camera_K: torch.Tensor,
        end_points: Dict[str, Any],
    ) -> torch.Tensor:
        """Returns grouped features [B,out_dim,Q]."""
        B, Cseed, Q = seed_features.shape
        if token_sel_idx.shape != (B, Q):
            raise ValueError(f"token_sel_idx must be [B,Q], got {tuple(token_sel_idx.shape)}")
        if seed_xyz.shape[:2] != (B, Q):
            raise ValueError(f"seed_xyz must be [B,Q,3], got {tuple(seed_xyz.shape)}")
        if top_view_rot.shape[:2] != (B, Q):
            raise ValueError(f"top_view_rot must be [B,Q,3,3], got {tuple(top_view_rot.shape)}")

        _, Cfeat, H, W = feat_map.shape
        if Cfeat != self.feat_dim:
            raise ValueError(f"feat_map channel={Cfeat}, expected feat_dim={self.feat_dim}")

        query = self.seed_proj(seed_features).transpose(1, 2).contiguous()  # [B,Q,C]
        query = query + self.xyz_embed(torch.nan_to_num(seed_xyz.float(), nan=0.0, posinf=0.0, neginf=0.0).to(query.dtype))
        query = query + self.view_rot_embed(top_view_rot.reshape(B, Q, 9).float().to(query.dtype))

        max_q = max(int(self.config.grouping_max_queries_per_chunk), 1)
        outputs = []
        valid_sum = seed_features.new_tensor(0.0)
        valid_count = 0
        radius_sum = seed_features.new_tensor(0.0)
        radius_count = 0
        ent_sum = seed_features.new_tensor(0.0)
        maxp_sum = seed_features.new_tensor(0.0)
        attn_count = 0

        # Additional grouping diagnostics.
        center_proj_err_sum = seed_features.new_tensor(0.0)
        center_proj_err_max = seed_features.new_tensor(0.0)
        center_proj_err_count = 0
        patch_fg_sum = seed_features.new_tensor(0.0)
        attn_fg_sum = seed_features.new_tensor(0.0)
        patch_grasp_sum = seed_features.new_tensor(0.0)
        attn_grasp_sum = seed_features.new_tensor(0.0)
        depth_delta_abs_sum = seed_features.new_tensor(0.0)
        attn_depth_delta_abs_sum = seed_features.new_tensor(0.0)
        patch_stat_count = 0
        attn_stat_count = 0

        first_patch_uv = None
        first_patch_attn = None
        first_patch_fg = None
        first_patch_grasp = None
        first_patch_depth_delta = None
        first_patch_valid = None
        first_center_uv = None
        first_center_proj_uv = None
        first_center_proj_err = None
        first_radius_dbg = None

        for qs in range(0, Q, max_q):
            qe = min(qs + max_q, Q)
            qn = qe - qs
            seed_xyz_c = seed_xyz[:, qs:qe]
            idx_c = token_sel_idx[:, qs:qe]
            rot_c = top_view_rot[:, qs:qe]
            query_c = query[:, qs:qe]

            grid, valid, radius_dbg, patch_uv, center_uv, center_proj_uv, center_proj_err, vec_y, vec_z = self._make_view_conditioned_grid(
                seed_xyz=seed_xyz_c,
                token_sel_idx=idx_c,
                top_view_rot=rot_c,
                depth_map=depth_map,
                camera_K=camera_K,
                H=H,
                W=W,
            )  # grid [B,qn,P,2]

            patch_feat = self._sample_map(feat_map, grid)  # [B,qn,P,Cfeat]
            depth_delta, fg_patch, grasp_patch, valid_f = self._sample_aux_maps(
                grid=grid,
                seed_xyz=seed_xyz_c,
                depth_map=depth_map,
                objectness_logits=objectness_logits,
                graspness_map=graspness_map,
                valid=valid,
            )
            offsets = self.unit_offsets.to(device=feat_map.device, dtype=feat_map.dtype).view(1, 1, -1, 2)
            offsets = offsets.expand(B, qn, -1, -1)
            token_in = torch.cat(
                [
                    patch_feat.to(dtype=query.dtype),
                    depth_delta.unsqueeze(-1).to(dtype=query.dtype),
                    fg_patch.unsqueeze(-1).to(dtype=query.dtype),
                    grasp_patch.unsqueeze(-1).to(dtype=query.dtype),
                    valid_f.unsqueeze(-1).to(dtype=query.dtype),
                    offsets.to(dtype=query.dtype),
                ],
                dim=-1,
            )
            kv = self.patch_proj(token_in).reshape(B * qn, -1, query.shape[-1]).contiguous()
            q = query_c.reshape(B * qn, 1, query.shape[-1]).contiguous()

            key_padding_mask = (~valid).reshape(B * qn, -1).contiguous()
            all_invalid = key_padding_mask.all(dim=-1)
            if bool(all_invalid.any()):
                key_padding_mask[all_invalid] = False

            attn_out, attn_w = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=True)
            # attn_w [B*qn,1,P] averaged over heads.
            q = self.norm1(q + attn_out)
            q = self.norm2(q + self.ffn(q))
            outputs.append(q.view(B, qn, -1))

            with torch.no_grad():
                valid_sum = valid_sum + valid_f.sum()
                valid_count += int(valid_f.numel())
                radius_sum = radius_sum + radius_dbg.detach().sum()
                radius_count += int(radius_dbg.numel())
                aw = attn_w.detach().squeeze(1).clamp_min(self.config.eps)  # [B*qn,P]
                ent_sum = ent_sum + _entropy(aw, dim=-1, eps=self.config.eps).sum()
                maxp_sum = maxp_sum + aw.max(dim=-1).values.sum()
                attn_count += int(aw.shape[0])

                center_proj_err_sum = center_proj_err_sum + center_proj_err.detach().sum()
                center_proj_err_max = torch.maximum(center_proj_err_max, center_proj_err.detach().max())
                center_proj_err_count += int(center_proj_err.numel())

                # Patch-level foreground/graspness mass. Patch means are valid-weighted.
                valid_den = valid_f.sum(dim=-1).clamp_min(1.0)  # [B,qn]
                patch_fg = (fg_patch.detach() * valid_f.detach()).sum(dim=-1) / valid_den
                patch_grasp = (grasp_patch.detach() * valid_f.detach()).sum(dim=-1) / valid_den
                patch_depth_abs = (depth_delta.detach().abs() * valid_f.detach()).sum(dim=-1) / valid_den
                patch_fg_sum = patch_fg_sum + patch_fg.sum()
                patch_grasp_sum = patch_grasp_sum + patch_grasp.sum()
                depth_delta_abs_sum = depth_delta_abs_sum + patch_depth_abs.sum()
                patch_stat_count += int(patch_fg.numel())

                aw_view = aw.view(B, qn, -1)
                # key_padding_mask makes invalid tokens receive zero mass, except all-invalid fallbacks.
                attn_fg = (aw_view * fg_patch.detach()).sum(dim=-1)
                attn_grasp = (aw_view * grasp_patch.detach()).sum(dim=-1)
                attn_depth_abs = (aw_view * depth_delta.detach().abs()).sum(dim=-1)
                attn_fg_sum = attn_fg_sum + attn_fg.sum()
                attn_grasp_sum = attn_grasp_sum + attn_grasp.sum()
                attn_depth_delta_abs_sum = attn_depth_delta_abs_sum + attn_depth_abs.sum()
                attn_stat_count += int(attn_fg.numel())

                if first_patch_uv is None and qs == 0:
                    nvis = min(int(self.config.vis_num_queries), qn)
                    first_patch_uv = patch_uv[0, :nvis].detach()
                    first_patch_attn = aw_view[0, :nvis].detach()
                    first_patch_fg = fg_patch[0, :nvis].detach()
                    first_patch_grasp = grasp_patch[0, :nvis].detach()
                    first_patch_depth_delta = depth_delta[0, :nvis].detach()
                    first_patch_valid = valid[0, :nvis].detach()
                    first_center_uv = center_uv[0, :nvis].detach()
                    first_center_proj_uv = center_proj_uv[0, :nvis].detach()
                    first_center_proj_err = center_proj_err[0, :nvis].detach()
                    first_radius_dbg = radius_dbg[0, :nvis].detach()
                    end_points["kview_debug_vec_y0"] = vec_y[0, :nvis].detach().cpu()
                    end_points["kview_debug_vec_z0"] = vec_z[0, :nvis].detach().cpu()

        x = torch.cat(outputs, dim=1).transpose(1, 2).contiguous()  # [B,C,Q]
        group_features = self.out_proj(x)  # [B,out_dim,Q]
        self._add_debug(
            end_points,
            valid_sum, valid_count, radius_sum, radius_count, ent_sum, maxp_sum, attn_count,
            first_patch_uv,
            center_proj_err_sum=center_proj_err_sum,
            center_proj_err_max=center_proj_err_max,
            center_proj_err_count=center_proj_err_count,
            patch_fg_sum=patch_fg_sum,
            attn_fg_sum=attn_fg_sum,
            patch_grasp_sum=patch_grasp_sum,
            attn_grasp_sum=attn_grasp_sum,
            depth_delta_abs_sum=depth_delta_abs_sum,
            attn_depth_delta_abs_sum=attn_depth_delta_abs_sum,
            patch_stat_count=patch_stat_count,
            attn_stat_count=attn_stat_count,
            first_patch_attn=first_patch_attn,
            first_patch_fg=first_patch_fg,
            first_patch_grasp=first_patch_grasp,
            first_patch_depth_delta=first_patch_depth_delta,
            first_patch_valid=first_patch_valid,
            first_center_uv=first_center_uv,
            first_center_proj_uv=first_center_proj_uv,
            first_center_proj_err=first_center_proj_err,
            first_radius_dbg=first_radius_dbg,
        )
        return group_features


# -----------------------------------------------------------------------------
# K-view transformer grasp head
# -----------------------------------------------------------------------------


class _WithinCenterViewTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, ffn_ratio: float = 4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, k_view: int) -> torch.Tensor:
        B, Q, C = x.shape
        if k_view <= 1:
            return self.norm2(x + self.ffn(x))
        if Q % k_view != 0:
            raise ValueError(f"Q={Q} is not divisible by k_view={k_view}")
        M = Q // k_view
        y = x.view(B, M, k_view, C).reshape(B * M, k_view, C).contiguous()
        out, _ = self.attn(y, y, y, need_weights=False)
        y = self.norm1(y + out)
        y = self.norm2(y + self.ffn(y))
        return y.view(B, M, k_view, C).reshape(B, Q, C).contiguous()


class KViewQueryTransformerGraspHeadDropout(nn.Module):
    def __init__(
        self,
        num_angle: int,
        num_depth: int,
        num_view: int,
        in_dim: int,
        config: KViewQueryTransformerConfig,
    ):
        super().__init__()
        self.num_angle = int(num_angle)
        self.num_depth = int(num_depth)
        self.num_view = int(num_view)
        self.config = config
        model_dim = int(config.head_model_dim)
        hidden_dim = int(config.head_hidden_dim)

        self.input_proj = nn.Conv1d(in_dim, model_dim, 1)
        self.input_norm = nn.LayerNorm(model_dim)
        self.view_embed = nn.Embedding(self.num_view, model_dim) if config.use_view_embedding else None
        self.xyz_embed = nn.Sequential(nn.Linear(3, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)) if config.use_xyz_embedding else None
        self.layers = nn.ModuleList([
            _WithinCenterViewTransformerLayer(
                dim=model_dim,
                num_heads=int(config.head_num_heads),
                dropout=float(config.head_attn_dropout),
                ffn_ratio=4.0,
            )
            for _ in range(int(config.head_num_layers))
        ])

        self.conv_angle_feature = nn.Conv1d(model_dim, hidden_dim, 1)
        self.conv_depth_feature = nn.Conv1d(model_dim, hidden_dim, 1)
        self.conv_width_feature = nn.Conv1d(model_dim, hidden_dim, 1)
        self.conv_score_feature = nn.Conv1d(model_dim, hidden_dim, 1)

        self.branch_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, dropout=float(config.head_attn_dropout), batch_first=True)
        self.branch_norm = nn.LayerNorm(hidden_dim)

        # Output-head dropout, same spirit as your local-interaction dropout head.
        self.angle_dropout = nn.Dropout(float(config.head_dropout_p))
        self.depth_dropout = nn.Dropout(float(config.head_dropout_p))
        self.width_dropout = nn.Dropout(float(config.head_dropout_p))
        self.score_dropout = nn.Dropout(float(config.head_dropout_p))
        self.view_quality_dropout = nn.Dropout(float(config.head_dropout_p))

        self.conv_angle = nn.Conv1d(hidden_dim, self.num_angle + 1, 1)
        self.conv_depth = nn.Conv1d(hidden_dim, self.num_depth + 1, 1)
        self.angle_ctx_proj = nn.Conv1d(self.num_angle + 1, hidden_dim, 1, bias=False)
        self.depth_ctx_proj = nn.Conv1d(self.num_depth + 1, hidden_dim, 1, bias=False)

        self.width_fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        self.score_fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        self.conv_width = nn.Conv1d(hidden_dim, 1, 1)
        self.conv_score = nn.Conv1d(hidden_dim, 6, 1)
        self.conv_view_quality = nn.Conv1d(hidden_dim, 1, 1) if config.predict_view_quality else None

    def _add_query_embeddings(self, x: torch.Tensor, end_points: Dict[str, Any]) -> torch.Tensor:
        B, Q, C = x.shape
        if self.view_embed is not None and "kview_query_view_inds" in end_points:
            inds = end_points["kview_query_view_inds"].long().clamp(0, self.num_view - 1)
            if inds.shape == (B, Q):
                x = x + self.view_embed(inds)
        if self.xyz_embed is not None and "xyz_graspable" in end_points:
            xyz = end_points["xyz_graspable"]
            if torch.is_tensor(xyz) and xyz.shape[:2] == (B, Q):
                x = x + self.xyz_embed(torch.nan_to_num(xyz.float(), nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype))
        return x

    def _interact_four_branches(self, a: torch.Tensor, d: torch.Tensor, w: torch.Tensor, s: torch.Tensor):
        B, C, Q = a.shape
        tokens = torch.stack([a.permute(0, 2, 1), d.permute(0, 2, 1), w.permute(0, 2, 1), s.permute(0, 2, 1)], dim=2)
        tokens = tokens.reshape(B * Q, 4, C).contiguous()
        out, _ = self.branch_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.branch_norm(tokens + out).view(B, Q, 4, C)
        return (
            tokens[:, :, 0].permute(0, 2, 1).contiguous(),
            tokens[:, :, 1].permute(0, 2, 1).contiguous(),
            tokens[:, :, 2].permute(0, 2, 1).contiguous(),
            tokens[:, :, 3].permute(0, 2, 1).contiguous(),
        )

    @torch.no_grad()
    def _add_debug(
        self,
        end_points: Dict[str, Any],
        angle_logits: torch.Tensor,
        depth_logits: torch.Tensor,
        score_pred: torch.Tensor,
        width_pred: torch.Tensor,
        view_quality: Optional[torch.Tensor],
    ) -> None:
        p = self.config.debug_prefix
        B, _, Q = angle_logits.shape
        A = self.num_angle
        D = self.num_depth

        angle_prob = F.softmax(angle_logits, dim=1)
        depth_prob = F.softmax(depth_logits, dim=1)
        score_prob = F.softmax(score_pred, dim=1)
        bins = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=score_pred.device, dtype=score_pred.dtype).view(1, 6, 1)
        score_expected = (score_prob * bins).sum(dim=1)  # [B,Q]

        # Standard scalar diagnostics.
        angle_ent = _entropy(angle_prob, dim=1)
        depth_ent = _entropy(depth_prob, dim=1)
        score_ent = _entropy(score_prob, dim=1)
        end_points[f"{p}GH angle entropy"] = angle_ent.mean().reshape(())
        end_points[f"{p}GH depth entropy"] = depth_ent.mean().reshape(())
        end_points[f"{p}GH score entropy"] = score_ent.mean().reshape(())
        end_points[f"{p}GH angle maxprob"] = angle_prob.max(dim=1).values.mean().reshape(())
        end_points[f"{p}GH depth maxprob"] = depth_prob.max(dim=1).values.mean().reshape(())
        end_points[f"{p}GH score expected"] = score_expected.mean().reshape(())
        end_points[f"{p}GH width mean"] = width_pred.mean().reshape(())

        # Prediction histograms. Include the invalid bin for angle/depth.
        angle_arg = angle_prob.argmax(dim=1)  # [B,Q], 0..A invalid=A
        depth_arg = depth_prob.argmax(dim=1)  # [B,Q], 0..D invalid=D
        score_arg = score_prob.argmax(dim=1)  # [B,Q], 0..5
        angle_hist = _bincount_float(angle_arg, A + 1)
        depth_hist = _bincount_float(depth_arg, D + 1)
        score_hist = _bincount_float(score_arg, 6)
        _add_hist_to_endpoints(end_points, "kview_debug_pred_angle_hist", angle_hist, f"{p}GH pred angle", invalid_idx=A)
        _add_hist_to_endpoints(end_points, "kview_debug_pred_depth_hist", depth_hist, f"{p}GH pred depth", invalid_idx=D)
        _add_hist_to_endpoints(end_points, "kview_debug_pred_score_hist", score_hist, f"{p}GH pred score")
        end_points[f"{p}GH pred depth01 ratio"] = ((depth_arg == 0) | (depth_arg == 1)).float().mean().reshape(())
        end_points[f"{p}GH pred angle valid maxprob"] = angle_prob[:, :A].max(dim=1).values.mean().reshape(())
        end_points[f"{p}GH pred depth valid maxprob"] = depth_prob[:, :D].max(dim=1).values.mean().reshape(())

        # Per-query arrays for NPZ/diagnostic scripts. Keep them on CPU only.
        end_points["kview_debug_pred_angle_argmax"] = angle_arg.detach().cpu()
        end_points["kview_debug_pred_depth_argmax"] = depth_arg.detach().cpu()
        end_points["kview_debug_pred_score_argmax"] = score_arg.detach().cpu()
        end_points["kview_debug_angle_entropy0"] = angle_ent[0].detach().cpu() if B > 0 else angle_ent.detach().cpu()
        end_points["kview_debug_depth_entropy0"] = depth_ent[0].detach().cpu() if B > 0 else depth_ent.detach().cpu()
        end_points["kview_debug_score_entropy0"] = score_ent[0].detach().cpu() if B > 0 else score_ent.detach().cpu()
        end_points["kview_debug_score_expected0"] = score_expected[0].detach().cpu() if B > 0 else score_expected.detach().cpu()

        # Query-level label histograms. Best-effort: direct labels if present;
        # otherwise derive from batch_grasp_score [B,Q,A,D].
        angle_label, depth_label, score_label = _derive_pose_labels_from_endpoints(end_points, A, D, B, Q)
        # c2_2 uses batch_valid_mask for angle/depth/score/width losses.
        # For angle diagnosis, log both all-query and valid-query histograms.
        # The valid-query histogram is the one that should be compared with
        # Angle Loss / Angle Acc.
        query_valid_mask = _extract_query_valid_mask(end_points, B, Q)
        if query_valid_mask is not None:
            end_points[f"{p}GH label valid ratio"] = query_valid_mask.float().mean().reshape(())
            end_points["kview_debug_query_valid_mask0"] = query_valid_mask[0].detach().cpu() if B > 0 else query_valid_mask.detach().cpu()

        if angle_label is not None:
            ah_all = _bincount_float(angle_label, A + 1)
            _add_hist_to_endpoints(end_points, "kview_debug_label_angle_hist_all", ah_all, f"{p}GH label angle all", invalid_idx=A)

            if query_valid_mask is not None and bool(query_valid_mask.any()):
                ah = _bincount_float(angle_label[query_valid_mask], A + 1)
            else:
                ah = ah_all

            # Historical key now means valid-mask filtered angle label histogram when
            # batch_valid_mask exists; this matches compute_angle_loss().
            _add_hist_to_endpoints(end_points, "kview_debug_label_angle_hist", ah, f"{p}GH label angle", invalid_idx=A)
            end_points["kview_debug_label_angle0"] = angle_label[0].detach().cpu() if B > 0 else angle_label.detach().cpu()
        if depth_label is not None:
            dh = _bincount_float(depth_label, D + 1)
            _add_hist_to_endpoints(end_points, "kview_debug_label_depth_hist", dh, f"{p}GH label depth", invalid_idx=D)
            end_points[f"{p}GH label depth01 ratio"] = ((depth_label == 0) | (depth_label == 1)).float().mean().reshape(())
            end_points["kview_debug_label_depth0"] = depth_label[0].detach().cpu() if B > 0 else depth_label.detach().cpu()
        if score_label is not None:
            sh = _bincount_float(score_label, 6)
            _add_hist_to_endpoints(end_points, "kview_debug_label_score_hist", sh, f"{p}GH label score")
            end_points["kview_debug_label_score0"] = score_label[0].detach().cpu() if B > 0 else score_label.detach().cpu()

        # Histograms conditioned on selected-view GT quality, when available.
        selected_gt = end_points.get("kview_query_selected_gt", None)
        if torch.is_tensor(selected_gt) and selected_gt.shape == (B, Q):
            bins_gt = [
                ("lt01", selected_gt < 0.1),
                ("01_03", (selected_gt >= 0.1) & (selected_gt < 0.3)),
                ("03_05", (selected_gt >= 0.3) & (selected_gt < 0.5)),
                ("gt05", selected_gt >= 0.5),
            ]
            for name, mask in bins_gt:
                if bool(mask.any()):
                    end_points[f"{p}GH pred angle0 {name}"] = (angle_arg[mask] == 0).float().mean().reshape(())
                    end_points[f"{p}GH pred depth01 {name}"] = (((depth_arg[mask] == 0) | (depth_arg[mask] == 1)).float().mean()).reshape(())
                    end_points[f"{p}GH score expected {name}"] = score_expected[mask].float().mean().reshape(())

                    # Label angle-bin-0 ratio in the same selected-view-GT bin.
                    # If batch_valid_mask exists, restrict to valid labels because c2_2
                    # angle loss ignores invalid queries.
                    if angle_label is not None:
                        lmask = mask
                        if query_valid_mask is not None:
                            lmask = lmask & query_valid_mask
                        if bool(lmask.any()):
                            end_points[f"{p}GH label angle0 {name}"] = (angle_label[lmask] == 0).float().mean().reshape(())
                            end_points[f"{p}GH label angle invalid {name}"] = (angle_label[lmask] == A).float().mean().reshape(())
                        else:
                            end_points[f"{p}GH label angle0 {name}"] = score_expected.new_tensor(0.0).reshape(())
                            end_points[f"{p}GH label angle invalid {name}"] = score_expected.new_tensor(0.0).reshape(())
                else:
                    z = score_expected.new_tensor(0.0).reshape(())
                    end_points[f"{p}GH pred angle0 {name}"] = z
                    end_points[f"{p}GH pred depth01 {name}"] = z
                    end_points[f"{p}GH score expected {name}"] = z
                    if angle_label is not None:
                        end_points[f"{p}GH label angle0 {name}"] = z
                        end_points[f"{p}GH label angle invalid {name}"] = z

        if view_quality is not None:
            vq = torch.sigmoid(view_quality.squeeze(1))
            end_points[f"{p}GH view quality"] = vq.mean().reshape(())
            end_points[f"{p}GH score expected x viewq"] = (score_expected * vq).mean().reshape(())
            end_points["kview_debug_view_quality0"] = vq[0].detach().cpu() if B > 0 else vq.detach().cpu()

        rank = end_points.get("kview_query_view_rank", None)
        if torch.is_tensor(rank) and rank.shape == score_expected.shape:
            for r in range(4):
                m = rank == r
                end_points[f"{p}GH score expected rank{r}"] = score_expected[m].mean().reshape(()) if bool(m.any()) else score_expected.new_tensor(0.0).reshape(())

    def forward(self, group_features: torch.Tensor, end_points: Dict[str, Any]) -> Dict[str, Any]:
        B, _, Q = group_features.shape
        k_view = int(end_points.get("kview_effective_k_int", 1))
        k_view = max(k_view, 1)

        x = self.input_proj(group_features).transpose(1, 2).contiguous()
        x = self.input_norm(x)
        x = self._add_query_embeddings(x, end_points)
        for layer in self.layers:
            x = layer(x, k_view=k_view)
        x = x.transpose(1, 2).contiguous()

        angle_features = self.conv_angle_feature(x)
        depth_features = self.conv_depth_feature(x)
        width_features = self.conv_width_feature(x)
        score_features = self.conv_score_feature(x)
        angle_features, depth_features, width_features, score_features = self._interact_four_branches(angle_features, depth_features, width_features, score_features)

        angle_logits = self.conv_angle(self.angle_dropout(angle_features))
        depth_logits = self.conv_depth(self.depth_dropout(depth_features))
        angle_prob = F.softmax(angle_logits, dim=1)
        depth_prob = F.softmax(depth_logits, dim=1)
        angle_ctx = self.angle_ctx_proj(angle_prob)
        depth_ctx = self.depth_ctx_proj(depth_prob)

        width_input = torch.cat([width_features, score_features, angle_ctx, depth_ctx], dim=1)
        width_features = width_features + self.width_fuse(width_input)
        width_pred = self.conv_width(self.width_dropout(width_features))

        score_input = torch.cat([score_features, width_features, angle_ctx, depth_ctx], dim=1)
        score_features = score_features + self.score_fuse(score_input)
        score_pred = self.conv_score(self.score_dropout(score_features))

        view_quality = None
        if self.conv_view_quality is not None:
            view_quality = self.conv_view_quality(self.view_quality_dropout(score_features))
            end_points["kview_view_quality_pred"] = view_quality

        end_points["grasp_angle_pred"] = angle_logits
        end_points["grasp_depth_pred"] = depth_logits
        end_points["grasp_score_pred"] = score_pred
        end_points["grasp_width_pred"] = width_pred
        self._add_debug(end_points, angle_logits, depth_logits, score_pred, width_pred, view_quality)
        return end_points


# -----------------------------------------------------------------------------
# Label helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def _backup_base_view_labels(end_points: Dict[str, Any]) -> Dict[str, Any]:
    if "batch_grasp_view_graspness" in end_points:
        end_points["batch_grasp_view_graspness_base"] = end_points["batch_grasp_view_graspness"].detach()
    return end_points


@torch.no_grad()
def _restore_base_view_labels_for_view_loss(end_points: Dict[str, Any]) -> Dict[str, Any]:
    if "batch_grasp_view_graspness" in end_points:
        end_points["batch_grasp_view_graspness_query"] = end_points["batch_grasp_view_graspness"].detach()
    if "batch_grasp_view_graspness_base" in end_points:
        end_points["batch_grasp_view_graspness"] = end_points["batch_grasp_view_graspness_base"]
    return end_points


@torch.no_grad()
def _add_kview_selected_view_label_debug(end_points: Dict[str, Any], prefix: str = "D: KV") -> Dict[str, Any]:
    label = end_points.get("batch_grasp_view_graspness_query", None)
    if label is None:
        label = end_points.get("batch_grasp_view_graspness", None)
    inds = end_points.get("grasp_top_view_inds", None)
    if not (torch.is_tensor(label) and torch.is_tensor(inds)):
        return end_points
    if label.dim() != 3 or inds.dim() != 2 or label.shape[:2] != inds.shape:
        return end_points
    V = label.shape[-1]
    selected_gt = torch.gather(label, dim=-1, index=inds.long().clamp(0, V - 1).unsqueeze(-1)).squeeze(-1)
    oracle_gt = label.max(dim=-1).values
    valid = torch.isfinite(selected_gt) & torch.isfinite(oracle_gt) & (oracle_gt > 1e-6)

    # Per-query values for selector visualization / later endpoint diagnosis.
    end_points["kview_query_selected_gt"] = selected_gt.detach()
    end_points["kview_query_oracle_gt"] = oracle_gt.detach()
    end_points["kview_query_view_regret"] = (oracle_gt - selected_gt).clamp_min(0.0).detach()

    # Histogram selected-GT into [0,.1), [.1,.3), [.3,.5), [.5,1].
    if bool(valid.any()):
        sg = selected_gt[valid].float()
        bins = torch.zeros(4, device=label.device, dtype=torch.float32)
        bins[0] = (sg < 0.1).float().sum()
        bins[1] = ((sg >= 0.1) & (sg < 0.3)).float().sum()
        bins[2] = ((sg >= 0.3) & (sg < 0.5)).float().sum()
        bins[3] = (sg >= 0.5).float().sum()
        end_points["kview_debug_selected_gt_hist"] = bins.detach().cpu()
        end_points[f"{prefix}Q selected GT"] = selected_gt[valid].mean().reshape(())
        end_points[f"{prefix}Q oracle GT"] = oracle_gt[valid].mean().reshape(())
        end_points[f"{prefix}Q view regret"] = (oracle_gt[valid] - selected_gt[valid]).clamp_min(0.0).mean().reshape(())
        end_points[f"{prefix}Q selected GT>0.1"] = (selected_gt[valid] > 0.1).float().mean().reshape(())
        end_points[f"{prefix}Q selected GT>0.3"] = (selected_gt[valid] > 0.3).float().mean().reshape(())
        end_points[f"{prefix}Q selected GT>0.5"] = (selected_gt[valid] > 0.5).float().mean().reshape(())
    else:
        z = label.new_tensor(0.0).reshape(())
        end_points["kview_debug_selected_gt_hist"] = torch.zeros(4, device=label.device, dtype=torch.float32).cpu()
        end_points[f"{prefix}Q selected GT"] = z
        end_points[f"{prefix}Q oracle GT"] = z
        end_points[f"{prefix}Q view regret"] = z
    return end_points


def _call_label_process(
    fn: Callable[..., Any],
    end_points: Dict[str, Any],
    kwargs: Optional[Dict[str, Any]],
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    kwargs = kwargs or {}
    out = fn(end_points, **kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    if isinstance(out, dict):
        return out.get("grasp_top_view_rot", None), out
    raise RuntimeError("label_process_fn must return (grasp_top_view_rot, end_points) or end_points")


# -----------------------------------------------------------------------------
# Complete module
# -----------------------------------------------------------------------------


class KViewQueryTransformerLocalGraspModule(nn.Module):
    """Complete replacement for view -> 2D local grouping -> grasp head.

    The module owns:
      - view_net
      - K-view selector
      - view-conditioned cross-attention grouping
      - K-view transformer grasp head
      - debug/visualization dumps

    It does not own process_grasp_labels because your model may use the normal
    or depth-compensated label function. Pass the chosen function at forward.
    """

    def __init__(
        self,
        view_net: nn.Module,
        num_view: int,
        num_angle: int,
        num_depth: int,
        seed_feature_dim: int,
        feat_dim: int,
        view_dirs: torch.Tensor,
        batch_viewpoint_params_to_matrix_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: Optional[KViewQueryTransformerConfig] = None,
    ):
        super().__init__()
        self.view = view_net
        self.num_view = int(num_view)
        self.num_angle = int(num_angle)
        self.num_depth = int(num_depth)
        self.seed_feature_dim = int(seed_feature_dim)
        self.feat_dim = int(feat_dim)
        self.config = config if config is not None else KViewQueryTransformerConfig()
        self._vis_iter = 0
        if self.config.vis_dir is not None:
            os.makedirs(self.config.vis_dir, exist_ok=True)

        self.selector = KViewQuerySelector(
            num_view=self.num_view,
            view_dirs=view_dirs,
            batch_viewpoint_params_to_matrix_fn=batch_viewpoint_params_to_matrix_fn,
            config=self.config,
        )
        self.group = ViewConditionedAttentionGrouping(
            seed_feature_dim=self.seed_feature_dim,
            feat_dim=self.feat_dim,
            out_dim=int(self.config.head_model_dim),
            config=self.config,
        )
        self.head = KViewQueryTransformerGraspHeadDropout(
            num_angle=self.num_angle,
            num_depth=self.num_depth,
            num_view=self.num_view,
            in_dim=int(self.config.head_model_dim),
            config=self.config,
        )

    def _call_view_net(
        self,
        seed_features: torch.Tensor,
        token_sel_idx: torch.Tensor,
        camera_K: torch.Tensor,
        depth_map: torch.Tensor,
        depth_prob: Optional[torch.Tensor],
        end_points: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        # Preferred signature used by GeometryAwareDenseFieldViewNet in your pasted code.
        try:
            return self.view(
                seed_features=seed_features,
                token_sel_idx=token_sel_idx,
                K=camera_K,
                depth_map=depth_map,
                depth_prob=depth_prob,
                end_points=end_points,
            )
        except TypeError:
            # Compatibility for the original ViewNet(seed_features, end_points).
            return self.view(seed_features, end_points)

    @torch.no_grad()
    def _save_debug(self, end_points: Dict[str, Any], img: Optional[torch.Tensor], image_hw: Tuple[int, int]) -> None:
        """Save module-level selector/grouping/head debug artifacts.

        Files:
          - kview_full_debug_itXXXXXX.npz
          - kview_selector_hist_itXXXXXX.png
          - kview_selector_overlay_itXXXXXX.png
          - kview_grouping_attn_overlay_itXXXXXX.png
          - kview_grouping_hist_itXXXXXX.png
          - kview_head_hist_itXXXXXX.png
          - kview_head_entropy_itXXXXXX.png
        """
        if self.config.vis_dir is None or img is None:
            return
        if self._vis_iter % int(self.config.vis_every) != 0:
            return
        if not _rank0_only():
            return
        os.makedirs(self.config.vis_dir, exist_ok=True)
        H, W = image_hw

        # ------------------------------------------------------------------
        # 1) NPZ dump: selector + grouping + head internals.
        # ------------------------------------------------------------------
        keys = [
            # selector / query
            "token_sel_idx", "xyz_graspable", "grasp_top_view_inds", "grasp_top_view_xyz",
            "kview_query_parent", "kview_query_view_inds", "kview_query_view_rank",
            "kview_query_view_prob", "kview_query_selected_gt", "kview_query_oracle_gt",
            "kview_query_view_regret", "kview_debug_selected_view_hist", "kview_debug_top1_view_hist",
            "kview_debug_selected_gt_hist", "kview_debug_view_entropy0", "kview_debug_view_margin0",
            # grouping / white-dot attention
            "kview_debug_patch_uv0", "kview_debug_patch_attn0", "kview_debug_patch_fg0",
            "kview_debug_patch_graspness0", "kview_debug_patch_depth_delta0", "kview_debug_patch_valid0",
            "kview_debug_center_uv0", "kview_debug_center_proj_uv0", "kview_debug_center_proj_err0",
            "kview_debug_radius_px0", "kview_debug_vec_y0", "kview_debug_vec_z0",
            # head predictions / labels
            "kview_view_quality_pred", "grasp_angle_pred", "grasp_depth_pred", "grasp_score_pred", "grasp_width_pred",
            "kview_debug_pred_angle_hist", "kview_debug_pred_depth_hist", "kview_debug_pred_score_hist",
            "kview_debug_label_angle_hist", "kview_debug_label_depth_hist", "kview_debug_label_score_hist",
            "kview_debug_pred_angle_argmax", "kview_debug_pred_depth_argmax", "kview_debug_pred_score_argmax",
            "kview_debug_label_angle0", "kview_debug_label_depth0", "kview_debug_label_score0",
            "kview_debug_query_valid_mask0", "kview_debug_label_angle_hist_all",
            "kview_debug_angle_entropy0", "kview_debug_depth_entropy0", "kview_debug_score_entropy0",
            "kview_debug_score_expected0", "kview_debug_view_quality0",
        ]
        if self.config.save_npz:
            dump = {}
            for k in keys:
                v = end_points.get(k, None)
                arr = _tensor_to_numpy(v)
                if arr is not None:
                    dump[k] = arr[0] if arr.ndim > 0 and k in ["grasp_angle_pred", "grasp_depth_pred", "grasp_score_pred", "grasp_width_pred", "kview_view_quality_pred"] else arr
            # Also dump scalar D: KV* entries for convenient inspection.
            for k, v in end_points.items():
                if isinstance(k, str) and k.startswith("D: KV") and torch.is_tensor(v):
                    arr = _tensor_to_numpy(v)
                    if arr is not None:
                        dump[k.replace("D: ", "D__").replace(" ", "_").replace(":", "_")] = arr
            if dump:
                np.savez_compressed(os.path.join(self.config.vis_dir, f"kview_full_debug_it{self._vis_iter:06d}.npz"), **dump)

        # Prepare image canvas.
        try:
            import cv2
            x = img[0].detach().float().cpu()
            x = x - x.min()
            x = x / (x.max() + 1e-6)
            base_canvas = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[..., ::-1].copy()
        except Exception:
            base_canvas = None

        # ------------------------------------------------------------------
        # 2) KView selector visualization.
        # ------------------------------------------------------------------
        if base_canvas is not None:
            try:
                import cv2
                canvas = base_canvas.copy()
                idx = end_points.get("token_sel_idx", None)
                prob = end_points.get("kview_query_view_prob", None)
                selected_gt = end_points.get("kview_query_selected_gt", None)
                rank = end_points.get("kview_query_view_rank", None)
                if torch.is_tensor(idx):
                    idx0 = idx[0].detach().long().cpu().numpy()
                    prob0 = prob[0].detach().float().cpu().numpy() if torch.is_tensor(prob) and prob.shape[:1] else np.ones_like(idx0, dtype=np.float32)
                    gt0 = selected_gt[0].detach().float().cpu().numpy() if torch.is_tensor(selected_gt) and selected_gt.shape == idx.shape else None
                    rank0 = rank[0].detach().long().cpu().numpy() if torch.is_tensor(rank) and rank.shape == idx.shape else np.zeros_like(idx0)
                    n = min(idx0.shape[0], int(self.config.vis_num_queries))
                    # Use a deterministic subset so overlay corresponds to saved patch preview when possible.
                    sel = np.arange(n, dtype=np.int64)
                    vals = gt0[sel] if gt0 is not None else prob0[sel]
                    vmax = float(np.nanmax(vals)) if vals.size else 1.0
                    vmax = max(vmax, 1e-6)
                    for t, val, r in zip(idx0[sel], vals, rank0[sel]):
                        u, v = int(t % W), int(t // W)
                        if 0 <= u < W and 0 <= v < H:
                            # Color by selected GT if available, otherwise selected probability.
                            cval = int(np.clip(float(val) / vmax, 0.0, 1.0) * 255)
                            color = tuple(int(x) for x in cv2.applyColorMap(np.array([[cval]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0])
                            radius = 2 if int(r) >= 0 else 1
                            cv2.circle(canvas, (u, v), radius, color, thickness=-1)
                cv2.imwrite(os.path.join(self.config.vis_dir, f"kview_selector_overlay_it{self._vis_iter:06d}.png"), canvas)
            except Exception:
                pass

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            ax = axes.reshape(-1)
            for a in ax:
                a.grid(True, alpha=0.2)
            sh = _tensor_to_numpy(end_points.get("kview_debug_selected_view_hist", None))
            th = _tensor_to_numpy(end_points.get("kview_debug_top1_view_hist", None))
            gh = _tensor_to_numpy(end_points.get("kview_debug_selected_gt_hist", None))
            if sh is not None:
                ax[0].bar(np.arange(len(sh)), sh)
                ax[0].set_title("selected/sampled view histogram")
                ax[0].set_xlabel("view idx")
            if th is not None:
                ax[1].bar(np.arange(len(th)), th)
                ax[1].set_title("top1 view histogram")
                ax[1].set_xlabel("view idx")
            if gh is not None:
                ax[2].bar(["<.1", ".1-.3", ".3-.5", ">=.5"], gh)
                ax[2].set_title("selected view GT bins")
            sg = _tensor_to_numpy(end_points.get("kview_query_selected_gt", None))
            if sg is not None:
                ax[3].hist(sg.reshape(-1), bins=30, range=(0, 1))
                ax[3].set_title("selected view GT distribution")
            fig.tight_layout()
            fig.savefig(os.path.join(self.config.vis_dir, f"kview_selector_hist_it{self._vis_iter:06d}.png"), dpi=160)
            plt.close(fig)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 3) White-dot / grouping visualization.
        # ------------------------------------------------------------------
        if base_canvas is not None:
            try:
                import cv2
                canvas = base_canvas.copy()
                patch_uv = _tensor_to_numpy(end_points.get("kview_debug_patch_uv0", None))
                patch_attn = _tensor_to_numpy(end_points.get("kview_debug_patch_attn0", None))
                patch_fg = _tensor_to_numpy(end_points.get("kview_debug_patch_fg0", None))
                center_uv = _tensor_to_numpy(end_points.get("kview_debug_center_uv0", None))
                center_proj = _tensor_to_numpy(end_points.get("kview_debug_center_proj_uv0", None))
                vec_y = _tensor_to_numpy(end_points.get("kview_debug_vec_y0", None))
                vec_z = _tensor_to_numpy(end_points.get("kview_debug_vec_z0", None))
                if patch_uv is not None:
                    n_patch = min(patch_uv.shape[0], int(getattr(self.config, "vis_num_patch_queries", 16)))
                    for qi in range(n_patch):
                        pts = patch_uv[qi].reshape(-1, 2)
                        aw = patch_attn[qi].reshape(-1) if patch_attn is not None and qi < patch_attn.shape[0] else np.ones((pts.shape[0],), dtype=np.float32)
                        fg = patch_fg[qi].reshape(-1) if patch_fg is not None and qi < patch_fg.shape[0] else None
                        denom = float(np.nanmax(aw) - np.nanmin(aw)) if aw.size else 0.0
                        awn = (aw - np.nanmin(aw)) / max(denom, 1e-6)
                        # Draw every point; color/size by attention, whiteness by fg if available.
                        for (u, v), av, j in zip(pts, awn, range(len(pts))):
                            uu, vv = int(round(float(u))), int(round(float(v)))
                            if 0 <= uu < W and 0 <= vv < H:
                                cval = int(np.clip(float(av), 0.0, 1.0) * 255)
                                color = cv2.applyColorMap(np.array([[cval]], dtype=np.uint8), cv2.COLORMAP_HOT)[0, 0]
                                if fg is not None:
                                    # Darken low foreground points, keep high-attn points visible.
                                    alpha = 0.35 + 0.65 * float(np.clip(fg[j], 0.0, 1.0))
                                    color = tuple(int(float(c) * alpha) for c in color)
                                rad = 1 + int(float(av) > 0.65)
                                cv2.circle(canvas, (uu, vv), rad, tuple(int(c) for c in color), thickness=-1)
                        if center_uv is not None and qi < center_uv.shape[0]:
                            cu, cv = int(round(center_uv[qi, 0])), int(round(center_uv[qi, 1]))
                            if 0 <= cu < W and 0 <= cv < H:
                                cv2.circle(canvas, (cu, cv), 3, (0, 0, 255), thickness=-1)
                                if vec_y is not None and qi < vec_y.shape[0]:
                                    py = (int(round(cu + vec_y[qi, 0])), int(round(cv + vec_y[qi, 1])))
                                    cv2.line(canvas, (cu, cv), py, (0, 255, 255), 1)
                                if vec_z is not None and qi < vec_z.shape[0]:
                                    pz = (int(round(cu + vec_z[qi, 0])), int(round(cv + vec_z[qi, 1])))
                                    cv2.line(canvas, (cu, cv), pz, (0, 255, 0), 1)
                        if center_proj is not None and center_uv is not None and qi < center_proj.shape[0]:
                            cuv = tuple(int(round(x)) for x in center_uv[qi])
                            puv = tuple(int(round(x)) for x in center_proj[qi])
                            if 0 <= puv[0] < W and 0 <= puv[1] < H and 0 <= cuv[0] < W and 0 <= cuv[1] < H:
                                cv2.circle(canvas, puv, 2, (255, 255, 0), thickness=-1)
                                cv2.line(canvas, cuv, puv, (255, 255, 0), 1)
                cv2.imwrite(os.path.join(self.config.vis_dir, f"kview_grouping_attn_overlay_it{self._vis_iter:06d}.png"), canvas)
            except Exception:
                pass

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            ax = axes.reshape(-1)
            for a in ax:
                a.grid(True, alpha=0.2)
            cpe = _tensor_to_numpy(end_points.get("kview_debug_center_proj_err0", None))
            rad = _tensor_to_numpy(end_points.get("kview_debug_radius_px0", None))
            aw = _tensor_to_numpy(end_points.get("kview_debug_patch_attn0", None))
            fg = _tensor_to_numpy(end_points.get("kview_debug_patch_fg0", None))
            gp = _tensor_to_numpy(end_points.get("kview_debug_patch_graspness0", None))
            dd = _tensor_to_numpy(end_points.get("kview_debug_patch_depth_delta0", None))
            if cpe is not None:
                ax[0].hist(cpe.reshape(-1), bins=30)
                ax[0].set_title("center projection error px")
            if rad is not None:
                ax[1].hist(rad.reshape(-1), bins=30)
                ax[1].set_title("patch radius px")
            if aw is not None:
                ax[2].hist(aw.reshape(-1), bins=40)
                ax[2].set_title("attention weights")
            if fg is not None:
                ax[3].hist(fg.reshape(-1), bins=30, range=(0, 1))
                ax[3].set_title("patch foreground prob")
            if gp is not None:
                ax[4].hist(gp.reshape(-1), bins=30)
                ax[4].set_title("patch graspness")
            if dd is not None:
                ax[5].hist(np.clip(np.abs(dd.reshape(-1)), 0, 5), bins=30)
                ax[5].set_title("abs depth delta")
            fig.tight_layout()
            fig.savefig(os.path.join(self.config.vis_dir, f"kview_grouping_hist_it{self._vis_iter:06d}.png"), dpi=160)
            plt.close(fig)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 4) Head internal visualization.
        # ------------------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            ax = axes.reshape(-1)
            for a in ax:
                a.grid(True, alpha=0.2)
            pairs = [
                ("angle", "kview_debug_pred_angle_hist", "kview_debug_label_angle_hist"),
                ("depth", "kview_debug_pred_depth_hist", "kview_debug_label_depth_hist"),
                ("score", "kview_debug_pred_score_hist", "kview_debug_label_score_hist"),
            ]
            for j, (title, pred_key, label_key) in enumerate(pairs):
                ph = _tensor_to_numpy(end_points.get(pred_key, None))
                lh = _tensor_to_numpy(end_points.get(label_key, None))
                if ph is not None:
                    xloc = np.arange(len(ph))
                    ax[j].bar(xloc - 0.18, ph, width=0.35, label="pred")
                    if lh is not None and len(lh) == len(ph):
                        ax[j].bar(xloc + 0.18, lh, width=0.35, label="label", alpha=0.75)
                    ax[j].set_title(f"{title} histogram")
                    ax[j].legend(fontsize=8)
            for j, key in enumerate(["kview_debug_angle_entropy0", "kview_debug_depth_entropy0", "kview_debug_score_entropy0"], start=3):
                arr = _tensor_to_numpy(end_points.get(key, None))
                if arr is not None:
                    ax[j].hist(arr.reshape(-1), bins=40)
                    ax[j].set_title(key.replace("kview_debug_", ""))
            fig.tight_layout()
            fig.savefig(os.path.join(self.config.vis_dir, f"kview_head_hist_it{self._vis_iter:06d}.png"), dpi=160)
            plt.close(fig)

            # Entropy / score scatter for batch item 0.
            ae = _tensor_to_numpy(end_points.get("kview_debug_angle_entropy0", None))
            de = _tensor_to_numpy(end_points.get("kview_debug_depth_entropy0", None))
            se = _tensor_to_numpy(end_points.get("kview_debug_score_expected0", None))
            if ae is not None and de is not None and se is not None:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].scatter(ae.reshape(-1), se.reshape(-1), s=3, alpha=0.35)
                ax[0].set_xlabel("angle entropy")
                ax[0].set_ylabel("score expected")
                ax[0].grid(True, alpha=0.2)
                ax[1].scatter(de.reshape(-1), se.reshape(-1), s=3, alpha=0.35)
                ax[1].set_xlabel("depth entropy")
                ax[1].set_ylabel("score expected")
                ax[1].grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(os.path.join(self.config.vis_dir, f"kview_head_entropy_it{self._vis_iter:06d}.png"), dpi=160)
                plt.close(fig)
        except Exception:
            pass

    @torch.no_grad()
    def _cache_kview_query_selected_gt(self, end_points):
        """
        Must be called after query-level process_grasp_labels(),
        before restoring base-M labels for view loss.
        """
        view_label = end_points["batch_grasp_view_graspness"]  # [B, Q, V]
        view_inds = end_points["grasp_top_view_inds"].long()   # [B, Q]

        if view_label.dim() != 3:
            raise RuntimeError(
                f"batch_grasp_view_graspness must be [B,Q,V], got {tuple(view_label.shape)}"
            )
        if view_inds.shape != view_label.shape[:2]:
            raise RuntimeError(
                f"grasp_top_view_inds shape {tuple(view_inds.shape)} does not match "
                f"batch_grasp_view_graspness[:2] {tuple(view_label.shape[:2])}"
            )

        selected_gt = torch.gather(
            view_label,
            dim=2,
            index=view_inds.unsqueeze(-1).clamp(0, view_label.shape[2] - 1),
        ).squeeze(-1)

        end_points["kview_query_selected_gt"] = selected_gt.detach()
        return end_points
    
    def forward(
        self,
        seed_features: torch.Tensor,
        seed_xyz: torch.Tensor,
        token_sel_idx: torch.Tensor,
        feat_map: torch.Tensor,
        depth_map: torch.Tensor,
        camera_K: torch.Tensor,
        end_points: Dict[str, Any],
        is_training: bool,
        process_grasp_labels_fn: Optional[Callable[..., Any]] = None,
        process_grasp_labels_kwargs: Optional[Dict[str, Any]] = None,
        topview_debug_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        depth_prob: Optional[torch.Tensor] = None,
        objectness_logits: Optional[torch.Tensor] = None,
        graspness_map: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run view prediction, K-view selection, attention grouping, and head.

        Args:
            seed_features: [B,C,M]
            seed_xyz:      [B,M,3]
            token_sel_idx: [B,M]
            feat_map:      [B,Cf,H,W]
            depth_map:     [B,1,H,W]
            camera_K:      [B,3,3]
            process_grasp_labels_fn:
                process_grasp_labels or process_grasp_labels_depth_cls_compensated.
                Required during training if your loss expects labels.
        """
        B, _, H, W = feat_map.shape

        # 1) View field.
        end_points, res_feat = self._call_view_net(
            seed_features=seed_features,
            token_sel_idx=token_sel_idx,
            camera_K=camera_K,
            depth_map=depth_map,
            depth_prob=depth_prob,
            end_points=end_points,
        )

        # Preserve the ViewNet-produced view index before any later query expansion
        # overwrites grasp_top_view_inds. In training, GeometryAwareDenseFieldViewNet
        # uses its own multinomial branch; A1/A2 can reuse this exact sampled index
        # to make the transformer version a clean replacement of only local analysis.
        viewnet_sampled_view_inds = end_points.get("grasp_top_view_inds", None)
        if torch.is_tensor(viewnet_sampled_view_inds):
            viewnet_sampled_view_inds = viewnet_sampled_view_inds.detach().clone()
            end_points["kview_viewnet_sampled_view_inds"] = viewnet_sampled_view_inds

        seed_features_base = seed_features + res_feat
        seed_xyz_base = seed_xyz
        token_sel_idx_base = token_sel_idx

        # Keep base state before query expansion.
        end_points["xyz_graspable"] = seed_xyz_base
        end_points["token_sel_idx"] = token_sel_idx_base
        end_points["token_sel_xyz"] = seed_xyz_base

        # 2) Base label pass for view loss and view diagnostics.
        if is_training and process_grasp_labels_fn is not None:
            _, end_points = _call_label_process(process_grasp_labels_fn, end_points, process_grasp_labels_kwargs)
            end_points = _backup_base_view_labels(end_points)
            if topview_debug_fn is not None:
                end_points = topview_debug_fn(end_points)

        # 3) K-view query selection / expansion.
        if "view_score" not in end_points:
            raise KeyError("KView module expects end_points['view_score'] from view_net.")
        forced_view_inds = None
        mode = str(self.config.mode).upper()
        if is_training and torch.is_tensor(viewnet_sampled_view_inds):
            if mode in ("A1", "A2") and bool(self.config.reuse_viewnet_sample_in_single_view_train):
                forced_view_inds = viewnet_sampled_view_inds
            elif mode == "A3" and bool(self.config.reuse_viewnet_sample_in_multiview_train):
                forced_view_inds = viewnet_sampled_view_inds

        seed_features_q, seed_xyz_q, token_sel_idx_q, view_rot_q, end_points = self.selector(
            seed_features=seed_features_base,
            seed_xyz=seed_xyz_base,
            token_sel_idx=token_sel_idx_base,
            view_score=end_points["view_score"],
            end_points=end_points,
            is_training=is_training,
            forced_view_inds=forced_view_inds,
        )

        # 4) Query-level label pass for angle/depth/width/score labels.
        if is_training and process_grasp_labels_fn is not None:
            _, end_points = _call_label_process(process_grasp_labels_fn, end_points, process_grasp_labels_kwargs)
            
            end_points = self._cache_kview_query_selected_gt(end_points)
            
            # Save query-level selected view label diagnostics, then restore only view labels.
            end_points["batch_grasp_view_graspness_query"] = end_points.get("batch_grasp_view_graspness", None)
            end_points = _add_kview_selected_view_label_debug(end_points, prefix=self.config.debug_prefix)
            end_points = _restore_base_view_labels_for_view_loss(end_points)
        elif not is_training:
            # In inference, selector already writes query-level grasp_top_view_rot.
            view_rot_q = end_points["grasp_top_view_rot"]

        # 5) View-conditioned local attention grouping.
        group_features = self.group(
            seed_features=seed_features_q,
            token_sel_idx=token_sel_idx_q,
            seed_xyz=seed_xyz_q,
            top_view_rot=view_rot_q,
            feat_map=feat_map,
            depth_map=depth_map,
            objectness_logits=objectness_logits,
            graspness_map=graspness_map,
            camera_K=camera_K,
            end_points=end_points,
        )

        # 6) K-view transformer grasp head.
        end_points = self.head(group_features, end_points)

        # 7) Internal visualization/debug dump.
        self._save_debug(end_points, img=img, image_hw=(H, W))
        self._vis_iter += 1
        return end_points


class SimpleAttentionModule(nn.Module):
    """Small wrapper around MultiheadAttention with residual + FFN.

    This is used only to let the A angle candidates of the same center compare
    with each other before producing candidate score/depth/width.
    """
    def __init__(self, dim: int, n_head: int = 4, dropout: float = 0.05, ffn_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,A,C]
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class CenterViewAngleCandidateDecoder(nn.Module):
    """Simple D4RT-style query decoder for center-view-angle candidates.

    Input:
        group_features_angle: [B,C,Q*A]
        end_points['kview_angle_query_base_q'] = Q
        end_points['kview_angle_query_num_angle'] = A

    Candidate outputs:
        grasp_depth_pred_angle: [B,D+1,Q,A]
        grasp_score_pred_angle: [B,6,Q,A]
        grasp_width_pred_angle: [B,1,Q,A]

    Compatibility outputs are collapsed by score-expected selected angle.
    """
    def __init__(
        self,
        num_angle: int,
        num_depth: int,
        in_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_p: float = 0.15,
        attn_dropout: float = 0.05,
        debug_prefix: str = "D: CVA",
    ):
        super().__init__()
        self.num_angle = int(num_angle)
        self.num_depth = int(num_depth)
        self.hidden_dim = int(hidden_dim)
        self.debug_prefix = debug_prefix

        self.input_proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.angle_embed = nn.Embedding(self.num_angle, hidden_dim)
        self.layers = nn.ModuleList([
            SimpleAttentionModule(hidden_dim, n_head=num_heads, dropout=attn_dropout)
            for _ in range(int(num_layers))
        ])

        # EconomicGrasp-style lightweight branch interaction.
        self.depth_proj = nn.Conv1d(hidden_dim, 64, 1)
        self.width_proj = nn.Conv1d(hidden_dim, 64, 1)
        self.score_proj = nn.Conv1d(hidden_dim, 64, 1)
        self.branch_attn = nn.MultiheadAttention(64, 1, dropout=attn_dropout, batch_first=True)

        self.depth_dropout = nn.Dropout(dropout_p)
        self.width_dropout = nn.Dropout(dropout_p)
        self.score_dropout = nn.Dropout(dropout_p)
        self.depth_head = nn.Conv1d(64, self.num_depth + 1, 1)
        self.width_head = nn.Conv1d(64, 1, 1)
        self.score_head = nn.Conv1d(64, 6, 1)

    @staticmethod
    def _expected_score(score_logits_angle: torch.Tensor) -> torch.Tensor:
        # score_logits_angle: [B,6,Q,A]
        bins = torch.tensor(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            device=score_logits_angle.device,
            dtype=score_logits_angle.dtype,
        ).view(1, 6, 1, 1)
        prob = F.softmax(score_logits_angle, dim=1)
        return (prob * bins).sum(dim=1)  # [B,Q,A]

    @staticmethod
    def _gather_angle(x: torch.Tensor, angle_idx: torch.Tensor) -> torch.Tensor:
        """Gather candidate tensor along final angle dimension.

        x: [B,C,Q,A]
        angle_idx: [B,Q]
        return: [B,C,Q]
        """
        B, C, Q, A = x.shape
        idx = angle_idx.long().clamp(0, A - 1).view(B, 1, Q, 1).expand(B, C, Q, 1)
        return torch.gather(x, dim=-1, index=idx).squeeze(-1)

    @torch.no_grad()
    def _add_debug(self, end_points: Dict[str, Any], score_angle: torch.Tensor, depth_angle: torch.Tensor, selected_angle: torch.Tensor) -> None:
        p = self.debug_prefix
        score_expected = self._expected_score(score_angle)  # [B,Q,A]
        B, Q, A = score_expected.shape
        end_points[f"{p} score angle expected"] = score_expected.mean()
        end_points[f"{p} selected angle bin0 ratio"] = (selected_angle == 0).float().mean()
        end_points[f"{p} selected angle mean"] = selected_angle.float().mean()
        end_points[f"{p} score max-angle margin"] = (
            torch.topk(score_expected, k=min(2, A), dim=-1).values[..., 0]
            - torch.topk(score_expected, k=min(2, A), dim=-1).values[..., -1]
        ).mean()

        # Candidate depth shallow statistics under selected angle.
        depth_sel = self._gather_angle(depth_angle, selected_angle)  # [B,D+1,Q]
        depth_idx = torch.argmax(depth_sel[:, :self.num_depth, :], dim=1)
        end_points[f"{p} selected depth01 ratio"] = (depth_idx <= 1).float().mean()

        # If extended labels are available, compare score-selected angle to best-score GT angle.
        label_score = end_points.get("batch_grasp_score_angle", None)
        valid = end_points.get("batch_grasp_angle_valid_mask", None)
        if torch.is_tensor(label_score) and label_score.shape == (B, Q, A):
            best_label_angle = torch.argmax(label_score.float(), dim=-1)
            if torch.is_tensor(valid) and valid.shape == (B, Q, A):
                m = valid.any(dim=-1)
            else:
                m = torch.ones((B, Q), dtype=torch.bool, device=label_score.device)
            if bool(m.any()):
                end_points[f"{p} selected angle acc vs label-score-best"] = (selected_angle[m] == best_label_angle[m]).float().mean()
                label_hist = _bincount_float(best_label_angle[m], A)
                _add_hist_to_endpoints(end_points, "cva_debug_label_best_angle_hist", label_hist.cpu(), f"{p} label best angle")

    def forward(self, group_features_angle: torch.Tensor, end_points: Dict[str, Any]) -> Dict[str, Any]:
        B, C, QA = group_features_angle.shape
        Q = int(end_points["kview_angle_query_base_q"])
        A = int(end_points["kview_angle_query_num_angle"])
        if A != self.num_angle or QA != Q * A:
            raise RuntimeError(f"Expected QA=Q*A ({Q}*{A}), got {QA}.")

        x = self.input_proj(group_features_angle)  # [B,H,Q*A]
        x = x.transpose(1, 2).contiguous().view(B, Q, A, self.hidden_dim)
        angle_ids = torch.arange(A, device=x.device).view(1, 1, A)
        x = x + self.angle_embed(angle_ids)

        # D4RT-style: decoder answers a set of query tokens.  Here the set is the
        # A angle candidates of one center-view query.
        x = x.view(B * Q, A, self.hidden_dim)
        for layer in self.layers:
            x = layer(x)
        x = x.view(B, Q, A, self.hidden_dim).reshape(B, Q * A, self.hidden_dim).transpose(1, 2).contiguous()

        depth_feat = self.depth_proj(x)
        width_feat = self.width_proj(x)
        score_feat = self.score_proj(x)

        # 3 branch interaction per candidate: [B*Q*A,3,64]
        depth_tok = depth_feat.permute(0, 2, 1).reshape(B * Q * A, 1, 64)
        width_tok = width_feat.permute(0, 2, 1).reshape(B * Q * A, 1, 64)
        score_tok = score_feat.permute(0, 2, 1).reshape(B * Q * A, 1, 64)
        tokens = torch.cat([depth_tok, width_tok, score_tok], dim=1)
        tokens, _ = self.branch_attn(tokens, tokens, tokens, need_weights=False)
        depth_feat = tokens[:, 0].view(B, Q * A, 64).permute(0, 2, 1).contiguous()
        width_feat = tokens[:, 1].view(B, Q * A, 64).permute(0, 2, 1).contiguous()
        score_feat = tokens[:, 2].view(B, Q * A, 64).permute(0, 2, 1).contiguous()

        depth_logits_flat = self.depth_head(self.depth_dropout(depth_feat))  # [B,D+1,Q*A]
        score_logits_flat = self.score_head(self.score_dropout(score_feat))  # [B,6,Q*A]
        width_flat = self.width_head(self.width_dropout(width_feat))         # [B,1,Q*A]

        depth_angle = depth_logits_flat.view(B, self.num_depth + 1, Q, A)
        score_angle = score_logits_flat.view(B, 6, Q, A)
        width_angle = width_flat.view(B, 1, Q, A)

        # Candidate tensors for new extended-angle losses.
        end_points["grasp_depth_pred_angle"] = depth_angle
        end_points["grasp_score_pred_angle"] = score_angle
        end_points["grasp_width_pred_angle"] = width_angle

        # Inference/compatibility collapse: select angle by candidate expected score.
        score_expected = self._expected_score(score_angle)  # [B,Q,A]
        selected_angle = torch.argmax(score_expected, dim=-1)  # [B,Q]
        end_points["cva_selected_angle"] = selected_angle
        end_points["cva_score_expected_angle"] = score_expected.detach()

        # Pseudo angle prediction for existing pred_decode: valid angle logits = score expected.
        angle_logits_valid = score_expected.permute(0, 2, 1).contiguous()  # [B,A,Q]
        invalid = angle_logits_valid.new_full((B, 1, Q), -20.0)
        end_points["grasp_angle_pred"] = torch.cat([angle_logits_valid, invalid], dim=1)
        end_points["grasp_depth_pred"] = self._gather_angle(depth_angle, selected_angle)
        end_points["grasp_score_pred"] = self._gather_angle(score_angle, selected_angle)
        end_points["grasp_width_pred"] = self._gather_angle(width_angle, selected_angle)

        self._add_debug(end_points, score_angle, depth_angle, selected_angle)
        return end_points


class CenterViewAngleQueryTransformerLocalGraspModule(nn.Module):
    """Minimal center-view-angle query module.

    It reuses the existing ViewNet and KView selector.  After a view is selected,
    it enumerates A in-plane angles, performs angle-conditioned grouping, and
    decodes candidate depth/width/score.
    """
    def __init__(
        self,
        view_net: nn.Module,
        num_view: int,
        num_angle: int,
        num_depth: int,
        seed_feature_dim: int,
        feat_dim: int,
        view_dirs: torch.Tensor,
        batch_viewpoint_params_to_matrix_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: Optional[KViewQueryTransformerConfig] = None,
    ):
        super().__init__()
        self.view = view_net
        self.num_view = int(num_view)
        self.num_angle = int(num_angle)
        self.num_depth = int(num_depth)
        self.seed_feature_dim = int(seed_feature_dim)
        self.feat_dim = int(feat_dim)
        self.config = config if config is not None else KViewQueryTransformerConfig()
        self.batch_viewpoint_params_to_matrix_fn = batch_viewpoint_params_to_matrix_fn

        self._vis_iter = 0
        if self.config.vis_dir is not None:
            os.makedirs(self.config.vis_dir, exist_ok=True)

        self.selector = KViewQuerySelector(
            num_view=self.num_view,
            view_dirs=view_dirs,
            batch_viewpoint_params_to_matrix_fn=batch_viewpoint_params_to_matrix_fn,
            config=self.config,
        )
        self.group = ViewConditionedAttentionGrouping(
            seed_feature_dim=self.seed_feature_dim,
            feat_dim=self.feat_dim,
            out_dim=int(self.config.head_model_dim),
            config=self.config,
        )
        self.decoder = CenterViewAngleCandidateDecoder(
            num_angle=self.num_angle,
            num_depth=self.num_depth,
            in_dim=int(self.config.head_model_dim),
            hidden_dim=int(self.config.head_model_dim),
            num_layers=max(int(self.config.head_num_layers), 1),
            num_heads=max(int(self.config.head_num_heads), 1),
            dropout_p=float(self.config.head_dropout_p),
            attn_dropout=float(self.config.head_attn_dropout),
            debug_prefix="D: CVA",
        )

    def _call_view_net(self, seed_features, token_sel_idx, camera_K, depth_map, depth_prob, end_points):
        try:
            return self.view(
                seed_features=seed_features,
                token_sel_idx=token_sel_idx,
                K=camera_K,
                depth_map=depth_map,
                depth_prob=depth_prob,
                end_points=end_points,
            )
        except TypeError:
            return self.view(seed_features, end_points)

    def _expand_angle_queries(self, seed_features_q, seed_xyz_q, token_sel_idx_q, view_xyz_q, end_points):
        B, C, Q = seed_features_q.shape
        A = self.num_angle
        device = seed_features_q.device
        dtype = seed_xyz_q.dtype

        seed_features_a = seed_features_q.unsqueeze(-1).expand(B, C, Q, A).reshape(B, C, Q * A).contiguous()
        seed_xyz_a = seed_xyz_q.unsqueeze(2).expand(B, Q, A, 3).reshape(B, Q * A, 3).contiguous()
        token_sel_idx_a = token_sel_idx_q.unsqueeze(-1).expand(B, Q, A).reshape(B, Q * A).contiguous()
        view_xyz_a = view_xyz_q.unsqueeze(2).expand(B, Q, A, 3).reshape(B, Q * A, 3).contiguous()

        angle_ids = torch.arange(A, device=device).view(1, 1, A).expand(B, Q, A).reshape(B, Q * A)
        # Use the same angle discretization as EconomicGrasp pred_decode: idx * pi / 12 when A=12.
        angle_rad = angle_ids.to(dtype) * (np.pi / float(A))
        view_rot_a = self.batch_viewpoint_params_to_matrix_fn(
            -view_xyz_a.reshape(-1, 3),
            angle_rad.reshape(-1),
        ).view(B, Q * A, 3, 3)

        end_points["kview_angle_query_base_q"] = int(Q)
        end_points["kview_angle_query_num_angle"] = int(A)
        end_points["kview_angle_query_angle_ids"] = angle_ids
        end_points["D: CVA base Q"] = torch.tensor(float(Q), device=device)
        end_points["D: CVA angle A"] = torch.tensor(float(A), device=device)
        end_points["D: CVA Qangle"] = torch.tensor(float(Q * A), device=device)
        return seed_features_a, seed_xyz_a, token_sel_idx_a, view_rot_a, end_points

    @staticmethod
    def _angle_color(angle_id: int, num_angle: int) -> Tuple[int, int, int]:
        """BGR color for OpenCV drawing."""
        try:
            import cv2
            hue = int(round(179.0 * (int(angle_id) % max(int(num_angle), 1)) / max(float(num_angle), 1.0)))
            hsv = np.uint8([[[hue, 220, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            return int(bgr[0]), int(bgr[1]), int(bgr[2])
        except Exception:
            return (0, 255, 255)

    @staticmethod
    def _make_canvas(img0: torch.Tensor, out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create a BGR uint8 canvas from img[0]."""
        if img0 is None:
            return None
        try:
            import cv2
            x = img0.detach().float().cpu()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            if x.shape[0] > 3:
                x = x[:3]
            x = x - x.amin()
            x = x / (x.amax() + 1e-6)
            canvas = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            canvas = canvas[..., ::-1].copy()  # RGB -> BGR
            H, W = out_hw
            if canvas.shape[:2] != (H, W):
                canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)
            return canvas
        except Exception:
            return None

    @torch.no_grad()
    def _save_cva_visualization(
        self,
        end_points: Dict[str, Any],
        img: Optional[torch.Tensor],
        image_hw: Tuple[int, int],
    ) -> None:
        """Save PNG-only diagnostics for center-view-angle query decoding.

        Saved files:
          - cva_overlay_itXXXXXX.png: centers colored by score-selected angle;
            red rings mark disagreement with label best-angle when labels exist.
          - cva_angle_hist_itXXXXXX.png: selected-angle histogram vs label best-angle.
          - cva_score_heatmap_itXXXXXX.png: predicted score over angle for selected queries;
            label score over angle is shown below when available.

        No .npy/.npz files are written by design.
        """
        cfg = self.config
        if cfg.vis_dir is None:
            return
        if int(cfg.vis_every) <= 0:
            return
        if self._vis_iter % int(cfg.vis_every) != 0:
            return
        if not _rank0_only():
            return

        os.makedirs(cfg.vis_dir, exist_ok=True)
        H, W = image_hw

        selected_angle = end_points.get("cva_selected_angle", None)       # [B,Q]
        score_expected = end_points.get("cva_score_expected_angle", None) # [B,Q,A]
        token_idx = end_points.get("token_sel_idx", None)                 # [B,Q]
        patch_uv = end_points.get("kview_debug_patch_uv0", None)          # [Nqa,P,2], first batch preview
        label_score = end_points.get("batch_grasp_score_angle", None)     # [B,Q,A]
        valid_angle = end_points.get("batch_grasp_angle_valid_mask", None)

        if not (torch.is_tensor(selected_angle) and torch.is_tensor(score_expected)):
            return

        A = int(score_expected.shape[-1])
        B = int(score_expected.shape[0])
        Q = int(score_expected.shape[1])
        selected0 = selected_angle[0].detach().long().cpu()
        score0 = score_expected[0].detach().float().cpu()
        score_sel0 = torch.gather(score0, 1, selected0.view(-1, 1).clamp(0, A - 1)).squeeze(1)

        if torch.is_tensor(valid_angle) and tuple(valid_angle.shape[:3]) == tuple(score_expected.shape):
            valid_q0 = valid_angle[0].detach().bool().cpu().any(dim=-1)
        else:
            valid_q0 = torch.ones(Q, dtype=torch.bool)

        if torch.is_tensor(label_score) and tuple(label_score.shape[:3]) == tuple(score_expected.shape):
            label0 = label_score[0].detach().float().cpu()
            label_best0 = torch.argmax(label0, dim=-1)
            has_label = True
        else:
            label0 = None
            label_best0 = None
            has_label = False

        # Pick visualization query indices: top predicted-score valid queries.
        q_pool = torch.where(valid_q0)[0]
        nvis = min(max(int(cfg.vis_num_queries), 1), int(q_pool.numel()) if int(q_pool.numel()) > 0 else Q)
        if int(q_pool.numel()) > 0:
            pool_scores = score_sel0[q_pool]
            top_local = torch.topk(pool_scores, k=min(nvis, int(q_pool.numel())), largest=True).indices
            q_sel = q_pool[top_local]
        else:
            q_sel = torch.linspace(0, Q - 1, steps=nvis).long()

        # ------------------------------------------------------------------
        # 1) Overlay: centers colored by selected angle + selected-angle patch.
        # ------------------------------------------------------------------
        canvas = self._make_canvas(img[0], (H, W)) if torch.is_tensor(img) else None
        if canvas is not None:
            try:
                import cv2
                # Draw query centers.
                if torch.is_tensor(token_idx):
                    idx0 = token_idx[0].detach().long().cpu()
                    for q in q_sel.tolist():
                        t = int(idx0[q])
                        u, v = int(t % W), int(t // W)
                        if not (0 <= u < W and 0 <= v < H):
                            continue
                        a = int(selected0[q])
                        color = self._angle_color(a, A)
                        rad = 2 + int(min(max(float(score_sel0[q]) * 4.0, 0.0), 4.0))
                        cv2.circle(canvas, (u, v), rad, color, thickness=-1, lineType=cv2.LINE_AA)
                        # Red ring if selected angle disagrees with label-score-best.
                        if has_label and int(label_best0[q]) != a:
                            cv2.circle(canvas, (u, v), rad + 3, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                # Draw selected-angle patch tokens for the first few base queries
                # for which kview_debug_patch_uv0 is available.  The grouping
                # preview is ordered as q0_a0..q0_aA, q1_a0.. .
                if torch.is_tensor(patch_uv):
                    puv = patch_uv.detach().float().cpu().numpy()
                    max_base = min(int(puv.shape[0]) // max(A, 1), 8, Q)
                    for q in range(max_base):
                        a = int(selected0[q])
                        qa = q * A + a
                        if qa < puv.shape[0]:
                            pts = puv[qa].reshape(-1, 2)
                            color = self._angle_color(a, A)
                            step = max(1, int(pts.shape[0]) // 24)
                            for uu, vv in pts[::step]:
                                u, v = int(round(float(uu))), int(round(float(vv)))
                                if 0 <= u < W and 0 <= v < H:
                                    cv2.circle(canvas, (u, v), 1, color, thickness=-1, lineType=cv2.LINE_AA)

                txt = f"CVA it={self._vis_iter} Q={Q} A={A} mean_sel_score={float(score_sel0[valid_q0].mean()) if bool(valid_q0.any()) else float(score_sel0.mean()):.3f}"
                cv2.putText(canvas, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(cfg.vis_dir, f"cva_overlay_it{self._vis_iter:06d}.png"), canvas)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 2) Angle histograms and 3) score-vs-angle heatmaps.
        # ------------------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m = valid_q0
            if not bool(m.any()):
                m = torch.ones_like(valid_q0)

            # Histogram: selected angle, and label-best angle if available.
            fig, ax = plt.subplots(figsize=(7.5, 3.0), dpi=160)
            xs = np.arange(A)
            pred_hist = np.bincount(selected0[m].numpy(), minlength=A).astype(np.float32)
            pred_hist = pred_hist / max(pred_hist.sum(), 1.0)
            ax.bar(xs - (0.18 if has_label else 0.0), pred_hist, width=(0.36 if has_label else 0.7), label="selected")
            if has_label:
                lab_hist = np.bincount(label_best0[m].numpy(), minlength=A).astype(np.float32)
                lab_hist = lab_hist / max(lab_hist.sum(), 1.0)
                ax.bar(xs + 0.18, lab_hist, width=0.36, label="label-best")
                acc = float((selected0[m] == label_best0[m]).float().mean())
                ax.set_title(f"CVA selected-angle distribution, acc={acc:.3f}")
            else:
                ax.set_title("CVA selected-angle distribution")
            ax.set_xlabel("angle bin")
            ax.set_ylabel("ratio")
            ax.set_xticks(xs)
            ax.set_ylim(0.0, max(float(pred_hist.max()) * 1.25, 0.05))
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(cfg.vis_dir, f"cva_angle_hist_it{self._vis_iter:06d}.png"))
            plt.close(fig)

            # Heatmap for up to 64 high-score queries.
            nheat = min(64, int(q_sel.numel()))
            qh = q_sel[:nheat]
            pred_h = score0[qh].numpy()  # [N,A]
            if has_label:
                lab_h = label0[qh].numpy()
                fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), dpi=160, sharex=True)
                im0 = axes[0].imshow(pred_h, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
                axes[0].set_title("predicted expected score over angle")
                axes[0].set_ylabel("query")
                im1 = axes[1].imshow(lab_h, aspect="auto", interpolation="nearest", vmin=0.0, vmax=max(1.0, float(np.nanmax(lab_h)) if lab_h.size else 1.0))
                axes[1].set_title("label score over angle")
                axes[1].set_ylabel("query")
                axes[1].set_xlabel("angle bin")
                fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.02)
                fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.02)
            else:
                fig, ax = plt.subplots(figsize=(8.0, 2.8), dpi=160)
                im0 = ax.imshow(pred_h, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
                ax.set_title("predicted expected score over angle")
                ax.set_ylabel("query")
                ax.set_xlabel("angle bin")
                fig.colorbar(im0, ax=ax, fraction=0.025, pad=0.02)
            fig.tight_layout()
            fig.savefig(os.path.join(cfg.vis_dir, f"cva_score_heatmap_it{self._vis_iter:06d}.png"))
            plt.close(fig)
        except Exception:
            return

    def forward(
        self,
        seed_features: torch.Tensor,
        seed_xyz: torch.Tensor,
        token_sel_idx: torch.Tensor,
        feat_map: torch.Tensor,
        depth_map: torch.Tensor,
        camera_K: torch.Tensor,
        end_points: Dict[str, Any],
        is_training: bool,
        process_grasp_labels_fn: Optional[Callable[..., Any]] = None,
        process_grasp_labels_kwargs: Optional[Dict[str, Any]] = None,
        topview_debug_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        depth_prob: Optional[torch.Tensor] = None,
        objectness_logits: Optional[torch.Tensor] = None,
        graspness_map: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # 1) View field.
        end_points, res_feat = self._call_view_net(
            seed_features=seed_features,
            token_sel_idx=token_sel_idx,
            camera_K=camera_K,
            depth_map=depth_map,
            depth_prob=depth_prob,
            end_points=end_points,
        )
        viewnet_sampled_view_inds = end_points.get("grasp_top_view_inds", None)
        if torch.is_tensor(viewnet_sampled_view_inds):
            viewnet_sampled_view_inds = viewnet_sampled_view_inds.detach().clone()
            end_points["kview_viewnet_sampled_view_inds"] = viewnet_sampled_view_inds

        seed_features_base = seed_features + res_feat
        seed_xyz_base = seed_xyz
        token_sel_idx_base = token_sel_idx

        end_points["xyz_graspable"] = seed_xyz_base
        end_points["token_sel_idx"] = token_sel_idx_base
        end_points["token_sel_xyz"] = seed_xyz_base

        # 2) Base label pass for view loss.
        if is_training and process_grasp_labels_fn is not None:
            _, end_points = _call_label_process(process_grasp_labels_fn, end_points, process_grasp_labels_kwargs)
            end_points = _backup_base_view_labels(end_points)
            if topview_debug_fn is not None:
                end_points = topview_debug_fn(end_points)

        # 3) Select center-view queries.
        mode = str(self.config.mode).upper()
        forced_view_inds = None
        if is_training and torch.is_tensor(viewnet_sampled_view_inds):
            if mode in ("A1", "A2") and bool(self.config.reuse_viewnet_sample_in_single_view_train):
                forced_view_inds = viewnet_sampled_view_inds
            elif mode == "A3" and bool(self.config.reuse_viewnet_sample_in_multiview_train):
                forced_view_inds = viewnet_sampled_view_inds

        seed_features_q, seed_xyz_q, token_sel_idx_q, view_rot_q, end_points = self.selector(
            seed_features=seed_features_base,
            seed_xyz=seed_xyz_base,
            token_sel_idx=token_sel_idx_base,
            view_score=end_points["view_score"],
            end_points=end_points,
            is_training=is_training,
            forced_view_inds=forced_view_inds,
        )
        view_xyz_q = end_points["grasp_top_view_xyz"]  # [B,Q,3]

        # 4) Query-level extended labels.  This must be a new label function that
        # writes batch_grasp_*_angle tensors.  It also writes base-compatible
        # batch_grasp_view_graspness for view debug/loss.
        if is_training and process_grasp_labels_fn is not None:
            _, end_points = _call_label_process(process_grasp_labels_fn, end_points, process_grasp_labels_kwargs)
            end_points["batch_grasp_view_graspness_query"] = end_points.get("batch_grasp_view_graspness", None)
            end_points = _add_kview_selected_view_label_debug(end_points, prefix=self.config.debug_prefix)
            end_points = _restore_base_view_labels_for_view_loss(end_points)

        # 5) Angle query expansion and grouping.
        seed_features_a, seed_xyz_a, token_sel_idx_a, view_rot_a, end_points = self._expand_angle_queries(
            seed_features_q, seed_xyz_q, token_sel_idx_q, view_xyz_q, end_points
        )

        group_features_a = self.group(
            seed_features=seed_features_a,
            token_sel_idx=token_sel_idx_a,
            seed_xyz=seed_xyz_a,
            top_view_rot=view_rot_a,
            feat_map=feat_map,
            depth_map=depth_map,
            objectness_logits=objectness_logits,
            graspness_map=graspness_map,
            camera_K=camera_K,
            end_points=end_points,
        )

        # 6) Candidate decoder.
        end_points = self.decoder(group_features_a, end_points)

        # 7) Internal PNG-only visualization.  No .npy/.npz files are saved.
        _, _, H, W = feat_map.shape
        self._save_cva_visualization(end_points, img=img, image_hw=(H, W))
        self._vis_iter += 1
        return end_points