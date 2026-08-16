"""Minimal privileged-depth teacher -> RGB student distillation.

All stages use the same deterministic image-space FPS selector so output KD is
not confounded by 3D-FPS/image-query seed mismatch.  The modality distinction
is explicit and restricted to the geometry depth consumed by the grasp model:

* Stage 0 teacher: RGB proposal features + clean synthetic ``gt_depth_m``; the
  DPT metric-depth decoder remains checkpoint-compatible but is frozen and
  bypassed.
* Stage 1 student: RGB -> DPT metric depth, trained with the existing GT losses.
* Stage 2 student: the same RGB-only model plus frozen Stage-0 output KD. The
  student selects image-FPS seeds autonomously, and the clean-depth teacher is
  evaluated at those exact student-selected image locations.

Thus the experiment isolates whether task-specific grasp outputs from privileged
clean geometry can improve the RGB-only CVA-CDF student, without adding feature
KD, ray losses, collision heads, material augmentation, or multi-view training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .economicgrasp_bip3d import economicgrasp_dpt


# Version 2 is the first contract in which Stage 0 is a true privileged
# clean-depth teacher rather than a predicted-depth self-distillation model.
DISTILL_CONTRACT_VERSION = 2


class economicgrasp_dpt_distill(economicgrasp_dpt):
    """EconomicGrasp-DPT with deterministic image-FPS sparse queries."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Ignore selector arguments from the earlier image-topk prototype so an
        # old launcher cannot silently reactivate a different seed path.
        kwargs.pop("seed_selection_mode", None)
        kwargs.pop("image_seed_nms_kernel", None)
        # The distillation experiment never uses the legacy train-only GT-XYZ
        # switch. Geometry privilege is controlled solely by depth source.
        kwargs["use_gt_xyz_for_train"] = False
        super().__init__(*args, seed_selection_mode="image_fps", **kwargs)


class economicgrasp_dpt_teacher(economicgrasp_dpt_distill):
    """Stage-0/2 privileged teacher using clean synthetic depth geometry."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("geometry_depth_source", None)
        kwargs["geometry_depth_source"] = "gt"
        kwargs["use_obs_depth"] = False
        kwargs["pose_depth_mode"] = "none"
        super().__init__(*args, **kwargs)


class economicgrasp_dpt_student(economicgrasp_dpt_distill):
    """Stage-1/2 RGB-only student using its predicted metric depth."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("geometry_depth_source", None)
        kwargs["geometry_depth_source"] = "pred"
        super().__init__(*args, **kwargs)


@dataclass(frozen=True)
class OutputDistillationConfig:
    """Weights and validity thresholds for privileged-teacher output KD."""

    overall_weight: float = 1.0
    # E1: proposal outputs are not privileged by clean geometry because they
    # are predicted before the spatial/depth enhancer. Keep their optional KD
    # switches for ablations, but disable them in the default experiment.
    objectness_weight: float = 0.0
    graspness_weight: float = 0.0
    depth_weight: float = 0.0
    view_weight: float = 1.0
    cdf_weight: float = 1.0
    width_weight: float = 0.1

    temperature: float = 1.0
    max_query_view_angle_deg: float = 35.0
    width_positive_threshold: float = 0.5
    min_depth: float = 0.2
    max_depth: float = 1.0
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Retain tensors required by the KD objective. In E2, the student first
# selects its own image-FPS seeds; ``kview_base_token_sel_idx`` is then passed
# to the frozen teacher as an exact seed override.
_DISTILL_TARGET_KEYS = (
    "objectness_score",
    "graspness_score",
    "depth_map_pred",
    "token_valid_mask",
    "view_score",
    "kview_base_token_sel_idx",
    "token_sel_idx",
    "grasp_top_view_xyz",
    "grasp_cdf_pred_angle_depth",
    "grasp_width_pred_angle_depth",
)


def extract_distillation_targets(
    teacher_end_points: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Detach the strict subset needed by stage-2 distillation."""
    targets: Dict[str, torch.Tensor] = {}
    missing = []
    for key in _DISTILL_TARGET_KEYS:
        value = teacher_end_points.get(key, None)
        if value is None:
            if key == "token_valid_mask":
                continue
            missing.append(key)
            continue
        if not torch.is_tensor(value):
            raise TypeError(
                f"Teacher endpoint {key!r} must be a tensor, got {type(value)}."
            )
        targets[key] = value.detach()
    if missing:
        raise KeyError(
            "Stage-2 teacher is missing required CDF endpoint(s): "
            + ", ".join(missing)
        )
    return targets


def load_checkpoint_state(
    model: nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = True,
    checkpoint_data: Any = None,
) -> Dict[str, Any]:
    """Load a state dict, optionally reusing prevalidated checkpoint data."""
    if checkpoint_data is None:
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be non-empty.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    else:
        checkpoint = checkpoint_data
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state, Mapping):
        raise TypeError(
            f"Checkpoint does not contain a state dict: {checkpoint_path}"
        )
    result = model.load_state_dict(state, strict=False)
    optional_prefixes = ("rgb_geometry_diagnostics.",)
    missing = [
        key for key in result.missing_keys
        if not key.startswith(optional_prefixes)
    ]
    unexpected = [
        key for key in result.unexpected_keys
        if not key.startswith(optional_prefixes)
    ]
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict checkpoint loading produced incompatible keys: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state}


def _zero_from_student(student_end_points: Mapping[str, Any]) -> torch.Tensor:
    for key in (
        "grasp_cdf_pred_angle_depth",
        "view_score",
        "objectness_score",
        "depth_map_pred",
    ):
        value = student_end_points.get(key, None)
        if torch.is_tensor(value):
            return value.sum() * 0.0
    raise KeyError("No differentiable student output is available for KD loss.")


def _normalize_view_score_shape(
    score: torch.Tensor,
    num_query: int,
) -> torch.Tensor:
    if score.dim() != 3:
        raise ValueError(
            f"view_score must be [B,Q,V] or [B,V,Q], got {tuple(score.shape)}"
        )
    if score.shape[1] == num_query:
        return score.contiguous()
    if score.shape[2] == num_query:
        return score.transpose(1, 2).contiguous()
    raise ValueError(
        f"Cannot align view_score {tuple(score.shape)} with Q={num_query}."
    )


def _gather_query_dim(
    tensor: torch.Tensor,
    match: torch.Tensor,
    query_dim: int,
) -> torch.Tensor:
    """Gather a teacher tensor along its query dimension with [B,Qs] indices."""
    if tensor.shape[0] != match.shape[0]:
        raise ValueError("Teacher tensor and query match batch sizes differ.")
    shape = list(tensor.shape)
    index_shape = shape.copy()
    index_shape[query_dim] = match.shape[1]
    view = [match.shape[0]] + [1] * (tensor.dim() - 1)
    view[query_dim] = match.shape[1]
    index = match.view(*view).expand(*index_shape)
    return torch.gather(tensor, query_dim, index)


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=torch.bool)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    if bool(mask.any()):
        return value[mask].mean()
    return zero


def _assert_shared_base_image_fps(
    student_idx: torch.Tensor,
    teacher_idx: torch.Tensor,
) -> torch.Tensor:
    """Require identical ordered image-FPS seeds and return equality mask."""
    if student_idx.dim() != 2 or teacher_idx.dim() != 2:
        raise ValueError(
            "Base image-FPS indices must be [B,M], got "
            f"student={tuple(student_idx.shape)}, teacher={tuple(teacher_idx.shape)}"
        )
    if student_idx.shape != teacher_idx.shape:
        raise RuntimeError(
            "Teacher/student image-FPS seed shape mismatch: "
            f"{tuple(student_idx.shape)} vs {tuple(teacher_idx.shape)}."
        )
    equal = student_idx == teacher_idx
    if not bool(equal.all()):
        mismatch = float((~equal).float().mean().item())
        raise RuntimeError(
            "Stage-2 requires exact shared image-FPS seeds, but the ordered "
            f"base indices differ at {100.0 * mismatch:.3f}% of positions."
        )
    return equal


def _same_seed_query_match(
    student_base_idx: torch.Tensor,
    teacher_base_idx: torch.Tensor,
    student_query_idx: torch.Tensor,
    student_view_xyz: torch.Tensor,
    teacher_query_idx: torch.Tensor,
    teacher_view_xyz: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match teacher views only within the same ordered image-FPS seed.

    K-view expansion is base-major in the CVA selector. After exact base-seed
    sharing, query tensors can be reshaped to [B,M,K,*]. Each student view is
    paired with the closest physical teacher view belonging to that same base
    image location. No global center matching or pixel-distance threshold is
    needed.
    """
    _assert_shared_base_image_fps(student_base_idx, teacher_base_idx)
    B, M = student_base_idx.shape
    if student_query_idx.dim() != 2 or teacher_query_idx.dim() != 2:
        raise ValueError("Expanded token indices must be [B,Q].")
    if student_query_idx.shape[0] != B or teacher_query_idx.shape[0] != B:
        raise ValueError("Expanded query batch size differs from base seeds.")
    Qs = int(student_query_idx.shape[1])
    Qt = int(teacher_query_idx.shape[1])
    if Qs % M != 0 or Qt % M != 0:
        raise RuntimeError(
            f"K-view query count must be divisible by M={M}; got Qs/Qt={Qs}/{Qt}."
        )
    Ks = Qs // M
    Kt = Qt // M
    if Ks <= 0 or Kt <= 0:
        raise RuntimeError("Teacher/student produced no K-view query.")

    s_idx = student_query_idx.view(B, M, Ks)
    t_idx = teacher_query_idx.view(B, M, Kt)
    expected_s = student_base_idx.unsqueeze(-1).expand_as(s_idx)
    expected_t = teacher_base_idx.unsqueeze(-1).expand_as(t_idx)
    if not bool((s_idx == expected_s).all()):
        raise RuntimeError(
            "Student K-view query ordering is not base-major or does not reuse "
            "the shared image-FPS base indices."
        )
    if not bool((t_idx == expected_t).all()):
        raise RuntimeError(
            "Teacher K-view query ordering is not base-major or does not reuse "
            "the shared image-FPS base indices."
        )

    if student_view_xyz.shape != (B, Qs, 3):
        raise ValueError(
            f"student_view_xyz must be {(B, Qs, 3)}, got {tuple(student_view_xyz.shape)}"
        )
    if teacher_view_xyz.shape != (B, Qt, 3):
        raise ValueError(
            f"teacher_view_xyz must be {(B, Qt, 3)}, got {tuple(teacher_view_xyz.shape)}"
        )

    s_view = F.normalize(
        student_view_xyz.detach().float(), dim=-1
    ).view(B, M, Ks, 3)
    t_view = F.normalize(
        teacher_view_xyz.detach().float(), dim=-1
    ).view(B, M, Kt, 3)
    cosine = torch.einsum("bmqc,bmkc->bmqk", s_view, t_view).clamp(-1.0, 1.0)
    local_match = cosine.argmax(dim=-1)
    parent_offset = (
        torch.arange(M, device=local_match.device, dtype=torch.long)
        .view(1, M, 1)
        .expand(B, M, Ks)
        * Kt
    )
    global_match = (parent_offset + local_match).reshape(B, Qs).contiguous()
    matched_cos = torch.gather(
        cosine, dim=-1, index=local_match.unsqueeze(-1)
    ).squeeze(-1)
    matched_cos = matched_cos.clamp(-1.0 + float(eps), 1.0 - float(eps))
    angle_deg = torch.rad2deg(torch.acos(matched_cos)).reshape(B, Qs)
    return global_match, angle_deg


def compute_output_distillation_loss(
    student_end_points: Dict[str, Any],
    teacher_targets: Mapping[str, torch.Tensor],
    config: OutputDistillationConfig,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute E1 output KD under exact student-driven shared image-FPS seeds."""
    zero = _zero_from_student(student_end_points)
    temperature = max(float(config.temperature), float(config.eps))

    required_student = (
        "objectness_score",
        "graspness_score",
        "depth_map_pred",
        "view_score",
        "kview_base_token_sel_idx",
        "token_sel_idx",
        "grasp_top_view_xyz",
        "grasp_cdf_pred_angle_depth",
        "grasp_width_pred_angle_depth",
    )
    missing = [key for key in required_student if key not in student_end_points]
    if missing:
        raise KeyError(
            "Stage-2 student is missing required CDF endpoint(s): "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # Dense pixel-aligned outputs.
    # ------------------------------------------------------------------
    s_obj = student_end_points["objectness_score"]
    t_obj = teacher_targets["objectness_score"].to(s_obj)
    if s_obj.shape != t_obj.shape:
        raise ValueError(
            f"Objectness KD shape mismatch: {tuple(s_obj.shape)} vs {tuple(t_obj.shape)}"
        )
    valid_tok = student_end_points.get(
        "token_valid_mask",
        teacher_targets.get("token_valid_mask", None),
    )
    if torch.is_tensor(valid_tok):
        valid_tok = valid_tok.to(device=s_obj.device).bool()
    else:
        valid_tok = torch.ones(
            s_obj.shape[0], s_obj.shape[-1], device=s_obj.device, dtype=torch.bool
        )

    obj_kl = F.kl_div(
        F.log_softmax(s_obj / temperature, dim=1),
        F.softmax(t_obj / temperature, dim=1),
        reduction="none",
    ).sum(dim=1) * (temperature ** 2)
    objectness_loss = _masked_mean(obj_kl, valid_tok, zero)

    s_grasp = student_end_points["graspness_score"]
    t_grasp = teacher_targets["graspness_score"].to(s_grasp)
    if s_grasp.shape != t_grasp.shape:
        raise ValueError(
            f"Graspness KD shape mismatch: {tuple(s_grasp.shape)} vs {tuple(t_grasp.shape)}"
        )
    grasp_map = F.smooth_l1_loss(s_grasp, t_grasp, reduction="none").squeeze(1)
    graspness_loss = _masked_mean(grasp_map, valid_tok, zero)

    # The RGB student already receives direct GT metric-depth supervision via
    # the unchanged supervised loss.  By default depth KD is disabled so Stage 2
    # measures task-output transfer rather than duplicating the same GT target.
    if float(config.depth_weight) > 0.0:
        s_depth = student_end_points["depth_map_pred"]
        t_depth = teacher_targets["depth_map_pred"].to(s_depth)
        if s_depth.shape != t_depth.shape:
            t_depth = F.interpolate(
                t_depth,
                size=s_depth.shape[-2:],
                mode="nearest",
            )
        depth_valid = (
            torch.isfinite(t_depth)
            & (t_depth > float(config.min_depth))
            & (t_depth < float(config.max_depth))
            & torch.isfinite(s_depth)
        )
        depth_map = F.smooth_l1_loss(s_depth, t_depth, reduction="none")
        depth_loss = _masked_mean(depth_map, depth_valid, zero)
    else:
        depth_loss = zero

    # ------------------------------------------------------------------
    # Exact same-base view field.
    # ------------------------------------------------------------------
    s_base_idx = student_end_points["kview_base_token_sel_idx"].long()
    t_base_idx = teacher_targets["kview_base_token_sel_idx"].to(
        device=s_base_idx.device, dtype=torch.long
    )
    shared_equal = _assert_shared_base_image_fps(s_base_idx, t_base_idx)
    s_view = _normalize_view_score_shape(
        student_end_points["view_score"], s_base_idx.shape[1]
    )
    t_view = _normalize_view_score_shape(
        teacher_targets["view_score"].to(s_view), t_base_idx.shape[1]
    )
    if s_view.shape != t_view.shape:
        raise ValueError(
            "Shared-seed view field shape mismatch: "
            f"student={tuple(s_view.shape)}, teacher={tuple(t_view.shape)}"
        )
    view_loss = F.smooth_l1_loss(s_view, t_view, reduction="mean")

    # ------------------------------------------------------------------
    # Same-center CVA query CDF and depth-wise width.
    # ------------------------------------------------------------------
    s_query_idx = student_end_points["token_sel_idx"].long()
    t_query_idx = teacher_targets["token_sel_idx"].to(
        device=s_query_idx.device, dtype=torch.long
    )
    s_query_view = student_end_points["grasp_top_view_xyz"]
    t_query_view = teacher_targets["grasp_top_view_xyz"].to(s_query_view)
    query_match, query_angle = _same_seed_query_match(
        student_base_idx=s_base_idx,
        teacher_base_idx=t_base_idx,
        student_query_idx=s_query_idx,
        student_view_xyz=s_query_view,
        teacher_query_idx=t_query_idx,
        teacher_view_xyz=t_query_view,
        eps=float(config.eps),
    )
    query_valid = query_angle <= float(config.max_query_view_angle_deg)

    s_cdf = student_end_points["grasp_cdf_pred_angle_depth"]
    t_cdf = teacher_targets["grasp_cdf_pred_angle_depth"].to(s_cdf)
    if s_cdf.dim() != 5 or t_cdf.dim() != 5:
        raise ValueError(
            "CDF outputs must be [B,T,Q,A,D], got "
            f"student={tuple(s_cdf.shape)}, teacher={tuple(t_cdf.shape)}"
        )
    t_cdf_matched = _gather_query_dim(t_cdf, query_match, query_dim=2)
    if s_cdf.shape != t_cdf_matched.shape:
        raise ValueError(
            "Matched CDF shape mismatch: "
            f"student={tuple(s_cdf.shape)}, teacher={tuple(t_cdf_matched.shape)}"
        )
    cdf_soft_target = torch.sigmoid(t_cdf_matched / temperature)
    cdf_map = F.binary_cross_entropy_with_logits(
        s_cdf / temperature,
        cdf_soft_target,
        reduction="none",
    ) * (temperature ** 2)
    cdf_valid = query_valid[:, None, :, None, None]
    cdf_loss = _masked_mean(cdf_map, cdf_valid, zero)

    s_width = student_end_points["grasp_width_pred_angle_depth"]
    t_width = teacher_targets["grasp_width_pred_angle_depth"].to(s_width)
    if s_width.dim() != 4 or t_width.dim() != 4:
        raise ValueError(
            "Width outputs must be [B,D,Q,A], got "
            f"student={tuple(s_width.shape)}, teacher={tuple(t_width.shape)}"
        )
    t_width_matched = _gather_query_dim(t_width, query_match, query_dim=2)
    if s_width.shape != t_width_matched.shape:
        raise ValueError(
            "Matched width shape mismatch: "
            f"student={tuple(s_width.shape)}, teacher={tuple(t_width_matched.shape)}"
        )
    teacher_positive = (
        cdf_soft_target.mean(dim=1).permute(0, 3, 1, 2)
        >= float(config.width_positive_threshold)
    )
    width_valid = query_valid.unsqueeze(1).unsqueeze(-1) & teacher_positive
    width_map = F.smooth_l1_loss(
        s_width, t_width_matched, reduction="none"
    )
    width_loss = _masked_mean(width_map, width_valid, zero)

    weighted = (
        float(config.objectness_weight) * objectness_loss
        + float(config.graspness_weight) * graspness_loss
        + float(config.depth_weight) * depth_loss
        + float(config.view_weight) * view_loss
        + float(config.cdf_weight) * cdf_loss
        + float(config.width_weight) * width_loss
    )
    total = float(config.overall_weight) * weighted

    student_end_points["B: KD Objectness Loss"] = objectness_loss
    student_end_points["B: KD Graspness Loss"] = graspness_loss
    student_end_points["B: KD Depth Loss"] = depth_loss
    student_end_points["B: KD View Loss"] = view_loss
    student_end_points["B: KD CDF Loss"] = cdf_loss
    student_end_points["B: KD Width Loss"] = width_loss
    student_end_points["A: Distill Loss"] = total

    with torch.no_grad():
        student_end_points["D: KD shared image-FPS ratio"] = (
            shared_equal.float().mean().reshape(())
        )
        student_end_points["D: KD query match ratio"] = (
            query_valid.float().mean().reshape(())
        )
        student_end_points["D: KD query view angle"] = (
            query_angle.mean().reshape(())
        )
        student_end_points["D: KD width positive ratio"] = (
            width_valid.float().mean().reshape(())
        )
        student_end_points["D: KD depth enabled"] = zero.new_tensor(
            float(config.depth_weight > 0.0)
        ).reshape(())

    return total, student_end_points
