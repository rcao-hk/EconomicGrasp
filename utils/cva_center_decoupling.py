"""Controlled center-decoupling diagnostics for the CVA-CDF model.

This module intentionally does not change the deployed model.  It reuses a
completed Stage-1 forward pass and intervenes *after* the center-view selector:

E00: read at predicted center, output predicted center (native baseline)
E01: keep native read/decoder outputs, replace only output translation by the
     reference center
E10: re-read/re-decode at the reference center, but output the predicted center
E11: re-read/re-decode at the reference center and output the reference center

The image feature map, predicted depth map, image-FPS token, selected approach
view, proposal maps and all model weights are held fixed.  This isolates the
roles of the center used to read local evidence and the center emitted as the
6-DoF grasp translation.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch


def _require_tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    value = mapping.get(key, None)
    if not torch.is_tensor(value):
        raise KeyError(f"Required center-diagnostic tensor {key!r} is missing.")
    return value


def gather_reference_centers_from_depth(
    end_points: Mapping[str, object],
    *,
    min_depth: float,
    max_depth: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backproject GT/reference depth at the exact native image-FPS tokens.

    Returns
    -------
    ref_xyz : [B,M,3]
        Reference centers. Invalid reference pixels fall back to the native
        predicted center so tensor shapes remain stable.
    ref_valid : [B,M] bool
        True only where the reference depth is finite and within range.
    ref_depth : [B,M]
        Raw gathered reference depth before fallback.
    """
    token_idx = _require_tensor(end_points, "kview_base_token_sel_idx").long()
    native_xyz = _require_tensor(end_points, "kview_base_xyz_graspable").float()
    gt_depth = _require_tensor(end_points, "gt_depth_m").float()
    K = _require_tensor(end_points, "K").float()

    if gt_depth.dim() == 3:
        gt_depth = gt_depth.unsqueeze(1)
    elif gt_depth.dim() == 4:
        gt_depth = gt_depth[:, :1]
    else:
        raise ValueError(f"gt_depth_m must be [B,H,W] or [B,1,H,W], got {tuple(gt_depth.shape)}")

    B, _, H, W = gt_depth.shape
    if token_idx.shape[0] != B or native_xyz.shape[:2] != token_idx.shape:
        raise ValueError(
            "Batch/query mismatch among GT depth, base token indices and base centers: "
            f"depth={tuple(gt_depth.shape)}, idx={tuple(token_idx.shape)}, xyz={tuple(native_xyz.shape)}"
        )
    if bool(((token_idx < 0) | (token_idx >= H * W)).any()):
        raise ValueError("kview_base_token_sel_idx contains an out-of-range pixel index.")

    flat = gt_depth[:, 0].reshape(B, H * W)
    ref_depth = torch.gather(flat, 1, token_idx)
    ref_valid = (
        torch.isfinite(ref_depth)
        & (ref_depth > float(min_depth))
        & (ref_depth < float(max_depth))
    )

    z = torch.where(ref_valid, ref_depth, native_xyz[..., 2]).clamp_min(1.0e-6)
    u = (token_idx % W).to(dtype=z.dtype)
    v = (token_idx // W).to(dtype=z.dtype)
    fx = K[:, 0, 0].unsqueeze(1).to(z)
    fy = K[:, 1, 1].unsqueeze(1).to(z)
    cx = K[:, 0, 2].unsqueeze(1).to(z)
    cy = K[:, 1, 2].unsqueeze(1).to(z)
    x = (u - cx) / fx.clamp_min(1.0e-6) * z
    y = (v - cy) / fy.clamp_min(1.0e-6) * z
    ref_xyz = torch.stack((x, y, z), dim=-1)
    return ref_xyz.contiguous(), ref_valid.contiguous(), ref_depth.contiguous()


def _top1_query_contract(end_points: Mapping[str, object]):
    """Extract the exact post-selector Top-1 query contract.

    The diagnostic deliberately forbids Top-K view expansion.  Otherwise E01
    translation replacement and E10/E11 query pairing would need rank-aware
    center expansion and would no longer isolate center depth as cleanly.
    """
    seed_features = _require_tensor(end_points, "kview_base_seed_features")
    token_idx = _require_tensor(end_points, "kview_base_token_sel_idx").long()
    native_xyz = _require_tensor(end_points, "kview_base_xyz_graspable").float()
    view_xyz = _require_tensor(end_points, "grasp_top_view_xyz").float()
    view_inds = _require_tensor(end_points, "grasp_top_view_inds").long()

    B, C, M = seed_features.shape
    if token_idx.shape != (B, M) or native_xyz.shape != (B, M, 3):
        raise ValueError("Malformed base CVA query contract.")
    if view_xyz.shape != (B, M, 3) or view_inds.shape != (B, M):
        raise RuntimeError(
            "Center-decoupling diagnostic requires deterministic Top-1 CVA inference. "
            f"Got base M={M}, view_xyz={tuple(view_xyz.shape)}, view_inds={tuple(view_inds.shape)}."
        )

    effective_k = end_points.get("kview_effective_k_int", 1)
    if torch.is_tensor(effective_k):
        effective_k = int(effective_k.detach().reshape(-1)[0].item())
    else:
        effective_k = int(effective_k)
    if effective_k != 1:
        raise RuntimeError(
            "E00/E01/E10/E11 v1 requires kview_effective_k_int == 1; "
            f"got {effective_k}. Disable Top-4 inference."
        )
    return seed_features, native_xyz, token_idx, view_xyz, view_inds


def rerun_cdf_with_read_center(
    model: torch.nn.Module,
    native_end_points: Mapping[str, object],
    *,
    read_center: torch.Tensor,
    output_center: torch.Tensor,
):
    """Re-run only angle expansion, local grouping and CDF/width decoding.

    View prediction and view selection are *not* re-run.  All image/depth
    memories are reused from the native forward pass.
    """
    module = getattr(model, "kview_grasp_module", None)
    if module is None:
        raise AttributeError("Model has no kview_grasp_module.")
    if not bool(getattr(model, "use_cdf", False)):
        raise RuntimeError("Center-decoupling diagnostic is defined for the CDF head only.")

    seed_features, native_xyz, token_idx, view_xyz, view_inds = _top1_query_contract(native_end_points)
    if read_center.shape != native_xyz.shape or output_center.shape != native_xyz.shape:
        raise ValueError(
            "read_center/output_center must match native center shape "
            f"{tuple(native_xyz.shape)}; got {tuple(read_center.shape)} / {tuple(output_center.shape)}"
        )

    feat_map = _require_tensor(native_end_points, "img_feat_dpt")
    depth_map = _require_tensor(native_end_points, "depth_map_used_for_geometry")
    camera_K = _require_tensor(native_end_points, "K")
    objectness = _require_tensor(native_end_points, "objectness_score")
    graspness = _require_tensor(native_end_points, "graspness_score")
    B, _, H, W = feat_map.shape
    if objectness.shape[-1] != H * W or graspness.shape[-1] != H * W:
        raise ValueError("Flattened proposal maps are not aligned with img_feat_dpt.")
    objectness_logits = objectness.view(B, objectness.shape[1], H, W).contiguous()
    graspness_map = graspness[:, :1].view(B, 1, H, W).contiguous()

    local_ep: Dict[str, object] = {}
    seed_a, xyz_a, token_a, rot_a, local_ep = module._expand_angle_queries(
        seed_features,
        read_center,
        token_idx,
        view_xyz,
        local_ep,
    )
    grouped = module.group(
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
    decoded = module.decoder(grouped, local_ep)

    # Build the smallest strict endpoint contract needed by the repository's
    # native CDF decoder.  The output center is intentionally independent of the
    # center used above for feature reading.
    decode_ep: Dict[str, object] = {
        "xyz_graspable": output_center,
        "grasp_top_view_xyz": view_xyz,
        "grasp_top_view_inds": view_inds,
        "grasp_cdf_pred_angle_depth": decoded["grasp_cdf_pred_angle_depth"],
        "grasp_width_pred_angle_depth": decoded["grasp_width_pred_angle_depth"],
        "D: CDF enabled": torch.ones((), device=output_center.device),
        "kview_effective_k_int": 1,
    }
    # Native query-index helper accepts these metadata when present.  For Top-1
    # they are copied verbatim to guarantee the same parent/rank semantics.
    for key in (
        "kview_query_parent",
        "kview_query_view_rank",
        "kview_query_view_inds",
        "kview_base_M",
        "kview_mode",
    ):
        if key in native_end_points:
            decode_ep[key] = native_end_points[key]
    return decode_ep, grouped


def replace_decoded_translation(
    grasp_preds,
    replacement_xyz: torch.Tensor,
):
    """E01 intervention: preserve native grasp decisions, replace xyz only."""
    if len(grasp_preds) != replacement_xyz.shape[0]:
        raise ValueError("Batch size mismatch in translation replacement.")
    outputs = []
    for b, pred in enumerate(grasp_preds):
        if pred.dim() != 2 or pred.shape[-1] != 17:
            raise ValueError(f"Decoded grasps must be [N,17], got {tuple(pred.shape)}")
        xyz = replacement_xyz[b]
        if pred.shape[0] != xyz.shape[0]:
            raise RuntimeError(
                "E01 requires one decoded Top-1 grasp per native image-FPS query: "
                f"pred={pred.shape[0]}, xyz={xyz.shape[0]}."
            )
        out = pred.clone()
        out[:, 13:16] = xyz.to(device=out.device, dtype=out.dtype)
        outputs.append(out)
    return outputs
