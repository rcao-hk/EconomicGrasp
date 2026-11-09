#!/usr/bin/env python3
"""Shared contracts for the scratch P2 gripper-conditioned CDF probes."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from exact_action_cdf_common import (
    CurrentCdfCheckpointContract,
    sha256_file,
    validate_current_stage1_cdf_checkpoint,
)
from models.p2_gripper_cdf_field import (
    P2_PROBE_VERSION,
    P2ScratchCdfMLP,
    P2FieldConfig,
    checkpoint_predictor_contract,
    validate_variant,
)


def validate_base_checkpoint(
    base_checkpoint_path: str,
    *,
    expected_pose_depth_mode: str = "global_film",
    expected_use_fuse_depth: bool = True,
) -> Tuple[Dict[str, Any], CurrentCdfCheckpointContract, str]:
    """Validate the frozen original Stage-1 RGB student used by every P2 variant."""
    base_checkpoint_path = os.path.abspath(base_checkpoint_path)
    checkpoint, contract = validate_current_stage1_cdf_checkpoint(
        base_checkpoint_path,
        expected_pose_depth_mode=expected_pose_depth_mode,
        expected_use_fuse_depth=expected_use_fuse_depth,
    )
    if checkpoint.get("exact_action_cdf_probe") is not None:
        raise RuntimeError(
            "P2 scratch variants require the untouched original Stage-1 checkpoint, "
            "not a P1 exact-action-updated checkpoint."
        )
    return checkpoint, contract, sha256_file(base_checkpoint_path)


class CdfHeadIoCapture:
    """Capture the exact input and raw Conv1d output of decoder.cdf_head."""

    def __init__(self, cdf_head: nn.Module) -> None:
        self.input_value: Optional[torch.Tensor] = None
        self.output_value: Optional[torch.Tensor] = None

        def hook(_module, args, output):
            if len(args) != 1 or not torch.is_tensor(args[0]) or not torch.is_tensor(output):
                raise RuntimeError("P2 CDF hook expected one tensor input/output")
            if self.input_value is not None:
                raise RuntimeError("P2 CDF head executed more than once in one forward")
            self.input_value = args[0].detach()
            self.output_value = output.detach()

        self.handle = cdf_head.register_forward_hook(hook)

    def reset(self) -> None:
        self.input_value = None
        self.output_value = None

    def pop(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_value is None or self.output_value is None:
            raise RuntimeError("P2 did not capture CDF head input/output")
        value = self.input_value, self.output_value
        self.reset()
        return value

    def close(self) -> None:
        self.handle.remove()


def assert_current_top1_rgb_output(
    end_points: Mapping[str, Any],
    *,
    m_point: int,
) -> Tuple[int, int, int, int]:
    required = (
        "D: Geometry depth source GT",
        "D: Depth head executed",
        "depth_map_used_for_geometry",
        "depth_net_pred",
        "img_feat_dpt",
        "grasp_cdf_pred_angle_depth",
        "grasp_width_pred_angle_depth",
        "xyz_graspable",
        "grasp_top_view_xyz",
        "grasp_top_view_inds",
        "token_sel_idx",
        "K",
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise RuntimeError(f"P2 current model output is missing {missing}")
    source_gt = bool(round(float(end_points["D: Geometry depth source GT"].item())))
    head_executed = bool(round(float(end_points["D: Depth head executed"].item())))
    if source_gt or not head_executed:
        raise RuntimeError("P2 did not execute RGB predicted-depth geometry")
    used = end_points["depth_map_used_for_geometry"]
    predicted = end_points["depth_net_pred"]
    if used.shape != predicted.shape or float((used - predicted).abs().max().item()) > 1e-6:
        raise RuntimeError("P2 geometry depth differs from predicted metric depth")
    cdf = end_points["grasp_cdf_pred_angle_depth"]
    width = end_points["grasp_width_pred_angle_depth"]
    if cdf.dim() != 5:
        raise RuntimeError(f"CDF endpoint must be [B,T,Q,A,D], got {tuple(cdf.shape)}")
    b, t, q, a, d = cdf.shape
    if b != 1:
        raise RuntimeError("P2 cache/inference requires batch size 1")
    if q != int(m_point):
        raise RuntimeError(f"P2 is Top-1 and requires Q=m_point={m_point}, got {q}")
    if width.shape != (b, d, q, a):
        raise RuntimeError(f"Width endpoint has incompatible shape {tuple(width.shape)}")
    return t, q, a, d


def save_p2_predictor_checkpoint(
    path: str,
    predictor: P2ScratchCdfMLP,
    *,
    variant: str,
    epoch: int,
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    source_base_checkpoint_sha256: str,
    field_config: P2FieldConfig,
    cache_contract: Mapping[str, Any],
    optimizer_state_dict: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "p2_gripper_cdf_probe": {
            "version": P2_PROBE_VERSION,
            "variant": validate_variant(variant),
            "source_base_checkpoint_sha256": str(source_base_checkpoint_sha256),
            "field_config": field_config.to_dict(),
            "field_config_sha256": field_config.sha256(),
            "predictor_contract": predictor.contract(),
            "objective": "ordinary_unbalanced_exact_action_cdf_bce_only",
            "training_initialization": "random_scratch_xavier_uniform",
            "three_layer_mlp": True,
            "no_stage1_or_p1_residual": True,
            "does_not_consume_p1_prediction": True,
            "no_ranking_loss": True,
            "no_hard_negative_loss_or_reweighting": True,
            "no_auxiliary_loss": True,
            "cache_contract": dict(cache_contract),
        },
        "epoch": int(epoch),
        "model_state_dict": copy.deepcopy(predictor.state_dict()),
        "train_metrics": dict(train_metrics),
        "val_metrics": dict(val_metrics),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = copy.deepcopy(dict(optimizer_state_dict))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_p2_predictor_checkpoint(
    checkpoint_path: str,
    *,
    expected_variant: str,
    expected_source_base_checkpoint_sha256: str,
    expected_field_config: Optional[P2FieldConfig] = None,
    map_location: str = "cpu",
) -> Tuple[Dict[str, Any], P2ScratchCdfMLP, P2FieldConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("P2 checkpoint is not a full scratch-predictor checkpoint")
    probe = checkpoint_predictor_contract(checkpoint)
    variant = validate_variant(str(probe.get("variant", "")))
    if variant != validate_variant(expected_variant):
        raise RuntimeError(f"P2 checkpoint variant={variant}, expected={expected_variant}")
    if not bool(probe.get("three_layer_mlp", False)):
        raise RuntimeError("P2 checkpoint does not declare the three-layer MLP contract")
    if not bool(probe.get("no_stage1_or_p1_residual", False)):
        raise RuntimeError("P2 checkpoint is a residual model and is rejected")
    if str(probe.get("source_base_checkpoint_sha256", "")) != str(
        expected_source_base_checkpoint_sha256
    ):
        raise RuntimeError("P2 checkpoint Base lineage mismatch")
    field_config = P2FieldConfig(**dict(probe.get("field_config", {})))
    if field_config.sha256() != str(probe.get("field_config_sha256", "")):
        raise RuntimeError("P2 checkpoint field config hash mismatch")
    if expected_field_config is not None and (
        field_config.sha256() != expected_field_config.sha256()
    ):
        raise RuntimeError("P2 checkpoint field config differs from requested config")
    contract = probe.get("predictor_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("P2 checkpoint lacks predictor contract")
    from models.p2_gripper_cdf_field import P2_FIELD_VERSION

    if str(contract.get("field_version", "")) != P2_FIELD_VERSION:
        raise RuntimeError(
            f"P2 predictor field version={contract.get('field_version')!r}, "
            f"expected={P2_FIELD_VERSION!r}"
        )
    predictor = P2ScratchCdfMLP(
        variant=variant,
        base_feature_dim=int(contract["base_feature_dim"]),
        image_feature_dim=int(contract["image_feature_dim"]),
        num_depths=int(contract["num_depths"]),
        num_thresholds=int(contract["num_thresholds"]),
        hidden_dim=int(contract["hidden_dim"]),
        increment_bias=float(contract["increment_bias"]),
    )
    result = predictor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"P2 predictor load mismatch: {result.missing_keys}/{result.unexpected_keys}"
        )
    return checkpoint, predictor, field_config
