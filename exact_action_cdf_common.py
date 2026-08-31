#!/usr/bin/env python3
"""Shared contract helpers for the current CVA-CDF exact-action probe.

This module intentionally supports only the corrected Stage-1 RGB student:

* CVA-CDF head (no legacy score/depth/collision head),
* predicted metric depth,
* deterministic image-space FPS,
* current privileged-depth checkpoint contract.

It contains no compatibility path for the old ``cva_joint_utility_v1`` cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE_SCHEMA_VERSION = "cva_exact_action_cdf_head_cache_v1_1"
PROBE_VERSION = "cva_exact_action_cdf_head_probe_v1_1"
DISTILL_CONTRACT_VERSION = 2
FRICTION_THRESHOLDS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)

CDF_HEAD_WEIGHT_SUFFIX = "kview_grasp_module.decoder.cdf_head.weight"
CDF_HEAD_BIAS_SUFFIX = "kview_grasp_module.decoder.cdf_head.bias"
WIDTH_HEAD_WEIGHT_SUFFIX = "kview_grasp_module.decoder.width_head.weight"


@dataclass(frozen=True)
class CurrentCdfCheckpointContract:
    checkpoint_path: str
    checkpoint_sha256: str
    distill_stage: int
    distill_contract_version: int
    seed_selection_mode: str
    geometry_depth_source: str
    depth_head_executed: bool
    pose_depth_mode: str
    use_fuse_depth: bool
    feature_dim: int
    num_depths: int
    num_thresholds: int
    cdf_head_weight_key: str
    cdf_head_bias_key: str
    width_head_weight_key: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_file(path: str, chunk_size: int = 8 << 20) -> str:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def atomic_save_json(payload: Mapping[str, Any], path: str) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(dict(payload)), f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def atomic_save_npz(path: str, arrays: Mapping[str, np.ndarray], compress: bool = False) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    saver = np.savez_compressed if bool(compress) else np.savez
    with open(tmp, "wb") as f:
        saver(f, **arrays)
    os.replace(tmp, path)


def read_full_checkpoint(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "The current exact-action probe requires a full corrected Stage-1 "
            "checkpoint with model_state_dict and contract metadata. Plain/legacy "
            "state dictionaries are rejected."
        )
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint['model_state_dict'] is not a mapping")
    return checkpoint


def _unique_key_ending_with(state: Mapping[str, Any], suffix: str) -> str:
    matches = [str(key) for key in state.keys() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one state key ending with {suffix!r}, found {matches}."
        )
    return matches[0]


def validate_current_stage1_cdf_checkpoint(
    checkpoint_path: str,
    *,
    expected_pose_depth_mode: Optional[str] = "global_film",
    expected_use_fuse_depth: Optional[bool] = True,
    expected_num_depths: int = 4,
    expected_num_thresholds: int = 6,
) -> Tuple[Dict[str, Any], CurrentCdfCheckpointContract]:
    """Validate and describe the only checkpoint family supported by the probe."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint = read_full_checkpoint(checkpoint_path)
    state = checkpoint["model_state_dict"]

    stage = int(checkpoint.get("distill_stage", -1))
    if stage != 1:
        raise RuntimeError(
            f"Exact-action head probing requires the current Stage-1 RGB student; "
            f"checkpoint distill_stage={stage}."
        )
    version = int(checkpoint.get("distill_contract_version", -1))
    if version != DISTILL_CONTRACT_VERSION:
        raise RuntimeError(
            "Checkpoint predates the corrected privileged-depth contract: "
            f"expected {DISTILL_CONTRACT_VERSION}, got {version}."
        )
    seed_mode = str(checkpoint.get("seed_selection_mode", ""))
    if seed_mode != "image_fps":
        raise RuntimeError(
            f"Current probe requires seed_selection_mode='image_fps', got {seed_mode!r}."
        )
    geometry_source = str(checkpoint.get("geometry_depth_source", ""))
    if geometry_source != "pred":
        raise RuntimeError(
            f"Current Stage-1 probe requires geometry_depth_source='pred', got "
            f"{geometry_source!r}."
        )
    head_executed = bool(checkpoint.get("depth_head_executed", False))
    if not head_executed:
        raise RuntimeError("Current Stage-1 checkpoint says the metric-depth head was bypassed.")
    if bool(checkpoint.get("legacy_dataset_use_gt_depth", True)):
        raise RuntimeError(
            "Checkpoint used the deprecated dataset --use_gt_depth path and is rejected."
        )

    pose_mode = str(checkpoint.get("pose_depth_mode", ""))
    if expected_pose_depth_mode is not None and pose_mode != str(expected_pose_depth_mode):
        raise RuntimeError(
            f"Checkpoint pose_depth_mode={pose_mode!r}; expected "
            f"{str(expected_pose_depth_mode)!r}."
        )
    if "use_fuse_depth" not in checkpoint:
        raise RuntimeError("Checkpoint has no use_fuse_depth metadata.")
    use_fuse_depth = bool(checkpoint["use_fuse_depth"])
    if expected_use_fuse_depth is not None and use_fuse_depth != bool(expected_use_fuse_depth):
        raise RuntimeError(
            f"Checkpoint use_fuse_depth={use_fuse_depth}; expected "
            f"{bool(expected_use_fuse_depth)}."
        )

    weight_key = _unique_key_ending_with(state, CDF_HEAD_WEIGHT_SUFFIX)
    bias_key = _unique_key_ending_with(state, CDF_HEAD_BIAS_SUFFIX)
    width_key = _unique_key_ending_with(state, WIDTH_HEAD_WEIGHT_SUFFIX)
    weight = state[weight_key]
    bias = state[bias_key]
    width_weight = state[width_key]
    if not (torch.is_tensor(weight) and torch.is_tensor(bias) and torch.is_tensor(width_weight)):
        raise TypeError("Current CDF/width head parameters must be tensors.")
    if weight.dim() != 3 or weight.shape[-1] != 1:
        raise RuntimeError(f"CDF head weight must be [D*T,C,1], got {tuple(weight.shape)}")
    if bias.dim() != 1 or bias.shape[0] != weight.shape[0]:
        raise RuntimeError(
            f"CDF head bias shape {tuple(bias.shape)} does not match weight "
            f"{tuple(weight.shape)}."
        )
    if width_weight.dim() != 3 or width_weight.shape[-1] != 1:
        raise RuntimeError(
            f"Depth-wise width head weight must be [D,C,1], got {tuple(width_weight.shape)}"
        )

    num_thresholds = int(expected_num_thresholds)
    if int(weight.shape[0]) % num_thresholds != 0:
        raise RuntimeError(
            f"CDF output channels={weight.shape[0]} are not divisible by T={num_thresholds}."
        )
    num_depths = int(weight.shape[0]) // num_thresholds
    feature_dim = int(weight.shape[1])
    if num_depths != int(expected_num_depths):
        raise RuntimeError(
            f"Current probe expects D={expected_num_depths}, checkpoint has D={num_depths}."
        )
    if int(width_weight.shape[0]) != num_depths or int(width_weight.shape[1]) != feature_dim:
        raise RuntimeError(
            "CDF and depth-wise width heads do not share the expected D/C dimensions: "
            f"cdf={tuple(weight.shape)}, width={tuple(width_weight.shape)}."
        )

    # Explicitly reject the old explicit-angle score/depth/collision decoder.
    forbidden_fragments = (
        "kview_grasp_module.decoder.score_head.",
        "kview_grasp_module.decoder.depth_head.",
        "kview_grasp_module.decoder.collision_head.",
    )
    forbidden = [
        str(key) for key in state.keys()
        if any(fragment in str(key) for fragment in forbidden_fragments)
    ]
    if forbidden:
        raise RuntimeError(
            "Checkpoint contains a legacy score/depth/collision candidate decoder and "
            f"is unsupported. Example keys: {forbidden[:6]}"
        )

    contract = CurrentCdfCheckpointContract(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        distill_stage=stage,
        distill_contract_version=version,
        seed_selection_mode=seed_mode,
        geometry_depth_source=geometry_source,
        depth_head_executed=head_executed,
        pose_depth_mode=pose_mode,
        use_fuse_depth=use_fuse_depth,
        feature_dim=feature_dim,
        num_depths=num_depths,
        num_thresholds=num_thresholds,
        cdf_head_weight_key=weight_key,
        cdf_head_bias_key=bias_key,
        width_head_weight_key=width_key,
    )
    return checkpoint, contract


def load_model_state_strict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load current CVA-CDF state, allowing only optional diagnostic modules."""
    result = model.load_state_dict(state, strict=False)
    optional_prefixes = ("rgb_geometry_diagnostics.",)
    missing = [key for key in result.missing_keys if not key.startswith(optional_prefixes)]
    unexpected = [key for key in result.unexpected_keys if not key.startswith(optional_prefixes)]
    if missing or unexpected:
        raise RuntimeError(
            "Strict current CVA-CDF checkpoint load failed: "
            f"missing={missing}, unexpected={unexpected}"
        )


def resolve_current_cdf_decoder(model: nn.Module) -> nn.Module:
    module = model
    for name in ("kview_grasp_module", "decoder"):
        if not hasattr(module, name):
            raise AttributeError(
                f"Current CVA-CDF model is missing module path component {name!r}."
            )
        module = getattr(module, name)
    if not hasattr(module, "cdf_head") or not isinstance(module.cdf_head, nn.Conv1d):
        raise TypeError("Current decoder.cdf_head must be nn.Conv1d.")
    if hasattr(module, "score_head") or hasattr(module, "depth_head"):
        raise RuntimeError("Resolved decoder is a legacy explicit-angle decoder, not CDF-only.")
    return module


def friction_to_cdf_target(
    friction: torch.Tensor,
    thresholds: Sequence[float] = FRICTION_THRESHOLDS,
) -> torch.Tensor:
    """Map minimum successful friction [*,D] to monotonic targets [*,D,T]."""
    threshold = friction.new_tensor(tuple(float(x) for x in thresholds))
    shape = [1] * friction.dim() + [len(thresholds)]
    threshold = threshold.view(*shape)
    f = friction.unsqueeze(-1)
    return ((f > 0.0) & (f <= threshold)).to(dtype=friction.dtype)


def monotonic_cdf_logits_from_raw(
    raw: torch.Tensor,
    increment_bias: float,
) -> torch.Tensor:
    """Convert raw [...,D,T] outputs to monotonic CDF logits."""
    if raw.dim() < 2 or raw.shape[-1] < 2:
        raise ValueError(f"raw must end in [D,T>=2], got {tuple(raw.shape)}")
    base = raw[..., :1]
    increments = F.softplus(raw[..., 1:] + float(increment_bias))
    return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


class CurrentCdfHeadOnly(nn.Module):
    """The existing current CDF Conv1d head, evaluated on cached head inputs.

    No auxiliary MLP, collision head, ranking head, or legacy logits are added.
    """

    def __init__(
        self,
        feature_dim: int,
        num_depths: int,
        num_thresholds: int,
        increment_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_depths = int(num_depths)
        self.num_thresholds = int(num_thresholds)
        self.increment_bias = float(increment_bias)
        self.cdf_head = nn.Conv1d(
            self.feature_dim,
            self.num_depths * self.num_thresholds,
            kernel_size=1,
        )

    def load_from_full_state(
        self,
        state: Mapping[str, torch.Tensor],
        contract: CurrentCdfCheckpointContract,
    ) -> None:
        with torch.no_grad():
            self.cdf_head.weight.copy_(state[contract.cdf_head_weight_key])
            self.cdf_head.bias.copy_(state[contract.cdf_head_bias_key])

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() != 2 or feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feature must be [N,{self.feature_dim}], got {tuple(feature.shape)}"
            )
        # Keep the deployed operator itself in the probe. Although a 1x1
        # Conv1d is mathematically a row-wise linear map, cuDNN Conv1d and
        # cuBLAS F.linear may use different TF32 kernels on Ampere GPUs. The
        # resulting ~1e-2 logit difference is numerical, not a cache-alignment
        # error. Running the actual Conv1d also makes the optimized parameters
        # follow the same operator used after merging into the full model.
        x = feature.transpose(0, 1).unsqueeze(0).contiguous()  # [1,C,N]
        raw_flat = self.cdf_head(x)                           # [1,D*T,N]
        raw = (
            raw_flat.squeeze(0)
            .transpose(0, 1)
            .contiguous()
            .view(-1, self.num_depths, self.num_thresholds)
        )
        return monotonic_cdf_logits_from_raw(raw, self.increment_bias)


def merge_cdf_head_into_full_checkpoint(
    base_checkpoint: Mapping[str, Any],
    contract: CurrentCdfCheckpointContract,
    head_model: CurrentCdfHeadOnly,
    *,
    probe_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return an inference checkpoint differing only in current cdf_head weights."""
    merged: Dict[str, Any] = {
        key: copy.deepcopy(value)
        for key, value in base_checkpoint.items()
        if key != "optimizer_state_dict"
    }
    state = dict(merged["model_state_dict"])
    state[contract.cdf_head_weight_key] = (
        head_model.cdf_head.weight.detach().cpu().clone()
    )
    state[contract.cdf_head_bias_key] = (
        head_model.cdf_head.bias.detach().cpu().clone()
    )
    merged["model_state_dict"] = state
    merged["exact_action_cdf_probe"] = {
        "version": PROBE_VERSION,
        "head_only_update": True,
        "updated_state_keys": [
            contract.cdf_head_weight_key,
            contract.cdf_head_bias_key,
        ],
        "base_checkpoint_path": contract.checkpoint_path,
        "base_checkpoint_sha256": contract.checkpoint_sha256,
        **dict(probe_metadata),
    }
    return merged
