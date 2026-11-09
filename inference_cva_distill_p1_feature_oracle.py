#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1: Student-query-conditioned privileged feature oracle.

This is an inference-only mechanism experiment for the current
EconomicGrasp-DPT CVA-CDF repository.

Core protocol
-------------
1. Run the RGB Stage-1 student normally.  The student owns:
     * deterministic image-FPS seed pixels;
     * predicted-depth physical 3D centers;
     * inference-time Top-1 approach views.
2. Run the frozen Stage-0 clean-geometry teacher at the exact same seed pixels,
   exact same approach views, and—critically—the exact *student physical 3D
   centers*.  The teacher still uses clean ``gt_depth_m`` for its spatial
   enhancer and local grouping support; the active ray-only ViewNet receives
   seed features formed by that clean-depth-enhanced representation.
3. Either:
     * decode the complete student-query-conditioned teacher (``teacher_full``);
       or
     * inject one teacher feature into the student's downstream tail:
         O1 ``o1_group``: output of ViewConditionedAttentionGrouping;
         O2 ``o2_angle``: output of the last CVA angle-transformer layer;
         O3 ``o3_cdf``: input feature of the final CDF head.
4. Decode with the repository's unmodified CDF decoder and save GraspNet-format
   predictions.  Optional model-free collision filtering is unchanged.

The data protocol is deliberately unchanged.  This script reuses the same
``GraspNetMultiDataset`` split, ``sample_interval``, clean-depth construction,
decoder, and collision post-processing as ``inference_cva_distill.py``.

Important interpretation
------------------------
Stage 0 and Stage 1 were trained separately.  Direct feature swapping therefore
tests both privileged information and latent-space compatibility.  Always run
``student`` and ``teacher_full`` controls alongside O1--O3:

* teacher_full improves, O1--O3 do not:
    the teacher is informative on the student query, but its latent basis is not
    directly compatible with the student tail.
* an O1/O2/O3 variant improves:
    that location is a viable representation-transfer site.
* teacher_full does not improve:
    the clean-geometry teacher itself is not better on the student's physical
    candidate support; feature KD at that site is unlikely to help.

This implementation is Top-1 only.  Top-4 would require preserving the exact
[B,M,K] query order and should be implemented as a separate controlled protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from graspnetAPI import GraspGroup
from torch import nn
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Parse P1-only flags before importing utils.arguments, whose parser consumes
# all remaining command-line arguments during import.
# ---------------------------------------------------------------------------


def _parse_p1_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--distill_stage",
        type=str,
        default="auto",
        choices=("auto", "1", "2"),
        help=(
            "Compatibility flag for existing inference launchers.  The actual "
            "student stage is read from checkpoint metadata; an explicit 1/2 "
            "must match it.  Stage 0 is the separate --teacher_checkpoint."
        ),
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Full Stage-0 privileged clean-geometry checkpoint.",
    )
    parser.add_argument(
        "--p1_oracle",
        type=str,
        default="o1_group",
        choices=(
            "student",
            "teacher_full",
            "o1_group",
            "o2_angle",
            "o3_cdf",
        ),
        help=(
            "student: unchanged Stage-1 control; "
            "teacher_full: full Stage-0 tail on the student physical query; "
            "o1_group: teacher grouped feature -> student CVA decoder; "
            "o2_angle: teacher post-angle-transformer feature -> student branches; "
            "o3_cdf: teacher pre-CDF feature -> student CDF head."
        ),
    )
    parser.add_argument(
        "--p1_allow_stage2_student",
        action="store_true",
        help=(
            "Permit a Stage-2 RGB checkpoint as the student.  The default "
            "requires Stage 1 so P1 remains a clean representation oracle."
        ),
    )
    parser.add_argument(
        "--p1_fix_student_width",
        action="store_true",
        help=(
            "After feature intervention, restore the native student's complete "
            "depth-wise width tensor.  This isolates CDF/action ranking from "
            "width changes.  Default: let the injected representation feed the "
            "student width branch normally."
        ),
    )
    parser.add_argument(
        "--p1_assert_atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for exact physical-center assertions.",
    )
    parser.add_argument(
        "--p1_max_batches",
        type=int,
        default=0,
        help="Smoke-test limit; 0 processes the complete selected split.",
    )
    parser.add_argument(
        "--p1_summary_filename",
        type=str,
        default="p1_feature_oracle_summary.json",
    )
    parser.add_argument(
        "--p1_protocol_filename",
        type=str,
        default="p1_feature_oracle_protocol.json",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


P1_ARGS = _parse_p1_args()


# Existing repository configuration and implementations.
from utils.arguments import cfgs  # noqa: E402
from utils.collision_detector import ModelFreeCollisionDetectorTorch  # noqa: E402
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn  # noqa: E402
from models.economicgrasp_bip3d import pred_decode_center_view_angle  # noqa: E402
from models.economicgrasp_dpt_distill import (  # noqa: E402
    DISTILL_CONTRACT_VERSION,
    economicgrasp_dpt_student,
    economicgrasp_dpt_teacher,
)


FEATURE_ORACLES = {"o1_group", "o2_angle", "o3_cdf"}
EXPECTED_REPOSITORY_COMMIT = "3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2"


# ---------------------------------------------------------------------------
# Dataset / checkpoint helpers.  These intentionally mirror
# inference_cva_distill.py rather than introducing a new protocol.
# ---------------------------------------------------------------------------


def _worker_init(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _build_subset(
    dataset: torch.utils.data.Dataset,
    sample_interval: float,
    annos_per_scene: int = 256,
) -> Tuple[torch.utils.data.Dataset, List[int]]:
    if sample_interval <= 0:
        raise ValueError(
            f"sample_interval must be positive, got {sample_interval}."
        )
    total = len(dataset)
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices: List[int] = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        indices.extend(range(start, end, stride))
    return Subset(dataset, indices), indices


def _move_fixed_inputs(
    batch: MutableMapping[str, Any],
    device: torch.device,
) -> MutableMapping[str, Any]:
    # The network remains point-cloud-free.  Captured points are read only by
    # the unchanged optional collision post-processor.
    for key in (
        "point_clouds",
        "cloud_colors",
        "coordinates_for_voxel",
    ):
        batch.pop(key, None)

    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Unexpected list-valued inference key {key!r}. "
                "Construct the test dataset with load_label=False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def _read_checkpoint(path: str, role: str) -> Tuple[Dict[str, Any], Mapping[str, torch.Tensor]]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{role} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"{role} must be a full Stage-0/1/2 checkpoint with metadata."
        )
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"{role} checkpoint has no valid model state dict: {path}")
    return checkpoint, state


def _require_metadata(checkpoint: Mapping[str, Any], key: str, role: str) -> Any:
    if key not in checkpoint:
        raise RuntimeError(f"{role} checkpoint is missing metadata key {key!r}.")
    return checkpoint[key]


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    role: str,
) -> Dict[str, Any]:
    version = int(_require_metadata(
        checkpoint, "distill_contract_version", role
    ))
    if version != DISTILL_CONTRACT_VERSION:
        raise RuntimeError(
            f"{role} uses distill_contract_version={version}; expected "
            f"{DISTILL_CONTRACT_VERSION}."
        )

    stage = int(_require_metadata(checkpoint, "distill_stage", role))
    seed_mode = str(_require_metadata(
        checkpoint, "seed_selection_mode", role
    ))
    geometry_source = str(_require_metadata(
        checkpoint, "geometry_depth_source", role
    ))
    depth_head_executed = bool(_require_metadata(
        checkpoint, "depth_head_executed", role
    ))
    pose_depth_mode = str(_require_metadata(
        checkpoint, "pose_depth_mode", role
    ))
    use_fuse_depth = bool(_require_metadata(
        checkpoint, "use_fuse_depth", role
    ))

    if seed_mode != "image_fps":
        raise RuntimeError(
            f"{role} must use seed_selection_mode='image_fps', got {seed_mode!r}."
        )
    if bool(checkpoint.get("legacy_dataset_use_gt_depth", True)):
        raise RuntimeError(
            f"{role} used the legacy dataset --use_gt_depth switch and violates "
            "the controlled crop/label protocol."
        )

    if role == "student":
        allowed = (1, 2) if P1_ARGS.p1_allow_stage2_student else (1,)
        if stage not in allowed:
            raise RuntimeError(
                f"P1 student stage must be one of {allowed}, got Stage {stage}."
            )
        requested_stage = str(P1_ARGS.distill_stage)
        if requested_stage != "auto" and int(requested_stage) != stage:
            raise RuntimeError(
                f"--distill_stage={requested_stage} disagrees with the "
                f"student checkpoint metadata Stage {stage}."
            )
        if geometry_source != "pred" or not depth_head_executed:
            raise RuntimeError(
                "The P1 student must execute RGB-predicted metric geometry."
            )
    elif role == "teacher":
        if stage != 0:
            raise RuntimeError(
                f"The P1 teacher must be Stage 0, got Stage {stage}."
            )
        if geometry_source != "gt" or depth_head_executed:
            raise RuntimeError(
                "The P1 teacher must use clean GT geometry and bypass the depth head."
            )
        if pose_depth_mode != "none":
            raise RuntimeError(
                f"Stage-0 teacher must use pose_depth_mode='none', got "
                f"{pose_depth_mode!r}."
            )
    else:
        raise ValueError(f"Unknown checkpoint role {role!r}.")

    return {
        "stage": stage,
        "seed_selection_mode": seed_mode,
        "geometry_depth_source": geometry_source,
        "depth_head_executed": depth_head_executed,
        "pose_depth_mode": pose_depth_mode,
        "use_fuse_depth": use_fuse_depth,
    }


def _load_checkpoint_strict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    role: str,
) -> None:
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
    if missing or unexpected:
        raise RuntimeError(
            f"Strict {role} checkpoint loading failed: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _assert_geometry_role(
    end_points: Mapping[str, Any],
    expected_source: str,
    *,
    context: str,
) -> None:
    required = (
        "D: Geometry depth source GT",
        "D: Depth head executed",
        "depth_map_used_for_geometry",
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise RuntimeError(
            f"{context}: missing geometry-role endpoint(s): {missing}."
        )

    source_is_gt = bool(round(float(
        end_points["D: Geometry depth source GT"].detach().item()
    )))
    depth_head_executed = bool(round(float(
        end_points["D: Depth head executed"].detach().item()
    )))
    expect_gt = expected_source == "gt"
    if source_is_gt != expect_gt or depth_head_executed == expect_gt:
        raise RuntimeError(
            f"{context}: wrong geometry path; expected={expected_source}, "
            f"source_is_gt={source_is_gt}, "
            f"depth_head_executed={depth_head_executed}."
        )

    used = end_points["depth_map_used_for_geometry"]
    if expect_gt:
        if "depth_net_pred" in end_points or "depth_head_raw_pred" in end_points:
            raise RuntimeError(
                f"{context}: Stage-0 unexpectedly executed the depth decoder."
            )
        gt = end_points.get("gt_depth_m")
        if not torch.is_tensor(gt):
            raise RuntimeError(f"{context}: Stage-0 has no tensor gt_depth_m.")
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 4:
            gt = gt[:, :1]
        else:
            raise RuntimeError(
                f"{context}: invalid gt_depth_m shape {tuple(gt.shape)}."
            )
        gt = torch.nan_to_num(
            gt.to(used), nan=0.0, posinf=0.0, neginf=0.0
        )
        if gt.shape != used.shape:
            raise RuntimeError(
                f"{context}: GT/used depth shapes differ: "
                f"{tuple(gt.shape)} vs {tuple(used.shape)}."
            )
        if float((gt - used).abs().max().item()) > P1_ARGS.p1_assert_atol:
            raise RuntimeError(
                f"{context}: clean geometry does not equal gt_depth_m."
            )
    else:
        pred = end_points.get("depth_net_pred")
        if not torch.is_tensor(pred):
            raise RuntimeError(
                f"{context}: RGB student did not export depth_net_pred."
            )
        if pred.shape != used.shape:
            raise RuntimeError(
                f"{context}: predicted/used depth shapes differ."
            )
        if float((pred - used).abs().max().item()) > P1_ARGS.p1_assert_atol:
            raise RuntimeError(
                f"{context}: student geometry is not its predicted depth."
            )


def _fresh_model_input(
    pristine: Mapping[str, Any],
) -> Dict[str, Any]:
    data = dict(pristine)
    for key in (
        "image_fps_seed_idx_override",
        "oracle_view_inds_override",
    ):
        data.pop(key, None)
    data["cva_compute_diagnostics"] = False
    data["geometry_compute_diagnostics"] = False
    data["cva_export_angle_feature"] = False
    data["cva_force_process_grasp_labels"] = False
    return data


# ---------------------------------------------------------------------------
# Query contract and exactness checks.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentQueryContract:
    base_seed_idx: torch.Tensor
    query_pixel_idx: torch.Tensor
    view_idx: torch.Tensor
    center_xyz: torch.Tensor

    @classmethod
    def from_end_points(
        cls,
        end_points: Mapping[str, Any],
        *,
        context: str,
    ) -> "StudentQueryContract":
        keys = (
            "kview_base_token_sel_idx",
            "token_sel_idx",
            "grasp_top_view_inds",
            "xyz_graspable",
        )
        missing = [
            key for key in keys
            if not torch.is_tensor(end_points.get(key))
        ]
        if missing:
            raise KeyError(
                f"{context}: missing query endpoint(s): {missing}."
            )

        base = end_points["kview_base_token_sel_idx"].detach().long()
        query = end_points["token_sel_idx"].detach().long()
        view = end_points["grasp_top_view_inds"].detach().long()
        center = end_points["xyz_graspable"].detach().float()

        if base.dim() != 2 or query.dim() != 2 or view.dim() != 2:
            raise RuntimeError(
                f"{context}: expected rank-2 seed/query/view tensors; got "
                f"base={tuple(base.shape)}, query={tuple(query.shape)}, "
                f"view={tuple(view.shape)}."
            )
        if not (base.shape == query.shape == view.shape):
            raise RuntimeError(
                f"{context}: P1 is Top-1 only; got base={tuple(base.shape)}, "
                f"query={tuple(query.shape)}, view={tuple(view.shape)}."
            )
        if center.shape != (*base.shape, 3):
            raise RuntimeError(
                f"{context}: center must be [B,M,3], got {tuple(center.shape)}."
            )
        if not torch.equal(base, query):
            raise RuntimeError(
                f"{context}: Top-1 image-FPS query pixels differ from base seeds."
            )
        if not torch.isfinite(center).all():
            raise FloatingPointError(f"{context}: non-finite query centers.")

        return cls(
            base_seed_idx=base,
            query_pixel_idx=query,
            view_idx=view,
            center_xyz=center,
        )


def _exact_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return 0.0
    return float((a == b).float().mean().item())


def _assert_same_query(
    reference: StudentQueryContract,
    candidate: StudentQueryContract,
    *,
    context: str,
) -> Dict[str, float]:
    seed_ratio = _exact_ratio(
        reference.base_seed_idx, candidate.base_seed_idx
    )
    pixel_ratio = _exact_ratio(
        reference.query_pixel_idx, candidate.query_pixel_idx
    )
    view_ratio = _exact_ratio(reference.view_idx, candidate.view_idx)
    if min(seed_ratio, pixel_ratio, view_ratio) < 1.0:
        raise RuntimeError(
            f"{context}: query mismatch; seed={seed_ratio:.6f}, "
            f"pixel={pixel_ratio:.6f}, view={view_ratio:.6f}."
        )

    center_abs = (
        reference.center_xyz.to(candidate.center_xyz)
        - candidate.center_xyz
    ).abs()
    center_max = float(center_abs.max().item())
    center_l2 = torch.linalg.vector_norm(
        reference.center_xyz.to(candidate.center_xyz)
        - candidate.center_xyz,
        dim=-1,
    )
    center_l2_mean = float(center_l2.mean().item())
    if center_max > P1_ARGS.p1_assert_atol:
        raise RuntimeError(
            f"{context}: physical centers differ; max_abs={center_max:.3e}, "
            f"atol={P1_ARGS.p1_assert_atol:.3e}."
        )
    return {
        "seed_exact_ratio": seed_ratio,
        "pixel_exact_ratio": pixel_ratio,
        "view_exact_ratio": view_ratio,
        "center_max_abs_m": center_max,
        "center_l2_mean_m": center_l2_mean,
    }


# ---------------------------------------------------------------------------
# Feature capture / intervention.
# ---------------------------------------------------------------------------


def _resolve_feature_site(
    model: nn.Module,
    oracle: str,
) -> Tuple[nn.Module, str]:
    core = getattr(model, "kview_grasp_module", None)
    if core is None:
        raise AttributeError("Model has no kview_grasp_module.")
    if not bool(getattr(core, "use_cdf", False)):
        raise RuntimeError("P1 supports only the CVA-CDF model.")
    decoder = getattr(core, "decoder", None)
    if decoder is None:
        raise AttributeError("CVA module has no decoder.")

    if oracle == "o1_group":
        module = getattr(core, "group", None)
        kind = "output"
    elif oracle == "o2_angle":
        layers = getattr(decoder, "layers", None)
        if layers is None or len(layers) == 0:
            raise AttributeError("CDF decoder has no angle-transformer layers.")
        module = layers[-1]
        kind = "output"
    elif oracle == "o3_cdf":
        module = getattr(decoder, "cdf_head", None)
        kind = "input"
    else:
        raise ValueError(f"{oracle!r} is not a feature-oracle site.")

    if not isinstance(module, nn.Module):
        raise TypeError(
            f"Resolved {oracle} site is not an nn.Module: {type(module)!r}."
        )
    return module, kind


@dataclass
class _FeatureBox:
    oracle: str
    value: Optional[torch.Tensor] = None
    calls: int = 0

    def store(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError(
                f"{self.oracle}: expected tensor feature, got {type(value)!r}."
            )
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError(
                f"{self.oracle}: feature site executed more than once."
            )
        self.value = value.detach().clone()

    def require(self) -> torch.Tensor:
        if self.calls != 1 or self.value is None:
            raise RuntimeError(
                f"{self.oracle}: expected exactly one captured feature; "
                f"calls={self.calls}."
            )
        return self.value


@contextlib.contextmanager
def _capture_feature(
    model: nn.Module,
    oracle: str,
) -> Iterator[_FeatureBox]:
    module, kind = _resolve_feature_site(model, oracle)
    box = _FeatureBox(oracle=oracle)

    if kind == "output":
        def _hook(
            _module: nn.Module,
            _inputs: Tuple[Any, ...],
            output: Any,
        ) -> None:
            box.store(output)

        handle = module.register_forward_hook(_hook)
    else:
        def _pre_hook(
            _module: nn.Module,
            inputs: Tuple[Any, ...],
        ) -> None:
            if not inputs:
                raise RuntimeError(
                    f"{oracle}: target module received no positional input."
                )
            box.store(inputs[0])

        handle = module.register_forward_pre_hook(_pre_hook)

    try:
        yield box
    finally:
        handle.remove()


def _coerce_replacement(
    replacement: torch.Tensor,
    native: torch.Tensor,
    *,
    oracle: str,
) -> torch.Tensor:
    if replacement.shape != native.shape:
        raise RuntimeError(
            f"{oracle}: teacher/student feature shapes differ: "
            f"teacher={tuple(replacement.shape)}, "
            f"student={tuple(native.shape)}."
        )
    converted = replacement.to(
        device=native.device,
        dtype=native.dtype,
    )
    if not torch.isfinite(converted).all():
        raise FloatingPointError(
            f"{oracle}: non-finite teacher replacement feature."
        )
    return converted


@dataclass
class _ReplacementBox:
    oracle: str
    calls: int = 0


@contextlib.contextmanager
def _replace_feature(
    model: nn.Module,
    oracle: str,
    replacement: torch.Tensor,
) -> Iterator[_ReplacementBox]:
    module, kind = _resolve_feature_site(model, oracle)
    box = _ReplacementBox(oracle=oracle)

    if kind == "output":
        def _hook(
            _module: nn.Module,
            _inputs: Tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            if not torch.is_tensor(output):
                raise TypeError(
                    f"{oracle}: native output is not a tensor."
                )
            box.calls += 1
            if box.calls > 1:
                raise RuntimeError(
                    f"{oracle}: replacement site executed more than once."
                )
            return _coerce_replacement(
                replacement,
                output,
                oracle=oracle,
            )

        handle = module.register_forward_hook(_hook)
    else:
        def _pre_hook(
            _module: nn.Module,
            inputs: Tuple[Any, ...],
        ) -> Tuple[Any, ...]:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise TypeError(
                    f"{oracle}: native module input is not a tensor."
                )
            box.calls += 1
            if box.calls > 1:
                raise RuntimeError(
                    f"{oracle}: replacement site executed more than once."
                )
            first = _coerce_replacement(
                replacement,
                inputs[0],
                oracle=oracle,
            )
            return (first, *inputs[1:])

        handle = module.register_forward_pre_hook(_pre_hook)

    try:
        yield box
    finally:
        handle.remove()

    if box.calls != 1:
        raise RuntimeError(
            f"{oracle}: expected exactly one feature replacement; "
            f"calls={box.calls}."
        )


# ---------------------------------------------------------------------------
# Force the teacher's sparse physical center to equal the student's center.
# This is the key difference from prior "same pixel + same view" diagnostics.
# ---------------------------------------------------------------------------


@dataclass
class _CenterOverrideBox:
    calls: int = 0
    native_teacher_xyz: Optional[torch.Tensor] = None


@contextlib.contextmanager
def _force_teacher_student_physical_centers(
    teacher: nn.Module,
    contract: StudentQueryContract,
) -> Iterator[_CenterOverrideBox]:
    method_name = "_select_graspable_seed_queries"
    original_bound = getattr(teacher, method_name, None)
    if original_bound is None:
        raise AttributeError(
            f"Teacher has no {method_name}; current repository API changed."
        )

    had_instance_attribute = method_name in teacher.__dict__
    original_instance_attribute = teacher.__dict__.get(method_name)
    box = _CenterOverrideBox()

    def _wrapped(
        _self: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        output = original_bound(*args, **kwargs)
        if not isinstance(output, tuple) or len(output) != 6:
            raise RuntimeError(
                "Unexpected _select_graspable_seed_queries return contract; "
                f"got {type(output)!r} with length "
                f"{len(output) if isinstance(output, tuple) else 'n/a'}."
            )

        (
            seed_features,
            native_xyz,
            token_idx,
            xyz_all,
            uv_all,
            graspable_num,
        ) = output
        if not torch.is_tensor(native_xyz) or not torch.is_tensor(token_idx):
            raise TypeError(
                "Teacher seed selector did not return tensor xyz/token indices."
            )
        if token_idx.shape != contract.base_seed_idx.shape:
            raise RuntimeError(
                "Teacher/student base seed shapes differ before center override: "
                f"{tuple(token_idx.shape)} vs "
                f"{tuple(contract.base_seed_idx.shape)}."
            )
        teacher_idx = token_idx.to(
            device=contract.base_seed_idx.device,
            dtype=torch.long,
        )
        if not torch.equal(teacher_idx, contract.base_seed_idx):
            mismatch = float(
                (teacher_idx != contract.base_seed_idx)
                .float()
                .mean()
                .item()
            )
            raise RuntimeError(
                "Teacher failed to reuse the student's exact image-FPS seeds; "
                f"mismatch={100.0 * mismatch:.4f}%."
            )
        if native_xyz.shape != contract.center_xyz.shape:
            raise RuntimeError(
                "Teacher native center/student center shapes differ: "
                f"{tuple(native_xyz.shape)} vs "
                f"{tuple(contract.center_xyz.shape)}."
            )

        box.calls += 1
        if box.calls > 1:
            raise RuntimeError(
                "Teacher sparse seed selector executed more than once."
            )
        box.native_teacher_xyz = native_xyz.detach().clone()
        forced_xyz = contract.center_xyz.to(
            device=native_xyz.device,
            dtype=native_xyz.dtype,
        ).detach()

        return (
            seed_features,
            forced_xyz,
            token_idx,
            xyz_all,
            uv_all,
            graspable_num,
        )

    setattr(
        teacher,
        method_name,
        types.MethodType(_wrapped, teacher),
    )
    try:
        yield box
    finally:
        if had_instance_attribute:
            setattr(
                teacher,
                method_name,
                original_instance_attribute,
            )
        else:
            delattr(teacher, method_name)

    if box.calls != 1 or box.native_teacher_xyz is None:
        raise RuntimeError(
            "Teacher physical-center override was not executed exactly once."
        )


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------


@dataclass
class MetricAverager:
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def update(
        self,
        metrics: Mapping[str, float],
        n: int = 1,
    ) -> None:
        for key, value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + value * int(n)
            self.counts[key] = self.counts.get(key, 0) + int(n)

    def averages(self) -> Dict[str, float]:
        return {
            key: self.sums[key] / max(self.counts.get(key, 0), 1)
            for key in sorted(self.sums)
        }


def _feature_statistics(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
) -> Dict[str, float]:
    if student_feature.shape != teacher_feature.shape:
        raise RuntimeError(
            "Cannot compare features with different shapes: "
            f"{tuple(student_feature.shape)} vs "
            f"{tuple(teacher_feature.shape)}."
        )
    s = student_feature.detach().float().reshape(-1)
    t = teacher_feature.detach().float().reshape(-1)
    finite = torch.isfinite(s) & torch.isfinite(t)
    s = s[finite]
    t = t[finite]
    if s.numel() == 0:
        raise FloatingPointError("No finite feature elements are available.")

    s_norm = torch.linalg.vector_norm(s)
    t_norm = torch.linalg.vector_norm(t)
    diff_norm = torch.linalg.vector_norm(s - t)
    cosine = torch.dot(s, t) / (
        s_norm.clamp_min(1.0e-12)
        * t_norm.clamp_min(1.0e-12)
    )
    return {
        "feature_cosine": float(cosine.item()),
        "feature_student_rms": float(torch.sqrt(s.square().mean()).item()),
        "feature_teacher_rms": float(torch.sqrt(t.square().mean()).item()),
        "feature_mean_abs_delta": float((s - t).abs().mean().item()),
        "feature_relative_l2_to_student": float(
            (diff_norm / s_norm.clamp_min(1.0e-12)).item()
        ),
        "feature_numel": float(s.numel()),
    }


def _prediction_delta_statistics(
    native: Mapping[str, Any],
    intervened: Mapping[str, Any],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, tag in (
        ("grasp_cdf_pred_angle_depth", "cdf"),
        ("grasp_width_pred_angle_depth", "width"),
    ):
        a = native.get(key)
        b = intervened.get(key)
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue
        if a.shape != b.shape:
            raise RuntimeError(
                f"{tag} output shapes differ: "
                f"{tuple(a.shape)} vs {tuple(b.shape)}."
            )
        delta = a.detach().float() - b.detach().float()
        metrics[f"{tag}_mean_abs_delta"] = float(
            delta.abs().mean().item()
        )
        metrics[f"{tag}_rms_delta"] = float(
            torch.sqrt(delta.square().mean()).item()
        )
    return metrics


def _native_teacher_center_statistics(
    native_teacher_xyz: torch.Tensor,
    student_xyz: torch.Tensor,
) -> Dict[str, float]:
    native = native_teacher_xyz.detach().float()
    student = student_xyz.detach().float().to(native)
    delta = native - student
    l2 = torch.linalg.vector_norm(delta, dim=-1)
    z = delta[..., 2].abs()
    return {
        "teacher_native_vs_student_center_l2_mean_m": float(
            l2.mean().item()
        ),
        "teacher_native_vs_student_center_l2_p90_m": float(
            torch.quantile(l2.reshape(-1), 0.90).item()
        ),
        "teacher_native_vs_student_center_z_abs_mean_m": float(
            z.mean().item()
        ),
        "teacher_native_vs_student_center_z_abs_p90_m": float(
            torch.quantile(z.reshape(-1), 0.90).item()
        ),
    }


def _copy_native_width(
    native: Mapping[str, Any],
    intervened: MutableMapping[str, Any],
) -> None:
    key = "grasp_width_pred_angle_depth"
    width = native.get(key)
    if not torch.is_tensor(width):
        raise KeyError(
            "--p1_fix_student_width requires native depth-wise width output."
        )
    intervened[key] = width.detach().clone()


# ---------------------------------------------------------------------------
# One P1 batch.
# ---------------------------------------------------------------------------


def _run_p1_batch(
    pristine: Mapping[str, Any],
    *,
    student: nn.Module,
    teacher: Optional[nn.Module],
    student_checked: bool,
    teacher_checked: bool,
) -> Tuple[Dict[str, Any], Dict[str, float], bool, bool]:
    oracle = P1_ARGS.p1_oracle
    metrics: Dict[str, float] = {}

    # Pass 1: native student owns the deployment-time query.
    if oracle in FEATURE_ORACLES:
        with _capture_feature(student, oracle) as native_capture:
            native_end_points = student(
                _fresh_model_input(pristine)
            )
        native_feature = native_capture.require()
    else:
        native_end_points = student(
            _fresh_model_input(pristine)
        )
        native_feature = None

    if not student_checked:
        _assert_geometry_role(
            native_end_points,
            "pred",
            context="P1 native student",
        )
        student_checked = True

    student_contract = StudentQueryContract.from_end_points(
        native_end_points,
        context="P1 native student",
    )

    if oracle == "student":
        metrics.update({
            "student_query_count_per_sample": float(
                student_contract.base_seed_idx.shape[1]
            ),
        })
        return (
            dict(native_end_points),
            metrics,
            student_checked,
            teacher_checked,
        )

    if teacher is None:
        raise RuntimeError(
            f"P1 oracle {oracle!r} requires a Stage-0 teacher."
        )

    # Pass 2: clean-geometry teacher at exact student pixel/view/3D center.
    teacher_input = _fresh_model_input(pristine)
    teacher_input["image_fps_seed_idx_override"] = (
        student_contract.base_seed_idx
    )
    teacher_input["oracle_view_inds_override"] = (
        student_contract.view_idx
    )

    with _force_teacher_student_physical_centers(
        teacher,
        student_contract,
    ) as center_box:
        if oracle in FEATURE_ORACLES:
            with _capture_feature(teacher, oracle) as teacher_capture:
                teacher_end_points = teacher(teacher_input)
            teacher_feature = teacher_capture.require()
        else:
            teacher_end_points = teacher(teacher_input)
            teacher_feature = None

    if not teacher_checked:
        _assert_geometry_role(
            teacher_end_points,
            "gt",
            context="P1 student-conditioned teacher",
        )
        teacher_checked = True

    teacher_contract = StudentQueryContract.from_end_points(
        teacher_end_points,
        context="P1 student-conditioned teacher",
    )
    metrics.update(_assert_same_query(
        student_contract,
        teacher_contract,
        context="P1 teacher/student",
    ))
    metrics.update(_native_teacher_center_statistics(
        center_box.native_teacher_xyz,
        student_contract.center_xyz,
    ))
    metrics["student_query_count_per_sample"] = float(
        student_contract.base_seed_idx.shape[1]
    )

    if oracle == "teacher_full":
        output_end_points = dict(teacher_end_points)
        if P1_ARGS.p1_fix_student_width:
            _copy_native_width(
                native_end_points,
                output_end_points,
            )
        metrics.update(_prediction_delta_statistics(
            native_end_points,
            output_end_points,
        ))
        return (
            output_end_points,
            metrics,
            student_checked,
            teacher_checked,
        )

    if native_feature is None or teacher_feature is None:
        raise RuntimeError(
            f"{oracle}: feature capture unexpectedly returned None."
        )
    metrics.update(_feature_statistics(
        native_feature,
        teacher_feature,
    ))

    # Pass 3: deterministic student rerun with exactly one feature replaced.
    with _replace_feature(
        student,
        oracle,
        teacher_feature,
    ):
        oracle_end_points = student(
            _fresh_model_input(pristine)
        )

    oracle_contract = StudentQueryContract.from_end_points(
        oracle_end_points,
        context=f"P1 {oracle} intervened student",
    )
    rerun_metrics = _assert_same_query(
        student_contract,
        oracle_contract,
        context=f"P1 {oracle} native/intervened student",
    )
    metrics.update({
        f"student_rerun_{key}": value
        for key, value in rerun_metrics.items()
    })

    if P1_ARGS.p1_fix_student_width:
        _copy_native_width(
            native_end_points,
            oracle_end_points,
        )
    metrics.update(_prediction_delta_statistics(
        native_end_points,
        oracle_end_points,
    ))
    return (
        dict(oracle_end_points),
        metrics,
        student_checked,
        teacher_checked,
    )


# ---------------------------------------------------------------------------
# Model construction and result saving.
# ---------------------------------------------------------------------------


def _build_models(
    device: torch.device,
) -> Tuple[
    nn.Module,
    Optional[nn.Module],
    Dict[str, Any],
    Dict[str, Any],
]:
    student_checkpoint, student_state = _read_checkpoint(
        str(cfgs.checkpoint_path or ""),
        "student",
    )
    teacher_checkpoint, teacher_state = _read_checkpoint(
        P1_ARGS.teacher_checkpoint,
        "teacher",
    )
    student_meta = _validate_checkpoint_contract(
        student_checkpoint,
        role="student",
    )
    teacher_meta = _validate_checkpoint_contract(
        teacher_checkpoint,
        role="teacher",
    )

    if student_meta["use_fuse_depth"] != teacher_meta["use_fuse_depth"]:
        raise RuntimeError(
            "Student and teacher checkpoints use different clean-depth "
            "construction: student use_fuse_depth="
            f"{student_meta['use_fuse_depth']}, teacher="
            f"{teacher_meta['use_fuse_depth']}."
        )
    requested_fuse = bool(getattr(cfgs, "use_fuse_depth", False))
    if requested_fuse != student_meta["use_fuse_depth"]:
        raise RuntimeError(
            "--use_fuse_depth must match both checkpoint contracts: "
            f"requested={requested_fuse}, checkpoint="
            f"{student_meta['use_fuse_depth']}."
        )

    common_kwargs = dict(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_cdf=True,
        # Multi-pass oracle inference deliberately disables model-side
        # visualization/diagnostic modules to avoid duplicate side effects.
        vis_dir=None,
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    )
    student = economicgrasp_dpt_student(
        **common_kwargs,
        use_obs_depth=False,
        pose_depth_mode=student_meta["pose_depth_mode"],
        camera_pose_key=str(
            student_checkpoint.get(
                "camera_pose_key",
                "camera_pose_vec",
            )
        ),
        camera_gravity_key=str(
            student_checkpoint.get(
                "camera_gravity_key",
                "camera_gravity_vec",
            )
        ),
        pose_hidden_dim=int(
            student_checkpoint.get("pose_hidden_dim", 64)
        ),
        ray_gravity_hidden_dim=int(
            student_checkpoint.get(
                "ray_gravity_hidden_dim",
                64,
            )
        ),
        ray_gravity_mid_dim=int(
            student_checkpoint.get(
                "ray_gravity_mid_dim",
                32,
            )
        ),
    ).to(device)
    teacher = economicgrasp_dpt_teacher(
        **common_kwargs,
    ).to(device)

    _load_checkpoint_strict(
        student,
        student_state,
        role="student",
    )
    _load_checkpoint_strict(
        teacher,
        teacher_state,
        role="teacher",
    )
    student.eval()
    teacher.eval()
    student.requires_grad_(False)
    teacher.requires_grad_(False)

    # Fail early if the requested injection site is absent/incompatible.
    if P1_ARGS.p1_oracle in FEATURE_ORACLES:
        student_site, student_kind = _resolve_feature_site(
            student,
            P1_ARGS.p1_oracle,
        )
        teacher_site, teacher_kind = _resolve_feature_site(
            teacher,
            P1_ARGS.p1_oracle,
        )
        if student_kind != teacher_kind:
            raise RuntimeError(
                "Teacher/student feature-site hook types differ."
            )
        del student_site, teacher_site

    return student, teacher, student_meta, teacher_meta


def _save_grasp_predictions(
    grasp_preds: List[torch.Tensor],
    *,
    batch_idx: int,
    sampled_indices: List[int],
    full_dataset: GraspNetMultiDataset,
    scene_list: List[str],
) -> int:
    saved = 0
    for sample_i, pred in enumerate(grasp_preds):
        subset_idx = batch_idx * cfgs.batch_size + sample_i
        if subset_idx >= len(sampled_indices):
            raise IndexError(
                f"Subset index {subset_idx} exceeds "
                f"{len(sampled_indices)}."
            )
        data_idx = sampled_indices[subset_idx]
        gg = GraspGroup(pred.detach().cpu().numpy())

        if cfgs.save_nocollision:
            out_dir = os.path.join(
                cfgs.save_dir + "_nocollision",
                scene_list[data_idx],
                cfgs.camera,
            )
            os.makedirs(out_dir, exist_ok=True)
            gg.save_npy(
                os.path.join(
                    out_dir,
                    f"{data_idx % 256:04d}.npy",
                )
            )

        if cfgs.collision_thresh > 0:
            cloud, _ = full_dataset.get_data(
                data_idx,
                return_raw_cloud=True,
            )
            detector = ModelFreeCollisionDetectorTorch(
                cloud.reshape(-1, 3),
                voxel_size=cfgs.collision_voxel_size,
            )
            collision = detector.detect(
                gg,
                approach_dist=0.05,
                collision_thresh=cfgs.collision_thresh,
            )
            gg = gg[~collision.detach().cpu().numpy()]

        out_dir = os.path.join(
            cfgs.save_dir,
            scene_list[data_idx],
            cfgs.camera,
        )
        os.makedirs(out_dir, exist_ok=True)
        gg.save_npy(
            os.path.join(
                out_dir,
                f"{data_idx % 256:04d}.npy",
            )
        )
        saved += 1
    return saved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )


def inference() -> None:
    if not bool(getattr(cfgs, "multi_modal", False)):
        raise RuntimeError("P1 CVA inference requires --multi_modal.")
    if not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("P1 supports only CVA-CDF; add --use_cdf.")
    if bool(getattr(cfgs, "use_top4_view_infer", False)):
        raise RuntimeError(
            "P1 v1 is Top-1 only. Remove --use_top4_view_infer."
        )
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError(
            "The current CVA-CDF model has no learned collision head."
        )
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError(
            "P1 Stage-1 deployment path is RGB-only; remove --use_obs_depth."
        )
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError(
            "Keep the legacy dataset --use_gt_depth switch disabled."
        )
    if not cfgs.checkpoint_path:
        raise ValueError(
            "--checkpoint_path must point to the Stage-1 student checkpoint."
        )
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required.")
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required.")
    if P1_ARGS.p1_assert_atol <= 0:
        raise ValueError("--p1_assert_atol must be positive.")
    if P1_ARGS.p1_max_batches < 0:
        raise ValueError("--p1_max_batches cannot be negative.")

    os.makedirs(cfgs.save_dir, exist_ok=True)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    student, teacher, student_meta, teacher_meta = _build_models(
        device
    )

    requested_fuse = bool(getattr(cfgs, "use_fuse_depth", False))
    full_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        split=cfgs.test_mode,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        # Unchanged controlled data protocol: identical RGB crop/labels;
        # gt_depth_m remains separately available to Stage 0.
        use_gt_depth=False,
        use_fuse_depth=requested_fuse,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
    )
    eval_dataset, sampled_indices = _build_subset(
        full_dataset,
        float(getattr(cfgs, "sample_interval", 1.0)),
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=cfgs.num_workers,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(cfgs.num_workers > 0),
    )
    scene_list = full_dataset.scene_list()

    protocol = {
        "protocol": "student-query-conditioned-feature-oracle-v1",
        "target_repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "oracle": P1_ARGS.p1_oracle,
        "requested_distill_stage": str(P1_ARGS.distill_stage),
        "student_checkpoint": os.path.abspath(str(cfgs.checkpoint_path)),
        "teacher_checkpoint": os.path.abspath(P1_ARGS.teacher_checkpoint),
        "student_checkpoint_metadata": student_meta,
        "teacher_checkpoint_metadata": teacher_meta,
        "student_query_ownership": {
            "seed_pixel": "student deterministic image-FPS",
            "physical_center": "student predicted-depth backprojection",
            "approach_view": "student inference Top-1",
            "angle_grid": "shared canonical A-angle enumeration",
            "depth_grid": "shared canonical D-depth enumeration",
        },
        "teacher_conditioning": {
            "image_seed_override": True,
            "approach_view_override": True,
            "physical_center_override": True,
            "geometry_depth": "clean gt_depth_m",
            "depth_decoder_executed": False,
        },
        "intervention_sites": {
            "o1_group": (
                "ViewConditionedAttentionGrouping output [B,C,Q*A]"
            ),
            "o2_angle": (
                "last CVA angle-transformer output [B*Q,A,H]"
            ),
            "o3_cdf": (
                "input to student final CDF head [B,64,Q*A]"
            ),
            "teacher_full": (
                "full teacher CDF/width tail on student query"
            ),
            "student": "unchanged Stage-1 control",
        },
        "student_tail_retained": (
            P1_ARGS.p1_oracle in FEATURE_ORACLES
        ),
        "native_student_width_restored": bool(
            P1_ARGS.p1_fix_student_width
        ),
        "top4": False,
        "dataset_root": os.path.abspath(cfgs.dataset_root),
        "camera": cfgs.camera,
        "split": cfgs.test_mode,
        "sample_interval": float(
            getattr(cfgs, "sample_interval", 1.0)
        ),
        "batch_size": int(cfgs.batch_size),
        "max_batches": int(P1_ARGS.p1_max_batches),
        "collision_thresh": float(cfgs.collision_thresh),
        "collision_voxel_size": float(cfgs.collision_voxel_size),
        "use_fuse_depth": requested_fuse,
        "data_protocol_changed": False,
    }
    _write_json(
        Path(cfgs.save_dir) / P1_ARGS.p1_protocol_filename,
        protocol,
    )

    print(
        "[P1] oracle={} split={} total={} selected={} batch={} "
        "collision={} fix_width={}".format(
            P1_ARGS.p1_oracle,
            cfgs.test_mode,
            len(full_dataset),
            len(eval_dataset),
            cfgs.batch_size,
            cfgs.collision_thresh,
            int(P1_ARGS.p1_fix_student_width),
        ),
        flush=True,
    )
    print(
        "[P1] query owner=student pixel+predicted-depth center+Top1 view; "
        "teacher geometry=clean GT; data protocol unchanged",
        flush=True,
    )

    start = time.perf_counter()
    processed = 0
    meters = MetricAverager()
    student_checked = False
    teacher_checked = False

    for batch_idx, batch in enumerate(dataloader):
        if (
            P1_ARGS.p1_max_batches > 0
            and batch_idx >= P1_ARGS.p1_max_batches
        ):
            break

        moved = _move_fixed_inputs(batch, device)
        # Every forward mutates its endpoint dictionary.  Preserve a pristine
        # shallow copy whose tensors are reused read-only across the three passes.
        pristine = dict(moved)

        with torch.inference_mode():
            output_end_points, batch_metrics, student_checked, teacher_checked = (
                _run_p1_batch(
                    pristine,
                    student=student,
                    teacher=teacher,
                    student_checked=student_checked,
                    teacher_checked=teacher_checked,
                )
            )
            grasp_preds = pred_decode_center_view_angle(
                output_end_points,
                use_cdf=True,
            )

        current_batch_size = len(grasp_preds)
        meters.update(
            batch_metrics,
            n=current_batch_size,
        )
        processed += _save_grasp_predictions(
            grasp_preds,
            batch_idx=batch_idx,
            sampled_indices=sampled_indices,
            full_dataset=full_dataset,
            scene_list=scene_list,
        )

        if batch_idx % 20 == 0:
            elapsed = time.perf_counter() - start
            avg = meters.averages()
            message = (
                f"[P1] batch={batch_idx}/{len(dataloader)} "
                f"samples={processed}/{len(eval_dataset)} "
                f"sec_per_sample={elapsed / max(processed, 1):.3f}"
            )
            if "feature_cosine" in avg:
                message += (
                    f" feature_cos={avg['feature_cosine']:.4f}"
                    f" feature_relL2="
                    f"{avg['feature_relative_l2_to_student']:.4f}"
                )
            if "cdf_mean_abs_delta" in avg:
                message += (
                    f" cdf_delta={avg['cdf_mean_abs_delta']:.5f}"
                )
            print(message, flush=True)

    elapsed = time.perf_counter() - start
    summary = {
        **protocol,
        "processed_samples": processed,
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / max(processed, 1),
        "diagnostic_means": meters.averages(),
        "complete_selected_split": (
            P1_ARGS.p1_max_batches == 0
            and processed == len(eval_dataset)
        ),
    }
    summary_path = (
        Path(cfgs.save_dir)
        / P1_ARGS.p1_summary_filename
    )
    _write_json(summary_path, summary)
    print(
        f"[P1] done: samples={processed}, "
        f"sec_per_sample={summary['seconds_per_sample']:.3f}",
        flush=True,
    )
    print(f"[P1] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    inference()
