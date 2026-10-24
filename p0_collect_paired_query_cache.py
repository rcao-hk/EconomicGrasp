#!/usr/bin/env python3
"""Collect a student-query-aligned teacher/student cache for P0-C and P0-D.

This collector uses the P0-only Stage-0 and Stage-1 wrappers from
``models/economicgrasp_dpt_p0.py``. The student owns the ordered image-FPS seed
pixels and selected view indices. The teacher reuses that exact image-space
query contract through existing model controls, then recomputes its 3D centers
from clean geometry. It exports paired CDF logits and intermediate features.

No repository source patch or runtime-hook installation is required.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--student_checkpoint", required=True)
    p.add_argument("--teacher_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="train", choices=("train", "test_seen", "test_similar", "test_novel"))
    p.add_argument("--scene_ids", default="")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--camera", default="realsense")
    p.add_argument("--num_point", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--graspness_mode", default="scene")
    p.add_argument(
        "--cva_label_folder",
        default=os.environ.get(
            "CVA_LABEL_FOLDER",
            os.environ.get(
                "CDF_LABEL_FOLDER",
                "economic_grasp_label_300views_extend_angle_cdf_depth",
            ),
        ),
        help=(
            "Common extended-angle CDF label directory relative to dataset_root. "
            "Must match the folder used by train_cva_distill_ddp.py."
        ),
    )
    p.add_argument("--gt_target_key", default="")
    p.add_argument("--gt_target_semantics", choices=("auto", "cdf", "friction", "quality_1p1_minus_mu"), default="auto")
    p.add_argument("--valid_mask_key", default="")
    p.add_argument("--support_iou_key", default="")
    p.add_argument("--capture_strict", type=int, choices=(0, 1), default=0)
    p.add_argument("--compress", type=int, choices=(0, 1), default=0)
    p.add_argument("--overwrite", type=int, choices=(0, 1), default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inspect_only", type=int, choices=(0, 1), default=0)
    return p


ARGS = parser().parse_args()
sys.argv[:] = [sys.argv[0]]

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pkd_p0.common import (
    CDF_THRESHOLDS,
    ContractError,
    annotation_ids,
    atomic_json_dump,
    atomic_npz_dump,
    scene_ids_for_split,
    seed_everything,
)
from pkd_p0.feature_capture import FeatureCapture
from pkd_p0.repo_adapter import (
    CDF_LOGIT_ALIASES,
    RepoImports,
    DeterministicSubset,
    build_current_model,
    build_dataset,
    dataset_index_records,
    extract_core_outputs,
    extract_query_override,
    forward_model,
    validate_cdf_batch_label_contract,
)


TARGET_ALIASES = (
    "grasp_cdf_label_angle_depth",
    "grasp_cdf_target_angle_depth",
    "batch_grasp_cdf",
    "grasp_cdf_label",
    "cdf_target",
    "cdf_targets",
)
CDF_BIN_ALIASES = (
    # Current CVA-CDF label processor publishes compact onset bins [B,Q,A,D].
    "batch_grasp_cdf_bins_angle_depth",
    "grasp_cdf_bins_angle_depth",
)
FRICTION_ALIASES = (
    "batch_grasp_friction",
    "grasp_friction_label",
    "batch_grasp_score",
    "grasp_score_label",
)
VALID_ALIASES = (
    "batch_grasp_cdf_valid_mask",
    "grasp_cdf_valid_mask",
    "common_valid_mask",
    "cdf_valid_mask",
    "kd_valid_mask",
)


def selected_records(dataset: Any) -> List[Tuple[int, int, int]]:
    scenes = set(scene_ids_for_split(ARGS.split, ARGS.scene_ids))
    annos = set(annotation_ids(float(ARGS.sample_interval)))
    records = [record for record in dataset_index_records(dataset) if record[1] in scenes and record[2] in annos]
    if int(ARGS.max_samples) > 0:
        records = records[: int(ARGS.max_samples)]
    return records


def find_tensor(mapping: Mapping[str, Any], explicit: str, aliases: Sequence[str]) -> Tuple[str, torch.Tensor]:
    if explicit:
        value = mapping.get(explicit)
        if not torch.is_tensor(value):
            raise KeyError(f"Explicit endpoint {explicit!r} is absent or not a tensor")
        return explicit, value
    for key in aliases:
        value = mapping.get(key)
        if torch.is_tensor(value):
            return key, value
    normalized_aliases = {re.sub(r"[^a-z0-9]", "", key.lower()) for key in aliases}
    for key, value in mapping.items():
        if torch.is_tensor(value) and re.sub(r"[^a-z0-9]", "", str(key).lower()) in normalized_aliases:
            return str(key), value
    raise KeyError(f"None of endpoints {list(aliases)} found. Available tensor keys: {sorted(str(k) for k,v in mapping.items() if torch.is_tensor(v))[:400]}")


def threshold_last(tensor: torch.Tensor) -> torch.Tensor:
    axes = [axis for axis, size in enumerate(tensor.shape) if int(size) == len(CDF_THRESHOLDS)]
    if not axes:
        raise ValueError(f"No CDF threshold axis T=6 in {tuple(tensor.shape)}")
    axis = tensor.ndim - 1 if tensor.shape[-1] == len(CDF_THRESHOLDS) else axes[0]
    return tensor.movedim(axis, -1)


def _cdf_from_compact_bins(
    bins: torch.Tensor,
    num_thresholds: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert current compact onset bins [B,Q,A,D] to [B,Q,A,D,T]."""
    bins = bins.long()
    ids = torch.arange(
        int(num_thresholds),
        device=bins.device,
        dtype=bins.dtype,
    )
    return (
        (bins.unsqueeze(-1) > 0)
        & (ids >= bins.unsqueeze(-1) - 1)
    ).to(dtype=dtype)


def target_cdf(
    output: Mapping[str, Any],
    student_logits: torch.Tensor,
) -> Tuple[str, torch.Tensor]:
    """Resolve the student's current GT CDF in threshold-last layout."""
    semantics = ARGS.gt_target_semantics

    # The current implementation stores an onset bin rather than six explicit
    # binary targets. Prefer that unambiguous representation in auto mode.
    if semantics in {"auto", "cdf"}:
        compact_error: Optional[Exception] = None
        try:
            explicit = ARGS.gt_target_key
            aliases = CDF_BIN_ALIASES if not explicit else (explicit,)
            key, value = find_tensor(output, explicit, aliases)
            if value.ndim == student_logits.ndim - 1:
                if tuple(value.shape) != tuple(student_logits.shape[:-1]):
                    raise ValueError(
                        f"Compact CDF bins {tuple(value.shape)} do not match "
                        f"logits {tuple(student_logits.shape)}."
                    )
                return key, _cdf_from_compact_bins(
                    value,
                    student_logits.shape[-1],
                    student_logits.dtype,
                )
            # An explicitly named endpoint may already contain T=6 targets.
            cdf = threshold_last(value.float())
            if tuple(cdf.shape) != tuple(student_logits.shape):
                cdf = torch.broadcast_to(cdf, student_logits.shape)
            return key, cdf
        except Exception as exc:
            compact_error = exc
            if semantics == "cdf" and ARGS.gt_target_key:
                raise

        try:
            key, value = find_tensor(output, ARGS.gt_target_key, TARGET_ALIASES)
            if value.ndim == student_logits.ndim - 1:
                if tuple(value.shape) != tuple(student_logits.shape[:-1]):
                    raise ValueError(
                        f"CDF target {tuple(value.shape)} does not match logits "
                        f"{tuple(student_logits.shape)}."
                    )
                return key, _cdf_from_compact_bins(
                    value,
                    student_logits.shape[-1],
                    student_logits.dtype,
                )
            cdf = threshold_last(value.float())
            if tuple(cdf.shape) != tuple(student_logits.shape):
                cdf = torch.broadcast_to(cdf, student_logits.shape)
            return key, cdf
        except Exception:
            if semantics == "cdf" or ARGS.gt_target_key:
                if compact_error is not None:
                    raise compact_error
                raise

    key, value = find_tensor(output, ARGS.gt_target_key, FRICTION_ALIASES)
    friction = value.float()
    if semantics == "auto":
        sample = friction.detach().reshape(-1)
        sample = sample[torch.isfinite(sample)][:10000]
        allowed = friction.new_tensor((-1.0, 0.0, *CDF_THRESHOLDS))
        distance = (
            torch.min(
                torch.abs(sample[:, None] - allowed[None, :]), dim=1
            ).values
            if sample.numel()
            else sample
        )
        if sample.numel() and float(
            (distance < 2e-3).float().mean().item()
        ) > 0.95:
            semantics = "friction"
        else:
            raise ContractError(
                f"Endpoint {key!r} is not an explicit CDF tensor and its "
                "values are ambiguous. Set --gt_target_semantics friction or "
                "quality_1p1_minus_mu explicitly."
            )
    if semantics == "quality_1p1_minus_mu":
        friction = torch.where(
            friction > 0.0,
            1.1 - friction,
            friction.new_full((), -1.0),
        )
    thresholds = friction.new_tensor(CDF_THRESHOLDS)
    cdf = (
        (friction.unsqueeze(-1) > 0.0)
        & (friction.unsqueeze(-1) <= thresholds)
    ).float()
    if tuple(cdf.shape) != tuple(student_logits.shape):
        cdf = torch.broadcast_to(cdf, student_logits.shape)
    return key, cdf


def resolve_valid_mask(
    output: Mapping[str, Any],
    target_shape: Sequence[int],
    explicit: str = "",
) -> Tuple[str, torch.Tensor]:
    """Resolve and broadcast a current CDF validity mask to [B,Q,A,D]."""
    try:
        key, value = find_tensor(output, explicit, VALID_ALIASES)
    except Exception:
        if explicit:
            raise
        reference = next(
            value for value in output.values() if torch.is_tensor(value)
        )
        return "all", torch.ones(
            tuple(int(x) for x in target_shape),
            device=reference.device,
            dtype=torch.bool,
        )
    mask = value.bool()
    while mask.ndim > len(target_shape) and mask.shape[-1] == 1:
        mask = mask[..., 0]
    while mask.ndim < len(target_shape):
        mask = mask.unsqueeze(-1)
    try:
        mask = torch.broadcast_to(mask, tuple(int(x) for x in target_shape))
    except RuntimeError as exc:
        raise ValueError(
            f"Validity endpoint {key!r} with shape {tuple(value.shape)} cannot "
            f"match CDF operation shape {tuple(target_shape)}."
        ) from exc
    return key, mask.bool()


def matched_center_z_error(student: Mapping[str, Any], teacher_native: Mapping[str, Any]) -> torch.Tensor:
    s = extract_core_outputs(student)
    t = extract_core_outputs(teacher_native)
    s_center = s["centers"]
    t_center = t["centers"]
    s_token = s.get("token_indices")
    t_token = t.get("token_indices")
    if tuple(s_center.shape) == tuple(t_center.shape) and s_token is not None and t_token is not None:
        # Match by token id independently for each batch item.
        errors = []
        for batch_index in range(s_center.shape[0]):
            lookup = {int(token): index for index, token in enumerate(t_token[batch_index].detach().cpu().tolist())}
            rows = []
            for index, token in enumerate(s_token[batch_index].detach().cpu().tolist()):
                teacher_index = lookup.get(int(token))
                rows.append(
                    torch.tensor(float("nan"), device=s_center.device)
                    if teacher_index is None
                    else torch.abs(s_center[batch_index, index, 2] - t_center[batch_index, teacher_index, 2])
                )
            errors.append(torch.stack(rows))
        return torch.stack(errors)
    if tuple(s_center.shape) == tuple(t_center.shape):
        return torch.abs(s_center[..., 2] - t_center[..., 2])
    raise ContractError(f"Cannot align native teacher/student centers: {tuple(s_center.shape)} vs {tuple(t_center.shape)}")


def feature_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def endpoint_feature_fallback(output: Mapping[str, Any], layer: str) -> Optional[torch.Tensor]:
    """Find a named endpoint feature when no module hook was available."""
    tokens = {
        "seed": (("seed", "feature"), ("graspable", "feature")),
        "pre_view": (("pre", "view", "feature"), ("view", "input", "feature")),
        "selected_view": (("selected", "view", "feature"), ("top", "view", "feature")),
        "local": (("local", "feature"), ("cva", "feature"), ("group", "feature")),
        "pre_cdf": (("pre", "cdf", "feature"), ("cdf", "head", "input")),
    }.get(layer, ())
    candidates: List[Tuple[str, torch.Tensor]] = []
    for key, value in output.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            continue
        lower = str(key).lower()
        if any(all(token in lower for token in pattern) for pattern in tokens):
            candidates.append((str(key), value))
    if len(candidates) == 1:
        return candidates[0][1].detach()
    return None


def augment_captured_features(captured: Dict[str, torch.Tensor], output: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    result = dict(captured)
    for layer in ("seed", "pre_view", "selected_view", "local", "pre_cdf"):
        if layer not in result:
            value = endpoint_feature_fallback(output, layer)
            if value is not None:
                result[layer] = value
    return result


def main() -> None:
    seed_everything(int(ARGS.seed))
    device = torch.device(ARGS.device)
    repo = RepoImports()
    student, _, student_contract = build_current_model(
        repo, checkpoint_path=ARGS.student_checkpoint, device=device,
        min_depth=float(ARGS.min_depth), max_depth=float(ARGS.max_depth), bin_num=int(ARGS.bin_num), is_training=True,
    )
    teacher, _, teacher_contract = build_current_model(
        repo, checkpoint_path=ARGS.teacher_checkpoint, device=device,
        min_depth=float(ARGS.min_depth), max_depth=float(ARGS.max_depth), bin_num=int(ARGS.bin_num), is_training=False,
    )
    if student_contract.distill_stage != 1 or teacher_contract.distill_stage != 0:
        raise ContractError("Expected Stage-1 student and Stage-0 teacher")
    # Keep the constructor's training contract so label processing and the
    # student view-sampling path match PKD training, while disabling dropout and
    # gradients for deterministic cache export.
    student.eval().requires_grad_(False)
    teacher.eval().requires_grad_(False)

    student_capture = FeatureCapture(student, strict=bool(ARGS.capture_strict))
    teacher_capture = FeatureCapture(teacher, strict=bool(ARGS.capture_strict))
    dataset = build_dataset(
        repo,
        dataset_root=ARGS.dataset_root,
        split=ARGS.split,
        camera=ARGS.camera,
        num_point=int(ARGS.num_point),
        min_depth=float(ARGS.min_depth),
        max_depth=float(ARGS.max_depth),
        bin_num=int(ARGS.bin_num),
        use_fuse_depth=student_contract.use_fuse_depth,
        graspness_mode=ARGS.graspness_mode,
        load_label=True,
        use_gt_depth=False,
        use_cdf=True,
        cva_label_folder=ARGS.cva_label_folder,
        num_angle=int(getattr(repo.cfgs, "num_angle", 12)),
        num_depth=int(getattr(repo.cfgs, "num_depth", 4)),
    )
    records = selected_records(dataset)
    subset = DeterministicSubset(dataset, [record[0] for record in records], int(ARGS.seed))
    loader = DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=max(0, int(ARGS.num_workers)),
        collate_fn=repo.collate_fn, pin_memory=False, persistent_workers=int(ARGS.num_workers) > 0,
    )
    output_dir = Path(ARGS.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract: Dict[str, Any] = {
        "experiment": "P0 paired student-query cache",
        "student": student_contract.to_dict(),
        "teacher": teacher_contract.to_dict(),
        "split": ARGS.split,
        "sample_interval": float(ARGS.sample_interval),
        "dataset_label_adapter": getattr(
            dataset, "pkd_p0_label_adapter", type(dataset).__name__
        ),
        "cva_label_folder": getattr(
            dataset, "pkd_p0_cdf_label_folder", ARGS.cva_label_folder
        ),
        "cdf_label_payload": (
            "grasp_cdf_bins_list + grasp_widths_depth_list + "
            "grasp_width_valids_depth_list + cdf_thresholds"
        ),
        "student_feature_hooks": student_capture.contract(),
        "teacher_feature_hooks": teacher_capture.contract(),
        "query_alignment": (
            "student ordered image-FPS pixels and selected views replayed in "
            "teacher; teacher XYZ recomputed from clean geometry"
        ),
    }

    first_contract_written = False
    completed = 0
    started = time.time()
    try:
        for local_index, batch in enumerate(loader):
            dataset_index, scene_id, anno_id = records[local_index]
            out_path = output_dir / f"scene_{scene_id:04d}" / f"ann_{anno_id:04d}.npz"
            if out_path.is_file() and not bool(ARGS.overwrite):
                continue
            sample_seed = int(ARGS.seed) + dataset_index * 1_000_003
            validate_cdf_batch_label_contract(batch)
            student_capture.clear()
            teacher_capture.clear()
            with torch.inference_mode():
                # Native teacher is needed only for center correspondence diagnostics
                # and deliberately skips label matching.
                teacher_native = forward_model(
                    repo,
                    teacher,
                    teacher_contract,
                    batch,
                    device=device,
                    seed=sample_seed,
                    force_process_grasp_labels=False,
                )
                teacher_capture.clear()
                # The student uses the same current compact-CDF labels as the
                # canonical Stage-1/2 trainer.
                student_output = forward_model(
                    repo,
                    student,
                    student_contract,
                    batch,
                    device=device,
                    seed=sample_seed,
                    force_process_grasp_labels=True,
                )
                student_features = augment_captured_features(dict(student_capture.values), student_output)
                query = extract_query_override(student_output)
                teacher_capture.clear()
                teacher_aligned = forward_model(
                    repo, teacher, teacher_contract, batch,
                    device=device,
                    seed=sample_seed,
                    forced_query=query,
                    require_override_marker=True,
                    force_process_grasp_labels=True,
                )
                teacher_features = augment_captured_features(dict(teacher_capture.values), teacher_aligned)

            student_logits = threshold_last(extract_core_outputs(student_output)["cdf_logits"].float())
            teacher_logits = threshold_last(extract_core_outputs(teacher_aligned)["cdf_logits"].float())
            if tuple(student_logits.shape) != tuple(teacher_logits.shape):
                raise ContractError(f"Aligned CDF shapes differ: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}")
            target_source = dict(batch)
            target_source.update(student_output)
            target_key, gt = target_cdf(target_source, student_logits)
            student_valid_key, student_valid = resolve_valid_mask(
                student_output,
                student_logits.shape[:-1],
                ARGS.valid_mask_key,
            )
            teacher_valid_key, teacher_valid = resolve_valid_mask(
                teacher_aligned,
                teacher_logits.shape[:-1],
                ARGS.valid_mask_key,
            )
            common_valid = student_valid & teacher_valid
            valid_key = f"student:{student_valid_key};teacher:{teacher_valid_key}"
            student_bce = F.binary_cross_entropy_with_logits(
                student_logits, gt, reduction="none"
            ).mean(dim=-1)
            teacher_bce = F.binary_cross_entropy_with_logits(
                teacher_logits, gt, reduction="none"
            ).mean(dim=-1)
            teacher_better = common_valid & (teacher_bce < student_bce)
            z_error_q = matched_center_z_error(student_output, teacher_native)

            arrays: Dict[str, np.ndarray] = {
                "student_cdf_logits": feature_array(student_logits),
                "teacher_cdf_logits": feature_array(teacher_logits),
                "gt_cdf": feature_array(gt),
                "valid_mask": common_valid.detach().cpu().numpy().astype(np.uint8),
                "common_valid": common_valid.detach().cpu().numpy().astype(np.uint8),
                "student_valid_mask": student_valid.detach().cpu().numpy().astype(np.uint8),
                "teacher_valid_mask": teacher_valid.detach().cpu().numpy().astype(np.uint8),
                "teacher_better": teacher_better.detach().cpu().numpy().astype(np.uint8),
                "center_z_error": feature_array(z_error_q),
                "scene_id": np.asarray([scene_id], dtype=np.int16),
                "anno_id": np.asarray([anno_id], dtype=np.int16),
                "dataset_idx": np.asarray([dataset_index], dtype=np.int32),
                "student_checkpoint_sha256": np.asarray(student_contract.sha256),
                "teacher_checkpoint_sha256": np.asarray(teacher_contract.sha256),
                "target_endpoint": np.asarray(target_key),
                "valid_endpoint": np.asarray(valid_key),
            }
            for layer, tensor in student_features.items():
                arrays[f"feature_{layer}_student"] = feature_array(tensor)
            for layer, tensor in teacher_features.items():
                arrays[f"feature_{layer}_teacher"] = feature_array(tensor)
            for canonical, tensor in extract_core_outputs(student_output).items():
                if canonical in {"cdf_logits", "width", "geometry_depth", "predicted_depth"}:
                    continue
                arrays[f"student_{canonical}"] = feature_array(tensor)
            for canonical, tensor in extract_core_outputs(teacher_aligned).items():
                if canonical in {"cdf_logits", "width", "geometry_depth", "predicted_depth"}:
                    continue
                arrays[f"teacher_{canonical}"] = feature_array(tensor)

            if bool(ARGS.inspect_only):
                contract.update({
                    "target_endpoint": target_key,
                    "valid_endpoint": valid_key,
                    "student_feature_hooks_after_forward": student_capture.contract(),
                    "teacher_feature_hooks_after_forward": teacher_capture.contract(),
                    "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                    "student_tensor_keys": sorted(str(key) for key, value in student_output.items() if torch.is_tensor(value)),
                    "teacher_tensor_keys": sorted(str(key) for key, value in teacher_aligned.items() if torch.is_tensor(value)),
                })
                atomic_json_dump(contract, output_dir / "inspect_contract.json")
                print(json.dumps(contract, indent=2, sort_keys=True))
                return

            atomic_npz_dump(out_path, compress=bool(ARGS.compress), **arrays)
            if not first_contract_written:
                contract.update({
                    "target_endpoint": target_key,
                    "valid_endpoint": valid_key,
                    "student_feature_hooks_after_forward": student_capture.contract(),
                    "teacher_feature_hooks_after_forward": teacher_capture.contract(),
                    "array_shapes_example": {key: list(value.shape) for key, value in arrays.items()},
                })
                atomic_json_dump(contract, output_dir / "paired_cache_contract.json")
                first_contract_written = True
            completed += 1
            print(
                f"[P0-CACHE] {completed}/{len(records)} scene={scene_id:04d} ann={anno_id:04d} "
                f"teacher_better={float(teacher_better[common_valid].float().mean().item()) if bool(common_valid.any()) else float('nan'):.3f} "
                f"elapsed={(time.time()-started)/60.0:.1f}m",
                flush=True,
            )
    finally:
        student_capture.close()
        teacher_capture.close()

    atomic_json_dump(
        {**contract, "status": "complete", "num_files_new": completed, "elapsed_seconds": time.time() - started},
        output_dir / "paired_cache_complete.json",
    )
    print(f"[DONE] paired cache: {output_dir}")


if __name__ == "__main__":
    main()
