#!/usr/bin/env python3
"""P0-B official-AP inference for selective privileged CDF replacement.

This script performs checkpoint-only paired inference with:

1. a frozen Stage-1 RGB/predicted-depth student;
2. a frozen Stage-0 RGB + clean-GT-depth teacher;
3. exact student image-FPS seeds and exact student selected views;
4. four CDF variants decoded through the repository's canonical
   ``pred_decode_center_view_angle(..., use_cdf=True)``;
5. the standard GraspNet dump layout used by ``eval.py``.

Only ``grasp_cdf_pred_angle_depth`` is replaced.  Student centers, selected
views, depth-wise width predictions, and every other endpoint remain fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup


def _parse_p0b_args() -> argparse.Namespace:
    """Consume P0-B-only flags before ``utils.arguments`` parses the rest."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--p0b_teacher_checkpoint", required=True)
    parser.add_argument("--p0b_student_checkpoint", required=True)
    parser.add_argument(
        "--p0b_variants",
        default="student,teacher_full,teacher_common,oracle_hybrid",
        help="Comma-separated subset of student,teacher_full,teacher_common,oracle_hybrid.",
    )
    parser.add_argument("--p0b_teacher_better_margin", type=float, default=0.0)
    parser.add_argument("--p0b_worker_rank", type=int, default=0)
    parser.add_argument("--p0b_world_size", type=int, default=1)
    parser.add_argument("--p0b_max_batches", type=int, default=0)
    parser.add_argument("--p0b_resume", action="store_true")
    parser.add_argument("--p0b_overwrite", action="store_true")
    parser.add_argument("--p0b_allow_stage2_student", action="store_true")
    parser.add_argument("--p0b_no_gate_csv", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


P0B_ARGS = _parse_p0b_args()

# ``economicgrasp_bip3d`` imports the repository-wide cfgs object, so all
# ordinary model/data/inference flags must remain in utils.arguments.
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import (
    DISTILL_CONTRACT_VERSION,
    economicgrasp_dpt_student,
    economicgrasp_dpt_teacher,
)
from models.loss_economicgrasp_depth_kview_transformer import (
    get_loss as get_loss_economicgrasp,
)
from models.p0b_oracle_hybrid import (
    P0B_VARIANTS,
    assert_exact_teacher_output,
    build_exact_teacher_input,
    build_p0b_cdf_variants,
    make_variant_end_points,
)


CVA_COMMON_CPU_LABEL_KEYS = {
    "object_poses_list",
    "grasp_points_list",
    "view_graspness_list",
    "top_view_index_list",
}
CVA_CDF_CPU_LABEL_KEYS = {
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
}
CVA_CPU_RESIDENT_LABEL_KEYS = CVA_COMMON_CPU_LABEL_KEYS | CVA_CDF_CPU_LABEL_KEYS
UNUSED_POINT_INPUT_KEYS = {
    "point_clouds",
    "cloud_colors",
    "coordinates_for_voxel",
}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _worker_init(worker_id: int) -> None:
    base_seed = np.random.get_state()[1][0]
    np.random.seed(int(base_seed) + int(worker_id))


def _parse_variants(text: str) -> Tuple[str, ...]:
    names = tuple(x.strip() for x in str(text).split(",") if x.strip())
    if not names:
        raise ValueError("--p0b_variants cannot be empty.")
    unknown = sorted(set(names) - set(P0B_VARIANTS))
    if unknown:
        raise ValueError(
            f"Unknown P0-B variants {unknown}; supported={list(P0B_VARIANTS)}."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate P0-B variants are not allowed: {names}.")
    return names


def _read_checkpoint(path: str, role: str) -> Tuple[Dict[str, Any], Mapping[str, torch.Tensor]]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{role} checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"P0-B requires a full {role} checkpoint with model_state_dict and "
            f"privileged-depth metadata: {path}"
        )
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"{role} model_state_dict is not a mapping: {path}")
    return checkpoint, state


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    role: str,
    expected_stages: Sequence[int],
    expected_geometry: str,
) -> None:
    stage = int(checkpoint.get("distill_stage", -1))
    if stage not in set(int(x) for x in expected_stages):
        raise RuntimeError(
            f"P0-B {role} checkpoint has distill_stage={stage}; expected "
            f"one of {list(expected_stages)}."
        )
    version = int(checkpoint.get("distill_contract_version", -1))
    if version != int(DISTILL_CONTRACT_VERSION):
        raise RuntimeError(
            f"P0-B {role} checkpoint contract mismatch: expected "
            f"{DISTILL_CONTRACT_VERSION}, got {version}."
        )
    if str(checkpoint.get("seed_selection_mode", "")) != "image_fps":
        raise RuntimeError(
            f"P0-B {role} checkpoint must use image_fps, got "
            f"{checkpoint.get('seed_selection_mode')!r}."
        )
    if str(checkpoint.get("geometry_depth_source", "")) != expected_geometry:
        raise RuntimeError(
            f"P0-B {role} checkpoint must use geometry_depth_source="
            f"{expected_geometry!r}, got "
            f"{checkpoint.get('geometry_depth_source')!r}."
        )
    expected_head = expected_geometry == "pred"
    if bool(checkpoint.get("depth_head_executed", not expected_head)) != expected_head:
        raise RuntimeError(
            f"P0-B {role} checkpoint has inconsistent depth_head_executed metadata."
        )
    if bool(checkpoint.get("legacy_dataset_use_gt_depth", True)):
        raise RuntimeError(
            f"P0-B {role} checkpoint used the legacy dataset use_gt_depth path."
        )
    if "use_fuse_depth" not in checkpoint:
        raise RuntimeError(f"P0-B {role} checkpoint has no use_fuse_depth metadata.")
    if expected_geometry == "gt" and str(checkpoint.get("pose_depth_mode", "")) != "none":
        raise RuntimeError(
            f"P0-B teacher must use pose_depth_mode='none', got "
            f"{checkpoint.get('pose_depth_mode')!r}."
        )


def _load_state_strict(model: torch.nn.Module, state: Mapping[str, torch.Tensor], role: str) -> None:
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
            f"Strict P0-B {role} loading produced incompatible keys: "
            f"missing={missing}, unexpected={unexpected}."
        )


def _assert_geometry_role(
    end_points: Mapping[str, Any],
    *,
    expected_source: str,
    context: str,
) -> None:
    required = (
        "D: Geometry depth source GT",
        "D: Depth head executed",
        "depth_map_used_for_geometry",
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise RuntimeError(f"{context}: missing geometry endpoints {missing}.")
    source_is_gt = bool(round(float(
        end_points["D: Geometry depth source GT"].detach().item()
    )))
    head_executed = bool(round(float(
        end_points["D: Depth head executed"].detach().item()
    )))
    expect_gt = expected_source == "gt"
    if source_is_gt != expect_gt or head_executed == expect_gt:
        raise RuntimeError(
            f"{context}: wrong geometry role; expected={expected_source}, "
            f"source_is_gt={source_is_gt}, head_executed={head_executed}."
        )


def _validate_label_contract(batch: Mapping[str, Any]) -> None:
    required = CVA_CPU_RESIDENT_LABEL_KEYS | {"cdf_thresholds"}
    missing = sorted(key for key in required if key not in batch)
    if missing:
        raise KeyError(f"P0-B CDF dataset is missing keys: {missing}.")
    batch_size = len(batch["object_poses_list"])
    for key in sorted(CVA_CPU_RESIDENT_LABEL_KEYS):
        value = batch[key]
        if not isinstance(value, (list, tuple)) or len(value) != batch_size:
            raise TypeError(
                f"P0-B {key} must be a batch list of length {batch_size}."
            )
    for batch_i in range(batch_size):
        num_objects = len(batch["object_poses_list"][batch_i])
        if num_objects <= 0:
            raise RuntimeError(f"P0-B sample {batch_i} has no object labels.")
        for key in sorted(CVA_CPU_RESIDENT_LABEL_KEYS):
            if len(batch[key][batch_i]) != num_objects:
                raise RuntimeError(
                    f"P0-B {key}[{batch_i}] has {len(batch[key][batch_i])} "
                    f"objects; expected {num_objects}."
                )


def _drop_unused_point_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    for key in UNUSED_POINT_INPUT_KEYS:
        batch.pop(key, None)
    return batch


def _recursive_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {k: _recursive_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        raise TypeError(
            "Unexpected nested value outside the explicit CPU-resident CDF label contract."
        )
    return value


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key, value in list(batch.items()):
        if key in CVA_CPU_RESIDENT_LABEL_KEYS:
            continue
        batch[key] = _recursive_to_device(value, device)
    return batch


def _assert_cpu_labels(batch: Mapping[str, Any]) -> None:
    for key in CVA_CPU_RESIDENT_LABEL_KEYS:
        value = batch.get(key)
        if value is None:
            continue
        for batch_i, sample in enumerate(value):
            for object_i, tensor in enumerate(sample):
                if not torch.is_tensor(tensor) or tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"P0-B {key}[{batch_i}][{object_i}] must remain a CPU tensor."
                    )


def _sample_indices(total: int, sample_fraction: float, annos_per_scene: int = 256) -> List[int]:
    fraction = float(sample_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0:
        raise ValueError(f"sample_interval must be positive, got {sample_fraction}.")
    if fraction >= 1.0:
        return list(range(total))
    stride = max(1, int(round(1.0 / fraction)))
    indices: List[int] = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        indices.extend(range(start, end, stride))
    return indices


def _shard_indices_like_distributed_sampler(
    indices: Sequence[int],
    scene_list: Sequence[str],
    *,
    rank: int,
    world_size: int,
) -> Tuple[List[int], List[str]]:
    """Use the no-padding part of DistributedSampler's deterministic order.

    Priority-1 used ``DistributedSampler(shuffle=False)``.  For the official
    full split (7,680 frames), the length is divisible by the usual three-GPU
    world size, so ``indices[rank::world_size]`` exactly reproduces each rank's
    sample order and local-batch composition without initializing DDP.  The
    no-padding rule also prevents two independent inference workers from
    writing the same prediction file on non-divisible pilot subsets.
    """
    worker_indices = list(indices)[rank::world_size]
    assigned_scenes: List[str] = []
    seen = set()
    for index in worker_indices:
        scene = str(scene_list[index])
        if scene not in seen:
            assigned_scenes.append(scene)
            seen.add(scene)
    return worker_indices, assigned_scenes


def _prediction_path(
    save_root: Path,
    variant: str,
    scene_name: str,
    camera: str,
    anno_id: int,
) -> Path:
    return save_root / variant / scene_name / camera / f"{anno_id:04d}.npy"


def _existing_policy_state(
    indices: Sequence[int],
    scene_list: Sequence[str],
    *,
    save_root: Path,
    camera: str,
    variants: Sequence[str],
    resume: bool,
    overwrite: bool,
) -> set[int]:
    """Return samples whose complete outputs may be skipped on resume.

    Do not remove completed samples from ``indices``.  The Priority-1 oracle
    gate uses batch-level threshold balancing, so filtering samples would change
    the local batch composition and could change the gate for a partially
    completed neighboring sample.  The DataLoader therefore keeps its original
    deterministic batches; fully completed batches can be skipped as a unit,
    while partial batches are forwarded in full and only incomplete samples are
    written.
    """
    if resume and overwrite:
        raise ValueError("--p0b_resume and --p0b_overwrite are mutually exclusive.")
    completed: set[int] = set()
    first_existing: Path | None = None
    for index in indices:
        scene = str(scene_list[index])
        anno = int(index % 256)
        paths = [
            _prediction_path(save_root, variant, scene, camera, anno)
            for variant in variants
        ]
        existing = [path.exists() for path in paths]
        if resume and all(existing):
            completed.add(int(index))
            continue
        if any(existing) and not (resume or overwrite):
            first_existing = paths[existing.index(True)]
            break
    if first_existing is not None:
        raise FileExistsError(
            f"P0-B output already exists: {first_existing}. Use --p0b_resume "
            "to skip complete samples or --p0b_overwrite to replace outputs."
        )
    return completed


def _checkpoint_identity(path: str, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "distill_stage": int(checkpoint.get("distill_stage", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "geometry_depth_source": str(checkpoint.get("geometry_depth_source", "")),
        "pose_depth_mode": str(checkpoint.get("pose_depth_mode", "")),
    }


def _save_grasp_group(
    pred: torch.Tensor,
    *,
    full_dataset: GraspNetMultiDataset,
    data_index: int,
    variant: str,
    save_root: Path,
    collision_detector: Any = None,
) -> None:
    array = pred.detach().cpu().numpy()
    if array.ndim != 2 or array.shape[1] != 17:
        raise RuntimeError(
            f"P0-B decoded grasp tensor must be [N,17], got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise RuntimeError(
            f"P0-B decoded non-finite grasp values for data index {data_index}, "
            f"variant={variant}."
        )
    gg = GraspGroup(array)
    scene_name = str(full_dataset.scene_list()[data_index])
    anno_id = int(data_index % 256)

    if bool(getattr(cfgs, "save_nocollision", False)):
        raw_path = _prediction_path(
            save_root,
            variant + "_nocollision",
            scene_name,
            cfgs.camera,
            anno_id,
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        gg.save_npy(str(raw_path))

    if float(getattr(cfgs, "collision_thresh", 0.0)) > 0.0:
        if collision_detector is None:
            raise RuntimeError(
                "P0-B collision filtering was requested without a shared detector."
            )
        collision = collision_detector.detect(
            gg,
            approach_dist=0.05,
            collision_thresh=float(cfgs.collision_thresh),
        )
        gg = gg[~collision.detach().cpu().numpy()]

    path = _prediction_path(
        save_root,
        variant,
        scene_name,
        cfgs.camera,
        anno_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gg.save_npy(str(path))


def _build_models(
    device: torch.device,
    teacher_checkpoint: Mapping[str, Any],
    teacher_state: Mapping[str, torch.Tensor],
    student_checkpoint: Mapping[str, Any],
    student_state: Mapping[str, torch.Tensor],
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    common = dict(
        min_depth=float(cfgs.min_depth),
        max_depth=float(cfgs.max_depth),
        bin_num=int(cfgs.bin_num),
        is_training=False,
        use_cdf=True,
        vis_dir=None,
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    )
    student = economicgrasp_dpt_student(
        **common,
        use_obs_depth=False,
        pose_depth_mode=str(student_checkpoint.get("pose_depth_mode", "none")),
        camera_pose_key=str(student_checkpoint.get("camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(
            student_checkpoint.get("camera_gravity_key", "camera_gravity_vec")
        ),
        pose_hidden_dim=int(student_checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(
            student_checkpoint.get("ray_gravity_hidden_dim", 64)
        ),
        ray_gravity_mid_dim=int(
            student_checkpoint.get("ray_gravity_mid_dim", 32)
        ),
    ).to(device)
    teacher = economicgrasp_dpt_teacher(**common).to(device)
    _load_state_strict(student, student_state, "student")
    _load_state_strict(teacher, teacher_state, "teacher")
    student.eval().requires_grad_(False)
    teacher.eval().requires_grad_(False)
    return student, teacher


def inference() -> None:
    variants = _parse_variants(P0B_ARGS.p0b_variants)
    margin = float(P0B_ARGS.p0b_teacher_better_margin)
    if margin < 0.0:
        raise ValueError("--p0b_teacher_better_margin must be non-negative.")
    rank = int(P0B_ARGS.p0b_worker_rank)
    world_size = int(P0B_ARGS.p0b_world_size)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid P0-B worker rank/world size: rank={rank}, world_size={world_size}."
        )
    if not bool(getattr(cfgs, "multi_modal", False)):
        raise RuntimeError("P0-B requires --multi_modal.")
    if not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("P0-B requires --use_cdf.")
    if bool(getattr(cfgs, "use_top4_view_infer", False)):
        raise RuntimeError(
            "P0-B exact-view oracle evaluation currently requires Top-1 view "
            "queries. Remove --use_top4_view_infer."
        )
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError("P0-B student must remain RGB/predicted-depth; remove --use_obs_depth.")
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError(
            "Keep dataset --use_gt_depth disabled; the teacher reads gt_depth_m internally."
        )
    if not cfgs.save_dir or not cfgs.test_mode:
        raise ValueError("P0-B requires --save_dir and --test_mode.")
    if cfgs.test_mode not in {"test_seen", "test_similar", "test_novel"}:
        raise ValueError(f"Unsupported P0-B test_mode={cfgs.test_mode!r}.")

    _seed_everything(int(getattr(cfgs, "seed", 0)) + rank)

    teacher_checkpoint, teacher_state = _read_checkpoint(
        P0B_ARGS.p0b_teacher_checkpoint,
        "teacher",
    )
    student_checkpoint, student_state = _read_checkpoint(
        P0B_ARGS.p0b_student_checkpoint,
        "student",
    )
    _validate_checkpoint(
        teacher_checkpoint,
        role="teacher",
        expected_stages=(0,),
        expected_geometry="gt",
    )
    allowed_student_stages = (1, 2) if P0B_ARGS.p0b_allow_stage2_student else (1,)
    _validate_checkpoint(
        student_checkpoint,
        role="student",
        expected_stages=allowed_student_stages,
        expected_geometry="pred",
    )

    teacher_fuse = bool(teacher_checkpoint["use_fuse_depth"])
    student_fuse = bool(student_checkpoint["use_fuse_depth"])
    if teacher_fuse != student_fuse:
        raise RuntimeError(
            "P0-B teacher/student use_fuse_depth metadata differs: "
            f"teacher={teacher_fuse}, student={student_fuse}."
        )
    requested_fuse = bool(getattr(cfgs, "use_fuse_depth", False))
    if requested_fuse != teacher_fuse:
        raise RuntimeError(
            "P0-B --use_fuse_depth must match both checkpoints: "
            f"requested={requested_fuse}, checkpoint={teacher_fuse}."
        )

    save_root = Path(cfgs.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    meta_dir = save_root / "_p0b_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        camera=cfgs.camera,
        split=cfgs.test_mode,
        num_points=int(cfgs.num_point),
        voxel_size=float(cfgs.voxel_size),
        remove_outlier=True,
        augment=False,
        load_label=True,
        use_gt_depth=False,
        use_fuse_depth=teacher_fuse,
        graspness_mode=str(cfgs.graspness_mode or "scene"),
        min_depth=float(cfgs.min_depth),
        max_depth=float(cfgs.max_depth),
        bin_num=int(cfgs.bin_num),
        depth_strides=1,
        extend_angle=True,
        load_grasp_payload=False,
    )
    dataset = CVAExtendedLabelAdapter(
        base_dataset,
        dataset_root=cfgs.dataset_root,
        use_cdf=True,
        label_folder=str(cfgs.cdf_label_folder),
        num_angle=int(cfgs.num_angle),
        num_depth=int(cfgs.num_depth),
    )
    scene_list = base_dataset.scene_list()

    all_indices = _sample_indices(
        len(dataset),
        float(getattr(cfgs, "sample_interval", 1.0)),
    )
    worker_indices, assigned_scenes = _shard_indices_like_distributed_sampler(
        all_indices,
        scene_list,
        rank=rank,
        world_size=world_size,
    )
    completed_indices = _existing_policy_state(
        worker_indices,
        scene_list,
        save_root=save_root,
        camera=str(cfgs.camera),
        variants=variants,
        resume=bool(P0B_ARGS.p0b_resume),
        overwrite=bool(P0B_ARGS.p0b_overwrite),
    )
    samples_requiring_output = len(worker_indices) - len(completed_indices)
    eval_dataset = Subset(dataset, worker_indices)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=int(cfgs.batch_size),
        shuffle=False,
        num_workers=max(int(cfgs.num_workers), 0),
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        persistent_workers=(int(cfgs.num_workers) > 0),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student, teacher = _build_models(
        device,
        teacher_checkpoint,
        teacher_state,
        student_checkpoint,
        student_state,
    )
    del teacher_state, student_state

    print(
        f"[P0-B][worker {rank}/{world_size}] split={cfgs.test_mode} "
        f"scenes={len(assigned_scenes)} samples={len(worker_indices)} "
        f"need_output={samples_requiring_output} "
        f"batch={cfgs.batch_size} variants={','.join(variants)}",
        flush=True,
    )
    print(
        f"[P0-B] student_stage={student_checkpoint['distill_stage']} "
        f"teacher_stage={teacher_checkpoint['distill_stage']} "
        f"use_fuse_depth={int(teacher_fuse)} margin={margin:g} "
        f"collision_thresh={float(cfgs.collision_thresh):g}",
        flush=True,
    )

    diagnostics_sum: Dict[str, float] = {}
    diagnostics_weight = 0
    gate_rows_by_index: Dict[int, Dict[str, Any]] = {}
    gate_path = meta_dir / f"worker_{rank:02d}_gate_summary.csv"
    if bool(P0B_ARGS.p0b_resume) and gate_path.is_file():
        with gate_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                gate_rows_by_index[int(row["data_index"])] = dict(row)
    processed = 0
    saved = 0
    skipped_complete = 0
    role_checked = False
    exact_min = {
        "seed": 1.0,
        "pixel": 1.0,
        "view": 1.0,
    }
    start_time = time.perf_counter()

    for batch_idx, raw_batch in enumerate(dataloader):
        if int(P0B_ARGS.p0b_max_batches) > 0 and batch_idx >= int(P0B_ARGS.p0b_max_batches):
            break
        dataset_idx_value = raw_batch.get("dataset_idx")
        if not torch.is_tensor(dataset_idx_value):
            raise KeyError("P0-B batch has no tensor dataset_idx for deterministic saving.")
        batch_data_indices = [
            int(value) for value in dataset_idx_value.detach().cpu().reshape(-1).tolist()
        ]
        local_offset = batch_idx * int(cfgs.batch_size)
        expected_batch_indices = [
            int(value)
            for value in worker_indices[local_offset:local_offset + len(batch_data_indices)]
        ]
        if batch_data_indices != expected_batch_indices:
            raise RuntimeError(
                "P0-B DataLoader order differs from deterministic worker indices: "
                f"batch={batch_data_indices}, expected={expected_batch_indices}."
            )
        if bool(P0B_ARGS.p0b_resume) and all(
            index in completed_indices for index in batch_data_indices
        ):
            skipped_complete += len(batch_data_indices)
            continue
        _validate_label_contract(raw_batch)
        batch = _drop_unused_point_inputs(dict(raw_batch))
        batch = _move_batch_to_device(batch, device)
        _assert_cpu_labels(batch)
        batch.pop("image_fps_seed_idx_override", None)
        batch.pop("oracle_view_inds_override", None)
        batch["cva_force_process_grasp_labels"] = True
        batch["cva_compute_diagnostics"] = False
        batch["geometry_compute_diagnostics"] = False
        batch["cva_export_angle_feature"] = False

        # Preserve a pristine top-level mapping before the student forward mutates
        # its endpoint dictionary. Nested CPU label lists are deliberately shared.
        teacher_source = dict(batch)

        with torch.no_grad():
            student_end_points = student(batch)
            student_end_points["epoch"] = 0
            _, student_end_points = get_loss_economicgrasp(
                student_end_points,
                use_cdf=True,
            )

            teacher_input = build_exact_teacher_input(
                teacher_source,
                student_end_points,
                force_process_grasp_labels=True,
            )
            teacher_end_points = teacher(teacher_input)
            teacher_end_points["epoch"] = 0
            _, teacher_end_points = get_loss_economicgrasp(
                teacher_end_points,
                use_cdf=True,
            )

            if not role_checked:
                _assert_geometry_role(
                    student_end_points,
                    expected_source="pred",
                    context="P0-B student",
                )
                _assert_geometry_role(
                    teacher_end_points,
                    expected_source="gt",
                    context="P0-B teacher",
                )
                role_checked = True

            exact = assert_exact_teacher_output(
                student_end_points,
                teacher_end_points,
            )
            exact_min["seed"] = min(
                exact_min["seed"],
                float(exact["kview_base_token_sel_idx"].item()),
            )
            exact_min["pixel"] = min(
                exact_min["pixel"],
                float(exact["token_sel_idx"].item()),
            )
            exact_min["view"] = min(
                exact_min["view"],
                float(exact["grasp_top_view_inds"].item()),
            )

            bundle = build_p0b_cdf_variants(
                student_end_points,
                teacher_end_points,
                teacher_better_margin=margin,
            )
            decoded: Dict[str, List[torch.Tensor]] = {}
            for variant in variants:
                variant_end_points = make_variant_end_points(
                    student_end_points,
                    bundle.logits_btqad[variant],
                )
                decoded[variant] = pred_decode_center_view_angle(
                    variant_end_points,
                    use_cdf=True,
                )

        batch_size_actual = len(next(iter(decoded.values())))
        if batch_size_actual != len(batch_data_indices):
            raise RuntimeError(
                f"P0-B decoded batch size {batch_size_actual} differs from "
                f"dataset_idx count {len(batch_data_indices)}."
            )
        for sample_i, data_index in enumerate(batch_data_indices):
            should_write = not (
                bool(P0B_ARGS.p0b_resume) and data_index in completed_indices
            )
            if should_write:
                collision_detector = None
                if float(getattr(cfgs, "collision_thresh", 0.0)) > 0.0:
                    cloud, _ = base_dataset.get_data(
                        data_index, return_raw_cloud=True
                    )
                    collision_detector = ModelFreeCollisionDetectorTorch(
                        cloud.reshape(-1, 3),
                        voxel_size=float(cfgs.collision_voxel_size),
                    )
                for variant in variants:
                    _save_grasp_group(
                        decoded[variant][sample_i],
                        full_dataset=base_dataset,
                        data_index=data_index,
                        variant=variant,
                        save_root=save_root,
                        collision_detector=collision_detector,
                    )
                saved += 1

            if not P0B_ARGS.p0b_no_gate_csv:
                gate_valid = bundle.gate_valid_bq[sample_i]
                teacher_better = bundle.teacher_better_bq[sample_i]
                common_valid = bundle.common_valid_bqad[sample_i]
                gate_rows_by_index[data_index] = {
                    "data_index": data_index,
                    "scene": str(scene_list[data_index]),
                    "anno_id": int(data_index % 256),
                    "num_queries": int(gate_valid.numel()),
                    "gate_valid_queries": int(gate_valid.sum().item()),
                    "teacher_better_queries": int(teacher_better.sum().item()),
                    "common_valid_elements": int(common_valid.sum().item()),
                    "total_angle_depth_elements": int(common_valid.numel()),
                }
            processed += 1

        for key, value in bundle.diagnostics.items():
            scalar = float(value.detach().item())
            diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + scalar * batch_size_actual
        diagnostics_weight += batch_size_actual

        if batch_idx % 20 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"[P0-B][worker {rank}] batch={batch_idx}/{len(dataloader)} "
                f"forwarded={processed}/{len(worker_indices)} saved={saved} "
                f"skipped={skipped_complete} "
                f"sec_per_sample={elapsed / max(processed, 1):.3f} "
                f"teacher_better={float(bundle.diagnostics['teacher_better_query_ratio']):.4f}",
                flush=True,
            )

        del decoded, bundle, teacher_end_points, student_end_points, teacher_input, teacher_source

    elapsed = time.perf_counter() - start_time
    diagnostic_means = {
        key: value / max(diagnostics_weight, 1)
        for key, value in sorted(diagnostics_sum.items())
    }
    summary = {
        "protocol": "P0-B-official-AP-oracle-hybrid-v1",
        "worker_rank": rank,
        "world_size": world_size,
        "split": str(cfgs.test_mode),
        "camera": str(cfgs.camera),
        "variants": list(variants),
        "teacher_better_margin": margin,
        "sample_interval_fraction": float(cfgs.sample_interval),
        "assigned_scenes": assigned_scenes,
        "sharding": "distributed_sampler_strided_without_padding",
        "batch_size": int(cfgs.batch_size),
        "num_workers": int(cfgs.num_workers),
        "processed_samples": processed,
        "saved_samples": saved,
        "skipped_complete_samples": skipped_complete,
        "worker_samples_total": len(worker_indices),
        "samples_requiring_output_at_start": samples_requiring_output,
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / max(processed, 1),
        "exact_min": exact_min,
        "collision_thresh": float(cfgs.collision_thresh),
        "student_checkpoint": _checkpoint_identity(
            P0B_ARGS.p0b_student_checkpoint,
            student_checkpoint,
        ),
        "teacher_checkpoint": _checkpoint_identity(
            P0B_ARGS.p0b_teacher_checkpoint,
            teacher_checkpoint,
        ),
        "diagnostic_means": diagnostic_means,
    }
    summary_path = meta_dir / f"worker_{rank:02d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if gate_rows_by_index and not P0B_ARGS.p0b_no_gate_csv:
        ordered_gate_rows = [
            gate_rows_by_index[index] for index in sorted(gate_rows_by_index)
        ]
        with gate_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(ordered_gate_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(ordered_gate_rows)

    print(
        f"[P0-B][worker {rank}] complete: forwarded={processed}, saved={saved}, "
        f"skipped={skipped_complete}, "
        f"elapsed={elapsed / 3600.0:.3f}h, exact={exact_min}, "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    inference()
