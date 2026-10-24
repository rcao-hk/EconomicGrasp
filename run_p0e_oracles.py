#!/usr/bin/env python3
"""Generate P0-E privileged grasp-field oracle dumps.

The script consumes ordinary GraspNet prediction dumps and writes four standard
prediction roots:

* ``student_original``;
* ``exact_action_rerank``;
* ``local_field_oracle``;
* ``proposal_union_oracle``.

It performs no neural-network inference. Every privileged label is computed for
an explicit physical ``[R, t, width, depth]`` action with the repository's
clean-CAD/DexNet evaluator. The generated roots can be evaluated unchanged with
``GraspNetEval``.
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
from p0e_oracle_common import (
    FIXED_INPUT_COLLISION_THRESH,
    FRICTION_THRESHOLDS,
    PROTOCOL_VERSION,
    FrameSpec,
    LocalPerturbation,
    apply_oracle_scores,
    atomic_save_json,
    atomic_save_npy,
    atomic_save_npz,
    build_frame_list,
    build_local_perturbations,
    deduplicate_physical_actions,
    friction_to_utility,
    grasp_path,
    load_grasps,
    link_or_copy,
    meta_path,
    parse_int_ranges,
    pick_best_local_actions,
    resolve_prediction_root,
    select_local_base_indices,
    shard_frames,
    stack_local_lattice,
    summarize_exact_labels,
    top_n_by_score,
    topk_metric,
    verify_fixed_input_collision_policy,
)


SUPPORTED_MODES = ("rerank", "local_field", "proposal_union")
MODE_TO_VARIANT = {
    "rerank": "exact_action_rerank",
    "local_field": "local_field_oracle",
    "proposal_union": "proposal_union_oracle",
}


def _parse_modes(text: str) -> Tuple[str, ...]:
    modes = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not modes:
        raise ValueError("--modes cannot be empty.")
    unknown = sorted(set(modes) - set(SUPPORTED_MODES))
    if unknown:
        raise ValueError(f"Unknown P0-E modes {unknown}; supported={SUPPORTED_MODES}.")
    if len(modes) != len(set(modes)):
        raise ValueError(f"Duplicate P0-E modes are unsupported: {modes}.")
    return modes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate exact-action, local-field, and proposal-union P0-E oracles.",
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument(
        "--student_dump",
        required=True,
        help=(
            "Student dump root. Accepts flat autonomous dumps, <root>/<split>/..., "
            "or a P0-B root with <root>/<split>/student/... ."
        ),
    )
    parser.add_argument(
        "--teacher_dump",
        default="",
        help=(
            "Autonomous clean-depth teacher root. Accepts flat or split-nested "
            "layouts. Required for proposal_union. P0-B teacher_full/common are "
            "intentionally rejected because they retain student proposals."
        ),
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument(
        "--split",
        required=True,
        choices=("test_seen", "test_similar", "test_novel"),
    )
    parser.add_argument(
        "--modes",
        default=",".join(SUPPORTED_MODES),
        help="Comma-separated subset of rerank,local_field,proposal_union.",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Use annotation IDs 0,K,2K,... when --anno_ids is empty.",
    )
    parser.add_argument(
        "--scene_ids",
        default="",
        help="Optional comma/range filter, e.g. 100-102,107.",
    )
    parser.add_argument(
        "--anno_ids",
        default="",
        help="Optional comma/range filter, e.g. 0,10,20-25.",
    )
    parser.add_argument("--worker_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument(
        "--shard_mode",
        choices=("auto", "scene", "frame"),
        default="auto",
        help=(
            "Worker sharding policy. 'scene' avoids loading the same CAD scene "
            "in every worker; 'auto' falls back to frame sharding for narrow pilots."
        ),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum frames for this worker; 0 means all assigned frames.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing input frames instead of failing.",
    )

    parser.add_argument(
        "--student_top_n",
        type=int,
        default=0,
        help="Retain only top-N student actions before all oracles; 0 keeps all.",
    )
    parser.add_argument(
        "--teacher_top_n",
        type=int,
        default=0,
        help="Retain only top-N teacher actions for proposal union; 0 keeps all.",
    )
    parser.add_argument(
        "--tie_break_eps",
        type=float,
        default=1.0e-4,
        help="Original-score tie breaker, smaller than one CDF utility bin.",
    )

    parser.add_argument(
        "--local_top_n_per_object",
        type=int,
        default=10,
        help="Local-search proposals retained per assigned object; <=0 keeps all.",
    )
    parser.add_argument(
        "--local_global_top_n",
        type=int,
        default=0,
        help="Optional global cap after per-object selection; 0 disables it.",
    )
    parser.add_argument("--local_translation_mm", type=float, default=5.0)
    parser.add_argument("--local_inplane_deg", type=float, default=15.0)
    parser.add_argument("--local_depth_delta_m", type=float, default=0.01)
    parser.add_argument("--local_width_delta_m", type=float, default=0.005)
    parser.add_argument(
        "--local_view_tilt_deg",
        type=float,
        default=0.0,
        help="Optional +/- local Y/Z tilt; 0 keeps the default 13-action lattice.",
    )
    parser.add_argument("--min_width_m", type=float, default=0.0)
    parser.add_argument("--max_width_m", type=float, default=0.10)
    parser.add_argument("--min_depth_m", type=float, default=0.0)
    parser.add_argument("--max_depth_m", type=float, default=0.10)

    parser.add_argument("--collision_chunk", type=int, default=512)
    parser.add_argument(
        "--fc_mode",
        choices=("official", "reuse_contacts"),
        default="reuse_contacts",
    )
    parser.add_argument(
        "--fc_verify_n",
        type=int,
        default=0,
        help="Verify N optimized force-closure labels per frame against stock official labels.",
    )
    parser.add_argument("--strict_evaluator", type=int, choices=(0, 1), default=1)
    parser.add_argument("--compress_meta", type=int, choices=(0, 1), default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    return parser.parse_args()


def _required_variants(modes: Sequence[str]) -> Tuple[str, ...]:
    return ("student_original", *(MODE_TO_VARIANT[mode] for mode in modes))


def _frame_outputs_complete(
    output_root: Path,
    frame: FrameSpec,
    variants: Sequence[str],
) -> bool:
    return all(
        grasp_path(output_root / variant, frame).is_file() for variant in variants
    ) and meta_path(output_root, frame).is_file()


def _guard_existing_outputs(
    *,
    output_root: Path,
    frames: Sequence[FrameSpec],
    variants: Sequence[str],
    resume: bool,
    overwrite: bool,
) -> None:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if resume or overwrite:
        return
    for frame in frames:
        paths = [grasp_path(output_root / variant, frame) for variant in variants]
        paths.append(meta_path(output_root, frame))
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"P0-E output already exists: {existing[0]}. Use --resume or --overwrite."
            )


def _save_variant(
    output_root: Path,
    variant: str,
    frame: FrameSpec,
    grasps: np.ndarray,
) -> Path:
    path = grasp_path(output_root / variant, frame)
    atomic_save_npy(path, grasps)
    return path


def _save_student_original(
    *,
    output_root: Path,
    frame: FrameSpec,
    source_path: Path,
    student_grasps: np.ndarray,
    student_top_n: int,
) -> Path:
    """Hard-link/copy unchanged student dumps when no Top-N truncation is used."""
    path = grasp_path(output_root / "student_original", frame)
    if int(student_top_n) <= 0:
        link_or_copy(source_path, path)
    else:
        atomic_save_npy(path, student_grasps)
    return path


def _eval_payload(result) -> Dict[str, np.ndarray]:
    return {
        "assigned_obj": np.asarray(result.assigned_obj, dtype=np.int16),
        "collision_or_empty": np.asarray(result.collision_or_empty, dtype=np.uint8),
        "pure_collision": np.asarray(result.pure_collision, dtype=np.uint8),
        "empty": np.asarray(result.empty, dtype=np.uint8),
        "friction": np.asarray(result.friction, dtype=np.float32),
    }


def _evaluate(
    evaluator: ExactGraspNetActionEvaluator,
    frame: FrameSpec,
    grasps: np.ndarray,
):
    started = time.perf_counter()
    result = evaluator.evaluate(frame.scene_id, frame.anno_id, grasps)
    return result, time.perf_counter() - started


def _frame_summary_base(
    *,
    frame: FrameSpec,
    student_grasps: np.ndarray,
    student_result,
    student_eval_sec: float,
) -> Dict[str, object]:
    labels = summarize_exact_labels(
        student_result.friction, student_result.collision_or_empty
    )
    return {
        "scene_id": frame.scene_id,
        "anno_id": frame.anno_id,
        "num_student_actions": int(student_grasps.shape[0]),
        "student_eval_sec": float(student_eval_sec),
        "student_top1_safe04": topk_metric(
            student_result.friction,
            student_grasps[:, 0],
            top_k=1,
            threshold=0.4,
        ),
        "student_top10_safe04": topk_metric(
            student_result.friction,
            student_grasps[:, 0],
            top_k=10,
            threshold=0.4,
        ),
        "student_top10_safe08": topk_metric(
            student_result.friction,
            student_grasps[:, 0],
            top_k=10,
            threshold=0.8,
        ),
        **{f"student_{key}": value for key, value in labels.items()},
    }


def _run_local_field(
    *,
    evaluator: ExactGraspNetActionEvaluator,
    frame: FrameSpec,
    student_grasps: np.ndarray,
    student_result,
    perturbations: Sequence[LocalPerturbation],
    args: argparse.Namespace,
):
    selected = select_local_base_indices(
        student_grasps,
        student_result.assigned_obj,
        top_n_per_object=int(args.local_top_n_per_object),
        global_top_n=int(args.local_global_top_n),
    )
    output_friction = np.asarray(student_result.friction, dtype=np.float32).copy()
    local_output = student_grasps.copy()
    best_perturb_id = np.zeros(selected.shape[0], dtype=np.int16)
    best_friction = output_friction[selected].copy()
    best_utility = friction_to_utility(best_friction)
    local_eval_sec = 0.0
    lattice = np.zeros((0, len(perturbations), 17), dtype=np.float32)
    local_assigned = np.zeros((0, len(perturbations)), dtype=np.int16)
    local_collision = np.zeros((0, len(perturbations)), dtype=np.uint8)
    local_pure_collision = np.zeros((0, len(perturbations)), dtype=np.uint8)
    local_empty = np.zeros((0, len(perturbations)), dtype=np.uint8)
    local_friction = np.zeros((0, len(perturbations)), dtype=np.float32)
    evaluator_actions = 0
    logical_actions = int(selected.size * len(perturbations))
    identity_reused = 0
    dedup_saved = 0

    if selected.size > 0:
        lattice = stack_local_lattice(
            student_grasps[selected],
            perturbations,
            min_width_m=float(args.min_width_m),
            max_width_m=float(args.max_width_m),
            min_depth_m=float(args.min_depth_m),
            max_depth_m=float(args.max_depth_m),
        )
        local_shape = lattice.shape[:2]
        local_assigned = np.zeros(local_shape, dtype=np.int16)
        local_collision = np.zeros(local_shape, dtype=np.uint8)
        local_pure_collision = np.zeros(local_shape, dtype=np.uint8)
        local_empty = np.zeros(local_shape, dtype=np.uint8)
        local_friction = np.full(local_shape, -1.0, dtype=np.float32)

        # Identity actions are normally byte-equivalent to the already evaluated
        # student actions. Reuse those exact labels instead of paying for another
        # CAD/collision/force-closure pass. If clipping changed an identity action,
        # that row remains in the evaluator mask and is handled normally.
        flat_lattice = lattice.reshape(-1, 17)
        eval_mask = np.ones(local_shape, dtype=bool)
        identity_same = np.all(
            np.isclose(
                lattice[:, 0, 1:16],
                student_grasps[selected, 1:16],
                rtol=0.0,
                atol=1.0e-7,
            ),
            axis=1,
        )
        identity_rows = np.flatnonzero(identity_same)
        if identity_rows.size:
            source_rows = selected[identity_rows]
            local_assigned[identity_rows, 0] = np.asarray(
                student_result.assigned_obj[source_rows], dtype=np.int16
            )
            local_collision[identity_rows, 0] = np.asarray(
                student_result.collision_or_empty[source_rows], dtype=np.uint8
            )
            local_pure_collision[identity_rows, 0] = np.asarray(
                student_result.pure_collision[source_rows], dtype=np.uint8
            )
            local_empty[identity_rows, 0] = np.asarray(
                student_result.empty[source_rows], dtype=np.uint8
            )
            local_friction[identity_rows, 0] = np.asarray(
                student_result.friction[source_rows], dtype=np.float32
            )
            eval_mask[identity_rows, 0] = False
            identity_reused = int(identity_rows.size)

        eval_flat_ids = np.flatnonzero(eval_mask.reshape(-1))
        if eval_flat_ids.size:
            requested_actions = flat_lattice[eval_flat_ids]
            unique_actions, inverse = deduplicate_physical_actions(requested_actions)
            evaluator_actions = int(unique_actions.shape[0])
            dedup_saved = int(requested_actions.shape[0] - unique_actions.shape[0])
            local_result, local_eval_sec = _evaluate(evaluator, frame, unique_actions)

            def expand(value, dtype):
                return np.asarray(value, dtype=dtype)[inverse]

            local_assigned.reshape(-1)[eval_flat_ids] = expand(
                local_result.assigned_obj, np.int16
            )
            local_collision.reshape(-1)[eval_flat_ids] = expand(
                local_result.collision_or_empty, np.uint8
            )
            local_pure_collision.reshape(-1)[eval_flat_ids] = expand(
                local_result.pure_collision, np.uint8
            )
            local_empty.reshape(-1)[eval_flat_ids] = expand(
                local_result.empty, np.uint8
            )
            local_friction.reshape(-1)[eval_flat_ids] = expand(
                local_result.friction, np.float32
            )

        (
            best_actions,
            best_friction,
            best_utility,
            best_perturb_id,
        ) = pick_best_local_actions(lattice, local_friction)
        local_output[selected] = best_actions
        output_friction[selected] = best_friction

    local_output, local_scores = apply_oracle_scores(
        local_output,
        output_friction,
        original_scores=student_grasps[:, 0],
        tie_break_eps=float(args.tie_break_eps),
    )
    base_utility = friction_to_utility(student_result.friction[selected])
    improved = best_utility > base_utility + 1.0e-8
    changed = best_perturb_id != 0
    perturb_names = np.asarray([item.name for item in perturbations], dtype="U64")
    name_counts = Counter(
        perturb_names[best_perturb_id].tolist() if best_perturb_id.size else []
    )

    metadata = {
        "local_selected_indices": selected.astype(np.int32),
        "local_lattice_actions": lattice.astype(np.float32),
        "local_lattice_assigned_obj": local_assigned,
        "local_lattice_collision_or_empty": local_collision,
        "local_lattice_pure_collision": local_pure_collision,
        "local_lattice_empty": local_empty,
        "local_lattice_friction": local_friction,
        "local_lattice_utility": friction_to_utility(local_friction).astype(np.float32),
        "local_best_perturb_id": best_perturb_id.astype(np.int16),
        "local_best_perturb_name": (
            perturb_names[best_perturb_id]
            if best_perturb_id.size
            else np.asarray([], dtype="U64")
        ),
        "local_base_friction": np.asarray(
            student_result.friction[selected], dtype=np.float32
        ),
        "local_best_friction": best_friction.astype(np.float32),
        "local_base_utility": base_utility.astype(np.float32),
        "local_best_utility": best_utility.astype(np.float32),
        "local_output_friction": output_friction.astype(np.float32),
        "local_output_oracle_utility": local_scores.utility.astype(np.float32),
    }
    summary = {
        "local_num_selected": int(selected.size),
        "local_num_perturbations": int(len(perturbations)),
        # Preserve the logical lattice size for comparison with v1.1.
        "local_num_evaluated_actions": logical_actions,
        "local_num_evaluator_actions_physical": int(evaluator_actions),
        "local_identity_labels_reused": int(identity_reused),
        "local_deduplicated_actions_saved": int(dedup_saved),
        "local_evaluator_action_reduction_ratio": (
            1.0 - float(evaluator_actions) / float(logical_actions)
            if logical_actions > 0
            else float("nan")
        ),
        "local_eval_sec": float(local_eval_sec),
        "local_improved_ratio": (
            float(improved.mean()) if improved.size else float("nan")
        ),
        "local_changed_ratio": (
            float(changed.mean()) if changed.size else float("nan")
        ),
        "local_mean_utility_gain_selected": (
            float((best_utility - base_utility).mean())
            if selected.size
            else float("nan")
        ),
        "local_max_utility_gain_selected": (
            float((best_utility - base_utility).max())
            if selected.size
            else float("nan")
        ),
        "local_top10_safe04": topk_metric(
            output_friction, local_output[:, 0], top_k=10, threshold=0.4
        ),
        "local_top10_safe08": topk_metric(
            output_friction, local_output[:, 0], top_k=10, threshold=0.8
        ),
        "local_best_perturb_counts": ";".join(
            f"{name}:{count}" for name, count in sorted(name_counts.items())
        ),
    }
    return local_output, metadata, summary


def _run_proposal_union(
    *,
    evaluator: ExactGraspNetActionEvaluator,
    frame: FrameSpec,
    student_grasps: np.ndarray,
    student_result,
    teacher_path: Path,
    args: argparse.Namespace,
):
    teacher_full = load_grasps(teacher_path)
    teacher, teacher_source_index = top_n_by_score(
        teacher_full, int(args.teacher_top_n)
    )
    teacher_result, teacher_eval_sec = _evaluate(evaluator, frame, teacher)

    union = np.concatenate([student_grasps, teacher], axis=0)
    union_friction = np.concatenate(
        [
            np.asarray(student_result.friction, dtype=np.float32),
            np.asarray(teacher_result.friction, dtype=np.float32),
        ],
        axis=0,
    )
    union_source = np.concatenate(
        [
            np.zeros(student_grasps.shape[0], dtype=np.int8),
            np.ones(teacher.shape[0], dtype=np.int8),
        ]
    )
    union_original_scores = union[:, 0].copy()
    union, union_scores = apply_oracle_scores(
        union,
        union_friction,
        original_scores=union_original_scores,
        tie_break_eps=float(args.tie_break_eps),
    )
    order = np.argsort(-union[:, 0], kind="stable")
    top50 = order[: min(50, order.size)]
    top10 = order[: min(10, order.size)]
    metadata = {
        "teacher_source_index": teacher_source_index.astype(np.int32),
        "teacher_friction": np.asarray(teacher_result.friction, dtype=np.float32),
        "teacher_collision_or_empty": np.asarray(
            teacher_result.collision_or_empty, dtype=np.uint8
        ),
        "teacher_assigned_obj": np.asarray(
            teacher_result.assigned_obj, dtype=np.int16
        ),
        "union_source": union_source,
        "union_friction": union_friction.astype(np.float32),
        "union_oracle_utility": union_scores.utility.astype(np.float32),
    }
    summary = {
        "num_teacher_actions": int(teacher.shape[0]),
        "teacher_eval_sec": float(teacher_eval_sec),
        "proposal_union_num_actions": int(union.shape[0]),
        "proposal_union_teacher_ratio_top10": (
            float(union_source[top10].mean()) if top10.size else float("nan")
        ),
        "proposal_union_teacher_ratio_top50": (
            float(union_source[top50].mean()) if top50.size else float("nan")
        ),
        "proposal_union_top10_safe04": topk_metric(
            union_friction, union[:, 0], top_k=10, threshold=0.4
        ),
        "proposal_union_top10_safe08": topk_metric(
            union_friction, union[:, 0], top_k=10, threshold=0.8
        ),
        **{
            f"teacher_{key}": value
            for key, value in summarize_exact_labels(
                teacher_result.friction, teacher_result.collision_or_empty
            ).items()
        },
    }
    return union, metadata, summary


def _write_worker_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in sorted(
            rows, key=lambda item: (int(item["scene_id"]), int(item["anno_id"]))
        ):
            writer.writerow({key: row.get(key, "") for key in keys})
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    modes = _parse_modes(args.modes)
    if "proposal_union" in modes and not str(args.teacher_dump).strip():
        raise ValueError("--teacher_dump is required when proposal_union is enabled.")
    if int(args.world_size) <= 0 or not 0 <= int(args.worker_rank) < int(args.world_size):
        raise ValueError(
            f"Invalid worker_rank/world_size: {args.worker_rank}/{args.world_size}."
        )
    if int(args.progress_every) <= 0:
        raise ValueError("--progress_every must be positive.")
    if int(args.collision_chunk) <= 0:
        raise ValueError("--collision_chunk must be positive.")
    if float(args.min_width_m) > float(args.max_width_m):
        raise ValueError("min_width_m cannot exceed max_width_m.")
    if float(args.min_depth_m) > float(args.max_depth_m):
        raise ValueError("min_depth_m cannot exceed max_depth_m.")

    student_requested_root = Path(args.student_dump).expanduser().resolve()
    teacher_requested_root = (
        Path(args.teacher_dump).expanduser().resolve()
        if str(args.teacher_dump).strip()
        else None
    )
    student_root, student_layout = resolve_prediction_root(
        student_requested_root,
        split=args.split,
        camera=args.camera,
        role="student",
    )
    teacher_root = None
    teacher_layout = ""
    if teacher_requested_root is not None:
        teacher_root, teacher_layout = resolve_prediction_root(
            teacher_requested_root,
            split=args.split,
            camera=args.camera,
            role="teacher",
        )
    student_collision_check = verify_fixed_input_collision_policy(student_root)
    teacher_collision_check = (
        verify_fixed_input_collision_policy(teacher_root)
        if teacher_root is not None
        else {
            "status": "not_applicable",
            "meta_dir": "",
            "worker_summaries": 0,
            "collision_thresh": FIXED_INPUT_COLLISION_THRESH,
        }
    )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    meta_root = output_root / "_p0e_meta"
    meta_root.mkdir(parents=True, exist_ok=True)

    scene_ids = parse_int_ranges(args.scene_ids)
    anno_ids = parse_int_ranges(args.anno_ids)
    all_frames = build_frame_list(
        split=args.split,
        camera=args.camera,
        sample_interval=int(args.sample_interval),
        scene_ids=scene_ids,
        anno_ids=anno_ids,
    )
    frames, effective_shard_mode = shard_frames(
        all_frames,
        rank=int(args.worker_rank),
        world_size=int(args.world_size),
        mode=str(args.shard_mode),
    )
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    variants = _required_variants(modes)
    _guard_existing_outputs(
        output_root=output_root,
        frames=frames,
        variants=variants,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )

    perturbations = build_local_perturbations(
        translation_mm=float(args.local_translation_mm),
        inplane_deg=float(args.local_inplane_deg),
        depth_delta_m=float(args.local_depth_delta_m),
        width_delta_m=float(args.local_width_delta_m),
        view_tilt_deg=float(args.local_view_tilt_deg),
    )
    contract = {
        "protocol": PROTOCOL_VERSION,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "student_dump_requested": str(student_requested_root),
        "student_dump_resolved": str(student_root),
        "student_dump_layout": student_layout,
        "teacher_dump_requested": (
            str(teacher_requested_root) if teacher_requested_root is not None else ""
        ),
        "teacher_dump_resolved": str(teacher_root) if teacher_root is not None else "",
        "teacher_dump_layout": teacher_layout,
        "fixed_input_collision_thresh": FIXED_INPUT_COLLISION_THRESH,
        "input_collision_policy": (
            "Both student and teacher dumps are assumed to be generated with the "
            "standard model-free collision filter at collision_thresh=0.01. P0-E "
            "does not re-run that captured-point-cloud filter; it applies the exact "
            "clean-CAD/DexNet evaluator to the already-postprocessed actions."
        ),
        "student_collision_policy_check": student_collision_check,
        "teacher_collision_policy_check": teacher_collision_check,
        "output_root": str(output_root),
        "camera": args.camera,
        "split": args.split,
        "modes": list(modes),
        "variants": list(variants),
        "sample_interval": int(args.sample_interval),
        "scene_ids": list(scene_ids),
        "anno_ids": list(anno_ids),
        "worker_rank": int(args.worker_rank),
        "world_size": int(args.world_size),
        "shard_mode_requested": str(args.shard_mode),
        "shard_mode_effective": effective_shard_mode,
        "num_global_frames": len(all_frames),
        "num_worker_frames": len(frames),
        "student_top_n": int(args.student_top_n),
        "teacher_top_n": int(args.teacher_top_n),
        "friction_thresholds": FRICTION_THRESHOLDS,
        "tie_break_eps": float(args.tie_break_eps),
        "local_top_n_per_object": int(args.local_top_n_per_object),
        "local_global_top_n": int(args.local_global_top_n),
        "local_perturbations": list(perturbations),
        "fc_mode": args.fc_mode,
        "fc_verify_n": int(args.fc_verify_n),
        "strict_evaluator": bool(args.strict_evaluator),
        "semantic_contract": {
            "student_original": "Autonomous student actions and original scores.",
            "exact_action_rerank": (
                "Identical student actions with exact clean-geometry CAD/DexNet scores."
            ),
            "local_field_oracle": (
                "Best action from an identity-first local gripper-frame lattice per "
                "selected student proposal, followed by exact oracle ranking."
            ),
            "proposal_union_oracle": (
                "Autonomous student plus autonomous clean-depth teacher proposals, "
                "followed by exact oracle ranking."
            ),
        },
    }
    atomic_save_json(
        meta_root / f"contract_{args.split}_worker_{int(args.worker_rank):02d}.json",
        contract,
    )

    evaluator = ExactGraspNetActionEvaluator(
        args.dataset_root,
        args.camera,
        split=args.split,
        collision_chunk=int(args.collision_chunk),
        fc_mode=args.fc_mode,
        verify_n=int(args.fc_verify_n),
        strict=bool(args.strict_evaluator),
    )

    summaries: List[Dict[str, object]] = []
    skipped_missing = 0
    skipped_complete = 0
    start = time.perf_counter()
    print(
        f"[P0-E][worker {args.worker_rank}/{args.world_size}] "
        f"split={args.split} frames={len(frames)} modes={','.join(modes)} "
        f"shard={effective_shard_mode} local_lattice={len(perturbations)} collision_thresh="
        f"{FIXED_INPUT_COLLISION_THRESH:g}",
        flush=True,
    )
    print(
        f"[P0-E][INPUT] student={student_root} ({student_layout}, "
        f"collision={student_collision_check['status']}) "
        f"teacher={teacher_root if teacher_root is not None else '-'} "
        f"({teacher_layout or '-'}, collision={teacher_collision_check['status']})",
        flush=True,
    )

    for frame_index, frame in enumerate(frames):
        if bool(args.resume) and _frame_outputs_complete(output_root, frame, variants):
            skipped_complete += 1
            continue

        student_path = grasp_path(student_root, frame)
        teacher_path = (
            grasp_path(teacher_root, frame) if teacher_root is not None else None
        )
        missing = []
        if not student_path.is_file():
            missing.append(str(student_path))
        if "proposal_union" in modes and (
            teacher_path is None or not teacher_path.is_file()
        ):
            missing.append(str(teacher_path))
        if missing:
            if bool(args.skip_missing):
                skipped_missing += 1
                print(f"[P0-E][SKIP] missing {missing}", flush=True)
                continue
            raise FileNotFoundError(f"Missing P0-E input files: {missing}")

        frame_started = time.perf_counter()
        student_full = load_grasps(student_path)
        student_grasps, student_source_index = top_n_by_score(
            student_full, int(args.student_top_n)
        )
        student_result, student_eval_sec = _evaluate(evaluator, frame, student_grasps)
        summary = _frame_summary_base(
            frame=frame,
            student_grasps=student_grasps,
            student_result=student_result,
            student_eval_sec=student_eval_sec,
        )
        frame_metadata: Dict[str, np.ndarray] = {
            "protocol": np.asarray(PROTOCOL_VERSION),
            "scene_id": np.asarray([frame.scene_id], dtype=np.int16),
            "anno_id": np.asarray([frame.anno_id], dtype=np.int16),
            "student_source_index": student_source_index.astype(np.int32),
            "student_original_score": student_grasps[:, 0].astype(np.float32),
            **{
                f"student_{key}": value
                for key, value in _eval_payload(student_result).items()
            },
        }
        _save_student_original(
            output_root=output_root,
            frame=frame,
            source_path=student_path,
            student_grasps=student_grasps,
            student_top_n=int(args.student_top_n),
        )

        if "rerank" in modes:
            reranked, rerank_scores = apply_oracle_scores(
                student_grasps,
                student_result.friction,
                original_scores=student_grasps[:, 0],
                tie_break_eps=float(args.tie_break_eps),
            )
            _save_variant(output_root, MODE_TO_VARIANT["rerank"], frame, reranked)
            frame_metadata["exact_action_oracle_utility"] = rerank_scores.utility
            frame_metadata["exact_action_oracle_score"] = rerank_scores.score
            summary.update(
                {
                    "rerank_top1_safe04": topk_metric(
                        student_result.friction,
                        reranked[:, 0],
                        top_k=1,
                        threshold=0.4,
                    ),
                    "rerank_top10_safe04": topk_metric(
                        student_result.friction,
                        reranked[:, 0],
                        top_k=10,
                        threshold=0.4,
                    ),
                    "rerank_top10_safe08": topk_metric(
                        student_result.friction,
                        reranked[:, 0],
                        top_k=10,
                        threshold=0.8,
                    ),
                }
            )

        if "local_field" in modes:
            local_output, local_metadata, local_summary = _run_local_field(
                evaluator=evaluator,
                frame=frame,
                student_grasps=student_grasps,
                student_result=student_result,
                perturbations=perturbations,
                args=args,
            )
            _save_variant(
                output_root,
                MODE_TO_VARIANT["local_field"],
                frame,
                local_output,
            )
            frame_metadata.update(local_metadata)
            summary.update(local_summary)

        if "proposal_union" in modes:
            assert teacher_path is not None
            union_output, union_metadata, union_summary = _run_proposal_union(
                evaluator=evaluator,
                frame=frame,
                student_grasps=student_grasps,
                student_result=student_result,
                teacher_path=teacher_path,
                args=args,
            )
            _save_variant(
                output_root,
                MODE_TO_VARIANT["proposal_union"],
                frame,
                union_output,
            )
            frame_metadata.update(union_metadata)
            summary.update(union_summary)

        summary["frame_sec"] = float(time.perf_counter() - frame_started)
        summary["student_input_path"] = str(student_path)
        summary["teacher_input_path"] = (
            str(teacher_path) if teacher_path is not None else ""
        )
        summaries.append(summary)
        atomic_save_npz(
            meta_path(output_root, frame),
            frame_metadata,
            compress=bool(args.compress_meta),
        )

        if frame_index % int(args.progress_every) == 0:
            elapsed = time.perf_counter() - start
            processed = len(summaries)
            print(
                f"[P0-E][worker {args.worker_rank}] frame={frame.key} "
                f"processed={processed}/{len(frames)} student={student_grasps.shape[0]} "
                f"sec_per_frame={elapsed / max(processed, 1):.2f}",
                flush=True,
            )

    summary_path = (
        meta_root / f"frame_summary_{args.split}_worker_{int(args.worker_rank):02d}.csv"
    )
    _write_worker_summary(summary_path, summaries)
    final = {
        **contract,
        "processed_frames": len(summaries),
        "skipped_complete": skipped_complete,
        "skipped_missing": skipped_missing,
        "elapsed_sec": float(time.perf_counter() - start),
        "summary_csv": str(summary_path),
    }
    atomic_save_json(
        meta_root / f"done_{args.split}_worker_{int(args.worker_rank):02d}.json",
        final,
    )
    print(
        f"[P0-E][DONE] worker={args.worker_rank} processed={len(summaries)} "
        f"skip_complete={skipped_complete} skip_missing={skipped_missing} "
        f"elapsed={(time.perf_counter() - start) / 60.0:.2f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
