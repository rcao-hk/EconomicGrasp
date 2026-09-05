#!/usr/bin/env python3
"""Shared utilities for P0-E privileged grasp-field oracle experiments.

P0-E evaluates three independent privileged upper bounds over physical 6-DoF
parallel-jaw actions:

1. exact-action reranking keeps student actions fixed and replaces only their
   scores with clean-geometry evaluator utility;
2. local-field refinement evaluates a small gripper-frame action lattice around
   selected student actions and retains the best action per proposal;
3. proposal union combines autonomous student and clean-depth teacher actions
   and ranks the union with the same clean-geometry utility.

All grasp arrays follow the GraspNet ``GraspGroup`` layout ``[N, 17]``:
``score, width, height, depth, rotation(9), translation(3), object_id``.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROTOCOL_VERSION = "p0e_privileged_grasp_field_oracles_v1_2"
FIXED_INPUT_COLLISION_THRESH = 0.01
FRICTION_THRESHOLDS = np.asarray(
    [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32
)
SUPPORTED_SPLITS = ("test_seen", "test_similar", "test_novel")
DEFAULT_VARIANTS = (
    "student_original",
    "exact_action_rerank",
    "local_field_oracle",
    "proposal_union_oracle",
)


@dataclass(frozen=True)
class FrameSpec:
    scene_id: int
    anno_id: int
    camera: str

    @property
    def scene_name(self) -> str:
        return f"scene_{self.scene_id:04d}"

    @property
    def key(self) -> str:
        return f"{self.scene_name}/{self.camera}/{self.anno_id:04d}"


@dataclass(frozen=True)
class LocalPerturbation:
    """One local action offset expressed in the gripper frame."""

    name: str
    translation_local_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_local_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_delta_m: float = 0.0
    width_delta_m: float = 0.0

    @property
    def is_identity(self) -> bool:
        values = (
            *self.translation_local_m,
            *self.rotation_local_rad,
            self.depth_delta_m,
            self.width_delta_m,
        )
        return all(abs(float(value)) < 1.0e-12 for value in values)


@dataclass(frozen=True)
class OracleScores:
    utility: np.ndarray
    score: np.ndarray


def scene_ids_for_split(split: str) -> range:
    if split == "test_seen":
        return range(100, 130)
    if split == "test_similar":
        return range(130, 160)
    if split == "test_novel":
        return range(160, 190)
    raise ValueError(f"Unsupported split {split!r}; expected one of {SUPPORTED_SPLITS}.")


def parse_int_ranges(text: str) -> Tuple[int, ...]:
    """Parse ``1,3-5`` into a sorted unique tuple."""

    values: set[int] = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending integer range {token!r} is unsupported.")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return tuple(sorted(values))


def build_frame_list(
    *,
    split: str,
    camera: str,
    sample_interval: int,
    scene_ids: Sequence[int] = (),
    anno_ids: Sequence[int] = (),
) -> List[FrameSpec]:
    if int(sample_interval) <= 0:
        raise ValueError("sample_interval must be positive.")
    split_scenes = set(scene_ids_for_split(split))
    if scene_ids:
        requested = set(int(value) for value in scene_ids)
        invalid = sorted(requested - split_scenes)
        if invalid:
            raise ValueError(f"scene_ids {invalid} do not belong to split {split!r}.")
        selected_scenes = sorted(requested)
    else:
        selected_scenes = sorted(split_scenes)

    if anno_ids:
        selected_annos = sorted(set(int(value) for value in anno_ids))
        invalid_annos = [value for value in selected_annos if not 0 <= value < 256]
        if invalid_annos:
            raise ValueError(f"Annotation IDs must be in [0,255], got {invalid_annos}.")
    else:
        selected_annos = list(range(0, 256, int(sample_interval)))

    return [
        FrameSpec(scene_id=scene_id, anno_id=anno_id, camera=str(camera))
        for scene_id in selected_scenes
        for anno_id in selected_annos
    ]


def shard_frames(
    frames: Sequence[FrameSpec],
    *,
    rank: int,
    world_size: int,
    mode: str = "auto",
) -> Tuple[List[FrameSpec], str]:
    """Shard frames while preserving evaluator scene-cache locality.

    ``frame`` reproduces the old strided frame sharding. ``scene`` assigns each
    scene to one worker so CAD meshes are loaded/cached by only one process.
    ``auto`` uses scene sharding when there are at least ``world_size`` scenes,
    and falls back to frame sharding for narrow smoke tests.
    """
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}.")
    mode = str(mode).strip().lower()
    if mode not in {"auto", "scene", "frame"}:
        raise ValueError(f"Unknown shard mode {mode!r}; expected auto, scene, or frame.")

    ordered = list(frames)
    scenes = sorted({int(frame.scene_id) for frame in ordered})
    effective = mode
    if mode == "auto":
        effective = "scene" if len(scenes) >= world_size else "frame"

    if effective == "frame":
        return ordered[rank::world_size], effective

    scene_slot = {scene_id: index % world_size for index, scene_id in enumerate(scenes)}
    return [
        frame for frame in ordered
        if scene_slot[int(frame.scene_id)] == rank
    ], effective


def deduplicate_physical_actions(
    grasps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate evaluator-equivalent actions and return an inverse map.

    The exact evaluator ignores the network score (column 0) and input object ID
    (column 16).  The physical signature therefore consists of width, height,
    grasp depth, rotation, and translation (columns 1:16).  Labels evaluated on
    the returned unique actions can be expanded with ``labels[inverse]``.
    """
    grasps = validate_grasps(grasps, context="deduplicate_physical_actions")
    if grasps.shape[0] == 0:
        return grasps.copy(), np.zeros(0, dtype=np.int64)
    signature = np.ascontiguousarray(grasps[:, 1:16], dtype=np.float32)
    _unique_signature, first_index, inverse = np.unique(
        signature, axis=0, return_index=True, return_inverse=True
    )
    return grasps[first_index].copy(), inverse.astype(np.int64, copy=False)


def _looks_like_prediction_root(
    root: os.PathLike[str] | str,
    *,
    split: str,
    camera: str,
) -> bool:
    """Return True when ``root`` directly contains scene prediction folders.

    We intentionally check directories instead of a particular annotation file so
    that sparse/sample-interval dumps resolve correctly.
    """
    root = Path(root)
    if not root.is_dir():
        return False
    for scene_id in scene_ids_for_split(split):
        if (root / f"scene_{scene_id:04d}" / str(camera)).is_dir():
            return True
    return False


def resolve_prediction_root(
    requested_root: os.PathLike[str] | str,
    *,
    split: str,
    camera: str,
    role: str,
) -> Tuple[Path, str]:
    """Resolve flat, split-nested, and P0-B student prediction layouts.

    Supported student inputs:

    1. ``root/scene_XXXX/<camera>/*.npy`` (flat autonomous dump);
    2. ``root/<split>/scene_XXXX/<camera>/*.npy``;
    3. ``root/<split>/student/scene_XXXX/<camera>/*.npy`` (P0-B root);
    4. ``root/student/scene_XXXX/<camera>/*.npy`` (P0-B split root);
    5. any of the above resolved subdirectories passed directly.

    Teacher inputs intentionally do *not* auto-resolve P0-B ``teacher_full`` or
    ``teacher_common`` because those variants are forced onto student proposals
    and cannot measure proposal recall.
    """
    role = str(role).strip().lower()
    if role not in {"student", "teacher"}:
        raise ValueError(f"role must be 'student' or 'teacher', got {role!r}.")

    requested = Path(requested_root).expanduser().resolve()
    if not requested.exists():
        raise FileNotFoundError(f"P0-E {role} dump root does not exist: {requested}")

    if role == "teacher":
        forbidden = {"teacher_full", "teacher_common", "oracle_hybrid"}
        bad = [part for part in requested.parts if part in forbidden]
        if bad:
            raise ValueError(
                "P0-E proposal union requires an autonomous Stage-0 teacher dump; "
                f"P0-B variant path {requested} is invalid (matched {bad[-1]!r})."
            )

    candidates: List[Tuple[Path, str]] = [
        (requested, "direct"),
        (requested / split, "split_nested"),
    ]
    if role == "student":
        candidates.extend(
            [
                (requested / split / "student", "p0b_root_student"),
                (requested / "student", "p0b_split_student"),
            ]
        )
    else:
        # Optional neutral wrapper name for autonomous teacher dumps.  We do not
        # accept P0-B teacher_full/common automatically.
        candidates.extend(
            [
                (requested / split / "teacher", "split_teacher"),
                (requested / "teacher", "teacher_wrapper"),
            ]
        )

    seen: set[str] = set()
    checked: List[str] = []
    for candidate, layout in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(f"{layout}: {candidate}")
        if _looks_like_prediction_root(candidate, split=split, camera=camera):
            return candidate.resolve(), layout

    hint = (
        " For a P0-B student root, pass the experiment root and P0-E will "
        "resolve <root>/<split>/student automatically."
        if role == "student"
        else " The teacher must be an autonomous Stage-0 dump, not P0-B teacher_full/common."
    )
    raise FileNotFoundError(
        f"Could not resolve P0-E {role} predictions for split={split!r}, "
        f"camera={camera!r} from {requested}. Checked: {checked}.{hint}"
    )


def verify_fixed_input_collision_policy(
    resolved_root: os.PathLike[str] | str,
    *,
    expected: float = FIXED_INPUT_COLLISION_THRESH,
) -> Dict[str, object]:
    """Verify P0-B collision metadata when it is available.

    Autonomous Stage-0/1/2 dumps do not currently carry a sidecar contract, so
    their collision policy cannot be inferred from `[N,17]` arrays alone.  P0-B
    dumps do have worker summaries; when a nearby `_p0b_meta` directory exists,
    this function strictly checks that every worker used the fixed threshold.
    """
    resolved = Path(resolved_root).resolve()
    candidate_meta_dirs: List[Path] = []
    for parent in (resolved, resolved.parent, resolved.parent.parent):
        meta = parent / "_p0b_meta"
        if meta not in candidate_meta_dirs:
            candidate_meta_dirs.append(meta)

    for meta_dir in candidate_meta_dirs:
        if not meta_dir.is_dir():
            continue
        summaries = sorted(meta_dir.glob("worker_*_summary.json"))
        if not summaries:
            continue
        values: List[float] = []
        for path in summaries:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "collision_thresh" not in payload:
                raise RuntimeError(
                    f"P0-B metadata {path} has no collision_thresh field."
                )
            values.append(float(payload["collision_thresh"]))
        bad = [value for value in values if abs(value - float(expected)) > 1e-9]
        if bad:
            raise RuntimeError(
                "P0-E fixed collision protocol requires collision_thresh="
                f"{float(expected):g}, but P0-B metadata under {meta_dir} contains "
                f"{sorted(set(bad))}."
            )
        return {
            "status": "verified_from_p0b_metadata",
            "meta_dir": str(meta_dir.resolve()),
            "worker_summaries": len(summaries),
            "collision_thresh": float(expected),
        }

    return {
        "status": "unverified_no_sidecar_metadata",
        "meta_dir": "",
        "worker_summaries": 0,
        "collision_thresh": float(expected),
    }


def grasp_path(root: os.PathLike[str] | str, frame: FrameSpec) -> Path:
    return Path(root) / frame.scene_name / frame.camera / f"{frame.anno_id:04d}.npy"


def meta_path(root: os.PathLike[str] | str, frame: FrameSpec) -> Path:
    return (
        Path(root)
        / "_p0e_meta"
        / "frame_cache"
        / frame.scene_name
        / frame.camera
        / f"{frame.anno_id:04d}.npz"
    )


def validate_grasps(array: np.ndarray, *, context: str) -> np.ndarray:
    grasps = np.asarray(array, dtype=np.float32)
    if grasps.ndim != 2 or grasps.shape[1] != 17:
        raise ValueError(f"{context}: expected [N,17], got {grasps.shape}.")
    if not np.isfinite(grasps).all():
        bad = int((~np.isfinite(grasps)).sum())
        raise ValueError(f"{context}: found {bad} non-finite values.")
    return grasps


def load_grasps(path: os.PathLike[str] | str) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return validate_grasps(np.load(path), context=str(path))


def top_n_by_score(grasps: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
    grasps = validate_grasps(grasps, context="top_n_by_score")
    n = int(grasps.shape[0])
    if int(top_n) <= 0 or int(top_n) >= n:
        indices = np.arange(n, dtype=np.int64)
        return grasps.copy(), indices
    order = np.argsort(-grasps[:, 0], kind="stable")[: int(top_n)]
    return grasps[order].copy(), order.astype(np.int64)


def friction_to_utility(
    friction: np.ndarray,
    thresholds: np.ndarray = FRICTION_THRESHOLDS,
) -> np.ndarray:
    """Map official friction labels to mean success over CDF thresholds.

    Collision, empty, and force-closure failure labels (``<=0``) receive zero.
    For a valid grasp with minimum successful friction ``mu``, utility is the
    fraction of thresholds greater than or equal to ``mu``. Sorting by this
    scalar is the exact per-candidate ordering for mean AP over the same
    thresholds.
    """

    friction = np.asarray(friction, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if thresholds.size == 0 or not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be a non-empty strictly increasing array.")
    valid = friction[..., None] > 0.0
    success = valid & (friction[..., None] <= thresholds)
    return success.mean(axis=-1, dtype=np.float32)


def _normalized_tie_break(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return scores.copy()
    finite = np.isfinite(scores)
    result = np.zeros_like(scores, dtype=np.float32)
    if not finite.any():
        return result
    values = scores[finite]
    lo, hi = float(values.min()), float(values.max())
    result[finite] = (values - lo) / (hi - lo) if hi > lo else 0.5
    return result


def oracle_scores(
    friction: np.ndarray,
    original_scores: np.ndarray,
    *,
    tie_break_eps: float = 1.0e-4,
) -> OracleScores:
    utility = friction_to_utility(friction).astype(np.float32, copy=False)
    original_scores = np.asarray(original_scores, dtype=np.float32).reshape(-1)
    if utility.shape != original_scores.shape:
        raise ValueError(
            f"friction/original score shape mismatch: {utility.shape} vs "
            f"{original_scores.shape}."
        )
    epsilon = float(tie_break_eps)
    if epsilon < 0.0 or epsilon >= 1.0 / max(len(FRICTION_THRESHOLDS), 1):
        raise ValueError(
            "tie_break_eps must be non-negative and smaller than one utility bin."
        )
    score = utility + epsilon * _normalized_tie_break(original_scores)
    return OracleScores(utility=utility, score=score.astype(np.float32))


def apply_oracle_scores(
    grasps: np.ndarray,
    friction: np.ndarray,
    *,
    original_scores: np.ndarray | None = None,
    tie_break_eps: float = 1.0e-4,
) -> Tuple[np.ndarray, OracleScores]:
    grasps = validate_grasps(grasps, context="apply_oracle_scores")
    if original_scores is None:
        original_scores = grasps[:, 0]
    scores = oracle_scores(
        friction,
        np.asarray(original_scores, dtype=np.float32),
        tie_break_eps=tie_break_eps,
    )
    result = grasps.copy()
    result[:, 0] = scores.score
    return result, scores


def _axis_rotation(axis: int, angle_rad: float) -> np.ndarray:
    c, s = math.cos(float(angle_rad)), math.sin(float(angle_rad))
    if axis == 0:
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32
        )
    if axis == 1:
        return np.asarray(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32
        )
    if axis == 2:
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
    raise ValueError(f"axis must be 0, 1, or 2, got {axis}.")


def local_rotation_matrix(rotation_local_rad: Sequence[float]) -> np.ndarray:
    angles = tuple(float(value) for value in rotation_local_rad)
    if len(angles) != 3:
        raise ValueError("rotation_local_rad must contain three values.")
    # Intrinsic XYZ offsets in the gripper coordinate system. GraspNet uses
    # local X as the approach axis, so rotation around X is in-plane rotation.
    return (
        _axis_rotation(0, angles[0])
        @ _axis_rotation(1, angles[1])
        @ _axis_rotation(2, angles[2])
    )


def apply_local_perturbation(
    grasps: np.ndarray,
    perturbation: LocalPerturbation,
    *,
    min_width_m: float = 0.0,
    max_width_m: float = 0.10,
    min_depth_m: float = 0.0,
    max_depth_m: float = 0.10,
) -> np.ndarray:
    result = validate_grasps(grasps, context="apply_local_perturbation").copy()
    if result.shape[0] == 0:
        return result

    rotations = result[:, 4:13].reshape(-1, 3, 3)
    translations = result[:, 13:16]
    local_delta = np.asarray(perturbation.translation_local_m, dtype=np.float32)
    if local_delta.shape != (3,):
        raise ValueError(
            f"translation_local_m must have shape (3,), got {local_delta.shape}."
        )
    translations += np.einsum("nij,j->ni", rotations, local_delta)

    delta_rotation = local_rotation_matrix(perturbation.rotation_local_rad)
    rotations = np.einsum("nij,jk->nik", rotations, delta_rotation)
    result[:, 4:13] = rotations.reshape(-1, 9)
    result[:, 13:16] = translations
    result[:, 1] = np.clip(
        result[:, 1] + float(perturbation.width_delta_m),
        float(min_width_m),
        float(max_width_m),
    )
    result[:, 3] = np.clip(
        result[:, 3] + float(perturbation.depth_delta_m),
        float(min_depth_m),
        float(max_depth_m),
    )
    return result


def build_local_perturbations(
    *,
    translation_mm: float = 5.0,
    inplane_deg: float = 15.0,
    depth_delta_m: float = 0.01,
    width_delta_m: float = 0.005,
    view_tilt_deg: float = 0.0,
) -> Tuple[LocalPerturbation, ...]:
    """Build an identity-first local action lattice."""

    translation = float(translation_mm) / 1000.0
    inplane = math.radians(float(inplane_deg))
    tilt = math.radians(float(view_tilt_deg))
    perturbations: List[LocalPerturbation] = [LocalPerturbation("identity")]

    if translation > 0.0:
        for axis, name in enumerate(("x", "y", "z")):
            for sign, suffix in ((1.0, "pos"), (-1.0, "neg")):
                delta = [0.0, 0.0, 0.0]
                delta[axis] = sign * translation
                perturbations.append(
                    LocalPerturbation(
                        f"translate_{name}_{suffix}",
                        translation_local_m=tuple(delta),
                    )
                )
    if inplane > 0.0:
        perturbations.extend(
            [
                LocalPerturbation(
                    "inplane_pos", rotation_local_rad=(inplane, 0.0, 0.0)
                ),
                LocalPerturbation(
                    "inplane_neg", rotation_local_rad=(-inplane, 0.0, 0.0)
                ),
            ]
        )
    if depth_delta_m > 0.0:
        perturbations.extend(
            [
                LocalPerturbation("depth_pos", depth_delta_m=float(depth_delta_m)),
                LocalPerturbation("depth_neg", depth_delta_m=-float(depth_delta_m)),
            ]
        )
    if width_delta_m > 0.0:
        perturbations.extend(
            [
                LocalPerturbation("width_pos", width_delta_m=float(width_delta_m)),
                LocalPerturbation("width_neg", width_delta_m=-float(width_delta_m)),
            ]
        )
    if tilt > 0.0:
        for axis, name in ((1, "view_y"), (2, "view_z")):
            for sign, suffix in ((1.0, "pos"), (-1.0, "neg")):
                angles = [0.0, 0.0, 0.0]
                angles[axis] = sign * tilt
                perturbations.append(
                    LocalPerturbation(
                        f"{name}_{suffix}", rotation_local_rad=tuple(angles)
                    )
                )

    names = [item.name for item in perturbations]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate perturbation names: {names}.")
    return tuple(perturbations)


def select_local_base_indices(
    grasps: np.ndarray,
    assigned_obj: np.ndarray,
    *,
    top_n_per_object: int,
    global_top_n: int,
) -> np.ndarray:
    """Select local-search actions without starving low-score objects."""

    grasps = validate_grasps(grasps, context="select_local_base_indices")
    assigned = np.asarray(assigned_obj, dtype=np.int64).reshape(-1)
    if assigned.shape[0] != grasps.shape[0]:
        raise ValueError("assigned_obj length does not match grasps.")
    if grasps.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    per_object = int(top_n_per_object)
    if per_object > 0:
        selected: List[int] = []
        for object_id in sorted(np.unique(assigned).tolist()):
            ids = np.flatnonzero(assigned == int(object_id))
            order = ids[np.argsort(-grasps[ids, 0], kind="stable")]
            selected.extend(order[:per_object].tolist())
        indices = np.asarray(sorted(set(selected)), dtype=np.int64)
    else:
        indices = np.arange(grasps.shape[0], dtype=np.int64)

    indices = indices[np.argsort(-grasps[indices, 0], kind="stable")]
    if int(global_top_n) > 0:
        indices = indices[: int(global_top_n)]
    return indices.astype(np.int64, copy=False)


def stack_local_lattice(
    base_grasps: np.ndarray,
    perturbations: Sequence[LocalPerturbation],
    *,
    min_width_m: float,
    max_width_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Return a local action lattice with shape ``[N,P,17]``."""

    base = validate_grasps(base_grasps, context="stack_local_lattice")
    if not perturbations:
        raise ValueError("At least one local perturbation is required.")
    if not perturbations[0].is_identity:
        raise ValueError("The first perturbation must be identity for stable ties.")
    return np.stack(
        [
            apply_local_perturbation(
                base,
                perturbation,
                min_width_m=min_width_m,
                max_width_m=max_width_m,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
            for perturbation in perturbations
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def pick_best_local_actions(
    lattice: np.ndarray,
    friction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the best perturbation per base action; identity wins ties."""

    lattice = np.asarray(lattice, dtype=np.float32)
    friction = np.asarray(friction, dtype=np.float32)
    if lattice.ndim != 3 or lattice.shape[-1] != 17:
        raise ValueError(f"lattice must be [N,P,17], got {lattice.shape}.")
    if friction.shape != lattice.shape[:2]:
        raise ValueError(
            f"friction shape {friction.shape} does not match lattice {lattice.shape[:2]}."
        )
    utility = friction_to_utility(friction)
    best_id = np.argmax(utility, axis=1).astype(np.int16)
    rows = np.arange(lattice.shape[0], dtype=np.int64)
    return (
        lattice[rows, best_id].copy(),
        friction[rows, best_id].copy(),
        utility[rows, best_id].copy(),
        best_id,
    )


def atomic_save_npy(path: os.PathLike[str] | str, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as file:
        np.save(file, np.asarray(array, dtype=np.float32))
    os.replace(temporary, path)


def atomic_save_npz(
    path: os.PathLike[str] | str,
    arrays: Mapping[str, np.ndarray],
    *,
    compress: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    saver = np.savez_compressed if bool(compress) else np.savez
    with temporary.open("wb") as file:
        saver(file, **arrays)
    os.replace(temporary, path)


def atomic_save_json(path: os.PathLike[str] | str, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")

    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, LocalPerturbation):
            return asdict(value)
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(convert(dict(payload)), file, indent=2, sort_keys=True)
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def topk_metric(
    friction: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    threshold: float,
) -> float:
    friction = np.asarray(friction, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if friction.shape != scores.shape:
        raise ValueError("friction and scores must have the same shape.")
    if friction.size == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")[: min(int(top_k), friction.size)]
    values = friction[order]
    return float(((values > 0.0) & (values <= float(threshold))).mean())


def summarize_exact_labels(
    friction: np.ndarray,
    collision_or_empty: np.ndarray,
) -> Dict[str, float]:
    friction = np.asarray(friction, dtype=np.float32).reshape(-1)
    collision = np.asarray(collision_or_empty, dtype=bool).reshape(-1)
    if friction.shape != collision.shape:
        raise ValueError("friction and collision_or_empty must have the same shape.")
    if friction.size == 0:
        return {
            "valid_ratio": float("nan"),
            "safe04_ratio": float("nan"),
            "safe08_ratio": float("nan"),
            "collision_or_empty_ratio": float("nan"),
            "utility_mean": float("nan"),
        }
    return {
        "valid_ratio": float((friction > 0.0).mean()),
        "safe04_ratio": float(((friction > 0.0) & (friction <= 0.4)).mean()),
        "safe08_ratio": float(((friction > 0.0) & (friction <= 0.8)).mean()),
        "collision_or_empty_ratio": float(collision.mean()),
        "utility_mean": float(friction_to_utility(friction).mean()),
    }
