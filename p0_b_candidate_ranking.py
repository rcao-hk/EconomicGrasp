#!/usr/bin/env python3
"""P0-B: decompose PKD gains into candidate-set quality and ranking quality.

The script evaluates every available candidate with the same GraspNet CAD /
DexNet machinery used by the existing exact-action cache. It reports:

* actual score-ranked AP proxy and official-style Top-1..50 precision average;
* oracle re-ranked AP on the same candidate set;
* good-candidate availability and object coverage;
* Top-1/10/50 collision, empty, and force-closure failure rates;
* optional matched-candidate score-swap analysis between two models.

Use pre-postprocessing sidecars (``*.p0_candidates.npz``) when available. Plain
``.npy`` inference dumps are also supported, but then the analysis describes
only post-processing survivors.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from pkd_p0.common import (
    CDF_THRESHOLDS,
    atomic_json_dump,
    atomic_npz_dump,
    filtered_kwargs,
    parse_scene_anno_from_path,
    rotation_geodesic_deg,
    sha256_file,
)


@dataclass
class CandidateEvaluation:
    assigned_obj: np.ndarray
    collision_or_empty: np.ndarray
    pure_collision: np.ndarray
    empty: np.ndarray
    friction: np.ndarray
    stats: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", required=True, choices=("test_seen", "test_similar", "test_novel", "train"))
    p.add_argument("--camera", default="realsense")
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=DIR; specify exactly two models for score-swap analysis.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--candidate_source", choices=("auto", "raw_sidecar", "final_npy"), default="auto")
    p.add_argument("--max_candidates", type=int, default=1024)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--sample_stride", type=int, default=1)
    p.add_argument("--top_k", default="1,10,50")
    p.add_argument("--thresholds", default=",".join(str(x) for x in CDF_THRESHOLDS))
    p.add_argument("--collision_chunk", type=int, default=512)
    p.add_argument("--fc_mode", choices=("official", "reuse_contacts", "reuse_contacts_binary"), default="reuse_contacts")
    p.add_argument("--fc_verify_n", type=int, default=0)
    p.add_argument("--skip_force_closure", type=int, choices=(0, 1), default=0)
    p.add_argument("--pad_to_50", type=int, choices=(0, 1), default=1)
    p.add_argument("--per_object_cap", type=int, default=10)
    p.add_argument("--match_translation_m", type=float, default=0.0075)
    p.add_argument("--match_rotation_deg", type=float, default=15.0)
    p.add_argument("--match_width_m", type=float, default=0.02)
    p.add_argument("--match_depth_m", type=float, default=0.011)
    p.add_argument("--save_candidate_rows", type=int, choices=(0, 1), default=0)
    p.add_argument("--resume", type=int, choices=(0, 1), default=1)
    p.add_argument("--strict", type=int, choices=(0, 1), default=1)
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of scene-level CPU evaluator processes. 0 uses all CPUs "
            "available in the current affinity mask; 1 keeps the old serial path."
        ),
    )
    p.add_argument(
        "--mp_start_method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
        help="Multiprocessing start method. spawn is the safest with DexNet/Open3D.",
    )
    p.add_argument(
        "--worker_max_scenes",
        type=int,
        default=1,
        help=(
            "Recycle a worker after this many scene tasks. 1 bounds memory because "
            "the evaluator caches CAD/DexNet models per scene; 0 disables recycling."
        ),
    )
    p.add_argument(
        "--worker_threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads inside each evaluator process.",
    )
    p.add_argument(
        "--progress_every_scenes",
        type=int,
        default=1,
        help="Parent-process progress interval in completed scenes.",
    )
    return p.parse_args()


def parse_models(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be NAME=DIR, got {value!r}")
        name, directory = value.split("=", 1)
        name = name.strip()
        path = Path(directory).expanduser().resolve()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate model name {name!r}")
        if path.is_file():
            raise FileNotFoundError(
                f"--model expects a candidate-dump directory, but {path} is a file. "
                "Do not pass a .tar checkpoint here. For dumps produced by "
                "p0_b_dump_stage1_uniform.sh, use .../pkd_p0_b_dumps/stage1 "
                "and .../pkd_p0_b_dumps/uniform."
            )
        if not path.is_dir():
            raise FileNotFoundError(
                f"Candidate-dump directory does not exist: {path}. Expected a "
                "model root containing test_seen/test_similar/test_novel."
            )
        result[name] = path
    return result


def parse_float_csv(text: str) -> np.ndarray:
    values = np.asarray([float(token) for token in text.split(",") if token.strip()], dtype=np.float32)
    if values.ndim != 1 or len(values) == 0 or np.any(np.diff(values) <= 0):
        raise ValueError(f"Invalid increasing threshold list {text!r}")
    return values


def parse_int_csv(text: str) -> List[int]:
    values = sorted({int(token) for token in text.split(",") if token.strip()})
    if not values or values[0] <= 0:
        raise ValueError(f"Invalid top-k list {text!r}")
    return values


def discover_files(root: Path, source: str) -> Dict[Tuple[int, int], Path]:
    files: List[Path] = []
    if source in {"auto", "raw_sidecar"}:
        files = sorted(root.rglob("*.p0_candidates.npz"))
        if source == "raw_sidecar" and not files:
            raise FileNotFoundError(f"No *.p0_candidates.npz under {root}")
    if not files and source in {"auto", "final_npy"}:
        files = [path for path in sorted(root.rglob("*.npy")) if not path.name.startswith("ap_")]
    result: Dict[Tuple[int, int], Path] = {}
    for path in files:
        scene_id, anno_id = parse_scene_anno_from_path(path)
        key = (scene_id, anno_id)
        if key in result:
            raise RuntimeError(f"Duplicate frame {key}: {result[key]} and {path}")
        result[key] = path
    if not result:
        raise FileNotFoundError(f"No candidate files under {root}")
    return result


def load_candidates(path: Path, max_candidates: int) -> Tuple[np.ndarray, str]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "raw_grasps" not in data:
                raise KeyError(f"{path} lacks raw_grasps")
            rows = np.asarray(data["raw_grasps"], dtype=np.float32)
        stage = "pre_postprocess"
    else:
        rows = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        stage = "final_dump"
    if rows.ndim != 2 or rows.shape[1] != 17:
        raise ValueError(f"Expected [N,17] grasp rows in {path}, got {rows.shape}")
    rows = rows[np.isfinite(rows).all(axis=1)]
    if len(rows):
        rows = rows[np.argsort(-rows[:, 0], kind="stable")]
    if max_candidates > 0:
        rows = rows[:max_candidates]
    return rows, stage


def build_evaluator(args: argparse.Namespace) -> Any:
    module = None
    import_errors = []
    for module_name in (
        "pkd_p0.graspnet_evaluator",
        "exact_action_graspnet_evaluator",
    ):
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as exc:
            # Only treat a missing candidate module as a fallback condition;
            # preserve errors raised by its internal dependencies.
            if exc.name != module_name:
                raise
            import_errors.append(f"{module_name}: {exc}")
    if module is None:
        raise ModuleNotFoundError(
            "No exact GraspNet evaluator is available for P0-B. Install "
            "pkd_p0/graspnet_evaluator.py from the P0-B hotfix. Attempts: "
            + "; ".join(import_errors)
        )
    candidate_names = ("ExactGraspNetActionEvaluator", "RawCandidateEvaluator")
    cls = next((getattr(module, name) for name in candidate_names if hasattr(module, name)), None)
    if cls is None:
        classes = [value for value in vars(module).values() if inspect.isclass(value) and hasattr(value, "evaluate")]
        if len(classes) != 1:
            raise RuntimeError(f"Cannot identify evaluator class; candidates={classes}")
        cls = classes[0]
    kwargs = {
        "dataset_root": args.dataset_root,
        "root": args.dataset_root,
        "camera": args.camera,
        "split": args.split,
        "collision_chunk": int(args.collision_chunk),
        "chunk": int(args.collision_chunk),
        "skip_force_closure": bool(args.skip_force_closure),
        "fc_mode": args.fc_mode,
        "fc_verify_n": int(args.fc_verify_n),
        "verify_n": int(args.fc_verify_n),
        "strict": bool(args.strict),
    }
    signature = inspect.signature(cls)
    selected = filtered_kwargs(cls, kwargs)
    positional: List[Any] = []
    if "dataset_root" not in signature.parameters and "root" not in signature.parameters:
        positional.append(args.dataset_root)
    evaluator = cls(*positional, **selected)
    return evaluator


def normalize_evaluation(result: Any) -> CandidateEvaluation:
    if isinstance(result, Mapping):
        mapping = dict(result)
    else:
        mapping = {
            key: getattr(result, key)
            for key in ("assigned_obj", "collision_or_empty", "pure_collision", "empty", "friction", "stats")
            if hasattr(result, key)
        }
    aliases = {
        "assigned_obj": ("assigned_obj", "assigned_object", "object_id"),
        "collision_or_empty": ("collision_or_empty", "collision", "collision_mask"),
        "pure_collision": ("pure_collision",),
        "empty": ("empty", "empty_mask"),
        "friction": ("friction", "friction_score", "score"),
    }
    values: Dict[str, np.ndarray] = {}
    for canonical, keys in aliases.items():
        value = next((mapping[key] for key in keys if key in mapping), None)
        if value is None:
            if canonical == "pure_collision" and "collision_or_empty" in values and "empty" in values:
                value = values["collision_or_empty"].astype(bool) & ~values["empty"].astype(bool)
            else:
                raise KeyError(f"Evaluator result lacks {canonical}; keys={sorted(mapping)}")
        values[canonical] = np.asarray(value)
    return CandidateEvaluation(
        assigned_obj=values["assigned_obj"].astype(np.int64).reshape(-1),
        collision_or_empty=values["collision_or_empty"].astype(bool).reshape(-1),
        pure_collision=values["pure_collision"].astype(bool).reshape(-1),
        empty=values["empty"].astype(bool).reshape(-1),
        friction=values["friction"].astype(np.float32).reshape(-1),
        stats=dict(mapping.get("stats", {})),
    )


def evaluate_rows(evaluator: Any, scene_id: int, anno_id: int, rows: np.ndarray) -> CandidateEvaluation:
    result = evaluator.evaluate(int(scene_id), int(anno_id), rows)
    normalized = normalize_evaluation(result)
    if len(normalized.friction) != len(rows):
        raise RuntimeError(f"Evaluator returned {len(normalized.friction)} rows for {len(rows)} candidates")
    return normalized


def cached_evaluate_rows(
    evaluator: Any,
    *,
    model: str,
    scene_id: int,
    anno_id: int,
    rows: np.ndarray,
    candidate_path: Path,
    cache_root: Path,
    resume: bool,
) -> CandidateEvaluation:
    cache_path = cache_root / model / f"scene_{scene_id:04d}" / f"ann_{anno_id:04d}.npz"
    candidate_sha = sha256_file(candidate_path)
    if resume and cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if str(np.asarray(data["candidate_sha256"]).reshape(-1)[0]) == candidate_sha and int(np.asarray(data["num_candidates"]).reshape(-1)[0]) == len(rows):
                    return CandidateEvaluation(
                        assigned_obj=np.asarray(data["assigned_obj"], dtype=np.int64),
                        collision_or_empty=np.asarray(data["collision_or_empty"], dtype=bool),
                        pure_collision=np.asarray(data["pure_collision"], dtype=bool),
                        empty=np.asarray(data["empty"], dtype=bool),
                        friction=np.asarray(data["friction"], dtype=np.float32),
                        stats={"cache_hit": True},
                    )
        except Exception:
            pass
    evaluation = evaluate_rows(evaluator, scene_id, anno_id, rows)
    atomic_npz_dump(
        cache_path,
        compress=False,
        assigned_obj=evaluation.assigned_obj.astype(np.int16),
        collision_or_empty=evaluation.collision_or_empty.astype(np.uint8),
        pure_collision=evaluation.pure_collision.astype(np.uint8),
        empty=evaluation.empty.astype(np.uint8),
        friction=evaluation.friction.astype(np.float32),
        candidate_sha256=np.asarray(candidate_sha),
        num_candidates=np.asarray([len(rows)], dtype=np.int32),
    )
    return evaluation


def success_matrix(friction: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    f = np.asarray(friction, dtype=np.float32).reshape(-1, 1)
    return (f > 0.0) & (f <= thresholds.reshape(1, -1))


def utility(friction: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return success_matrix(friction, thresholds).mean(axis=1).astype(np.float32)


def ap_from_order(success: np.ndarray, order: np.ndarray, *, max_k: int = 50, pad: bool = True) -> float:
    ordered = success[order[:max_k]].astype(np.float64)
    if pad and len(ordered) < max_k:
        ordered = np.concatenate([ordered, np.zeros((max_k - len(ordered), success.shape[1]), dtype=np.float64)], axis=0)
    if len(ordered) == 0:
        return 0.0
    cumulative_precision = np.cumsum(ordered, axis=0) / np.arange(1, len(ordered) + 1, dtype=np.float64)[:, None]
    return float(cumulative_precision.mean())


def topk_rate(values: np.ndarray, order: np.ndarray, k: int, *, pad: bool = False) -> float:
    selected = np.asarray(values)[order[:k]].astype(np.float64)
    if pad and len(selected) < k:
        selected = np.concatenate([selected, np.ones(k - len(selected), dtype=np.float64)])
    return float(selected.mean()) if len(selected) else float("nan")


def topk_utility(values: np.ndarray, order: np.ndarray, k: int) -> float:
    selected = np.asarray(values)[order[:k]]
    return float(selected.mean()) if len(selected) else float("nan")


def object_coverage(assigned_obj: np.ndarray, valid: np.ndarray, order: np.ndarray, k: int) -> int:
    ids = assigned_obj[order[:k]]
    mask = valid[order[:k]] & (ids >= 0)
    return int(len(np.unique(ids[mask])))


def capped_order(values: np.ndarray, assigned_obj: np.ndarray, per_object_cap: int) -> np.ndarray:
    """Sort descending and apply the GraspNet per-object prediction cap."""
    global_order = np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")
    if per_object_cap <= 0:
        return global_order
    counts: Dict[int, int] = {}
    kept: List[int] = []
    for index in global_order:
        object_id = int(assigned_obj[index])
        count = counts.get(object_id, 0)
        if count >= per_object_cap:
            continue
        counts[object_id] = count + 1
        kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def frame_metrics(
    model: str,
    scene_id: int,
    anno_id: int,
    path: Path,
    stage: str,
    rows: np.ndarray,
    evaluation: CandidateEvaluation,
    thresholds: np.ndarray,
    top_ks: Sequence[int],
    pad: bool,
    per_object_cap: int,
    include_detailed: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]] :
    scores = rows[:, 0].astype(np.float64)
    success = success_matrix(evaluation.friction, thresholds)
    util = success.mean(axis=1)
    score_order = capped_order(scores, evaluation.assigned_obj, per_object_cap)
    # Small score tie-break keeps deterministic ordering among equal exact utilities.
    oracle_values = util + 1e-9 * (scores - scores.min()) / (np.ptp(scores) + 1e-12)
    oracle_order = capped_order(oracle_values, evaluation.assigned_obj, per_object_cap)
    actual_ap = ap_from_order(success, score_order, max_k=50, pad=pad)
    oracle_ap = ap_from_order(success, oracle_order, max_k=50, pad=pad)
    valid = evaluation.friction > 0.0
    summary: Dict[str, Any] = {
        "model": model,
        "scene_id": scene_id,
        "anno_id": anno_id,
        "candidate_stage": stage,
        "candidate_file": str(path),
        "num_candidates": int(len(rows)),
        "actual_ap": actual_ap,
        "oracle_ap": oracle_ap,
        "oracle_gap": oracle_ap - actual_ap,
        "valid_ratio_all": float(valid.mean()) if len(valid) else float("nan"),
        "safe04_count_all": int(((evaluation.friction > 0) & (evaluation.friction <= 0.4)).sum()),
        "safe08_count_all": int(((evaluation.friction > 0) & (evaluation.friction <= 0.8)).sum()),
        "valid_object_coverage_all": int(len(np.unique(evaluation.assigned_obj[valid & (evaluation.assigned_obj >= 0)]))),
    }
    for k in top_ks:
        summary[f"top{k}_utility"] = topk_utility(util, score_order, k)
        summary[f"top{k}_oracle_utility"] = topk_utility(util, oracle_order, k)
        summary[f"top{k}_collision_or_empty"] = topk_rate(evaluation.collision_or_empty, score_order, k)
        summary[f"top{k}_pure_collision"] = topk_rate(evaluation.pure_collision, score_order, k)
        summary[f"top{k}_empty"] = topk_rate(evaluation.empty, score_order, k)
        summary[f"top{k}_eval_fail"] = topk_rate(evaluation.friction <= 0.0, score_order, k)
        summary[f"top{k}_object_coverage"] = object_coverage(evaluation.assigned_obj, valid, score_order, k)

    rows_out: List[Dict[str, Any]] = []
    if include_detailed:
        for rank, index in enumerate(score_order, start=1):
            rows_out.append({
                "model": model,
                "scene_id": scene_id,
                "anno_id": anno_id,
                "candidate_index": int(index),
                "predicted_rank": int(rank),
                "score": float(scores[index]),
                "width": float(rows[index, 1]),
                "depth": float(rows[index, 3]),
                "tx": float(rows[index, 13]),
                "ty": float(rows[index, 14]),
                "tz": float(rows[index, 15]),
                "assigned_obj": int(evaluation.assigned_obj[index]),
                "friction": float(evaluation.friction[index]),
                "utility": float(util[index]),
                "collision_or_empty": int(evaluation.collision_or_empty[index]),
                "pure_collision": int(evaluation.pure_collision[index]),
                "empty": int(evaluation.empty[index]),
            })
    return summary, rows_out


def greedy_match(
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    *,
    translation_m: float,
    rotation_deg: float,
    width_m: float,
    depth_m: float,
) -> List[Tuple[int, int, float, float]]:
    """Greedy one-to-one SE(3) matching, sorted by translation then rotation."""
    if len(rows_a) == 0 or len(rows_b) == 0:
        return []
    ta = rows_a[:, 13:16]
    tb = rows_b[:, 13:16]
    candidates: List[Tuple[float, float, int, int]] = []
    # Vectorized by A row to avoid an NxM rotation tensor in memory.
    rb = rows_b[:, 4:13].reshape(-1, 3, 3)
    for i in range(len(rows_a)):
        distance = np.linalg.norm(tb - ta[i], axis=1)
        js = np.flatnonzero(distance <= translation_m)
        if len(js) == 0:
            continue
        if width_m > 0:
            js = js[np.abs(rows_b[js, 1] - rows_a[i, 1]) <= width_m]
        if depth_m > 0 and len(js):
            js = js[np.abs(rows_b[js, 3] - rows_a[i, 3]) <= depth_m]
        if len(js) == 0:
            continue
        ra = np.repeat(rows_a[i, 4:13].reshape(1, 3, 3), len(js), axis=0)
        angle = rotation_geodesic_deg(ra, rb[js])
        for local, j in enumerate(js):
            if angle[local] <= rotation_deg:
                candidates.append((float(distance[j]), float(angle[local]), i, int(j)))
    candidates.sort()
    used_a, used_b = set(), set()
    matches: List[Tuple[int, int, float, float]] = []
    for distance, angle, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, distance, angle))
    return matches


def matched_score_swap(
    name_a: str,
    rows_a: np.ndarray,
    eval_a: CandidateEvaluation,
    name_b: str,
    rows_b: np.ndarray,
    thresholds: np.ndarray,
    args: argparse.Namespace,
    scene_id: int,
    anno_id: int,
) -> Dict[str, Any]:
    matches = greedy_match(
        rows_a,
        rows_b,
        translation_m=float(args.match_translation_m),
        rotation_deg=float(args.match_rotation_deg),
        width_m=float(args.match_width_m),
        depth_m=float(args.match_depth_m),
    )
    if not matches:
        return {
            "candidate_model": name_a,
            "score_model": name_b,
            "scene_id": scene_id,
            "anno_id": anno_id,
            "matched": 0,
            "candidate_count": int(len(rows_a)),
            "match_ratio": 0.0,
            "own_score_ap_common": float("nan"),
            "other_score_ap_common": float("nan"),
        }
    ia = np.asarray([item[0] for item in matches], dtype=np.int64)
    ib = np.asarray([item[1] for item in matches], dtype=np.int64)
    success = success_matrix(eval_a.friction[ia], thresholds)
    own_order = capped_order(rows_a[ia, 0], eval_a.assigned_obj[ia], int(args.per_object_cap))
    other_order = capped_order(rows_b[ib, 0], eval_a.assigned_obj[ia], int(args.per_object_cap))
    return {
        "candidate_model": name_a,
        "score_model": name_b,
        "scene_id": scene_id,
        "anno_id": anno_id,
        "matched": int(len(matches)),
        "candidate_count": int(len(rows_a)),
        "match_ratio": float(len(matches) / max(len(rows_a), 1)),
        "translation_mean_m": float(np.mean([item[2] for item in matches])),
        "rotation_mean_deg": float(np.mean([item[3] for item in matches])),
        "own_score_ap_common": ap_from_order(success, own_order, max_k=50, pad=bool(args.pad_to_50)),
        "other_score_ap_common": ap_from_order(success, other_order, max_k=50, pad=bool(args.pad_to_50)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_keys)
        groups.setdefault(key, []).append(row)
    result: List[Dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        record = {field: value for field, value in zip(group_keys, key)}
        numeric_keys = sorted({field for member in members for field, value in member.items() if isinstance(value, (int, float, np.number)) and field not in group_keys})
        record["num_frames"] = len(members)
        for field in numeric_keys:
            values = np.asarray([float(member[field]) for member in members if field in member and math.isfinite(float(member[field]))], dtype=np.float64)
            if len(values):
                record[field] = float(values.mean())
        result.append(record)
    return result


# ---------------------------------------------------------------------------
# Scene-level multiprocessing
# ---------------------------------------------------------------------------

_WORKER_STATE: Dict[str, Any] = {}


def available_cpu_count() -> int:
    """Return CPUs available to the current affinity/cgroup allocation."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def set_thread_limits(num_threads: int) -> None:
    """Avoid nested BLAS/OpenMP oversubscription in each process."""
    value = str(max(1, int(num_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = value


def build_scene_tasks(
    frame_keys: Sequence[Tuple[int, int]],
    files_by_model: Mapping[str, Mapping[Tuple[int, int], Path]],
) -> List[Dict[str, Any]]:
    """Group frames by scene so one evaluator process reuses one CAD scene."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    model_names = list(files_by_model)
    for scene_id, anno_id in frame_keys:
        grouped.setdefault(int(scene_id), []).append(
            {
                "anno_id": int(anno_id),
                "paths": {
                    name: str(files_by_model[name][(scene_id, anno_id)])
                    for name in model_names
                },
            }
        )
    tasks: List[Dict[str, Any]] = []
    for scene_id in sorted(grouped):
        tasks.append(
            {
                "scene_id": int(scene_id),
                "frames": sorted(grouped[scene_id], key=lambda row: row["anno_id"]),
            }
        )
    return tasks


def init_scene_worker(config: Mapping[str, Any]) -> None:
    """Initialize process-local configuration; construct evaluator lazily."""
    global _WORKER_STATE
    args = SimpleNamespace(**dict(config["args"]))
    set_thread_limits(int(args.worker_threads))
    _WORKER_STATE = {
        "args": args,
        "model_names": tuple(config["model_names"]),
        "thresholds": np.asarray(config["thresholds"], dtype=np.float32),
        "top_ks": tuple(int(value) for value in config["top_ks"]),
        "out_dir": Path(config["out_dir"]),
        # Build inside process_scene_task's try/except. A Pool initializer
        # exception can otherwise trigger an unhelpful worker-respawn loop.
        "evaluator": None,
    }


def process_scene_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate every selected frame in one scene inside one CPU process."""
    try:
        state = _WORKER_STATE
        if not state:
            raise RuntimeError("P0-B worker state was not initialized")
        args = state["args"]
        evaluator = state.get("evaluator")
        if evaluator is None:
            evaluator = build_evaluator(args)
            state["evaluator"] = evaluator
        model_names: Tuple[str, ...] = state["model_names"]
        thresholds: np.ndarray = state["thresholds"]
        top_ks: Tuple[int, ...] = state["top_ks"]
        out_dir: Path = state["out_dir"]

        scene_id = int(task["scene_id"])
        summaries: List[Dict[str, Any]] = []
        swaps: List[Dict[str, Any]] = []
        candidate_rows: List[Dict[str, Any]] = []
        cache_hits = 0
        evaluated_candidates = 0
        started = time.time()

        for frame in task["frames"]:
            anno_id = int(frame["anno_id"])
            loaded: Dict[str, Tuple[np.ndarray, str, CandidateEvaluation, Path]] = {}
            for name in model_names:
                path = Path(frame["paths"][name])
                rows, stage = load_candidates(path, int(args.max_candidates))
                evaluation = cached_evaluate_rows(
                    evaluator,
                    model=name,
                    scene_id=scene_id,
                    anno_id=anno_id,
                    rows=rows,
                    candidate_path=path,
                    cache_root=out_dir / "evaluation_cache",
                    resume=bool(args.resume),
                )
                cache_hits += int(bool(evaluation.stats.get("cache_hit", False)))
                evaluated_candidates += int(len(rows))
                summary, detailed = frame_metrics(
                    name,
                    scene_id,
                    anno_id,
                    path,
                    stage,
                    rows,
                    evaluation,
                    thresholds,
                    top_ks,
                    bool(args.pad_to_50),
                    int(args.per_object_cap),
                    include_detailed=bool(args.save_candidate_rows),
                )
                summaries.append(summary)
                if bool(args.save_candidate_rows):
                    candidate_rows.extend(detailed)
                loaded[name] = (rows, stage, evaluation, path)

            if len(model_names) == 2:
                name_a, name_b = model_names
                rows_a, _, eval_a, _ = loaded[name_a]
                rows_b, _, eval_b, _ = loaded[name_b]
                swaps.append(
                    matched_score_swap(
                        name_a,
                        rows_a,
                        eval_a,
                        name_b,
                        rows_b,
                        thresholds,
                        args,
                        scene_id,
                        anno_id,
                    )
                )
                swaps.append(
                    matched_score_swap(
                        name_b,
                        rows_b,
                        eval_b,
                        name_a,
                        rows_a,
                        thresholds,
                        args,
                        scene_id,
                        anno_id,
                    )
                )

        candidate_shard = ""
        if bool(args.save_candidate_rows):
            shard_dir = out_dir / "_candidate_row_shards"
            candidate_path = shard_dir / f"scene_{scene_id:04d}.csv"
            write_csv(candidate_path, candidate_rows)
            candidate_shard = str(candidate_path)

        return {
            "ok": True,
            "scene_id": scene_id,
            "num_frames": len(task["frames"]),
            "summaries": summaries,
            "swaps": swaps,
            "candidate_shard": candidate_shard,
            "cache_hits": cache_hits,
            "evaluated_candidates": evaluated_candidates,
            "elapsed_seconds": time.time() - started,
            "worker_pid": os.getpid(),
        }
    except BaseException:
        return {
            "ok": False,
            "scene_id": int(task.get("scene_id", -1)),
            "worker_pid": os.getpid(),
            "traceback": traceback.format_exc(),
        }


def concatenate_csv_shards(shards: Sequence[Path], output_path: Path) -> None:
    """Concatenate deterministic per-scene candidate-row CSV shards."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted({Path(path) for path in shards})
    if not ordered:
        output_path.write_text("", encoding="utf-8")
        return
    header: Optional[List[str]] = None
    with open(output_path, "w", newline="", encoding="utf-8") as output:
        writer: Optional[csv.DictWriter] = None
        for shard in ordered:
            with open(shard, "r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                fields = list(reader.fieldnames or [])
                if not fields:
                    continue
                if header is None:
                    header = fields
                    writer = csv.DictWriter(output, fieldnames=header)
                    writer.writeheader()
                elif fields != header:
                    raise RuntimeError(
                        f"Candidate-row shard schema mismatch: {shard} has "
                        f"{fields}, expected {header}"
                    )
                assert writer is not None
                writer.writerows(reader)


def collect_result(
    result: Mapping[str, Any],
    *,
    summaries: List[Dict[str, Any]],
    swaps: List[Dict[str, Any]],
    candidate_shards: List[Path],
) -> Tuple[int, int]:
    if not bool(result.get("ok", False)):
        raise RuntimeError(
            f"P0-B scene {int(result.get('scene_id', -1)):04d} failed in "
            f"worker {result.get('worker_pid')}\n{result.get('traceback', '')}"
        )
    summaries.extend(result["summaries"])
    swaps.extend(result["swaps"])
    shard = str(result.get("candidate_shard", ""))
    if shard:
        candidate_shards.append(Path(shard))
    return int(result["num_frames"]), int(result.get("cache_hits", 0))


def main() -> None:
    args = parse_args()
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be >= 0")
    if int(args.worker_max_scenes) < 0:
        raise ValueError("--worker_max_scenes must be >= 0")
    if int(args.worker_threads) <= 0:
        raise ValueError("--worker_threads must be positive")
    if int(args.progress_every_scenes) <= 0:
        raise ValueError("--progress_every_scenes must be positive")

    # Set this before spawn workers import NumPy/SciPy/DexNet.
    set_thread_limits(int(args.worker_threads))

    models = parse_models(args.model)
    thresholds = parse_float_csv(args.thresholds)
    top_ks = parse_int_csv(args.top_k)
    files_by_model = {
        name: discover_files(path, args.candidate_source)
        for name, path in models.items()
    }
    common_frames = set.intersection(
        *(set(files) for files in files_by_model.values())
    )
    if int(args.sample_stride) > 1:
        common_frames = {
            key
            for key in common_frames
            if key[1] % int(args.sample_stride) == 0
        }
    frame_keys = sorted(common_frames)
    if int(args.max_frames) > 0:
        frame_keys = frame_keys[: int(args.max_frames)]
    if not frame_keys:
        raise RuntimeError("No common candidate frames across models")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_tasks = build_scene_tasks(frame_keys, files_by_model)
    if not scene_tasks:
        raise RuntimeError("No scene tasks were built from common frames")

    requested_workers = int(args.num_workers)
    if requested_workers == 0:
        requested_workers = available_cpu_count()
    num_workers = max(1, min(requested_workers, len(scene_tasks)))
    if int(args.fc_verify_n) > 0 and num_workers > 1:
        print(
            "[P0-B][WARN] --fc_verify_n is applied independently in every "
            "worker. Use --num_workers 1 for a minimal verification smoke run.",
            flush=True,
        )

    print(
        f"[P0-B][START] split={args.split} source={args.candidate_source} "
        f"frames={len(frame_keys)} scenes={len(scene_tasks)} "
        f"workers={num_workers} available_cpus={available_cpu_count()} "
        f"worker_threads={int(args.worker_threads)} "
        f"mp_start={args.mp_start_method} "
        f"worker_max_scenes={int(args.worker_max_scenes)}",
        flush=True,
    )
    print(
        "[P0-B][NOTE] Parallelism is scene-level: one process evaluates all "
        "selected frames of a scene and reuses that scene's CAD/DexNet models.",
        flush=True,
    )

    config = {
        "args": dict(vars(args)),
        "model_names": list(models),
        "thresholds": thresholds.tolist(),
        "top_ks": list(top_ks),
        "out_dir": str(out_dir),
    }
    summaries: List[Dict[str, Any]] = []
    swaps: List[Dict[str, Any]] = []
    candidate_shards: List[Path] = []
    processed_frames = 0
    completed_scenes = 0
    total_cache_hits = 0
    started = time.time()

    if num_workers == 1:
        init_scene_worker(config)
        for task in scene_tasks:
            result = process_scene_task(task)
            frames_done, cache_hits = collect_result(
                result,
                summaries=summaries,
                swaps=swaps,
                candidate_shards=candidate_shards,
            )
            processed_frames += frames_done
            total_cache_hits += cache_hits
            completed_scenes += 1
            if (
                completed_scenes % int(args.progress_every_scenes) == 0
                or completed_scenes == len(scene_tasks)
            ):
                print(
                    f"[P0-B] scenes={completed_scenes}/{len(scene_tasks)} "
                    f"frames={processed_frames}/{len(frame_keys)} "
                    f"last_scene={int(result['scene_id']):04d} "
                    f"worker={result['worker_pid']} "
                    f"scene_time={float(result['elapsed_seconds']):.1f}s "
                    f"cache_hits={total_cache_hits} "
                    f"elapsed={(time.time() - started) / 60.0:.1f}m",
                    flush=True,
                )
    else:
        context = mp.get_context(str(args.mp_start_method))
        max_tasks = (
            None
            if int(args.worker_max_scenes) == 0
            else int(args.worker_max_scenes)
        )
        pool = context.Pool(
            processes=num_workers,
            initializer=init_scene_worker,
            initargs=(config,),
            maxtasksperchild=max_tasks,
        )
        try:
            iterator = pool.imap_unordered(
                process_scene_task,
                scene_tasks,
                chunksize=1,
            )
            for result in iterator:
                frames_done, cache_hits = collect_result(
                    result,
                    summaries=summaries,
                    swaps=swaps,
                    candidate_shards=candidate_shards,
                )
                processed_frames += frames_done
                total_cache_hits += cache_hits
                completed_scenes += 1
                if (
                    completed_scenes % int(args.progress_every_scenes) == 0
                    or completed_scenes == len(scene_tasks)
                ):
                    print(
                        f"[P0-B][MP] scenes={completed_scenes}/{len(scene_tasks)} "
                        f"frames={processed_frames}/{len(frame_keys)} "
                        f"last_scene={int(result['scene_id']):04d} "
                        f"worker={result['worker_pid']} "
                        f"scene_time={float(result['elapsed_seconds']):.1f}s "
                        f"cache_hits={total_cache_hits} "
                        f"elapsed={(time.time() - started) / 60.0:.1f}m",
                        flush=True,
                    )
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()

    # imap_unordered returns scenes in completion order; sort all public outputs
    # so serial and parallel runs are byte-order deterministic modulo timings.
    summaries.sort(
        key=lambda row: (
            str(row["model"]),
            int(row["scene_id"]),
            int(row["anno_id"]),
        )
    )
    swaps.sort(
        key=lambda row: (
            str(row["candidate_model"]),
            str(row["score_model"]),
            int(row["scene_id"]),
            int(row["anno_id"]),
        )
    )

    per_scene = aggregate(summaries, ("model", "scene_id"))
    overall = aggregate(summaries, ("model",))
    write_csv(out_dir / "per_frame.csv", summaries)
    write_csv(out_dir / "per_scene.csv", per_scene)
    write_csv(out_dir / "summary.csv", overall)
    write_csv(out_dir / "matched_score_swap.csv", swaps)
    if bool(args.save_candidate_rows):
        concatenate_csv_shards(
            candidate_shards,
            out_dir / "candidate_rows.csv",
        )

    atomic_json_dump(
        {
            "experiment": "P0-B candidate/ranking decomposition",
            "parallel_version": "p0_b_scene_multiprocessing_v1_2",
            "models": {name: str(path) for name, path in models.items()},
            "split": args.split,
            "num_frames": len(frame_keys),
            "num_scenes": len(scene_tasks),
            "thresholds": thresholds.tolist(),
            "top_k": top_ks,
            "candidate_source": args.candidate_source,
            "max_candidates": int(args.max_candidates),
            "per_object_cap": int(args.per_object_cap),
            "parallel": {
                "num_workers": num_workers,
                "available_cpus": available_cpu_count(),
                "worker_threads": int(args.worker_threads),
                "mp_start_method": str(args.mp_start_method),
                "worker_max_scenes": int(args.worker_max_scenes),
                "granularity": "scene",
                "evaluation_cache_hits": int(total_cache_hits),
            },
            "match_thresholds": {
                "translation_m": float(args.match_translation_m),
                "rotation_deg": float(args.match_rotation_deg),
                "width_m": float(args.match_width_m),
                "depth_m": float(args.match_depth_m),
            },
            "overall": overall,
            "elapsed_seconds": time.time() - started,
        },
        out_dir / "summary.json",
    )
    print(f"[DONE] P0-B outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
