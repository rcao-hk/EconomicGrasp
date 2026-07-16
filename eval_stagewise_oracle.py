"""Evaluate first-generation CVA stage-wise oracle dumps on GraspNet-1Billion.

The stock GraspNet evaluator expects all 256 frames of a scene.  This script
instead scans the files actually present in each mode, so the same evaluator can
be used on a sparse ``sample_interval`` diagnostic subset.

For each mode it reports official AP/AP0.4/AP0.8 and a *selected-pool evaluator
oracle*: after official ``eval_grasp`` preselection, retained candidates are
re-ranked by their evaluator friction utility.  This isolates residual ranking
error inside the retained pool.  It is not a raw-candidate oracle because
``eval_grasp`` has already applied NMS and score-dependent per-object/global
preselection.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from graspnetAPI import GraspGroup, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    create_table_points,
    eval_grasp,
    transform_points,
    voxel_sample_points,
)


FRICTIONS = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
CUMULATIVE_CHAIN = [
    "s0_base",
    "s1_oracle_view",
    "s2_oracle_view_angle",
    "s3_oracle_view_angle_depth",
    "s4_oracle_view_operation",
    "s5_oracle_view_operation_labelrank",
]
EXPECTED_SCENE_RANGES = {
    "test_seen": (100, 129),
    "test_similar": (130, 159),
    "test_novel": (160, 189),
    "test": (100, 189),
}
SPLIT_FIRST_SCENE = {
    "test_seen": 100,
    "test_similar": 130,
    "test_novel": 160,
    "test": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CVA stage-wise oracle decomposition."
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--base_save_dir", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument(
        "--split",
        default="test_seen",
        choices=["test_seen", "test_similar", "test_novel", "test"],
    )
    parser.add_argument(
        "--modes",
        default="auto",
        help="Comma-separated mode names, or 'auto' from _stagewise_modes.json.",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_width", type=float, default=0.1)
    parser.add_argument(
        "--sample_policy",
        choices=["intersection", "union"],
        default="intersection",
        help="Use samples present in every mode (recommended) or their union.",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <base_save_dir>/_stagewise_eval",
    )
    parser.add_argument(
        "--skip_selected_pool_oracle",
        action="store_true",
        help="Skip evaluator-based reranking inside the retained candidate pool.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Number of CPU worker processes. Parallelization is scene-wise so "
            "each worker loads a scene model once and evaluates all sampled "
            "annotations/modes from that scene."
        ),
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help=(
            "OMP/MKL/OpenBLAS thread limit inside each worker. Keep this at 1 "
            "when using multiple worker processes to avoid oversubscription."
        ),
    )
    parser.add_argument(
        "--mp_start_method",
        choices=["fork", "spawn", "forkserver"],
        default=("fork" if "fork" in mp.get_all_start_methods() else "spawn"),
        help="Multiprocessing start method. Linux default: fork.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from stagewise_oracle_per_sample.partial.csv and the "
            "progress manifest in output_dir. Completed scenes are skipped."
        ),
    )
    return parser.parse_args()


def _load_mode_roots(base_save_dir: str, modes_arg: str) -> Dict[str, str]:
    mode_file = os.path.join(base_save_dir, "_stagewise_modes.json")
    mode_roots: Dict[str, str] = {}
    if os.path.isfile(mode_file):
        with open(mode_file, "r") as f:
            payload = json.load(f)
        mode_roots = {str(k): str(v) for k, v in payload["modes"].items()}

    if modes_arg != "auto":
        requested = [x.strip() for x in modes_arg.split(",") if x.strip()]
        if not requested:
            raise ValueError("--modes produced an empty mode list.")
        if mode_roots:
            missing = [m for m in requested if m not in mode_roots]
            if missing:
                raise KeyError(f"Requested modes absent from manifest: {missing}")
            mode_roots = {m: mode_roots[m] for m in requested}
        else:
            mode_roots = {
                mode: (
                    base_save_dir
                    if mode == "s0_base"
                    else f"{base_save_dir}_{mode}"
                )
                for mode in requested
            }

    if not mode_roots:
        raise FileNotFoundError(
            f"No mode manifest found at {mode_file}; pass --modes explicitly."
        )
    for mode, root in mode_roots.items():
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Mode '{mode}' directory does not exist: {root}")
    return mode_roots


def _manifest_sample_set(root: str) -> set[Tuple[int, int]] | None:
    """Convert the inference manifest's split-relative indices to scene/anno."""
    path = os.path.join(root, "_sampled_indices.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    split = str(payload.get("test_mode", ""))
    if split not in SPLIT_FIRST_SCENE:
        print(f"[WARN] Unknown manifest split '{split}' in {path}; ignoring manifest.")
        return None
    first_scene = SPLIT_FIRST_SCENE[split]
    return {
        (first_scene + int(data_idx) // 256, int(data_idx) % 256)
        for data_idx in payload.get("sampled_indices", [])
    }


def _scan_mode_samples(root: str, camera: str) -> Dict[Tuple[int, int], str]:
    pattern = os.path.join(root, "scene_*", camera, "*.npy")
    expected = _manifest_sample_set(root)
    out: Dict[Tuple[int, int], str] = {}
    for path in glob.glob(pattern):
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            scene_id = int(scene_name.split("_")[-1])
            anno_id = int(stem)
        except ValueError:
            continue
        key = (scene_id, anno_id)
        if expected is not None and key not in expected:
            continue
        out[key] = path
    return out


def _resolve_samples(sample_maps, policy: str) -> List[Tuple[int, int]]:
    sets = [set(x) for x in sample_maps.values()]
    if not sets:
        return []
    samples = set.intersection(*sets) if policy == "intersection" else set.union(*sets)
    return sorted(samples)


def _warn_scene_range(samples: Sequence[Tuple[int, int]], split: str) -> None:
    lo, hi = EXPECTED_SCENE_RANGES[split]
    bad = sorted({scene for scene, _ in samples if not (lo <= scene <= hi)})
    if bad:
        print(
            f"[WARN] {len(bad)} scene IDs fall outside {split} range "
            f"[{lo}, {hi}]: {bad[:10]}{'...' if len(bad) > 10 else ''}"
        )


def _clip_grasp_widths(gg: GraspGroup, max_width: float) -> GraspGroup:
    arr = gg.grasp_group_array
    if arr.size:
        arr[:, 1] = np.clip(arr[:, 1], 0.0, max_width)
        gg.grasp_group_array = arr
    return gg


def _accuracy_from_eval_scores(
    scores: np.ndarray,
    top_k: int,
    frictions: np.ndarray = FRICTIONS,
) -> np.ndarray:
    """Reproduce official prefix precision, including padding by denominator k."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    acc = np.zeros((top_k, len(frictions)), dtype=np.float32)
    if scores.size == 0:
        return acc

    for j, friction in enumerate(frictions):
        success = ((scores > 0.0) & (scores <= friction)).astype(np.float32)
        prefix = np.cumsum(success)
        for k in range(1, top_k + 1):
            n = min(k, scores.size)
            correct = float(prefix[n - 1]) if n else 0.0
            acc[k - 1, j] = correct / float(k)
    return acc


def _evaluator_utility(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return (
        ((scores[:, None] > 0.0) & (scores[:, None] <= FRICTIONS[None, :]))
        .astype(np.float32)
        .mean(axis=1)
    )


def _topk_stats(
    sorted_scores: np.ndarray,
    sorted_collision: np.ndarray,
    prefix: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(sorted_scores)
    for k in [1, 10, 50]:
        kk = min(k, n)
        if kk == 0:
            for name in ["collision", "eval_fail", "success04", "success08"]:
                out[f"{prefix}_top{k}_{name}"] = 0.0
            continue
        quality = sorted_scores[:kk]
        collision = sorted_collision[:kk]
        out[f"{prefix}_top{k}_collision"] = float(
            np.mean(collision.astype(np.float32))
        )
        out[f"{prefix}_top{k}_eval_fail"] = float(np.mean(quality <= 0.0))
        out[f"{prefix}_top{k}_success04"] = float(
            np.mean((quality > 0.0) & (quality <= 0.4))
        )
        out[f"{prefix}_top{k}_success08"] = float(
            np.mean((quality > 0.0) & (quality <= 0.8))
        )
    return out


def _empty_result(top_k: int) -> Dict[str, Any]:
    zero = np.zeros((top_k, len(FRICTIONS)), dtype=np.float32)
    result: Dict[str, Any] = {
        "official_acc": zero,
        "selected_pool_oracle_acc": zero.copy(),
        "candidate_count": 0,
        "selected_pool_fail_ratio": 0.0,
        "selected_pool_collision_ratio": 0.0,
    }
    for prefix in ["official", "oracle_rank"]:
        result.update(
            _topk_stats(
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=bool),
                prefix,
            )
        )
    return result


def _evaluate_one_mode_sample(
    grasp_path: str,
    model_sampled_list: Sequence[np.ndarray],
    dexmodel_list: Sequence[Any],
    pose_list: Sequence[np.ndarray],
    table_trans: np.ndarray,
    config: Dict[str, Any],
    top_k: int,
    max_width: float,
    selected_pool_oracle: bool,
) -> Dict[str, Any]:
    gg = _clip_grasp_widths(GraspGroup().from_npy(grasp_path), max_width)
    if len(gg) == 0:
        return _empty_result(top_k)

    grasp_list, score_list, collision_list = eval_grasp(
        gg,
        model_sampled_list,
        dexmodel_list,
        pose_list,
        config,
        table=table_trans,
        voxel_size=0.008,
        TOP_K=top_k,
    )
    nonempty = [idx for idx, x in enumerate(grasp_list) if len(x) != 0]
    if not nonempty:
        return _empty_result(top_k)

    grasps = np.concatenate([grasp_list[j] for j in nonempty], axis=0)
    scores = np.concatenate([score_list[j] for j in nonempty], axis=0)
    collisions = np.concatenate(
        [collision_list[j] for j in nonempty], axis=0
    ).astype(bool)
    confidence = grasps[:, 0]

    official_order = np.argsort(-confidence, kind="stable")
    official_scores = scores[official_order]
    official_collision = collisions[official_order]
    official_acc = _accuracy_from_eval_scores(official_scores, top_k)

    if selected_pool_oracle:
        utility = _evaluator_utility(scores)
        # np.lexsort uses the last key as primary: utility first, confidence tie-break.
        oracle_order = np.lexsort((-confidence, -utility))
        oracle_scores = scores[oracle_order]
        oracle_collision = collisions[oracle_order]
        oracle_acc = _accuracy_from_eval_scores(oracle_scores, top_k)
    else:
        oracle_scores = official_scores
        oracle_collision = official_collision
        oracle_acc = official_acc.copy()

    result: Dict[str, Any] = {
        "official_acc": official_acc,
        "selected_pool_oracle_acc": oracle_acc,
        "candidate_count": int(len(scores)),
        "selected_pool_fail_ratio": float(np.mean(scores <= 0.0)),
        "selected_pool_collision_ratio": float(
            np.mean(collisions.astype(np.float32))
        ),
    }
    result.update(_topk_stats(official_scores, official_collision, "official"))
    result.update(_topk_stats(oracle_scores, oracle_collision, "oracle_rank"))
    return result


def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(set().union(*(set(row) for row in rows)))
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _save_json(payload: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _load_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _mean_key(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = [float(x[key]) for x in rows if key in x and np.isfinite(float(x[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _summarize_rows(rows, ordered_modes: Sequence[str]):
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    summary: List[Dict[str, Any]] = []
    base_ap = None
    previous_chain_ap = None
    for mode in ordered_modes:
        current = by_mode.get(mode, [])
        if not current:
            continue
        official_ap = _mean_key(current, "official_ap")
        rank_oracle_ap = _mean_key(current, "selected_pool_oracle_ap")
        if mode == "s0_base":
            base_ap = official_ap
        stage_gain = float("nan")
        if mode in CUMULATIVE_CHAIN:
            if previous_chain_ap is not None:
                stage_gain = official_ap - previous_chain_ap
            previous_chain_ap = official_ap

        summary.append(
            {
                "mode": mode,
                "num_samples": len(current),
                "official_ap": official_ap,
                "official_ap04": _mean_key(current, "official_ap04"),
                "official_ap08": _mean_key(current, "official_ap08"),
                "selected_pool_oracle_ap": rank_oracle_ap,
                "ranking_gap_selected_pool": rank_oracle_ap - official_ap,
                "delta_vs_s0_base": (
                    official_ap - base_ap if base_ap is not None else float("nan")
                ),
                "stage_gain_vs_previous": stage_gain,
                "candidate_count": _mean_key(current, "candidate_count"),
                "official_top1_collision": _mean_key(
                    current, "official_top1_collision"
                ),
                "official_top10_collision": _mean_key(
                    current, "official_top10_collision"
                ),
                "official_top50_collision": _mean_key(
                    current, "official_top50_collision"
                ),
                "official_top1_eval_fail": _mean_key(
                    current, "official_top1_eval_fail"
                ),
                "official_top10_eval_fail": _mean_key(
                    current, "official_top10_eval_fail"
                ),
                "official_top10_success08": _mean_key(
                    current, "official_top10_success08"
                ),
                "official_top50_success08": _mean_key(
                    current, "official_top50_success08"
                ),
            }
        )
    return summary



_WORKER_STATE: Dict[str, Any] = {}


def _set_worker_thread_limits(num_threads: int) -> None:
    value = str(max(1, int(num_threads)))
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[name] = value


def _worker_init(
    dataset_root: str,
    camera: str,
    split: str,
    top_k: int,
    max_width: float,
    selected_pool_oracle: bool,
    threads_per_worker: int,
) -> None:
    _set_worker_thread_limits(threads_per_worker)
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "evaluator": GraspNetEval(dataset_root, camera, split=split),
            "config": get_config(),
            "table": create_table_points(
                1.0,
                1.0,
                0.05,
                dx=-0.5,
                dy=-0.5,
                dz=-0.05,
                grid_size=0.008,
            ),
            "top_k": int(top_k),
            "max_width": float(max_width),
            "selected_pool_oracle": bool(selected_pool_oracle),
        }
    )


def _scene_task(
    scene_id: int,
    anno_entries: Sequence[Tuple[int, Sequence[Tuple[str, str]]]],
) -> Dict[str, Any]:
    """Evaluate all sampled annotations/modes of one scene.

    Scene-wise task granularity is deliberate: scene CAD models and DexNet
    objects are loaded and voxelized once per worker task, rather than once per
    annotation. This is substantially faster than sample-wise multiprocessing.
    """
    if not _WORKER_STATE:
        raise RuntimeError("Worker state is not initialized.")

    evaluator = _WORKER_STATE["evaluator"]
    config = _WORKER_STATE["config"]
    table = _WORKER_STATE["table"]
    top_k = _WORKER_STATE["top_k"]
    max_width = _WORKER_STATE["max_width"]
    selected_pool_oracle = _WORKER_STATE["selected_pool_oracle"]

    model_list, dexmodel_list, _ = evaluator.get_scene_models(scene_id, ann_id=0)
    model_sampled_list = [voxel_sample_points(model, 0.008) for model in model_list]

    rows: List[Dict[str, Any]] = []
    for anno_id, mode_paths in anno_entries:
        _, pose_list, camera_pose, align_mat = evaluator.get_model_poses(
            scene_id, anno_id
        )
        table_trans = transform_points(
            table, np.linalg.inv(np.matmul(align_mat, camera_pose))
        )

        for mode, path in mode_paths:
            result = _evaluate_one_mode_sample(
                grasp_path=path,
                model_sampled_list=model_sampled_list,
                dexmodel_list=dexmodel_list,
                pose_list=pose_list,
                table_trans=table_trans,
                config=config,
                top_k=top_k,
                max_width=max_width,
                selected_pool_oracle=selected_pool_oracle,
            )
            official_acc = result.pop("official_acc")
            oracle_acc = result.pop("selected_pool_oracle_acc")
            row: Dict[str, Any] = {
                "mode": mode,
                "scene_id": int(scene_id),
                "anno_id": int(anno_id),
                "official_ap": float(official_acc.mean()),
                "official_ap04": float(official_acc[:, 1].mean()),
                "official_ap08": float(official_acc[:, 3].mean()),
                "selected_pool_oracle_ap": float(oracle_acc.mean()),
                "selected_pool_oracle_ap04": float(oracle_acc[:, 1].mean()),
                "selected_pool_oracle_ap08": float(oracle_acc[:, 3].mean()),
            }
            row.update(result)
            rows.append(row)

    return {
        "scene_id": int(scene_id),
        "num_samples": int(len(anno_entries)),
        "rows": rows,
    }


def _build_scene_tasks(
    samples: Sequence[Tuple[int, int]],
    modes: Sequence[str],
    sample_maps: Mapping[str, Mapping[Tuple[int, int], str]],
    sample_policy: str,
) -> List[Tuple[int, List[Tuple[int, List[Tuple[str, str]]]]]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for scene_id, anno_id in samples:
        grouped[int(scene_id)].append(int(anno_id))

    tasks: List[Tuple[int, List[Tuple[int, List[Tuple[str, str]]]]]] = []
    for scene_id in sorted(grouped):
        entries: List[Tuple[int, List[Tuple[str, str]]]] = []
        for anno_id in sorted(grouped[scene_id]):
            mode_paths: List[Tuple[str, str]] = []
            for mode in modes:
                path = sample_maps[mode].get((scene_id, anno_id))
                if path is None:
                    if sample_policy == "intersection":
                        raise RuntimeError(
                            f"Missing intersection sample: mode={mode}, "
                            f"scene={scene_id}, anno={anno_id}"
                        )
                    continue
                mode_paths.append((mode, path))
            if mode_paths:
                entries.append((anno_id, mode_paths))
        if entries:
            tasks.append((scene_id, entries))
    return tasks


def _progress_payload(
    *,
    args: argparse.Namespace,
    modes: Sequence[str],
    total_scenes: int,
    total_samples: int,
    completed_scenes: Sequence[int],
    completed_samples: int,
    status: str,
    elapsed_sec: float,
) -> Dict[str, Any]:
    return {
        "status": status,
        "dataset_root": os.path.abspath(args.dataset_root),
        "base_save_dir": os.path.abspath(args.base_save_dir),
        "camera": args.camera,
        "split": args.split,
        "modes": list(modes),
        "num_workers": int(args.num_workers),
        "threads_per_worker": int(args.threads_per_worker),
        "mp_start_method": args.mp_start_method,
        "total_scenes": int(total_scenes),
        "total_samples": int(total_samples),
        "completed_scenes": [int(x) for x in sorted(completed_scenes)],
        "completed_samples": int(completed_samples),
        "elapsed_sec": float(elapsed_sec),
        "updated_at_unix": float(time.time()),
    }


def _validate_resume_progress(
    progress: Mapping[str, Any],
    args: argparse.Namespace,
    modes: Sequence[str],
) -> None:
    expected = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "base_save_dir": os.path.abspath(args.base_save_dir),
        "camera": args.camera,
        "split": args.split,
        "modes": list(modes),
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(
                f"Cannot resume: progress field '{key}' differs. "
                f"saved={progress.get(key)!r}, current={value!r}"
            )

def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1.")
    if args.threads_per_worker < 1:
        raise ValueError("--threads_per_worker must be >= 1.")
    if args.top_k != 50:
        print(
            "[WARN] Current official eval_grasp uses index 49 in its global "
            "candidate cutoff; TOP_K=50 is the protocol-matched setting."
        )

    # Set limits before constructing the process pool. Spawn/forkserver workers
    # inherit these variables before importing NumPy/BLAS libraries.
    _set_worker_thread_limits(args.threads_per_worker)

    mode_roots = _load_mode_roots(args.base_save_dir, args.modes)
    modes = list(mode_roots)
    sample_maps = {
        mode: _scan_mode_samples(root, args.camera)
        for mode, root in mode_roots.items()
    }
    samples = _resolve_samples(sample_maps, args.sample_policy)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No common .npy samples found across mode directories.")
    _warn_scene_range(samples, args.split)

    output_dir = args.output_dir or os.path.join(
        args.base_save_dir, "_stagewise_eval"
    )
    os.makedirs(output_dir, exist_ok=True)

    partial_path = os.path.join(
        output_dir, "stagewise_oracle_per_sample.partial.csv"
    )
    progress_path = os.path.join(output_dir, "stagewise_oracle_progress.json")
    per_sample_path = os.path.join(output_dir, "stagewise_oracle_per_sample.csv")
    summary_path = os.path.join(output_dir, "stagewise_oracle_summary.csv")
    json_path = os.path.join(output_dir, "stagewise_oracle_summary.json")

    tasks = _build_scene_tasks(samples, modes, sample_maps, args.sample_policy)
    all_scene_ids = [scene_id for scene_id, _ in tasks]

    rows: List[Dict[str, Any]] = []
    completed_scenes: set[int] = set()
    completed_samples = 0
    if args.resume:
        if not os.path.isfile(progress_path) or not os.path.isfile(partial_path):
            raise FileNotFoundError(
                "--resume requires both progress and partial CSV files: "
                f"{progress_path}, {partial_path}"
            )
        with open(progress_path) as f:
            saved_progress = json.load(f)
        _validate_resume_progress(saved_progress, args, modes)
        completed_scenes = {
            int(x) for x in saved_progress.get("completed_scenes", [])
        }
        rows = _load_csv(partial_path)
        completed_samples = int(saved_progress.get("completed_samples", 0))
        tasks = [task for task in tasks if task[0] not in completed_scenes]
        print(
            f"[RESUME] Loaded {len(rows)} rows; "
            f"skipping {len(completed_scenes)} completed scenes."
        )
    else:
        for path in [partial_path, progress_path, per_sample_path, summary_path, json_path]:
            if os.path.isfile(path):
                os.remove(path)

    print(f"Modes: {modes}")
    print(f"Files per mode: { {m: len(v) for m, v in sample_maps.items()} }")
    print(f"Evaluation samples ({args.sample_policy}): {len(samples)}")
    print(f"Scene tasks: {len(all_scene_ids)} total, {len(tasks)} pending")
    print(
        f"CPU parallelism: {args.num_workers} process worker(s), "
        f"{args.threads_per_worker} thread(s)/worker, "
        f"start_method={args.mp_start_method}"
    )
    print(f"Progress manifest: {progress_path}")
    print(f"Incremental rows:  {partial_path}")

    start_time = time.time()
    _save_json(
        _progress_payload(
            args=args,
            modes=modes,
            total_scenes=len(all_scene_ids),
            total_samples=len(samples),
            completed_scenes=completed_scenes,
            completed_samples=completed_samples,
            status="running",
            elapsed_sec=0.0,
        ),
        progress_path,
    )

    initargs = (
        args.dataset_root,
        args.camera,
        args.split,
        args.top_k,
        args.max_width,
        not args.skip_selected_pool_oracle,
        args.threads_per_worker,
    )

    def consume_result(result: Mapping[str, Any]) -> None:
        nonlocal completed_samples
        scene_id = int(result["scene_id"])
        scene_rows = list(result["rows"])
        rows.extend(scene_rows)
        completed_scenes.add(scene_id)
        completed_samples += int(result["num_samples"])
        rows.sort(key=lambda x: (int(x["scene_id"]), int(x["anno_id"]), str(x["mode"])))

        # Checkpoint after every completed scene. This both exposes live progress
        # in output_dir and permits a deterministic scene-wise resume after failure.
        _save_csv(rows, partial_path)
        elapsed = time.time() - start_time
        _save_json(
            _progress_payload(
                args=args,
                modes=modes,
                total_scenes=len(all_scene_ids),
                total_samples=len(samples),
                completed_scenes=completed_scenes,
                completed_samples=completed_samples,
                status="running",
                elapsed_sec=elapsed,
            ),
            progress_path,
        )
        rate = completed_samples / elapsed if elapsed > 0 else 0.0
        print(
            f"[DONE] scene_{scene_id:04d}: "
            f"{len(scene_rows)} mode-sample rows; "
            f"samples {completed_samples}/{len(samples)}; "
            f"scenes {len(completed_scenes)}/{len(all_scene_ids)}; "
            f"{rate:.2f} samples/s",
            flush=True,
        )

    if tasks:
        if args.num_workers == 1:
            _worker_init(*initargs)
            for scene_id, entries in tasks:
                consume_result(_scene_task(scene_id, entries))
        else:
            ctx = mp.get_context(args.mp_start_method)
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=initargs,
            ) as executor:
                future_to_scene = {
                    executor.submit(_scene_task, scene_id, entries): scene_id
                    for scene_id, entries in tasks
                }
                try:
                    for future in as_completed(future_to_scene):
                        scene_id = future_to_scene[future]
                        try:
                            consume_result(future.result())
                        except Exception as exc:
                            for other in future_to_scene:
                                other.cancel()
                            raise RuntimeError(
                                f"Evaluation failed for scene_{scene_id:04d}"
                            ) from exc
                finally:
                    # ProcessPoolExecutor context waits for running workers; partial
                    # results from already completed scenes remain checkpointed.
                    pass

    ordered_modes = [m for m in CUMULATIVE_CHAIN if m in modes] + [
        m for m in modes if m not in CUMULATIVE_CHAIN
    ]
    summary = _summarize_rows(rows, ordered_modes)

    _save_csv(rows, per_sample_path)
    _save_csv(summary, summary_path)
    elapsed = time.time() - start_time
    _save_json(
        {
            "dataset_root": args.dataset_root,
            "camera": args.camera,
            "split": args.split,
            "top_k": args.top_k,
            "sample_policy": args.sample_policy,
            "num_samples": len(samples),
            "num_workers": args.num_workers,
            "threads_per_worker": args.threads_per_worker,
            "mp_start_method": args.mp_start_method,
            "elapsed_sec": elapsed,
            "mode_roots": mode_roots,
            "summary": summary,
            "selected_pool_oracle_definition": (
                "Official eval_grasp NMS/object-wise/global preselection, "
                "then rerank retained candidates by mean success across "
                "friction thresholds. Not a raw-candidate oracle."
            ),
        },
        json_path,
    )
    _save_json(
        _progress_payload(
            args=args,
            modes=modes,
            total_scenes=len(all_scene_ids),
            total_samples=len(samples),
            completed_scenes=completed_scenes,
            completed_samples=completed_samples,
            status="complete",
            elapsed_sec=elapsed,
        ),
        progress_path,
    )
    if os.path.isfile(partial_path):
        os.remove(partial_path)

    print("\nStage-wise oracle summary")
    for row in summary:
        print(
            f"{row['mode']:<43s} "
            f"AP={row['official_ap']:.4f} "
            f"stage={row['stage_gain_vs_previous']:+.4f} "
            f"delta={row['delta_vs_s0_base']:+.4f} "
            f"pool-oracle={row['selected_pool_oracle_ap']:.4f} "
            f"rank-gap={row['ranking_gap_selected_pool']:+.4f}"
        )
    print(f"\nElapsed: {elapsed / 60.0:.2f} min")
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_sample_path}")
    print(f"Saved: {json_path}")
    print(f"Progress: {progress_path}")


if __name__ == "__main__":
    main()
