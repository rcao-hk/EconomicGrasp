#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed one-case GraspNet evaluation diagnosis.

This script is intended to explain why two grasp prediction files can look
visually similar in a PLY viewer but obtain very different GraspNet AP.

It reproduces the key logic of GraspNetEval.eval_scene / eval_utils.eval_grasp:
  1) width clipping to [0, max_width]
  2) NMS
  3) nearest-object assignment by grasp center
  4) top-10 per object + global top-50 confidence threshold
  5) collision / empty-grasp filtering
  6) Dex-Net force-closure quality score
  7) AP computation over top-k and friction thresholds

Outputs:
  - summary.json
  - <method>_ranked_eval.csv
  - matched_pairs_topK.csv
  - ap_curves.png
  - optional <method>_eval_quality_topK.ply, colored by true eval outcome

Example:
python diagnose_graspnet_eval_case.py \
  --dataset-root /data/robotarm/dataset/graspnet \
  --camera realsense \
  --scene-id 139 --ann-id 200 \
  --method1-name baseline \
  --method1-dump-root /path/to/baseline/dump \
  --method2-name dpt_spatial \
  --method2-dump-root /path/to/dpt/dump \
  --output-dir /tmp/diag_scene139_ann200 \
  --save-eval-ply
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import fnmatch
import re
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

# Robust import for different graspnetAPI layouts.
try:
    from graspnetAPI.graspnet_eval import GraspNetEval  # type: ignore
    from graspnetAPI.grasp import GraspGroup  # type: ignore
    from graspnetAPI.utils.config import get_config  # type: ignore
    from graspnetAPI.utils.eval_utils import (  # type: ignore
        create_table_points,
        transform_points,
        voxel_sample_points,
        compute_closest_points,
        collision_detection,
        get_grasp_score,
        get_scene_name,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import graspnetAPI internals. Run this script inside the "
        "same environment where the GraspNet evaluation script works. "
        f"Original error: {repr(e)}"
    )

FRICTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
# Keep this as native Python floats, not np.float32.
# graspnetAPI.utils.eval_utils.get_grasp_score() rounds each value and uses it
# as a dictionary key. np.float32 keys can fail Python dict lookup even when
# printed as 1.2 because their hash differs from the native float key.
FC_LIST_FOR_QUALITY = [1.2, 1.0, 0.8, 0.6, 0.4, 0.2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch diagnose GraspNet evaluation results and cached end_points for one or more splits. "
            "It reuses the same detailed evaluator logic as the single-case diagnostic script, "
            "but loops over test_seen/test_similar/test_novel samples."
        )
    )
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--camera", type=str, default="realsense", choices=["realsense", "kinect"])
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test_seen", "test_similar", "test_novel"],
        choices=["test_seen", "test_similar", "test_novel", "test"],
        help="Splits to diagnose. Use any subset, e.g. --splits test_similar test_novel.",
    )

    # Cache/dump layout. The compact inference script can be run either with one
    # output root per split:
    #   <root>/<split>/<method>/{final_grasps,raw_grasps,collisions,end_points}
    # or with one shared output root for all splits:
    #   <root>/<method>/{final_grasps,raw_grasps,collisions,end_points}
    # This script supports both layouts. For your current layout, use:
    #   --cache-root-template vis/baseline_vs_dpt_spatial
    p.add_argument(
        "--cache-root-template",
        type=str,
        default="",
        help=(
            "Template for the inference output root. Supports three forms: "
            "(1) shared root: vis/baseline_vs_dpt_spatial -> <root>/<method>; "
            "(2) split root: vis/baseline_vs_dpt_spatial/{split} -> <root>/<split>/<method>; "
            "(3) method root: vis/baseline_vs_dpt_spatial/{method} -> <root>/<method>. "
            "Supports {split} and {method}."
        ),
    )
    p.add_argument(
        "--cache-layout",
        type=str,
        default="auto",
        choices=["auto", "shared", "split"],
        help=(
            "How to resolve --cache-root/--cache-root-template when it does not contain {method}. "
            "shared means <root>/<method>; split means <root>/<split>/<method> unless the template "
            "already contains {split}; auto first tries split layout if it exists, then shared layout. "
            "For vis/baseline_vs_dpt_spatial/baseline/final_grasps/scene_xxxx, use shared or auto."
        ),
    )
    p.add_argument(
        "--cache-root",
        type=str,
        default="",
        help=(
            "Fallback common cache root. In auto mode the script tries <cache-root>/{split}/<method> "
            "if it exists, then <cache-root>/<method>."
        ),
    )
    p.add_argument("--method1-name", type=str, default="baseline")
    p.add_argument("--method2-name", type=str, default="dpt_spatial")
    p.add_argument("--method1-cache-root-template", type=str, default="", help="Optional method-specific cache-root template; supports {split} and {method}.")
    p.add_argument("--method2-cache-root-template", type=str, default="", help="Optional method-specific cache-root template; supports {split} and {method}.")
    p.add_argument("--method1-dump-root-template", type=str, default="", help="Optional final-grasp dump-root template if no cache root is available; supports {split}.")
    p.add_argument("--method2-dump-root-template", type=str, default="", help="Optional final-grasp dump-root template if no cache root is available; supports {split}.")
    p.add_argument("--sample-index-template", type=str, default="", help="Optional sample_index.csv template; supports {split}.")

    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-width", type=float, default=0.1)
    p.add_argument("--voxel-size", type=float, default=0.008)
    p.add_argument("--nms-trans-th", type=float, default=0.03)
    p.add_argument("--nms-rot-deg", type=float, default=30.0)
    p.add_argument("--sample-interval", type=float, default=0.1)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--annos-per-scene", type=int, default=256)
    p.add_argument("--use-sample-index", action="store_true", default=True, help="Use sample_index.csv when it exists.")
    p.add_argument("--ignore-sample-index", action="store_true", help="Ignore sample_index.csv and regenerate scene/ann list from --sample-interval.")
    p.add_argument(
        "--no-sample-index-fallback",
        action="store_true",
        help=(
            "Do not fall back to generated scene/ann samples when a shared sample_index.csv exists "
            "but contains no rows for the current split. The default fallback is safer for shared "
            "cache roots reused across split runs."
        ),
    )
    p.add_argument("--scenes", type=str, default="", help="Optional comma-separated scene ids to keep, e.g. 139,143,165.")
    p.add_argument("--ann-ids", type=str, default="", help="Optional comma-separated original ann ids to keep, e.g. 0,10,200.")
    p.add_argument("--max-samples-per-split", type=int, default=0, help="Debug option; 0 means all selected samples.")
    p.add_argument("--skip-missing", action="store_true", default=True, help="Skip samples with missing prediction/cache files instead of raising.")
    p.add_argument("--strict-missing", action="store_true", help="Raise if any requested sample is missing.")

    # Endpoint analysis.
    p.add_argument("--analyze-endpoints", action="store_true", default=True)
    p.add_argument("--no-analyze-endpoints", action="store_true")
    p.add_argument("--endpoint-max-array-elems", type=int, default=200000)
    p.add_argument("--torch-load-weights-only", action="store_true")
    p.add_argument(
        "--important-endpoint-patterns",
        type=str,
        default=(
            "D:*,*score*,*objectness*,*graspness*,*view*,*angle*,*depth*,*width*,"
            "seed_xyz,xyz_all*,uv_all*,*top*,*select*,*sel*idx*,*token*idx*"
        ),
        help="Comma-separated fnmatch patterns used to build endpoint wide/delta tables.",
    )
    p.add_argument("--endpoint-wide-stats", type=str, default="mean,std,median,min,max,numel,finite_count")
    p.add_argument("--endpoint-max-wide-keys", type=int, default=120)
    p.add_argument("--save-endpoint-long", action="store_true", default=True)
    p.add_argument("--no-save-endpoint-long", action="store_true")

    # Grasp result analysis outputs.
    p.add_argument("--save-rank-rows", action="store_true", default=True)
    p.add_argument("--no-save-rank-rows", action="store_true")
    p.add_argument("--save-per-object-rows", action="store_true", default=True)
    p.add_argument("--no-save-per-object-rows", action="store_true")
    p.add_argument("--save-matched-pairs", action="store_true", help="Save top-K matched-pair rows for every sample. Can be large.")
    p.add_argument(
        "--save-grasp-component-rows",
        action="store_true",
        default=True,
        help=(
            "Save one row per evaluator-ranked grasp with reverse mapping back to final/raw grasp row, "
            "source token/seed index, selected view, angle/depth/score/width head statistics, "
            "and local endpoint variables when available."
        ),
    )
    p.add_argument("--no-save-grasp-component-rows", action="store_true", help="Disable grasp_component_rows.csv output.")
    p.add_argument("--component-top-k", type=int, default=0, help="Number of ranked grasps per sample to trace. 0 means use --top-k.")
    p.add_argument("--component-match-dist-scale", type=float, default=0.005, help="Center distance scale in meters used for rank->final/raw grasp matching.")
    p.add_argument("--component-match-rot-scale-deg", type=float, default=5.0, help="Rotation angle scale in degrees used for rank->final/raw grasp matching.")
    p.add_argument("--component-match-score-scale", type=float, default=0.05, help="Predicted score scale used for grasp matching.")
    p.add_argument("--component-match-width-scale", type=float, default=0.005, help="Width scale in meters used for grasp matching.")
    p.add_argument("--component-match-depth-scale", type=float, default=0.005, help="Depth scale in meters used for grasp matching.")
    p.add_argument("--match-top-k", type=int, default=50)
    p.add_argument("--match-dist-th", type=float, default=0.02)
    p.add_argument("--match-rot-deg-th", type=float, default=20.0)
    p.add_argument("--save-aggregate-ap-curves", action="store_true", default=True)
    p.add_argument("--no-save-aggregate-ap-curves", action="store_true")

    # Parallel scene-level execution. Each worker handles exactly one scene task
    # at a time, reusing loaded object models across annos within that scene.
    p.add_argument("--num-scene-workers", type=int, default=4, help="Number of parallel scene workers. Each worker processes one scene task.")
    p.add_argument("--parallel-backend", type=str, default="process", choices=["process", "thread", "none"], help="process is recommended for Dex-Net/Open3D evaluation; thread is available for debugging.")
    p.add_argument("--scene-output-dir", type=str, default="scene_outputs", help="Subdirectory under output-dir for per-scene incremental outputs.")
    p.add_argument("--no-scene-output-files", action="store_true", help="Do not write per-scene intermediate CSV files.")
    p.add_argument("--progress-every-scenes", type=int, default=1, help="Print progress every N completed scene tasks.")

    return p.parse_args()


def npy_path(dump_root: str, direct_npy: str, scene_id: int, ann_id: int, camera: str) -> Path:
    if direct_npy:
        return Path(direct_npy)
    root = Path(dump_root)
    scene = f"scene_{scene_id:04d}"
    candidates = [
        root / scene / camera / f"{ann_id:04d}.npy",
        root / scene / camera / f"grasp_{ann_id:04d}.npy",
        root / scene / camera / f"pred_{ann_id:04d}.npy",
        root / scene / f"{ann_id:04d}.npy",
        root / f"{scene}_{camera}_{ann_id:04d}.npy",
        root / f"{scene}_{ann_id:04d}.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not resolve grasp npy. Tried:\n" + "\n".join(str(x) for x in candidates))


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def summarize_array(x: np.ndarray, prefix: str) -> Dict[str, Any]:
    x = np.asarray(x)
    out: Dict[str, Any] = {}
    if x.size == 0:
        out[f"{prefix}_count"] = 0
        return out
    out[f"{prefix}_count"] = int(x.size)
    out[f"{prefix}_min"] = safe_float(np.nanmin(x))
    out[f"{prefix}_p25"] = safe_float(np.nanpercentile(x, 25))
    out[f"{prefix}_median"] = safe_float(np.nanmedian(x))
    out[f"{prefix}_mean"] = safe_float(np.nanmean(x))
    out[f"{prefix}_p75"] = safe_float(np.nanpercentile(x, 75))
    out[f"{prefix}_max"] = safe_float(np.nanmax(x))
    return out


def clip_width_like_eval(gg: Any, max_width: float) -> Dict[str, Any]:
    arr = gg.grasp_group_array.copy()
    widths = arr[:, 1] if arr.size else np.array([], dtype=np.float32)
    raw_stats = {
        "num_raw_grasps": int(arr.shape[0]),
        "num_width_lt_0": int(np.sum(widths < 0)) if widths.size else 0,
        "num_width_gt_max": int(np.sum(widths > max_width)) if widths.size else 0,
    }
    raw_stats.update(summarize_array(arr[:, 0] if arr.size else np.array([]), "raw_conf"))
    raw_stats.update(summarize_array(widths, "raw_width"))
    raw_stats.update(summarize_array(arr[:, 3] if arr.size else np.array([]), "raw_depth"))
    if arr.size:
        arr[arr[:, 1] < 0, 1] = 0.0
        arr[arr[:, 1] > max_width, 1] = max_width
    gg.grasp_group_array = arr
    return raw_stats


def build_force_closure_quality_config(config: Dict[str, Any]) -> Dict[float, Any]:
    # Local import here to keep imports close to the original eval_utils implementation.
    try:
        from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory  # type: ignore
    except Exception:
        # Some installations expose the same package through another relative path only;
        # in that case importing get_grasp_score usually means the factory is importable
        # from the same absolute path used above. Re-raise if it fails.
        raise

    out: Dict[float, Any] = {}
    for value_fc in FC_LIST_FOR_QUALITY:
        value_fc = round(float(value_fc), 2)
        config["metrics"]["force_closure"]["friction_coef"] = value_fc
        out[value_fc] = GraspQualityConfigFactory.create_config(config["metrics"]["force_closure"])
    return out


def eval_grasp_detailed(
    grasp_group: Any,
    models: Sequence[np.ndarray],
    dexnet_models: Sequence[Any],
    poses: Sequence[np.ndarray],
    config: Dict[str, Any],
    table: Optional[np.ndarray],
    voxel_size: float,
    top_k: int,
    nms_trans_th: float,
    nms_rot_rad: float,
) -> Dict[str, Any]:
    """Reimplementation of eval_utils.eval_grasp with extra introspection."""
    num_models = len(models)
    stats: Dict[str, Any] = {}

    # NMS exactly as GraspNet evaluation, except thresholds are exposed.
    gg_nms = grasp_group.nms(nms_trans_th, nms_rot_rad)
    stats["num_after_nms"] = int(len(gg_nms))

    # Transform model points into camera frame and build object segmentation mask.
    model_trans_list: List[np.ndarray] = []
    seg_mask: List[np.ndarray] = []
    for i, model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        model_trans_list.append(model_trans)
        seg_mask.append(i * np.ones(model_trans.shape[0], dtype=np.int32))

    if len(model_trans_list) == 0 or len(gg_nms) == 0:
        return {
            "stats": stats,
            "ranked_grasps": np.zeros((0, 17), dtype=np.float32),
            "ranked_obj_ids": np.zeros((0,), dtype=np.int32),
            "ranked_eval_scores": np.zeros((0,), dtype=np.float32),
            "ranked_collision": np.zeros((0,), dtype=bool),
            "ranked_empty": np.zeros((0,), dtype=bool),
            "ap_matrix": np.zeros((top_k, len(FRICTIONS)), dtype=np.float32),
        }

    scene = np.concatenate(model_trans_list, axis=0)
    seg = np.concatenate(seg_mask, axis=0)

    # Assign grasps to nearest object model point.
    assigned_indices = compute_closest_points(gg_nms.translations, scene)
    model_to_grasp = seg[assigned_indices]
    obj_assignment_counts = {str(i): int(np.sum(model_to_grasp == i)) for i in range(num_models)}
    stats["obj_assignment_counts_after_nms"] = obj_assignment_counts

    # Per-object top-10 and global top-K confidence threshold, matching eval_utils.eval_grasp.
    pre_grasp_list: List[np.ndarray] = []
    for i in range(num_models):
        grasp_i = gg_nms[model_to_grasp == i]
        grasp_i.sort_by_score()
        arr_i = grasp_i[:10].grasp_group_array
        arr_i = np.asarray(arr_i, dtype=np.float32).reshape((-1, 17)) if arr_i.size else np.zeros((0, 17), dtype=np.float32)
        pre_grasp_list.append(arr_i)

    stats["per_object_pre_top10_counts"] = {str(i): int(x.shape[0]) for i, x in enumerate(pre_grasp_list)}
    nonempty = [x for x in pre_grasp_list if x.shape[0] > 0]
    if len(nonempty) == 0:
        return {
            "stats": stats,
            "ranked_grasps": np.zeros((0, 17), dtype=np.float32),
            "ranked_obj_ids": np.zeros((0,), dtype=np.int32),
            "ranked_eval_scores": np.zeros((0,), dtype=np.float32),
            "ranked_collision": np.zeros((0,), dtype=bool),
            "ranked_empty": np.zeros((0,), dtype=bool),
            "ap_matrix": np.zeros((top_k, len(FRICTIONS)), dtype=np.float32),
        }

    all_pre = np.vstack(nonempty)
    remain_order = np.argsort(all_pre[:, 0])[::-1]
    min_score = float(all_pre[remain_order[min(top_k - 1, len(remain_order) - 1)], 0])
    stats["global_topk_min_confidence_after_per_object_top10"] = min_score

    grasp_list_by_obj: List[np.ndarray] = []
    obj_ids_by_obj: List[np.ndarray] = []
    for i in range(num_models):
        arr_i = pre_grasp_list[i]
        if arr_i.shape[0] == 0:
            grasp_list_by_obj.append(arr_i)
            obj_ids_by_obj.append(np.zeros((0,), dtype=np.int32))
            continue
        keep = arr_i[:, 0] >= min_score
        kept = arr_i[keep]
        grasp_list_by_obj.append(kept)
        obj_ids_by_obj.append(i * np.ones((kept.shape[0],), dtype=np.int32))

    stats["per_object_selected_counts"] = {str(i): int(x.shape[0]) for i, x in enumerate(grasp_list_by_obj)}

    # Collision detection. The original evaluation appends table points to object model points.
    scene_for_collision = scene
    if table is not None:
        scene_for_collision = np.concatenate([scene_for_collision, table], axis=0)

    collision_mask_list, empty_mask_list, dexgrasp_list = collision_detection(
        grasp_list_by_obj,
        model_trans_list,
        dexnet_models,
        poses,
        scene_for_collision,
        outlier=0.05,
        return_dexgrasps=True,
    )

    fc_config = build_force_closure_quality_config(config)
    eval_scores_by_obj: List[np.ndarray] = []
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = np.asarray(collision_mask_list[i], dtype=bool)
        dexgrasps = dexgrasp_list[i]
        scores: List[float] = []
        for grasp_id in range(len(dexgrasps)):
            if collision_mask[grasp_id] or dexgrasps[grasp_id] is None:
                scores.append(-1.0)
            else:
                scores.append(float(get_grasp_score(dexgrasps[grasp_id], dexnet_model, FC_LIST_FOR_QUALITY, fc_config)))
        eval_scores_by_obj.append(np.asarray(scores, dtype=np.float32))

    if len([x for x in grasp_list_by_obj if x.shape[0] > 0]) == 0:
        all_grasps = np.zeros((0, 17), dtype=np.float32)
        all_obj_ids = np.zeros((0,), dtype=np.int32)
        all_scores = np.zeros((0,), dtype=np.float32)
        all_collision = np.zeros((0,), dtype=bool)
        all_empty = np.zeros((0,), dtype=bool)
    else:
        all_grasps = np.vstack([x for x in grasp_list_by_obj if x.shape[0] > 0]).astype(np.float32)
        all_obj_ids = np.concatenate([x for x in obj_ids_by_obj if x.shape[0] > 0]).astype(np.int32)
        all_scores = np.concatenate([x for x in eval_scores_by_obj if x.shape[0] > 0]).astype(np.float32)
        all_collision = np.concatenate([np.asarray(x, dtype=bool) for x in collision_mask_list if len(x) > 0]).astype(bool)
        all_empty = np.concatenate([np.asarray(x, dtype=bool) for x in empty_mask_list if len(x) > 0]).astype(bool)

    # Scene-level sorting by predicted confidence, exactly as eval_scene.
    if all_grasps.shape[0] > 0:
        order = np.argsort(-all_grasps[:, 0])
        all_grasps = all_grasps[order]
        all_obj_ids = all_obj_ids[order]
        all_scores = all_scores[order]
        all_collision = all_collision[order]
        all_empty = all_empty[order]

    ap_matrix = compute_ap_matrix(all_scores, top_k=top_k)
    stats.update(summarize_ranked(all_scores, all_collision, all_empty, all_grasps, top_k))

    return {
        "stats": stats,
        "ranked_grasps": all_grasps,
        "ranked_obj_ids": all_obj_ids,
        "ranked_eval_scores": all_scores,
        "ranked_collision": all_collision,
        "ranked_empty": all_empty,
        "ap_matrix": ap_matrix,
    }


def compute_ap_matrix(scores: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    acc = np.zeros((top_k, len(FRICTIONS)), dtype=np.float32)
    for fric_idx, fric in enumerate(FRICTIONS):
        for k in range(top_k):
            cur = scores[: min(k + 1, len(scores))]
            ok = ((cur <= fric + 1e-6) & (cur > 0)).astype(np.int32)
            acc[k, fric_idx] = float(np.sum(ok)) / float(k + 1)
    return acc


def summarize_ranked(
    scores: np.ndarray,
    collision: np.ndarray,
    empty: np.ndarray,
    grasps: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = int(min(top_k, len(scores)))
    out["num_eval_selected"] = int(len(scores))
    out["num_eval_topk_available"] = n
    if len(scores) == 0:
        out.update({"AP": 0.0, "AP0.4": 0.0, "AP0.8": 0.0})
        return out
    ap = compute_ap_matrix(scores, top_k)
    out["AP"] = float(np.mean(ap))
    out["AP0.4"] = float(np.mean(ap[:, FRICTIONS.index(0.4)]))
    out["AP0.8"] = float(np.mean(ap[:, FRICTIONS.index(0.8)]))
    for kk in [1, 5, 10, 20, 50]:
        m = min(kk, len(scores))
        if m <= 0:
            continue
        out[f"top{kk}_collision_rate"] = float(np.mean(collision[:m]))
        out[f"top{kk}_empty_rate"] = float(np.mean(empty[:m]))
        out[f"top{kk}_success_rate_fric04"] = float(np.mean((scores[:m] <= 0.4 + 1e-6) & (scores[:m] > 0)))
        out[f"top{kk}_success_rate_fric08"] = float(np.mean((scores[:m] <= 0.8 + 1e-6) & (scores[:m] > 0)))
        out[f"top{kk}_success_rate_fric12"] = float(np.mean((scores[:m] <= 1.2 + 1e-6) & (scores[:m] > 0)))
    out.update(summarize_array(scores[:n], "eval_score_topk"))
    out.update(summarize_array(grasps[:n, 0] if grasps.size else np.array([]), "pred_conf_topk"))
    out.update(summarize_array(grasps[:n, 1] if grasps.size else np.array([]), "width_topk"))
    out.update(summarize_array(grasps[:n, 3] if grasps.size else np.array([]), "depth_topk"))
    return out


def make_ranked_rows(result: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    grasps = result["ranked_grasps"]
    obj_ids = result["ranked_obj_ids"]
    scores = result["ranked_eval_scores"]
    collision = result["ranked_collision"]
    empty = result["ranked_empty"]
    rows: List[Dict[str, Any]] = []
    max_n = min(top_k, len(scores))
    cum_success = {f: 0 for f in FRICTIONS}
    for i in range(max_n):
        s = float(scores[i])
        row: Dict[str, Any] = {
            "rank": i + 1,
            "pred_conf": float(grasps[i, 0]),
            "width": float(grasps[i, 1]),
            "depth": float(grasps[i, 3]),
            "tx": float(grasps[i, 13]),
            "ty": float(grasps[i, 14]),
            "tz": float(grasps[i, 15]),
            "assigned_obj_local_id": int(obj_ids[i]),
            "eval_friction_score": s,
            "collision_or_empty": bool(collision[i]),
            "empty": bool(empty[i]),
        }
        for f in FRICTIONS:
            ok = bool((s <= f + 1e-6) and (s > 0))
            cum_success[f] += int(ok)
            row[f"success_fric_{f:.1f}"] = ok
            row[f"cum_precision_fric_{f:.1f}"] = float(cum_success[f]) / float(i + 1)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def rotation_angle_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    R = np.matmul(R1.T, R2)
    c = (float(np.trace(R)) - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(np.arccos(c))


def match_topk(result1: Dict[str, Any], result2: Dict[str, Any], top_k: int, dist_th: float, rot_th_rad: float) -> List[Dict[str, Any]]:
    g1 = result1["ranked_grasps"][:top_k]
    g2 = result2["ranked_grasps"][:top_k]
    s1 = result1["ranked_eval_scores"][:top_k]
    s2 = result2["ranked_eval_scores"][:top_k]
    c1 = result1["ranked_collision"][:top_k]
    c2 = result2["ranked_collision"][:top_k]
    rows: List[Dict[str, Any]] = []
    used2 = set()
    for i in range(len(g1)):
        center1 = g1[i, 13:16]
        R1 = g1[i, 4:13].reshape(3, 3)
        best_j = None
        best_key = None
        best_dist = None
        best_rot = None
        for j in range(len(g2)):
            if j in used2:
                continue
            center2 = g2[j, 13:16]
            dist = float(np.linalg.norm(center1 - center2))
            if dist > dist_th:
                continue
            R2 = g2[j, 4:13].reshape(3, 3)
            rot = rotation_angle_rad(R1, R2)
            if rot > rot_th_rad:
                continue
            key = (dist, rot)
            if best_key is None or key < best_key:
                best_key = key
                best_j = j
                best_dist = dist
                best_rot = rot
        if best_j is not None:
            used2.add(best_j)
            rows.append({
                "method1_rank": i + 1,
                "method2_rank": best_j + 1,
                "center_dist_m": best_dist,
                "rot_angle_deg": float(best_rot * 180.0 / np.pi),
                "method1_pred_conf": float(g1[i, 0]),
                "method2_pred_conf": float(g2[best_j, 0]),
                "method1_eval_friction_score": float(s1[i]),
                "method2_eval_friction_score": float(s2[best_j]),
                "method1_collision_or_empty": bool(c1[i]),
                "method2_collision_or_empty": bool(c2[best_j]),
                "delta_rank_method2_minus_method1": int(best_j + 1 - (i + 1)),
                "delta_pred_conf_method2_minus_method1": float(g2[best_j, 0] - g1[i, 0]),
            })
    return rows


def quality_color(score: float, collision: bool) -> Tuple[float, float, float]:
    if collision or score <= 0:
        return (0.25, 0.25, 0.25)  # collision / empty / failed
    if score <= 0.4:
        return (0.0, 0.85, 0.0)    # strong grasp
    if score <= 0.8:
        return (1.0, 0.8, 0.0)     # medium grasp
    return (1.0, 0.35, 0.0)        # weak but force-closure at high friction


def save_eval_quality_ply(path: Path, result: Dict[str, Any], top_k: int, sample_points: int) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required for --save-eval-ply")
    grasps = result["ranked_grasps"][:top_k]
    scores = result["ranked_eval_scores"][:top_k]
    collision = result["ranked_collision"][:top_k]
    if len(grasps) == 0:
        return
    gg = GraspGroup(grasps.copy())
    geoms = gg.to_open3d_geometry_list()
    combo = o3d.geometry.PointCloud()
    for geom, score, col in zip(geoms, scores, collision):
        if hasattr(geom, "sample_points_uniformly"):
            sampled = geom.sample_points_uniformly(number_of_points=sample_points)
        else:
            sampled = geom
        sampled.paint_uniform_color(quality_color(float(score), bool(col)))
        combo += sampled
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), combo)


def plot_ap_curves(out_path: Path, res1: Dict[str, Any], res2: Dict[str, Any], name1: str, name2: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ks = np.arange(1, res1["ap_matrix"].shape[0] + 1)
    plt.figure(figsize=(9, 5))
    for fric in [0.4, 0.8, 1.2]:
        idx = FRICTIONS.index(fric)
        plt.plot(ks, res1["ap_matrix"][:, idx], label=f"{name1} fric<={fric}")
        plt.plot(ks, res2["ap_matrix"][:, idx], linestyle="--", label=f"{name2} fric<={fric}")
    plt.xlabel("Top-k rank")
    plt.ylabel("Cumulative precision")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()



# -----------------------------------------------------------------------------
# Cached inference artifact / endpoint analysis helpers
# -----------------------------------------------------------------------------
def resolve_cache_file(cache_root: str, subdir: str, suffix: str, scene_id: int, ann_id: int, camera: str) -> Optional[Path]:
    if not cache_root:
        return None
    root = Path(cache_root)
    scene = f"scene_{scene_id:04d}"
    candidates = [
        root / subdir / scene / camera / f"{ann_id:04d}{suffix}",
        root / scene / camera / f"{ann_id:04d}{suffix}",
        root / subdir / scene / f"{ann_id:04d}{suffix}",
        root / f"{scene}_{camera}_{ann_id:04d}{suffix}",
        root / f"{scene}_{ann_id:04d}{suffix}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_method_artifacts(
    cache_root: str,
    dump_root: str,
    direct_npy: str,
    direct_raw_npy: str,
    direct_endpoint: str,
    direct_collision_npz: str,
    scene_id: int,
    ann_id: int,
    camera: str,
) -> Dict[str, Optional[Path]]:
    final_npy: Optional[Path] = Path(direct_npy) if direct_npy else None
    if final_npy is None or not final_npy.exists():
        cached = resolve_cache_file(cache_root, "final_grasps", ".npy", scene_id, ann_id, camera)
        if cached is not None:
            final_npy = cached
    if (final_npy is None or not final_npy.exists()) and dump_root:
        final_npy = npy_path(dump_root, "", scene_id, ann_id, camera)

    raw_npy: Optional[Path] = Path(direct_raw_npy) if direct_raw_npy else None
    if raw_npy is None or not raw_npy.exists():
        raw_npy = resolve_cache_file(cache_root, "raw_grasps", ".npy", scene_id, ann_id, camera)

    endpoint: Optional[Path] = Path(direct_endpoint) if direct_endpoint else None
    if endpoint is None or not endpoint.exists():
        endpoint = resolve_cache_file(cache_root, "end_points", ".pth", scene_id, ann_id, camera)
    if endpoint is None or not endpoint.exists():
        endpoint = resolve_cache_file(cache_root, "end_points", ".pth.gz", scene_id, ann_id, camera)

    collision_npz: Optional[Path] = Path(direct_collision_npz) if direct_collision_npz else None
    if collision_npz is None or not collision_npz.exists():
        collision_npz = resolve_cache_file(cache_root, "collisions", ".npz", scene_id, ann_id, camera)

    if final_npy is None or not final_npy.exists():
        raise FileNotFoundError(
            f"Could not find final grasp npy for scene_{scene_id:04d}/{camera}/{ann_id:04d}. "
            f"Use --methodX-cache-root, --methodX-dump-root, or --methodX-npy."
        )
    return {
        "final_npy": final_npy,
        "raw_npy": raw_npy if raw_npy is not None and raw_npy.exists() else None,
        "endpoint": endpoint if endpoint is not None and endpoint.exists() else None,
        "collision_npz": collision_npz if collision_npz is not None and collision_npz.exists() else None,
    }


def load_endpoint_payload(path: Path, weights_only: bool = False) -> Dict[str, Any]:
    import torch

    def _load_from_fileobj(fobj):
        try:
            return torch.load(fobj, map_location="cpu", weights_only=weights_only)
        except TypeError:
            return torch.load(fobj, map_location="cpu")

    if str(path).endswith(".gz"):
        with gzip.open(str(path), "rb") as f:
            payload = _load_from_fileobj(f)
    else:
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=weights_only)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and "end_points" in payload:
        return payload
    if isinstance(payload, dict):
        return {"meta": {}, "end_points": payload}
    return {"meta": {}, "end_points": {"payload": payload}}


def tensor_to_numpy_small(x: Any, max_elems: int = 200000) -> Optional[np.ndarray]:
    try:
        import torch
        if torch.is_tensor(x):
            if x.numel() > max_elems:
                return x.detach().cpu().reshape(-1)[:max_elems].numpy()
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        if x.size > max_elems:
            return x.reshape(-1)[:max_elems]
        return x
    return None


def endpoint_value_summary(key: str, value: Any, max_elems: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {"key": key, "type": type(value).__name__}
    try:
        import torch
        if torch.is_tensor(value):
            row.update({
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            })
            arr = tensor_to_numpy_small(value, max_elems=max_elems)
        elif isinstance(value, np.ndarray):
            row.update({
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.size),
            })
            arr = tensor_to_numpy_small(value, max_elems=max_elems)
        elif isinstance(value, (list, tuple)):
            row.update({"len": int(len(value))})
            arr = None
        elif isinstance(value, dict):
            row.update({"len": int(len(value)), "subkeys": ";".join(list(map(str, value.keys()))[:20])})
            arr = None
        else:
            row.update({"repr": repr(value)[:300]})
            arr = None
        if arr is not None:
            arr = np.asarray(arr)
            finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else np.zeros_like(arr, dtype=bool)
            row["sampled_elems_for_stats"] = int(arr.size)
            row["finite_count"] = int(np.sum(finite))
            if np.any(finite):
                a = arr[finite].astype(np.float64)
                row.update({
                    "min": float(np.min(a)),
                    "p01": float(np.percentile(a, 1)),
                    "p25": float(np.percentile(a, 25)),
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a)),
                    "median": float(np.median(a)),
                    "p75": float(np.percentile(a, 75)),
                    "p99": float(np.percentile(a, 99)),
                    "max": float(np.max(a)),
                })
    except Exception as e:
        row["summary_error"] = repr(e)
    return row


def summarize_endpoint_payload(payload: Dict[str, Any], max_elems: int) -> List[Dict[str, Any]]:
    ep = payload.get("end_points", {})
    rows: List[Dict[str, Any]] = []
    if not isinstance(ep, dict):
        return [{"key": "<end_points>", "type": type(ep).__name__, "repr": repr(ep)[:300]}]
    for key in sorted(ep.keys()):
        rows.append(endpoint_value_summary(str(key), ep[key], max_elems=max_elems))
    return rows


def compare_endpoint_payloads(payload1: Dict[str, Any], payload2: Dict[str, Any], max_elems: int) -> List[Dict[str, Any]]:
    ep1 = payload1.get("end_points", {})
    ep2 = payload2.get("end_points", {})
    if not isinstance(ep1, dict) or not isinstance(ep2, dict):
        return []
    rows: List[Dict[str, Any]] = []
    common = sorted(set(ep1.keys()) & set(ep2.keys()))
    for key in common:
        a = tensor_to_numpy_small(ep1[key], max_elems=max_elems)
        b = tensor_to_numpy_small(ep2[key], max_elems=max_elems)
        row: Dict[str, Any] = {"key": str(key), "method1_type": type(ep1[key]).__name__, "method2_type": type(ep2[key]).__name__}
        if a is None or b is None:
            row["comparable"] = False
            rows.append(row)
            continue
        row["method1_shape"] = list(np.asarray(a).shape)
        row["method2_shape"] = list(np.asarray(b).shape)
        if np.asarray(a).shape != np.asarray(b).shape:
            row["comparable"] = False
            rows.append(row)
            continue
        try:
            af = np.asarray(a, dtype=np.float64).reshape(-1)
            bf = np.asarray(b, dtype=np.float64).reshape(-1)
            mask = np.isfinite(af) & np.isfinite(bf)
            row["comparable"] = True
            row["finite_count"] = int(np.sum(mask))
            if np.any(mask):
                d = bf[mask] - af[mask]
                row.update({
                    "mean_abs_diff_m2_minus_m1": float(np.mean(np.abs(d))),
                    "median_abs_diff_m2_minus_m1": float(np.median(np.abs(d))),
                    "max_abs_diff_m2_minus_m1": float(np.max(np.abs(d))),
                    "mean_signed_diff_m2_minus_m1": float(np.mean(d)),
                    "std_signed_diff_m2_minus_m1": float(np.std(d)),
                })
                if np.std(af[mask]) > 1e-12 and np.std(bf[mask]) > 1e-12:
                    row["pearson_corr"] = float(np.corrcoef(af[mask], bf[mask])[0, 1])
        except Exception as e:
            row["comparable"] = False
            row["compare_error"] = repr(e)
        rows.append(row)
    return rows


def load_raw_collision_summary(artifacts: Dict[str, Optional[Path]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if artifacts.get("raw_npy") is not None:
        try:
            gg = GraspGroup().from_npy(str(artifacts["raw_npy"]))
            arr = np.asarray(gg.grasp_group_array, dtype=np.float32)
            out["raw_npy_path"] = str(artifacts["raw_npy"])
            out["raw_grasp_count"] = int(arr.shape[0])
            if arr.shape[0] > 0:
                out.update(summarize_array(arr[:, 0], "raw_pred_conf"))
                out.update(summarize_array(arr[:, 1], "raw_width"))
                out.update(summarize_array(arr[:, 3], "raw_depth"))
        except Exception as e:
            out["raw_npy_error"] = repr(e)
    if artifacts.get("collision_npz") is not None:
        try:
            z = np.load(str(artifacts["collision_npz"]), allow_pickle=True)
            out["collision_npz_path"] = str(artifacts["collision_npz"])
            if "collision_mask" in z:
                mask = np.asarray(z["collision_mask"], dtype=bool)
                out["model_free_collision_count"] = int(np.sum(mask))
                out["model_free_collision_rate"] = float(np.mean(mask)) if mask.size else 0.0
            for key in ["raw_count", "final_count", "collision_count", "collision_thresh", "collision_voxel_size", "approach_dist"]:
                if key in z:
                    val = np.asarray(z[key]).reshape(-1)
                    if val.size == 1:
                        out[key] = float(val[0]) if np.issubdtype(val.dtype, np.number) else str(val[0])
        except Exception as e:
            out["collision_npz_error"] = repr(e)
    return out

# -----------------------------------------------------------------------------
# Reverse mapping: evaluator-ranked grasp -> raw decoded grasp -> token/seed/head variables
# -----------------------------------------------------------------------------
def _num_array(x: Any) -> Optional[np.ndarray]:
    if x is None or isinstance(x, str):
        return None
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        a = np.asarray(x)
        return None if a.dtype == object else a
    except Exception:
        return None


def _ep(payload: Optional[Dict[str, Any]], key: str) -> Optional[np.ndarray]:
    if not payload:
        return None
    ep = payload.get("end_points", {})
    if not isinstance(ep, dict) or key not in ep:
        return None
    return _num_array(ep.get(key))


def _flat_at(arr: Optional[np.ndarray], idx: Optional[int]) -> Optional[float]:
    if arr is None or idx is None:
        return None
    try:
        f = np.asarray(arr).reshape(-1)
        i = int(idx)
        if i < 0 or i >= f.size:
            return None
        v = float(f[i])
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _vec_at(arr: Optional[np.ndarray], idx: Optional[int], channel_first: bool = True) -> Optional[np.ndarray]:
    if arr is None or idx is None:
        return None
    try:
        a = np.asarray(arr)
        i = int(idx)
        if a.ndim == 0:
            return np.asarray([float(a)], dtype=np.float64)
        if a.ndim == 1:
            return np.asarray([a[i]], dtype=np.float64) if 0 <= i < a.shape[0] else None
        if a.ndim == 2:
            if channel_first and a.shape[0] <= 64 and 0 <= i < a.shape[1]:
                return np.asarray(a[:, i], dtype=np.float64)
            if 0 <= i < a.shape[0]:
                return np.asarray(a[i, :], dtype=np.float64)
            if 0 <= i < a.shape[1]:
                return np.asarray(a[:, i], dtype=np.float64)
        if a.ndim >= 3:
            # For [1,H,W] / [C,H,W], use flattened spatial index.
            if a.shape[0] <= 64:
                flat = a.reshape(a.shape[0], -1)
                if 0 <= i < flat.shape[1]:
                    return np.asarray(flat[:, i], dtype=np.float64)
            f = a.reshape(-1)
            return np.asarray([f[i]], dtype=np.float64) if 0 <= i < f.size else None
    except Exception:
        return None
    return None


def _softmax(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        x = np.asarray(v, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return None
        x = x - np.nanmax(x)
        e = np.exp(x)
        s = np.nansum(e)
        return e / s if np.isfinite(s) and s > 0 else None
    except Exception:
        return None


def _entropy(p: Optional[np.ndarray]) -> Optional[float]:
    if p is None:
        return None
    q = np.asarray(p, dtype=np.float64)
    q = q[np.isfinite(q) & (q > 0)]
    return float(-np.sum(q * np.log(q + 1e-12))) if q.size else None


def _add_head(row: Dict[str, Any], prefix: str, vec: Optional[np.ndarray], selected: Optional[int] = None, values: Optional[Sequence[float]] = None) -> None:
    if vec is None:
        return
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return
    row[f"{prefix}_num_channels"] = int(v.size)
    row[f"{prefix}_argmax"] = int(np.nanargmax(v))
    row[f"{prefix}_max_logit"] = float(np.nanmax(v))
    row[f"{prefix}_logit_mean"] = float(np.nanmean(v))
    row[f"{prefix}_logit_std"] = float(np.nanstd(v))
    p = _softmax(v)
    if p is not None:
        row[f"{prefix}_maxprob"] = float(np.nanmax(p))
        row[f"{prefix}_prob_argmax"] = int(np.nanargmax(p))
        row[f"{prefix}_entropy"] = _entropy(p)
        if values is not None and len(values) == len(p):
            row[f"{prefix}_expected"] = float(np.sum(np.asarray(values, dtype=np.float64) * p))
    if selected is not None and 0 <= int(selected) < v.size:
        si = int(selected)
        row[f"{prefix}_selected_idx"] = si
        row[f"{prefix}_selected_logit"] = float(v[si])
        if p is not None:
            row[f"{prefix}_selected_prob"] = float(p[si])


def _rot_deg(R_ref: np.ndarray, R_cand: np.ndarray) -> np.ndarray:
    R0 = np.asarray(R_ref, dtype=np.float64).reshape(3, 3)
    Rc = np.asarray(R_cand, dtype=np.float64).reshape((-1, 3, 3))
    rel = np.matmul(np.transpose(Rc, (0, 2, 1)), R0[None])
    tr = np.einsum("nii->n", rel)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _match_grasp(q: np.ndarray, cand: np.ndarray, args: argparse.Namespace, max_width: Optional[float]) -> Tuple[int, Dict[str, Any]]:
    if cand is None or len(cand) == 0:
        return -1, {"match_error": "empty_candidates"}
    q0 = np.asarray(q, dtype=np.float64).copy().reshape(-1)
    c0 = np.asarray(cand, dtype=np.float64).copy()
    if max_width is not None:
        q0[1] = np.clip(q0[1], 0.0, float(max_width))
        c0[:, 1] = np.clip(c0[:, 1], 0.0, float(max_width))
    center = np.linalg.norm(c0[:, 13:16] - q0[13:16][None], axis=1)
    rot = _rot_deg(q0[4:13], c0[:, 4:13])
    score = np.abs(c0[:, 0] - q0[0])
    width = np.abs(c0[:, 1] - q0[1])
    depth = np.abs(c0[:, 3] - q0[3])
    cost = center / max(args.component_match_dist_scale, 1e-9) + rot / max(args.component_match_rot_scale_deg, 1e-9) + score / max(args.component_match_score_scale, 1e-9) + width / max(args.component_match_width_scale, 1e-9) + depth / max(args.component_match_depth_scale, 1e-9)
    j = int(np.nanargmin(cost))
    return j, {
        "match_cost": float(cost[j]),
        "match_center_dist_m": float(center[j]),
        "match_rot_deg": float(rot[j]),
        "match_score_abs": float(score[j]),
        "match_width_abs": float(width[j]),
        "match_depth_abs": float(depth[j]),
    }


def _load_grasps(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists():
        return None
    try:
        gg = GraspGroup().from_npy(str(path))
        return np.asarray(gg.grasp_group_array, dtype=np.float32).reshape((-1, 17))
    except Exception:
        return None


def _load_collision_arrays(artifacts: Dict[str, Optional[Path]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if artifacts.get("collision_npz") is not None:
        try:
            z = np.load(str(artifacts["collision_npz"]), allow_pickle=True)
            for k in ["raw_grasps", "final_grasps", "collision_mask"]:
                if k in z:
                    out[k] = np.asarray(z[k])
        except Exception as e:
            out["collision_npz_load_error"] = repr(e)
    if "raw_grasps" not in out:
        r = _load_grasps(artifacts.get("raw_npy"))
        if r is not None:
            out["raw_grasps"] = r
    if "final_grasps" not in out:
        f = _load_grasps(artifacts.get("final_npy"))
        if f is not None:
            out["final_grasps"] = f
    if "collision_mask" in out:
        out["collision_mask"] = np.asarray(out["collision_mask"], dtype=bool).reshape(-1)
    return out


def _final_to_raw_map(final_arr: np.ndarray, raw_arr: Optional[np.ndarray], mask: Optional[np.ndarray], args: argparse.Namespace) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
    fmap: Dict[int, int] = {}
    fmet: Dict[int, Dict[str, Any]] = {}
    if raw_arr is None or final_arr is None or len(raw_arr) == 0 or len(final_arr) == 0:
        return fmap, fmet
    if mask is not None and len(mask) == len(raw_arr):
        keep = np.flatnonzero(~mask)
        exp = raw_arr[keep]
        if exp.shape == final_arr.shape:
            diff = float(np.nanmax(np.abs(exp[:, [0, 1, 3, 13, 14, 15]] - final_arr[:, [0, 1, 3, 13, 14, 15]]))) if len(final_arr) else 0.0
            if np.isfinite(diff) and diff < 1e-4:
                for i, ridx in enumerate(keep):
                    fmap[i] = int(ridx)
                    fmet[i] = {"final_to_raw_direct_from_collision_mask": True, "final_to_raw_match_abs_max": diff}
                return fmap, fmet
        for i, row in enumerate(final_arr):
            j, m = _match_grasp(row, exp, args, max_width=args.max_width)
            if j >= 0 and j < len(keep):
                fmap[i] = int(keep[j])
                fmet[i] = {f"final_to_raw_{k}": v for k, v in m.items()}
        return fmap, fmet
    for i, row in enumerate(final_arr):
        j, m = _match_grasp(row, raw_arr, args, max_width=args.max_width)
        if j >= 0:
            fmap[i] = int(j)
            fmet[i] = {f"final_to_raw_{k}": v for k, v in m.items()}
    return fmap, fmet


def _nearest_xyz(q: Optional[np.ndarray], c: Optional[np.ndarray]) -> Tuple[Optional[int], Optional[float]]:
    if q is None or c is None:
        return None, None
    try:
        qq = np.asarray(q, dtype=np.float64).reshape(1, 3)
        cc = np.asarray(c, dtype=np.float64).reshape((-1, 3))
        d = np.linalg.norm(cc - qq, axis=1)
        j = int(np.nanargmin(d))
        return j, float(d[j])
    except Exception:
        return None, None


def _map_hw(payload: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    for k in ["depth_map_pred", "gt_depth_m", "sensor_depth_m", "obs_depth_m_used", "sensor_depth_m_used"]:
        a = _ep(payload, k)
        if a is not None and np.asarray(a).ndim >= 2:
            h, w = np.asarray(a).shape[-2:]
            if h > 1 and w > 1:
                return int(h), int(w)
    return None, None


def _add_objectness_graspness(row: Dict[str, Any], payload: Optional[Dict[str, Any]], idx: Optional[int], prefix: str) -> None:
    if idx is None:
        return
    obj = _ep(payload, "objectness_score")
    vec = _vec_at(obj, idx, channel_first=True)
    _add_head(row, f"{prefix}_objectness", vec)
    if vec is not None and vec.size >= 2:
        p = _softmax(vec)
        if p is not None and p.size >= 2:
            row[f"{prefix}_objectness_fg_prob"] = float(p[1])
            row[f"{prefix}_objectness_pred"] = int(np.argmax(p))
    gr = _ep(payload, "graspness_score")
    val = _flat_at(gr, idx)
    if val is not None:
        row[f"{prefix}_graspness_score"] = val
    for k in ["dbg_grasp_raw", "dbg_grasp_sel", "dbg_mask_obj", "dbg_mask_pred", "dbg_objectness_pred", "depth_confidence_pred"]:
        val = _flat_at(_ep(payload, k), idx)
        if val is not None:
            row[f"{prefix}_{k}"] = val


def _add_depth_at(row: Dict[str, Any], payload: Optional[Dict[str, Any]], pix: Optional[int], prefix: str) -> None:
    if pix is None:
        return
    h, w = _map_hw(payload)
    row[f"{prefix}_pixel_idx"] = int(pix)
    if h is not None and w is not None:
        row[f"{prefix}_pixel_u"] = int(pix) % w
        row[f"{prefix}_pixel_v"] = int(pix) // w
    for k in ["gt_depth_m", "sensor_depth_m", "sensor_depth_m_used", "obs_depth_m_used", "depth_map_pred", "depth_tok_pred", "depth_net_pred", "depth_head_raw_pred", "depth_refined_correction", "depth_residual_pred", "depth_confidence_pred", "dbg_depth_valid"]:
        val = _flat_at(_ep(payload, k), pix)
        if val is not None:
            row[f"{prefix}_{k}"] = val


def _endpoint_components(payload: Optional[Dict[str, Any]], raw_idx: Optional[int], g: np.ndarray) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if payload is None or raw_idx is None:
        return row
    ri = int(raw_idx)
    row["source_raw_idx"] = ri
    source_xyz = None
    for k in ["token_sel_xyz", "xyz_graspable"]:
        a = _ep(payload, k)
        if a is not None and np.asarray(a).ndim == 2 and 0 <= ri < np.asarray(a).shape[0]:
            xyz = np.asarray(a, dtype=np.float64)[ri, :3]
            row[f"{k}_x"], row[f"{k}_y"], row[f"{k}_z"] = map(float, xyz)
            if source_xyz is None:
                source_xyz = xyz
    if source_xyz is not None:
        row["ranked_center_to_source_xyz_dist_m"] = float(np.linalg.norm(np.asarray(g[13:16], dtype=np.float64) - source_xyz))

    token_idx = None
    tok = _ep(payload, "token_sel_idx")
    if tok is not None:
        t = np.asarray(tok).reshape(-1)
        if 0 <= ri < t.size:
            token_idx = int(t[ri])
            row["token_sel_idx"] = token_idx
            _add_objectness_graspness(row, payload, token_idx, "token")
            _add_depth_at(row, payload, token_idx, "token")

    seed_idx = None
    seed_all = _ep(payload, "eco_seed_xyz")
    if seed_all is None:
        seed_all = _ep(payload, "point_clouds")
    if source_xyz is not None and seed_all is not None:
        seed_idx, seed_dist = _nearest_xyz(source_xyz, seed_all)
        if seed_idx is not None:
            row["seed_global_idx"] = int(seed_idx)
            row["seed_global_match_dist_m"] = float(seed_dist or 0.0)
            _add_objectness_graspness(row, payload, seed_idx, "seed")
            img_idxs = _ep(payload, "img_idxs")
            if img_idxs is not None:
                ii = np.asarray(img_idxs).reshape(-1)
                if 0 <= seed_idx < ii.size:
                    pix = int(ii[seed_idx])
                    row["seed_img_idx"] = pix
                    _add_depth_at(row, payload, pix, "seed")
    elif token_idx is None:
        _add_objectness_graspness(row, payload, ri, "source")

    view_idx = None
    vi = _ep(payload, "grasp_top_view_inds")
    if vi is not None:
        vf = np.asarray(vi).reshape(-1)
        if 0 <= ri < vf.size:
            view_idx = int(vf[ri])
            row["grasp_top_view_idx"] = view_idx
    for k in ["eco_view_top_idx", "eco_view_top_score", "view_ray_align"]:
        val = _flat_at(_ep(payload, k), ri)
        if val is not None:
            row[k] = val
    _add_head(row, "view_score", _vec_at(_ep(payload, "view_score"), ri, channel_first=False), selected=view_idx)
    _add_head(row, "angle_head", _vec_at(_ep(payload, "grasp_angle_pred"), ri, channel_first=True))
    try:
        d_bin = int(round(float(g[3]) / 0.01)) - 1
        d_bin = d_bin if d_bin >= 0 else None
    except Exception:
        d_bin = None
    row["decoded_depth_bin_from_grasp"] = d_bin
    _add_head(row, "depth_head", _vec_at(_ep(payload, "grasp_depth_pred"), ri, channel_first=True), selected=d_bin)
    sv = _vec_at(_ep(payload, "grasp_score_pred"), ri, channel_first=True)
    _add_head(row, "score_head", sv, values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if sv is not None and sv.size == 6 else None)
    wv = _vec_at(_ep(payload, "grasp_width_pred"), ri, channel_first=True)
    _add_head(row, "width_head", wv)
    if wv is not None and wv.size:
        row["width_head_raw_value"] = float(wv.reshape(-1)[0])
    return row


def make_grasp_component_rows(result: Dict[str, Any], artifacts: Dict[str, Optional[Path]], endpoint_payload: Optional[Dict[str, Any]], args: argparse.Namespace, top_k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ranked = np.asarray(result.get("ranked_grasps", np.zeros((0, 17))), dtype=np.float32)
    obj_ids = np.asarray(result.get("ranked_obj_ids", np.zeros((0,), dtype=np.int32)))
    eval_scores = np.asarray(result.get("ranked_eval_scores", np.zeros((0,), dtype=np.float32)))
    collision = np.asarray(result.get("ranked_collision", np.zeros((0,), dtype=bool)), dtype=bool)
    empty = np.asarray(result.get("ranked_empty", np.zeros((0,), dtype=bool)), dtype=bool)
    n = min(int(top_k), len(ranked))
    if n <= 0:
        return rows
    cdata = _load_collision_arrays(artifacts)
    final_arr = cdata.get("final_grasps")
    raw_arr = cdata.get("raw_grasps")
    mask = cdata.get("collision_mask")
    final_arr = np.asarray(final_arr, dtype=np.float32).reshape((-1, 17)) if final_arr is not None and len(final_arr) else np.zeros((0, 17), dtype=np.float32)
    raw_arr = np.asarray(raw_arr, dtype=np.float32).reshape((-1, 17)) if raw_arr is not None and len(raw_arr) else None
    mask = np.asarray(mask, dtype=bool).reshape(-1) if mask is not None else None
    f2r, f2r_met = _final_to_raw_map(final_arr, raw_arr, mask, args)
    for i in range(n):
        g = ranked[i]
        fidx, met = _match_grasp(g, final_arr, args, max_width=args.max_width) if len(final_arr) else (-1, {"match_error": "final_grasps_missing"})
        ridx = f2r.get(fidx) if fidx is not None and fidx >= 0 else None
        row: Dict[str, Any] = {
            "rank": int(i + 1),
            "pred_conf": float(g[0]),
            "width": float(g[1]),
            "height": float(g[2]),
            "depth": float(g[3]),
            "tx": float(g[13]),
            "ty": float(g[14]),
            "tz": float(g[15]),
            "assigned_obj_local_id": int(obj_ids[i]) if i < len(obj_ids) else None,
            "eval_friction_score": float(eval_scores[i]) if i < len(eval_scores) else None,
            "collision_or_empty": bool(collision[i]) if i < len(collision) else None,
            "empty": bool(empty[i]) if i < len(empty) else None,
            "matched_final_idx": int(fidx) if fidx is not None and fidx >= 0 else None,
            "matched_raw_idx": int(ridx) if ridx is not None else None,
            "raw_grasp_count": int(len(raw_arr)) if raw_arr is not None else None,
            "final_grasp_count": int(len(final_arr)),
        }
        row.update({f"rank_to_final_{k}": v for k, v in met.items()})
        if fidx in f2r_met:
            row.update(f2r_met[fidx])
        if mask is not None and ridx is not None and 0 <= int(ridx) < len(mask):
            row["model_free_collision_at_raw_idx"] = bool(mask[int(ridx)])
        if ridx is not None:
            row.update(_endpoint_components(endpoint_payload, int(ridx), g))
        rows.append(row)
    return rows


def maybe_make_2d_image(value: Any) -> Optional[np.ndarray]:
    arr = tensor_to_numpy_small(value, max_elems=10_000_000)
    if arr is None:
        return None
    arr = np.asarray(arr)
    # Remove singleton dims.
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # Common cases: CxHxW, HxWxC, NxK. For score tensors, visualize max/mean over channel-like dim.
        if arr.shape[0] in {1, 2, 3, 4, 6, 12, 300}:
            return np.nanmax(arr.astype(np.float32), axis=0)
        if arr.shape[-1] in {1, 2, 3, 4, 6, 12, 300}:
            return np.nanmax(arr.astype(np.float32), axis=-1)
        # Fallback: first slice.
        return arr.reshape(arr.shape[0], -1).astype(np.float32) if arr.shape[0] < 2048 and arr.shape[1] < 2048 else None
    return None


def save_endpoint_heatmaps(out_dir: Path, method_name: str, payload: Dict[str, Any], key_patterns: Sequence[str]) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    ep = payload.get("end_points", {})
    if not isinstance(ep, dict):
        return []
    saved: List[str] = []
    heat_dir = out_dir / "endpoint_heatmaps" / method_name
    heat_dir.mkdir(parents=True, exist_ok=True)
    pats = [p.strip() for p in key_patterns if p.strip()]
    for key, val in ep.items():
        if pats and not any(pat in str(key) for pat in pats):
            continue
        img = maybe_make_2d_image(val)
        if img is None:
            continue
        if img.ndim != 2:
            continue
        try:
            plt.figure(figsize=(6, 5))
            plt.imshow(img)
            plt.title(str(key))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            p = heat_dir / f"{str(key).replace('/', '_')}.png"
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(str(p))
        except Exception:
            try:
                plt.close()
            except Exception:
                pass
    return saved



# -----------------------------------------------------------------------------
# Batch split-level diagnosis helpers
# -----------------------------------------------------------------------------
SPLIT_TO_SCENES = {
    "test_seen": list(range(100, 130)),
    "test_similar": list(range(130, 160)),
    "test_novel": list(range(160, 190)),
    "test": list(range(100, 190)),
}


def split_csv_ints(x: str) -> Optional[set]:
    x = (x or "").strip()
    if not x:
        return None
    return {int(v.strip()) for v in x.split(",") if v.strip()}


def split_patterns(x: str) -> List[str]:
    return [v.strip() for v in (x or "").split(",") if v.strip()]


def format_template(template: str, split: str, method: str = "") -> str:
    if not template:
        return ""
    return template.format(split=split, method=method)


def template_has_field(template: str, field: str) -> bool:
    """Return True if a format template contains a given named field."""
    if not template:
        return False
    return ("{" + field) in template


def cache_root_has_artifacts(path: Path) -> bool:
    """
    Heuristic for a valid method cache root.

    Important: do NOT treat an empty directory as valid. Older runs may have
    created split-specific empty roots such as
        <root>/test_seen/<method>/
    while the real cache is shared at
        <root>/<method>/.
    If we accept the empty split directory, every sample becomes missing.
    Therefore this function requires at least one saved artifact file under the
    standard subdirectories.
    """
    if not path:
        return False
    path = Path(path)
    if not path.exists():
        return False
    checks = [
        ("final_grasps", "*.npy"),
        ("raw_grasps", "*.npy"),
        ("collisions", "*.npz"),
        ("end_points", "*.pth"),
        ("end_points", "*.pth.gz"),
    ]
    for subdir, pattern in checks:
        root = path / subdir
        if root.exists():
            try:
                if next(root.rglob(pattern), None) is not None:
                    return True
            except Exception:
                pass
    return False


def choose_cache_root(candidates: Sequence[Path]) -> str:
    """Choose the first candidate containing real artifacts, otherwise last candidate."""
    clean: List[Path] = []
    seen = set()
    for c in candidates:
        c = Path(c)
        key = str(c)
        if key and key not in seen:
            clean.append(c)
            seen.add(key)
    for c in clean:
        if cache_root_has_artifacts(c):
            return str(c)
    return str(clean[-1]) if clean else ""


def method_cache_candidates_from_root(root_template_or_root: str, args: argparse.Namespace, split: str, method_name: str) -> List[Path]:
    """
    Build possible method cache roots.

    Supported layouts:
      shared root: <root>/<method>/{final_grasps,...}
      split root:  <root>/<split>/<method>/{final_grasps,...}
      method root: <root-template-with-{method}>/{final_grasps,...}

    If the template already contains {split}, formatted root is treated as a split root.
    If it contains {method}, formatted root is treated as the method root directly.
    """
    if not root_template_or_root:
        return []

    contains_split = template_has_field(root_template_or_root, "split")
    contains_method = template_has_field(root_template_or_root, "method")
    formatted = Path(format_template(root_template_or_root, split=split, method=method_name))

    if contains_method:
        return [formatted]

    # If the user passes a concrete method root by mistake, accept it.
    # This is useful for --methodX-cache-root-template but also harmless here.
    if formatted.name == method_name and (
        (formatted / "final_grasps").exists()
        or (formatted / "raw_grasps").exists()
        or (formatted / "end_points").exists()
        or (formatted / "collisions").exists()
    ):
        return [formatted]

    candidates: List[Path] = []
    layout = getattr(args, "cache_layout", "auto")

    if contains_split:
        # Example: vis/root/{split} -> vis/root/test_similar/<method>
        candidates.append(formatted / method_name)
    else:
        if layout == "split":
            candidates.append(formatted / split / method_name)
            candidates.append(formatted / method_name)  # fallback
        elif layout == "shared":
            candidates.append(formatted / method_name)
            candidates.append(formatted / split / method_name)  # fallback
        else:
            # auto: prefer split-specific if it exists, otherwise shared.
            candidates.append(formatted / split / method_name)
            candidates.append(formatted / method_name)
    return candidates


def resolve_method_cache_root_batch(args: argparse.Namespace, split: str, method_idx: int, method_name: str) -> str:
    # Method-specific templates are interpreted as method roots if they contain {method},
    # otherwise as exact roots for that method. This avoids accidental <root>/<method>/<method>.
    tmpl = args.method1_cache_root_template if method_idx == 1 else args.method2_cache_root_template
    if tmpl:
        if template_has_field(tmpl, "method"):
            return format_template(tmpl, split=split, method=method_name)
        formatted = Path(format_template(tmpl, split=split, method=method_name))
        # If a method-specific template points to a base/common root, still try appending method.
        # Otherwise prefer the exact path.
        candidates = [formatted, formatted / method_name]
        return choose_cache_root(candidates)

    if args.cache_root_template:
        candidates = method_cache_candidates_from_root(args.cache_root_template, args, split, method_name)
        return choose_cache_root(candidates)

    if args.cache_root:
        candidates = method_cache_candidates_from_root(args.cache_root, args, split, method_name)
        return choose_cache_root(candidates)

    return ""


def resolve_base_cache_root_batch(args: argparse.Namespace, split: str) -> str:
    """
    Resolve the base/root containing sample_index.csv.

    For shared layout it is <root>. For split layout it is <root>/<split>.
    If cache-root-template contains {method}, use the parent of the resolved method root.
    """
    root_template_or_root = args.cache_root_template or args.cache_root
    if not root_template_or_root:
        return ""

    contains_split = template_has_field(root_template_or_root, "split")
    contains_method = template_has_field(root_template_or_root, "method")

    if contains_method:
        p = Path(format_template(root_template_or_root, split=split, method=args.method1_name))
        return str(p.parent)

    formatted = Path(format_template(root_template_or_root, split=split, method=""))
    layout = getattr(args, "cache_layout", "auto")

    candidates: List[Path] = []
    if contains_split:
        candidates.append(formatted)
    else:
        if layout == "split":
            candidates.append(formatted / split)
            candidates.append(formatted)
        elif layout == "shared":
            candidates.append(formatted)
            candidates.append(formatted / split)
        else:
            candidates.append(formatted / split)
            candidates.append(formatted)

    # Prefer an existing sample_index.csv root if possible.
    for c in candidates:
        if (c / "sample_index.csv").exists():
            return str(c)
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[-1]) if candidates else ""


def resolve_dump_root_batch(args: argparse.Namespace, split: str, method_idx: int) -> str:
    tmpl = args.method1_dump_root_template if method_idx == 1 else args.method2_dump_root_template
    return format_template(tmpl, split=split) if tmpl else ""


def resolve_sample_index_path(args: argparse.Namespace, split: str, method1_cache_root: str, method2_cache_root: str) -> Optional[Path]:
    if args.sample_index_template:
        p = Path(format_template(args.sample_index_template, split=split))
        return p if p.exists() else None
    base = resolve_base_cache_root_batch(args, split)
    candidates: List[Path] = []
    if base:
        candidates.append(Path(base) / "sample_index.csv")
    for r in [method1_cache_root, method2_cache_root]:
        if r:
            candidates.append(Path(r).parent / "sample_index.csv")
            candidates.append(Path(r) / "sample_index.csv")
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_samples_for_split(split: str, sample_interval: float, sample_offset: int, annos_per_scene: int) -> List[Dict[str, Any]]:
    scenes = SPLIT_TO_SCENES[split]
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")
    if sample_interval >= 1.0:
        ann_ids = list(range(annos_per_scene))
    else:
        stride = max(1, int(round(1.0 / sample_interval)))
        offset = int(sample_offset) % stride
        ann_ids = list(range(offset, annos_per_scene, stride))
    rows = []
    subset_idx = 0
    for scene_id in scenes:
        for ann_id in ann_ids:
            rows.append({
                "split": split,
                "scene_id": int(scene_id),
                "scene_name": f"scene_{scene_id:04d}",
                "ann_id": int(ann_id),
                "subset_index": subset_idx,
                "camera": "",
            })
            subset_idx += 1
    return rows


def normalize_sample_rows(raw_rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        scene_name = str(r.get("scene_name", ""))
        scene_id = r.get("scene_id", None)
        if scene_id in (None, "") and scene_name.startswith("scene_"):
            scene_id = int(scene_name.split("_")[-1])
        ann_id = r.get("ann_id", r.get("orig_ann_id", None))
        if ann_id in (None, ""):
            # Last-resort fallback: data_idx modulo 256.
            ann_id = int(r.get("data_idx", 0)) % 256
        rows.append({
            "split": split,
            "subset_index": int(float(r.get("subset_index", len(rows)))) if str(r.get("subset_index", "")).strip() else len(rows),
            "data_idx": int(float(r.get("data_idx", -1))) if str(r.get("data_idx", "")).strip() else -1,
            "scene_name": scene_name or f"scene_{int(scene_id):04d}",
            "scene_id": int(scene_id),
            "ann_id": int(ann_id),
            "camera": r.get("camera", ""),
            "test_mode": r.get("test_mode", split),
        })
    return rows


def get_samples_for_split(args: argparse.Namespace, split: str, method1_cache_root: str, method2_cache_root: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    scenes_keep = split_csv_ints(args.scenes)
    anns_keep = split_csv_ints(args.ann_ids)
    sample_index_path = None
    rows: List[Dict[str, Any]]
    if (not args.ignore_sample_index) and args.use_sample_index:
        sample_index_path = resolve_sample_index_path(args, split, method1_cache_root, method2_cache_root)
    if sample_index_path is not None:
        rows = normalize_sample_rows(read_csv_dicts(sample_index_path), split)
        # Keep only rows from this split's scene range, in case a shared sample_index exists.
        split_scenes = set(SPLIT_TO_SCENES[split])
        rows = [r for r in rows if int(r["scene_id"]) in split_scenes]
        if (len(rows) == 0) and (not args.no_sample_index_fallback):
            print(
                f"[warn] sample_index.csv exists but has no rows for {split}: {sample_index_path}. "
                "Falling back to generated scene/ann ids from --sample-interval."
            )
            rows = generate_samples_for_split(split, float(args.sample_interval), int(args.sample_offset), int(args.annos_per_scene))
            sample_index_path = None
    else:
        rows = generate_samples_for_split(split, float(args.sample_interval), int(args.sample_offset), int(args.annos_per_scene))
    if scenes_keep is not None:
        rows = [r for r in rows if int(r["scene_id"]) in scenes_keep]
    if anns_keep is not None:
        rows = [r for r in rows if int(r["ann_id"]) in anns_keep]
    rows = sorted(rows, key=lambda r: (int(r["scene_id"]), int(r["ann_id"])))
    if int(args.max_samples_per_split) > 0:
        rows = rows[: int(args.max_samples_per_split)]
    return rows, str(sample_index_path) if sample_index_path is not None else None


def sanitize_col(x: str, max_len: int = 120) -> str:
    x = re.sub(r"[^0-9a-zA-Z_]+", "_", str(x)).strip("_")
    if not x:
        x = "key"
    return x[:max_len]


def is_important_key(key: str, patterns: Sequence[str]) -> bool:
    k = str(key)
    kl = k.lower()
    for pat in patterns:
        pl = pat.lower()
        if fnmatch.fnmatch(k, pat) or fnmatch.fnmatch(kl, pl):
            return True
        # Make literal items like "depth" usable as substring patterns.
        if ("*" not in pat) and ("?" not in pat) and (pl in kl):
            return True
    return False


def get_endpoint_summary_rows(payload: Dict[str, Any], max_elems: int) -> List[Dict[str, Any]]:
    rows = payload.get("endpoint_summaries", None)
    if isinstance(rows, list):
        return [dict(r) for r in rows if isinstance(r, dict)]
    return summarize_endpoint_payload(payload, max_elems=max_elems)


def endpoint_wide_from_summary_rows(rows: List[Dict[str, Any]], patterns: Sequence[str], stats: Sequence[str], max_keys: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    kept = 0
    for r in rows:
        key = str(r.get("key", ""))
        if not key or not is_important_key(key, patterns):
            continue
        if kept >= max_keys:
            break
        sk = sanitize_col(key)
        for stat in stats:
            if stat in r and r[stat] not in (None, ""):
                out[f"ep__{sk}__{sanitize_col(stat)}"] = r[stat]
        if "shape" in r:
            out[f"ep__{sk}__shape"] = r.get("shape")
        if "dtype" in r:
            out[f"ep__{sk}__dtype"] = r.get("dtype")
        kept += 1
    out["endpoint_important_key_count"] = kept
    return out


def endpoint_summary_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if "key" in r:
            out[str(r["key"])] = r
    return out


def endpoint_delta_rows(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    patterns: Sequence[str],
    stats: Sequence[str],
    meta: Dict[str, Any],
    method1_name: str,
    method2_name: str,
) -> List[Dict[str, Any]]:
    m1 = endpoint_summary_map(rows1)
    m2 = endpoint_summary_map(rows2)
    out: List[Dict[str, Any]] = []
    for key in sorted(set(m1.keys()) & set(m2.keys())):
        if not is_important_key(key, patterns):
            continue
        r1, r2 = m1[key], m2[key]
        row: Dict[str, Any] = {**meta, "key": key, "method1": method1_name, "method2": method2_name}
        row["method1_shape"] = r1.get("shape")
        row["method2_shape"] = r2.get("shape")
        row["method1_dtype"] = r1.get("dtype")
        row["method2_dtype"] = r2.get("dtype")
        for stat in stats:
            v1 = safe_float(r1.get(stat))
            v2 = safe_float(r2.get(stat))
            if v1 is not None:
                row[f"method1_{stat}"] = v1
            if v2 is not None:
                row[f"method2_{stat}"] = v2
            if v1 is not None and v2 is not None:
                row[f"delta_{stat}_m2_minus_m1"] = v2 - v1
        out.append(row)
    return out


def make_object_summary_rows(
    result: Dict[str, Any],
    obj_list: Sequence[int],
    base_meta: Dict[str, Any],
    method_name: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    grasps = result["ranked_grasps"][:top_k]
    obj_ids = result["ranked_obj_ids"][:top_k]
    scores = result["ranked_eval_scores"][:top_k]
    collision = result["ranked_collision"][:top_k]
    empty = result["ranked_empty"][:top_k]
    rows: List[Dict[str, Any]] = []
    for local_id, global_obj_id in enumerate(obj_list):
        mask = obj_ids == local_id
        idx = np.where(mask)[0]
        row: Dict[str, Any] = {
            **base_meta,
            "method": method_name,
            "assigned_obj_local_id": int(local_id),
            "graspnet_obj_id": int(global_obj_id),
            "selected_topk_count": int(mask.sum()),
        }
        if idx.size > 0:
            sc = scores[idx]
            col = collision[idx]
            emp = empty[idx]
            g = grasps[idx]
            row.update({
                "first_rank": int(idx.min() + 1),
                "mean_rank": float(np.mean(idx + 1)),
                "collision_rate": float(np.mean(col)),
                "empty_rate": float(np.mean(emp)),
                "success_rate_fric04": float(np.mean((sc <= 0.4 + 1e-6) & (sc > 0))),
                "success_rate_fric08": float(np.mean((sc <= 0.8 + 1e-6) & (sc > 0))),
                "success_rate_fric12": float(np.mean((sc <= 1.2 + 1e-6) & (sc > 0))),
            })
            row.update(summarize_array(sc, "eval_score"))
            row.update(summarize_array(g[:, 0], "pred_conf"))
            row.update(summarize_array(g[:, 1], "width"))
            row.update(summarize_array(g[:, 3], "depth"))
        rows.append(row)
    return rows


def add_meta_to_rows(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    method: Optional[str] = None,
    **extra_meta: Any,
) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(meta)
        if method is not None:
            rr["method"] = method
        rr.update(extra_meta)
        rr.update(r)
        out.append(rr)
    return out


def flatten_method_summary(base_meta: Dict[str, Any], method_name: str, artifacts: Dict[str, Optional[Path]], raw_stats: Dict[str, Any], collision_stats: Dict[str, Any], detailed: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {**base_meta, "method": method_name}
    row["final_npy_path"] = str(artifacts["final_npy"]) if artifacts.get("final_npy") is not None else None
    row["raw_npy_path"] = str(artifacts["raw_npy"]) if artifacts.get("raw_npy") is not None else None
    row["collision_npz_path"] = str(artifacts["collision_npz"]) if artifacts.get("collision_npz") is not None else None
    row["endpoint_path"] = str(artifacts["endpoint"]) if artifacts.get("endpoint") is not None else None
    row.update(raw_stats)
    # Clarify the confusing name inherited from clip_width_like_eval.
    if "num_raw_grasps" in raw_stats:
        row["final_grasp_npy_count"] = raw_stats["num_raw_grasps"]
    row.update(collision_stats)
    row.update(detailed.get("stats", {}))
    return row


def numeric_value(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return float(x)
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def aggregate_group(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_keys)].append(r)
    out: List[Dict[str, Any]] = []
    for key_vals, rs in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        row = {k: v for k, v in zip(group_keys, key_vals)}
        row["num_rows"] = len(rs)
        all_keys = sorted(set().union(*(r.keys() for r in rs)))
        for k in all_keys:
            if k in group_keys:
                continue
            vals = [numeric_value(r.get(k)) for r in rs]
            vals = [v for v in vals if v is not None]
            if vals and len(vals) >= max(1, int(0.5 * len(rs))):
                arr = np.asarray(vals, dtype=np.float64)
                row[f"{k}__mean"] = float(np.mean(arr))
                row[f"{k}__median"] = float(np.median(arr))
                row[f"{k}__std"] = float(np.std(arr))
        out.append(row)
    return out


def make_delta_row(meta: Dict[str, Any], row1: Dict[str, Any], row2: Dict[str, Any], method1_name: str, method2_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {**meta, "method1": method1_name, "method2": method2_name}
    keys_of_interest = [
        "AP", "AP0.4", "AP0.8",
        "top1_collision_rate", "top5_collision_rate", "top10_collision_rate", "top20_collision_rate", "top50_collision_rate",
        "top1_success_rate_fric04", "top5_success_rate_fric04", "top10_success_rate_fric04", "top20_success_rate_fric04", "top50_success_rate_fric04",
        "top1_success_rate_fric08", "top5_success_rate_fric08", "top10_success_rate_fric08", "top20_success_rate_fric08", "top50_success_rate_fric08",
        "model_free_collision_rate", "raw_grasp_count", "final_grasp_npy_count", "num_after_nms", "num_eval_selected",
        "pred_conf_topk_mean", "width_topk_mean", "depth_topk_mean", "eval_score_topk_mean",
    ]
    for k in keys_of_interest:
        v1 = numeric_value(row1.get(k))
        v2 = numeric_value(row2.get(k))
        if v1 is not None:
            out[f"method1_{k}"] = v1
        if v2 is not None:
            out[f"method2_{k}"] = v2
        if v1 is not None and v2 is not None:
            out[f"delta_{k}_m2_minus_m1"] = v2 - v1
    return out


def plot_aggregate_ap_curves(out_dir: Path, aggregate_mats: Dict[Tuple[str, str], np.ndarray], method_names: Sequence[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    curve_dir = out_dir / "aggregate_ap_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted({k[0] for k in aggregate_mats.keys()}):
        plt.figure(figsize=(9, 5))
        for method in method_names:
            mat = aggregate_mats.get((split, method))
            if mat is None:
                continue
            ks = np.arange(1, mat.shape[0] + 1)
            for fric in [0.4, 0.8, 1.2]:
                idx = FRICTIONS.index(fric)
                ls = "-" if method == method_names[0] else "--"
                plt.plot(ks, mat[:, idx], linestyle=ls, label=f"{method} fric<={fric}")
        plt.xlabel("Top-k rank")
        plt.ylabel("Mean cumulative precision")
        plt.ylim(0, 1.02)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(curve_dir / f"{split}_ap_curves.png", dpi=180)
        plt.close()



def _ns_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return dict(vars(args))


def _empty_scene_result(split: str, scene_id: int) -> Dict[str, Any]:
    return {
        "split": split,
        "scene_id": int(scene_id),
        "status": "ok",
        "elapsed_sec": 0.0,
        "per_sample_method_rows": [],
        "per_sample_delta_rows": [],
        "per_object_rows": [],
        "rank_rows_all": [],
        "matched_pair_rows_all": [],
        "grasp_component_rows_all": [],
        "endpoint_long_rows": [],
        "endpoint_wide_rows": [],
        "endpoint_delta_long_rows": [],
        "missing_rows": [],
        "aggregate_ap_sums": {},
        "aggregate_ap_counts": {},
        "processed_pair_samples": 0,
        "requested_samples": 0,
    }


def _process_scene_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry point. One task corresponds to one split/scene."""
    t0 = time.time()
    args = argparse.Namespace(**task["args"])
    split = str(task["split"])
    scene_id = int(task["scene_id"])
    samples = list(task["samples"])
    result = _empty_scene_result(split, scene_id)
    result["requested_samples"] = len(samples)

    try:
        # Avoid OpenMP oversubscription when many scene workers run concurrently.
        # Respect user's explicit setting if already provided.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        ge = GraspNetEval(args.dataset_root, args.camera, split="test")
        config = get_config()
        base_table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=args.voxel_size)
        important_patterns = split_patterns(args.important_endpoint_patterns)
        endpoint_stats = split_patterns(args.endpoint_wide_stats)

        model_list, dexmodel_list, obj_list = ge.get_scene_models(scene_id, ann_id=0)
        model_sampled_list = [voxel_sample_points(model, args.voxel_size) for model in model_list]

        m1_cache_root = task.get("method1_cache_root", "")
        m2_cache_root = task.get("method2_cache_root", "")
        m1_dump_root = task.get("method1_dump_root", "")
        m2_dump_root = task.get("method2_dump_root", "")

        aggregate_ap_sums: Dict[str, np.ndarray] = {}
        aggregate_ap_counts: Dict[str, int] = {}

        for sample in samples:
            ann_id = int(sample["ann_id"])
            base_meta = {
                "split": split,
                "scene_id": scene_id,
                "scene_name": f"scene_{scene_id:04d}",
                "ann_id": ann_id,
                "camera": args.camera,
                "sample_index_within_split": int(sample.get("sample_index_within_split", sample.get("subset_index", -1))),
            }
            try:
                _, pose_list, camera_pose, align_mat = ge.get_model_poses(scene_id, ann_id)
                table_trans = transform_points(base_table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
            except Exception as e:
                row = {**base_meta, "stage": "load_pose_or_table", "error": repr(e), "traceback": traceback.format_exc(limit=5)}
                result["missing_rows"].append(row)
                if args.skip_missing:
                    continue
                raise

            method_specs = [
                (args.method1_name, m1_cache_root, m1_dump_root, 1),
                (args.method2_name, m2_cache_root, m2_dump_root, 2),
            ]
            method_rows: Dict[str, Dict[str, Any]] = {}
            method_results: Dict[str, Dict[str, Any]] = {}
            endpoint_summary_by_method: Dict[str, List[Dict[str, Any]]] = {}

            for method_name, cache_root, dump_root, method_idx in method_specs:
                try:
                    artifacts = resolve_method_artifacts(
                        cache_root=cache_root,
                        dump_root=dump_root,
                        direct_npy="",
                        direct_raw_npy="",
                        direct_endpoint="",
                        direct_collision_npz="",
                        scene_id=scene_id,
                        ann_id=ann_id,
                        camera=args.camera,
                    )
                except Exception as e:
                    row = {**base_meta, "method": method_name, "stage": "resolve_artifacts", "error": repr(e)}
                    result["missing_rows"].append(row)
                    if args.skip_missing:
                        continue
                    raise

                try:
                    gg = GraspGroup().from_npy(str(artifacts["final_npy"]))
                    raw_stats = clip_width_like_eval(gg, args.max_width)
                    detailed = eval_grasp_detailed(
                        gg,
                        model_sampled_list,
                        dexmodel_list,
                        pose_list,
                        config,
                        table_trans,
                        voxel_size=args.voxel_size,
                        top_k=args.top_k,
                        nms_trans_th=args.nms_trans_th,
                        nms_rot_rad=args.nms_rot_deg / 180.0 * np.pi,
                    )
                    collision_stats = load_raw_collision_summary(artifacts)
                    method_row = flatten_method_summary(base_meta, method_name, artifacts, raw_stats, collision_stats, detailed)
                    method_rows[method_name] = method_row
                    method_results[method_name] = detailed
                    result["per_sample_method_rows"].append(method_row)

                    mat = np.asarray(detailed["ap_matrix"], dtype=np.float64)
                    if method_name not in aggregate_ap_sums:
                        aggregate_ap_sums[method_name] = np.zeros_like(mat, dtype=np.float64)
                        aggregate_ap_counts[method_name] = 0
                    aggregate_ap_sums[method_name] += mat
                    aggregate_ap_counts[method_name] += 1

                    if args.save_rank_rows:
                        result["rank_rows_all"].extend(add_meta_to_rows(make_ranked_rows(detailed, args.top_k), base_meta, method=method_name))
                    if args.save_per_object_rows:
                        result["per_object_rows"].extend(make_object_summary_rows(detailed, obj_list, base_meta, method_name, args.top_k))

                    payload = None
                    need_endpoint_payload = bool(args.analyze_endpoints or args.save_grasp_component_rows)
                    if need_endpoint_payload and artifacts.get("endpoint") is not None:
                        try:
                            payload = load_endpoint_payload(Path(artifacts["endpoint"]), weights_only=bool(args.torch_load_weights_only))
                        except Exception as e:
                            result["missing_rows"].append({**base_meta, "method": method_name, "stage": "load_endpoint", "endpoint_path": str(artifacts.get("endpoint")), "error": repr(e)})
                    elif need_endpoint_payload:
                        result["missing_rows"].append({**base_meta, "method": method_name, "stage": "endpoint_missing", "error": "endpoint cache not found"})

                    if args.analyze_endpoints and payload is not None:
                        try:
                            ep_rows = get_endpoint_summary_rows(payload, max_elems=int(args.endpoint_max_array_elems))
                            endpoint_summary_by_method[method_name] = ep_rows
                            ep_meta = {**base_meta, "method": method_name, "endpoint_path": str(artifacts["endpoint"])}
                            if args.save_endpoint_long:
                                result["endpoint_long_rows"].extend(add_meta_to_rows(ep_rows, ep_meta))
                            wide = {**ep_meta}
                            wide.update(endpoint_wide_from_summary_rows(ep_rows, important_patterns, endpoint_stats, int(args.endpoint_max_wide_keys)))
                            result["endpoint_wide_rows"].append(wide)
                        except Exception as e:
                            result["missing_rows"].append({**base_meta, "method": method_name, "stage": "summarize_endpoint", "endpoint_path": str(artifacts.get("endpoint")), "error": repr(e)})

                    if args.save_grasp_component_rows:
                        try:
                            ctopk = int(args.component_top_k) if int(args.component_top_k) > 0 else int(args.top_k)
                            comp_rows = make_grasp_component_rows(detailed, artifacts, payload, args, top_k=ctopk)
                            result["grasp_component_rows_all"].extend(add_meta_to_rows(comp_rows, base_meta, method=method_name, endpoint_path=str(artifacts.get("endpoint")) if artifacts.get("endpoint") is not None else None))
                        except Exception as e:
                            result["missing_rows"].append({**base_meta, "method": method_name, "stage": "make_grasp_component_rows", "endpoint_path": str(artifacts.get("endpoint")), "error": repr(e), "traceback": traceback.format_exc(limit=5)})
                except Exception as e:
                    row = {**base_meta, "method": method_name, "stage": "evaluate_method", "error": repr(e), "traceback": traceback.format_exc(limit=5)}
                    result["missing_rows"].append(row)
                    if args.skip_missing:
                        continue
                    raise

            if args.method1_name in method_rows and args.method2_name in method_rows:
                result["per_sample_delta_rows"].append(make_delta_row(base_meta, method_rows[args.method1_name], method_rows[args.method2_name], args.method1_name, args.method2_name))
                if args.save_matched_pairs and args.method1_name in method_results and args.method2_name in method_results:
                    pairs = match_topk(
                        method_results[args.method1_name],
                        method_results[args.method2_name],
                        top_k=args.match_top_k,
                        dist_th=args.match_dist_th,
                        rot_th_rad=args.match_rot_deg_th / 180.0 * np.pi,
                    )
                    result["matched_pair_rows_all"].extend(add_meta_to_rows(pairs, base_meta))
                if args.analyze_endpoints and args.method1_name in endpoint_summary_by_method and args.method2_name in endpoint_summary_by_method:
                    result["endpoint_delta_long_rows"].extend(endpoint_delta_rows(
                        endpoint_summary_by_method[args.method1_name],
                        endpoint_summary_by_method[args.method2_name],
                        important_patterns,
                        endpoint_stats,
                        base_meta,
                        args.method1_name,
                        args.method2_name,
                    ))
                result["processed_pair_samples"] += 1

        # Convert aggregate matrices to pickle-safe and JSON-safe-ish containers.
        result["aggregate_ap_sums"] = {k: v for k, v in aggregate_ap_sums.items()}
        result["aggregate_ap_counts"] = aggregate_ap_counts
        result["elapsed_sec"] = float(time.time() - t0)
        return result
    except Exception as e:
        result["status"] = "scene_failed"
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()
        result["elapsed_sec"] = float(time.time() - t0)
        return result


def _append_rows(dst: Dict[str, List[Dict[str, Any]]], res: Dict[str, Any]) -> None:
    for key in [
        "per_sample_method_rows",
        "per_sample_delta_rows",
        "per_object_rows",
        "rank_rows_all",
        "matched_pair_rows_all",
        "grasp_component_rows_all",
        "endpoint_long_rows",
        "endpoint_wide_rows",
        "endpoint_delta_long_rows",
        "missing_rows",
    ]:
        dst[key].extend(res.get(key, []))


def _write_scene_result_files(out_dir: Path, res: Dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "no_scene_output_files", False):
        return
    split = str(res.get("split", "unknown"))
    scene_id = int(res.get("scene_id", -1))
    sdir = out_dir / args.scene_output_dir / split / f"scene_{scene_id:04d}"
    sdir.mkdir(parents=True, exist_ok=True)
    write_csv(sdir / "per_sample_method_summary.csv", res.get("per_sample_method_rows", []))
    write_csv(sdir / "per_sample_delta_summary.csv", res.get("per_sample_delta_rows", []))
    if args.save_per_object_rows:
        write_csv(sdir / "per_object_summary.csv", res.get("per_object_rows", []))
    if args.save_rank_rows:
        write_csv(sdir / "rank_eval_rows.csv", res.get("rank_rows_all", []))
    if args.save_matched_pairs:
        write_csv(sdir / "matched_pairs_topK_all.csv", res.get("matched_pair_rows_all", []))
    if args.save_grasp_component_rows:
        write_csv(sdir / "grasp_component_rows.csv", res.get("grasp_component_rows_all", []))
    if args.analyze_endpoints:
        if args.save_endpoint_long:
            write_csv(sdir / "endpoint_key_summary_long.csv", res.get("endpoint_long_rows", []))
        write_csv(sdir / "endpoint_variable_wide.csv", res.get("endpoint_wide_rows", []))
        write_csv(sdir / "endpoint_stat_delta_long.csv", res.get("endpoint_delta_long_rows", []))
    write_csv(sdir / "missing_or_error_rows.csv", res.get("missing_rows", []))
    overview = {
        "split": split,
        "scene_id": scene_id,
        "status": res.get("status"),
        "elapsed_sec": res.get("elapsed_sec"),
        "requested_samples": res.get("requested_samples"),
        "processed_pair_samples": res.get("processed_pair_samples"),
        "num_missing_or_error_rows": len(res.get("missing_rows", [])),
        "error": res.get("error"),
    }
    with (sdir / "scene_overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, default=str)


def _build_scene_tasks(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    split_sample_index_paths: Dict[str, Any] = {}
    requested_by_split: Dict[str, int] = {}
    scene_task_rows: List[Dict[str, Any]] = []

    for split in args.splits:
        m1_cache_root = resolve_method_cache_root_batch(args, split, 1, args.method1_name)
        m2_cache_root = resolve_method_cache_root_batch(args, split, 2, args.method2_name)
        m1_dump_root = resolve_dump_root_batch(args, split, 1)
        m2_dump_root = resolve_dump_root_batch(args, split, 2)
        samples, sample_index_path = get_samples_for_split(args, split, m1_cache_root, m2_cache_root)
        split_sample_index_paths[split] = sample_index_path
        requested_by_split[split] = len(samples)
        print(f"\n[{split}] samples={len(samples)} sample_index={sample_index_path}", flush=True)
        print(f"  method1_cache_root={m1_cache_root}", flush=True)
        print(f"  method2_cache_root={m2_cache_root}", flush=True)

        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for i, sample in enumerate(samples):
            ss = dict(sample)
            ss["sample_index_within_split"] = int(i)
            grouped[int(ss["scene_id"])].append(ss)

        for scene_id in sorted(grouped.keys()):
            task = {
                "args": _ns_to_dict(args),
                "split": split,
                "scene_id": int(scene_id),
                "samples": grouped[scene_id],
                "method1_cache_root": m1_cache_root,
                "method2_cache_root": m2_cache_root,
                "method1_dump_root": m1_dump_root,
                "method2_dump_root": m2_dump_root,
            }
            tasks.append(task)
            scene_task_rows.append({
                "split": split,
                "scene_id": int(scene_id),
                "num_samples": len(grouped[scene_id]),
                "method1_cache_root": m1_cache_root,
                "method2_cache_root": m2_cache_root,
                "sample_index_path": sample_index_path,
            })

    meta = {
        "split_sample_index_paths": split_sample_index_paths,
        "requested_by_split": requested_by_split,
        "scene_task_rows": scene_task_rows,
    }
    return tasks, meta


def main() -> None:
    args = parse_args()
    if args.strict_missing:
        args.skip_missing = False
    if args.no_analyze_endpoints:
        args.analyze_endpoints = False
    if args.no_save_rank_rows:
        args.save_rank_rows = False
    if args.no_save_per_object_rows:
        args.save_per_object_rows = False
    if args.no_save_endpoint_long:
        args.save_endpoint_long = False
    if args.no_save_aggregate_ap_curves:
        args.save_aggregate_ap_curves = False
    if args.no_save_grasp_component_rows:
        args.save_grasp_component_rows = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, task_meta = _build_scene_tasks(args)
    write_csv(out_dir / "planned_scene_tasks.csv", task_meta["scene_task_rows"])

    started = {
        "dataset_root": args.dataset_root,
        "camera": args.camera,
        "splits": args.splits,
        "method1": args.method1_name,
        "method2": args.method2_name,
        "top_k": int(args.top_k),
        "frictions": FRICTIONS,
        "num_scene_tasks": len(tasks),
        "requested_by_split": task_meta["requested_by_split"],
        "sample_index_paths": task_meta["split_sample_index_paths"],
        "parallel_backend": args.parallel_backend,
        "num_scene_workers": int(args.num_scene_workers),
        "note": "This file is written before scene processing starts so output-dir is not empty during long runs.",
    }
    with (out_dir / "batch_diagnosis_started.json").open("w", encoding="utf-8") as f:
        json.dump(started, f, indent=2, default=str)

    print(f"\nPlanned {len(tasks)} scene tasks. Outputs will be written incrementally under {out_dir}", flush=True)
    if len(tasks) == 0:
        print("No scene tasks to process. Check --splits, --sample-interval, --scenes, --ann-ids, and sample_index.csv.", flush=True)

    all_rows: Dict[str, List[Dict[str, Any]]] = {
        "per_sample_method_rows": [],
        "per_sample_delta_rows": [],
        "per_object_rows": [],
        "rank_rows_all": [],
        "matched_pair_rows_all": [],
        "grasp_component_rows_all": [],
        "endpoint_long_rows": [],
        "endpoint_wide_rows": [],
        "endpoint_delta_long_rows": [],
        "missing_rows": [],
    }
    aggregate_ap_sums: Dict[Tuple[str, str], np.ndarray] = {}
    aggregate_ap_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    progress_path = out_dir / "progress.jsonl"

    t_all = time.time()
    completed = 0
    total_processed = 0
    total_requested = sum(len(t["samples"]) for t in tasks)

    def consume_result(res: Dict[str, Any]) -> None:
        nonlocal completed, total_processed
        completed += 1
        total_processed += int(res.get("processed_pair_samples", 0) or 0)
        _append_rows(all_rows, res)
        _write_scene_result_files(out_dir, res, args)

        split = str(res.get("split", ""))
        for method, mat_sum in res.get("aggregate_ap_sums", {}).items():
            key = (split, method)
            mat_sum = np.asarray(mat_sum, dtype=np.float64)
            if key not in aggregate_ap_sums:
                aggregate_ap_sums[key] = np.zeros_like(mat_sum, dtype=np.float64)
            aggregate_ap_sums[key] += mat_sum
            aggregate_ap_counts[key] += int(res.get("aggregate_ap_counts", {}).get(method, 0))

        prog = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_scene_tasks": completed,
            "total_scene_tasks": len(tasks),
            "split": res.get("split"),
            "scene_id": res.get("scene_id"),
            "status": res.get("status"),
            "elapsed_sec": res.get("elapsed_sec"),
            "requested_samples": res.get("requested_samples"),
            "processed_pair_samples": res.get("processed_pair_samples"),
            "missing_or_error_rows": len(res.get("missing_rows", [])),
            "error": res.get("error"),
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(prog, default=str) + "\n")
        every = max(1, int(args.progress_every_scenes))
        if completed % every == 0 or completed == len(tasks):
            print(
                f"[progress] {completed}/{len(tasks)} scenes done | "
                f"last={res.get('split')} scene_{int(res.get('scene_id', -1)):04d} "
                f"status={res.get('status')} processed_pairs={res.get('processed_pair_samples')} "
                f"elapsed={float(res.get('elapsed_sec', 0.0)):.1f}s",
                flush=True,
            )

    max_workers = max(1, int(args.num_scene_workers))
    if args.parallel_backend == "none" or max_workers <= 1:
        for task in tasks:
            consume_result(_process_scene_task(task))
    else:
        Executor = ProcessPoolExecutor if args.parallel_backend == "process" else ThreadPoolExecutor
        with Executor(max_workers=min(max_workers, len(tasks) if tasks else max_workers)) as ex:
            fut_to_task = {ex.submit(_process_scene_task, task): task for task in tasks}
            for fut in as_completed(fut_to_task):
                task = fut_to_task[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = _empty_scene_result(str(task.get("split", "unknown")), int(task.get("scene_id", -1)))
                    res["status"] = "future_failed"
                    res["error"] = repr(e)
                    res["traceback"] = traceback.format_exc()
                    res["missing_rows"].append({"split": res["split"], "scene_id": res["scene_id"], "stage": "future_failed", "error": repr(e)})
                consume_result(res)

    # Write combined outputs.
    write_csv(out_dir / "per_sample_method_summary.csv", all_rows["per_sample_method_rows"])
    write_csv(out_dir / "per_sample_delta_summary.csv", all_rows["per_sample_delta_rows"])
    if args.save_per_object_rows:
        write_csv(out_dir / "per_object_summary.csv", all_rows["per_object_rows"])
    if args.save_rank_rows:
        write_csv(out_dir / "rank_eval_rows.csv", all_rows["rank_rows_all"])
    if args.save_matched_pairs:
        write_csv(out_dir / "matched_pairs_topK_all.csv", all_rows["matched_pair_rows_all"])
    if args.save_grasp_component_rows:
        write_csv(out_dir / "grasp_component_rows.csv", all_rows["grasp_component_rows_all"])
    if args.analyze_endpoints:
        if args.save_endpoint_long:
            write_csv(out_dir / "endpoint_key_summary_long.csv", all_rows["endpoint_long_rows"])
        write_csv(out_dir / "endpoint_variable_wide.csv", all_rows["endpoint_wide_rows"])
        write_csv(out_dir / "endpoint_stat_delta_long.csv", all_rows["endpoint_delta_long_rows"])
    write_csv(out_dir / "missing_or_error_rows.csv", all_rows["missing_rows"])

    split_summary = aggregate_group(all_rows["per_sample_method_rows"], ["split", "method"])
    scene_summary = aggregate_group(all_rows["per_sample_method_rows"], ["split", "scene_id", "method"])
    write_csv(out_dir / "per_split_method_summary.csv", split_summary)
    write_csv(out_dir / "per_scene_method_summary.csv", scene_summary)

    delta_sorted = sorted(
        all_rows["per_sample_delta_rows"],
        key=lambda r: numeric_value(r.get("delta_AP_m2_minus_m1")) if numeric_value(r.get("delta_AP_m2_minus_m1")) is not None else 0.0,
    )
    write_csv(out_dir / "top_method2_worse_by_AP.csv", delta_sorted)
    write_csv(out_dir / "top_method2_better_by_AP.csv", list(reversed(delta_sorted)))

    aggregate_mats: Dict[Tuple[str, str], np.ndarray] = {}
    ap_dir = out_dir / "aggregate_ap_matrices"
    ap_dir.mkdir(parents=True, exist_ok=True)
    for key, mat_sum in aggregate_ap_sums.items():
        cnt = int(aggregate_ap_counts.get(key, 0))
        if cnt <= 0:
            continue
        mat = mat_sum / float(cnt)
        aggregate_mats[key] = mat
        split, method = key
        np.save(ap_dir / f"{split}_{method}_ap_matrix.npy", mat)
    if args.save_aggregate_ap_curves:
        plot_aggregate_ap_curves(out_dir, aggregate_mats, [args.method1_name, args.method2_name])

    overview = {
        "dataset_root": args.dataset_root,
        "camera": args.camera,
        "splits": args.splits,
        "method1": args.method1_name,
        "method2": args.method2_name,
        "top_k": int(args.top_k),
        "frictions": FRICTIONS,
        "requested_samples": int(total_requested),
        "processed_pair_samples": int(total_processed),
        "num_scene_tasks": len(tasks),
        "parallel_backend": args.parallel_backend,
        "num_scene_workers": int(args.num_scene_workers),
        "elapsed_sec": float(time.time() - t_all),
        "sample_index_paths": task_meta["split_sample_index_paths"],
        "num_missing_or_error_rows": int(len(all_rows["missing_rows"])),
        "outputs": {
            "planned_scene_tasks": str(out_dir / "planned_scene_tasks.csv"),
            "progress": str(progress_path),
            "scene_outputs": str(out_dir / args.scene_output_dir),
            "per_sample_method_summary": str(out_dir / "per_sample_method_summary.csv"),
            "per_sample_delta_summary": str(out_dir / "per_sample_delta_summary.csv"),
            "per_split_method_summary": str(out_dir / "per_split_method_summary.csv"),
            "per_scene_method_summary": str(out_dir / "per_scene_method_summary.csv"),
            "per_object_summary": str(out_dir / "per_object_summary.csv") if args.save_per_object_rows else None,
            "rank_eval_rows": str(out_dir / "rank_eval_rows.csv") if args.save_rank_rows else None,
            "grasp_component_rows": str(out_dir / "grasp_component_rows.csv") if args.save_grasp_component_rows else None,
            "endpoint_variable_wide": str(out_dir / "endpoint_variable_wide.csv") if args.analyze_endpoints else None,
            "endpoint_stat_delta_long": str(out_dir / "endpoint_stat_delta_long.csv") if args.analyze_endpoints else None,
            "top_method2_worse_by_AP": str(out_dir / "top_method2_worse_by_AP.csv"),
            "top_method2_better_by_AP": str(out_dir / "top_method2_better_by_AP.csv"),
            "missing_or_error_rows": str(out_dir / "missing_or_error_rows.csv"),
        },
    }
    with (out_dir / "batch_diagnosis_overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, default=str)

    print(json.dumps(overview, indent=2, default=str), flush=True)
    if total_processed == 0:
        print("\n[warn] processed_pair_samples is 0. Check missing_or_error_rows.csv and planned_scene_tasks.csv first.", flush=True)
    print(f"\nSaved batch diagnostics to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
