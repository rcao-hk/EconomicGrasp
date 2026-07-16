#!/usr/bin/env python3
"""Detailed evaluator diagnostics for CVA joint-utility scorer dumps.

The script supports two levels:

FAST (default)
    Calls the official GraspNet ``eval_grasp`` routine.  This exports:
      * top-1 / top-10 / top-50 collision-or-empty
      * top-1 / top-10 / top-50 evaluator-fail
      * top-10 success@0.8 (and success@0.4)
      * official AP / AP@0.4 / AP@0.8
      * selected-pool evaluator-oracle AP and ranking gap
      * one row per retained evaluator candidate

FULL-POOL (``--full_pool_eval 1``)
    Evaluates every post-NMS candidate in the saved dump before the official
    per-object Top-10 and global Top-50 admission steps.  It additionally
    exports:
      * per-object good-candidate admission at friction <=0.8 / <=0.4
      * candidate-level good admission recall
      * object-level "a good candidate existed and at least one was admitted"
      * raw-pool oracle AP and admission gap

The standard joint-utility dump is already post model-free collision filtering,
so full-pool admission measures scorer/NMS/evaluator admission among candidates
that survived deployment-time filtering.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from graspnetAPI import GraspGroup, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    collision_detection,
    compute_closest_points,
    create_table_points,
    eval_grasp,
    get_grasp_score,
    transform_points,
    voxel_sample_points,
)

try:
    from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D
    from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
except Exception:  # full-pool official mode still works without these direct imports
    PointGraspMetrics3D = None
    GraspQualityConfigFactory = None


FRICTIONS = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
SCENE_RANGES = {
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CVA joint-utility rank/admission diagnostics.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dump_dir", required=True, help="Joint-utility GraspNet dump root for one split.")
    p.add_argument("--camera", default="realsense")
    p.add_argument("--split", required=True, choices=sorted(SCENE_RANGES))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_width", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threads_per_worker", type=int, default=1)
    p.add_argument("--mp_start_method", choices=["fork", "spawn", "forkserver"], default="spawn")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--full_pool_eval", type=int, default=0,
                   help="Evaluate every post-NMS candidate to measure admission (slow).")
    p.add_argument("--full_pool_max_per_object", type=int, default=0,
                   help="0=all post-NMS candidates; >0 caps candidates/object by predicted score (approximate).")
    p.add_argument("--eval_chunk", type=int, default=512)
    p.add_argument("--fc_mode", choices=["official", "reuse_contacts", "reuse_contacts_binary"],
                   default="reuse_contacts")
    p.add_argument("--fc_verify_n", type=int, default=0)
    p.add_argument("--strict", type=int, default=1)
    return p.parse_args()


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _save_json(payload: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(payload)), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _save_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted(set().union(*(set(r.keys()) for r in rows)))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)


def _set_threads(n: int) -> None:
    n = max(1, int(n))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(n)


def _clip_grasp_widths(gg: GraspGroup, max_width: float) -> GraspGroup:
    if len(gg):
        gg.widths = np.clip(gg.widths, 0.0, float(max_width))
    return gg


def _accuracy_from_scores(scores: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    acc = np.zeros((top_k, len(FRICTIONS)), dtype=np.float32)
    if scores.size == 0:
        return acc
    for j, mu in enumerate(FRICTIONS):
        ok = ((scores > 0.0) & (scores <= mu)).astype(np.float32)
        prefix = np.cumsum(ok)
        for k in range(1, top_k + 1):
            n = min(k, scores.size)
            acc[k - 1, j] = (float(prefix[n - 1]) if n else 0.0) / float(k)
    return acc


def _utility(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    return (((s[:, None] > 0.0) & (s[:, None] <= FRICTIONS[None, :]))
            .astype(np.float32).mean(axis=1))


def _topk_stats(scores: np.ndarray, collision: np.ndarray, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = np.asarray(scores).reshape(-1)
    c = np.asarray(collision, dtype=bool).reshape(-1)
    for k in [1, 10, 50]:
        n = min(k, len(s))
        if n == 0:
            for key in ["collision", "eval_fail", "success04", "success08"]:
                out[f"{prefix}_top{k}_{key}"] = 0.0
            continue
        ss = s[:n]
        cc = c[:n]
        out[f"{prefix}_top{k}_collision"] = float(np.mean(cc.astype(np.float32)))
        out[f"{prefix}_top{k}_eval_fail"] = float(np.mean(ss <= 0.0))
        out[f"{prefix}_top{k}_success04"] = float(np.mean((ss > 0.0) & (ss <= 0.4)))
        out[f"{prefix}_top{k}_success08"] = float(np.mean((ss > 0.0) & (ss <= 0.8)))
    return out


def _grasp_params(row: np.ndarray) -> Dict[str, Any]:
    r = np.asarray(row, dtype=np.float32)
    return {
        "pred_score": float(r[0]),
        "width": float(r[1]),
        "height": float(r[2]),
        "depth": float(r[3]),
        "tx": float(r[13]),
        "ty": float(r[14]),
        "tz": float(r[15]),
    }


def _discover_samples(dump_dir: str, camera: str, split: str, max_samples: int) -> List[Tuple[int, int, str]]:
    allowed = set(SCENE_RANGES[split])
    samples: List[Tuple[int, int, str]] = []
    for path in glob.glob(os.path.join(os.path.abspath(dump_dir), "scene_*", camera, "*.npy")):
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        try:
            sid = int(scene_name.split("_")[-1])
            aid = int(os.path.splitext(os.path.basename(path))[0])
        except ValueError:
            continue
        if sid not in allowed:
            continue
        samples.append((sid, aid, path))
    samples.sort(key=lambda x: (x[0], x[1]))
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def _official_selection_indices(conf: np.ndarray, assigned: np.ndarray, num_obj: int, top_k: int) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """Mirror official NMS-output admission: per-object top10 + global score cutoff."""
    pre: List[np.ndarray] = []
    for obj in range(num_obj):
        ids = np.flatnonzero(assigned == obj)
        if len(ids):
            order = np.argsort(-conf[ids], kind="stable")
            ids = ids[order[: min(10, len(order))]]
        pre.append(ids)
    nonempty = [x for x in pre if len(x)]
    if not nonempty:
        return [np.zeros(0, dtype=np.int64) for _ in range(num_obj)], np.zeros(0, dtype=np.int64), float("inf")
    pooled = np.concatenate(nonempty)
    order = np.argsort(-conf[pooled], kind="stable")
    kth = min(top_k - 1, len(order) - 1)
    threshold = float(conf[pooled[order[kth]]])
    selected_by_obj = [ids[conf[ids] >= threshold] if len(ids) else ids for ids in pre]
    selected = np.concatenate([x for x in selected_by_obj if len(x)]) if any(len(x) for x in selected_by_obj) else np.zeros(0, dtype=np.int64)
    return selected_by_obj, selected, threshold


class FullPoolEvaluator:
    def __init__(self, root: str, camera: str, split: str, chunk: int,
                 fc_mode: str, fc_verify_n: int, strict: bool) -> None:
        self.eval = GraspNetEval(root, camera, split=split)
        self.config = get_config()
        self.chunk = max(1, int(chunk))
        self.fc_mode = str(fc_mode)
        self.fc_verify_n = max(0, int(fc_verify_n))
        self.strict = bool(strict)
        self.table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        self.fc_desc = np.asarray([1.2, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
        self.fc_asc = self.fc_desc[::-1].copy()
        self.fc_configs: Dict[float, Any] = {}
        if self.fc_mode != "official":
            if PointGraspMetrics3D is None or GraspQualityConfigFactory is None:
                raise ImportError("Direct DexNet imports unavailable; use --fc_mode official")
        for mu in self.fc_desc:
            key = round(float(mu), 2)
            self.config["metrics"]["force_closure"]["friction_coef"] = key
            self.fc_configs[key] = GraspQualityConfigFactory.create_config(
                self.config["metrics"]["force_closure"]
            )

    def _quality_contacts(self, grasp: Any, obj: Any, contacts: Sequence[Any], mu: float) -> bool:
        return bool(PointGraspMetrics3D.grasp_quality(
            grasp, obj, self.fc_configs[round(float(mu), 2)], contacts=contacts
        ))

    def _score(self, grasp: Any, obj: Any) -> float:
        if self.fc_mode == "official":
            return float(get_grasp_score(grasp, obj, self.fc_desc, self.fc_configs))
        base_cfg = self.fc_configs[round(float(self.fc_desc[0]), 2)]
        found, contacts = grasp.close_fingers(
            obj, check_approach=bool(getattr(base_cfg, "check_approach", False)), vis=False
        )
        if not found:
            return -1.0
        use_binary = self.fc_mode == "reuse_contacts_binary" and len(contacts) == 2
        if use_binary:
            lo, hi = 0, len(self.fc_asc)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._quality_contacts(grasp, obj, contacts, float(self.fc_asc[mid])):
                    hi = mid
                else:
                    lo = mid + 1
            return -1.0 if lo >= len(self.fc_asc) else round(float(self.fc_asc[lo]), 2)
        prev = False
        for i, mu in enumerate(self.fc_desc):
            current = self._quality_contacts(grasp, obj, contacts, float(mu))
            if prev and not current:
                return round(float(self.fc_desc[i - 1]), 2)
            if current and i == len(self.fc_desc) - 1:
                return round(float(mu), 2)
            if i == 0 and not current:
                return -1.0
            prev = current
        return -1.0

    def evaluate_all(self, grasps: np.ndarray, models_obj: Sequence[np.ndarray], dexmodels: Sequence[Any],
                     poses: Sequence[np.ndarray], camera_pose: np.ndarray, align_mat: np.ndarray,
                     max_per_object: int) -> Dict[str, Any]:
        n = len(grasps)
        models_cam = [transform_points(m, poses[i]) for i, m in enumerate(models_obj)]
        scene = np.concatenate(models_cam, axis=0)
        seg = np.concatenate([np.full(len(m), i, dtype=np.int64) for i, m in enumerate(models_cam)])
        nearest = compute_closest_points(grasps[:, 13:16], scene)
        assigned = seg[nearest]

        keep = np.ones(n, dtype=bool)
        if max_per_object > 0:
            keep[:] = False
            for obj in range(len(models_cam)):
                ids = np.flatnonzero(assigned == obj)
                if len(ids):
                    ids = ids[np.argsort(-grasps[ids, 0], kind="stable")[:max_per_object]]
                    keep[ids] = True
            grasps = grasps[keep]
            assigned = assigned[keep]

        table_cam = transform_points(self.table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        scene_table = np.concatenate([scene, table_cam], axis=0)
        collision = np.zeros(len(grasps), dtype=bool)
        empty = np.zeros(len(grasps), dtype=bool)
        friction = np.full(len(grasps), -1.0, dtype=np.float32)
        verify_left = self.fc_verify_n

        for obj in range(len(models_cam)):
            obj_ids = np.flatnonzero(assigned == obj)
            for start in range(0, len(obj_ids), self.chunk):
                ids = obj_ids[start:start + self.chunk]
                if not len(ids):
                    continue
                coll_l, empty_l, dex_l = collision_detection(
                    [grasps[ids]], [models_cam[obj]], [dexmodels[obj]], [poses[obj]],
                    scene_table, outlier=0.05, return_dexgrasps=True,
                )
                c = np.asarray(coll_l[0], dtype=bool)
                e = np.asarray(empty_l[0], dtype=bool)
                collision[ids] = c
                empty[ids] = e
                for local, global_id in enumerate(ids):
                    if c[local] or dex_l[0][local] is None:
                        continue
                    score = self._score(dex_l[0][local], dexmodels[obj])
                    friction[global_id] = score
                    if self.fc_mode != "official" and verify_left > 0:
                        official = float(get_grasp_score(
                            dex_l[0][local], dexmodels[obj], self.fc_desc, self.fc_configs
                        ))
                        verify_left -= 1
                        if not np.isclose(score, official, atol=1e-6, rtol=0.0):
                            msg = f"Force-closure mismatch optimized={score} official={official}"
                            if self.strict:
                                raise RuntimeError(msg)
                            print(f"[WARN] {msg}", flush=True)
        return {
            "grasps": grasps,
            "assigned": assigned,
            "collision": collision,
            "empty": empty,
            "friction": friction,
            "capped": bool(max_per_object > 0),
        }


_WORKER: Dict[str, Any] = {}


def _worker_init(args_dict: Dict[str, Any]) -> None:
    _set_threads(int(args_dict["threads_per_worker"]))
    evaluator = GraspNetEval(args_dict["dataset_root"], args_dict["camera"], split=args_dict["split"])
    full = None
    if bool(args_dict["full_pool_eval"]):
        full = FullPoolEvaluator(
            args_dict["dataset_root"], args_dict["camera"], args_dict["split"],
            args_dict["eval_chunk"], args_dict["fc_mode"], args_dict["fc_verify_n"],
            bool(args_dict["strict"]),
        )
    _WORKER.clear()
    _WORKER.update({"args": args_dict, "evaluator": evaluator, "full": full, "config": get_config(),
                    "table": create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)})


def _fast_eval(path: str, models_sampled: Sequence[np.ndarray], dexmodels: Sequence[Any], poses: Sequence[np.ndarray],
               table_cam: np.ndarray, obj_ids: Sequence[int], scene_id: int, anno_id: int) -> Dict[str, Any]:
    args = _WORKER["args"]
    gg = _clip_grasp_widths(GraspGroup().from_npy(path), args["max_width"])
    if len(gg) == 0:
        return {"sample": {"scene_id": scene_id, "anno_id": anno_id}, "rank": [], "object": []}
    grasp_list, score_list, collision_list = eval_grasp(
        gg, models_sampled, dexmodels, poses, _WORKER["config"], table=table_cam,
        voxel_size=0.008, TOP_K=args["top_k"],
    )
    rows: List[Dict[str, Any]] = []
    object_rows: List[Dict[str, Any]] = []
    flat_g, flat_s, flat_c, flat_o, flat_orank = [], [], [], [], []
    for obj, grasps in enumerate(grasp_list):
        s = np.asarray(score_list[obj], dtype=np.float32)
        c = np.asarray(collision_list[obj], dtype=bool)
        g = np.asarray(grasps, dtype=np.float32)
        order = np.argsort(-g[:, 0], kind="stable") if len(g) else np.zeros(0, dtype=np.int64)
        good08 = (s > 0) & (s <= 0.8)
        object_rows.append({
            "scene_id": scene_id, "anno_id": anno_id, "local_object_idx": obj,
            "object_id": int(obj_ids[obj]) if obj < len(obj_ids) else obj,
            "selected_pool_count": int(len(g)),
            "selected_good08_count": int(good08.sum()),
            "selected_has_good08": int(bool(good08.any())),
            "full_pool_available": 0,
        })
        for rank_local, j in enumerate(order, start=1):
            flat_g.append(g[j]); flat_s.append(s[j]); flat_c.append(c[j]); flat_o.append(obj); flat_orank.append(rank_local)
    if flat_g:
        g = np.asarray(flat_g); s = np.asarray(flat_s); c = np.asarray(flat_c); o = np.asarray(flat_o); orank = np.asarray(flat_orank)
        official_order = np.argsort(-g[:, 0], kind="stable")
        util = _utility(s)
        oracle_order = np.lexsort((-g[:, 0], -util))
        official_acc = _accuracy_from_scores(s[official_order], args["top_k"])
        oracle_acc = _accuracy_from_scores(s[oracle_order], args["top_k"])
        oracle_rank = np.empty(len(g), dtype=np.int64); oracle_rank[oracle_order] = np.arange(1, len(g)+1)
        for global_rank, j in enumerate(official_order, start=1):
            row = {
                "scene_id": scene_id, "anno_id": anno_id,
                "local_object_idx": int(o[j]),
                "object_id": int(obj_ids[int(o[j])]) if int(o[j]) < len(obj_ids) else int(o[j]),
                "object_selected_rank": int(orank[j]),
                "global_rank_pred": global_rank,
                "global_rank_oracle": int(oracle_rank[j]),
                "eval_friction_score": float(s[j]),
                "evaluator_utility": float(util[j]),
                "collision_or_empty": int(bool(c[j])),
                "eval_fail": int(s[j] <= 0),
                "success04": int((s[j] > 0) and (s[j] <= 0.4)),
                "success08": int((s[j] > 0) and (s[j] <= 0.8)),
            }
            row.update(_grasp_params(g[j])); rows.append(row)
        sample = {
            "scene_id": scene_id, "anno_id": anno_id,
            "official_ap": float(official_acc.mean()),
            "official_ap04": float(official_acc[:, 1].mean()),
            "official_ap08": float(official_acc[:, 3].mean()),
            "selected_pool_oracle_ap": float(oracle_acc.mean()),
            "selected_pool_oracle_ap04": float(oracle_acc[:, 1].mean()),
            "selected_pool_oracle_ap08": float(oracle_acc[:, 3].mean()),
            "selected_pool_oracle_gap": float(oracle_acc.mean() - official_acc.mean()),
            "selected_pool_count": int(len(g)),
            **_topk_stats(s[official_order], c[official_order], "official"),
        }
    else:
        sample = {"scene_id": scene_id, "anno_id": anno_id, "official_ap": 0.0,
                  "selected_pool_oracle_ap": 0.0, "selected_pool_oracle_gap": 0.0,
                  **_topk_stats(np.zeros(0), np.zeros(0, dtype=bool), "official")}
    return {"sample": sample, "rank": rows, "object": object_rows}


def _full_eval(path: str, models_obj: Sequence[np.ndarray], dexmodels: Sequence[Any], poses: Sequence[np.ndarray],
               camera_pose: np.ndarray, align_mat: np.ndarray, obj_ids: Sequence[int], scene_id: int, anno_id: int) -> Dict[str, Any]:
    args = _WORKER["args"]
    gg = _clip_grasp_widths(GraspGroup().from_npy(path), args["max_width"])
    gg = gg.nms(0.03, 30.0 / 180.0 * math.pi)
    arr = np.asarray(gg.grasp_group_array, dtype=np.float32)
    if len(arr) == 0:
        return {"sample": {"scene_id": scene_id, "anno_id": anno_id}, "rank": [], "object": []}
    result = _WORKER["full"].evaluate_all(
        arr, models_obj, dexmodels, poses, camera_pose, align_mat,
        int(args["full_pool_max_per_object"]),
    )
    g = result["grasps"]; assigned = result["assigned"]; c = result["collision"]; e = result["empty"]; s = result["friction"]
    conf = g[:, 0]
    num_obj = len(models_obj)
    selected_by_obj, selected, threshold = _official_selection_indices(conf, assigned, num_obj, args["top_k"])
    pred_order = selected[np.argsort(-conf[selected], kind="stable")] if len(selected) else selected
    util = _utility(s)
    selected_oracle_order = selected[np.lexsort((-conf[selected], -util[selected]))] if len(selected) else selected

    # Raw-pool oracle: evaluator utility chooses each object's best 10, then global best 50.
    raw_oracle_obj: List[np.ndarray] = []
    for obj in range(num_obj):
        ids = np.flatnonzero(assigned == obj)
        if len(ids):
            order = np.lexsort((-conf[ids], -util[ids]))
            ids = ids[order[: min(10, len(order))]]
        raw_oracle_obj.append(ids)
    pooled = np.concatenate([x for x in raw_oracle_obj if len(x)]) if any(len(x) for x in raw_oracle_obj) else np.zeros(0, dtype=np.int64)
    raw_oracle_order = pooled[np.lexsort((-conf[pooled], -util[pooled]))][:args["top_k"]] if len(pooled) else pooled

    official_acc = _accuracy_from_scores(s[pred_order], args["top_k"])
    selected_oracle_acc = _accuracy_from_scores(s[selected_oracle_order], args["top_k"])
    raw_oracle_acc = _accuracy_from_scores(s[raw_oracle_order], args["top_k"])

    selected_set = set(int(x) for x in selected.tolist())
    selected_orank: Dict[int, int] = {}
    for obj, ids in enumerate(selected_by_obj):
        order = ids[np.argsort(-conf[ids], kind="stable")] if len(ids) else ids
        for r, idx in enumerate(order, start=1):
            selected_orank[int(idx)] = r
    oracle_rank = {int(idx): r for r, idx in enumerate(selected_oracle_order, start=1)}
    rank_rows: List[Dict[str, Any]] = []
    for r, idx in enumerate(pred_order, start=1):
        obj = int(assigned[idx])
        row = {
            "scene_id": scene_id, "anno_id": anno_id,
            "local_object_idx": obj,
            "object_id": int(obj_ids[obj]) if obj < len(obj_ids) else obj,
            "object_selected_rank": int(selected_orank.get(int(idx), -1)),
            "global_rank_pred": r,
            "global_rank_oracle": int(oracle_rank.get(int(idx), -1)),
            "eval_friction_score": float(s[idx]),
            "evaluator_utility": float(util[idx]),
            "collision_or_empty": int(bool(c[idx])), "empty": int(bool(e[idx])),
            "eval_fail": int(s[idx] <= 0),
            "success04": int((s[idx] > 0) and (s[idx] <= 0.4)),
            "success08": int((s[idx] > 0) and (s[idx] <= 0.8)),
        }
        row.update(_grasp_params(g[idx])); rank_rows.append(row)

    object_rows: List[Dict[str, Any]] = []
    for obj in range(num_obj):
        raw_ids = np.flatnonzero(assigned == obj)
        admitted = np.asarray(selected_by_obj[obj], dtype=np.int64)
        raw_good08 = raw_ids[(s[raw_ids] > 0) & (s[raw_ids] <= 0.8)]
        adm_good08 = admitted[(s[admitted] > 0) & (s[admitted] <= 0.8)] if len(admitted) else admitted
        raw_good04 = raw_ids[(s[raw_ids] > 0) & (s[raw_ids] <= 0.4)]
        adm_good04 = admitted[(s[admitted] > 0) & (s[admitted] <= 0.4)] if len(admitted) else admitted
        best_raw = float(util[raw_ids].max()) if len(raw_ids) else 0.0
        best_adm = float(util[admitted].max()) if len(admitted) else 0.0
        object_rows.append({
            "scene_id": scene_id, "anno_id": anno_id, "local_object_idx": obj,
            "object_id": int(obj_ids[obj]) if obj < len(obj_ids) else obj,
            "raw_nms_count": int(len(raw_ids)), "selected_pool_count": int(len(admitted)),
            "raw_good08_count": int(len(raw_good08)), "admitted_good08_count": int(len(adm_good08)),
            "good08_candidate_admission_recall": float(len(adm_good08) / len(raw_good08)) if len(raw_good08) else np.nan,
            "raw_has_good08": int(len(raw_good08) > 0), "admitted_has_good08": int(len(adm_good08) > 0),
            "good08_object_admission_success": int(len(raw_good08) > 0 and len(adm_good08) > 0),
            "raw_good04_count": int(len(raw_good04)), "admitted_good04_count": int(len(adm_good04)),
            "good04_candidate_admission_recall": float(len(adm_good04) / len(raw_good04)) if len(raw_good04) else np.nan,
            "raw_has_good04": int(len(raw_good04) > 0), "admitted_has_good04": int(len(adm_good04) > 0),
            "good04_object_admission_success": int(len(raw_good04) > 0 and len(adm_good04) > 0),
            "best_raw_utility": best_raw, "best_admitted_utility": best_adm,
            "admission_utility_regret": best_raw - best_adm,
            "full_pool_available": 1,
            "full_pool_capped": int(result["capped"]),
        })

    sample = {
        "scene_id": scene_id, "anno_id": anno_id,
        "official_ap": float(official_acc.mean()),
        "official_ap04": float(official_acc[:, 1].mean()),
        "official_ap08": float(official_acc[:, 3].mean()),
        "selected_pool_oracle_ap": float(selected_oracle_acc.mean()),
        "selected_pool_oracle_ap04": float(selected_oracle_acc[:, 1].mean()),
        "selected_pool_oracle_ap08": float(selected_oracle_acc[:, 3].mean()),
        "selected_pool_oracle_gap": float(selected_oracle_acc.mean() - official_acc.mean()),
        "raw_pool_oracle_ap": float(raw_oracle_acc.mean()),
        "raw_pool_oracle_ap04": float(raw_oracle_acc[:, 1].mean()),
        "raw_pool_oracle_ap08": float(raw_oracle_acc[:, 3].mean()),
        "admission_gap": float(raw_oracle_acc.mean() - selected_oracle_acc.mean()),
        "raw_nms_count": int(len(g)), "selected_pool_count": int(len(selected)),
        "global_score_threshold": threshold,
        "full_pool_capped": int(result["capped"]),
        **_topk_stats(s[pred_order], c[pred_order], "official"),
    }
    return {"sample": sample, "rank": rank_rows, "object": object_rows}


def _scene_task(scene_id: int, entries: Sequence[Tuple[int, str]]) -> Dict[str, Any]:
    ev: GraspNetEval = _WORKER["evaluator"]
    args = _WORKER["args"]
    models_obj, dexmodels, _ = ev.get_scene_models(scene_id, ann_id=0)
    models_sampled = [voxel_sample_points(m, 0.008) for m in models_obj]
    sample_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []
    object_rows: List[Dict[str, Any]] = []
    for anno_id, path in entries:
        obj_ids, poses, camera_pose, align_mat = ev.get_model_poses(scene_id, anno_id)
        try:
            obj_ids = [int(x) for x in np.asarray(obj_ids).reshape(-1).tolist()]
        except Exception:
            obj_ids = list(range(len(models_obj)))
        table_cam = transform_points(_WORKER["table"], np.linalg.inv(np.matmul(align_mat, camera_pose)))
        if bool(args["full_pool_eval"]):
            result = _full_eval(path, models_sampled, dexmodels, poses, camera_pose, align_mat,
                                obj_ids, scene_id, anno_id)
        else:
            result = _fast_eval(path, models_sampled, dexmodels, poses, table_cam,
                                obj_ids, scene_id, anno_id)
        sample_rows.append(result["sample"]); rank_rows.extend(result["rank"]); object_rows.extend(result["object"])
    return {"scene_id": scene_id, "sample": sample_rows, "rank": rank_rows, "object": object_rows}


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if np.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else None


def _aggregate(sample_rows: Sequence[Mapping[str, Any]], object_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "split": args.split, "dump_dir": os.path.abspath(args.dump_dir),
        "num_samples": len(sample_rows), "full_pool_eval": bool(args.full_pool_eval),
        "full_pool_max_per_object": int(args.full_pool_max_per_object),
    }
    keys = [
        "official_ap", "official_ap04", "official_ap08", "selected_pool_oracle_ap",
        "selected_pool_oracle_gap", "raw_pool_oracle_ap", "admission_gap",
        "official_top1_collision", "official_top10_collision", "official_top50_collision",
        "official_top1_eval_fail", "official_top10_eval_fail", "official_top50_eval_fail",
        "official_top10_success04", "official_top10_success08", "official_top50_success08",
    ]
    for key in keys:
        value = _mean(sample_rows, key)
        if value is not None:
            summary[key] = value
    full_objects = [r for r in object_rows if int(r.get("full_pool_available", 0)) == 1]
    if full_objects:
        for mu in ["08", "04"]:
            available = [r for r in full_objects if int(r.get(f"raw_has_good{mu}", 0)) == 1]
            summary[f"objects_with_good{mu}_raw"] = len(available)
            summary[f"good{mu}_object_admission_rate"] = (
                float(np.mean([int(r[f"admitted_has_good{mu}"]) for r in available])) if available else None
            )
            raw_good = sum(int(r[f"raw_good{mu}_count"]) for r in full_objects)
            admitted_good = sum(int(r[f"admitted_good{mu}_count"]) for r in full_objects)
            summary[f"good{mu}_candidate_admission_recall"] = (
                float(admitted_good / raw_good) if raw_good else None
            )
        summary["mean_admission_utility_regret"] = _mean(full_objects, "admission_utility_regret")
    return summary


def _shard_paths(output_dir: str, scene_id: int) -> Tuple[str, str, str]:
    root = os.path.join(output_dir, "_scene_shards")
    return (
        os.path.join(root, f"scene_{scene_id:04d}_sample.csv"),
        os.path.join(root, f"scene_{scene_id:04d}_rank.csv"),
        os.path.join(root, f"scene_{scene_id:04d}_object.csv"),
    )


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_scene_shard(output_dir: str, result: Mapping[str, Any]) -> None:
    scene_id = int(result["scene_id"])
    sample_p, rank_p, object_p = _shard_paths(output_dir, scene_id)
    _save_csv(result["sample"], sample_p)
    _save_csv(result["rank"], rank_p)
    _save_csv(result["object"], object_p)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(os.path.abspath(args.dump_dir), "_joint_utility_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    samples = _discover_samples(args.dump_dir, args.camera, args.split, args.max_samples)
    if not samples:
        raise FileNotFoundError(f"No dump samples under {args.dump_dir}/scene_*/{args.camera}/*.npy")
    grouped: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for sid, aid, path in samples:
        grouped[sid].append((aid, path))
    tasks_all = [(sid, sorted(entries)) for sid, entries in sorted(grouped.items())]

    sample_path = os.path.join(output_dir, "per_sample_summary.csv")
    rank_path = os.path.join(output_dir, "rank_eval_rows.csv")
    object_path = os.path.join(output_dir, "per_object_admission.csv")
    summary_path = os.path.join(output_dir, "diagnostic_summary.json")
    if args.overwrite:
        for p in [sample_path, rank_path, object_path, summary_path]:
            if os.path.exists(p): os.remove(p)

    args_dict = vars(args).copy()
    sample_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []
    object_rows: List[Dict[str, Any]] = []
    tasks: List[Tuple[int, List[Tuple[int, str]]]] = []
    for task in tasks_all:
        sid = int(task[0])
        sp, rp, op = _shard_paths(output_dir, sid)
        if args.resume and os.path.isfile(sp) and os.path.isfile(rp) and os.path.isfile(op):
            sample_rows.extend(_read_csv(sp)); rank_rows.extend(_read_csv(rp)); object_rows.extend(_read_csv(op))
            print(f"[RESUME] scene={sid:04d}", flush=True)
        else:
            tasks.append(task)

    started = time.time()
    if args.num_workers <= 1:
        _worker_init(args_dict)
        for task in tasks:
            result = _scene_task(*task)
            _save_scene_shard(output_dir, result)
            sample_rows.extend(result["sample"]); rank_rows.extend(result["rank"]); object_rows.extend(result["object"])
            print(f"[DIAG] scene={task[0]:04d} samples={len(sample_rows)}/{len(samples)}", flush=True)
    else:
        ctx = mp.get_context(args.mp_start_method)
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx,
                                 initializer=_worker_init, initargs=(args_dict,)) as ex:
            futures = {ex.submit(_scene_task, *task): task[0] for task in tasks}
            for fut in as_completed(futures):
                result = fut.result()
                _save_scene_shard(output_dir, result)
                sample_rows.extend(result["sample"]); rank_rows.extend(result["rank"]); object_rows.extend(result["object"])
                print(f"[DIAG] scene={futures[fut]:04d} samples={len(sample_rows)}/{len(samples)}", flush=True)

    sample_rows.sort(key=lambda r: (int(r["scene_id"]), int(r["anno_id"])))
    rank_rows.sort(key=lambda r: (int(r["scene_id"]), int(r["anno_id"]), int(r["global_rank_pred"])))
    object_rows.sort(key=lambda r: (int(r["scene_id"]), int(r["anno_id"]), int(r["local_object_idx"])))
    _save_csv(sample_rows, sample_path); _save_csv(rank_rows, rank_path); _save_csv(object_rows, object_path)
    summary = _aggregate(sample_rows, object_rows, args)
    summary["elapsed_minutes"] = (time.time() - started) / 60.0
    _save_json(summary, summary_path)
    _save_csv([summary], os.path.join(output_dir, "diagnostic_summary.csv"))
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    print(f"[SAVE] {sample_path}\n[SAVE] {rank_path}\n[SAVE] {object_path}\n[SAVE] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
