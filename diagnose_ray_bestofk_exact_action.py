#!/usr/bin/env python3
"""Frozen K-center exact-action diagnostic for RGB-only CVA-CDF.

For every native image-FPS query, construct fixed camera-z hypotheses around the
Stage-1 predicted center, reread the frozen local CVA evidence at each physical
center, and decode one grasp per center. No network parameter is trained.

Primary comparisons per ray:
  native       : zero-offset Stage-1 grasp
  raw_selected : best center by the frozen decoded CDF score
  oracle       : best center by post-hoc exact CAD/DexNet action utility

The oracle is diagnostic only.  It is never fed back into the model.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="test_seen", choices=("test_seen", "test_similar", "test_novel"))
    p.add_argument("--camera", default="realsense")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_points", type=int, default=20000)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--pose_depth_mode", default="global_film", choices=("none", "global_film", "ray_gravity_film"))
    p.add_argument("--offsets_mm", default="-40,-20,-10,0,10,20,40")
    p.add_argument("--query_eval_num", type=int, default=128)
    p.add_argument("--query_eval_mode", default="topk_uniform", choices=("all", "topk", "uniform", "topk_uniform"))
    p.add_argument("--fc_mode", default="reuse_contacts", choices=("reuse_contacts", "official"))
    p.add_argument("--verify_n", type=int, default=0)
    p.add_argument("--noop_check_samples", type=int, default=2)
    p.add_argument("--noop_atol", type=float, default=5e-5)
    p.add_argument("--profile_timing", action="store_true")
    p.add_argument("--save_raw_grasps", action="store_true")
    return p.parse_args()


ARGS = _parse_args()

# Reuse the audited center-decoupling helpers without letting its legacy parser
# consume this diagnostic's arguments.
_base_argv = [
    sys.argv[0],
    "--dataset_root", ARGS.dataset_root,
    "--checkpoint_path", ARGS.checkpoint_path,
    "--output_dir", ARGS.output_dir,
    "--split", ARGS.split,
    "--camera", ARGS.camera,
    "--sample_interval", str(ARGS.sample_interval),
    "--max_samples", str(ARGS.max_samples),
    "--num_workers", str(ARGS.num_workers),
    "--num_points", str(ARGS.num_points),
    "--min_depth", str(ARGS.min_depth),
    "--max_depth", str(ARGS.max_depth),
    "--bin_num", str(ARGS.bin_num),
    "--pose_depth_mode", ARGS.pose_depth_mode,
    "--fc_mode", ARGS.fc_mode,
    "--verify_n", str(ARGS.verify_n),
    "--query_eval_num", str(ARGS.query_eval_num),
    "--query_eval_mode", ARGS.query_eval_mode,
]
sys.argv = _base_argv

import diagnose_cva_center_decoupling as base
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from utils.cva_center_decoupling import (
    assert_native_reread_equivalent,
    rerun_cdf_with_read_center,
)
from utils.ray_bestofk_diagnostic import (
    build_ray_center_hypotheses,
    friction_utility,
    gather_kn,
    parse_offsets_mm,
    select_exact_oracle,
    select_raw_score,
)


def _success(friction, threshold):
    f = np.asarray(friction, dtype=np.float32)
    return np.isfinite(f) & (f > 0.0) & (f <= float(threshold) + 1e-6)


def _mean_bool(x):
    x = np.asarray(x, dtype=np.float32)
    return float(x.mean()) if x.size else float("nan")


def _evaluate_grid(evaluator, scene_id, anno_id, grasps_kn17, valid_kn):
    K, N, D = grasps_kn17.shape
    if D != 17 or valid_kn.shape != (K, N):
        raise ValueError("Malformed KxN grasp grid.")
    flat = grasps_kn17.reshape(K * N, 17)
    valid_flat = valid_kn.reshape(K * N)
    ids = np.flatnonzero(valid_flat)

    friction = np.full(K * N, np.nan, dtype=np.float32)
    assigned = np.full(K * N, -1, dtype=np.int64)
    collision = np.full(K * N, -1, dtype=np.int8)
    pure_collision = np.full(K * N, -1, dtype=np.int8)
    empty = np.full(K * N, -1, dtype=np.int8)
    t0 = time.perf_counter()
    result = evaluator.evaluate(scene_id, anno_id, flat[ids])
    eval_sec = time.perf_counter() - t0
    friction[ids] = result.friction
    assigned[ids] = result.assigned_obj
    collision[ids] = result.collision_or_empty.astype(np.int8)
    pure_collision[ids] = result.pure_collision.astype(np.int8)
    empty[ids] = result.empty.astype(np.int8)

    matrices = {
        "friction": friction.reshape(K, N),
        "assigned_obj": assigned.reshape(K, N),
        "collision_or_empty": collision.reshape(K, N),
        "pure_collision": pure_collision.reshape(K, N),
        "empty": empty.reshape(K, N),
    }
    timing = {
        "eval_sec": float(eval_sec),
        "collision_sec": float(result.stats.get("collision_sec", 0.0)),
        "force_closure_sec": float(result.stats.get("force_closure_sec", 0.0)),
        "num_valid_actions": int(ids.size),
    }
    return matrices, timing


def _selected_metrics(mats, k_idx):
    f = gather_kn(mats["friction"], k_idx)
    return {
        "friction": f,
        "utility": friction_utility(f),
        "success04": _success(f, 0.4),
        "success08": _success(f, 0.8),
        "collision_or_empty": gather_kn(mats["collision_or_empty"], k_idx),
        "pure_collision": gather_kn(mats["pure_collision"], k_idx),
        "empty": gather_kn(mats["empty"], k_idx),
        "assigned_obj": gather_kn(mats["assigned_obj"], k_idx),
    }


def _write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows generated for {path}")
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _aggregate_selected(rows, prefix):
    return {
        "success04": float(np.mean([r[f"{prefix}_success04"] for r in rows])),
        "success08": float(np.mean([r[f"{prefix}_success08"] for r in rows])),
        "utility": float(np.mean([r[f"{prefix}_utility"] for r in rows])),
        "collision_or_empty": float(np.mean([r[f"{prefix}_collision_or_empty"] for r in rows])),
        "pure_collision": float(np.mean([r[f"{prefix}_pure_collision"] for r in rows])),
        "empty": float(np.mean([r[f"{prefix}_empty"] for r in rows])),
    }


def main():
    offsets = parse_offsets_mm(ARGS.offsets_mm)
    zero_candidates = [i for i, x in enumerate(offsets) if abs(x) < 1e-9]
    if len(zero_candidates) != 1:
        raise ValueError("Exactly one zero-offset native control is required.")
    zero_k = zero_candidates[0]

    base._configure_legacy_cfg()
    out_root = Path(ARGS.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = GraspNetMultiDataset(
        ARGS.dataset_root,
        split=ARGS.split,
        camera=ARGS.camera,
        num_points=ARGS.num_points,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
    )
    indices = base._subset_indices(len(dataset), ARGS.sample_interval, ARGS.max_samples)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        num_workers=ARGS.num_workers, collate_fn=collate_fn,
        pin_memory=False, persistent_workers=(ARGS.num_workers > 0),
    )

    model = economicgrasp_dpt_student(
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=ARGS.pose_depth_mode,
        use_cdf=True,
        vis_dir=None,
    ).to(device)
    base._load_checkpoint_strict(model, ARGS.checkpoint_path)
    model.eval()

    evaluator = ExactGraspNetActionEvaluator(
        ARGS.dataset_root, ARGS.camera, split=ARGS.split,
        fc_mode=ARGS.fc_mode, verify_n=ARGS.verify_n, strict=True,
    )

    per_query = []
    per_sample = []
    timing_sum = defaultdict(float)
    noop_max = defaultdict(float)
    raw_offset_hist = Counter()
    oracle_offset_hist = Counter()
    offset_acc = {float(o): defaultdict(float) for o in offsets}
    offset_count = {float(o): 0 for o in offsets}

    cand_fields = [
        "split", "scene_id", "anno_id", "dataset_idx", "query_id", "eval_query_rank",
        "offset_index", "offset_mm", "candidate_valid", "center_shift_mm", "raw_score",
        "friction", "utility", "success04", "success08", "collision_or_empty",
        "pure_collision", "empty", "assigned_obj", "is_native", "selected_raw", "selected_oracle",
    ]
    cand_file = gzip.open(out_root / "per_candidate.csv.gz", "wt", newline="")
    cand_writer = csv.DictWriter(cand_file, fieldnames=cand_fields)
    cand_writer.writeheader()

    try:
        for local_i, batch in enumerate(loader):
            dataset_idx = indices[local_i]
            batch = base._move_batch(batch, device)
            batch["cva_export_angle_feature"] = False
            batch["cva_compute_diagnostics"] = False
            batch["geometry_compute_diagnostics"] = False

            if ARGS.profile_timing:
                base._cuda_sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                ep = model(batch)
                e00_t = pred_decode_center_view_angle(ep, use_cdf=True)
            if ARGS.profile_timing:
                base._cuda_sync(device)
            native_sec = time.perf_counter() - t0

            native_xyz = ep["kview_base_xyz_graspable"].float()
            token_idx = ep["kview_base_token_sel_idx"].long()
            depth_map = ep["depth_map_used_for_geometry"]
            _, _, H, W = depth_map.shape
            centers, center_valid = build_ray_center_hypotheses(
                native_xyz, token_idx, ep["K"], (H, W), offsets,
                ARGS.min_depth, ARGS.max_depth,
            )
            if not bool(center_valid[zero_k].all()):
                raise RuntimeError("Native zero-offset center unexpectedly invalid.")

            if local_i < max(0, int(ARGS.noop_check_samples)):
                with torch.inference_mode():
                    noop_ep, _ = rerun_cdf_with_read_center(
                        model, ep, read_center=native_xyz, output_center=native_xyz
                    )
                    metrics = assert_native_reread_equivalent(
                        ep, noop_ep, atol=ARGS.noop_atol
                    )
                    noop_decoded = pred_decode_center_view_angle(noop_ep, use_cdf=True)[0]
                decoded_diff = float((noop_decoded - e00_t[0]).abs().max().item())
                if decoded_diff > float(ARGS.noop_atol):
                    raise RuntimeError(
                        f"Native no-op decoded replay mismatch {decoded_diff:.3e} > {ARGS.noop_atol:.3e}"
                    )
                metrics["noop_decoded_max_abs"] = decoded_diff
                for k, v in metrics.items():
                    noop_max[k] = max(noop_max[k], float(v))

            if ARGS.profile_timing:
                base._cuda_sync(device)
            t_ray = time.perf_counter()
            grasp_by_k = []
            with torch.inference_mode():
                for k, offset in enumerate(offsets):
                    if k == zero_k:
                        grasp_by_k.append(e00_t[0])
                    else:
                        epk, _ = rerun_cdf_with_read_center(
                            model, ep, read_center=centers[k], output_center=centers[k]
                        )
                        grasp_by_k.append(pred_decode_center_view_angle(epk, use_cdf=True)[0])
            if ARGS.profile_timing:
                base._cuda_sync(device)
            ray_sec = time.perf_counter() - t_ray

            e00 = e00_t[0]
            all_native_eligible = torch.ones(e00.shape[0], dtype=torch.bool, device=e00.device)
            eval_idx = base._select_eval_queries(
                e00, all_native_eligible,
                query_eval_num=ARGS.query_eval_num,
                mode=ARGS.query_eval_mode,
                valid_only=False,
            )
            grasps = torch.stack(
                [g.index_select(0, eval_idx) for g in grasp_by_k], dim=0
            ).detach().cpu().numpy().astype(np.float32)
            valid_kn = center_valid[:, 0].index_select(1, eval_idx).detach().cpu().numpy().astype(bool)
            centers_kn3 = centers[:, 0].index_select(1, eval_idx).detach().cpu().numpy().astype(np.float32)
            native_sel = native_xyz[0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
            center_shift = np.linalg.norm(centers_kn3 - native_sel[None, :, :], axis=-1) * 1000.0

            scene_id = int(batch["scene_idx"].reshape(-1)[0].item())
            anno_id = int(batch["anno_idx"].reshape(-1)[0].item())
            mats, eval_timing = _evaluate_grid(
                evaluator, scene_id, anno_id, grasps, valid_kn
            )
            utility = friction_utility(mats["friction"])
            raw_score = grasps[:, :, 0]
            raw_k = select_raw_score(raw_score, valid_kn)
            oracle_k = select_exact_oracle(utility, raw_score, valid_kn)
            native_k = np.full(len(raw_k), zero_k, dtype=np.int64)

            native_m = _selected_metrics(mats, native_k)
            raw_m = _selected_metrics(mats, raw_k)
            oracle_m = _selected_metrics(mats, oracle_k)
            any04 = np.any(_success(mats["friction"], 0.4) & valid_kn, axis=0)
            any08 = np.any(_success(mats["friction"], 0.8) & valid_kn, axis=0)

            selected_query = eval_idx.detach().cpu().numpy().astype(np.int64)
            for j, qid in enumerate(selected_query):
                row = {
                    "split": ARGS.split, "scene_id": scene_id, "anno_id": anno_id,
                    "dataset_idx": dataset_idx, "query_id": int(qid), "eval_query_rank": j,
                    "num_valid_centers": int(valid_kn[:, j].sum()),
                    "native_offset_mm": float(offsets[zero_k]),
                    "raw_offset_mm": float(offsets[int(raw_k[j])]),
                    "oracle_offset_mm": float(offsets[int(oracle_k[j])]),
                    "raw_matches_oracle": int(raw_k[j] == oracle_k[j]),
                    "raw_changed_from_native": int(raw_k[j] != zero_k),
                    "oracle_changed_from_native": int(oracle_k[j] != zero_k),
                    "any_success04": int(any04[j]), "any_success08": int(any08[j]),
                }
                for name, met, kk in (
                    ("native", native_m, native_k), ("raw", raw_m, raw_k), ("oracle", oracle_m, oracle_k)
                ):
                    row[f"{name}_raw_score"] = float(raw_score[int(kk[j]), j])
                    row[f"{name}_friction"] = float(met["friction"][j])
                    row[f"{name}_utility"] = float(met["utility"][j])
                    row[f"{name}_success04"] = int(met["success04"][j])
                    row[f"{name}_success08"] = int(met["success08"][j])
                    row[f"{name}_collision_or_empty"] = int(met["collision_or_empty"][j])
                    row[f"{name}_pure_collision"] = int(met["pure_collision"][j])
                    row[f"{name}_empty"] = int(met["empty"][j])
                    row[f"{name}_assigned_obj"] = int(met["assigned_obj"][j])
                row["oracle_headroom_utility"] = row["oracle_utility"] - row["native_utility"]
                row["raw_selection_regret"] = row["oracle_utility"] - row["raw_utility"]
                for thr in ("04", "08"):
                    n_ok = bool(row[f"native_success{thr}"])
                    r_ok = bool(row[f"raw_success{thr}"])
                    o_ok = bool(row[f"oracle_success{thr}"])
                    row[f"raw_rescue{thr}"] = int((not n_ok) and r_ok)
                    row[f"raw_harm{thr}"] = int(n_ok and (not r_ok))
                    row[f"oracle_rescue{thr}"] = int((not n_ok) and o_ok)
                    row[f"oracle_harm{thr}"] = int(n_ok and (not o_ok))
                per_query.append(row)

                raw_offset_hist[float(offsets[int(raw_k[j])])] += 1
                oracle_offset_hist[float(offsets[int(oracle_k[j])])] += 1
                for k, offset in enumerate(offsets):
                    valid = bool(valid_kn[k, j])
                    friction = float(mats["friction"][k, j]) if valid else float("nan")
                    cand_writer.writerow({
                        "split": ARGS.split, "scene_id": scene_id, "anno_id": anno_id,
                        "dataset_idx": dataset_idx, "query_id": int(qid), "eval_query_rank": j,
                        "offset_index": k, "offset_mm": float(offset),
                        "candidate_valid": int(valid), "center_shift_mm": float(center_shift[k, j]),
                        "raw_score": float(raw_score[k, j]), "friction": friction,
                        "utility": float(utility[k, j]) if valid else float("nan"),
                        "success04": int(_success(np.asarray([friction]), 0.4)[0]) if valid else -1,
                        "success08": int(_success(np.asarray([friction]), 0.8)[0]) if valid else -1,
                        "collision_or_empty": int(mats["collision_or_empty"][k, j]),
                        "pure_collision": int(mats["pure_collision"][k, j]),
                        "empty": int(mats["empty"][k, j]),
                        "assigned_obj": int(mats["assigned_obj"][k, j]),
                        "is_native": int(k == zero_k), "selected_raw": int(k == raw_k[j]),
                        "selected_oracle": int(k == oracle_k[j]),
                    })
                    if valid:
                        key = float(offset)
                        offset_count[key] += 1
                        offset_acc[key]["success04"] += float(_success(np.asarray([friction]), 0.4)[0])
                        offset_acc[key]["success08"] += float(_success(np.asarray([friction]), 0.8)[0])
                        offset_acc[key]["utility"] += float(utility[k, j])
                        offset_acc[key]["collision_or_empty"] += float(mats["collision_or_empty"][k, j])
                        offset_acc[key]["pure_collision"] += float(mats["pure_collision"][k, j])
                        offset_acc[key]["empty"] += float(mats["empty"][k, j])

            n = len(selected_query)
            per_sample.append({
                "split": ARGS.split, "scene_id": scene_id, "anno_id": anno_id, "dataset_idx": dataset_idx,
                "num_queries": n,
                "native_success08": _mean_bool(native_m["success08"]),
                "raw_success08": _mean_bool(raw_m["success08"]),
                "oracle_success08": _mean_bool(oracle_m["success08"]),
                "any_success08": _mean_bool(any08),
                "native_success04": _mean_bool(native_m["success04"]),
                "raw_success04": _mean_bool(raw_m["success04"]),
                "oracle_success04": _mean_bool(oracle_m["success04"]),
                "any_success04": _mean_bool(any04),
                "native_utility": float(native_m["utility"].mean()),
                "raw_utility": float(raw_m["utility"].mean()),
                "oracle_utility": float(oracle_m["utility"].mean()),
                "raw_match_oracle": float(np.mean(raw_k == oracle_k)),
                "raw_change_rate": float(np.mean(raw_k != zero_k)),
                "oracle_change_rate": float(np.mean(oracle_k != zero_k)),
                "native_forward_sec": native_sec, "ray_reread_sec": ray_sec,
                "exact_eval_sec": eval_timing["eval_sec"],
                "collision_sec": eval_timing["collision_sec"],
                "force_closure_sec": eval_timing["force_closure_sec"],
            })

            timing_sum["native_forward_sec"] += native_sec
            timing_sum["ray_reread_sec"] += ray_sec
            for k, v in eval_timing.items():
                if k != "num_valid_actions":
                    timing_sum[k] += float(v)

            if ARGS.save_raw_grasps:
                raw_dir = out_root / "raw_grids" / f"scene_{scene_id:04d}"
                raw_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    raw_dir / f"{anno_id:04d}.npz",
                    grasps=grasps, valid=valid_kn, offsets_mm=np.asarray(offsets, dtype=np.float32),
                    query_id=selected_query, raw_k=raw_k, oracle_k=oracle_k,
                    friction=mats["friction"], utility=utility,
                )

            if local_i % 20 == 0:
                print(
                    f"[RAY-BESTK] {local_i+1}/{len(indices)} scene={scene_id:04d} anno={anno_id:04d} "
                    f"queries={n} K={len(offsets)} native08={native_m['success08'].mean():.3f} "
                    f"raw08={raw_m['success08'].mean():.3f} oracle08={oracle_m['success08'].mean():.3f} "
                    f"t_eval={eval_timing['eval_sec']:.2f}s",
                    flush=True,
                )
    finally:
        cand_file.close()

    _write_csv(out_root / "per_query.csv", per_query)
    _write_csv(out_root / "per_sample_summary.csv", per_sample)

    offset_rows = []
    total_q = len(per_query)
    for offset in offsets:
        key = float(offset)
        n = offset_count[key]
        row = {
            "offset_mm": key, "num_valid_candidates": n,
            "valid_fraction": float(n / max(total_q, 1)),
            "raw_selected_count": int(raw_offset_hist[key]),
            "raw_selected_fraction": float(raw_offset_hist[key] / max(total_q, 1)),
            "oracle_selected_count": int(oracle_offset_hist[key]),
            "oracle_selected_fraction": float(oracle_offset_hist[key] / max(total_q, 1)),
        }
        for metric in ("success04", "success08", "utility", "collision_or_empty", "pure_collision", "empty"):
            row[metric] = float(offset_acc[key][metric] / max(n, 1))
        offset_rows.append(row)
    _write_csv(out_root / "offset_summary.csv", offset_rows)

    aggregate = {
        "native": _aggregate_selected(per_query, "native"),
        "raw_selected": _aggregate_selected(per_query, "raw"),
        "oracle": _aggregate_selected(per_query, "oracle"),
        "any_success04": float(np.mean([r["any_success04"] for r in per_query])),
        "any_success08": float(np.mean([r["any_success08"] for r in per_query])),
        "raw_matches_oracle": float(np.mean([r["raw_matches_oracle"] for r in per_query])),
        "raw_changed_from_native": float(np.mean([r["raw_changed_from_native"] for r in per_query])),
        "oracle_changed_from_native": float(np.mean([r["oracle_changed_from_native"] for r in per_query])),
        "oracle_headroom_utility": float(np.mean([r["oracle_headroom_utility"] for r in per_query])),
        "raw_selection_regret": float(np.mean([r["raw_selection_regret"] for r in per_query])),
    }
    rescue_harm = {}
    for policy in ("raw", "oracle"):
        for thr in ("04", "08"):
            rescue_harm[f"{policy}_rescue{thr}"] = float(np.mean([r[f"{policy}_rescue{thr}"] for r in per_query]))
            rescue_harm[f"{policy}_harm{thr}"] = float(np.mean([r[f"{policy}_harm{thr}"] for r in per_query]))

    summary = {
        "protocol": "frozen Stage-1 K-center ray exact-action best-of-K diagnostic v1",
        "checkpoint": os.path.abspath(ARGS.checkpoint_path),
        "split": ARGS.split,
        "camera": ARGS.camera,
        "sample_interval": ARGS.sample_interval,
        "offsets_mm": list(offsets),
        "zero_offset_index": zero_k,
        "query_eval_num": ARGS.query_eval_num,
        "query_eval_mode": ARGS.query_eval_mode,
        "num_samples": len(per_sample),
        "num_queries": len(per_query),
        "noop_replay_max": dict(noop_max),
        "aggregate": aggregate,
        "rescue_harm": rescue_harm,
        "raw_offset_hist": {str(k): int(v) for k, v in sorted(raw_offset_hist.items())},
        "oracle_offset_hist": {str(k): int(v) for k, v in sorted(oracle_offset_hist.items())},
        "timing_total_sec": {k: float(v) for k, v in timing_sum.items()},
        "timing_per_sample_sec": {k: float(v / max(len(per_sample), 1)) for k, v in timing_sum.items()},
        "interpretation": {
            "candidate_headroom": "oracle - native",
            "selection_gap": "oracle - raw_selected",
            "oracle_warning": "oracle uses post-hoc CAD/DexNet labels and is diagnostic only",
        },
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
