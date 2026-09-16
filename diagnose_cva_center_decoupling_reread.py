#!/usr/bin/env python3
"""Corrected E10/E11-only rerun for the CVA-CDF center-decoupling study.

This script is intended to repair/re-run only the two variants affected by the
old counterfactual grouping-input mismatch:

  E10: read at reference center, output native center
  E11: read at reference center, output reference center

E00 is still computed internally because it defines the native query set and the
deterministic query-selection policy, but E00/E01 are NOT sent through the
expensive exact-action evaluator. Existing E00/E01 results from the previous run
remain valid and can be joined with this output by
(split, scene_id, anno_id, query_id).

Before evaluating E10/E11, the first ``--noop_check_samples`` frames perform a
native-center reread and require the replayed CDF/width tensors to match the
native outputs within ``--noop_atol``. This fail-fast audit prevents another
silent evidence-map confound.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--split",
        default="test_seen",
        choices=("test_seen", "test_similar", "test_novel"),
    )
    p.add_argument("--camera", default="realsense")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_points", type=int, default=20000)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument(
        "--pose_depth_mode",
        default="global_film",
        choices=("none", "global_film", "ray_gravity_film"),
    )
    p.add_argument("--fc_mode", default="reuse_contacts", choices=("reuse_contacts", "official"))
    p.add_argument("--verify_n", type=int, default=0)
    p.add_argument("--query_eval_num", type=int, default=128)
    p.add_argument(
        "--query_eval_mode",
        default="topk_uniform",
        choices=("all", "topk", "uniform", "topk_uniform"),
    )
    p.add_argument("--eval_valid_only", action="store_true")
    p.add_argument("--profile_timing", action="store_true")
    p.add_argument("--save_raw_grasps", action="store_true")
    p.add_argument(
        "--noop_check_samples",
        type=int,
        default=2,
        help="Number of initial frames for native-center no-op replay audit; 0 disables.",
    )
    p.add_argument(
        "--noop_atol",
        type=float,
        default=5.0e-5,
        help="Maximum allowed absolute CDF/width difference in no-op replay.",
    )
    return p.parse_args()


ARGS = _parse_args()

# Reuse the established dataset/query-selection/checkpoint helpers. The base
# module parses argv at import time, so give it only options it understands.
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
if ARGS.eval_valid_only:
    _base_argv.append("--eval_valid_only")
if ARGS.profile_timing:
    _base_argv.append("--profile_timing")
if ARGS.save_raw_grasps:
    _base_argv.append("--save_raw_grasps")
sys.argv = _base_argv

import diagnose_cva_center_decoupling as base
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactActionEvalResult, ExactGraspNetActionEvaluator
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from utils.cva_center_decoupling import (
    assert_native_reread_equivalent,
    gather_reference_centers_from_depth,
    replace_decoded_translation,
    rerun_cdf_with_read_center,
)

# Base helper imports reset sys.argv; it is no longer needed after this point.
VARIANTS = ("E10", "E11")


def _evaluate_two_once(
    evaluator: ExactGraspNetActionEvaluator,
    scene_id: int,
    anno_id: int,
    arrays: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, ExactActionEvalResult], Dict[str, float]]:
    sizes = [int(arrays[v].shape[0]) for v in VARIANTS]
    if len(set(sizes)) != 1:
        raise RuntimeError(f"E10/E11 query counts differ: {dict(zip(VARIANTS, sizes))}")
    n = sizes[0]
    if n == 0:
        results = {v: evaluator.evaluate(scene_id, anno_id, arrays[v]) for v in VARIANTS}
        return results, {"eval_sec": 0.0, "collision_sec": 0.0, "force_closure_sec": 0.0}

    merged = np.concatenate([arrays[v] for v in VARIANTS], axis=0)
    t0 = time.perf_counter()
    merged_result = evaluator.evaluate(scene_id, anno_id, merged)
    elapsed = time.perf_counter() - t0
    results = {
        "E10": base._slice_eval_result(merged_result, 0, n),
        "E11": base._slice_eval_result(merged_result, n, 2 * n),
    }
    return results, {
        "eval_sec": float(elapsed),
        "collision_sec": float(merged_result.stats.get("collision_sec", 0.0)),
        "force_closure_sec": float(merged_result.stats.get("force_closure_sec", 0.0)),
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows generated for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
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
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=ARGS.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(ARGS.num_workers > 0),
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
        ARGS.dataset_root,
        ARGS.camera,
        split=ARGS.split,
        fc_mode=ARGS.fc_mode,
        verify_n=ARGS.verify_n,
        strict=True,
    )

    per_query: List[dict] = []
    per_sample: List[dict] = []
    timing_sum = defaultdict(float)
    noop_max = defaultdict(float)
    total_native_queries = 0
    total_eval_queries = 0
    total_ref_valid = 0

    for local_i, batch in enumerate(loader):
        dataset_idx = indices[local_i]
        batch = base._move_batch(batch, device)
        batch["cva_export_angle_feature"] = False
        batch["cva_compute_diagnostics"] = False
        batch["geometry_compute_diagnostics"] = False

        if ARGS.profile_timing:
            base._cuda_sync(device)
        t_native = time.perf_counter()
        with torch.inference_mode():
            ep = model(batch)
            e00_t = pred_decode_center_view_angle(ep, use_cdf=True)
        if ARGS.profile_timing:
            base._cuda_sync(device)
        native_sec = time.perf_counter() - t_native

        native_xyz = ep["kview_base_xyz_graspable"].float()
        ref_xyz, ref_valid, ref_depth = gather_reference_centers_from_depth(
            ep, min_depth=ARGS.min_depth, max_depth=ARGS.max_depth
        )

        # Fail-fast replay audit before trusting any E10/E11 result.
        if local_i < max(0, int(ARGS.noop_check_samples)):
            with torch.inference_mode():
                noop_ep, _ = rerun_cdf_with_read_center(
                    model,
                    ep,
                    read_center=native_xyz,
                    output_center=native_xyz,
                )
            metrics = assert_native_reread_equivalent(
                ep, noop_ep, atol=float(ARGS.noop_atol)
            )
            for key, value in metrics.items():
                noop_max[key] = max(noop_max[key], float(value))
            noop_decoded = pred_decode_center_view_angle(noop_ep, use_cdf=True)[0]
            decoded_diff = float((noop_decoded - e00_t[0]).abs().max().item())
            noop_max["noop_decoded_max_abs"] = max(
                noop_max["noop_decoded_max_abs"], decoded_diff
            )
            if decoded_diff > float(ARGS.noop_atol):
                raise RuntimeError(
                    "Native-center no-op replay changed decoded grasps: "
                    f"max_abs={decoded_diff:.3e} > atol={ARGS.noop_atol:.3e}"
                )

        if ARGS.profile_timing:
            base._cuda_sync(device)
        t_reread = time.perf_counter()
        with torch.inference_mode():
            ep10, _ = rerun_cdf_with_read_center(
                model,
                ep,
                read_center=ref_xyz,
                output_center=native_xyz,
            )
            e10_t = pred_decode_center_view_angle(ep10, use_cdf=True)
            e11_t = replace_decoded_translation(e10_t, ref_xyz)
        if ARGS.profile_timing:
            base._cuda_sync(device)
        reread_sec = time.perf_counter() - t_reread

        valid_all = ref_valid[0].bool()
        eval_idx = base._select_eval_queries(
            e00_t[0],
            valid_all,
            query_eval_num=ARGS.query_eval_num,
            mode=ARGS.query_eval_mode,
            valid_only=ARGS.eval_valid_only,
        )
        n_native = int(e00_t[0].shape[0])
        total_native_queries += n_native
        total_ref_valid += int(valid_all.sum().item())
        total_eval_queries += int(eval_idx.numel())

        arrays_t = {"E10": e10_t[0], "E11": e11_t[0]}
        arrays = {
            v: arrays_t[v].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
            for v in VARIANTS
        }
        # E10/E11 must differ by translation only.
        keep = np.r_[0:13, 16]
        nonxyz = (
            float(np.max(np.abs(arrays["E10"][:, keep] - arrays["E11"][:, keep])))
            if arrays["E10"].size else 0.0
        )
        if nonxyz > 2.0e-5:
            raise RuntimeError(
                f"E10/E11 changed non-translation fields: max_abs={nonxyz:.3e}"
            )

        scene_id = int(batch["scene_idx"].reshape(-1)[0].item())
        anno_id = int(batch["anno_idx"].reshape(-1)[0].item())
        if ARGS.save_raw_grasps:
            for variant, arr in arrays.items():
                base._save_raw(out_root, variant, scene_id, anno_id, arr)

        results, eval_timing = _evaluate_two_once(
            evaluator, scene_id, anno_id, arrays
        )
        timing_sum["native_forward_sec"] += native_sec
        timing_sum["counterfactual_reread_sec"] += reread_sec
        for key, value in eval_timing.items():
            timing_sum[key] += float(value)

        selected = eval_idx.detach().cpu().numpy().astype(np.int64)
        valid = valid_all.index_select(0, eval_idx).detach().cpu().numpy().astype(bool)
        ref_z = ref_depth[0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
        native = native_xyz[0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
        reference = ref_xyz[0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
        shift_mm = np.linalg.norm(reference - native, axis=-1) * 1000.0
        token_idx = ep["kview_base_token_sel_idx"][0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.int64)
        view_idx = ep["grasp_top_view_inds"][0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.int64)
        n = int(eval_idx.numel())

        for local_q in range(n):
            native_q = int(selected[local_q])
            for variant in VARIANTS:
                r = results[variant]
                friction = float(r.friction[local_q])
                per_query.append({
                    "split": ARGS.split,
                    "scene_id": scene_id,
                    "anno_id": anno_id,
                    "dataset_idx": dataset_idx,
                    "query_id": native_q,
                    "eval_query_rank": local_q,
                    "variant": variant,
                    "token_idx": int(token_idx[local_q]),
                    "view_idx": int(view_idx[local_q]),
                    "ref_valid": int(valid[local_q]),
                    "native_z_m": float(native[local_q, 2]),
                    "ref_z_m": float(ref_z[local_q]),
                    "center_shift_mm": float(shift_mm[local_q]),
                    "pred_score": float(arrays[variant][local_q, 0]),
                    "friction": friction,
                    "success_04": int(friction > 0.0 and friction <= 0.4 + 1e-6),
                    "success_08": int(friction > 0.0 and friction <= 0.8 + 1e-6),
                    "collision_or_empty": int(r.collision_or_empty[local_q]),
                    "pure_collision": int(r.pure_collision[local_q]),
                    "empty": int(r.empty[local_q]),
                    "assigned_obj": int(r.assigned_obj[local_q]),
                })

        sample_row = {
            "split": ARGS.split,
            "scene_id": scene_id,
            "anno_id": anno_id,
            "dataset_idx": dataset_idx,
            "num_native_queries": n_native,
            "num_eval_queries": n,
            "native_ref_valid_ratio": float(valid_all.float().mean().item()),
            "eval_ref_valid_ratio": float(valid.mean()) if n else float("nan"),
            "center_shift_mm_mean": float(shift_mm.mean()) if n else float("nan"),
            "center_shift_mm_median": float(np.median(shift_mm)) if n else float("nan"),
            "native_forward_sec": float(native_sec),
            "counterfactual_reread_sec": float(reread_sec),
            "exact_eval_sec": float(eval_timing["eval_sec"]),
            "collision_sec": float(eval_timing["collision_sec"]),
            "force_closure_sec": float(eval_timing["force_closure_sec"]),
            "e10_e11_nonxyz_max_abs": nonxyz,
        }
        for variant in VARIANTS:
            r = results[variant]
            sample_row[f"{variant}_success04"] = base._mean_bool(base._success(r.friction, 0.4))
            sample_row[f"{variant}_success08"] = base._mean_bool(base._success(r.friction, 0.8))
            sample_row[f"{variant}_collision"] = base._mean_bool(r.collision_or_empty)
            sample_row[f"{variant}_empty"] = base._mean_bool(r.empty)
        per_sample.append(sample_row)

        if local_i % 20 == 0:
            print(
                f"[REREAD-DIAG] {local_i + 1}/{len(indices)} "
                f"scene={scene_id:04d} anno={anno_id:04d} native={n_native} eval={n} "
                f"valid={valid_all.float().mean().item():.3f} "
                f"t_native={native_sec:.2f}s t_reread={reread_sec:.2f}s "
                f"t_eval={eval_timing['eval_sec']:.2f}s",
                flush=True,
            )

    _write_csv(out_root / "per_query.csv", per_query)
    _write_csv(out_root / "per_sample_summary.csv", per_sample)

    aggregate = {}
    for variant in VARIANTS:
        rows = [r for r in per_query if r["variant"] == variant]
        friction = np.asarray([r["friction"] for r in rows], dtype=np.float32)
        collision = np.asarray([r["collision_or_empty"] for r in rows], dtype=np.float32)
        empty = np.asarray([r["empty"] for r in rows], dtype=np.float32)
        aggregate[variant] = {
            "num_queries": len(rows),
            "success04": base._mean_bool(base._success(friction, 0.4)),
            "success08": base._mean_bool(base._success(friction, 0.8)),
            "collision": base._mean_bool(collision),
            "empty": base._mean_bool(empty),
        }

    summary = {
        "protocol": "E10/E11 corrected reread v1",
        "checkpoint": os.path.abspath(ARGS.checkpoint_path),
        "split": ARGS.split,
        "camera": ARGS.camera,
        "sample_interval": ARGS.sample_interval,
        "query_eval_num": ARGS.query_eval_num,
        "query_eval_mode": ARGS.query_eval_mode,
        "eval_valid_only": bool(ARGS.eval_valid_only),
        "num_samples": len(per_sample),
        "num_native_queries": total_native_queries,
        "num_reference_valid_queries": total_ref_valid,
        "num_evaluated_queries": total_eval_queries,
        "aggregate": aggregate,
        "noop_replay_max": dict(noop_max),
        "timing_total_sec": dict(timing_sum),
        "timing_per_sample_sec": {
            key: float(value / max(len(per_sample), 1))
            for key, value in timing_sum.items()
        },
        "join_key_for_previous_e00_e01": [
            "split", "scene_id", "anno_id", "query_id"
        ],
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
