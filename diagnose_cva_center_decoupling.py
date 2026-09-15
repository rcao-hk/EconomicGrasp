#!/usr/bin/env python3
"""E00/E01/E10/E11 diagnostic for EconomicGrasp-DPT CVA-CDF.

The script performs an inference-only, same-query causal intervention after the
CVA center-view selector.  It uses the rendered/fused GT depth *only* to define
an evaluation-time reference center at the exact native image-FPS pixel; GT
geometry never enters the native feature/depth forward path.

Fast-path design
----------------
1. E10 and E11 share one counterfactual local re-read/CDF decode.  E11 is formed
   from E10 by replacing translation only.
2. All four variants are concatenated and sent through the exact-action
   evaluator once per frame, avoiding four repeated scene/evaluator setups.
3. Reference-invalid queries can be excluded before exact evaluation.
4. A deterministic paired query subset can be evaluated.  Selection uses only
   E00/native information, so the treatment variants cannot influence which
   queries enter the diagnostic.

Outputs:
  per_query.csv          exact-action result for every evaluated query/variant
  per_sample_summary.csv sample-level paired summary
  summary.json           aggregate paired statistics, timing and invariants
  raw_grasps/...         optional [N,17] evaluated variant arrays

The GraspNet/Dex-Net evaluator is used only here, never in model training.
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
from typing import Dict, List, Mapping, Sequence, Tuple

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
    p.add_argument("--pose_depth_mode", default="none", choices=("none", "global_film", "ray_gravity_film"))
    p.add_argument("--save_raw_grasps", action="store_true")
    p.add_argument("--fc_mode", default="reuse_contacts", choices=("reuse_contacts", "official"))
    p.add_argument("--verify_n", type=int, default=0)
    p.add_argument(
        "--query_eval_num",
        type=int,
        default=128,
        help="Number of native queries evaluated per frame; 0 means all queries.",
    )
    p.add_argument(
        "--query_eval_mode",
        default="topk_uniform",
        choices=("all", "topk", "uniform", "topk_uniform"),
        help="Deterministic paired query selection based only on E00/native outputs.",
    )
    p.add_argument(
        "--eval_valid_only",
        action="store_true",
        help="Exclude reference-invalid queries before exact-action evaluation.",
    )
    p.add_argument(
        "--profile_timing",
        action="store_true",
        help="Synchronize CUDA around model stages and export timing diagnostics.",
    )
    return p.parse_args()


ARGS = _parse_args()
# economicgrasp_bip3d imports the repository's legacy global cfg parser.  The
# diagnostic owns its CLI, so prevent the legacy parser from consuming it.
sys.argv = [sys.argv[0]]

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactActionEvalResult, ExactGraspNetActionEvaluator
from models.economicgrasp_bip3d import economicgrasp_dpt, pred_decode_center_view_angle
from utils.arguments import cfgs
from utils.cva_center_decoupling import (
    gather_reference_centers_from_depth,
    replace_decoded_translation,
    rerun_cdf_with_read_center,
)


VARIANTS = ("E00", "E01", "E10", "E11")


def _configure_legacy_cfg() -> None:
    """Set only architecture switches that must be deterministic for this test."""
    cfgs.use_top4_view_infer = False
    cfgs.kview_mode = "A1"
    cfgs.kview_k = 1
    cfgs.use_cdf = True
    cfgs.use_obs_depth = False
    cfgs.pose_depth_mode = ARGS.pose_depth_mode


def _load_checkpoint_strict(model: torch.nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    result = model.load_state_dict(state, strict=False)
    optional = ("rgb_geometry_diagnostics.",)
    missing = [k for k in result.missing_keys if not k.startswith(optional)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(optional)]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch in center diagnostic: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _subset_indices(total: int, interval: float, max_samples: int) -> List[int]:
    if interval <= 0.0 or interval > 1.0:
        raise ValueError("sample_interval must lie in (0,1].")
    stride = max(1, int(round(1.0 / interval)))
    out: List[int] = []
    for start in range(0, total, 256):
        out.extend(range(start, min(start + 256, total), stride))
    if max_samples > 0:
        out = out[:max_samples]
    return out


def _move_batch(batch: dict, device: torch.device) -> dict:
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=False)
        elif isinstance(v, (list, tuple)):
            raise TypeError(
                f"Diagnostic expects load_label=False fixed tensors, but key {k!r} is list-valued."
            )
    return batch


def _success(friction: np.ndarray, threshold: float) -> np.ndarray:
    friction = np.asarray(friction, dtype=np.float32)
    return (friction > 0.0) & (friction <= float(threshold) + 1.0e-6)


def _mean_bool(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    return float(arr.mean()) if arr.size else float("nan")


def _save_raw(root: Path, variant: str, scene_id: int, anno_id: int, arr: np.ndarray) -> None:
    path = root / "raw_grasps" / variant / f"scene_{scene_id:04d}"
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"{anno_id:04d}.npy", np.asarray(arr, dtype=np.float32))


def _assert_pair_invariants(grasp_arrays: Dict[str, np.ndarray], tol: float = 2e-5) -> Dict[str, float]:
    e00, e01 = grasp_arrays["E00"], grasp_arrays["E01"]
    e10, e11 = grasp_arrays["E10"], grasp_arrays["E11"]
    if not (e00.shape == e01.shape == e10.shape == e11.shape):
        raise RuntimeError(f"Variant shape mismatch: {[x.shape for x in grasp_arrays.values()]}")
    keep = np.r_[0:13, 16]
    d01 = float(np.max(np.abs(e00[:, keep] - e01[:, keep]))) if e00.size else 0.0
    d1011 = float(np.max(np.abs(e10[:, keep] - e11[:, keep]))) if e10.size else 0.0
    if d01 > tol or d1011 > tol:
        raise RuntimeError(
            "Center intervention changed non-translation outputs unexpectedly: "
            f"E00/E01={d01:.3e}, E10/E11={d1011:.3e}"
        )
    return {"e00_e01_nonxyz_max_abs": d01, "e10_e11_nonxyz_max_abs": d1011}


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _uniform_pick(sorted_indices: torch.Tensor, count: int) -> torch.Tensor:
    """Pick evenly spaced entries from an already filtered/sorted index vector."""
    n = int(sorted_indices.numel())
    count = min(max(int(count), 0), n)
    if count <= 0:
        return sorted_indices[:0]
    if count >= n:
        return sorted_indices
    pos = torch.linspace(0, n - 1, steps=count, device=sorted_indices.device)
    pos = torch.round(pos).long().clamp(0, n - 1)
    # linspace+round can duplicate positions for very small n/count edge cases.
    pos = torch.unique(pos, sorted=True)
    if pos.numel() < count:
        used = torch.zeros(n, device=sorted_indices.device, dtype=torch.bool)
        used[pos] = True
        fill = torch.nonzero(~used, as_tuple=False).squeeze(1)[: count - pos.numel()]
        pos = torch.sort(torch.cat((pos, fill), dim=0)).values
    return sorted_indices.index_select(0, pos[:count])


def _select_eval_queries(
    native_grasps: torch.Tensor,
    ref_valid: torch.Tensor,
    *,
    query_eval_num: int,
    mode: str,
    valid_only: bool,
) -> torch.Tensor:
    """Select a deterministic same-query subset using E00/native information only."""
    if native_grasps.dim() != 2 or native_grasps.shape[-1] != 17:
        raise ValueError(f"native_grasps must be [N,17], got {tuple(native_grasps.shape)}")
    n = int(native_grasps.shape[0])
    if ref_valid.shape != (n,):
        raise ValueError(f"ref_valid must be [{n}], got {tuple(ref_valid.shape)}")

    eligible = torch.arange(n, device=native_grasps.device, dtype=torch.long)
    if valid_only:
        eligible = eligible[ref_valid.bool()]
    if eligible.numel() == 0:
        return eligible

    target = int(query_eval_num)
    if target <= 0 or target >= int(eligible.numel()) or mode == "all":
        return eligible

    score = native_grasps[:, 0].float()
    if mode == "topk":
        order = torch.argsort(score.index_select(0, eligible), descending=True, stable=True)
        return eligible.index_select(0, order[:target])
    if mode == "uniform":
        return _uniform_pick(eligible, target)
    if mode == "topk_uniform":
        n_top = target // 2
        n_uniform = target - n_top
        order = torch.argsort(score.index_select(0, eligible), descending=True, stable=True)
        ranked = eligible.index_select(0, order)
        top = ranked[:n_top]
        if n_uniform <= 0:
            return top
        remaining = ranked[n_top:]
        uniform = _uniform_pick(remaining, n_uniform)
        return torch.cat((top, uniform), dim=0)
    raise ValueError(f"Unsupported query_eval_mode={mode!r}")


def _slice_eval_result(result: ExactActionEvalResult, start: int, end: int) -> ExactActionEvalResult:
    """Split one concatenated exact-evaluator result back into one variant."""
    return ExactActionEvalResult(
        assigned_obj=result.assigned_obj[start:end].copy(),
        collision_or_empty=result.collision_or_empty[start:end].copy(),
        pure_collision=result.pure_collision[start:end].copy(),
        empty=result.empty[start:end].copy(),
        friction=result.friction[start:end].copy(),
        stats=dict(result.stats),
    )


def _evaluate_variants_once(
    evaluator: ExactGraspNetActionEvaluator,
    scene_id: int,
    anno_id: int,
    arrays: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, ExactActionEvalResult], Dict[str, float]]:
    """Evaluate concatenated E00/E01/E10/E11 candidates in one scene pass."""
    sizes = [int(arrays[v].shape[0]) for v in VARIANTS]
    if len(set(sizes)) != 1:
        raise RuntimeError(f"Variant query counts differ: {dict(zip(VARIANTS, sizes))}")
    n = sizes[0]
    if n == 0:
        return {
            v: evaluator.evaluate(scene_id, anno_id, arrays[v]) for v in VARIANTS
        }, {"collision_sec": 0.0, "force_closure_sec": 0.0, "eval_sec": 0.0}

    merged = np.concatenate([arrays[v] for v in VARIANTS], axis=0)
    t0 = time.perf_counter()
    merged_result = evaluator.evaluate(scene_id, anno_id, merged)
    eval_sec = time.perf_counter() - t0
    results: Dict[str, ExactActionEvalResult] = {}
    for i, v in enumerate(VARIANTS):
        results[v] = _slice_eval_result(merged_result, i * n, (i + 1) * n)
    timing = {
        "eval_sec": float(eval_sec),
        "collision_sec": float(merged_result.stats.get("collision_sec", 0.0)),
        "force_closure_sec": float(merged_result.stats.get("force_closure_sec", 0.0)),
    }
    return results, timing


def main() -> None:
    _configure_legacy_cfg()
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
    indices = _subset_indices(len(dataset), ARGS.sample_interval, ARGS.max_samples)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=ARGS.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(ARGS.num_workers > 0),
    )

    model = economicgrasp_dpt(
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=ARGS.pose_depth_mode,
        use_cdf=True,
        vis_dir=None,
    ).to(device)
    _load_checkpoint_strict(model, ARGS.checkpoint_path)
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
    invariant_max = defaultdict(float)
    timing_sum = defaultdict(float)
    total_native_queries = 0
    total_eval_queries = 0
    total_ref_valid = 0

    for local_i, batch in enumerate(loader):
        dataset_idx = indices[local_i]
        batch = _move_batch(batch, device)
        batch["cva_export_angle_feature"] = False
        batch["cva_compute_diagnostics"] = False
        batch["geometry_compute_diagnostics"] = False

        if ARGS.profile_timing:
            _cuda_sync(device)
        t_native = time.perf_counter()
        with torch.inference_mode():
            ep = model(batch)
            e00_t = pred_decode_center_view_angle(ep, use_cdf=True)
        if ARGS.profile_timing:
            _cuda_sync(device)
        native_sec = time.perf_counter() - t_native

        ref_xyz, ref_valid, ref_depth = gather_reference_centers_from_depth(
            ep, min_depth=ARGS.min_depth, max_depth=ARGS.max_depth
        )
        native_xyz = ep["kview_base_xyz_graspable"].float()
        e01_t = replace_decoded_translation(e00_t, ref_xyz)

        if ARGS.profile_timing:
            _cuda_sync(device)
        t_reread = time.perf_counter()
        with torch.inference_mode():
            # E10/E11 share exactly the same reference-center evidence read and
            # decoder output.  Decode once at the native output center, then form
            # E11 by translation replacement only.
            ep10, _ = rerun_cdf_with_read_center(
                model, ep, read_center=ref_xyz, output_center=native_xyz
            )
            e10_t = pred_decode_center_view_angle(ep10, use_cdf=True)
            e11_t = replace_decoded_translation(e10_t, ref_xyz)
        if ARGS.profile_timing:
            _cuda_sync(device)
        reread_sec = time.perf_counter() - t_reread

        all_arrays_t = {
            "E00": e00_t[0],
            "E01": e01_t[0],
            "E10": e10_t[0],
            "E11": e11_t[0],
        }
        n_native = int(all_arrays_t["E00"].shape[0])
        valid_all = ref_valid[0].bool()
        eval_idx = _select_eval_queries(
            all_arrays_t["E00"],
            valid_all,
            query_eval_num=ARGS.query_eval_num,
            mode=ARGS.query_eval_mode,
            valid_only=ARGS.eval_valid_only,
        )
        total_native_queries += n_native
        total_ref_valid += int(valid_all.sum().item())
        total_eval_queries += int(eval_idx.numel())

        arrays = {
            v: all_arrays_t[v].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
            for v in VARIANTS
        }
        inv = _assert_pair_invariants(arrays)
        for k, v in inv.items():
            invariant_max[k] = max(invariant_max[k], v)

        scene_id = int(batch["scene_idx"].reshape(-1)[0].item())
        anno_id = int(batch["anno_idx"].reshape(-1)[0].item())
        if ARGS.save_raw_grasps:
            for variant, arr in arrays.items():
                _save_raw(out_root, variant, scene_id, anno_id, arr)

        results, eval_timing = _evaluate_variants_once(
            evaluator, scene_id, anno_id, arrays
        )
        timing_sum["native_forward_sec"] += native_sec
        timing_sum["counterfactual_reread_sec"] += reread_sec
        for k, v in eval_timing.items():
            timing_sum[k] += float(v)

        selected = eval_idx.detach().cpu().numpy().astype(np.int64)
        valid = valid_all.index_select(0, eval_idx).detach().cpu().numpy().astype(bool)
        ref_z = ref_depth[0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
        native = native_xyz[0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
        reference = ref_xyz[0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.float32)
        center_shift_mm = np.linalg.norm(reference - native, axis=-1) * 1000.0
        token_idx = ep["kview_base_token_sel_idx"][0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.int64)
        view_idx = ep["grasp_top_view_inds"][0].index_select(0, eval_idx).detach().cpu().numpy().astype(np.int64)

        n = arrays["E00"].shape[0]
        if not (len(selected) == len(valid) == len(ref_z) == len(token_idx) == len(view_idx) == n):
            raise RuntimeError("Evaluated query/reference arrays lost same-query alignment.")

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
                    "center_shift_mm": float(center_shift_mm[local_q]),
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
            "native_ref_valid_ratio": float(valid_all.float().mean().item()) if n_native else float("nan"),
            "eval_ref_valid_ratio": float(valid.mean()) if n else float("nan"),
            "center_shift_mm_mean": float(center_shift_mm.mean()) if n else float("nan"),
            "center_shift_mm_median": float(np.median(center_shift_mm)) if n else float("nan"),
            "native_forward_sec": float(native_sec),
            "counterfactual_reread_sec": float(reread_sec),
            "exact_eval_sec": float(eval_timing["eval_sec"]),
            "collision_sec": float(eval_timing["collision_sec"]),
            "force_closure_sec": float(eval_timing["force_closure_sec"]),
        }
        for variant in VARIANTS:
            r = results[variant]
            sample_row[f"{variant}_success04"] = _mean_bool(_success(r.friction, 0.4))
            sample_row[f"{variant}_success08"] = _mean_bool(_success(r.friction, 0.8))
            sample_row[f"{variant}_collision"] = _mean_bool(r.collision_or_empty)
            sample_row[f"{variant}_empty"] = _mean_bool(r.empty)
        per_sample.append(sample_row)

        if local_i % 20 == 0:
            print(
                f"[CENTER-DIAG] {local_i + 1}/{len(indices)} "
                f"scene={scene_id:04d} anno={anno_id:04d} "
                f"native={n_native} eval={n} valid={valid_all.float().mean().item():.3f} "
                f"t_native={native_sec:.2f}s t_reread={reread_sec:.2f}s "
                f"t_eval={eval_timing['eval_sec']:.2f}s",
                flush=True,
            )

    def write_csv(path: Path, rows: List[dict]) -> None:
        if not rows:
            raise RuntimeError(f"No rows generated for {path}")
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_root / "per_query.csv", per_query)
    write_csv(out_root / "per_sample_summary.csv", per_sample)

    def _variant_rows(variant: str) -> List[dict]:
        return [r for r in per_query if r["variant"] == variant]

    aggregate = {}
    for variant in VARIANTS:
        rows = _variant_rows(variant)
        friction = np.asarray([r["friction"] for r in rows], dtype=np.float32)
        collision = np.asarray([r["collision_or_empty"] for r in rows], dtype=np.float32)
        empty = np.asarray([r["empty"] for r in rows], dtype=np.float32)
        aggregate[variant] = {
            "num_queries": int(len(rows)),
            "success04": _mean_bool(_success(friction, 0.4)),
            "success08": _mean_bool(_success(friction, 0.8)),
            "collision": _mean_bool(collision),
            "empty": _mean_bool(empty),
        }

    def delta(a: str, b: str, metric: str) -> float:
        return float(aggregate[a][metric] - aggregate[b][metric])

    summary = {
        "protocol": "E00/E01/E10/E11 post-selector center decoupling v2-fast",
        "checkpoint": os.path.abspath(ARGS.checkpoint_path),
        "split": ARGS.split,
        "camera": ARGS.camera,
        "sample_interval": ARGS.sample_interval,
        "query_eval_num": int(ARGS.query_eval_num),
        "query_eval_mode": ARGS.query_eval_mode,
        "eval_valid_only": bool(ARGS.eval_valid_only),
        "num_samples": len(per_sample),
        "num_native_queries": int(total_native_queries),
        "num_reference_valid_queries": int(total_ref_valid),
        "num_evaluated_queries": int(total_eval_queries),
        "evaluated_fraction": float(total_eval_queries / max(total_native_queries, 1)),
        "invariants": dict(invariant_max),
        "aggregate": aggregate,
        "paired_deltas": {
            "E01_minus_E00_success04": delta("E01", "E00", "success04"),
            "E01_minus_E00_success08": delta("E01", "E00", "success08"),
            "E01_minus_E00_collision": delta("E01", "E00", "collision"),
            "E11_minus_E01_success04": delta("E11", "E01", "success04"),
            "E11_minus_E01_success08": delta("E11", "E01", "success08"),
            "E11_minus_E01_collision": delta("E11", "E01", "collision"),
            "E10_minus_E00_success04": delta("E10", "E00", "success04"),
            "E10_minus_E00_success08": delta("E10", "E00", "success08"),
            "E10_minus_E00_collision": delta("E10", "E00", "collision"),
            "E11_minus_E00_success04": delta("E11", "E00", "success04"),
            "E11_minus_E00_success08": delta("E11", "E00", "success08"),
            "E11_minus_E00_collision": delta("E11", "E00", "collision"),
        },
        "timing_total_sec": {k: float(v) for k, v in timing_sum.items()},
        "timing_per_sample_sec": {
            k: float(v / max(len(per_sample), 1)) for k, v in timing_sum.items()
        },
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
