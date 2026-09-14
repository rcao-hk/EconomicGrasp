#!/usr/bin/env python3
"""E00/E01/E10/E11 diagnostic for EconomicGrasp-DPT CVA-CDF.

The script performs an inference-only, same-query causal intervention after the
CVA center-view selector.  It uses the rendered/fused GT depth *only* to define
an evaluation-time reference center at the exact native image-FPS pixel; GT
geometry never enters the native feature/depth forward path.

Outputs:
  per_query.csv          exact-action result for every paired query/variant
  per_sample_summary.csv sample-level paired summary on reference-valid queries
  summary.json           aggregate paired statistics and invariants
  raw_grasps/...         optional [N,17] variant arrays before any postprocess

The GraspNet/Dex-Net evaluator is used only here, never in model training.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

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
    return p.parse_args()


ARGS = _parse_args()
# economicgrasp_bip3d imports the repository's legacy global cfg parser.  The
# diagnostic owns its CLI, so prevent the legacy parser from consuming it.
sys.argv = [sys.argv[0]]

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
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
    # Geometry diagnostics are optional runtime-only modules on some revisions.
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


def _mean_bool(x: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.asarray(x, dtype=np.float32)[mask].mean())


def _save_raw(root: Path, variant: str, scene_id: int, anno_id: int, arr: np.ndarray) -> None:
    path = root / "raw_grasps" / variant / f"scene_{scene_id:04d}"
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"{anno_id:04d}.npy", np.asarray(arr, dtype=np.float32))


def _assert_pair_invariants(grasp_arrays: Dict[str, np.ndarray], tol: float = 2e-5) -> Dict[str, float]:
    e00, e01 = grasp_arrays["E00"], grasp_arrays["E01"]
    e10, e11 = grasp_arrays["E10"], grasp_arrays["E11"]
    if not (e00.shape == e01.shape == e10.shape == e11.shape):
        raise RuntimeError(f"Variant shape mismatch: {[x.shape for x in grasp_arrays.values()]}")
    # score,width,height,depth,rotation,obj-id must be unchanged within a
    # read-center pair; only xyz may differ.
    keep = np.r_[0:13, 16]
    d01 = float(np.max(np.abs(e00[:, keep] - e01[:, keep]))) if e00.size else 0.0
    d1011 = float(np.max(np.abs(e10[:, keep] - e11[:, keep]))) if e10.size else 0.0
    if d01 > tol or d1011 > tol:
        raise RuntimeError(
            "Center intervention changed non-translation outputs unexpectedly: "
            f"E00/E01={d01:.3e}, E10/E11={d1011:.3e}"
        )
    return {"e00_e01_nonxyz_max_abs": d01, "e10_e11_nonxyz_max_abs": d1011}


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

    for local_i, batch in enumerate(loader):
        dataset_idx = indices[local_i]
        batch = _move_batch(batch, device)
        batch["cva_export_angle_feature"] = False
        batch["cva_compute_diagnostics"] = False
        batch["geometry_compute_diagnostics"] = False

        with torch.no_grad():
            ep = model(batch)
            e00_t = pred_decode_center_view_angle(ep, use_cdf=True)
            ref_xyz, ref_valid, ref_depth = gather_reference_centers_from_depth(
                ep, min_depth=ARGS.min_depth, max_depth=ARGS.max_depth
            )
            native_xyz = ep["kview_base_xyz_graspable"].float()
            e01_t = replace_decoded_translation(e00_t, ref_xyz)

            ep10, _ = rerun_cdf_with_read_center(
                model, ep, read_center=ref_xyz, output_center=native_xyz
            )
            e10_t = pred_decode_center_view_angle(ep10, use_cdf=True)
            ep11, _ = rerun_cdf_with_read_center(
                model, ep, read_center=ref_xyz, output_center=ref_xyz
            )
            e11_t = pred_decode_center_view_angle(ep11, use_cdf=True)

        arrays = {
            "E00": e00_t[0].detach().cpu().numpy().astype(np.float32),
            "E01": e01_t[0].detach().cpu().numpy().astype(np.float32),
            "E10": e10_t[0].detach().cpu().numpy().astype(np.float32),
            "E11": e11_t[0].detach().cpu().numpy().astype(np.float32),
        }
        inv = _assert_pair_invariants(arrays)
        for k, v in inv.items():
            invariant_max[k] = max(invariant_max[k], v)

        scene_id = int(batch["scene_idx"].reshape(-1)[0].item())
        anno_id = int(batch["anno_idx"].reshape(-1)[0].item())
        if ARGS.save_raw_grasps:
            for variant, arr in arrays.items():
                _save_raw(out_root, variant, scene_id, anno_id, arr)

        results = {
            variant: evaluator.evaluate(scene_id, anno_id, arr)
            for variant, arr in arrays.items()
        }
        valid = ref_valid[0].detach().cpu().numpy().astype(bool)
        ref_z = ref_depth[0].detach().cpu().numpy().astype(np.float32)
        native = native_xyz[0].detach().cpu().numpy().astype(np.float32)
        reference = ref_xyz[0].detach().cpu().numpy().astype(np.float32)
        center_shift_mm = np.linalg.norm(reference - native, axis=-1) * 1000.0
        token_idx = ep["kview_base_token_sel_idx"][0].detach().cpu().numpy().astype(np.int64)
        view_idx = ep["grasp_top_view_inds"][0].detach().cpu().numpy().astype(np.int64)

        n = arrays["E00"].shape[0]
        if not (len(valid) == len(ref_z) == len(token_idx) == len(view_idx) == n):
            raise RuntimeError("Top-1 query/reference arrays lost same-query alignment.")

        for q in range(n):
            for variant in VARIANTS:
                r = results[variant]
                friction = float(r.friction[q])
                per_query.append({
                    "split": ARGS.split,
                    "scene_id": scene_id,
                    "anno_id": anno_id,
                    "dataset_idx": dataset_idx,
                    "query_id": q,
                    "variant": variant,
                    "token_idx": int(token_idx[q]),
                    "view_idx": int(view_idx[q]),
                    "ref_valid": int(valid[q]),
                    "native_z_m": float(native[q, 2]),
                    "ref_z_m": float(ref_z[q]),
                    "center_shift_mm": float(center_shift_mm[q]),
                    "pred_score": float(arrays[variant][q, 0]),
                    "friction": friction,
                    "success_04": int(friction > 0.0 and friction <= 0.4 + 1e-6),
                    "success_08": int(friction > 0.0 and friction <= 0.8 + 1e-6),
                    "collision_or_empty": int(r.collision_or_empty[q]),
                    "pure_collision": int(r.pure_collision[q]),
                    "empty": int(r.empty[q]),
                    "assigned_obj": int(r.assigned_obj[q]),
                })

        sample_row = {
            "split": ARGS.split,
            "scene_id": scene_id,
            "anno_id": anno_id,
            "dataset_idx": dataset_idx,
            "num_queries": n,
            "ref_valid_ratio": float(valid.mean()) if n else float("nan"),
            "center_shift_mm_mean": float(center_shift_mm[valid].mean()) if valid.any() else float("nan"),
            "center_shift_mm_median": float(np.median(center_shift_mm[valid])) if valid.any() else float("nan"),
        }
        for variant in VARIANTS:
            r = results[variant]
            sample_row[f"{variant}_success04"] = _mean_bool(_success(r.friction, 0.4), valid)
            sample_row[f"{variant}_success08"] = _mean_bool(_success(r.friction, 0.8), valid)
            sample_row[f"{variant}_collision"] = _mean_bool(r.collision_or_empty, valid)
            sample_row[f"{variant}_empty"] = _mean_bool(r.empty, valid)
        per_sample.append(sample_row)

        if local_i % 20 == 0:
            print(
                f"[CENTER-DIAG] {local_i + 1}/{len(indices)} "
                f"scene={scene_id:04d} anno={anno_id:04d} valid={valid.mean():.3f}",
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

    valid_rows = [r for r in per_query if int(r["ref_valid"]) == 1]
    summary = {
        "protocol": "E00/E01/E10/E11 post-selector center decoupling v1",
        "checkpoint": os.path.abspath(ARGS.checkpoint_path),
        "split": ARGS.split,
        "camera": ARGS.camera,
        "sample_interval": ARGS.sample_interval,
        "num_samples": len(per_sample),
        "num_query_variant_rows": len(per_query),
        "num_reference_valid_variant_rows": len(valid_rows),
        "invariants": dict(invariant_max),
        "variants": {},
    }
    for variant in VARIANTS:
        rows = [r for r in valid_rows if r["variant"] == variant]
        if not rows:
            continue
        summary["variants"][variant] = {
            "queries": len(rows),
            "success04": float(np.mean([r["success_04"] for r in rows])),
            "success08": float(np.mean([r["success_08"] for r in rows])),
            "collision_or_empty": float(np.mean([r["collision_or_empty"] for r in rows])),
            "empty": float(np.mean([r["empty"] for r in rows])),
            "score_mean": float(np.mean([r["pred_score"] for r in rows])),
        }

    # Paired causal deltas: translation effect, read-evidence effect, combined.
    by_key = defaultdict(dict)
    for r in valid_rows:
        by_key[(r["scene_id"], r["anno_id"], r["query_id"])][r["variant"]] = r
    pairs = [v for v in by_key.values() if all(k in v for k in VARIANTS)]
    summary["paired"] = {}
    for lhs, rhs, name in (
        ("E00", "E01", "output_center_only_E01_minus_E00"),
        ("E01", "E11", "reread_given_ref_output_E11_minus_E01"),
        ("E00", "E10", "read_center_only_E10_minus_E00"),
        ("E00", "E11", "combined_E11_minus_E00"),
    ):
        summary["paired"][name] = {
            "success04_delta": float(np.mean([p[rhs]["success_04"] - p[lhs]["success_04"] for p in pairs])),
            "success08_delta": float(np.mean([p[rhs]["success_08"] - p[lhs]["success_08"] for p in pairs])),
            "collision_delta": float(np.mean([p[rhs]["collision_or_empty"] - p[lhs]["collision_or_empty"] for p in pairs])),
        }

    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
