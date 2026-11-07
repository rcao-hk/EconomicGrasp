#!/usr/bin/env python3
"""Point RayRoPE P2: model-forward fixed-action sampling + exact CAD/Dex-Net labels.

Run against the *existing* checkpoint. No training, no GT depth as RGB model
input and no score-based pre-evaluation collision filter. The same physical
candidate is evaluated once and its label is reused across sampling groups.

Custom --diag_* flags are consumed BEFORE importing utils.arguments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

# This entrypoint lives below the repository root, but imports the existing
# root-level model/inference modules without altering their default behavior.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_diag_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diag_output_dir", required=True)
    parser.add_argument("--diag_frame_manifest_csv", default="")
    parser.add_argument("--diag_scene_ids", default="", help="Absolute scene IDs, e.g. 100,101,133.")
    parser.add_argument("--diag_anno_ids", default="0,128", help="Per-scene frame indices.")
    parser.add_argument("--diag_max_frames", type=int, default=18)
    parser.add_argument("--diag_query_anchors", type=int, default=16)
    parser.add_argument("--diag_nonwinners_per_query", type=int, default=1)
    parser.add_argument("--diag_full_lattice_per_anchor", type=int, choices=(0, 1), default=0)
    parser.add_argument("--diag_uniform_budget", type=int, default=0)
    parser.add_argument("--diag_width_grid", default="0.04,0.06,0.08,0.10")
    parser.add_argument("--diag_insertion_width", choices=("fixed", "predicted", "both"), default="fixed")
    parser.add_argument("--diag_seed", type=int, default=20261106)
    parser.add_argument("--diag_eval_fc_mode", choices=("official", "reuse_contacts"), default="reuse_contacts")
    parser.add_argument("--diag_verify_n", type=int, default=10)
    parser.add_argument("--diag_eval_chunk", type=int, default=256)
    parser.add_argument("--diag_skip_eval", type=int, choices=(0, 1), default=0)
    diag, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *remaining]
    return diag


DIAG = _parse_diag_args()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from diagnostics.rayrope_action_lattice import (
    build_p2_actions, decode_cdf_lattice, physical_action_id,
    select_p2_queries, count_width_status,
)
from inference_cva import _load_checkpoint_strict, _move_fixed_inputs
from models.economicgrasp_bip3d import economicgrasp_dpt, pred_decode_center_view_angle
from models.grasp_designs import design_options
from utils.arguments import cfgs
from utils.label_generation import batch_viewpoint_params_to_matrix


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _integers(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _sample_indices(full_dataset, mode: str):
    scenes = full_dataset.scene_list()
    if len(scenes) != len(full_dataset):
        raise RuntimeError("Dataset scene_list() must map 1:1 to samples.")
    requested = []
    if DIAG.diag_frame_manifest_csv:
        with open(DIAG.diag_frame_manifest_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not {"scene_id", "anno_id"}.issubset(set(reader.fieldnames or [])):
                raise ValueError("Frame manifest needs scene_id,anno_id.")
            requested = [(int(r["scene_id"]), int(r["anno_id"])) for r in reader]
    else:
        ids = set(_integers(DIAG.diag_scene_ids)) if DIAG.diag_scene_ids else None
        annos = _integers(DIAG.diag_anno_ids)
        if not annos or any(a < 0 or a >= 256 for a in annos):
            raise ValueError("Expected annotation IDs in [0,255]")
        available_scenes = sorted({int(str(s)[-4:]) for s in scenes})
        requested = [(s, a) for s in available_scenes if ids is None or s in ids for a in annos]
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate scene_id/anno_id in requested frame set.")
    # Strictly ground global scene/anno against the actual subset index.
    by_frame = {}
    for idx, name in enumerate(scenes):
        key = (int(str(name)[-4:]), idx % 256)
        if key in by_frame:
            raise RuntimeError(f"Duplicate dataset frame mapping: {key}")
        by_frame[key] = idx
    missing = [key for key in requested if key not in by_frame]
    if missing:
        raise ValueError(f"Requested frames absent in dataset: {missing[:10]}")
    if DIAG.diag_max_frames < 1:
        raise ValueError("--diag_max_frames must be positive.")
    return [by_frame[key] for key in requested[:DIAG.diag_max_frames]]


def _build_model(device):
    if not bool(cfgs.multi_modal) or not bool(cfgs.use_cdf):
        raise RuntimeError("Requires --multi_modal --use_cdf.")
    if not bool(getattr(cfgs, "enable_point_rayrope", False)):
        raise RuntimeError("Requires --enable_point_rayrope on the intended Point RayRoPE checkpoint.")
    if bool(cfgs.use_obs_depth) or bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError("RGB model inference must not use observed/GT depth.")
    if bool(getattr(cfgs, "use_top4_view_infer", False)):
        raise RuntimeError("Current P2 lattice targets one native top-view per query; disable top4.")
    if cfgs.kview_mode != "A1":
        raise RuntimeError("P2 lattice requires --kview_mode A1.")
    if not cfgs.checkpoint_path or not os.path.isfile(cfgs.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint missing: {cfgs.checkpoint_path}")
    model = economicgrasp_dpt(
        **design_options(cfgs),
        min_depth=cfgs.min_depth, max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num, is_training=False,
        use_obs_depth=False,
        pose_depth_mode=str(cfgs.pose_depth_mode),
        camera_pose_key=str(getattr(cfgs, "camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(getattr(cfgs, "camera_gravity_key", "camera_gravity_vec")),
        pose_hidden_dim=int(getattr(cfgs, "pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(getattr(cfgs, "ray_gravity_hidden_dim", 64)),
        ray_gravity_mid_dim=int(getattr(cfgs, "ray_gravity_mid_dim", 32)),
        use_cdf=True, vis_dir=None,
    ).to(device)
    _load_checkpoint_strict(model, cfgs.checkpoint_path)
    model.eval()
    return model


def _atomic_csv(path: Path, rows):
    if not rows:
        raise ValueError("No candidate rows exported")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main():
    if not cfgs.test_mode or not cfgs.dataset_root:
        raise ValueError("Pass --test_mode test_seen|test_similar|test_novel and --dataset_root.")
    if int(cfgs.batch_size) != 1:
        raise RuntimeError("P2 runner requires --batch_size 1 for per-frame exact action identity.")
    max_width = float(cfgs.grasp_max_width)
    width_grid = [float(w) for w in DIAG.diag_width_grid.split(",") if w.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_dataset = GraspNetMultiDataset(
        cfgs.dataset_root, split=cfgs.test_mode, camera=cfgs.camera,
        num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False,
    )
    indices = _sample_indices(full_dataset, cfgs.test_mode)
    if not indices:
        raise ValueError("No frames selected")
    loader = DataLoader(Subset(full_dataset, indices), batch_size=1, shuffle=False,
                        num_workers=cfgs.num_workers, collate_fn=collate_fn,
                        persistent_workers=(cfgs.num_workers > 0))
    model = _build_model(device)
    evaluator = None
    if not DIAG.diag_skip_eval:
        from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
        evaluator = ExactGraspNetActionEvaluator(
            cfgs.dataset_root, cfgs.camera, split=cfgs.test_mode,
            collision_chunk=DIAG.diag_eval_chunk, fc_mode=DIAG.diag_eval_fc_mode,
            verify_n=DIAG.diag_verify_n, strict=True,
        )
    output_dir = Path(DIAG.diag_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, frame_log = [], []
    for frame_no, (batch, dataset_idx) in enumerate(zip(loader, indices)):
        scene_name = full_dataset.scene_list()[dataset_idx]
        scene_id = int(str(scene_name)[-4:]); anno_id = int(dataset_idx % 256)
        sample = f"scene{scene_id:04d}_frame{anno_id:04d}"
        batch = _move_fixed_inputs(batch, device)
        batch["cva_export_angle_feature"] = False
        batch["cva_compute_diagnostics"] = False
        batch["geometry_compute_diagnostics"] = False
        with torch.inference_mode():
            ep = model(batch)
            lattice = decode_cdf_lattice(
                ep, 0, batch_viewpoint_params_to_matrix,
                max_width_m=max_width,
            )
        # Critical parity contract: the per-query lattice winner must decode
        # to precisely the existing production CVA-CDF prediction. All P2
        # controlled actions are derived from this verified lattice.
        with torch.inference_mode():
            native = pred_decode_center_view_angle(ep, use_cdf=True)[0].detach().cpu().numpy()
        winners = lattice.grasp[lattice.query_winner]
        if winners.shape != native.shape:
            raise RuntimeError(f"Native decoder parity failed: {winners.shape} != {native.shape}")
        err = float(np.max(np.abs(winners - native)))
        if err > 1e-5 or not np.isfinite(err):
            raise RuntimeError(f"Native decoder parity max abs error {err:.8g} > 1e-5; check width/rotation/score contract.")
        seed = DIAG.diag_seed + scene_id * 1009 + anno_id * 101
        selected = select_p2_queries(
            lattice, num_query_anchors=DIAG.diag_query_anchors,
            nonwinner_per_query=DIAG.diag_nonwinners_per_query,
            full_lattice_per_anchor=bool(DIAG.diag_full_lattice_per_anchor),
            uniform_budget=DIAG.diag_uniform_budget or None, seed=seed,
        )
        records = build_p2_actions(
            lattice, selected, widths_m=width_grid,
            insertion_width=DIAG.diag_insertion_width,
            max_width_m=max_width,
        )
        if DIAG.diag_insertion_width == "both":
            from diagnostics.rayrope_action_lattice import validate_paired_insertion_records
            validate_paired_insertion_records(records)
        unique_grasps = {}
        for item in records:
            unique_grasps.setdefault(item.candidate_id, item.row)
        ordered_ids = sorted(unique_grasps)
        unique_rows = np.stack([unique_grasps[k] for k in ordered_ids])
        label_by_id = {}
        if evaluator is not None:
            labels = evaluator.evaluate(scene_id, anno_id, unique_rows)
            for i, candidate_id in enumerate(ordered_ids):
                friction = float(labels.friction[i])
                label_by_id[candidate_id] = {
                    "friction": friction,
                    "pure_collision": int(labels.pure_collision[i]),
                    "collision_or_empty": int(labels.collision_or_empty[i]),
                    "empty": int(labels.empty[i]),
                    "assigned_obj_id": int(labels.assigned_obj[i]),
                }
        for item in records:
            cid = item.candidate_id
            row = item.row
            label = label_by_id.get(cid, {})
            friction = float(label.get("friction", float("nan")))
            bits = {f"success_mu_{mu:.1f}": int(friction > 0 and friction <= mu + 1e-7)
                    if evaluator is not None else "" for mu in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)}
            quality = float(np.mean(list(bits.values()))) if evaluator is not None else ""
            csv_row = {
                "sample_id": sample, "split": cfgs.test_mode,
                "scene_id": scene_id, "anno_id": anno_id,
                "condition": item.condition, "intervention": item.condition,
                "candidate_id": cid, "parent_id": item.parent_id,
                "query_id": item.query_id, "angle_id": item.angle_id,
                "depth_id": item.depth_id, "query_winner": int(item.query_winner),
                "score": float(row[0]),
                "width_m": float(row[1]), "width_mode": item.width_mode,
                "width_unclamped_m": item.raw_width_m,
                "width_predicted_decoded_m": item.original_width_m,
                "height_m": float(row[2]), "insertion_depth_m": float(row[3]),
                **{f"r{i}{j}": float(row[4+i*3+j]) for i in range(3) for j in range(3)},
                "center_x": float(row[13]), "center_y": float(row[14]), "center_z": float(row[15]),
                "object_id": float(row[16]),
                "friction": friction if evaluator is not None else "",
                "quality": quality,
                "collision_or_empty": label.get("collision_or_empty", ""),
                "pure_collision": label.get("pure_collision", ""),
                "empty": label.get("empty", ""),
                "assigned_obj_id": label.get("assigned_obj_id", ""),
                **bits,
            }
            all_rows.append(csv_row)
        log = {
            "sample_id": sample, "dataset_index": dataset_idx,
            "num_query_lattice": lattice.num_queries,
            "num_angles": lattice.num_angles, "num_depths": lattice.num_depths,
            "num_lattice_actions": len(lattice.grasp),
            "sampled_query_anchors": selected["anchor_query_ids"],
            "num_exported_rows": len(records), "num_unique_physical_actions": len(unique_grasps),
            "decoded_width_audit": count_width_status(lattice.grasp[:, 1], max_width),
        }
        frame_log.append(log)
        print(f"[P2] {frame_no+1}/{len(indices)} {sample} "
              f"{len(records)} exported / {len(unique_grasps)} evaluated", flush=True)
    output_csv = output_dir / ("candidate_lattice_unlabeled.csv" if evaluator is None else "candidate_lattice_labeled.csv")
    _atomic_csv(output_csv, all_rows)
    manifest = {
        "version": "p2-lattice-v1",
        "checkpoint_sha256": _sha256(cfgs.checkpoint_path),
        "checkpoint_path": os.path.abspath(cfgs.checkpoint_path),
        "dataset_root": os.path.abspath(cfgs.dataset_root),
        "test_mode": cfgs.test_mode, "camera": cfgs.camera,
        "inference_uses_observed_or_GT_depth": False,
        "collision_prefilter": False,
        "evaluator": None if evaluator is None else "GraspNet CAD/table/DexNet exact-action",
        "fc_mode": DIAG.diag_eval_fc_mode if evaluator is not None else "unlabeled",
        "grasp_max_width_m": max_width,
        "width_grid_m": width_grid,
        "insertion_sweep_width": DIAG.diag_insertion_width,
        "p3_matched_insertion_modes": DIAG.diag_insertion_width == "both",
        "query_anchors": DIAG.diag_query_anchors,
        "nonwinner_per_query": DIAG.diag_nonwinners_per_query,
        "full_lattice_per_anchor": bool(DIAG.diag_full_lattice_per_anchor),
        "uniform_budget": DIAG.diag_uniform_budget or DIAG.diag_query_anchors * (1+DIAG.diag_nonwinners_per_query),
        "seed": DIAG.diag_seed,
        "frames": frame_log, "candidate_csv": str(output_csv),
        "note": "Sampling groups differ in candidate definition. Compare paired query-winner/nonwinner within the same anchors; width/insertion sweeps change physical action and require exact reevaluation.",
    }
    _atomic_json(output_dir / "lattice_manifest.json", manifest)
    print(f"[P2] Saved {output_csv} and lattice_manifest.json")


if __name__ == "__main__":
    main()
