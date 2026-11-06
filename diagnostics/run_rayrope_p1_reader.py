#!/usr/bin/env python3
"""P1 reader-only depth dose response with orthogonal coords/mask controls.

Every intervention evaluates exactly the SAME native physical grasps. Only
their CDF scores are reread at the original query, angle, depth indices;
native executed width is frozen and labels are reused from exact evaluation.
The model's new width predictions are logged but NEVER used to alter actions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _consume():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--diag_output_dir", required=True)
    p.add_argument("--diag_frame_manifest_csv", default="")
    p.add_argument("--diag_scene_ids", default="")
    p.add_argument("--diag_anno_ids", default="0,128")
    p.add_argument("--diag_max_frames", type=int, default=18)
    p.add_argument("--diag_fixed_actions", type=int, default=64)
    p.add_argument("--diag_biases_mm", default="-10,-5,0,5,10,15,20")
    p.add_argument("--diag_modes", default="coords_only,mask_only,both")
    p.add_argument("--diag_seed", type=int, default=20261106)
    p.add_argument("--diag_eval_fc_mode", choices=("official", "reuse_contacts"), default="reuse_contacts")
    p.add_argument("--diag_verify_n", type=int, default=10)
    p.add_argument("--diag_eval_chunk", type=int, default=256)
    p.add_argument("--diag_skip_eval", type=int, choices=(0, 1), default=0)
    args, remaining = p.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *remaining]
    return args


DIAG = _consume()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from diagnostics.rayrope_action_lattice import decode_cdf_lattice, physical_action_id
from inference_cva import _load_checkpoint_strict, _move_fixed_inputs
from models.economicgrasp_bip3d import economicgrasp_dpt, pred_decode_center_view_angle
from models.grasp_designs import design_options
from utils.arguments import cfgs
from utils.label_generation import batch_viewpoint_params_to_matrix


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _frames(dataset):
    scene_list = dataset.scene_list()
    if len(scene_list) != len(dataset):
        raise RuntimeError("Dataset scene_list must map each sample 1:1.")
    lookup = {(int(str(s)[-4:]), idx % 256): idx for idx,s in enumerate(scene_list)}
    if len(lookup) != len(dataset):
        raise RuntimeError("Duplicate (scene,anno) in dataset.")
    if DIAG.diag_frame_manifest_csv:
        with open(DIAG.diag_frame_manifest_csv, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if not {"scene_id", "anno_id"}.issubset(rd.fieldnames or []):
                raise ValueError("Frame CSV must include scene_id,anno_id")
            keys = [(int(r["scene_id"]), int(r["anno_id"])) for r in rd]
    else:
        ids = set(_parse_ints(DIAG.diag_scene_ids)) if DIAG.diag_scene_ids else None
        annos = _parse_ints(DIAG.diag_anno_ids)
        keys = [(s, a) for s in sorted({x[0] for x in lookup}) if ids is None or s in ids for a in annos]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated frame in manifest")
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise ValueError(f"Frames not in dataset: {missing[:10]}")
    if DIAG.diag_max_frames < 1 or DIAG.diag_fixed_actions < 1:
        raise ValueError("Nonpositive frames/actions")
    return [lookup[k] for k in keys[:DIAG.diag_max_frames]]


def _model(device):
    if not (cfgs.multi_modal and cfgs.use_cdf and cfgs.enable_point_rayrope):
        raise RuntimeError("P1 requires --multi_modal --use_cdf --enable_point_rayrope.")
    if cfgs.use_obs_depth or cfgs.use_gt_depth or cfgs.kview_mode != "A1" or cfgs.use_top4_view_infer:
        raise RuntimeError("P1 requires predicted depth, A1 single top view, no observed/GT depth.")
    model = economicgrasp_dpt(
        **design_options(cfgs), min_depth=cfgs.min_depth, max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num, is_training=False, use_obs_depth=False,
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
    reader = model.kview_grasp_module.group
    if not hasattr(reader, "set_diagnostic"):
        raise RuntimeError("Checkpoint model grouping lacks inference-only RayRoPE set_diagnostic.")
    return model, reader


def _write_csv(path, rows):
    if not rows:
        raise ValueError("No P1 rows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _native_score_strata(lattice, n, seed):
    # Select from the Q native per-query winners only, with equal budget
    # across sorted score ranks. Stratification is entirely determined by
    # the unperturbed scores; the chosen physical actions never change.
    winners = np.flatnonzero(lattice.query_winner)
    if len(winners) < n:
        raise ValueError(f"Need {n} native winners but only {len(winners)} available")
    ordered = winners[np.argsort(lattice.grasp[winners, 0], kind="stable")]
    rng = np.random.default_rng(seed)
    buckets = np.array_split(ordered, n)
    return [int(rng.choice(b)) for b in buckets]


def main():
    if cfgs.batch_size != 1:
        raise RuntimeError("Use --batch_size 1.")
    modes = [m.strip() for m in DIAG.diag_modes.split(",") if m.strip()]
    if not modes or any(m not in ("coords_only", "mask_only", "both") for m in modes):
        raise ValueError("diag_modes must be coords_only,mask_only,both")
    biases = [float(x) for x in DIAG.diag_biases_mm.split(",") if x.strip()]
    if not biases or any(not np.isfinite(x) for x in biases):
        raise ValueError("Invalid reader biases")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = GraspNetMultiDataset(
        cfgs.dataset_root, split=cfgs.test_mode, camera=cfgs.camera,
        num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False)
    indices = _frames(dataset)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=cfgs.num_workers,
                        shuffle=False, collate_fn=collate_fn,
                        persistent_workers=(cfgs.num_workers > 0))
    model,reader = _model(device)
    evaluator = None
    if not DIAG.diag_skip_eval:
        from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
        evaluator = ExactGraspNetActionEvaluator(
            cfgs.dataset_root, cfgs.camera, split=cfgs.test_mode,
            collision_chunk=DIAG.diag_eval_chunk, fc_mode=DIAG.diag_eval_fc_mode,
            verify_n=DIAG.diag_verify_n, strict=True,
        )
    outdir = Path(DIAG.diag_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows, stats = [], []
    try:
        for index,(batch,data_idx) in enumerate(zip(loader, indices)):
            scene = int(str(dataset.scene_list()[data_idx])[-4:])
            anno = int(data_idx % 256)
            sample = f"scene{scene:04d}_frame{anno:04d}"
            batch = _move_fixed_inputs(batch, device)
            batch["cva_export_angle_feature"]=False
            batch["cva_compute_diagnostics"]=False
            batch["geometry_compute_diagnostics"]=False
            # Model may append endpoints to its input dictionary; separate
            # shallow dicts keep the same immutable RGB tensors for every
            # intervention without carrying stale intermediate endpoints.
            def execute(mode=None, bias_m=0.0):
                reader.set_diagnostic(mode, bias_m)
                with torch.inference_mode():
                    ep=model(dict(batch))
                    lattice=decode_cdf_lattice(ep, 0, batch_viewpoint_params_to_matrix,
                                               max_width_m=float(cfgs.grasp_max_width))
                diagnostics=dict(reader.last_diagnostic_stats)
                return lattice,diagnostics
            native,_=execute(None,0.0)
            native_pred = pred_decode_center_view_angle(dict(ep) if False else ep, use_cdf=True) if False else None
            fixed = _native_score_strata(
                native, DIAG.diag_fixed_actions, DIAG.diag_seed + scene*1009 + anno*101)
            original_actions = native.grasp[fixed]
            labels = None
            if evaluator is not None:
                labels = evaluator.evaluate(scene, anno, original_actions)
            conditions = [("native", None, 0.0)]
            for bias_mm in biases:
                if bias_mm == 0:
                    continue
                name = ("p" if bias_mm > 0 else "m") + f"{abs(bias_mm):g}mm"
                for mode in modes:
                    conditions.append((f"{mode}_{name}", mode, bias_mm * 0.001))
            for cond,mode,bias_m in conditions:
                lattice, debug = (native, {}) if mode is None else execute(mode,bias_m)
                # Control: same seed, top view, angle, insertion across passes.
                if lattice.grasp.shape != native.grasp.shape:
                    raise RuntimeError(f"P1 lattice shape changed: {cond}")
                if not np.allclose(lattice.grasp[fixed,3:16],
                                   original_actions[:,3:16], atol=1e-6, rtol=0):
                    raise RuntimeError(f"P1 geometry/pose changed in reader-only intervention {cond}")
                stats.append({"sample":sample,"condition":cond,**debug})
                for slot,source_idx in enumerate(fixed):
                    grasp=original_actions[slot]
                    friction=float(labels.friction[slot]) if labels is not None else float("nan")
                    bits={f"success_mu_{mu:.1f}": int(friction>0 and friction<=mu+1e-7) if labels is not None else ""
                          for mu in (0.2,0.4,0.6,0.8,1.0,1.2)}
                    out = {
                        "sample_id":sample, "scene_id":scene, "anno_id":anno, "split":cfgs.test_mode,
                        "condition":cond, "intervention":cond,
                        "candidate_id":physical_action_id(grasp),
                        "query_id":int(native.query_id[source_idx]),
                        "angle_id":int(native.angle_id[source_idx]),
                        "depth_id":int(native.depth_id[source_idx]),
                        "score":float(lattice.grasp[source_idx,0]),
                        "native_score":float(grasp[0]),
                        "width_m":float(grasp[1]), "height_m":float(grasp[2]),
                        "insertion_depth_m":float(grasp[3]),
                        "width_predicted_under_intervention_m":float(lattice.grasp[source_idx,1]),
                        **{f"r{i}{j}":float(grasp[4+i*3+j]) for i in range(3) for j in range(3)},
                        "center_x":float(grasp[13]),"center_y":float(grasp[14]),"center_z":float(grasp[15]),
                        "friction":friction if labels is not None else "",
                        "quality":float(np.mean(list(bits.values()))) if labels is not None else "",
                        "collision_or_empty":int(labels.collision_or_empty[slot]) if labels is not None else "",
                        "pure_collision":int(labels.pure_collision[slot]) if labels is not None else "",
                        "empty":int(labels.empty[slot]) if labels is not None else "",
                        **bits,
                    }
                    all_rows.append(out)
            print(f"[P1] {index+1}/{len(indices)} {sample} {len(fixed)} fixed actions; {len(conditions)} modes",flush=True)
    finally:
        reader.set_diagnostic(None)
    csvpath = outdir / ("p1_fixed_actions_unlabeled.csv" if evaluator is None else "p1_fixed_actions_labeled.csv")
    _write_csv(csvpath, all_rows)
    _write_json(outdir / "p1_interventions.json", {
        "checkpoint_sha256":_sha256(cfgs.checkpoint_path),
        "test_mode":cfgs.test_mode, "camera":cfgs.camera,
        "num_frames":len(indices), "num_fixed_actions_per_frame":DIAG.diag_fixed_actions,
        "biases_mm":biases, "modes":modes,
        "depth_intervention_applied_only_to_rayrope_reader":True,
        "physical_actions_and_evaluator_labels_frozen":True,
        "model_width_predictions_may_change_but_executed_width_is_frozen":True,
        "candidate_sampling":"native per-query winners, score-rank stratified",
        "evaluation":"CAD/DexNet exact action" if evaluator else "not evaluated",
        "mask_statistics":stats,
        "candidate_csv":str(csvpath),
        "note":"This isolates reader positional coordinates and in-front mask; it is not a placement or full-system depth intervention.",
    })
    print(f"[P1] Saved {csvpath} and p1_interventions.json")


if __name__ == "__main__":
    main()
