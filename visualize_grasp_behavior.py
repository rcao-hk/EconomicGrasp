#!/usr/bin/env python3
"""Visualize EconomicGrasp-DPT/CVA/CDF and DCR behavior on fixed GraspNet views.

Default protocol:
  camera: realsense
  splits: test_seen,test_similar,test_novel
  frames per scene: 0,128,255

The runner is scene-shardable across GPUs.  Heavy evaluator inspection is
optional through --items all / evaluator and is never required for lightweight
training-time hooks.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dcr_cva_common import DCR_VERSION, anchored_score
from e1e2_common import (VERSION, load_torch, make_dataset, move_batch, seed_all,
                         seed_for, select_centers)
from tools.dcr_visual_probe import inspect_dcr_case, inspect_e1_like_case
from tools.grasp_behavior_viz import (
    BehaviorVizWriter, depth_to_points, imagenet_rgb, make_contact_sheet,
    parse_items, save_center_cdf_panels, save_center_selection_motion,
    save_corruption_motion, save_depth_bundle, save_feature_bundle, save_grasp_overlay,
    save_grasp_scene_ply, save_local_patch_overlay, save_proposal_bundle,
    save_query_response, save_query_scalar_overlay, save_rgb, save_spatial_bundle,
    save_stage1_angle_depth, save_view_response, write_csv, write_html_index,
    write_json,
)

SPLIT_SCENES = {
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--dcr-checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--e1-checkpoint", default="",
                   help="Optional E1 checkpoint for E1-vs-DCR center-selection panels.")
    p.add_argument("--air-checkpoint", default="",
                   help="Optional DCR-AIR checkpoint for AIR residual/selection panels.")
    p.add_argument("--camera", default="realsense", choices=("realsense", "kinect"))
    p.add_argument("--splits", default="test_seen,test_similar,test_novel")
    p.add_argument("--frames", default="0,128,255")
    p.add_argument("--cases",
                   default="nominal,bias:-25,bias:25,smooth:10",
                   help="Comma-separated corruption cases. nominal is always first.")
    p.add_argument("--items", default="core",
                   help="light/core/all or comma list: rgb,depth,pointcloud,proposal,"
                        "feature,spatial,view,cdf,query_response,local,grasps,"
                        "corruption_delta,evaluator,air")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--local-queries", type=int, default=8)
    p.add_argument("--query-limit", type=int, default=0)
    p.add_argument("--query-chunk", type=int, default=64)
    p.add_argument("--point-stride", type=int, default=3)
    p.add_argument("--eval-methods", default="dcr",
                   help="Comma list from native,dcr,e1,air; only used when evaluator is enabled.")
    p.add_argument("--eval-cases", default="nominal",
                   help="Comma-separated cases sent to detailed Dex-Net evaluation. Use all to evaluate every case.")
    p.add_argument("--eval-topk", type=int, default=50)
    p.add_argument("--eval-workers", type=int, default=1,
                   help="Reserved for compatibility; detailed per-frame evaluator is sequential.")
    p.add_argument("--eval-voxel-size", type=float, default=.008)
    p.add_argument("--nms-trans-th", type=float, default=.03)
    p.add_argument("--nms-rot-deg", type=float, default=30.)
    p.add_argument("--max-width", type=float, default=.1)
    p.add_argument("--scene-ids", default="",
                   help="Optional comma-separated global scene ids.")
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2051)
    p.add_argument("--resume", action="store_true")
    return p


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> List[int]:
    out = sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})
    return out


def load_dcr(stage1: str, dcr: str, device):
    from models.economicgrasp_cva_air import load_frozen_dcr
    model, protocol = load_frozen_dcr(stage1, dcr, device)
    return model, protocol


def load_e1_as_zero_dcr(stage1: str, e1_checkpoint: str, device,
                        fallback_group_chunk: int = 512):
    from models.economicgrasp_cva_centers import load_reference
    from models.economicgrasp_cva_dcr import DecoupledCenterRankingCVA
    ck = load_torch(e1_checkpoint)
    if ck.get("version") != VERSION:
        raise RuntimeError("E1 visualization checkpoint must be an E1/E2 checkpoint")
    protocol = ck["protocol"]
    cfg = protocol["config"]
    model = DecoupledCenterRankingCVA(
        load_reference(stage1, device),
        protocol["offsets_mm"],
        cfg.get("group_chunk", fallback_group_chunk),
        rank_hidden=128, rank_bound=.5, rank_seed=2052).to(device)
    model.warm_start_corrector(ck["model"])
    model.eval().requires_grad_(False)
    del ck
    return model


def load_air_reader(dcr_model, air_checkpoint: str, device):
    if not air_checkpoint:
        return None, None
    from dcr_air_common import AIR_VERSION
    from models.economicgrasp_cva_air import ActionImageDCR
    ck = load_torch(air_checkpoint)
    if ck.get("version") != AIR_VERSION:
        raise RuntimeError("Unsupported AIR checkpoint")
    cfg = ck["protocol"]["config"]
    model = ActionImageDCR(
        dcr_model, hidden=cfg["hidden"],
        bound=cfg["residual_bound"], seed=cfg["seed"]).to(device)
    model.load_learned_state(ck["model"])
    model.eval()
    protocol = ck["protocol"]
    del ck
    return model, protocol


def selected_action_from_logits(logits, bundle, zero, offsets):
    utility = logits.sigmoid().mean(-1)
    selected = select_centers(utility, bundle["valid"], zero)
    q = torch.arange(len(selected), device=selected.device)
    out = bundle["actions"][selected, q].detach().clone()
    out[:, 0] = bundle["actions"][zero, :, 0]
    return out, selected, offsets[selected], utility


def _subset_raw_item(ds, lookup, sid, aid):
    return ds[lookup[(sid, aid)]]


def _batch_from_raw_item(item, device):
    from dataset.graspnet_dataset import collate_fn
    keys = (
        "img", "K", "camera_pose_vec", "camera_gravity_vec",
        "scene_idx", "anno_idx", "token_valid_mask",
    )
    return move_batch(
        collate_fn([{k: item[k] for k in keys if k in item}]),
        device)


class DetailedEvaluator:
    """Current-scene-only GraspNet/Dex-Net inspector."""

    def __init__(self, dataset_root: str, camera: str, voxel_size: float,
                 topk: int, nms_trans: float, nms_rot_deg: float,
                 max_width: float):
        from graspnetAPI.graspnet_eval import GraspNetEval
        from graspnetAPI.utils.config import get_config
        from graspnetAPI.utils.eval_utils import create_table_points
        self.ge = GraspNetEval(dataset_root, camera, split="test")
        self.config = get_config()
        self.base_table = create_table_points(
            1.0, 1.0, .05, dx=-.5, dy=-.5, dz=-.05, grid_size=voxel_size)
        self.voxel_size = float(voxel_size)
        self.topk = int(topk)
        self.nms_trans = float(nms_trans)
        self.nms_rot = float(nms_rot_deg) / 180. * np.pi
        self.max_width = float(max_width)
        self.scene_id = None
        self.models = None
        self.dexmodels = None
        self.obj_list = None

    def set_scene(self, sid: int):
        if self.scene_id == int(sid):
            return
        self.clear()
        from graspnetAPI.utils.eval_utils import voxel_sample_points
        models, dexmodels, obj_list = self.ge.get_scene_models(int(sid), ann_id=0)
        self.models = [voxel_sample_points(m, self.voxel_size) for m in models]
        self.dexmodels = dexmodels
        self.obj_list = obj_list
        self.scene_id = int(sid)

    def evaluate(self, ann_id: int, grasps: np.ndarray):
        from graspnetAPI.grasp import GraspGroup
        from graspnetAPI.utils.eval_utils import transform_points
        from diagnose_graspnet_eval import clip_width_like_eval, eval_grasp_detailed
        _, pose_list, camera_pose, align_mat = self.ge.get_model_poses(
            self.scene_id, int(ann_id))
        table = transform_points(
            self.base_table,
            np.linalg.inv(np.matmul(align_mat, camera_pose)))
        gg = GraspGroup(np.asarray(grasps, np.float32).copy())
        clip_width_like_eval(gg, self.max_width)
        return eval_grasp_detailed(
            gg, self.models, self.dexmodels, pose_list, self.config, table,
            voxel_size=self.voxel_size, top_k=self.topk,
            nms_trans_th=self.nms_trans, nms_rot_rad=self.nms_rot)

    def clear(self):
        self.models = None
        self.dexmodels = None
        self.obj_list = None
        self.scene_id = None
        gc.collect()


def save_eval_artifacts(out, method, result, rgb, K, scene_points, scene_colors,
                        topk: int):
    from diagnose_graspnet_eval import make_ranked_rows
    method_dir = Path(out) / "evaluator" / method
    method_dir.mkdir(parents=True, exist_ok=True)
    write_csv(method_dir / "ranked_eval.csv", make_ranked_rows(result, topk))
    save_grasp_overlay(
        method_dir / "ranked_eval_overlay.png", rgb,
        result["ranked_grasps"], K,
        title=f"{method}: evaluator-ranked grasps",
        topk=topk, eval_scores=result["ranked_eval_scores"],
        collision=result["ranked_collision"])
    save_grasp_scene_ply(
        method_dir / "ranked_eval_scene.ply",
        scene_points, scene_colors, result["ranked_grasps"],
        topk=topk, eval_scores=result["ranked_eval_scores"],
        collision=result["ranked_collision"])
    write_json(method_dir / "summary.json", result["stats"])


def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Bad shard id/count")
    if min(args.topk, args.local_queries, args.query_chunk,
           args.point_stride, args.eval_topk) < 1:
        raise ValueError("Positive visualization counts required")
    seed_all(args.seed)
    device = torch.device(args.device)

    splits = parse_csv(args.splits)
    frames = parse_int_csv(args.frames)
    if any(x not in (0, 128, 255) for x in frames):
        # Custom values are allowed; this message is only recorded in protocol.
        pass
    cases = parse_csv(args.cases)
    cases = list(dict.fromkeys(["nominal"] + cases))
    items = parse_items(args.items)
    selected_scenes = set(parse_int_csv(args.scene_ids)) if args.scene_ids else None

    dcr, dcr_protocol = load_dcr(
        args.stage1_checkpoint, args.dcr_checkpoint, device)
    e1_model = (load_e1_as_zero_dcr(
        args.stage1_checkpoint, args.e1_checkpoint, device,
        dcr_protocol["config"].get("group_chunk", 512))
        if args.e1_checkpoint else None)
    air_model, air_protocol = load_air_reader(
        dcr, args.air_checkpoint, device) if args.air_checkpoint else (None, None)

    out_root = Path(args.output_root)
    writer = BehaviorVizWriter(out_root, items=args.items, every=1)

    protocol = {
        "camera": args.camera,
        "splits": splits,
        "frames": frames,
        "cases": cases,
        "items": sorted(items),
        "topk": args.topk,
        "local_queries": args.local_queries,
        "query_limit": args.query_limit,
        "query_chunk": args.query_chunk,
        "point_stride": args.point_stride,
        "eval_cases": args.eval_cases,
        "dcr_checkpoint": str(args.dcr_checkpoint),
        "e1_checkpoint": str(args.e1_checkpoint) if args.e1_checkpoint else None,
        "air_checkpoint": str(args.air_checkpoint) if args.air_checkpoint else None,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "note": "realsense fixed-view behavior visualization; evaluator is diagnostic, not official split AP",
    }
    write_json(out_root / f"protocol_shard{args.shard_id}.json", protocol)

    tasks: List[Tuple[str, int]] = []
    for split in splits:
        if split not in SPLIT_SCENES:
            raise ValueError(split)
        for sid in SPLIT_SCENES[split]:
            if selected_scenes is None or sid in selected_scenes:
                tasks.append((split, sid))
    if args.max_scenes:
        tasks = tasks[:args.max_scenes]
    tasks = [x for i, x in enumerate(tasks)
             if i % args.num_shards == args.shard_id]

    evaluator = None
    if "evaluator" in items:
        evaluator = DetailedEvaluator(
            args.dataset_root, args.camera, args.eval_voxel_size,
            args.eval_topk, args.nms_trans_th, args.nms_rot_deg,
            args.max_width)

    index_groups: Dict[str, List[Tuple[str, Path]]] = {}
    summary_rows = []
    datasets = {}

    for split, sid in tasks:
        if split not in datasets:
            datasets[split] = make_dataset(args.dataset_root, split, args.camera)
        ds, lookup = datasets[split]
        if evaluator is not None:
            evaluator.set_scene(sid)

        for aid in frames:
            frame_root = out_root / split / f"scene_{sid:04d}" / f"ann_{aid:04d}"
            done = frame_root / "_SUCCESS.json"
            if args.resume and done.is_file():
                continue

            raw_item = _subset_raw_item(ds, lookup, sid, aid)
            batch = _batch_from_raw_item(raw_item, device)
            rgb = imagenet_rgb(batch["img"])
            K = batch["K"]

            from models.economicgrasp_cva_centers import extract_depth_features
            pack = extract_depth_features(dcr.reference, batch)
            nominal_reference = None
            case_groups = {}

            for case in cases:
                case_dir = writer.frame_dir(split, sid, aid, case=case)
                snapshot = inspect_dcr_case(
                    dcr, batch, case, seed_for(2030, sid, aid, case),
                    query_limit=args.query_limit,
                    query_chunk=args.query_chunk,
                    depth_pack=pack, local_queries=args.local_queries)
                snapshot["bundle_batch_K"] = K

                # Correct CDF panel with the actual DCR offset grid.
                if "cdf" in items:
                    save_center_cdf_panels(
                        case_dir / "dcr_center_cdf.png",
                        snapshot["cdf_logits"],
                        dcr.corrector.offsets_mm,
                        snapshot["top_query_indices"].cpu().tolist(),
                        selected=snapshot["selected"])

                images: List[Tuple[str, Path]] = []
                files: List[Tuple[str, Path]] = []

                if "rgb" in items:
                    p = case_dir / "rgb.png"
                    save_rgb(p, rgb, "RGB input")
                    images.append(("RGB", p))

                if "depth" in items or "pointcloud" in items:
                    save_depth_bundle(
                        case_dir, rgb, K,
                        snapshot["nominal_depth"], snapshot["active_depth"],
                        sensor=raw_item.get("sensor_depth_m"),
                        rendered=raw_item.get("gt_depth_m"),
                        point_stride=args.point_stride)
                    for name in (
                            "depth_pred_nominal.png", "depth_active.png",
                            "depth_corruption_delta_mm.png", "depth_sensor.png",
                            "depth_rendered.png",
                            "depth_pred_nominal_minus_sensor_mm.png",
                            "depth_pred_nominal_minus_rendered_mm.png",
                            "depth_active_minus_sensor_mm.png",
                            "depth_active_minus_rendered_mm.png"):
                        p = case_dir / name
                        if p.is_file():
                            images.append((name, p))
                    for name in (
                            "pointcloud_pred_nominal.ply", "pointcloud_active.ply",
                            "pointcloud_sensor.ply", "pointcloud_rendered.ply"):
                        p = case_dir / name
                        if p.is_file():
                            files.append((name, p))

                if "proposal" in items:
                    save_proposal_bundle(case_dir, rgb, snapshot["proposal_logits"])
                    images.extend([
                        ("Objectness", case_dir / "objectness_overlay.png"),
                        ("Graspness", case_dir / "graspness_overlay.png"),
                    ])

                if "feature" in items:
                    save_feature_bundle(
                        case_dir, snapshot["feature_pre"], snapshot["feature_post"])
                    images.extend([
                        ("Pre-enhancer feature PCA", case_dir / "feature_pre_enhancer_pca.png"),
                        ("Post-enhancer feature PCA", case_dir / "feature_post_enhancer_pca.png"),
                    ])

                if "spatial" in items:
                    save_spatial_bundle(case_dir, rgb, snapshot["spatial_aux"])
                    for name in ("spatial_gate_overlay.png",
                                 "spatial_delta_abs_overlay.png",
                                 "spatial_update_abs_overlay.png"):
                        p = case_dir / name
                        if p.is_file():
                            images.append((name, p))

                if "view" in items:
                    save_view_response(
                        case_dir, rgb, snapshot["stage1_end_points"],
                        snapshot["bundle"])
                    for name in ("view_entropy.png", "view_margin.png"):
                        p = case_dir / name
                        if p.is_file():
                            images.append((name, p))

                if "cdf" in items:
                    save_stage1_angle_depth(
                        case_dir / "stage1_angle_depth.png",
                        snapshot["stage1_end_points"],
                        snapshot["bundle"]["query_ids"],
                        snapshot["top_query_indices"].cpu().tolist())
                    images.extend([
                        ("DCR center CDF", case_dir / "dcr_center_cdf.png"),
                        ("Stage-1 angle/depth CDF", case_dir / "stage1_angle_depth.png"),
                    ])

                if "query_response" in items:
                    save_query_response(
                        case_dir, rgb, snapshot["bundle"]["token_ids"],
                        snapshot["stage1_score"],
                        snapshot["selected_offsets_mm"],
                        snapshot["local_utility"])
                    images.extend([
                        ("Stage-1 query score", case_dir / "query_stage1_score.png"),
                        ("Selected offset", case_dir / "query_selected_offset_mm.png"),
                        ("Best local utility", case_dir / "query_best_local_utility.png"),
                    ])

                    q_rank = torch.arange(
                        len(snapshot["selected"]),
                        device=snapshot["selected"].device)
                    selected_rank_residual = snapshot["rank_residual"][
                        snapshot["selected"], q_rank]
                    anchored = anchored_score(
                        snapshot["stage1_score"], selected_rank_residual, 1.0)
                    save_query_scalar_overlay(
                        case_dir / "dcr_rank_residual_selected.png",
                        rgb, snapshot["bundle"]["token_ids"],
                        selected_rank_residual,
                        "DCR learned rank log-odds residual (diagnostic)",
                        cmap="coolwarm", symmetric=True)
                    save_query_scalar_overlay(
                        case_dir / "dcr_anchored_minus_stage1_score.png",
                        rgb, snapshot["bundle"]["token_ids"],
                        anchored - snapshot["stage1_score"],
                        "Anchored score - frozen Stage-1 score (diagnostic)",
                        cmap="coolwarm", symmetric=True)
                    images.extend([
                        ("DCR learned rank residual",
                         case_dir / "dcr_rank_residual_selected.png"),
                        ("Anchored score shift",
                         case_dir / "dcr_anchored_minus_stage1_score.png"),
                    ])

                    motion_stats = save_center_selection_motion(
                        case_dir / "dcr_center_correction_motion.png",
                        rgb, K, snapshot["bundle"]["actions"],
                        snapshot["selected"], dcr.zero,
                        dcr.corrector.offsets_mm,
                        title="DCR native→selected")
                    write_json(case_dir / "dcr_center_correction_motion.json",
                               motion_stats)
                    images.append(("DCR center correction motion",
                                   case_dir / "dcr_center_correction_motion.png"))

                if "local" in items:
                    p = case_dir / "local_attention_overlay.png"
                    save_local_patch_overlay(
                        p, rgb, snapshot["local_debug"],
                        max_queries=args.local_queries)
                    images.append(("Local CVA attention", p))

                methods: Dict[str, np.ndarray] = {
                    "native": snapshot["outputs"]["native"].cpu().numpy(),
                    "dcr": snapshot["outputs"]["stage1"].cpu().numpy(),
                }

                e1_info = None
                if e1_model is not None:
                    e1_info = inspect_e1_like_case(
                        e1_model, batch, snapshot["bundle"],
                        snapshot["active_depth"], pack,
                        query_chunk=args.query_chunk)
                    q = torch.arange(len(e1_info["selected"]), device=device)
                    e1_action = snapshot["bundle"]["actions"][
                        e1_info["selected"], q].detach().clone()
                    e1_action[:, 0] = snapshot["stage1_score"]
                    methods["e1"] = e1_action.cpu().numpy()
                    if "query_response" in items:
                        save_query_response(
                            case_dir / "e1", rgb,
                            snapshot["bundle"]["token_ids"],
                            snapshot["stage1_score"],
                            e1_info["selected_offsets_mm"],
                            e1_info["local_utility"])
                        delta_off = (
                            snapshot["selected_offsets_mm"] -
                            e1_info["selected_offsets_mm"])
                        save_query_scalar_overlay(
                            case_dir / "e1_vs_dcr_offset_delta_mm.png",
                            rgb, snapshot["bundle"]["token_ids"], delta_off,
                            "DCR selected offset - E1 selected offset [mm]",
                            symmetric=True)
                        save_center_selection_motion(
                            case_dir / "e1" / "center_correction_motion.png",
                            rgb, K, snapshot["bundle"]["actions"],
                            e1_info["selected"], e1_model.zero,
                            e1_model.corrector.offsets_mm,
                            title="E1 native→selected")
                        images.extend([
                            ("E1 selected offset",
                             case_dir / "e1" / "query_selected_offset_mm.png"),
                            ("DCR-E1 offset delta",
                             case_dir / "e1_vs_dcr_offset_delta_mm.png"),
                            ("E1 center correction motion",
                             case_dir / "e1" / "center_correction_motion.png"),
                        ])

                air_info = None
                if air_model is not None and "air" in items:
                    h, w = rgb.shape[:2]
                    residual, diag = air_model.reader(
                        snapshot["feature_pre"].detach(), K, (h, w),
                        snapshot["bundle"]["actions"],
                        snapshot["bundle"]["valid"])
                    fused = snapshot["cdf_logits"] + residual[..., None]
                    air_action, air_sel, air_off, air_u = selected_action_from_logits(
                        fused, snapshot["bundle"], air_model.zero,
                        air_model.base.corrector.offsets_mm)
                    methods["air"] = air_action.cpu().numpy()
                    air_info = {
                        "residual": residual, "diag": diag,
                        "selected": air_sel, "offset": air_off, "utility": air_u,
                    }
                    save_query_response(
                        case_dir / "air", rgb,
                        snapshot["bundle"]["token_ids"],
                        snapshot["stage1_score"], air_off, air_u)
                    save_query_scalar_overlay(
                        case_dir / "air" / "air_logit_residual_selected.png",
                        rgb, snapshot["bundle"]["token_ids"],
                        residual[air_sel, torch.arange(
                            len(air_sel), device=air_sel.device)],
                        "AIR scalar logit residual at selected center",
                        cmap="coolwarm", symmetric=True)
                    save_query_scalar_overlay(
                        case_dir / "air" / "air_minus_dcr_offset_mm.png",
                        rgb, snapshot["bundle"]["token_ids"],
                        air_off - snapshot["selected_offsets_mm"],
                        "AIR selected offset - DCR selected offset [mm]",
                        symmetric=True)
                    save_center_selection_motion(
                        case_dir / "air" / "center_correction_motion.png",
                        rgb, K, snapshot["bundle"]["actions"],
                        air_sel, air_model.zero,
                        air_model.base.corrector.offsets_mm,
                        title="AIR native→selected")
                    images.extend([
                        ("AIR selected offset",
                         case_dir / "air" / "query_selected_offset_mm.png"),
                        ("AIR residual",
                         case_dir / "air" / "air_logit_residual_selected.png"),
                        ("AIR-DCR offset delta",
                         case_dir / "air" / "air_minus_dcr_offset_mm.png"),
                        ("AIR center correction motion",
                         case_dir / "air" / "center_correction_motion.png"),
                    ])

                active_pts, active_cols = depth_to_points(
                    snapshot["active_depth"], K, rgb,
                    stride=args.point_stride)

                if "grasps" in items:
                    for method, arr in methods.items():
                        p = case_dir / f"grasps_{method}_overlay.png"
                        save_grasp_overlay(
                            p, rgb, arr, K,
                            title=f"{method} Top-{args.topk}",
                            topk=args.topk)
                        images.append((f"{method} grasps", p))
                        ply = case_dir / f"grasps_{method}_scene.ply"
                        if save_grasp_scene_ply(
                                ply, active_pts, active_cols, arr,
                                topk=args.topk):
                            files.append((f"{method} scene PLY", ply))

                if case == "nominal":
                    nominal_reference = {
                        "bundle": {
                            "token_ids": snapshot["bundle"]["token_ids"].detach().cpu(),
                            "native": snapshot["bundle"]["native"].detach().cpu(),
                        },
                        "dcr_action": snapshot["outputs"]["stage1"].detach().cpu().numpy(),
                    }
                elif "corruption_delta" in items and nominal_reference is not None:
                    stats = save_corruption_motion(
                        case_dir / "corruption_native_motion.png",
                        rgb, K, nominal_reference["bundle"],
                        snapshot["bundle"])
                    write_json(case_dir / "corruption_motion.json", stats)
                    images.append(("Nominal→corrupted Stage-1 center motion",
                                   case_dir / "corruption_native_motion.png"))

                    if "grasps" in items:
                        save_grasp_overlay(
                            case_dir / "grasps_nominal_dcr_reference.png",
                            rgb, nominal_reference["dcr_action"], K,
                            title="Nominal DCR reference", topk=args.topk)
                        images.append(("Nominal DCR reference",
                                       case_dir / "grasps_nominal_dcr_reference.png"))

                eval_summaries = {}
                eval_cases = set(parse_csv(args.eval_cases))
                evaluate_this_case = (
                    evaluator is not None and
                    ("all" in eval_cases or case in eval_cases))
                if evaluate_this_case:
                    eval_methods = set(parse_csv(args.eval_methods))
                    for method, arr in methods.items():
                        if method not in eval_methods:
                            continue
                        detailed = evaluator.evaluate(aid, arr)
                        save_eval_artifacts(
                            case_dir, method, detailed, rgb, K,
                            active_pts, active_cols, args.eval_topk)
                        eval_summaries[method] = detailed["stats"]
                        images.append((
                            f"{method} evaluator outcome",
                            case_dir / "evaluator" / method /
                            "ranked_eval_overlay.png"))

                case_manifest = {
                    "split": split, "scene_id": sid, "anno_id": aid,
                    "case": case,
                    "num_queries": int(len(snapshot["selected"])),
                    "dcr_move_rate": float(
                        (snapshot["selected"] != dcr.zero).float().mean()),
                    "dcr_mean_offset_mm": float(
                        snapshot["selected_offsets_mm"].float().mean()),
                    "stage1_score_mean": float(snapshot["stage1_score"].mean()),
                    "local_utility_mean": float(snapshot["local_utility"].mean()),
                    "evaluator": eval_summaries,
                    "air_enabled": air_info is not None,
                    "e1_enabled": e1_info is not None,
                }
                write_json(case_dir / "manifest.json", case_manifest)

                sheet_items = images[:12]
                make_contact_sheet(
                    case_dir / "overview.png", sheet_items,
                    cols=3, thumb_width=380)
                images.insert(0, ("Overview", case_dir / "overview.png"))
                case_groups[case] = images + files
                summary_rows.append(case_manifest)

                del snapshot, e1_info, air_info, methods
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # One browser page per scene/frame, grouped by corruption case.
            write_html_index(
                frame_root / "index.html",
                f"{split} scene_{sid:04d} ann_{aid:04d}",
                case_groups, base_dir=frame_root)
            write_json(done, {
                "split": split, "scene_id": sid, "anno_id": aid,
                "cases": cases, "items": sorted(items),
            })
            index_groups.setdefault(split, []).append(
                (f"scene_{sid:04d} ann_{aid:04d}",
                 frame_root / "index.html"))
            del batch, raw_item, pack
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if evaluator is not None:
            evaluator.clear()
        gc.collect()

    datasets.clear()
    write_csv(out_root / f"summary_shard{args.shard_id}.csv", summary_rows)
    write_html_index(
        out_root / f"index_shard{args.shard_id}.html",
        "EconomicGrasp behavior visualization",
        index_groups, base_dir=out_root)
    print(
        f"[GRASP-VIZ] shard {args.shard_id}/{args.num_shards}: "
        f"{len(summary_rows)} case visualizations -> {out_root}",
        flush=True)


if __name__ == "__main__":
    main()
