#!/usr/bin/env python3
"""P0 diagnostic: can center mismatch be transported into grasp depth?

This diagnostic never calls the GraspNet/Dex-Net evaluator. It pairs a trained
RGB/predicted-depth student with the Stage-0 clean-depth teacher at the exact
same image-FPS seed and exact same selected view, then decomposes their physical
center mismatch into approach-parallel and approach-lateral components.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def _consume_p0_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--p0_student_checkpoint", required=True)
    parser.add_argument(
        "--p0_split",
        choices=("test_seen", "test_similar", "test_novel"),
        default="test_seen",
    )
    parser.add_argument("--p0_output_dir", default=None)
    parser.add_argument("--p0_max_batches", type=int, default=0)
    parser.add_argument("--p0_point_match_thresh_mm", type=float, default=5.0)
    parser.add_argument("--p0_tolerated_parallel_mm", type=float, default=30.0)
    parser.add_argument("--p0_depth_start_mm", type=float, default=10.0)
    parser.add_argument("--p0_depth_interval_mm", type=float, default=10.0)
    parser.add_argument("--p0_query_sample_per_image", type=int, default=16)
    parser.add_argument("--p0_save_query_rows", type=int, choices=(0, 1), default=1)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    return args


P0 = _consume_p0_args()

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import train_cva_distill_ddp as base
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_dpt_distill import (
    DISTILL_CONTRACT_VERSION,
    economicgrasp_dpt_student,
    economicgrasp_dpt_teacher,
    load_checkpoint_state,
)
from models.p0_pose_space_cdf_transport import (
    P0TransportConfig,
    compute_p0_pose_space_transport_diagnostics,
)

cfgs = base.cfgs

DISTANCE_HIST_EDGES_MM = [
    0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0,
    float("inf"),
]
SHIFT_HIST_LABELS = ["<=-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", ">=5"]


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _validate_common_protocol() -> None:
    if int(base.DISTILL_ARGS.distill_stage) != 2:
        raise RuntimeError("P0 requires --distill_stage 2.")
    if not str(base.DISTILL_ARGS.teacher_checkpoint).strip():
        raise RuntimeError("P0 requires a Stage-0 --teacher_checkpoint.")
    if not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("P0 is a CVA-CDF diagnostic; add --use_cdf.")
    if not bool(getattr(cfgs, "multi_modal", False)):
        raise RuntimeError("P0 uses the EconomicGrasp-DPT CVA path; add --multi_modal.")
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError("P0 student must use predicted depth; remove --use_obs_depth.")
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError("Do not use the legacy dataset --use_gt_depth switch in P0.")
    if bool(getattr(cfgs, "use_depth_comp", False)):
        raise RuntimeError("P0 measures compensation counterfactually; remove --use_depth_comp.")
    if bool(getattr(cfgs, "pin_memory", False)):
        raise RuntimeError("Remove --pin_memory for CVA variable-size labels.")
    if getattr(cfgs, "checkpoint_path", None):
        raise RuntimeError("Use --p0_student_checkpoint, not --checkpoint_path.")
    if bool(getattr(cfgs, "resume", False)):
        raise RuntimeError("P0 is read-only; remove --resume.")


def _validate_student_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    stage = checkpoint.get("distill_stage", None)
    if stage is not None and int(stage) not in (1, 2):
        raise RuntimeError(f"P0 student checkpoint must be Stage 1 or 2, got {stage!r}.")
    contract = checkpoint.get("distill_contract_version", None)
    if contract is not None and int(contract) != int(DISTILL_CONTRACT_VERSION):
        raise RuntimeError(
            f"P0 requires distill_contract_version={DISTILL_CONTRACT_VERSION}, got {contract!r}."
        )
    source = checkpoint.get("geometry_depth_source", None)
    if source is not None and str(source) != "pred":
        raise RuntimeError(f"P0 student must use predicted geometry depth, got {source!r}.")
    seed_mode = checkpoint.get("seed_selection_mode", None)
    if seed_mode is not None and str(seed_mode) != "image_fps":
        raise RuntimeError(f"P0 requires image_fps student queries, got {seed_mode!r}.")
    saved_fuse = checkpoint.get("use_fuse_depth", None)
    if saved_fuse is not None and bool(saved_fuse) != bool(cfgs.use_fuse_depth):
        raise RuntimeError("Student checkpoint/use_fuse_depth mismatch.")
    saved_pose = checkpoint.get("pose_depth_mode", None)
    current_pose = str(getattr(cfgs, "pose_depth_mode", "none") or "none")
    if saved_pose is not None and str(saved_pose) != current_pose:
        raise RuntimeError(
            f"Student checkpoint pose_depth_mode={saved_pose!r}, current={current_pose!r}."
        )


def _validate_teacher_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if int(checkpoint.get("distill_stage", -1)) != 0:
        raise RuntimeError("P0 teacher checkpoint must be Stage 0.")
    if int(checkpoint.get("distill_contract_version", -1)) != int(DISTILL_CONTRACT_VERSION):
        raise RuntimeError("P0 teacher checkpoint has an incompatible distillation contract.")
    if str(checkpoint.get("seed_selection_mode", "")) != "image_fps":
        raise RuntimeError("P0 teacher checkpoint must use image_fps.")
    if str(checkpoint.get("geometry_depth_source", "")) != "gt":
        raise RuntimeError("P0 teacher checkpoint must use geometry_depth_source='gt'.")
    if bool(checkpoint.get("depth_head_executed", True)):
        raise RuntimeError("P0 teacher checkpoint says its depth head executed.")
    if str(checkpoint.get("pose_depth_mode", "")) != "none":
        raise RuntimeError("P0 Stage-0 teacher must use pose_depth_mode='none'.")
    if bool(checkpoint.get("legacy_dataset_use_gt_depth", True)):
        raise RuntimeError("P0 rejects legacy dataset --use_gt_depth Stage-0 checkpoints.")
    if bool(checkpoint.get("use_fuse_depth", False)) != bool(cfgs.use_fuse_depth):
        raise RuntimeError("Teacher checkpoint/use_fuse_depth mismatch.")


def _build_dataset(split: str) -> CVAExtendedLabelAdapter:
    base_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        camera=cfgs.camera,
        split=split,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        voxel_size=cfgs.voxel_size,
        use_gt_depth=False,
        use_fuse_depth=cfgs.use_fuse_depth,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        depth_strides=1,
        extend_angle=True,
        load_grasp_payload=False,
    )
    label_folder = str(
        getattr(cfgs, "cva_label_folder", "")
        or os.environ.get("CVA_LABEL_FOLDER", "")
        or os.environ.get("CDF_LABEL_FOLDER", "")
        or getattr(
            cfgs,
            "cdf_label_folder",
            "economic_grasp_label_300views_extend_angle_cdf_depth",
        )
    )
    return CVAExtendedLabelAdapter(
        base_dataset,
        dataset_root=cfgs.dataset_root,
        use_cdf=True,
        label_folder=label_folder,
        num_angle=cfgs.num_angle,
        num_depth=cfgs.num_depth,
    )


def _build_models(device: torch.device):
    common = dict(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        use_depth_comp=False,
        use_cdf=True,
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    )
    student = economicgrasp_dpt_student(
        **common,
        is_training=True,
        use_obs_depth=False,
        pose_depth_mode=str(getattr(cfgs, "pose_depth_mode", "none") or "none"),
        vis_dir=None,
    )
    teacher = economicgrasp_dpt_teacher(**common, is_training=False, vis_dir=None)

    student_path = os.path.abspath(P0.p0_student_checkpoint)
    teacher_path = os.path.abspath(str(base.DISTILL_ARGS.teacher_checkpoint))
    if not os.path.isfile(student_path):
        raise FileNotFoundError(student_path)
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(teacher_path)

    student_checkpoint = torch.load(student_path, map_location="cpu")
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu")
    if not isinstance(student_checkpoint, Mapping):
        raise TypeError("P0 student checkpoint must be a full checkpoint mapping.")
    if not isinstance(teacher_checkpoint, Mapping):
        raise TypeError("P0 teacher checkpoint must be a full checkpoint mapping.")
    _validate_student_checkpoint(student_checkpoint)
    _validate_teacher_checkpoint(teacher_checkpoint)
    load_checkpoint_state(student, student_path, strict=True, checkpoint_data=student_checkpoint)
    load_checkpoint_state(teacher, teacher_path, strict=True, checkpoint_data=teacher_checkpoint)
    student.to(device).eval().requires_grad_(False)
    teacher.to(device).eval().requires_grad_(False)
    return student, teacher, dict(student_checkpoint), dict(teacher_checkpoint)


class _MetricAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, float] = {}

    def update(self, stats) -> None:
        for key, (numerator, denominator) in stats.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(numerator.detach().item())
            self.counts[key] = self.counts.get(key, 0.0) + float(denominator.detach().item())

    def batch_means(self, stats) -> Dict[str, float]:
        out = {}
        for key, (numerator, denominator) in stats.items():
            den = float(denominator.detach().item())
            out[key] = float(numerator.detach().item()) / den if den > 0 else float("nan")
        return out

    def reduce(self, device: torch.device, distributed: bool) -> Dict[str, float]:
        local_keys = sorted(set(self.sums) | set(self.counts))
        if distributed and dist.is_initialized():
            gathered: List[List[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_keys)
            all_keys = sorted(set(k for keys in gathered if keys for k in keys))
        else:
            all_keys = local_keys
        buf = torch.zeros((len(all_keys), 2), device=device, dtype=torch.float64)
        for i, key in enumerate(all_keys):
            buf[i, 0] = float(self.sums.get(key, 0.0))
            buf[i, 1] = float(self.counts.get(key, 0.0))
        if distributed and dist.is_initialized() and buf.numel() > 0:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        out = {}
        for i, key in enumerate(all_keys):
            den = float(buf[i, 1].item())
            out[key] = float(buf[i, 0].item() / den) if den > 0 else float("nan")
        return out


class _HistAccumulator:
    def __init__(self) -> None:
        self.distance: Dict[str, torch.Tensor] = {}
        self.shift: Dict[str, torch.Tensor] = {}

    def add_distance(self, name: str, values_m: torch.Tensor) -> None:
        values_mm = values_m.detach().float().reshape(-1) * 1000.0
        inner = torch.tensor(DISTANCE_HIST_EDGES_MM[1:-1], device=values_mm.device, dtype=values_mm.dtype)
        idx = torch.bucketize(values_mm, inner, right=False)
        counts = torch.bincount(idx, minlength=len(DISTANCE_HIST_EDGES_MM) - 1).to(torch.float64)
        if name not in self.distance:
            self.distance[name] = torch.zeros_like(counts)
        self.distance[name] += counts

    def add_shift(self, name: str, shift_bins: torch.Tensor) -> None:
        shift = shift_bins.detach().long().reshape(-1)
        idx = torch.where(
            shift <= -5,
            torch.zeros_like(shift),
            torch.where(shift >= 5, torch.full_like(shift, 10), shift + 5),
        )
        counts = torch.bincount(idx, minlength=11).to(torch.float64)
        if name not in self.shift:
            self.shift[name] = torch.zeros_like(counts)
        self.shift[name] += counts

    def reduce(self, distributed: bool) -> Dict[str, Any]:
        for table in (self.distance, self.shift):
            for key in sorted(table):
                if distributed and dist.is_initialized():
                    dist.all_reduce(table[key], op=dist.ReduceOp.SUM)
        return {
            "distance_mm": {
                name: {"edges_mm": DISTANCE_HIST_EDGES_MM, "counts": counts.detach().cpu().tolist()}
                for name, counts in sorted(self.distance.items())
            },
            "shift_bins": {
                name: {"labels": SHIFT_HIST_LABELS, "counts": counts.detach().cpu().tolist()}
                for name, counts in sorted(self.shift.items())
            },
        }


def _open_csv(path: str, fieldnames: Iterable[str], gzip_output: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = gzip.open(path, "wt", newline="") if gzip_output else open(path, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
    writer.writeheader()
    return handle, writer


def _sample_query_indices(num_queries: int, n: int, device: torch.device) -> torch.Tensor:
    n = min(max(int(n), 0), int(num_queries))
    if n <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if n == num_queries:
        return torch.arange(num_queries, device=device)
    return torch.linspace(0, num_queries - 1, steps=n, device=device).round().long().unique()


def _query_rows(result, student_ep, *, rank: int, batch_idx: int, n_per_image: int):
    q = result["per_query"]
    seed_idx = student_ep["token_sel_idx"].detach().long()
    view_idx = student_ep["grasp_top_view_inds"].detach().long()
    B, Q = seed_idx.shape
    selected = _sample_query_indices(Q, n_per_image, seed_idx.device)
    scalar_keys = [
        "student_valid", "teacher_valid", "common_valid",
        "center_error_m", "center_parallel_m", "center_lateral_m",
        "center_shift_bins", "center_action_residual_m", "center_depth_retained_fraction",
        "label_point_dist_m", "label_parallel_m", "label_lateral_m",
        "label_shift_bins", "label_action_residual_m", "label_comp_continuous",
        "label_comp_nearest", "recovered_continuous", "recovered_nearest",
        "matched_gt_point_same", "teacher_cdf_bce_before", "teacher_cdf_bce_after",
        "teacher_cdf_bce_paired", "teacher_cdf_bce_improved",
    ]
    cpu = {key: q[key].detach().cpu() for key in scalar_keys}
    seed_cpu = seed_idx.detach().cpu()
    view_cpu = view_idx.detach().cpu()
    selected_cpu = selected.detach().cpu().tolist()
    for b in range(B):
        for qi in selected_cpu:
            row: Dict[str, Any] = {
                "rank": rank, "batch_idx": batch_idx, "batch_item": b,
                "query_idx": int(qi), "token_sel_idx": int(seed_cpu[b, qi].item()),
                "view_idx": int(view_cpu[b, qi].item()),
            }
            for key in scalar_keys:
                value = cpu[key][b, qi].item()
                if key.endswith("_m"):
                    row[key[:-2] + "_mm"] = float(value) * 1000.0
                elif isinstance(value, bool):
                    row[key] = int(value)
                else:
                    row[key] = value
            yield row


def _metric_display(metrics: Mapping[str, float]) -> Dict[str, float]:
    out = dict(metrics)
    for key, value in list(metrics.items()):
        if key.endswith("_mean_m"):
            out[key[:-2] + "_mm"] = value * 1000.0
    return out


def main() -> None:
    _validate_common_protocol()
    distributed, rank, local_rank, world_size, device = base.setup_distributed()
    main_process = rank == 0
    base.seed_everything(int(getattr(cfgs, "seed", 0)), rank)

    output_dir = P0.p0_output_dir or os.path.join(cfgs.log_dir, f"p0_pose_space_cdf_transport_{P0.p0_split}")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    query_handle = None
    batch_handle = None
    try:
        dataset = _build_dataset(P0.p0_split)
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            if distributed else None
        )
        loader = DataLoader(
            dataset,
            batch_size=cfgs.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=max(int(getattr(cfgs, "eval_num_workers", 1)), 0),
            worker_init_fn=base.my_worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

        student, teacher, student_ckpt, teacher_ckpt = _build_models(device)
        config = P0TransportConfig(
            point_match_thresh_m=float(P0.p0_point_match_thresh_mm) * 1.0e-3,
            tolerated_parallel_m=float(P0.p0_tolerated_parallel_mm) * 1.0e-3,
            depth_start_m=float(P0.p0_depth_start_mm) * 1.0e-3,
            depth_interval_m=float(P0.p0_depth_interval_mm) * 1.0e-3,
            num_depth=int(cfgs.num_depth),
        )

        protocol = {
            "protocol": "p0-pose-space-cdf-transport-diagnostic-v1",
            "git_head": _git_head(),
            "split": P0.p0_split,
            "student_checkpoint": os.path.abspath(P0.p0_student_checkpoint),
            "student_distill_stage": student_ckpt.get("distill_stage"),
            "teacher_checkpoint": os.path.abspath(str(base.DISTILL_ARGS.teacher_checkpoint)),
            "teacher_distill_stage": teacher_ckpt.get("distill_stage"),
            "student_geometry": "pred",
            "teacher_geometry": "gt",
            "seed_alignment": "teacher exact student image-FPS override",
            "view_alignment": "teacher exact student Top-1 view override",
            "student_view_policy": "eval Top-1",
            "transport_sign": "delta=dot(target-center, approach); d_student=d_source+delta",
            "depth_transport": "nearest class, floor(delta/interval + 0.5), no boundary clipping",
            "point_match_thresh_mm": P0.p0_point_match_thresh_mm,
            "tolerated_parallel_mm": P0.p0_tolerated_parallel_mm,
            "depth_start_mm": P0.p0_depth_start_mm,
            "depth_interval_mm": P0.p0_depth_interval_mm,
            "num_depth": int(cfgs.num_depth),
            "use_fuse_depth": bool(cfgs.use_fuse_depth),
            "pose_depth_mode": str(getattr(cfgs, "pose_depth_mode", "none") or "none"),
            "camera": str(cfgs.camera),
            "batch_size": int(cfgs.batch_size),
            "world_size": int(world_size),
            "max_batches_per_rank": int(P0.p0_max_batches),
            "dexnet_or_graspnet_evaluator_called": False,
        }
        if main_process:
            _write_json(os.path.join(output_dir, "protocol.json"), protocol)
            print(json.dumps(protocol, indent=2), flush=True)

        batch_writer = None
        query_writer = None
        metrics_acc = _MetricAccumulator()
        hist_acc = _HistAccumulator()
        local_batches = 0
        local_images = 0

        for batch_idx, batch in enumerate(loader):
            if int(P0.p0_max_batches) > 0 and batch_idx >= int(P0.p0_max_batches):
                break
            base.validate_batch_label_contract(batch, use_cdf=True)
            batch = base.drop_unused_point_inputs(batch)
            batch = base.move_batch_to_device(batch, device=device, use_cdf=True, non_blocking=False)
            base.assert_cpu_resident_label_lists(batch, use_cdf=True)

            student_input = dict(batch)
            student_input.pop("image_fps_seed_idx_override", None)
            student_input.pop("oracle_view_inds_override", None)
            student_input["cva_force_process_grasp_labels"] = True
            student_input["cva_compute_diagnostics"] = False
            student_input["geometry_compute_diagnostics"] = False
            student_input["cva_export_angle_feature"] = False

            teacher_input = dict(batch)
            teacher_input["cva_force_process_grasp_labels"] = True
            teacher_input["cva_compute_diagnostics"] = False
            teacher_input["geometry_compute_diagnostics"] = False
            teacher_input["cva_export_angle_feature"] = False

            with torch.no_grad():
                student_ep = student(student_input)
                if batch_idx == 0:
                    base.assert_geometry_depth_contract(
                        student_ep, expected_source="pred", context=f"P0 student split={P0.p0_split}"
                    )
                teacher_input["image_fps_seed_idx_override"] = student_ep["kview_base_token_sel_idx"].detach().long()
                teacher_input["oracle_view_inds_override"] = student_ep["grasp_top_view_inds"].detach().long()
                teacher_ep = teacher(teacher_input)
                if batch_idx == 0:
                    base.assert_geometry_depth_contract(
                        teacher_ep, expected_source="gt", context=f"P0 teacher split={P0.p0_split}"
                    )
                result = compute_p0_pose_space_transport_diagnostics(student_ep, teacher_ep, config)

            metrics_acc.update(result["stats"])
            batch_metrics = metrics_acc.batch_means(result["stats"])
            local_batches += 1
            local_images += int(student_ep["token_sel_idx"].shape[0])

            pq = result["per_query"]
            hist_acc.add_distance("center_error", pq["center_error_m"])
            hist_acc.add_distance("center_lateral", pq["center_lateral_m"])
            hist_acc.add_distance("center_action_residual_nearest", pq["center_action_residual_m"])
            hist_acc.add_distance("label_point_dist", pq["label_point_dist_m"])
            hist_acc.add_distance("label_lateral", pq["label_lateral_m"])
            hist_acc.add_distance("label_action_residual_nearest", pq["label_action_residual_m"])
            hist_acc.add_shift("center_shift", result["center_shift_bins_bq"])
            hist_acc.add_shift("label_shift", result["label_shift_bins_bq"])

            if batch_writer is None:
                metric_names = sorted(batch_metrics)
                batch_path = os.path.join(output_dir, f"batch_summary_rank{rank:02d}.csv")
                batch_handle, batch_writer = _open_csv(
                    batch_path, ["rank", "batch_idx", "batch_size"] + metric_names
                )
            row = {
                "rank": rank,
                "batch_idx": batch_idx,
                "batch_size": int(student_ep["token_sel_idx"].shape[0]),
            }
            row.update(batch_metrics)
            batch_writer.writerow(row)
            batch_handle.flush()

            if bool(P0.p0_save_query_rows) and int(P0.p0_query_sample_per_image) > 0:
                rows = list(_query_rows(
                    result, student_ep, rank=rank, batch_idx=batch_idx,
                    n_per_image=int(P0.p0_query_sample_per_image),
                ))
                if rows and query_writer is None:
                    query_path = os.path.join(output_dir, f"query_sample_rank{rank:02d}.csv.gz")
                    query_handle, query_writer = _open_csv(query_path, rows[0].keys(), gzip_output=True)
                if query_writer is not None:
                    query_writer.writerows(rows)
                    query_handle.flush()

            if main_process and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
                display = _metric_display(batch_metrics)
                print(
                    f"[P0][{P0.p0_split}] batch={batch_idx + 1}/{len(loader)} "
                    f"common={display.get('common_valid_query_ratio', float('nan')):.4f} "
                    f"IoU={display.get('support_iou_before', float('nan')):.4f}->"
                    f"{display.get('support_iou_after_nearest', float('nan')):.4f} "
                    f"recovered={display.get('p0_recovered_nearest_fraction_of_missing_teacher_support', float('nan')):.4f} "
                    f"center_lat_mm={display.get('center_lateral_mean_mm', float('nan')):.2f}",
                    flush=True,
                )

        metrics = metrics_acc.reduce(device, distributed)
        histograms = hist_acc.reduce(distributed)
        count_buf = torch.tensor([float(local_batches), float(local_images)], device=device, dtype=torch.float64)
        if distributed and dist.is_initialized():
            dist.all_reduce(count_buf, op=dist.ReduceOp.SUM)
        global_batches = int(round(float(count_buf[0].item())))
        global_images = int(round(float(count_buf[1].item())))

        if main_process:
            display_metrics = _metric_display(metrics)
            decision = {
                "support_iou_gain_continuous": metrics.get("support_iou_after_continuous", float("nan"))
                - metrics.get("support_iou_before", float("nan")),
                "support_iou_gain_nearest": metrics.get("support_iou_after_nearest", float("nan"))
                - metrics.get("support_iou_before", float("nan")),
                "teacher_support_recall_gain_nearest": metrics.get("teacher_support_recall_after_nearest", float("nan"))
                - metrics.get("teacher_support_recall_before", float("nan")),
                "teacher_cdf_bce_delta_after_minus_before": metrics.get("teacher_cdf_bce_after_nearest_transport", float("nan"))
                - metrics.get("teacher_cdf_bce_before_transport", float("nan")),
            }
            summary = {
                "protocol": protocol,
                "global_batches": global_batches,
                "global_images": global_images,
                "metrics": metrics,
                "display_metrics": display_metrics,
                "decision_deltas": decision,
                "histograms": histograms,
                "notes": [
                    "Recovered-query metrics are conservative: a student-invalid query counts as recoverable only when the clean-depth teacher supports the exact same selected view.",
                    "Continuous recovery removes only the approach-parallel center error; nearest recovery also includes 1-cm depth-bin quantization and finite 4-bin depth support.",
                    "CDF BCE before/after transport is evaluated only on destination candidates that already have a valid student GT target. Recovered queries are not assigned synthetic evaluator labels.",
                ],
            }
            _write_json(os.path.join(output_dir, "summary.json"), summary)
            print("[P0] summary:", json.dumps(_json_safe(summary), indent=2), flush=True)
    finally:
        if query_handle is not None:
            query_handle.close()
        if batch_handle is not None:
            batch_handle.close()
        base.cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
