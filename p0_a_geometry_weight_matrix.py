#!/usr/bin/env python3
"""P0-A: Geometry x Weight causal matrix for current PKD checkpoints.

Cells:
    S_P: Stage-1 student weights + predicted metric geometry
    S_G: Stage-1 student weights + clean/GT geometry
    T_P: Stage-0 teacher weights + the same Stage-1 predicted geometry
    T_G: Stage-0 teacher weights + clean/GT geometry

The runner executes all four cells on every selected frame. This shares the
same data sample and the same deterministic random seed across cells, avoiding
four independently sampled inference runs. Final grasp dumps use the current
CVA-CDF decoder and the same post-processing. A pre-postprocessing sidecar is
also saved for P0-B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--student_checkpoint", required=True)
    p.add_argument("--teacher_checkpoint", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--split", choices=("train", "test_seen", "test_similar", "test_novel"), required=True)
    p.add_argument("--camera", default="realsense")
    p.add_argument("--scene_ids", default="")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--num_point", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--graspness_mode", default="scene")
    p.add_argument("--collision_thresh", type=float, default=0.01)
    p.add_argument("--collision_voxel_size", type=float, default=0.01)
    p.add_argument("--save_pre_postprocess", type=int, choices=(0, 1), default=1)
    p.add_argument("--overwrite", type=int, choices=(0, 1), default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tf32", type=int, choices=(0, 1), default=0)
    return p


# Parse before importing EconomicGrasp because utils.arguments parses argv.
ARGS = build_parser().parse_args()
sys.argv[:] = [sys.argv[0]]

import numpy as np
import torch
from torch.utils.data import DataLoader

from pkd_p0.common import (
    ContractError,
    annotation_ids,
    atomic_json_dump,
    atomic_npz_dump,
    load_current_checkpoint,
    parse_csv_ints,
    scene_ids_for_split,
    seed_everything,
)
from pkd_p0.repo_adapter import (
    DEPTH_USED_ALIASES,
    RepoImports,
    DeterministicSubset,
    build_current_model,
    build_dataset,
    dataset_index_records,
    decode_current,
    extract_core_outputs,
    find_point_cloud,
    forward_model,
    postprocess_grasps,
)


CELL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("S_P", "student_predicted_geometry"),
    ("S_G", "student_gt_geometry"),
    ("T_P", "teacher_predicted_geometry"),
    ("T_G", "teacher_gt_geometry"),
)


def _output_path(root: Path, cell: str, split: str, scene_id: int, anno_id: int, camera: str) -> Path:
    return root / cell / split / f"scene_{scene_id:04d}" / camera / f"{anno_id:04d}.npy"


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".p0_candidates.npz")


def _selected_records(dataset: Any) -> List[Tuple[int, int, int]]:
    requested_scenes = set(scene_ids_for_split(ARGS.split, ARGS.scene_ids))
    requested_annos = set(annotation_ids(float(ARGS.sample_interval)))
    records = [
        record for record in dataset_index_records(dataset)
        if record[1] in requested_scenes and record[2] in requested_annos
    ]
    if int(ARGS.max_samples) > 0:
        records = records[: int(ARGS.max_samples)]
    return records


def _batch_first(value: torch.Tensor) -> torch.Tensor:
    return value[:1] if value.ndim > 0 else value


def _depth_from_output(output: Mapping[str, Any]) -> torch.Tensor:
    core = extract_core_outputs(output)
    return core["geometry_depth"].detach()


def _save_cell(
    repo: RepoImports,
    output: Mapping[str, Any],
    *,
    cell: str,
    split: str,
    scene_id: int,
    anno_id: int,
    point_cloud: np.ndarray,
    output_root: Path,
    checkpoint_sha256: str,
    geometry_source: str,
) -> Dict[str, Any]:
    decoded = decode_current(repo, output)
    if len(decoded) != 1:
        raise RuntimeError(f"P0-A uses batch_size=1; decoder returned {len(decoded)} samples")
    raw = decoded[0]
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().numpy()
    raw = np.asarray(raw, dtype=np.float32)
    path = _output_path(output_root, cell, split, scene_id, anno_id, ARGS.camera)
    path.parent.mkdir(parents=True, exist_ok=True)

    final_rows, counts = postprocess_grasps(
        repo,
        raw,
        point_cloud=point_cloud,
        collision_thresh=float(ARGS.collision_thresh),
        collision_voxel_size=float(ARGS.collision_voxel_size),
        apply_nms=True,
    )
    np.save(path, final_rows)

    core = extract_core_outputs(output)
    if bool(ARGS.save_pre_postprocess):
        arrays: Dict[str, np.ndarray] = {
            "raw_grasps": raw.astype(np.float32),
            "final_grasps": final_rows.astype(np.float32),
            "scene_id": np.asarray([scene_id], dtype=np.int16),
            "anno_id": np.asarray([anno_id], dtype=np.int16),
            "cell": np.asarray(cell),
            "geometry_source": np.asarray(geometry_source),
            "checkpoint_sha256": np.asarray(checkpoint_sha256),
        }
        for name, tensor in core.items():
            value = tensor.detach().cpu().numpy()
            arrays[name] = value.astype(np.float32) if np.issubdtype(value.dtype, np.floating) else value
        atomic_npz_dump(_sidecar_path(path), compress=False, **arrays)

    return {
        "cell": cell,
        "scene_id": scene_id,
        "anno_id": anno_id,
        "path": str(path),
        **counts,
    }


def main() -> None:
    seed_everything(int(ARGS.seed))
    if bool(ARGS.tf32) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = torch.device(ARGS.device)
    repo = RepoImports()
    _, student_contract_check = load_current_checkpoint(ARGS.student_checkpoint, expected_stage=1)
    _, teacher_contract_check = load_current_checkpoint(ARGS.teacher_checkpoint, expected_stage=0)
    if student_contract_check.seed_selection_mode != teacher_contract_check.seed_selection_mode:
        raise ContractError("Teacher/student seed-selection modes differ")

    student, _, student_contract = build_current_model(
        repo,
        checkpoint_path=ARGS.student_checkpoint,
        device=device,
        min_depth=float(ARGS.min_depth),
        max_depth=float(ARGS.max_depth),
        bin_num=int(ARGS.bin_num),
        is_training=False,
    )
    teacher, _, teacher_contract = build_current_model(
        repo,
        checkpoint_path=ARGS.teacher_checkpoint,
        device=device,
        min_depth=float(ARGS.min_depth),
        max_depth=float(ARGS.max_depth),
        bin_num=int(ARGS.bin_num),
        is_training=False,
    )

    dataset = build_dataset(
        repo,
        dataset_root=ARGS.dataset_root,
        split=ARGS.split,
        camera=ARGS.camera,
        num_point=int(ARGS.num_point),
        min_depth=float(ARGS.min_depth),
        max_depth=float(ARGS.max_depth),
        bin_num=int(ARGS.bin_num),
        use_fuse_depth=student_contract.use_fuse_depth,
        graspness_mode=ARGS.graspness_mode,
        load_label=False,
        use_gt_depth=False,
    )
    records = _selected_records(dataset)
    if not records:
        raise RuntimeError("No frames selected")
    indices = [record[0] for record in records]
    subset = DeterministicSubset(dataset, indices, int(ARGS.seed))
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(ARGS.num_workers)),
        collate_fn=repo.collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=int(ARGS.num_workers) > 0,
    )

    output_root = Path(ARGS.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract_payload = {
        "experiment": "P0-A Geometry x Weight matrix",
        "cells": {cell: description for cell, description in CELL_SPECS},
        "student": student_contract.to_dict(),
        "teacher": teacher_contract.to_dict(),
        "dataset_root": str(Path(ARGS.dataset_root).expanduser().resolve()),
        "split": ARGS.split,
        "scene_ids": sorted({scene for _, scene, _ in records}),
        "annotation_ids": sorted({anno for _, _, anno in records}),
        "sample_interval": float(ARGS.sample_interval),
        "collision_thresh": float(ARGS.collision_thresh),
        "collision_voxel_size": float(ARGS.collision_voxel_size),
        "tf32": bool(ARGS.tf32),
        "seed": int(ARGS.seed),
    }
    atomic_json_dump(contract_payload, output_root / "p0_a_contract.json")

    completed = 0
    started = time.time()
    frame_records: List[Dict[str, Any]] = []
    for local_index, batch in enumerate(loader):
        dataset_index, scene_id, anno_id = records[local_index]
        expected_paths = [
            _output_path(output_root, cell, ARGS.split, scene_id, anno_id, ARGS.camera)
            for cell, _ in CELL_SPECS
        ]
        if not bool(ARGS.overwrite) and all(path.is_file() for path in expected_paths):
            print(f"[SKIP] scene={scene_id:04d} ann={anno_id:04d}", flush=True)
            continue

        point_cloud = find_point_cloud(batch, 0)
        sample_seed = int(ARGS.seed) + int(dataset_index) * 1_000_003

        # Run the two native cells first and extract the exact geometry tensors
        # actually used by each checkpoint. This avoids guessing a dataset key.
        with torch.inference_mode():
            t_g = forward_model(
                repo, teacher, teacher_contract, batch,
                device=device, seed=sample_seed,
            )
            gt_geometry = _depth_from_output(t_g)
            s_p = forward_model(
                repo, student, student_contract, batch,
                device=device, seed=sample_seed,
            )
            predicted_geometry = _depth_from_output(s_p)
            if tuple(gt_geometry.shape) != tuple(predicted_geometry.shape):
                raise ContractError(
                    f"GT/predicted geometry depth shapes differ: {tuple(gt_geometry.shape)} vs {tuple(predicted_geometry.shape)}"
                )
            s_g = forward_model(
                repo, student, student_contract, batch,
                device=device, seed=sample_seed,
                geometry_override=gt_geometry,
                require_override_marker=True,
            )
            t_p = forward_model(
                repo, teacher, teacher_contract, batch,
                device=device, seed=sample_seed,
                geometry_override=predicted_geometry,
                require_override_marker=True,
            )

        cell_outputs = {
            "S_P": (s_p, student_contract, "predicted"),
            "S_G": (s_g, student_contract, "gt"),
            "T_P": (t_p, teacher_contract, "predicted_from_stage1"),
            "T_G": (t_g, teacher_contract, "gt"),
        }
        for cell, (output, contract, geometry_source) in cell_outputs.items():
            path = _output_path(output_root, cell, ARGS.split, scene_id, anno_id, ARGS.camera)
            if path.is_file() and not bool(ARGS.overwrite):
                continue
            record = _save_cell(
                repo,
                output,
                cell=cell,
                split=ARGS.split,
                scene_id=scene_id,
                anno_id=anno_id,
                point_cloud=point_cloud,
                output_root=output_root,
                checkpoint_sha256=contract.sha256,
                geometry_source=geometry_source,
            )
            frame_records.append(record)

        completed += 1
        print(
            f"[P0-A] {completed}/{len(records)} scene={scene_id:04d} ann={anno_id:04d} "
            f"elapsed={(time.time() - started) / 60.0:.1f}m",
            flush=True,
        )

    atomic_json_dump(
        {
            **contract_payload,
            "status": "complete",
            "completed_new_frames": completed,
            "elapsed_seconds": time.time() - started,
            "frame_records": frame_records,
        },
        output_root / f"p0_a_complete_{ARGS.split}.json",
    )
    print(f"[DONE] P0-A dumps under {output_root}", flush=True)


if __name__ == "__main__":
    main()
