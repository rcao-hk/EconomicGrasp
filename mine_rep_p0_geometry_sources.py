#!/usr/bin/env python3
"""Mine Rep-P0 fixed-action geometry-source cache with bounded host memory.

Memory-safety contract
----------------------
1. Frames are sharded by *scene*, not round-robin by frame.
2. Exact-evaluator scene_cache retains only the current scene.
3. The CPU checkpoint/state_dict is released immediately after model loading.
4. DataLoader workers default to zero; persistent workers are never used.
5. Scene transitions explicitly clear evaluator/CAD caches, run gc, trim glibc,
   and empty the CUDA allocator cache.
6. RSS/system-memory/evaluator-cache diagnostics are printed periodically.

Resume contract
---------------
With --resume, an existing cache is structurally/protocol validated and skipped.
Writes are atomic (temporary file + os.replace), so SIGKILL cannot turn a
partially written file into a valid ann_XXXX.npz. --repair_invalid_cache can
regenerate incompatible/corrupt files during resume.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument(
        "--split",
        default="train",
        choices=("train", "test_seen", "test_similar", "test_novel"),
    )
    p.add_argument("--camera", default="realsense")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--query_eval_num", type=int, default=64)
    p.add_argument(
        "--query_eval_mode",
        default="topk_uniform",
        choices=("all", "topk", "uniform", "topk_uniform"),
    )
    p.add_argument("--offsets_mm", default="-40,-20,-10,0,10,20,40")
    p.add_argument("--sources", default="pred,sensor,rendered,cad_full")
    p.add_argument("--voxel_size", type=float, default=0.005)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--repair_invalid_cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fc_mode",
        default="reuse_contacts",
        choices=("reuse_contacts", "official"),
    )
    p.add_argument("--verify_n", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=20)
    p.add_argument("--memory_report_every", type=int, default=10)
    return p.parse_args()


ARGS = parse_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from rep_p0_geometry_common import (
    DescriptorConfig,
    GEOMETRY_SOURCES,
    REP_P0_CACHE_CONTRACT_VERSION,
    REP_P0_EVIDENCE_VOXEL_SIZE,
    backproject_depth_map,
    build_evidence_table,
    build_translation_ray_actions,
    describe_actions,
    friction_utility,
    load_full_cad_scene_cloud,
    parse_offsets_mm,
    prepare_cad_models_for_evidence,
    scene_sharded_indices,
    select_query_indices,
    zero_offset_index,
)
from utils.arguments import cfgs


def _read_kib(path: str, keys):
    out = {}
    try:
        lines = Path(path).read_text().splitlines()
    except Exception:
        return out
    wanted = set(keys)
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key not in wanted:
            continue
        parts = value.strip().split()
        if not parts:
            continue
        try:
            out[key] = float(parts[0])
        except ValueError:
            pass
    return out


def memory_snapshot():
    proc = _read_kib(
        "/proc/self/status",
        ("VmRSS", "VmHWM", "VmSize"),
    )
    sysmem = _read_kib(
        "/proc/meminfo",
        ("MemAvailable", "MemFree", "SwapFree", "SwapTotal"),
    )
    snap = {
        "rss_gib": proc.get("VmRSS", 0.0) / 1024.0 / 1024.0,
        "hwm_gib": proc.get("VmHWM", 0.0) / 1024.0 / 1024.0,
        "vmsize_gib": proc.get("VmSize", 0.0) / 1024.0 / 1024.0,
        "mem_available_gib": sysmem.get("MemAvailable", 0.0) / 1024.0 / 1024.0,
        "swap_free_gib": sysmem.get("SwapFree", 0.0) / 1024.0 / 1024.0,
        "swap_total_gib": sysmem.get("SwapTotal", 0.0) / 1024.0 / 1024.0,
    }
    if torch.cuda.is_available():
        snap["cuda_alloc_gib"] = torch.cuda.memory_allocated() / (1024.0 ** 3)
        snap["cuda_reserved_gib"] = torch.cuda.memory_reserved() / (1024.0 ** 3)
    else:
        snap["cuda_alloc_gib"] = 0.0
        snap["cuda_reserved_gib"] = 0.0
    return snap


def report_memory(tag: str, evaluator=None):
    m = memory_snapshot()
    cache_size = len(evaluator.scene_cache) if evaluator is not None else 0
    print(
        f"[REP-P0-MEM] {tag} "
        f"rss={m['rss_gib']:.2f}GiB hwm={m['hwm_gib']:.2f}GiB "
        f"avail={m['mem_available_gib']:.2f}GiB "
        f"swap_free={m['swap_free_gib']:.2f}/{m['swap_total_gib']:.2f}GiB "
        f"cuda_alloc={m['cuda_alloc_gib']:.2f}GiB "
        f"cuda_reserved={m['cuda_reserved_gib']:.2f}GiB "
        f"eval_scene_cache={cache_size}",
        flush=True,
    )


def release_process_memory(*, cuda: bool = True):
    """Release unreachable Python/NumPy memory and return glibc arenas to OS."""
    gc.collect()
    if cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def move_batch(batch, device):
    for key in ("point_clouds", "cloud_colors", "coordinates_for_voxel"):
        batch.pop(key, None)
    for key, value in list(batch.items()):
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Unexpected list-valued key {key!r}; use load_label=False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def load_checkpoint_model(path: str, device):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(
            "Rep-P0 requires a full EconomicGrasp checkpoint with model_state_dict."
        )

    source = str(ckpt.get("geometry_depth_source", "pred"))
    if source not in ("", "pred"):
        raise RuntimeError(
            "Rep-P0 action generator must be RGB-predicted geometry; "
            f"checkpoint source={source!r}."
        )

    meta = {
        "pose_mode": str(ckpt.get("pose_depth_mode", "global_film")),
        "use_fuse_depth": bool(ckpt.get("use_fuse_depth", False)),
        "camera_pose_key": str(ckpt.get("camera_pose_key", "camera_pose_vec")),
        "camera_gravity_key": str(
            ckpt.get("camera_gravity_key", "camera_gravity_vec")
        ),
        "pose_hidden_dim": int(ckpt.get("pose_hidden_dim", 64)),
        "ray_gravity_hidden_dim": int(
            ckpt.get("ray_gravity_hidden_dim", 64)
        ),
        "ray_gravity_mid_dim": int(
            ckpt.get("ray_gravity_mid_dim", 32)
        ),
    }

    cfgs.use_top4_view_infer = False
    cfgs.kview_mode = "A1"
    cfgs.kview_k = 1
    cfgs.use_cdf = True
    cfgs.use_obs_depth = False
    cfgs.pose_depth_mode = meta["pose_mode"]

    model = economicgrasp_dpt_student(
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=meta["pose_mode"],
        camera_pose_key=meta["camera_pose_key"],
        camera_gravity_key=meta["camera_gravity_key"],
        pose_hidden_dim=meta["pose_hidden_dim"],
        ray_gravity_hidden_dim=meta["ray_gravity_hidden_dim"],
        ray_gravity_mid_dim=meta["ray_gravity_mid_dim"],
        use_cdf=True,
        vis_dir=None,
    ).to(device)

    state = ckpt["model_state_dict"]
    result = model.load_state_dict(state, strict=False)
    optional = ("rgb_geometry_diagnostics.",)
    missing = [k for k in result.missing_keys if not k.startswith(optional)]
    unexpected = [
        k for k in result.unexpected_keys if not k.startswith(optional)
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    # Critical for multi-GPU mining: do not retain a full CPU checkpoint/state
    # copy in every miner process.
    del state
    del ckpt
    release_process_memory(cuda=False)

    model.eval()
    return model, meta["pose_mode"], meta["use_fuse_depth"]


def evaluate_actions(evaluator, scene_id, anno_id, actions, valid):
    K, Q, _ = actions.shape
    friction = np.full((K, Q), np.nan, dtype=np.float32)
    assigned = np.full((K, Q), -1, dtype=np.int64)
    collision = np.full((K, Q), -1, dtype=np.int8)
    pure = np.full((K, Q), -1, dtype=np.int8)
    empty = np.full((K, Q), -1, dtype=np.int8)

    ids = np.flatnonzero(valid.reshape(-1))
    if len(ids):
        res = evaluator.evaluate(
            scene_id,
            anno_id,
            actions.reshape(-1, 17)[ids],
        )
        friction.reshape(-1)[ids] = res.friction
        assigned.reshape(-1)[ids] = res.assigned_obj
        collision.reshape(-1)[ids] = res.collision_or_empty.astype(np.int8)
        pure.reshape(-1)[ids] = res.pure_collision.astype(np.int8)
        empty.reshape(-1)[ids] = res.empty.astype(np.int8)
        return friction, assigned, collision, pure, empty, res.stats

    return friction, assigned, collision, pure, empty, {}


def _scalar_string(value) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError("Expected scalar string field.")
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def validate_resume_cache(
    path: Path,
    *,
    scene_id: int,
    anno_id: int,
    offsets: np.ndarray,
    sources,
    voxel_size: float,
    query_eval_num: int,
    query_eval_mode: str,
):
    """Return (valid, reason, contract_kind) without modifying the cache."""
    try:
        with np.load(path, allow_pickle=False) as d:
            required = {
                "actions",
                "valid",
                "friction",
                "utility",
                "offsets_mm",
                "zero_index",
                "scene_id",
                "anno_id",
                "query_ids",
                "native_score",
                "evidence_voxel_size",
            }
            missing = sorted(required - set(d.files))
            if missing:
                return False, f"missing keys {missing}", "invalid"

            cached_scene = int(np.asarray(d["scene_id"]).reshape(-1)[0])
            cached_anno = int(np.asarray(d["anno_id"]).reshape(-1)[0])
            if cached_scene != int(scene_id) or cached_anno != int(anno_id):
                return (
                    False,
                    f"scene/anno mismatch cache={cached_scene}/{cached_anno} "
                    f"expected={scene_id}/{anno_id}",
                    "invalid",
                )

            old_voxel = float(
                np.asarray(d["evidence_voxel_size"]).reshape(-1)[0]
            )
            if abs(old_voxel - float(voxel_size)) > 1e-9:
                return (
                    False,
                    f"voxel mismatch {old_voxel:.6f} vs {voxel_size:.6f}",
                    "invalid",
                )

            old_offsets = np.asarray(d["offsets_mm"], dtype=np.float32)
            if old_offsets.shape != offsets.shape or not np.allclose(
                old_offsets, offsets, atol=1e-7, rtol=0.0
            ):
                return False, "offsets mismatch", "invalid"

            actions = np.asarray(d["actions"])
            valid = np.asarray(d["valid"])
            friction = np.asarray(d["friction"])
            utility = np.asarray(d["utility"])
            if (
                actions.ndim != 3
                or actions.shape[-1] != 17
                or valid.shape != actions.shape[:2]
                or friction.shape != valid.shape
                or utility.shape != valid.shape
            ):
                return False, "malformed action/label shapes", "invalid"

            qids = np.asarray(d["query_ids"]).reshape(-1)
            if actions.shape[1] != len(qids):
                return False, "query dimension mismatch", "invalid"
            if query_eval_mode != "all" and query_eval_num > 0:
                if actions.shape[1] != int(query_eval_num):
                    return (
                        False,
                        f"query count {actions.shape[1]} != {query_eval_num}",
                        "invalid",
                    )

            for source in sources:
                key = f"feat_{source}"
                if key not in d.files:
                    return False, f"missing {key}", "invalid"
                feat = np.asarray(d[key])
                if feat.shape[:2] != valid.shape or feat.ndim != 3:
                    return False, f"malformed {key} shape {feat.shape}", "invalid"

            if "cache_contract_version" in d.files:
                version = int(
                    np.asarray(d["cache_contract_version"]).reshape(-1)[0]
                )
                if version != REP_P0_CACHE_CONTRACT_VERSION:
                    return (
                        False,
                        f"cache contract {version} != "
                        f"{REP_P0_CACHE_CONTRACT_VERSION}",
                        "invalid",
                    )
                if "query_eval_mode" not in d.files:
                    return False, "v2 cache missing query_eval_mode", "invalid"
                cached_mode = _scalar_string(d["query_eval_mode"])
                if cached_mode != query_eval_mode:
                    return (
                        False,
                        f"query mode {cached_mode} != {query_eval_mode}",
                        "invalid",
                    )
                return True, "validated", "v2"

            # Caches mined immediately before the memory-bounded update did not
            # have a contract-version field, but are safe to reuse if every
            # structural/source/5-mm check above passes.
            return True, "legacy 5-mm cache structurally validated", "legacy"

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", "corrupt"


def atomic_save_npz(path: Path, payload):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def source_points(
    source: str,
    *,
    ep,
    batch,
    Kcam,
    evaluator,
    scene_id,
    anno_id,
    voxel_size,
    min_depth,
    max_depth,
    prepared_cad_models,
    evidence_table,
):
    if source == "pred":
        pred_depth = ep["depth_map_used_for_geometry"][0]
        if pred_depth.dim() == 3:
            pred_depth = pred_depth[0]
        return backproject_depth_map(
            pred_depth.detach().cpu().numpy(),
            Kcam,
            voxel_size=voxel_size,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    if source == "sensor":
        return backproject_depth_map(
            batch["sensor_depth_m"][0].detach().cpu().numpy(),
            Kcam,
            voxel_size=voxel_size,
            min_depth=0.05,
            max_depth=2.0,
        )

    if source == "rendered":
        return backproject_depth_map(
            batch["gt_depth_m"][0].detach().cpu().numpy(),
            Kcam,
            voxel_size=voxel_size,
            min_depth=0.05,
            max_depth=2.0,
        )

    if source == "cad_full":
        return load_full_cad_scene_cloud(
            evaluator,
            scene_id,
            anno_id,
            voxel_size,
            prepared_models=prepared_cad_models,
            table_points=evidence_table,
        )

    raise KeyError(source)


def main():
    if ARGS.resume and ARGS.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if ARGS.repair_invalid_cache and not ARGS.resume:
        raise ValueError("--repair_invalid_cache requires --resume.")

    sources = tuple(x.strip() for x in ARGS.sources.split(",") if x.strip())
    unknown = sorted(set(sources) - set(GEOMETRY_SOURCES))
    if unknown:
        raise ValueError(
            f"Unknown geometry source(s): {unknown}; supported={GEOMETRY_SOURCES}"
        )
    if not sources:
        raise ValueError("At least one source is required.")

    if abs(
        float(ARGS.voxel_size) - float(REP_P0_EVIDENCE_VOXEL_SIZE)
    ) > 1e-9:
        print(
            f"[REP-P0][WARN] non-canonical evidence voxel size "
            f"{ARGS.voxel_size:.6f} m; formal Rep-P0 uses "
            f"{REP_P0_EVIDENCE_VOXEL_SIZE:.6f} m.",
            flush=True,
        )

    offsets = parse_offsets_mm(ARGS.offsets_mm)
    zidx = zero_offset_index(offsets)
    if ARGS.num_shards < 1 or not (0 <= ARGS.shard_id < ARGS.num_shards):
        raise ValueError("Invalid shard_id/num_shards.")
    if ARGS.num_workers > 0:
        print(
            f"[REP-P0][WARN] num_workers={ARGS.num_workers}; memory-bounded "
            "formal mining uses NUM_WORKERS=0.",
            flush=True,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report_memory("before_checkpoint")
    model, pose_mode, use_fuse_depth = load_checkpoint_model(
        ARGS.checkpoint_path, device
    )
    report_memory("after_checkpoint_cpu_release")

    dataset = GraspNetMultiDataset(
        ARGS.dataset_root,
        split=ARGS.split,
        camera=ARGS.camera,
        num_points=20000,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=use_fuse_depth,
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
    )

    indices = scene_sharded_indices(
        len(dataset),
        ARGS.sample_interval,
        ARGS.shard_id,
        ARGS.num_shards,
        frames_per_scene=256,
        max_samples=ARGS.max_samples,
    )
    if not indices:
        print(
            f"[REP-P0-MINE] shard {ARGS.shard_id}/{ARGS.num_shards} "
            f"owns no selected frames for split={ARGS.split}.",
            flush=True,
        )

    owned_scene_positions = sorted({int(i) // 256 for i in indices})
    print(
        f"[REP-P0-MINE] split={ARGS.split} "
        f"shard={ARGS.shard_id}/{ARGS.num_shards} "
        f"scene_positions={owned_scene_positions} "
        f"selected_frames={len(indices)} "
        f"num_workers={ARGS.num_workers} resume={int(ARGS.resume)}",
        flush=True,
    )

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=ARGS.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )

    evaluator = ExactGraspNetActionEvaluator(
        ARGS.dataset_root,
        ARGS.camera,
        split=ARGS.split,
        fc_mode=ARGS.fc_mode,
        verify_n=ARGS.verify_n,
        strict=True,
    )
    desc_cfg = DescriptorConfig(voxel_size=ARGS.voxel_size)
    evidence_table = (
        build_evidence_table(ARGS.voxel_size)
        if "cad_full" in sources
        else None
    )

    out_root = Path(ARGS.output_root) / ARGS.split
    out_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment": "Rep-P0 fixed-action geometry-source diagnosis",
        "cache_contract_version": REP_P0_CACHE_CONTRACT_VERSION,
        "split": ARGS.split,
        "camera": ARGS.camera,
        "checkpoint": os.path.abspath(ARGS.checkpoint_path),
        "sample_interval": ARGS.sample_interval,
        "query_eval_num": ARGS.query_eval_num,
        "query_eval_mode": ARGS.query_eval_mode,
        "offsets_mm": offsets.tolist(),
        "zero_index": zidx,
        "action_contract": (
            "native R/width/height/insertion-depth fixed; "
            "translation shifts on same camera ray"
        ),
        "sources": list(sources),
        "voxel_size": ARGS.voxel_size,
        "descriptor_feature_dim": desc_cfg.feature_dim,
        "descriptor_hist_bins": list(desc_cfg.hist_bins),
        "pose_depth_mode": pose_mode,
        "use_fuse_depth_for_rendered_target": use_fuse_depth,
        "sharding": "scene_level_modulo",
        "shard_id": ARGS.shard_id,
        "num_shards": ARGS.num_shards,
        "num_workers": ARGS.num_workers,
        "resume": bool(ARGS.resume),
    }
    with (out_root / f"protocol_shard_{ARGS.shard_id:02d}.json").open("w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)

    processed = 0
    resumed = 0
    repaired = 0
    total_actions = 0
    total_eval_sec = 0.0
    start_all = time.perf_counter()

    current_scene = None
    prepared_cad_models = None

    try:
        for local_i, batch in enumerate(loader):
            batch = move_batch(batch, device)
            scene_id = int(batch["scene_idx"].reshape(-1)[0].item())
            anno_id = int(batch["anno_idx"].reshape(-1)[0].item())

            if current_scene != scene_id:
                if current_scene is not None:
                    evaluator.scene_cache.clear()
                    prepared_cad_models = None
                    release_process_memory()
                    report_memory(
                        f"after_release_scene_{current_scene:04d}",
                        evaluator,
                    )

                current_scene = scene_id
                evaluator.scene_cache.clear()

                # CAD evidence is loaded lazily only when the first
                # non-resumed frame of this scene is actually processed.
                prepared_cad_models = None

                report_memory(
                    f"enter_scene_{scene_id:04d}",
                    evaluator,
                )

            out_dir = out_root / f"scene_{scene_id:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"ann_{anno_id:04d}.npz"

            # Remove only stale temp files for this exact frame. Scene-level
            # ownership guarantees no other shard writes this frame.
            for stale in out_dir.glob(out_path.name + ".tmp.*"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass

            if out_path.exists():
                if ARGS.overwrite:
                    pass
                elif ARGS.resume:
                    valid_cache, reason, kind = validate_resume_cache(
                        out_path,
                        scene_id=scene_id,
                        anno_id=anno_id,
                        offsets=offsets,
                        sources=sources,
                        voxel_size=ARGS.voxel_size,
                        query_eval_num=ARGS.query_eval_num,
                        query_eval_mode=ARGS.query_eval_mode,
                    )
                    if valid_cache:
                        resumed += 1
                        if (
                            ARGS.progress_every > 0
                            and (processed + resumed) % ARGS.progress_every == 0
                        ):
                            print(
                                f"[REP-P0-RESUME] split={ARGS.split} "
                                f"shard={ARGS.shard_id}/{ARGS.num_shards} "
                                f"done={processed+resumed}/{len(indices)} "
                                f"resumed={resumed} last={scene_id:04d}/{anno_id:04d} "
                                f"contract={kind}",
                                flush=True,
                            )
                        continue

                    if ARGS.repair_invalid_cache:
                        print(
                            f"[REP-P0-RESUME][REPAIR] {out_path}: {reason}",
                            flush=True,
                        )
                        out_path.unlink(missing_ok=True)
                        repaired += 1
                    else:
                        raise RuntimeError(
                            f"Existing Rep-P0 cache failed resume validation: "
                            f"{out_path}: {reason}. Use "
                            "--repair_invalid_cache to regenerate only invalid "
                            "files, or --overwrite for a full rewrite."
                        )
                else:
                    raise FileExistsError(
                        f"Cache already exists: {out_path}. "
                        "Use --resume or --overwrite."
                    )

            batch["cva_export_angle_feature"] = False
            batch["cva_compute_diagnostics"] = False
            batch["geometry_compute_diagnostics"] = False

            with torch.inference_mode():
                ep = model(batch)
                native_all = pred_decode_center_view_angle(
                    ep, use_cdf=True
                )[0]

            qidx = select_query_indices(
                native_all,
                ARGS.query_eval_num,
                ARGS.query_eval_mode,
            )
            native = (
                native_all.index_select(0, qidx)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            actions, valid = build_translation_ray_actions(
                native,
                offsets,
                min_depth=ARGS.min_depth,
                max_depth=ARGS.max_depth,
            )
            if not np.all(valid[zidx]):
                raise RuntimeError(
                    "Native candidate became invalid in Rep-P0 action generation."
                )

            t_eval = time.perf_counter()
            (
                friction,
                assigned,
                collision,
                pure,
                empty,
                eval_stats,
            ) = evaluate_actions(
                evaluator,
                scene_id,
                anno_id,
                actions,
                valid,
            )
            total_eval_sec += time.perf_counter() - t_eval
            utility = friction_utility(friction)

            payload = {
                "cache_contract_version": np.asarray(
                    REP_P0_CACHE_CONTRACT_VERSION, dtype=np.int16
                ),
                "actions": actions.astype(np.float32),
                "valid": valid.astype(np.uint8),
                "friction": friction.astype(np.float32),
                "utility": utility.astype(np.float32),
                "assigned_obj": assigned.astype(np.int16),
                "collision_or_empty": collision.astype(np.int8),
                "pure_collision": pure.astype(np.int8),
                "empty": empty.astype(np.int8),
                "offsets_mm": offsets.astype(np.float32),
                "zero_index": np.asarray(zidx, dtype=np.int16),
                "scene_id": np.asarray(scene_id, dtype=np.int16),
                "anno_id": np.asarray(anno_id, dtype=np.int16),
                "query_ids": (
                    qidx.detach().cpu().numpy().astype(np.int16)
                ),
                "native_score": native[:, 0].astype(np.float32),
                "evidence_voxel_size": np.asarray(
                    ARGS.voxel_size, dtype=np.float32
                ),
                "query_eval_num": np.asarray(
                    ARGS.query_eval_num, dtype=np.int16
                ),
                "query_eval_mode": np.asarray(ARGS.query_eval_mode),
                "sources_csv": np.asarray(",".join(sources)),
            }

            Kcam = (
                batch["K"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # Lazy one-scene CAD evidence cache. Fully resumed scenes never
            # load raw CAD evidence at all.
            if "cad_full" in sources and prepared_cad_models is None:
                prepared_cad_models = prepare_cad_models_for_evidence(
                    evaluator,
                    scene_id,
                    ARGS.voxel_size,
                )
                release_process_memory(cuda=False)
                report_memory(
                    f"cad_ready_scene_{scene_id:04d}",
                    evaluator,
                )

            # Peak-memory control: materialize exactly one geometry source,
            # extract its descriptor, then release that point cloud before the
            # next source.
            for source in sources:
                pts = source_points(
                    source,
                    ep=ep,
                    batch=batch,
                    Kcam=Kcam,
                    evaluator=evaluator,
                    scene_id=scene_id,
                    anno_id=anno_id,
                    voxel_size=ARGS.voxel_size,
                    min_depth=ARGS.min_depth,
                    max_depth=ARGS.max_depth,
                    prepared_cad_models=prepared_cad_models,
                    evidence_table=evidence_table,
                )
                feat = describe_actions(
                    pts,
                    actions,
                    valid,
                    desc_cfg,
                )
                payload[f"feat_{source}"] = feat.astype(np.float16)
                payload[f"points_{source}"] = np.asarray(
                    len(pts), dtype=np.int32
                )
                del feat, pts

            atomic_save_npz(out_path, payload)

            processed += 1
            total_actions += int(valid.sum())

            del payload, utility, friction, assigned, collision, pure, empty
            del actions, valid, native, native_all, qidx, ep, Kcam, batch

            if (
                ARGS.memory_report_every > 0
                and processed % ARGS.memory_report_every == 0
            ):
                release_process_memory()
                report_memory(
                    f"progress_{processed}_scene_{scene_id:04d}",
                    evaluator,
                )

            if ARGS.progress_every > 0 and processed % ARGS.progress_every == 0:
                elapsed = time.perf_counter() - start_all
                print(
                    f"[REP-P0-MINE] split={ARGS.split} "
                    f"shard={ARGS.shard_id}/{ARGS.num_shards} "
                    f"processed={processed} resumed={resumed} "
                    f"repaired={repaired} total={len(indices)} "
                    f"scene={scene_id:04d} ann={anno_id:04d} "
                    f"valid_actions={total_actions} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    finally:
        evaluator.scene_cache.clear()
        prepared_cad_models = None
        evidence_table = None
        release_process_memory()
        report_memory("final_release", evaluator)

    summary = {
        **protocol,
        "selected_frames_for_shard": len(indices),
        "processed_frames": processed,
        "resumed_frames": resumed,
        "repaired_frames": repaired,
        "completed_frames": processed + resumed,
        "valid_actions_newly_processed": total_actions,
        "exact_eval_sec_newly_processed": total_eval_sec,
        "wall_sec": time.perf_counter() - start_all,
        "final_memory": memory_snapshot(),
    }
    with (
        out_root / f"summary_shard_{ARGS.shard_id:02d}.json"
    ).open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
