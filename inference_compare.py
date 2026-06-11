#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer two EconomicGrasp-style methods on a sampled GraspNet split and save:
  1) raw decoded grasps before model-free collision filtering;
  2) final grasps after model-free collision filtering, in GraspNetEval dump format;
  3) model-free collision masks and metadata;
  4) per-sample end_points caches for later diagnosis.

The script is intentionally close to the usual EconomicGrasp inference script, but
runs two methods sequentially to avoid keeping both networks on GPU at once.

Example:
python infer_two_methods_save_endpoints.py \
  --dataset-root /data/robotarm/dataset/graspnet \
  --test-mode test_similar \
  --camera realsense \
  --output-root ./infer_cache/baseline_vs_dpt \
  --batch-size 4 \
  --sample-interval 0.1 \
  --method1-name baseline \
  --method1-type baseline \
  --method1-checkpoint /path/to/economicgrasp.pth \
  --method2-name dpt_spatial \
  --method2-type dpt \
  --method2-checkpoint /path/to/dpt.pth \
  --method2-use-obs-depth \
  --collision-thresh 0.01

Final evaluable dump roots:
  <output-root>/<method-name>/final_grasps/

Raw decoded grasp roots:
  <output-root>/<method-name>/raw_grasps/

Endpoint roots:
  <output-root>/<method-name>/end_points/
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gc
import gzip
import importlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from graspnetAPI import GraspGroup  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import graspnetAPI.GraspGroup. Run inside the EconomicGrasp/GraspNet environment. "
        f"Original error: {repr(e)}"
    )

from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn  # type: ignore
from utils.collision_detector import ModelFreeCollisionDetectorTorch  # type: ignore


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if str(v).lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Dataset / dataloader.
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--test-mode", type=str, default="test_similar", help="e.g. test_seen/test_similar/test_novel/test")
    p.add_argument("--camera", type=str, default="realsense", choices=["realsense", "kinect"])
    p.add_argument("--num-point", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sample-interval", type=float, default=0.1)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--annos-per-scene", type=int, default=256)
    p.add_argument(
        "--dataset-kind",
        type=str,
        default="auto",
        choices=["auto", "single", "multi"],
        help="auto uses GraspNetMultiDataset if either method type needs RGB/DPT inputs.",
    )
    p.add_argument("--remove-outlier", type=str2bool, default=True)

    # Output.
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-raw-grasps", type=str2bool, default=True)
    p.add_argument("--save-final-grasps", type=str2bool, default=True)
    p.add_argument("--save-collision", type=str2bool, default=True)
    p.add_argument("--save-endpoints", type=str2bool, default=True)
    p.add_argument(
        "--endpoint-save-mode",
        type=str,
        default="compact",
        choices=["full", "compact", "summary"],
        help=(
            "full: save all convertible top-level end_points unless skipped; "
            "compact: save only diagnostic keys/patterns and skip large feature tensors; "
            "summary: save only per-key statistics, no tensor payloads."
        ),
    )
    p.add_argument(
        "--endpoint-keys",
        type=str,
        default="",
        help=(
            "Comma-separated exact end_points keys to save. When set, it overrides compact patterns. "
            "Use this for a strict whitelist."
        ),
    )
    p.add_argument(
        "--endpoint-patterns",
        type=str,
        default=(
            "*score*,*objectness*,*graspness*,*view*,*angle*,*depth*,*width*,"
            "seed_xyz,seed_inds,xyz_all*,uv_all*,*top*,*select*,*sel*idx*,*token*idx*"
        ),
        help="Comma-separated fnmatch patterns used in compact mode.",
    )
    p.add_argument(
        "--endpoint-skip-keys",
        type=str,
        default="",
        help="Comma-separated exact end_points keys to skip.",
    )
    p.add_argument(
        "--endpoint-skip-patterns",
        type=str,
        default=(
            "*feat*,*feature*,seed_features*,fp*_features*,*backbone*,*graph*,*sparse*,"
            "*rgb*,*image*,*color*,cloud*,point_clouds,coors,coordinates"
        ),
        help="Comma-separated fnmatch patterns to skip before saving tensors.",
    )
    p.add_argument(
        "--endpoint-max-tensor-numel",
        type=int,
        default=800000,
        help=(
            "Maximum numel of a per-sample tensor/array saved in compact mode. "
            "Larger values are summarized but not stored. 800k is enough for 448x448 maps and 1024x300 view scores."
        ),
    )
    p.add_argument(
        "--endpoint-float-dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Floating tensor dtype used in endpoint cache. fp16 usually halves file size without hurting diagnosis.",
    )
    p.add_argument(
        "--endpoint-compress",
        type=str,
        default="gzip",
        choices=["none", "gzip"],
        help="Compress endpoint payloads. gzip is slower but can substantially reduce dense maps.",
    )
    p.add_argument(
        "--endpoint-include-grasps",
        action="store_true",
        help="Also duplicate raw/final grasps and model-free collision mask inside each endpoint file. Usually unnecessary because they are saved separately.",
    )
    p.add_argument(
        "--endpoint-save-filter",
        type=str,
        default="",
        help=(
            "Optional comma-separated scene:ann pairs for endpoint saving only, e.g. 139:200,143:150. "
            "Grasp npy and collision npz are still saved for all evaluated samples."
        ),
    )
    p.add_argument(
        "--save-batch-endpoints",
        action="store_true",
        help="Also save full batch end_points. Usually not needed and can be very large.",
    )

    # Model-free collision filtering; this is the usual pre-eval collision filtering,
    # not the GraspNetEval object/table collision test.
    p.add_argument("--collision-thresh", type=float, default=0.0)
    p.add_argument("--collision-voxel-size", type=float, default=0.01)
    p.add_argument("--approach-dist", type=float, default=0.05)

    # Common model args.
    p.add_argument("--min-depth", type=float, default=0.2)
    p.add_argument("--max-depth", type=float, default=1.0)
    p.add_argument("--bin-num", type=int, default=256)
    p.add_argument("--seed-feat-dim", type=int, default=512)
    p.add_argument("--vis-dir", type=str, default=None)
    p.add_argument("--vis-every", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--strict-load", type=str2bool, default=True)

    # Method 1.
    p.add_argument("--method1-name", type=str, default="baseline")
    p.add_argument("--method1-type", type=str, default="baseline", choices=["baseline", "dpt", "economicgrasp_obs_depth","dpt_direct", "bip3d", "custom"])
    p.add_argument("--method1-checkpoint", type=str, required=True)
    p.add_argument("--method1-use-obs-depth", action="store_true")
    p.add_argument("--method1-model-import", type=str, default="", help="For custom: module:callable")
    p.add_argument("--method1-pred-decode-import", type=str, default="", help="For custom: module:callable")

    # Method 2.
    p.add_argument("--method2-name", type=str, default="dpt")
    p.add_argument("--method2-type", type=str, default="dpt", choices=["baseline", "dpt", "economicgrasp_obs_depth", "dpt_direct", "bip3d", "custom"])
    p.add_argument("--method2-checkpoint", type=str, required=True)
    p.add_argument("--method2-use-obs-depth", action="store_true")
    p.add_argument("--method2-model-import", type=str, default="", help="For custom: module:callable")
    p.add_argument("--method2-pred-decode-import", type=str, default="", help="For custom: module:callable")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def my_worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(np.random.get_state()[1][0] + worker_id)


def method_needs_multi_dataset(method_type: str) -> bool:
    return method_type in {"dpt", "dpt_direct", "bip3d", "custom"}


def build_dataset(args: argparse.Namespace):
    if args.dataset_kind == "multi":
        use_multi = True
    elif args.dataset_kind == "single":
        use_multi = False
    else:
        use_multi = method_needs_multi_dataset(args.method1_type) or method_needs_multi_dataset(args.method2_type)

    cls = GraspNetMultiDataset if use_multi else GraspNetDataset
    dataset = cls(
        args.dataset_root,
        split=str(args.test_mode),
        camera=args.camera,
        num_points=args.num_point,
        remove_outlier=bool(args.remove_outlier),
        augment=False,
        load_label=False,
    )
    return dataset, use_multi


def build_sample_indices(total: int, sample_interval: float, annos_per_scene: int, sample_offset: int) -> List[int]:
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")
    if sample_interval >= 1.0:
        return list(range(total))
    stride = max(1, int(round(1.0 / sample_interval)))
    offset = int(sample_offset) % stride
    indices: List[int] = []
    num_scenes = (total + annos_per_scene - 1) // annos_per_scene
    for scene_idx in range(num_scenes):
        start = scene_idx * annos_per_scene
        end = min((scene_idx + 1) * annos_per_scene, total)
        for local_ann in range(offset, end - start, stride):
            indices.append(start + local_ann)
    return indices


def build_dataloader(args: argparse.Namespace):
    full_dataset, use_multi = build_dataset(args)
    sampled_indices = build_sample_indices(
        total=len(full_dataset),
        sample_interval=float(args.sample_interval),
        annos_per_scene=int(args.annos_per_scene),
        sample_offset=int(args.sample_offset),
    )
    eval_dataset = Subset(full_dataset, sampled_indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn,
    )
    return full_dataset, eval_dataset, loader, sampled_indices, use_multi


def get_scene_list(full_dataset: Any) -> Sequence[str]:
    scene_list = full_dataset.scene_list()
    if len(scene_list) == len(full_dataset):
        return scene_list
    # Fallback for dataset implementations that return unique scene names only.
    expanded: List[str] = []
    for scene in scene_list:
        expanded.extend([scene] * 256)
    return expanded[: len(full_dataset)]



# -----------------------------------------------------------------------------
# Legacy argparse isolation
# -----------------------------------------------------------------------------
def build_legacy_argv_for_project_parser(
    args: argparse.Namespace,
    method_name: str,
    method_type: str,
    checkpoint: str,
    use_obs_depth: bool,
) -> List[str]:
    """Build argv compatible with the original utils.arguments parser.

    Some project modules import ``utils.arguments.cfgs`` at import time. That
    parser only accepts the original underscore-style CLI arguments. This script
    uses its own hyphen-style arguments, so dynamic model imports can otherwise
    crash with ``unrecognized arguments``. We temporarily expose only this legacy
    argv while importing models.
    """
    save_dir = str(Path(args.output_root) / method_name / "final_grasps")
    argv = [
        sys.argv[0],
        "--dataset_root", str(args.dataset_root),
        "--camera", str(args.camera),
        "--num_point", str(args.num_point),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--test_mode", str(args.test_mode),
        "--checkpoint_path", str(checkpoint),
        "--save_dir", save_dir,
        "--sample_interval", str(args.sample_interval),
        "--collision_thresh", str(args.collision_thresh),
        "--collision_voxel_size", str(args.collision_voxel_size),
        "--min_depth", str(args.min_depth),
        "--max_depth", str(args.max_depth),
        "--bin_num", str(args.bin_num),
    ]
    if method_type in {"dpt", "dpt_direct", "bip3d", "custom"}:
        argv.append("--multi_modal")
    if use_obs_depth:
        argv.append("--use_obs_depth")
    return argv


@contextmanager
def temporary_sys_argv(argv: Sequence[str]):
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def patch_legacy_cfgs_if_loaded(
    args: argparse.Namespace,
    method_name: str,
    method_type: str,
    checkpoint: str,
    use_obs_depth: bool,
) -> None:
    """Patch utils.arguments.cfgs if some imported module has already created it."""
    try:
        legacy_args = importlib.import_module("utils.arguments")
        cfgs = getattr(legacy_args, "cfgs", None)
        if cfgs is None:
            return
        setattr(cfgs, "dataset_root", str(args.dataset_root))
        setattr(cfgs, "camera", str(args.camera))
        setattr(cfgs, "num_point", int(args.num_point))
        setattr(cfgs, "batch_size", int(args.batch_size))
        setattr(cfgs, "num_workers", int(args.num_workers))
        setattr(cfgs, "test_mode", str(args.test_mode))
        setattr(cfgs, "checkpoint_path", str(checkpoint))
        setattr(cfgs, "save_dir", str(Path(args.output_root) / method_name / "final_grasps"))
        setattr(cfgs, "sample_interval", float(args.sample_interval))
        setattr(cfgs, "collision_thresh", float(args.collision_thresh))
        setattr(cfgs, "collision_voxel_size", float(args.collision_voxel_size))
        setattr(cfgs, "min_depth", float(args.min_depth))
        setattr(cfgs, "max_depth", float(args.max_depth))
        setattr(cfgs, "bin_num", int(args.bin_num))
        setattr(cfgs, "multi_modal", method_type in {"dpt", "dpt_direct", "bip3d", "custom"})
        setattr(cfgs, "use_obs_depth", bool(use_obs_depth))
        setattr(cfgs, "vis_dir", args.vis_dir)
        setattr(cfgs, "vis_every", int(args.vis_every))
    except Exception as e:
        print(f"[warn] Failed to patch utils.arguments.cfgs: {repr(e)}")

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def import_symbol(spec: str):
    if ":" not in spec:
        raise ValueError(f"Import spec must be 'module:callable', got {spec}")
    module_name, symbol_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def build_model_and_decoder(
    method_name: str,
    method_type: str,
    checkpoint: str,
    args: argparse.Namespace,
    use_obs_depth: bool,
    model_import: str = "",
    pred_decode_import: str = "",
):
    legacy_argv = build_legacy_argv_for_project_parser(
        args=args,
        method_name=method_name,
        method_type=method_type,
        checkpoint=checkpoint,
        use_obs_depth=use_obs_depth,
    )

    # Isolate the original project parser while importing modules. Some model
    # files transitively import utils.arguments.cfgs at import time; without this
    # block, that parser sees this script's --dataset-root/--methodX-* args and
    # exits with "unrecognized arguments".
    with temporary_sys_argv(legacy_argv):
        patch_legacy_cfgs_if_loaded(args, method_name, method_type, checkpoint, use_obs_depth)

        if method_type == "baseline":
            from models.economicgrasp import economicgrasp, pred_decode  # type: ignore
            net = economicgrasp(seed_feat_dim=int(args.seed_feat_dim), is_training=False)
            return net, pred_decode

        if method_type == "economicgrasp_obs_depth":
            from models.economicgrasp_depth import economicgrasp_depth_baseline, pred_decode  # type: ignore
            net = economicgrasp_depth_baseline(seed_feat_dim=512, 
                                    depth_stride=1, 
                                    min_depth=args.min_depth, 
                                    max_depth=args.max_depth, 
                                    is_training=False,
                                    use_obs_depth=True, 
                                    vis_dir=args.vis_dir, 
                                    vis_every=int(args.vis_every))
            return net, pred_decode

        if method_type == "dpt":
            from models.economicgrasp_bip3d import economicgrasp_dpt  # type: ignore
            # from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode  # type: ignore
            from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
            net = economicgrasp_dpt(
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                bin_num=int(args.bin_num),
                is_training=False,
                use_obs_depth=bool(use_obs_depth),
                vis_dir=args.vis_dir,
                vis_every=int(args.vis_every),
            )
            return net, pred_decode

        if method_type == "dpt_direct":
            from models.economicgrasp_bip3d import economicgrasp_dpt_direct  # type: ignore
            from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode  # type: ignore
            net = economicgrasp_dpt_direct(
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                bin_num=int(args.bin_num),
                is_training=False,
                vis_dir=args.vis_dir,
                vis_every=int(args.vis_every),
            )
            return net, pred_decode

        if method_type == "bip3d":
            from models.economicgrasp_bip3d import economicgrasp_bip3d  # type: ignore
            from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode  # type: ignore
            net = economicgrasp_bip3d(
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                bin_num=int(args.bin_num),
                is_training=False,
                vis_dir=args.vis_dir,
                vis_every=int(args.vis_every),
            )
            return net, pred_decode

        if method_type == "custom":
            if not model_import or not pred_decode_import:
                raise ValueError("custom method requires --methodX-model-import and --methodX-pred-decode-import")
            builder = import_symbol(model_import)
            pred_decode = import_symbol(pred_decode_import)
            # Custom builder is expected to either return a module or accept no args.
            net = builder()
            return net, pred_decode

    raise ValueError(f"Unsupported method_type: {method_type}")

def normalize_state_dict_keys(state: Mapping[str, Any], strip_module: bool) -> Dict[str, Any]:
    if not strip_module:
        return dict(state)
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_checkpoint(net: torch.nn.Module, checkpoint_path: str, strict: bool) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint does not contain a state_dict-like object: {checkpoint_path}")
    try:
        net.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        state2 = normalize_state_dict_keys(state, strip_module=True)
        try:
            net.load_state_dict(state2, strict=strict)
        except RuntimeError:
            raise e


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------
def move_batch_to_device(batch_data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key in list(batch_data.keys()):
        val = batch_data[key]
        if "list" in key and isinstance(val, list):
            for i in range(len(val)):
                if isinstance(val[i], list):
                    for j in range(len(val[i])):
                        if torch.is_tensor(val[i][j]):
                            val[i][j] = val[i][j].to(device)
                elif torch.is_tensor(val[i]):
                    val[i] = val[i].to(device)
        elif "graph" in key and isinstance(val, list):
            for i in range(len(val)):
                if torch.is_tensor(val[i]):
                    val[i] = val[i].to(device)
        elif torch.is_tensor(val):
            batch_data[key] = val.to(device)
    return batch_data


def as_numpy_grasps(pred: Any) -> np.ndarray:
    if torch.is_tensor(pred):
        arr = pred.detach().cpu().numpy()
    else:
        arr = np.asarray(pred)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 0:
        arr = arr.reshape(0, 17)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected decoded grasp shape: {arr.shape}")
    return arr


def split_csv_arg(x: str) -> Optional[set]:
    x = (x or "").strip()
    if not x:
        return None
    return {v.strip() for v in x.split(",") if v.strip()}


def split_csv_patterns(x: str) -> List[str]:
    x = (x or "").strip()
    if not x:
        return []
    return [v.strip() for v in x.split(",") if v.strip()]


def match_any_pattern(key: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(key, pat) for pat in patterns)


def tensor_basic_summary(key: str, value: Any, batch_i: int, batch_size: int, max_stats_elems: int = 200000) -> Dict[str, Any]:
    """Small statistical summary without copying full large tensors to CPU."""
    row: Dict[str, Any] = {"key": str(key), "type": type(value).__name__}
    try:
        if torch.is_tensor(value):
            x = value.detach()
            row.update({
                "shape_before_slice": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "numel_before_slice": int(x.numel()),
            })
            if x.ndim > 0 and x.shape[0] == batch_size:
                x = x[batch_i]
                row["sliced_batch_dim"] = True
            else:
                row["sliced_batch_dim"] = False
            row.update({"shape": list(x.shape), "numel": int(x.numel())})
            if x.numel() > 0 and (x.is_floating_point() or x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)):
                flat = x.reshape(-1)
                if flat.numel() > max_stats_elems:
                    flat = flat[:max_stats_elems]
                    row["stats_sampled"] = True
                else:
                    row["stats_sampled"] = False
                arr = flat.detach().float().cpu().numpy()
                finite = np.isfinite(arr)
                row["finite_count"] = int(finite.sum())
                if finite.any():
                    a = arr[finite]
                    row.update({
                        "min": float(np.min(a)),
                        "mean": float(np.mean(a)),
                        "std": float(np.std(a)),
                        "median": float(np.median(a)),
                        "max": float(np.max(a)),
                    })
            return row
        if isinstance(value, np.ndarray):
            x = value
            row.update({"shape_before_slice": list(x.shape), "dtype": str(x.dtype), "numel_before_slice": int(x.size)})
            if x.ndim > 0 and x.shape[0] == batch_size:
                x = x[batch_i]
                row["sliced_batch_dim"] = True
            else:
                row["sliced_batch_dim"] = False
            row.update({"shape": list(x.shape), "numel": int(x.size)})
            if x.size > 0 and np.issubdtype(x.dtype, np.number):
                flat = x.reshape(-1)
                if flat.size > max_stats_elems:
                    flat = flat[:max_stats_elems]
                    row["stats_sampled"] = True
                else:
                    row["stats_sampled"] = False
                finite = np.isfinite(flat)
                row["finite_count"] = int(finite.sum())
                if finite.any():
                    a = flat[finite].astype(np.float64)
                    row.update({"min": float(np.min(a)), "mean": float(np.mean(a)), "std": float(np.std(a)), "median": float(np.median(a)), "max": float(np.max(a))})
            return row
        if isinstance(value, (list, tuple)):
            row["len"] = int(len(value))
        elif isinstance(value, dict):
            row["len"] = int(len(value))
            row["subkeys"] = ";".join(list(map(str, value.keys()))[:50])
        else:
            row["repr"] = repr(value)[:300]
    except Exception as e:
        row["summary_error"] = repr(e)
    return row


def maybe_cast_float_tensor(x: torch.Tensor, float_dtype: str) -> torch.Tensor:
    if not x.is_floating_point():
        return x
    if float_dtype == "fp16":
        return x.to(torch.float16)
    if float_dtype == "bf16":
        return x.to(torch.bfloat16)
    return x.to(torch.float32)


def maybe_cast_float_array(x: np.ndarray, float_dtype: str) -> np.ndarray:
    if not np.issubdtype(x.dtype, np.floating):
        return x
    if float_dtype == "fp16":
        return x.astype(np.float16, copy=False)
    # NumPy has no stable portable bfloat16 for np.save/torch.save use; keep fp32.
    return x.astype(np.float32, copy=False)


def to_cpu_slice_compact(obj: Any, batch_i: int, batch_size: int, float_dtype: str, max_tensor_numel: Optional[int]) -> Any:
    """Convert tensors/arrays/lists/dicts to CPU and slice per-sample when possible.

    If max_tensor_numel is not None and the per-sample tensor remains larger, return a marker
    instead of storing the large payload.
    """
    if torch.is_tensor(obj):
        x = obj.detach()
        if x.ndim > 0 and x.shape[0] == batch_size:
            x = x[batch_i]
        if max_tensor_numel is not None and int(x.numel()) > int(max_tensor_numel):
            return f"<skipped_large_tensor shape={tuple(x.shape)} dtype={x.dtype} numel={int(x.numel())}>"
        x = maybe_cast_float_tensor(x, float_dtype).cpu().contiguous()
        return x
    if isinstance(obj, np.ndarray):
        x = obj
        if x.ndim > 0 and x.shape[0] == batch_size:
            x = x[batch_i]
        if max_tensor_numel is not None and int(x.size) > int(max_tensor_numel):
            return f"<skipped_large_array shape={tuple(x.shape)} dtype={x.dtype} numel={int(x.size)}>"
        return np.ascontiguousarray(maybe_cast_float_array(np.array(x), float_dtype))
    if isinstance(obj, dict):
        return {str(k): to_cpu_slice_compact(v, batch_i, batch_size, float_dtype, max_tensor_numel) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) == batch_size:
            return to_cpu_slice_compact(obj[batch_i], batch_i=0, batch_size=1, float_dtype=float_dtype, max_tensor_numel=max_tensor_numel)
        return [to_cpu_slice_compact(v, batch_i, batch_size, float_dtype, max_tensor_numel) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return repr(obj)


def should_save_endpoint_key(
    key: str,
    mode: str,
    include_keys: Optional[set],
    skip_keys: Optional[set],
    include_patterns: Sequence[str],
    skip_patterns: Sequence[str],
) -> bool:
    if skip_keys is not None and key in skip_keys:
        return False
    if match_any_pattern(key, skip_patterns):
        return False
    if mode == "summary":
        return False
    if include_keys is not None:
        return key in include_keys
    if mode == "full":
        return True
    # compact mode
    return match_any_pattern(key, include_patterns)


def atomic_torch_save(payload: Dict[str, Any], path: Path, compress: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if compress == "gzip":
        with gzip.open(str(tmp), "wb", compresslevel=3) as f:
            torch.save(payload, f)
    else:
        torch.save(payload, str(tmp))
    tmp.replace(path)


def save_sample_endpoints(
    path: Path,
    end_points: Dict[str, Any],
    batch_i: int,
    batch_size: int,
    meta: Dict[str, Any],
    raw_grasps: np.ndarray,
    final_grasps: np.ndarray,
    collision_mask: np.ndarray,
    include_keys: Optional[set],
    skip_keys: Optional[set],
    include_patterns: Sequence[str],
    skip_patterns: Sequence[str],
    save_mode: str,
    float_dtype: str,
    max_tensor_numel: int,
    compress: str,
    include_grasps: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_ep: Dict[str, Any] = {}
    summaries: List[Dict[str, Any]] = []

    for key, val in end_points.items():
        key_s = str(key)
        summaries.append(tensor_basic_summary(key_s, val, batch_i=batch_i, batch_size=batch_size))
        if not should_save_endpoint_key(
            key_s,
            mode=save_mode,
            include_keys=include_keys,
            skip_keys=skip_keys,
            include_patterns=include_patterns,
            skip_patterns=skip_patterns,
        ):
            continue
        try:
            saved_ep[key_s] = to_cpu_slice_compact(
                val,
                batch_i=batch_i,
                batch_size=batch_size,
                float_dtype=float_dtype,
                max_tensor_numel=max_tensor_numel if save_mode == "compact" else None,
            )
        except Exception as e:
            saved_ep[key_s] = f"<failed_to_serialize: {repr(e)}>"

    payload: Dict[str, Any] = {
        "meta": {
            **meta,
            "endpoint_save_mode": save_mode,
            "endpoint_float_dtype": float_dtype,
            "endpoint_max_tensor_numel": int(max_tensor_numel),
            "endpoint_compress": compress,
            "saved_endpoint_keys": sorted(saved_ep.keys()),
        },
        "end_points": saved_ep,
        "endpoint_summaries": summaries,
    }
    if include_grasps:
        payload.update({
            "raw_grasps": raw_grasps.astype(np.float32),
            "final_grasps": final_grasps.astype(np.float32),
            "model_free_collision_mask": collision_mask.astype(bool),
        })
    atomic_torch_save(payload, path, compress=compress)


def save_grasp_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gg = GraspGroup(arr.astype(np.float32))
    gg.save_npy(str(path))


def save_collision_npz(path: Path, raw: np.ndarray, final: np.ndarray, collision_mask: np.ndarray, meta: Dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        raw_grasps=raw.astype(np.float32),
        final_grasps=final.astype(np.float32),
        collision_mask=collision_mask.astype(bool),
        raw_count=np.array([raw.shape[0]], dtype=np.int64),
        final_count=np.array([final.shape[0]], dtype=np.int64),
        collision_count=np.array([int(np.sum(collision_mask))], dtype=np.int64),
        collision_thresh=np.array([float(args.collision_thresh)], dtype=np.float32),
        collision_voxel_size=np.array([float(args.collision_voxel_size)], dtype=np.float32),
        approach_dist=np.array([float(args.approach_dist)], dtype=np.float32),
        data_idx=np.array([int(meta["data_idx"])], dtype=np.int64),
        ann_id=np.array([int(meta["ann_id"])], dtype=np.int64),
        scene_name=np.array([str(meta["scene_name"])]),
        camera=np.array([str(meta["camera"])]),
        method_name=np.array([str(meta["method_name"])]),
    )


def write_sample_map(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def parse_endpoint_save_filter(x: str) -> Optional[set]:
    x = (x or "").strip()
    if not x:
        return None
    out = set()
    for item in x.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --endpoint-save-filter item: {item}. Expected scene:ann, e.g. 139:200")
        scene_s, ann_s = item.split(":", 1)
        out.add((int(scene_s), int(ann_s)))
    return out


def run_one_method(
    args: argparse.Namespace,
    method_name: str,
    method_type: str,
    checkpoint: str,
    use_obs_depth: bool,
    model_import: str,
    pred_decode_import: str,
    full_dataset: Any,
    loader: DataLoader,
    sampled_indices: Sequence[int],
    scene_list: Sequence[str],
    device: torch.device,
) -> None:
    print(f"\n========== Infer method: {method_name} ({method_type}) ==========")
    method_root = Path(args.output_root) / method_name
    raw_root = method_root / "raw_grasps"
    final_root = method_root / "final_grasps"
    collision_root = method_root / "collisions"
    endpoint_root = method_root / "end_points"
    batch_endpoint_root = method_root / "batch_end_points"

    net, pred_decode = build_model_and_decoder(
        method_name=method_name,
        method_type=method_type,
        checkpoint=checkpoint,
        args=args,
        use_obs_depth=use_obs_depth,
        model_import=model_import,
        pred_decode_import=pred_decode_import,
    )
    load_checkpoint(net, checkpoint, strict=bool(args.strict_load))
    net.to(device)
    net.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    include_keys = split_csv_arg(args.endpoint_keys)
    skip_keys = split_csv_arg(args.endpoint_skip_keys)
    include_patterns = split_csv_patterns(args.endpoint_patterns)
    skip_patterns = split_csv_patterns(args.endpoint_skip_patterns)
    endpoint_suffix = ".pth.gz" if args.endpoint_compress == "gzip" else ".pth"
    endpoint_filter = parse_endpoint_save_filter(args.endpoint_save_filter)
    if endpoint_filter is not None:
        print(f"[{method_name}] endpoint saving is restricted to {len(endpoint_filter)} scene:ann pairs: {sorted(endpoint_filter)[:10]}")

    tic = time.time()
    total_samples = 0
    for batch_idx, batch_data in enumerate(loader):
        batch_data = move_batch_to_device(batch_data, device)
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        cur_bs = len(grasp_preds)
        total_samples += cur_bs

        if args.save_batch_endpoints:
            # Save a CPU copy of the full batch for rare debugging. This can be large.
            batch_payload = {
                "batch_idx": int(batch_idx),
                "sampled_indices": [int(sampled_indices[batch_idx * args.batch_size + i]) for i in range(cur_bs)],
                "end_points": to_cpu_slice_compact(
                    end_points,
                    batch_i=0,
                    batch_size=1,
                    float_dtype=str(args.endpoint_float_dtype),
                    max_tensor_numel=int(args.endpoint_max_tensor_numel),
                ),
            }
            batch_endpoint_root.mkdir(parents=True, exist_ok=True)
            torch.save(batch_payload, str(batch_endpoint_root / f"batch_{batch_idx:06d}.pth"))

        for i in range(cur_bs):
            subset_data_idx = batch_idx * int(args.batch_size) + i
            data_idx = int(sampled_indices[subset_data_idx])
            scene_name = str(scene_list[data_idx])
            ann_id = int(data_idx % int(args.annos_per_scene))
            scene_id = int(scene_name.split("_")[-1]) if scene_name.startswith("scene_") else -1

            rel_npy = Path(scene_name) / args.camera / f"{ann_id:04d}.npy"
            rel_npz = Path(scene_name) / args.camera / f"{ann_id:04d}.npz"
            rel_pth = Path(scene_name) / args.camera / f"{ann_id:04d}{endpoint_suffix}"

            meta: Dict[str, Any] = {
                "method_name": method_name,
                "method_type": method_type,
                "checkpoint": checkpoint,
                "data_idx": data_idx,
                "subset_data_idx": int(subset_data_idx),
                "batch_idx": int(batch_idx),
                "batch_i": int(i),
                "scene_name": scene_name,
                "scene_id": int(scene_id),
                "ann_id": int(ann_id),
                "camera": args.camera,
                "sample_interval": float(args.sample_interval),
                "sample_offset": int(args.sample_offset),
                "test_mode": str(args.test_mode),
            }

            raw_arr = as_numpy_grasps(grasp_preds[i])
            if args.save_raw_grasps:
                save_grasp_npy(raw_root / rel_npy, raw_arr)

            if float(args.collision_thresh) > 0 and raw_arr.shape[0] > 0:
                gg = GraspGroup(raw_arr.copy())
                cloud, _ = full_dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=float(args.collision_voxel_size),
                )
                collision_mask_t = mfcdetector.detect(
                    gg,
                    approach_dist=float(args.approach_dist),
                    collision_thresh=float(args.collision_thresh),
                )
                collision_mask = collision_mask_t.detach().cpu().numpy().astype(bool)
                final_gg = gg[~collision_mask]
                final_arr = np.asarray(final_gg.grasp_group_array, dtype=np.float32)
            else:
                collision_mask = np.zeros((raw_arr.shape[0],), dtype=bool)
                final_arr = raw_arr.copy()

            if args.save_final_grasps:
                save_grasp_npy(final_root / rel_npy, final_arr)
            if args.save_collision:
                save_collision_npz(collision_root / rel_npz, raw_arr, final_arr, collision_mask, meta, args)
            if args.save_endpoints and (endpoint_filter is None or (int(scene_id), int(ann_id)) in endpoint_filter):
                save_sample_endpoints(
                    endpoint_root / rel_pth,
                    end_points=end_points,
                    batch_i=i,
                    batch_size=cur_bs,
                    meta=meta,
                    raw_grasps=raw_arr,
                    final_grasps=final_arr,
                    collision_mask=collision_mask,
                    include_keys=include_keys,
                    skip_keys=skip_keys,
                    include_patterns=include_patterns,
                    skip_patterns=skip_patterns,
                    save_mode=str(args.endpoint_save_mode),
                    float_dtype=str(args.endpoint_float_dtype),
                    max_tensor_numel=int(args.endpoint_max_tensor_numel),
                    compress=str(args.endpoint_compress),
                    include_grasps=bool(args.endpoint_include_grasps),
                )

        if batch_idx % 10 == 0:
            toc = time.time()
            print(
                f"[{method_name}] batch {batch_idx:05d}/{len(loader):05d}, "
                f"samples={total_samples}, avg_time_per_batch={(toc - tic) / max(1, 10):.4f}s"
            )
            tic = time.time()

    # Release GPU memory before the next method.
    del net
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Finished {method_name}. Saved to {method_root}")


def main() -> None:
    args = parse_args()
    if int(args.batch_size) != 4:
        print(f"[WARN] batch_size is {args.batch_size}; user requested batch size can be 4. Use --batch-size 4 if needed.")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    full_dataset, eval_dataset, loader, sampled_indices, use_multi = build_dataloader(args)
    scene_list = get_scene_list(full_dataset)

    sample_rows: List[Dict[str, Any]] = []
    for local_idx, data_idx in enumerate(sampled_indices):
        scene_name = str(scene_list[int(data_idx)])
        ann_id = int(data_idx % int(args.annos_per_scene))
        sample_rows.append({
            "subset_index": int(local_idx),
            "data_idx": int(data_idx),
            "scene_name": scene_name,
            "scene_id": int(scene_name.split("_")[-1]) if scene_name.startswith("scene_") else -1,
            "ann_id": int(ann_id),
            "camera": args.camera,
            "test_mode": args.test_mode,
        })
    write_sample_map(out_root / "sample_index.csv", sample_rows)

    run_meta = {
        "dataset_root": args.dataset_root,
        "test_mode": args.test_mode,
        "camera": args.camera,
        "num_point": int(args.num_point),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "sample_interval": float(args.sample_interval),
        "sample_offset": int(args.sample_offset),
        "annos_per_scene": int(args.annos_per_scene),
        "total_test_samples": int(len(full_dataset)),
        "evaluated_samples": int(len(eval_dataset)),
        "dataset_kind_used": "multi" if use_multi else "single",
        "method1": {"name": args.method1_name, "type": args.method1_type, "checkpoint": args.method1_checkpoint},
        "method2": {"name": args.method2_name, "type": args.method2_type, "checkpoint": args.method2_checkpoint},
        "endpoint_cache": {
            "save_mode": args.endpoint_save_mode,
            "float_dtype": args.endpoint_float_dtype,
            "compress": args.endpoint_compress,
            "max_tensor_numel": int(args.endpoint_max_tensor_numel),
            "include_grasps_in_endpoint": bool(args.endpoint_include_grasps),
            "patterns": args.endpoint_patterns,
            "skip_patterns": args.endpoint_skip_patterns,
            "exact_keys": args.endpoint_keys,
            "exact_skip_keys": args.endpoint_skip_keys,
            "save_filter": args.endpoint_save_filter,
        },
        "output_structure": {
            "final_grasps": "<output-root>/<method>/final_grasps/scene_xxxx/<camera>/xxxx.npy",
            "raw_grasps": "<output-root>/<method>/raw_grasps/scene_xxxx/<camera>/xxxx.npy",
            "collisions": "<output-root>/<method>/collisions/scene_xxxx/<camera>/xxxx.npz",
            "end_points": "<output-root>/<method>/end_points/scene_xxxx/<camera>/xxxx.pth or xxxx.pth.gz",
        },
    }
    with (out_root / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(json.dumps(run_meta, indent=2))

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    run_one_method(
        args=args,
        method_name=args.method1_name,
        method_type=args.method1_type,
        checkpoint=args.method1_checkpoint,
        use_obs_depth=args.method1_use_obs_depth,
        model_import=args.method1_model_import,
        pred_decode_import=args.method1_pred_decode_import,
        full_dataset=full_dataset,
        loader=loader,
        sampled_indices=sampled_indices,
        scene_list=scene_list,
        device=device,
    )
    run_one_method(
        args=args,
        method_name=args.method2_name,
        method_type=args.method2_type,
        checkpoint=args.method2_checkpoint,
        use_obs_depth=args.method2_use_obs_depth,
        model_import=args.method2_model_import,
        pred_decode_import=args.method2_pred_decode_import,
        full_dataset=full_dataset,
        loader=loader,
        sampled_indices=sampled_indices,
        scene_list=scene_list,
        device=device,
    )

    print(f"\nAll done. Final evaluable dump roots:")
    print(f"  {out_root / args.method1_name / 'final_grasps'}")
    print(f"  {out_root / args.method2_name / 'final_grasps'}")


if __name__ == "__main__":
    main()
