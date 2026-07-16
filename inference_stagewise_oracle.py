"""Stage-wise oracle decomposition for the first-generation CVA Transformer.

The script performs two deterministic forward passes for each batch:

1. the normal model-selected view;
2. a label-best-view override for the same selected center queries.

It then decodes a cumulative chain of operation oracles and writes one standard
GraspNet dump directory per mode.  GT labels are used only after the prediction
pass or to override the selected view in pass 2; this is a diagnostic script and
must never be reported as a normal inference result.

Environment controls (no change to utils.arguments is required):
  CVA_ORACLE_FACTOR_MODES=1  additionally emit orthogonal factor-only modes.
  CVA_ORACLE_SAVE_META=1     save compact per-query NPZ metadata.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup

from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import (
    GraspNetDataset,
    GraspNetMultiDataset,
    collate_fn,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


EMIT_FACTOR_MODES = _env_flag("CVA_ORACLE_FACTOR_MODES", False)
SAVE_ORACLE_META = _env_flag("CVA_ORACLE_SAVE_META", False)


# -----------------------------------------------------------------------------
# Dataset / dataloader
# -----------------------------------------------------------------------------

def my_worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataset(args):
    # Required by process_grasp_labels_extend_angle during eval diagnostics.
    dataset_cls = GraspNetMultiDataset if args.multi_modal else GraspNetDataset
    return dataset_cls(
        args.dataset_root,
        split=str(args.test_mode),
        camera=args.camera,
        num_points=args.num_point,
        remove_outlier=True,
        augment=False,
        load_label=True,
        extend_angle=cfgs.extend_angle
    )


def build_eval_subset(dataset, sample_interval: float, annos_per_scene: int = 256):
    total = len(dataset)
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices: List[int] = []
    num_scenes = (total + annos_per_scene - 1) // annos_per_scene
    for scene_idx in range(num_scenes):
        start = scene_idx * annos_per_scene
        end = min((scene_idx + 1) * annos_per_scene, total)
        indices.extend(start + x for x in range(0, end - start, stride))
    return Subset(dataset, indices), indices


def build_dataloader(args):
    full_dataset = build_dataset(args)
    sample_interval = float(getattr(args, "sample_interval", 1.0))
    eval_dataset, sampled_indices = build_eval_subset(
        full_dataset, sample_interval
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 2)),
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn,
    )
    return full_dataset, eval_dataset, loader, sampled_indices


FULL_TEST_DATASET, TEST_DATASET, TEST_DATALOADER, SAMPLED_INDICES = (
    build_dataloader(cfgs)
)
SCENE_LIST = FULL_TEST_DATASET.scene_list()

print(f"Total test samples: {len(FULL_TEST_DATASET)}")
print(f"Evaluated samples:  {len(TEST_DATASET)}")
print(f"sample_interval:    {getattr(cfgs, 'sample_interval', 1.0)}")
print("load_label:         True (required for stage-wise oracle)")
print(f"factor modes:       {EMIT_FACTOR_MODES}")
print(f"save compact meta:  {SAVE_ORACLE_META}")


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

if not cfgs.multi_modal:
    raise ValueError(
        "This implementation targets the multi-modal first-generation "
        "economicgrasp_dpt CVA Transformer. Set --multi_modal."
    )
if int(getattr(cfgs, "kview_k", 1)) != 1:
    raise ValueError(
        "The first-generation stage-wise decoder assumes one selected view "
        "per center (kview_k=1). Use a kview_k=1 checkpoint/config for this "
        "decomposition; RotNet/top-L requires a separate implementation."
    )

from models.economicgrasp_bip3d import (  # noqa: E402
    economicgrasp_dpt,
    pred_decode_center_view_angle_oracle,
)

net = economicgrasp_dpt(
    min_depth=cfgs.min_depth,
    max_depth=cfgs.max_depth,
    bin_num=cfgs.bin_num,
    is_training=False,
    use_obs_depth=bool(getattr(cfgs, "use_obs_depth", False)),
    vis_dir=getattr(cfgs, "vis_dir", None),
    vis_every=int(getattr(cfgs, "vis_every", 1000)),
    oracle_diag=True,
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(DEVICE)

checkpoint = torch.load(cfgs.checkpoint_path, map_location="cpu")
state_dict = (
    checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    else checkpoint
)
try:
    net.load_state_dict(state_dict)
except RuntimeError:
    # Common DDP checkpoint fallback.
    if state_dict and all(str(k).startswith("module.") for k in state_dict):
        stripped = {str(k)[7:]: v for k, v in state_dict.items()}
        net.load_state_dict(stripped)
    else:
        raise
print(f"-> loaded checkpoint {cfgs.checkpoint_path}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _move_batch_to_device(batch_data: MutableMapping[str, Any], device) -> None:
    """Mirror the repository's nested-list transfer convention."""
    for key in list(batch_data.keys()):
        value = batch_data[key]
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device)
        elif "graph" in key:
            for i in range(len(value)):
                value[i] = value[i].to(device)
        elif torch.is_tensor(value):
            batch_data[key] = value.to(device)


def _copy_input_dict(batch_data: Mapping[str, Any]) -> Dict[str, Any]:
    # The network adds keys to end_points but does not require tensor clones.
    return dict(batch_data)


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _mode_save_root(base_dir: str, mode: str) -> str:
    return base_dir if mode == "s0_base" else f"{base_dir}_{mode}"


def _save_sample_manifest(
    root: str,
    sampled_indices: Iterable[int],
    sample_interval: float,
    test_mode: str,
    camera: str,
) -> None:
    sampled_indices = list(sampled_indices)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "_sampled_indices.json"), "w") as f:
        json.dump(
            {
                "test_mode": str(test_mode),
                "camera": str(camera),
                "sample_interval": float(sample_interval),
                "num_samples": len(sampled_indices),
                "sampled_indices": [int(x) for x in sampled_indices],
            },
            f,
            indent=2,
        )


def _save_debug_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = sorted(set().union(*(set(r.keys()) for r in rows)))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _as_bqv(x: torch.Tensor, q: int, v: int, name: str) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"{name} must be 3D, got {tuple(x.shape)}")
    if x.shape[1:] == (q, v):
        return x
    if x.shape[1:] == (v, q):
        return x.transpose(1, 2).contiguous()
    raise ValueError(
        f"Cannot canonicalize {name} to [B,Q,V]=[B,{q},{v}], "
        f"got {tuple(x.shape)}"
    )


def _derive_oracle_view_indices(end_points: Mapping[str, Any]):
    """Return label-best view indices with model-view fallback per query.

    This is a training-label proposal oracle, not an evaluator-exact oracle.
    """
    required = [
        "view_score",
        "grasp_top_view_inds",
        "batch_grasp_view_graspness",
    ]
    missing = [k for k in required if k not in end_points]
    if missing:
        raise KeyError("Cannot derive oracle view; missing: " + ", ".join(missing))

    pred_idx = end_points["grasp_top_view_inds"].detach().long()
    B, Q = pred_idx.shape
    raw_view_score = end_points["view_score"].detach()
    V = raw_view_score.shape[2] if raw_view_score.shape[1] == Q else raw_view_score.shape[1]
    view_score = _as_bqv(raw_view_score, Q, V, "view_score")
    view_label = _as_bqv(
        end_points["batch_grasp_view_graspness"].detach(),
        Q,
        V,
        "batch_grasp_view_graspness",
    ).to(view_score)

    finite = torch.isfinite(view_label)
    neg_inf = torch.finfo(view_label.dtype).min
    safe_label = torch.where(
        finite, view_label, torch.full_like(view_label, neg_inf)
    )
    oracle_value, oracle_idx_raw = safe_label.max(dim=-1)

    min_quality = float(getattr(cfgs, "oracle_view_min_label", 1e-6))
    valid_query = finite.any(dim=-1) & (oracle_value > min_quality)
    batch_valid = end_points.get("batch_valid_mask", None)
    if torch.is_tensor(batch_valid) and batch_valid.shape == (B, Q):
        valid_query &= batch_valid.bool().to(valid_query.device)

    oracle_idx = torch.where(valid_query, oracle_idx_raw.long(), pred_idx)
    pred_gt = torch.gather(
        view_label, -1, pred_idx.clamp(0, V - 1).unsqueeze(-1)
    ).squeeze(-1)

    topk = min(5, V)
    gt_topk = torch.topk(safe_label, k=topk, dim=-1).indices
    pred_in_top5 = (gt_topk == pred_idx.unsqueeze(-1)).any(dim=-1)
    denom = valid_query.float().sum(dim=1).clamp_min(1.0)
    diag = {
        "oracle_view_valid_ratio": valid_query.float().mean(dim=1),
        "pred_view_eq_oracle_ratio": (pred_idx == oracle_idx).float().mean(dim=1),
        "pred_view_in_gt_top5_ratio": pred_in_top5.float().mean(dim=1),
        "pred_view_gt_score_mean": torch.where(
            valid_query, pred_gt, torch.zeros_like(pred_gt)
        ).sum(dim=1) / denom,
        "oracle_view_gt_score_mean": torch.where(
            valid_query, oracle_value, torch.zeros_like(oracle_value)
        ).sum(dim=1) / denom,
    }
    compact = {
        "pred_view_idx": pred_idx,
        "oracle_view_idx": oracle_idx,
        "oracle_view_valid": valid_query,
        "pred_view_gt_score": pred_gt,
        "oracle_view_gt_score": oracle_value,
    }
    return oracle_idx, valid_query, diag, compact


def _assert_same_queries(
    base_end_points: Mapping[str, Any],
    oracle_end_points: Mapping[str, Any],
) -> None:
    for key in ["token_sel_idx", "xyz_graspable"]:
        a = base_end_points.get(key)
        b = oracle_end_points.get(key)
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise KeyError(f"Both passes must expose tensor key '{key}'.")
        if a.shape != b.shape:
            raise RuntimeError(
                f"Two-pass query mismatch for {key}: {tuple(a.shape)} vs "
                f"{tuple(b.shape)}"
            )
        same = torch.equal(a, b) if key == "token_sel_idx" else torch.allclose(
            a, b, atol=1e-7, rtol=1e-6
        )
        if not same:
            raise RuntimeError(
                f"Normal and oracle-view passes selected different {key}. "
                "RNG state is restored between passes; inspect other "
                "nondeterministic operations before interpreting the result."
            )


def _combine_mode_outputs(base_modes, view_modes):
    required = {
        "base",
        "oracle_angle",
        "oracle_depth",
        "oracle_angle_depth",
        "oracle_operation",
        "oracle_label_rank",
        "oracle_operation_label_rank",
    }
    for name, modes in [("base pass", base_modes), ("oracle-view pass", view_modes)]:
        missing = sorted(required - set(modes))
        if missing:
            raise KeyError(f"{name} decoder is missing modes: {missing}")

    out = {
        "s0_base": base_modes["base"],
        "s1_oracle_view": view_modes["base"],
        "s2_oracle_view_angle": view_modes["oracle_angle"],
        "s3_oracle_view_angle_depth": view_modes["oracle_angle_depth"],
        "s4_oracle_view_operation": view_modes["oracle_operation"],
        "s5_oracle_view_operation_labelrank": view_modes[
            "oracle_operation_label_rank"
        ],
    }
    if EMIT_FACTOR_MODES:
        out.update(
            {
                "f_oracle_angle_only": base_modes["oracle_angle"],
                "f_oracle_depth_only": base_modes["oracle_depth"],
                "f_oracle_angle_depth": base_modes["oracle_angle_depth"],
                "f_oracle_operation": base_modes["oracle_operation"],
                "f_oracle_labelrank_basepose": base_modes["oracle_label_rank"],
                "f_oracle_operation_labelrank": base_modes[
                    "oracle_operation_label_rank"
                ],
            }
        )
    return out


def _save_compact_meta_npz(
    out_path: str,
    base_end_points: Mapping[str, Any],
    view_end_points: Mapping[str, Any],
    sample_i: int,
    view_compact: Mapping[str, torch.Tensor],
) -> None:
    payload: Dict[str, np.ndarray] = {}
    for key, value in view_compact.items():
        payload[key] = value[sample_i].detach().cpu().numpy()

    for prefix, ep in [("base", base_end_points), ("view", view_end_points)]:
        for key in ["token_sel_idx", "xyz_graspable", "grasp_top_view_inds"]:
            value = ep.get(key)
            if torch.is_tensor(value):
                payload[f"{prefix}_{key}"] = value[sample_i].detach().cpu().numpy()
        meta = ep.get("cva_oracle_meta")
        if isinstance(meta, list) and sample_i < len(meta):
            for key, value in meta[sample_i].items():
                if torch.is_tensor(value):
                    payload[f"{prefix}_{key}"] = value.detach().cpu().numpy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)


# -----------------------------------------------------------------------------
# Main inference
# -----------------------------------------------------------------------------

def inference() -> None:
    print_interval = int(getattr(cfgs, "print_interval", 20))
    sample_interval = float(getattr(cfgs, "sample_interval", 1.0))
    net.eval()
    os.makedirs(cfgs.save_dir, exist_ok=True)
    if sample_interval < 1.0:
        print(
            f"[WARN] sample_interval={sample_interval}; use clean output "
            "directories to avoid stale full-evaluation files."
        )

    all_debug_rows: List[Dict[str, Any]] = []
    modes_seen: Optional[List[str]] = None
    tic = time.time()

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        _move_batch_to_device(batch_data, DEVICE)
        rng_before = _capture_rng_state()

        with torch.no_grad():
            base_input = _copy_input_dict(batch_data)
            base_input["oracle_diag_enable"] = True
            base_end_points = net(base_input)
            base_modes = pred_decode_center_view_angle_oracle(
                base_end_points, return_dict=True
            )

        rng_after_base = _capture_rng_state()
        oracle_view_idx, oracle_view_valid, view_diag, view_compact = (
            _derive_oracle_view_indices(base_end_points)
        )

        _restore_rng_state(rng_before)
        with torch.no_grad():
            view_input = _copy_input_dict(batch_data)
            view_input["oracle_diag_enable"] = True
            view_input["oracle_view_inds_override"] = oracle_view_idx
            view_end_points = net(view_input)
            view_modes = pred_decode_center_view_angle_oracle(
                view_end_points, return_dict=True
            )
        _restore_rng_state(rng_after_base)

        _assert_same_queries(base_end_points, view_end_points)
        grasp_preds_by_mode = _combine_mode_outputs(base_modes, view_modes)
        modes = list(grasp_preds_by_mode)

        if modes_seen is None:
            modes_seen = modes
            mode_roots = {}
            for mode in modes:
                root = _mode_save_root(cfgs.save_dir, mode)
                mode_roots[mode] = root
                _save_sample_manifest(
                    root,
                    SAMPLED_INDICES,
                    sample_interval,
                    cfgs.test_mode,
                    cfgs.camera,
                )
            with open(os.path.join(cfgs.save_dir, "_stagewise_modes.json"), "w") as f:
                json.dump(
                    {
                        "base_save_dir": cfgs.save_dir,
                        "test_mode": str(cfgs.test_mode),
                        "camera": str(cfgs.camera),
                        "modes": mode_roots,
                        "cumulative_chain": [
                            "s0_base",
                            "s1_oracle_view",
                            "s2_oracle_view_angle",
                            "s3_oracle_view_angle_depth",
                            "s4_oracle_view_operation",
                            "s5_oracle_view_operation_labelrank",
                        ],
                        "factor_modes_enabled": EMIT_FACTOR_MODES,
                    },
                    f,
                    indent=2,
                )
            print(f"[stage-wise oracle modes] {modes}")

        cur_bs = len(grasp_preds_by_mode[modes[0]])
        base_debug = base_end_points.get("cva_oracle_debug_rows", [])
        view_debug = view_end_points.get("cva_oracle_debug_rows", [])

        for i in range(cur_bs):
            subset_idx = batch_idx * cfgs.batch_size + i
            data_idx = SAMPLED_INDICES[subset_idx]
            scene_name = SCENE_LIST[data_idx]
            anno_id = data_idx % 256

            gg_by_mode = {
                mode: GraspGroup(
                    grasp_preds_by_mode[mode][i].detach().cpu().numpy()
                )
                for mode in modes
            }

            if bool(getattr(cfgs, "save_nocollision", False)):
                for mode, gg in gg_by_mode.items():
                    root = _mode_save_root(cfgs.save_dir, mode) + "_nocollision"
                    out_dir = os.path.join(root, scene_name, cfgs.camera)
                    os.makedirs(out_dir, exist_ok=True)
                    gg.save_npy(os.path.join(out_dir, f"{anno_id:04d}.npy"))

            detector = None
            if float(getattr(cfgs, "collision_thresh", 0.0)) > 0:
                cloud, _ = FULL_TEST_DATASET.get_data(
                    data_idx, return_raw_cloud=True
                )
                detector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size,
                )

            row: Dict[str, Any] = {
                "split": str(cfgs.test_mode),
                "scene": str(scene_name),
                "data_idx": int(data_idx),
                "anno_id": int(anno_id),
            }
            for key, value in view_diag.items():
                row[key] = float(value[i].detach().cpu())
            if isinstance(base_debug, list) and i < len(base_debug):
                row.update({f"base_{k}": v for k, v in base_debug[i].items()})
            if isinstance(view_debug, list) and i < len(view_debug):
                row.update({f"view_{k}": v for k, v in view_debug[i].items()})

            # Oracle modes change the physical pose, so each mode requires its
            # own model-free collision mask.  Reusing the base mask is invalid.
            for mode, gg in gg_by_mode.items():
                raw_count = len(gg)
                if detector is not None and raw_count > 0:
                    collision_mask = detector.detect(
                        gg,
                        approach_dist=0.05,
                        collision_thresh=cfgs.collision_thresh,
                    ).detach().cpu().numpy().astype(bool)
                    gg = gg[~collision_mask]
                    row[f"{mode}_model_free_collision_ratio"] = float(
                        collision_mask.mean() if collision_mask.size else 0.0
                    )
                else:
                    row[f"{mode}_model_free_collision_ratio"] = 0.0

                row[f"{mode}_raw_count"] = int(raw_count)
                row[f"{mode}_final_count"] = int(len(gg))
                root = _mode_save_root(cfgs.save_dir, mode)
                out_dir = os.path.join(root, scene_name, cfgs.camera)
                os.makedirs(out_dir, exist_ok=True)
                gg.save_npy(os.path.join(out_dir, f"{anno_id:04d}.npy"))

            if SAVE_ORACLE_META:
                _save_compact_meta_npz(
                    os.path.join(
                        cfgs.save_dir,
                        "_oracle_meta",
                        scene_name,
                        cfgs.camera,
                        f"{anno_id:04d}.npz",
                    ),
                    base_end_points,
                    view_end_points,
                    i,
                    view_compact,
                )

            all_debug_rows.append(row)

        if batch_idx % max(print_interval, 1) == 0:
            toc = time.time()
            print(
                f"Oracle batch {batch_idx}/{len(TEST_DATALOADER)}; "
                f"elapsed={toc - tic:.2f}s"
            )
            tic = toc

    out_csv = os.path.join(
        cfgs.save_dir,
        f"_stagewise_oracle_diag_{cfgs.test_mode}_"
        f"si{getattr(cfgs, 'sample_interval', 1.0)}.csv",
    )
    _save_debug_csv(all_debug_rows, out_csv)
    print(f"[diag] saved {out_csv}")


if __name__ == "__main__":
    inference()
