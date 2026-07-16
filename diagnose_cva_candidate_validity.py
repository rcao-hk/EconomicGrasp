#!/usr/bin/env python3
"""Candidate-level validity diagnostics and frozen probes for first-generation CVA.

This script implements four diagnostics on the *predicted-view-conditioned* CVA
candidate lattice:

1. Safe-alternative recoverability for invalid high-ranked grasps.
2. Mixed-depth collision / empty / validity ratios.
3. Agreement among the learned collision head, training collision labels, and
   the official GraspNet CAD/DexNet evaluator.
4. Frozen-feature linear probes for official collision, validity, and safe
   grasp utility.

Important
---------
- This is a diagnostic program. It uses GT scene CAD models and object poses to
  label candidates and must never be used as normal test-time inference.
- It targets the first CVA implementation with one selected view per center
  (kview_k=1), not the RotNet/top-L implementation.
- It expects the stage-wise-oracle model patch: ``economicgrasp_dpt(...,
  oracle_diag=True)`` must be supported so eval-time labels can be attached.
- Raw candidates are evaluated *without* the official score-dependent top-10
  pruning. This is deliberate: safe alternatives at the same center must not be
  hidden by evaluator preselection.

The custom ``--diag_*`` options are consumed before importing the repository's
``utils.arguments`` module, so no change to the project argument parser is
required. All ordinary EconomicGrasp options remain available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _consume_custom_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diag_output_dir", default=None)
    parser.add_argument("--diag_scene_ids", default="", help="Absolute scene IDs, comma-separated.")
    parser.add_argument("--diag_anno_ids", default="", help="Annotation IDs, comma-separated.")
    parser.add_argument("--diag_max_samples", type=int, default=60)
    parser.add_argument("--diag_max_centers", type=int, default=24)
    parser.add_argument("--diag_top_angles", type=int, default=12)
    parser.add_argument("--diag_top_depths", type=int, default=4)
    parser.add_argument("--diag_eval_chunk", type=int, default=192)
    parser.add_argument("--diag_safe_mus", default="0.4,0.8")
    parser.add_argument("--diag_rank_ks", default="10,50")
    parser.add_argument("--diag_seed", type=int, default=0)
    parser.add_argument("--diag_threads", type=int, default=1)
    parser.add_argument("--diag_save_candidate_rows", type=int, default=1)
    parser.add_argument("--diag_skip_force_closure", type=int, default=0)
    parser.add_argument("--diag_probe", type=int, default=1)
    parser.add_argument("--diag_probe_max_rows", type=int, default=180000)
    parser.add_argument("--diag_probe_val_fraction", type=float, default=0.25)
    parser.add_argument("--diag_probe_feature_key", default="auto")
    parser.add_argument("--diag_probe_hook_module", default="")
    parser.add_argument(
        "--diag_probe_hook_source",
        choices=["input", "output"],
        default="output",
    )
    parser.add_argument("--diag_strict", type=int, default=1)
    custom, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    return custom


DIAG = _consume_custom_args()

# Limit nested BLAS/OpenMP parallelism. Candidate evaluation is already chunky.
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name, str(max(1, int(DIAG.diag_threads))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from graspnetAPI import GraspGroup, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    collision_detection,
    compute_closest_points,
    create_table_points,
    get_grasp_score,
    transform_points,
    voxel_sample_points,
)
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory

from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn
from utils.label_generation import batch_viewpoint_params_to_matrix


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    return obj


def _save_json(payload: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_safe_json(dict(payload)), f, indent=2)
    os.replace(tmp, path)


def _save_csv(rows: Sequence[Mapping[str, Any]], path: str, gzip_output: bool = False) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(set().union(*(set(r.keys()) for r in rows)))
    opener = gzip.open if gzip_output else open
    mode = "wt" if gzip_output else "w"
    with opener(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _move_batch_to_device(batch: MutableMapping[str, Any], device: torch.device) -> None:
    for key in list(batch.keys()):
        value = batch[key]
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device)
        elif "graph" in key:
            for i in range(len(value)):
                value[i] = value[i].to(device)
        elif torch.is_tensor(value):
            batch[key] = value.to(device)


def _worker_init(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _entropy(prob: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def _binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(score).astype(np.float64)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    out: Dict[str, float] = {"n": float(len(y)), "positive_ratio": float(y.mean()) if len(y) else float("nan")}
    if len(y) == 0:
        return out
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    out.update(
        {
            "accuracy": float((tp + tn) / max(len(y), 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
        }
    )
    if np.unique(y).size >= 2:
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")

    # Equal-width ECE, sufficient for a diagnostic calibration check.
    ece = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(s[mask].mean()) - float(y[mask].mean()))
    out["ece10"] = float(ece)
    return out


def _spearman(y: np.ndarray, score: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(y) < 2:
        return float("nan")
    value = spearmanr(y, score, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else float("nan")


def _ndcg_and_topk(
    utility: np.ndarray,
    score: np.ndarray,
    sample_ids: np.ndarray,
    ks: Sequence[int] = (10, 50),
) -> Dict[str, float]:
    from sklearn.metrics import ndcg_score

    rows: Dict[str, List[float]] = defaultdict(list)
    for sid in np.unique(sample_ids):
        m = sample_ids == sid
        if m.sum() < 2:
            continue
        u = utility[m].astype(np.float64)
        s = score[m].astype(np.float64)
        try:
            rows["ndcg50"].append(float(ndcg_score(u[None, :], s[None, :], k=min(50, len(u)))))
        except ValueError:
            pass
        order = np.argsort(-s, kind="stable")
        for k in ks:
            kk = min(k, len(order))
            if kk:
                rows[f"top{k}_invalid"].append(float(np.mean(u[order[:kk]] <= 0.0)))
                rows[f"top{k}_safe08"].append(float(np.mean(u[order[:kk]] >= 0.5)))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in rows.items()}


# -----------------------------------------------------------------------------
# Dataset / model
# -----------------------------------------------------------------------------


def _build_dataset():
    cls = GraspNetMultiDataset if cfgs.multi_modal else GraspNetDataset
    return cls(
        cfgs.dataset_root,
        split=str(cfgs.test_mode),
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=True,
        extend_angle=cfgs.extend_angle,
    )


def _select_indices(dataset) -> Tuple[List[int], List[str]]:
    scene_ids = set(_csv_ints(DIAG.diag_scene_ids))
    anno_ids = set(_csv_ints(DIAG.diag_anno_ids))
    scene_list = list(dataset.scene_list())
    interval = float(getattr(cfgs, "sample_interval", 1.0))
    stride = 1 if interval >= 1.0 else max(1, int(round(1.0 / interval)))

    selected: List[int] = []
    for data_idx, scene_name in enumerate(scene_list):
        try:
            sid = int(str(scene_name).split("_")[-1])
        except ValueError:
            continue
        anno = data_idx % 256
        if scene_ids and sid not in scene_ids:
            continue
        if anno_ids:
            if anno not in anno_ids:
                continue
        elif anno % stride != 0:
            continue
        selected.append(data_idx)
        if DIAG.diag_max_samples > 0 and len(selected) >= DIAG.diag_max_samples:
            break
    if not selected:
        raise RuntimeError("No diagnostic samples selected. Check scene/annotation filters.")
    return selected, scene_list


def _load_model(device: torch.device):
    if not cfgs.multi_modal:
        raise ValueError("This diagnostic targets the multi-modal CVA model; pass --multi_modal.")
    if int(getattr(cfgs, "kview_k", 1)) != 1:
        raise ValueError("This script targets first-generation CVA with --kview_k 1, not RotNet/top-L.")

    from models.economicgrasp_bip3d import economicgrasp_dpt

    try:
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
    except TypeError as exc:
        raise RuntimeError(
            "economicgrasp_dpt does not accept oracle_diag=True. Apply the "
            "stage-wise-oracle model patch before running this script."
        ) from exc

    checkpoint = torch.load(cfgs.checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        net.load_state_dict(state)
    except RuntimeError:
        if state and all(str(k).startswith("module.") for k in state):
            net.load_state_dict({str(k)[7:]: v for k, v in state.items()})
        else:
            raise
    net.to(device).eval()
    return net


class _HookCapture:
    def __init__(self, net: torch.nn.Module, module_path: str, source: str):
        self.tensor: Optional[torch.Tensor] = None
        self.handle = None
        self.module_path = module_path
        if not module_path:
            return
        modules = dict(net.named_modules())
        if module_path not in modules:
            candidates = [x for x in modules if module_path.lower() in x.lower()]
            raise KeyError(
                f"Probe hook module '{module_path}' not found. Partial matches: {candidates[:30]}"
            )

        def first_tensor(obj: Any) -> Optional[torch.Tensor]:
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, Mapping):
                vals = [first_tensor(v) for v in obj.values()]
            elif isinstance(obj, (list, tuple)):
                vals = [first_tensor(v) for v in obj]
            else:
                vals = []
            vals = [x for x in vals if torch.is_tensor(x)]
            return max(vals, key=lambda x: x.numel()) if vals else None

        def hook(_module, inputs, output):
            obj = inputs if source == "input" else output
            t = first_tensor(obj)
            self.tensor = None if t is None else t.detach()

        self.handle = modules[module_path].register_forward_hook(hook)

    def close(self):
        if self.handle is not None:
            self.handle.remove()


# -----------------------------------------------------------------------------
# Official raw-candidate evaluator
# -----------------------------------------------------------------------------


@dataclass
class EvalResult:
    assigned_obj: np.ndarray
    collision_or_empty: np.ndarray
    empty: np.ndarray
    pure_collision: np.ndarray
    friction: np.ndarray


class RawCandidateEvaluator:
    """Official CAD collision/DexNet quality without score-dependent pruning."""

    def __init__(self, root: str, camera: str, split: str, chunk: int, skip_force_closure: bool):
        self.eval = GraspNetEval(root, camera, split=split)
        self.config = get_config()
        self.chunk = max(1, int(chunk))
        self.skip_fc = bool(skip_force_closure)
        self.table = create_table_points(
            1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008
        )
        self.scene_cache: Dict[int, Tuple[List[np.ndarray], List[Any]]] = {}
        self.fc_list = np.asarray([1.2, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
        self.fc_configs: Dict[float, Any] = {}
        if not self.skip_fc:
            for mu in self.fc_list:
                key = round(float(mu), 2)
                self.config["metrics"]["force_closure"]["friction_coef"] = key
                self.fc_configs[key] = GraspQualityConfigFactory.create_config(
                    self.config["metrics"]["force_closure"]
                )

    def _scene_models(self, scene_id: int) -> Tuple[List[np.ndarray], List[Any]]:
        if scene_id not in self.scene_cache:
            models, dexmodels, _ = self.eval.get_scene_models(scene_id, ann_id=0)
            sampled = [voxel_sample_points(x, 0.008) for x in models]
            self.scene_cache[scene_id] = (sampled, dexmodels)
        return self.scene_cache[scene_id]

    def evaluate(self, scene_id: int, anno_id: int, grasps: np.ndarray) -> EvalResult:
        n = len(grasps)
        if n == 0:
            z_i = np.zeros((0,), dtype=np.int64)
            z_b = np.zeros((0,), dtype=bool)
            z_f = np.zeros((0,), dtype=np.float32)
            return EvalResult(z_i, z_b, z_b, z_b, z_f)

        models_obj, dexmodels = self._scene_models(scene_id)
        _, poses, camera_pose, align_mat = self.eval.get_model_poses(scene_id, anno_id)
        models_cam = [transform_points(m, poses[i]) for i, m in enumerate(models_obj)]
        scene = np.concatenate(models_cam, axis=0)
        seg = np.concatenate(
            [np.full(len(m), i, dtype=np.int64) for i, m in enumerate(models_cam)], axis=0
        )
        nearest = compute_closest_points(grasps[:, 13:16], scene)
        assigned = seg[nearest]
        table_cam = transform_points(self.table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        scene_with_table = np.concatenate([scene, table_cam], axis=0)

        collision = np.zeros(n, dtype=bool)
        empty = np.zeros(n, dtype=bool)
        friction = np.full(n, -1.0, dtype=np.float32)

        for obj_idx in range(len(models_cam)):
            obj_rows = np.flatnonzero(assigned == obj_idx)
            for start in range(0, len(obj_rows), self.chunk):
                ids = obj_rows[start : start + self.chunk]
                if len(ids) == 0:
                    continue
                chunk_grasps = grasps[ids]
                coll_l, empty_l, dex_l = collision_detection(
                    [chunk_grasps],
                    [models_cam[obj_idx]],
                    [dexmodels[obj_idx]],
                    [poses[obj_idx]],
                    scene_with_table,
                    outlier=0.05,
                    return_dexgrasps=True,
                )
                c = np.asarray(coll_l[0], dtype=bool)
                e = np.asarray(empty_l[0], dtype=bool)
                collision[ids] = c
                empty[ids] = e
                if self.skip_fc:
                    friction[ids] = np.where(c, -1.0, 1.2).astype(np.float32)
                    continue
                dexgrasps = dex_l[0]
                for local_i, global_i in enumerate(ids):
                    if c[local_i] or dexgrasps[local_i] is None:
                        friction[global_i] = -1.0
                    else:
                        friction[global_i] = float(
                            get_grasp_score(
                                dexgrasps[local_i],
                                dexmodels[obj_idx],
                                self.fc_list,
                                self.fc_configs,
                            )
                        )

        pure_collision = collision & (~empty)
        return EvalResult(assigned, collision, empty, pure_collision, friction)


# -----------------------------------------------------------------------------
# Candidate construction and feature extraction
# -----------------------------------------------------------------------------


def _score_expected(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # logits: [C,Q,A]
    prob = F.softmax(logits, dim=0)
    bins = torch.linspace(0.0, 1.0, logits.shape[0], device=logits.device, dtype=logits.dtype)
    expected = (prob * bins[:, None, None]).sum(dim=0)
    return expected, prob


def _token_scalar(
    end_points: Mapping[str, Any],
    batch_i: int,
    key: str,
    q: int,
    default: float = 0.0,
) -> torch.Tensor:
    token_idx = end_points.get("token_sel_idx")
    value = end_points.get(key)
    if not (torch.is_tensor(token_idx) and torch.is_tensor(value)):
        return torch.full((q,), default, device=end_points["xyz_graspable"].device)
    flat = value[batch_i].float()
    if flat.dim() > 1:
        flat = flat.reshape(-1)
    idx = token_idx[batch_i].long().clamp(0, max(flat.numel() - 1, 0))
    return flat.index_select(0, idx)


def _query_aux(end_points: Mapping[str, Any], b: int, q: int) -> Dict[str, torch.Tensor]:
    device = end_points["xyz_graspable"].device
    token_idx = end_points.get("token_sel_idx")

    # Graspness used in seed selection.
    if torch.is_tensor(end_points.get("dbg_grasp_sel")):
        graspness = _token_scalar(end_points, b, "dbg_grasp_sel", q, 0.0)
    elif torch.is_tensor(end_points.get("graspness_score")):
        graspness = _token_scalar(end_points, b, "graspness_score", q, 0.0)
    else:
        graspness = torch.zeros(q, device=device)

    objectness_fg = torch.zeros(q, device=device)
    obj = end_points.get("objectness_score")
    if torch.is_tensor(obj) and torch.is_tensor(token_idx):
        prob = F.softmax(obj[b].float(), dim=0)[1].reshape(-1)
        idx = token_idx[b].long().clamp(0, prob.numel() - 1)
        objectness_fg = prob.index_select(0, idx)

    view_score = end_points.get("view_score")
    view_entropy = torch.zeros(q, device=device)
    view_margin = torch.zeros(q, device=device)
    view_selected_score = torch.zeros(q, device=device)
    if torch.is_tensor(view_score):
        cur = view_score[b].float()
        if cur.dim() == 2:
            if cur.shape[0] == q:
                qv = cur
            elif cur.shape[1] == q:
                qv = cur.t().contiguous()
            else:
                qv = None
            if qv is not None:
                vp = F.softmax(qv, dim=-1)
                view_entropy = _entropy(vp, dim=-1) / max(math.log(float(qv.shape[-1])), 1e-6)
                if qv.shape[-1] >= 2:
                    top2 = torch.topk(qv, 2, dim=-1).values
                    view_margin = top2[:, 0] - top2[:, 1]
                inds = end_points["grasp_top_view_inds"][b].long()
                view_selected_score = qv.gather(-1, inds[:, None]).squeeze(-1)

    align = end_points.get("view_ray_align")
    if torch.is_tensor(align) and align.shape[:2] == (end_points["xyz_graspable"].shape[0], q):
        ray_align = align[b].float().abs().clamp(0.0, 1.0)
    else:
        center = F.normalize(end_points["xyz_graspable"][b].float(), dim=-1)
        approach = F.normalize(-end_points["grasp_top_view_xyz"][b].float(), dim=-1)
        ray_align = (center * approach).sum(-1).abs().clamp(0.0, 1.0)

    return {
        "graspness": graspness,
        "objectness_fg": objectness_fg,
        "view_entropy": view_entropy,
        "view_margin": view_margin,
        "view_selected_score": view_selected_score,
        "ray_align": ray_align,
    }


def _first_auto_feature_key(end_points: Mapping[str, Any]) -> Optional[str]:
    preferred = [
        "cva_candidate_features",
        "cva_query_features",
        "grasp_query_features_angle",
        "query_features_angle",
        "kview_query_features",
        "grasp_feature_angle",
    ]
    for key in preferred:
        if torch.is_tensor(end_points.get(key)):
            return key
    tokens = ("cva", "query", "candidate", "grasp_feature")
    for key, value in end_points.items():
        if torch.is_tensor(value) and any(t in key.lower() for t in tokens):
            if value.dim() >= 3:
                return key
    return None


def _canonical_hidden(
    tensor: Optional[torch.Tensor],
    b: int,
    q: int,
    a: int,
) -> Optional[torch.Tensor]:
    """Canonicalize a hidden tensor to [Q,A,C] when possible.

    The hook may expose either a batched tensor or an already batch-sliced
    tensor.  We first recognize unbatched candidate layouts, then strip a
    leading batch dimension only for higher-rank layouts.
    """
    if tensor is None:
        return None
    x = tensor.detach()

    def convert(y: torch.Tensor) -> Optional[torch.Tensor]:
        # [C,Q,A]
        if y.dim() == 3 and y.shape[1:] == (q, a):
            return y.permute(1, 2, 0).contiguous().float()
        # [Q,A,C]
        if y.dim() == 3 and y.shape[:2] == (q, a):
            return y.float()
        # [Q*A,C]
        if y.dim() == 2 and y.shape[0] == q * a:
            return y.reshape(q, a, -1).float()
        # [C,Q*A]
        if y.dim() == 2 and y.shape[1] == q * a:
            return y.t().reshape(q, a, -1).float()
        # Query-only [Q,C] or [C,Q], repeated across angles.
        if y.dim() == 2 and y.shape[0] == q:
            return y[:, None, :].expand(q, a, y.shape[1]).contiguous().float()
        if y.dim() == 2 and y.shape[1] == q:
            return y.t()[:, None, :].expand(q, a, y.shape[0]).contiguous().float()
        return None

    direct = convert(x)
    if direct is not None:
        return direct
    if x.dim() >= 3 and x.shape[0] > b:
        return convert(x[b])
    return None


@dataclass
class CandidateBatch:
    grasps: np.ndarray
    center_local: np.ndarray
    center_global: np.ndarray
    angle: np.ndarray
    depth: np.ndarray
    base_row_by_center: np.ndarray
    pred_depth_by_center_angle: np.ndarray
    pred_score: np.ndarray
    collision_prob: np.ndarray
    train_collision_label: np.ndarray
    features: np.ndarray
    feature_names: List[str]
    hidden: Optional[np.ndarray]


def _build_candidates(
    end_points: Mapping[str, Any],
    batch_i: int,
    hook_tensor: Optional[torch.Tensor],
) -> CandidateBatch:
    center_all = end_points["xyz_graspable"][batch_i].float()
    top_view_all = end_points["grasp_top_view_xyz"][batch_i].float()
    score_logits = end_points["grasp_score_pred_angle"][batch_i].float()  # [6,Q,A]
    depth_logits = end_points["grasp_depth_pred_angle"][batch_i].float()  # [D+1,Q,A]
    width_pred = end_points["grasp_width_pred_angle"][batch_i].float()    # [1,Q,A]
    q, a = score_logits.shape[1], score_logits.shape[2]
    d = min(int(getattr(cfgs, "num_depth", depth_logits.shape[0] - 1)), depth_logits.shape[0] - 1)

    expected, score_prob = _score_expected(score_logits)
    center_score = expected.max(dim=-1).values
    num_centers = min(max(1, int(DIAG.diag_max_centers)), q)
    center_ids = torch.topk(center_score, k=num_centers, largest=True).indices

    top_a = min(max(1, int(DIAG.diag_top_angles)), a)
    angle_ids = torch.topk(expected.index_select(0, center_ids), k=top_a, dim=-1).indices
    top_d = min(max(1, int(DIAG.diag_top_depths)), d)

    depth_prob = F.softmax(depth_logits, dim=0)
    pred_depth_full = depth_logits[:d].argmax(dim=0)  # [Q,A]
    aux = _query_aux(end_points, batch_i, q)

    col_logits = end_points.get("grasp_collision_pred_angle")
    if torch.is_tensor(col_logits):
        col_q_a = col_logits[batch_i].float().squeeze(0)
        col_prob_q_a = torch.sigmoid(col_q_a)
    else:
        col_q_a = torch.zeros((q, a), device=center_all.device)
        col_prob_q_a = torch.full((q, a), 0.5, device=center_all.device)

    train_col = end_points.get("batch_grasp_collision_angle")
    if torch.is_tensor(train_col):
        train_col_q_a = train_col[batch_i].float()
    else:
        train_col_q_a = torch.full((q, a), float("nan"), device=center_all.device)

    hidden_tensor = None
    key = DIAG.diag_probe_feature_key
    if key == "auto":
        auto = _first_auto_feature_key(end_points)
        hidden_tensor = end_points.get(auto) if auto else hook_tensor
    elif key:
        hidden_tensor = end_points.get(key)
        if hidden_tensor is None:
            hidden_tensor = hook_tensor
    else:
        hidden_tensor = hook_tensor
    hidden_qac = _canonical_hidden(hidden_tensor, batch_i, q, a)

    grasp_rows: List[torch.Tensor] = []
    feat_rows: List[torch.Tensor] = []
    hidden_rows: List[torch.Tensor] = []
    local_center: List[int] = []
    global_center: List[int] = []
    angle_rows: List[int] = []
    depth_rows: List[int] = []
    score_rows: List[float] = []
    col_rows: List[float] = []
    train_col_rows: List[float] = []
    base_row_by_center = np.full(num_centers, -1, dtype=np.int64)

    feature_names = (
        [f"score_prob_{i}" for i in range(score_prob.shape[0])]
        + [f"depth_prob_{i}" for i in range(depth_prob.shape[0])]
        + [
            "score_expected",
            "candidate_depth_prob",
            "depth_entropy",
            "angle_entropy",
            "width_m",
            "collision_logit",
            "collision_prob",
            "graspness",
            "objectness_fg",
            "view_entropy",
            "view_margin",
            "view_selected_score",
            "ray_align",
            "center_x",
            "center_y",
            "center_z",
            "view_x",
            "view_y",
            "view_z",
            "angle_sin",
            "angle_cos",
            "depth_norm",
        ]
    )

    for lc, qid_t in enumerate(center_ids):
        qid = int(qid_t.item())
        base_a = int(expected[qid].argmax().item())
        base_d = int(pred_depth_full[qid, base_a].item())
        # Include selected top angles; force base angle in case of ties/truncation.
        aset = [int(x) for x in angle_ids[lc].tolist()]
        if base_a not in aset:
            aset[-1] = base_a
        aset = list(dict.fromkeys(aset))
        for aid in aset:
            if top_d >= d:
                dset = list(range(d))
            else:
                dset = [int(x) for x in torch.topk(depth_prob[:d, qid, aid], k=top_d).indices.tolist()]
                if int(pred_depth_full[qid, aid].item()) not in dset:
                    dset[-1] = int(pred_depth_full[qid, aid].item())
                dset = list(dict.fromkeys(dset))

            angle_rad = float(aid) * math.pi / float(a)
            approach = -top_view_all[qid : qid + 1]
            angle_tensor = torch.tensor([angle_rad], device=center_all.device, dtype=center_all.dtype)
            rot = batch_viewpoint_params_to_matrix(approach, angle_tensor).reshape(9)
            width_m = torch.clamp(1.2 * width_pred[0, qid, aid] / 10.0, 0.0, float(getattr(cfgs, "grasp_max_width", 0.1)))
            angle_distribution = F.softmax(expected[qid], dim=-1)
            angle_ent = _entropy(angle_distribution, dim=-1)
            depth_ent = _entropy(depth_prob[:, qid, aid], dim=0)

            for did in dset:
                score = expected[qid, aid]
                depth_m = (float(did) + 1.0) * 0.01
                row = torch.cat(
                    [
                        score.reshape(1),
                        width_m.reshape(1),
                        torch.tensor([0.02, depth_m], device=center_all.device, dtype=center_all.dtype),
                        rot,
                        center_all[qid],
                        torch.tensor([-1.0], device=center_all.device, dtype=center_all.dtype),
                    ]
                )
                grasp_rows.append(row)
                feat = torch.cat(
                    [
                        score_prob[:, qid, aid],
                        depth_prob[:, qid, aid],
                        torch.stack(
                            [
                                score,
                                depth_prob[did, qid, aid],
                                depth_ent,
                                angle_ent,
                                width_m,
                                col_q_a[qid, aid],
                                col_prob_q_a[qid, aid],
                                aux["graspness"][qid],
                                aux["objectness_fg"][qid],
                                aux["view_entropy"][qid],
                                aux["view_margin"][qid],
                                aux["view_selected_score"][qid],
                                aux["ray_align"][qid],
                                center_all[qid, 0],
                                center_all[qid, 1],
                                center_all[qid, 2],
                                top_view_all[qid, 0],
                                top_view_all[qid, 1],
                                top_view_all[qid, 2],
                                torch.tensor(math.sin(angle_rad), device=center_all.device),
                                torch.tensor(math.cos(angle_rad), device=center_all.device),
                                torch.tensor(float(did) / max(d - 1, 1), device=center_all.device),
                            ]
                        ),
                    ]
                )
                feat_rows.append(feat.float())
                if hidden_qac is not None:
                    hidden_rows.append(hidden_qac[qid, aid].float())
                local_center.append(lc)
                global_center.append(qid)
                angle_rows.append(aid)
                depth_rows.append(did)
                score_rows.append(float(score.item()))
                col_rows.append(float(col_prob_q_a[qid, aid].item()))
                train_col_rows.append(float(train_col_q_a[qid, aid].item()))
                if aid == base_a and did == base_d:
                    base_row_by_center[lc] = len(grasp_rows) - 1

    if np.any(base_row_by_center < 0):
        raise RuntimeError("Candidate beam did not retain every base angle/depth candidate.")

    return CandidateBatch(
        grasps=torch.stack(grasp_rows).detach().cpu().numpy().astype(np.float32),
        center_local=np.asarray(local_center, dtype=np.int64),
        center_global=np.asarray(global_center, dtype=np.int64),
        angle=np.asarray(angle_rows, dtype=np.int64),
        depth=np.asarray(depth_rows, dtype=np.int64),
        base_row_by_center=base_row_by_center,
        pred_depth_by_center_angle=pred_depth_full.index_select(0, center_ids).detach().cpu().numpy(),
        pred_score=np.asarray(score_rows, dtype=np.float32),
        collision_prob=np.asarray(col_rows, dtype=np.float32),
        train_collision_label=np.asarray(train_col_rows, dtype=np.float32),
        features=torch.stack(feat_rows).detach().cpu().numpy().astype(np.float32),
        feature_names=feature_names,
        hidden=(torch.stack(hidden_rows).detach().cpu().numpy().astype(np.float32) if hidden_rows else None),
    )


# -----------------------------------------------------------------------------
# Per-sample diagnostics
# -----------------------------------------------------------------------------


def _friction_utility(friction: np.ndarray) -> np.ndarray:
    mus = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
    f = friction.reshape(-1, 1)
    return ((f > 0) & (f <= mus.reshape(1, -1))).mean(axis=1).astype(np.float32)


def _sample_diagnostics(
    cand: CandidateBatch,
    ev: EvalResult,
    rank_ks: Sequence[int],
    safe_mus: Sequence[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_rows = cand.base_row_by_center
    base_score = cand.pred_score[base_rows]
    order_centers = np.argsort(-base_score, kind="stable")
    center_rows: List[Dict[str, Any]] = []

    # Matrix lookup per center/angle/depth.
    lookup: Dict[Tuple[int, int, int], int] = {
        (int(c), int(a), int(d)): i
        for i, (c, a, d) in enumerate(zip(cand.center_local, cand.angle, cand.depth))
    }

    for rank0, lc in enumerate(order_centers):
        base_row = int(base_rows[lc])
        base_a = int(cand.angle[base_row])
        base_d = int(cand.depth[base_row])
        base_invalid = bool(ev.friction[base_row] <= 0.0)
        row: Dict[str, Any] = {
            "center_local": int(lc),
            "center_global": int(cand.center_global[base_row]),
            "base_rank": int(rank0 + 1),
            "base_score": float(cand.pred_score[base_row]),
            "base_angle": base_a,
            "base_depth": base_d,
            "base_friction": float(ev.friction[base_row]),
            "base_collision_or_empty": int(ev.collision_or_empty[base_row]),
            "base_empty": int(ev.empty[base_row]),
            "base_pure_collision": int(ev.pure_collision[base_row]),
            "base_fc_fail": int(base_invalid and not ev.collision_or_empty[base_row]),
        }
        for mu in safe_mus:
            safe = (ev.friction > 0.0) & (ev.friction <= float(mu))
            # Alternative angle uses each angle's *predicted* depth.
            angle_ok = False
            for aid in np.unique(cand.angle[cand.center_local == lc]):
                if int(aid) == base_a:
                    continue
                pd = int(cand.pred_depth_by_center_angle[lc, int(aid)])
                idx = lookup.get((int(lc), int(aid), pd))
                if idx is not None and safe[idx]:
                    angle_ok = True
                    break
            depth_ok = any(
                safe[idx]
                for (c, a, d), idx in lookup.items()
                if c == int(lc) and a == base_a and d != base_d
            )
            any_ok = any(
                safe[idx]
                for (c, a, d), idx in lookup.items()
                if c == int(lc) and not (a == base_a and d == base_d)
            )
            row[f"recover_angle_mu{mu:g}"] = int(base_invalid and angle_ok)
            row[f"recover_depth_mu{mu:g}"] = int(base_invalid and depth_ok)
            row[f"recover_any_mu{mu:g}"] = int(base_invalid and any_ok)
        center_rows.append(row)

    # Mixed-depth states are computed per evaluated center-angle pair.
    mixed_rows: List[Dict[str, Any]] = []
    for lc in np.unique(cand.center_local):
        for aid in np.unique(cand.angle[cand.center_local == lc]):
            ids = np.flatnonzero((cand.center_local == lc) & (cand.angle == aid))
            if len(ids) < 2:
                continue
            coll = ev.pure_collision[ids]
            coe = ev.collision_or_empty[ids]
            emp = ev.empty[ids]
            valid = ev.friction[ids] > 0.0
            mixed_rows.append(
                {
                    "center_local": int(lc),
                    "center_global": int(cand.center_global[ids[0]]),
                    "angle": int(aid),
                    "num_depths": int(len(ids)),
                    "mixed_pure_collision": int(coll.any() and (~coll).any()),
                    "mixed_collision_or_empty": int(coe.any() and (~coe).any()),
                    "mixed_empty": int(emp.any() and (~emp).any()),
                    "mixed_validity": int(valid.any() and (~valid).any()),
                    "any_safe04": int(np.any((ev.friction[ids] > 0) & (ev.friction[ids] <= 0.4))),
                    "any_safe08": int(np.any((ev.friction[ids] > 0) & (ev.friction[ids] <= 0.8))),
                }
            )

    summary: Dict[str, Any] = {
        "num_candidates": int(len(cand.grasps)),
        "num_centers": int(len(base_rows)),
        "candidate_collision_or_empty_ratio": float(ev.collision_or_empty.mean()),
        "candidate_empty_ratio": float(ev.empty.mean()),
        "candidate_pure_collision_ratio": float(ev.pure_collision.mean()),
        "candidate_valid_ratio": float(np.mean(ev.friction > 0.0)),
        "candidate_safe04_ratio": float(np.mean((ev.friction > 0) & (ev.friction <= 0.4))),
        "candidate_safe08_ratio": float(np.mean((ev.friction > 0) & (ev.friction <= 0.8))),
    }
    for k in rank_ks:
        subset = [r for r in center_rows if r["base_rank"] <= k]
        invalid = [r for r in subset if r["base_friction"] <= 0.0]
        summary[f"base_top{k}_invalid_ratio"] = float(len(invalid) / max(len(subset), 1))
        summary[f"base_top{k}_collision_or_empty_ratio"] = float(
            np.mean([r["base_collision_or_empty"] for r in subset]) if subset else float("nan")
        )
        for mu in safe_mus:
            denom = max(len(invalid), 1)
            for typ in ["angle", "depth", "any"]:
                summary[f"top{k}_invalid_recover_{typ}_mu{mu:g}"] = float(
                    sum(r[f"recover_{typ}_mu{mu:g}"] for r in invalid) / denom
                )
    if mixed_rows:
        for key in ["mixed_pure_collision", "mixed_collision_or_empty", "mixed_empty", "mixed_validity"]:
            summary[f"{key}_ratio"] = float(np.mean([r[key] for r in mixed_rows]))
    return summary, center_rows, mixed_rows


# -----------------------------------------------------------------------------
# Probe fitting
# -----------------------------------------------------------------------------


def _fit_probes(
    X: np.ndarray,
    hidden: Optional[np.ndarray],
    feature_names: Sequence[str],
    targets: Mapping[str, np.ndarray],
    utility: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    out_dir: str,
) -> Dict[str, Any]:
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(DIAG.diag_seed)
    n = len(X)
    max_rows = max(1000, int(DIAG.diag_probe_max_rows))
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        X = X[idx]
        hidden = hidden[idx] if hidden is not None else None
        targets = {k: v[idx] for k, v in targets.items()}
        utility = utility[idx]
        groups = groups[idx]
        sample_ids = sample_ids[idx]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(DIAG.diag_probe_val_fraction),
        random_state=int(DIAG.diag_seed),
    )
    train_idx, val_idx = next(splitter.split(X, groups=groups))
    feature_sets = {"output": (X, list(feature_names))}
    if hidden is not None and hidden.ndim == 2 and hidden.shape[0] == X.shape[0]:
        feature_sets["output_plus_hidden"] = (
            np.concatenate([X, hidden], axis=1),
            list(feature_names) + [f"hidden_{i}" for i in range(hidden.shape[1])],
        )

    results: Dict[str, Any] = {
        "num_rows": int(len(X)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "train_groups": sorted(set(int(x) for x in groups[train_idx])),
        "val_groups": sorted(set(int(x) for x in groups[val_idx])),
        "feature_sets": {},
    }

    for fs_name, (features, names) in feature_sets.items():
        fs_result: Dict[str, Any] = {}
        for task, y in targets.items():
            y = y.astype(np.int64)
            if np.unique(y[train_idx]).size < 2 or np.unique(y[val_idx]).size < 2:
                fs_result[task] = {"error": "Task has one class in train or validation split."}
                continue
            model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=800,
                            solver="lbfgs",
                            random_state=int(DIAG.diag_seed),
                        ),
                    ),
                ]
            )
            model.fit(features[train_idx], y[train_idx])
            score = model.predict_proba(features[val_idx])[:, 1]
            metrics = _binary_metrics(y[val_idx], score)
            # Ranking score should always mean "better grasp" for this section.
            ranking_score = 1.0 - score if task == "collision" else score
            metrics.update(_ndcg_and_topk(utility[val_idx], ranking_score, sample_ids[val_idx]))
            metrics["utility_spearman"] = _spearman(utility[val_idx], ranking_score)
            fs_result[task] = metrics
            joblib.dump(model, os.path.join(out_dir, f"probe_{fs_name}_{task}.joblib"))

            coef = model.named_steps["clf"].coef_.reshape(-1)
            top = np.argsort(-np.abs(coef))[: min(50, len(coef))]
            coef_rows = [
                {"feature": names[i], "coefficient": float(coef[i]), "abs_coefficient": float(abs(coef[i]))}
                for i in top
            ]
            _save_csv(coef_rows, os.path.join(out_dir, f"probe_{fs_name}_{task}_coefficients.csv"))
        results["feature_sets"][fs_name] = fs_result
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    random.seed(DIAG.diag_seed)
    np.random.seed(DIAG.diag_seed)
    torch.manual_seed(DIAG.diag_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DIAG.diag_seed)

    out_dir = DIAG.diag_output_dir or os.path.join(cfgs.save_dir, f"candidate_validity_{cfgs.test_mode}")
    os.makedirs(out_dir, exist_ok=True)
    safe_mus = _csv_floats(DIAG.diag_safe_mus)
    rank_ks = _csv_ints(DIAG.diag_rank_ks)

    dataset = _build_dataset()
    indices, scene_list = _select_indices(dataset)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(cfgs.batch_size),
        shuffle=False,
        num_workers=int(getattr(cfgs, "num_workers", 2)),
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = _load_model(device)

    modules_path = os.path.join(out_dir, "model_modules.txt")
    with open(modules_path, "w") as f:
        for name, module in net.named_modules():
            f.write(f"{name}\t{module.__class__.__name__}\n")

    hook = _HookCapture(net, DIAG.diag_probe_hook_module, DIAG.diag_probe_hook_source)
    raw_eval = RawCandidateEvaluator(
        cfgs.dataset_root,
        cfgs.camera,
        str(cfgs.test_mode),
        DIAG.diag_eval_chunk,
        bool(DIAG.diag_skip_force_closure),
    )

    sample_rows: List[Dict[str, Any]] = []
    center_rows_all: List[Dict[str, Any]] = []
    mixed_rows_all: List[Dict[str, Any]] = []
    candidate_rows_all: List[Dict[str, Any]] = []
    X_all: List[np.ndarray] = []
    H_all: List[np.ndarray] = []
    target_all: Dict[str, List[np.ndarray]] = defaultdict(list)
    utility_all: List[np.ndarray] = []
    group_all: List[np.ndarray] = []
    sample_id_all: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    hidden_consistent = True
    endpoint_inventory_saved = False

    start_time = time.time()
    global_sample = 0
    try:
        for batch_idx, batch in enumerate(loader):
            _move_batch_to_device(batch, device)
            with torch.no_grad():
                end_points = net(dict(batch))

            if not endpoint_inventory_saved:
                inventory = {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                    for k, v in end_points.items()
                    if torch.is_tensor(v)
                }
                _save_json(inventory, os.path.join(out_dir, "endpoint_tensor_inventory.json"))
                endpoint_inventory_saved = True

            bs = end_points["xyz_graspable"].shape[0]
            for b in range(bs):
                subset_pos = batch_idx * int(cfgs.batch_size) + b
                if subset_pos >= len(indices):
                    continue
                data_idx = indices[subset_pos]
                scene_name = str(scene_list[data_idx])
                scene_id = int(scene_name.split("_")[-1])
                anno_id = int(data_idx % 256)

                cand = _build_candidates(end_points, b, hook.tensor)
                ev = raw_eval.evaluate(scene_id, anno_id, cand.grasps)
                summary, centers, mixed = _sample_diagnostics(cand, ev, rank_ks, safe_mus)
                sample_key = f"{scene_id:04d}_{anno_id:04d}"
                row = {
                    "split": str(cfgs.test_mode),
                    "scene_id": scene_id,
                    "anno_id": anno_id,
                    "sample_key": sample_key,
                    **summary,
                }
                sample_rows.append(row)
                for r in centers:
                    center_rows_all.append({"split": str(cfgs.test_mode), "scene_id": scene_id, "anno_id": anno_id, **r})
                for r in mixed:
                    mixed_rows_all.append({"split": str(cfgs.test_mode), "scene_id": scene_id, "anno_id": anno_id, **r})

                utility = _friction_utility(ev.friction)
                X_all.append(cand.features)
                if cand.hidden is None:
                    hidden_consistent = False
                else:
                    H_all.append(cand.hidden)
                target_all["collision"].append(ev.collision_or_empty.astype(np.int64))
                target_all["valid"].append((ev.friction > 0.0).astype(np.int64))
                target_all["safe08"].append(((ev.friction > 0.0) & (ev.friction <= 0.8)).astype(np.int64))
                utility_all.append(utility)
                group_all.append(np.full(len(cand.grasps), scene_id, dtype=np.int64))
                sample_id_all.append(np.full(len(cand.grasps), global_sample, dtype=np.int64))
                feature_names = cand.feature_names

                if DIAG.diag_save_candidate_rows:
                    base_row_set = set(cand.base_row_by_center.tolist())
                    for j in range(len(cand.grasps)):
                        predicted_depth_for_angle = int(
                            cand.depth[j]
                            == cand.pred_depth_by_center_angle[cand.center_local[j], cand.angle[j]]
                        )
                        candidate_rows_all.append(
                            {
                                "split": str(cfgs.test_mode),
                                "scene_id": scene_id,
                                "anno_id": anno_id,
                                "center_local": int(cand.center_local[j]),
                                "center_global": int(cand.center_global[j]),
                                "angle": int(cand.angle[j]),
                                "depth": int(cand.depth[j]),
                                "is_base_candidate": int(j in base_row_set),
                                "is_predicted_depth_for_angle": predicted_depth_for_angle,
                                "pred_score": float(cand.pred_score[j]),
                                "collision_prob": float(cand.collision_prob[j]),
                                "train_collision_label": float(cand.train_collision_label[j]),
                                "assigned_obj": int(ev.assigned_obj[j]),
                                "official_collision_or_empty": int(ev.collision_or_empty[j]),
                                "official_empty": int(ev.empty[j]),
                                "official_pure_collision": int(ev.pure_collision[j]),
                                "official_fc_fail": int(ev.friction[j] <= 0 and not ev.collision_or_empty[j]),
                                "official_friction": float(ev.friction[j]),
                                "official_utility": float(utility[j]),
                            }
                        )

                global_sample += 1
                print(
                    f"[{global_sample}/{len(indices)}] scene={scene_id:04d} anno={anno_id:04d} "
                    f"cand={len(cand.grasps)} valid={summary['candidate_valid_ratio']:.3f} "
                    f"mixed_depth={summary.get('mixed_collision_or_empty_ratio', float('nan')):.3f} "
                    f"elapsed={(time.time()-start_time)/60:.1f}m"
                )

                # Crash-safe progressive outputs.
                _save_csv(sample_rows, os.path.join(out_dir, "sample_diagnostics.partial.csv"))
                _save_csv(center_rows_all, os.path.join(out_dir, "center_recoverability.partial.csv"))
                _save_csv(mixed_rows_all, os.path.join(out_dir, "mixed_depth.partial.csv"))

    finally:
        hook.close()

    _save_csv(sample_rows, os.path.join(out_dir, "sample_diagnostics.csv"))
    _save_csv(center_rows_all, os.path.join(out_dir, "center_recoverability.csv"))
    _save_csv(mixed_rows_all, os.path.join(out_dir, "mixed_depth.csv"))
    if candidate_rows_all:
        _save_csv(candidate_rows_all, os.path.join(out_dir, "candidate_outcomes.csv.gz"), gzip_output=True)

    X = np.concatenate(X_all, axis=0)
    utility = np.concatenate(utility_all)
    groups = np.concatenate(group_all)
    sample_ids = np.concatenate(sample_id_all)
    targets = {k: np.concatenate(v) for k, v in target_all.items()}
    hidden = np.concatenate(H_all, axis=0) if hidden_consistent and H_all else None

    np.savez_compressed(
        os.path.join(out_dir, "probe_dataset.npz"),
        X=X,
        hidden=(hidden if hidden is not None else np.empty((len(X), 0), dtype=np.float32)),
        utility=utility,
        groups=groups,
        sample_ids=sample_ids,
        **targets,
    )
    _save_json({"feature_names": feature_names or [], "hidden_dim": 0 if hidden is None else hidden.shape[1]}, os.path.join(out_dir, "probe_features.json"))

    # Collision head / label agreement.  The angle-level head is compared most
    # directly at each angle's predicted depth; all-depth results are also kept
    # to reveal granularity mismatch.
    candidate_data = candidate_rows_all
    agreement: Dict[str, Any] = {}
    if candidate_data:
        arr_p = np.asarray([r["collision_prob"] for r in candidate_data], dtype=np.float32)
        arr_off = np.asarray([r["official_collision_or_empty"] for r in candidate_data], dtype=np.int64)
        arr_empty = np.asarray([r["official_empty"] for r in candidate_data], dtype=np.int64)
        arr_fc = np.asarray([r["official_fc_fail"] for r in candidate_data], dtype=np.int64)
        pred_depth_mask = np.asarray([r["is_predicted_depth_for_angle"] for r in candidate_data], dtype=bool)
        base_mask = np.asarray([r["is_base_candidate"] for r in candidate_data], dtype=bool)
        agreement["official_failure_composition_all"] = {
            "collision_or_empty_ratio": float(arr_off.mean()),
            "empty_ratio": float(arr_empty.mean()),
            "force_closure_fail_ratio": float(arr_fc.mean()),
        }
        agreement["collision_head_vs_official_all_depths"] = _binary_metrics(arr_off, arr_p)
        if pred_depth_mask.any():
            agreement["collision_head_vs_official_predicted_depth"] = _binary_metrics(
                arr_off[pred_depth_mask], arr_p[pred_depth_mask]
            )
        if base_mask.any():
            agreement["collision_head_vs_official_base_candidate"] = _binary_metrics(
                arr_off[base_mask], arr_p[base_mask]
            )

        train_label = np.asarray([r["train_collision_label"] for r in candidate_data], dtype=np.float32)
        finite = np.isfinite(train_label) & pred_depth_mask
        if finite.any():
            agreement["training_label_vs_official_predicted_depth"] = _binary_metrics(
                arr_off[finite], train_label[finite]
            )
            agreement["collision_head_vs_training_label"] = _binary_metrics(
                train_label[finite] > 0.5, arr_p[finite]
            )
    _save_json(agreement, os.path.join(out_dir, "collision_agreement.json"))

    probe_result: Dict[str, Any] = {"disabled": True}
    if DIAG.diag_probe:
        probe_result = _fit_probes(
            X,
            hidden,
            feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            targets,
            utility,
            groups,
            sample_ids,
            out_dir,
        )
        _save_json(probe_result, os.path.join(out_dir, "probe_metrics.json"))

    global_summary: Dict[str, Any] = {
        "split": str(cfgs.test_mode),
        "num_samples": len(sample_rows),
        "num_candidates": int(len(X)),
        "hidden_probe_available": hidden is not None,
        "elapsed_minutes": float((time.time() - start_time) / 60.0),
        "sample_metric_means": {},
        "collision_agreement": agreement,
        "probe": probe_result,
    }
    if sample_rows:
        numeric_keys = [
            k for k in sample_rows[0]
            if k not in {"split", "scene_id", "anno_id", "sample_key"}
        ]
        for key in numeric_keys:
            vals = [float(r[key]) for r in sample_rows if key in r and np.isfinite(float(r[key]))]
            if vals:
                global_summary["sample_metric_means"][key] = float(np.mean(vals))
    _save_json(global_summary, os.path.join(out_dir, "summary.json"))

    print(f"\n[DONE] Outputs: {out_dir}")
    print("Key files:")
    print("  summary.json")
    print("  sample_diagnostics.csv")
    print("  center_recoverability.csv")
    print("  mixed_depth.csv")
    print("  collision_agreement.json")
    print("  probe_metrics.json")
    if hidden is None:
        print(
            "[NOTE] No hidden CVA feature was found. The output-feature probe still ran. "
            "Inspect model_modules.txt / endpoint_tensor_inventory.json and rerun with "
            "--diag_probe_hook_module or --diag_probe_feature_key for a hidden-feature probe."
        )


if __name__ == "__main__":
    main()
