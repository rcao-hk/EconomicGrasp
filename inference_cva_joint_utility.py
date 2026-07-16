#!/usr/bin/env python3
"""Grasp inference with a frozen evaluator-aligned CVA joint utility scorer.

This script targets the first-generation center-view-angle CVA Transformer
(`kview_k=1`).  It expands the model's top-K views, constructs explicit
center-view-angle-depth candidates, scores them with a trained
EvaluatorAlignedJointUtilityScorer, retains a fixed number of candidates per
center, applies the repository's model-free collision filter, and saves
GraspNet-compatible `.npy` dumps.

The scorer is used before view/angle/depth collapse.  No GT labels or official
benchmark evaluator are used during inference.

Required project modifications
------------------------------
1. `models/kview_query_transformer.py` must expose
   `end_points['cva_angle_feature']` with shape [B,Q,A,C].
2. The runtime view-override + score-pinning fix must be applied to
   `models/economicgrasp_bip3d.py`, so forced top-K view passes are respected.
3. `models/cva_joint_utility.py` must be installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def _consume_joint_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ju_scorer_checkpoint", required=True)
    p.add_argument("--ju_variant_name", default="joint_utility")
    p.add_argument("--ju_top_views", type=int, default=4)
    p.add_argument("--ju_top_angles", type=int, default=4)
    p.add_argument("--ju_top_depths", type=int, default=4)
    p.add_argument("--ju_per_center_budget", type=int, default=4)
    p.add_argument("--ju_scorer_chunk", type=int, default=65536)
    p.add_argument("--ju_collision_chunk", type=int, default=1024)
    p.add_argument("--ju_amp", type=int, default=1)
    p.add_argument("--ju_strict", type=int, default=1)
    p.add_argument("--ju_seed", type=int, default=0)
    p.add_argument("--ju_override_key", default="oracle_view_inds_override")
    p.add_argument("--ju_save_candidate_meta", type=int, default=0)
    p.add_argument("--ju_save_nocollision", type=int, default=0)
    p.add_argument("--ju_min_utility", type=float, default=-1.0)
    p.add_argument("--ju_max_samples", type=int, default=-1)
    custom, rest = p.parse_known_args(sys.argv[1:])
    sys.argv[:] = [sys.argv[0], *rest]
    return custom


JU = _consume_joint_args()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from graspnetAPI import GraspGroup
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn
from utils.label_generation import batch_viewpoint_params_to_matrix
from models.economicgrasp_bip3d import economicgrasp_dpt
from models.cva_joint_utility import load_joint_utility_scorer


STRICT = bool(JU.ju_strict)
SAVE_META = bool(JU.ju_save_candidate_meta)
SAVE_RAW = bool(JU.ju_save_nocollision)


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _save_json(payload: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(payload)), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _save_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted(set().union(*(set(r.keys()) for r in rows)))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


def _build_dataset(args):
    cls = GraspNetMultiDataset if args.multi_modal else GraspNetDataset
    return cls(
        args.dataset_root,
        split=str(args.test_mode),
        camera=args.camera,
        num_points=args.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )


def _build_subset(dataset, sample_interval: float, annos_per_scene: int = 256):
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be >0, got {sample_interval}")
    total = len(dataset)
    if sample_interval >= 1.0:
        ids = list(range(total))
    else:
        stride = max(1, int(round(1.0 / sample_interval)))
        ids: List[int] = []
        for start in range(0, total, annos_per_scene):
            end = min(start + annos_per_scene, total)
            ids.extend(start + i for i in range(0, end - start, stride))
    if int(JU.ju_max_samples) > 0:
        ids = ids[: int(JU.ju_max_samples)]
    return Subset(dataset, ids), ids


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
            batch[key] = value.to(device, non_blocking=True)


def _capture_rng() -> Dict[str, Any]:
    state: Dict[str, Any] = {"cpu": torch.random.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _copy_inputs(batch: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(batch)


def _canonical_view_score(end_points: Mapping[str, Any]) -> torch.Tensor:
    score = end_points["view_score"]
    q = int(end_points["xyz_graspable"].shape[1])
    if score.dim() != 3:
        raise ValueError(f"view_score must be 3D, got {tuple(score.shape)}")
    if score.shape[1] == q:
        return score
    if score.shape[2] == q:
        return score.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot canonicalize view_score={tuple(score.shape)} to [B,Q,V], Q={q}")


def _expected_quality(score_logits: torch.Tensor) -> torch.Tensor:
    """Convert [S,Q,A] legacy score logits to [Q,A] expected quality."""
    bins = torch.linspace(
        0.0, 1.0, score_logits.shape[0],
        device=score_logits.device, dtype=score_logits.dtype,
    )
    return (F.softmax(score_logits.float(), dim=0) * bins[:, None, None]).sum(dim=0)


def _assert_same_centers(base: Mapping[str, Any], other: Mapping[str, Any], tag: str) -> None:
    failures: List[str] = []
    if torch.is_tensor(base.get("token_sel_idx")) and torch.is_tensor(other.get("token_sel_idx")):
        if not torch.equal(base["token_sel_idx"], other["token_sel_idx"]):
            failures.append("token_sel_idx")
    if not torch.allclose(base["xyz_graspable"], other["xyz_graspable"], atol=1e-7, rtol=0.0):
        failures.append("xyz_graspable")
    if failures:
        raise RuntimeError(f"Forced-view pass {tag} changed center/query selection: {failures}")


def _run_forced_view(
    net: torch.nn.Module,
    batch: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    forced_view: torch.Tensor,
    base: Mapping[str, Any],
    tag: str,
) -> Dict[str, Any]:
    _restore_rng(rng_state)
    inp = _copy_inputs(batch)
    inp[str(JU.ju_override_key)] = forced_view
    out = net(inp)
    _assert_same_centers(base, out, tag)
    used = out.get("grasp_top_view_inds")
    if not torch.is_tensor(used):
        raise KeyError("Forced pass did not expose grasp_top_view_inds")
    ratio = float(used.long().eq(forced_view.long()).float().mean().item())
    if ratio < 0.999999:
        msg = (
            f"Forced view override not respected for {tag}: ratio={ratio:.6f}. "
            "Apply the runtime override + view-score pinning fix."
        )
        if STRICT:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}", flush=True)
    return out


def _top_view_indices(base: Mapping[str, Any], k: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    score = _canonical_view_score(base).float()
    if not torch.isfinite(score).all():
        raise RuntimeError("view_score contains non-finite values")
    base_used = base["grasp_top_view_inds"].long()
    k = max(1, min(int(k), int(score.shape[-1])))

    row_max = score.amax(dim=-1)
    row_min = score.amin(dim=-1)
    base_score = score.gather(-1, base_used.unsqueeze(-1)).squeeze(-1)
    span = (row_max - row_min).abs()
    tol = torch.maximum(torch.full_like(span, 1e-6), span * 1e-6)
    score_equiv = (row_max - base_score).abs() <= tol
    true_bad = ~score_equiv
    if bool(true_bad.any()):
        max_gap = float((row_max - base_score)[true_bad].max().item())
        msg = f"Base view is not score-equivalent to row maximum: count={int(true_bad.sum())}, max_gap={max_gap:.6g}"
        if STRICT:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}", flush=True)

    if k == 1:
        result = base_used.unsqueeze(-1)
    else:
        extra_score = score.clone()
        extra_score.scatter_(-1, base_used.unsqueeze(-1), float("-inf"))
        extra = torch.topk(extra_score, k=k - 1, dim=-1, largest=True, sorted=True).indices
        result = torch.cat([base_used.unsqueeze(-1), extra], dim=-1)
    info = {
        "base_argmax_index_agreement": float(base_used.eq(torch.argmax(score, dim=-1)).float().mean().item()),
        "base_score_equivalent_ratio": float(score_equiv.float().mean().item()),
    }
    return result, info


def _gather_q_angle(x: torch.Tensor, angle_idx: torch.Tensor) -> torch.Tensor:
    """Gather x [Q,A,C] with angle_idx [Q,Ta] -> [Q,Ta,C]."""
    q = x.shape[0]
    qid = torch.arange(q, device=x.device)[:, None].expand_as(angle_idx)
    return x[qid, angle_idx]


def _make_scorer_candidates(
    end_points: Mapping[str, Any],
    base_view_score: torch.Tensor,
    b: int,
    view_rank: int,
    scorer,
) -> Dict[str, torch.Tensor]:
    hidden_all = end_points.get("cva_angle_feature")
    if not torch.is_tensor(hidden_all):
        raise KeyError(
            "Missing end_points['cva_angle_feature']; apply expose_cva_angle_feature.patch."
        )
    hidden = hidden_all[b].float()                              # [Q,A,C]
    score_raw = end_points["grasp_score_pred_angle"][b].float()  # [S,Q,A]
    depth_raw = end_points["grasp_depth_pred_angle"][b].float()  # [D+1,Q,A]
    width = end_points["grasp_width_pred_angle"][b, 0].float()   # [Q,A]
    centers = end_points["xyz_graspable"][b].float()             # [Q,3]
    view_xyz = end_points["grasp_top_view_xyz"][b].float()       # [Q,3]
    view_id = end_points["grasp_top_view_inds"][b].long()        # [Q]
    collision_all = end_points.get("grasp_collision_pred_angle")
    if torch.is_tensor(collision_all):
        legacy_collision = collision_all[b, 0].float()
    else:
        legacy_collision = torch.zeros_like(width)
        if scorer.config.use_legacy_collision:
            raise KeyError(
                "The scorer checkpoint expects legacy collision logits, but the base model "
                "did not emit grasp_collision_pred_angle. Use the collision checkpoint and "
                "pass --kview_use_collision."
            )

    q, a, feat_dim = hidden.shape
    score_dim = score_raw.shape[0]
    d_total = depth_raw.shape[0]
    d_phys = min(int(getattr(cfgs, "num_depth", d_total - 1)), d_total - 1)
    if d_phys != int(scorer.config.num_depths):
        raise ValueError(f"Physical depth count={d_phys}, scorer expects {scorer.config.num_depths}")
    if a != int(scorer.config.num_angles):
        raise ValueError(f"Angle count={a}, scorer expects {scorer.config.num_angles}")
    if feat_dim != int(scorer.config.angle_feature_dim):
        raise ValueError(f"CVA feature dim={feat_dim}, scorer expects {scorer.config.angle_feature_dim}")

    ta = max(1, min(int(JU.ju_top_angles), a))
    td = max(1, min(int(JU.ju_top_depths), d_phys))
    quality = _expected_quality(score_raw)  # [Q,A]
    angle_idx = torch.topk(quality, k=ta, dim=1, largest=True, sorted=True).indices

    hidden_sel = _gather_q_angle(hidden, angle_idx)  # [Q,Ta,C]
    score_qas = score_raw.permute(1, 2, 0).contiguous()
    score_sel = _gather_q_angle(score_qas, angle_idx)  # [Q,Ta,S]
    depth_qad = depth_raw.permute(1, 2, 0).contiguous()
    depth_sel = _gather_q_angle(depth_qad, angle_idx)  # [Q,Ta,D+1]
    width_sel = torch.gather(width, 1, angle_idx)
    coll_sel = torch.gather(legacy_collision, 1, angle_idx)
    quality_sel = torch.gather(quality, 1, angle_idx)

    depth_prob = F.softmax(depth_sel, dim=-1)[..., :d_phys]
    depth_prob_sel, depth_idx = torch.topk(
        depth_prob, k=td, dim=-1, largest=True, sorted=True
    )  # [Q,Ta,Td]

    used_view_score = base_view_score[b].gather(1, view_id[:, None]).squeeze(1)
    leading = (q, ta, td)
    q_idx = torch.arange(q, device=hidden.device)[:, None, None].expand(*leading)
    angle_3 = angle_idx[:, :, None].expand(*leading)

    scorer_batch: Dict[str, torch.Tensor] = {
        "angle_feature": hidden_sel[:, :, None, :].expand(q, ta, td, feat_dim).reshape(-1, feat_dim),
        "score_logits": score_sel[:, :, None, :].expand(q, ta, td, score_dim).reshape(-1, score_dim),
        "depth_logits": depth_sel[:, :, None, :].expand(q, ta, td, d_total).reshape(-1, d_total),
        "width_raw": width_sel[:, :, None].expand(*leading).reshape(-1),
        "legacy_collision_logit": coll_sel[:, :, None].expand(*leading).reshape(-1),
        "angle_id": angle_3.reshape(-1),
        "depth_id": depth_idx.reshape(-1),
        "view_rank": torch.full((q * ta * td,), int(view_rank), device=hidden.device, dtype=torch.long),
        "view_score": used_view_score[:, None, None].expand(*leading).reshape(-1),
        "center_xyz": centers[:, None, None, :].expand(q, ta, td, 3).reshape(-1, 3),
        "view_xyz": view_xyz[:, None, None, :].expand(q, ta, td, 3).reshape(-1, 3),
    }
    return {
        "scorer_batch": scorer_batch,
        "q_idx": q_idx,
        "angle_id": angle_3,
        "depth_id": depth_idx,
        "view_id": view_id[:, None, None].expand(*leading),
        "view_xyz": view_xyz[:, None, None, :].expand(q, ta, td, 3),
        "center_xyz": centers,
        "width_raw": width_sel[:, :, None].expand(*leading),
        "legacy_quality": quality_sel[:, :, None].expand(*leading),
        "legacy_depth_prob": depth_prob_sel,
        "shape": torch.tensor([q, ta, td], device=hidden.device, dtype=torch.long),
    }


def _score_in_chunks(scorer, batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    n = int(batch["angle_id"].numel())
    chunk = max(1, int(JU.ju_scorer_chunk))
    parts: Dict[str, List[torch.Tensor]] = {
        "utility": [], "collision_logit": [], "empty_logit": [], "cdf_prob": [],
    }
    amp_enabled = bool(JU.ju_amp) and device.type == "cuda"
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cur = {k: v[start:end] for k, v in batch.items()}
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = scorer.forward_flat(cur)
        for key in parts:
            parts[key].append(out[key].float())
    return {key: torch.cat(value, dim=0) for key, value in parts.items()}


def _attach_scores(part: Dict[str, torch.Tensor], scorer, device: torch.device) -> Dict[str, torch.Tensor]:
    shape = tuple(int(x) for x in part.pop("shape").tolist())
    scored = _score_in_chunks(scorer, part.pop("scorer_batch"), device)
    part["utility"] = scored["utility"].reshape(*shape)
    part["pred_collision_prob"] = torch.sigmoid(scored["collision_logit"]).reshape(*shape)
    part["pred_empty_prob"] = torch.sigmoid(scored["empty_logit"]).reshape(*shape)
    part["cdf_prob"] = scored["cdf_prob"].reshape(*shape, scored["cdf_prob"].shape[-1])
    return part


def _merge_parts(parts: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not parts:
        raise ValueError("No candidate view parts")
    q = int(parts[0]["utility"].shape[0])
    merged: Dict[str, torch.Tensor] = {"center_xyz": parts[0]["center_xyz"]}
    scalar_keys = [
        "utility", "pred_collision_prob", "pred_empty_prob", "q_idx",
        "angle_id", "depth_id", "view_id", "width_raw", "legacy_quality",
        "legacy_depth_prob",
    ]
    for key in scalar_keys:
        merged[key] = torch.cat([p[key].reshape(q, -1) for p in parts], dim=1)
    merged["view_xyz"] = torch.cat([p["view_xyz"].reshape(q, -1, 3) for p in parts], dim=1)
    merged["cdf_prob"] = torch.cat([p["cdf_prob"].reshape(q, -1, p["cdf_prob"].shape[-1]) for p in parts], dim=1)
    return merged


def _select_per_center(merged: Mapping[str, torch.Tensor], budget: int) -> Dict[str, torch.Tensor]:
    utility = merged["utility"]
    q, n = utility.shape
    k = max(1, min(int(budget), n))
    keep = torch.topk(utility, k=k, dim=1, largest=True, sorted=True).indices
    out: Dict[str, torch.Tensor] = {"center_xyz": merged["center_xyz"]}
    for key in [
        "utility", "pred_collision_prob", "pred_empty_prob", "q_idx", "angle_id",
        "depth_id", "view_id", "width_raw", "legacy_quality", "legacy_depth_prob",
    ]:
        out[key] = torch.gather(merged[key], 1, keep)
    out["view_xyz"] = torch.gather(
        merged["view_xyz"], 1, keep[:, :, None].expand(q, k, 3)
    )
    out["cdf_prob"] = torch.gather(
        merged["cdf_prob"], 1,
        keep[:, :, None].expand(q, k, merged["cdf_prob"].shape[-1]),
    )
    return out


def _decode(selected: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    q_idx = selected["q_idx"].reshape(-1).long()
    angle_id = selected["angle_id"].reshape(-1).long()
    depth_id = selected["depth_id"].reshape(-1).long()
    utility = selected["utility"].reshape(-1).float()
    width_raw = selected["width_raw"].reshape(-1).float()
    view_xyz = selected["view_xyz"].reshape(-1, 3).float()
    centers = selected["center_xyz"].index_select(0, q_idx).float()

    angle = angle_id.float() * (math.pi / float(getattr(cfgs, "num_angle", 12)))
    rotation = batch_viewpoint_params_to_matrix(-view_xyz, angle).reshape(-1, 9)
    width_m = torch.clamp(
        1.2 * width_raw / 10.0,
        min=0.0,
        max=float(getattr(cfgs, "grasp_max_width", 0.1)),
    )
    depth_m = (depth_id.float() + 1.0) * 0.01
    height = torch.full_like(utility, 0.02)
    obj_id = torch.full_like(utility, -1.0)
    grasp = torch.cat(
        [
            utility[:, None], width_m[:, None], height[:, None], depth_m[:, None],
            rotation, centers, obj_id[:, None],
        ],
        dim=1,
    )
    meta = {
        "q_idx": q_idx,
        "view_id": selected["view_id"].reshape(-1).long(),
        "angle_id": angle_id,
        "depth_id": depth_id,
        "utility": utility,
        "pred_collision_prob": selected["pred_collision_prob"].reshape(-1).float(),
        "pred_empty_prob": selected["pred_empty_prob"].reshape(-1).float(),
        "legacy_quality": selected["legacy_quality"].reshape(-1).float(),
        "legacy_depth_prob": selected["legacy_depth_prob"].reshape(-1).float(),
        "cdf_prob": selected["cdf_prob"].reshape(-1, selected["cdf_prob"].shape[-1]).float(),
    }
    return grasp, meta


def _detect_collision_chunked(detector, gg: GraspGroup) -> np.ndarray:
    n = len(gg)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    chunk = max(1, int(JU.ju_collision_chunk))
    masks = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cur = detector.detect(
            gg[start:end], approach_dist=0.05, collision_thresh=cfgs.collision_thresh
        )
        masks.append(cur.detach().cpu().numpy().astype(bool))
    return np.concatenate(masks, axis=0)


def _meta_rows(meta: Mapping[str, torch.Tensor], collision: np.ndarray | None) -> List[Dict[str, Any]]:
    cpu = {k: v.detach().cpu().numpy() for k, v in meta.items() if torch.is_tensor(v)}
    rows: List[Dict[str, Any]] = []
    n = len(cpu["utility"])
    for i in range(n):
        row: Dict[str, Any] = {
            "raw_candidate_idx": i,
            "q_idx": int(cpu["q_idx"][i]),
            "view_id": int(cpu["view_id"][i]),
            "angle_id": int(cpu["angle_id"][i]),
            "depth_id": int(cpu["depth_id"][i]),
            "utility": float(cpu["utility"][i]),
            "pred_collision_prob": float(cpu["pred_collision_prob"][i]),
            "pred_empty_prob": float(cpu["pred_empty_prob"][i]),
            "legacy_quality": float(cpu["legacy_quality"][i]),
            "legacy_depth_prob": float(cpu["legacy_depth_prob"][i]),
            "model_free_collision": bool(collision[i]) if collision is not None else False,
        }
        for j in range(cpu["cdf_prob"].shape[-1]):
            row[f"cdf_{j}"] = float(cpu["cdf_prob"][i, j])
        rows.append(row)
    return rows


def _load_base_model(device: torch.device):
    net = economicgrasp_dpt(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=bool(getattr(cfgs, "use_obs_depth", False)),
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    ).to(device)
    checkpoint = torch.load(cfgs.checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        net.load_state_dict(state)
    except RuntimeError:
        if state and all(str(k).startswith("module.") for k in state):
            net.load_state_dict({str(k)[7:]: v for k, v in state.items()})
        else:
            raise
    return net.eval()


def main() -> None:
    if not cfgs.multi_modal:
        raise ValueError("Joint utility inference requires --multi_modal")
    if int(getattr(cfgs, "kview_k", 1)) != 1:
        raise ValueError("This script targets first-generation CVA with kview_k=1")

    torch.manual_seed(int(JU.ju_seed))
    np.random.seed(int(JU.ju_seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_dataset = _build_dataset(cfgs)
    eval_dataset, sampled_indices = _build_subset(
        full_dataset, float(getattr(cfgs, "sample_interval", 1.0))
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=int(cfgs.batch_size),
        shuffle=False,
        num_workers=int(getattr(cfgs, "num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(getattr(cfgs, "num_workers", 2)) > 0,
        worker_init_fn=_worker_init_fn,
        collate_fn=collate_fn,
    )
    scene_list = full_dataset.scene_list()

    net = _load_base_model(device)
    scorer = load_joint_utility_scorer(JU.ju_scorer_checkpoint, device=device, strict=True)
    if int(JU.ju_top_views) > int(scorer.config.max_view_rank):
        raise ValueError(
            f"ju_top_views={JU.ju_top_views} exceeds scorer max_view_rank={scorer.config.max_view_rank}"
        )

    output_root = os.path.abspath(cfgs.save_dir)
    os.makedirs(output_root, exist_ok=True)
    config = {
        "variant_name": str(JU.ju_variant_name),
        "base_checkpoint": str(cfgs.checkpoint_path),
        "scorer_checkpoint": os.path.abspath(JU.ju_scorer_checkpoint),
        "scorer_config": scorer.config.to_dict(),
        "test_mode": str(cfgs.test_mode),
        "camera": str(cfgs.camera),
        "sample_interval": float(getattr(cfgs, "sample_interval", 1.0)),
        "num_samples": len(sampled_indices),
        "top_views": int(JU.ju_top_views),
        "top_angles": int(JU.ju_top_angles),
        "top_depths": int(JU.ju_top_depths),
        "per_center_budget": int(JU.ju_per_center_budget),
        "collision_thresh": float(cfgs.collision_thresh),
        "min_utility": float(JU.ju_min_utility),
    }
    _save_json(config, os.path.join(output_root, "_joint_utility_infer_config.json"))

    cursor = 0
    rows: List[Dict[str, Any]] = []
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        _move_batch_to_device(batch, device)
        rng = _capture_rng()
        with torch.inference_mode():
            base = net(_copy_inputs(batch))
        if "cva_angle_feature" not in base:
            raise KeyError("Base model did not expose cva_angle_feature")

        base_view_score = _canonical_view_score(base).detach()
        top_view, view_info = _top_view_indices(base, int(JU.ju_top_views))
        bs = int(base["xyz_graspable"].shape[0])
        per_batch_parts: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(bs)]

        for rank in range(top_view.shape[-1]):
            if rank == 0:
                current = base
            else:
                forced = top_view[:, :, rank].long()
                with torch.inference_mode():
                    current = _run_forced_view(
                        net, batch, rng, forced, base, f"view_rank_{rank}"
                    )
            for b in range(bs):
                part = _make_scorer_candidates(
                    current, base_view_score, b, rank, scorer
                )
                per_batch_parts[b].append(_attach_scores(part, scorer, device))
            if rank > 0:
                del current

        for b in range(bs):
            if cursor >= len(sampled_indices):
                raise RuntimeError("Sample cursor exceeded subset length")
            data_idx = int(sampled_indices[cursor])
            cursor += 1
            scene_name = scene_list[data_idx]
            anno_id = int(data_idx % 256)

            merged = _merge_parts(per_batch_parts[b])
            selected = _select_per_center(merged, int(JU.ju_per_center_budget))
            grasp_tensor, meta = _decode(selected)

            if float(JU.ju_min_utility) >= 0:
                keep = meta["utility"] >= float(JU.ju_min_utility)
                grasp_tensor = grasp_tensor[keep]
                meta = {k: v[keep] for k, v in meta.items()}

            gg = GraspGroup(grasp_tensor.detach().cpu().numpy())
            if SAVE_RAW:
                raw_dir = os.path.join(output_root, "_nocollision", scene_name, cfgs.camera)
                os.makedirs(raw_dir, exist_ok=True)
                gg.save_npy(os.path.join(raw_dir, f"{anno_id:04d}.npy"))

            collision_mask = None
            if float(cfgs.collision_thresh) > 0:
                cloud, _ = full_dataset.get_data(data_idx, return_raw_cloud=True)
                detector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size
                )
                collision_mask = _detect_collision_chunked(detector, gg)
                gg = gg[~collision_mask]

            out_dir = os.path.join(output_root, scene_name, cfgs.camera)
            os.makedirs(out_dir, exist_ok=True)
            gg.save_npy(os.path.join(out_dir, f"{anno_id:04d}.npy"))

            base_view = base["grasp_top_view_inds"][b].long()
            selected_view = meta["view_id"]
            selected_q = meta["q_idx"]
            row = {
                "split": str(cfgs.test_mode),
                "scene": str(scene_name),
                "anno_id": anno_id,
                "data_idx": data_idx,
                "candidate_count_raw": int(meta["utility"].numel()),
                "candidate_count_final": int(len(gg)),
                "utility_mean": float(meta["utility"].mean().item()) if meta["utility"].numel() else 0.0,
                "utility_max": float(meta["utility"].max().item()) if meta["utility"].numel() else 0.0,
                "pred_collision_mean": float(meta["pred_collision_prob"].mean().item()) if meta["utility"].numel() else 0.0,
                "pred_empty_mean": float(meta["pred_empty_prob"].mean().item()) if meta["utility"].numel() else 0.0,
                "non_top1_view_ratio": float(
                    selected_view.ne(base_view.index_select(0, selected_q)).float().mean().item()
                ) if selected_view.numel() else 0.0,
                "model_free_collision_ratio": float(collision_mask.mean()) if collision_mask is not None and collision_mask.size else 0.0,
                **view_info,
            }
            rows.append(row)

            if SAVE_META:
                meta_dir = os.path.join(output_root, "_candidate_meta", scene_name, cfgs.camera)
                os.makedirs(meta_dir, exist_ok=True)
                _save_csv(
                    _meta_rows(meta, collision_mask),
                    os.path.join(meta_dir, f"{anno_id:04d}.csv"),
                )

        if batch_idx % 10 == 0:
            print(
                f"[INFER] batch={batch_idx:04d} samples={cursor}/{len(eval_dataset)} "
                f"elapsed={(time.time()-start_time)/60.0:.1f}m",
                flush=True,
            )
            _save_csv(rows, os.path.join(output_root, "_joint_utility_infer_rows.partial.csv"))

    if cursor != len(eval_dataset):
        raise RuntimeError(f"Processed {cursor} samples, expected {len(eval_dataset)}")

    _save_csv(rows, os.path.join(output_root, "_joint_utility_infer_rows.csv"))
    summary: Dict[str, Any] = {**config, "elapsed_minutes": (time.time() - start_time) / 60.0}
    for key in [
        "candidate_count_raw", "candidate_count_final", "utility_mean", "utility_max",
        "pred_collision_mean", "pred_empty_mean", "non_top1_view_ratio",
        "model_free_collision_ratio",
    ]:
        values = [float(r[key]) for r in rows]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    summary["status"] = "complete"
    _save_json(summary, os.path.join(output_root, "_joint_utility_infer_complete.json"))
    print(f"[DONE] {output_root}", flush=True)


if __name__ == "__main__":
    main()
