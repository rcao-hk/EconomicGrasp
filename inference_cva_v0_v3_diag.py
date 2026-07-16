#!/usr/bin/env python3
"""Inference-only V0--V3 diagnostics for first-generation CVA Transformer.

Variants
--------
V0 ``v0_base``
    Original first-generation CVA decode: predicted top-1 view, score-selected
    angle, predicted depth, predicted width, and the original expected quality
    score. The implementation is checked against the repository decoder.

V1 ``v1_collision_angle``
    Uses the current angle-level collision probability *before* angle collapse:

        risk(q,a) = quality(q,a) * (1 - p_collision(q,a)) ** beta

    The angle is selected by ``risk``. Depth and width are decoded as in V0.
    The same risk-adjusted value is used for global ranking.

V2 ``v2_a2d2``
    Under the predicted top-1 view, retains top-2 risk-adjusted angles and the
    top-2 valid depth classes for each angle. It emits four candidates per
    center. Candidate score is:

        risk(q,a) * p_depth(d | q,a) ** alpha

V3 ``v3_v4a2d2``
    Retains the predicted top-4 views. For each view it reruns the actual CVA
    grouping/head using ``oracle_view_inds_override`` and constructs the same
    top-2 angle x top-2 depth lattice. To keep candidate budget and center
    coverage matched to V2, it retains the top-4 candidates *per center* across
    the resulting 16 candidates.

This is a diagnostic program, not a new benchmark protocol. V3 uses several
counterfactual forward passes from the same single image. It does not use GT,
but it is computationally more expensive than the standard model.

Requirements
------------
- First-generation CVA checkpoint/config (``kview_k=1``), not RotNet-CVA.
- A collision-head checkpoint for V1--V3.
- The fixed runtime-view-override model path. A forced pass must expose
  ``grasp_top_view_inds`` equal to ``oracle_view_inds_override``; otherwise the
  script stops rather than silently producing an invalid V3 result.

Custom ``--beam_*`` arguments are consumed before importing the repository's
``utils.arguments`` parser, so no project parser modification is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def _consume_custom_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--beam_variants", default="v0_base,v1_collision_angle,v2_a2d2,v3_v4a2d2")
    p.add_argument("--beam_top_views", type=int, default=4)
    p.add_argument("--beam_top_angles", type=int, default=2)
    p.add_argument("--beam_top_depths", type=int, default=2)
    p.add_argument("--beam_v3_per_center_budget", type=int, default=4)
    p.add_argument("--beam_collision_power", type=float, default=1.0)
    p.add_argument("--beam_depth_power", type=float, default=1.0)
    p.add_argument("--beam_collision_logit_scale", type=float, default=1.0)
    p.add_argument("--beam_collision_logit_bias", type=float, default=0.0)
    p.add_argument("--beam_require_collision_head", type=int, default=1)
    p.add_argument("--beam_save_nocollision", type=int, default=0)
    p.add_argument("--beam_save_candidate_meta", type=int, default=1)
    p.add_argument("--beam_collision_chunk", type=int, default=1024)
    p.add_argument("--beam_strict", type=int, default=1)
    p.add_argument("--beam_seed", type=int, default=0)
    p.add_argument("--beam_override_key", default="oracle_view_inds_override")
    custom, rest = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    return custom


BEAM = _consume_custom_args()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from graspnetAPI import GraspGroup
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn
from utils.label_generation import batch_viewpoint_params_to_matrix
from models.economicgrasp_bip3d import economicgrasp_dpt, pred_decode_center_view_angle


ALLOWED_VARIANTS = (
    "v0_base",
    "v1_collision_angle",
    "v2_a2d2",
    "v3_v4a2d2",
)
VARIANTS = [x.strip() for x in str(BEAM.beam_variants).split(",") if x.strip()]
unknown = sorted(set(VARIANTS) - set(ALLOWED_VARIANTS))
if unknown:
    raise ValueError(f"Unknown --beam_variants entries: {unknown}; allowed={ALLOWED_VARIANTS}")
if not VARIANTS:
    raise ValueError("At least one beam variant is required")

STRICT = bool(BEAM.beam_strict)
REQUIRE_COLLISION = bool(BEAM.beam_require_collision_head)
SAVE_RAW = bool(BEAM.beam_save_nocollision)
SAVE_META = bool(BEAM.beam_save_candidate_meta)


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(payload)), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _save_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(set().union(*(set(r.keys()) for r in rows)))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")
    total = len(dataset)
    if sample_interval >= 1.0:
        idx = list(range(total))
        return dataset, idx
    stride = max(1, int(round(1.0 / sample_interval)))
    idx: List[int] = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        idx.extend(start + i for i in range(0, end - start, stride))
    return Subset(dataset, idx), idx


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


def _capture_rng() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
    }
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
    x = end_points["view_score"]
    q = int(end_points["xyz_graspable"].shape[1])
    if x.dim() != 3:
        raise ValueError(f"view_score must be 3D, got {tuple(x.shape)}")
    if x.shape[1] == q:
        return x
    if x.shape[2] == q:
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot canonicalize view_score {tuple(x.shape)} to [B,Q,V], Q={q}")


def _expected_quality(score_logits: torch.Tensor) -> torch.Tensor:
    """score_logits [C,Q,A] -> expected quality [Q,A]."""
    c = score_logits.shape[0]
    bins = torch.linspace(0.0, 1.0, steps=c, device=score_logits.device, dtype=score_logits.dtype)
    return (F.softmax(score_logits, dim=0) * bins[:, None, None]).sum(dim=0)


def _collision_probability(end_points: Mapping[str, Any], b: int, q: int, a: int) -> torch.Tensor:
    logits = end_points.get("grasp_collision_pred_angle")
    if not torch.is_tensor(logits):
        if REQUIRE_COLLISION:
            raise KeyError(
                "V1--V3 require grasp_collision_pred_angle [B,1,Q,A]. "
                "Use the collision-head checkpoint and pass --kview_use_collision."
            )
        return torch.zeros((q, a), device=end_points["xyz_graspable"].device)
    cur = logits[b].float()
    if cur.dim() == 3 and cur.shape[0] == 1:
        cur = cur.squeeze(0)
    if cur.shape != (q, a):
        raise ValueError(f"collision logits must canonicalize to [Q,A]={q,a}, got {tuple(cur.shape)}")
    z = float(BEAM.beam_collision_logit_scale) * cur + float(BEAM.beam_collision_logit_bias)
    return torch.sigmoid(z)


def _gather_2d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [Q,A], idx [Q,K] -> [Q,K]."""
    return torch.gather(x, 1, idx)


def _make_lattice(
    end_points: Mapping[str, Any],
    b: int,
    top_angles: int,
    top_depths: int,
) -> Dict[str, torch.Tensor]:
    centers = end_points["xyz_graspable"][b].float()             # [Q,3]
    view_xyz = end_points["grasp_top_view_xyz"][b].float()      # [Q,3]
    view_idx = end_points["grasp_top_view_inds"][b].long()      # [Q]
    score_logits = end_points["grasp_score_pred_angle"][b].float()  # [C,Q,A]
    depth_logits = end_points["grasp_depth_pred_angle"][b].float()  # [D+1,Q,A]
    width_pred = end_points["grasp_width_pred_angle"][b].float()    # [1,Q,A]

    _, q, a = score_logits.shape
    d = max(1, min(int(getattr(cfgs, "num_depth", depth_logits.shape[0] - 1)), depth_logits.shape[0] - 1))
    ta = max(1, min(int(top_angles), a))
    td = max(1, min(int(top_depths), d))

    quality = _expected_quality(score_logits)                    # [Q,A]
    collision_prob = _collision_probability(end_points, b, q, a)
    risk = quality * (1.0 - collision_prob).clamp_min(0.0).pow(float(BEAM.beam_collision_power))

    angle_idx = torch.topk(risk, k=ta, dim=1, largest=True, sorted=True).indices  # [Q,Ta]
    quality_sel = _gather_2d(quality, angle_idx)
    risk_sel = _gather_2d(risk, angle_idx)
    collision_sel = _gather_2d(collision_prob, angle_idx)
    width_sel = torch.gather(width_pred.squeeze(0), 1, angle_idx)

    # The invalid/dummy depth class participates in the softmax denominator but
    # cannot be decoded as a physical depth. This penalizes low valid-depth mass.
    depth_prob = F.softmax(depth_logits, dim=0)
    valid_prob = depth_prob[:d].permute(1, 2, 0).contiguous()     # [Q,A,D]
    q_idx_qa = torch.arange(q, device=centers.device)[:, None].expand(q, ta)
    valid_prob_sel = valid_prob[q_idx_qa, angle_idx]              # [Q,Ta,D]
    depth_prob_sel, depth_idx = torch.topk(valid_prob_sel, k=td, dim=2, largest=True, sorted=True)

    q_idx = torch.arange(q, device=centers.device)[:, None, None].expand(q, ta, td)
    a_idx = angle_idx[:, :, None].expand(q, ta, td)
    risk_3 = risk_sel[:, :, None].expand(q, ta, td)
    quality_3 = quality_sel[:, :, None].expand(q, ta, td)
    collision_3 = collision_sel[:, :, None].expand(q, ta, td)
    width_3 = width_sel[:, :, None].expand(q, ta, td)
    view_idx_3 = view_idx[:, None, None].expand(q, ta, td)
    candidate_score = risk_3 * depth_prob_sel.clamp_min(0.0).pow(float(BEAM.beam_depth_power))

    return {
        "score": candidate_score,
        "quality": quality_3,
        "risk": risk_3,
        "collision_prob": collision_3,
        "depth_prob": depth_prob_sel,
        "q_idx": q_idx,
        "angle_idx": a_idx,
        "depth_idx": depth_idx,
        "width_raw": width_3,
        "view_idx": view_idx_3,
        "centers": centers,
        "view_xyz": view_xyz,
        "num_angle": torch.tensor(a, device=centers.device),
    }


def _select_per_center(lattice: Dict[str, torch.Tensor], budget: int) -> Dict[str, torch.Tensor]:
    """Select top budget candidates per center across lattice axes 1..N."""
    score = lattice["score"].reshape(lattice["score"].shape[0], -1)
    q, n = score.shape
    k = max(1, min(int(budget), n))
    keep = torch.topk(score, k=k, dim=1, largest=True, sorted=True).indices
    out: Dict[str, torch.Tensor] = {}
    for key in [
        "score", "quality", "risk", "collision_prob", "depth_prob",
        "q_idx", "angle_idx", "depth_idx", "width_raw", "view_idx",
    ]:
        flat = lattice[key].reshape(q, -1)
        out[key] = torch.gather(flat, 1, keep)
    out["centers"] = lattice["centers"]
    out["view_xyz_candidates"] = lattice["view_xyz_candidates"].reshape(q, -1, 3).gather(
        1, keep[:, :, None].expand(q, k, 3)
    ) if "view_xyz_candidates" in lattice else lattice["view_xyz"][:, None, :].expand(q, k, 3)
    out["num_angle"] = lattice["num_angle"]
    return out


def _flatten_lattice(lattice: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    q = lattice["score"].shape[0]
    out = {}
    for key in [
        "score", "quality", "risk", "collision_prob", "depth_prob",
        "q_idx", "angle_idx", "depth_idx", "width_raw", "view_idx",
    ]:
        out[key] = lattice[key].reshape(q, -1)
    k = out["score"].shape[1]
    out["centers"] = lattice["centers"]
    out["view_xyz_candidates"] = lattice["view_xyz"][:, None, :].expand(q, k, 3)
    out["num_angle"] = lattice["num_angle"]
    return out


def _decode_flat(flat: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    q, k = flat["score"].shape
    q_idx = flat["q_idx"].reshape(-1).long()
    angle_idx = flat["angle_idx"].reshape(-1).long()
    depth_idx = flat["depth_idx"].reshape(-1).long()
    score = flat["score"].reshape(-1).float()
    width_raw = flat["width_raw"].reshape(-1).float()
    view_xyz = flat["view_xyz_candidates"].reshape(-1, 3).float()
    centers = flat["centers"].index_select(0, q_idx).float()
    num_angle = int(flat["num_angle"].item())

    angle = angle_idx.float() * (math.pi / float(num_angle))
    rot = batch_viewpoint_params_to_matrix(-view_xyz, angle).reshape(-1, 9)
    width_m = torch.clamp(
        1.2 * width_raw / 10.0,
        min=0.0,
        max=float(getattr(cfgs, "grasp_max_width", 0.1)),
    )
    depth_m = (depth_idx.float() + 1.0) * 0.01
    height = torch.full_like(score, 0.02)
    obj_id = torch.full_like(score, -1.0)
    grasp = torch.cat(
        [
            score[:, None], width_m[:, None], height[:, None], depth_m[:, None],
            rot, centers, obj_id[:, None],
        ],
        dim=1,
    )
    meta = {
        "q_idx": q_idx,
        "angle_idx": angle_idx,
        "depth_idx": depth_idx,
        "view_idx": flat["view_idx"].reshape(-1).long(),
        "quality": flat["quality"].reshape(-1).float(),
        "risk": flat["risk"].reshape(-1).float(),
        "collision_prob": flat["collision_prob"].reshape(-1).float(),
        "depth_prob": flat["depth_prob"].reshape(-1).float(),
        "score": score,
    }
    return grasp, meta


def _decode_v0(end_points: Mapping[str, Any], b: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    score_logits = end_points["grasp_score_pred_angle"][b].float()
    depth_logits = end_points["grasp_depth_pred_angle"][b].float()
    width_pred = end_points["grasp_width_pred_angle"][b].float()
    centers = end_points["xyz_graspable"][b].float()
    view_xyz = end_points["grasp_top_view_xyz"][b].float()
    view_idx = end_points["grasp_top_view_inds"][b].long()
    _, q, a = score_logits.shape
    d = max(1, min(int(getattr(cfgs, "num_depth", depth_logits.shape[0] - 1)), depth_logits.shape[0] - 1))
    quality = _expected_quality(score_logits)
    angle_idx = quality.argmax(dim=1)
    q_idx = torch.arange(q, device=centers.device)
    depth_idx = depth_logits[:d, q_idx, angle_idx].argmax(dim=0)
    width_raw = width_pred[0, q_idx, angle_idx]
    collision_all = _collision_probability(end_points, b, q, a)
    collision_prob = collision_all[q_idx, angle_idx]
    angle = angle_idx.float() * (math.pi / float(a))
    rot = batch_viewpoint_params_to_matrix(-view_xyz, angle).reshape(q, 9)
    width_m = torch.clamp(1.2 * width_raw / 10.0, 0.0, float(getattr(cfgs, "grasp_max_width", 0.1)))
    depth_m = (depth_idx.float() + 1.0) * 0.01
    score = quality[q_idx, angle_idx]
    grasp = torch.cat(
        [
            score[:, None], width_m[:, None], torch.full((q, 1), 0.02, device=score.device),
            depth_m[:, None], rot, centers, torch.full((q, 1), -1.0, device=score.device),
        ],
        dim=1,
    )
    meta = {
        "q_idx": q_idx,
        "angle_idx": angle_idx,
        "depth_idx": depth_idx,
        "view_idx": view_idx,
        "quality": score,
        "risk": score * (1.0 - collision_prob).pow(float(BEAM.beam_collision_power)),
        "collision_prob": collision_prob,
        "depth_prob": F.softmax(depth_logits, dim=0)[depth_idx, q_idx, angle_idx],
        "score": score,
    }
    return grasp, meta


def _decode_v1(end_points: Mapping[str, Any], b: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lattice = _make_lattice(end_points, b, top_angles=1, top_depths=1)
    flat = _flatten_lattice(lattice)
    # V1 does not use depth probability in the final score; it only changes
    # angle selection/ranking through the collision-adjusted risk.
    flat["score"] = flat["risk"]
    return _decode_flat(flat)


def _decode_v2(end_points: Mapping[str, Any], b: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lattice = _make_lattice(
        end_points, b,
        top_angles=int(BEAM.beam_top_angles),
        top_depths=int(BEAM.beam_top_depths),
    )
    return _decode_flat(_flatten_lattice(lattice))


def _merge_v3_lattices(lattices: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not lattices:
        raise ValueError("No V3 view lattices")
    q = lattices[0]["score"].shape[0]
    merged: Dict[str, torch.Tensor] = {}
    for key in [
        "score", "quality", "risk", "collision_prob", "depth_prob",
        "q_idx", "angle_idx", "depth_idx", "width_raw", "view_idx",
    ]:
        merged[key] = torch.cat([x[key].reshape(q, -1) for x in lattices], dim=1)
    merged["view_xyz_candidates"] = torch.cat(
        [
            x["view_xyz"][:, None, :].expand(q, x["score"].reshape(q, -1).shape[1], 3)
            for x in lattices
        ],
        dim=1,
    )
    merged["centers"] = lattices[0]["centers"]
    merged["num_angle"] = lattices[0]["num_angle"]
    return merged


def _assert_same_centers(base: Mapping[str, Any], other: Mapping[str, Any], tag: str) -> None:
    checks = []
    if torch.is_tensor(base.get("token_sel_idx")) and torch.is_tensor(other.get("token_sel_idx")):
        checks.append(("token_sel_idx", torch.equal(base["token_sel_idx"], other["token_sel_idx"])))
    checks.append(("xyz_graspable", torch.allclose(base["xyz_graspable"], other["xyz_graspable"], atol=1e-7, rtol=0.0)))
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise RuntimeError(f"Counterfactual pass {tag} changed center/query selection: {failed}")


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
    inp[str(BEAM.beam_override_key)] = forced_view
    out = net(inp)
    _assert_same_centers(base, out, tag)
    used = out.get("grasp_top_view_inds")
    if not torch.is_tensor(used):
        raise KeyError("Forced pass did not expose grasp_top_view_inds")
    respected = used.long().eq(forced_view.long())
    ratio = float(respected.float().mean().item())
    if ratio < 0.999999:
        msg = (
            f"Forced view override was not respected for {tag}: ratio={ratio:.6f}. "
            "Apply the runtime view override + view-score pinning model fix."
        )
        if STRICT:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
    return out


def _manifest(root: str, sampled_indices: Iterable[int]) -> None:
    _save_json(
        {
            "test_mode": str(cfgs.test_mode),
            "camera": str(cfgs.camera),
            "sample_interval": float(getattr(cfgs, "sample_interval", 1.0)),
            "num_samples": len(list(sampled_indices)),
            "sampled_indices": [int(x) for x in sampled_indices],
        },
        os.path.join(root, "_sampled_indices.json"),
    )


def _meta_summary(meta: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    out = {
        "candidate_count_raw": int(meta["score"].numel()),
        "unique_centers": int(torch.unique(meta["q_idx"]).numel()),
        "score_mean": float(meta["score"].mean().item()),
        "score_max": float(meta["score"].max().item()),
        "quality_mean": float(meta["quality"].mean().item()),
        "collision_prob_mean": float(meta["collision_prob"].mean().item()),
        "collision_prob_max": float(meta["collision_prob"].max().item()),
        "depth_prob_mean": float(meta["depth_prob"].mean().item()),
    }
    if meta["view_idx"].numel():
        out["unique_view_indices"] = int(torch.unique(meta["view_idx"]).numel())
    return out




def _detect_collision_chunked(detector, gg: GraspGroup) -> np.ndarray:
    n = len(gg)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    chunk = max(1, int(BEAM.beam_collision_chunk))
    masks = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cur = detector.detect(
            gg[start:end],
            approach_dist=0.05,
            collision_thresh=cfgs.collision_thresh,
        )
        masks.append(cur.detach().cpu().numpy().astype(bool))
    return np.concatenate(masks, axis=0)

def _save_candidate_meta(meta: Mapping[str, torch.Tensor], path: str, collision_mask: np.ndarray | None) -> None:
    n = int(meta["score"].numel())
    rows = []
    cpu = {k: v.detach().cpu().numpy() for k, v in meta.items() if torch.is_tensor(v)}
    for i in range(n):
        rows.append(
            {
                "raw_candidate_idx": i,
                "q_idx": int(cpu["q_idx"][i]),
                "view_idx": int(cpu["view_idx"][i]),
                "angle_idx": int(cpu["angle_idx"][i]),
                "depth_idx": int(cpu["depth_idx"][i]),
                "score": float(cpu["score"][i]),
                "quality": float(cpu["quality"][i]),
                "risk": float(cpu["risk"][i]),
                "collision_prob": float(cpu["collision_prob"][i]),
                "depth_prob": float(cpu["depth_prob"][i]),
                "model_free_collision": bool(collision_mask[i]) if collision_mask is not None else False,
            }
        )
    _save_csv(rows, path)


def main() -> None:
    if not cfgs.multi_modal:
        raise ValueError("This diagnostic targets economicgrasp_dpt with --multi_modal")
    if int(getattr(cfgs, "kview_k", 1)) != 1:
        raise ValueError("V0--V3 target first-generation CVA with kview_k=1, not RotNet/top-L")

    torch.manual_seed(int(BEAM.beam_seed))
    np.random.seed(int(BEAM.beam_seed))

    full_dataset = _build_dataset(cfgs)
    eval_dataset, sampled_indices = _build_subset(full_dataset, float(getattr(cfgs, "sample_interval", 1.0)))
    loader = DataLoader(
        eval_dataset,
        batch_size=int(cfgs.batch_size),
        shuffle=False,
        num_workers=int(getattr(cfgs, "num_workers", 2)),
        worker_init_fn=_worker_init_fn,
        collate_fn=collate_fn,
    )
    scene_list = full_dataset.scene_list()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    try:
        net.load_state_dict(state)
    except RuntimeError:
        if state and all(str(k).startswith("module.") for k in state):
            net.load_state_dict({str(k)[7:]: v for k, v in state.items()})
        else:
            raise
    net.eval()

    split_root = os.path.abspath(cfgs.save_dir)
    os.makedirs(split_root, exist_ok=True)
    for variant in VARIANTS:
        _manifest(os.path.join(split_root, variant), sampled_indices)

    config = {
        "variants": VARIANTS,
        "top_views": int(BEAM.beam_top_views),
        "top_angles": int(BEAM.beam_top_angles),
        "top_depths": int(BEAM.beam_top_depths),
        "v3_per_center_budget": int(BEAM.beam_v3_per_center_budget),
        "collision_power": float(BEAM.beam_collision_power),
        "depth_power": float(BEAM.beam_depth_power),
        "collision_logit_scale": float(BEAM.beam_collision_logit_scale),
        "collision_logit_bias": float(BEAM.beam_collision_logit_bias),
        "require_collision_head": REQUIRE_COLLISION,
        "model_free_collision_chunk": int(BEAM.beam_collision_chunk),
        "checkpoint_path": str(cfgs.checkpoint_path),
        "test_mode": str(cfgs.test_mode),
        "sample_interval": float(getattr(cfgs, "sample_interval", 1.0)),
        "collision_thresh": float(cfgs.collision_thresh),
        "candidate_budget_note": (
            "V2 emits top_angles*top_depths candidates per center; "
            "V3 retains v3_per_center_budget candidates per center across top-view beam."
        ),
    }
    _save_json(config, os.path.join(split_root, "_beam_diag_config.json"))

    all_rows: List[Dict[str, Any]] = []
    cursor = 0
    start_time = time.time()
    checked_v0 = False

    for batch_idx, batch in enumerate(loader):
        _move_batch_to_device(batch, device)
        rng = _capture_rng()
        with torch.no_grad():
            base = net(_copy_inputs(batch))

        if REQUIRE_COLLISION and "grasp_collision_pred_angle" not in base:
            raise KeyError(
                "The model did not emit grasp_collision_pred_angle. "
                "Pass --kview_use_collision and load the collision checkpoint."
            )

        # V0 exact-decoder equivalence check.
        v0_repo = pred_decode_center_view_angle(base)
        if not checked_v0:
            for b in range(len(v0_repo)):
                ours, _ = _decode_v0(base, b)
                if not torch.allclose(ours, v0_repo[b], atol=1e-6, rtol=1e-6):
                    diff = float((ours - v0_repo[b]).abs().max().item())
                    raise RuntimeError(f"V0 decoder mismatch with repository decoder; max_abs_diff={diff}")
            checked_v0 = True
            print("[PASS] V0 decoder matches pred_decode_center_view_angle")

        # Base view top-K. Rank 0 reuses the base forward; remaining ranks are
        # counterfactual passes with identical center/query selection.
        view_score = _canonical_view_score(base)
        k_view = max(1, min(int(BEAM.beam_top_views), int(view_score.shape[-1])))
        top_view_idx = torch.topk(view_score, k=k_view, dim=-1, largest=True, sorted=True).indices
        bs = int(base["xyz_graspable"].shape[0])
        # Keep only compact candidate lattices from counterfactual passes.
        # Retaining four full endpoint dictionaries can multiply the DPT/CVA
        # activation memory and cause avoidable OOMs during long inference.
        v3_lattices_by_b: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(bs)]
        if "v3_v4a2d2" in VARIANTS:
            base_used = base["grasp_top_view_inds"].long()
            if not torch.equal(base_used, top_view_idx[:, :, 0].long()):
                ratio = float(base_used.eq(top_view_idx[:, :, 0]).float().mean().item())
                msg = f"Base used view differs from view_score top1: agreement={ratio:.6f}"
                if STRICT:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}")
            for b in range(bs):
                v3_lattices_by_b[b].append(
                    _make_lattice(
                        base, b,
                        top_angles=int(BEAM.beam_top_angles),
                        top_depths=int(BEAM.beam_top_depths),
                    )
                )
            for r in range(1, k_view):
                forced = top_view_idx[:, :, r].long()
                with torch.no_grad():
                    out = _run_forced_view(net, batch, rng, forced, base, f"view_rank_{r}")
                for b in range(bs):
                    v3_lattices_by_b[b].append(
                        _make_lattice(
                            out, b,
                            top_angles=int(BEAM.beam_top_angles),
                            top_depths=int(BEAM.beam_top_depths),
                        )
                    )
                del out

        for b in range(bs):
            data_idx = sampled_indices[cursor]
            cursor += 1
            scene_name = scene_list[data_idx]
            anno_id = int(data_idx % 256)

            decoded: Dict[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}
            if "v0_base" in VARIANTS:
                decoded["v0_base"] = _decode_v0(base, b)
            if "v1_collision_angle" in VARIANTS:
                decoded["v1_collision_angle"] = _decode_v1(base, b)
            if "v2_a2d2" in VARIANTS:
                decoded["v2_a2d2"] = _decode_v2(base, b)
            if "v3_v4a2d2" in VARIANTS:
                merged = _merge_v3_lattices(v3_lattices_by_b[b])
                selected = _select_per_center(merged, int(BEAM.beam_v3_per_center_budget))
                decoded["v3_v4a2d2"] = _decode_flat(selected)

            cloud = None
            detector = None
            if float(cfgs.collision_thresh) > 0:
                cloud, _ = full_dataset.get_data(data_idx, return_raw_cloud=True)
                detector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size
                )

            # Base selected angle for diagnostics.
            _, v0_meta = _decode_v0(base, b)
            for variant, (grasp_tensor, meta) in decoded.items():
                gg = GraspGroup(grasp_tensor.detach().cpu().numpy())

                if SAVE_RAW:
                    raw_dir = os.path.join(split_root, f"{variant}_nocollision", scene_name, cfgs.camera)
                    os.makedirs(raw_dir, exist_ok=True)
                    gg.save_npy(os.path.join(raw_dir, f"{anno_id:04d}.npy"))

                collision_mask = None
                if detector is not None:
                    collision_mask = _detect_collision_chunked(detector, gg)
                    gg = gg[~collision_mask]

                out_dir = os.path.join(split_root, variant, scene_name, cfgs.camera)
                os.makedirs(out_dir, exist_ok=True)
                gg.save_npy(os.path.join(out_dir, f"{anno_id:04d}.npy"))

                row: Dict[str, Any] = {
                    "split": str(cfgs.test_mode),
                    "scene": str(scene_name),
                    "anno_id": anno_id,
                    "data_idx": int(data_idx),
                    "variant": variant,
                    "candidate_count_final": int(len(gg)),
                    "model_free_collision_ratio": (
                        float(collision_mask.mean()) if collision_mask is not None and collision_mask.size else 0.0
                    ),
                }
                row.update(_meta_summary(meta))
                if variant == "v1_collision_angle":
                    row["angle_changed_vs_v0_ratio"] = float(
                        meta["angle_idx"].long().ne(v0_meta["angle_idx"].long()).float().mean().item()
                    )
                if variant == "v3_v4a2d2":
                    base_view = base["grasp_top_view_inds"][b].long()
                    selected_view = meta["view_idx"].reshape(-1)
                    selected_q = meta["q_idx"].reshape(-1)
                    row["non_top1_view_candidate_ratio"] = float(
                        selected_view.ne(base_view.index_select(0, selected_q)).float().mean().item()
                    )
                all_rows.append(row)

                if SAVE_META:
                    meta_dir = os.path.join(split_root, "_candidate_meta", variant, scene_name, cfgs.camera)
                    os.makedirs(meta_dir, exist_ok=True)
                    _save_candidate_meta(
                        meta,
                        os.path.join(meta_dir, f"{anno_id:04d}.csv"),
                        collision_mask,
                    )

        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"[batch {batch_idx:04d}] processed={cursor}/{len(eval_dataset)} "
                f"elapsed={elapsed/60.0:.1f} min"
            )
            _save_csv(all_rows, os.path.join(split_root, "_beam_diag_rows.partial.csv"))

    if cursor != len(eval_dataset):
        raise RuntimeError(f"Processed sample count mismatch: {cursor} vs {len(eval_dataset)}")

    _save_csv(all_rows, os.path.join(split_root, "_beam_diag_rows.csv"))
    summary_rows = []
    for variant in VARIANTS:
        cur = [r for r in all_rows if r["variant"] == variant]
        row = {"variant": variant, "num_samples": len(cur)}
        for key in [
            "candidate_count_raw", "candidate_count_final", "unique_centers",
            "score_mean", "score_max", "quality_mean", "collision_prob_mean",
            "collision_prob_max", "depth_prob_mean", "model_free_collision_ratio",
            "angle_changed_vs_v0_ratio", "non_top1_view_candidate_ratio",
        ]:
            vals = [float(r[key]) for r in cur if key in r and r[key] != ""]
            if vals:
                row[f"mean_{key}"] = float(np.mean(vals))
        summary_rows.append(row)
    _save_csv(summary_rows, os.path.join(split_root, "_beam_diag_summary.csv"))
    _save_json(
        {
            **config,
            "num_samples": cursor,
            "elapsed_minutes": (time.time() - start_time) / 60.0,
            "summary_rows": summary_rows,
            "status": "complete",
        },
        os.path.join(split_root, "_beam_diag_complete.json"),
    )
    print(f"[DONE] {split_root}")


if __name__ == "__main__":
    main()
