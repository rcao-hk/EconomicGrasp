"""Pure deterministic candidate construction and P2 selection for Point RayRoPE.

No graspnetAPI, data loader or model import at module scope. This module
preserves the EconomicGrasp CDF decode convention:
  grasp = [score, width, height, insertion, R(9), center(3), object_id].
The CDF score is mean(sigmoid(logits)) across the six friction thresholds.
Width is 1.2 * raw_width / 10, clamped to [0, grasp_max_width].
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch


@dataclass
class Lattice:
    grasp: np.ndarray             # [Q*A*D,17] float32
    query_id: np.ndarray          # [N] int32
    angle_id: np.ndarray          # [N] int32
    depth_id: np.ndarray          # [N] int32
    query_winner: np.ndarray      # [N] bool
    raw_width_m: np.ndarray       # [N] float32 (before clamp)
    num_queries: int
    num_angles: int
    num_depths: int


def physical_action_id(grasp: np.ndarray) -> str:
    """Stable *physical* ID independent of score and object ID."""
    arr = np.asarray(grasp, dtype=np.float32)
    if arr.shape != (17,) or not np.isfinite(arr[1:16]).all():
        raise ValueError("physical_action_id needs finite grasp row [17]")
    # Stored in GraspNet's float32 output convention. A width-only change
    # changes the physical ID; a score-only intervention must not.
    return hashlib.blake2b(arr[1:16].tobytes(), digest_size=12).hexdigest()


def decode_cdf_lattice(
    endpoints: dict,
    batch_index: int,
    rotation_fn: Callable,
    *,
    max_width_m: float = 0.1,
    num_queries: int | None = None,
) -> Lattice:
    """Unpruned physical Q x A x D lattice for the selected native top-view.

    Args:
        endpoints: model forward output, including xyz_graspable,
            grasp_top_view_xyz, grasp_cdf_pred_angle_depth and
            grasp_width_pred_angle_depth.
        batch_index: local batch index.
        rotation_fn: existing batch_viewpoint_params_to_matrix API.
        max_width_m: MUST match the checkpoint's inference setting.
        num_queries: optional explicit prefix cap for smoke tests.
    """
    if not (0 < max_width_m <= 0.3):
        raise ValueError("max_width_m must be in (0,0.3]")
    required = ("xyz_graspable", "grasp_top_view_xyz",
                "grasp_cdf_pred_angle_depth", "grasp_width_pred_angle_depth")
    absent = [k for k in required if k not in endpoints]
    if absent:
        raise KeyError(f"Missing endpoints: {absent}")
    center = endpoints["xyz_graspable"][batch_index].detach().float()
    view = endpoints["grasp_top_view_xyz"][batch_index].detach().float()
    cdf = endpoints["grasp_cdf_pred_angle_depth"][batch_index].detach().float()
    widths = endpoints["grasp_width_pred_angle_depth"][batch_index].detach().float()
    if center.ndim != 2 or center.shape[-1] != 3 or view.shape != center.shape:
        raise ValueError("center/view must each be [Q,3]")
    if cdf.ndim != 4 or widths.ndim != 3:
        raise ValueError("CDF must be [T,Q,A,D], width [D,Q,A]")
    t, q, a, d = cdf.shape
    if widths.shape != (d, q, a) or center.shape[0] != q:
        raise ValueError("Mismatched query/angle/depth dimensions")
    if t != 6:
        raise ValueError(f"Expected six CDF friction logits, got {t}")
    if num_queries is not None:
        if not 1 <= num_queries <= q:
            raise ValueError(f"num_queries must be in [1,{q}]")
        q = num_queries
        center = center[:q]
        view = view[:q]
        cdf = cdf[:, :q]
        widths = widths[:, :q]

    utility = torch.sigmoid(cdf).mean(dim=0)  # [Q,A,D]
    width_unclamped = (1.2 * widths.permute(1, 2, 0) / 10.0).contiguous()
    width_metric = width_unclamped.clamp(0.0, max_width_m)
    winner_flat = utility.reshape(q, a * d).argmax(dim=-1)  # native tie order
    query_id = torch.arange(q, device=center.device).reshape(q, 1, 1).expand(q, a, d)
    angle_id = torch.arange(a, device=center.device).reshape(1, a, 1).expand(q, a, d)
    depth_id = torch.arange(d, device=center.device).reshape(1, 1, d).expand(q, a, d)
    selected = angle_id * d + depth_id == winner_flat[:, None, None]

    angles_qa = (torch.arange(a, device=center.device).float() * (np.pi / a))
    approach_qa = (-view[:, None, :].expand(q, a, 3)).reshape(q * a, 3)
    theta_qa = angles_qa[None, :].expand(q, a).reshape(-1)
    rotation_qa = rotation_fn(approach_qa, theta_qa).reshape(q, a, 3, 3)
    rotation_qad = rotation_qa[:, :, None].expand(q, a, d, 3, 3)
    c_qad = center[:, None, None, :].expand(q, a, d, 3)

    n = q * a * d
    one = torch.ones((q, a, d, 1), device=center.device, dtype=torch.float32)
    arrays = [
        utility.unsqueeze(-1),
        width_metric.unsqueeze(-1),
        0.02 * one,
        (0.01 * (depth_id.float() + 1.0)).unsqueeze(-1),
        rotation_qad.reshape(q, a, d, 9),
        c_qad,
        -one,
    ]
    grasp = torch.cat(arrays, dim=-1).reshape(n, 17).cpu().numpy().astype(np.float32, copy=False)
    if grasp.shape != (n, 17) or not np.isfinite(grasp).all():
        raise ValueError("Invalid decoded full grasp lattice")
    return Lattice(
        grasp=grasp, query_id=query_id.reshape(-1).cpu().numpy().astype(np.int32),
        angle_id=angle_id.reshape(-1).cpu().numpy().astype(np.int32),
        depth_id=depth_id.reshape(-1).cpu().numpy().astype(np.int32),
        query_winner=selected.reshape(-1).cpu().numpy().astype(bool),
        raw_width_m=width_unclamped.reshape(-1).cpu().numpy().astype(np.float32),
        num_queries=q, num_angles=a, num_depths=d,
    )


def _random(rng: np.random.Generator, seq: Sequence[int], n: int) -> List[int]:
    if n <= 0:
        return []
    if len(seq) < n:
        raise ValueError(f"Not enough candidates: {len(seq)} < {n}")
    return [int(x) for x in rng.choice(np.asarray(seq), size=n, replace=False)]


def select_p2_queries(
    lattice: Lattice,
    *,
    num_query_anchors: int = 16,
    nonwinner_per_query: int = 1,
    full_lattice_per_anchor: bool = False,
    uniform_budget: int | None = None,
    seed: int = 20261106,
) -> Dict[str, List[int]]:
    """Match winner/nonwinner at identical sampled query anchors.

    The uniform group comes from the entire Q*A*D lattice, not the
    argmax-pruned per-query winner set. Each group is a distinct
    candidate-selection protocol; compare them descriptively, not as
    a pure sampling-method causal ablation.
    """
    if lattice.num_queries < 1 or lattice.num_angles * lattice.num_depths < 2:
        raise ValueError("A query requires at least two angle/depth candidates")
    if not 1 <= num_query_anchors <= lattice.num_queries:
        raise ValueError("Bad num_query_anchors")
    ad = lattice.num_angles * lattice.num_depths
    if not 1 <= nonwinner_per_query < ad:
        raise ValueError("nonwinner_per_query must be between 1 and A*D-1")
    rng = np.random.default_rng(seed)
    anchors = sorted(_random(rng, list(range(lattice.num_queries)), num_query_anchors))
    winner, nonwinner, full = [], [], []
    for q in anchors:
        candidates = list(range(q * ad, (q + 1) * ad))
        w = [i for i in candidates if lattice.query_winner[i]]
        if len(w) != 1:
            raise AssertionError(f"Expected exactly one winner for query {q}")
        others = [i for i in candidates if i != w[0]]
        winner.extend(w)
        nonwinner.extend(_random(rng, others, nonwinner_per_query))
        if full_lattice_per_anchor:
            full.extend(candidates)
    n_uniform = (num_query_anchors * (1 + nonwinner_per_query)
                 if uniform_budget is None else uniform_budget)
    if not 1 <= n_uniform <= len(lattice.grasp):
        raise ValueError("uniform_budget outside complete lattice")
    return {
        "anchor_query_ids": anchors,
        "query_winner": winner,
        "query_nonwinner": nonwinner,
        "lattice_uniform": _random(rng, list(range(len(lattice.grasp))), n_uniform),
        "anchor_full_lattice": full,
    }


@dataclass
class ActionRecord:
    row: np.ndarray
    condition: str
    parent_id: str
    query_id: int
    angle_id: int
    depth_id: int
    query_winner: bool
    raw_width_m: float
    width_mode: str
    original_width_m: float

    @property
    def candidate_id(self) -> str:
        return physical_action_id(self.row)


def build_p2_actions(
    lattice: Lattice,
    selected: Dict[str, List[int]],
    *,
    widths_m: Sequence[float] = (0.04, 0.06, 0.08, 0.10),
    insertion_width: str = "fixed",
    max_width_m: float = 0.1,
) -> List[ActionRecord]:
    """Construct controlled width and insertion sweeps at shared Q/A anchors.

    Width sweep: t,R,d held fixed, only w changes; predicted score is
    intentionally constant, since the existing head is not conditioned
    on executed width.

    Insertion sweep: t,R and (by default) w held fixed, d changes and
    the existing CDF utility is read at the matching depth bin.
    With insertion_width='predicted', w also changes, so label the
    intervention as a compound (d,w) sweep.
    """
    if insertion_width not in ("fixed", "predicted"):
        raise ValueError("insertion_width must be 'fixed' or 'predicted'")
    if not widths_m or any((not np.isfinite(w)) or (w <= 0) or (w > max_width_m+1e-8) for w in widths_m):
        raise ValueError("width grid must lie inside configured gripper width range")
    n = len(lattice.grasp)
    a, d = lattice.num_angles, lattice.num_depths
    result: List[ActionRecord] = []

    def push(row, condition, parent, source_idx, width_mode="predicted"):
        result.append(ActionRecord(
            row=np.asarray(row, dtype=np.float32).copy(),
            condition=condition,
            parent_id=parent,
            query_id=int(lattice.query_id[source_idx]),
            angle_id=int(lattice.angle_id[source_idx]),
            depth_id=int(lattice.depth_id[source_idx]),
            query_winner=bool(lattice.query_winner[source_idx]),
            raw_width_m=float(lattice.raw_width_m[source_idx]),
            original_width_m=float(lattice.grasp[source_idx, 1]),
            width_mode=width_mode,
        ))

    for condition in ("query_winner", "query_nonwinner", "lattice_uniform", "anchor_full_lattice"):
        for idx in selected.get(condition, []):
            if not 0 <= idx < n:
                raise ValueError(f"Candidate index {idx} outside lattice [0,{n})")
            parent = f"q{int(lattice.query_id[idx]):04d}-a{int(lattice.angle_id[idx]):02d}-d{int(lattice.depth_id[idx]):02d}"
            push(lattice.grasp[idx], condition, parent, idx)

    for idx in selected["query_winner"]:
        q = int(lattice.query_id[idx]); angle = int(lattice.angle_id[idx])
        base_width = float(lattice.grasp[idx, 1])
        family = f"q{q:04d}-a{angle:02d}-d{int(lattice.depth_id[idx]):02d}"
        # Include the decoder's width for a fair width-sweep reference.
        # Duplicate physical actions are deduplicated within a condition later.
        for width in [base_width, *widths_m]:
            row = lattice.grasp[idx].copy()
            row[1] = float(width)
            mode = "predicted" if abs(width - base_width) < 1e-7 else "fixed"
            push(row, "width_sweep", family, idx, width_mode=mode)

        # Same grasp center/orientation and width for every insertion bin.
        # Model CDF utility varies with d, while width is held fixed.
        for depth_id in range(d):
            source = q * (a * d) + angle * d + depth_id
            row = lattice.grasp[source].copy()
            if insertion_width == "fixed":
                row[1] = base_width
            push(row,
                 "insertion_sweep_fixed_width" if insertion_width == "fixed" else "insertion_sweep_predicted_width",
                 f"q{q:04d}-a{angle:02d}-w{base_width:.6f}", source,
                 width_mode="fixed" if insertion_width == "fixed" else "predicted")

    # Within each experimental condition, one physical action has one quality.
    # Keep the first occurrence (for width sweep, the predicted-width reference).
    distinct = []
    seen = set()
    for item in result:
        key = (item.condition, item.candidate_id)
        if key not in seen:
            distinct.append(item)
            seen.add(key)
    return distinct


def count_width_status(widths_m: Sequence[float], cap_m: float, tol_m: float=1e-6) -> dict:
    w = np.asarray(widths_m, dtype=float)
    if not (np.isfinite(w).all() and 0 < cap_m <= 0.3 and tol_m >= 0):
        raise ValueError("Invalid widths/cap/tolerance")
    return {
        "n": int(len(w)),
        "below_cap": int(np.sum(w < cap_m-tol_m)),
        "near_cap": int(np.sum(np.abs(w-cap_m) <= tol_m)),
        "above_cap": int(np.sum(w > cap_m+tol_m)),
        "max_width_m": float(np.max(w)) if len(w) else None,
        "cap_m": float(cap_m), "tol_m": float(tol_m),
    }
