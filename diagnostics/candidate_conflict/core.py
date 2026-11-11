"""Pure NumPy utilities for GraspNet NMS-conflict and quality-oracle analysis.

An edge means only that the official GraspNet NMS geometry rule would suppress
one candidate when the other is visited first. It is NOT a claim that the two
grasps are physically incompatible.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MU = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float64)
SCHEMA = "graspnet-candidate-conflict-v1"
GRASP_DIM = 17


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def frame_key(scene_id: int, anno_id: int) -> str:
    return f"scene_{int(scene_id):04d}_{int(anno_id):04d}"


def load_grasps(path: str | Path, max_width: float = 0.1) -> tuple[np.ndarray, dict]:
    g = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if g.ndim != 2 or g.shape[1] != GRASP_DIM:
        raise ValueError(f"{path}: expected N x 17 grasp array, got {g.shape}")
    if not np.isfinite(g).all():
        raise ValueError(f"{path}: non-finite grasp values")
    raw_width = g[:, 1].copy()
    g = g.copy()
    g[:, 1] = np.clip(g[:, 1], 0.0, float(max_width))
    stats = {
        "num_candidates": int(len(g)),
        "num_width_clipped_low": int(np.sum(raw_width < 0)),
        "num_width_clipped_high": int(np.sum(raw_width > max_width)),
    }
    return g, stats


def physical_ids(grasps: np.ndarray) -> np.ndarray:
    """Hash executed geometry (width/height/depth/R/t), excluding score/object id."""
    g = np.asarray(grasps, dtype="<f4")
    if g.ndim != 2 or g.shape[1] != GRASP_DIM or not np.isfinite(g).all():
        raise ValueError("Expected finite N x 17 grasps")
    return np.asarray([
        hashlib.sha256(row[1:16].tobytes()).hexdigest() for row in g
    ], dtype="<U64")


def quality_bits(friction: np.ndarray, collision_or_empty: np.ndarray) -> np.ndarray:
    f = np.asarray(friction, dtype=np.float64).reshape(-1)
    bad = np.asarray(collision_or_empty, dtype=bool).reshape(-1)
    if f.shape != bad.shape or not np.isfinite(f).all():
        raise ValueError("Invalid exact-action labels")
    return ((f[:, None] > 0.0) & (f[:, None] <= MU[None, :] + 1e-6) &
            (~bad[:, None])).astype(np.uint8)


def build_conflict_edges(
    grasps: np.ndarray,
    translation_thresh: float = 0.03,
    rotation_thresh_rad: float = np.deg2rad(30.0),
) -> np.ndarray:
    """Build the undirected geometric conflict graph used by GraspNet NMS.

    Official GraspGroup.nms delegates to grasp_nms. This implements the
    corresponding translation <= threshold AND geodesic rotation <= threshold
    relation; formal runs must verify greedy output against the installed
    GraspGroup.nms implementation for every frame.
    """
    g = np.asarray(grasps, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != GRASP_DIM:
        raise ValueError("Expected N x 17 grasps")
    n = len(g)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    xyz = g[:, 13:16]
    try:
        from scipy.spatial import cKDTree
        pairs = cKDTree(xyz).query_pairs(float(translation_thresh), output_type="ndarray")
    except ImportError as exc:
        raise RuntimeError("scipy is required for full conflict-graph construction") from exc
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    rot = g[:, 4:13].reshape(-1, 3, 3)
    rel = np.matmul(np.transpose(rot[pairs[:, 0]], (0, 2, 1)), rot[pairs[:, 1]])
    trace = np.trace(rel, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    keep = angle <= float(rotation_thresh_rad) + 1e-12
    edges = pairs[keep]
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return edges


def connected_components(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(num_nodes, dtype=np.int64)
    rank = np.zeros(num_nodes, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        ru, rv = find(int(u)), find(int(v))
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
    roots = np.asarray([find(i) for i in range(num_nodes)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def adjacency_lists(num_nodes: int, edges: np.ndarray) -> list[list[int]]:
    adj = [[] for _ in range(num_nodes)]
    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def greedy_nms(
    scores: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy NMS over a precomputed conflict graph, matching score order.

    Returns (kept_indices_in_visit_order, suppressor_index_per_node,
    visit_order). The score ordering is np.argsort(score)[::-1], as used by
    GraspGroup.sort_by_score in the official API.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("NMS scores must be finite")
    n = len(scores)
    adj = adjacency_lists(n, edges)
    order = np.argsort(scores)[::-1]
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    suppressed = np.zeros(n, dtype=bool)
    suppressor = np.full(n, -1, dtype=np.int64)
    kept = []
    for i in order:
        i = int(i)
        if suppressed[i]:
            continue
        kept.append(i)
        for j in adj[i]:
            if not suppressed[j] and position[j] > position[i]:
                suppressed[j] = True
                suppressor[j] = i
    return np.asarray(kept, dtype=np.int64), suppressor, order.astype(np.int64)


def verify_official_nms(grasps: np.ndarray, kept: np.ndarray,
                        translation_thresh: float = 0.03,
                        rotation_thresh_rad: float = np.deg2rad(30.0),
                        atol: float = 1e-6) -> float:
    """Return max absolute parity error; raise if installed GraspNet NMS differs."""
    try:
        from graspnetAPI import GraspGroup
    except Exception as exc:
        raise RuntimeError("Run official NMS parity check inside the GraspNet API environment") from exc
    official = GraspGroup(np.asarray(grasps, dtype=np.float64).copy()).nms(
        float(translation_thresh), float(rotation_thresh_rad)
    ).grasp_group_array
    local = np.asarray(grasps, dtype=np.float64)[np.asarray(kept, dtype=np.int64)]
    if official.shape != local.shape:
        raise RuntimeError(f"Official/local NMS count mismatch: {official.shape} vs {local.shape}")
    diff = float(np.max(np.abs(official - local))) if official.size else 0.0
    if not np.isfinite(diff) or diff > atol:
        raise RuntimeError(f"Official/local NMS parity failed: max_abs={diff:.9g} > {atol}")
    return diff


def graph_stats(num_nodes: int, edges: np.ndarray, component_id: np.ndarray,
                kept: np.ndarray) -> dict:
    component_sizes = np.bincount(component_id, minlength=int(component_id.max()) + 1) if num_nodes else np.zeros(0, dtype=int)
    degree = np.zeros(num_nodes, dtype=np.int64)
    if len(edges):
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(edges)),
        "num_components": int(len(component_sizes)),
        "num_nms_survivors": int(len(kept)),
        "num_suppressed": int(num_nodes - len(kept)),
        "mean_degree": float(degree.mean()) if num_nodes else 0.0,
        "p95_degree": float(np.percentile(degree, 95)) if num_nodes else 0.0,
        "max_degree": int(degree.max()) if num_nodes else 0,
        "component_size_p50": float(np.median(component_sizes)) if len(component_sizes) else 0.0,
        "component_size_p90": float(np.percentile(component_sizes, 90)) if len(component_sizes) else 0.0,
        "component_size_max": int(component_sizes.max()) if len(component_sizes) else 0,
        "component_sizes": component_sizes.astype(int).tolist(),
    }


def q_priority(q_count: np.ndarray, native_scores: np.ndarray) -> np.ndarray:
    q = np.asarray(q_count, dtype=np.float64).reshape(-1)
    s = np.asarray(native_scores, dtype=np.float64).reshape(-1)
    if q.shape != s.shape:
        raise ValueError("Quality/score shape mismatch")
    if len(s) == 0:
        return s.copy()
    lo, hi = float(np.min(s)), float(np.max(s))
    normalized = (s - lo) / max(hi - lo, 1e-12)
    # Six-threshold quality is the primary key; native confidence only breaks ties.
    return q + 1e-6 * normalized


def official_ap_matrix(
    grasps: np.ndarray,
    assigned_obj: np.ndarray,
    collision_or_empty: np.ndarray,
    friction: np.ndarray,
    selected: np.ndarray,
    ranking_scores: np.ndarray,
    top_k: int = 50,
    per_object_top: int = 10,
) -> np.ndarray:
    """Reproduce GraspNetEval.eval_scene's per-frame selection and AP calculation.

    Input candidates are the pre-official-NMS dump after the experiment's
    sensor-collision prefilter. Exact-action labels are evaluated before NMS,
    which is valid because object assignment/physical quality are candidate-local.
    """
    g = np.asarray(grasps, dtype=np.float64)
    obj = np.asarray(assigned_obj, dtype=np.int64).reshape(-1)
    bad = np.asarray(collision_or_empty, dtype=bool).reshape(-1)
    friction = np.asarray(friction, dtype=np.float64).reshape(-1)
    ranking = np.asarray(ranking_scores, dtype=np.float64).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    n = len(g)
    if g.shape != (n, GRASP_DIM) or any(x.shape != (n,) for x in (obj, bad, friction, ranking)):
        raise ValueError("Official AP simulation input shape mismatch")
    if not np.isfinite(ranking).all() or not np.isfinite(friction).all():
        raise ValueError("Non-finite ranking/quality labels")
    out = np.zeros((top_k, len(MU)), dtype=np.float64)
    if len(selected) == 0:
        return out

    # GraspNet eval_grasp: assign by object first, then score-sort and top-10/object.
    pre_by_obj = []
    for object_id in np.unique(obj[selected]):
        ids = selected[obj[selected] == object_id]
        order = np.argsort(ranking[ids])[::-1]
        pre_by_obj.append(ids[order[:per_object_top]])
    nonempty = [x for x in pre_by_obj if len(x)]
    if not nonempty:
        return out
    all_pre = np.concatenate(nonempty)
    remain_order = np.argsort(ranking[all_pre])[::-1]
    min_index = min(top_k - 1, len(remain_order) - 1)
    min_score = float(ranking[all_pre[remain_order[min_index]]])

    # Official implementation uses >= threshold, so ties can leave > top_k rows.
    kept_by_obj = [x[ranking[x] >= min_score] for x in pre_by_obj if len(x)]
    ranked = np.concatenate([x for x in kept_by_obj if len(x)])
    final_order = np.argsort(-ranking[ranked])
    ranked = ranked[final_order]

    valid = (~bad[ranked]) & (friction[ranked] > 0.0)
    for j, mu in enumerate(MU):
        success = valid & (friction[ranked] <= mu + 1e-6)
        cumulative = np.cumsum(success.astype(np.float64))
        for k in range(top_k):
            out[k, j] = cumulative[min(k, len(cumulative) - 1)] / float(k + 1) if len(cumulative) else 0.0
    return out


def component_fixed_cardinality_oracle(
    num_nodes: int,
    edges: np.ndarray,
    component_id: np.ndarray,
    baseline_keep: np.ndarray,
    q_count: np.ndarray,
    native_scores: np.ndarray,
    max_component_nodes: int = 20,
) -> tuple[np.ndarray, dict]:
    """Exact max-Q independent set at baseline NMS cardinality per small component.

    Large components are left at their baseline NMS selection. The returned set
    is always feasible and has exactly the same survivor count in each component.
    If any component exceeds max_component_nodes, this is a hybrid diagnostic,
    NOT a globally exact MWIS oracle.
    """
    q = np.asarray(q_count, dtype=np.int64).reshape(-1)
    score = np.asarray(native_scores, dtype=np.float64).reshape(-1)
    base = np.asarray(baseline_keep, dtype=np.int64).reshape(-1)
    if q.shape != (num_nodes,) or score.shape != (num_nodes,) or component_id.shape != (num_nodes,):
        raise ValueError("Component oracle input shape mismatch")
    adj = adjacency_lists(num_nodes, edges)
    chosen = []
    exact_components = 0
    fallback_components = 0
    exact_nodes = 0
    total_nodes = num_nodes

    for cid in np.unique(component_id):
        nodes = np.flatnonzero(component_id == cid)
        base_local = base[component_id[base] == cid]
        target_k = len(base_local)
        if target_k <= 0:
            raise RuntimeError("A non-empty conflict component must retain at least one NMS node")
        if len(nodes) > max_component_nodes:
            chosen.extend(base_local.tolist())
            fallback_components += 1
            continue

        local_index = {int(node): i for i, node in enumerate(nodes)}
        local_adj = []
        for node in nodes:
            mask = 0
            for other in adj[int(node)]:
                if other in local_index:
                    mask |= 1 << local_index[other]
            local_adj.append(mask)

        @lru_cache(maxsize=None)
        def solve(mask: int, k: int):
            if k == 0:
                return (0, 0.0, ())
            if mask.bit_count() < k:
                return None
            # Choose the remaining vertex with the highest induced degree.
            candidates = [i for i in range(len(nodes)) if mask & (1 << i)]
            v = max(candidates, key=lambda i: (local_adj[i] & mask).bit_count())
            without_v = mask & ~(1 << v)
            best = solve(without_v, k)
            include_mask = without_v & ~local_adj[v]
            inc = solve(include_mask, k - 1)
            if inc is not None:
                candidate = (inc[0] + int(q[nodes[v]]),
                             inc[1] + float(score[nodes[v]]),
                             tuple(sorted(inc[2] + (int(nodes[v]),))))
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[1] > best[1] + 1e-12
                ) or (
                    candidate[0] == best[0] and abs(candidate[1] - best[1]) <= 1e-12
                    and candidate[2] < best[2]
                ):
                    best = candidate
            return best

        result = solve((1 << len(nodes)) - 1, target_k)
        if result is None:
            raise RuntimeError("Fixed-cardinality independent set unexpectedly infeasible")
        chosen.extend(result[2])
        exact_components += 1
        exact_nodes += len(nodes)

    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    # Validate feasibility and count preservation.
    if len(chosen) != len(base):
        raise RuntimeError("Component oracle changed total NMS survivor count")
    selected_set = set(chosen.tolist())
    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        if int(u) in selected_set and int(v) in selected_set:
            raise RuntimeError("Component oracle produced a conflicting set")
    return chosen, {
        "exact_components": exact_components,
        "fallback_components": fallback_components,
        "exact_nodes": exact_nodes,
        "total_nodes": total_nodes,
        "exact_node_coverage": float(exact_nodes / total_nodes) if total_nodes else 1.0,
        "max_component_nodes": int(max_component_nodes),
        "is_globally_exact": fallback_components == 0,
        "same_survivor_count": True,
    }


def scene_cluster_bootstrap(scene_values: dict[int, float], n_boot: int = 5000,
                            seed: int = 0) -> dict:
    scenes = np.asarray(sorted(scene_values), dtype=np.int64)
    values = np.asarray([scene_values[int(s)] for s in scenes], dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n_scenes": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ids = rng.integers(0, len(values), size=len(values))
        boot[i] = values[ids].mean()
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "n_scenes": int(len(values)),
        "scene_values": {str(int(s)): float(v) for s, v in zip(scenes, values)},
    }
