"""P0 pure action-set attribution: AP-stage replay and changed-competitor diagnostics.

All quantities here are *pathwise counterfactuals*, not order-independent
causal effects. The only labels required are the union of compiled NMS
survivors across compared models. Physical candidate identities are fixed.
"""
from __future__ import annotations

import hashlib
import numpy as np

from diagnostics.candidate_conflict.core import (
    official_ap_matrix, build_conflict_edges,
)
from .core import physical_ids


def common_bank(runs):
    """Reject any unpaired frame before comparing scores/NMS.

    Dict[run_id, post-sensor-pre-NMS N×17], in the same query order. For
    rank_executed the executed geometry is frozen by experimental contract.
    """
    if len(runs) < 2:
        raise ValueError("Need at least two models")
    original = None
    for name, values in runs.items():
        g = np.asarray(values)
        if g.ndim != 2 or g.shape[1] != 17 or not np.isfinite(g).all():
            raise ValueError(f"Invalid N×17 dump for {name}")
        if original is None:
            original = g.copy()
        elif g.shape != original.shape or not np.array_equal(g[:, 1:16], original[:, 1:16]):
            raise ValueError("Physical candidates differ: rank_executed cannot change geometry/filter")
    return original


def compiled_keep(grasps, run_nms):
    """Use the installed compiled NMS, tagging the otherwise unused object id."""
    g = np.asarray(grasps, dtype=np.float64)
    if g.shape == (0, 17):
        return np.empty(0, dtype=np.int64)
    tagged = g.copy()
    tagged[:, 16] = np.arange(len(g))
    kept = np.asarray(run_nms(tagged.copy()), dtype=np.float64)
    if kept.ndim != 2 or kept.shape[1] != 17:
        raise RuntimeError("Compiled NMS returned malformed result")
    ids = kept[:, 16].astype(np.int64)
    if (np.any(ids < 0) or np.any(ids >= len(g)) or
            len(np.unique(ids)) != len(ids) or
            not np.array_equal(tagged[ids, :16], kept[:, :16])):
        raise RuntimeError("Compiled NMS changed action identity/score")
    return ids


def final_ids(keep, scores, objects, per_object=10, k=50):
    """Mirror official per-object top10/global-score-threshold behavior.

    Sort/tie policy matches diagnostics.candidate_conflict.core.official_ap_matrix.
    The returned length can exceed k if exact threshold ties exist.
    """
    keep = np.asarray(keep, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64)
    objects = np.asarray(objects, dtype=np.int64)
    if len(keep) == 0:
        return np.empty(0, dtype=np.int64)
    grouped = []
    for obj in np.unique(objects[keep]):
        ids = keep[objects[keep] == obj]
        grouped.append(ids[np.argsort(scores[ids])[::-1][:per_object]])
    pool = np.concatenate(grouped)
    rank = np.argsort(scores[pool])[::-1]
    threshold = float(scores[pool[rank[min(k - 1, len(rank) - 1)]]])
    chosen = np.concatenate([arr[scores[arr] >= threshold] for arr in grouped])
    return chosen[np.argsort(-scores[chosen])]


def ap_from_fixed_membership(ranked, invalid, friction, k=50):
    """Evaluate a fixed final membership in a supplied order WITHOUT re-trimming.

    Official AP prefixes count missing positions as failures. This is the
    exact fixed-membership ranking counterfactual, even when official score
    ties made the baseline final set larger than K.
    """
    ranked=np.asarray(ranked,dtype=np.int64).reshape(-1)
    f=np.asarray(friction,dtype=np.float64)[ranked]
    bad=np.asarray(invalid,dtype=bool)[ranked]
    if not np.isfinite(f).all():
        raise ValueError("Nonfinite quality labels")
    out=np.zeros((k,6),dtype=np.float64)
    for j,mu in enumerate((.2,.4,.6,.8,1.,1.2)):
        success=(~bad)&(f>0)&(f<=mu+1e-6)
        cumulative=np.cumsum(success,dtype=np.float64)
        for idx in range(k):
            if len(cumulative):
                out[idx,j]=cumulative[min(idx,len(cumulative)-1)]/(idx+1)
    return out


def arrange_labels(num_nodes, selected, eval_result):
    """Fill only labeled NMS-surviving nodes. Unlabeled ones cannot be selected."""
    ids = np.asarray(selected, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Label evaluation requires unique node indices")
    if ids.size and (ids.min() < 0 or ids.max() >= num_nodes):
        raise ValueError("Invalid label indices")
    assigned = np.full(num_nodes, -1, dtype=np.int64)
    bad = np.ones(num_nodes, dtype=bool)
    friction = np.full(num_nodes, -1., dtype=np.float64)
    for field in ("assigned_obj", "collision_or_empty", "friction"):
        if len(np.asarray(getattr(eval_result, field))) != len(ids):
            raise ValueError(f"Evaluator {field} length mismatch")
    assigned[ids] = np.asarray(eval_result.assigned_obj, dtype=np.int64)
    bad[ids] = np.asarray(eval_result.collision_or_empty, dtype=bool)
    friction[ids] = np.asarray(eval_result.friction, dtype=np.float64)
    if not np.isfinite(friction).all():
        raise ValueError("Nonfinite exact friction values")
    return assigned, bad, friction


def q_from_labels(friction, invalid):
    f = np.asarray(friction, dtype=np.float64).reshape(-1)
    bad = np.asarray(invalid, dtype=bool).reshape(-1)
    if f.shape != bad.shape:
        raise ValueError("Invalid label arrays")
    return (((f[:, None] > 0.) &
            (f[:, None] <= np.array([.2, .4, .6, .8, 1., 1.2])[None] + 1e-6) &
            (~bad[:, None])).sum(axis=1).astype(np.uint8))


def pairwise_changed_conflicts(base, lost, gained):
    """Geometry-only cross edges between NMS-lost and newly kept actions.

    Never call an edge an actual suppressor assignment. Compile-check the
    returned edges separately before making any physical explanation.
    """
    lost = np.asarray(sorted(set(map(int, lost))), dtype=np.int64)
    gained = np.asarray(sorted(set(map(int, gained))), dtype=np.int64)
    if not len(lost) or not len(gained):
        return np.zeros((0, 2), dtype=np.int64)
    ix = np.concatenate([lost, gained])
    pairs = build_conflict_edges(np.asarray(base)[ix])
    keep = (pairs[:, 0] < len(lost)) & (pairs[:, 1] >= len(lost))
    # build_conflict_edges gives ascending indices, so the opposite orientation
    # is not possible in this concatenation.
    matched = pairs[keep]
    return np.stack((ix[matched[:, 0]], ix[matched[:, 1]]), axis=-1).astype(np.int64)


def replay(base, scores_left, scores_right, keep_left, keep_right,
           assigned, invalid, friction, *, official_left=None,
           official_right=None, tolerance=1e-6):
    """Four stages: original -> same final set, new score -> same NMS set,
    new score/reselect budgets -> new NMS set, new score. Not causal Shapley.
    """
    g = np.asarray(base)
    n = len(g)
    sl = np.asarray(scores_left, dtype=np.float64).reshape(-1)
    sr = np.asarray(scores_right, dtype=np.float64).reshape(-1)
    kl = np.asarray(keep_left, dtype=np.int64).reshape(-1)
    kr = np.asarray(keep_right, dtype=np.int64).reshape(-1)
    if sl.shape != (n,) or sr.shape != (n,) or not np.isfinite(sl).all() or not np.isfinite(sr).all():
        raise ValueError("Score vector mismatch")
    labeled = np.zeros(n, dtype=bool)
    labeled[np.union1d(kl, kr)] = True
    if not np.isfinite(friction).all():
        raise ValueError("Nonfinite labels")
    if not len(kl) or not len(kr):
        # Empty surviving sets are still evaluable; exact labels may be empty.
        pass
    lfinal = final_ids(kl, sl, assigned)
    rfinal = final_ids(kr, sr, assigned)
    ahyp = final_ids(kl, sr, assigned)
    all_required = np.union1d(np.union1d(kl, kr), np.union1d(lfinal, ahyp))
    if not np.all(labeled[all_required]):
        raise RuntimeError("Missing exact labels for AP-participating nodes")

    def ap(sel, rank):
        return official_ap_matrix(g, assigned, invalid, friction, sel, rank)
    a0 = ap(kl, sl)         # Original native model-side choice and score
    fixed_rank=lfinal[np.argsort(-sr[lfinal])]
    a1 = ap_from_fixed_membership(fixed_rank, invalid, friction)  # No cap/NMS recomputation
    a2 = ap(kl, sr)         # Fix NMS survivors; reapply per-obj/global budgets
    a3 = ap(kr, sr)         # New NMS survivors, new rank
    for name, current, expected in (
        ("left", a0, official_left), ("right", a3, official_right)):
        if expected is not None:
            expected = np.asarray(expected)
            if current.shape != expected.shape:
                raise RuntimeError(f"{name} official AP shape differs")
            error = float(np.max(np.abs(current - expected)))
            if error > tolerance:
                raise RuntimeError(f"{name} official AP parity failed: {error:.9g}")
    q = q_from_labels(friction, invalid)
    akeep, bkeep = set(kl.tolist()), set(kr.tolist())
    afinal, bfinal = set(lfinal.tolist()), set(rfinal.tolist())
    nms_lost, nms_gained = akeep - bkeep, bkeep - akeep
    final_lost, final_gained = afinal - bfinal, bfinal - afinal
    cross = pairwise_changed_conflicts(g, nms_lost, nms_gained)
    actual_inversions = [
        (int(i), int(j)) for i, j in cross
        if q[i] > 0 and q[j] == 0 and sl[i] > sl[j] and sr[i] < sr[j]
    ]
    stages = np.array([a.mean() * 100. for a in (a0, a1, a2, a3)])
    return dict(
        stages_pp=stages,
        rank_fixed_final_pp=float(stages[1] - stages[0]),
        quota_reselection_pp=float(stages[2] - stages[1]),
        nms_set_replacement_pp=float(stages[3] - stages[2]),
        total_pp=float(stages[3] - stages[0]),
        num_source_candidates=n,
        nms_left=len(kl), nms_right=len(kr),
        final_left=len(lfinal), final_right=len(rfinal),
        nms_lost=len(nms_lost), nms_gained=len(nms_gained),
        final_lost=len(final_lost), final_gained=len(final_gained),
        final_lost_q_mean=float(q[list(final_lost)].mean()/6) if final_lost else None,
        final_gained_q_mean=float(q[list(final_gained)].mean()/6) if final_gained else None,
        lost_positive_nms=int(np.sum(q[list(nms_lost)] > 0)) if nms_lost else 0,
        gained_zero_nms=int(np.sum(q[list(nms_gained)] == 0)) if nms_gained else 0,
        geometric_lost_gained_conflicts=len(cross),
        positive_lost_zero_gained_score_inversions=len(actual_inversions),
        inversion_pairs=actual_inversions,
        left_final_ids=lfinal,
        right_final_ids=rfinal,
        left_nms_keep=kl,
        right_nms_keep=kr,
        quality=q,
        conflicts=cross,
    )
