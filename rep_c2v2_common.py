"""Rep-C2-v2 shared definitions: full-path proposal labels and RGB verification.

The v1 Rep-C2 verifier was trained on fixed physical actions while perturbing
only the reader observation. C2-v2 instead learns on the actual full-path
problem: joint depth error regenerates the Stage-1 anchor and A1 fixed-0
proposes a physical translation correction.

C2-v2 deliberately separates the *proposal* from *verification*:
  proposal     = frozen A1 fixed-0 candidate
  verification = decide beneficial / equivalent / harmful vs native

The verifier can use a pre-enhancer RGB feature map. This map is captured
before any metric-depth spatial enhancement, so it is not another read of the
same predicted metric geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from rep_a_common import cdf_targets
from rep_fullpath_runtime import case_key


VARIANTS = ("score", "rgb", "rgb_only")
CLASS_NAMES = ("harmful", "equivalent", "beneficial")
HARMFUL, EQUIVALENT, BENEFICIAL = 0, 1, 2


def strings(x) -> list[str]:
    return [str(v) for v in np.asarray(x).tolist()]


def row_index(methods, policies, method="A1", policy="fixed_0") -> int:
    hits = [
        i for i, (m, p) in enumerate(zip(strings(methods), strings(policies)))
        if m == method and p == policy
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one {method}/{policy} row, got {hits}")
    return hits[0]


def scorer_index(names, scorer="A1") -> int:
    names = strings(names)
    if scorer not in names:
        raise KeyError(f"{scorer!r} not in scorer_names={names}")
    return names.index(scorer)


def class_from_delta(delta, eps=1e-7):
    delta = np.asarray(delta, np.float32)
    out = np.full(delta.shape, EQUIVALENT, np.int64)
    out[delta < -eps] = HARMFUL
    out[delta > eps] = BENEFICIAL
    return out


def compact_from_source(payload, labels, scorer="A1", policy="fixed_0", eps=1e-7):
    """Extract moved A1 proposals and fresh labels from an existing full-path run.

    Returns only queries where A1 fixed-0 actually proposes a non-native action.
    The exact-evaluation file must contain native and proposed actions.
    """
    actions = np.asarray(payload["actions"], np.float32)
    valid = np.asarray(payload["valid"], bool)
    if actions.ndim != 3 or actions.shape[-1] != 17 or valid.shape != actions.shape[:2]:
        raise ValueError("Bad full-path action/valid shape")
    k, q = valid.shape
    z = int(np.asarray(payload["zero_index"]).reshape(()))
    qq = np.arange(q)
    row = row_index(payload["output_methods"], payload["output_policies"], scorer, policy)
    proposal = np.asarray(payload["selected"][row], np.int64)
    move = proposal != z
    ids = np.flatnonzero(move)

    si = scorer_index(payload["scorer_names"], scorer)
    probs = np.asarray(payload["probabilities"], np.float32)
    if probs.ndim != 4 or probs.shape[:3] != (len(strings(payload["scorer_names"])), k, q):
        raise ValueError(f"Bad probability tensor {probs.shape}")

    evaluated = np.asarray(labels["evaluated_mask"], bool)
    friction = np.asarray(labels["friction"], np.float32)
    if evaluated.shape != valid.shape or friction.shape != valid.shape:
        raise ValueError("Bad exact-evaluation shape")
    if ids.size:
        if not evaluated[z, ids].all() or not evaluated[proposal[ids], ids].all():
            raise RuntimeError(
                "C2-v2 needs fresh exact labels for native + A1 fixed-0 proposal. "
                "Run evaluate_rep_fullpath.py with LABEL_SCOPE=selected or all."
            )
    y = cdf_targets(friction)
    util = y.mean(-1)
    native_u = util[z, ids] if ids.size else np.empty(0, np.float32)
    prop_u = util[proposal[ids], ids] if ids.size else np.empty(0, np.float32)
    delta = (prop_u-native_u).astype(np.float32)
    offsets = np.asarray(payload["offsets_mm"], np.float32)
    return {
        "query_pos": ids.astype(np.int64),
        "query_ids": np.asarray(payload["query_ids"], np.int64)[ids],
        "actions": (
            np.stack((actions[z, ids], actions[proposal[ids], ids]), axis=0)
            if ids.size else np.empty((2,0,17), np.float32)
        ),
        "probabilities": (
            np.stack((probs[si,z,ids], probs[si,proposal[ids],ids]), axis=0)
            if ids.size else np.empty((2,0,6), np.float32)
        ),
        "proposal_k": proposal[ids].astype(np.int64),
        "offsets_mm": offsets[proposal[ids]].astype(np.float32),
        "original_native_score": np.asarray(payload["original_native_score"], np.float32)[ids],
        "native_utility": native_u.astype(np.float32),
        "proposal_utility": prop_u.astype(np.float32),
        "delta_utility": delta,
        "target_class": class_from_delta(delta, eps),
    }


def source_eval_path(source_root, inference_path, split):
    p = Path(inference_path)
    return Path(source_root) / "evaluation" / split / p.parent.name / p.name


def frame_cache_path(cache_root, split, sid, aid):
    return Path(cache_root) / split / f"scene_{int(sid):04d}" / f"ann_{int(aid):04d}.npz"


def source_files(source_root, split, mode="joint", cases=("nominal","bias:-20","bias:20")):
    root = Path(source_root)
    wanted = set(cases)
    result = []
    for path in sorted((root/"inference"/split).glob("scene_*/ann_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            m, c = str(z["mode"]), str(z["case"])
        if m == mode and c in wanted:
            result.append(path)
    if not result:
        raise FileNotFoundError(f"No {mode}/{wanted} inference files under {root}/{split}")
    return result


def select_move_subset(margins, limit):
    """Deterministic mixed high-margin/uniform subset of moved proposals."""
    margins = np.asarray(margins, np.float64)
    n = len(margins)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)
    rank = np.argsort(-margins, kind="stable")
    n_top = limit // 2
    top = rank[:n_top]
    tail = rank[n_top:]
    need = limit-len(top)
    if need <= 0:
        return top
    pos = np.rint(np.linspace(0, len(tail)-1, need)).astype(int)
    return np.concatenate((top, tail[pos])).astype(np.int64)


@dataclass
class GateMetrics:
    threshold: float
    mean_gain: float
    accept_rate: float
    beneficial_retention: float
    harmful_rejection: float
    accept_precision: float
    beneficial_count: int
    harmful_count: int
    equivalent_count: int


def gate_metrics(prob_beneficial, delta, threshold):
    p = np.asarray(prob_beneficial, np.float64)
    d = np.asarray(delta, np.float64)
    if p.shape != d.shape:
        raise ValueError("gate score/delta shape mismatch")
    accept = p >= threshold
    benefit = d > 1e-7
    harm = d < -1e-7
    equiv = ~(benefit | harm)
    gain = np.where(accept, d, 0.)
    return GateMetrics(
        threshold=float(threshold),
        mean_gain=float(gain.mean()) if len(gain) else 0.,
        accept_rate=float(accept.mean()) if len(accept) else 0.,
        beneficial_retention=float((accept&benefit).sum()/max(1,benefit.sum())),
        harmful_rejection=float(1-(accept&harm).sum()/max(1,harm.sum())),
        accept_precision=float((accept&benefit).sum()/max(1,accept.sum())),
        beneficial_count=int(benefit.sum()),
        harmful_count=int(harm.sum()),
        equivalent_count=int(equiv.sum()),
    )


def tune_threshold(prob_beneficial, delta):
    rows = [gate_metrics(prob_beneficial, delta, t) for t in np.linspace(0,1,101)]
    # Primary target is actual relative utility on full-path proposals.
    # Tie-break toward retaining useful corrections, then rejecting harm.
    best = max(
        rows,
        key=lambda x: (
            x.mean_gain,
            x.beneficial_retention,
            x.harmful_rejection,
            -x.accept_rate,
        ),
    )
    return best, rows


def official_dump_path(root, method, mode, case, split, camera, sid, aid):
    return (
        Path(root)/"dump"/method/"verified"/mode/case_key(case)/split
        /f"scene_{int(sid):04d}"/camera/f"{int(aid):04d}.npy"
    )
