#!/usr/bin/env python3
"""Build the two missing cells of the Rep full-path 2x2 action/score decomposition.

The existing full-query formal run already contains two diagonal cells:

    Stage1 action + Stage1 score  -> dump/stage1_native/native/...
    A1 action     + A1 score      -> dump/A1/<policy>/...

This script creates only the two hybrid cells, so the expensive official
GraspNet evaluation need not be repeated for redundant copies:

    A1 action     + Stage1 score  -> dump/decomp_A1Action_Stage1Score/<policy>/...
    Stage1 action + A1 score      -> dump/decomp_Stage1Action_A1Score/shared/...

"A1 score" for the native-action cell is the A1 score of the zero-offset
hypothesis for the same query. "Stage1 score" is the original score of that
query's regenerated native Stage-1 grasp.

The script never runs the detector or the CAD evaluator. It only rewrites
already-produced full-path inference arrays. It is intended for QUERY_LIMIT=0
formal runs and verifies the two pre-existing diagonal dumps when available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from rep_a_common import SPLITS, atomic_file, save_json
from rep_fullpath_runtime import case_key


ACTION_ONLY_METHOD = "decomp_A1Action_Stage1Score"
SCORE_ONLY_METHOD = "decomp_Stage1Action_A1Score"
BASELINE_METHOD = "stage1_native"
FULL_METHOD = "A1"
SHARED_POLICY = "shared"


def _as_strings(x) -> list[str]:
    return [str(v) for v in np.asarray(x).tolist()]


def _one_row(methods, policies, method: str, policy: str) -> int:
    hits = [
        i
        for i, (m, p) in enumerate(zip(methods, policies))
        if m == method and p == policy
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected exactly one output row for {method}/{policy}, got {hits}"
        )
    return hits[0]


def _check_grasp_array(name: str, x: np.ndarray, q: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (q, 17):
        raise ValueError(f"{name} must be [Q,17], got {x.shape}")
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return x


def compose_2x2(
    payload: Dict[str, np.ndarray],
    *,
    scorer: str = "A1",
    policies: Iterable[str] = ("fixed_0", "val_selected"),
    score_atol: float = 1e-6,
):
    """Return baseline, score-only, action-only and full arrays.

    The score-only cell is policy-independent because the physical action stays
    at the zero-offset/native hypothesis and no move threshold is applied to its
    score. Action-only/full cells depend on the A1 selection policy.
    """
    actions = np.asarray(payload["actions"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=bool)
    if actions.ndim != 3 or actions.shape[-1] != 17:
        raise ValueError(f"actions must be [K,Q,17], got {actions.shape}")
    if valid.shape != actions.shape[:2]:
        raise ValueError("valid/action shape mismatch")
    k, q = valid.shape
    zero = int(np.asarray(payload["zero_index"]).reshape(()))
    if not 0 <= zero < k or not valid[zero].all():
        raise ValueError("Invalid zero/native hypothesis")

    scorer_names = _as_strings(payload["scorer_names"])
    if scorer not in scorer_names:
        raise KeyError(f"Scorer {scorer!r} not present; available={scorer_names}")
    scorer_i = scorer_names.index(scorer)

    probabilities = np.asarray(payload["probabilities"])
    if probabilities.ndim != 4 or probabilities.shape[:3] != (len(scorer_names), k, q):
        raise ValueError(
            "probabilities must be [num_scorers,K,Q,6]-like; "
            f"got {probabilities.shape}"
        )
    if probabilities.shape[-1] < 1 or not np.isfinite(probabilities[scorer_i]).all():
        raise FloatingPointError("Invalid scorer probabilities")

    native_score = np.asarray(payload["original_native_score"], dtype=np.float32)
    if native_score.shape != (q,) or not np.isfinite(native_score).all():
        raise ValueError("original_native_score must be finite [Q]")

    # Baseline: original regenerated native Stage-1 action and score.
    native_action = actions[zero].copy()
    baseline = native_action.copy()
    baseline[:, 0] = native_score

    # Score-only: keep native physical action, replace only ranking confidence
    # by A1's zero-offset score for the same query.
    a1_zero_score = probabilities[scorer_i, zero].mean(-1).astype(np.float32)
    score_only = native_action.copy()
    score_only[:, 0] = a1_zero_score

    methods = _as_strings(payload["output_methods"])
    output_policies = _as_strings(payload["output_policies"])
    selected_all = np.asarray(payload["selected"], dtype=np.int64)
    rank_scores = np.asarray(payload["rank_scores"], dtype=np.float32)
    if selected_all.shape != rank_scores.shape or selected_all.shape[1] != q:
        raise ValueError("selected/rank_scores shape mismatch")

    per_policy = {}
    qq = np.arange(q)
    for policy in policies:
        row = _one_row(methods, output_policies, scorer, policy)
        sel = selected_all[row]
        if np.any((sel < 0) | (sel >= k)):
            raise ValueError(f"{scorer}/{policy} has out-of-range selection")
        if not valid[sel, qq].all():
            raise ValueError(f"{scorer}/{policy} selected an invalid hypothesis")

        selected_score = probabilities[scorer_i, sel, qq].mean(-1).astype(np.float32)
        if not np.allclose(selected_score, rank_scores[row], atol=score_atol, rtol=0):
            err = float(np.max(np.abs(selected_score - rank_scores[row])))
            raise RuntimeError(
                f"{scorer}/{policy} score replay mismatch; max_abs={err}"
            )

        # Action-only: A1 chooses the physical hypothesis, but ranking across
        # output grasps is still the original Stage-1 per-query ranking.
        action_only = actions[sel, qq].copy()
        action_only[:, 0] = native_score

        # Full A1: both selected physical action and A1 selected score.
        full = actions[sel, qq].copy()
        full[:, 0] = rank_scores[row]

        per_policy[str(policy)] = {
            "action_only": _check_grasp_array("action_only", action_only, q),
            "full": _check_grasp_array("full", full, q),
            "selected": sel.copy(),
        }

    return {
        "baseline": _check_grasp_array("baseline", baseline, q),
        "score_only": _check_grasp_array("score_only", score_only, q),
        "a1_zero_score": a1_zero_score,
        "per_policy": per_policy,
    }


def _existing_corner(
    root: Path,
    method: str,
    policy: str,
    mode: str,
    case: str,
    split: str,
    scene_id: int,
    camera: str,
    anno_id: int,
) -> Path:
    return (
        root
        / "dump"
        / method
        / policy
        / mode
        / case_key(case)
        / split
        / f"scene_{scene_id:04d}"
        / camera
        / f"{anno_id:04d}.npy"
    )


def _write_or_verify(path: Path, array: np.ndarray, overwrite: bool, atol: float) -> str:
    array = np.asarray(array, dtype=np.float32)
    if path.exists() and not overwrite:
        old = np.load(path, allow_pickle=False)
        if old.shape != array.shape or not np.allclose(old, array, atol=atol, rtol=0):
            diff = np.inf if old.shape != array.shape else float(np.max(np.abs(old - array)))
            raise RuntimeError(f"Existing decomposition dump differs: {path}; max_abs={diff}")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_file(path) as f:
        np.save(f, array, allow_pickle=False)
    return "written"


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", required=True, help="Existing QUERY_LIMIT=0 formal full-path run")
    p.add_argument("--splits", default="test_seen,test_similar,test_novel")
    p.add_argument("--cases", default="", help="Default: all cases in protocol.json")
    p.add_argument("--mode", default="joint")
    p.add_argument("--scorer", default="A1")
    p.add_argument("--policies", default="fixed_0,val_selected")
    p.add_argument("--camera", default="", help="Default: protocol camera")
    p.add_argument("--score-atol", type=float, default=1e-6)
    p.add_argument("--dump-atol", type=float, default=1e-6)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--skip-corner-check",
        action="store_true",
        help="Do not verify existing stage1_native and A1 full dumps",
    )
    args = p.parse_args()

    root = Path(args.work_root)
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if int(protocol.get("query_limit", -1)) != 0:
        raise RuntimeError(
            "2x2 official decomposition requires the full-query formal run "
            "(protocol query_limit=0)."
        )
    if args.mode not in protocol.get("modes", []):
        raise ValueError(f"Mode {args.mode!r} not in protocol modes={protocol.get('modes')}")

    splits = _parse_csv(args.splits)
    for split in splits:
        if split not in SPLITS or split == "train":
            raise ValueError(f"Unsupported split: {split}")
    cases = _parse_csv(args.cases) if args.cases else list(protocol["cases"])
    missing_cases = [c for c in cases if c not in protocol["cases"]]
    if missing_cases:
        raise ValueError(f"Cases not in formal protocol: {missing_cases}")
    policies = _parse_csv(args.policies)
    if not policies:
        raise ValueError("At least one A1 policy is required")
    camera = args.camera or protocol["camera"]

    input_files = []
    for split in splits:
        input_files.extend(
            sorted((root / "inference" / split).glob("scene_*/ann_*.npz"))
        )
    if not input_files:
        raise FileNotFoundError("No full-path inference NPZ files found")

    stats = {
        "processed_inputs": 0,
        "written": 0,
        "verified": 0,
        "corner_checks": 0,
        "per_split_case": {},
    }

    for path in input_files:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        split = str(d["split"])
        mode = str(d["mode"])
        case = str(d["case"])
        if split not in splits or mode != args.mode or case not in cases:
            continue

        scene_id = int(np.asarray(d["scene_id"]).reshape(()))
        anno_id = int(np.asarray(d["anno_id"]).reshape(()))
        parts = compose_2x2(
            d,
            scorer=args.scorer,
            policies=policies,
            score_atol=args.score_atol,
        )

        # Verify that our definitions reproduce the already existing diagonal
        # cells exactly enough before creating hybrids.
        if not args.skip_corner_check:
            baseline_path = _existing_corner(
                root, BASELINE_METHOD, "native", mode, case, split,
                scene_id, camera, anno_id
            )
            if not baseline_path.is_file():
                raise FileNotFoundError(f"Missing existing baseline dump: {baseline_path}")
            baseline_old = np.load(baseline_path, allow_pickle=False)
            if baseline_old.shape != parts["baseline"].shape or not np.allclose(
                baseline_old, parts["baseline"], atol=args.dump_atol, rtol=0
            ):
                raise RuntimeError(f"Baseline corner mismatch: {baseline_path}")
            stats["corner_checks"] += 1

            for policy in policies:
                full_path = _existing_corner(
                    root, FULL_METHOD, policy, mode, case, split,
                    scene_id, camera, anno_id
                )
                if not full_path.is_file():
                    raise FileNotFoundError(f"Missing existing full A1 dump: {full_path}")
                full_old = np.load(full_path, allow_pickle=False)
                expected = parts["per_policy"][policy]["full"]
                if full_old.shape != expected.shape or not np.allclose(
                    full_old, expected, atol=args.dump_atol, rtol=0
                ):
                    raise RuntimeError(f"Full A1 corner mismatch: {full_path}")
                stats["corner_checks"] += 1

        # Hybrid cell 1: native action + A1 zero-hypothesis score.
        score_only_path = _existing_corner(
            root, SCORE_ONLY_METHOD, SHARED_POLICY, mode, case, split,
            scene_id, camera, anno_id
        )
        status = _write_or_verify(
            score_only_path, parts["score_only"], args.overwrite, args.dump_atol
        )
        stats[status] += 1

        # Hybrid cell 2: A1 selected action + original Stage-1 score.
        for policy in policies:
            action_only_path = _existing_corner(
                root, ACTION_ONLY_METHOD, policy, mode, case, split,
                scene_id, camera, anno_id
            )
            status = _write_or_verify(
                action_only_path,
                parts["per_policy"][policy]["action_only"],
                args.overwrite,
                args.dump_atol,
            )
            stats[status] += 1

        stats["processed_inputs"] += 1
        key = f"{split}/{case}"
        stats["per_split_case"][key] = stats["per_split_case"].get(key, 0) + 1

    if stats["processed_inputs"] == 0:
        raise RuntimeError("No inference files matched requested split/mode/case filters")

    manifest = {
        "version": 1,
        "work_root": str(root),
        "source_protocol": protocol,
        "scorer": args.scorer,
        "mode": args.mode,
        "splits": splits,
        "cases": cases,
        "policies": policies,
        "semantics": {
            "baseline": {
                "action": "Stage1 native / zero-offset action",
                "score": "Stage1 original per-query score",
                "existing_dump": "stage1_native/native",
            },
            "action_only": {
                "action": "A1-selected physical hypothesis",
                "score": "Stage1 original score of the same query",
                "generated_method": ACTION_ONLY_METHOD,
            },
            "score_only": {
                "action": "Stage1 native / zero-offset action",
                "score": "A1 zero-offset hypothesis mean CDF probability",
                "generated_method": SCORE_ONLY_METHOD,
                "generated_policy": SHARED_POLICY,
                "note": "Policy-independent; no move threshold is applied to the native action.",
            },
            "full": {
                "action": "A1-selected physical hypothesis",
                "score": "A1 selected-hypothesis score",
                "existing_dump": "A1/<policy>",
            },
        },
        "stats": stats,
        "next_step": (
            "Run eval_rep_fullpath_official.py (or scripts/run_rep_fullpath.sh "
            "with PHASES=official, RESUME=1) on the same WORK_ROOT. Existing "
            "official cells resume/skip; only the new hybrid folders need evaluation."
        ),
    }
    save_json(root / "decomposition_2x2_manifest.json", manifest)

    print(
        "[2X2] built hybrids: "
        f"inputs={stats['processed_inputs']} written={stats['written']} "
        f"verified={stats['verified']} corner_checks={stats['corner_checks']}"
    )
    print(f"[2X2] manifest: {root/'decomposition_2x2_manifest.json'}")
    print(
        "[2X2] evaluate with: PHASES=official RESUME=1 "
        "bash scripts/run_rep_fullpath.sh"
    )


if __name__ == "__main__":
    main()
