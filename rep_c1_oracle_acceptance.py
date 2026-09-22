#!/usr/bin/env python3
"""Rep-C1: oracle upper bound for accepting/rejecting an A1 correction.

This experiment does NOT learn a gate. It fixes the A1 proposed correction
from an existing full-path run and uses fresh exact-action labels only to ask:

    If we knew perfectly when A1's proposed move is better than native,
    how much utility / official AP could an accept-or-reject mechanism recover?

For official AP, two score controls are emitted:
  * stage1: accepted/rejected actions keep the original Stage-1 query score.
  * a1: accepted actions use A1 selected score; rejected native actions use
        A1's zero-offset/native-hypothesis score.

The oracle label is test-time privileged information and is an upper bound, not
a deployable method. Ties stay native. No CAD/DexNet call occurs here; labels
must already exist in SOURCE_ROOT/evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rep_a_common import atomic_file, cdf_targets, digest, save_json
from rep_fullpath_runtime import case_key


METHODS = {
    "stage1": "rep_c1_oracle_accept_stage1score",
    "a1": "rep_c1_oracle_accept_a1score",
}


def _strings(x):
    return [str(v) for v in np.asarray(x).tolist()]


def _row_index(methods, policies, method, policy):
    hits = [
        i for i, (m, p) in enumerate(zip(methods, policies))
        if m == method and p == policy
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one row for {method}/{policy}, got {hits}")
    return hits[0]


def oracle_accept(payload, labels, scorer="A1", policy="fixed_0", eps=1e-7):
    """Return oracle accept decisions and both score controls for one frame."""
    actions = np.asarray(payload["actions"], np.float32)
    valid = np.asarray(payload["valid"], bool)
    if actions.ndim != 3 or actions.shape[-1] != 17:
        raise ValueError(f"Bad actions shape: {actions.shape}")
    k, q = valid.shape
    zero = int(np.asarray(payload["zero_index"]).reshape(()))
    qq = np.arange(q)
    methods = _strings(payload["output_methods"])
    policies = _strings(payload["output_policies"])
    row = _row_index(methods, policies, scorer, policy)
    proposal = np.asarray(payload["selected"][row], np.int64)
    if not valid[proposal, qq].all() or not valid[zero].all():
        raise RuntimeError("A1/native contains invalid actions")

    evaluated = np.asarray(labels["evaluated_mask"], bool)
    if evaluated.shape != valid.shape:
        raise ValueError("evaluated_mask shape mismatch")
    needed = np.zeros_like(valid)
    needed[zero] = True
    needed[proposal, qq] = True
    if not evaluated[needed].all():
        raise RuntimeError(
            "C1 needs exact labels for native and the A1-selected action. "
            "Rerun evaluate_rep_fullpath.py with LABEL_SCOPE=selected or all."
        )

    friction = np.asarray(labels["friction"], np.float32)
    y = cdf_targets(friction)
    utility = y.mean(-1)
    native_u = utility[zero]
    proposal_u = utility[proposal, qq]
    proposed_move = proposal != zero
    beneficial = proposed_move & (proposal_u > native_u + eps)
    harmful = proposed_move & (proposal_u < native_u - eps)
    tied = proposed_move & ~(beneficial | harmful)
    accepted = beneficial
    selected = np.where(accepted, proposal, zero).astype(np.int64)

    stage1_score = np.asarray(payload["original_native_score"], np.float32)
    if stage1_score.shape != (q,) or not np.isfinite(stage1_score).all():
        raise ValueError("Bad original_native_score")

    scorer_names = _strings(payload["scorer_names"])
    if scorer not in scorer_names:
        raise KeyError(f"{scorer} not found in {scorer_names}")
    scorer_i = scorer_names.index(scorer)
    probs = np.asarray(payload["probabilities"], np.float32)
    if probs.shape[:3] != (len(scorer_names), k, q):
        raise ValueError(f"Bad probabilities shape: {probs.shape}")
    zero_score = probs[scorer_i, zero].mean(-1)
    selected_score = np.asarray(payload["rank_scores"][row], np.float32)
    a1_score = np.where(accepted, selected_score, zero_score).astype(np.float32)

    selected_u = utility[selected, qq]
    native_s08 = y[zero, :, 3]
    selected_s08 = y[selected, qq, 3]
    proposal_s08 = y[proposal, qq, 3]

    result = {
        "selected": selected,
        "proposal": proposal,
        "stage1_score": stage1_score,
        "a1_score": a1_score,
        "native_utility": native_u,
        "proposal_utility": proposal_u,
        "selected_utility": selected_u,
        "proposed_move": proposed_move,
        "beneficial": beneficial,
        "harmful": harmful,
        "tied": tied,
        "accepted": accepted,
        "native_s08": native_s08,
        "proposal_s08": proposal_s08,
        "selected_s08": selected_s08,
    }
    return result


def _dump_path(root, method, policy, mode, case, split, camera, sid, aid):
    return (
        Path(root) / "dump" / method / policy / mode / case_key(case) / split
        / f"scene_{sid:04d}" / camera / f"{aid:04d}.npy"
    )


def _write_dump(path, actions, selected, score, overwrite=False):
    q = np.arange(actions.shape[1])
    out = actions[selected, q].copy()
    out[:, 0] = score
    if path.exists() and not overwrite:
        old = np.load(path, allow_pickle=False)
        if old.shape != out.shape or not np.allclose(old, out, atol=1e-6, rtol=0):
            raise RuntimeError(f"Existing C1 dump differs: {path}")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_file(path) as f:
        np.save(f, out, allow_pickle=False)
    return "written"


def _frame_metrics(res):
    move = res["proposed_move"]
    n = len(move)
    accepted = res["accepted"]
    harm = res["harmful"]
    benefit = res["beneficial"]
    native_u = res["native_utility"]
    prop_u = res["proposal_utility"]
    sel_u = res["selected_utility"]
    ns = res["native_s08"]
    ps = res["proposal_s08"]
    ss = res["selected_s08"]
    proposal_count = int(move.sum())
    return {
        "num_queries": n,
        "proposal_count": proposal_count,
        "proposal_rate": float(move.mean()),
        "beneficial_proposal_count": int(benefit.sum()),
        "harmful_proposal_count": int(harm.sum()),
        "tied_proposal_count": int(res["tied"].sum()),
        "oracle_accept_count": int(accepted.sum()),
        "oracle_accept_rate": float(accepted.mean()),
        "oracle_accept_given_proposal": float(accepted.sum() / proposal_count) if proposal_count else 0.0,
        "native_utility": float(native_u.mean()),
        "a1_proposal_utility": float(prop_u.mean()),
        "oracle_accept_utility": float(sel_u.mean()),
        "a1_proposal_gain": float((prop_u-native_u).mean()),
        "oracle_accept_gain": float((sel_u-native_u).mean()),
        "oracle_gain_over_a1": float((sel_u-prop_u).mean()),
        "native_success08": float(ns.mean()),
        "a1_proposal_success08": float(ps.mean()),
        "oracle_accept_success08": float(ss.mean()),
        "prevented_harm08": float(((ps < ns) & (~accepted)).mean()),
        "retained_rescue08": float(((ps > ns) & accepted).mean()),
    }


def _aggregate(rows):
    if not rows:
        raise ValueError("No rows")
    n = sum(r["num_queries"] for r in rows)
    p = sum(r["proposal_count"] for r in rows)
    counts = (
        "proposal_count", "beneficial_proposal_count", "harmful_proposal_count",
        "tied_proposal_count", "oracle_accept_count"
    )
    out = {k: int(sum(r[k] for r in rows)) for k in counts}
    for k in (
        "native_utility", "a1_proposal_utility", "oracle_accept_utility",
        "a1_proposal_gain", "oracle_accept_gain", "oracle_gain_over_a1",
        "native_success08", "a1_proposal_success08", "oracle_accept_success08",
        "prevented_harm08", "retained_rescue08",
    ):
        out[k] = sum(r[k]*r["num_queries"] for r in rows)/n
    out["proposal_rate"] = out["proposal_count"]/n
    out["oracle_accept_rate"] = out["oracle_accept_count"]/n
    out["oracle_accept_given_proposal"] = out["oracle_accept_count"]/p if p else 0.0
    out["num_queries"] = n
    out["num_frames"] = len(rows)
    return out


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=keys, restval="")
    w.writeheader()
    w.writerows(rows)
    with atomic_file(path) as f:
        f.write(buf.getvalue().encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--splits", default="test_seen,test_similar,test_novel")
    p.add_argument("--cases", default="", help="Default: all cases in source protocol")
    p.add_argument("--mode", default="joint")
    p.add_argument("--scorer", default="A1")
    p.add_argument("--policies", default="fixed_0,val_selected")
    p.add_argument("--score-sources", default="stage1,a1", choices=None)
    p.add_argument("--tie-eps", type=float, default=1e-7)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src = Path(args.source_root)
    out = Path(args.output_root)
    protocol = json.loads((src/"protocol.json").read_text())
    splits = [x for x in args.splits.split(",") if x]
    policies = [x for x in args.policies.split(",") if x]
    score_sources = [x for x in args.score_sources.split(",") if x]
    if any(x not in METHODS for x in score_sources):
        raise ValueError(f"score-sources must be subset of {tuple(METHODS)}")
    cases = [x for x in args.cases.split(",") if x] if args.cases else list(protocol["cases"])
    for c in cases:
        if c not in protocol["cases"]:
            raise ValueError(f"Case {c!r} not in source protocol")

    c1_protocol = dict(protocol)
    c1_protocol["rep_c1"] = {
        "source_root": str(src.resolve()),
        "source_protocol_digest": digest(protocol),
        "oracle": "Accept A1 fixed proposal iff exact utility(candidate) > exact utility(native); ties keep native",
        "scorer": args.scorer,
        "policies": policies,
        "score_sources": score_sources,
        "tie_eps": args.tie_eps,
        "privileged_test_labels": True,
        "claim": "upper bound for accept/reject only; NOT deployable",
    }
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/"protocol.json", c1_protocol)

    rows = []
    written = verified = 0
    for split in splits:
        files = sorted((src/"inference"/split).glob("scene_*/ann_*.npz"))
        if not files:
            raise FileNotFoundError(f"No source inference files for {split}")
        for inf in files:
            with np.load(inf, allow_pickle=False) as z:
                d = {k:z[k] for k in z.files}
            case, mode = str(d["case"]), str(d["mode"])
            if case not in cases or mode != args.mode:
                continue
            ev = src/"evaluation"/split/inf.parent.name/inf.name
            if not ev.is_file():
                raise FileNotFoundError(
                    f"Missing fresh exact labels: {ev}. "
                    "C1 does not use old fixed-action labels."
                )
            with np.load(ev, allow_pickle=False) as z:
                lab = {k:z[k] for k in z.files}
            sid = int(np.asarray(d["scene_id"]).reshape(()))
            aid = int(np.asarray(d["anno_id"]).reshape(()))
            for policy in policies:
                res = oracle_accept(d, lab, args.scorer, policy, args.tie_eps)
                row = {
                    "split":split, "scene_id":sid, "anno_id":aid,
                    "mode":mode, "case":case, "scorer":args.scorer,
                    "proposal_policy":policy, **_frame_metrics(res)
                }
                rows.append(row)
                for score_source in score_sources:
                    path = _dump_path(
                        out, METHODS[score_source], policy, mode, case, split,
                        protocol["camera"], sid, aid
                    )
                    status = _write_dump(
                        path, d["actions"], res["selected"],
                        res[f"{score_source}_score"], args.overwrite
                    )
                    written += int(status=="written")
                    verified += int(status=="verified")

    if not rows:
        raise RuntimeError("No C1 scenarios matched requested filters")
    _write_csv(out/"per_frame.csv", rows)
    groups = sorted({(r["split"],r["case"],r["proposal_policy"]) for r in rows})
    summary = []
    for split,case,policy in groups:
        rr = [r for r in rows if (r["split"],r["case"],r["proposal_policy"])==(split,case,policy)]
        summary.append({
            "split":split, "case":case, "proposal_policy":policy, **_aggregate(rr)
        })
    _write_csv(out/"comparison.csv", summary)
    save_json(out/"comparison.json", summary)
    save_json(out/"manifest.json", {
        "processed_frame_policy_rows":len(rows),
        "dumps_written":written,
        "dumps_verified":verified,
        "official_ready":int(protocol.get("query_limit",-1))==0,
        "note":"Official AP on these oracle dumps is an upper bound using test labels."
    })
    print(f"[REP-C1] rows={len(rows)} written={written} verified={verified}")
    print(f"[REP-C1] {out/'comparison.csv'}")


if __name__ == "__main__":
    main()
