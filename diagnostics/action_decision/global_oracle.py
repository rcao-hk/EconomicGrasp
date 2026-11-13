"""P2: global-budget, conflict-constrained known-quality candidate selection.

Requires the accepted graph/label caches from candidate_conflict. CP-SAT is
optional at import time and mandatory when solving. No test labels are used to
train a model. This optimizes summed six-mu success counts, NOT GraspNet AP.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics.candidate_conflict.core import (
    SCHEMA, greedy_nms, official_ap_matrix, physical_ids, q_priority,
    scene_cluster_bootstrap, sha256_file, write_json,
)
from diagnostics.candidate_conflict.build import load_verified_manifest as read_graph
from diagnostics.candidate_conflict.label import load_verified_manifest as read_labels
from diagnostics.candidate_conflict.analyze import _load_label


def restricted_pool(q, score, native_kept, max_candidates):
    """Keep every native NMS survivor plus top true-Q candidates.

    A capped pool is a diagnostic subset, never a full-candidate optimum.
    """
    n = len(q)
    if max_candidates <= 0 or n <= max_candidates:
        return np.arange(n, dtype=np.int64), False
    order = np.lexsort((-score, -q))
    selected = set(order[:max_candidates].tolist())
    selected.update(np.asarray(native_kept, dtype=np.int64).tolist())
    return np.array(sorted(selected), dtype=np.int64), True


def feasible_quality_greedy(q, scores, edges, objects, budget=50, per_object=10):
    """A feasible baseline with both NMS conflicts and evaluator quotas."""
    q = np.asarray(q, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    objects = np.asarray(objects, dtype=np.int64)
    if len(q) != len(scores) or len(q) != len(objects):
        raise ValueError("Candidate arrays must align")
    n = len(q)
    adj = [[] for _ in range(n)]
    for a, b in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    order = np.argsort(q_priority(q, scores))[::-1]
    blocked = np.zeros(n, dtype=bool)
    counts = defaultdict(int)
    result = []
    for i in order:
        i = int(i)
        if blocked[i] or counts[int(objects[i])] >= per_object:
            continue
        result.append(i)
        counts[int(objects[i])] += 1
        blocked[i] = True
        blocked[adj[i]] = True
        if len(result) == budget:
            break
    return np.array(result, dtype=np.int64)


def solve_set(q, scores, edges, objects, budget=50, per_object=10,
              time_limit=5.0, workers=1, max_candidates=0):
    """CP-SAT max sum(Q_i), global budget and per-object cardinality.

    Status OPTIMAL certifies the *sum-Q objective* only. FEASIBLE/UNKNOWN
    denotes solver-limited or capped-pool results; none certifies optimal AP.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("P2 requires ortools; install it in the diagnostics environment") from exc
    q = np.asarray(q, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    obj = np.asarray(objects, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = len(q)
    if not (len(scores) == len(obj) == n) or np.any((q < 0) | (q > 6)):
        raise ValueError("Invalid candidate scores / exact quality")
    if budget < 1 or per_object < 1 or time_limit <= 0 or workers < 1:
        raise ValueError("Invalid selection resource/quotas")
    native_kept, _, _ = greedy_nms(scores, edges)
    pool, restricted = restricted_pool(q, scores, native_kept, max_candidates)
    local = {int(x): i for i, x in enumerate(pool)}
    small_edges = np.asarray(
        [(local[int(a)], local[int(b)]) for a, b in edges if int(a) in local and int(b) in local],
        dtype=np.int64,
    ).reshape(-1, 2)
    q_small, s_small, obj_small = q[pool], scores[pool], obj[pool]
    initial = feasible_quality_greedy(q_small, s_small, small_edges, obj_small, budget, per_object)
    if len(pool) == 0:
        return np.zeros(0, dtype=np.int64), dict(
            status="OPTIMAL", num_input=0, num_considered=0, restricted=False,
            objective_count=0, best_bound_count=0, solve_seconds=0.0,
            same_quality_greedy_count=0, certifies_full_pool_sum_q=True)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(len(pool))]
    model.Add(sum(x) <= budget)
    for a, b in small_edges:
        model.Add(x[int(a)] + x[int(b)] <= 1)
    for oid in np.unique(obj_small):
        ids = np.flatnonzero(obj_small == oid)
        model.Add(sum(x[int(i)] for i in ids) <= per_object)
    # Tie-breaking cannot overturn one whole friction-success threshold:
    # max tie term (50*99) is strictly lower than quality-unit multiplier.
    scores_norm = ((s_small - s_small.min()) /
                   max(float(np.ptp(s_small)), 1e-12))
    tie = np.rint(scores_norm * 99).astype(np.int64)
    model.Maximize(sum((10000 * int(q_small[i]) + int(tie[i])) * x[i]
                       for i in range(len(pool))))
    hint_set = set(initial.tolist())
    for i in range(len(pool)):
        model.AddHint(x[i], int(i in hint_set))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        picked = np.asarray([i for i in range(len(pool)) if solver.BooleanValue(x[i])], dtype=np.int64)
        if int(q_small[picked].sum()) < int(q_small[initial].sum()):
            raise RuntimeError("CP-SAT incumbent below deterministic feasible hint")
        chosen = pool[picked]
    else:
        chosen = pool[initial]
    chosen_set = set(chosen.tolist())
    for a, b in edges:
        if int(a) in chosen_set and int(b) in chosen_set:
            raise RuntimeError("Optimizer returned conflicting grasps")
    if len(chosen) > budget:
        raise RuntimeError("Global budget violation")
    if any(np.sum(obj[chosen] == oid) > per_object for oid in np.unique(obj[chosen])):
        raise RuntimeError("Per-object quota violation")
    certified = (status == cp_model.OPTIMAL and not restricted)
    result = dict(
        status=solver.StatusName(status), num_input=n,
        num_considered=len(pool), restricted=restricted,
        num_selected=len(chosen), budget=budget, per_object=per_object,
        objective_count=int(q[chosen].sum()),
        same_quality_greedy_count=int(q_small[initial].sum()),
        solve_seconds=float(solver.WallTime()),
        best_objective_scaled=float(solver.ObjectiveValue()) if status in
            (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        best_bound_scaled=float(solver.BestObjectiveBound()) if status in
            (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        certifies_full_pool_sum_q=certified,
        disclaimer="Optimality refers to sum-Q, not official AP",
    )
    return chosen, result


def analyze_frame(graph_file, graph_sha, labels, time_limit, workers,
                  max_candidates, budget, per_object):
    if sha256_file(graph_file) != graph_sha:
        raise RuntimeError(f"Graph SHA mismatch: {graph_file}")
    with np.load(graph_file, allow_pickle=False) as g:
        grasps = np.asarray(g["grasps"], dtype=np.float32)
        ids = np.asarray(g["physical_id"]).astype(str)
        if not np.array_equal(ids, physical_ids(grasps)):
            raise ValueError("Physical action identity mismatch")
        edges = np.asarray(g["edges"], dtype=np.int64)
        native_keep = np.asarray(g["nms_keep"], dtype=np.int64)
    lookup = {pid: i for i, pid in enumerate(np.asarray(labels["physical_id"]).astype(str))}
    try:
        inds = np.array([lookup[pid] for pid in ids], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("An action lacks an exact label") from exc
    q = np.asarray(labels["quality_count"], dtype=np.int64)[inds]
    obj = np.asarray(labels["assigned_obj"], dtype=np.int64)[inds]
    fric = np.asarray(labels["friction"], dtype=np.float64)[inds]
    bad = np.asarray(labels["collision_or_empty"], dtype=bool)[inds]
    native_score = grasps[:, 0].astype(np.float64)
    qscore = q_priority(q, native_score)
    q_nms, _, _ = greedy_nms(qscore, edges)
    greedy_budget = feasible_quality_greedy(q, native_score, edges, obj, budget, per_object)
    optimized, stats = solve_set(q, native_score, edges, obj, budget, per_object,
                                 time_limit, workers, max_candidates)
    ap_native = official_ap_matrix(grasps, obj, bad, fric, native_keep, native_score)
    ap_quality_nms = official_ap_matrix(grasps, obj, bad, fric, q_nms, qscore)
    ap_greedy_budget = official_ap_matrix(grasps, obj, bad, fric, greedy_budget, qscore)
    ap_optimized = official_ap_matrix(grasps, obj, bad, fric, optimized, qscore)
    # A sum-Q optimum does not guarantee nondecreasing AP.
    return dict(native_ap=float(ap_native.mean()*100),
                quality_greedy_ap=float(ap_quality_nms.mean()*100),
                budget_greedy_ap=float(ap_greedy_budget.mean()*100),
                optimized_ap=float(ap_optimized.mean()*100),
                quality_count_delta=int(q[optimized].sum()-q[greedy_budget].sum()),
                ap_delta_vs_qnms_pp=float((ap_optimized-ap_quality_nms).mean()*100),
                ap_delta_vs_budget_pp=float((ap_optimized-ap_greedy_budget).mean()*100),
                solver=stats), ap_native


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-root", required=True)
    p.add_argument("--labels-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", action="append", default=[], help="MM/MR/RM/RR, default all")
    p.add_argument("--sample-interval", type=int, default=10,
                   help="Default pilot uses 26 frames/scene. For full use 1.")
    p.add_argument("--max-scenes", type=int, default=0, help="Smoke only")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="0 for full pool; >0 reports restricted non-full optimum")
    p.add_argument("--time-limit", type=float, default=5.)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--per-object", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=3000)
    args=p.parse_args()
    gm=read_graph(Path(args.graph_root)/"graph_manifest.json")
    lm=read_labels(Path(args.labels_root)/"label_manifest.json")
    if gm["manifest_sha256"]!=lm["graph_manifest_sha256"]:
        raise ValueError("Graph and labels must belong to the same experiment")
    if not gm["complete"] or not lm["complete"]:
        raise ValueError("P2 requires full graph and exact-label source manifests")
    methods=args.method or sorted(gm["methods"])
    if any(x not in gm["methods"] for x in methods) or len(methods)!=len(set(methods)):
        raise ValueError("Invalid method list")
    if args.sample_interval<=0 or args.bootstrap<=0:
        raise ValueError("Invalid sampling/bootstrap settings")
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Output must be empty")
    out.mkdir(parents=True,exist_ok=True)
    label_lookup={}
    for scene in lm["scene_records"]:
        for r in scene["frames"]:
            label_lookup[r["contract"]["frame_key"]]=r
    rows=[]
    scene_values=defaultdict(lambda:defaultdict(list))
    for method in methods:
        official_path=gm["methods"][method].get("official_ap_path")
        if not official_path:
            raise RuntimeError(f"{method}: missing official AP reference for replay parity")
        official_path=Path(official_path)
        if not official_path.is_file(): raise FileNotFoundError(official_path)
        official=np.asarray(np.load(official_path,allow_pickle=False),dtype=np.float64)
        if official.shape!=(30,256,50,6) or not np.isfinite(official).all():
            raise RuntimeError(f"{method}: wrong official AP array shape")
        first_scene=int(min(e["scene"] for e in gm["methods"][method]["entries"]))
        scenes=sorted(set(r["scene"] for r in gm["methods"][method]["entries"]))
        if args.max_scenes: scenes=scenes[:args.max_scenes]
        for e in gm["methods"][method]["entries"]:
            scene,anno=int(e["scene"]),int(e["anno"])
            if scene not in scenes or anno % args.sample_interval: continue
            lr=label_lookup[e["frame_key"]]
            labels=_load_label(Path(args.labels_root),lr,gm["manifest_sha256"])
            record,native_ap=analyze_frame(
                Path(args.graph_root)/e["graph_path"],e["graph_sha256"],
                labels,args.time_limit,args.workers,args.max_candidates,
                args.budget,args.per_object)
            err=float(np.max(np.abs(native_ap-official[scene-first_scene,anno])))
            if err>1e-5:
                raise RuntimeError(
                    f"{method} scene={scene} anno={anno}: official native AP replay "
                    f"mismatch {err:.8g} (tol 1e-5); abandon P2 oracle interpretation")
            record["native_ap_parity_max_abs"]=err
            row=dict(method=method,scene=scene,anno=anno,**{
                k:v for k,v in record.items() if k!="solver"},**record["solver"])
            rows.append(row)
            scene_values[method][scene].append(record["ap_delta_vs_qnms_pp"])
            if len(rows)%100==0:print(f"[P2] {len(rows)} frame-methods",flush=True)
    if not rows:raise ValueError("No frame-method pairs processed")
    with (out/"per_frame.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={}
    for m,scenes in scene_values.items():
        values={s:float(np.mean(v)) for s,v in scenes.items()}
        subset=[r for r in rows if r["method"]==m]
        summary[m]=dict(
            frames=len(subset),scenes=len(values),
            solver_status={x:sum(r["status"]==x for r in subset) for x in sorted({r["status"] for r in subset})},
            full_pool_optimal_fraction=float(np.mean([r["certifies_full_pool_sum_q"] for r in subset])),
            restricted_fraction=float(np.mean([r["restricted"] for r in subset])),
            native_ap_parity_max_abs=max(r["native_ap_parity_max_abs"] for r in subset),
            official_ap_sha256=sha256_file(
                gm["methods"][m]["official_ap_path"]),
            delta_vs_qgreedy=scene_cluster_bootstrap(values,args.bootstrap,0),
            mean_ap={k:float(np.mean([r[k] for r in subset])) for k in
                     ("native_ap","quality_greedy_ap","budget_greedy_ap","optimized_ap")},
        )
    write_json(out/"summary.json",dict(
        scope="SUM-Q constrained selection; NOT AP optimum or deployment",
        source_graph_manifest=gm["manifest_sha256"],
        source_label_manifest=lm["manifest_sha256"],
        source_method=methods,interval=args.sample_interval,
        parameters=dict(time_limit=args.time_limit,max_candidates=args.max_candidates,
                        budget=args.budget,per_object=args.per_object),
        results=summary))
    print(json.dumps({k:v["mean_ap"] for k,v in summary.items()},indent=2))


if __name__=="__main__":
    main()
