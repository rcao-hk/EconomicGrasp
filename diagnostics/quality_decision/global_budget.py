"""P2: global-budget grasp selection using exactly the SAME quality-labelled pool.

Two levels:
- deterministic Q-greedy independent-set selection with global K and object caps;
- optional CP-SAT maximizing the exact six-mu, top-50 rank-weighted AP on a
  capped subset. CP-SAT reports OPTIMAL/FEASIBLE and pool-truncation limits.
This is an offline oracle, NOT a deployable detector or a universal upper bound.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnostics.candidate_conflict.build import load_verified_manifest as load_graph
from diagnostics.candidate_conflict.label import load_verified_manifest as load_labels
from diagnostics.candidate_conflict.analyze import _load_label
from diagnostics.candidate_conflict.core import (greedy_nms, official_ap_matrix,
                                                  physical_ids, q_priority,
                                                  scene_cluster_bootstrap, sha256_file)
from .core import SCHEMA, MU, digest

TOPK=50
MAX_PER_OBJECT=10


def global_greedy_quality(ids, edges, quality, obj, original_scores,
                          top_k=TOPK, max_per_object=MAX_PER_OBJECT):
    """One-pass greedy independent set, enforcing object quotas and total K.

    Unlike NMS-then-cap, the object and global budgets are active during
    selection. Always feasible but not guaranteed optimal.
    """
    ids=np.asarray(ids,dtype=np.int64)
    q=np.asarray(quality,dtype=np.int64)
    scores=np.asarray(original_scores,dtype=np.float64)
    obj=np.asarray(obj,dtype=np.int64)
    if not np.isin(ids,np.arange(len(q))).all():
        raise ValueError("Unknown candidate ids")
    order=ids[np.argsort(q_priority(q[ids],scores[ids]))[::-1]]
    adj=[set() for _ in range(len(q))]
    for u,v in np.asarray(edges,dtype=np.int64).reshape(-1,2):
        adj[int(u)].add(int(v));adj[int(v)].add(int(u))
    chosen=[]; chosen_set=set();counts={}
    for i in order:
        i=int(i);o=int(obj[i])
        if counts.get(o,0)>=max_per_object:
            continue
        if chosen_set & adj[i]:
            continue
        chosen.append(i);chosen_set.add(i);counts[o]=counts.get(o,0)+1
        if len(chosen)>=top_k:
            break
    return np.asarray(chosen,dtype=np.int64)


def exact_ap_rank_weights(top_k=TOPK, mu_count=6):
    """One AP-success at rank r contributes sum_{k=r}^{K-1} 1/(k+1)/(K*6)."""
    k=np.arange(1,top_k+1,dtype=np.float64)
    return np.cumsum((1/k)[::-1])[::-1]/(top_k*mu_count)


def cpsat_ap_oracle(ids, edges, success_bits, obj, time_limit=10.,
                    top_k=TOPK, max_per_object=MAX_PER_OBJECT):
    """Optimize exact rank-weighted mean AP within explicitly supplied pool.

    The full-scene problem can be computationally prohibitive. Input ids are
    normally capped to at most 80-128 candidates; OPTIMAL proves optimality
    only for these ids. FEASIBLE is merely a lower-bound solution.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("Optional CP-SAT requires: pip install ortools") from exc
    ids=np.asarray(ids,dtype=np.int64)
    bits=np.asarray(success_bits,dtype=np.int32)
    obj=np.asarray(obj,dtype=np.int64)
    if len(ids)==0:
        return np.zeros(0,dtype=np.int64),dict(status="OPTIMAL",best_objective=0.,best_bound=0.)
    if len(np.unique(ids))!=len(ids):
        raise ValueError("Duplicate candidate indices")
    n=len(ids); slots=min(top_k,n)
    chosen_map={int(v):j for j,v in enumerate(ids)}
    model=cp_model.CpModel()
    x=[model.NewBoolVar(f"selected_{i}") for i in range(n)]
    y=[[model.NewBoolVar(f"slot_{i}_{r}") for r in range(slots)] for i in range(n)]
    for i in range(n):
        model.Add(sum(y[i])==x[i])
    occupancy=[]
    for r in range(slots):
        cur=sum(y[i][r] for i in range(n))
        model.Add(cur<=1)
        occupancy.append(cur)
    for r in range(slots-1):
        model.Add(occupancy[r]>=occupancy[r+1])
    for object_id in np.unique(obj[ids]):
        members=[i for i in range(n) if obj[ids[i]]==object_id]
        model.Add(sum(x[i] for i in members)<=max_per_object)
    for u,v in np.asarray(edges,dtype=np.int64).reshape(-1,2):
        a,b=chosen_map.get(int(u)),chosen_map.get(int(v))
        if a is not None and b is not None:
            model.Add(x[a]+x[b]<=1)
    weights=np.maximum(1,np.rint(exact_ap_rank_weights(top_k)*1e8).astype(np.int64))
    quality=bits[ids].sum(axis=1)
    if np.any((quality<0)|(quality>6)):
        raise ValueError("Invalid six-mu quality bits")
    model.Maximize(sum(int(quality[i]*weights[r])*y[i][r]
                       for i in range(n) for r in range(slots)))
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=float(time_limit)
    solver.parameters.num_search_workers=1
    solver.parameters.random_seed=0
    status=solver.Solve(model)
    names={cp_model.OPTIMAL:"OPTIMAL",cp_model.FEASIBLE:"FEASIBLE",
           cp_model.UNKNOWN:"UNKNOWN",cp_model.INFEASIBLE:"INFEASIBLE"}
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        return np.zeros(0,dtype=np.int64),dict(status=names.get(status,str(status)),
                                                best_objective=None,best_bound=None,
                                                solved_pool_n=n)
    ranked=[]
    for r in range(slots):
        found=[int(ids[i]) for i in range(n) if solver.Value(y[i][r])]
        if len(found)>1:
            raise RuntimeError("CP-SAT returned two actions in one rank")
        ranked+=found
    result=np.asarray(ranked,dtype=np.int64)
    return result,dict(status=names.get(status,str(status)),solved_pool_n=n,
                       best_objective=float(solver.ObjectiveValue()),
                       best_bound=float(solver.BestObjectiveBound()),
                       objective_scale=1e8,
                       objective_exactness="integer-approximated six-mu AP")


def ap_for_ranked(grasps, obj, bad, friction, ranked):
    ranked=np.asarray(ranked,dtype=np.int64)
    n=len(grasps)
    # Strictly decreasing synthetic priority for the chosen rank assignment.
    scores=np.zeros(n,dtype=np.float64)
    if len(ranked):
        scores[ranked]=len(ranked)-np.arange(len(ranked),dtype=np.float64)
    return official_ap_matrix(grasps,obj,bad,friction,ranked,scores)


def verify_non_conflicting(ids, edges, obj, k=TOPK, per_obj=MAX_PER_OBJECT):
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)>k or len(np.unique(ids))!=len(ids):
        raise ValueError("Budget/duplicate violation")
    included=set(map(int,ids))
    for u,v in np.asarray(edges,dtype=np.int64).reshape(-1,2):
        if int(u) in included and int(v) in included:
            raise ValueError("Conflicting chosen actions")
    u,counts=np.unique(np.asarray(obj)[ids],return_counts=True)
    if np.any(counts>per_obj):
        raise ValueError("Per-object budget violated")
    return True


def analyze_frame(graph, labels, max_solver_nodes=96, solver="off", solver_seconds=10.):
    g=np.asarray(graph["grasps"],dtype=np.float32)
    ids=np.asarray(graph["physical_id"]).astype(str)
    if not np.array_equal(physical_ids(g),ids):
        raise ValueError("Graph action identity mismatch")
    label_idx={str(x):i for i,x in enumerate(labels["physical_id"].astype(str))}
    try:
        ix=np.asarray([label_idx[x] for x in ids],dtype=np.int64)
    except KeyError as e:
        raise RuntimeError(f"Candidate lacks exact action label: {e}") from e
    obj=np.asarray(labels["assigned_obj"],dtype=np.int64)[ix]
    bad=np.asarray(labels["collision_or_empty"],dtype=bool)[ix]
    friction=np.asarray(labels["friction"],dtype=np.float64)[ix]
    bits=np.asarray(labels["success_bits"],dtype=np.int32)[ix]
    if np.any(bits.sum(-1)>6):
        raise ValueError("Malformed six-friction bits")
    q=bits.sum(-1).astype(np.int64)
    edges=np.asarray(graph["edges"],dtype=np.int64)
    native=np.asarray(graph["nms_keep"],dtype=np.int64)
    priority=q_priority(q,g[:,0])
    qnms,_suppressor,_=greedy_nms(priority,edges)
    global_greedy=global_greedy_quality(np.arange(len(g)),edges,q,obj,g[:,0])
    verify_non_conflicting(global_greedy,edges,obj)
    # Fixed-NMS-set Q ranking: compare with the old label-aware oracle.
    ap_native=official_ap_matrix(g,obj,bad,friction,native,g[:,0])
    ap_qnms=official_ap_matrix(g,obj,bad,friction,qnms,priority)
    ap_greedy=ap_for_ranked(g,obj,bad,friction,global_greedy)
    result=dict(native_ap_percent=float(100*ap_native.mean()),
                quality_nms_ap_percent=float(100*ap_qnms.mean()),
                global_greedy_ap_percent=float(100*ap_greedy.mean()),
                native_survivors=len(native),q_nms_survivors=len(qnms),
                global_greedy_survivors=len(global_greedy),
                num_candidates=len(g),num_conflicts=len(edges))
    if solver=="cp_sat":
        sort=np.argsort(priority)[::-1][:max_solver_nodes]
        small=np.asarray(sort,dtype=np.int64)
        restricted_greedy=global_greedy_quality(small,edges,q,obj,g[:,0])
        verify_non_conflicting(restricted_greedy,edges,obj)
        opt,info=cpsat_ap_oracle(small,edges,bits,obj,solver_seconds)
        if info["status"] not in ("OPTIMAL","FEASIBLE"):
            raise RuntimeError("CP-SAT did not return a feasible selection; no oracle result")
        verify_non_conflicting(opt,edges,obj)
        ap_opt=ap_for_ranked(g,obj,bad,friction,opt)
        ap_restricted=ap_for_ranked(g,obj,bad,friction,restricted_greedy)
        result.update(cp_sat_ap_percent=float(100*ap_opt.mean()),
                      restricted_greedy_ap_percent=float(100*ap_restricted.mean()),
                      cp_sat_minus_restricted_greedy_pp=float(100*(ap_opt.mean()-ap_restricted.mean())),
                      cp_sat_status=info["status"],
                      cp_sat_candidate_fraction=float(len(small)/max(1,len(g))),
                      cp_sat_scope=f"restricted top-{max_solver_nodes} quality candidates",
                      cp_sat_objective_bound=info["best_bound"],
                      cp_sat_objective_value=info["best_objective"])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-root",required=True)
    p.add_argument("--labels-root",required=True)
    p.add_argument("--method",default="MM")
    p.add_argument("--output",required=True)
    p.add_argument("--solver",choices=("off","cp_sat"),default="off")
    p.add_argument("--max-solver-nodes",type=int,default=96)
    p.add_argument("--solver-seconds",type=float,default=10.)
    p.add_argument("--frames-per-scene",type=int,default=2,
                   help="2 for preregistered budgeted pilot, 0 for all frames")
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261113)
    p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args()
    graph_root=Path(a.graph_root);labels_root=Path(a.labels_root)
    gm=load_graph(graph_root/"graph_manifest.json")
    lm=load_labels(labels_root/"label_manifest.json")
    if gm["manifest_sha256"]!=lm.get("graph_manifest_sha256"):
        raise ValueError("Graph-label experiment mismatch")
    if not a.allow_partial and (not gm.get("complete") or not lm.get("complete")):
        raise ValueError("Formal analysis requires full graph/label manifests")
    if a.method not in gm["methods"] or a.method not in lm["methods"]:
        raise ValueError("Method not covered by labels")
    if a.frames_per_scene<0 or a.max_solver_nodes<1 or a.bootstrap<1:
        raise ValueError("Invalid constraints")
    frames=sorted(gm["methods"][a.method]["entries"],key=lambda x:(x["scene"],x["anno"]))
    if a.frames_per_scene:
        chosen={}
        for entry in frames:
            chosen.setdefault(entry["scene"],[]).append(entry)
        frames=[r for scene,records in sorted(chosen.items())
                for r in np.asarray(records,dtype=object)[np.linspace(
                    0,len(records)-1,min(a.frames_per_scene,len(records)),dtype=int)]]
    labels_by_key={}
    for scene in lm["scene_records"]:
        for rec in scene["frames"]:
            labels_by_key[rec["contract"]["frame_key"]]=rec
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use new P2 output directory")
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for k,entry in enumerate(frames,1):
        path=graph_root/entry["graph_path"]
        if sha256_file(path)!=entry["graph_sha256"]:
            raise ValueError("Graph corrupted")
        with np.load(path,allow_pickle=False) as z:
            graph={x:z[x] for x in z.files}
        labelrec=labels_by_key.get(entry["frame_key"])
        if labelrec is None:raise ValueError("Missing label frame")
        labels=_load_label(labels_root,labelrec,gm["manifest_sha256"])
        result=analyze_frame(graph,labels,max_solver_nodes=a.max_solver_nodes,
                             solver=a.solver,solver_seconds=a.solver_seconds)
        rows.append(dict(scene=int(entry["scene"]),anno=int(entry["anno"]),**result))
        if k%32==0:print(f"[P2] {k}/{len(frames)}",flush=True)
    metric_keys=[k for k in rows[0] if k.endswith("_ap_percent")]
    summary={}
    for metric in metric_keys:
        by_scene={scene:float(np.mean([r[metric] for r in rows if r["scene"]==scene]))
                  for scene in sorted(set(r["scene"] for r in rows))}
        summary[metric]=scene_cluster_bootstrap(by_scene,a.bootstrap,a.seed)
    import csv
    with (root/"per_frame.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(schema=SCHEMA,kind="P2_global_budget_oracle",
                graph_manifest_sha256=gm["manifest_sha256"],
                label_manifest_sha256=lm["manifest_sha256"],
                split=gm["split"],method=a.method,frames=len(rows),
                scene_count=len(set(r["scene"] for r in rows)),
                frames_per_scene=a.frames_per_scene,solver=a.solver,
                max_solver_nodes=a.max_solver_nodes,solver_seconds=a.solver_seconds,
                metric_by_scene=summary,
                caveat="All Q labels from held-out tests are diagnostic-only. "
                       "CP-SAT exactness is restricted to capped candidate pool and solver OPTIMAL status. "
                       "No claim of global 6D grasp AP optimum.")
    (root/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v["mean"] for k,v in summary.items()},indent=2))


if __name__=="__main__":
    main()
