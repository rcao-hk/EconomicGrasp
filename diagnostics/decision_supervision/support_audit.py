"""P1 train-only support audit: is the U∪D reservoir missing hard, valid actions?

Select NEW native executed actions from full 1024-query snapshot WITHOUT
reading their CAD labels. Label only a fixed budget per frame using official
exact evaluator. This is a diagnostic acquisition audit, NOT training.

The old cache contains all four depths for U∪D families. New acquisitions
label native executed depth ONLY; no claims about other insertion depths.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rare_grasp.core import frame_seed
from diagnostics.candidate_conflict.core import (
    build_conflict_edges, scene_cluster_bootstrap,
)
from .attribution_core import compiled_keep
from .core import BASE_SHA, SCHEMA, digest, sha, physical_ids, write_json, write_npz, native_rows
from .data import load_plan, read_pack, frame_name
from .runtime import make_runtime, filtered_indices

STRATA=("new_top50_survivor","new_top50_conflict_neighbor",
        "new_high_score_other","new_random_control","new_fallback")
DEFAULT_QUOTAS=(16,24,16,8)


def _native_known(pack):
    rows=native_rows(pack)
    ids=np.asarray(pack["family"][rows],dtype=np.int64)
    data={int(q):dict(count=int(pack["count"][idx]),score=float(pack["grasps"][idx,0]),
                      physical_id=str(pack["physical_id"][idx]))
          for q,idx in zip(ids,rows)}
    return data


def _sample_new(eligible, scores, native_keep, edges, *, old, quotas=DEFAULT_QUOTAS,
                seed=0):
    """Return new query IDs, disjoint source-population strata, and audit.

    eligible: full original query indices that survived sensor filtering.
    edges: [E,2] in *eligible-array index space* (geometric undirected).
    No quality labels are read during sampling.
    """
    eligible=np.asarray(eligible,dtype=np.int64)
    score=np.asarray(scores,dtype=np.float64)
    old=set(map(int,old))
    if len(eligible)!=len(np.unique(eligible)) or not np.isfinite(score).all():
        raise ValueError("Invalid full-frame sensor survivors")
    if len(quotas)!=4 or sum(quotas)!=64:
        raise ValueError("Preregistered acquisition budget: 16+24+16+8=64")
    n=len(eligible)
    keep_indices=np.asarray(native_keep,dtype=np.int64)
    if len(keep_indices) and (keep_indices.min()<0 or keep_indices.max()>=n):
        raise ValueError("Invalid native compiled NMS survivor indices")
    top={int(x) for x in eligible[keep_indices[:50]]}
    nodes=set(map(int,eligible))
    if not top.issubset(nodes):
        raise ValueError("Native top50 outside sensor-filtered candidate pool")
    neighbor=set()
    leader_neighbors={}
    for u,v in np.asarray(edges,dtype=np.int64).reshape(-1,2):
        a,b=int(eligible[u]),int(eligible[v])
        if a in top and b not in top:
            neighbor.add(b);leader_neighbors.setdefault(b,set()).add(a)
        if b in top and a not in top:
            neighbor.add(a);leader_neighbors.setdefault(a,set()).add(b)
    threshold=float(np.quantile(score, .75)) if n else None
    available=nodes-old
    # scores is indexed like eligible, not by raw 1024-index query id.
    scores_by_query={int(q):float(v) for q,v in zip(eligible,score)}
    def sort_desc(items):
        return sorted(items,key=lambda q:(-scores_by_query[q],q))
    survivor=sort_desc(top&available)
    high=sort_desc({q for q in available if q not in top and q not in neighbor
                    and scores_by_query[q]>=threshold}) if threshold is not None else []
    rng=np.random.default_rng(seed)
    random_pool=list(rng.permutation(sorted(available-top-neighbor-set(high))))
    chosen=[];labels=[];deficit=[]
    # Pick survivors FIRST. Then prefer a competitor if its top50 endpoint is
    # actually labeled already (old U∪D) or is among newly selected survivors.
    first=survivor[:quotas[0]]
    chosen.extend(first);labels.extend([STRATA[0]]*len(first))
    deficit.append(quotas[0]-len(first))
    available_leaders=old|set(chosen)
    comp=sorted(neighbor&available,
                key=lambda q:(-int(bool(leader_neighbors.get(q,set()) & available_leaders)),
                              -scores_by_query[q],q))
    for pool,quota,name in zip((comp,high,random_pool),quotas[1:],STRATA[1:4]):
        accepted=[q for q in pool if q not in chosen][:quota]
        chosen.extend(accepted);labels.extend([name]*len(accepted))
        deficit.append(quota-len(accepted))
    missing=64-len(chosen)
    fallback=list(rng.permutation(sorted(available-set(chosen))))[:missing]
    chosen.extend(map(int,fallback));labels.extend(["new_fallback"]*len(fallback))
    if len(chosen)!=len(set(chosen)) or len(chosen)>64 or set(chosen)&old:
        raise RuntimeError("Invalid new-action budget or duplicate")
    audit=dict(sensor_valid_n=n,old_cached_n=len(old),
               cached_sensor_valid_n=len(nodes&old),
               top50_n=len(top),top50_cached_n=len(top&old),
               competition_nodes_n=len(neighbor),
               competition_cached_n=len(neighbor&old),
               high_score_nodes_n=sum(v>=threshold for v in score) if n else 0,
               high_score_cached_n=sum(scores_by_query[q]>=threshold for q in nodes&old) if n else 0,
               unseen_sensor_valid_n=len(available),
               quantile75=threshold,quotas=list(quotas),
               strata_selected={name:int(labels.count(name)) for name in STRATA[:4]},
               quota_deficits=dict(zip(STRATA[:4],deficit)),
               fallback_n=len(fallback),new_labeled_n=len(chosen))
    return np.asarray(chosen,dtype=np.int64),np.asarray(labels,dtype="<U40"),audit,leader_neighbors


def _frame(args,rt,compiled,entry,root,contract,resume):
    row=entry["row"]
    if row["split"]!="train" or not 0<=int(row["scene"])<100:
        raise ValueError("P1 labeling forbidden on official test scenes")
    pack=read_pack(entry)
    known=_native_known(pack)
    snap=rt.snapshot(row,int(args.snapshot_seed))
    bank=np.asarray(snap["grasps"])
    if bank.shape!=(1024,4,17):
        raise ValueError("Require all 1024 native query candidates")
    f=np.asarray(pack["family"],dtype=int)
    d=np.asarray(pack["depth"],dtype=int)
    existing=bank[f,d]
    if (not np.array_equal(physical_ids(existing),pack["physical_id"].astype(str))
          or float(np.max(np.abs(existing[:,:16]-pack["grasps"][:,:16])))>1e-6
          or float(np.max(np.abs(np.asarray(snap["features"])[f,d]-pack["features"])))>1e-6
          or float(np.max(np.abs(np.asarray(snap["base_logit"])[f,d]-pack["base_logit"])))>1e-6):
        raise RuntimeError("Accepted action/feature cache parity failed")
    winners=bank[np.arange(1024),bank[...,0].argmax(1)].copy()
    filtered=np.asarray(rt.sensor_collision_filter(row,winners))
    qids=filtered_indices(winners,filtered)
    nms=compiled_keep(filtered,compiled)
    if len(filtered):
        edges=build_conflict_edges(filtered)
    else:edges=np.empty((0,2),dtype=np.int64)
    native_valid=set(map(int,qids))
    expected_mask=np.isin(np.unique(f),qids)
    if not np.array_equal(np.asarray(pack["native_sensor_valid"],dtype=bool),expected_mask):
        raise RuntimeError("Sensor-valid parity differs from accepted relation trace")
    seed=int(frame_seed(row,args.seed))
    query_ids,strata,audit,leaders=_sample_new(
        qids,filtered[:,0],nms,edges,old=known,seed=seed)
    if len(query_ids)<args.min_new_actions:
        raise RuntimeError(f"Insufficient new native actions {len(query_ids)} "
                           f"< {args.min_new_actions}: {frame_name(row)}")
    source_map={int(q):i for i,q in enumerate(qids)}
    label_input=filtered[np.asarray([source_map[int(q)] for q in query_ids],dtype=int)]
    name=frame_name(row)
    npz=root/"frames"/(name+".npz")
    side=npz.with_suffix(".json")
    frame_contract=dict(schema=SCHEMA,kind="P1_support_audit_frame",
                        plan_sha256=contract["plan_sha256"],
                        script_sha256=contract["script_sha256"],
                        source_pack_sha256=entry["sha256"],
                        scene=row["scene"],anno=row["anno"],domain=row["domain"],
                        base_checkpoint_sha256=BASE_SHA,
                        native_bank_physical_sha256=digest(physical_ids(winners).tolist()),
                        sensor_valid_ids_sha256=digest(qids.tolist()),
                        compiled_survivors_sha256=digest(nms.tolist()),
                        requested_new_families=64,original_snapshot_seed=args.snapshot_seed)
    if npz.exists() or side.exists():
        if not resume or not (npz.is_file() and side.is_file()):
            raise FileExistsError(f"Existing/incomplete P1 support cache: {npz}")
        meta=json.loads(side.read_text(encoding="utf-8"))
        if (meta["contract_sha256"]!=digest(frame_contract) or sha(npz)!=meta["sha256"]):
            raise RuntimeError("P1 resume cannot reuse different source/actions")
        with np.load(npz,allow_pickle=False) as z:
            if (not np.array_equal(z["query_ids"],query_ids) or
                    not np.array_equal(z["physical_id"].astype(str),physical_ids(label_input))):
                raise RuntimeError("P1 resume sample identity mismatch")
        return meta
    # Only newly selected action labels are expensive; an exact resume does
    # not re-run CAD/Dex-Net.
    if len(query_ids):
        labels=rt.label(row,label_input.astype(np.float64))
        count=np.asarray(labels["count"],dtype=np.int64)
        bits=np.asarray(labels["bits"],dtype=np.uint8)
        if count.shape!=(len(query_ids),) or bits.shape!=(len(query_ids),6) or not np.array_equal(bits.sum(1),count):
            raise RuntimeError("New labels are incomplete/mismatched")
    else:
        labels={k:np.empty(0) for k in ("count","friction","collision","empty")}
        count=np.empty(0,dtype=np.int64);bits=np.empty((0,6),dtype=np.uint8)
    score_by_query={int(q):float(v) for q,v in zip(qids,filtered[:,0])}
    new_known={int(q):int(c) for q,c in zip(query_ids,count)}
    pair_records=[]
    # Only these NEW-vs-known / NEW-vs-NEW pairs are attributable to new labels.
    for q in query_ids:
        q=int(q)
        if q not in leaders:continue
        for leader in sorted(leaders[q]):
            label_leader=(known.get(leader) or {}).get("count")
            if label_leader is None:label_leader=new_known.get(leader)
            if label_leader is None:continue
            pair_records.append((q,int(leader),int(new_known[q]),int(label_leader)))
    if pair_records:
        pair_arr=np.asarray(pair_records,dtype=np.int64).reshape(-1,4)
    else:pair_arr=np.empty((0,4),dtype=np.int64)
    if len(pair_arr):
        # Confirm a fixed, deterministic subset of graph edges with compiled NMS.
        for q,leader,_,_ in pair_arr[:args.max_compiled_pair_checks]:
            ix=[source_map[int(q)],source_map[int(leader)]]
            if len(compiled(filtered[ix].astype(np.float64)))!=1:
                raise RuntimeError("Graph competitor pair disagrees with installed compiled NMS")
    old_high_q0=[(q,value["count"]) for q,value in known.items()
                 if q in native_valid and score_by_query[q]>=audit["quantile75"]
                 and value["count"]==0]
    new_high_q0=[q for q,c in zip(query_ids,count)
                 if score_by_query[int(q)]>=audit["quantile75"] and int(c)==0]
    pure_collision=np.asarray(labels["collision"],dtype=bool)
    empty=np.asarray(labels["empty"],dtype=bool)
    if pure_collision.shape!=count.shape or empty.shape!=count.shape:
        raise ValueError("Exact evaluator collision/empty shape mismatch")
    audit.update(
        sampled_q0_n=int(np.sum(count==0)),sampled_q_positive_n=int(np.sum(count>0)),
        sampled_q0_collision_n=int(np.sum((count==0)&pure_collision)),
        sampled_q0_empty_n=int(np.sum((count==0)&(~pure_collision)&empty)),
        sampled_q0_other_n=int(np.sum((count==0)&(~pure_collision)&(~empty))),
        sampled_high_score_q0_n=len(new_high_q0),
        cached_high_score_q0_n=len(old_high_q0),
        new_competition_pairs_with_both_q=int(len(pair_arr)),
        new_competition_pairs_quality_distinct=int(np.sum(pair_arr[:,2]!=pair_arr[:,3])),
        new_competition_q0_vs_positive=int(np.sum((pair_arr[:,2]==0)&(pair_arr[:,3]>0))),
        sampled_q0_by_stratum={name:int(np.sum(count[strata==name]==0)) for name in STRATA},
        sample_count_by_stratum={name:int(np.sum(strata==name)) for name in STRATA})
    values=dict(query_ids=query_ids,physical_id=physical_ids(label_input),
                grasps=np.asarray(label_input,dtype=np.float64),
                source_score=np.asarray([score_by_query[int(q)] for q in query_ids]),
                strata=strata,quality_count=count,success_bits=bits,
                friction=np.asarray(labels["friction"],dtype=np.float64),
                collision=np.asarray(labels["collision"],dtype=np.uint8),
                empty=np.asarray(labels["empty"],dtype=np.uint8),
                comparable_competition_pairs=pair_arr,
                sensor_valid_query_ids=qids,
                native_nms_keep_query_ids=qids[nms])
    write_npz(npz,values)
    meta=dict(schema=SCHEMA,kind="P1_support_audit_frame",row=row,
              source_pack_sha256=entry["sha256"],
              path=str(npz.resolve()),sha256=sha(npz),
              contract=frame_contract,contract_sha256=digest(frame_contract),
              audit=audit,new_actions=len(query_ids))
    write_json(side,meta)
    return meta


def _selected(plan,domains,frames_per_scene):
    if frames_per_scene not in (1,2,4):
        raise ValueError("Pre-registered frames-per-scene: 1, 2 or 4")
    groups={}
    for entry in plan["entries"]:
        r=entry["row"]
        if r["domain"] not in domains:continue
        if not (0<=int(r["scene"])<100) or r["split"]!="train":
            raise ValueError("Test scene contamination")
        groups.setdefault((r["domain"],int(r["scene"])),[]).append(entry)
    expected={(d,s) for d in domains for s in range(100)}
    if set(groups)!=expected:
        raise ValueError("Support audit requires all 100 train scenes for each selected domain")
    selected=[]
    for key in sorted(groups):
        es=sorted(groups[key],key=lambda e:int(e["row"]["anno"]))
        if len(es)!=4:
            raise ValueError(f"Source plan must contain 4 accepted frames per {key}")
        indices={1:[0],2:[0,3],4:[0,1,2,3]}[frames_per_scene]
        selected.extend(es[i] for i in indices)
    return selected


def mine(args):
    if not 0<=args.shard_index<args.shards or args.max_compiled_pair_checks<0:
        raise ValueError("Invalid shards or compiler audit budget")
    if args.limit<0:raise ValueError("limit must be nonnegative")
    p=load_plan(args.plan)
    if p["base_checkpoint_sha256"]!=BASE_SHA:
        raise ValueError("Historical checkpoint differs")
    domains=tuple(args.domains.split(","))
    if not set(domains)<=set(("original","gntrans")) or not domains:
        raise ValueError("Select original and/or gntrans")
    all_selected=_selected(p,domains,args.frames_per_scene)
    chosen=[e for i,e in enumerate(all_selected) if i%args.shards==args.shard_index]
    if args.limit:
        chosen=chosen[:args.limit]
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rt,identity=make_runtime(args)
    if rt.config_digest!=p["base_config_digest"]:
        raise RuntimeError("Accepted GVAR runtime differs")
    if int(args.snapshot_seed)!=int(p["trace_contract"]["source_snapshot_seed"]):
        raise ValueError("Snapshot seed differs from accepted trace source")
    from graspnetAPI import GraspGroup
    import grasp_nms
    import inspect
    def compiled(g):
        return GraspGroup(g).nms(.03,float(np.pi/6)).grasp_group_array
    contract=dict(schema=SCHEMA,kind="P1_missing_support_audit",
                  plan_sha256=p["plan_sha256"],source_index_sha256=p["source_index_sha256"],
                  base_checkpoint_sha256=BASE_SHA,bound_config_sha256=sha(args.config),
                  runtime_identity=identity,compiled_nms_sha256=sha(inspect.getfile(grasp_nms)),
                  source_evaluator_sha256=sha("exact_action_graspnet_evaluator.py"),
                  script_sha256=sha(__file__),domains=domains,
                  frames_per_scene=args.frames_per_scene,seed=args.seed,
                  snapshot_seed=args.snapshot_seed,
                  new_native_actions_budget=64,quotas=DEFAULT_QUOTAS,
                  no_test_labels=True,selection_uses_only_native_scores_and_geometry=True)
    entries=[]
    for i,entry in enumerate(chosen,1):
        meta=_frame(args,rt,compiled,entry,root,contract,args.resume)
        entries.append(meta)
        if i%16==0 or i==len(chosen):
            print(f"[P1 support] {i}/{len(chosen)} {frame_name(entry['row'])} "
                  f"new_native_actions={meta['new_actions']}",flush=True)
    artifact=dict(schema=SCHEMA,kind="P1_support_shard",
                  contract=contract,contract_sha256=digest(contract),
                  shard_index=args.shard_index,shards=args.shards,
                  partial=bool(args.limit),entries=entries,
                  entries_sha256=[r["sha256"] for r in entries])
    artifact["manifest_sha256"]=digest(artifact)
    path=root/f"support_shard_{args.shard_index:02d}_of_{args.shards:02d}.json"
    if path.exists() and not args.resume:
        raise FileExistsError(path)
    write_json(path,artifact)
    return artifact


def summarize(args):
    root=Path(args.root)
    manifests=sorted(root.glob("support_shard_*_of_*.json"))
    if not manifests:raise FileNotFoundError("Missing P1 support shard manifests")
    contract=None;seen=set();rows=[];shas={}
    shard_expected=None
    for path in manifests:
        m=json.loads(path.read_text(encoding="utf-8"))
        h=m.pop("manifest_sha256",None)
        if digest(m)!=h or m.get("schema")!=SCHEMA or m.get("kind")!="P1_support_shard":
            raise ValueError("Invalid P1 shard signature")
        m["manifest_sha256"]=h
        if contract is None:contract=m["contract"]
        elif contract!=m["contract"]:raise ValueError("Shards use different P1 labeling contracts")
        if shard_expected is None:shard_expected=m["shards"]
        elif shard_expected!=m["shards"]:raise ValueError("Shards count mismatch")
        if m.get("partial") and not args.allow_partial:
            raise ValueError("Formal aggregate excludes smoke shards")
        shas[str(path)]=sha(path)
        for r in m["entries"]:
            row=r["row"]
            key=(row["domain"],row["scene"],row["anno"])
            if key in seen:raise ValueError("Duplicate frame across support shards")
            seen.add(key)
            path_np=Path(r["path"])
            if not path_np.is_file() or sha(path_np)!=r["sha256"]:
                raise ValueError(f"Missing/modified support-label frame: {path_np}")
            rows.append(dict(domain=row["domain"],scene=int(row["scene"]),anno=int(row["anno"]),
                             new_actions=int(r["new_actions"]),**r["audit"]))
    expected=100*len(contract["domains"])*contract["frames_per_scene"]
    if not args.allow_partial and (len(rows)!=expected or len(manifests)!=shard_expected):
        raise ValueError(f"Expected all {expected} train-domain frames and {shard_expected} shards")
    if not rows:raise ValueError("Empty support audit")
    total_new=sum(r["new_actions"] for r in rows)
    by_domain={}
    for domain in contract["domains"]:
        part=[r for r in rows if r["domain"]==domain]
        by_domain[domain]=dict(
            frames=len(part),
            total_sensor_valid=sum(r["sensor_valid_n"] for r in part),
            cached_sensor_valid=sum(r["cached_sensor_valid_n"] for r in part),
            missed_top50=sum(r["top50_n"]-r["top50_cached_n"] for r in part),
            missed_competitors=sum(r["competition_nodes_n"]-r["competition_cached_n"] for r in part),
            newly_labeled=sum(r["new_actions"] for r in part),
            new_high_score_q0=sum(r["sampled_high_score_q0_n"] for r in part),
            known_high_score_q0=sum(r["cached_high_score_q0_n"] for r in part),
            sampled_q0_collision=sum(r["sampled_q0_collision_n"] for r in part),
            sampled_q0_empty=sum(r["sampled_q0_empty_n"] for r in part),
            sampled_q0_other=sum(r["sampled_q0_other_n"] for r in part),
            paired_competition_edges=sum(r["new_competition_pairs_with_both_q"] for r in part),
            paired_distinct_quality=sum(r["new_competition_pairs_quality_distinct"] for r in part),
            paired_new_zero_vs_positive_leader=sum(r["new_competition_q0_vs_positive"] for r in part),
            q0_new_sampled=sum(r["sampled_q0_n"] for r in part),
            by_stratum={name:dict(
                labeled=sum(r["sample_count_by_stratum"][name] for r in part),
                q0=sum(r["sampled_q0_by_stratum"][name] for r in part))
                for name in STRATA})
    # Fixed scene groups: scorer training / scorer validation / previously
    # inspected audit. All are backbone-train scenes, not an independent test.
    scene_group_bounds={"probe_fit_scenes_0_79":(0,80),
                        "probe_validation_scenes_80_89":(80,90),
                        "reused_audit_scenes_90_99":(90,100)}
    by_scene_group={}
    for name,(lo,hi) in scene_group_bounds.items():
        part=[r for r in rows if lo <= r["scene"] < hi]
        by_scene_group[name]=dict(
            frames=len(part),new_labels=sum(r["new_actions"] for r in part),
            missed_top50=sum(r["top50_n"]-r["top50_cached_n"] for r in part),
            missed_conflict_neighbors=sum(r["competition_nodes_n"]-r["competition_cached_n"] for r in part),
            new_high_score_zero_quality=sum(r["sampled_high_score_q0_n"] for r in part),
            paired_distinct_quality=sum(r["new_competition_pairs_quality_distinct"] for r in part),
            new_zero_vs_positive_leader=sum(r["new_competition_q0_vs_positive"] for r in part))
    summary=dict(schema=SCHEMA,kind="P1_support_summary",
                 complete=(len(rows)==expected and len(manifests)==shard_expected and not args.allow_partial),
                 frames=len(rows),expected_frames=expected,new_labels=total_new,
                 contract=contract,shard_sha256=shas,by_domain=by_domain,
                 by_scene_group=by_scene_group,
                 limitations=["Labels acquired only for selected new native executed depths, not all unseen families or four depths.",
                              "Sampling by score/competition is not IID; q=0 rates describe sampled strata, not global candidate prevalence.",
                              "Existing U∪D four-depth quality and full sensor candidate geometry are reused without test labels.",
                              "All scenes are training scenes of frozen GVAR, and the Novel test set is never used for acquisition."])
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"summary.json",summary)
    with (out/"per_frame.csv").open("w",newline="",encoding="utf-8") as f:
        # Flatten nested stratum metrics as JSON strings; preserve full details in frame sidecars.
        flat=[]
        for r in rows:
            flat.append({k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v
                         for k,v in r.items()})
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=["# P1 高分零质量／竞争监督支持域审计", "",
           f"旧模型训练域帧：{len(rows)}；新增实际执行深度标签：{total_new}。",
           "本审计不训练；所有新增标签仅属于训练场景。抽样率不是完整未知候选总体的 q0 真值比例。","",
           "| Domain | Sensor-valid | Existing covered | Missed native top50 | Missed conflict neighbors | Newly labeled | New high-score q0 | Labeled new competition pairs |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for domain,r in by_domain.items():
        lines.append(f"| {domain} | {r['total_sensor_valid']} | {r['cached_sensor_valid']} | "
                     f"{r['missed_top50']} | {r['missed_competitors']} | "
                     f"{r['newly_labeled']} | {r['new_high_score_q0']} | "
                     f"{r['paired_competition_edges']} |")
    lines.extend(["", "高分阈值在每帧sensor-valid集合的 native score 第75百分位计算。",
                  "新竞争对要求两端实际质量都已知；几何边是NMS条件，并不等同于历史有向suppressor。"])
    (out/"RESULTS_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="cmd",required=True)
    minep=sp.add_parser("mine")
    minep.add_argument("--plan",required=True)
    minep.add_argument("--config",required=True)
    minep.add_argument("--dataset-root",required=True)
    minep.add_argument("--gntrans-root",required=True)
    minep.add_argument("--factory",default="rare_grasp.runtime:Runtime")
    minep.add_argument("--device",default="cuda:0")
    minep.add_argument("--output",required=True)
    minep.add_argument("--domains",default="original,gntrans")
    minep.add_argument("--frames-per-scene",type=int,choices=(1,2,4),default=2)
    minep.add_argument("--snapshot-seed",type=int,default=20261113)
    minep.add_argument("--seed",type=int,default=20261114)
    minep.add_argument("--shards",type=int,default=1)
    minep.add_argument("--shard-index",type=int,default=0)
    minep.add_argument("--max-compiled-pair-checks",type=int,default=32)
    minep.add_argument("--min-new-actions",type=int,default=1)
    minep.add_argument("--limit",type=int,default=0)
    minep.add_argument("--resume",action="store_true")
    ag=sp.add_parser("summarize")
    ag.add_argument("--root",required=True)
    ag.add_argument("--output",required=True)
    ag.add_argument("--allow-partial",action="store_true")
    args=p.parse_args()
    return mine(args) if args.cmd=="mine" else summarize(args)


if __name__=="__main__":
    main()
