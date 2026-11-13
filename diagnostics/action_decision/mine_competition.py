"""Optional P1b: label native high-ranked and suppressed query families.

Each frame reuses the original 32 uniform-family exact labels, adds 32
decision-relevant families selected WITHOUT reading new true quality, and
re-evaluates only genuinely new executed physical actions. Captures matching
GVAR hidden features simultaneously. It writes RARE-compatible train labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rare_grasp.core import (
    SCHEMA as RARE_SCHEMA, physical_ids, sha256, digest, write_json,
    frame_key,validate_cache,success_bits,
)
from rare_grasp.pipeline import load_index, arrays, atomic_npz
from diagnostics.candidate_conflict.core import build_conflict_edges,greedy_nms
from diagnostics.action_decision.run_quality import _capture_full
from diagnostics.action_decision.feature_cache import SCHEMA as FEATURE_SCHEMA


def choose_queries(snapshot, uniform_ids, filtered, high=32):
    """Selection uses predicted score, NMS edges and collision filter only.

    Captures both suppressor and suppressed queries. Never sees true Q.
    """
    g=np.asarray(snapshot["grasps"],dtype=np.float32)
    if g.shape!=(1024,4,17):raise RuntimeError("Expected full native 1024×4 grasps")
    score=g[...,0].max(1)
    d=g[...,0].argmax(1)
    native=g[np.arange(1024),d]
    keep_pids=set(physical_ids(filtered))
    native_pids=physical_ids(native)
    ids=np.asarray([i for i,pid in enumerate(native_pids) if pid in keep_pids],dtype=int)
    selected=[]
    num_conflicts=0
    if len(ids):
        candidates=native[ids]
        edges=build_conflict_edges(candidates)
        kept,suppressor,order=greedy_nms(score[ids],edges)
        num_conflicts=int(len(edges))
        # Highest-scoring surviving queries and their actual suppressed
        # neighbors, in the native greedy order (before objective selection).
        for j in kept:
            q=int(ids[j])
            selected.append(q)
            others=np.flatnonzero(suppressor==int(j))
            others=others[np.argsort(score[ids[others]])[::-1]]
            selected.extend(ids[others].astype(int).tolist())
            if len(selected)>=high*4:break
        selected.extend(ids[order].astype(int).tolist())
    selected.extend(np.argsort(score)[::-1].astype(int).tolist())
    uniform=set(int(x) for x in np.asarray(uniform_ids).reshape(-1))
    extra=[];seen=set(uniform)
    for q in selected:
        q=int(q)
        if q not in seen:
            extra.append(q);seen.add(q)
        if len(extra)>=high:break
    if len(extra)<high:
        raise RuntimeError("Unable to sample unique extra query families")
    all_ids=np.asarray(sorted(uniform|set(extra)),dtype=np.int64)
    if len(all_ids)!=len(uniform)+high:
        raise RuntimeError("Selection family budget changed")
    return all_ids,np.asarray(extra,dtype=np.int64),dict(
        initial_prefilter_queries=len(ids),
        native_conflict_edges=num_conflicts,
        selected_uniform=len(uniform),selected_extra=len(extra),
        chosen_extra_prefilter=int(sum(x in set(ids.tolist()) for x in extra)))


def build_one(rt,entry,base_seed,original_index_sha,output,extra_count,smoke):
    row=entry["row"];key=frame_key(row)
    label_path=Path(output)/"labels"/(key+".npz")
    feature_path=Path(output)/"features"/"frames"/(key+".npz")
    label_meta=label_path.with_suffix(".json")
    if any(x.exists() for x in (label_path,feature_path,label_meta)):
        if not (label_path.exists() and label_meta.exists() and feature_path.exists()):
            raise RuntimeError(f"Incomplete existing P1b cache {key}")
        meta=json.loads(label_meta.read_text(encoding="utf-8"))
        if meta["source_label_sha256"]!=entry["sha256"] or meta["source_index_sha256"]!=original_index_sha:
            raise RuntimeError(f"P1b source identity mismatch {key}")
        if sha256(label_path)!=meta["npz_sha256"] or sha256(feature_path)!=meta["reader_npz_sha256"]:
            raise RuntimeError(f"P1b cached bytes changed {key}")
        return meta

    source=arrays(entry)
    source_q=np.unique(source["family"])
    if len(source_q)!=32:
        raise ValueError("P1b requires existing 32-family original uniform cache")
    snap,full,full_added=_capture_full(rt,row,base_seed)
    # Validate uniform source action and seed by physical identity.
    old_q=np.asarray(source["family"],dtype=np.int64)
    old_d=np.asarray(source["depth"],dtype=np.int64)
    old_grasp=snap["grasps"][old_q,old_d]
    if not np.array_equal(physical_ids(old_grasp),source["physical_id"]):
        raise RuntimeError(f"Source uniform action bank changed: {key}")
    native_winner=snap["grasps"][np.arange(1024),snap["grasps"][...,0].argmax(1)]
    filtered=rt.sensor_collision_filter(row,native_winner)
    qids,extra,stats=choose_queries(snap,source_q,filtered,extra_count)
    all_q=np.repeat(qids,4);all_d=np.tile(np.arange(4,dtype=np.int64),len(qids))
    flat_idx=all_q*4+all_d
    grasps=np.asarray(snap["grasps"][all_q,all_d],dtype=np.float32)
    pids=physical_ids(grasps)
    old_lookup={pid:i for i,pid in enumerate(source["physical_id"])}
    genuinely_new=np.array([i for i,pid in enumerate(pids) if pid not in old_lookup],dtype=np.int64)
    new_result=rt.label(row,grasps[genuinely_new]) if len(genuinely_new) else None

    n=len(grasps)
    count=np.zeros(n,dtype=np.int64)
    bits=np.zeros((n,6),dtype=np.uint8)
    friction=np.full(n,-1,dtype=np.float32)
    collision=np.zeros(n,dtype=bool)
    empty=np.zeros(n,dtype=bool)
    for j,pid in enumerate(pids):
        if pid in old_lookup:
            i=old_lookup[pid]
            for target,source_value in (
                (count,source["count"]),(bits,source["bits"]),
                (friction,source["friction"]),(collision,source["collision"]),
                (empty,source["empty"])):
                target[j]=source_value[i]
    if len(genuinely_new):
        for target,name in ((count,"count"),(bits,"bits"),
                            (friction,"friction"),(collision,"collision"),
                            (empty,"empty")):
            target[genuinely_new]=new_result[name]

    label=dict(
        features=snap["features"][all_q,all_d].astype(np.float32),
        base_logit=snap["base_logit"][all_q,all_d].astype(np.float32),
        family=all_q,depth=all_d,grasps=grasps,
        physical_id=pids,native_angle=snap["angles"][all_q],
        count=count,bits=bits,friction=friction,collision=collision,empty=empty,
    )
    validate_cache(label)
    reader=dict(reader=full_added["reader"][flat_idx].astype(np.float16),
                reader_delta=full_added["reader_delta"][flat_idx].astype(np.float16),
                family=all_q,depth=all_d,physical_id=pids)

    original_contract=entry.get("contract")  # source index entry has no contract
    contract=dict(schema=RARE_SCHEMA,
                  protocol_sha256=rt.config.get("rare_frame_protocol_sha256",
                                        "") or rt.config.get("source_mix_protocol",{}).get("rare_frame_protocol_sha256",""),
                  partition="train",config_digest=rt.config_digest,
                  config=rt.config,families=len(qids),seed=base_seed,
                  angle_policy="frozen_native_winner",
                  sampling="uniform32_plus_native_prefilter_NMS_competitors32",
                  fc_mode="official",
                  labels="exact_executed_actions_no_legacy_mask",
                  collision_prefilter=False,
                  feature_dtype="float32",
                  dataset_root=str(Path(rt.root).resolve()),
                  gntrans_root=str(Path(rt.trans_root).resolve()))
    # Use ORIGINAL rare source protocol fingerprint verbatim; avoid depending on
    # a possibly absent checkpoint-internal metadata synonym.
    label_path.parent.mkdir(parents=True,exist_ok=True)
    feature_path.parent.mkdir(parents=True,exist_ok=True)
    atomic_npz(label_path,**label);atomic_npz(feature_path,**reader)
    rec=dict(row=row,contract=contract,contract_hash=digest(contract),
             npz_sha256=sha256(label_path),
             reader_npz_sha256=sha256(feature_path),
             source_label_sha256=entry["sha256"],
             source_index_sha256=original_index_sha,
             new_exact_actions=int(len(genuinely_new)),
             repeated_exact_actions=n-len(genuinely_new),
             selection_diagnostics=stats)
    write_json(label_meta,rec)
    print(f"[P1b] {key}: {n} actions, reused={rec['repeated_exact_actions']}, "
          f"new={rec['new_exact_actions']}, native_conflict_edges={stats['native_conflict_edges']}",flush=True)
    return rec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument("--index",required=True,help="Original 32-uniform-family RARE train_index.json")
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--gntrans-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--num-shards",type=int,default=1)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--high-families",type=int,default=32)
    p.add_argument("--smoke-scenes",default="",help="Comma scene IDs, smoke only")
    p.add_argument("--max-per-scene",type=int,default=0,help="Smoke only")
    a=p.parse_args()
    from rare_grasp.runtime import Runtime, read_bound_config
    idx=load_index(a.index,"train",allow_partial=False)
    conf=read_bound_config(a.config)
    if digest(conf)!=idx["contract"]["config_digest"] or conf["backbone"]!="point_gvar":
        raise ValueError("P1b training label source must match exact bound GVAR")
    if a.high_families!=32:
        raise ValueError("Main P1b budget requires 32 uniform + 32 high-rank queries")
    rows=idx["entries"]
    if a.smoke_scenes:
        scenes={int(x) for x in a.smoke_scenes.split(",") if x}
        if not scenes or any(x<0 or x>=100 for x in scenes):
            raise ValueError("Invalid smoke scenes")
        rows=[e for e in rows if int(e["row"]["scene"]) in scenes]
    if a.max_per_scene:
        if not a.smoke_scenes: raise ValueError("Frame limit only for declared smoke")
        seen={};reduced=[]
        for e in rows:
            key=(e["row"]["domain"],e["row"]["scene"])
            if seen.get(key,0)>=a.max_per_scene:continue
            reduced.append(e);seen[key]=seen.get(key,0)+1
        rows=reduced
    if not 0<=a.shard_index<a.num_shards:
        raise ValueError("Invalid shards")
    rows=[e for i,e in enumerate(rows) if i%a.num_shards==a.shard_index]
    if not rows:raise ValueError("Empty shard")
    rt=Runtime(a.config,a.dataset_root,a.gntrans_root,a.device)
    original_seed=int(idx["contract"]["seed"])
    # Patch in the source RARE protocol digest as a runtime-only field.
    rt.config["rare_frame_protocol_sha256"]=idx["contract"]["protocol_sha256"]
    records=[]
    for e in rows:
        records.append(build_one(rt,e,original_seed,idx["index_sha256"],
                                 a.output,a.high_families,bool(a.smoke_scenes)))
    totals=dict(frames=len(records),
                new_exact_actions=sum(r["new_exact_actions"] for r in records),
                reused_exact_actions=sum(r["repeated_exact_actions"] for r in records),
                native_conflict_edges=sum(r["selection_diagnostics"]["native_conflict_edges"] for r in records),
                no_native_conflict_frames=sum(r["selection_diagnostics"]["native_conflict_edges"]==0 for r in records))
    print(json.dumps(totals,indent=2))

if __name__=="__main__":
    main()
