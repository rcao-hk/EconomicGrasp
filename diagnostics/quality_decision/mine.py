"""P1: equal-budget uniform vs decision-aware exact-action supervision.

Select 64 complete 4-depth families per frame in BOTH arms. Decision arm uses
32 shared uniform anchors + 32 high-ranked NMS competitors; uniform arm uses
the same first 32 anchors + 32 additional uniform anchors. Labels are computed
once on the union; only train-scene data may be mined.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rare_grasp.core import (make_protocol, frame_key, frame_seed, physical_ids,
                             validate_cache)
from rare_grasp.pipeline import atomic_npz
from .core import SCHEMA, digest, sha
from diagnostics.candidate_conflict.core import build_conflict_edges, greedy_nms


def choose_queries(snapshot, rt, row, seed, count=64):
    if count!=64:
        raise ValueError("Preregistered P1 uses 64 families for each arm")
    bank=np.asarray(snapshot["grasps"],dtype=np.float32)
    if bank.shape!=(1024,4,17):
        raise ValueError("Expected full 1024x4x17 grasp bank")
    rng=np.random.default_rng(frame_seed(row,seed))
    uniform=rng.permutation(1024)[:count].astype(np.int64)
    winners=bank[np.arange(1024),np.argmax(bank[...,0],axis=1)].copy()
    filtered=rt.sensor_collision_filter(row,winners)
    original_id=physical_ids(winners)
    mapping={}
    for q,pid in enumerate(original_id):
        mapping.setdefault(str(pid),q)
    query_ids=[]
    for pid in physical_ids(filtered):
        if str(pid) not in mapping:
            raise RuntimeError("Collision-filter output is not a subset of frozen executed actions")
        query_ids.append(mapping[str(pid)])
    qs=np.asarray(query_ids,dtype=np.int64)
    if len(qs)!=len(filtered):
        raise ValueError("Candidate mapping mismatch")
    if len(filtered):
        edges=build_conflict_edges(filtered)
        keep,suppressor,order=greedy_nms(filtered[:,0],edges)
        top=keep[:50]
        top_set=set(int(x) for x in top)
        # Give genuinely suppressed alternatives first priority. Otherwise
        # top-32 survivor sampling would exhaust the budget before observing a
        # SINGLE conflict pair and could not test decision-aware learning.
        suppressed=np.flatnonzero(suppressor>=0)
        suppressed=sorted((int(i) for i in suppressed
                            if int(suppressor[i]) in top_set),
                          key=lambda i:-float(filtered[i,0]))
        candidates=([int(qs[i]) for i in suppressed[:16]]
                    + [int(qs[i]) for i in top]
                    + [int(qs[i]) for i in suppressed[16:]]
                    + [int(qs[i]) for i in order])
    else:
        candidates=[]
    chosen=list(map(int,uniform[:32]))
    used=set(chosen)
    focused=[]
    for q in candidates:
        if q not in used:
            focused.append(q);used.add(q)
        if len(focused)==32:
            break
    # If graph is sparse, fill missing positions by same-frame uniform samples.
    for q in uniform[32:]:
        if len(focused)==32: break
        if int(q) not in used:
            focused.append(int(q));used.add(int(q))
    if len(focused)<32:
        for q in rng.permutation(1024):
            if int(q) not in used:
                focused.append(int(q));used.add(int(q))
            if len(focused)==32: break
    decision=np.asarray(sorted(chosen+focused),dtype=np.int64)
    uniform=np.sort(uniform)
    if len(decision)!=64 or len(np.unique(decision))!=64 or len(np.unique(uniform))!=64:
        raise RuntimeError("Decision and uniform arms must have equal 64-family budgets")
    audit=dict(pre_collision=1024,post_collision=len(filtered),
               native_nms_survivors=len(keep) if len(filtered) else 0,
               competition_selected=len(focused),
               unique_families_across_arms=len(np.union1d(uniform,decision)),
               shared_first_32=len(set(chosen) & set(map(int,decision))),
               suppressed_queries_selected=len(
                   set(focused) & set(int(qs[i]) for i in suppressed)) if len(filtered) else 0,
               native_survivor_queries_selected=len(
                   set(focused) & set(int(qs[i]) for i in top)) if len(filtered) else 0)
    return uniform,decision,audit


def mine_frame(rt,row,seed,root,contract,resume):
    key=frame_key(row)
    paths={strategy:root/"cache"/f"{key}.{strategy}.npz"
           for strategy in ("uniform","decision")}
    metas={strategy:paths[strategy].with_suffix(".json") for strategy in paths}
    if any(x.exists() for x in (*paths.values(),*metas.values())):
        if not resume or not all(p.exists() and metas[s].exists() for s,p in paths.items()):
            raise FileExistsError(f"Incomplete/existing P1 frame {key}")
        entries=[]
        for strategy,p in paths.items():
            meta=json.loads(metas[strategy].read_text())
            if (meta.get("contract_hash")!=digest(contract) or
                    meta.get("row")!=row or sha(p)!=meta["sha256"]):
                raise RuntimeError(f"P1 resume mismatch: {p}")
            entries.append(dict(row=row,file=str(p.resolve()),sha256=sha(p),
                                strategy=strategy))
        return entries,{"reused":True}

    snap=rt.snapshot(row,seed)
    uniform,decision,audit=choose_queries(snap,rt,row,seed)
    bank=snap["grasps"]
    union=np.union1d(uniform,decision)
    union_grasps=bank[union].reshape(-1,17).copy()
    pids=physical_ids(union_grasps)
    unique_ids,first_index,inverse=np.unique(pids,return_index=True,return_inverse=True)
    unique_grasps=union_grasps[first_index]
    label=rt.label(row,unique_grasps)
    if len(label["count"]) != len(unique_grasps):
        raise RuntimeError("Exact evaluator returned wrong number of labels")
    all_labels={k:np.asarray(v)[inverse] for k,v in label.items()}
    id_to_pos={str(pid):i for i,pid in enumerate(pids)}
    out=[]
    for strategy,ids in (("uniform",uniform),("decision",decision)):
        grasps=bank[ids].reshape(-1,17).copy()
        pos=np.asarray([id_to_pos[str(pid)] for pid in physical_ids(grasps)])
        arrays=dict(features=snap["features"][ids].reshape(-1,64).astype(np.float32),
                    base_logit=snap["base_logit"][ids].reshape(-1).astype(np.float32),
                    grasps=grasps,physical_id=physical_ids(grasps),
                    family=np.repeat(ids,4).astype(np.int64),
                    depth=np.tile(np.arange(4),len(ids)).astype(np.int64),
                    native_angle=np.repeat(snap["angles"][ids],4),
                    **{k:v[pos] for k,v in all_labels.items()})
        validate_cache(arrays)
        p=paths[strategy];p.parent.mkdir(parents=True,exist_ok=True)
        atomic_npz(p,**arrays)
        meta=dict(schema=SCHEMA,contract_hash=digest(contract),
                  row=row,strategy=strategy,sha256=sha(p),
                  native_parity_max_abs=snap["native_parity_max_abs"],
                  sample_audit=audit)
        tmp=metas[strategy].with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta,indent=2,ensure_ascii=False));os.replace(tmp,metas[strategy])
        out.append(dict(row=row,file=str(p.resolve()),sha256=sha(p),strategy=strategy))
    return out,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,help="Strict RARE MIX50 bound GVAR config")
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--gntrans-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=20261113)
    p.add_argument("--frames-per-scene",type=int,default=4)
    p.add_argument("--scene-start",type=int,default=0)
    p.add_argument("--scene-end",type=int,default=90,
                   help="Default train probe 0..79 and validation probe 80..89")
    p.add_argument("--shard-count",type=int,default=1)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if not 0<=a.scene_start<a.scene_end<=100 or a.frames_per_scene<1:
        raise ValueError("Invalid train scene sampling")
    if not 0<=a.shard_index<a.shard_count:
        raise ValueError("Invalid shard configuration")
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.config,a.dataset_root,a.gntrans_root,a.device)
    if rt.config["backbone"]!="point_gvar" or float(rt.config.get("mix_training_fraction",.1))!=.5:
        raise ValueError("P1 is conditioned on the verified 50%+50% GVAR baseline")
    manifest=make_protocol(.5)
    rows=[r for r in manifest["train"]
          if a.scene_start<=r["scene"]<a.scene_end]
    groups={}
    for r in rows: groups.setdefault((r["domain"],r["scene"]),[]).append(r)
    selected=[]
    for key in sorted(groups):
        options=sorted(groups[key],key=lambda r:r["anno"])
        if len(options)<a.frames_per_scene:
            raise ValueError(f"Scene {key} too small")
        ind=np.linspace(0,len(options)-1,a.frames_per_scene,dtype=int)
        selected.extend(options[int(i)] for i in ind)
    selected=[r for i,r in enumerate(selected) if i%a.shard_count==a.shard_index]
    if not selected:raise ValueError("Empty worker shard")
    contract=dict(schema=SCHEMA,kind="decision-cache",sample_count=64,
                  shared_uniform_families=32,decision_families=32,
                  selection="native_post_sensor_NMS_survivors_and_suppressed_neighbors",
                  checkpoint_sha256=rt.config["checkpoint_sha256"],
                  config_digest=rt.config_digest,
                  source_manifest_sha256=manifest["protocol_sha256"],
                  source_geometry="RGB predicted depth",
                  exact_labels=True,seed=a.seed,
                  cache_model="frozen PointRayRoPE+GVAR36",
                  source_core_sha256=sha(Path(__file__).with_name("core.py")),
                  source_mining_sha256=sha(Path(__file__)),
                  dataset_root=str(Path(a.dataset_root).resolve()),
                  gntrans_root=str(Path(a.gntrans_root).resolve()))
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    indexpath=root/f"index_shard{a.shard_index:02d}_of{a.shard_count:02d}.json"
    if indexpath.exists() and not a.resume:raise FileExistsError(indexpath)
    entries=[]
    for i,row in enumerate(selected,1):
        result,audit=mine_frame(rt,row,a.seed,root,contract,a.resume)
        entries.extend(result)
        print(f"[P1 mine] {i}/{len(selected)} {frame_key(row)} {audit}",flush=True)
    data=dict(schema=SCHEMA,kind="decision-cache-index",
              config_digest=rt.config_digest,
              base_checkpoint_sha256=rt.config["checkpoint_sha256"],
              contract=contract,entries=entries,
              expected_entries=len(selected)*2,complete=True)
    data["index_sha256"]=digest(data)
    indexpath.write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print(f"Wrote {len(entries)} cache entries to {indexpath}")


if __name__=="__main__":
    main()
