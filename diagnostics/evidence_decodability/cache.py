"""Build immutable, native-depth-only action evidence caches for P0/P1.

Reuse the accepted 800-frame U∪D plan and 400-frame support_audit labels.
No new CAD labels. Only original training scenes 0..99 are eligible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA, physical_ids
from diagnostics.decision_supervision.data import load_plan, read_pack, frame_name
from diagnostics.decision_supervision.runtime import make_runtime
from diagnostics.candidate_conflict.core import build_conflict_edges
from .core import SCHEMA, check_bank, digest, sha256, write_npz, write_json
from .capture import GVARCapture, privileged_geometry_from_depth


def _choose_frames(plan):
    """Fixed first/last image per (domain, scene): 100 scenes x 2 domains x 2."""
    groups={}
    for e in plan["entries"]:
        r=e["row"]
        if r["split"]!="train" or not 0<=int(r["scene"])<100:
            raise ValueError("Training-label cache cannot contain test scenes")
        key=(str(r["domain"]),int(r["scene"]))
        groups.setdefault(key,[]).append(e)
    if set(groups)!={(d,s) for d in ("original","gntrans") for s in range(100)}:
        raise ValueError("Expected Original/GN-Trans for all 100 training scenes")
    selected=[]
    for group in sorted(groups):
        entries=sorted(groups[group],key=lambda e:int(e["row"]["anno"]))
        if len(entries)!=4:
            raise ValueError(f"Expected 4 accepted images per {group}")
        selected.extend([entries[0],entries[-1]])
    return selected


def _support_file(root,name, expected_source_sha):
    path=Path(root)/"frames"/f"{name}.npz"
    meta=path.with_suffix(".json")
    if not path.exists() or not meta.is_file():
        raise FileNotFoundError(f"Missing exact-action support: {path}")
    info=json.loads(meta.read_text(encoding="utf-8"))
    if info.get("sha256")!=sha256(path):
        raise RuntimeError(f"Support-label SHA mismatch: {path}")
    if info.get("source_pack_sha256")!=expected_source_sha:
        raise RuntimeError(f"New/old native cache provenance mismatch: {path}")
    with np.load(path,allow_pickle=False) as z:
        x={k:z[k] for k in z.files}
    n=len(x["query_ids"])
    needed=("physical_id","grasps","quality_count","success_bits","collision","empty",
            "source_score","strata","sensor_valid_query_ids","native_nms_keep_query_ids")
    if any(k not in x for k in needed) or any(len(x[k])!=n for k in
                    ("physical_id","grasps","quality_count","success_bits",
                     "collision","empty","source_score","strata")):
        raise ValueError(f"Incomplete new native-action labels: {path}")
    if n!=64:
        raise ValueError(f"Accepted P1 support audit must contain 64 new actions/frame: {n}")
    if x["success_bits"].shape!=(n,6) or not np.array_equal(
            x["success_bits"].sum(1),x["quality_count"]):
        raise ValueError("Quality bits/count mismatch for new actions")
    if not np.array_equal(physical_ids(x["grasps"]),x["physical_id"].astype(str)):
        raise RuntimeError("New labels do not match physical actions")
    return x,path,info


def _old_native(pack):
    from diagnostics.decision_supervision.core import native_rows
    ids=native_rows(pack)
    return dict(q=np.asarray(pack["family"][ids],dtype=np.int64),
                d=np.asarray(pack["depth"][ids],dtype=np.int64),
                score=np.asarray(pack["grasps"][ids,0],dtype=np.float32),
                grasps=np.asarray(pack["grasps"][ids]),
                ids=np.asarray(pack["physical_id"][ids]).astype(str),
                count=np.asarray(pack["count"][ids],dtype=np.int64),
                bits=np.asarray(pack["bits"][ids],dtype=np.uint8),
                collision=np.asarray(pack["collision"][ids],dtype=np.uint8),
                empty=np.asarray(pack["empty"][ids],dtype=np.uint8),
                sensor_valid=np.asarray(pack["native_sensor_valid"],dtype=bool))


def _nms_roles(sensor_valid_queries,snap,top_queries,selected_ids):
    """Infer undirected top50-neighbor status from the same native geometry."""
    valid=np.asarray(sensor_valid_queries,dtype=np.int64)
    top=set(map(int,np.asarray(top_queries,dtype=np.int64)[:50]))
    bank=np.asarray(snap["grasps"])
    d=bank[...,0].argmax(1)
    n=len(valid)
    if n:
        g=bank[valid,d[valid]]
        edges=build_conflict_edges(g)
    else:edges=np.empty((0,2),dtype=np.int64)
    neighbor=set()
    for u,v in edges:
        a,b=int(valid[u]),int(valid[v])
        if a in top and b not in top:neighbor.add(b)
        if b in top and a not in top:neighbor.add(a)
    return np.asarray([2 if int(q) in top else 1 if int(q) in neighbor else 0
                       for q in selected_ids],dtype=np.int8)


def _join_and_validate(snap,old,new):
    q=np.concatenate([old["q"],np.asarray(new["query_ids"],dtype=np.int64)])
    if len(np.unique(q))!=len(q):
        raise RuntimeError("Old and new native query IDs overlap")
    bank=np.asarray(snap["grasps"])
    if bank.shape!=(1024,4,17):
        raise RuntimeError("Strict native 1024x4x17 bank required")
    d_native=bank[...,0].argmax(1)
    d=d_native[q]
    if not np.array_equal(d[:len(old["d"])],old["d"]):
        raise RuntimeError("Old cache native insertion no longer matches model")
    g=bank[q,d]
    oldn=len(old["q"])
    if (not np.array_equal(physical_ids(g[:oldn]),old["ids"])
            or not np.array_equal(physical_ids(g[oldn:]),new["physical_id"].astype(str))):
        raise RuntimeError("Exact native grasp identity mismatch across checkpoints/cache")
    if np.max(np.abs(g[:oldn,:16]-old["grasps"][:,:16]))>1e-6:
        raise RuntimeError("Old native action geometry/score differs")
    if len(new["query_ids"]) and np.max(np.abs(
            g[oldn:,:16]-new["grasps"][:,:16]))>1e-6:
        raise RuntimeError("New native action geometry/score differs")
    nfeat=np.asarray(snap["features"])[old["q"],old["d"]]
    if "features" in old and np.max(np.abs(nfeat-old["features"]))>1e-6:
        raise RuntimeError("Old GVAR cached feature mismatch")
    return q,d,g


def build_one(rt, entry, args, root, contract):
    row=entry["row"]
    name=frame_name(row)
    oldpack=read_pack(entry)
    old=_old_native(oldpack)
    # Cache already contains post-GVAR 64D score features for audit parity.
    from diagnostics.decision_supervision.core import native_rows
    old["features"]=np.asarray(oldpack["features"][native_rows(oldpack)])
    old["logit"]=np.asarray(oldpack["base_logit"][native_rows(oldpack)])
    new,new_path,new_meta=_support_file(args.support_root,name,entry["sha256"])
    out=root/"frames"/f"{name}.npz"
    side=out.with_suffix(".json")
    reference_path=(Path(args.reference_root)/row["domain"]/
                    f"scene_{row['scene']:04d}"/"realsense"/f"{int(row['anno']):04d}.npy"
                    if args.reference_root else None)
    if reference_path is not None and not reference_path.is_file():
        raise FileNotFoundError(f"Missing aligned reference: {reference_path}")
    frame_contract=dict(schema=SCHEMA,plan_sha256=contract["plan_sha256"],
                        source_pack_sha256=entry["sha256"],
                        support_file_sha256=sha256(new_path),
                        base_checkpoint_sha256=BASE_SHA,seed=contract["snapshot_seed"],
                        evidence_code_sha256=contract["evidence_code_sha256"],
                        capture_code_sha256=contract["capture_code_sha256"],
                        reference_config=contract["reference_config"],
                        reference_depth_sha256=sha256(reference_path) if reference_path else None,
                        row=row)
    if out.exists() or side.exists():
        if not args.resume or not (out.is_file() and side.is_file()):
            raise FileExistsError(f"Existing/incomplete feature cache: {out}")
        meta=json.loads(side.read_text(encoding="utf-8"))
        if meta.get("contract_sha256")!=digest(frame_contract) or sha256(out)!=meta.get("sha256"):
            raise RuntimeError(f"Feature cache resume provenance differs: {out}")
        with np.load(out,allow_pickle=False) as z:
            saved={k:z[k] for k in z.files}
        check_bank(saved)
        return meta
    with GVARCapture(rt) as capture:
        snap=rt.snapshot(row,int(contract["snapshot_seed"]))
    q,d,g=_join_and_validate(snap,old,new)
    f=np.asarray(snap["features"])[old["q"],old["d"]]
    if np.max(np.abs(f-old["features"]))>1e-6:
        raise RuntimeError("Old 64D GVAR score feature parity failed")
    b=np.asarray(snap["base_logit"])[old["q"],old["d"]]
    if np.max(np.abs(b-old["logit"]))>1e-6:
        raise RuntimeError("Old CDF base logit parity failed")
    captured=capture.chosen(snap,q,d)
    n_old=len(old["q"])
    bits=np.concatenate([old["bits"],np.asarray(new["success_bits"],dtype=np.uint8)])
    counts=np.concatenate([old["count"],np.asarray(new["quality_count"],dtype=np.int64)])
    collision=np.concatenate([old["collision"],np.asarray(new["collision"],dtype=np.uint8)])
    empty=np.concatenate([old["empty"],np.asarray(new["empty"],dtype=np.uint8)])
    sensor_valid=np.concatenate([
        old["sensor_valid"],
        np.ones(len(new["query_ids"]),dtype=bool)])
    sensor_ids=np.asarray(new["sensor_valid_query_ids"],dtype=np.int64)
    actual=np.isin(old["q"],sensor_ids)
    if not np.array_equal(actual,old["sensor_valid"]):
        raise RuntimeError("Sensor-valid status differs from accepted U/D trace")
    nms_roles=_nms_roles(sensor_ids,snap,new["native_nms_keep_query_ids"],q)
    source=np.concatenate([np.zeros(n_old,dtype=np.uint8),
                           np.ones(len(new["query_ids"]),dtype=np.uint8)])
    values=dict(query_id=q,depth_id=d,native_score=g[:,0].astype(np.float32),
                grasps=g.astype(np.float32),
                physical_id=physical_ids(g),source_new=source,
                sensor_valid=sensor_valid.astype(np.uint8),nms_role=nms_roles,
                score64=np.asarray(snap["features"])[q,d].astype(np.float32),
                base_logit=np.asarray(snap["base_logit"])[q,d].astype(np.float32),
                quality_count=counts,quality_bits=bits,
                pure_collision=collision,empty=empty,
                native_score_percentile=np.asarray([
                    np.mean(np.asarray(snap["grasps"])[sensor_ids,
                        np.asarray(snap["grasps"])[sensor_ids,:,0].argmax(1),0]
                        <=s) if len(sensor_ids) else 0. for s in g[:,0]],dtype=np.float32),
                **captured)
    if args.reference_root:
        if int(row["scene"])>=100:
            raise ValueError("Never export privileged reference geometry on test scenes")
        ref=np.load(reference_path,allow_pickle=False)
        values["reference_geometry"]=privileged_geometry_from_depth(
            rt,capture,snap,q,d,ref,args.reference_unit)
    check_bank(values)
    write_npz(out,**values)
    meta=dict(schema=SCHEMA,kind="native-evidence-frame",row=row,
              path=str(out.resolve()),sha256=sha256(out),
              contract=frame_contract,contract_sha256=digest(frame_contract),
              old_actions=n_old,new_actions=len(new["query_ids"]),
              has_reference=bool(args.reference_root),
              hidden_dim=int(values["pre_reader"].shape[-1]))
    write_json(side,meta)
    return meta


def build(args):
    plan=load_plan(args.plan)
    if plan.get("base_checkpoint_sha256")!=BASE_SHA:
        raise ValueError("Wrong accepted historical GVAR base")
    summary=json.loads(Path(args.support_summary).read_text(encoding="utf-8"))
    if not summary.get("complete") or summary.get("contract",{}).get("plan_sha256")!=plan["plan_sha256"]:
        raise ValueError("Only complete, accepted P1 train support labels may be used")
    root=Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("Use new evidence-cache root or explicit --resume")
    root.mkdir(parents=True,exist_ok=True)
    rt,rt_id=make_runtime(args)
    if rt.config_digest!=plan["base_config_digest"]:
        raise RuntimeError("Frozen runtime/config not equal to support plan")
    frames=_choose_frames(plan)
    if args.limit:
        frames=frames[:args.limit]
    contract=dict(schema=SCHEMA,kind="evidence-cache",
                  plan_sha256=plan["plan_sha256"],
                  source_index_sha256=plan["source_index_sha256"],
                  base_checkpoint_sha256=BASE_SHA,
                  runtime=rt_id,reference_config=dict(root=str(Path(args.reference_root).resolve())
                           if args.reference_root else None,unit=args.reference_unit),
                  evidence_code_sha256=sha256(__file__),
                  capture_code_sha256=sha256(Path(__file__).with_name("capture.py")),
                  core_code_sha256=sha256(Path(__file__).with_name("core.py")),
                  support_summary_sha256=sha256(args.support_summary),
                  snapshot_seed=int(plan["trace_contract"]["source_snapshot_seed"]),
                  test_labels_used=False,train_scene_range=[0,100])
    records=[]
    for i,e in enumerate(frames,1):
        rec=build_one(rt,e,args,root,contract)
        records.append(rec)
        if i%16==0 or i==len(frames):
            print(f"[evidence-cache] {i}/{len(frames)} {frame_name(e['row'])} "
                  f"old={rec['old_actions']} new={rec['new_actions']}",flush=True)
    mf=dict(schema=SCHEMA,kind="evidence-cache-manifest",
            complete=not bool(args.limit) and len(records)==400,
            contract=contract,records=records,num_frames=len(records),
            total_actions=sum(x["old_actions"]+x["new_actions"] for x in records))
    mf["manifest_sha256"]=digest(mf)
    output=root/"manifest.json"
    if output.exists():
        if not args.resume or json.loads(output.read_text())!=mf:
            raise RuntimeError("Changed evidence manifest on resume")
    else:
        write_json(output,mf)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True)
    p.add_argument("--support-root",required=True)
    p.add_argument("--support-summary",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--gntrans-root",required=True)
    p.add_argument("--factory",default="rare_grasp.runtime:Runtime")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output",required=True)
    p.add_argument("--reference-root",default=None,
                   help="Optional pre-aligned train-scene reference depth ONLY, not for official inference")
    p.add_argument("--reference-unit",choices=("m","mm"),default="m")
    p.add_argument("--limit",type=int,default=0,help="Smoke only, cannot be formal training")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.limit<0:
        raise ValueError("Invalid smoke limit")
    build(args)


if __name__=="__main__":
    main()
