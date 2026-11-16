"""Materialize source-verified matched native/score64/all physical-action labels.

Uses accepted post-sensor pre-NMS 17D dumps, compiled GraspNet NMS and exact
CAD quality. Reuses a signed label cache when available; otherwise requires
an EXPLICIT --allow-relabel action. Never retrains or changes geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA, physical_ids
from diagnostics.decision_supervision.attribution_core import compiled_keep, final_ids
from rare_grasp.core import success_bits
from diagnostics.candidate_conflict.core import official_ap_matrix
from .contract import (SCHEMA, METHODS, SPLITS, sha256, digest, save_npz,
                       save_json, validate_frame, load_manifest)

KNOWN_SOURCE_SCHEMAS=("physical-evidence-decoding-v1","decision-supervision-v1")


def _source(root, split, *, require_official=True):
    root=Path(root)
    mf_path=root/"inference.json"
    mf=json.loads(mf_path.read_text(encoding="utf-8"))
    if (mf.get("schema") not in KNOWN_SOURCE_SCHEMAS or
        mf.get("base_checkpoint_sha256")!=BASE_SHA or
        not mf.get("complete") or mf.get("split")!=split):
        raise ValueError(f"Inference baseline/protocol mismatch: {root}")
    if len(mf.get("records",[]))!=int(mf.get("num_frames",mf.get("frames",-1))):
        raise ValueError(f"Incomplete inference records: {root}")
    if "selection" in mf and "rank_executed" not in mf["selection"] and \
            mf.get("probe_sha256") is not None:
        raise ValueError("Require exactly frozen executed-depth ranking")
    if require_official:
        eval_path=root/"evaluation.json"
        ev=json.loads(eval_path.read_text(encoding="utf-8"))
        if ev.get("inference_sha256")!=sha256(mf_path):
            raise ValueError(f"Official evaluation and inference hash mismatch: {root}")
        if "ap_sha256" in ev:
            ap_path=root/f"ap_{split}_realsense.npy"
            if sha256(ap_path)!=ev["ap_sha256"]:
                raise ValueError("Official AP array source modified")
    keyed={}
    for rec in mf["records"]:
        ident=(int(rec["scene"]),int(rec["anno"]))
        if ident in keyed:
            raise ValueError("Duplicate source frame")
        if "path" not in rec or "sha256" not in rec:
            raise ValueError("Source needs per-frame path and SHA256")
        keyed[ident]=rec
    return mf,root,keyed


def _read_dump(source, key):
    mf,root,keyed=source
    rec=keyed[key]
    path=root/rec["path"]
    if sha256(path)!=rec["sha256"]:
        raise ValueError(f"Changed 17D inference dump: {path}")
    g=np.asarray(np.load(path,allow_pickle=False),dtype=np.float64)
    if g.ndim!=2 or g.shape[1]!=17 or not np.isfinite(g).all():
        raise ValueError("Malformed physical grasp dump")
    return g


def _load_label_index(path):
    if not path:return None
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    signature=obj.pop("index_sha256",None)
    if signature!=digest(obj) or obj.get("schema")!="gvar-exact-label-inventory-v1":
        raise ValueError("Exact-label reuse manifest signature/schema failed")
    records={}
    for r in obj["records"]:
        key=(str(r["dataset"]),str(r["split"]),str(r["scene"]),int(r["anno"]))
        if key in records:raise ValueError("Duplicate exact-label frame")
        records[key]=r
    return records


def _reuse_labels(index, key, chosen):
    item=index.get(key)
    if item is None:return None
    path=Path(item["file"])
    if sha256(path)!=item["sha256"]:
        raise ValueError(f"Accepted exact label cache modified: {path}")
    with np.load(path,allow_pickle=False) as z:
        required=("physical_id","assigned_obj","pure_collision","empty","friction")
        if any(x not in z for x in required):
            raise ValueError("Label cache missing physical failure fields; cannot infer them")
        old={k:z[k] for k in required}
    ids=np.asarray(old["physical_id"]).astype(str)
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate physical ID in source exact labels")
    location={name:i for i,name in enumerate(ids)}
    try:
        selected=np.asarray([location[name] for name in chosen],dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Requested exact physical action absent from label cache: {exc}") from exc
    return {k:np.asarray(v)[selected] for k,v in old.items() if k!="physical_id"}


def _eval_labels(evaluator,scene,anno,grasps):
    if not len(grasps):
        raise ValueError("No compiled NMS survivors to label")
    answer=evaluator.evaluate(scene,anno,np.asarray(grasps,dtype=np.float64))
    fields={k:np.asarray(getattr(answer,k)) for k in
            ("assigned_obj","pure_collision","empty","friction")}
    if any(len(v)!=len(grasps) for v in fields.values()):
        raise ValueError("Exact evaluator returned incomplete action labels")
    return fields


def _frame(methods,key,compiled,exact,index,dataset,split,a):
    arrays={name:_read_dump(value,key) for name,value in methods.items()}
    baseline=arrays["native"]
    n=len(baseline)
    if not n:
        raise RuntimeError("Empty post-sensor grasp pool requires separate zero-AP protocol")
    for name,g in arrays.items():
        if g.shape!=baseline.shape or not np.array_equal(g[:,1:16],baseline[:,1:16]):
            raise RuntimeError(f"Frozen physical actions mismatch at {key}: {name}")
    nms={name:compiled_keep(g,compiled) for name,g in arrays.items()}
    chosen=np.unique(np.concatenate(list(nms.values())))
    if not len(chosen):
        raise RuntimeError("No official NMS survivors")
    physical=physical_ids(baseline[chosen])
    lbl_key=(dataset,split,str(key[0]),int(key[1]))
    labels=_reuse_labels(index,lbl_key,physical) if index is not None else None
    if labels is None:
        if not a.allow_relabel:
            raise ValueError("Signed exact-label cache missing; use explicit --allow-relabel")
        labels=_eval_labels(exact,key[0],key[1],baseline[chosen])
    bad=(np.asarray(labels["pure_collision"],dtype=bool)|
         np.asarray(labels["empty"],dtype=bool))
    bits=success_bits(np.asarray(labels["friction"],dtype=np.float64),bad)
    if bits.shape!=(len(chosen),6):
        raise RuntimeError("Six-threshold exact quality mismatch")
    pct=(baseline[:,0,None] <= baseline[None,:,0]).mean(axis=0)
    out=dict(
        grasps=baseline[chosen].astype(np.float32),
        physical_id=physical,
        native_percentile=pct[chosen].astype(np.float32),
        score_native=baseline[chosen,0].astype(np.float32),
        score_score64=arrays["score64"][chosen,0].astype(np.float32),
        score_all=arrays["all"][chosen,0].astype(np.float32),
        quality_bits=bits.astype(np.uint8),
        assigned_obj=np.asarray(labels["assigned_obj"],dtype=np.int32),
        pure_collision=np.asarray(labels["pure_collision"],dtype=np.uint8),
        empty=np.asarray(labels["empty"],dtype=np.uint8),
    )
    objects=_full_obj(n,chosen,out["assigned_obj"])
    invalid=np.ones(n,dtype=bool)
    invalid[chosen]=bad
    friction=np.full(n,-1.0,dtype=np.float64)
    friction[chosen]=np.asarray(labels["friction"],dtype=np.float64)
    ap_replay={}
    for method in METHODS:
        selected=final_ids(nms[method],arrays[method][:,0],objects)
        # final_ids only reads chosen (NMS-surviving) labels;
        # other positions carry -1 but cannot influence selected set.
        for k in (10,50):
            out[f"top{k}_{method}"]=np.isin(chosen,selected[:k]).astype(np.uint8)
        ap_replay[method]=official_ap_matrix(arrays[method],objects,invalid,
                                               friction,nms[method],arrays[method][:,0])
    validate_frame(out)
    return out,dict(frame=str(key),full_candidates=n,labeled=len(chosen),
                    label_source="signed_cache" if index is not None and lbl_key in index else "strict_official_relabel",
                    nms_survivors={m:len(v) for m,v in nms.items()}),ap_replay


def _full_obj(n,chosen,obj):
    result=np.full(n,-1,dtype=np.int64)
    result[np.asarray(chosen,dtype=np.int64)]=np.asarray(obj,dtype=np.int64)
    return result


def _attach_geometry(rt, row, physical, a, snapshot_seed):
    """Replay frozen predicted GVAR geometry on the same physical native actions.

    Never infer geometry error without a second, separately supplied aligned
    CAD/reference raster; that input is offline analysis only.
    """
    from diagnostics.evidence_decodability.capture import (
        GVARCapture, privileged_geometry_from_depth
    )
    with GVARCapture(rt) as captured:
        snap=rt.snapshot(row,int(snapshot_seed))
    bank=np.asarray(snap["grasps"])
    if bank.shape!=(1024,4,17):
        raise RuntimeError("Different native GVAR query bank on diagnostic replay")
    depth=bank[...,0].argmax(1)
    native=bank[np.arange(1024),depth]
    ids=physical_ids(native)
    positions={}
    for i,ident in enumerate(ids):
        if ident in positions:
            raise RuntimeError("Ambiguous native physical action -> query identity mapping")
        positions[ident]=i
    try:
        q=np.asarray([positions[str(i)] for i in physical],dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"GVAR replay physical action not in frozen snapshot: {exc}") from exc
    d=depth[q]
    out=captured.chosen(snap,q,d)
    values={"pred_geometry":out["pred_geometry"]}
    if a.reference_depth_root:
        path=(Path(a.reference_depth_root)/row["domain"]/
              f"scene_{row['scene']:04d}"/"realsense"/f"{int(row['anno']):04d}.npy")
        if not path.is_file():
            raise FileNotFoundError(f"Offline reference depth missing: {path}")
        depth_ref=np.load(path,allow_pickle=False)
        values["reference_geometry"]=privileged_geometry_from_depth(
            rt,captured,snap,q,d,depth_ref,unit=a.reference_unit)
        values["_reference_sha256"]=sha256(path)
    return values


def collect(a):
    if a.split not in SPLITS or a.split=="external_holdout":
        raise ValueError("External holdouts must use a verified external adapter")
    if a.split.startswith("test_") and a.dataset != "GraspNet":
        raise ValueError("Built-in Official evaluation only supports GraspNet")
    runs=json.loads(Path(a.runs).read_text(encoding="utf-8"))
    if a.split=="probe_validation" and a.dataset not in ("GraspNet","GNTrans"):
        raise ValueError("Built-in validation replay only supports declared GraspNet/ GNTrans domains")
    if set(runs)!=set(METHODS):
        raise ValueError("Supply accepted inference roots for native, score64 and all")
    official=a.split.startswith("test_")
    sources={m:_source(path,a.split,require_official=official) for m,path in runs.items()}
    if a.split=="probe_validation":
        expected_domain="gntrans" if a.dataset=="GNTrans" else "original"
        for name,src in sources.items():
            if src[0].get("domain")!=expected_domain:
                raise ValueError(f"Validation domain mismatch for {name}")
    geometry_runtime=None
    if a.capture_pred_geometry:
        from diagnostics.decision_supervision.runtime import make_runtime
        if not a.config or not a.dataset_root:
            raise ValueError("Full frozen GVAR bound config and dataset root required for geometry replay")
        geometry_runtime,identity=make_runtime(a)
        for m in METHODS:
            src=sources[m][0]
            if src["base_config_digest"]!=geometry_runtime.config_digest:
                raise ValueError("Frozen geometry replay checkpoint/config differs from accepted inference")
    official_arrays={}
    if official:
        start={"test_seen":100,"test_similar":130,"test_novel":160}[a.split]
        for name,(_,root,_) in sources.items():
            path=root/f"ap_{a.split}_realsense.npy"
            ap=np.load(path,allow_pickle=False)
            if (ap.ndim!=4 or ap.shape[0]!=30 or ap.shape[-2:]!=(50,6)
                    or not np.isfinite(ap).all()):
                raise ValueError(f"Accepted official AP shape invalid: {path}")
            official_arrays[name]=ap
    same={k:len(v[2]) for k,v in sources.items()}
    if len(set(same.values()))!=1:
        raise ValueError("Different source frame coverage")
    keys=sorted(sources["native"][2])
    for m in METHODS:
        if set(keys)!=set(sources[m][2]):
            raise ValueError("Source models evaluated different frame IDs")
    # All use exactly the same frozen base and source handling.
    fields=("base_checkpoint_sha256","base_config_digest","interval",
            "snapshot_seed","collision")
    for field in fields:
        values=[s[0].get(field) for s in sources.values()]
        if len({json.dumps(v,sort_keys=True) for v in values})!=1:
            raise ValueError(f"Source protocols differ in {field}")
    if a.limit and not 0<a.limit<len(keys):
        raise ValueError("Smoke limit must be positive and less than complete frame count")
    if a.limit:keys=keys[:a.limit]
    from graspnetAPI import GraspGroup
    import grasp_nms
    def compiled(g):
        return GraspGroup(g).nms(.03,float(np.pi/6)).grasp_group_array
    index=_load_label_index(a.label_manifest)
    evaluator=None
    if a.allow_relabel:
        from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
        evaluator=ExactGraspNetActionEvaluator(
            a.dataset_root,"realsense",split="train" if a.split=="probe_validation" else a.split,
            fc_mode="official",strict=True)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new collector output root")
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,key in enumerate(keys,1):
        arr,audit,replayed=_frame(sources,key,compiled,evaluator,index,a.dataset,a.split,a)
        if geometry_runtime is not None:
            row=dict(domain=("gntrans" if a.dataset=="GNTrans" else "original"),
                     scene=key[0],anno=key[1],
                     split=("train" if a.split=="probe_validation" else a.split),
                     partition=("train" if a.split=="probe_validation" else "test"))
            geo=_attach_geometry(geometry_runtime,row,arr["physical_id"],a,
                                 sources["native"][0]["snapshot_seed"])
            audit["reference_depth_sha256"]=geo.pop("_reference_sha256",None)
            arr.update(geo)
            validate_frame(arr)
        if official:
            for name,ap in replayed.items():
                idx=key[0]-start
                step=int(sources[name][0]["interval"])
                if key[1]%step:
                    raise ValueError("Saved inference annotation grid differs from official AP")
                expected=official_arrays[name][idx,key[1]//step]
                if expected.shape!=(50,6):
                    raise ValueError("Frame official AP matrix malformed")
                err=float(np.max(np.abs(ap-expected)))
                if err>a.ap_tolerance:
                    raise RuntimeError(f"Official AP parity failed: {key}/{name}: {err:.9g}")
                audit[f"ap_parity_maxabs_{name}"]=err
        p=root/"frames"/f"scene_{key[0]:04d}"/f"{key[1]:04d}.npz"
        save_npz(p,**arr)
        rows.append(dict(dataset=a.dataset,split=a.split,scene=key[0],anno=key[1],
                         file=str(p.resolve()),sha256=sha256(p),audit=audit))
        if i%26==0 or i==len(keys):
            print(f"[matched evidence] {i}/{len(keys)} {a.split}",flush=True)
    meta=dict(schema=SCHEMA,base_checkpoint_sha256=BASE_SHA,
              complete=not bool(a.limit),
              provenance=dict(original_source_roots=runs,
                              original_runs_sha256=sha256(a.runs),
                              label_manifest_sha256=sha256(a.label_manifest) if a.label_manifest else None,
                              source="GraspNet post-sensor-pre-NMS compiled-solver survivor-union",
                              test_labels_for_analysis_only=True,
                              verified_official_source=official,
                              prior_development_use=True,
              predicted_geometry_source=("strict_frozen_GVAR_reader_replay" if a.capture_pred_geometry else None),
              reference_geometry_offline_only=bool(a.reference_depth_root)),
              frames=rows)
    meta["manifest_sha256"]=digest(meta)
    save_json(root/"manifest.json",meta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",required=True,
                   help='JSON: {"native":"/accepted/run","score64":"...","all":"..."}')
    p.add_argument("--dataset",default="GraspNet")
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--config",default=None,help="Strict frozen GVAR bound config if capturing predicted geometry")
    p.add_argument("--gntrans-root",default="")
    p.add_argument("--factory",default="rare_grasp.runtime:Runtime")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--capture-pred-geometry",action="store_true",
                   help="Replay actual frozen GVAR role geometry for aligned selected actions")
    p.add_argument("--reference-depth-root",default=None,
                   help="Optional offline aligned reference depth rasters; forbidden as network input")
    p.add_argument("--reference-unit",choices=("m","mm"),default="m"),
    p.add_argument("--label-manifest",default=None,
                   help="Signed per-frame exact-label NPZ index with all physical IDs")
    p.add_argument("--ap-tolerance",type=float,default=1e-6,
                   help="Formal per-frame official AP replay must pass")
    p.add_argument("--allow-relabel",action="store_true",
                   help="Explicitly pay CAD evaluator cost for missing exact labels")
    p.add_argument("--limit",type=int,default=0,help="Smoke only")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if not a.label_manifest and not a.allow_relabel:
        p.error("Require signed label reuse or explicit --allow-relabel")
    if a.reference_depth_root and not a.capture_pred_geometry:
        p.error("Reference geometry requires matching predicted geometry capture")
    if not 0<a.ap_tolerance<=1e-6:
        p.error("Do not loosen existing official AP parity tolerance")
    collect(a)


if __name__=="__main__":
    main()
