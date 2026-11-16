"""Create signed provenance ledger or an external common-action bundle manifest.

This tool never invents scene membership or independent evaluation status:
source lists, dataset identity and evaluator provenance are user-supplied,
hashed and checked against the actual new per-frame action files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA
from .contract import SCHEMA, digest, load_manifest, load_frame, save_json, sha256
from .holdout import PROV_SCHEMA, scene_key


def _read_keys(path):
    x=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(x,list) or not x or not all(isinstance(k,str) and "::" in k for k in x):
        raise ValueError(f"Expected explicit nonempty dataset::scene JSON list: {path}")
    if len(set(x))!=len(x):
        raise ValueError("Duplicate scene identifiers")
    return sorted(x)


def ledger(a):
    base=_read_keys(a.base_scenes)
    dev=_read_keys(a.development_scenes)
    # These lists are a claimed provenance lower bound, verified against
    # their actual files by holdout.py. An auditor must also check completeness.
    review=json.loads(Path(a.object_review).read_text(encoding="utf-8"))
    if not isinstance(review,dict) or "documented" not in review:
        raise ValueError("Object/CAD overlap review must explicitly describe its evidence")
    data=dict(
        schema=PROV_SCHEMA,base_checkpoint_sha256=BASE_SHA,
        base_training_scene_keys=base,development_scene_keys=dev,
        base_training_manifest_path=str(Path(a.base_scenes).resolve()),
        base_training_manifest_sha256=sha256(a.base_scenes),
        development_log_path=str(Path(a.development_scenes).resolve()),
        development_log_sha256=sha256(a.development_scenes),
        object_overlap_review={**review,
                               "source_review_sha256":sha256(a.object_review)},
        warning="The ledger checks completeness against declared files and known GraspNet history only; a third party must verify all other prior uses.")
    data["ledger_sha256"]=digest(data)
    save_json(a.output,data)
    print(f"[ledger] {len(base)} base-train keys, {len(dev)} development keys")


def external(a):
    records=json.loads(Path(a.frame_list).read_text(encoding="utf-8"))
    if not isinstance(records,list) or not records:
        raise ValueError("Explicit external frame record list required")
    if not a.dataset_name or a.dataset_name=="GraspNet":
        raise ValueError("Built-in GraspNet is not a new external holdout")
    seen=set();frames=[]
    for row in records:
        required=("scene","anno","file","sha256")
        if any(k not in row for k in required):
            raise ValueError(f"External frame row missing fields: {row}")
        key=(str(row["scene"]),int(row["anno"]))
        if key in seen:raise ValueError(f"Repeated external frame: {key}")
        seen.add(key)
        item=dict(dataset=a.dataset_name,split="external_holdout",
                  scene=str(row["scene"]),anno=int(row["anno"]),
                  file=str(Path(row["file"]).resolve()),sha256=str(row["sha256"]))
        # Exact physical action and CAD quality are validated through the
        # standard strict loader, not by external dataset-name heuristics.
        load_frame(item)
        frames.append(item)
    prov=dict(
        origin="external_manifest",
        dataset_source_sha256=sha256(a.dataset_provenance),
        dataset_source=str(Path(a.dataset_provenance).resolve()),
        external_evaluator_name=a.evaluator_name,
        prior_development_use=(a.prior_use=="used"),
        untouched_self_declaration=(a.prior_use=="never"),
        source_frame_index_sha256=sha256(a.frame_list),
        needs_human_independence_audit=True,
        note="External adapter must generate the same native/score64/all scores and physically exact labels on each action; this tool does not infer missing sensor/camera/evaluator semantics.")
    obj=dict(schema=SCHEMA,base_checkpoint_sha256=BASE_SHA,
             complete=True,frames=frames,provenance=prov)
    obj["manifest_sha256"]=digest(obj)
    save_json(a.output,obj)
    print(f"[external] {len(frames)} verified common-action frames in {a.dataset_name}")


def label_index(a):
    """Index EXISTING exact-action label NPZ files; no evaluator is run."""
    items=json.loads(Path(a.label_files).read_text(encoding="utf-8"))
    if not isinstance(items,list) or not items:
        raise ValueError("Require explicit list of existing label NPZ files")
    seen=set()
    records=[]
    for row in items:
        for key in ("dataset","split","scene","anno","file","sha256"):
            if key not in row:
                raise ValueError(f"Label inventory missing {key}")
        key=(str(row["dataset"]),str(row["split"]),str(row["scene"]),int(row["anno"]))
        if key in seen:
            raise ValueError("Duplicate per-frame label inventory")
        seen.add(key)
        path=Path(row["file"])
        if not path.is_file() or sha256(path)!=row["sha256"]:
            raise ValueError(f"Changed exact label NPZ: {path}")
        with np.load(path,allow_pickle=False) as z:
            required=("physical_id","assigned_obj","pure_collision","empty","friction")
            if any(x not in z for x in required):
                raise ValueError(f"Label NPZ missing required original fields {required}: {path}")
            ids=np.asarray(z["physical_id"]).astype(str)
            n=len(ids)
            if (n!=len(set(ids)) or
                any(np.asarray(z[k]).shape!=(n,) for k in required)):
                raise ValueError(f"Exact-label candidate identity/size inconsistent: {path}")
            if not np.isfinite(z["friction"]).all():
                raise ValueError(f"Nonfinite exact CAD friction labels: {path}")
        records.append(dict(dataset=key[0],split=key[1],scene=key[2],anno=key[3],
                            file=str(path.resolve()),sha256=row["sha256"]))
    result=dict(schema="gvar-exact-label-inventory-v1",
                source_inventory_sha256=sha256(a.label_files),records=records)
    result["index_sha256"]=digest(result)
    save_json(a.output,result)
    print(f"[label-index] {len(records)} exact signed frame-label bundles")


def merge(a):
    """Combine multiple complete labeled matched-action manifests by source SHA."""
    if len(a.manifest)<2:
        raise ValueError("Need at least two separately signed original manifests")
    merged=[]
    seen=set()
    originals={}
    for path in a.manifest:
        src=load_manifest(path,complete=True)
        originals[str(Path(path).resolve())]=sha256(path)
        for rec in src["frames"]:
            key=(rec["dataset"],rec["split"],str(rec["scene"]),int(rec["anno"]))
            if key in seen:
                raise ValueError(f"Overlapping validation/test frame in merge: {key}")
            seen.add(key)
            # Verify every raw action-label NPZ, not just parent hashes.
            load_frame(rec)
            merged.append(rec)
    merged.sort(key=lambda r:(str(r["dataset"]),str(r["split"]),
                               str(r["scene"]),int(r["anno"])))
    result=dict(schema=SCHEMA,base_checkpoint_sha256=BASE_SHA,
                complete=True,frames=merged,
                provenance=dict(origin="validated_manifest_merge",
                                source_manifest_sha256=originals,
                                prior_development_use=True,
                                test_labels_for_analysis_only=True))
    result["manifest_sha256"]=digest(result)
    save_json(a.output,result)
    print(f"[merge] {len(merged)} verified same-action labeled frames")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    l=sub.add_parser("ledger")
    l.add_argument("--base-scenes",required=True,help="Actual dataset::scene JSON list for ALL base-training scenes")
    l.add_argument("--development-scenes",required=True,help="All previously inspected dataset::scene keys")
    l.add_argument("--object-review",required=True,help="JSON evidence/review for CAD/object overlap, do not invent")
    l.add_argument("--output",required=True)
    e=sub.add_parser("external")
    e.add_argument("--dataset-name",required=True)
    e.add_argument("--frame-list",required=True,
                   help="JSON list {scene,anno,file,sha256}, each NPZ uses shared exact action bundle schema")
    e.add_argument("--dataset-provenance",required=True)
    e.add_argument("--evaluator-name",required=True)
    e.add_argument("--prior-use",choices=("never","used"),required=True)
    e.add_argument("--output",required=True)
    m=sub.add_parser("merge")
    m.add_argument("--manifest",action="append",required=True)
    m.add_argument("--output",required=True)
    labels=sub.add_parser("labels")
    labels.add_argument("--label-files",required=True,
                        help="JSON list of already computed exact CAD label frame files with SHA")
    labels.add_argument("--output",required=True)
    a=p.parse_args()
    {"ledger":ledger,"external":external,"labels":label_index,"merge":merge}[a.command](a)


if __name__=="__main__":
    main()
