"""P1b preregistered CAD-object split for matched near-only/near+far training.

Do NOT infer an "unseen" benchmark from prior four inspected CAD assets. A
hash-signed train/eval inventory and explicit overlap review are required;
object-mesh SHA identity and render RGB health are verified before training.
This registration creates NO trainable model and consumes NO eval labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .factorial_contract import (PLAN_SCHEMA,sha256,digest,write_json,
                                  verify_fixed_physics,source_image)
from .scale_cues_p1 import rgb_quality

SCHEMA="depth-distance-cad-object-split-v1"
SCHEDULE=[.35,.45,.55,.70,.85]


def _read_inventory(path, *, allow_procedural=False):
    items=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items,list) or not items:
        raise ValueError("Expected explicit nonempty CAD render source JSON list")
    records=[];seen=set()
    for item in items:
        if not isinstance(item,dict) or not {"manifest","sha256"}.issubset(item):
            raise ValueError("Every CAD source must have manifest + actual SHA")
        p=Path(item["manifest"])
        if not p.is_file() or sha256(p)!=item["sha256"]:
            raise ValueError(f"Changed/missing CAD source: {p}")
        raw=json.loads(p.read_text(encoding="utf-8"))
        verify_fixed_physics(raw)
        obj=raw["plan"]["object"]
        if obj["kind"]!="mesh" and not allow_procedural:
            raise ValueError("Formal controlled training requires real signed CAD mesh")
        obj_id=(obj["sha256"] if obj["kind"]=="mesh" else digest(obj))
        if obj_id in seen:raise ValueError("Duplicate object identity in CAD source list")
        seen.add(obj_id)
        grid=raw["plan"]["distances_m"]
        if not np.allclose(grid,SCHEDULE,atol=1e-9,rtol=0):
            raise ValueError("Matched experiment requires 0.35/0.45/0.55/0.70/0.85m")
        q=[]
        for frame in raw["views"]:
            rgb,z,mask=source_image(frame)
            q.append(rgb_quality(rgb,mask))
        records.append(dict(manifest=str(p.resolve()),sha256=item["sha256"],
                            object_sha=obj_id,source_plan_sha256=digest(raw["plan"]),
                            rgb_qc_min_std=float(min(v["rgb_std"] for v in q)),
                            rgb_qc_max_white=float(max(v["rgb_white_fraction"] for v in q)),
                            rgb_qc_min_contrast=float(min(v["rgb_fg_bg_contrast"] for v in q))))
    return records


def read_registered(path, *, require_test=True):
    p=Path(path)
    split=json.loads(p.read_text(encoding="utf-8"))
    signature=split.pop("registration_sha256",None)
    if signature!=digest(split) or split.get("schema")!=SCHEMA:
        raise ValueError("CAD object split registration hash/schema mismatch")
    split["registration_sha256"]=signature
    train=split.get("train",[]);test=split.get("eval",[])
    if len(train)<2 or (require_test and len(test)<1):
        raise ValueError("Require at least two train CAD assets and one disjoint evaluation CAD")
    train_id={r["object_sha"] for r in train}
    test_id={r["object_sha"] for r in test}
    if len(train_id)!=len(train) or len(test_id)!=len(test) or train_id&test_id:
        raise ValueError("CAD object SHA train/eval overlap")
    if split.get("frozen_base_checkpoint_sha256")!=(
        "2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675"):
        raise ValueError("Wrong frozen metric depth base")
    review=Path(split["object_overlap_review_file"])
    if not review.is_file() or sha256(review)!=split["object_overlap_review_sha256"]:
        raise ValueError("Registered CAD overlap review source changed")
    for r in train+test:
        p=Path(r["manifest"])
        if not p.is_file() or sha256(p)!=r["sha256"]:
            raise ValueError(f"CAD source modified since split registration: {p}")
    return split


def register(a):
    train=_read_inventory(a.train_sources,allow_procedural=a.smoke)
    test=_read_inventory(a.eval_sources,allow_procedural=a.smoke)
    if set(x["object_sha"] for x in train)&set(x["object_sha"] for x in test):
        raise ValueError("Train/eval CAD hashes overlap; never train on same object")
    if len(train)<2 or len(test)<1:
        raise ValueError("Formal near/far budget needs >=2 train and >=1 eval CAD")
    overlap=Path(a.object_overlap_review)
    if not overlap.is_file() or len(overlap.read_text(encoding="utf-8"))<10:
        raise ValueError("Real CAD object lineage/duplicate review file required")
    info=dict(schema=SCHEMA,
        train=train,eval=test,
        train_source_index_sha256=sha256(a.train_sources),
        eval_source_index_sha256=sha256(a.eval_sources),
        object_overlap_review_sha256=sha256(overlap),
        object_overlap_review_file=str(overlap.resolve()),
        fixed_physical_distances_m=SCHEDULE,
        frozen_base_checkpoint_sha256=(
            "2c7a4097b6350935640d249aa1c0ef2f2871d454417a6d1675"),
        crop_policies=["fixed_full","fixed_roi","adaptive_workspace"],
        status="synthetic_smoke" if a.smoke else "CAD_OBJECT_TRAIN_TEST_DISJOINT_ONLY",
        development_untouched=False,
        caution=[
            "CAD SHA disjointness does not verify semantic duplication or mesh variants",
            "The original four CAD assets were already inspected: split is NOT untouched if reusing them",
            "No evaluation object or distance used to select hyperparameters after registration",
            "The B renderer must hold RGB/material/light/camera fixed across distance; image QC was checked",
            "All results of fine-tuning are SYNTHETIC depth metrics, not real GC6D grasp AP"
        ])
    info["registration_sha256"]=digest(info)
    write_json(a.output,info)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sources",required=True,help='JSON array {"manifest","sha256"}')
    p.add_argument("--eval-sources",required=True)
    p.add_argument("--object-overlap-review",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--smoke",action="store_true",help="Only permits procedural CAD smoke")
    register(p.parse_args())


if __name__=="__main__":
    main()
