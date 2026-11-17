"""Register independently rendered, different-shape/material CAD eval objects.

Fail closed on the original CAD2/5 train and 8/9 eval mesh identities. A user
must supply real new CAD render sources and a documented lineage/usage review.
"Untouched" is a human declaration and never proven by hashing alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .distance_split_p1 import read_registered,SCHEDULE
from .factorial_contract import (
    sha256,digest,write_json,source_image,verify_fixed_physics
)
from .scale_cues_p1 import rgb_quality

SCHEMA="physical-distance-budget-new-cad-holdout-v1"


def register(a):
    original=read_registered(a.registration)
    entries=json.loads(Path(a.sources_index).read_text(encoding="utf-8"))
    if not isinstance(entries,list) or not entries:
        raise ValueError("Explicit newly rendered CAD source index required")
    review=Path(a.lineage_review)
    if not review.is_file() or len(review.read_text(encoding="utf-8"))<30:
        raise ValueError("An actual CAD instance/mesh history and category review file is required")
    old={r["object_sha"] for r in original["train"]+original["eval"]}
    objects=[];seen=set()
    for row in entries:
        if not isinstance(row,dict) or any(k not in row for k in (
            "manifest","sha256","semantic_group","prior_development_use")):
            raise ValueError("CAD source needs manifest/sha256/group/prior-use declaration")
        p=Path(row["manifest"])
        if not p.is_file() or sha256(p)!=row["sha256"]:
            raise ValueError(f"CAD render source SHA invalid: {p}")
        source=json.loads(p.read_text(encoding="utf-8"))
        verify_fixed_physics(source,strict_files=True)
        plan=source["plan"]
        obj=plan["object"]
        if obj["kind"]!="mesh":
            raise ValueError("Only real SHA-verified CAD meshes, never procedural smoke")
        identity=obj["sha256"]
        if identity in old or identity in seen:
            raise ValueError("CAD was used in existing train/test experiment or duplicated")
        seen.add(identity)
        if not row["semantic_group"] or not isinstance(row["semantic_group"],str):
            raise ValueError("User must supply actual CAD semantic group, no invented labels")
        if row["prior_development_use"] not in ("never","used"):
            raise ValueError("Specify never or used; cannot infer untouched status")
        if source["plan"]["distances_m"]!=SCHEDULE:
            raise ValueError("New test CAD physical distance grid differs from prior")
        if digest(plan["camera"])!=original["camera_protocol_sha256"]:
            raise ValueError("Different camera K/rotation confounds new holdout")
        if digest(plan["render"])!=original["render_protocol_sha256"]:
            raise ValueError("Different Cycles samples/seed confounds holdout")
        appearance=plan.get("appearance",{})
        # Explicit appearance is a material-control attribute. It cannot be
        # inferred from semantic group or object SHA; source manifest records it.
        material_key=digest(appearance if appearance else
                            dict(base_color_linear=[.63,.32,.21],roughness=.85))
        for view in source["views"]:
            rgb,z,mask=source_image(view)
            rgb_quality(rgb,mask)
            if view.get("rgb_exposure_ev")!=-6.:
                raise ValueError("New rendered CAD must use corrected -6EV source, not rejected bright render")
            if float(view.get("projection_max_error_px",float("inf")))>.25:
                raise ValueError("Physical K projection quality gate failed")
            if float(view.get("raycast_z_mae_m",float("inf")))>.008:
                raise ValueError("Rendered optical-depth raycast parity failed")
        objects.append(dict(object_sha=identity,
             original_mesh_file=obj["path"],original_mesh_sha=obj["sha256"],
             semantic_group=row["semantic_group"],
             prior_development_use=row["prior_development_use"],
             material_source_sha256=material_key,
             material=appearance,
             source_manifest=str(p.resolve()),source_sha256=row["sha256"],
             physical_render_origin="corrected -6EV Blender Cycles",
             note="CAD hash disjointness does NOT prove disjoint semantic instances"))
    groups={r["semantic_group"] for r in objects}
    materials={r["material_source_sha256"] for r in objects}
    if not a.exploratory:
        if len(objects)<4 or len(groups)<3 or len(materials)<2:
            raise ValueError("Formal source registration needs >=4 new CAD, >=3 semantic groups, >=2 actual materials")
        if any(x["prior_development_use"]!="never" for x in objects):
            raise ValueError("Some new CAD already participated in prior method development")
    result=dict(schema=SCHEMA,
        status="exploratory_source_extension" if a.exploratory
                else "PROVISIONAL_NEW_CAD_NOT_PRIORLY_INSPECTED_SELF_DECLARED",
        source_list_sha256=sha256(a.sources_index),
        original_registration_path=str(Path(a.registration).resolve()),
        original_registration_sha256=sha256(a.registration),
        original_registration_identity_sha256=original["registration_sha256"],
        lineage_review_file=str(review.resolve()),
        lineage_review_sha256=sha256(review),
        source_render_objects=objects,
        object_count=len(objects),semantic_groups=sorted(groups),
        material_appearance_count=len(materials),
        original_inspected_mesh_exclusions=sorted(old),
        limitations=[
            "Never previously used is a human declaration and remains provisional",
            "Different CAD file SHA alone does not preclude near-duplicate geometry/category",
            "Materials and lighting differ only BETWEEN registered CAD; each CAD's physical distance sequence fixes material and illumination",
            "Synthetic Blender RGB/depth does not prove real GC6D or RGB-only robot performance",
            "No training/eval ground-truth leakage or new grasp AP in this registrar"
        ])
    result["manifest_sha256"]=digest(result)
    write_json(a.output,result)
    return result


def read_holdout(path,registration,*,require_formal=False):
    f=Path(path)
    info=json.loads(f.read_text(encoding="utf-8"))
    signature=info.pop("manifest_sha256",None)
    if signature!=digest(info) or info.get("schema")!=SCHEMA:
        raise ValueError("Invalid signed external CAD holdout")
    info["manifest_sha256"]=signature
    if info["original_registration_sha256"]!=sha256(registration):
        raise ValueError("Original train CAD changed after external CAD registration")
    if require_formal and not info["status"].startswith("PROVISIONAL_"):
        raise ValueError("Formal new-CAD results require predeclared new object categories/materials")
    review=Path(info["lineage_review_file"])
    if not review.is_file() or sha256(review)!=info["lineage_review_sha256"]:
        raise ValueError("CAD lineage review source modified")
    for obj in info["source_render_objects"]:
        f=Path(obj["source_manifest"])
        if not f.is_file() or sha256(f)!=obj["source_sha256"]:
            raise ValueError("External rendered CAD source changed")
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--sources-index",required=True,
                   help='JSON list of {manifest,sha256,semantic_group,prior_development_use}')
    p.add_argument("--lineage-review",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--exploratory",action="store_true")
    register(p.parse_args())


if __name__=="__main__":
    main()
