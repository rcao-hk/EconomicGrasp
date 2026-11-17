"""Fail-closed preflight BEFORE new mixed-crop GPU training.

The historical ROI near/wide heads were already trained on 117 and their
evaluation files were produced by a local, separately audited worktree.
Never silently reconstruct those checkpoints or fabricate their provenance.
Require exact registration SHA and matching physical/object/protocol identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .distance_crosscrop_core import MAIN_STEPS,MAIN_PIXELS,MAIN_SEED,schedule
from .distance_crosscrop_compare import load_evaluation
from .distance_split_p1 import read_registered
from .factorial_contract import sha256,digest,write_json,verify_fixed_physics

SCHEMA="physical-distance-crosscrop-preflight-v1"


def verify_rois(registration,near_root,wide_root):
    reg=read_registered(registration)
    sha=sha256(registration)
    nr=load_evaluation(near_root,"near_only",sha)
    wr=load_evaluation(wide_root,"near_mid_far",sha)
    init=nr["base_head_parameter_sha256"]
    if init!=wr["base_head_parameter_sha256"]:
        raise ValueError("Two historical ROI heads have different frozen initial DPT")
    fields=("registration_identity_sha256","base_checkpoint_sha256",
         "initial_depth_head_sha256","steps","pixels_per_step",
         "total_supervised_pixels","learning_rate","weight_decay",
         "smooth_l1_beta","gradient_clip","crop_policy","seed",
         "optimizer","trainable_module","trainable_params","train_object_sequence")
    a=nr["train_protocol"];b=wr["train_protocol"]
    if not isinstance(a,dict) or not isinstance(b,dict):
        raise ValueError("Missing accepted historical ROI model training protocol")
    for key in fields:
        if a.get(key)!=b.get(key):
            raise ValueError(f"Historical ROI near/wide differ: {key}")
    if (a["steps"]!=MAIN_STEPS or a["pixels_per_step"]!=MAIN_PIXELS or
        a["seed"]!=MAIN_SEED or a["crop_policy"]!="fixed_roi" or
        a["learning_rate"]!=2e-5 or a["weight_decay"]!=1e-4 or
        a["smooth_l1_beta"]!=.03 or a["gradient_clip"]!=1.):
        raise ValueError("Historical training budget/config differs from registered multi3 experiment")
    if a["initial_depth_head_sha256"]!=init:
        raise ValueError("Historical training initial model does not equal evaluated frozen base")
    if a["train_registration_sha256"]!=sha or b["train_registration_sha256"]!=sha:
        raise ValueError("The actual source CAD registration file changed")
    train_objects=[item["object_sha"] for item in reg["train"]]
    if len(train_objects)!=2:
        raise ValueError("Primary 2x2 follow-up requires exactly two original training CADs")
    expected=schedule(MAIN_STEPS,MAIN_SEED,2,"near_only","roi")
    selected=[train_objects[x["object_index"]] for x in expected]
    if selected!=a["train_object_sequence"]:
        raise ValueError("Historical training CAD order not derived from recorded fixed seed")
    for arm,p in (("near_only",a),("near_mid_far",b)):
        expected=schedule(MAIN_STEPS,MAIN_SEED,2,arm,"roi")
        if p["train_distance_indices"]!=[x["distance_id"] for x in expected]:
            raise ValueError("Historical ROI distance sample schedule does not match same-seed experiment")
    for entry in reg["train"]+reg["eval"]:
        source=json.loads(Path(entry["manifest"]).read_text(encoding="utf-8"))
        verify_fixed_physics(source,strict_files=True)
    evalids=[r["object_sha"] for r in reg["eval"]]
    if evalids!=[x["object_sha"] for x in nr["object_analysis"]] or \
       evalids!=[x["object_sha"] for x in wr["object_analysis"]]:
        raise ValueError("Historical ROI evaluation excluded or reordered registered CAD")
    return dict(schema=SCHEMA,verified=True,
        source_registration_sha256=sha,
        source_registration_identity_sha256=reg["registration_sha256"],
        roi_near_eval_sha256=sha256(Path(near_root)/"summary.json"),
        roi_wide_eval_sha256=sha256(Path(wide_root)/"summary.json"),
        initial_depth_head_sha256=init,
        frozen_base_checkpoint_sha256=a["base_checkpoint_sha256"],
        expected_train_object_sequence=selected,
        historical_steps=MAIN_STEPS,historical_pixels_per_step=MAIN_PIXELS,
        historical_seed=MAIN_SEED,
        train_cads=train_objects,eval_cads=evalids,
        note="Verified CAD source for both ROI arms before adding orthogonally crossed training crops")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--roi-near",required=True)
    p.add_argument("--roi-wide",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=verify_rois(a.registration,a.roi_near,a.roi_wide)
    result["manifest_sha256"]=digest(result)
    write_json(a.output,result)
    print(f"[crosscrop preflight] verified trainCAD={len(result['train_cads'])} "
          f"evalCAD={len(result['eval_cads'])} historical ROI steps={MAIN_STEPS}")


if __name__=="__main__":
    main()
