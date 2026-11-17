"""Preflight 200/600 study against the EXACT accepted original 200-step data.

Don't fabricate historical heads: require actual last.pt bytes plus SHA256 and
their signed CAD test results. The original registration must remain bytewise
unchanged and every underlying CAD render must still pass physical source QC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .distance_split_p1 import read_registered
from .distance_crosscrop_preflight import verify_rois
from .distance_crosscrop_compare import load_evaluation
from .distance_crosscrop_core import verify_four_arm_protocols
from .factorial_contract import sha256,digest,write_json
from .distance_budget_core import budget_schedule,assert_600_pair
from diagnostics.decision_supervision.core import BASE_SHA

SCHEMA="physical-distance-crop-budget-preflight-v1"


def _check_last(path,declared_sha,expected,registration_sha,initial):
    p=Path(path)
    if not p.is_file() or p.name!="last.pt" or sha256(p)!=declared_sha:
        raise ValueError(f"Historical accepted last.pt missing or SHA changed: {path}")
    checkpoint=torch.load(p,map_location="cpu",weights_only=True)
    if checkpoint.get("kind")!="DPT_head_only":
        raise ValueError("Historical DPT head checkpoint not the original loss/target")
    proto=checkpoint.get("protocol",{})
    if (proto.get("arm")!="near_mid_far" or
        proto.get("crop_policy")!=expected or
        proto.get("steps")!=200 or
        proto.get("pixels_per_step")!=256 or
        proto.get("base_checkpoint_sha256")!=BASE_SHA or
        proto.get("initial_depth_head_sha256")!=initial or
        proto.get("train_registration_sha256")!=registration_sha):
        raise ValueError("Historical 200-step DPT-head-only source mismatch")
    return dict(path=str(p.resolve()),sha256=declared_sha,
                protocol=proto)


def preflight(a):
    reg=read_registered(a.registration)
    if reg["frozen_base_checkpoint_sha256"]!=BASE_SHA or len(reg["train"])!=2:
        raise ValueError("Original historical train registration must be 2 fixed CAD/known base")
    # The original ROI near+wide source also checks all train/eval render SHA,
    # 200-step object selection, crop and camera physics.
    verify_rois(a.registration,a.roi_near_eval,a.roi_wide_eval)
    old_reports={
        "roi_near":load_evaluation(a.roi_near_eval,"near_only",sha256(a.registration)),
        "roi_wide":load_evaluation(a.roi_wide_eval,"near_mid_far",sha256(a.registration)),
        "multi_near":load_evaluation(a.multi_near_eval,"near_only",sha256(a.registration)),
        "multi_wide":load_evaluation(a.multi_wide_eval,"near_mid_far",sha256(a.registration)),
    }
    protocols={name:item["train_protocol"] for name,item in old_reports.items()}
    check=verify_four_arm_protocols(protocols,sha256(a.registration))
    base_sha=protocols["roi_wide"]["initial_depth_head_sha256"]
    heads={
        "roi200":_check_last(a.roi200_last,a.roi200_sha,"fixed_roi",
                             sha256(a.registration),base_sha),
        "multi200":_check_last(a.multi200_last,a.multi200_sha,"multi3",
                               sha256(a.registration),base_sha),
    }
    # No unknown last.pt should be accepted as one of the four arms.
    if heads["roi200"]["protocol"]!=protocols["roi_wide"] or \
       heads["multi200"]["protocol"]!=protocols["multi_wide"]:
        raise ValueError("Historical last.pt does not correspond to signed ROI/Multi evaluation")
    future_roi=budget_schedule("roi",600,protocols["roi_wide"]["seed"],2)
    future_multi=budget_schedule("multi3",600,protocols["multi_wide"]["seed"],2)
    expectation=assert_600_pair(future_roi,future_multi)
    registration_sha=sha256(a.registration)
    result=dict(schema=SCHEMA,verified=True,
        original_registration_path=str(Path(a.registration).resolve()),
        original_registration_sha256=registration_sha,
        registration_identity_sha256=reg["registration_sha256"],
        initial_depth_head_sha256=base_sha,
        base_checkpoint_sha256=BASE_SHA,
        old_eval_sha256={
            name:sha256(Path(getattr(a,name+"_eval"))/"summary.json")
            for name in old_reports},
        old_head_sources=heads,
        verified_historical_four_arm_protocol=check,
        expected_600_step_design=expectation,
        historical_CAD_train=[x["object_sha"] for x in reg["train"]],
        historical_CAD_eval=[x["object_sha"] for x in reg["eval"]],
        notes=[
            "Original 200step source hash must not change; do not rerun it and substitute a new checkpoint",
            "The old four CADs were already inspected; all existing results are exploratory",
            "No benchmark/CAD test results are used to select a training checkpoint or learning rate",
            "600 steps are new from-base runs, NOT a continuation of an old 200-step optimizer state",
        ])
    result["manifest_sha256"]=digest(result)
    write_json(a.output,result)
    return result


def read_preflight(path,registration):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    signed=data.pop("manifest_sha256",None)
    if signed!=digest(data) or data.get("schema")!=SCHEMA or not data.get("verified"):
        raise ValueError("Budget preflight signature invalid")
    data["manifest_sha256"]=signed
    if data["original_registration_sha256"]!=sha256(registration):
        raise ValueError("Source training CAD registration changed")
    for name,v in data["old_head_sources"].items():
        if not Path(v["path"]).is_file() or sha256(v["path"])!=v["sha256"]:
            raise ValueError(f"Old {name} last.pt bytes modified")
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    for name in ("roi_near","roi_wide","multi_near","multi_wide"):
        p.add_argument("--"+name.replace("_","-")+"-eval",dest=name+"_eval",required=True)
    p.add_argument("--roi200-last",required=True)
    p.add_argument("--roi200-sha",required=True)
    p.add_argument("--multi200-last",required=True)
    p.add_argument("--multi200-sha",required=True)
    p.add_argument("--output",required=True)
    preflight(p.parse_args())


if __name__=="__main__":
    main()
