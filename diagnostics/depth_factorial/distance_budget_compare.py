"""Object-paired 2×2 compute×crop-depth study: original 200 versus NEW 600.

The four combinations are ROI200, Multi200, ROI600 and Multi600. The two
600 runs start independently from the frozen base; NOT optimizer resumes.
Object-level bootstrap only; the historical two eval CAD are exploratory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .distance_budget_preflight import read_preflight
from .distance_budget_core import budget_schedule,assert_600_pair
from .distance_crosscrop_compare import load_evaluation,_cells,_ci
from .distance_split_p1 import read_registered
from .factorial_contract import CROP_POLICIES,sha256,digest,write_json

SCHEMA="physical-distance-crop-budget-2x2-v1"
ARMS=("roi200","multi200","roi600","multi600")
METRICS=("mae","absrel","slope","delta_abs_error","crop_distance_interaction_did")


def _validate_600(protocol,design,preflight,registration):
    if protocol is None:
        raise ValueError("600step fixed-last training protocol missing")
    old=preflight["old_head_sources"]["roi200"]["protocol"]
    constant=("registration_identity_sha256","base_checkpoint_sha256",
              "initial_depth_head_sha256","learning_rate","weight_decay",
              "smooth_l1_beta","gradient_clip","seed","optimizer",
              "trainable_module","trainable_params")
    for key in constant:
        if protocol.get(key)!=old.get(key):
            raise ValueError(f"600-step and historic 200-step contracts differ: {key}")
    if (protocol.get("steps")!=600 or protocol.get("pixels_per_step")!=256 or
        protocol.get("total_supervised_pixels")!=153600 or
        protocol.get("crop_policy")!=("fixed_roi" if design=="roi" else "multi3") or
        protocol.get("train_registration_sha256")!=sha256(registration)):
        raise ValueError("600-step source budget or crop design not preregistered")
    expected=budget_schedule(design,600,int(protocol["seed"]),
                            objects=len(preflight["historical_CAD_train"]))
    names=preflight["historical_CAD_train"]
    obj=[names[r["object_index"]] for r in expected]
    if obj!=protocol["train_object_sequence"]:
        raise ValueError("600step CAD object sequence mismatch")
    if [r["distance_id"] for r in expected]!=protocol["train_distance_indices"]:
        raise ValueError("600step physical distance source mismatch")
    if protocol["crop_policy"]=="multi3":
        if [r["crop_policy"] for r in expected]!=protocol.get("train_crop_indices"):
            raise ValueError("600step Latin-square crop exposure mismatch")
    if protocol.get("frozen_parameters_sha256_before")!=protocol.get(
            "frozen_parameters_sha256_after"):
        raise ValueError("DINO/pose-FiLM/grasp parameters changed in new training")


def _cells_identity(report):
    return _cells(report)


def run(a):
    if a.bootstrap<1:
        raise ValueError("Bootstrap count must be positive")
    pre=read_preflight(a.preflight,a.registration)
    reg=read_registered(a.registration)
    dirs={arm:Path(getattr(a,arm)) for arm in ARMS}
    reports={arm:load_evaluation(path,"near_mid_far",sha256(a.registration))
             for arm,path in dirs.items()}
    if sha256(dirs["roi200"]/"summary.json")!=pre["old_eval_sha256"]["roi_wide"] or \
       sha256(dirs["multi200"]/"summary.json")!=pre["old_eval_sha256"]["multi_wide"]:
        raise ValueError("200step accepted ROI/Multi evaluation changed")
    ids=("registration_identity_sha256","base_checkpoint_sha256",
         "base_head_parameter_sha256")
    for key in ids:
        if len({str(v.get(key)) for v in reports.values()})!=1:
            raise ValueError(f"Different initial base or registration: {key}")
    _validate_600(reports["roi600"]["train_protocol"],"roi",pre,a.registration)
    _validate_600(reports["multi600"]["train_protocol"],"multi3",pre,a.registration)
    cads=[r["object_sha"] for r in reports["roi200"]["object_analysis"]]
    if set(cads)!={r["object_sha"] for r in reg["eval"]} and not a.external_holdout:
        raise ValueError("Original 2-CAD exploratory evaluator identities changed")
    if any([x["object_sha"] for x in report["object_analysis"]]!=cads
           for report in reports.values()):
        raise ValueError("Four training conditions must evaluate identical CAD in identical order")
    roots={arm:_cells_identity(v) for arm,v in reports.items()}
    if any(set(roots[arm])!=set(roots["roi200"]) for arm in ARMS):
        raise ValueError("Physical CAD/distance/crop candidate sets unequal")
    # All differences are per CAD first, then equal-CAD averages and bootstrap.
    results=[]
    for di in range(5):
        for crop in CROP_POLICIES:
            for metric in METRICS:
                if di==0 and metric not in ("mae","absrel"):continue
                contrasts={key:[] for key in (
                    "budget_roi","budget_multi","crop_at200","crop_at600",
                    "budget_crop_interaction","per_crop_exposure_roi200_vs_multi600")}
                for cad in cads:
                    data={arm:roots[arm][cad,di,crop][metric] for arm in ARMS}
                    contrasts["budget_roi"].append(data["roi600"]-data["roi200"])
                    contrasts["budget_multi"].append(data["multi600"]-data["multi200"])
                    contrasts["crop_at200"].append(data["multi200"]-data["roi200"])
                    contrasts["crop_at600"].append(data["multi600"]-data["roi600"])
                    contrasts["budget_crop_interaction"].append(
                        (data["multi600"]-data["multi200"])-(data["roi600"]-data["roi200"]))
                    contrasts["per_crop_exposure_roi200_vs_multi600"].append(
                        data["multi600"]-data["roi200"])
                record=dict(distance_id=di,distance_m=(.35,.45,.55,.70,.85)[di],
                            eval_crop=crop,metric=metric,cad_objects=len(cads))
                for j,(key,vals) in enumerate(contrasts.items()):
                    record[key]=_ci(vals,a.bootstrap,a.seed+di*127+j*17)
                results.append(record)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new signed budget comparison output")
    root.mkdir(parents=True,exist_ok=True)
    with (root/"per_cell_budget_contrasts.csv").open("w",newline="",encoding="utf-8") as f:
        header=["distance_id","distance_m","eval_crop","metric","cad_objects"]
        for key in results[0]:
            if key not in header:
                header.extend([key+"_mean",key+"_ci95"])
        w=csv.DictWriter(f,fieldnames=header);w.writeheader()
        for r in results:
            line={k:r[k] for k in header[:5]}
            for key in r:
                if key not in header[:5]:
                    line[key+"_mean"]=r[key]["mean"]
                    line[key+"_ci95"]=str(r[key]["ci95"])
            w.writerow(line)
    report=dict(schema=SCHEMA,status="complete",
        original_registration_sha256=sha256(a.registration),
        original_roi_preflight_sha256=sha256(a.preflight),
        evaluation_set="new_declared_CAD" if a.external_holdout else "original_inspected_CAD",
        evaluation_CADs=cads,
        original_MIX50_checkpoint_sha256=pre["base_checkpoint_sha256"],
        model_initial_head_sha256=pre["initial_depth_head_sha256"],
        source_eval_sha256={k:sha256(v/"summary.json") for k,v in dirs.items()},
        study_structure=[
            "Compare ROI200 vs Multi200 and ROI600 vs Multi600 at matched total updates",
            "Compare ROI200 vs Multi600 at matched ROI-crop update count but NOT matched total compute",
            "Compute separate budget and crop main effects and their interaction",
            "Use fixed LAST 200/600, not best validation epoch",
            "No fine-tuning on any evaluated CAD",
        ],
        per_cell=results,
        limitations=[
            "Previously inspected CAD 8/9 remains exploratory and is not untouched",
            "Small CAD count: bootstrap over physical CAD, not point pixels",
            "Multi600 vs ROI200 controls ROI-crop exposures but confounds total update count",
            "ROI600 vs Multi600 controls total update count but ROI exposure differs",
            "Neither comparison alone proves interfering gradients are causal",
            "No real GC6D/GraspNet grasp AP",
        ])
    report["manifest_sha256"]=digest(report)
    write_json(root/"summary.json",report)
    lines=["# P1 物距×Crop预算控制：200/600-step、ROI/Multi","",
       f"评估对象：{len(cads)} CAD；评估来源：{report['evaluation_set']}。",
       "历史两组200-step固定LAST直接复用；新600-step两组从同一MIX50基座独立训练。","",
       "| 物距(m) | 测试Crop | 指标 | ROI600−ROI200 | Multi600−Multi200 | 交互 |",
       "|---|---|---|---:|---:|---:|"]
    for item in results:
        if item["distance_id"]!=4 or item["metric"] not in ("mae","slope","delta_abs_error"):
            continue
        lines.append(f"| {item['distance_m']:.2f} | {item['eval_crop']} | "
                     f"{item['metric']} | {item['budget_roi']['mean']:+.5f} | "
                     f"{item['budget_multi']['mean']:+.5f} | "
                     f"{item['budget_crop_interaction']['mean']:+.5f} |")
    lines.extend(["","请分别检查全距离MAE、同点β和Δz误差的方向。单独远距MAE下降不能证明物理距离响应正确。",
        "两CAD或少量新CAD的区间不足以声称未知类别普遍泛化。没有新增grasp AP。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--preflight",required=True)
    for arm in ARMS:p.add_argument("--"+arm.replace("_","-"),dest=arm,required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=4000)
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--external-holdout",action="store_true",
                   help="Only for genuinely separately registered external-CAD eval; does not confer untouched status")
    run(p.parse_args())


if __name__=="__main__":
    main()
