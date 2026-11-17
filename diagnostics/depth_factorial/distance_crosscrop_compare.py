"""2×2 Distance(near/wide) × TrainCrop(ROI/multi3) on CAD eval objects.

ROI near/wide are historical accepted reference heads, not retrained.
New mixed-crop near/wide are compared using identical rendered eval CAD
5 physical distances × 3 crop policies × same-world 3D surface metrics.

Reports both factor MAIN effects and their interaction (difference of
difference of wide-minus-near) using paired CAD objects as the unit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .distance_crosscrop_core import verify_four_arm_protocols
from .factorial_contract import CROP_POLICIES,digest,sha256,write_json

SCHEMA="physical-distance-crosscrop-2x2-v1"
ARMS=("roi_near","roi_wide","multi_near","multi_wide")
METRICS=("mae","absrel","slope","delta_abs_error",
         "crop_distance_interaction_did")


def load_evaluation(root,expected_arm,registration_sha):
    root=Path(root)
    path=root/"summary.json"
    report=json.loads(path.read_text(encoding="utf-8"))
    signed=report.pop("manifest_sha256",None)
    if (signed!=digest(report) or
        report.get("schema")!="physical-distance-head-eval-v1" or
        report.get("status")!="complete" or
        report.get("arm")!=expected_arm):
        raise ValueError(f"Invalid fixed LAST evaluation summary: {path}")
    report["manifest_sha256"]=signed
    if sha256(root/"per_condition.csv")!=report.get("per_condition_sha256"):
        raise ValueError("Tampered per-condition CAD evaluation")
    if report["registration_sha256"]!=registration_sha:
        raise ValueError("Different preregistered CAD evaluation split")
    if len(report.get("object_analysis",[]))<1:
        raise ValueError("No CAD-object evaluation")
    for entry in report["object_analysis"]:
        source=Path(entry["analysis_dir"])/"summary.json"
        if not source.is_file() or sha256(source)!=entry["analysis_sha256"]:
            raise ValueError("Same-world surface CAD analysis bytes modified")
        matches=entry.get("same_surface_rows",[])
        if len(matches)!=4*len(CROP_POLICIES):
            raise ValueError("Missing 4 nonbaseline distances × 3 eval crop conditions")
    return report


def _ci(values,draws,seed):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty/nonfinite CAD object contrasts")
    if len(x)<2:return dict(mean=float(x.mean()),ci95=None,n_cad=len(x))
    rng=np.random.default_rng(seed)
    ids=rng.integers(0,len(x),size=(draws,len(x)))
    return dict(mean=float(x.mean()),
                ci95=np.quantile(x[ids].mean(axis=1),[.025,.975]).tolist(),
                n_cad=len(x))


def _cells(report):
    out={}
    for obj in report["object_analysis"]:
        cad=obj["object_sha"]
        for r in obj["per_condition"]:
            key=(cad,int(r["distance_id"]),r["crop_policy"])
            if key in out:raise ValueError("Repeated CAD distance/crop input")
            out[key]={k:float(r[k]) for k in ("mae","absrel")}
        for r in obj["same_surface_rows"]:
            key=(cad,int(r["distance_id"]),r["crop_policy"])
            if key not in out:raise ValueError("World-surface cell without depth metrics")
            for metric,source in (("slope","slope"),
                        ("delta_abs_error","mean_delta_abs_error"),
                        ("crop_distance_interaction_did","crop_distance_interaction_did")):
                out[key][metric]=float(r[source])
    return out


def run(a):
    if a.bootstrap<1:raise ValueError("bootstrap must be >=1")
    run_paths={arm:getattr(a,arm) for arm in ARMS}
    base_sha=sha256(a.registration)
    labels={"roi_near":"near_only","roi_wide":"near_mid_far",
            "multi_near":"near_only","multi_wide":"near_mid_far"}
    evals={arm:load_evaluation(p,labels[arm],base_sha)
           for arm,p in run_paths.items()}
    roots=list(evals.values())
    identities=("registration_identity_sha256","base_checkpoint_sha256",
                "base_head_parameter_sha256","eval_objects")
    for key in identities:
        if len({str(r[key]) for r in roots})!=1:
            raise ValueError(f"Different physical CAD or model identity: {key}")
    cad_list=[r["object_sha"] for r in roots[0]["object_analysis"]]
    for arm,report in evals.items():
        if [r["object_sha"] for r in report["object_analysis"]]!=cad_list:
            raise ValueError(f"Different test CAD object ordering: {arm}")
    protocols={arm:report["train_protocol"] for arm,report in evals.items()}
    check=verify_four_arm_protocols(protocols,base_sha)
    expected_design={"roi_near":"fixed_roi","roi_wide":"fixed_roi",
                     "multi_near":"multi3","multi_wide":"multi3"}
    for arm,report in evals.items():
        if report["train_crop_policy"]!=expected_design[arm]:
            raise ValueError(f"Unexpected fixed/test crop training assignment: {arm}")
    cells={arm:_cells(report) for arm,report in evals.items()}
    if any(set(cells[arm])!=set(cells["roi_near"]) for arm in ARMS):
        raise ValueError("All 4 trained models must evaluate EXACTLY same CAD×distance×crop")
    records=[]
    for di in range(5):
        for eval_crop in CROP_POLICIES:
            for metric in METRICS:
                if di==0 and metric not in ("mae","absrel"):
                    continue
                deltas=dict(
                    wide_effect_roi=[],wide_effect_multi=[],
                    crop_effect_near=[],crop_effect_wide=[],interaction=[])
                for cad in cad_list:
                    values={arm:cells[arm][cad,di,eval_crop][metric]
                            for arm in ARMS}
                    d_roi=values["roi_wide"]-values["roi_near"]
                    d_multi=values["multi_wide"]-values["multi_near"]
                    c_near=values["multi_near"]-values["roi_near"]
                    c_wide=values["multi_wide"]-values["roi_wide"]
                    deltas["wide_effect_roi"].append(d_roi)
                    deltas["wide_effect_multi"].append(d_multi)
                    deltas["crop_effect_near"].append(c_near)
                    deltas["crop_effect_wide"].append(c_wide)
                    deltas["interaction"].append(d_multi-d_roi)
                record=dict(distance_id=di,distance_m=(.35,.45,.55,.70,.85)[di],
                            eval_crop=eval_crop,metric=metric,
                            cad_objects=len(cad_list))
                for k,vs in deltas.items():
                    record[k]=_ci(vs,a.bootstrap,a.seed+di*17+len(k))
                records.append(record)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    fields=("distance_id","distance_m","eval_crop","metric","cad_objects",
            "wide_effect_roi","wide_effect_multi","crop_effect_near",
            "crop_effect_wide","interaction")
    with (root/"per_cell_factor_effects.csv").open("w",newline="",encoding="utf-8") as f:
        keys=["distance_id","distance_m","eval_crop","metric","cad_objects"]
        for m in fields[5:]:
            keys.extend([f"{m}_mean",f"{m}_ci95"])
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
        for item in records:
            row={k:item[k] for k in keys[:5]}
            for name in fields[5:]:
                row[f"{name}_mean"]=item[name]["mean"]
                row[f"{name}_ci95"]=str(item[name]["ci95"])
            wr.writerow(row)
    out=dict(schema=SCHEMA,status="complete",
        original_registration_sha256=base_sha,
        verified_equal_budget=check,
        test_cad_objects=cad_list,
        source_eval_manifest_sha256={
            arm:sha256(Path(path)/"summary.json") for arm,path in run_paths.items()},
        records=records,
        interpretation=[
            "Negative MAE interaction means multi3 improved far-distance supervision effect relative to fixed ROI",
            "Positive beta interaction means multi3 improved distance response above the ROI-only wide-minus-near effect",
            "Only two previously inspected CAD eval objects: object CIs are weak exploratory ranges",
            "No model selection or new grasp AP, and do not call this untouched CAD benchmark",
            "A training crop effect can couple appearance, resolution and effective K; no unique causal K mechanism proven",
        ])
    out["manifest_sha256"]=digest(out)
    write_json(root/"summary.json",out)
    lines=["# P1 2×2物距覆盖×训练Crop联合诊断","",
        "历史：ROI-near / ROI-wide；新增：MultiCrop-near / MultiCrop-wide。",
        "四组相同初始模型、CAD、优化器步数、监督像素、样本序列；宽域和crop预先正交编排。",
        "",
        "| 目标物距 | 测试crop | 指标 | Wide增益（ROI训练） | Wide增益（Multi训练） | 交互（Multi−ROI） |",
        "|---|---|---|---:|---:|---:|"]
    for row in records:
        if row["distance_id"]!=4 or row["metric"] not in ("mae","slope","delta_abs_error"):
            continue
        lines.append(f"| {row['distance_m']:.2f} | {row['eval_crop']} | "
                     f"{row['metric']} | {row['wide_effect_roi']['mean']:+.5f} | "
                     f"{row['wide_effect_multi']['mean']:+.5f} | "
                     f"{row['interaction']['mean']:+.5f} |")
    lines.extend(["","先检查adaptive β是否保持正向、所有crop远距离MAE是否同时改善，以及近距离性能是否崩溃。",
        "被使用的合成CAD只有训练对象2与测试对象2，本结果不构成真实GC6D或未知类别的验证。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    for arm in ARMS:p.add_argument("--"+arm.replace("_","-"),required=True,dest=arm)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=4000)
    p.add_argument("--seed",type=int,default=20261117)
    run(p.parse_args())


if __name__=="__main__":
    main()
