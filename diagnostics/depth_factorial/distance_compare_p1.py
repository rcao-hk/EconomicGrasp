"""P1b object-paired, equal-budget near-only vs near+far depth analysis.

Compares full 5×3 evaluation cells and same-world-surface β, depth-change
MAE and distance×crop DiD on the SAME held-out CAD objects. Bootstrap unit
is CAD object (never pixels). With one eval object CI is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .factorial_contract import CROP_POLICIES,sha256,digest,write_json
from .distance_eval_p1 import EVAL_SCHEMA

SCHEMA="physical-distance-equal-budget-comparison-v1"


def _read(path,arm):
    path=Path(path)
    file=path/"summary.json"
    obj=json.loads(file.read_text(encoding="utf-8"))
    signature=obj.pop("manifest_sha256",None)
    if signature!=digest(obj) or obj.get("schema")!=EVAL_SCHEMA or obj.get("arm")!=arm:
        raise ValueError(f"Unverified evaluation for arm {arm}: {file}")
    obj["manifest_sha256"]=signature
    if obj.get("status")!="complete" or obj.get("evaluation_used_for_training") is not False:
        raise ValueError("Incomplete/contaminated frozen evaluation")
    if sha256(path/"per_condition.csv")!=obj["per_condition_sha256"]:
        raise ValueError("Evaluation CSV changed")
    return obj


def _paired_object_ci(v,draws,seed):
    x=np.asarray(v,dtype=np.float64)
    if not np.isfinite(x).all() or not len(x):raise ValueError("Invalid object metric")
    if len(x)==1:return dict(mean=float(x[0]),ci95=None,objects=1)
    rng=np.random.default_rng(seed)
    ids=rng.integers(0,len(x),size=(draws,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[ids].mean(axis=1),[.025,.975]).tolist(),
                objects=int(len(x)))


def run(a):
    if a.bootstrap<1:raise ValueError("Invalid bootstrap")
    paths=dict(native=a.native,near_only=a.near,near_mid_far=a.wide)
    reports={arm:_read(path,arm) for arm,path in paths.items()}
    base=reports["native"]
    for arm,obj in reports.items():
        if obj["registration_sha256"]!=base["registration_sha256"] or \
           obj["base_checkpoint_sha256"]!=base["base_checkpoint_sha256"]:
            raise ValueError("Different CAD split or frozen base across experiments")
    near=reports["near_only"]["train_protocol"]
    wide=reports["near_mid_far"]["train_protocol"]
    if not isinstance(near,dict) or not isinstance(wide,dict):
        raise ValueError("Missing trained head protocols")
    constants=("registration_identity_sha256","base_checkpoint_sha256",
       "initial_depth_head_sha256","steps","pixels_per_step",
       "total_supervised_pixels","learning_rate","weight_decay",
       "smooth_l1_beta","gradient_clip","crop_policy","seed",
       "optimizer","trainable_module","trainable_params","train_object_sequence")
    for key in constants:
        if near.get(key)!=wide.get(key):
            raise ValueError(f"Unfair near/far training protocols differ in {key}")
    n=int(near["steps"])
    if (near["train_distance_indices"]!=[0]*n or
        wide["train_distance_indices"]!=[([0,2,4][i%3]) for i in range(n)]):
        raise ValueError("Distance is not the ONLY treatment factor")
    objs=[r["object_sha"] for r in base["object_analysis"]]
    if any([r["object_sha"] for r in reports[arm]["object_analysis"]]!=objs
           for arm in paths):
        raise ValueError("Three arms must evaluate identical CAD objects")
    per=[]
    for arm,report in reports.items():
        for obj in report["object_analysis"]:
            for r in obj["per_condition"]:
                per.append(dict(arm=arm,object_sha=obj["object_sha"],
                                distance_id=r["distance_id"],crop_policy=r["crop_policy"],
                                mae=float(r["mae"]),absrel=float(r["absrel"]),
                                median_pred=float(r["median_pred"]),
                                median_reference=float(r["median_reference"])))
            for row in obj["same_surface_rows"]:
                per.append(dict(arm=arm,object_sha=obj["object_sha"],
                                distance_id=row["distance_id"],
                                crop_policy=row["crop_policy"],
                                slope=float(row["slope"]),
                                delta_abs_error=float(row["mean_delta_abs_error"]),
                                crop_distance_interaction_did=float(row["crop_distance_interaction_did"])))
    ckey={(r["arm"],r["object_sha"],r["distance_id"],r["crop_policy"]):
          [] for r in per}
    for r in per:ckey[(r["arm"],r["object_sha"],r["distance_id"],r["crop_policy"])].append(r)
    records=[]
    for distance_id in range(5):
        for policy in CROP_POLICIES:
            for target in ("mae","absrel","median_pred","slope",
                           "delta_abs_error","crop_distance_interaction_did"):
                if distance_id==0 and target in (
                        "slope","delta_abs_error","crop_distance_interaction_did"):
                    continue
                differences={arm:[] for arm in ("near_only","near_mid_far")}
                contrast=[]
                for obj in objs:
                    values={}
                    for arm in paths:
                        items=ckey.get((arm,obj,distance_id,policy),[])
                        matches=[r[target] for r in items if target in r]
                        if len(matches)!=1:raise ValueError(
                            f"Missing/duplicate matched metric {target}/{arm}/{obj}/{distance_id}/{policy}")
                        values[arm]=matches[0]
                    for arm in differences:
                        differences[arm].append(values[arm]-values["native"])
                    contrast.append(values["near_mid_far"]-values["near_only"])
                stat=_paired_object_ci(contrast,a.bootstrap,a.seed+distance_id)
                records.append(dict(distance_id=distance_id,crop_policy=policy,
                    metric=target,wide_minus_near=stat,
                    near_minus_native=_paired_object_ci(
                        differences["near_only"],a.bootstrap,a.seed+11),
                    wide_minus_native=_paired_object_ci(
                        differences["near_mid_far"],a.bootstrap,a.seed+22)))
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in records:
        row=dict(distance_id=r["distance_id"],crop_policy=r["crop_policy"],
                 metric=r["metric"])
        for name in ("wide_minus_near","near_minus_native","wide_minus_native"):
            v=r[name];row[f"{name}_mean"]=v["mean"]
            row[f"{name}_ci95"]=str(v["ci95"])
        rows.append(row)
    with (root/"paired_object_contrasts.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(schema=SCHEMA,status="complete",
       registration_sha256=base["registration_sha256"],
       base_checkpoint_sha256=base["base_checkpoint_sha256"],
       training_equal_budget_verified=True,
       steps=n,pixels_per_step=near["pixels_per_step"],
       trained_crop_policy=near["crop_policy"],
       evaluation_objects=len(objs),model_seed=near["seed"],
       eval_sources={arm:sha256(Path(paths[arm])/"summary.json") for arm in paths},
       paired=records,
       limitations=[
           "Four originally inspected CAD objects and any subsets are NOT untouched",
           "Object-level CIs require multiple truly distinct evaluation CADs",
           "Near-only repeats near distance at exactly same optimizer/pixel budget",
           "Synthetic depth performance is not a GC6D real grasp success/AP improvement",
           "RGB render photometry and object CAD identity must be externally audited",
           "No hyperparameter/model selection on these evaluation results"
       ])
    result["manifest_sha256"]=digest(result)
    write_json(root/"summary.json",result)
    lines=["# P1b near-only 与 near+far：等预算、CAD对象隔离比较","",
           f"训练步数={n}，每步物理深度像素数={near['pixels_per_step']}，"
           f"测试CAD对象={len(objs)}，训练crop={near['crop_policy']}。","",
           "| 距离index | Crop | 指标 | Wide−Near | 对象级CI |",
           "|---|---|---|---:|---|"]
    for r in records:
        if r["metric"] not in ("mae","slope","delta_abs_error"):continue
        v=r["wide_minus_near"]
        lines.append(f"| {r['distance_id']} | {r['crop_policy']} | {r['metric']} | "
                     f"{v['mean']:+.5f} | {v['ci95']} |")
    lines.extend(["","最终测试权重必须为固定last.pt，不按GC6D/Novel的AP选模型。",
                  "训练两arm仅改变物理距离，未训练DINO/pose-FiLM/GVAR/grasp模块。",
                  "本对照的数据是合成渲染场景，不能视为真实跨域Depth Head能力已经改善。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",required=True)
    p.add_argument("--near",required=True)
    p.add_argument("--wide",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=4000)
    p.add_argument("--seed",type=int,default=20261117)
    run(p.parse_args())


if __name__=="__main__":
    main()
