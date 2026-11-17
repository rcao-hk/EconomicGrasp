"""Aggregate multiple independently specified physical-object distance×crop runs.

The unit of uncertainty is the CAD/object plan, not an image, camera position,
or hundreds of matched pixels. One-object experiments report no CI.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .factorial_contract import digest,sha256,write_json
from .factorial_analyze import SCHEMA

AGG_SCHEMA="physical-distance-crop-object-aggregate-v1"
METRICS=("slope","mean_delta_abs_error","mean_target_abs_error",
         "crop_distance_interaction_did","mean_true_abs_motion",
         "mean_pred_abs_motion")


def run(a):
    paths=json.loads(Path(a.analyses_index).read_text(encoding="utf-8"))
    if not isinstance(paths,list) or not paths:
        raise ValueError("Supply explicit nonempty result directories JSON array")
    records=[]
    seen=set();expected=None
    for entry in paths:
        f=Path(entry)/"summary.json"
        report=json.loads(f.read_text(encoding="utf-8"))
        signature=report.pop("manifest_sha256",None)
        if signature!=digest(report) or report.get("schema")!=SCHEMA or not report.get("complete"):
            raise ValueError(f"Unverified physical distance×crop analysis: {f}")
        report["manifest_sha256"]=signature
        plan=report["fixed_physical_source"]
        obj=plan["object"]
        ident=("mesh:"+obj["sha256"] if obj["kind"]=="mesh"
               else "box:"+digest(obj))
        if ident in seen:
            raise ValueError("Duplicated CAD geometry cannot count as independent object")
        seen.add(ident)
        grid=tuple(plan["distances_m"])
        if expected is None:
            expected=dict(distances=grid,checkpoint=report["base_checkpoint_sha256"])
        elif expected["distances"]!=grid or expected["checkpoint"]!=report["base_checkpoint_sha256"]:
            raise ValueError("Distance schedule or frozen checkpoint differs across objects")
        for row in report["rows"]:
            if row["distance_id"] not in range(1,len(grid)):
                raise ValueError("Unexpected crossed distance")
            records.append(dict(object_sha=ident,object_result_sha256=sha256(f),
                                **{key:row[key] for key in
                                   ("distance_id","distance_m","crop_policy",
                                    "n_common_surface_points",*METRICS)}))
    output=[]
    keys=sorted({(r["distance_id"],r["crop_policy"]) for r in records})
    for index,key in enumerate(keys):
        vals=[r for r in records if (r["distance_id"],r["crop_policy"])==key]
        if len(vals)!=len(seen):
            raise ValueError(f"Missing distance×crop cell for some objects: {key}")
        row=dict(distance_id=key[0],distance_m=vals[0]["distance_m"],
                 crop_policy=key[1],objects=len(vals))
        for j,metric in enumerate(METRICS):
            v=np.asarray([r[metric] for r in vals],dtype=np.float64)
            if not np.isfinite(v).all():raise ValueError("Nonfinite object response")
            row[metric]=float(v.mean())
            if len(v)<2:
                row[f"{metric}_ci95"]=None
            else:
                rng=np.random.default_rng(a.seed+index*13+j)
                samples=rng.integers(0,len(v),size=(a.bootstrap,len(v)))
                row[f"{metric}_ci95"]=np.quantile(v[samples].mean(1),[.025,.975]).tolist()
        output.append(row)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("New aggregate output root required")
    root.mkdir(parents=True,exist_ok=True)
    with (root/"per_object_distance_crop.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    mf=dict(schema=AGG_SCHEMA,objects=len(seen),
            physical_distance_m=list(expected["distances"]),
            base_checkpoint_sha256=expected["checkpoint"],
            source_index_sha256=sha256(a.analyses_index),
            bootstrap_unit="independent object geometry, NOT view/point",
            per_cell=output,
            limitations=[
                "Distinct CAD file hashes need not imply different semantic objects",
                "Repeated objects, CAD variants and material/style domains require manual audit",
                "CAD render appearance differs from real GC6D; no real-world improvement",
                "No scale fitting, no finetuning, no grasp AP results"
            ])
    mf["manifest_sha256"]=digest(mf)
    write_json(root/"summary.json",mf)
    with (root/"REPORT_CN.md").open("w",encoding="utf-8") as f:
        f.write("# B：跨独立CAD物体的物距×crop作用汇总\n\n")
        f.write(f"物体数：{len(seen)}；场景／CAD几何是置信区间重采样单位。\n\n")
        f.write("| 距离(m) | crop | 同点斜率β | Δz MAE(m) | distance×crop DiD(m) |\n|---|---|---:|---:|---:|\n")
        for r in output:
            f.write(f"| {r['distance_m']:.3f} | {r['crop_policy']} | "
                    f"{r['slope']:+.4f} | {r['mean_delta_abs_error']:.4f} | "
                    f"{r['crop_distance_interaction_did']:+.4f} |\n")
        f.write("\n合成实验不代表真实相机/真实物体域。未训练或拟合任何校准参数。\n")
    return mf


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analyses-index",required=True,
                   help="JSON list of verified per-object B analysis output directories")
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=4000)
    p.add_argument("--seed",type=int,default=20261117)
    a=p.parse_args()
    if a.bootstrap<1:p.error("bootstrap must be positive")
    run(a)


if __name__=="__main__":
    main()
