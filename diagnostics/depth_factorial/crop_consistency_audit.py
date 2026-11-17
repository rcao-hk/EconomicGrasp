"""P0: same-physical-ray cross-crop failure audit on signed P1b CAD eval dumps.

Runs with existing native + fixed-ROI near/wide eval, and optionally both new
multi3 arms. Every crop reads the *same* original render object pixels. The
data are test/analysis only: no training, no correction, no new 6D grasp AP.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnostics.depth_followup.multiview import sample_anchors
from .factorial_contract import (
    CROP_POLICIES,INFER_SCHEMA,sha256,digest,source_image,verify_fixed_physics,
    write_json,
)
from .distance_split_p1 import read_registered
from .crop_consistency_core import sample_crop_predictions,evaluate_shared_points

SCHEMA="physical-ray-crosscrop-error-audit-v1"
REQUIRED=("native","roi_near","roi_wide")
OPTIONAL=("multi_near","multi_wide")


def load_eval_run(path,reg,expected_arm):
    root=Path(path)
    report_file=root/"summary.json"
    report=json.loads(report_file.read_text(encoding="utf-8"))
    sig=report.pop("manifest_sha256",None)
    if sig!=digest(report) or report.get("schema")!="physical-distance-head-eval-v1" or (
        report.get("status")!="complete"):
        raise ValueError(f"Invalid signed P1b evaluation {root}")
    report["manifest_sha256"]=sig
    if (report["registration_identity_sha256"]!=reg["registration_sha256"] or
        report["base_checkpoint_sha256"]!=reg["frozen_base_checkpoint_sha256"]):
        raise ValueError("Test CAD registration or frozen base mismatch")
    if report["arm"]!=expected_arm:
        raise ValueError(f"Wrong head arm from {root}: {report['arm']}")
    if sha256(root/"per_condition.csv")!=report["per_condition_sha256"]:
        raise ValueError("Changed per-condition exact signed eval CSV")
    reg_eval={r["object_sha"]:r for r in reg["eval"]}
    reports={}
    for item in report["object_analysis"]:
        object_sha=item["object_sha"]
        if object_sha not in reg_eval or object_sha in reports:
            raise ValueError("Missing or duplicated heldout CAD object")
        frame_root=root/f"cad_{object_sha[:16]}"
        file=frame_root/"manifest.json"
        if sha256(file)!=item["prediction_manifest_sha256"]:
            raise ValueError("Per-object frozen prediction manifest changed")
        mf=json.loads(file.read_text(encoding="utf-8"))
        signed=mf.pop("manifest_sha256",None)
        if signed!=digest(mf) or mf.get("schema")!=INFER_SCHEMA or not mf["complete"]:
            raise ValueError("Invalid per-object frozen source prediction schema/hash")
        mf["manifest_sha256"]=signed
        srcfile=Path(reg_eval[object_sha]["manifest"])
        if (mf["source_manifest_sha256"]!=sha256(srcfile)
              or len(mf["records"])!=15):
            raise ValueError("Per-object signed render source or 5x3 crop cells changed")
        source=json.loads(srcfile.read_text(encoding="utf-8"))
        verify_fixed_physics(source,strict_files=True)
        bycell={}
        for cell in mf["records"]:
            key=(cell["distance_id"],cell["crop_policy"])
            if key in bycell or key[0] not in range(5) or key[1] not in CROP_POLICIES:
                raise ValueError("Repeated/invalid source crop depth file")
            p=Path(cell["prediction_file"])
            if not p.is_file() or sha256(p)!=cell["prediction_sha256"]:
                raise ValueError("Changed exact raw 448 RGB-predicted depth")
            if cell["render_source_sha256"]!=source["views"][key[0]]["files"]:
                raise ValueError("Wrong physical-render image/depth/seg labels")
            pred=np.asarray(np.load(p,allow_pickle=False),dtype=np.float32)
            if pred.shape!=(448,448) or not np.isfinite(pred).all() or (pred<=0).any():
                raise ValueError("Invalid 448 predicted depth")
            bycell[key]=dict(pred=pred,crop_box=cell["crop_box"])
        if len(bycell)!=15:raise ValueError("Missing physical 5×3 data cell")
        reports[object_sha]=dict(source=source,cells=bycell)
    if set(reports)!=set(reg_eval):
        raise ValueError("Evaluation report excluded a preregistered CAD object")
    return report,reports


def _csv(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def run(a):
    paths=json.loads(Path(a.eval_index).read_text(encoding="utf-8"))
    if not isinstance(paths,dict) or not set(REQUIRED).issubset(paths) or (
        set(paths)-set(REQUIRED+OPTIONAL)):
        raise ValueError("Need native, roi_near, roi_wide and optionally both multi arms")
    if ("multi_near" in paths)!=("multi_wide" in paths):
        raise ValueError("Mixed-crop comparison requires both near and wide arms")
    reg=read_registered(a.registration)
    expected={"native":"native","roi_near":"near_only",
              "roi_wide":"near_mid_far","multi_near":"near_only",
              "multi_wide":"near_mid_far"}
    evals={key:load_eval_run(path,reg,expected[key])
           for key,path in paths.items()}
    results=[];rows=[];pixel_counts=[]
    for obj_ix,item in enumerate(reg["eval"]):
        obj=item["object_sha"]
        baseline=evals["native"][1][obj]
        for di,view in enumerate(baseline["source"]["views"]):
            _,ref,mask=source_image(view)
            u,v,labels=sample_anchors(ref,np.asarray(mask,dtype=np.int32),
                    scene=obj_ix,img_id=di,max_points=a.points_per_frame,
                    points_per_object=a.points_per_frame)
            raw_ref=ref[v,u]
            matched={}
            for arm,(_,groups) in evals.items():
                frame=groups[obj]
                if frame["source"]["plan_sha256"]!=baseline["source"]["plan_sha256"]:
                    raise ValueError("Different CAD physics for two evaluated model arms")
                data={crop:frame["cells"][di,crop] for crop in CROP_POLICIES}
                matched[arm]=sample_crop_predictions(u,v,raw_ref,data,
                                          min_points=a.min_points)
            selected=matched["native"]["accepted_raw_index"]
            for arm,data in matched.items():
                if not np.array_equal(data["accepted_raw_index"],selected):
                    raise RuntimeError("Different physical object rays across model arms")
                for row in evaluate_shared_points(data):
                    rows.append(dict(object_sha=obj,arm=arm,
                                     distance_id=di,
                                     distance_m=float(view["distance_m"]),**row))
            pixel_counts.append(dict(object_sha=obj,distance_id=di,
                                     raw_samples=len(u),shared_pixels=len(selected)))
        print(f"[P0 crosscrop] object {obj_ix+1}/{len(reg['eval'])} {obj[:12]}",flush=True)
    # First average by CAD geometry, then by object for all comparisons.
    stats=[]
    for arm in paths:
        for crop in CROP_POLICIES:
            for di in range(5):
                subset=[r for r in rows if r["arm"]==arm and
                        r["crop_policy"]==crop and r["distance_id"]==di]
                if len(subset)!=len(reg["eval"]):
                    raise RuntimeError("Incomplete common physical point results")
                stats.append(dict(arm=arm,crop_policy=crop,distance_id=di,
                    distance_m=float(subset[0]["distance_m"]),
                    objects=len(subset),
                    mean_abs_crosscrop_m=float(np.mean([
                        r["shared_three_crop_range_m"] for r in subset])),
                    mean_mae_m=float(np.mean([r["mae"] for r in subset])),
                    mean_bias_m=float(np.mean([r["mean_bias"] for r in subset])),
                    mean_pred_m=float(np.mean([r["median_prediction"] for r in subset]))))
    if a.baseline_dir:
        # Audit that adding multi3 models did not alter the underlying
        # physical pixel set, crop boxes, original native or ROI predictions.
        past=Path(a.baseline_dir)
        old=json.loads((past/"summary.json").read_text(encoding="utf-8"))
        signature=old.pop("manifest_sha256",None)
        if (signature!=digest(old) or old.get("schema")!=SCHEMA):
            raise ValueError("P0 prior audit provenance invalid")
        if old["registered_cad_source_sha256"]!=sha256(a.registration):
            raise ValueError("Different P0 physical CAD registration")
        for arm in REQUIRED:
            if old["eval_sources"].get(arm)!=sha256(Path(paths[arm])/"summary.json"):
                raise ValueError(f"P0 previously audited {arm} model source changed")
        with (past/"per_object_distance_crop_arm.csv").open(
                newline="",encoding="utf-8") as f:
            baseline=list(csv.DictReader(f))
        key=lambda r:(r["object_sha"],r["arm"],str(r["distance_id"]),r["crop_policy"])
        current={key(r):r for r in rows if r["arm"] in REQUIRED}
        if len(current)!=len(baseline):
            raise ValueError("Different baseline CAD object×crop×distance coverage")
        for r in baseline:
            new=current.get(key(r))
            if new is None:
                raise ValueError("Previously evaluated CAD physical point set missing")
            for field in ("points","mae","mean_bias","median_prediction",
                          "median_reference","shared_three_crop_std_m",
                          "shared_three_crop_range_m"):
                if not np.isclose(float(r[field]),float(new[field]),
                                  rtol=0,atol=1e-10):
                    raise RuntimeError(f"Original P0 baseline altered after multi3: {key(r)}/{field}")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    _csv(root/"per_object_distance_crop_arm.csv",rows)
    _csv(root/"same_raw_pixel_counts.csv",pixel_counts)
    _csv(root/"per_distance_crop_arm.csv",stats)
    manifest=dict(schema=SCHEMA,status="complete",
        source_eval_index_sha256=sha256(a.eval_index),
        registered_cad_source_sha256=sha256(a.registration),
        eval_sources={key:sha256(Path(path)/"summary.json")
                      for key,path in paths.items()},
        objects=len(reg["eval"]),distances=5,crops=list(CROP_POLICIES),
        arms=list(paths),per_cell=stats,
        same_world_surface_definition=(
            "identical raw CAD render camera pixels across all crops and arms"),
        training_updates=0,
        limitations=[
            "CAD and evaluated 2/5 train,8/9 eval objects have been previously inspected",
            "Pixel/point matches within each CAD are correlated, not IID observations",
            "Crop may alter visual context/resolution, no unique cause identified",
            "Only synthetic CAD with privileged render masks, not real RGB-only deployment",
            "No adjusted RGB metric depth or grasp AP generated"])
    manifest["manifest_sha256"]=digest(manifest)
    write_json(root/"summary.json",manifest)
    lines=["# P0 同一物理表面点的跨Crop预测一致性","",
           "原RGB/世界表面点相同，只改变crop；先在每对象每距离统计，随后等权平均CAD。",
           "| Arm | Crop | 物距(m) | 平均绝对深度误差(m) | 同点三crop间深度range(m) |",
           "|---|---|---:|---:|---:|"]
    for r in stats:
        if r["distance_id"] not in (0,4):continue
        lines.append(f"| {r['arm']} | {r['crop_policy']} | {r['distance_m']:.2f} | "
                     f"{r['mean_mae_m']:.4f} | {r['mean_abs_crosscrop_m']:.4f} |")
    lines.extend(["","此处的跨crop差是相同相机、相同物理点上的预测差，非相机运动真实距离变化。",
                  "原CAD已被查看；仅是机制审计，不能证明系统跨未见物体泛化。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--eval-index",required=True,
                   help='JSON {"native":"/abs","roi_near":"/abs","roi_wide":"/abs", optional multi_near/multi_wide}')
    p.add_argument("--baseline-dir",default=None,
                   help="Prior 3-arm same-point audit must remain byte-numerically consistent after new training")
    p.add_argument("--points-per-frame",type=int,default=256)
    p.add_argument("--min-points",type=int,default=32)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.points_per_frame<32 or a.min_points<1:
        p.error("Invalid source physical pixel budget")
    run(a)


if __name__=="__main__":
    main()
