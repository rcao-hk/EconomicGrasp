"""B: same-world-surface paired distance × crop response and interaction.

Frozen RGB-depth predictions are compared across a fixed 3D scene and real
perspective camera translations. Correspondences derive only from rendered
camera-z depth/instance labels and verified camera poses, NEVER predicted z.
Three crop policies evaluate exactly the same retained world surface points.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics.depth_followup.multiview import match_pair,sample_anchors
from .factorial_contract import (CROP_POLICIES,SOURCE_SCHEMA,
                                 sha256,digest,write_json,
                                 source_image,verify_fixed_physics)
from .factorial_run import SCHEMA as INFER_SCHEMA

SCHEMA="physical-distance-crop-factorial-analysis-v1"


def _load(source_file,pred_file):
    src=json.loads(Path(source_file).read_text(encoding="utf-8"))
    contract=verify_fixed_physics(src,strict_files=True)
    source_sha=sha256(source_file)
    result=json.loads(Path(pred_file).read_text(encoding="utf-8"))
    sig=result.pop("manifest_sha256",None)
    if sig!=digest(result) or result.get("schema")!=INFER_SCHEMA or not result.get("complete"):
        raise ValueError("Incomplete/falsified frozen cross-crop predictions")
    result["manifest_sha256"]=sig
    if result.get("source_manifest_sha256")!=source_sha:
        raise ValueError("Render source and frozen depth inference not the same")
    if set(result.get("crop_policies",[]))!=set(CROP_POLICIES):
        raise ValueError("Missing crossed crop policies")
    if len(result["records"])!=len(src["views"])*len(CROP_POLICIES):
        raise ValueError("Missing physical distance x crop cell")
    by={}
    for record in result["records"]:
        key=(record["distance_id"],record["crop_policy"])
        if key in by:raise ValueError("Duplicate factorial cell")
        by[key]=record
        file=Path(record["prediction_file"])
        if not file.is_file() or sha256(file)!=record["prediction_sha256"]:
            raise ValueError(f"Changed frozen depth prediction: {file}")
        if record["distance_id"] not in range(len(src["views"])):
            raise ValueError("Unexpected camera position index")
        if not np.isclose(record["distance_m"],src["views"][record["distance_id"]]["distance_m"],atol=1e-9):
            raise ValueError("Camera distance not bound to frozen prediction")
        if record["render_source_sha256"]!=src["views"][record["distance_id"]]["files"]:
            raise ValueError("Rendered source RGB/depth/seg changed")
    return src,result,by


def _read_view(src,index,policy,record):
    image,ref,mask=source_image(src["views"][index])
    with np.load(record["prediction_file"],allow_pickle=False) as error:
        pred=error["pred_depth"]
    if pred.shape!=(448,448) or not np.isfinite(pred).all() or (pred<=0).any():
        raise ValueError("Invalid frozen 448 predicted depth")
    K=np.asarray(src["views"][index]["K"],dtype=np.float32)
    T=np.asarray(src["views"][index]["T_c2w"],dtype=np.float32)
    return dict(scene=0,img_id=index,anno=index,
        depth_raw_m=ref,seg_raw=np.asarray(mask,dtype=np.int32),
        pred_depth=np.asarray(pred,dtype=np.float32),
        raw_K=K,K=np.asarray(record["K"],dtype=np.float32),
        crop_box=np.asarray(record["crop_box"],dtype=np.int64),T_c2w=T)


def _slope(pairs,min_motion):
    if not pairs:return None
    x=np.asarray([p["delta_reference_m"] for p in pairs])
    y=np.asarray([p["delta_predicted_m"] for p in pairs])
    mask=(np.abs(x)>=min_motion)
    if int(mask.sum())<8:return None
    d=float(np.dot(x[mask],x[mask]))
    return float(np.dot(x[mask],y[mask])/d) if d>1e-9 else None


def _write_csv(path,rows):
    if not rows:return
    fields=sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=fields)
        wr.writeheader();wr.writerows(rows)


def run(a):
    if a.min_motion<=0 or a.max_points<16 or a.abs_tol<=0 or a.xyz_tol<=0:
        raise ValueError("Invalid physical surface verification thresholds")
    src,inferred,index=_load(a.source_manifest,a.frozen_manifest)
    n=len(src["views"])
    source_by_policy={}
    for policy in CROP_POLICIES:
        source_by_policy[policy]=_read_view(src,0,policy,index[0,policy])
    # Same world-surface anchor pixels across ALL crop policies.
    base=source_by_policy["fixed_full"]
    u,v,label=sample_anchors(base["depth_raw_m"],base["seg_raw"],
        scene=0,img_id=0,points_per_object=a.max_points,max_points=a.max_points)
    rows=[];per_pair=[]
    for index_distance in range(1,n):
        label_maps={}
        controls={}
        for policy in CROP_POLICIES:
            target=_read_view(src,index_distance,policy,index[index_distance,policy])
            matches,counters=match_pair(source_by_policy[policy],target,u,v,label,
                abs_tol=a.abs_tol,rel_tol=a.rel_tol,xyz_tol=a.xyz_tol)
            index_by_id={(int(r["src_u"]),int(r["src_v"]),int(r["dst_img_id"])):r
                         for r in matches}
            if len(index_by_id)!=len(matches):
                raise ValueError("Duplicate physical world point match")
            label_maps[policy]=index_by_id
            controls[policy]=counters
        shared=set.intersection(*(set(label_maps[p]) for p in CROP_POLICIES))
        if len(shared)<a.min_shared:
            raise RuntimeError(f"Too few mutually visible physical points at distance {index_distance}: {len(shared)}")
        for key in sorted(shared):
            condition={p:label_maps[p][key] for p in CROP_POLICIES}
            reference=condition["fixed_full"]["delta_reference_m"]
            for mode in CROP_POLICIES:
                if abs(condition[mode]["delta_reference_m"]-reference)>1e-6:
                    raise RuntimeError("Reference physical change differs across crop policies")
            for policy in CROP_POLICIES:
                row=condition[policy]
                full=condition["fixed_full"]
                target_error=abs(row["target_error_m"])
                source_error=abs(row["source_error_m"])
                per_pair.append(dict(distance_id=index_distance,
                    distance_m=float(src["views"][index_distance]["distance_m"]),
                    distance_delta_m=float(src["views"][index_distance]["distance_m"]-
                                           src["views"][0]["distance_m"]),
                    crop_policy=policy,src_u=key[0],src_v=key[1],
                    ref_delta=reference,pred_delta=row["delta_predicted_m"],
                    delta_abs_error=abs(row["delta_predicted_m"]-reference),
                    target_abs_error=target_error,source_abs_error=source_error,
                    # Within same surface point, compare adaptive vs full
                    # crop response after subtracting the baseline crop gap.
                    crop_gain_target_vs_full=target_error-abs(full["target_error_m"]),
                    crop_gain_source_vs_full=source_error-abs(full["source_error_m"]),
                    crop_interaction_did_vs_full=(
                        target_error-abs(full["target_error_m"]))-(
                        source_error-abs(full["source_error_m"])),
                    source_prediction=row["pred_depth_src_m"],
                    target_prediction=row["pred_depth_dst_m"],
                    true_target=row["true_depth_dst_m"]))
        for policy in CROP_POLICIES:
            sample=[r for r in per_pair if r["distance_id"]==index_distance
                    and r["crop_policy"]==policy]
            if len(sample)!=len(shared):
                raise RuntimeError("Factorial cropped matching not complete")
            sx=np.array([r["ref_delta"] for r in sample])
            sy=np.array([r["pred_delta"] for r in sample])
            selected=np.abs(sx)>=a.min_motion
            if selected.sum()<8:
                raise RuntimeError("No real metric camera translation in matched response")
            slope=float(np.dot(sx[selected],sy[selected])/np.dot(sx[selected],sx[selected]))
            rows.append(dict(
                distance_id=index_distance,
                distance_m=src["views"][index_distance]["distance_m"],
                distance_delta_m=src["views"][index_distance]["distance_m"]-
                                 src["views"][0]["distance_m"],
                crop_policy=policy,n_common_surface_points=len(shared),
                n_with_geometric_motion=int(selected.sum()),
                slope=slope,
                mean_true_abs_motion=float(np.mean(np.abs(sx[selected]))),
                mean_pred_abs_motion=float(np.mean(np.abs(sy[selected]))),
                mean_delta_abs_error=float(np.mean(
                    np.abs(sy[selected]-sx[selected]))),
                mean_target_abs_error=float(np.mean([r["target_abs_error"] for r in sample])),
                mean_source_abs_error=float(np.mean([r["source_abs_error"] for r in sample])),
                crop_gap_vs_full_target=float(np.mean([r["crop_gain_target_vs_full"] for r in sample])),
                crop_distance_interaction_did=float(np.mean(
                    [r["crop_interaction_did_vs_full"] for r in sample])),
                fov_counts=controls[policy],
            ))
        print(f"[B same-world surface] physical distance {index_distance+1}/{n} "
              f"common={len(shared)}",flush=True)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    _write_csv(root/"per_distance_crop.csv",[
        {k:v for k,v in row.items() if k!="fov_counts"} for row in rows])
    _write_csv(root/"per_common_surface.csv",per_pair)
    result=dict(schema=SCHEMA,complete=True,
        source_render_manifest_sha256=sha256(a.source_manifest),
        frozen_inference_manifest_sha256=sha256(a.frozen_manifest),
        base_checkpoint_sha256=inferred["base_checkpoint_sha256"],
        fixed_physical_source=src["plan"],
        main_factor="physical translation of the camera only; fixed object and K",
        secondary_factor="crop policy on the same rendered source RGB",
        source_distance_m=float(src["views"][0]["distance_m"]),
        rows=rows,n_shared_pairs=len(per_pair)//len(CROP_POLICIES),
        common_surface_selection="intersection across all crop conditions",
        calibration_oracle_used=False,reference_used_for_visibility_and_workspace_crop=True,
        limitations=[
            "All depth predictions frozen. No updates or AP training/selection.",
            "Camera/object/material/light held fixed except physical camera translation",
            "Renderer appearance may not represent real GC6D photometry",
            "Rendered object depth/seg are privileged workspace crop and ground-truth diagnostics",
            "Single CAD or procedural mesh yields per-object results, not independent object-level CIs",
            "The interaction is descriptive on same-world-surface points, not proof of a universal crop cause",
        ])
    result["manifest_sha256"]=digest(result)
    write_json(root/"summary.json",result)
    lines=["# B：真实相机平移距离 × crop 交互（冻结单目深度）","",
       "视角通过3D CAD渲染生成；相机K、朝向、材质、光照、物体位姿不变，仅相机沿光轴平移。",
       "同一个源表面点经过真实深度/实例mask重投影后，三个crop策略取交集，避免不同像素集合的比较。",
       "",
       "| 远距(m) | Δ物距(m) | Crop | 同点β | Δz MAE(m) | target MAE(m) | distance×crop DiD(m) |",
       "|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['distance_m']:.3f} | {r['distance_delta_m']:.3f} | "
                     f"{r['crop_policy']} | {r['slope']:+.4f} | "
                     f"{r['mean_delta_abs_error']:.4f} | "
                     f"{r['mean_target_abs_error']:.4f} | "
                     f"{r['crop_distance_interaction_did']:+.4f} |")
    lines.extend(["","DiD是同一可见CAD表面点的目标crop误差相对于固定全图的差距，减去源距离时同点差距。",
                  "没有配备真实独立物体多样性时，单个renderer对象结果不能推广到全GC6D。",
                  "没有做米制scale校准、没训练模型、没评估新的grasp AP。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest",required=True)
    p.add_argument("--frozen-manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--min-motion",type=float,default=.03)
    p.add_argument("--max-points",type=int,default=256)
    p.add_argument("--min-shared",type=int,default=32)
    p.add_argument("--abs-tol",type=float,default=.006)
    p.add_argument("--rel-tol",type=float,default=.008)
    p.add_argument("--xyz-tol",type=float,default=.012)
    a=p.parse_args()
    run(a)


if __name__=="__main__":
    main()
