"""P1a: audit existing *corrected* physical Distance×Crop render scale cues.

No new training. Input: an explicit, hashed list of accepted corrected B
source/inference/analysis manifests. Rescans RGB saturation and foreground
contrast to reject the superseded near-white render batch. Measures projected
object dimensions, source crop pixel occupancy, effective K', predicted depth
and matched-surface β, without tuning any model weights or distances.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .factorial_contract import (CROP_POLICIES,sha256,digest,write_json,
                                  source_image,verify_fixed_physics)
from .factorial_analyze import _load
from diagnostics.frozen_depth_response.analyze import spearman,slope

SCHEMA="physical-distance-crop-scale-cues-v1"


def rgb_quality(rgb,fg,*,max_white=.96,min_std=5.,min_contrast=5.):
    """Independent image-content gate, not a learned model-scoring criterion."""
    image=np.asarray(rgb)
    mask=np.asarray(fg,dtype=bool)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError("Expected exact 8-bit rendered RGB")
    if mask.shape!=image.shape[:2] or not mask.any() or (~mask).sum()<100:
        raise ValueError("Insufficient object/background for RGB quality")
    white=float(((image>=250).all(axis=2)).mean())
    std=float(np.std(image.astype(np.float64)))
    fg_mean=image[mask].astype(np.float64).mean(0)
    bg_mean=image[~mask].astype(np.float64).mean(0)
    contrast=float(np.linalg.norm(fg_mean-bg_mean)/np.sqrt(3))
    if white>max_white or std<min_std or contrast<min_contrast:
        raise RuntimeError(f"Reject saturated/near-constant render: "
                           f"white={white:.4f},std={std:.3f},contrast={contrast:.3f}")
    return dict(rgb_white_fraction=white,rgb_std=std,rgb_fg_bg_contrast=contrast)


def read_inventory(path, *, require_accepted=True):
    srcfile=Path(path)
    inventory=json.loads(srcfile.read_text(encoding="utf-8"))
    if not isinstance(inventory,list) or not inventory:
        raise ValueError("Supply JSON list of signed corrected CAD B artifacts")
    seen=set()
    accepted=[]
    for item in inventory:
        required=("source_manifest","source_sha256","frozen_manifest",
                  "frozen_sha256","analysis_manifest","analysis_sha256")
        if any(key not in item for key in required):
            raise ValueError(f"B source inventory lacks signed files {required}")
        for prefix,expected in (("source",item["source_sha256"]),
                                ("frozen",item["frozen_sha256"]),
                                ("analysis",item["analysis_sha256"])):
            file=Path(item[f"{prefix}_manifest"])
            if not file.is_file() or sha256(file)!=expected:
                raise ValueError(f"{prefix} manifest missing or modified: {file}")
        if require_accepted:
            f=Path(item.get("acceptance_evidence",""))
            if not f.is_file() or sha256(f)!=item.get("acceptance_sha256"):
                raise ValueError("Corrected B independent acceptance file/hash missing")
        src,freeze,_=_load(item["source_manifest"],item["frozen_manifest"])
        if src["plan"]["object"]["kind"]!="mesh":
            raise ValueError("Formal B scale audit excludes procedural smoke box")
        with Path(item["analysis_manifest"]).open(encoding="utf-8") as f:
            analysis=json.load(f)
        sig=analysis.pop("manifest_sha256",None)
        from .factorial_analyze import SCHEMA as ANALYSIS_SCHEMA
        if sig!=digest(analysis) or analysis.get("schema")!=ANALYSIS_SCHEMA:
            raise ValueError("B same-world-point analysis missing signature")
        analysis["manifest_sha256"]=sig
        if (analysis.get("source_render_manifest_sha256")!=item["source_sha256"] or
            analysis.get("frozen_inference_manifest_sha256")!=item["frozen_sha256"]):
            raise ValueError("B physical source/prediction/analysis chain not paired")
        obj=src["plan"]["object"]["sha256"]
        if obj in seen:raise ValueError("Duplicate CAD asset in inventory")
        seen.add(obj)
        accepted.append(dict(identity=obj,source=src,frozen=freeze,
                             analysis=analysis,item=item))
    return accepted,sha256(srcfile)


def _bbox(mask):
    ys,xs=np.nonzero(np.asarray(mask,dtype=bool))
    if not len(xs):raise ValueError("Visible projected CAD silhouette empty")
    return int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)


def _crop_metrics(mask,box):
    H,W=mask.shape
    x0,y0,x1,y1=map(int,box)
    if not 0<=x0<x1<=W or not 0<=y0<y1<=H:
        raise ValueError("Crop coordinates outside physical render")
    bb=_bbox(mask)
    cl=mask[y0:y1,x0:x1]
    ys,xs=np.nonzero(cl)
    if not len(xs):
        raise ValueError("Crop loses entire CAD silhouette")
    # These are only projected pixel-scale proxies; CAD dimensions are never
    # inferred from semantic category or guessed "ground-truth" object size.
    cw,ch=x1-x0,y1-y0
    full_w,full_h=bb[2]-bb[0],bb[3]-bb[1]
    return dict(projected_bbox_w_px=full_w,projected_bbox_h_px=full_h,
        bbox_width_as_fraction_of_full=float(full_w/W),
        bbox_height_as_fraction_of_full=float(full_h/H),
        crop_w_px=cw,crop_h_px=ch,
        crop_area_fraction=float(cw*ch/(W*H)),
        object_bbox_w_on_448=float((xs.max()-xs.min()+1)*448./cw),
        object_bbox_h_on_448=float((ys.max()-ys.min()+1)*448./ch),
        object_fraction_of_448=float(cl.mean()),
        foreground_visible_fraction=float(cl.sum()/mask.sum()))


def _perframe_predictions(src,frozen,item):
    predictions={}
    for rec in frozen["records"]:
        path=Path(rec["prediction_file"])
        if not path.is_file() or sha256(path)!=rec["prediction_sha256"]:
            raise ValueError("Changed frozen prediction map")
        p=np.asarray(np.load(path,allow_pickle=False),dtype=np.float32)
        if p.shape!=(448,448) or not np.isfinite(p).all() or np.any(p<=0):
            raise ValueError("Invalid frozen RGB predicted depth")
        predictions[(rec["distance_id"],rec["crop_policy"])]=(p,rec)
    rows=[]
    for di,view in enumerate(src["views"]):
        rgb,ref,mask=source_image(view)
        rgbcheck=rgb_quality(rgb,mask)
        W,H=src["plan"]["camera"]["width"],src["plan"]["camera"]["height"]
        if rgb.shape[:2]!=(H,W):
            raise ValueError("Physical render image resolution not fixed")
        for crop in CROP_POLICIES:
            p,rec=predictions[(di,crop)]
            box=tuple(rec["crop_box"])
            cm=_crop_metrics(mask,box)
            x0,y0,x1,y1=box
            foreground=cv2.resize(mask[y0:y1,x0:x1].astype("uint8"),
                                   (448,448),interpolation=cv2.INTER_NEAREST)>0
            reference=cv2.resize(ref[y0:y1,x0:x1],
                                 (448,448),interpolation=cv2.INTER_NEAREST)
            good=foreground&(reference>0)
            if good.sum()<128:
                raise RuntimeError("B scale audit has too few valid silhouette pixels")
            Kn=np.asarray(rec["K"],dtype=float)
            fx=float(view["K"][0][0]);fy=float(view["K"][1][1])
            if abs(Kn[0,0]-fx*448/(x1-x0))>.002 or \
               abs(Kn[1,1]-fy*448/(y1-y0))>.002:
                raise RuntimeError("Crop effective K not from physical K'=A K")
            rows.append(dict(object_sha=item["identity"],distance_id=di,
                distance_m=float(view["distance_m"]),crop_policy=crop,
                fx_crop=float(Kn[0,0]),fy_crop=float(Kn[1,1]),
                pred_median=float(np.median(p[good])),
                ref_median=float(np.median(reference[good])),
                pred_reference_ratio=float(np.median(p[good]/reference[good])),
                mae=float(np.mean(np.abs(p[good]-reference[good]))),
                **rgbcheck,**cm))
    return rows


def _csv(path,rows):
    if not rows:return
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w",newline="",encoding="utf-8") as file:
        w=csv.DictWriter(file,fieldnames=fields);w.writeheader();w.writerows(rows)


def run(a):
    groups,index_hash=read_inventory(a.index,require_accepted=not a.smoke)
    if not a.smoke and len(groups)!=4:
        raise ValueError("Primary P1a must audit exactly four accepted corrected B CAD renders; do not select a favorable subset")
    rows=[];analysis_rows=[]
    for item in groups:
        object_rows=_perframe_predictions(item["source"],item["frozen"],item)
        rows.extend(object_rows)
        for m in item["analysis"]["rows"]:
            analysis_rows.append(dict(object_sha=item["identity"],
                                      distance_id=m["distance_id"],
                                      crop_policy=m["crop_policy"],
                                      matched_surface_beta=m["slope"],
                                      matched_delta_mae=m["mean_delta_abs_error"],
                                      crop_interaction_did=m["crop_distance_interaction_did"]))
    traces={}
    for obj in sorted({r["object_sha"] for r in rows}):
        for crop in CROP_POLICIES:
            sub=sorted([r for r in rows if r["object_sha"]==obj and
                        r["crop_policy"]==crop],key=lambda r:r["distance_id"])
            if len(sub)!=5:
                raise ValueError("Expected same preregistered 5 physical distances and all crops")
            def values(key):return [r[key] for r in sub]
            distances=values("distance_m")
            if distances!=[.35,.45,.55,.70,.85]:
                raise ValueError("Primary B corrected 0.35–0.85m grid must be unchanged")
            traces[f"{obj}/{crop}"]=dict(
                projected_width_vs_distance=spearman(distances,values("projected_bbox_w_px")),
                crop_occupancy_vs_distance=spearman(distances,values("object_fraction_of_448")),
                effective_fx_vs_distance=spearman(distances,values("fx_crop")),
                prediction_vs_distance=spearman(distances,values("pred_median")),
                distance_slope_pred=slope(distances,values("pred_median"),min_span=.1),
                projected_width_ratio_far_near=float(sub[-1]["projected_bbox_w_px"]/
                                                     sub[0]["projected_bbox_w_px"]),
                occupied_fraction_ratio_far_near=float(sub[-1]["object_fraction_of_448"]/
                                                     sub[0]["object_fraction_of_448"]),
                effective_fx_ratio_far_near=float(sub[-1]["fx_crop"]/sub[0]["fx_crop"]),
                predicted_depth_change_far_near=float(sub[-1]["pred_median"]-sub[0]["pred_median"]),
                native_reference_change_far_near=float(sub[-1]["ref_median"]-sub[0]["ref_median"]))
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    _csv(root/"per_object_distance_crop.csv",rows)
    _csv(root/"paired_surface_beta.csv",analysis_rows)
    result=dict(schema=SCHEMA,source_index_sha256=index_hash,
        status="smoke" if a.smoke else "complete",
        cad_objects=len(groups),conditions=len(rows),
        physical_distance_grid_m=[.35,.45,.55,.70,.85],
        by_object_crop=traces,
        object_level_aggregates={
            name:{crop:float(np.mean([x[name] for key,x in traces.items()
                  if key.endswith("/"+crop)]))
                  for crop in CROP_POLICIES}
            for name in ("projected_width_ratio_far_near",
                         "occupied_fraction_ratio_far_near",
                         "effective_fx_ratio_far_near",
                         "predicted_depth_change_far_near")},
        limitations=[
           "Only four already inspected corrected CAD render objects; descriptive mechanism audit",
           "RGB saturation/contrast rechecked; old near-white B renders rejected",
           "Object pixel silhouette is an image-scale proxy, NOT known metric CAD size",
           "Adaptive workspace crop consumes privileged rendered depth and object mask",
           "Camera K and physical object/world pose are fixed across distance",
           "Spearman over only five distances per object is descriptive, not object-independent inference",
           "No depth finetuning, no focal causal model, no new grasp AP"],
        model_updates=0)
    write_json(root/"summary.json",result)
    lines=["# P1a 物理距离×Crop的成像尺度证据审计","",
        f"修正曝光CAD对象：{len(groups)}；每对象5物理距离×3种crop；未训练。",
        "| Crop | 平均远近投影宽比 | 448物体像素占比远近比 | 有效fx远近比 | 预测距离变化(m) |",
        "|---|---:|---:|---:|---:|"]
    for crop in CROP_POLICIES:
        v=result["object_level_aggregates"]
        lines.append(f"| {crop} | {v['projected_width_ratio_far_near'][crop]:.3f} | "
             f"{v['occupied_fraction_ratio_far_near'][crop]:.3f} | "
             f"{v['effective_fx_ratio_far_near'][crop]:.3f} | "
             f"{v['predicted_depth_change_far_near'][crop]:+.4f} |")
    lines.extend(["","这些统计对原4 CAD是探索性，不证明真实GC6D唯一根因。",
                  "同一视角仅改变crop；物距真实变化只能来自经验证的相机平移渲染。",
                  "没有训练/深度倍率选择/新的抓取AP。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",required=True,
       help="JSON signed accepted corrected B source/frozen/analysis manifests (all 4 CAD)")
    p.add_argument("--output",required=True)
    p.add_argument("--smoke",action="store_true",help="Only skips corrected acceptance evidence file")
    run(p.parse_args())


if __name__=="__main__":
    main()
