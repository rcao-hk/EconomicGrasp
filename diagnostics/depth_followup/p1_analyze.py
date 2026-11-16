"""P1 scene-clustered same-3D-point metric distance response.

Crucially, correspondence gates use measured sensor depth/seg and verified
C2W, not predicted depth. Each anchor has its own baseline depth prediction;
the difference in target/reference depth is measured for that SAME surface.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics.frozen_depth_response.analyze import spearman
from .common import digest,sha256,write_json

SCHEMA="gc6d-same-surface-track-v1"
FIELDS=("scene","src_img_id","dst_img_id","src_u","src_v","seg_label",
        "true_depth_src_m","true_depth_dst_m","pred_depth_src_m",
        "pred_depth_dst_m","delta_reference_m","delta_predicted_m",
        "source_error_m","target_error_m",
        "abs_target_geometry_reprojection_m","ref_pose_distance_m",
        "crop_fx_src","crop_fx_dst")


def _matches(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if set(FIELDS)-set(reader.fieldnames or []):
            raise ValueError(f"Missing P1 measured correspondence fields: {path}")
        out=[]
        for r in reader:
            row={}
            for k in FIELDS:
                if k in ("scene","src_img_id","dst_img_id","src_u",
                         "src_v","seg_label"):
                    row[k]=int(r[k])
                else:
                    row[k]=float(r[k])
                if k not in ("scene","src_img_id","dst_img_id","src_u",
                             "src_v","seg_label") and not np.isfinite(row[k]):
                    raise ValueError(f"Nonfinite tracked surface depth {path}")
            out.append(row)
    return out


def slope_origin(x,y, min_samples=10, min_span=.03):
    """Anchored depth-change slope, no global depth scale fitted."""
    xx=np.asarray(x,dtype=np.float64)
    yy=np.asarray(y,dtype=np.float64)
    valid=np.isfinite(xx)&np.isfinite(yy)&(np.abs(xx)>=min_span)
    if valid.sum()<min_samples:return None
    x=xx[valid];y=yy[valid]
    denom=float(np.dot(x,x))
    return float(np.dot(x,y)/denom) if denom>1e-10 else None


def _summarize_scene(rows,min_motion=.03,min_scene_pairs=40,min_track_views=3):
    if not rows:
        return dict(usable=False,pairs=0,slope=None,matched_tracks=0,
                    multi_view_tracks=0,mean_true_abs_shift=None,
                    mean_pred_abs_shift=None,mean_delta_error=None,
                    sign_agreement=None,rank_corr=None)
    x=np.asarray([r["delta_reference_m"] for r in rows],dtype=float)
    y=np.asarray([r["delta_predicted_m"] for r in rows],dtype=float)
    groups=defaultdict(list)
    for r in rows:
        key=(r["src_img_id"],r["src_u"],r["src_v"])
        groups[key].append(r)
    multi=0;track_slopes=[]
    for k,g in groups.items():
        if len({r["dst_img_id"] for r in g})>=min_track_views:
            multi+=1
            delta=np.asarray([r["delta_reference_m"] for r in g])
            pred=np.asarray([r["delta_predicted_m"] for r in g])
            s=slope_origin(delta,pred,min_samples=2,min_span=min_motion)
            if s is not None:
                track_slopes.append(s)
    usable=len(rows)>=min_scene_pairs and np.sum(np.abs(x)>=min_motion)>=10
    slope=(slope_origin(x,y,min_samples=10,min_span=min_motion)
           if usable else None)
    motion=np.abs(x)>=min_motion
    return dict(
        usable=bool(usable and slope is not None),
        pairs=len(rows),matched_tracks=len(groups),multi_view_tracks=multi,
        tracks_with_slope=len(track_slopes),
        mean_track_slope=float(np.mean(track_slopes)) if track_slopes else None,
        median_track_slope=float(np.median(track_slopes)) if track_slopes else None,
        slope=slope,
        rank_corr=spearman(x[motion],y[motion]) if motion.sum()>=4 else None,
        mean_true_abs_shift=float(np.mean(np.abs(x))),
        mean_pred_abs_shift=float(np.mean(np.abs(y))),
        median_abs_true_shift=float(np.median(np.abs(x))),
        median_abs_pred_shift=float(np.median(np.abs(y))),
        mean_delta_error=float(np.mean(np.abs(x-y))),
        mean_bias_change=float(np.mean(y-x)),
        sign_agreement=float(np.mean(np.sign(x[motion])==np.sign(y[motion])))
                       if motion.any() else None,
        sufficient_motion_pairs=int(motion.sum()),
        ref_camera_translation_mean=float(np.mean([r["ref_pose_distance_m"] for r in rows])),
        mean_reference_depth=float(np.mean([r["true_depth_dst_m"] for r in rows])),
        mean_predicted_depth=float(np.mean([r["pred_depth_dst_m"] for r in rows])),
        source_bias=float(np.mean([r["source_error_m"] for r in rows])),
        target_bias=float(np.mean([r["target_error_m"] for r in rows])),
    )


def _bootstrap_scenes(data,key,draws,seed):
    vals=np.asarray([r[key] for r in data if r.get("usable") and
                     r.get(key) is not None and np.isfinite(r[key])],
                    dtype=np.float64)
    if len(vals)==0:
        return dict(mean=None,ci95=None,scenes=0)
    if len(vals)==1:
        return dict(mean=float(vals[0]),ci95=None,scenes=1)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(vals),size=(draws,len(vals)))
    boot=vals[idx].mean(axis=1)
    return dict(mean=float(vals.mean()),median=float(np.median(vals)),
                ci95=np.quantile(boot,[.025,.975]).tolist(),scenes=len(vals))


def _csv(path,rows):
    if not rows:return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(rows)


def run(a):
    if a.bootstrap<1 or a.min_motion<=0 or a.min_scene_pairs<1:
        raise ValueError("Invalid P1 analysis thresholds")
    manifest_file=Path(a.tracks)/"manifest.json"
    m=json.loads(manifest_file.read_text(encoding="utf-8"))
    signature=m.pop("manifest_sha256",None)
    if (signature!=digest(m) or m.get("schema")!=SCHEMA or
        not m.get("complete") or m.get("matched_scenes")!=78):
        raise ValueError("P1 requires accepted full78 scene matching manifest")
    m["manifest_sha256"]=signature
    records=[];audits=[]
    seen=set()
    for arow in m["audits"]:
        scene=int(arow["scene"])
        if scene in seen:raise RuntimeError("Duplicate GC6D scene")
        seen.add(scene)
        path=Path(arow["matching_file"])
        if not path.is_file() or sha256(path)!=arow["matching_sha256"]:
            raise RuntimeError(f"Changed tracked same-3D surface file: {path}")
        match_rows=_matches(path) if arow["has_matches"] else []
        if len(match_rows)!=int(arow["matched_pairs"]):
            raise ValueError("Saved match count and CSV disagree")
        for r in match_rows:
            if r["scene"]!=scene:raise ValueError("Cross-scene surface match contamination")
        summary=_summarize_scene(match_rows,a.min_motion,a.min_scene_pairs,a.min_track_views)
        summary.update(scene=scene,c2w_translation_m=
                       arow["c2w_scene_gt_parity"]["median_translation_m"],
                       c2w_rotation_deg=arow["c2w_scene_gt_parity"]["median_rotation_deg"])
        records.append(summary)
        audits.append(dict(scene=scene,
             visibility=arow["visibility_counts"],pairs=arow["matched_pairs"]))
    if len(records)!=78:
        raise ValueError("Full78 track results not present")
    accepted=[r for r in records if r["usable"]]
    metrics=("slope","mean_track_slope","mean_delta_error","mean_bias_change",
             "mean_true_abs_shift","mean_pred_abs_shift","sign_agreement",
             "rank_corr","source_bias","target_bias","median_abs_true_shift",
             "median_abs_pred_shift")
    means={key:_bootstrap_scenes(records,key,a.bootstrap,a.seed+i)
           for i,key in enumerate(metrics)}
    counters=defaultdict(int)
    for audit in audits:
        for k,v in audit["visibility"].items():counters[k]+=int(v)
    result=dict(schema="gc6d-p1-same-surface-report-v1",
        source_track_manifest_sha256=sha256(manifest_file),
        source_cache_identity_sha256=m["cache_identity_sha256"],
        all_scenes=len(records),scenes_with_matches=sum(r["pairs"]>0 for r in records),
        scenes_with_valid_slope=len(accepted),
        n_matched_pairs=sum(r["pairs"] for r in records),
        total_matched_tracks=sum(r.get("matched_tracks",0) for r in records),
        visibility_gate_totals=dict(counters),
        min_motion_m=a.min_motion,min_scene_pairs=a.min_scene_pairs,
        min_track_views=a.min_track_views,
        scene_clustered=means,per_scene=records,
        conclusions_not_allowed=[
            "Cannot extrapolate quality/AP from depth-only same-surface tracking",
            "Accepted points are measured-depth and visibility-selected, not unbiased all surface pixels",
            "GC6D 78 scenes were previously inspected, not fresh independent validation",
            "Raw equal segmentation codes are a conservative gate, not confirmed unique CAD object identity",
            "No CAD/rendered reference was used; no global scale fitted",
            "Matched correspondence across viewpoints does not isolate camera focal length from appearance and crop",
        ])
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"summary.json",result)
    _csv(out/"per_scene.csv",records)
    _csv(out/"visibility_counts.csv",[
        dict(scene=audit["scene"],pairs=audit["pairs"],**audit["visibility"])
        for audit in audits])
    lines=["# P1 GC6D同一物理表面跨视角深度响应","",
       f"已验收78场景，具有可解释距离斜率的场景：{len(accepted)}/78。",
       f"成功物理表面对应数：{result['n_matched_pairs']}。",
       "",
       "| 指标 | 场景均值 | 95%场景bootstrap区间 |",
       "|---|---:|---|"]
    for key,score in means.items():
        ci=score["ci95"];formatted=("NA" if ci is None else
                         f"[{ci[0]:+.5f}, {ci[1]:+.5f}]")
        lines.append(f"| {key} | {score['mean']} | {formatted} |")
    lines.extend(["","可见性由目标传感器深度、原分割标签一致性和3D回投误差共同确认。",
                  "这是在同一源3D表面点上的距离变化，不是前景区域中位数在不同视角间的比较。",
                  "曲线对场景bootstrap而不是对百万级像素bootstrap；未达运动跨度阈值的对应不进入斜率。",
                  "场景不足或可见重叠过少必须作为数据覆盖限制汇报，而不是把缺失场景斜率设置为零。",
                  "未训练、未拟合深度校正倍率、未评估新grasp AP。"])
    (out/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=4000)
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--min-motion",type=float,default=.03,
                   help="Require absolute same-point reference z shift >=3cm")
    p.add_argument("--min-scene-pairs",type=int,default=40)
    p.add_argument("--min-track-views",type=int,default=3)
    run(p.parse_args())


if __name__=="__main__":
    main()
