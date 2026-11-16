"""Scene-clustered distance/pose/crop response analysis.

All model weights frozen; all reference values are OFFLINE diagnostics.
This script does not fit any depth scale/camera calibration and produces
no modified grasp centers or official GraspNet/GC6D grasp AP.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rare_grasp.core import sha256


def _read_rows(path):
    with Path(path).open("r",encoding="utf-8",newline="") as stream:
        raw=list(csv.DictReader(stream))
    if not raw:raise ValueError("No saved depth response rows")
    names=("scene","img_id","anno","sensor_valid_pixels","sensor_median_pred",
           "sensor_median_reference","sensor_median_ratio","sensor_median_error",
           "sensor_mae","sensor_absrel","same_ray_median_delta",
           "same_ray_median_abs_delta","same_ray_median_ratio",
           "common_ray_n","overlap_fraction","pose_delta_deg",
           "K_only_maxabs","K_fx","K_fy","K_fx_modified","K_fy_modified",
           "rendered_median_reference","pose_film_gamma_abs_mean",
           "pose_film_beta_abs_mean","pose_film_gamma_shift",
           "pose_film_beta_shift")
    parsed=[]
    for r in raw:
        x=dict(r)
        for name in names:
            v=x.get(name)
            if v is None or v=="":
                x[name]=None
            else:
                x[name]=(int(v) if name in ("scene","img_id","anno",
                    "sensor_valid_pixels","common_ray_n") else float(v))
        parsed.append(x)
    return parsed


def rank_average(x):
    """Average rank for ties, deterministic and SciPy-free."""
    x=np.asarray(x,dtype=np.float64)
    if len(x)==0:return x
    order=np.argsort(x,kind="mergesort")
    out=np.empty(len(x),dtype=np.float64)
    values=x[order]
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and values[j]==values[i]:j+=1
        out[order[i:j]]=(i+1+j)/2.
        i=j
    return out


def spearman(x,y):
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    if x.shape!=y.shape or len(x)<3:
        return None
    x=rank_average(x);y=rank_average(y)
    x-=x.mean();y-=y.mean()
    denom=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/denom) if denom>0 else None


def slope(x,y,min_span=.03):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.shape!=y.shape or len(x)<3:
        return None
    if float(x.max()-x.min())<min_span:
        return None
    d=x-x.mean()
    denom=float(np.dot(d,d))
    return float(np.dot(d,y-y.mean())/denom) if denom>1e-12 else None


def _bootstrap_scene(values,draws=5000,seed=20261116):
    vals=np.asarray([v for v in values if v is not None and np.isfinite(v)],
                    dtype=np.float64)
    if len(vals)==0:return dict(mean=None,ci95=None,scenes=0)
    if len(vals)==1:return dict(mean=float(vals[0]),ci95=None,scenes=1)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(vals),size=(draws,len(vals)))
    boot=vals[idx].mean(1)
    return dict(mean=float(vals.mean()),
                median=float(np.median(vals)),
                ci95=np.quantile(boot,[.025,.975]).tolist(),
                scenes=len(vals))


def summarize(rows,manifest,*,bootstrap=5000,seed=20261116,min_distance_span=.03):
    if bootstrap<1:raise ValueError("Bootstrap must be positive")
    original=[r for r in rows if r["intervention"]=="identity"]
    if not original:raise ValueError("Missing native baseline")
    keys=[(r["scene"],r["img_id"]) for r in original]
    if len(keys)!=len(set(keys)):
        raise RuntimeError("Duplicate native scene/view rows")
    if manifest["status"]=="complete" and len(original)!=manifest["expected_frames"]:
        raise RuntimeError("Formal depth baseline frame coverage incomplete")
    by_scene=defaultdict(list)
    for r in original:by_scene[r["scene"]].append(r)
    response=[]
    for scene,records in sorted(by_scene.items()):
        valid=[r for r in records if r["sensor_median_pred"] is not None and
               r["sensor_median_reference"] is not None]
        pred=np.array([r["sensor_median_pred"] for r in valid],dtype=float)
        ref=np.array([r["sensor_median_reference"] for r in valid],dtype=float)
        response.append(dict(scene=scene,frames=len(records),valid_frames=len(valid),
            reference_span_m=float(np.ptp(ref)) if len(ref) else None,
            predicted_span_m=float(np.ptp(pred)) if len(pred) else None,
            within_scene_slope=slope(ref,pred,min_distance_span),
            within_scene_spearman=spearman(ref,pred),
            median_ref=float(np.median(ref)) if len(ref) else None,
            median_pred=float(np.median(pred)) if len(pred) else None,
            median_ratio=float(np.median([r["sensor_median_ratio"] for r in valid]))
                  if valid else None,
            median_abs_error=float(np.median([r["sensor_mae"] for r in valid]))
                  if valid else None))
    all_valid=[r for r in original if r["sensor_median_pred"] is not None and
               r["sensor_median_reference"] is not None]
    all_ref=[r["sensor_median_reference"] for r in all_valid]
    all_pred=[r["sensor_median_pred"] for r in all_valid]
    raw_corr=spearman(all_ref,all_pred)
    raw_slope=slope(all_ref,all_pred,min_distance_span)
    xcenter=[];ycenter=[]
    for s,records in by_scene.items():
        valid=[r for r in records if r["sensor_median_pred"] is not None]
        if not valid:continue
        xx=np.array([r["sensor_median_reference"] for r in valid])
        yy=np.array([r["sensor_median_pred"] for r in valid])
        xcenter.extend((xx-xx.mean()).tolist())
        ycenter.extend((yy-yy.mean()).tolist())
    centered_slope=slope(xcenter,ycenter,min_distance_span/2) if len(xcenter)>=3 else None
    matched=defaultdict(list)
    for r in rows:
        if r["intervention"]!="identity":
            matched[r["intervention"]].append(r)
    mode_stats={}
    for mode,recs in sorted(matched.items()):
        if len({(r["scene"],r["img_id"]) for r in recs})!=len(recs):
            raise RuntimeError(f"Intervention repeated within same scene/view: {mode}")
        group=defaultdict(list)
        for r in recs:group[r["scene"]].append(r)
        metrics=("same_ray_median_delta","same_ray_median_abs_delta",
                 "same_ray_median_ratio","sensor_median_error",
                 "overlap_fraction","pose_delta_deg","K_only_maxabs",
                 "pose_film_gamma_shift","pose_film_beta_shift")
        summary={key:_bootstrap_scene(
            [float(np.mean([r[key] for r in items if r[key] is not None]))
             if any(r[key] is not None for r in items) else None
             for items in group.values()],bootstrap,seed+i)
             for i,key in enumerate(metrics)}
        summary["frames"]=len(recs);summary["scenes"]=len(group)
        summary["valid_same_ray_frames"]=sum(r["same_ray_median_abs_delta"] is not None for r in recs)
        mode_stats[mode]=summary
    all_ref_stats=_bootstrap_scene([r["median_ref"] for r in response],bootstrap,seed)
    all_pred_stats=_bootstrap_scene([r["median_pred"] for r in response],bootstrap,seed+1)
    all_ratio_stats=_bootstrap_scene([r["median_ratio"] for r in response],bootstrap,seed+2)
    slope_stats=_bootstrap_scene([r["within_scene_slope"] for r in response],bootstrap,seed+3)
    rank_stats=_bootstrap_scene([r["within_scene_spearman"] for r in response],bootstrap,seed+4)
    return dict(
        schema="frozen-gvar-depth-camera-response-summary-v1",
        source_manifest_sha256=manifest["output_csv_sha256"],
        source_status=manifest["status"],
        scenes=len(response),identity_frames=len(original),
        median_reference_scene_mean=all_ref_stats,
        median_predicted_scene_mean=all_pred_stats,
        median_ratio_scene_mean=all_ratio_stats,
        across_frame_spearman=raw_corr,across_frame_slope=raw_slope,
        scene_centered_slope=centered_slope,
        within_scene_slope=slope_stats,
        within_scene_spearman=rank_stats,
        scenes_with_valid_slope=sum(r["within_scene_slope"] is not None for r in response),
        min_distance_span_m=min_distance_span,
        interventions=mode_stats,
        per_scene=response,
        interpretation_limits=[
            "Observed sensor depth is OFFLINE reference; reference was not an RGB model input",
            "GC6D preprocessing crop/workspace uses observed sensor depth; pipeline is not sensor-depth-free",
            "Digital crop/zoom with K'=AK does not emulate a different optical focal length",
            "K-only parity is a known global_film code-path negative control, not learned camera invariance",
            "Pose-swap changes metadata with fixed image, an OOD stress intervention only",
            "Neither a scale correction nor a new grasp AP was fitted or evaluated",
            "Previously inspected 78 GC6D scenes are exploratory, not a new untouched benchmark",
            "Medians and slopes describe fixed model responses, not causal attribution of failure to K or pose",
        ])


def write_csv(path,rows):
    keys=list(rows[0])
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,help="Directory with per_frame_response.csv and manifest.json")
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261116)
    p.add_argument("--min-distance-span",type=float,default=.03)
    a=p.parse_args()
    root=Path(a.run)
    mf=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    path=root/"per_frame_response.csv"
    if mf.get("output_csv_sha256")!=sha256(path):
        raise RuntimeError("Frozen response result hash changed")
    rows=_read_rows(path)
    result=summarize(rows,mf,bootstrap=a.bootstrap,seed=a.seed,
                     min_distance_span=a.min_distance_span)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use new analysis output directory")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.json").open("w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    write_csv(out/"per_scene_distance.csv",result["per_scene"])
    lines=["# 冻结 GVAR 米制深度：距离／裁剪／相机条件响应","",
        f"源运行状态：{result['source_status']}；场景{result['scenes']}，帧{result['identity_frames']}。",
        f"跨帧参考／预测Spearman: {result['across_frame_spearman']}",
        f"基于场景去中心化的距离响应斜率: {result['scene_centered_slope']}",
        f"场景内可计算斜率数量: {result['scenes_with_valid_slope']}",
        "",
        "| 干预 | 帧 | 同射线median绝对深度变化(m) | 同射线深度比值 | 相机pose差(deg) |",
        "|---|---:|---:|---:|---:|"]
    for name,stats in result["interventions"].items():
        def fmt(key):
            v=stats[key]["mean"]
            return "NA" if v is None else f"{v:.5f}"
        lines.append(f"| {name} | {stats['frames']} | {fmt('same_ray_median_abs_delta')} | "
                     f"{fmt('same_ray_median_ratio')} | {fmt('pose_delta_deg')} |")
    lines.extend(["","## 解释边界",
        "本轮未训练、不拟合校准倍率、不执行GraspNet/GC6D抓取AP。",
        "仅把crop/zoom解释为数字图像坐标变换，不等价真实相机焦距变化。",
        "传感器参考深度只离线评价；原始GC6D crop及工作空间筛选仍使用传感器深度。",
        "姿态替换固定图像，仅分析FiLM敏感性，非真实相机运动。"])
    (out/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"scenes":result["scenes"],
                      "within_scene_slope":result["within_scene_slope"],
                      "interventions":{k:v["same_ray_median_abs_delta"]
                                       for k,v in result["interventions"].items()}},
                      indent=2))


if __name__=="__main__":
    main()
