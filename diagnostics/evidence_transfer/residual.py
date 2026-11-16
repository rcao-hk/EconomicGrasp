"""P0: paired score64-vs-all physical-quality residuals and failure attribution.

No training and no model selection. Source actions are IDENTICAL for both
methods. The native/top10 membership masks are descriptive, not a causal NMS
decomposition. Previously inspected Novel scenes remain exploratory evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contract import (
    SCHEMA, METHODS, failure_classes, geometry_error,
    load_manifest, load_frame, pose_attributes, sha256, save_json,
)


def _write_csv(path, rows):
    if not rows:
        return
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def per_action(frame, record):
    q=np.asarray(frame["quality_bits"],dtype=float).sum(axis=1)/6.
    scores={m:np.asarray(frame[f"score_{m}"],dtype=float) for m in METHODS}
    classes=failure_classes(frame)
    poses=pose_attributes(frame["grasps"])
    geom_err=geometry_error(frame)
    rows=[]
    for i in range(len(q)):
        row=dict(
            dataset=str(record["dataset"]),split=str(record["split"]),
            scene=str(record["scene"]),anno=int(record["anno"]),
            assigned_obj=int(frame["assigned_obj"][i]),
            physical_id=str(frame["physical_id"][i]),
            quality=float(q[i]),quality_count=int(q[i]*6+1e-6),
            failure=int(classes[i]),
            native_percentile=float(frame["native_percentile"][i]),
            native_high=int(frame["native_percentile"][i]>=.75),
            score_native=float(scores["native"][i]),
            score_score64=float(scores["score64"][i]),
            score_all=float(scores["all"][i]),
            delta_score=float(scores["all"][i]-scores["score64"][i]),
            delta_sq_error=float((scores["all"][i]-q[i])**2-
                                 (scores["score64"][i]-q[i])**2),
            sq_error_all=float((scores["all"][i]-q[i])**2),
            sq_error_score64=float((scores["score64"][i]-q[i])**2),
            overestimate_all=float(scores["all"][i]-q[i]),
            overestimate_score64=float(scores["score64"][i]-q[i]),
            high_overestimate_all=int(q[i]==0 and scores["all"][i]>=.5),
            high_overestimate_score64=int(q[i]==0 and scores["score64"][i]>=.5),
            q_strong=int(q[i]>=5/6-1e-7),
            width_m=float(frame["grasps"][i,1]),
            insertion_m=float(frame["grasps"][i,3]),
            center_z_m=float(frame["grasps"][i,15]),
            **{name:int(vals[i]) for name,vals in poses.items()},
        )
        for method in METHODS:
            for k in (10,50):
                row[f"top{k}_{method}"]=int(frame[f"top{k}_{method}"][i])
        if geom_err is not None:
            row["geometry_abs_dz_error"]=float(geom_err[i]) if np.isfinite(geom_err[i]) else None
        rows.append(row)
    return rows


def _safe_mean(rows, key):
    values=[float(r[key]) for r in rows if key in r and r[key] is not None
            and np.isfinite(r[key])]
    return float(np.mean(values)) if values else None


def summarize_group(rows):
    if not rows:
        raise ValueError("Cannot summarize empty group")
    q=np.array([r["quality"] for r in rows])
    zero=[r for r in rows if r["failure"]!=0]
    collision=[r for r in rows if r["failure"]==1]
    empty=[r for r in rows if r["failure"]==2]
    other=[r for r in rows if r["failure"]==3]
    high=[r for r in rows if r["native_high"] and r["failure"]!=0]
    strong=[r for r in rows if r["q_strong"]]
    return dict(
        actions=len(rows),
        mean_quality=float(q.mean()),
        mean_delta_score=_safe_mean(rows,"delta_score"),
        delta_mse=_safe_mean(rows,"delta_sq_error"),
        score64_mse=_safe_mean(rows,"sq_error_score64"),
        all_mse=_safe_mean(rows,"sq_error_all"),
        zero_actions=len(zero),collision_actions=len(collision),
        empty_actions=len(empty),other_zero_actions=len(other),
        high_native_zero_actions=len(high),strong_quality_actions=len(strong),
        zero_all_score=_safe_mean(zero,"score_all"),
        zero_score64_score=_safe_mean(zero,"score_score64"),
        collision_delta_score=_safe_mean(collision,"delta_score"),
        empty_delta_score=_safe_mean(empty,"delta_score"),
        other_zero_delta_score=_safe_mean(other,"delta_score"),
        strong_delta_score=_safe_mean(strong,"delta_score"),
        high_failure_delta_score=_safe_mean(high,"delta_score"),
        zero_overestimate_rate_all=_safe_mean(zero,"high_overestimate_all"),
        zero_overestimate_rate_score64=_safe_mean(zero,"high_overestimate_score64"),
        top10_native_quality=_safe_mean([r for r in rows if r["top10_native"]],"quality"),
        top10_all_quality=_safe_mean([r for r in rows if r["top10_all"]],"quality"),
        top10_score64_quality=_safe_mean([r for r in rows if r["top10_score64"]],"quality"),
        strong_in_native_top10=sum(r["top10_native"] and r["q_strong"] for r in rows),
        strong_in_all_top10=sum(r["top10_all"] and r["q_strong"] for r in rows),
    )


def _inversions(frame):
    """Same-object high-value-vs-failed pairs within matched survivor union.

    Exclude invalid/unassigned object IDs. This measures score-order errors
    in the labeled pool, NOT actual directed historical NMS suppressions.
    """
    obj=np.asarray(frame["assigned_obj"],dtype=np.int64)
    q=np.asarray(frame["quality_bits"],dtype=int).sum(axis=1)
    percent=np.asarray(frame["native_percentile"],dtype=float)
    old=np.asarray(frame["score_score64"],dtype=float)
    new=np.asarray(frame["score_all"],dtype=float)
    wrong_a=wrong_b=rescued=newly_wrong=0
    for oid in np.unique(obj):
        if int(oid)<0:continue
        bad=np.flatnonzero((obj==oid)&(q==0)&(percent>=.75))
        good=np.flatnonzero((obj==oid)&(q>=5))
        if not len(bad) or not len(good):continue
        i=np.repeat(bad,len(good))
        j=np.tile(good,len(bad))
        wa=old[i]>=old[j]
        wb=new[i]>=new[j]
        wrong_a+=int(wa.sum());wrong_b+=int(wb.sum())
        rescued+=int((wa&~wb).sum())
        newly_wrong+=int((~wa&wb).sum())
    return dict(wrong_score64=wrong_a,wrong_all=wrong_b,
                new_wrong_pairs=newly_wrong,rescued_pairs=rescued)


def _group_by(rows, keys, min_n):
    grouped=defaultdict(list)
    for row in rows:
        key=tuple(row[field] for field in keys)
        grouped[key].append(row)
    output=[]
    for key,group in sorted(grouped.items(),key=lambda z:str(z[0])):
        if len(group)<min_n:continue
        output.append({**dict(zip(keys,key)),**summarize_group(group)})
    return output


def _bootstrap_scene_difference(scene_rows, key, reps, seed):
    valid=[x for x in scene_rows if x.get(key) is not None]
    if not valid:return dict(mean=None,ci95=None,scenes=0)
    vals=np.array([float(x[key]) for x in valid],dtype=float)
    rng=np.random.default_rng(seed)
    if len(vals)==1:
        return dict(mean=float(vals.mean()),ci95=None,scenes=1)
    picks=rng.integers(0,len(vals),size=(reps,len(vals)))
    ci=np.quantile(vals[picks].mean(axis=1),[.025,.975])
    return dict(mean=float(vals.mean()),ci95=ci.tolist(),scenes=len(vals))


def run(a):
    if a.bootstrap<1 or a.min_cell<1:raise ValueError("Invalid bootstrap/minimum cell")
    manifest=load_manifest(a.manifest)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    actions=[];frames=[]
    for ix,rec in enumerate(manifest["frames"],1):
        x=load_frame(rec)
        ar=per_action(x,rec)
        inv=_inversions(x)
        actions.extend(ar)
        frames.append(dict(dataset=rec["dataset"],split=rec["split"],
                           scene=str(rec["scene"]),anno=int(rec["anno"]),
                           **summarize_group(ar),**inv))
        if ix%64==0 or ix==len(manifest["frames"]):
            print(f"[P0 residual] {ix}/{len(manifest['frames'])}",flush=True)
    _write_csv(root/"per_action.csv",actions)
    _write_csv(root/"per_frame.csv",frames)
    scenes=_group_by(frames,("dataset","split","scene"),1)
    _write_csv(root/"per_scene.csv",scenes)
    # Object ID is scene-local; never pool object IDs across different scenes.
    objects=_group_by(actions,("dataset","split","scene","assigned_obj"),a.min_cell)
    _write_csv(root/"per_object.csv",objects)
    axes=("failure","native_high","q_strong","polar_bin","width_bin",
          "center_depth_bin","insertion_bin")
    strata={}
    for axis in axes:
        grouped=_group_by(actions,("dataset","split",axis),a.min_cell)
        _write_csv(root/f"strata_{axis}.csv",grouped)
        strata[axis]=len(grouped)
    stats={}
    for dataset,split in sorted({(r["dataset"],r["split"]) for r in frames}):
        records=[r for r in frames if r["dataset"]==dataset and r["split"]==split]
        # First aggregate by scene, then bootstrap scenes: never pretend each
        # frame, object or candidate is an IID new environment.
        sr=_group_by(records,("scene",),1)
        stats[f"{dataset}/{split}"]={
            "frames":len(records),"scenes":len(sr),
            "delta_mse":_bootstrap_scene_difference(sr,"delta_mse",a.bootstrap,a.seed),
            "collision_delta_score":_bootstrap_scene_difference(sr,"collision_delta_score",a.bootstrap,a.seed+1),
            "empty_delta_score":_bootstrap_scene_difference(sr,"empty_delta_score",a.bootstrap,a.seed+2),
            "strong_delta_score":_bootstrap_scene_difference(sr,"strong_delta_score",a.bootstrap,a.seed+3),
            "high_failure_delta_score":_bootstrap_scene_difference(sr,"high_failure_delta_score",a.bootstrap,a.seed+4),
            "inversions":{k:int(sum(r[k] for r in records)) for k in
                ("wrong_score64","wrong_all","new_wrong_pairs","rescued_pairs")},
        }
    report=dict(schema=SCHEMA,kind="paired_physical_residual",
                manifest_sha256=manifest["manifest_sha256"],
                frames=len(frames),actions=len(actions),
                scene_conditional=stats,conditional_strata_files=strata,
                provenance=manifest.get("provenance",{}),
                notes=["Same-action paired scores and CAD labels; no model training",
                       "Geometry error only when paired explicit reference_geometry exists",
                       "Per-object IDs are scene-local assignments, not category labels",
                       "Top10 flags describe saved compiled official-prefix choices; no directed NMS causal attribution",
                       "Already inspected test scenes remain exploratory"])
    save_json(root/"summary.json",report)
    lines=["# 冻结 GVAR 同动作评分残差与失败类型审计","",
           "对比score64和all在完全相同物理动作的质量预测，不重训、不选模。","",
           "| 数据集/划分 | 场景 | ΔMSE(all-score64) | Collision Δscore | Empty Δscore | 强质量 Δscore |",
           "|---|---:|---:|---:|---:|---:|"]
    for k,v in stats.items():
        fmt=lambda d: "NA" if d["mean"] is None else f"{d['mean']:+.6f}"
        lines.append(f"| {k} | {v['scenes']} | {fmt(v['delta_mse'])} | "
                     f"{fmt(v['collision_delta_score'])} | {fmt(v['empty_delta_score'])} | "
                     f"{fmt(v['strong_delta_score'])} |")
    lines.extend(["", "各区间对场景进行bootstrap，场景数量不能由动作数代替。",
                  "测试集多次查看，结果只用于机制探索。若未提供reference_geometry，不输出预测几何误差归因。"])
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--min-cell",type=int,default=30)
    p.add_argument("--bootstrap",type=int,default=3000)
    p.add_argument("--seed",type=int,default=20261116)
    run(p.parse_args())


if __name__=="__main__":
    main()
